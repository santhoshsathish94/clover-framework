#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/pebbling/w2"
cd "$W2"

cat << 'EOF' > matmul_exact.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/*
 * Exact 0-1 BFS solver for the 20-node 2x2 Matrix Multiplication DAG
 * under Hong & Kung (1981) Section 2 rules:
 *   Nodes 0..3:  A00, A01, A10, A11 (inputs)
 *   Nodes 4..7:  B00, B01, B10, B11 (inputs)
 *   Nodes 8..15: products p_{i,j,k} = A_{i,k} * B_{k,j}
 *   Nodes 16..19: outputs C_{i,j} = p_{i,j,0} + p_{i,j,1}
 *
 * State:
 *   red_mask  (16 bits: nodes 0..15; outputs 16..19 have no successors so R3(o)+R2(o)+R4(o,red) is immediate)
 *   blue_prod (8 bits: products 8..15 currently in blue)
 *   blue_out  (4 bits: outputs 16..19 currently in blue; inputs 0..7 are always in blue until final R4 cleanup)
 *
 * Dead-pebble dominance (exact):
 *   Once output c = 16 + idx is in blue_out, its two products (8 + 2*idx, 8 + 2*idx + 1) have no remaining
 *   uncomputed successors, so free R4 removes them from red_mask and blue_prod immediately.
 *   Also, any input u in 0..7 whose two product successors both belong to completed outputs is removed from red_mask.
 */

static const int prod_preds[8][2] = {
    {0, 4}, {1, 6}, /* C00: A00*B00, A01*B10 */
    {0, 5}, {1, 7}, /* C01: A00*B01, A01*B11 */
    {2, 4}, {3, 6}, /* C10: A10*B00, A11*B10 */
    {2, 5}, {3, 7}  /* C11: A10*B01, A11*B11 */
};

static inline uint32_t clean_state(uint32_t red, uint32_t bprod, uint32_t bout) {
    for (int c = 0; c < 4; c++) {
        if (bout & (1u << c)) {
            uint32_t pbits = (3u << (2 * c));
            bprod &= ~pbits;
            red &= ~(pbits << 8);
        }
    }
    /* Clean inputs in red whose two products are both in completed outputs or already computed */
    for (int u = 0; u < 8; u++) {
        if (red & (1u << u)) {
            int needed = 0;
            for (int p = 0; p < 8; p++) {
                if (prod_preds[p][0] == u || prod_preds[p][1] == u) {
                    int c = p >> 1;
                    if (!(bout & (1u << c)) && !(red & (1u << (8 + p))) && !(bprod & (1u << p))) {
                        needed = 1;
                    }
                }
            }
            if (!needed) red &= ~(1u << u);
        }
    }
    return red | (bprod << 16) | (bout << 24);
}

#define HT_SIZE (1 << 22)
#define HT_MASK (HT_SIZE - 1)

static uint32_t ht_key[HT_SIZE];
static uint16_t ht_dist[HT_SIZE];
static uint32_t *dq;
#define DQ_CAP (1 << 22)
#define DQ_MASK (DQ_CAP - 1)

static inline void push_state(uint32_t ns, uint16_t nd, int cost_delta,
                              uint32_t *head, uint32_t *tail, uint64_t *visited) {
    uint32_t hi = (ns * 2654435761u) & HT_MASK;
    while (ht_key[hi] != 0xffffffffu && ht_key[hi] != ns) hi = (hi + 1) & HT_MASK;
    if (ht_key[hi] == 0xffffffffu || nd < ht_dist[hi]) {
        if (ht_key[hi] == 0xffffffffu) {
            ht_key[hi] = ns;
            (*visited)++;
        }
        ht_dist[hi] = nd;
        if (cost_delta == 0) {
            (*head)--;
            dq[(*head) & DQ_MASK] = ns;
        } else {
            dq[(*tail) & DQ_MASK] = ns;
            (*tail)++;
        }
    }
}

int solve_matmul(int S, int max_bprod, uint64_t *states_out) {
    if (S <= 2) { *states_out = 0; return -1; }
    memset(ht_key, 0xff, sizeof(ht_key));
    memset(ht_dist, 0xff, sizeof(ht_dist));

    dq = (uint32_t *)malloc(DQ_CAP * sizeof(uint32_t));
    uint32_t head = 0, tail = 0;

    *states_out = 0;
    uint32_t init = clean_state(0, 0, 0);
    push_state(init, 0, 1, &head, &tail, states_out);

    while (head != tail) {
        uint32_t st = dq[head & DQ_MASK];
        head++;
        uint32_t idx = (st * 2654435761u) & HT_MASK;
        while (ht_key[idx] != st) idx = (idx + 1) & HT_MASK;
        uint16_t d = ht_dist[idx];

        uint32_t red = st & 0xffffu;
        uint32_t bprod = (st >> 16) & 0xffu;
        uint32_t bout = (st >> 24) & 0xfu;

        if (bout == 0xfu) {
            free(dq);
            return (int)d;
        }

        int rc = __builtin_popcount(red);

        /* Dominance 1: if any output C_c has both products in red and rc < S, immediately R3(C_c)+R2(C_c)+R4(C_c,red) */
        if (rc < S) {
            int fired = 0;
            for (int c = 0; c < 4; c++) {
                if (bout & (1u << c)) continue;
                uint32_t p0 = 1u << (8 + 2 * c);
                uint32_t p1 = 1u << (8 + 2 * c + 1);
                if ((red & p0) && (red & p1)) {
                    uint32_t ns = clean_state(red, bprod, bout | (1u << c));
                    push_state(ns, d + 1, 1, &head, &tail, states_out);
                    fired = 1;
                    break;
                }
            }
            if (fired) continue;
        }

        /* Cost 0: R3 on any product p in 0..7 if preds in red and rc < S */
        if (rc < S) {
            for (int p = 0; p < 8; p++) {
                int c = p >> 1;
                if (bout & (1u << c)) continue;
                uint32_t pbit = 1u << (8 + p);
                if ((red & pbit) || (bprod & (1u << p))) continue;
                int u0 = prod_preds[p][0], u1 = prod_preds[p][1];
                if ((red & (1u << u0)) && (red & (1u << u1))) {
                    uint32_t ns = clean_state(red | pbit, bprod, bout);
                    push_state(ns, d, 0, &head, &tail, states_out);
                }
            }
        }

        /* Cost 0: R4(red) on any input 0..7 in red (deleting an unspilled product 8..15 before its output C_c is a strict no-op) */
        int r4_limit = (max_bprod >= 99) ? 16 : 8;
        for (int v = 0; v < r4_limit; v++) {
            if (red & (1u << v)) {
                uint32_t ns = clean_state(red ^ (1u << v), bprod, bout);
                push_state(ns, d, 0, &head, &tail, states_out);
            }
        }

        /* Cost 1: R1 on any input 0..7 or blue product 8..15 if rc < S */
        if (rc < S) {
            for (int u = 0; u < 8; u++) {
                if (!(red & (1u << u))) {
                    uint32_t ns = clean_state(red | (1u << u), bprod, bout);
                    push_state(ns, d + 1, 1, &head, &tail, states_out);
                }
            }
            for (int p = 0; p < 8; p++) {
                if ((bprod & (1u << p)) && !(red & (1u << (8 + p)))) {
                    uint32_t ns = clean_state(red | (1u << (8 + p)), bprod ^ (1u << p), bout);
                    push_state(ns, d + 1, 1, &head, &tail, states_out);
                }
            }
        }

        /* Cost 1: R2(p) + R4(p, red) on any product p in red */
        int bp_cap = (max_bprod >= 99) ? 8 : max_bprod;
        if (__builtin_popcount(bprod) < bp_cap) {
            for (int p = 0; p < 8; p++) {
                if ((red & (1u << (8 + p))) && !(bprod & (1u << p))) {
                    uint32_t ns = clean_state(red ^ (1u << (8 + p)), bprod | (1u << p), bout);
                    push_state(ns, d + 1, 1, &head, &tail, states_out);
                }
            }
        }
    }
    free(dq);
    return -1;
}

int main(void) {
    printf("=== 2x2 MATRIX MULTIPLY DAG (20 nodes: 8 inputs, 8 products, 4 outputs) EXACT Q*(S) ===\n");
    fflush(stdout);
    for (int S = 1; S <= 8; S++) {
        uint64_t st_b0 = 0, st_b1 = 0, st_full = 0;
        int q_b0 = solve_matmul(S, 0, &st_b0);
        int q_b1 = solve_matmul(S, 1, &st_b1);
        int q_full = solve_matmul(S, 99, &st_full);
        printf("  S=%d: exact_unpruned Q*=%d (states=%llu) | max_bprod=1: Q=%d (states=%llu) | max_bprod=0: Q=%d (states=%llu)\n",
               S, q_full, (unsigned long long)st_full, q_b1, (unsigned long long)st_b1, q_b0, (unsigned long long)st_b0);
        fflush(stdout);
    }
    return 0;
}
EOF

gcc -O3 matmul_exact.c -o matmul_exact
./matmul_exact
