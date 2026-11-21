/*
 * Reproducible Experiments 1, 2 & 3: Giant-Integer Neural Layer Lab (x-cruncher v1.6.1)
 *
 * Build & run (from repository root):
 *   gcc -O3 -std=c11 -fopenmp research/giant-integer-trunk-lm/experiments/exp_giant_int_layer.c \
 *       clover-ai-system/studio/clover/_xcruncher_c/src/xcruncher.c \
 *       -I clover-ai-system/studio/clover/_xcruncher_c -lm -o exp_giant_int_layer
 *   ./exp_giant_int_layer
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "include/xcruncher.h"

static void pack_slot24(uint8_t *buf, size_t slot_idx, uint32_t val) {
    size_t byte_off = slot_idx * 3;
    buf[byte_off + 0] = (uint8_t)(val & 0xFFu);
    buf[byte_off + 1] = (uint8_t)((val >> 8) & 0xFFu);
    buf[byte_off + 2] = (uint8_t)((val >> 16) & 0xFFu);
}

static uint32_t unpack_slot24(const uint8_t *buf, size_t max_bytes, size_t slot_idx) {
    size_t byte_off = slot_idx * 3;
    uint32_t v = 0;
    if (byte_off + 0 < max_bytes) v |= (uint32_t)buf[byte_off + 0];
    if (byte_off + 1 < max_bytes) v |= ((uint32_t)buf[byte_off + 1] << 8);
    if (byte_off + 2 < max_bytes) v |= ((uint32_t)buf[byte_off + 2] << 16);
    return v;
}

int main(void) {
    printf("=== x-cruncher Giant-Integer Neural Layer Lab ===\n");
    printf("Active ISA: %s\n\n", xc_active_isa_name());

    /* Exp 1: Unstructured Dense Matrix as 1 Giant Multiply (R=64, K=256, T=16) */
    {
        const size_t R = 64, K = 256, T = 16;
        uint8_t *W_mat = (uint8_t *)malloc(R * K);
        uint8_t *X_mat = (uint8_t *)malloc(T * K);
        uint32_t *Y_ref = (uint32_t *)calloc(R * T, sizeof(uint32_t));
        for (size_t i = 0; i < R * K; i++) W_mat[i] = (uint8_t)((i * 7 + 3) & 0xF);
        for (size_t i = 0; i < T * K; i++) X_mat[i] = (uint8_t)((i * 13 + 5) & 0xFF);

        double t_mm0 = omp_get_wtime();
        for (int rep = 0; rep < 100; rep++) {
            for (size_t r = 0; r < R; r++)
                for (size_t t = 0; t < T; t++) {
                    uint32_t s = 0;
                    for (size_t k = 0; k < K; k++) s += (uint32_t)W_mat[r * K + k] * (uint32_t)X_mat[t * K + k];
                    Y_ref[r * T + t] = s;
                }
        }
        double ms_mm = (omp_get_wtime() - t_mm0) * 1000.0 / 100.0;

        size_t w_slots = R * T * 2 * K, x_slots = T * 2 * K;
        size_t w_u64 = (w_slots * 3 + 7) / 8, x_u64 = (x_slots * 3 + 7) / 8, p_u64 = w_u64 + x_u64;
        uint64_t *W_big = (uint64_t *)calloc(w_u64, 8);
        uint64_t *X_big = (uint64_t *)calloc(x_u64, 8);
        uint64_t *P_big = (uint64_t *)calloc(p_u64, 8);

        for (size_t r = 0; r < R; r++)
            for (size_t k = 0; k < K; k++)
                pack_slot24((uint8_t *)W_big, r * (T * 2 * K) + (K - 1 - k), W_mat[r * K + k]);
        for (size_t t = 0; t < T; t++)
            for (size_t k = 0; k < K; k++)
                pack_slot24((uint8_t *)X_big, t * (2 * K) + k, X_mat[t * K + k]);

        xc_plan_t *plan = xc_plan_create(x_u64, w_u64);
        xc_trunk_t *w_trunk = xc_trunk_create(plan, W_big, w_u64);
        xc_mul_trunk(plan, P_big, X_big, x_u64, w_trunk);

        double t0 = omp_get_wtime();
        for (int it = 0; it < 5; it++) xc_mul_trunk(plan, P_big, X_big, x_u64, w_trunk);
        double ms_xc = (omp_get_wtime() - t0) * 1000.0 / 5.0;

        size_t mismatches = 0;
        for (size_t r = 0; r < R; r++)
            for (size_t t = 0; t < T; t++)
                if (unpack_slot24((const uint8_t *)P_big, p_u64 * 8, (r * T + t) * (2 * K) + (K - 1)) != Y_ref[r * T + t])
                    mismatches++;

        printf("[Exp 1: Unstructured Dense Matrix (R=%zu, K=%zu, T=%zu)]\n", R, K, T);
        printf("  Raw weights = %.1f KB -> Padded Giant W_big = %.1f KB\n", (R * K * 0.5) / 1024.0, w_u64 * 8 / 1024.0);
        printf("  Direct MatMul: %.4f ms | 1-Shot xc_mul_trunk: %.3f ms | Exact matches: %zu / %zu\n\n",
               ms_mm, ms_xc, R * T - mismatches, R * T);

        xc_trunk_destroy(w_trunk); xc_plan_destroy(plan);
        free(W_mat); free(X_mat); free(Y_ref); free(W_big); free(X_big); free(P_big);
    }

    /* Exp 2: Structured Toeplitz Layer (K=2048 x 2048, T=256 tokens) */
    {
        const size_t K = 2048, T = 256;
        uint8_t *w_vec = (uint8_t *)malloc(2 * K);
        uint8_t *X_mat = (uint8_t *)malloc(T * K);
        for (size_t i = 0; i < 2 * K; i++) w_vec[i] = (uint8_t)((i * 11 + 1) & 0xF);
        for (size_t i = 0; i < T * K; i++) X_mat[i] = (uint8_t)((i * 17 + 3) & 0xFF);

        uint32_t *Y_ref = (uint32_t *)calloc(T * K, sizeof(uint32_t));
        double t_ref0 = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < T; t++)
            for (size_t r = 0; r < K; r++) {
                uint32_t s = 0;
                for (size_t k = 0; k < K; k++) s += (uint32_t)w_vec[r + (K - 1 - k)] * (uint32_t)X_mat[t * K + k];
                Y_ref[t * K + r] = s;
            }
        double ms_ref = (omp_get_wtime() - t_ref0) * 1000.0;

        size_t w_u64 = (2 * K * 3 + 7) / 8, x_u64 = (T * 2 * K * 3 + 7) / 8, p_u64 = w_u64 + x_u64;
        uint64_t *W_big = (uint64_t *)calloc(w_u64, 8);
        uint64_t *X_big = (uint64_t *)calloc(x_u64, 8);
        uint64_t *P_big = (uint64_t *)calloc(p_u64, 8);
        for (size_t i = 0; i < 2 * K; i++) pack_slot24((uint8_t *)W_big, i, w_vec[i]);
        for (size_t t = 0; t < T; t++)
            for (size_t k = 0; k < K; k++)
                pack_slot24((uint8_t *)X_big, t * (2 * K) + k, X_mat[t * K + k]);

        xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);
        double t_xc0 = omp_get_wtime();
        for (int it = 0; it < 5; it++) xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);
        double ms_xc = (omp_get_wtime() - t_xc0) * 1000.0 / 5.0;

        size_t mismatches = 0;
        for (size_t t = 0; t < T; t++)
            for (size_t r = 0; r < K; r++)
                if (unpack_slot24((const uint8_t *)P_big, p_u64 * 8, t * (2 * K) + (K - 1) + r) != Y_ref[t * K + r])
                    mismatches++;

        printf("[Exp 2: Toeplitz Layer K=2048x2048, T=256]\n");
        printf("  16-Core MatMul: %.3f ms | 1-Shot x-cruncher: %.3f ms (%.1fx FASTER) | Exact: %zu/%zu\n\n",
               ms_ref, ms_xc, ms_ref / ms_xc, T * K - mismatches, T * K);
        free(w_vec); free(X_mat); free(Y_ref); free(W_big); free(X_big); free(P_big);
    }

    /* Exp 3: 1,000,000-Token Global Causal Sequence Mixing */
    {
        const size_t T = 1000000;
        size_t u64_len = (T * 3 + 7) / 8;
        uint64_t *W_big = (uint64_t *)calloc(u64_len, 8);
        uint64_t *X_big = (uint64_t *)calloc(u64_len, 8);
        uint64_t *P_big = (uint64_t *)calloc(2 * u64_len, 8);
        uint8_t *w_tap = (uint8_t *)malloc(T), *x_tok = (uint8_t *)malloc(T);
        for (size_t i = 0; i < T; i++) {
            w_tap[i] = (uint8_t)((i < 4096) ? ((i * 5 + 1) & 0xF) : ((i % 97 == 0) ? 3 : 1));
            x_tok[i] = (uint8_t)((i * 9 + 7) & 0xF);
            pack_slot24((uint8_t *)W_big, i, w_tap[i]);
            pack_slot24((uint8_t *)X_big, i, x_tok[i]);
        }

        xc_plan_t *plan = xc_plan_create(u64_len, u64_len);
        xc_trunk_t *trunk = xc_trunk_create(plan, W_big, u64_len);
        xc_mul_trunk(plan, P_big, X_big, u64_len, trunk);

        double t_trk0 = omp_get_wtime();
        for (int it = 0; it < 5; it++) xc_mul_trunk(plan, P_big, X_big, u64_len, trunk);
        double ms_trk = (omp_get_wtime() - t_trk0) * 1000.0 / 5.0;

        size_t check_pos[4] = {0, 12345, 500000, 999999}, bad = 0;
        for (int ci = 0; ci < 4; ci++) {
            size_t t = check_pos[ci];
            uint32_t ref_s = 0;
            for (size_t j = 0; j <= t; j++) ref_s += (uint32_t)w_tap[j] * (uint32_t)x_tok[t - j];
            if (unpack_slot24((const uint8_t *)P_big, 2 * u64_len * 8, t) != ref_s) bad++;
        }
        printf("[Exp 3: 1,000,000-Token Global Causal Mixing]\n");
        printf("  1M-Token RAM = %.2f MB | 1-Shot xc_mul_trunk = %.2f ms | Exact checks: %zu/4\n",
               u64_len * 8 / (1024.0 * 1024.0), ms_trk, 4 - bad);

        xc_trunk_destroy(trunk); xc_plan_destroy(plan);
        free(w_tap); free(x_tok); free(W_big); free(X_big); free(P_big);
    }
    return 0;
}
