/* Independent worst-case carry/overflow audit for Kronecker + Salem-Spencer packing. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include "include/xcruncher.h"

static void put24(uint8_t *b, size_t i, uint32_t v) { b[3*i] = v & 0xFF; b[3*i+1] = (v >> 8) & 0xFF; b[3*i+2] = (v >> 16) & 0xFF; }
static uint32_t get24(const uint8_t *b, size_t i) { return b[3*i] | (b[3*i+1] << 8) | ((uint32_t)b[3*i+2] << 16); }

static void exp3_case(size_t T, uint32_t wv, uint32_t xv, const char *tag) {
    size_t n = (T * 3 + 7) / 8;
    uint64_t *W = calloc(n, 8), *X = calloc(n, 8), *P = calloc(2 * n, 8);
    for (size_t i = 0; i < T; i++) { put24((uint8_t *)W, i, wv); put24((uint8_t *)X, i, xv); }
    xc_mul_u64(P, X, n, W, n);
    size_t first_bad = (size_t)-1, bad = 0;
    for (size_t t = 0; t < T; t++) {
        uint64_t ref = (uint64_t)(t + 1) * wv * xv;
        if ((uint64_t)get24((uint8_t *)P, t) != ref) { bad++; if (first_bad == (size_t)-1) first_bad = t; }
    }
    printf("[Exp3-layout 24-bit %s] T=%zu w=%u x=%u: wrong slots=%zu/%zu first_bad_t=%zd (theory first bad t = %llu)\n",
           tag, T, wv, xv, bad, T, (ssize_t)first_bad, (unsigned long long)((1ull << 24) - 1) / (wv * xv));
    free(W); free(X); free(P);
}

static void ss_case(size_t K, size_t T) {
    const size_t e4[4] = {0, 1, 3, 4}, S = 9;
    size_t wn = (K * S * 4 + 7) / 8, xn = (T * S * 4 + 7) / 8;
    uint64_t *W = calloc(wn, 8), *X = calloc(xn, 8), *P = calloc(wn + xn, 8);
    uint32_t *wl = (uint32_t *)W, *xl = (uint32_t *)X;
    for (size_t p = 0; p < 4; p++) {
        for (size_t j = 0; j < K; j++) wl[j * S + e4[p]] = 15;
        for (size_t t = 0; t < T; t++) xl[t * S + e4[p]] = 255;
    }
    xc_mul_u64(P, X, xn, W, wn);
    const uint32_t *pl = (const uint32_t *)P;
    size_t bad = 0;
    for (size_t t = 0; t < T; t++)
        for (size_t p = 0; p < 4; p++) {
            uint64_t lim = (t + 1 < K) ? t + 1 : K;
            if ((uint64_t)pl[t * S + 2 * e4[p]] != lim * 15 * 255) bad++;
        }
    printf("[Salem-Spencer P=4 32-bit, w=15 x=255] K=%zu T=%zu: wrong diagonal outputs=%zu/%zu (2*K*3825=%llu, 2^32=4294967296)\n",
           K, T, bad, 4 * T, (unsigned long long)(2ull * K * 3825));
    free(W); free(X); free(P);
}

static void circ_case(size_t B, size_t T, int random) {
    size_t wn = (2 * B * 4 + 7) / 8, xn = (T * 2 * B * 4 + 7) / 8;
    uint64_t *W = calloc(wn, 8), *X = calloc(xn, 8), *P = calloc(wn + xn, 8);
    uint8_t *c = malloc(B), *xq = malloc(T * B);
    srand(7);
    for (size_t k = 0; k < B; k++) c[k] = random ? rand() % 16 : 15;
    for (size_t i = 0; i < T * B; i++) xq[i] = random ? rand() % 256 : 255;
    uint32_t *ws = (uint32_t *)W, *xs = (uint32_t *)X;
    for (size_t m = 0; m < 2 * B; m++) ws[m] = c[(m + 1) % B];
    for (size_t t = 0; t < T; t++) for (size_t k = 0; k < B; k++) xs[t * 2 * B + k] = xq[t * B + k];
    xc_mul_u64(P, X, xn, W, wn);
    const uint32_t *ps = (const uint32_t *)P;
    size_t bad = 0;
    for (size_t t = 0; t < T; t++)
        for (size_t r = 0; r < B; r++) {
            uint64_t ref = 0;
            for (size_t k = 0; k < B; k++) ref += (uint64_t)xq[t * B + k] * c[(r + B - k) % B];
            if (ps[t * 2 * B + (B - 1) + r] != ref) bad++;
        }
    printf("[Block-circulant B=%zu T=%zu %s] wrong outputs=%zu/%zu\n", B, T, random ? "random" : "worst-case 15x255", bad, T * B);
    free(W); free(X); free(P); free(c); free(xq);
}

int main(void) {
    exp3_case(1000000, 15, 15, "worst u4xu4");
    exp3_case(74565, 15, 15, "at bound");
    exp3_case(74566, 15, 15, "bound+1");
    ss_case(32, 4096);
    ss_case(4096, 32768);
    ss_case(561433, 561450);
    ss_case(561434, 561450);
    circ_case(64, 128, 0);
    circ_case(64, 128, 1);
    return 0;
}
