/*
 * Step B: Base-3 Salem-Spencer AP-Free Multi-Channel Packing inside One Giant Integer
 *
 * Exponents e = {0, 1, 3, 4} (numbers with only {0, 1} in base 3):
 *   - Diagonal channel convolutions land on 2*e = {0, 2, 6, 8} (base-3 digits in {0, 2}).
 *   - All 6 cross-channel terms w_u * x_v (u != v) land on {1, 3, 4, 5, 7} (at least one '1' in base 3).
 *   - Zero cross-channel collision using 9 lanes/token instead of 16.
 *
 * Build & run:
 *   gcc -O3 -std=c11 -fopenmp research/giant-integer-trunk-lm/experiments/exp_salem_spencer_multichannel.c \
 *       clover-ai-system/studio/clover/_xcruncher_c/src/xcruncher.c \
 *       -I clover-ai-system/studio/clover/_xcruncher_c -lm -o exp_salem_spencer
 *   ./exp_salem_spencer
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "include/xcruncher.h"

int main(void) {
    printf("=== Step B: Salem-Spencer AP-Free Multi-Channel Giant-Integer Packing ===\n");
    const size_t e4[4] = {0, 1, 3, 4};
    const size_t STRIDE4 = 9;
    const size_t C2 = 256, T2 = 32768, K2 = 4096;

    uint8_t *W2 = (uint8_t *)malloc(C2 * K2);
    uint8_t *X2 = (uint8_t *)malloc(C2 * T2);
    for (size_t i = 0; i < C2 * K2; i++) W2[i] = (uint8_t)((i * 3 + 1) & 0xF);
    for (size_t i = 0; i < C2 * T2; i++) X2[i] = (uint8_t)((i * 5 + 2) & 0xF);

    double t_r2 = omp_get_wtime();
    uint64_t chk_ref2 = 0;
    #pragma omp parallel for reduction(+:chk_ref2) schedule(static)
    for (size_t c = 0; c < C2; c++) {
        const uint8_t *wc = W2 + c * K2, *xc = X2 + c * T2;
        for (size_t t = 0; t < T2; t++) {
            uint32_t s = 0;
            size_t lim = (t + 1 < K2) ? (t + 1) : K2;
            for (size_t j = 0; j < lim; j++) s += (uint32_t)wc[j] * (uint32_t)xc[t - j];
            if (t == T2 - 1) chk_ref2 += s;
        }
    }
    double ms_ref2 = (omp_get_wtime() - t_r2) * 1000.0;

    size_t NG2 = C2 / 4;
    size_t w4_u64 = (K2 * STRIDE4 * 4 + 7) / 8, x4_u64 = (T2 * STRIDE4 * 4 + 7) / 8;
    xc_plan_t *pl4 = xc_plan_create(x4_u64, w4_u64);
    uint64_t *W4_tmp = (uint64_t *)calloc(w4_u64, 8);
    uint64_t *X4_tmp = (uint64_t *)calloc(x4_u64, 8);
    uint64_t *P4_tmp = (uint64_t *)calloc(w4_u64 + x4_u64, 8);
    xc_trunk_t **trunks4 = (xc_trunk_t **)malloc(NG2 * sizeof(xc_trunk_t *));
    for (size_t g = 0; g < NG2; g++) {
        memset(W4_tmp, 0, w4_u64 * 8);
        uint32_t *wl = (uint32_t *)W4_tmp;
        for (size_t p = 0; p < 4; p++)
            for (size_t j = 0; j < K2; j++)
                wl[j * STRIDE4 + e4[p]] = W2[(g * 4 + p) * K2 + j];
        trunks4[g] = xc_trunk_create(pl4, W4_tmp, w4_u64);
    }

    double t_p4 = omp_get_wtime();
    uint64_t chk_p4 = 0;
    for (size_t g = 0; g < NG2; g++) {
        memset(X4_tmp, 0, x4_u64 * 8);
        uint32_t *xl = (uint32_t *)X4_tmp;
        for (size_t p = 0; p < 4; p++) {
            size_t off = e4[p];
            const uint8_t *xc = X2 + (g * 4 + p) * T2;
            for (size_t t = 0; t < T2; t++) xl[t * STRIDE4 + off] = xc[t];
        }
        xc_mul_trunk(pl4, P4_tmp, X4_tmp, x4_u64, trunks4[g]);
        const uint32_t *pl_out = (const uint32_t *)P4_tmp;
        for (size_t p = 0; p < 4; p++) chk_p4 += pl_out[(T2 - 1) * STRIDE4 + 2 * e4[p]];
    }
    double ms_p4 = (omp_get_wtime() - t_p4) * 1000.0;

    printf("  16-Core Direct Loop:           %.2f ms (chk=%llu)\n", ms_ref2, (unsigned long long)chk_ref2);
    printf("  P=4 Salem-Spencer (64 Trunks): %.2f ms (%.1fx FASTER, chk=%llu, Exact=%s)\n",
           ms_p4, ms_ref2 / ms_p4, (unsigned long long)chk_p4, (chk_p4 == chk_ref2) ? "YES" : "NO");

    for (size_t g = 0; g < NG2; g++) xc_trunk_destroy(trunks4[g]);
    xc_plan_destroy(pl4);
    free(trunks4); free(W4_tmp); free(X4_tmp); free(P4_tmp); free(W2); free(X2);
    return 0;
}
