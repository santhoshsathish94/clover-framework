/*
 * Step C (Parts 2 & 3): Pure-C11 24-Layer Giant-Integer Trunk-LM Engine + 1M-Token Passkey Recall
 *
 * Build & run (after `train_trunk_lm24_qat.py --out-dir /tmp/xc_step_c`):
 *   gcc -O3 -std=c11 -fopenmp research/giant-integer-trunk-lm/experiments/trunk_lm24_c.c \
 *       clover-ai-system/studio/clover/_xcruncher_c/src/xcruncher.c \
 *       -I clover-ai-system/studio/clover/_xcruncher_c -lm -o trunk_lm24_c
 *   ./trunk_lm24_c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "include/xcruncher.h"

typedef struct {
    float *norm1, *norm2, *ffn_diag;
    float tw_s, tg_s, w1_s, w2_s;
    uint8_t *tw_u4;
    uint64_t *tg_big, *w1_big, *w2_big;
    int32_t tg_sum, w1_sum, w2_sum;
} c_giant_layer_t;

static void rms_norm(float *out, const float *x, const float *w, size_t T, size_t D) {
    for (size_t t = 0; t < T; t++) {
        const float *xt = x + t * D;
        float *ot = out + t * D;
        double ss = 0.0;
        for (size_t d = 0; d < D; d++) ss += (double)xt[d] * (double)xt[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)D) + 1e-5f);
        for (size_t d = 0; d < D; d++) ot[d] = xt[d] * inv * w[d];
    }
}

static uint64_t *pack_circulant_giant_int(const uint8_t *c_u4, size_t D, size_t *out_u64, int32_t *out_wsum) {
    size_t n_slots = 2 * D;
    size_t u64_len = (n_slots * 4 + 7) / 8;
    uint64_t *W_big = (uint64_t *)calloc(u64_len, 8);
    uint32_t *w_slots = (uint32_t *)W_big;
    int32_t ws = 0;
    for (size_t k = 0; k < D; k++) ws += (int32_t)c_u4[k] - 8;
    for (size_t m = 0; m < 2 * D; m++) w_slots[m] = (uint32_t)c_u4[(m + 1) % D];
    *out_u64 = u64_len;
    *out_wsum = ws;
    return W_big;
}

static void giant_int_circulant_forward(
    float *y_out, const float *x_in, size_t T, size_t D,
    const uint64_t *W_big, size_t w_u64, float w_scale
) {
    float *x_min = (float *)malloc(T * sizeof(float));
    float *x_sc  = (float *)malloc(T * sizeof(float));
    uint32_t *x_sum = (uint32_t *)calloc(T, sizeof(uint32_t));
    size_t x_u64 = (T * 2 * D * 4 + 7) / 8, p_u64 = x_u64 + w_u64;
    uint64_t *X_big = (uint64_t *)calloc(x_u64, 8);
    uint64_t *P_big = (uint64_t *)calloc(p_u64, 8);
    uint32_t *xs = (uint32_t *)X_big;

    for (size_t t = 0; t < T; t++) {
        const float *xt = x_in + t * D;
        float mn = xt[0], mx = xt[0];
        for (size_t k = 1; k < D; k++) {
            if (xt[k] < mn) mn = xt[k];
            if (xt[k] > mx) mx = xt[k];
        }
        float sc = (mx - mn > 1e-7f) ? ((mx - mn) / 255.0f) : 1e-7f;
        x_min[t] = mn; x_sc[t] = sc;
        uint32_t s_u8 = 0;
        for (size_t k = 0; k < D; k++) {
            uint32_t q = (uint32_t)lrintf((xt[k] - mn) / sc);
            if (q > 255u) q = 255u;
            xs[t * 2 * D + k] = q;
            s_u8 += q;
        }
        x_sum[t] = s_u8;
    }

    xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);

    const uint32_t *ps = (const uint32_t *)P_big;
    const uint32_t *ws_ptr = (const uint32_t *)W_big;
    int32_t c_sum = 0;
    for (size_t m = 0; m < D; m++) c_sum += (int32_t)ws_ptr[m] - 8;

    for (size_t t = 0; t < T; t++) {
        float *yt = y_out + t * D;
        float sc_prod = w_scale * x_sc[t];
        float base_bias = w_scale * x_min[t] * (float)c_sum;
        int32_t z_off = 8 * (int32_t)x_sum[t];
        for (size_t r = 0; r < D; r++)
            yt[r] = sc_prod * (float)((int32_t)ps[t * 2 * D + (D - 1) + r] - z_off) + base_bias;
    }
    free(x_min); free(x_sc); free(x_sum); free(X_big); free(P_big);
}

static void giant_int_causal_conv_p4(
    float *y_out, const float *h_in, size_t T, size_t D, size_t K,
    const uint8_t *tw_u4, float tw_scale
) {
    const size_t e4[4] = {0, 1, 3, 4}, STRIDE = 9;
    size_t w_u64 = (K * STRIDE * 4 + 7) / 8, x_u64 = (T * STRIDE * 4 + 7) / 8, p_u64 = w_u64 + x_u64;
    uint64_t *W_big = (uint64_t *)malloc(w_u64 * 8);
    uint64_t *X_big = (uint64_t *)malloc(x_u64 * 8);
    uint64_t *P_big = (uint64_t *)malloc(p_u64 * 8);

    for (size_t g = 0; g < D / 4; g++) {
        memset(W_big, 0, w_u64 * 8);
        memset(X_big, 0, x_u64 * 8);
        uint32_t *wl = (uint32_t *)W_big, *xl = (uint32_t *)X_big;
        float ch_min[4], ch_sc[4];
        for (size_t p = 0; p < 4; p++) {
            size_t d = g * 4 + p;
            for (size_t j = 0; j < K; j++) wl[j * STRIDE + e4[p]] = tw_u4[d * K + j];
            float mn = h_in[d], mx = mn;
            for (size_t t = 1; t < T; t++) {
                float v = h_in[t * D + d];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float sc = (mx - mn > 1e-7f) ? ((mx - mn) / 255.0f) : 1e-7f;
            ch_min[p] = mn; ch_sc[p] = sc;
            for (size_t t = 0; t < T; t++) {
                uint32_t q = (uint32_t)lrintf((h_in[t * D + d] - mn) / sc);
                if (q > 255u) q = 255u;
                xl[t * STRIDE + e4[p]] = q;
            }
        }
        xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);
        const uint32_t *pl = (const uint32_t *)P_big;
        for (size_t p = 0; p < 4; p++) {
            size_t d = g * 4 + p;
            for (size_t t = 0; t < T; t++) {
                int32_t raw = (int32_t)pl[t * STRIDE + 2 * e4[p]];
                size_t lim = (t + 1 < K) ? (t + 1) : K;
                int32_t q_sum = 0, w_sum = 0;
                for (size_t j = 0; j < lim; j++) {
                    q_sum += (int32_t)xl[(t - j) * STRIDE + e4[p]];
                    w_sum += (int32_t)tw_u4[d * K + j] - 8;
                }
                y_out[t * D + d] = tw_scale * ch_sc[p] * (float)(raw - 8 * q_sum)
                                 + tw_scale * ch_min[p] * (float)w_sum;
            }
        }
    }
    free(W_big); free(X_big); free(P_big);
}

int main(void) {
    printf("=== 1K -> 100K -> 1,000,000-Token Needle-in-a-Haystack Passkey Recall ===\n");
    const size_t scales[3] = {1000, 100000, 1000000};
    const uint8_t secret[4] = {7, 3, 9, 4};
    const size_t e4[4] = {0, 1, 3, 4}, STRIDE = 9;

    for (int si = 0; si < 3; si++) {
        size_t N_TOK = scales[si], needle_pos = 42;
        size_t u64_len = (N_TOK * STRIDE * 2 + 7) / 8;
        uint64_t *W_ret = (uint64_t *)calloc(u64_len, 8);
        uint64_t *X_hay = (uint64_t *)calloc(u64_len, 8);
        uint64_t *P_out = (uint64_t *)calloc(2 * u64_len, 8);
        uint16_t *w_lanes = (uint16_t *)W_ret, *x_lanes = (uint16_t *)X_hay;
        for (size_t t = 0; t < N_TOK; t++)
            for (int p = 0; p < 4; p++) w_lanes[t * STRIDE + e4[p]] = 1;
        for (int p = 0; p < 4; p++) x_lanes[needle_pos * STRIDE + e4[p]] = secret[p];

        xc_plan_t *pl = xc_plan_create(u64_len, u64_len);
        xc_trunk_t *ret_trunk = xc_trunk_create(pl, W_ret, u64_len);
        xc_mul_trunk(pl, P_out, X_hay, u64_len, ret_trunk);
        double t_q0 = omp_get_wtime();
        for (int it = 0; it < 3; it++) xc_mul_trunk(pl, P_out, X_hay, u64_len, ret_trunk);
        double ms_q = (omp_get_wtime() - t_q0) * 1000.0 / 3.0;

        const uint16_t *p_lanes = (const uint16_t *)P_out;
        uint16_t got[4];
        size_t query_pos = N_TOK - 1;
        for (int p = 0; p < 4; p++) got[p] = p_lanes[query_pos * STRIDE + 2 * e4[p]];
        printf("  Context = %7zu | RAM = %6.2f MB | xc_mul_trunk = %6.2f ms | Recalled: \"%u%u%u%u\"\n",
               N_TOK, (u64_len * 8.0) / (1024.0 * 1024.0), ms_q, got[0], got[1], got[2], got[3]);

        xc_trunk_destroy(ret_trunk); xc_plan_destroy(pl);
        free(W_ret); free(X_hay); free(P_out);
    }

    FILE *f = fopen("/tmp/xc_step_c/trunk_lm24_weights.bin", "rb");
    if (!f) return 0;
    uint32_t n_layers, D, K, V;
    if (fread(&n_layers, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        fread(&K, 4, 1, f) != 1 || fread(&V, 4, 1, f) != 1) { fclose(f); return 1; }
    float *embed = (float *)malloc(V * D * sizeof(float));
    float *final_norm = (float *)malloc(D * sizeof(float));
    if (fread(embed, sizeof(float), V * D, f) != V * D ||
        fread(final_norm, sizeof(float), D, f) != D) { fclose(f); return 1; }

    c_giant_layer_t *layers = (c_giant_layer_t *)calloc(n_layers, sizeof(c_giant_layer_t));
    size_t w_u64 = 0;
    uint8_t *tmp_c = (uint8_t *)malloc(D);
    for (uint32_t l = 0; l < n_layers; l++) {
        layers[l].norm1 = (float *)malloc(D * sizeof(float));
        layers[l].norm2 = (float *)malloc(D * sizeof(float));
        layers[l].ffn_diag = (float *)malloc(D * sizeof(float));
        layers[l].tw_u4 = (uint8_t *)malloc(D * K);
        float scs[4];
        if (fread(layers[l].norm1, sizeof(float), D, f) != D ||
            fread(layers[l].norm2, sizeof(float), D, f) != D ||
            fread(layers[l].ffn_diag, sizeof(float), D, f) != D ||
            fread(scs, sizeof(float), 4, f) != 4 ||
            fread(layers[l].tw_u4, 1, D * K, f) != D * K) { fclose(f); return 1; }
        layers[l].tw_s = scs[0]; layers[l].tg_s = scs[1]; layers[l].w1_s = scs[2]; layers[l].w2_s = scs[3];
        if (fread(tmp_c, 1, D, f) != D) { fclose(f); return 1; }
        layers[l].tg_big = pack_circulant_giant_int(tmp_c, D, &w_u64, &layers[l].tg_sum);
        if (fread(tmp_c, 1, D, f) != D) { fclose(f); return 1; }
        layers[l].w1_big = pack_circulant_giant_int(tmp_c, D, &w_u64, &layers[l].w1_sum);
        if (fread(tmp_c, 1, D, f) != D) { fclose(f); return 1; }
        layers[l].w2_big = pack_circulant_giant_int(tmp_c, D, &w_u64, &layers[l].w2_sum);
    }
    fclose(f); free(tmp_c);

    const char *prompt = "Clover is a way of working with ";
    uint8_t seq[256];
    size_t T = strlen(prompt);
    memcpy(seq, prompt, T);
    float *x = (float *)malloc(256 * D * sizeof(float));
    float *h = (float *)malloc(256 * D * sizeof(float));
    float *gate = (float *)malloc(256 * D * sizeof(float));
    float *conv = (float *)malloc(256 * D * sizeof(float));
    float *u_buf = (float *)malloc(256 * D * sizeof(float));
    float *y_buf = (float *)malloc(256 * D * sizeof(float));

    for (size_t t = 0; t < T; t++) memcpy(x + t * D, embed + seq[t] * D, D * sizeof(float));
    for (uint32_t l = 0; l < n_layers; l++) {
        rms_norm(h, x, layers[l].norm1, T, D);
        giant_int_circulant_forward(gate, h, T, D, layers[l].tg_big, w_u64, layers[l].tg_s);
        giant_int_causal_conv_p4(conv, h, T, D, K, layers[l].tw_u4, layers[l].tw_s);
        for (size_t i = 0; i < T * D; i++) x[i] += (1.0f / (1.0f + expf(-gate[i]))) * conv[i];
        rms_norm(h, x, layers[l].norm2, T, D);
        giant_int_circulant_forward(u_buf, h, T, D, layers[l].w1_big, w_u64, layers[l].w1_s);
        for (size_t t = 0; t < T; t++)
            for (size_t d = 0; d < D; d++) {
                float val = u_buf[t * D + d] * layers[l].ffn_diag[d];
                u_buf[t * D + d] = val / (1.0f + expf(-val));
            }
        giant_int_circulant_forward(y_buf, u_buf, T, D, layers[l].w2_big, w_u64, layers[l].w2_s);
        for (size_t i = 0; i < T * D; i++) x[i] += y_buf[i];
    }
    rms_norm(h, x + (T - 1) * D, final_norm, 1, D);
    float logits[128], ref_lg[128];
    for (uint32_t v = 0; v < V; v++) {
        double s = 0.0;
        for (size_t d = 0; d < D; d++) s += (double)h[d] * (double)embed[v * D + d];
        logits[v] = (float)s;
    }
    FILE *rf = fopen("/tmp/xc_step_c/ref_logits.bin", "rb");
    if (rf && fread(ref_lg, sizeof(float), 128, rf) == 128) {
        fclose(rf);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < 128; i++) { dot += (double)logits[i] * ref_lg[i]; na += (double)logits[i] * logits[i]; nb += (double)ref_lg[i] * ref_lg[i]; }
        printf("  24-Layer C11 x-cruncher vs PyTorch QAT Logit Cosine: %.6f\n", dot / (sqrt(na * nb) + 1e-12));
    }
    return 0;
}
