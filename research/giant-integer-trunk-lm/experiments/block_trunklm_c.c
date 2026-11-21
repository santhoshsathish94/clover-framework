/*
 * Pure-C11 24-Layer Block-Circulant + Salem-Spencer `Block-TrunkLM-24` Verifier (x-cruncher v1.6.1)
 *
 * Loads `block_trunklm24.bin` exported by `distill_omni_to_block_trunklm.py`, executes all 24 layers
 * (8x8 B=64 4-bit Block-Circulant projections + K=32 P=4 Salem-Spencer causal convolutions, including
 * optional Butterfly-Transpose cross-block routing) purely via `xc_mul_u64`, and verifies logit cosine
 * parity against PyTorch QAT on unseen `LFM3-Omni-2B` validation tokens.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "include/xcruncher.h"

typedef struct {
    float *c_s;           /* [nb * nb] per-block scale */
    uint64_t **w_big;     /* [nb * nb] pre-packed length-2B Circulant Giant Integers */
    int32_t *c_sum;       /* [nb * nb] signed weight sum per block */
} c_block_circ_proj_t;

typedef struct {
    float *norm1, *norm2, *diag_scale, *tw_s;
    uint8_t *tw_u4;       /* [D * K] */
    c_block_circ_proj_t gate_proj, w1, w2;
} c_block_layer_t;

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

static void load_block_proj(FILE *f, c_block_circ_proj_t *p, size_t nb, size_t B, size_t *out_wu64) {
    size_t n_blocks = nb * nb;
    size_t w_u64 = (2 * B * 4 + 7) / 8;
    *out_wu64 = w_u64;
    p->c_s = (float *)malloc(n_blocks * sizeof(float));
    p->w_big = (uint64_t **)malloc(n_blocks * sizeof(uint64_t *));
    p->c_sum = (int32_t *)malloc(n_blocks * sizeof(int32_t));
    uint8_t *buf = (uint8_t *)malloc(n_blocks * B);
    if (fread(p->c_s, sizeof(float), n_blocks, f) != n_blocks ||
        fread(buf, 1, n_blocks * B, f) != n_blocks * B) {
        fprintf(stderr, "Failed reading block projection\n");
        exit(1);
    }
    for (size_t bi = 0; bi < n_blocks; bi++) {
        const uint8_t *cb = buf + bi * B;
        p->w_big[bi] = (uint64_t *)calloc(w_u64, 8);
        uint32_t *ws = (uint32_t *)p->w_big[bi];
        int32_t s = 0;
        for (size_t k = 0; k < B; k++) s += (int32_t)cb[k] - 8;
        for (size_t m = 0; m < 2 * B; m++) ws[m] = (uint32_t)cb[(m + 1) % B];
        p->c_sum[bi] = s;
    }
    free(buf);
}

static void block_circulant_forward_xc(
    float *y_out, const float *x_in, size_t T, size_t D, size_t B,
    const c_block_circ_proj_t *proj, size_t w_u64
) {
    size_t nb = D / B;
    memset(y_out, 0, T * D * sizeof(float));
    size_t x_u64 = (T * 2 * B * 4 + 7) / 8;
    size_t p_u64 = x_u64 + w_u64;

    /* Pre-quantize each input block i in 0..nb-1 once into a Giant Integer X_blk[i] across all T tokens! */
    uint64_t **X_blk = (uint64_t **)malloc(nb * sizeof(uint64_t *));
    float *x_min = (float *)malloc(nb * T * sizeof(float));
    float *x_sc  = (float *)malloc(nb * T * sizeof(float));
    uint32_t *x_sum = (uint32_t *)malloc(nb * T * sizeof(uint32_t));

    for (size_t ib = 0; ib < nb; ib++) {
        X_blk[ib] = (uint64_t *)calloc(x_u64, 8);
        uint32_t *xs = (uint32_t *)X_blk[ib];
        for (size_t t = 0; t < T; t++) {
            const float *xt = x_in + t * D + ib * B;
            float mn = xt[0], mx = xt[0];
            for (size_t k = 1; k < B; k++) {
                if (xt[k] < mn) mn = xt[k];
                if (xt[k] > mx) mx = xt[k];
            }
            float sc = (mx - mn > 1e-7f) ? ((mx - mn) / 255.0f) : 1e-7f;
            x_min[ib * T + t] = mn;
            x_sc[ib * T + t] = sc;
            uint32_t su = 0;
            for (size_t k = 0; k < B; k++) {
                uint32_t q = (uint32_t)lrintf((xt[k] - mn) / sc);
                if (q > 255u) q = 255u;
                xs[t * 2 * B + k] = q;
                su += q;
            }
            x_sum[ib * T + t] = su;
        }
    }

    #pragma omp parallel
    {
        uint64_t *P_big = (uint64_t *)malloc(p_u64 * 8);
        #pragma omp for schedule(static)
        for (size_t ob = 0; ob < nb; ob++) {
            for (size_t ib = 0; ib < nb; ib++) {
                size_t bi = ob * nb + ib;
                memset(P_big, 0, p_u64 * 8);
                xc_mul_u64(P_big, X_blk[ib], x_u64, proj->w_big[bi], w_u64);
                const uint32_t *ps = (const uint32_t *)P_big;
                float ws = proj->c_s[bi];
                int32_t cs = proj->c_sum[bi];
                for (size_t t = 0; t < T; t++) {
                    float *yt = y_out + t * D + ob * B;
                    float sc_prod = ws * x_sc[ib * T + t];
                    float bias = ws * x_min[ib * T + t] * (float)cs;
                    int32_t z_off = 8 * (int32_t)x_sum[ib * T + t];
                    for (size_t r = 0; r < B; r++) {
                        yt[r] += sc_prod * (float)((int32_t)ps[t * 2 * B + (B - 1) + r] - z_off) + bias;
                    }
                }
            }
        }
        free(P_big);
    }

    for (size_t ib = 0; ib < nb; ib++) free(X_blk[ib]);
    free(X_blk); free(x_min); free(x_sc); free(x_sum);
}

static void causal_conv_salem_spencer_xc(
    float *y_out, const float *h_in, size_t T, size_t D, size_t K,
    const uint8_t *tw_u4, const float *tw_s
) {
    const size_t e4[4] = {0, 1, 3, 4}, STRIDE = 9;
    size_t w_u64 = (K * STRIDE * 4 + 7) / 8, x_u64 = (T * STRIDE * 4 + 7) / 8, p_u64 = w_u64 + x_u64;

    #pragma omp parallel
    {
        uint64_t *W_big = (uint64_t *)malloc(w_u64 * 8);
        uint64_t *X_big = (uint64_t *)malloc(x_u64 * 8);
        uint64_t *P_big = (uint64_t *)malloc(p_u64 * 8);

        #pragma omp for schedule(static)
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
                float ws = tw_s[d];
                for (size_t t = 0; t < T; t++) {
                    int32_t raw = (int32_t)pl[t * STRIDE + 2 * e4[p]];
                    size_t lim = (t + 1 < K) ? (t + 1) : K;
                    int32_t q_sum = 0, w_sum = 0;
                    for (size_t j = 0; j < lim; j++) {
                        q_sum += (int32_t)xl[(t - j) * STRIDE + e4[p]];
                        w_sum += (int32_t)tw_u4[d * K + j] - 8;
                    }
                    y_out[t * D + d] = ws * ch_sc[p] * (float)(raw - 8 * q_sum)
                                     + ws * ch_min[p] * (float)w_sum;
                }
            }
        }
        free(W_big); free(X_big); free(P_big);
    }
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "/tmp/xc_pilot_omni";
    char path[512];
    snprintf(path, sizeof(path), "%s/block_trunklm24.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("No checkpoint at %s (skipping live verify)\n", path); return 0; }

    uint32_t n_layers, D, B, K, V, use_bp;
    if (fread(&n_layers, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        fread(&B, 4, 1, f) != 1 || fread(&K, 4, 1, f) != 1 ||
        fread(&V, 4, 1, f) != 1 || fread(&use_bp, 4, 1, f) != 1) return 1;

    float *embed = (float *)malloc((size_t)V * D * sizeof(float));
    float *final_norm = (float *)malloc(D * sizeof(float));
    if (fread(embed, sizeof(float), (size_t)V * D, f) != (size_t)V * D ||
        fread(final_norm, sizeof(float), D, f) != D) return 1;

    size_t nb = D / B, w_u64 = 0;
    c_block_layer_t *layers = (c_block_layer_t *)calloc(n_layers, sizeof(c_block_layer_t));
    for (uint32_t l = 0; l < n_layers; l++) {
        layers[l].norm1 = (float *)malloc(D * sizeof(float));
        layers[l].norm2 = (float *)malloc(D * sizeof(float));
        layers[l].diag_scale = (float *)malloc(D * sizeof(float));
        layers[l].tw_s = (float *)malloc(D * sizeof(float));
        layers[l].tw_u4 = (uint8_t *)malloc((size_t)D * K);
        if (fread(layers[l].norm1, sizeof(float), D, f) != D ||
            fread(layers[l].norm2, sizeof(float), D, f) != D ||
            fread(layers[l].diag_scale, sizeof(float), D, f) != D ||
            fread(layers[l].tw_s, sizeof(float), D, f) != D ||
            fread(layers[l].tw_u4, 1, (size_t)D * K, f) != (size_t)D * K) return 1;
        load_block_proj(f, &layers[l].gate_proj, nb, B, &w_u64);
        load_block_proj(f, &layers[l].w1, nb, B, &w_u64);
        load_block_proj(f, &layers[l].w2, nb, B, &w_u64);
    }
    fclose(f);

    size_t T = 32;
    int32_t tok_seq[32];
    snprintf(path, sizeof(path), "%s/sample_tokens.bin", dir);
    FILE *tf = fopen(path, "rb");
    if (!tf || fread(tok_seq, 4, T, tf) != T) return 1;
    fclose(tf);

    float *x = (float *)malloc(T * D * sizeof(float));
    float *h = (float *)malloc(T * D * sizeof(float));
    float *gate = (float *)malloc(T * D * sizeof(float));
    float *conv = (float *)malloc(T * D * sizeof(float));
    float *u_buf = (float *)malloc(T * D * sizeof(float));
    float *u_perm = (float *)malloc(T * D * sizeof(float));
    float *y_buf = (float *)malloc(T * D * sizeof(float));

    double t0 = omp_get_wtime();
    for (size_t t = 0; t < T; t++) memcpy(x + t * D, embed + (size_t)tok_seq[t] * D, D * sizeof(float));
    for (uint32_t l = 0; l < n_layers; l++) {
        rms_norm(h, x, layers[l].norm1, T, D);
        block_circulant_forward_xc(gate, h, T, D, B, &layers[l].gate_proj, w_u64);
        causal_conv_salem_spencer_xc(conv, h, T, D, K, layers[l].tw_u4, layers[l].tw_s);
        for (size_t i = 0; i < T * D; i++) x[i] += (1.0f / (1.0f + expf(-gate[i]))) * conv[i];

        rms_norm(h, x, layers[l].norm2, T, D);
        block_circulant_forward_xc(u_buf, h, T, D, B, &layers[l].w1, w_u64);
        for (size_t t = 0; t < T; t++)
            for (size_t d = 0; d < D; d++) {
                float val = u_buf[t * D + d] * layers[l].diag_scale[d];
                u_buf[t * D + d] = val / (1.0f + expf(-val));
            }
        const float *w2_in = u_buf;
        if (use_bp && D == 512) {
            for (size_t t = 0; t < T; t++)
                for (size_t a = 0; a < 8; a++)
                    for (size_t b_idx = 0; b_idx < 8; b_idx++)
                        for (size_t c = 0; c < 8; c++)
                            u_perm[t * 512 + (b_idx * 64 + a * 8 + c)] = u_buf[t * 512 + (a * 64 + b_idx * 8 + c)];
            w2_in = u_perm;
        }
        block_circulant_forward_xc(y_buf, w2_in, T, D, B, &layers[l].w2, w_u64);
        for (size_t i = 0; i < T * D; i++) x[i] += y_buf[i];
    }
    double ms = (omp_get_wtime() - t0) * 1000.0;

    rms_norm(h, x + (T - 1) * D, final_norm, 1, D);
    float *logits = (float *)malloc(V * sizeof(float));
    float *ref_lg = (float *)malloc(V * sizeof(float));
    for (uint32_t v = 0; v < V; v++) {
        double s = 0.0;
        for (size_t d = 0; d < D; d++) s += (double)h[d] * (double)embed[(size_t)v * D + d];
        logits[v] = (float)s;
    }

    snprintf(path, sizeof(path), "%s/ref_val_logits.bin", dir);
    FILE *rf = fopen(path, "rb");
    if (rf && fread(ref_lg, sizeof(float), V, rf) == V) {
        fclose(rf);
        double dot = 0, na = 0, nb_n = 0;
        int top_c = 0, top_py = 0;
        for (uint32_t v = 0; v < V; v++) {
            dot += (double)logits[v] * ref_lg[v];
            na += (double)logits[v] * logits[v];
            nb_n += (double)ref_lg[v] * ref_lg[v];
            if (logits[v] > logits[top_c]) top_c = (int)v;
            if (ref_lg[v] > ref_lg[top_py]) top_py = (int)v;
        }
        printf("=== Pure-C11 24-Layer Block-TrunkLM-24 Verification (x-cruncher) ===\n");
        printf("  24-Layer Forward Time (T=%zu, D=%u, B=%u): %.2f ms\n", T, D, B, ms);
        printf("  Logit Cosine vs PyTorch Hard 4-Bit QAT:    %.6f\n", dot / (sqrt(na * nb_n) + 1e-12));
        printf("  Top-1 Token ID Parity (C11 vs PyTorch):    %d vs %d (%s)\n",
               top_c, top_py, (top_c == top_py) ? "EXACT MATCH!" : "DIFF");
    }
    return 0;
}
