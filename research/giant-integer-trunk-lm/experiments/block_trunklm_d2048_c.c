/*
 * Full-Scale Pure-C11 24-Layer `Block-TrunkLM-24` (`D = 2048, B = 64, K = 64`) Verifier & Engine
 * (`TRKD` nibble-packed 3.75 MiB 4-bit Giant-Integer Trunk format, x-cruncher v1.6.1)
 *
 * Executes ALL 24 layers purely via `xc_mul_u64` Giant-Integer multiplications:
 *   - 32x32 grid of B=64 4-bit Circulant Giant Integers + O(D) in/out gain vectors
 *   - [32, 32, 2] Butterfly-Transpose cross-block routing between w1 and w2
 *   - K=64 Salem-Spencer P=4 ({0, 1, 3, 4}) Multi-Channel Causal Time-Trunks
 *   - Multi-position logit cosine + Top-1 token parity check against PyTorch Hard 4-Bit QAT
 *   - Live autoregressive token generation directly from Pure-C11 Giant-Integer layers
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "include/xcruncher.h"

typedef struct {
    float *in_gain;       /* [D] */
    float *out_gain;      /* [D] */
    float *c_s;           /* [nb * nb] */
    uint64_t **w_big;     /* [nb * nb] length-2B Circulant Giant Integers */
    int32_t *c_sum;       /* [nb * nb] */
} trkd_proj_t;

typedef struct {
    float *norm1, *norm2, *diag_scale, *tw_s;
    uint8_t *tw_u4;       /* [D * K] unpacked from nibbles [0..15] */
    trkd_proj_t gate_proj, w1, w2;
} trkd_layer_t;

static void unpack_nibbles(uint8_t *dst, const uint8_t *src, size_t n_vals) {
    for (size_t i = 0; i < n_vals / 2; i++) {
        dst[2 * i + 0] = (uint8_t)(src[i] & 0x0Fu);
        dst[2 * i + 1] = (uint8_t)((src[i] >> 4) & 0x0Fu);
    }
}

static void rms_norm_eps(float *out, const float *x, const float *w, size_t T, size_t D, float eps) {
    for (size_t t = 0; t < T; t++) {
        const float *xt = x + t * D;
        float *ot = out + t * D;
        double ss = 0.0;
        for (size_t d = 0; d < D; d++) ss += (double)xt[d] * (double)xt[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)D) + eps);
        for (size_t d = 0; d < D; d++) ot[d] = xt[d] * inv * w[d];
    }
}

static int load_trkd_proj(FILE *f, trkd_proj_t *p, size_t D, size_t nb, size_t B, size_t *out_wu64) {
    size_t n_blocks = nb * nb;
    size_t w_u64 = (2 * B * 4 + 7) / 8;
    *out_wu64 = w_u64;
    p->in_gain  = (float *)malloc(D * sizeof(float));
    p->out_gain = (float *)malloc(D * sizeof(float));
    p->c_s      = (float *)malloc(n_blocks * sizeof(float));
    p->w_big    = (uint64_t **)calloc(n_blocks, sizeof(uint64_t *));
    p->c_sum    = (int32_t *)malloc(n_blocks * sizeof(int32_t));
    size_t packed_bytes = (n_blocks * B) / 2;
    uint8_t *packed = (uint8_t *)malloc(packed_bytes);
    uint8_t *unpacked = (uint8_t *)malloc(n_blocks * B);

    if (fread(p->in_gain, sizeof(float), D, f) != D ||
        fread(p->out_gain, sizeof(float), D, f) != D ||
        fread(p->c_s, sizeof(float), n_blocks, f) != n_blocks ||
        fread(packed, 1, packed_bytes, f) != packed_bytes) {
        free(packed); free(unpacked);
        return -1;
    }
    unpack_nibbles(unpacked, packed, n_blocks * B);
    for (size_t bi = 0; bi < n_blocks; bi++) {
        const uint8_t *cb = unpacked + bi * B;
        p->w_big[bi] = (uint64_t *)calloc(w_u64, 8);
        uint32_t *ws = (uint32_t *)p->w_big[bi];
        int32_t s = 0;
        for (size_t k = 0; k < B; k++) s += (int32_t)cb[k] - 8;
        for (size_t m = 0; m < 2 * B; m++) ws[m] = (uint32_t)cb[(m + 1) % B];
        p->c_sum[bi] = s;
    }
    free(packed);
    free(unpacked);
    return 0;
}

static void trkd_proj_forward_xc(
    float *y_out, const float *x_in, size_t T, size_t D, size_t B,
    const trkd_proj_t *proj, size_t w_u64
) {
    size_t nb = D / B;
    memset(y_out, 0, T * D * sizeof(float));
    size_t x_u64 = (T * 2 * B * 4 + 7) / 8;
    size_t p_u64 = x_u64 + w_u64;

    float *x_scaled = (float *)malloc(T * D * sizeof(float));
    for (size_t t = 0; t < T; t++)
        for (size_t d = 0; d < D; d++)
            x_scaled[t * D + d] = x_in[t * D + d] * proj->in_gain[d];

    uint64_t **X_blk = (uint64_t **)malloc(nb * sizeof(uint64_t *));
    float *x_min = (float *)malloc(nb * T * sizeof(float));
    float *x_sc  = (float *)malloc(nb * T * sizeof(float));
    uint32_t *x_sum = (uint32_t *)malloc(nb * T * sizeof(uint32_t));

    for (size_t ib = 0; ib < nb; ib++) {
        X_blk[ib] = (uint64_t *)calloc(x_u64, 8);
        uint32_t *xs = (uint32_t *)X_blk[ib];
        for (size_t t = 0; t < T; t++) {
            const float *xt = x_scaled + t * D + ib * B;
            float mn = xt[0], mx = xt[0];
            for (size_t k = 1; k < B; k++) {
                if (xt[k] < mn) mn = xt[k];
                if (xt[k] > mx) mx = xt[k];
            }
            /* 16-bit activation quantization [0..65535] fits in 32-bit slots (15*65535*64 = 6.29e7 < 2^32!) for ultra-high precision */
            float sc = (mx - mn > 1e-8f) ? ((mx - mn) / 65535.0f) : 1e-8f;
            x_min[ib * T + t] = mn;
            x_sc[ib * T + t] = sc;
            uint32_t su = 0;
            for (size_t k = 0; k < B; k++) {
                uint32_t q = (uint32_t)lrintf((xt[k] - mn) / sc);
                if (q > 65535u) q = 65535u;
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
                    int64_t z_off = 8LL * (int64_t)x_sum[ib * T + t];
                    for (size_t r = 0; r < B; r++) {
                        int64_t raw = (int64_t)ps[t * 2 * B + (B - 1) + r];
                        yt[r] += sc_prod * (float)(raw - z_off) + bias;
                    }
                }
            }
        }
        free(P_big);
    }

    for (size_t t = 0; t < T; t++)
        for (size_t d = 0; d < D; d++)
            y_out[t * D + d] *= proj->out_gain[d];

    for (size_t ib = 0; ib < nb; ib++) free(X_blk[ib]);
    free(x_scaled); free(X_blk); free(x_min); free(x_sc); free(x_sum);
}

static void trkd_causal_conv_p4_xc(
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
                float sc = (mx - mn > 1e-8f) ? ((mx - mn) / 4095.0f) : 1e-8f;
                ch_min[p] = mn; ch_sc[p] = sc;
                for (size_t t = 0; t < T; t++) {
                    uint32_t q = (uint32_t)lrintf((h_in[t * D + d] - mn) / sc);
                    if (q > 4095u) q = 4095u;
                    xl[t * STRIDE + e4[p]] = q;
                }
            }
            xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);
            const uint32_t *pl = (const uint32_t *)P_big;
            for (size_t p = 0; p < 4; p++) {
                size_t d = g * 4 + p;
                float ws = tw_s[d];
                for (size_t t = 0; t < T; t++) {
                    int64_t raw = (int64_t)pl[t * STRIDE + 2 * e4[p]];
                    size_t lim = (t + 1 < K) ? (t + 1) : K;
                    int64_t q_sum = 0, w_sum = 0;
                    for (size_t j = 0; j < lim; j++) {
                        q_sum += (int64_t)xl[(t - j) * STRIDE + e4[p]];
                        w_sum += (int64_t)tw_u4[d * K + j] - 8;
                    }
                    y_out[t * D + d] = ws * ch_sc[p] * (float)(raw - 8LL * q_sum)
                                     + ws * ch_min[p] * (float)w_sum;
                }
            }
        }
        free(W_big); free(X_big); free(P_big);
    }
}

static void run_24_layers_xc(
    float *x, size_t T, size_t D, size_t B, size_t K, uint32_t n_layers, uint32_t use_bp, float eps,
    const trkd_layer_t *layers, size_t w_u64
) {
    size_t nb = D / B;
    size_t sub = D / (nb * nb);
    float *h = (float *)malloc(T * D * sizeof(float));
    float *gate = (float *)malloc(T * D * sizeof(float));
    float *conv = (float *)malloc(T * D * sizeof(float));
    float *u_buf = (float *)malloc(T * D * sizeof(float));
    float *u_perm = (float *)malloc(T * D * sizeof(float));
    float *y_buf = (float *)malloc(T * D * sizeof(float));

    for (uint32_t l = 0; l < n_layers; l++) {
        rms_norm_eps(h, x, layers[l].norm1, T, D, eps);
        trkd_proj_forward_xc(gate, h, T, D, B, &layers[l].gate_proj, w_u64);
        trkd_causal_conv_p4_xc(conv, h, T, D, K, layers[l].tw_u4, layers[l].tw_s);
        for (size_t i = 0; i < T * D; i++) {
            float sig = 1.0f / (1.0f + expf(-gate[i]));
            x[i] += sig * conv[i];
        }

        rms_norm_eps(h, x, layers[l].norm2, T, D, eps);
        trkd_proj_forward_xc(u_buf, h, T, D, B, &layers[l].w1, w_u64);
        for (size_t t = 0; t < T; t++)
            for (size_t d = 0; d < D; d++) {
                float val = u_buf[t * D + d] * layers[l].diag_scale[d];
                u_buf[t * D + d] = val / (1.0f + expf(-val));
            }
        const float *w2_in = u_buf;
        if (use_bp && sub > 0) {
            for (size_t t = 0; t < T; t++)
                for (size_t a = 0; a < nb; a++)
                    for (size_t b_i = 0; b_i < nb; b_i++)
                        for (size_t c = 0; c < sub; c++)
                            u_perm[t * D + ((b_i * nb + a) * sub + c)] = u_buf[t * D + ((a * nb + b_i) * sub + c)];
            w2_in = u_perm;
        }
        trkd_proj_forward_xc(y_buf, w2_in, T, D, B, &layers[l].w2, w_u64);
        for (size_t i = 0; i < T * D; i++) x[i] += y_buf[i];
    }

    free(h); free(gate); free(conv); free(u_buf); free(u_perm); free(y_buf);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <model.trkd> <embed_bf16.bin> <eval_bundle.bin> [gen_steps]\n", argv[0]);
        return 0;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open trkd"); return 1; }
    char magic[4];
    uint32_t n_layers, D, B, K, V, use_bp;
    float eps;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TRKD", 4) != 0 ||
        fread(&n_layers, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        fread(&B, 4, 1, f) != 1 || fread(&K, 4, 1, f) != 1 ||
        fread(&V, 4, 1, f) != 1 || fread(&use_bp, 4, 1, f) != 1 ||
        fread(&eps, 4, 1, f) != 1) { fclose(f); return 1; }

    float *final_norm = (float *)malloc(D * sizeof(float));
    if (fread(final_norm, sizeof(float), D, f) != D) { fclose(f); return 1; }

    size_t nb = D / B, w_u64 = 0;
    trkd_layer_t *layers = (trkd_layer_t *)calloc(n_layers, sizeof(trkd_layer_t));
    size_t packed_tw_len = ((size_t)D * K) / 2;
    uint8_t *packed_tw = (uint8_t *)malloc(packed_tw_len);
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
            fread(packed_tw, 1, packed_tw_len, f) != packed_tw_len) return 1;
        unpack_nibbles(layers[l].tw_u4, packed_tw, (size_t)D * K);
        if (load_trkd_proj(f, &layers[l].gate_proj, D, nb, B, &w_u64) != 0 ||
            load_trkd_proj(f, &layers[l].w1, D, nb, B, &w_u64) != 0 ||
            load_trkd_proj(f, &layers[l].w2, D, nb, B, &w_u64) != 0) return 1;
    }
    free(packed_tw);
    fclose(f);

    /* Load BF16 embeddings [V, D] */
    FILE *ef = fopen(argv[2], "rb");
    if (!ef) { perror("open embed_bf16"); return 1; }
    size_t emb_elems = (size_t)V * D;
    uint16_t *emb_bf16 = (uint16_t *)malloc(emb_elems * 2);
    if (fread(emb_bf16, 2, emb_elems, ef) != emb_elems) return 1;
    fclose(ef);

    /* Load eval bundle: uint32 T, int32 tokens[T], float ref_hidden[T * D], int32 ref_top1[T] */
    FILE *vf = fopen(argv[3], "rb");
    if (!vf) { perror("open eval_bundle"); return 1; }
    uint32_t T_eval = 0;
    if (fread(&T_eval, 4, 1, vf) != 1) return 1;
    int32_t *toks = (int32_t *)malloc(T_eval * 4);
    float *ref_h  = (float *)malloc((size_t)T_eval * D * sizeof(float));
    int32_t *ref_top1 = (int32_t *)malloc(T_eval * 4);
    if (fread(toks, 4, T_eval, vf) != T_eval ||
        fread(ref_h, sizeof(float), (size_t)T_eval * D, vf) != (size_t)T_eval * D ||
        fread(ref_top1, 4, T_eval, vf) != T_eval) return 1;
    fclose(vf);

    float *x = (float *)malloc((size_t)(T_eval + 64) * D * sizeof(float));
    float *h_out = (float *)malloc((size_t)T_eval * D * sizeof(float));
    for (size_t t = 0; t < T_eval; t++) {
        const uint16_t *erow = emb_bf16 + (size_t)toks[t] * D;
        for (size_t d = 0; d < D; d++) {
            uint32_t u = ((uint32_t)erow[d]) << 16;
            float fv; memcpy(&fv, &u, 4);
            x[t * D + d] = fv;
        }
    }

    double t0 = omp_get_wtime();
    run_24_layers_xc(x, T_eval, D, B, K, n_layers, use_bp, eps, layers, w_u64);
    rms_norm_eps(h_out, x, final_norm, T_eval, D, eps);
    double ms = (omp_get_wtime() - t0) * 1000.0;

    /* Check cosine of final normed hidden states AND exact 128,000-vocab Top-1 token IDs across ALL T_eval positions! */
    double min_cos = 1.0, sum_cos = 0.0;
    uint32_t top1_matches = 0;
    for (size_t t = 0; t < T_eval; t++) {
        double dot = 0, na = 0, nb_n = 0;
        for (size_t d = 0; d < D; d++) {
            double a = h_out[t * D + d], b = ref_h[t * D + d];
            dot += a * b; na += a * a; nb_n += b * b;
        }
        double c = dot / (sqrt(na * nb_n) + 1e-12);
        if (c < min_cos) min_cos = c;
        sum_cos += c;

        int best_v = 0;
        float best_l = -1e30f;
        const float *ht = h_out + t * D;
        #pragma omp parallel
        {
            int loc_v = 0; float loc_l = -1e30f;
            #pragma omp for nowait schedule(static)
            for (uint32_t v = 0; v < V; v++) {
                const uint16_t *er = emb_bf16 + (size_t)v * D;
                float s = 0.0f;
                for (size_t d = 0; d < D; d++) {
                    uint32_t u = ((uint32_t)er[d]) << 16;
                    float fv; memcpy(&fv, &u, 4);
                    s += ht[d] * fv;
                }
                if (s > loc_l) { loc_l = s; loc_v = (int)v; }
            }
            #pragma omp critical
            {
                if (loc_l > best_l) { best_l = loc_l; best_v = loc_v; }
            }
        }
        if (best_v == ref_top1[t]) top1_matches++;
    }

    printf("=== Pure-C11 Full-Scale D=%u Block-TrunkLM-24 Verification (x-cruncher %s) ===\n", D, xc_active_isa_name());
    printf("  24-Layer Giant-Int Forward (%u tokens): %.2f ms\n", T_eval, ms);
    printf("  Final-State Cosine vs PyTorch (mean / min): %.6f / %.6f\n", sum_cos / T_eval, min_cos);
    printf("  128,000-Vocab Top-1 Token Parity:           %u / %u (%.1f%%)\n",
           top1_matches, T_eval, 100.0 * top1_matches / T_eval);

    /* Optional greedy token generation directly in Pure-C11 x-cruncher! */
    int gen_steps = (argc > 4) ? atoi(argv[4]) : 0;
    if (gen_steps > 0 && gen_steps <= 64) {
        int32_t *g_toks = (int32_t *)malloc((T_eval + gen_steps) * 4);
        memcpy(g_toks, toks, T_eval * 4);
        float *hn = (float *)malloc(D * sizeof(float));
        printf("  Generated Token IDs (%d steps):", gen_steps);
        for (int gs = 0; gs < gen_steps; gs++) {
            size_t cur_T = T_eval + gs;
            for (size_t t = 0; t < cur_T; t++) {
                const uint16_t *erow = emb_bf16 + (size_t)g_toks[t] * D;
                for (size_t d = 0; d < D; d++) {
                    uint32_t u = ((uint32_t)erow[d]) << 16;
                    float fv; memcpy(&fv, &u, 4);
                    x[t * D + d] = fv;
                }
            }
            run_24_layers_xc(x, cur_T, D, B, K, n_layers, use_bp, eps, layers, w_u64);
            rms_norm_eps(hn, x + (cur_T - 1) * D, final_norm, 1, D, eps);
            int best_v = 0; float best_l = -1e30f;
            #pragma omp parallel
            {
                int lv = 0; float ll = -1e30f;
                #pragma omp for nowait schedule(static)
                for (uint32_t v = 0; v < V; v++) {
                    const uint16_t *er = emb_bf16 + (size_t)v * D;
                    float s = 0.0f;
                    for (size_t d = 0; d < D; d++) {
                        uint32_t u = ((uint32_t)er[d]) << 16;
                        float fv; memcpy(&fv, &u, 4);
                        s += hn[d] * fv;
                    }
                    if (s > ll) { ll = s; lv = (int)v; }
                }
                #pragma omp critical
                { if (ll > best_l) { best_l = ll; best_v = lv; } }
            }
            g_toks[cur_T] = best_v;
            printf(" %d", best_v);
            fflush(stdout);
        }
        printf("\n");
        FILE *gf = fopen("c11_generated_ids.bin", "wb");
        if (gf) { fwrite(g_toks, 4, T_eval + gen_steps, gf); fclose(gf); }
        free(g_toks); free(hn);
    }

    free(final_norm); free(emb_bf16); free(toks); free(ref_h); free(ref_top1); free(x); free(h_out);
    return 0;
}
