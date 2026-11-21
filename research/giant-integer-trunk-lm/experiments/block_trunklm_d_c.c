/*
 * Pure-C11 24-Layer `Block-TrunkLM-24` Variant D Verifier (x-cruncher v1.6.1)
 *
 * Variant D = Block-Circulant 4-bit trunks with per-projection fixed index permutations and O(D)
 * diagonal input/output gains (y = d_out * BlockCirc(d_in * x[perm])), K=32 4-bit causal taps packed
 * with Base-3 Salem-Spencer lanes {0,1,3,4}. Every 4-bit trunk product executes via `xc_mul_u64`.
 *
 * Verifies every position of every exported validation sequence against PyTorch hard-4-bit logits
 * (raw cosine, centred cosine, top-1 parity).
 *
 * Build: gcc -O3 -march=native -std=c11 -fopenmp block_trunklm_d_c.c xcruncher.c -I <xc dir> -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "include/xcruncher.h"

typedef struct {
    uint16_t *perm;       /* [D] input gather index */
    float *d_in, *d_out;  /* [D] diagonal gains */
    float *c_s;           /* [nb * nb] per-block weight scale */
    uint64_t **w_big;     /* [nb * nb] length-2B circulant Giant Integers (32-bit slots) */
    int32_t *c_sum;       /* [nb * nb] signed weight sum per block */
} proj_t;

typedef struct {
    float *norm1, *norm2, *diag_scale, *tw_s;
    uint8_t *tw_u4;       /* [D * K] */
    proj_t gate_proj, w1, w2;
} layer_t;

static size_t g_nb;

static int rd(void *dst, size_t sz, size_t n, FILE *f) { return fread(dst, sz, n, f) == n; }

static void rms_norm(float *out, const float *x, const float *w, size_t T, size_t D, float eps) {
    for (size_t t = 0; t < T; t++) {
        const float *xt = x + t * D;
        float *ot = out + t * D;
        double ss = 0.0;
        for (size_t d = 0; d < D; d++) ss += (double)xt[d] * (double)xt[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)D) + eps);
        for (size_t d = 0; d < D; d++) ot[d] = xt[d] * inv * w[d];
    }
}

static int load_proj(FILE *f, proj_t *p, size_t D, size_t B) {
    size_t nb = D / B, n_blocks = nb * nb, w_u64 = (2 * B * 4 + 7) / 8;
    p->perm = malloc(D * sizeof(uint16_t));
    p->d_in = malloc(D * sizeof(float));
    p->d_out = malloc(D * sizeof(float));
    p->c_s = malloc(n_blocks * sizeof(float));
    p->c_sum = malloc(n_blocks * sizeof(int32_t));
    p->w_big = calloc(n_blocks, sizeof(uint64_t *));
    uint8_t *buf = malloc(n_blocks * B);
    if (!p->perm || !p->d_in || !p->d_out || !p->c_s || !p->c_sum || !p->w_big || !buf) { free(buf); return 0; }
    if (!rd(p->perm, sizeof(uint16_t), D, f) || !rd(p->d_in, sizeof(float), D, f) || !rd(p->d_out, sizeof(float), D, f) ||
        !rd(p->c_s, sizeof(float), n_blocks, f) || !rd(buf, 1, n_blocks * B, f)) { free(buf); return 0; }
    for (size_t i = 0; i < D; i++) if (p->perm[i] >= D) { free(buf); return 0; }
    for (size_t bi = 0; bi < n_blocks; bi++) {
        const uint8_t *cb = buf + bi * B;
        p->w_big[bi] = calloc(w_u64, 8);
        if (!p->w_big[bi]) { free(buf); return 0; }
        uint32_t *ws = (uint32_t *)p->w_big[bi];
        int32_t s = 0;
        for (size_t k = 0; k < B; k++) { if (cb[k] > 15) { free(buf); return 0; } s += (int32_t)cb[k] - 8; }
        /* slot m holds c[(m+1) % B]; output slot (B-1)+r then equals sum_k x[k] * c[(r-k) mod B]. */
        for (size_t m = 0; m < 2 * B; m++) ws[m] = (uint32_t)cb[(m + 1) % B];
        p->c_sum[bi] = s;
    }
    free(buf);
    return 1;
}

static void free_proj(proj_t *p) {
    if (p->w_big) for (size_t i = 0; i < g_nb * g_nb; i++) free(p->w_big[i]);
    free(p->w_big); free(p->perm); free(p->d_in); free(p->d_out); free(p->c_s); free(p->c_sum);
}

/* y = BlockCirc(x) with 8-bit per-token/per-block asymmetric activations and 4-bit weights, exact via xc_mul_u64. */
static void block_circ_xc(float *y_out, const float *x_in, size_t T, size_t D, size_t B, const proj_t *proj) {
    size_t nb = D / B, w_u64 = (2 * B * 4 + 7) / 8, x_u64 = (T * 2 * B * 4 + 7) / 8, p_u64 = x_u64 + w_u64;
    memset(y_out, 0, T * D * sizeof(float));
    uint64_t **X_blk = malloc(nb * sizeof(uint64_t *));
    float *x_min = malloc(nb * T * sizeof(float)), *x_sc = malloc(nb * T * sizeof(float));
    uint32_t *x_sum = malloc(nb * T * sizeof(uint32_t));
    for (size_t ib = 0; ib < nb; ib++) {
        X_blk[ib] = calloc(x_u64, 8);
        uint32_t *xs = (uint32_t *)X_blk[ib];
        for (size_t t = 0; t < T; t++) {
            const float *xt = x_in + t * D + ib * B;
            float mn = xt[0], mx = xt[0];
            for (size_t k = 1; k < B; k++) { if (xt[k] < mn) mn = xt[k]; if (xt[k] > mx) mx = xt[k]; }
            float sc = (mx - mn > 1e-7f) ? ((mx - mn) / 255.0f) : 1e-7f;
            x_min[ib * T + t] = mn; x_sc[ib * T + t] = sc;
            uint32_t su = 0;
            for (size_t k = 0; k < B; k++) {
                long qi = lrintf((xt[k] - mn) / sc);
                uint32_t q = (qi < 0) ? 0u : (qi > 255 ? 255u : (uint32_t)qi);
                xs[t * 2 * B + k] = q;
                su += q;
            }
            x_sum[ib * T + t] = su;
        }
    }
    #pragma omp parallel
    {
        uint64_t *P_big = malloc(p_u64 * 8);
        #pragma omp for schedule(static)
        for (size_t ob = 0; ob < nb; ob++) {
            for (size_t ib = 0; ib < nb; ib++) {
                size_t bi = ob * nb + ib;
                xc_mul_u64(P_big, X_blk[ib], x_u64, proj->w_big[bi], w_u64);
                const uint32_t *ps = (const uint32_t *)P_big;
                float ws = proj->c_s[bi];
                int32_t cs = proj->c_sum[bi];
                for (size_t t = 0; t < T; t++) {
                    float *yt = y_out + t * D + ob * B;
                    float sc_prod = ws * x_sc[ib * T + t], bias = ws * x_min[ib * T + t] * (float)cs;
                    int32_t z_off = 8 * (int32_t)x_sum[ib * T + t];
                    for (size_t r = 0; r < B; r++)
                        yt[r] += sc_prod * (float)((int32_t)ps[t * 2 * B + (B - 1) + r] - z_off) + bias;
                }
            }
        }
        free(P_big);
    }
    for (size_t ib = 0; ib < nb; ib++) free(X_blk[ib]);
    free(X_blk); free(x_min); free(x_sc); free(x_sum);
}

/* y = d_out * BlockCirc(d_in * x[perm]) */
static void proj_forward(float *y, const float *x, float *tmp, size_t T, size_t D, size_t B, const proj_t *p) {
    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < D; k++) tmp[t * D + k] = x[t * D + p->perm[k]] * p->d_in[k];
    block_circ_xc(y, tmp, T, D, B, p);
    for (size_t t = 0; t < T; t++)
        for (size_t k = 0; k < D; k++) y[t * D + k] *= p->d_out[k];
}

static void causal_conv_ss_xc(float *y_out, const float *h_in, size_t T, size_t D, size_t K,
                              const uint8_t *tw_u4, const float *tw_s) {
    const size_t e4[4] = {0, 1, 3, 4}, STRIDE = 9;
    size_t w_u64 = (K * STRIDE * 4 + 7) / 8, x_u64 = (T * STRIDE * 4 + 7) / 8, p_u64 = w_u64 + x_u64;
    #pragma omp parallel
    {
        uint64_t *W_big = malloc(w_u64 * 8), *X_big = malloc(x_u64 * 8), *P_big = malloc(p_u64 * 8);
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
                for (size_t t = 1; t < T; t++) { float v = h_in[t * D + d]; if (v < mn) mn = v; if (v > mx) mx = v; }
                float sc = (mx - mn > 1e-7f) ? ((mx - mn) / 255.0f) : 1e-7f;
                ch_min[p] = mn; ch_sc[p] = sc;
                for (size_t t = 0; t < T; t++) {
                    long qi = lrintf((h_in[t * D + d] - mn) / sc);
                    xl[t * STRIDE + e4[p]] = (qi < 0) ? 0u : (qi > 255 ? 255u : (uint32_t)qi);
                }
            }
            xc_mul_u64(P_big, X_big, x_u64, W_big, w_u64);
            const uint32_t *pl = (const uint32_t *)P_big;
            for (size_t p = 0; p < 4; p++) {
                size_t d = g * 4 + p;
                float ws = tw_s[d];
                int32_t q_sum = 0, w_sum = 0;
                for (size_t t = 0; t < T; t++) {
                    /* running window sums over the causal taps j = 0..min(t, K-1) */
                    q_sum += (int32_t)xl[t * STRIDE + e4[p]];
                    if (t >= K) q_sum -= (int32_t)xl[(t - K) * STRIDE + e4[p]];
                    if (t < K) w_sum += (int32_t)tw_u4[d * K + t] - 8;
                    int32_t raw = (int32_t)pl[t * STRIDE + 2 * e4[p]];
                    y_out[t * D + d] = ws * ch_sc[p] * (float)(raw - 8 * q_sum) + ws * ch_min[p] * (float)w_sum;
                }
            }
        }
        free(W_big); free(X_big); free(P_big);
    }
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[1024];
    int rc = 1;
    snprintf(path, sizeof(path), "%s/block_trunklm24_d.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "No checkpoint at %s\n", path); return 1; }
    char magic[4];
    uint32_t hdr[5];
    float eps;
    if (!rd(magic, 1, 4, f) || memcmp(magic, "BTD1", 4) != 0 || !rd(hdr, 4, 5, f) || !rd(&eps, 4, 1, f)) {
        fprintf(stderr, "Bad header\n"); fclose(f); return 1;
    }
    size_t L = hdr[0], D = hdr[1], B = hdr[2], K = hdr[3], V = hdr[4];
    if (L == 0 || L > 1024 || B == 0 || D % B || D % 4 || D > 65535 || K == 0 || K > 4096 || V == 0) {
        fprintf(stderr, "Unsupported shape L=%zu D=%zu B=%zu K=%zu V=%zu\n", L, D, B, K, V); fclose(f); return 1;
    }
    g_nb = D / B;
    float *embed = malloc(V * D * sizeof(float)), *final_norm = malloc(D * sizeof(float)), *h_bias = malloc(D * sizeof(float));
    layer_t *layers = calloc(L, sizeof(layer_t));
    int ok = embed && final_norm && h_bias && layers && rd(embed, sizeof(float), V * D, f) &&
             rd(final_norm, sizeof(float), D, f) && rd(h_bias, sizeof(float), D, f);
    for (size_t l = 0; ok && l < L; l++) {
        layer_t *ly = &layers[l];
        ly->norm1 = malloc(D * 4); ly->norm2 = malloc(D * 4); ly->diag_scale = malloc(D * 4); ly->tw_s = malloc(D * 4);
        ly->tw_u4 = malloc(D * K);
        ok = ly->norm1 && ly->norm2 && ly->diag_scale && ly->tw_s && ly->tw_u4 &&
             rd(ly->norm1, 4, D, f) && rd(ly->norm2, 4, D, f) && rd(ly->diag_scale, 4, D, f) &&
             rd(ly->tw_s, 4, D, f) && rd(ly->tw_u4, 1, D * K, f) &&
             load_proj(f, &ly->gate_proj, D, B) && load_proj(f, &ly->w1, D, B) && load_proj(f, &ly->w2, D, B);
    }
    fclose(f);
    uint32_t n_seq = 0, T = 0;
    int32_t *toks = NULL;
    float *ref = NULL, *x = NULL, *h = NULL, *g = NULL, *cv = NULL, *u = NULL, *y = NULL, *tmp = NULL, *lg = NULL;
    if (!ok) { fprintf(stderr, "Failed reading checkpoint\n"); goto done; }

    snprintf(path, sizeof(path), "%s/sample_d.bin", dir);
    FILE *sf = fopen(path, "rb");
    if (!sf) { fprintf(stderr, "No sample file\n"); goto done; }
    ok = rd(&n_seq, 4, 1, sf) && rd(&T, 4, 1, sf) && n_seq > 0 && n_seq < 1024 && T > 0 && T < 65536;
    if (ok) {
        toks = malloc((size_t)n_seq * T * sizeof(int32_t));
        ref = malloc((size_t)n_seq * T * V * sizeof(float));
        ok = toks && ref && rd(toks, 4, (size_t)n_seq * T, sf) && rd(ref, 4, (size_t)n_seq * T * V, sf);
    }
    fclose(sf);
    if (!ok) { fprintf(stderr, "Bad sample file\n"); goto done; }
    for (size_t i = 0; i < (size_t)n_seq * T; i++) if (toks[i] < 0 || (size_t)toks[i] >= V) { fprintf(stderr, "Token OOB\n"); goto done; }

    size_t TD = (size_t)T * D;
    x = malloc(TD * 4); h = malloc(TD * 4); g = malloc(TD * 4); cv = malloc(TD * 4);
    u = malloc(TD * 4); y = malloc(TD * 4); tmp = malloc(TD * 4); lg = malloc(V * 4);
    if (!x || !h || !g || !cv || !u || !y || !tmp || !lg) goto done;

    double cos_min = 2.0, cos_sum = 0.0, ccos_min = 2.0, ccos_sum = 0.0, ms_tot = 0.0;
    size_t top_match = 0, npos = 0;
    for (uint32_t s = 0; s < n_seq; s++) {
        double t0 = omp_get_wtime();
        for (size_t t = 0; t < T; t++) memcpy(x + t * D, embed + (size_t)toks[s * T + t] * D, D * 4);
        for (size_t l = 0; l < L; l++) {
            const layer_t *ly = &layers[l];
            rms_norm(h, x, ly->norm1, T, D, eps);
            proj_forward(g, h, tmp, T, D, B, &ly->gate_proj);
            causal_conv_ss_xc(cv, h, T, D, K, ly->tw_u4, ly->tw_s);
            for (size_t i = 0; i < TD; i++) x[i] += (1.0f / (1.0f + expf(-g[i]))) * cv[i];
            rms_norm(h, x, ly->norm2, T, D, eps);
            proj_forward(u, h, tmp, T, D, B, &ly->w1);
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < D; d++) {
                    float v = u[t * D + d] * ly->diag_scale[d];
                    u[t * D + d] = v / (1.0f + expf(-v));
                }
            proj_forward(y, u, tmp, T, D, B, &ly->w2);
            for (size_t i = 0; i < TD; i++) x[i] += y[i];
        }
        ms_tot += (omp_get_wtime() - t0) * 1000.0;
        rms_norm(h, x, final_norm, T, D, eps);
        for (size_t i = 0; i < TD; i++) h[i] += h_bias[i % D];
        for (size_t t = 0; t < T; t++) {
            const float *ht = h + t * D, *rt = ref + ((size_t)s * T + t) * V;
            double mc = 0, mr = 0;
            size_t tc = 0, tr = 0;
            for (size_t v = 0; v < V; v++) {
                double acc = 0.0;
                const float *ev = embed + v * D;
                for (size_t d = 0; d < D; d++) acc += (double)ht[d] * ev[d];
                lg[v] = (float)acc;
                mc += lg[v]; mr += rt[v];
                if (lg[v] > lg[tc]) tc = v;
                if (rt[v] > rt[tr]) tr = v;
            }
            mc /= (double)V; mr /= (double)V;
            double dot = 0, na = 0, nb = 0, cd = 0, ca = 0, cb = 0;
            for (size_t v = 0; v < V; v++) {
                dot += (double)lg[v] * rt[v]; na += (double)lg[v] * lg[v]; nb += (double)rt[v] * rt[v];
                double a = lg[v] - mc, b = rt[v] - mr;
                cd += a * b; ca += a * a; cb += b * b;
            }
            double c = dot / (sqrt(na * nb) + 1e-30), cc = cd / (sqrt(ca * cb) + 1e-30);
            cos_sum += c; ccos_sum += cc; npos++;
            if (c < cos_min) cos_min = c;
            if (cc < ccos_min) ccos_min = cc;
            top_match += (tc == tr);
        }
    }
    printf("=== Pure-C11 x-cruncher Block-TrunkLM-24 Variant D Verification ===\n");
    printf("  Layers=%zu D=%zu B=%zu K=%zu V=%zu | %u seqs x T=%u = %zu positions\n", L, D, B, K, V, n_seq, T, npos);
    printf("  24-layer prefill (T=%u): %.2f ms/seq (xc_mul_u64)\n", T, ms_tot / n_seq);
    printf("  Logit cosine vs PyTorch hard-4-bit: mean %.6f | min %.6f\n", cos_sum / npos, cos_min);
    printf("  Centred logit cosine:               mean %.6f | min %.6f\n", ccos_sum / npos, ccos_min);
    printf("  Top-1 parity: %zu / %zu positions\n", top_match, npos);
    rc = (cos_min >= 0.999) ? 0 : 2;
done:
    for (size_t l = 0; layers && l < L; l++) {
        layer_t *ly = &layers[l];
        free(ly->norm1); free(ly->norm2); free(ly->diag_scale); free(ly->tw_s); free(ly->tw_u4);
        free_proj(&ly->gate_proj); free_proj(&ly->w1); free_proj(&ly->w2);
    }
    free(layers); free(embed); free(final_norm); free(h_bias); free(toks); free(ref);
    free(x); free(h); free(g); free(cv); free(u); free(y); free(tmp); free(lg);
    return rc;
}
