#define _GNU_SOURCE
/* Idea 3 (human, 2026-10-18): keep the already computed vectors P_k = value_k * x (the whole product vector of one weight value with the
   token's activations, built once per token) and build each row's product vectors by SELECTING lanes from them with vector operations,
   no multiply per weight. A group's table holds 16 values, so a chunk of 8 columns is built from at most 16 masked loads: for each of
   the group's candidate values, compare the 8 columns' levels with it and load those lanes of P_k. The additions are the shipped
   kernel's (accumulators A = [l0|l4], B = [l8|l12]); P_k holds the same double products, so the bits must equal the shipped result.

     expert-select-bench DATASET_DIR [REPS]                                                      */
#define LUT_AS_LIB
#include "expert-lut-bench.c"

static uint8_t (*g_cand)[16]; static uint8_t *g_ncand;   /* per pair table: the global levels of its (up to) 16 values */

static void build_candidates(const Root *root)
{
    g_cand = calloc(root->map_count, sizeof *g_cand); g_ncand = calloc(root->map_count, 1);
    for (unsigned t = 0; t < root->map_count; t++) {
        unsigned n = 0;
        for (unsigned code = 0; code < 256; code++) {
            const uint16_t l = g_lv[t][code];
            for (int h = 0; h < 2; h++) {
                const unsigned lev = h ? (l >> 8) : (l & 255);
                unsigned i = 0; while (i < n && g_cand[t][i] != lev) i++;
                if (i == n) { if (n == 16) bail("a pair table has more than 16 values"); g_cand[t][n++] = (uint8_t)lev; }
            }
        }
        g_ncand[t] = (uint8_t)n;
    }
}

/* P: g_lp rows of `width` doubles (level-major, so the 8 columns of a chunk are contiguous in each P_k). */
static void select_matrix(const Root *root, unsigned expert, unsigned matrix, const double *x, float *out, double *P, double *build_s)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    const double b0 = clk();
    for (unsigned k = 0; k < g_dict_n; k++) for (unsigned c = 0; c < width; c++) P[(size_t)k * width + c] = g_dict[k] * x[c];
    *build_s = clk() - b0;
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
            __m512d a = _mm512_setzero_pd(), b = _mm512_setzero_pd();
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const unsigned ref = root_u16(refs + (begin + selectors[coordinate / 32]) * 2);
                const uint16_t *lv = g_lv[ref];
                const unsigned char *p = codes + coordinate / 2;
                const uint64_t la = (uint64_t)lv[p[0]] | ((uint64_t)lv[p[1]] << 16) | ((uint64_t)lv[p[2]] << 32) | ((uint64_t)lv[p[3]] << 48);
                const uint64_t lb = (uint64_t)lv[p[4]] | ((uint64_t)lv[p[5]] << 16) | ((uint64_t)lv[p[6]] << 32) | ((uint64_t)lv[p[7]] << 48);
                const __m512i va = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128((long long)la)), vb = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128((long long)lb));
                __m512d ra = _mm512_setzero_pd(), rb = _mm512_setzero_pd();
                const unsigned n = g_ncand[ref];
                for (unsigned i = 0; i < n; i++) {
                    const unsigned k = g_cand[ref][i];
                    const double *pk = P + (size_t)k * width + coordinate;
                    ra = _mm512_mask_loadu_pd(ra, _mm512_cmpeq_epi64_mask(va, _mm512_set1_epi64(k)), pk);
                    rb = _mm512_mask_loadu_pd(rb, _mm512_cmpeq_epi64_mask(vb, _mm512_set1_epi64(k)), pk + 8);
                }
                a = _mm512_add_pd(a, ra); b = _mm512_add_pd(b, rb);
            }
            const __m512d t = _mm512_add_pd(a, b);
            double sums[4];
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1)));
            out[block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-select-bench DATASET_DIR [REPS]\n", stderr); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (getenv("EK_THREADS")) nth = atoi(getenv("EK_THREADS"));
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    build_dictionary(root); build_candidates(root);
    printf("dictionary: %u values; per-group candidates: up to 16 masked loads per 8 columns; P table per token = %u levels x 3584 columns = %.2f MB\n",
        g_dict_n, g_dict_n, (double)g_dict_n * 3584 * 8 / 1e6);
    unsigned long touch = 0;
    for (unsigned e = 0; e < (unsigned)nth; e++) for (size_t i = 0; i < ROOT_EXPERT_RAW; i += 4096) touch += root->direct[(size_t)e * ROOT_EXPERT_RAW + i];
    if (touch == 1) puts("");
    static double conv[MAXTH][LATW];
    uint64_t s = 88172645463325252ULL;
    for (int t = 0; t < nth; t++) for (unsigned i = 0; i < LATW; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        conv[t][i] = (double)(float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0);
    }
    static float o_ship[MAXTH][3584], o_sel[MAXTH][3584];
    static double *Ps[MAXTH];
    for (int t = 0; t < nth; t++) Ps[t] = aligned_alloc(64, (size_t)g_lp * 3584 * 8);
    for (int matrix = 0; matrix < 3; matrix += 2) {
        const unsigned rows = matrix == 2 ? 3584 : 3072, width = matrix == 2 ? 3072 : 3584;
        const double macs = (double)rows * width;
        const double *cin[1] = { conv[0] }; float *oo[1] = { o_ship[0] };
        ky1(root, 0, matrix, cin, oo);
        double best_s = 1e9, best_z = 1e9, best_l = 1e9, build = 0, bs;
        for (int r = 0; r < reps; r++) { double t = clk(); ky1(root, 0, matrix, cin, oo); t = clk() - t; if (t < best_s) best_s = t; }
        { float *oz[1] = { o_sel[1] }; kz_1_1(root, 0, matrix, cin, oz);
          for (int r = 0; r < reps; r++) { double t = clk(); kz_1_1(root, 0, matrix, cin, oz); t = clk() - t; if (t < best_z) best_z = t; } }
        select_matrix(root, 0, matrix, conv[0], o_sel[0], Ps[0], &bs);
        for (int r = 0; r < reps; r++) { double t = clk(); select_matrix(root, 0, matrix, conv[0], o_sel[0], Ps[0], &bs); t = clk() - t; if (t < best_l) { best_l = t; build = bs; } }
        const int same = !memcmp(o_ship[0], o_sel[0], rows * sizeof(float));
        printf("\nexpert matrix %d (%u x %u), 1 token, 1 thread\n", matrix, rows, width);
        printf("  shipped 256-bit %.3f ms (%.1f GMAC/s); 512-bit %.3f ms\n", best_s * 1e3, macs / best_s / 1e9, best_z * 1e3);
        printf("  vector select from precomputed product vectors %.3f ms (%.1f GMAC/s; building the P vectors %.3f ms of it) = %.2fx the shipped time; bits vs shipped: %s\n",
            best_l * 1e3, macs / best_l / 1e9, build * 1e3, best_l / best_s, same ? "identical" : "DIFFERENT");
        double bn_s = 1e9, bn_l = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const double *ci[1] = { conv[th] }; float *o1[1] = { o_ship[th] }; ky1(root, (unsigned)th, (unsigned)matrix, ci, o1); }
            t = clk() - t; if (r >= 0 && t < bn_s) bn_s = t;
        }
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); double b_; select_matrix(root, (unsigned)th, (unsigned)matrix, conv[th], o_sel[th], Ps[th], &b_); }
            t = clk() - t; if (r >= 0 && t < bn_l) bn_l = t;
        }
        int same_all = 1;
        for (int th = 0; th < nth; th++) { const double *ci[1] = { conv[th] }; float *o1[1] = { o_ship[th] }; ky1(root, (unsigned)th, (unsigned)matrix, ci, o1);
            if (memcmp(o_ship[th], o_sel[th], rows * sizeof(float))) same_all = 0; }
        printf("  %d threads, %d experts: shipped %.3f ms, vector select %.3f ms = %.2fx the time; bits identical on all experts: %s\n",
            nth, nth, bn_s * 1e3, bn_l * 1e3, bn_l / bn_s, same_all ? "yes" : "NO");
    }
    return 0;
}
