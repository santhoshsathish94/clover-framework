#define _GNU_SOURCE
/* Plain lookup tables, no hashing (human's direction 2026-10-18): the product of every weight level with every activation of
   every token is a table entry; a multiply becomes one gather.
   Part 1: experts, 4 tokens per pass (the weight levels of a step are decoded once and gathered from each token's own table),
           bits compared with the shipped kernel; time per token against the shipped 4-token kernels.
   Part 2: trunk (layer 1's gate matrix, real 8-bit codes), table of code * activation per token. The trunk kernel uses a fused
           multiply-add and adds into float, so a table of rounded float products changes the bits: this part measures SPEED ONLY and
           does not compare bits.

     expert-lut2-bench DATASET_DIR [REPS]                                                      */
#define LUT_AS_LIB
#include "expert-lut-bench.c"

enum { G4 = 4 };
static void lut_matrix4(const Root *root, unsigned expert, unsigned matrix, const double *const *x, float *const *out, double *tab, __m512d (*acc)[2 * G4])
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    const unsigned lp = g_lp;
    const size_t tsz = (size_t)CT * lp;
    for (unsigned r = 0; r < rows; r++) for (int i = 0; i < 2 * G4; i++) acc[r][i] = _mm512_setzero_pd();
    int lane_base[8]; for (int l = 0; l < 8; l++) lane_base[l] = l * (int)lp;
    const __m256i lanes = _mm256_loadu_si256((const __m256i *)lane_base);
    for (unsigned c0 = 0; c0 < width; c0 += CT) {
        for (int g = 0; g < G4; g++)
            for (unsigned c = 0; c < CT; c++) { const double xv = x[g][c0 + c]; for (unsigned l = 0; l < lp; l++) tab[g * tsz + (size_t)c * lp + l] = g_dict[l] * xv; }
        for (unsigned block = 0; block < rows / 64; block++) {
            const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
                (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
            for (unsigned row = 0; row < 64; row++) {
                const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
                __m512d a[G4], b[G4];
                for (int g = 0; g < G4; g++) { a[g] = acc[block * 64 + row][g * 2]; b[g] = acc[block * 64 + row][g * 2 + 1]; }
                for (unsigned s = 0; s < CT / 16; s++) {
                    const unsigned coordinate = c0 + s * 16;
                    const uint16_t *lv = g_lv[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                    const unsigned char *p = codes + coordinate / 2;
                    const uint64_t la = (uint64_t)lv[p[0]] | ((uint64_t)lv[p[1]] << 16) | ((uint64_t)lv[p[2]] << 32) | ((uint64_t)lv[p[3]] << 48);
                    const uint64_t lb = (uint64_t)lv[p[4]] | ((uint64_t)lv[p[5]] << 16) | ((uint64_t)lv[p[6]] << 32) | ((uint64_t)lv[p[7]] << 48);
                    const __m256i ia = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)la)), lanes), _mm256_set1_epi32((int)(s * 16 * lp)));
                    const __m256i ib = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)lb)), lanes), _mm256_set1_epi32((int)((s * 16 + 8) * lp)));
                    for (int g = 0; g < G4; g++) {
                        a[g] = _mm512_add_pd(a[g], _mm512_i32gather_pd(ia, tab + g * tsz, 8));
                        b[g] = _mm512_add_pd(b[g], _mm512_i32gather_pd(ib, tab + g * tsz, 8));
                    }
                }
                for (int g = 0; g < G4; g++) { acc[block * 64 + row][g * 2] = a[g]; acc[block * 64 + row][g * 2 + 1] = b[g]; }
            }
        }
    }
    for (unsigned r = 0; r < rows; r++)
        for (int g = 0; g < G4; g++) {
            const __m512d t = _mm512_add_pd(acc[r][g * 2], acc[r][g * 2 + 1]);
            double sums[4];
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1)));
            out[g][r] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
        }
}

/* ---- trunk: `rows` rows of a record's 8-bit codes (row stride `cols`), one token's vector x ---- */
enum { TCT = 32 };
static void trunk_lut(const unsigned char *ids, unsigned rows, unsigned cols, const float *x, float *tab, __m512 *acc, float *out)
{
    for (unsigned r = 0; r < rows; r++) acc[r] = _mm512_setzero_ps();
    int lane_base[16]; for (int l = 0; l < 16; l++) lane_base[l] = l * 256;
    const __m512i lanes = _mm512_loadu_si512((const void *)lane_base);
    for (unsigned c0 = 0; c0 < cols; c0 += TCT) {
        for (unsigned c = 0; c < TCT; c++) { const float xv = x[c0 + c]; for (unsigned u = 0; u < 256; u++) tab[(size_t)c * 256 + u] = (float)(int8_t)u * xv; }
        for (unsigned r = 0; r < rows; r++) {
            __m512 a = acc[r];
            const unsigned char *row_ids = ids + (size_t)r * cols + c0;
            for (unsigned s = 0; s < TCT / 16; s++) {
                const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(row_ids + s * 16))), lanes),
                    _mm512_set1_epi32((int)(s * 16 * 256)));
                a = _mm512_add_ps(a, _mm512_i32gather_ps(idx, tab, 4));
            }
            acc[r] = a;
        }
    }
    for (unsigned r = 0; r < rows; r++) out[r] = _mm512_reduce_add_ps(acc[r]);
}
/* The shipped trunk arithmetic (4 rows x 4 tokens, fused multiply-add into 16 float lanes), as in dotz4x4 of layer-batch.c. */
static void trunk_ship4(const unsigned char *ids, unsigned rows, unsigned cols, const float *const *x, float *const *out)
{
    for (unsigned row = 0; row < rows; row += 4) {
        __m512 a[4][4];
        for (int r = 0; r < 4; r++) for (int g = 0; g < 4; g++) a[r][g] = _mm512_setzero_ps();
        for (unsigned c = 0; c < cols; c += 16) {
            __m512 w[4];
            for (int r = 0; r < 4; r++) w[r] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(ids + (size_t)(row + r) * cols + c))));
            for (int r = 0; r < 4; r++) for (int g = 0; g < 4; g++) a[r][g] = _mm512_fmadd_ps(w[r], _mm512_loadu_ps(x[g] + c), a[r][g]);
        }
        for (int r = 0; r < 4; r++) for (int g = 0; g < 4; g++) out[g][row + r] = _mm512_reduce_add_ps(a[r][g]);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-lut2-bench DATASET_DIR [REPS]\n", stderr); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (getenv("EK_THREADS")) nth = atoi(getenv("EK_THREADS"));
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    build_dictionary(root);
    printf("dictionary: %u distinct weight values; expert tables: 4 tokens x %u columns x %u levels = %.0f KB per tile\n", g_dict_n, CT, g_lp, 4.0 * CT * g_lp * 8 / 1024);
    unsigned long touch = 0;
    for (unsigned e = 0; e < (unsigned)nth; e++) for (size_t i = 0; i < ROOT_EXPERT_RAW; i += 4096) touch += root->direct[(size_t)e * ROOT_EXPERT_RAW + i];
    if (touch == 1) puts("");

    /* ---- part 1: experts, 4 tokens ---- */
    static double conv[MAXTH][G4][LATW];
    uint64_t s = 88172645463325252ULL;
    for (int t = 0; t < nth; t++) for (int g = 0; g < G4; g++) for (unsigned i = 0; i < LATW; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        conv[t][g][i] = (double)(float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0);
    }
    static float o_ship[MAXTH][G4][3584], o_lut[MAXTH][G4][3584];
    static double *tabs[MAXTH]; static __m512d (*accs[MAXTH])[2 * G4];
    for (int t = 0; t < nth; t++) { tabs[t] = aligned_alloc(64, (size_t)G4 * CT * g_lp * 8); accs[t] = aligned_alloc(64, 3584 * sizeof(__m512d) * 2 * G4); }
    for (int matrix = 0; matrix < 3; matrix += 2) {
        const unsigned rows = matrix == 2 ? 3584 : 3072, width = matrix == 2 ? 3072 : 3584;
        const double macs = (double)rows * width * G4;
        const double *cin[G4]; float *oo[G4], *ol[G4];
        for (int g = 0; g < G4; g++) { cin[g] = conv[0][g]; oo[g] = o_ship[0][g]; ol[g] = o_lut[0][g]; }
        ky4(root, 0, matrix, cin, oo);
        double best_s = 1e9, best_z = 1e9, best_l = 1e9;
        for (int r = 0; r < reps; r++) { double t = clk(); ky4(root, 0, matrix, cin, oo); t = clk() - t; if (t < best_s) best_s = t; }
        { float *oz[G4]; for (int g = 0; g < G4; g++) oz[g] = o_lut[1][g];
          kz_2_4(root, 0, matrix, cin, oz);
          for (int r = 0; r < reps; r++) { double t = clk(); kz_2_4(root, 0, matrix, cin, oz); t = clk() - t; if (t < best_z) best_z = t; } }
        lut_matrix4(root, 0, matrix, cin, ol, tabs[0], accs[0]);
        for (int r = 0; r < reps; r++) { double t = clk(); lut_matrix4(root, 0, matrix, cin, ol, tabs[0], accs[0]); t = clk() - t; if (t < best_l) best_l = t; }
        int same = 1; for (int g = 0; g < G4; g++) if (memcmp(o_ship[0][g], o_lut[0][g], rows * sizeof(float))) same = 0;
        printf("\nexperts, matrix %d (%u x %u), 4 tokens, 1 thread\n", matrix, rows, width);
        printf("  shipped 256-bit 1 row x 4 tokens  %8.3f ms  (%.3f ms per token)\n", best_s * 1e3, best_s * 1e3 / G4);
        printf("  512-bit 2 rows x 4 tokens         %8.3f ms  (%.3f ms per token)\n", best_z * 1e3, best_z * 1e3 / G4);
        printf("  lookup table, 4 tokens            %8.3f ms  (%.3f ms per token)  = %.2fx the shipped time; bits vs shipped: %s\n", best_l * 1e3, best_l * 1e3 / G4, best_l / best_s, same ? "identical" : "DIFFERENT");
        double bn_s = 1e9, bn_l = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const double *ci[G4]; float *o4[G4]; for (int g = 0; g < G4; g++) { ci[g] = conv[th][g]; o4[g] = o_ship[th][g]; } ky4(root, (unsigned)th, (unsigned)matrix, ci, o4); }
            t = clk() - t; if (r >= 0 && t < bn_s) bn_s = t;
        }
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const double *ci[G4]; float *o4[G4]; for (int g = 0; g < G4; g++) { ci[g] = conv[th][g]; o4[g] = o_lut[th][g]; } lut_matrix4(root, (unsigned)th, (unsigned)matrix, ci, o4, tabs[th], accs[th]); }
            t = clk() - t; if (r >= 0 && t < bn_l) bn_l = t;
        }
        int same_all = 1;
        for (int th = 0; th < nth; th++) {
            const double *ci[G4]; float *o4[G4]; for (int g = 0; g < G4; g++) { ci[g] = conv[th][g]; o4[g] = o_ship[th][g]; } ky4(root, (unsigned)th, (unsigned)matrix, ci, o4);
            for (int g = 0; g < G4; g++) if (memcmp(o_ship[th][g], o_lut[th][g], rows * sizeof(float))) same_all = 0;
        }
        printf("  %d threads, %d experts x 4 tokens: shipped %.3f ms (%.0f GMAC/s), lookup %.3f ms (%.0f GMAC/s) = %.2fx the time; bits identical on all: %s\n",
            nth, nth, bn_s * 1e3, macs * nth / bn_s / 1e9, bn_l * 1e3, macs * nth / bn_l / 1e9, bn_l / bn_s, same_all ? "yes" : "NO");
    }

    /* ---- part 2: trunk, speed only ---- */
    {
        const TransformerRecord *rec = &tr->records[6];
        const unsigned cols = rec->columns, rows_all = rec->rows, rows = 2048;
        const unsigned char *ids = rec->data + (size_t)rows_all * 4;
        unsigned long tt = 0; for (size_t i = 0; i < (size_t)rows * cols; i += 4096) tt += ids[i];
        if (tt == 1) puts("");
        static float x4[MAXTH][4][8192], outs[MAXTH][4][2048], outl[MAXTH][2048];
        for (int t = 0; t < nth; t++) for (int g = 0; g < 4; g++) for (unsigned i = 0; i < cols; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            x4[t][g][i] = (float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0);
        }
        float *tab[MAXTH]; __m512 *ac[MAXTH];
        for (int t = 0; t < nth; t++) { tab[t] = aligned_alloc(64, (size_t)TCT * 256 * 4); ac[t] = aligned_alloc(64, 2048 * sizeof(__m512)); }
        const float *xs[4] = { x4[0][0], x4[0][1], x4[0][2], x4[0][3] }; float *os[4] = { outs[0][0], outs[0][1], outs[0][2], outs[0][3] };
        trunk_ship4(ids, rows, cols, xs, os);
        double best_s = 1e9, best_l = 1e9;
        for (int r = 0; r < reps; r++) { double t = clk(); trunk_ship4(ids, rows, cols, xs, os); t = clk() - t; if (t < best_s) best_s = t; }
        trunk_lut(ids, rows, cols, x4[0][0], tab[0], ac[0], outl[0]);
        for (int r = 0; r < reps; r++) { double t = clk(); trunk_lut(ids, rows, cols, x4[0][0], tab[0], ac[0], outl[0]); t = clk() - t; if (t < best_l) best_l = t; }
        const double macs1 = (double)rows * cols;
        printf("\ntrunk, layer 1 gate matrix, first %u of %u rows x %u columns, real codes, 1 thread (SPEED ONLY, bits not compared)\n", rows, rows_all, cols);
        printf("  shipped 4 rows x 4 tokens (fma)   %8.3f ms for 4 tokens = %.3f ms per token (%.1f GMAC/s)\n", best_s * 1e3, best_s * 1e3 / 4, macs1 * 4 / best_s / 1e9);
        printf("  lookup table, 1 token             %8.3f ms per token (%.1f GMAC/s) = %.1fx the shipped time per token\n", best_l * 1e3, macs1 / best_l / 1e9, best_l / (best_s / 4));
        double bn_s = 1e9, bn_l = 1e9;
        for (int r = -1; r < 3; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const float *xx[4] = { x4[th][0], x4[th][1], x4[th][2], x4[th][3] }; float *oo4[4] = { outs[th][0], outs[th][1], outs[th][2], outs[th][3] }; trunk_ship4(ids, rows, cols, xx, oo4); }
            t = clk() - t; if (r >= 0 && t < bn_s) bn_s = t;
        }
        for (int r = -1; r < 3; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); trunk_lut(ids, rows, cols, x4[th][0], tab[th], ac[th], outl[th]); }
            t = clk() - t; if (r >= 0 && t < bn_l) bn_l = t;
        }
        printf("  %d threads, each its own tokens: shipped %.3f ms for 4 tokens each (%.0f GMAC/s), lookup %.3f ms for 1 token each (%.0f GMAC/s) = %.1fx the time per token\n",
            nth, bn_s * 1e3, macs1 * 4 * nth / bn_s / 1e9, bn_l * 1e3, macs1 * nth / bn_l / 1e9, bn_l / (bn_s / 4));
    }
    return 0;
}
