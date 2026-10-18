#define _GNU_SOURCE
/* Product-table experiment for the expert matrices (human's idea, 2026-10-18): the 4-bit experts hold only a few dozen distinct
   float values, so for one token's vector x the products value * x[column] repeat for every row. Here, per column tile, the table
   T[column][level] = dictionary[level] * x[column] is built once (the same double multiply the kernel does) and each row's
   multiply is replaced by a gather from it; the additions, their order and the final fold are unchanged, so the bits must equal
   the shipped kernel's. Compared on real expert data with the shipped 256-bit kernel and the 512-bit kernels of
   expert-kernel-bench.c (included here, its main renamed).

     expert-lut-bench DATASET_DIR [REPS]                                                      */
#define main ekb_main
#include "expert-kernel-bench.c"
#undef main

enum { CT = 64, MAXLEV = 256 };
static double g_dict[MAXLEV]; static unsigned g_dict_n, g_lp;
static uint16_t (*g_lv)[256];   /* per pair table, per code byte: the levels of its two columns (low byte = first) */

static void build_dictionary(const Root *root)
{
    unsigned n = 0; unsigned bits[MAXLEV];
    for (unsigned t = 0; t < root->map_count; t++)
        for (unsigned code = 0; code < 256; code++)
            for (unsigned h = 0; h < 2; h++) {
                float v = root->pairs[t][code][h]; unsigned b; memcpy(&b, &v, 4);
                unsigned i = 0; while (i < n && bits[i] != b) i++;
                if (i == n) { if (n == MAXLEV) bail("more than 256 distinct weight values"); bits[n++] = b; }
            }
    g_dict_n = n; g_lp = (n + 7) / 8 * 8;
    for (unsigned i = 0; i < n; i++) { float v; memcpy(&v, &bits[i], 4); g_dict[i] = (double)v; }
    for (unsigned i = n; i < g_lp; i++) g_dict[i] = 0.0;
    g_lv = malloc((size_t)root->map_count * 256 * sizeof(uint16_t));
    for (unsigned t = 0; t < root->map_count; t++)
        for (unsigned code = 0; code < 256; code++) {
            unsigned lv[2];
            for (unsigned h = 0; h < 2; h++) {
                float v = root->pairs[t][code][h]; unsigned b; memcpy(&b, &v, 4);
                unsigned i = 0; while (bits[i] != b) i++;
                lv[h] = i;
            }
            g_lv[t][code] = (uint16_t)(lv[0] | (lv[1] << 8));
        }
}

/* One token, one expert matrix. `tab` holds CT * g_lp doubles; `acc` rows x 2 zmm. Returns the time spent building tables in *build_s. */
static void lut_matrix(const Root *root, unsigned expert, unsigned matrix, const double *x, float *out, double *tab, __m512d (*acc)[2], double *build_s)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    const unsigned lp = g_lp;
    for (unsigned r = 0; r < rows; r++) { acc[r][0] = _mm512_setzero_pd(); acc[r][1] = _mm512_setzero_pd(); }
    int lane_base[8]; for (int l = 0; l < 8; l++) lane_base[l] = l * (int)lp;
    const __m256i lanes = _mm256_loadu_si256((const __m256i *)lane_base);
    double built = 0;
    for (unsigned c0 = 0; c0 < width; c0 += CT) {
        const double b0 = clk();
        for (unsigned c = 0; c < CT; c++) { const double xv = x[c0 + c]; for (unsigned l = 0; l < lp; l++) tab[(size_t)c * lp + l] = g_dict[l] * xv; }
        built += clk() - b0;
        for (unsigned block = 0; block < rows / 64; block++) {
            const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
                (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
            for (unsigned row = 0; row < 64; row++) {
                const unsigned char *codes = data + row * width / 2;
                const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
                __m512d a = acc[block * 64 + row][0], b = acc[block * 64 + row][1];
                for (unsigned s = 0; s < CT / 16; s++) {
                    const unsigned coordinate = c0 + s * 16;
                    const uint16_t *lv = g_lv[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                    const unsigned char *p = codes + coordinate / 2;
                    const uint64_t la = (uint64_t)lv[p[0]] | ((uint64_t)lv[p[1]] << 16) | ((uint64_t)lv[p[2]] << 32) | ((uint64_t)lv[p[3]] << 48);
                    const uint64_t lb = (uint64_t)lv[p[4]] | ((uint64_t)lv[p[5]] << 16) | ((uint64_t)lv[p[6]] << 32) | ((uint64_t)lv[p[7]] << 48);
                    const __m256i colbase = _mm256_set1_epi32((int)(s * 16 * lp));
                    const __m256i ia = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)la)), lanes), colbase);
                    const __m256i ib = _mm256_add_epi32(_mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)lb)), lanes),
                        _mm256_set1_epi32((int)((s * 16 + 8) * lp)));
                    a = _mm512_add_pd(a, _mm512_i32gather_pd(ia, tab, 8));
                    b = _mm512_add_pd(b, _mm512_i32gather_pd(ib, tab, 8));
                }
                acc[block * 64 + row][0] = a; acc[block * 64 + row][1] = b;
            }
        }
    }
    for (unsigned r = 0; r < rows; r++) {
        const __m512d t = _mm512_add_pd(acc[r][0], acc[r][1]);
        double sums[4];
        _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1)));
        out[r] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
    }
    *build_s = built;
}

/* ---- hash list of canonical keys (human's design): every product weight * activation is looked up under the key (smaller operand
   bits, larger operand bits) of the two floats; a hit returns the stored double product, a miss multiplies once and stores it. The
   additions are the shipped kernel's, scalar: column c goes into accumulator c % 16, and the fold is the shipped one, so the bits
   must be the same. `use_hash` 0 is the same scalar loop with the plain multiply, to separate "scalar" from "hash". ---- */
enum { HBITS = 19 };
typedef struct { uint64_t key; double val; } HEntry;
static void hash_matrix(const Root *root, unsigned expert, unsigned matrix, const double *x, float *out, HEntry *tab, int use_hash,
    unsigned long *hits, unsigned long *misses, double *clear_s)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    if (use_hash) { const double c0 = clk(); memset(tab, 0, sizeof(HEntry) << HBITS); *clear_s = clk() - c0; }
    unsigned xbits[LATW];
    for (unsigned c = 0; c < width; c++) { const float f = (float)x[c]; memcpy(&xbits[c], &f, 4); }
    unsigned long h = 0, m = 0;
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
            double acc[16] = {0};
            for (unsigned c = 0; c < width; c++) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[c / 32]) * 2)];
                const float w = pair[codes[c / 2]][c & 1];
                double p;
                if (use_hash) {
                    unsigned wb; memcpy(&wb, &w, 4);
                    const unsigned xb = xbits[c];
                    const uint64_t key = (wb < xb ? ((uint64_t)wb << 32) | xb : ((uint64_t)xb << 32) | wb) + 1;
                    size_t i = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - HBITS));
                    for (;; i = (i + 1) & ((1u << HBITS) - 1)) {
                        if (tab[i].key == key) { p = tab[i].val; h++; break; }
                        if (!tab[i].key) { p = (double)w * x[c]; tab[i].key = key; tab[i].val = p; m++; break; }
                    }
                } else p = (double)w * x[c];
                acc[c & 15] = acc[c & 15] + p;
            }
            double s4[4];
            for (int i = 0; i < 4; i++) s4[i] = (acc[i] + acc[8 + i]) + (acc[4 + i] + acc[12 + i]);
            out[block * 64 + row] = (float)((s4[0] + s4[2]) + (s4[1] + s4[3]));
        }
    }
    *hits = h; *misses = m;
}

#ifdef LUT_AS_LIB
#define LUT_MAIN lut1_main
#else
#define LUT_MAIN main
#endif
int LUT_MAIN(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-lut-bench DATASET_DIR [REPS]\n", stderr); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (getenv("EK_THREADS")) nth = atoi(getenv("EK_THREADS"));
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    build_dictionary(root);
    printf("dictionary: %u distinct weight values over %u pair tables (padded to %u); table per column tile: %u columns x %u levels = %.0f KB\n",
        g_dict_n, root->map_count, g_lp, CT, g_lp, (double)CT * g_lp * 8 / 1024);
    unsigned long touch = 0;
    for (unsigned e = 0; e < (unsigned)nth; e++) for (size_t i = 0; i < ROOT_EXPERT_RAW; i += 4096) touch += root->direct[(size_t)e * ROOT_EXPERT_RAW + i];
    if (touch == 1) puts("");

    static double conv[MAXTH][LATW];
    uint64_t s = 88172645463325252ULL;
    for (int t = 0; t < nth; t++) for (unsigned i = 0; i < LATW; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        conv[t][i] = (double)(float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0);   /* a float widened, as the layer's latent is */
    }
    static float o_ship[MAXTH][3584], o_lut[MAXTH][3584];
    static double *tabs[MAXTH]; static __m512d (*accs[MAXTH])[2];
    for (int t = 0; t < nth; t++) { tabs[t] = aligned_alloc(64, (size_t)CT * g_lp * 8); accs[t] = aligned_alloc(64, 3584 * sizeof(__m512d) * 2); }
    static HEntry *htabs[MAXTH];
    for (int t = 0; t < nth; t++) { htabs[t] = aligned_alloc(64, sizeof(HEntry) << HBITS); if (!htabs[t]) bail("hash list allocation failed"); }

    for (int matrix = 0; matrix < 3; matrix += 2) {
        const unsigned rows = matrix == 2 ? 3584 : 3072, width = matrix == 2 ? 3072 : 3584;
        printf("\nexpert matrix %d: %u x %u, one token\n", matrix, rows, width);
        const double *cin[1] = { conv[0] }; float *oo[1] = { o_ship[0] };
        ky1(root, 0, matrix, cin, oo);
        double bs, best_ship = 1e9, best_z = 1e9, best_lut = 1e9, best_build = 1e9;
        for (int r = 0; r < reps; r++) { double t = clk(); ky1(root, 0, matrix, cin, oo); t = clk() - t; if (t < best_ship) best_ship = t; }
        { float *oz[1] = { o_lut[1] }; kz_1_1(root, 0, matrix, cin, oz);
          for (int r = 0; r < reps; r++) { double t = clk(); kz_1_1(root, 0, matrix, cin, oz); t = clk() - t; if (t < best_z) best_z = t; } }
        lut_matrix(root, 0, matrix, conv[0], o_lut[0], tabs[0], accs[0], &bs);
        for (int r = 0; r < reps; r++) {
            double t = clk(); lut_matrix(root, 0, matrix, conv[0], o_lut[0], tabs[0], accs[0], &bs); t = clk() - t;
            if (t < best_lut) { best_lut = t; best_build = bs; }
        }
        const int same = !memcmp(o_ship[0], o_lut[0], rows * sizeof(float));
        const double macs = (double)rows * width;
        printf("  shipped 256-bit 1 row x 1 token   %8.3f ms  %7.1f GMAC/s\n", best_ship * 1e3, macs / best_ship / 1e9);
        printf("  512-bit 1 row x 1 token           %8.3f ms  %7.1f GMAC/s\n", best_z * 1e3, macs / best_z / 1e9);
        printf("  product table (gather)            %8.3f ms  %7.1f GMAC/s   (building the tables %.3f ms of it)   bits vs shipped: %s\n",
            best_lut * 1e3, macs / best_lut / 1e9, best_build * 1e3, same ? "identical" : "DIFFERENT");
        printf("  product table / shipped: %.2fx the time\n", best_lut / best_ship);
        {   /* the hash list of canonical keys, and the same scalar loop without it */
            static float o_h[3584]; unsigned long hh = 0, mm = 0; double clr = 0, best_h = 1e9, best_s = 1e9, best_clear = 0;
            for (int r = 0; r < 3; r++) {
                double t = clk(); hash_matrix(root, 0, matrix, conv[0], o_h, htabs[0], 1, &hh, &mm, &clr); t = clk() - t;
                if (t < best_h) { best_h = t; best_clear = clr; }
            }
            const int same_h = !memcmp(o_ship[0], o_h, rows * sizeof(float));
            static float o_s[3584]; unsigned long u1, u2; double u3;
            for (int r = 0; r < 3; r++) {
                double t = clk(); hash_matrix(root, 0, matrix, conv[0], o_s, htabs[0], 0, &u1, &u2, &u3); t = clk() - t;
                if (t < best_s) best_s = t;
            }
            const int same_s = !memcmp(o_ship[0], o_s, rows * sizeof(float));
            printf("  scalar loop, plain multiply       %8.3f ms  %7.1f GMAC/s   bits vs shipped: %s\n", best_s * 1e3, macs / best_s / 1e9, same_s ? "identical" : "DIFFERENT");
            printf("  hash list (canonical key)         %8.3f ms  %7.1f GMAC/s   (clearing the list %.3f ms of it)   hits %lu of %lu lookups = %.2f%%, %lu entries stored   bits vs shipped: %s\n",
                best_h * 1e3, macs / best_h / 1e9, best_clear * 1e3, hh, hh + mm, 100.0 * (double)hh / (double)(hh + mm), mm, same_h ? "identical" : "DIFFERENT");
            printf("  hash list / shipped: %.1fx the time; hash list / scalar plain: %.2fx\n", best_h / best_ship, best_h / best_s);
            double bn_h = 1e9;
            for (int r = -1; r < 2; r++) {
                double t = clk();
#pragma omp parallel num_threads(nth)
                { const int th = omp_get_thread_num(); static float oth[MAXTH][3584]; unsigned long a, b; double c; hash_matrix(root, (unsigned)th, (unsigned)matrix, conv[th], oth[th], htabs[th], 1, &a, &b, &c); }
                t = clk() - t; if (r >= 0 && t < bn_h) bn_h = t;
            }
            printf("  hash list, %d threads, %d experts: %.3f ms (%.1f GMAC/s)\n", nth, nth, bn_h * 1e3, macs * nth / bn_h / 1e9);
        }

        /* all threads, one expert each, one token each */
        double bn_ship = 1e9, bn_lut = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const double *ci[1] = { conv[th] }; float *o1[1] = { o_ship[th] }; ky1(root, (unsigned)th, (unsigned)matrix, ci, o1); }
            t = clk() - t; if (r >= 0 && t < bn_ship) bn_ship = t;
        }
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); double b_; lut_matrix(root, (unsigned)th, (unsigned)matrix, conv[th], o_lut[th], tabs[th], accs[th], &b_); }
            t = clk() - t; if (r >= 0 && t < bn_lut) bn_lut = t;
        }
        int same_all = 1;
        for (int th = 0; th < nth; th++) {
            const double *ci[1] = { conv[th] }; float *o1[1] = { o_ship[th] }; ky1(root, (unsigned)th, (unsigned)matrix, ci, o1);
            if (memcmp(o_ship[th], o_lut[th], rows * sizeof(float))) same_all = 0;
        }
        printf("  %d threads, %d experts: shipped %.3f ms (%.0f GMAC/s), product table %.3f ms (%.0f GMAC/s) = %.2fx the time; bits identical on all %d experts: %s\n",
            nth, nth, bn_ship * 1e3, macs * nth / bn_ship / 1e9, bn_lut * 1e3, macs * nth / bn_lut / 1e9, bn_lut / bn_ship, nth, same_all ? "yes" : "NO");
    }
    return 0;
}
