#define _GNU_SOURCE
/* Bucket arithmetic for the expert matrices (human's idea, 2026-10-18): per row, add the activations that meet the same weight value
   into one bucket per value, then multiply each bucket by its value once (66 multiplies per row instead of 3,584). The additions
   run in a different ORDER from the shipped kernel (all columns of one value first), so the bits are not guaranteed to match: this
   bench measures speed and how many float outputs differ. Buckets are doubles, as the shipped accumulators are, so a difference
   can only appear when two double results straddle a float rounding boundary.
   T tokens share every decode: buckets are T-wide vectors (T = 1, 4 or 8), activations are stored token-interleaved.

     expert-bucket-bench DATASET_DIR [REPS]                                                      */
#define LUT_AS_LIB
#include "expert-lut-bench.c"

enum { MAXT_B = 8 };
/* x: [column][T] doubles. out: [token][row]. */
#define BUCKET_KERNEL(NAME, T, VEC) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, const double *xi, float *const *out) \
{ \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    double bucket[MAXLEV][T] __attribute__((aligned(64))); \
    for (unsigned block = 0; block < rows / 64; block++) { \
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW + \
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
        for (unsigned row = 0; row < 64; row++) { \
            const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32; \
            memset(bucket, 0, sizeof(double) * T * g_lp); \
            for (unsigned c = 0; c < width; c += 2) { \
                const uint16_t *lv = g_lv[root_u16(refs + (begin + selectors[c / 32]) * 2)]; \
                const uint16_t l2 = lv[codes[c / 2]]; \
                VEC(bucket[l2 & 255], xi + (size_t)c * T); \
                VEC(bucket[l2 >> 8], xi + (size_t)(c + 1) * T); \
            } \
            for (int t = 0; t < T; t++) { \
                double total = 0.0; \
                for (unsigned k = 0; k < g_dict_n; k++) total = total + g_dict[k] * bucket[k][t]; \
                out[t][block * 64 + row] = (float)total; \
            } \
        } \
    } \
}
#define VEC1(B, X) ((B)[0] = (B)[0] + (X)[0])
#define VEC4(B, X) _mm256_store_pd((B), _mm256_add_pd(_mm256_load_pd(B), _mm256_loadu_pd(X)))
#define VEC8(B, X) _mm512_store_pd((B), _mm512_add_pd(_mm512_load_pd(B), _mm512_loadu_pd(X)))
BUCKET_KERNEL(bucket1, 1, VEC1)
BUCKET_KERNEL(bucket4, 4, VEC4)
BUCKET_KERNEL(bucket8, 8, VEC8)

typedef void (*BucketFn)(const Root *, unsigned, unsigned, const double *, float *const *);

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-bucket-bench DATASET_DIR [REPS]\n", stderr); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (getenv("EK_THREADS")) nth = atoi(getenv("EK_THREADS"));
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    build_dictionary(root);
    printf("dictionary: %u distinct weight values (buckets per row)\n", g_dict_n);
    unsigned long touch = 0;
    for (unsigned e = 0; e < (unsigned)nth; e++) for (size_t i = 0; i < ROOT_EXPERT_RAW; i += 4096) touch += root->direct[(size_t)e * ROOT_EXPERT_RAW + i];
    if (touch == 1) puts("");

    static double xs[MAXTH][MAXT_B][LATW];            /* [thread][token][column] for the shipped kernels */
    static double xi[MAXTH][LATW * MAXT_B];           /* token-interleaved for the bucket kernels */
    static float o_ship[MAXTH][MAXT_B][3584], o_bkt[MAXTH][MAXT_B][3584];
    uint64_t s = 88172645463325252ULL;
    struct { const char *name; BucketFn fn; int T; } variants[] = { {"bucket, 1 token ", bucket1, 1}, {"bucket, 4 tokens", bucket4, 4}, {"bucket, 8 tokens", bucket8, 8} };

    for (int matrix = 0; matrix < 3; matrix += 2) {
        const unsigned rows = matrix == 2 ? 3584 : 3072, width = matrix == 2 ? 3072 : 3584;
        printf("\nexpert matrix %d: %u x %u\n", matrix, rows, width);
        /* speed, 1 thread, expert 0 */
        for (int t = 0; t < 8; t++) for (unsigned i = 0; i < LATW; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            xs[0][t][i] = (double)(float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0);
            for (int tt = 0; tt < 1; tt++) xi[0][(size_t)i * 8 + t] = xs[0][t][i];
        }
        double ship4 = 1e9, ship1 = 1e9, shipz4 = 1e9, shipz8 = 1e9;
        { const double *c4[4] = { xs[0][0], xs[0][1], xs[0][2], xs[0][3] }; float *o4[4] = { o_ship[0][0], o_ship[0][1], o_ship[0][2], o_ship[0][3] };
          const double *c8[8]; float *o8[8]; for (int g = 0; g < 8; g++) { c8[g] = xs[0][g]; o8[g] = o_bkt[1][g]; }
          const double *c1[1] = { xs[0][0] }; float *o1[1] = { o_ship[0][0] };
          ky4(root, 0, matrix, c4, o4); ky1(root, 0, matrix, c1, o1); kz_2_4(root, 0, matrix, c4, o4); kz_1_8(root, 0, matrix, c8, o8);
          for (int r = 0; r < reps; r++) {
              double t = clk(); ky1(root, 0, matrix, c1, o1); t = clk() - t; if (t < ship1) ship1 = t;
              t = clk(); ky4(root, 0, matrix, c4, o4); t = clk() - t; if (t < ship4) ship4 = t;
              t = clk(); kz_2_4(root, 0, matrix, c4, o4); t = clk() - t; if (t < shipz4) shipz4 = t;
              t = clk(); kz_1_8(root, 0, matrix, c8, o8); t = clk() - t; if (t < shipz8) shipz8 = t;
          } }
        printf("  shipped per token, 1 thread: 256-bit 1 token %.3f ms; 256-bit 4 tokens %.3f ms; 512-bit 2 rows x 4 tokens %.3f ms; 512-bit 1 row x 8 tokens %.3f ms\n",
            ship1 * 1e3, ship4 * 1e3 / 4, shipz4 * 1e3 / 4, shipz8 * 1e3 / 8);
        for (int v = 0; v < 3; v++) {
            const int T = variants[v].T;
            double xt[LATW * MAXT_B] __attribute__((aligned(64)));
            for (unsigned i = 0; i < LATW; i++) for (int t = 0; t < T; t++) xt[(size_t)i * T + t] = xs[0][t][i];
            float *ob[MAXT_B]; for (int t = 0; t < T; t++) ob[t] = o_bkt[0][t];
            variants[v].fn(root, 0, matrix, xt, ob);
            double best = 1e9;
            for (int r = 0; r < reps; r++) { double t = clk(); variants[v].fn(root, 0, matrix, xt, ob); t = clk() - t; if (t < best) best = t; }
            printf("  %s %8.3f ms per token  (%.2fx the best shipped per token at this token count, 1 thread)\n", variants[v].name, best * 1e3 / T,
                best / T / (T == 1 ? ship1 : T == 4 ? shipz4 / 4 : shipz8 / 8));
        }
        /* drift: 16 experts x 8 tokens x 3 input sets, bucket-8 against the shipped per-token kernel */
        unsigned long outputs = 0, differ = 0; double max_rel = 0, max_ulps = 0;
        for (int set = 0; set < 3; set++) {
            for (int t = 0; t < nth; t++) for (int g = 0; g < 8; g++) for (unsigned i = 0; i < LATW; i++) {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                xs[t][g][i] = (double)(float)(((double)(s >> 11) / 9007199254740992.0 - 0.5) * (set == 0 ? 2.0 : set == 1 ? 0.2 : 8.0));
                xi[t][(size_t)i * 8 + g] = xs[t][g][i];
            }
#pragma omp parallel num_threads(nth)
            {
                const int th = omp_get_thread_num();
                float *ob[8]; for (int g = 0; g < 8; g++) ob[g] = o_bkt[th][g];
                bucket8(root, (unsigned)th, (unsigned)matrix, xi[th], ob);
                for (int g = 0; g < 8; g++) { const double *c1[1] = { xs[th][g] }; float *o1[1] = { o_ship[th][g] }; ky1(root, (unsigned)th, (unsigned)matrix, c1, o1); }
            }
            for (int th = 0; th < nth; th++) for (int g = 0; g < 8; g++) for (unsigned r = 0; r < rows; r++) {
                const float a = o_ship[th][g][r], b = o_bkt[th][g][r];
                outputs++;
                if (memcmp(&a, &b, 4)) {
                    differ++;
                    const double rel = fabs((double)a - (double)b) / (fabs((double)a) + 1e-30); if (rel > max_rel) max_rel = rel;
                    int ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
                    const double u = fabs((double)ia - (double)ib); if (u > max_ulps) max_ulps = u;
                }
            }
        }
        printf("  float outputs that differ from the shipped kernel: %lu of %lu (%.6f%%); largest difference %.3g relative, %.0f ulp\n",
            differ, outputs, 100.0 * (double)differ / (double)outputs, max_rel, max_ulps);
        /* 16 threads, 16 experts, 8 tokens each */
        double bs = 1e9, bb = 1e9;
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); const double *c8[8]; float *o8[8]; for (int g = 0; g < 8; g++) { c8[g] = xs[th][g]; o8[g] = o_ship[th][g]; } kz_1_8(root, (unsigned)th, (unsigned)matrix, c8, o8); }
            t = clk() - t; if (r >= 0 && t < bs) bs = t;
        }
        for (int r = -1; r < reps; r++) {
            double t = clk();
#pragma omp parallel num_threads(nth)
            { const int th = omp_get_thread_num(); float *ob[8]; for (int g = 0; g < 8; g++) ob[g] = o_bkt[th][g]; bucket8(root, (unsigned)th, (unsigned)matrix, xi[th], ob); }
            t = clk() - t; if (r >= 0 && t < bb) bb = t;
        }
        printf("  %d threads, %d experts x 8 tokens: shipped 512-bit 1 row x 8 tokens %.3f ms, bucket 8 tokens %.3f ms (%.2fx the time)\n", nth, nth, bs * 1e3, bb * 1e3, bb / bs);
    }
    return 0;
}
