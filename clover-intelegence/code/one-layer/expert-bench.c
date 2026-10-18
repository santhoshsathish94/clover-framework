#define _GNU_SOURCE
#define TRANSFORMER_NO_MAIN
/* Expert execution arrangements for layer 1, on real routing, experts locked in RAM.

   A  pod code as shipped: every thread works on ONE expert at a time (three parallel
      regions per expert), experts one after another.
   B  one expert per thread: a token's 16 experts run concurrently, each on its own thread.
   C  tokens grouped by expert over a window of W tokens: each distinct expert is read once
      and applied to every token in the window that picked it, G tokens per weight pass,
      experts spread over the threads.

   B and C use a single-threaded copy of root_project's arithmetic in the same order, so
   every (token, expert) output is compared bit for bit with A.
     expert-bench DATASET_DIR RESID_F32 S0_F32 NTOKENS                                        */
#include "transformer-1.c"
#include <omp.h>
#include <sys/mman.h>
#include <time.h>

enum { LATW = 3584, EXPW = 3072, INW = 7168, THREADS = 16 };

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}

static void bail(const char *what) { fprintf(stderr, "expert-bench: %s\n", what); exit(1); }

static inline __attribute__((always_inline)) void proj_core(const Root *root, unsigned expert, unsigned matrix,
    const double *const *conv, float *const *out, const int G)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m256d l0[8], l4[8], l8[8], l12[8];
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                l0[g] = _mm256_setzero_pd(); l4[g] = _mm256_setzero_pd();
                l8[g] = _mm256_setzero_pd(); l12[g] = _mm256_setzero_pd();
            }
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
#pragma GCC unroll 8
                for (int g = 0; g < G; g++) {
                    const double *source = conv[g] + coordinate;
                    l0[g] = _mm256_add_pd(l0[g], _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                    l4[g] = _mm256_add_pd(l4[g], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                    l8[g] = _mm256_add_pd(l8[g], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                    l12[g] = _mm256_add_pd(l12[g], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
                }
            }
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                double sums[4];
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(l0[g], l8[g]), _mm256_add_pd(l4[g], l12[g])));
                out[g][block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
            }
        }
    }
}

typedef void (*ProjFn)(const Root *, unsigned, unsigned, const double *const *, float *const *);
static void proj_g1(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 1); }
static void proj_g2(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 2); }
static void proj_g4(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 4); }
static void proj_g8(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 8); }
static ProjFn proj_for(int g) { return g == 8 ? proj_g8 : g == 4 ? proj_g4 : g == 2 ? proj_g2 : proj_g1; }

/* One expert applied to n tokens by the calling thread, G tokens per pass over its weights. */
static void expert_apply(const Root *root, unsigned expert, const unsigned *token, const unsigned *rank,
    unsigned n, const double (*lat)[LATW], float (*out)[16][LATW], int gmax)
{
    float gate[8][EXPW], up[8][EXPW];
    double act[8][EXPW];
    unsigned done = 0;
    while (done < n) {
        int g = gmax;
        while (g > (int)(n - done)) g >>= 1;
        const double *cin[8], *ain[8];
        float *go[8], *uo[8], *dout[8];
        for (int k = 0; k < g; k++) { cin[k] = lat[token[done + k]]; go[k] = gate[k]; uo[k] = up[k]; }
        proj_for(g)(root, expert, 0, cin, go);
        proj_for(g)(root, expert, 1, cin, uo);
        for (int k = 0; k < g; k++) {
            transformer_activation(gate[k], up[k], EXPW);
            for (unsigned i = 0; i < EXPW; i++) act[k][i] = (double)gate[k][i];
            ain[k] = act[k];
            dout[k] = out[token[done + k]][rank[done + k]];
        }
        proj_for(g)(root, expert, 2, ain, dout);
        done += (unsigned)g;
    }
}

typedef struct { unsigned expert, count, start; } Group;
static int by_count_desc(const void *a, const void *b)
{
    const Group *x = a, *y = b;
    return x->count != y->count ? (x->count < y->count ? 1 : -1) : (x->expert > y->expert) - (x->expert < y->expert);
}

/* Tokens [first, first+count) grouped by expert; experts spread over the threads. */
static unsigned run_window(const Root *root, unsigned first, unsigned count, unsigned (*sel)[16],
    const double (*lat)[LATW], float (*out)[16][LATW], int gmax, double *members_sum)
{
    static unsigned fill[896];
    unsigned total[896] = {0};
    for (unsigned t = first; t < first + count; t++) for (unsigned r = 0; r < 16; r++) total[sel[t][r]]++;
    unsigned *ptok = malloc((size_t)count * 16 * sizeof(unsigned)), *prank = malloc((size_t)count * 16 * sizeof(unsigned));
    Group *groups = malloc(896 * sizeof(Group));
    if (!ptok || !prank || !groups) bail("allocation failed");
    unsigned n = 0, offset = 0;
    for (unsigned e = 0; e < 896; e++) if (total[e]) { groups[n++] = (Group){e, total[e], offset}; offset += total[e]; fill[e] = 0; }
    for (unsigned g = 0; g < n; g++) fill[groups[g].expert] = groups[g].start;
    for (unsigned t = first; t < first + count; t++)
        for (unsigned r = 0; r < 16; r++) { unsigned e = sel[t][r]; ptok[fill[e]] = t; prank[fill[e]] = r; fill[e]++; }
    qsort(groups, n, sizeof(Group), by_count_desc);
#pragma omp parallel for schedule(dynamic, 1) num_threads(THREADS)
    for (unsigned g = 0; g < n; g++)
        expert_apply(root, groups[g].expert, ptok + groups[g].start, prank + groups[g].start, groups[g].count, lat, out, gmax);
    *members_sum += (double)count * 16.0 / (double)n;
    free(ptok); free(prank); free(groups);
    return n;
}

int main(int argc, char **argv)
{
    if (argc != 5) { fputs("usage: expert-bench DATASET_DIR RESID_F32 S0_F32 NTOKENS\n", stderr); return 2; }
    const unsigned N = (unsigned)atoi(argv[4]);
    float *resid = malloc((size_t)N * INW * sizeof(float)), *s0 = malloc((size_t)N * INW * sizeof(float));
    if (!resid || !s0) bail("allocation failed");
    FILE *f = fopen(argv[2], "rb"); if (!f || fread(resid, sizeof(float), (size_t)N * INW, f) != (size_t)N * INW) bail("resid file"); fclose(f);
    f = fopen(argv[3], "rb");       if (!f || fread(s0, sizeof(float), (size_t)N * INW, f) != (size_t)N * INW) bail("s0 file"); fclose(f);

    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    TransformerSequence *seq = transformer_sequence_create(tr);
    if (!seq) bail("sequence failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    double mark = clk();
    if (mlock((void *)root->direct, root->direct_bytes)) { perror("mlock"); return 1; }
    printf("layer 1: %.2f GB locked in %.2f s, %d threads\n", (double)root->direct_bytes / 1e9, clk() - mark, THREADS);

    /* Real routing and real expert inputs, from the pod itself. */
    float (*latf)[LATW] = malloc((size_t)N * LATW * sizeof(float));
    double (*lat)[LATW] = malloc((size_t)N * LATW * sizeof(double));
    unsigned (*sel)[16] = malloc((size_t)N * 16 * sizeof(unsigned));
    float (*ref)[16][LATW] = malloc((size_t)N * 16 * LATW * sizeof(float));
    float (*out)[16][LATW] = malloc((size_t)N * 16 * LATW * sizeof(float));
    float (*ref2)[16][LATW] = malloc((size_t)N * 16 * LATW * sizeof(float));
    if (!latf || !lat || !sel || !ref || !out || !ref2) bail("allocation failed");
    float layer_out[INW];
    mark = clk();
    for (unsigned t = 0; t < N; t++) {
        if (!transformer_process(tr, seq, resid + (size_t)t * INW, s0 + (size_t)t * INW, layer_out)) bail("process failed");
        memcpy(latf[t], seq->latent, sizeof latf[t]);
        for (unsigned r = 0; r < 16; r++) sel[t][r] = seq->selected[r];
        for (unsigned i = 0; i < LATW; i++) lat[t][i] = (double)latf[t][i];
    }
    printf("captured routing for %u tokens through the pod in %.1f s\n", N, clk() - mark);
    {
        static unsigned seen[896]; unsigned distinct = 0;
        for (unsigned t = 0; t < N; t++) for (unsigned r = 0; r < 16; r++) if (!seen[sel[t][r]]++) distinct++;
        printf("%u of 896 experts used over the %u tokens\n\n", distinct, N);
    }

    /* A: shipped arrangement. Pass 1 makes the reference outputs, pass 2 is timed. */
    for (int pass = 0; pass < 2; pass++) {
        float (*dst)[16][LATW] = pass ? ref2 : ref;
        mark = clk();
        for (unsigned t = 0; t < N; t++) {
            memcpy(seq->latent, latf[t], sizeof seq->latent);
            for (unsigned r = 0; r < 16; r++) {
                if (!transformer_expert(tr, seq, sel[t][r])) bail("reference expert failed");
                memcpy(dst[t][r], seq->expert_down, sizeof dst[t][r]);
            }
        }
        if (pass) {
            double spent = clk() - mark;
            printf("%-34s %8.3f ms/token   (%.0f GB/s of expert weights)   reference %s\n",
                "A  pod, threads share one expert", 1000 * spent / N, (double)N * 16 * ROOT_EXPERT_RAW / spent / 1e9,
                memcmp(ref, ref2, (size_t)N * 16 * LATW * sizeof(float)) ? "NOT repeatable" : "repeatable");
            printf("\n%-34s %-3s %10s %10s %9s %12s %s\n", "arrangement", "G", "ms/token", "GB/s", "vs A", "tokens/expert", "bits vs A");
            fflush(stdout);
            const double a_ms = 1000 * spent / N;
            for (unsigned W = 1; W <= N; W = W < 8 ? 8 : W * 4 > N && W < N ? N : W * 4) {
                for (int G = 1; G <= 8; G *= 2) {
                    if (W == 1 && G > 1) continue;
                    memset(out, 0, (size_t)N * 16 * LATW * sizeof(float));
                    for (int round = 0; round < 2; round++) {          /* round 0 warms, round 1 is timed */
                        double members = 0, bytes = 0; unsigned windows = 0;
                        mark = clk();
                        for (unsigned first = 0; first < N; first += W) {
                            unsigned count = N - first < W ? N - first : W;
                            if (W == 1) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(THREADS)
                                for (unsigned r = 0; r < 16; r++) {
                                    unsigned token = first, rank = r;
                                    expert_apply(root, sel[first][r], &token, &rank, 1, lat, out, 1);
                                }
                                members += 1; bytes += 16.0 * ROOT_EXPERT_RAW; windows++;
                            } else {
                                unsigned used = run_window(root, first, count, sel, lat, out, G, &members);
                                bytes += (double)used * ROOT_EXPERT_RAW; windows++;
                            }
                        }
                        double spent2 = clk() - mark;
                        if (round == 0) continue;
                        int same = !memcmp(ref, out, (size_t)N * 16 * LATW * sizeof(float));
                        char name[64];
                        snprintf(name, sizeof name, W == 1 ? "B  one expert per thread, 1 token" : "C  grouped by expert, window %u", W);
                        printf("%-34s %-3d %10.3f %10.1f %8.2fx %12.2f %s\n", name, G, 1000 * spent2 / N, bytes / spent2 / 1e9,
                            a_ms / (1000 * spent2 / N), members / windows, same ? "identical" : "DIFFERENT");
                        fflush(stdout);
                    }
                }
                if (W == N) break;
            }
        }
    }
    return 0;
}
