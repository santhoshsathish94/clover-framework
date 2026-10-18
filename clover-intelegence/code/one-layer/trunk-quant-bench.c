#define _GNU_SOURCE
#define TRANSFORMER_NO_MAIN
/* How well can the trunk weights be held in 4 bits? Layer 1's gate matrix (first 2,048 of 12,288 rows x 7,168 columns, the real 8-bit codes) is
   re-quantised in several ways; for each: relative RMS error of the weights, and of the matrix-vector product with random activations
   (uniform and heavy-tailed), and the bits per weight.

     trunk-quant-bench DATASET_DIR                                                      */
#include "transformer-1.c"
#include <time.h>

static const float fp4[8] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6 };
static const float nf4[16] = { -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f, 0.0796f, 0.1609f, 0.2461f, 0.3379f, 0.4407f, 0.5626f, 0.7230f, 1.0f };

/* returns the quantised-and-dequantised value of q (an int8 code) for the group whose largest |code| is m; `mode` picks the format */
typedef struct { const char *name; int mode; unsigned group; double bits; } Fmt;
enum { UNI_INT_STEP, UNI_FLOAT, FP4_FLOAT, NF4_FLOAT, UNI5_FLOAT, UNI6_FLOAT };

static void quant_group(const int8_t *q, unsigned g, int mode, float *out)
{
    float m = 0; for (unsigned j = 0; j < g; j++) { float a = fabsf((float)q[j]); if (a > m) m = a; }
    if (m == 0) { for (unsigned j = 0; j < g; j++) out[j] = 0; return; }
    switch (mode) {
    case UNI_INT_STEP: { const int step = ((int)m + 6) / 7; for (unsigned j = 0; j < g; j++) { int v = (int)lrintf((float)q[j] / step); v = v < -8 ? -8 : v > 7 ? 7 : v; out[j] = (float)(v * step); } break; }
    case UNI_FLOAT: { const float s = m / 7.0f; for (unsigned j = 0; j < g; j++) { int v = (int)lrintf(q[j] / s); v = v < -8 ? -8 : v > 7 ? 7 : v; out[j] = v * s; } break; }
    case UNI5_FLOAT: { const float s = m / 15.0f; for (unsigned j = 0; j < g; j++) { int v = (int)lrintf(q[j] / s); v = v < -16 ? -16 : v > 15 ? 15 : v; out[j] = v * s; } break; }
    case UNI6_FLOAT: { const float s = m / 31.0f; for (unsigned j = 0; j < g; j++) { int v = (int)lrintf(q[j] / s); v = v < -32 ? -32 : v > 31 ? 31 : v; out[j] = v * s; } break; }
    case FP4_FLOAT: { const float s = m / 6.0f;
        for (unsigned j = 0; j < g; j++) { const float a = fabsf(q[j]) / s; int best = 0; float bd = 1e30f;
            for (int k = 0; k < 8; k++) { const float d = fabsf(a - fp4[k]); if (d < bd) { bd = d; best = k; } }
            out[j] = (q[j] < 0 ? -fp4[best] : fp4[best]) * s; } break; }
    case NF4_FLOAT: { const float s = m;
        for (unsigned j = 0; j < g; j++) { const float a = q[j] / s; int best = 0; float bd = 1e30f;
            for (int k = 0; k < 16; k++) { const float d = fabsf(a - nf4[k]); if (d < bd) { bd = d; best = k; } }
            out[j] = nf4[best] * s; } break; }
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: trunk-quant-bench DATASET_DIR\n", stderr); return 2; }
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) { fputs("open failed\n", stderr); return 1; }
    const TransformerRecord *rec = &tr->records[6];
    const unsigned cols = rec->columns, rows = 2048;
    const int8_t *ids = (const int8_t *)(rec->data + (size_t)rec->rows * 4);
    static const Fmt fmts[] = {
        {"current: uniform 4-bit, integer step, group 64", UNI_INT_STEP, 64, 4.125}, {"uniform 4-bit, float scale, group 64", UNI_FLOAT, 64, 4.25},
        {"uniform 4-bit, float scale, group 32", UNI_FLOAT, 32, 4.5}, {"FP4 (E2M1), float scale, group 64", FP4_FLOAT, 64, 4.25},
        {"FP4 (E2M1), float scale, group 32", FP4_FLOAT, 32, 4.5}, {"NF4 (normal-float), float scale, group 64", NF4_FLOAT, 64, 4.25},
        {"NF4, group 32", NF4_FLOAT, 32, 4.5}, {"uniform 5-bit, float scale, group 64", UNI5_FLOAT, 64, 5.25}, {"uniform 6-bit, float scale, group 64", UNI6_FLOAT, 64, 6.25} };
    float *x[2] = { malloc(cols * 4), malloc(cols * 4) };
    uint64_t s = 88172645463325252ULL;
    for (unsigned c = 0; c < cols; c++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; const double u = (double)(s >> 11) / 9007199254740992.0 - 0.5;
        x[0][c] = (float)(2.0 * u);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17; const double v = (double)(s >> 11) / 9007199254740992.0;
        x[1][c] = (float)(((v < 0.02) ? 12.0 : 1.0) * (u * 2.0));   /* 2% outliers 12x larger */
    }
    double wn = 0; for (size_t i = 0; i < (size_t)rows * cols; i++) wn += (double)ids[i] * ids[i];
    printf("layer 1 gate matrix, %u rows x %u columns, real 8-bit codes\n", rows, cols);
    printf("  %-52s %6s  %14s  %14s %14s\n", "format", "bits", "weight error", "output error", "output error");
    printf("  %-52s %6s  %14s  %14s %14s\n", "", "per w", "(rel RMS)", "uniform x", "x with outliers");
    float *buf = malloc(cols * 4);
    for (unsigned f = 0; f < sizeof fmts / sizeof *fmts; f++) {
        double we = 0, oe[2] = {0, 0}, on[2] = {0, 0};
        for (unsigned r = 0; r < rows; r++) {
            const int8_t *q = ids + (size_t)r * cols;
            double dot_e[2] = {0, 0}, dot_x[2] = {0, 0};
            for (unsigned g0 = 0; g0 < cols; g0 += fmts[f].group) {
                quant_group(q + g0, fmts[f].group, fmts[f].mode, buf);
                for (unsigned j = 0; j < fmts[f].group; j++) {
                    const double d = (double)buf[j] - q[g0 + j]; we += d * d;
                    for (int k = 0; k < 2; k++) { dot_e[k] += (double)buf[j] * x[k][g0 + j]; dot_x[k] += (double)q[g0 + j] * x[k][g0 + j]; }
                }
            }
            for (int k = 0; k < 2; k++) { const double d = dot_e[k] - dot_x[k]; oe[k] += d * d; on[k] += dot_x[k] * dot_x[k]; }
        }
        printf("  %-52s %6.3f  %14.4e  %14.4e %14.4e\n", fmts[f].name, fmts[f].bits, sqrt(we / wn), sqrt(oe[0] / on[0]), sqrt(oe[1] / on[1]));
    }
    return 0;
}
