#define _GNU_SOURCE
#define SERVER_NO_MAIN
/* One projection, opened up with real numbers: layer 0, first token of the real prompt.
   Shows the matrices (rows x columns, scale, any bias), one weight row against the input vector
   term by term, and the same dot product done independently in double precision.

     projection-show CODE_DIR TOKEN_ID                                                      */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "numeric-table.h"
#include "server.c"

enum { W = SERVER_WIDTH, R = SERVER_ROWS };

static void row_stats(const unsigned char *ids, unsigned cols, const float *palette)
{
    int lo = 1000, hi = -1000; unsigned zeros = 0; double mean_abs = 0;
    for (unsigned i = 0; i < cols; i++) {
        int v = (int)palette[ids[i]];
        if (v < lo) lo = v; if (v > hi) hi = v;
        zeros += v == 0; mean_abs += fabs((double)v);
    }
    printf("    codes in this row: min %d, max %d, mean |code| %.1f, zero codes %u of %u\n", lo, hi, mean_abs / cols, zeros, cols);
}

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: projection-show CODE_DIR TOKEN_ID\n", stderr); return 2; }
    char qkv[4096], trunk[4096], seed[4096];
    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", argv[1]);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", argv[1]);
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", argv[1]);
    Server *s = NULL;
    if (!server_open(qkv, trunk, &s)) { fputs("server_open failed\n", stderr); return 1; }
    ServerInput *in = server_input_open(seed);
    float emb[W], x[W];
    if (!in || !server_input(in, (uint32_t)atoi(argv[2]), emb)) { fputs("lookup failed\n", stderr); return 1; }
    server_normalize(x, emb, s->qkv + 88, W);

    printf("token id %s -> embedding vector of %d numbers; first 6: ", argv[2], W);
    for (int i = 0; i < 6; i++) printf("% .5f ", emb[i]);
    printf("\nafter the layer's input norm (RMS-normalise, times a learned gain per coordinate), first 6: ");
    for (int i = 0; i < 6; i++) printf("% .5f ", x[i]);
    printf("\n\nThe matrices layer 0 applies (each row is one weight array, stored as 8-bit codes plus one float scale per row):\n");
    static const struct { unsigned id; const char *what; } list[] = {
        {11, "beta        (one write strength per head)"}, {12, "decay, down (7168 -> 128)"},
        {13, "decay, up   (128 -> 12288)"}, {6, "output gate (7168 -> 12288)"},
        {7, "attention output projection (12288 -> 7168)"}, {26, "dense gate (7168 -> 33792)"},
        {27, "dense up   (7168 -> 33792)"}, {28, "dense down (33792 -> 7168)"}};
    for (unsigned i = 0; i < sizeof list / sizeof *list; i++) {
        const ServerRecord *r = &s->records[list[i].id];
        printf("  record %2u  %-48s rows %5u x columns %5u   bytes %10llu = rows*4 (scales) + rows*columns (codes): %s\n",
            list[i].id, list[i].what, r->rows, r->columns, (unsigned long long)r->length,
            (unsigned long long)r->length == (unsigned long long)r->rows * 4 + (unsigned long long)r->rows * r->columns ? "yes, no bias" : "NO, something else is stored");
    }
    printf("  Q, K, V    (7168 -> 12288 each; per row: 4-byte scale, 4 convolution taps (16 bytes), then 7168 codes = %d bytes per row)\n", SERVER_QKV_COMPONENT);

    const unsigned char *rec = s->qkv + SERVER_QKV_PREFIX;
    const float scale = server_f32(rec);
    const unsigned char *ids = rec + 20;
    printf("\nQ, row 0 (the first of 12288 weight arrays):\n  scale %.9g; convolution taps %.5f %.5f %.5f %.5f\n",
        scale, server_f32(rec + 4), server_f32(rec + 8), server_f32(rec + 12), server_f32(rec + 16));
    row_stats(ids, W, s->palette);
    printf("  first 6 terms, weight code x input coordinate:\n");
    for (int i = 0; i < 6; i++) printf("    column %d: code %4d x % .5f = % .6f\n", i, (int)s->palette[ids[i]], x[i], s->palette[ids[i]] * x[i]);
    double exact = 0;
    for (unsigned i = 0; i < W; i++) exact += (double)s->palette[ids[i]] * (double)x[i];
    float kernel = server_project_row(x, ids, W, s->palette, scale);
    printf("  all %d terms added up, times the row's scale: %.9g (the server's kernel, float)\n", W, kernel);
    printf("  the same sum in double precision, times the scale: %.9g (relative difference %.2e)\n", exact * scale,
        fabs((double)kernel - exact * scale) / fabs(exact * scale));

    printf("\nOne row of the matrix gives ONE output number. 12288 rows give the 12288-number Q vector = 96 heads x 128.\nFirst 6 outputs (raw Q, before the convolution): ");
    for (int row = 0; row < 6; row++) {
        const unsigned char *rr = s->qkv + SERVER_QKV_PREFIX + (size_t)row * 3 * SERVER_QKV_COMPONENT;
        printf("% .5f ", server_project_row(x, rr + 20, W, s->palette, server_f32(rr)));
    }
    printf("\n");
    {   /* Same multiplications, different order of adding: all 12288 Q rows. */
        unsigned differ_seq = 0, differ_fma = 0; double worst_seq = 0, worst_fma = 0;
        double sq_exact = 0, sq_lane = 0, sq_seq = 0, max_lane = 0, max_seq = 0, max_diff = 0, sq_once = 0, max_once = 0;
        unsigned differ_once = 0;
        for (unsigned row = 0; row < R; row++) {
            const unsigned char *rr = s->qkv + SERVER_QKV_PREFIX + (size_t)row * 3 * SERVER_QKV_COMPONENT;
            const unsigned char *wid = rr + 20; const float sc = server_f32(rr);
            const float lanes16 = server_project_row(x, wid, W, s->palette, sc);
            float seq = 0, fused = 0;   /* one running total, column by column, like adding x_j * column_j into the outputs */
            double truth = 0;
            for (unsigned j = 0; j < W; j++) { seq = seq + s->palette[wid[j]] * x[j]; fused = fmaf(s->palette[wid[j]], x[j], fused); truth += (double)s->palette[wid[j]] * (double)x[j]; }
            seq *= sc; fused *= sc; truth *= sc;
            sq_exact += truth * truth;
            const float once = (float)truth;   /* accumulate in double, round to float once at the end */
            if (memcmp(&once, &lanes16, 4)) differ_once++;
            { double eo = fabs((double)once - truth); sq_once += eo * eo; if (eo > max_once) max_once = eo; }
            double el = fabs((double)lanes16 - truth), es = fabs((double)seq - truth), ed = fabs((double)seq - (double)lanes16);
            sq_lane += el * el; sq_seq += es * es;
            if (el > max_lane) max_lane = el; if (es > max_seq) max_seq = es; if (ed > max_diff) max_diff = ed;
            if (memcmp(&seq, &lanes16, 4)) { differ_seq++; double d = fabs((double)seq - lanes16) / fabs((double)lanes16); if (d > worst_seq) worst_seq = d; }
            if (memcmp(&fused, &lanes16, 4)) { differ_fma++; double d = fabs((double)fused - lanes16) / fabs((double)lanes16); if (d > worst_fma) worst_fma = d; }
        }
        printf("\nSame 12288 x 7168 multiplications, added in a different order (one running total per output, column by column):\n"
               "  multiply then add:    %u of %u outputs differ in the last bits from the server's 16-lane order; worst relative difference %.2e\n"
               "  fused multiply-add:   %u of %u outputs differ; worst relative difference %.2e\n", differ_seq, R, worst_seq, differ_fma, R, worst_fma);
        const double rms_exact = sqrt(sq_exact / R);
        printf("Against the exact answer (double precision), outputs have typical size (RMS) %.5f:\n"
               "  server's 16-lane float order: RMS error %.2e, largest error %.2e\n"
               "  column-by-column float order: RMS error %.2e, largest error %.2e\n"
               "  largest gap between the two float orders: %.2e (%.2e of the typical output size)\n",
               rms_exact, sqrt(sq_lane / R), max_lane, sqrt(sq_seq / R), max_seq, max_diff, max_diff / rms_exact);
        printf("  added in double precision and rounded to float once at the end: RMS error %.2e, largest error %.2e;\n"
               "    %u of %u of those results differ from the server's float result\n",
               sqrt(sq_once / R), max_once, differ_once, R);
    }
    {   /* Why: float addition rounds after every step. */
        volatile float big = 16777216.0f, one = 1.0f;
        float left = (big + one) + one, right = big + (one + one);
        printf("\nFloat32 example: 16777216 + 1 + 1 added left to right = %.0f; adding the two ones first = %.0f\n", left, right);
    }
    {   /* The smallest projection that takes the whole 7168 input: beta, row 0 (of 96). Step by step. */
        const ServerRecord *rb = &s->records[11];
        const unsigned char *bids = rb->data + (size_t)rb->rows * 4;
        const float bscale = server_f32(rb->data);
        float lanes[16] = {0};
        printf("\n=== ONE array x array, from start to finish: beta, row 0 (record 11 has %u rows; this is the first) ===\n", rb->rows);
        printf("input x: %d numbers; weight row 0: %d codes; row scale %.9g\n", W, W, bscale);
        printf("The computer works in blocks of 16 columns: %d blocks. Lane l (0..15) keeps its own running total of columns l, l+16, l+32, ...\n", W / 16);
        for (unsigned block = 0; block < W / 16; block++) {
            const unsigned c = block * 16;
            if (block == 0) printf("\nBlock 0 (columns 0..15): lane, code, x, product, lane total after this block\n");
            for (unsigned l = 0; l < 16; l++) {
                const float before = lanes[l];
                lanes[l] = fmaf(s->palette[bids[c + l]], x[c + l], lanes[l]);
                if (block == 0) printf("  lane %2u: code %4d x % .5f = % .6f   total % .6f\n", l, (int)s->palette[bids[c + l]], x[c + l], s->palette[bids[c + l]] * x[c + l], lanes[l]);
                (void)before;
            }
            if (block == 1) { printf("\nLane totals after block 1 (columns 0..31): "); for (int l = 0; l < 16; l++) printf("% .4f ", lanes[l]); printf("\n"); }
        }
        printf("\nLane totals after all %d blocks (each lane added %d products): ", W / 16, W / 16);
        for (int l = 0; l < 16; l++) printf("% .4f ", lanes[l]);
        float combined[8];
        for (int l = 0; l < 8; l++) combined[l] = lanes[l] + lanes[l + 8];
        float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5], sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7];
        const float total = (sum0 + sum2) + (sum1 + sum3);
        const float out = total * bscale;
        const float kernel = server_project_row(x, bids, W, s->palette, bscale);
        printf("\nFold 16 lanes to 8 (lane l + lane l+8): ");
        for (int l = 0; l < 8; l++) printf("% .4f ", combined[l]);
        printf("\nFold 8 to 4: % .4f % .4f % .4f % .4f ->  then (s0+s2)+(s1+s3) = % .6f\n", sum0, sum1, sum2, sum3, total);
        printf("Times the row scale %.9g = % .9g   (the server's kernel gives % .9g: %s)\n", bscale, out, kernel,
            memcmp(&out, &kernel, 4) ? "DIFFERENT" : "bit-identical");
        printf("Cost of this one output: %d fused multiply-adds (%d vector instructions of 16 at a time), 15 additions to fold the lanes, 1 multiply by the scale, %d weight bytes read.\n", W, W / 16, W);
        printf("beta for head 0 = sigmoid(%.6f) = %.6f; the other 95 rows give the other 95 heads' write strengths from the same x.\n", out, server_sigmoid(out));
    }
    server_input_close(in);
    return 0;
}
