/* gate-profile.c - recover the KDA output gate from records already captured.
 *
 * The engine computes gate[c] = attention[c] * sigmoid(Wx)[c], and the trace holds the
 * attention after its per-head norm (local 5007) and the result (local 5008). Dividing
 * one by the other gives back sigmoid(Wx) without recording it, so the gate's actual
 * distribution is available from captures that are already on disk.
 *
 * The norm ratio reported earlier is a summary of this and can hide the shape: a gate
 * that is 0.9 everywhere and one that is 1.0 on a tenth of its channels and 0 elsewhere
 * both shrink the norm, for completely different reasons.
 *
 * build: gcc -O2 -o gate-profile gate-profile.c -lm
 * run:   ./gate-profile <dump.bin> <position>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 16384

static int find_record(const char *path, int local, int position, float *values, unsigned *count)
{
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    int header[4];
    while (fread(header, sizeof header, 1, file) == 1) {
        unsigned length = (unsigned)header[3];
        if (length > WIDTH) { fclose(file); return 0; }
        if (header[1] == local && header[2] == position) {
            int ok = fread(values, sizeof *values, length, file) == length;
            *count = length;
            fclose(file);
            return ok;
        }
        if (fseek(file, (long)length * 4, SEEK_CUR)) break;
    }
    fclose(file);
    return 0;
}

static int ascending(const void *left, const void *right)
{
    float a = *(const float *)left, b = *(const float *)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: %s <dump.bin> <position>\n", argv[0]); return 2; }
    int position = atoi(argv[2]);

    static float attention[WIDTH], gated[WIDTH], ratio[WIDTH];
    unsigned na = 0, ng = 0;
    if (!find_record(argv[1], 5007, position, attention, &na)) { printf("NA\tno 5007\n"); return 1; }
    if (!find_record(argv[1], 5008, position, gated, &ng)) { printf("NA\tno 5008\n"); return 1; }
    if (na != ng) { printf("NA\tlength %u vs %u\n", na, ng); return 1; }

    /* Where the attention is essentially zero the quotient is noise, not a gate value. */
    unsigned kept = 0, open_half = 0, shut_tenth = 0;
    double total = 0.0;
    for (unsigned i = 0; i < na; i++) {
        if (fabsf(attention[i]) < 1e-6f) continue;
        float value = gated[i] / attention[i];
        if (!isfinite(value)) continue;
        ratio[kept++] = value;
        total += value;
        if (value > 0.5f) open_half++;
        if (value < 0.1f) shut_tenth++;
    }
    if (!kept) { printf("NA\tno usable coordinates\n"); return 1; }
    qsort(ratio, kept, sizeof *ratio, ascending);

    /* The mean gate and the fraction of the norm that survives it disagree wildly
       between layers, which can only happen if the gate is correlated with where the
       attention actually has magnitude. Energy-weighted RMS is exactly the surviving
       norm fraction; plain RMS is what it would be if the two were independent. Their
       ratio is the alignment, above one when the gate opens on the coordinates the
       attention is using and below one when it closes on them. */
    double plain = 0.0, weighted = 0.0, energy = 0.0;
    for (unsigned i = 0; i < na; i++) {
        if (fabsf(attention[i]) < 1e-6f) continue;
        double g = gated[i] / attention[i];
        if (!isfinite(g)) continue;
        double a2 = (double)attention[i] * attention[i];
        plain += g * g;
        weighted += g * g * a2;
        energy += a2;
    }
    double plain_rms = sqrt(plain / kept);
    double weighted_rms = energy > 0.0 ? sqrt(weighted / energy) : 0.0;

    printf("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.3f\t%.3f\t%u\t%.4f\t%.4f\t%.3f\n",
        total / kept,                 /* mean gate */
        ratio[kept / 2],              /* median */
        ratio[kept / 10],             /* 10th percentile */
        ratio[kept - 1 - kept / 10],  /* 90th percentile */
        ratio[kept - 1],              /* max */
        (double)open_half / kept,     /* fraction above 0.5 */
        (double)shut_tenth / kept,    /* fraction below 0.1 */
        kept,
        plain_rms,                    /* RMS gate, unweighted */
        weighted_rms,                 /* RMS gate weighted by attention energy */
        plain_rms > 0.0 ? weighted_rms / plain_rms : 0.0);
    return 0;
}
