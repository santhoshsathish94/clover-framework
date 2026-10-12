/* gate-detail.c - is the gate itself constant, and is the alignment all heads or a few?
 *
 * The whole-layer alignment at pod 12 barely moves with position or prompt, which is
 * strange for a quantity the engine computes from the input. Two things would explain
 * it and they are distinguishable: the gate vector itself being near-fixed, or the
 * alignment being carried by a handful of heads that happen to be stable.
 *
 * mode "cos"  <a.bin> <posA> <b.bin> <posB>   cosine between the two gate vectors
 * mode "head" <a.bin> <pos>                   per-head alignment, sorted, with a summary
 *
 * build: gcc -O2 -o gate-detail gate-detail.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 16384
#define HEADS 96
#define HEAD 128

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

/* Recovers sigmoid(Wx) from the two recorded vectors, leaving NAN where the attention
   is too small for the quotient to mean anything. */
static int load_gate(const char *path, int position, float *gate, unsigned *count)
{
    static float attention[WIDTH], gated[WIDTH];
    unsigned na = 0, ng = 0;
    if (!find_record(path, 5007, position, attention, &na)) return 0;
    if (!find_record(path, 5008, position, gated, &ng)) return 0;
    if (na != ng) return 0;
    for (unsigned i = 0; i < na; i++)
        gate[i] = fabsf(attention[i]) < 1e-6f ? NAN : gated[i] / attention[i];
    *count = na;
    return 1;
}

static int descending(const void *left, const void *right)
{
    float a = *(const float *)left, b = *(const float *)right;
    return a < b ? 1 : a > b ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s cos <a.bin> <posA> <b.bin> <posB> | head <a.bin> <pos>\n", argv[0]); return 2; }

    if (!strcmp(argv[1], "cos") && argc == 6) {
        static float a[WIDTH], b[WIDTH];
        unsigned na = 0, nb = 0;
        if (!load_gate(argv[2], atoi(argv[3]), a, &na)) { printf("NA\n"); return 1; }
        if (!load_gate(argv[4], atoi(argv[5]), b, &nb)) { printf("NA\n"); return 1; }
        if (na != nb) { printf("NA\n"); return 1; }
        double dot = 0.0, sa = 0.0, sb = 0.0;
        unsigned kept = 0;
        for (unsigned i = 0; i < na; i++) {
            if (isnan(a[i]) || isnan(b[i])) continue;
            dot += (double)a[i] * b[i]; sa += (double)a[i] * a[i]; sb += (double)b[i] * b[i];
            kept++;
        }
        if (!kept || sa <= 0.0 || sb <= 0.0) { printf("NA\n"); return 1; }
        printf("%.6f\t%u\n", dot / (sqrt(sa) * sqrt(sb)), kept);
        return 0;
    }

    if (!strcmp(argv[1], "head") && argc == 4) {
        static float gate[WIDTH], attention[WIDTH], per_head[HEADS], sorted[HEADS], share[HEADS];
        unsigned na = 0, ng = 0;
        if (!load_gate(argv[2], atoi(argv[3]), gate, &ng)) { printf("NA\n"); return 1; }
        if (!find_record(argv[2], 5007, atoi(argv[3]), attention, &na)) { printf("NA\n"); return 1; }
        if (na < HEADS * HEAD) { printf("NA\tshort\n"); return 1; }
        unsigned above = 0;
        double all_energy = 0.0;
        for (unsigned h = 0; h < HEADS; h++) {
            double plain = 0.0, weighted = 0.0, energy = 0.0;
            unsigned kept = 0;
            for (unsigned c = 0; c < HEAD; c++) {
                unsigned i = h * HEAD + c;
                if (isnan(gate[i])) continue;
                double g = gate[i], a2 = (double)attention[i] * attention[i];
                plain += g * g; weighted += g * g * a2; energy += a2; kept++;
            }
            per_head[h] = (kept && energy > 0.0 && plain > 0.0)
                ? (float)(sqrt(weighted / energy) / sqrt(plain / kept)) : NAN;
            share[h] = (float)energy;
            all_energy += energy;
            if (!isnan(per_head[h]) && per_head[h] > 1.0f) above++;
        }

        /* A median near one cannot produce a whole-layer alignment of 1.7 on its own.
           Comparing the plain mean of the head alignments against the same mean weighted
           by each head's share of the attention energy says whether the aligned heads
           are also the ones carrying the signal. */
        double plain_mean = 0.0, weighted_mean = 0.0;
        unsigned counted = 0;
        for (unsigned h = 0; h < HEADS; h++) {
            if (isnan(per_head[h])) continue;
            plain_mean += per_head[h];
            weighted_mean += (double)per_head[h] * share[h];
            counted++;
        }
        plain_mean = counted ? plain_mean / counted : 0.0;
        weighted_mean = all_energy > 0.0 ? weighted_mean / all_energy : 0.0;

        memcpy(sorted, per_head, sizeof sorted);
        qsort(sorted, HEADS, sizeof *sorted, descending);
        printf("heads>1 %2u/96 | max %.3f med %.3f min %.3f | mean %.3f energy-weighted %.3f ratio %.3f\n",
            above, sorted[0], sorted[HEADS / 2], sorted[HEADS - 1],
            plain_mean, weighted_mean, plain_mean > 0.0 ? weighted_mean / plain_mean : 0.0);
        return 0;
    }

    fprintf(stderr, "bad arguments\n");
    return 2;
}
