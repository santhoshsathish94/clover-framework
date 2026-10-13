/* variance-profile.c - how much of a layer's attention output is a constant?
 *
 * Substituting one cached vector works at every layer individually, which was taken as
 * "the attention is near-constant". That phrasing may be hiding the shape: a vector can
 * sit close to a fixed direction while still carrying a small varying part that does
 * the useful work. Decomposing it says which.
 *
 * Pools every position from every dump given, computes the mean vector, and reports how
 * much energy sits in the mean against how much is left in the deviations. A layer that
 * is genuinely constant puts nearly all of it in the mean.
 *
 * build: gcc -O2 -o variance-profile variance-profile.c -lm
 * run:   ./variance-profile <local> <dump.bin> [more dumps...]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 16384
#define SAMPLES 64

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <local> <dump.bin> [more...]\n", argv[0]); return 2; }
    int want = atoi(argv[1]);

    static float sample[SAMPLES][WIDTH], mean[WIDTH];
    unsigned count = 0, width = 0;

    for (int a = 2; a < argc && count < SAMPLES; a++) {
        FILE *file = fopen(argv[a], "rb");
        if (!file) continue;
        int header[4];
        while (count < SAMPLES && fread(header, sizeof header, 1, file) == 1) {
            unsigned length = (unsigned)header[3];
            if (length > WIDTH) break;
            if (header[1] == want) {
                if (fread(sample[count], sizeof(float), length, file) != length) break;
                if (!width) width = length;
                if (width == length) count++;
                continue;
            }
            if (fseek(file, (long)length * 4, SEEK_CUR)) break;
        }
        fclose(file);
    }
    if (count < 2) { printf("NA\tonly %u samples\n", count); return 1; }

    for (unsigned i = 0; i < width; i++) {
        double total = 0.0;
        for (unsigned s = 0; s < count; s++) total += sample[s][i];
        mean[i] = (float)(total / count);
    }

    double mean_energy = 0.0;
    for (unsigned i = 0; i < width; i++) mean_energy += (double)mean[i] * mean[i];

    /* Deviation energy averaged over samples, and the worst single deviation, because an
       average can hide one position behaving differently from the rest. */
    double deviation_total = 0.0, deviation_worst = 0.0, sample_total = 0.0;
    for (unsigned s = 0; s < count; s++) {
        double deviation = 0.0, magnitude = 0.0;
        for (unsigned i = 0; i < width; i++) {
            double d = (double)sample[s][i] - mean[i];
            deviation += d * d;
            magnitude += (double)sample[s][i] * sample[s][i];
        }
        deviation_total += deviation;
        sample_total += magnitude;
        if (deviation > deviation_worst) deviation_worst = deviation;
    }
    double deviation_mean = deviation_total / count;
    double energy_mean = sample_total / count;

    printf("%u\t%.4f\t%.4f\t%.4f\t%.4f\n",
        count,
        sqrt(mean_energy),                      /* norm of the mean vector */
        sqrt(deviation_mean),                   /* rms deviation from it */
        mean_energy / energy_mean,              /* share of energy in the mean */
        sqrt(deviation_worst));                 /* worst single deviation */
    return 0;
}
