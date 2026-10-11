#define _GNU_SOURCE
/* What each stage actually does to the vector, rather than how long it takes.

   transformer_process records the layer input as stage 0 and the residual after every
   one of the 120 stages, plus the snapshots at 1000+slot, the fold weights at
   2000+stage, the router weights at 3000 and what the fold produced at 4000+stage.
   A record is four int32 (owner, local, position, count) then count float32.

   For each record this prints the L2 norm, the cosine against the layer input, and the
   largest single coordinate change since the previous residual, so a stage that does
   nothing to the residual is visible as plainly as one that rewrites it. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 7168 };

static double norm_of(const float *v, unsigned n)
{
    double sum = 0.0;
    for (unsigned i = 0; i < n; i++) sum += (double)v[i] * (double)v[i];
    return sqrt(sum);
}

static double cosine(const float *a, const float *b, unsigned n)
{
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (unsigned i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: vector-walk DUMP [POSITION]\n", stderr); return 2; }
    const long want = argc > 2 ? atol(argv[2]) : 0;
    FILE *file = fopen(argv[1], "rb");
    if (!file) { perror("open"); return 1; }

    float *input = malloc(WIDTH * sizeof(float));
    float *previous = malloc(WIDTH * sizeof(float));
    float *values = malloc(32768 * sizeof(float));
    if (!input || !previous || !values) return 1;
    int have_input = 0, have_previous = 0;
    unsigned long shown = 0;

    printf("position %ld\n", want);
    printf("%6s %7s %12s %10s %12s  %s\n", "stage", "count", "L2 norm", "cos(input)", "max change", "note");
    for (;;) {
        int32_t header[4];
        if (fread(header, sizeof header, 1, file) != 1) break;
        const int local = header[1], position = header[2];
        const unsigned count = (unsigned)header[3];
        if (count > 32768) { fputs("record too wide\n", stderr); break; }
        if (fread(values, sizeof(float), count, file) != count) break;
        if (position != want) continue;

        const char *note = "";
        if (local == 0) { memcpy(input, values, count * sizeof(float)); have_input = 1; note = "layer input"; }
        else if (local >= 1000 && local < 1008) note = "snapshot in";
        else if (local >= 2000 && local < 3000) note = "fold weights";
        else if (local == 3000) note = "router weights";
        else if (local >= 4000 && local < 5000) note = "fold produced";

        double change = -1.0;
        const int residual = local > 0 && local <= 120;
        if (residual && have_previous && count == WIDTH)
            for (unsigned i = 0; i < count; i++) {
                double d = fabs((double)values[i] - (double)previous[i]);
                if (d > change) change = d;
            }

        double cos_input = have_input && count == WIDTH ? cosine(values, input, count) : 0.0;
        printf("%6d %7u %12.4f ", local, count, norm_of(values, count));
        if (count == WIDTH && have_input) printf("%10.6f ", cos_input); else printf("%10s ", "-");
        if (change >= 0.0) printf("%12.6g  ", change); else printf("%12s  ", "-");
        printf("%s", note);
        if (count <= 16) {
            printf("  [");
            for (unsigned i = 0; i < count; i++) printf("%s%.4f", i ? " " : "", values[i]);
            printf("]");
        }
        printf("\n");

        if (residual && count == WIDTH) { memcpy(previous, values, count * sizeof(float)); have_previous = 1; }
        if (local == 0 && count == WIDTH) { memcpy(previous, values, count * sizeof(float)); have_previous = 1; }
        shown++;
    }
    printf("\n%lu records for position %ld\n", shown, want);
    fclose(file);
    return 0;
}
