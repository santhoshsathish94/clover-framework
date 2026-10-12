/* vector-compare.c - how far apart two prompts are at each layer.
 *
 * The norm trajectory turned out to be nearly prompt independent, so magnitude is
 * not where the content lives. This reads the same record from two CLOVER_VECTORS
 * dumps and reports the angle between them, which is.
 *
 * build: gcc -O2 -o vector-compare vector-compare.c -lm
 * run:   ./vector-compare <a.bin> <b.bin> <local> <position> [position-in-b]
 *
 * Passing the same file twice with two positions compares one token against
 * another inside a single prompt.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define WIDTH 7168

/* Returns 1 and fills values when the record is found. */
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

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s <a.bin> <b.bin> <local> <position> [position-in-b]\n", argv[0]);
        return 2;
    }
    int local = atoi(argv[3]), position = atoi(argv[4]);
    int other = argc == 6 ? atoi(argv[5]) : position;

    static float a[WIDTH], b[WIDTH];
    unsigned na = 0, nb = 0;
    if (!find_record(argv[1], local, position, a, &na)) { printf("NA\tmissing\t%s\n", argv[1]); return 1; }
    if (!find_record(argv[2], local, other, b, &nb)) { printf("NA\tmissing\t%s\n", argv[2]); return 1; }
    if (na != nb) { printf("NA\tlength %u vs %u\n", na, nb); return 1; }

    double dot = 0.0, sa = 0.0, sb = 0.0, far = 0.0;
    for (unsigned i = 0; i < na; i++) {
        dot += (double)a[i] * b[i];
        sa += (double)a[i] * a[i];
        sb += (double)b[i] * b[i];
        double gap = fabs((double)a[i] - b[i]);
        if (gap > far) far = gap;
    }
    double cosine = (sa > 0.0 && sb > 0.0) ? dot / (sqrt(sa) * sqrt(sb)) : 0.0;
    printf("%.6f\t%.4f\t%.4f\t%.4f\n", cosine, sqrt(sa), sqrt(sb), far);
    return 0;
}
