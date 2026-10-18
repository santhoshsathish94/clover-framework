#define _GNU_SOURCE
/* Prints the structure of the expert weight values: the dictionary, and for several pair tables the 16 values by code. */
#define LUT_AS_LIB
#include "expert-lut-bench.c"

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-dict-print DATASET_DIR\n", stderr); return 2; }
    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    build_dictionary(root);
    printf("dictionary (%u values, sorted):", g_dict_n);
    double sorted[MAXLEV]; memcpy(sorted, g_dict, g_dict_n * sizeof(double));
    for (unsigned i = 0; i < g_dict_n; i++) for (unsigned j = i + 1; j < g_dict_n; j++) if (sorted[j] < sorted[i]) { double t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
    for (unsigned i = 0; i < g_dict_n; i++) printf(" %.9g", sorted[i]);
    printf("\n");
    for (unsigned t = 0; t < root->map_count && t < 400; t += 57) {
        printf("table %3u by low nibble (pair[b][0] for b = 0..15):", t);
        for (unsigned b = 0; b < 16; b++) printf(" %.7g", (double)root->pairs[t][b][0]);
        printf("\n          high nibble (pair[16*b][1], b = 0..15):");
        for (unsigned b = 0; b < 16; b++) printf(" %.7g", (double)root->pairs[t][b * 16][1]);
        printf("\n");
    }
    return 0;
}
