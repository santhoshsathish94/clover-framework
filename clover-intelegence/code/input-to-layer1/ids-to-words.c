#define _GNU_SOURCE
/* ids -> words: writes the text that the recorded token ids stand for, then tokenises that text again and checks
   that the same ids come back (so the recorded layer values can be used to check a run that starts from words).

     ids-to-words TOKFILES IDS_FILE WORDS_OUT                                                              */
#include "k3_tok.h"

int main(int argc, char **argv)
{
    if (argc != 4) { fputs("usage: ids-to-words TOKFILES IDS_FILE WORDS_OUT\n", stderr); return 2; }
    Tok tok; k3_tok_load(&tok, argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { fputs("cannot read ids\n", stderr); return 1; }
    static int ids[8192]; int n = 0, v;
    while (n < 8192 && fscanf(f, "%d", &v) == 1) { ids[n++] = v; int ch = fgetc(f); if (ch != ',' && ch != '\n' && ch != ' ' && ch != EOF) ungetc(ch, f); }
    fclose(f);
    const int cap = n * 64 + 64;
    char *text = malloc((size_t)cap);
    const int len = tok_decode(&tok, ids, n, text, cap);
    f = fopen(argv[3], "wb");
    if (!f || fwrite(text, 1, (size_t)len, f) != (size_t)len) { fputs("cannot write words\n", stderr); return 1; }
    fclose(f);
    int *back = malloc((size_t)(n + 64) * sizeof *back);
    const int m = tok_encode(&tok, text, len, back, n + 64);
    printf("%d ids -> %d bytes of text -> %d ids again; same ids: %s\n", n, len, m, m == n && !memcmp(back, ids, (size_t)n * sizeof *ids) ? "yes" : "NO");
    return 0;
}
