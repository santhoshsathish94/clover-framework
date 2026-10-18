#define _GNU_SOURCE
/* Shows how text becomes token ids: one line of input per test string.   tok-show TOKFILES_DIR LINES_FILE */
#include "k3_tok.h"

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: tok-show TOKFILES_DIR LINES_FILE\n", stderr); return 2; }
    Tok tok; k3_tok_load(&tok, argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { fputs("cannot read lines file\n", stderr); return 1; }
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        if (len && line[len - 1] == '\n') line[--len] = 0;
        int ids[1024];
        int n = tok_encode(&tok, line, (int)len, ids, 1024);
        printf("\"%s\"  ->  %d token%s:", line, n, n == 1 ? "" : "s");
        for (int i = 0; i < n; i++) {
            char piece[256];
            int l = tok_decode(&tok, &ids[i], 1, piece, sizeof piece);
            printf("  %d=\"%.*s\"", ids[i], l, piece);
        }
        printf("\n");
    }
    fclose(f);
    return 0;
}
