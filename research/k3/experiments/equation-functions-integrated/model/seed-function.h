#ifndef K3_SEED_FUNCTION_H
#define K3_SEED_FUNCTION_H

static unsigned long long seed_function_calls;

void seed(SeedInput *reader, unsigned token, uint16_t output[7168]) {
    if (!reader || !output || reader->rows != 163840 || reader->width != 7168 || token >= reader->rows)
        die("seed lookup dimensions or token invalid");
    seed_read_row(reader, token, output);
    seed_function_calls++;
}

#endif