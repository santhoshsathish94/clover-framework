#include <omp.h>

typedef struct {
    unsigned rows, blocks, workers;
    uint64_t values, payload_bytes;
    unsigned char blocks_hash[32];
} FruitPass;

static FruitPass fruit_passes[2];
static unsigned fruit_pass_count;

void fruit(float *output, const float *input, int expression)
{
    if (!fruit_enabled() || expression < 0 || expression > 1 || fruit_passes[expression].rows)
        die("fruit projection state invalid");
    const double started = now_s();
    const unsigned blocks = VOCAB / 16;
    unsigned char *hashes = calloc(blocks, 32), *seen = calloc(blocks, 1);
    if (!hashes || !seen || VOCAB % 16) die("fruit coverage allocation/dimensions invalid");
    uint64_t payload_bytes = 0;
    unsigned workers = 0;
#pragma omp parallel reduction(+:payload_bytes)
    {
        SeedInput reader;
        seed_open(&reader, getenv("K3_FRUIT_HEAD"));
        if (reader.rows != VOCAB || reader.width != E || reader.block_rows != 16 || reader.blocks != blocks)
            die("fruit output dimensions differ");
        char identity[65];
        for (unsigned byte = 0; byte < 32; byte++) snprintf(identity + byte * 2, 3, "%02x", reader.source_sha[byte]);
        if (strcmp(identity, "11c1f1c09a8e0db55547b5e68ebfd1d8e3b503bee56c4e1312ef55ecd3e5580f"))
            die("fruit output tensor identity differs");
#pragma omp single
        workers = (unsigned)omp_get_num_threads();
#pragma omp for schedule(static)
        for (unsigned block = 0; block < blocks; block++) {
            seed_load_block(&reader, block);
            SHA256(reader.raw, (size_t)16 * E * 2, hashes + (size_t)block * 32);
            seen[block] = 1;
            for (unsigned local_row = 0; local_row < 16; local_row++) {
                const unsigned row = block * 16 + local_row;
                const unsigned char *weight_row = reader.raw + (size_t)local_row * E * 2;
                double lanes[16] = {0};
                for (unsigned start = 0; start < E; start += 16) {
                    for (unsigned lane = 0; lane < 16; lane++) {
                        const unsigned coordinate = start + lane;
                        const unsigned word = weight_row[coordinate * 2] | (unsigned)weight_row[coordinate * 2 + 1] << 8;
                        const float weight = endpoint_bf16((uint16_t)word);
                        const float value = expression ? endpoint_normalized((int)coordinate) : input[coordinate];
                        lanes[lane] += (double)weight * (double)value;
                    }
                }
                double groups[4];
                for (unsigned group = 0; group < 4; group++)
                    groups[group] = (lanes[group] + lanes[group + 4]) + (lanes[group + 8] + lanes[group + 12]);
                output[row] = (float)((groups[0] + groups[1]) + (groups[2] + groups[3]));
            }
        }
        payload_bytes += reader.payload_bytes_read;
        seed_close(&reader);
    }
    for (unsigned block = 0; block < blocks; block++) if (!seen[block]) die("fruit block not consumed");
    FruitPass *pass = &fruit_passes[expression];
    pass->rows = VOCAB;
    pass->blocks = blocks;
    pass->workers = workers;
    pass->values = (uint64_t)VOCAB * E;
    pass->payload_bytes = payload_bytes;
    SHA256(hashes, (size_t)blocks * 32, pass->blocks_hash);
    fruit_pass_count++;
    free(hashes);
    free(seen);
    if (!expression) {
        op_add(OP_B, started, (int64_t)VOCAB * E * 2);
    }
}

static void fruit_report(void)
{
    if (!fruit_enabled()) return;
    const unsigned expected_passes = (endpoint_mode() & 2) ? 2 : 1;
    if (fruit_pass_count != expected_passes || !fruit_guard_applied ||
        (getenv("K3_FRUIT_GUARD_PROBE") && !fruit_guard_probe_passed)) die("fruit not fully exercised");
    const char *path = getenv("K3_FRUIT_REPORT");
    if (!path) die("K3_FRUIT_REPORT required");
    FILE *report = fopen(path, "wb");
    if (!report) die("fruit report open failed");
    fprintf(report, "{\"gate\":\"PASS\",\"original_head_protected_bytes\":%zu,"
            "\"unprotected_boundary_bytes\":%zu,\"guard_probe_sigsegv\":%s,\"passes\":[",
            fruit_guard_bytes, fruit_boundary_bytes, fruit_guard_probe_passed ? "true" : "false");
    for (unsigned index = 0; index < expected_passes; index++) {
        const FruitPass *pass = &fruit_passes[index];
        char hash[65];
        for (unsigned byte = 0; byte < 32; byte++) snprintf(hash + byte * 2, 3, "%02x", pass->blocks_hash[byte]);
        fprintf(report, "%s{\"consumer\":\"%s\",\"rows\":%u,\"values\":%llu,\"blocks\":%u,"
                "\"workers\":%u,\"payload_bytes_read\":%llu,\"ordered_block_hashes_sha256\":\"%s\"}",
                index ? "," : "", index ? "tail_expression" : "head_projection", pass->rows,
                (unsigned long long)pass->values, pass->blocks, pass->workers,
                (unsigned long long)pass->payload_bytes, hash);
    }
    fprintf(report, "]}\n");
    if (fclose(report)) die("fruit report close failed");
}