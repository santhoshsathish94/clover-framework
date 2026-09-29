#include <signal.h>
#include <sys/wait.h>
#include "seed-reader.h"
#include "seed-function.h"

static uint16_t seed_model_rows[NPOS][E];
static int seed_model_ready, seed_guard_applied, seed_guard_probe_passed;
static size_t seed_guard_bytes, seed_boundary_bytes;
static uint64_t seed_model_blocks, seed_model_payload_bytes;

static int seed_model_enabled(void)
{
    return getenv("K3_SEED_INPUT") != NULL;
}

static void seed_model_prepare(const int *ids)
{
    SeedInput input;
    seed_open(&input, getenv("K3_SEED_INPUT"));
    if (input.rows != VOCAB || input.width != E) die("seed model dimensions differ");
    char source_hash[65];
    for (unsigned index = 0; index < 32; index++) snprintf(source_hash + index * 2, 3, "%02x", input.source_sha[index]);
    if (strcmp(source_hash, "4a79cdabdab6826b994aff69d90a72aa35dae32e90f8acaf1ce312c7bdc4a487"))
        die("seed input tensor identity differs");
    for (int position = 0; position < NPOS; position++) {
        if (ids[position] < 0 || ids[position] >= VOCAB) die("seed model token out of range");
        seed(&input, (unsigned)ids[position], seed_model_rows[position]);
    }
    seed_model_blocks = input.blocks_decoded;
    seed_model_payload_bytes = input.payload_bytes_read;
    seed_close(&input);
    const char *path = getenv("K3_SEED_ROWS_REPORT");
    if (!path) die("K3_SEED_ROWS_REPORT required");
    FILE *report = fopen(path, "wb");
    if (!report) die("seed input rows report open");
    for (int position = 0; position < NPOS; position++)
        for (int coordinate = 0; coordinate < E; coordinate++) {
            const unsigned word = seed_model_rows[position][coordinate];
            const unsigned char bytes[2] = {word, word >> 8};
            if (fwrite(bytes, 1, 2, report) != 2) die("seed input rows report write");
        }
    if (fclose(report)) die("seed input rows report close");
    seed_model_ready = 1;
}

static void seed_guard_mapping(int file_id)
{
    if (!seed_model_enabled() || file_id != mrec[0].file_id || seed_guard_applied) return;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || mrec[0].off < 0 || mrec[0].nbytes <= 0 ||
        (uint64_t)mrec[0].off + (uint64_t)mrec[0].nbytes > fsize[file_id]) die("seed guard source range invalid");
    const size_t start = ((size_t)mrec[0].off + (size_t)page - 1) / (size_t)page * (size_t)page;
    const size_t end = ((size_t)mrec[0].off + (size_t)mrec[0].nbytes) / (size_t)page * (size_t)page;
    if (end <= start || mprotect(fmap[file_id] + start, end - start, PROT_NONE)) die("seed source guard failed");
    seed_guard_bytes = end - start;
    seed_boundary_bytes = (size_t)mrec[0].nbytes - seed_guard_bytes;
    seed_guard_applied = 1;
    if (getenv("K3_SEED_GUARD_PROBE")) {
        const pid_t child = fork();
        if (child < 0) die("seed source guard fork failed");
        if (!child) {
            const struct rlimit limit = {0, 0};
            if (setrlimit(RLIMIT_CORE, &limit)) _exit(3);
            const volatile unsigned char *protected_bytes = fmap[file_id] + start;
            const unsigned char observed = *protected_bytes;
            _exit(observed ? 4 : 5);
        }
        int status;
        if (waitpid(child, &status, 0) != child || !WIFSIGNALED(status) || WTERMSIG(status) != SIGSEGV)
            die("seed source guard did not reject read");
        seed_guard_probe_passed = 1;
    }
}

static void seed_model_report(void)
{
    if (!seed_model_enabled()) return;
    if (!seed_model_ready || !seed_guard_applied || (getenv("K3_SEED_GUARD_PROBE") && !seed_guard_probe_passed))
        die("seed integration was not exercised");
    const char *path = getenv("K3_SEED_REPORT");
    if (!path) die("K3_SEED_REPORT required");
    FILE *report = fopen(path, "wb");
    if (!report) die("seed model report open");
    fprintf(report,
        "{\"gate\":\"PASS\",\"source\":\"seed\",\"rows_loaded\":%d,\"values_loaded\":%d,"
        "\"blocks_decoded\":%llu,\"payload_bytes_read\":%llu,\"retained_row_bytes\":%zu,"
        "\"original_embedding_protected_bytes\":%zu,\"unprotected_boundary_bytes\":%zu,"
        "\"guard_probe_sigsegv\":%s,\"entry_uses_seed_rows\":true}\n",
        NPOS, NPOS * E, (unsigned long long)seed_model_blocks, (unsigned long long)seed_model_payload_bytes,
        sizeof(seed_model_rows), seed_guard_bytes, seed_boundary_bytes, seed_guard_probe_passed ? "true" : "false");
    if (fclose(report)) die("seed model report close");
}