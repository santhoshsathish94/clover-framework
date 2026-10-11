#define _GNU_SOURCE
/* One layer, the same six positions, twice: once with its experts evicted and once
   with them read in and locked down. The difference is what this layer spends waiting
   for storage, which is what a pod holding its own experts in RAM would not pay.
   fadvise is used rather than dropping the whole machine's cache, so nothing else
   on the box is disturbed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int transformer_open(const char *dataset, void **model);
void transformer_close(void *model);
void *transformer_sequence_create(const void *model);
void transformer_sequence_close(void *sequence);
int transformer_process(void *model, void *sequence, const float *input,
    const float *snapshots, unsigned snapshot_count, float *output, float *output_snapshots);

enum { WIDTH = 7168, POSITIONS = 6, SNAPSHOTS_IN = 4 };

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

static size_t resident_bytes(const char *path)
{
    int handle = open(path, O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) { if (handle >= 0) close(handle); return 0; }
    long page = sysconf(_SC_PAGESIZE);
    size_t pages = ((size_t)info.st_size + (size_t)page - 1) / (size_t)page, live = 0;
    void *mapped = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_SHARED, handle, 0);
    if (mapped != MAP_FAILED) {
        unsigned char *state = malloc(pages);
        if (state && !mincore(mapped, (size_t)info.st_size, state))
            for (size_t p = 0; p < pages; p++) live += state[p] & 1;
        free(state);
        munmap(mapped, (size_t)info.st_size);
    }
    close(handle);
    return live * (size_t)page;
}

static double run(void *model, const float *input, const float *snapshots)
{
    void *sequence = transformer_sequence_create(model);
    if (!sequence) { fputs("sequence failed\n", stderr); exit(1); }
    static float output[WIDTH], output_snapshots[8 * WIDTH];
    double mark = now();
    for (unsigned position = 0; position < POSITIONS; position++)
        if (!transformer_process(model, sequence, input, snapshots, SNAPSHOTS_IN, output, output_snapshots)) {
            fputs("process failed\n", stderr); exit(1);
        }
    double spent = now() - mark;
    transformer_sequence_close(sequence);
    return spent;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: layer-cost DATASET_DIR EXPERTS_DIRECT\n", stderr); return 2; }

    void *model = NULL;
    if (!transformer_open(argv[1], &model)) { fputs("open failed\n", stderr); return 1; }

    static float input[WIDTH], snapshots[SNAPSHOTS_IN * WIDTH];
    uint64_t state = 2463534242ULL;
    for (unsigned i = 0; i < WIDTH; i++) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        input[i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
    }
    for (unsigned i = 0; i < SNAPSHOTS_IN * WIDTH; i++) snapshots[i] = input[i % WIDTH] * 0.5f;

    int handle = open(argv[2], O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) { fputs("cannot open experts\n", stderr); return 1; }
    size_t bytes = (size_t)info.st_size;

    (void)posix_fadvise(handle, 0, 0, POSIX_FADV_DONTNEED);
    printf("experts file          %.2f GB\n", (double)bytes / 1e9);
    printf("resident after evict  %.2f GB\n", (double)resident_bytes(argv[2]) / 1e9);
    double cold = run(model, input, snapshots);
    printf("cold pass             %8.3f s   %7.1f ms per position\n", cold, 1000 * cold / POSITIONS);
    printf("resident after pass   %.2f GB\n", (double)resident_bytes(argv[2]) / 1e9);

    void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
    if (mapped == MAP_FAILED) { fputs("map failed\n", stderr); return 1; }
    double mark = now();
    if (mlock(mapped, bytes)) { perror("mlock"); return 1; }
    printf("\npinned %.2f GB in      %8.3f s\n", (double)bytes / 1e9, now() - mark);
    printf("resident after pin    %.2f GB\n", (double)resident_bytes(argv[2]) / 1e9);

    double warm = run(model, input, snapshots);
    printf("resident pass         %8.3f s   %7.1f ms per position\n", warm, 1000 * warm / POSITIONS);
    printf("\nwaiting on storage    %8.3f s   %.1f%% of the cold pass\n", cold - warm, 100 * (cold - warm) / cold);
    printf("speedup               %8.2fx\n", cold / warm);

    munlock(mapped, bytes);
    munmap(mapped, bytes);
    close(handle);
    transformer_close(model);
    return 0;
}
