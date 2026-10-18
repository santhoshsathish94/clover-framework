#define _GNU_SOURCE
/* How fast can the host read layer-1 experts from disk with this access pattern?
   Reads E randomly chosen experts (17,547,264 bytes each, in random order) from experts.direct, T threads taking
   (expert, chunk) units dynamically, with the file evicted from the page cache first (buffered modes) or bypassing it
   (direct mode). Modes: mmap (touch every 4 KB page of the chunk, the way the layer reads today), pread (buffered),
   direct (O_DIRECT).
     disk-read-bench FILE E T MODE CHUNK_BYTES [REPS]                                              */
#include <fcntl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EXPERT 17547264UL

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 6) { fputs("usage: disk-read-bench FILE E T mmap|pread|direct CHUNK_BYTES [REPS]\n", stderr); return 2; }
    const char *path = argv[1], *mode = argv[4];
    const unsigned E = (unsigned)atoi(argv[2]), T = (unsigned)atoi(argv[3]);
    size_t chunk = (size_t)atol(argv[5]);
    const int reps = argc > 6 ? atoi(argv[6]) : 3;
    const int direct = !strcmp(mode, "direct"), use_mmap = !strcmp(mode, "mmap");
    if (chunk > EXPERT) chunk = EXPERT;
    if (chunk % 4096) { fputs("chunk must be a multiple of 4096\n", stderr); return 2; }

    int fd = open(path, O_RDONLY | (direct ? O_DIRECT : 0));
    int fd_plain = direct ? open(path, O_RDONLY) : fd;
    struct stat st;
    if (fd < 0 || fd_plain < 0 || fstat(fd, &st)) { perror("open"); return 1; }
    const unsigned total_experts = (unsigned)(st.st_size / EXPERT);
    if (E > total_experts) { fputs("E larger than the number of experts in the file\n", stderr); return 2; }
    unsigned *order = malloc(total_experts * sizeof *order);
    for (unsigned i = 0; i < total_experts; i++) order[i] = i;
    uint64_t s = 88172645463325252ULL;
    for (unsigned i = total_experts - 1; i > 0; i--) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; unsigned j = (unsigned)(s % (i + 1)); unsigned t = order[i]; order[i] = order[j]; order[j] = t; }

    const unsigned per = (unsigned)((EXPERT + chunk - 1) / chunk), units = E * per;
    unsigned char *map = use_mmap ? mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0) : NULL;
    if (use_mmap && map == MAP_FAILED) { perror("mmap"); return 1; }
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        if (!direct) { if (use_mmap) madvise(map, st.st_size, MADV_DONTNEED); posix_fadvise(fd_plain, 0, 0, POSIX_FADV_DONTNEED); }
        unsigned long sink = 0;
        const double t0 = clk();
#pragma omp parallel num_threads(T) reduction(+ : sink)
        {
            unsigned char *buf = NULL;
            if (!use_mmap && posix_memalign((void **)&buf, 4096, chunk)) abort();
#pragma omp for schedule(dynamic, 1)
            for (unsigned u = 0; u < units; u++) {
                const size_t offset = (size_t)order[u / per] * EXPERT + (size_t)(u % per) * chunk;
                const size_t length = EXPERT - (size_t)(u % per) * chunk < chunk ? EXPERT - (size_t)(u % per) * chunk : chunk;
                if (use_mmap) { for (size_t i = 0; i < length; i += 4096) sink += map[offset + i]; }
                else { size_t got = 0; while (got < length) { ssize_t n = pread(fd, buf, length - got, (off_t)(offset + got)); if (n <= 0) abort(); got += (size_t)n; } sink += buf[0]; }
            }
            free(buf);
        }
        const double sec = clk() - t0;
        if (sec < best) best = sec;
        if (sink == 123456789UL) puts("");
    }
    printf("%-6s experts %4u (%.2f GB)  threads %2u  chunk %8zu  best of %d: %.3f s = %.2f GB/s\n", mode, E, (double)E * EXPERT / 1e9, T,
        chunk, reps, best, (double)E * EXPERT / best / 1e9);
    return 0;
}
