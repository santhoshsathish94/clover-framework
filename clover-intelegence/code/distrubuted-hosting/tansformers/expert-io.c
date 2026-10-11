/* Measures how fast one layer's experts actually arrive, three ways.

   The layer mmaps experts.direct and faults pages in on the compute threads, with
   posix_madvise(WILLNEED) as the only hint. That is arm A. Arm B reads the same bytes
   with one pread per expert across threads; arm C splits each expert into chunks for
   more requests in flight. Same bytes, same checksum, so the only difference is how
   the request reaches the device. */
#define _GNU_SOURCE
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

enum { EXPERT_RAW = 17547264, EXPERTS = 896 };

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Touch one byte per 4 KB page; the page is the unit the kernel faults in. */
static uint64_t touch(const unsigned char *base, size_t bytes)
{
    uint64_t sum = 0;
    for (size_t offset = 0; offset < bytes; offset += 4096) sum += base[offset];
    return sum;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "experts.direct";
    int arm = argc > 2 ? atoi(argv[2]) : 0;
    unsigned first = argc > 3 ? (unsigned)atoi(argv[3]) : 0;
    unsigned count = argc > 4 ? (unsigned)atoi(argv[4]) : 16;
    unsigned chunks = argc > 5 ? (unsigned)atoi(argv[5]) : 8;
    if (!count || first + count > EXPERTS) return 1;

    int handle = open(path, O_RDONLY);
    if (handle < 0) { perror("open"); return 1; }
    const double bytes = (double)count * EXPERT_RAW;
    uint64_t sum = 0;
    double seconds = 0.0;
    const char *name = "?";

    if (arm == 0) {
        name = "A mmap + madvise WILLNEED, fault on touch";
        size_t span = (size_t)EXPERTS * EXPERT_RAW;
        unsigned char *map = mmap(NULL, span, PROT_READ, MAP_SHARED, handle, 0);
        if (map == MAP_FAILED) { perror("mmap"); return 1; }
        double start = now();
        for (unsigned e = 0; e < count; e++)
            (void)posix_madvise(map + (size_t)(first + e) * EXPERT_RAW, EXPERT_RAW, POSIX_MADV_WILLNEED);
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (unsigned e = 0; e < count; e++)
            sum += touch(map + (size_t)(first + e) * EXPERT_RAW, EXPERT_RAW);
        seconds = now() - start;
        munmap(map, span);
    } else if (arm == 1) {
        name = "B one pread per expert, threads over experts";
        unsigned char *buffer = malloc((size_t)count * EXPERT_RAW);
        if (!buffer) return 1;
        double start = now();
#pragma omp parallel for schedule(static) reduction(+:sum)
        for (unsigned e = 0; e < count; e++) {
            unsigned char *target = buffer + (size_t)e * EXPERT_RAW;
            size_t got = 0;
            while (got < EXPERT_RAW) {
                ssize_t n = pread(handle, target + got, EXPERT_RAW - got,
                    (off_t)(first + e) * EXPERT_RAW + (off_t)got);
                if (n <= 0) break;
                got += (size_t)n;
            }
            sum += touch(target, EXPERT_RAW);
        }
        seconds = now() - start;
        free(buffer);
    } else {
        name = "C chunked pread, more requests in flight";
        unsigned char *buffer = malloc((size_t)count * EXPERT_RAW);
        if (!buffer) return 1;
        const size_t piece = (EXPERT_RAW + chunks - 1) / chunks;
        double start = now();
#pragma omp parallel for schedule(static) collapse(2) reduction(+:sum)
        for (unsigned e = 0; e < count; e++) {
            for (unsigned c = 0; c < chunks; c++) {
                size_t offset = (size_t)c * piece;
                if (offset >= EXPERT_RAW) continue;
                size_t want = EXPERT_RAW - offset < piece ? EXPERT_RAW - offset : piece;
                unsigned char *target = buffer + (size_t)e * EXPERT_RAW + offset;
                size_t got = 0;
                while (got < want) {
                    ssize_t n = pread(handle, target + got, want - got,
                        (off_t)(first + e) * EXPERT_RAW + (off_t)(offset + got));
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                sum += touch(target, want);
            }
        }
        seconds = now() - start;
        free(buffer);
    }

    close(handle);
    printf("%-44s %6.0f MB  %7.3f s  %7.2f GB/s  threads %2d  checksum %llu\n",
        name, bytes / 1e6, seconds, bytes / seconds / 1e9, omp_get_max_threads(),
        (unsigned long long)sum);
    return 0;
}
