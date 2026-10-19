/* Reads a big file with O_DIRECT (no page cache) and reports bytes per second for a sweep of block sizes and thread counts.
   build: gcc -O2 -Wall -Wextra -pthread nvme-read-rate.c -o nvme-read-rate
   usage: nvme-read-rate FILE [GIB_PER_POINT=8] [sequential|random]
   FILE should be a big dataset file (expert-N/experts.direct is 14.64 GiB). A figure is the bytes pread really returned divided by the wall time;
   a short or failed read stops the run and says so, it is never counted. Reads only, writes nothing. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct { int fd; size_t block; uint64_t first, span, reads, seed, bytes; int random, failed; } Job;

static uint64_t next_random(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (double)t.tv_sec + (double)t.tv_nsec / 1e9; }

static void *worker(void *arg)
{
    Job *j = arg; void *buffer;
    if (posix_memalign(&buffer, 4096, j->block)) { fprintf(stderr, "no memory for a %zu byte buffer\n", j->block); j->failed = 1; return NULL; }
    for (uint64_t r = 0; r < j->reads; r++) {
        const uint64_t index = j->first + (j->random ? next_random(&j->seed) % j->span : r % j->span);
        const ssize_t n = pread(j->fd, buffer, j->block, (off_t)(index * j->block));
        if (n != (ssize_t)j->block) {
            fprintf(stderr, "read of %zu bytes at block %llu returned %zd (%s)\n", j->block, (unsigned long long)index, n, n < 0 ? strerror(errno) : "short read");
            j->failed = 1; break;
        }
        j->bytes += (uint64_t)n;
    }
    free(buffer);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s FILE [GIB_PER_POINT=8] [sequential|random]\n", argv[0]); return 2; }
    const double gib = argc > 2 ? atof(argv[2]) : 8;
    const int random = argc > 3 && !strcmp(argv[3], "random");
    if (gib <= 0) { fprintf(stderr, "GIB_PER_POINT must be positive\n"); return 2; }
    const int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open with O_DIRECT"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    if (st.st_size < (off_t)(1ULL << 30)) { fprintf(stderr, "%s is only %lld bytes: use a file of at least 1 GiB\n", argv[1], (long long)st.st_size); return 1; }
    const size_t block_mib[] = { 1, 4, 16 };
    const int thread_counts[] = { 1, 2, 4, 8, 16, 32 };
    printf("file %s, %.2f GiB, %s reads, %.1f GiB per point, O_DIRECT\n", argv[1], (double)st.st_size / 1073741824.0, random ? "random" : "sequential", gib);
    printf("%8s %8s %12s %10s\n", "block", "threads", "GB/s", "seconds");
    for (size_t b = 0; b < sizeof block_mib / sizeof *block_mib; b++) {
        for (size_t c = 0; c < sizeof thread_counts / sizeof *thread_counts; c++) {
            const size_t block = block_mib[b] << 20; const int threads = thread_counts[c];
            const uint64_t total_blocks = (uint64_t)st.st_size / block;
            if (total_blocks < (uint64_t)threads * 4) { printf("%5zu MiB %8d   skipped: the file is too small for this point\n", block_mib[b], threads); continue; }
            uint64_t reads = (uint64_t)(gib * 1073741824.0 / (double)block / threads);
            if (reads < 1) reads = 1;
            Job jobs[32]; pthread_t thread[32];
            const uint64_t span = random ? total_blocks : total_blocks / (uint64_t)threads;
            const double start = now();
            for (int t = 0; t < threads; t++) {
                jobs[t] = (Job){ .fd = fd, .block = block, .first = random ? 0 : (uint64_t)t * span, .span = span, .reads = reads,
                                 .seed = 88172645463325252ULL + (uint64_t)t * 7919 + block, .random = random };
                if (pthread_create(&thread[t], NULL, worker, &jobs[t]) != 0) { fprintf(stderr, "cannot start thread %d\n", t); return 1; }
            }
            uint64_t bytes = 0; int failed = 0;
            for (int t = 0; t < threads; t++) { pthread_join(thread[t], NULL); bytes += jobs[t].bytes; failed |= jobs[t].failed; }
            const double seconds = now() - start;
            if (failed) { printf("%5zu MiB %8d   FAILED: see the message above; this and later points are not valid\n", block_mib[b], threads); return 1; }
            if (bytes != reads * (uint64_t)threads * block) { printf("%5zu MiB %8d   FAILED: %llu bytes counted, %llu expected\n", block_mib[b], threads, (unsigned long long)bytes, (unsigned long long)(reads * (uint64_t)threads * block)); return 1; }
            printf("%5zu MiB %8d %12.2f %10.2f\n", block_mib[b], threads, (double)bytes / seconds / 1e9, seconds);
        }
    }
    close(fd);
    return 0;
}
