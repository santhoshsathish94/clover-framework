// What the host side can move, one leg at a time, with a chosen number of threads.
//
//   ram-copy copy THREADS [GIB=4]                      memory to memory: each thread copies its own slice of a GIB-sized buffer to another, 3 passes, best pass
//   ram-copy read FILE THREADS [GIB=4] direct|buffered [START_GIB=0]
//                                                       file to memory: THREADS threads each read their own slice of GIB gibibytes of FILE starting at START_GIB,
//                                                       64 MiB at a time, into one buffer. direct = O_DIRECT (the disk itself), buffered = through the page cache
//
// Run under numactl to place the threads and the memory: numactl --cpunodebind=0 --membind=0 ram-copy copy 8 4
// build: gcc -O2 -pthread -o ram-copy ram-copy.c
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

typedef struct { unsigned char *dst; const unsigned char *src; size_t bytes; int fd; off_t offset; } Job;

static void *copy_worker(void *p) { Job *j = p; memcpy(j->dst, j->src, j->bytes); return NULL; }
static void *read_worker(void *p)
{
    Job *j = p;
    const size_t chunk = 64u << 20;
    for (size_t done = 0; done < j->bytes; ) {
        size_t n = j->bytes - done < chunk ? j->bytes - done : chunk;
        ssize_t k = pread(j->fd, j->dst + done, n, j->offset + (off_t)done);
        if (k <= 0) { fprintf(stderr, "read failed at %zu\n", done); exit(1); }
        done += (size_t)k;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: ram-copy copy THREADS [GIB] | read FILE THREADS [GIB] direct|buffered [START_GIB]\n"); return 2; }
    const int is_copy = !strcmp(argv[1], "copy");
    const int threads = atoi(argv[is_copy ? 2 : 3]);
    const double gib = argc > (is_copy ? 3 : 4) ? atof(argv[is_copy ? 3 : 4]) : 4.0;
    const size_t total = (size_t)(gib * 1073741824.0) / 4096 / threads * 4096 * threads, slice = total / threads;
    unsigned char *a = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), *b = NULL;
    if (a == MAP_FAILED) { perror("mmap"); return 1; }
    memset(a, 1, total);   // touch every page now, on the node the process is bound to
    pthread_t th[256]; Job jobs[256];
    if (is_copy) {
        b = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        memset(b, 2, total);
        double best = 1e9;
        for (int pass = 0; pass < 3; pass++) {
            const double t0 = now();
            for (int i = 0; i < threads; i++) { jobs[i] = (Job){ b + i * slice, a + i * slice, slice, -1, 0 }; pthread_create(&th[i], NULL, copy_worker, &jobs[i]); }
            for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
            const double t = now() - t0; if (t < best) best = t;
        }
        printf("copy, %d thread(s), %.1f GiB: %.2f GB/s copied (%.2f GB/s of memory traffic, read plus write)\n", threads, total / 1073741824.0, total / 1e9 / best, 2.0 * total / 1e9 / best);
    } else {
        const int direct = !strcmp(argv[5], "direct");
        const off_t start = (off_t)((argc > 6 ? atof(argv[6]) : 0.0) * 1073741824.0) / 4096 * 4096;
        int fd = open(argv[2], O_RDONLY | (direct ? O_DIRECT : 0));
        if (fd < 0) { perror("open"); return 1; }
        const double t0 = now();
        for (int i = 0; i < threads; i++) { jobs[i] = (Job){ a + i * slice, NULL, slice, fd, start + (off_t)(i * slice) }; pthread_create(&th[i], NULL, read_worker, &jobs[i]); }
        for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
        const double t = now() - t0;
        printf("read %s, %d thread(s), %.1f GiB from %s: %.2f s = %.2f GB/s\n", direct ? "direct" : "buffered", threads, total / 1073741824.0, argv[2], t, total / 1e9 / t);
    }
    return 0;
}
