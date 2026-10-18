#define _GNU_SOURCE
/* How fast can 16 experts (what one decode token needs from a layer) be read from the layer's expert file with direct reads?
   Each repetition picks 16 different experts at random and reads all 17.5 MB of each; the page cache is bypassed (O_DIRECT).
   mode 0: one expert is one task and one thread reads it start to end in CHUNK pieces (what the engine does now);
   mode 1: every CHUNK of every expert is a task, so one expert is read by several threads at once.
     expert-read-bench FILE THREADS CHUNK_KB MODE REPS                                                                   */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define EXPERT 17547264ULL
#define EXPERTS 896

static int fd, chunk_kb, mode;
static size_t chunk;
static unsigned picked[16];
static unsigned next_task, tasks;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static void *worker(void *arg)
{
    unsigned char *buf = aligned_alloc(4096, chunk);
    (void)arg;
    for (;;) {
        const unsigned task = __atomic_fetch_add(&next_task, 1, __ATOMIC_RELAXED);
        if (task >= tasks) break;
        if (mode == 0) {
            const off_t base = (off_t)((uint64_t)picked[task] * EXPERT);
            for (size_t off = 0; off < EXPERT; off += chunk) {
                size_t len = EXPERT - off < chunk ? EXPERT - off : chunk;
                if (pread(fd, buf, len, base + (off_t)off) != (ssize_t)len) { perror("pread"); exit(1); }
            }
        } else {
            const unsigned per = (unsigned)((EXPERT + chunk - 1) / chunk);
            const unsigned e = task / per, c = task % per;
            const size_t off = (size_t)c * chunk, len = EXPERT - off < chunk ? EXPERT - off : chunk;
            if (pread(fd, buf, len, (off_t)((uint64_t)picked[e] * EXPERT + off)) != (ssize_t)len) { perror("pread"); exit(1); }
        }
    }
    free(buf);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 6) { fputs("usage: expert-read-bench FILE THREADS CHUNK_KB MODE REPS\n", stderr); return 2; }
    const int threads = atoi(argv[2]); chunk_kb = atoi(argv[3]); mode = atoi(argv[4]); const int reps = atoi(argv[5]);
    chunk = (size_t)chunk_kb * 1024;
    fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }
    const unsigned per = (unsigned)((EXPERT + chunk - 1) / chunk);
    tasks = mode == 0 ? 16 : 16 * per;
    uint64_t s = 88172645463325252ULL;
    double sum = 0, best = 1e9;
    for (int r = 0; r < reps; r++) {
        for (unsigned i = 0; i < 16; i++) {
            unsigned e, again;
            do { s ^= s << 13; s ^= s >> 7; s ^= s << 17; e = (unsigned)(s % EXPERTS); again = 0;
                 for (unsigned j = 0; j < i; j++) if (picked[j] == e) again = 1; } while (again);
            picked[i] = e;
        }
        next_task = 0;
        pthread_t th[64];
        const double t0 = now();
        for (int i = 0; i < threads; i++) pthread_create(&th[i], NULL, worker, NULL);
        for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
        const double dt = now() - t0;
        if (r) { sum += dt; if (dt < best) best = dt; }   /* the first repetition is a warm-up */
    }
    const double mean = sum / (reps - 1), bytes = 16.0 * EXPERT;
    printf("threads %2d chunk %5d KB mode %d: mean %6.2f ms (%5.2f GB/s)  best %6.2f ms (%5.2f GB/s)\n", threads, chunk_kb, mode,
        1000 * mean, bytes / mean / 1e9, 1000 * best, bytes / best / 1e9);
    return 0;
}
