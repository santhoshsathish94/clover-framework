/* Smoke test for libcopyq.so (Rust): one kind-0, one kind-1 and one kind-2 job to card 7, plus the act_hold gate. Card 7 only.
   build: gcc -O1 -I include -I /usr/local/cuda-12.6/include -o smoke_rust smoke_rust.c -ldl -L/usr/local/cuda-12.6/lib64 -lcudart -Wl,-rpath,/usr/local/cuda-12.6/lib64
   run:   ./smoke_rust target/release/libcopyq.so */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "copyq.h"

#define RAW ((size_t)17547264)
#define CARD 7
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static int failures = 0;
static void check(int ok, const char *what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); if (!ok) failures++; }

static int wait_flag(int *f, double secs)
{
    struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        if (__atomic_load_n(f, __ATOMIC_ACQUIRE)) return 1;
        clock_gettime(CLOCK_MONOTONIC, &t);
        if ((t.tv_sec - t0.tv_sec) + 1e-9 * (t.tv_nsec - t0.tv_nsec) > secs) return 0;
        usleep(500);
    }
}
static void fill(unsigned char *p, size_t n, unsigned seed) { for (size_t i = 0; i < n; i++) p[i] = (unsigned char)((i * 2654435761u + seed * 40503u) >> 13); }
static int get(int *f) { return __atomic_load_n(f, __ATOMIC_ACQUIRE); }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s libcopyq.so\n", argv[0]); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    void (*init)(const cq_config *) = (void (*)(const cq_config *))dlsym(h, "cq_init");
    void (*enq)(const cq_job *) = (void (*)(const cq_job *))dlsym(h, "cq_enqueue");
    if (!init || !enq) { fprintf(stderr, "dlsym failed\n"); return 2; }

    CK(cudaSetDevice(CARD));
    unsigned char *file_a = (unsigned char *)malloc(RAW * 2), *chk = (unsigned char *)malloc(RAW);
    fill(file_a, RAW, 1); fill(file_a + RAW, RAW, 2);                  /* record 0 = pattern 1, record 1 = pattern 2 */
    char path[] = "/tmp/copyq_smoke_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0 || write(fd, file_a, RAW * 2) != (ssize_t)(RAW * 2)) { perror("temp file"); return 2; }
    unlink(path);

    unsigned char *slot, *victim1, *victim2, *dev;
    CK(cudaMallocHost((void **)&slot, RAW)); CK(cudaMallocHost((void **)&victim1, RAW)); CK(cudaMallocHost((void **)&victim2, RAW));
    memset(slot, 0, RAW); memset(victim1, 0, RAW); memset(victim2, 0, RAW);
    CK(cudaMalloc((void **)&dev, RAW));
    unsigned char *pat3 = (unsigned char *)malloc(RAW); fill(pat3, RAW, 3);
    CK(cudaMemcpy(dev, pat3, RAW, cudaMemcpyHostToDevice));             /* the card slot starts with pattern 3 */

    int hold = 0;
    cq_config cfg; memset(&cfg, 0, sizeof cfg);
    cfg.expert_raw = RAW; cfg.readers = 8; cfg.cards = 8; cfg.prefill = 0; cfg.slab = NULL; cfg.act_hold = &hold;
    init(&cfg);

    if (argc > 2 && !strcmp(argv[2], "badfd")) {   /* error path: expect a stderr message and exit status 1 from a worker thread */
        int pb = 0; cq_job jb; memset(&jb, 0, sizeof jb);
        jb.kind = 0; jb.fd = -1; jb.off = 0; jb.slot = slot; jb.p_state = &pb;
        enq(&jb);
        sleep(5);
        printf("NOT REACHED if the queue exits on a failed pread\n");
        return 3;
    }

    /* kind 0: drive record 0 -> RAM slot */
    int p0 = 0; cq_job j; memset(&j, 0, sizeof j);
    j.kind = 0; j.fd = fd; j.off = 0; j.slot = slot; j.p_state = &p0;
    enq(&j);
    check(wait_flag(&p0, 20), "kind 0: p_state set within 20 s");
    check(memcmp(slot, file_a, RAW) == 0, "kind 0: RAM slot equals drive record 0");

    /* kind 1: drive record 1 -> bounce -> card; victim (pattern 3) first copied down */
    int wb1 = 0, rel1 = 0, arr1 = 0;
    memset(&j, 0, sizeof j);
    j.kind = 1; j.dev = CARD; j.fd = fd; j.off = (off_t)RAW; j.dst = dev; j.victim_host = victim1; j.wb_state = &wb1; j.rel_state = &rel1; j.arr_state = &arr1;
    enq(&j);
    check(wait_flag(&arr1, 20), "kind 1: arr_state set within 20 s");
    check(get(&wb1) == 1 && get(&rel1) == 1 && get(&arr1) == 1, "kind 1: wb_state, rel_state, arr_state all 1");
    CK(cudaMemcpy(chk, dev, RAW, cudaMemcpyDeviceToHost));
    check(memcmp(chk, file_a + RAW, RAW) == 0, "kind 1: card holds drive record 1");
    check(memcmp(victim1, pat3, RAW) == 0, "kind 1: victim_host holds the card's previous bytes (pattern 3)");

    /* kind 2: RAM slot (record 0) -> card; victim (record 1) first copied down; rel_state NULL */
    int wb2 = 0, arr2 = 0;
    memset(&j, 0, sizeof j);
    j.kind = 2; j.dev = CARD; j.host_src = slot; j.dst = dev; j.victim_host = victim2; j.wb_state = &wb2; j.rel_state = NULL; j.arr_state = &arr2;
    enq(&j);
    check(wait_flag(&arr2, 20), "kind 2: arr_state set within 20 s");
    check(get(&wb2) == 1, "kind 2: wb_state 1");
    CK(cudaMemcpy(chk, dev, RAW, cudaMemcpyDeviceToHost));
    check(memcmp(chk, file_a, RAW) == 0, "kind 2: card holds the RAM slot (drive record 0)");
    check(memcmp(victim2, file_a + RAW, RAW) == 0, "kind 2: victim_host holds the previous card bytes (record 1)");

    /* act_hold gate: a kind 2 job without victim must not finish while hold != 0 */
    int arr3 = 0;
    __atomic_store_n(&hold, 1, __ATOMIC_RELEASE);
    memset(&j, 0, sizeof j);
    j.kind = 2; j.dev = CARD; j.host_src = slot; j.dst = dev; j.arr_state = &arr3;
    enq(&j);
    usleep(300000);
    check(get(&arr3) == 0, "gate: arr_state still 0 after 300 ms with act_hold set");
    __atomic_store_n(&hold, 0, __ATOMIC_RELEASE);
    check(wait_flag(&arr3, 20), "gate: arr_state set after act_hold cleared");

    printf("%s (%d failure%s)\n", failures ? "SMOKE FAILED" : "SMOKE OK", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
