// Smoke test of libcopyq_ref.so through copyq.h: kind 0 (drive -> RAM slot), kind 2 with a victim copy (RAM slot -> card 7), kind 1 (drive -> bounce -> card 7).
// build: g++ -O2 -std=c++17 -I../include -I/usr/local/cuda-12.6/include smoke_ref.cpp -o smoke_ref -L/usr/local/cuda-12.6/lib64 -lcudart -ldl -pthread
// run:   ./smoke_ref ./libcopyq_ref.so [card, default 7]
#include "copyq.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)
static const size_t RAW = 17547264;

static int wait_flag(int *f, const char *name)
{
    struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
    while (__atomic_load_n(f, __ATOMIC_ACQUIRE) == 0) {
        clock_gettime(CLOCK_MONOTONIC, &t);
        if (t.tv_sec - t0.tv_sec > 60) { printf("FAIL: timeout waiting for %s\n", name); return 0; }
        usleep(200);
    }
    return 1;
}
static void fill(unsigned char *p, size_t n, unsigned seed) { for (size_t i = 0; i < n; i++) p[i] = (unsigned char)((i * 2654435761u) >> 13) ^ (unsigned char)seed ^ (unsigned char)(i >> 20); }
static size_t first_diff(const unsigned char *a, const unsigned char *b, size_t n) { for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return i; return n; }

int main(int argc, char **argv)
{
    const char *lib = argc > 1 ? argv[1] : "./libcopyq_ref.so";
    const int card = argc > 2 ? atoi(argv[2]) : 7;
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    void (*init)(const cq_config *) = (void (*)(const cq_config *))dlsym(h, "cq_init");
    void (*enq)(const cq_job *) = (void (*)(const cq_job *))dlsym(h, "cq_enqueue");
    if (!init || !enq) { fprintf(stderr, "dlsym failed\n"); return 2; }

    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    printf("devices %d, using card %d\n", ndev, card);
    CK(cudaSetDevice(card));

    // Source file: three records of RAW bytes (record 0 for kind 0, record 2 for kind 1).
    char path[] = "/tmp/copyq_smoke_XXXXXX";
    int fd = mkstemp(path); if (fd < 0) { perror("mkstemp"); return 2; }
    unsigned char *file_data = (unsigned char *)malloc(3 * RAW);
    for (int r = 0; r < 3; r++) fill(file_data + r * RAW, RAW, 11 + r);
    if (write(fd, file_data, 3 * RAW) != (ssize_t)(3 * RAW)) { perror("write"); return 2; }
    unlink(path);

    static int p_state = 0, wb_state = 0, arr2 = 0, rel2 = 0, arr1 = 0;
    static int act_hold = 0;
    cq_config cfg = {}; cfg.expert_raw = RAW; cfg.readers = 4; cfg.cards = 8; cfg.prefill = 0; cfg.slab = NULL; cfg.act_hold = &act_hold;
    init(&cfg);
    int ok = 1;

    // kind 0: file record 0 -> pinned RAM slot.
    unsigned char *slot = NULL; CK(cudaHostAlloc((void **)&slot, RAW, cudaHostAllocDefault)); memset(slot, 0, RAW);
    cq_job j0 = {}; j0.kind = 0; j0.fd = fd; j0.off = 0; j0.slot = slot; j0.p_state = &p_state;
    enq(&j0);
    if (!wait_flag(&p_state, "kind 0 p_state")) return 1;
    size_t d0 = first_diff(slot, file_data, RAW);
    printf("kind 0: p_state=%d, RAM slot vs file record 0: %s (first diff %zu of %zu)\n", p_state, d0 == RAW ? "IDENTICAL" : "DIFFERENT", d0, RAW);
    ok &= d0 == RAW;

    // kind 2: pinned RAM slot -> card; the card first holds "old" bytes, which must come down into victim_host first.
    unsigned char *dst = NULL; CK(cudaMalloc((void **)&dst, RAW));
    unsigned char *old_bytes = (unsigned char *)malloc(RAW); fill(old_bytes, RAW, 99);
    CK(cudaMemcpy(dst, old_bytes, RAW, cudaMemcpyHostToDevice));
    unsigned char *victim = NULL; CK(cudaHostAlloc((void **)&victim, RAW, cudaHostAllocDefault)); memset(victim, 0, RAW);
    cq_job j2 = {}; j2.kind = 2; j2.dev = card; j2.dst = dst; j2.host_src = slot; j2.victim_host = victim; j2.wb_state = &wb_state; j2.arr_state = &arr2; j2.rel_state = &rel2;
    enq(&j2);
    if (!wait_flag(&arr2, "kind 2 arr_state")) return 1;
    int wb_seen = __atomic_load_n(&wb_state, __ATOMIC_ACQUIRE), rel_seen = __atomic_load_n(&rel2, __ATOMIC_ACQUIRE);
    unsigned char *back = (unsigned char *)malloc(RAW);
    CK(cudaMemcpy(back, dst, RAW, cudaMemcpyDeviceToHost));
    size_t d2 = first_diff(back, file_data, RAW), dv = first_diff(victim, old_bytes, RAW);
    printf("kind 2: wb_state=%d rel_state=%d arr_state=%d; card read back vs record 0: %s; victim_host vs old card bytes: %s\n", wb_seen, rel_seen, arr2,
           d2 == RAW ? "IDENTICAL" : "DIFFERENT", dv == RAW ? "IDENTICAL" : "DIFFERENT");
    ok &= d2 == RAW && dv == RAW && wb_seen == 1 && rel_seen == 1 && arr2 == 1;

    // kind 1: file record 2 -> bounce -> card (no victim, no rel).
    cq_job j1 = {}; j1.kind = 1; j1.dev = card; j1.fd = fd; j1.off = (off_t)(2 * RAW); j1.dst = dst; j1.arr_state = &arr1;
    enq(&j1);
    if (!wait_flag(&arr1, "kind 1 arr_state")) return 1;
    CK(cudaMemcpy(back, dst, RAW, cudaMemcpyDeviceToHost));
    size_t d1 = first_diff(back, file_data + 2 * RAW, RAW);
    printf("kind 1: arr_state=%d; card read back vs record 2: %s\n", arr1, d1 == RAW ? "IDENTICAL" : "DIFFERENT");
    ok &= d1 == RAW && arr1 == 1;

    printf("%s\n", ok ? "SMOKE PASS" : "SMOKE FAIL");
    return ok ? 0 : 1;
}
