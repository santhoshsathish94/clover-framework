/* Demonstration for M52 (job file offset: e * raw as a 32-bit multiply), not part of the driver: one sy_ensure call with the REAL record size (17547264 bytes) and a SPARSE file of
   896 records (15.7 GB apparent, ~0 on disk) that has marker bytes only at the start and the end of record e. Fake ops, no GPU, no CUDA. Shows what the driver's small records cannot.
   build: gcc -O1 -std=gnu11 -Wall -Wextra -pthread -I ../../../include -I ../../../../slotcache/include -o m52_real_size m52_real_size.c -ldl
   run:   ./m52_real_size <libsupply.so> <sparse file path> <expert e>     (exit 0 = the bytes are right, 1 = wrong bytes, the library's own fatal exit status 1 is also possible) */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "supply.h"

#define RAW 17547264UL
#define EXPERTS 896
static unsigned char *g_slot;
static int ok0(void) { return 0; }
static int ss(void *s) { (void)s; return 0; }
static int sd(int d) { (void)d; return 0; }
static int d2h(void *d, const void *s, size_t n, void *st) { (void)d; (void)s; (void)n; (void)st; return 0; }
static int h2d(void *d, const void *s, size_t n, void *st) { (void)st; memcpy(d, s, n); return 0; }
static int er(void *e, void *s) { (void)e; (void)s; return 0; }
static int swe(void *s, void *e, unsigned f) { (void)s; (void)e; (void)f; return 0; }
static void chk(int st, const char *e) { if (st) { fprintf(stderr, "cuda error %d at %s\n", st, e); exit(1); } }
static unsigned char *scf(sc_cache *c, int k) { (void)c; (void)k; return NULL; }
static void sct(sc_cache *c, int k, float w) { (void)c; (void)k; (void)w; }
static int scw(const sc_cache *c, int k) { (void)c; (void)k; return 1; }
static unsigned char *sca(sc_cache *c, int k, const int *p, size_t np, float h, int *vk, float *vh) { (void)c; (void)k; (void)p; (void)np; (void)h; (void)vh; *vk = -1; return g_slot; }
static unsigned char *scr(sc_cache *c, int k) { (void)c; (void)k; return NULL; }
static void screl(sc_cache *c, unsigned char *s) { (void)c; (void)s; }
static void pt(int k) { (void)k; }
static void aw(int k) { (void)k; }
static unsigned char *ha(int k, float h, int *vk, float *vh) { (void)k; (void)h; (void)vh; *vk = -1; return NULL; }

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s lib file e\n", argv[0]); return 2; }
    const int e = atoi(argv[3]);
    const int fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0 || ftruncate(fd, (off_t)EXPERTS * (off_t)RAW) != 0) { perror("file"); return 2; }
    unsigned char m1[64], m2[64];
    for (int i = 0; i < 64; i++) { m1[i] = (unsigned char)(0xA0 + i % 16); m2[i] = (unsigned char)(0x50 + i % 16); }
    const off_t base = (off_t)e * (off_t)RAW;
    if (pwrite(fd, m1, 64, base) != 64 || pwrite(fd, m2, 64, base + (off_t)RAW - 64) != 64) { perror("marker"); return 2; }
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    void (*init)(const sy_ops *, const sy_config *) = (void (*)(const sy_ops *, const sy_config *))dlsym(h, "sy_init");
    void (*ens)(sy_pool *, sy_host *, int, int, const int *, size_t, const int *, int) = (void (*)(sy_pool *, sy_host *, int, int, const int *, size_t, const int *, int))dlsym(h, "sy_ensure");
    if (!init || !ens) { fprintf(stderr, "missing symbol\n"); return 2; }
    g_slot = calloc(1, RAW);
    sy_ops ops; memset(&ops, 0, sizeof ops);
    ops.device_sync = ok0; ops.stream_sync = ss; ops.set_device = sd; ops.memcpy_d2h_async = d2h; ops.memcpy_h2d_async = h2d; ops.event_record = er; ops.stream_wait_event = swe; ops.check = chk;
    ops.sc_find = scf; ops.sc_touch = sct; ops.sc_worth = scw; ops.sc_acquire = sca; ops.sc_remove = scr; ops.sc_release = screl; ops.pend_take = pt; ops.arr_wait = aw; ops.host_acquire = ha;
    sy_config cfg; memset(&cfg, 0, sizeof cfg); cfg.experts = EXPERTS; cfg.expert_raw = RAW;
    init(&ops, &cfg);
    sy_pool pool; memset(&pool, 0, sizeof pool);
    pool.readers = 1; pool.cache = (sc_cache *)(void *)g_slot;   /* never dereferenced: the fakes ignore it */
    pool.bounce[0] = malloc(RAW); pool.h_eptr = calloc(EXPERTS, sizeof(unsigned char *));
    sy_host host; memset(&host, 0, sizeof host);
    int *counts = calloc(EXPERTS, sizeof(int));
    ens(&pool, &host, 0, fd, &e, 1, counts, 1);
    const int good = memcmp(g_slot, m1, 64) == 0 && memcmp(g_slot + RAW - 64, m2, 64) == 0;
    printf("expert %d offset (64-bit) %lld: bytes on the card %s\n", e, (long long)base, good ? "RIGHT" : "WRONG");
    close(fd); unlink(argv[2]);
    return good ? 0 : 1;
}
