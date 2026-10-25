/* Smoke test of include/poolinit.h against either library: ref/libpoolinit_ref.so (original text) or target/release/libpoolinit.so (Rust). One file, two builds (see ref/build.sh and NOTES.md).
   Fake ops record every call; the expected values were derived by hand from decode.cuh@4d8d891 (derivations in the comments) before the first run. The memory of the host cache is real
   (mmap), with 64 KB "experts": one 32-slot chunk is 2 MiB. No CUDA is touched. */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include "poolinit.h"

static int g_checks, g_failed;
static char errbuf[8192];
#define CK(c) do { g_checks++; if (!(c)) { g_failed++; fprintf(stdout, "FAIL line %d: %s\n  (captured stderr: %s)\n", __LINE__, #c, errbuf); } } while (0)

/* ---- recording fakes */
static char ev[2048][112]; static int nev;
static unsigned char *spares[4096]; static int nsp;
static void add(const char *fmt, ...) { va_list ap; va_start(ap, fmt); vsnprintf(ev[nev < 2047 ? nev : 2047], sizeof ev[0], fmt, ap); va_end(ap); nev++; }
static int call_no, fail_at;
static int maybe_fail(void) { call_no++; return call_no == fail_at ? 2 : 0; }
static size_t g_free, g_total = (size_t)1 << 30;
static unsigned long long darena, harena, hnd;
static void *reg_ptr[64]; static int nreg;
static int f_set_device(int d) { add("set_device %d", d); return maybe_fail(); }
static int f_mem_get_info(size_t *f, size_t *t) { add("mem_get_info"); *f = g_free; *t = g_total; return maybe_fail(); }
static int f_device_malloc(void **p, size_t n) { add("malloc %zu -> %llx", n, darena); *p = (void *)(uintptr_t)darena; darena += ((unsigned long long)n + 511) & ~511ULL; return maybe_fail(); }
static int f_host_register(void *p, size_t n, unsigned fl) { add("register %zu %u", n, fl); if (nreg < 64) reg_ptr[nreg++] = p; return maybe_fail(); }
static int f_host_alloc(void **p, size_t n, unsigned fl) { add("host_alloc %zu %u -> %llx", n, fl, harena); *p = (void *)(uintptr_t)harena; harena += n; return maybe_fail(); }
static int f_stream_create(void **s, unsigned fl) { add("stream %u -> %llx", fl, hnd); *s = (void *)(uintptr_t)hnd; hnd += 0x10; return maybe_fail(); }
static int f_event_create(void **e, unsigned fl) { add("event %u -> %llx", fl, hnd); *e = (void *)(uintptr_t)hnd; hnd += 0x10; return maybe_fail(); }
static const char *f_error_string(int c) { static char b[32]; snprintf(b, sizeof b, "fake error %d", c); return b; }
static void f_add_spare(sc_cache *c, unsigned char *p) { (void)c; add("spare"); if (nsp < 4096) spares[nsp++] = p; }
static pi_ops OPS = { f_set_device, f_mem_get_info, f_device_malloc, f_host_register, f_host_alloc, f_stream_create, f_event_create, f_error_string, f_add_spare };
static sc_cache *const CACHE = (sc_cache *)(uintptr_t)0x1;
static void reset(void) { nev = nsp = nreg = call_no = fail_at = 0; darena = 0x700000000000ULL; harena = 0x7e0000000000ULL; hnd = 0x1000; }
static int is(int i, const char *fmt, ...)
{
    char e[112]; va_list ap; va_start(ap, fmt); vsnprintf(e, sizeof e, fmt, ap); va_end(ap);
    const int ok = i >= 0 && i < nev && i < 2048 && !strcmp(ev[i], e);
    if (!ok) fprintf(stdout, "  event %d: got '%s' expected '%s'\n", i, i < nev && i < 2048 ? ev[i] : "(none)", e);
    return ok;
}

/* ---- stderr capture: fd 2 into a temp file during a call; fatal cases run in a forked child */
static pi_pool_out *g_atexit_pool; static pi_host_out *g_atexit_host;
static void atexit_dump(void)   /* runs in the forked child after the library's exit(1): what the out struct and the fakes hold at that moment */
{
    if (g_atexit_pool) fprintf(stderr, "ATEXIT dev=%d slots=%d readers=%d ready=%d ev=%d sp=%d\n", g_atexit_pool->dev, g_atexit_pool->slots, g_atexit_pool->readers, g_atexit_pool->ready, nev, nsp);
    if (g_atexit_host) fprintf(stderr, "ATEXIT cap=%zu ready=%d ev=%d sp=%d\n", g_atexit_host->cap_slots, g_atexit_host->ready, nev, nsp);
}
static int run_capture(void (*fn)(void *), void *arg, int fatal)   /* returns the exit status (-1: in process), stderr in errbuf */
{
    char tmpl[] = "/tmp/poolinit_smoke_XXXXXX"; const int fd = mkstemp(tmpl); if (fd < 0) { perror("mkstemp"); exit(2); }
    int status = -1;
    fflush(stdout); fflush(stderr);
    if (fatal) {
        const pid_t pid = fork();
        if (pid == 0) { dup2(fd, 2); atexit(atexit_dump); fn(arg); fprintf(stderr, "RETURNED\n"); _exit(0); }
        int st = 0; waitpid(pid, &st, 0); status = WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st);
    } else {
        const int save = dup(2); dup2(fd, 2); fn(arg); fflush(stderr); dup2(save, 2); close(save);
    }
    const ssize_t n = pread(fd, errbuf, sizeof errbuf - 1, 0); errbuf[n > 0 ? n : 0] = 0;
    close(fd); unlink(tmpl);
    return status;
}

static pi_pool_out PO; static pi_host_out HO; static pi_pool_cfg PC; static pi_host_cfg HC;
static void do_pool(void *a) { (void)a; pi_pool_init(&OPS, &PC, &PO); }
static void do_host(void *a) { (void)a; pi_host_init(&OPS, &HC, &HO); }
static void fresh_pool(int dev, size_t reserve, int readers, size_t raw, int experts, size_t freebytes)
{
    reset(); memset(&PO, 0, sizeof PO); memset(PO.rstream, 0xA5, sizeof PO.rstream); memset(PO.bounce, 0xA5, sizeof PO.bounce); memset(&PO.st, 0xA5, sizeof PO.st);
    PC = (pi_pool_cfg){ dev, reserve, readers, raw, experts, CACHE, "x/decode.cuh" }; g_free = freebytes; g_atexit_pool = &PO; g_atexit_host = NULL;
}
static void fresh_host(size_t bytes, size_t raw)
{
    reset(); memset(&HO, 0, sizeof HO); HC = (pi_host_cfg){ bytes, raw, CACHE, "x/decode.cuh" }; g_atexit_host = &HO; g_atexit_pool = NULL;
}

int main(void)
{
    const size_t RAW = 65536;

    /* A: card pool, 3 readers. free = reserve(1 MiB) + 100 experts + 5000 bytes: usable = 100*64K + 5000, slots = 100. Expected calls: set_device 3, mem_get_info, malloc 100*65536 = 6553600
       (0x640000) at the arena start 0x700000000000, 100 spares base+99*raw .. base+0, stream copy (handle 0x1000), then per reader a stream (0x1010, 0x1020, 0x1030) and a 64 KB host_alloc
       (0x7e0000000000, +0x10000, +0x20000), event 0x1040 (flags 2), malloc 10*8 = 80 at 0x700000640000. */
    fresh_pool(3, 1 << 20, 3, RAW, 10, ((size_t)1 << 20) + 100 * RAW + 5000);
    CK(run_capture(do_pool, NULL, 0) == -1 && errbuf[0] == 0);
    CK(nev == 112);
    CK(is(0, "set_device 3") && is(1, "mem_get_info") && is(2, "malloc 6553600 -> 700000000000"));
    int spare_ok = nsp == 100;
    for (int j = 0; j < 100 && spare_ok; j++) spare_ok = spare_ok && spares[j] == (unsigned char *)(uintptr_t)(0x700000000000ULL + (unsigned long long)(99 - j) * RAW) && !strcmp(ev[3 + j], "spare");
    CK(spare_ok);
    CK(is(103, "stream 1 -> 1000") && is(104, "stream 1 -> 1010") && is(105, "host_alloc 65536 0 -> 7e0000000000") && is(106, "stream 1 -> 1020") && is(107, "host_alloc 65536 0 -> 7e0000010000"));
    CK(is(108, "stream 1 -> 1030") && is(109, "host_alloc 65536 0 -> 7e0000020000") && is(110, "event 2 -> 1040") && is(111, "malloc 80 -> 700000640000"));
    CK(PO.dev == 3 && PO.slots == 100 && PO.readers == 3 && PO.ready == 1 && PO.base == (unsigned char *)(uintptr_t)0x700000000000ULL);
    CK(PO.copy == (void *)(uintptr_t)0x1000 && PO.rstream[0] == (void *)(uintptr_t)0x1010 && PO.rstream[1] == (void *)(uintptr_t)0x1020 && PO.rstream[2] == (void *)(uintptr_t)0x1030);
    CK(PO.bounce[0] == (unsigned char *)(uintptr_t)0x7e0000000000ULL && PO.bounce[1] == (unsigned char *)(uintptr_t)0x7e0000010000ULL && PO.bounce[2] == (unsigned char *)(uintptr_t)0x7e0000020000ULL);
    CK(PO.written_back == (void *)(uintptr_t)0x1040 && PO.d_eptr == (const unsigned char **)(uintptr_t)0x700000640000ULL);
    { int same = 1; for (int i = 3; i < 32; i++) same = same && PO.rstream[i] == (void *)0xA5A5A5A5A5A5A5A5ULL && PO.bounce[i] == (unsigned char *)0xA5A5A5A5A5A5A5A5ULL;
      CK(same); pi_expert_stats s; memset(&s, 0xA5, sizeof s); CK(!memcmp(&PO.st, &s, sizeof s)); }

    /* A2: exactly 64 slots is accepted; 40 readers become 32: calls = 3 + 64 spares + (1 + 32*2) streams/allocs + event + malloc = 3 + 64 + 65 + 2 = 134 */
    fresh_pool(0, 0, 40, RAW, 896, 64 * RAW);
    CK(run_capture(do_pool, NULL, 0) == -1 && PO.slots == 64 && PO.readers == 32 && PO.ready == 1 && nev == 134 && nsp == 64);
    CK(is(2, "malloc 4194304 -> 700000000000") && is(133, "malloc 7168 -> 700000400000"));   /* 64*65536 = 0x400000; 896*8 = 7168 */
    CK(PO.rstream[31] != NULL && PO.bounce[31] != NULL);
    /* A3: negative readers: none made; calls = 2 + 1 malloc + 64 spares + 1 stream + event + malloc */
    fresh_pool(0, 0, -2, RAW, 4, 64 * RAW + 1);
    CK(run_capture(do_pool, NULL, 0) == -1 && PO.readers == -2 && PO.rstream[0] == (void *)0xA5A5A5A5A5A5A5A5ULL && nev == 2 + 1 + 64 + 1 + 1 + 1);

    /* B: 63 slots (free = 63*raw + 65535 after the reserve): the card is refused with the original's text, exit status 1; an atexit handler sees dev 3, slots 63 (assigned before the test), ready 0,
       and only the two calls set_device and mem_get_info were made */
    fresh_pool(3, 1 << 20, 3, RAW, 10, ((size_t)1 << 20) + 63 * RAW + 65535);
    CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, "card 3: only 63 expert slots fit after the reserve\nATEXIT dev=3 slots=63 readers=3 ready=0 ev=2 sp=0\n"));
    /* B2: free below the reserve: usable 0, slots 0 */
    fresh_pool(5, 1 << 20, 1, RAW, 10, 12345);
    CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, "card 5: only 0 expert slots fit after the reserve\nATEXIT dev=5 slots=0 readers=1 ready=0 ev=2 sp=0\n"));
    /* E: a failing op: the k-th CUDA call (spares are not CUDA calls) fails with code 2; the line is that of the CHECK in decode.cuh@4d8d891. Events seen: k for k <= 3, else 100 spares + k;
       slots is 0 until mem_get_info has returned (k >= 3 sees 100), spares 100 from k = 4 on. */
    { static const int line[12] = { 454, 455, 459, 461, 463, 464, 463, 464, 463, 464, 466, 467 };
      for (int k = 1; k <= 12; k++) {
          fresh_pool(1, 0, 3, RAW, 10, 100 * RAW); fail_at = k; char want[160];
          snprintf(want, sizeof want, "x/decode.cuh:%d: fake error 2\nATEXIT dev=1 slots=%d readers=3 ready=0 ev=%d sp=%d\n", line[k - 1], k >= 3 ? 100 : 0, k <= 3 ? k : 100 + k, k >= 4 ? 100 : 0);
          CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, want));
      } }
    /* E2: no src_file: "decode.cuh:" */
    fresh_pool(1, 0, 3, RAW, 10, 100 * RAW); PC.src_file = NULL; fail_at = 1;
    CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, "decode.cuh:454: fake error 2\nATEXIT dev=1 slots=0 readers=3 ready=0 ev=1 sp=0\n"));
    /* E3: expert_raw 0 and a NULL op are refused before anything is done (the out struct is untouched: dev 0, readers 0) */
    fresh_pool(1, 0, 3, RAW, 10, 100 * RAW); PC.expert_raw = 0;
    CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, "poolinit: expert_raw is 0\nATEXIT dev=0 slots=0 readers=0 ready=0 ev=0 sp=0\n"));
    fresh_pool(1, 0, 3, RAW, 10, 100 * RAW); OPS.event_create = NULL;
    CK(run_capture(do_pool, NULL, 1) == 1 && !strcmp(errbuf, "poolinit: ops->event_create is NULL\nATEXIT dev=0 slots=0 readers=0 ready=0 ev=0 sp=0\n"));
    OPS.event_create = f_event_create;

    /* C: host cache, real memory. bytes = 3*32*raw + 100*raw: cap_slots 196 -> n_chunks 6 (6*32 = 192, the other 4 slots are dropped); chunk_bytes = 32*65536 = 2097152.
       Calls: set_device 0, then 6 registers (2097152, flag 1), then 192 spares: chunk 5 slot 31 first, chunk 0 slot 0 last, where "chunk i" is the i-th registered region. Line: 192 slots, 192*65536/1e9 = 0.0126 -> "0.0". */
    fresh_host(3 * 32 * RAW + 100 * RAW, RAW);
    CK(run_capture(do_host, NULL, 0) == -1);
    CK(HO.cap_slots == 192 && HO.ready == 1 && nev == 1 + 6 + 192 && nreg == 6 && nsp == 192);
    CK(is(0, "set_device 0") && is(1, "register 2097152 1") && is(6, "register 2097152 1") && is(7, "spare"));
    { int ok = 1; for (int i = 0; i < 6; i++) { ok = ok && ((uintptr_t)reg_ptr[i] % 4096 == 0); for (int j = 0; j < i; j++) ok = ok && ((uintptr_t)reg_ptr[i] + 2097152 <= (uintptr_t)reg_ptr[j] || (uintptr_t)reg_ptr[j] + 2097152 <= (uintptr_t)reg_ptr[i]); }
      CK(ok); int sok = 1; for (int k = 0; k < 192; k++) sok = sok && spares[k] == (unsigned char *)reg_ptr[5 - k / 32] + (size_t)(31 - k % 32) * RAW; CK(sok);
      int zero = 1; for (int i = 0; i < 6; i++) for (size_t b = 0; b < 2097152; b += 4096) zero = zero && ((unsigned char *)reg_ptr[i])[b] == 0; CK(zero); }
    { const char *pre = "K3GPU: host expert cache of 192 slots (0.0 GB) pinned in "; const size_t n = strlen(pre);
      CK(!strncmp(errbuf, pre, n)); const char *p = errbuf + n; int digits = 0; while (*p >= '0' && *p <= '9') { p++; digits++; }
      CK(digits >= 1 && p[0] == '.' && p[1] >= '0' && p[1] <= '9' && !strcmp(p + 2, " s\n")); }
    /* D: no chunk: 31 slots (31*raw + 5 bytes) -> n_chunks 0: only set_device 0, no register, no spare, cap_slots 0, ready 1, still the line */
    fresh_host(31 * RAW + 5, RAW);
    CK(run_capture(do_host, NULL, 0) == -1 && HO.cap_slots == 0 && HO.ready == 1 && nev == 1 && is(0, "set_device 0") && !strncmp(errbuf, "K3GPU: host expert cache of 0 slots (0.0 GB) pinned in ", 55));
    fresh_host(RAW - 1, RAW);   /* bytes below one expert: cap_slots 0 before and after */
    CK(run_capture(do_host, NULL, 0) == -1 && HO.cap_slots == 0 && HO.ready == 1 && nev == 1);
    /* H-E: the first CUDA call fails: "x/decode.cuh:428: ..."; the atexit handler sees cap_slots 196 (not yet trimmed to 192), ready 1 and 1 event. The 4th call (the 3rd register) fails at line 429
       with 4 events and no spare added yet. */
    fresh_host(3 * 32 * RAW + 100 * RAW, RAW); fail_at = 1;
    CK(run_capture(do_host, NULL, 1) == 1 && !strcmp(errbuf, "x/decode.cuh:428: fake error 2\nATEXIT cap=196 ready=1 ev=1 sp=0\n"));
    fresh_host(3 * 32 * RAW + 100 * RAW, RAW); fail_at = 4;
    CK(run_capture(do_host, NULL, 1) == 1 && !strcmp(errbuf, "x/decode.cuh:429: fake error 2\nATEXIT cap=196 ready=1 ev=4 sp=0\n"));

    printf("%s (%d checks, %d failed)\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
