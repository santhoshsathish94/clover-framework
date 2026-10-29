/* Smoke test of include/engcfg.h against either library (ref/libengcfg_ref.so or target/release/libengcfg.so): one hand-derived scenario per function.
   Expected values come from the header text and from arithmetic done by hand (derivations in the comments), not from running either library.
   Functions that write to stderr or exit run in a child process with stderr redirected to a file. Prints "PASS (n checks, 0 failed)" and exits 0, or FAIL and exits 1. */
#include "engcfg.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* ABI layout, LP64, offsets worked out by hand from the field order of include/engcfg.h (the Rust tests check the same numbers on their side) */
_Static_assert(sizeof(ec_settings) == 88 && offsetof(ec_settings, reserve) == 16 && offsetof(ec_settings, policy) == 24 && offsetof(ec_settings, host_gb) == 32 &&
               offsetof(ec_settings, host_bytes) == 40 && offsetof(ec_settings, routecache) == 48 && offsetof(ec_settings, lookahead) == 56 && offsetof(ec_settings, vq_cap) == 64 &&
               offsetof(ec_settings, vq_from) == 72 && offsetof(ec_settings, layout) == 80 && offsetof(ec_settings, prefill) == 84, "ec_settings layout");
_Static_assert(sizeof(ec_call) == 40 && offsetof(ec_call, experts) == 8 && offsetof(ec_call, n_experts) == 16 && offsetof(ec_call, counts) == 24 && offsetof(ec_call, n_counts) == 32, "ec_call layout");
_Static_assert(sizeof(ec_card_stats) == 64 && offsetof(ec_card_stats, needed) == 8 && offsetof(ec_card_stats, written_back) == 40 && offsetof(ec_card_stats, t_read) == 48 &&
               offsetof(ec_card_stats, t_copy) == 56, "ec_card_stats layout");
_Static_assert(sizeof(ec_summary) == 784 && offsetof(ec_summary, card) == 8 && offsetof(ec_summary, copyq_rust) == 520 && offsetof(ec_summary, cq_jobs) == 528 &&
               offsetof(ec_summary, slotcache_rust) == 552 && offsetof(ec_summary, routetable_rust) == 584 && offsetof(ec_summary, rc_calls) == 592 && offsetof(ec_summary, rc_pos) == 608 &&
               offsetof(ec_summary, pf_planned) == 624 && offsetof(ec_summary, t_pwait) == 656 && offsetof(ec_summary, vq) == 672 && offsetof(ec_summary, vq_ram) == 680 &&
               offsetof(ec_summary, t_vwait) == 720 && offsetof(ec_summary, n_calls) == 736 && offsetof(ec_summary, t_calls) == 760, "ec_summary layout");

static int g_checks = 0, g_failed = 0;
static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) { g_failed++; printf("  FAIL: %s\n", what); }
}
static void check_str(const char *got, const char *want, const char *what)
{
    int ok = got && strcmp(got, want) == 0;
    g_checks++;
    if (!ok) { g_failed++; printf("  FAIL: %s\n    got  [%s]\n    want [%s]\n", what, got ? got : "(null)", want); }
}

/* runs fn in a child; stderr goes to `err`; returns the exit code (or 128+signal) */
static int run_child(int (*fn)(void), char *err, size_t cap)
{
    char path[] = "/tmp/engcfg_smoke_XXXXXX";
    int fd = mkstemp(path);
    pid_t pid;
    int st = 0;
    err[0] = 0;
    if (fd < 0) { perror("mkstemp"); exit(2); }
    fflush(stdout);
    pid = fork();
    if (pid == 0) {
        dup2(fd, 2);
        exit(fn());
    }
    waitpid(pid, &st, 0);
    {
        ssize_t n = pread(fd, err, cap - 1, 0);
        err[n < 0 ? 0 : n] = 0;
    }
    close(fd); unlink(path);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

static void clear_env(void)
{
    static const char *names[] = { "K3GPU_CARDS", "K3GPU_READERS", "K3GPU_VERBOSE", "K3GPU_RESERVE_MB", "K3GPU_POLICY", "K3GPU_CHUNKS", "K3GPU_HOST_GB", "K3GPU_ROUTECACHE",
        "K3GPU_LOOKAHEAD", "K3GPU_VQUEUE", "K3GPU_VQ_INFLIGHT", "K3GPU_VQ_FROM", "K3GPU_LAYOUT", "K3GPU_PREFILL", "K3GPU_TRACE", "K3GPU_COPYQ", "K3GPU_COPYQ_LIB",
        "K3GPU_SLOTCACHE", "K3GPU_SLOTCACHE_LIB", "K3GPU_ROUTETABLE", "K3GPU_ROUTETABLE_LIB", "ENGCFG_T" };
    for (size_t i = 0; i < sizeof names / sizeof *names; i++) unsetenv(names[i]);
}

static int child_sel_copyq_unset_built(void) { return 10 + ec_selector_mode(EC_COPYQ, 1); }
static int child_sel_slot_rust_built(void)
{
    int m = ec_selector_mode(EC_SLOTCACHE, 1);
    const char *lib = ec_selector_lib(EC_SLOTCACHE);
    if (m != EC_MODE_RUST || strcmp(lib, "libslotcache.so")) return 3;
    setenv("K3GPU_SLOTCACHE_LIB", "/tmp/x.so", 1);
    lib = ec_selector_lib(EC_SLOTCACHE);
    ec_selector_announce_rust(EC_SLOTCACHE, lib);
    return 10 + m;
}
static int child_sel_rt_bad(void) { return 10 + ec_selector_mode(EC_ROUTETABLE, 1); }
static int child_sel_copyq_notbuilt_rust(void) { return 10 + ec_selector_mode(EC_COPYQ, 0); }
static int child_sel_copyq_notbuilt_empty(void) { return 10 + ec_selector_mode(EC_COPYQ, 0); }
static int child_trace_unset(void) { return ec_open_trace() == NULL ? 10 : 11; }
static int child_trace_bad(void) { FILE *f = ec_open_trace(); return f ? 11 : 10; }
static int child_prefill0(void) { ec_prefill_start(0); return 10; }
static int child_prefill3(void) { ec_prefill_start(3); return 10; }
static int child_msgs(void)
{
    ec_settings s;
    memset(&s, 0, sizeof s);
    s.routecache = "/r.trace"; s.lookahead = 92; s.vqueue = 1; s.vq_cap = 256; s.cards = 8; s.host_gb = 150; s.readers = 16;
    ec_print_route_cache(&s, 5612, 1234);
    ec_print_start_line(&s);
    return 10;
}
static int child_force(void)
{
    /* hand case: top 2, n 3 tokens. experts 10,20,30 with counts 3,2,1 (sum 6 = 3*2, each <= 3). Dealing: s=0,1,2 -> tokens 0,1,2 get 10 at sel[0],sel[2],sel[4];
       s=3,4 -> tokens 0,1 get 20 at sel[1],sel[3]; s=5 -> token 2 gets 30 at sel[5]. Expected [10,20,10,20,10,30]. */
    int experts[3] = { 10, 20, 30 }, counts[3] = { 3, 2, 1 };
    ec_call c = { 7, experts, 3, counts, 3 };
    int sel[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    int want[8] = { 10, 20, 10, 20, 10, 30, -1, -1 };
    ec_force_routes(0, 1, &c, 7, 3, sel, 2);
    if (memcmp(sel, want, sizeof sel)) return 3;
    {   /* wrong layer: sel untouched, ONE message; a second failing call (counts missing) is silent */
        int sel2[2] = { -5, -5 };
        ec_call d = { 7, experts, 3, NULL, 0 };
        ec_force_routes(0, 1, &c, 8, 3, sel2, 2);
        ec_force_routes(0, 1, &d, 7, 3, sel2, 2);
        ec_force_routes(1, 1, NULL, 7, 3, sel2, 2);
        if (sel2[0] != -5 || sel2[1] != -5) return 4;
    }
    return 10;
}
static int child_force_first_no_counts(void)
{
    int experts[2] = { 1, 2 }, counts[1] = { 6 }, sel[6] = { 0 };
    ec_call c = { 3, experts, 2, counts, 1 };
    ec_force_routes(4, 9, &c, 3, 3, sel, 2);   /* pos 4 < size 9, layer matches, 1 count for 2 experts -> "no counts" */
    ec_force_routes(4, 9, &c, 3, 3, sel, 2);
    return 10;
}
static int child_force_nofit(void)
{
    int experts[2] = { 1, 2 }, counts[2] = { 4, 2 }, sel[6] = { 0 };
    ec_call c = { 3, experts, 2, counts, 2 };
    ec_force_routes(2, 3, &c, 3, 3, sel, 2);   /* count 4 > n 3 -> does not fit */
    return 10;
}
static int child_force_end(void)
{
    int sel[1] = { 0 };
    ec_force_routes(5, 5, NULL, 3, 3, sel, 2);   /* pos == size */
    return 10;
}
static int child_summary_small(void)
{
    ec_summary s;
    memset(&s, 0, sizeof s);
    s.cards = 2; s.expert_raw = 17547264;
    s.card[0].ready = 1; s.card[0].slots = 100; s.card[0].needed = 10; s.card[0].hit_card = 5; s.card[0].hit_host = 3; s.card[0].from_drive = 2; s.card[0].written_back = 1;
    s.card[0].t_read = 1.25; s.card[0].t_copy = 0.5;
    s.cq_jobs[0] = 1; s.cq_jobs[1] = 2; s.cq_jobs[2] = 3; s.sc_calls[0] = 4; s.sc_calls[1] = 5; s.sc_calls[2] = 6; s.rt_nu_calls = 7;
    s.n_calls[0] = 3; s.t_calls[0] = 0.0005;
    ec_print_summary(&s);
    return 10;
}
static int child_summary_zero(void)
{
    ec_summary s;
    memset(&s, 0, sizeof s);
    s.cards = 8; s.copyq_rust = 1; s.slotcache_rust = 1; s.routetable_rust = 1; s.rc_calls = 5612; s.vq = 1; s.n_calls[1] = 2;
    ec_print_summary(&s);   /* no ready card: only the three lines */
    return 10;
}

int main(void)
{
    char err[8192];
    int rc;
    clear_env();

    /* env_int: "12" -> 12; "" and unset -> default 42; " 7x" -> 7; "0x10" -> 0; "+5" -> 5; "x" -> 0; "-3" -> -3;
       99999999999 = 23 * 2^32 + 1215752191 -> low 32 bits 1215752191 (positive); 4294967297 -> 1; 2147483648 -> INT_MIN; 9223372036854775808 (> LONG_MAX) saturates, (int)LONG_MAX = -1 */
    check(ec_env_int("ENGCFG_T", 42) == 42, "env_int unset");
    setenv("ENGCFG_T", "", 1); check(ec_env_int("ENGCFG_T", 42) == 42, "env_int empty");
    setenv("ENGCFG_T", "12", 1); check(ec_env_int("ENGCFG_T", 42) == 12, "env_int 12");
    setenv("ENGCFG_T", " 7x", 1); check(ec_env_int("ENGCFG_T", 42) == 7, "env_int ' 7x'");
    setenv("ENGCFG_T", "0x10", 1); check(ec_env_int("ENGCFG_T", 42) == 0, "env_int 0x10");
    setenv("ENGCFG_T", "+5", 1); check(ec_env_int("ENGCFG_T", 42) == 5, "env_int +5");
    setenv("ENGCFG_T", "x", 1); check(ec_env_int("ENGCFG_T", 42) == 0, "env_int x");
    setenv("ENGCFG_T", "-3", 1); check(ec_env_int("ENGCFG_T", 42) == -3, "env_int -3");
    setenv("ENGCFG_T", "99999999999", 1); check(ec_env_int("ENGCFG_T", 42) == 1215752191, "env_int 99999999999");
    setenv("ENGCFG_T", "4294967297", 1); check(ec_env_int("ENGCFG_T", 42) == 1, "env_int 2^32+1");
    setenv("ENGCFG_T", "2147483648", 1); check(ec_env_int("ENGCFG_T", 42) == INT32_MIN, "env_int 2^31");
    setenv("ENGCFG_T", "9223372036854775808", 1); check(ec_env_int("ENGCFG_T", 42) == -1, "env_int LONG_MAX+1");
    unsetenv("ENGCFG_T");

    /* settings, all defaults, 4 devices: cards min(min(8,4),8) = 4; reserve 2500<<20 = 2621440000; host_bytes 150<<30 = 161061273600 */
    {
        ec_settings s;
        memset(&s, 0xAA, sizeof s);
        ec_settings_from_env(&s, 4);
        check(s.cards == 4 && s.readers == 16 && s.verbose == 0, "defaults: cards readers verbose");
        check(s.reserve == 2621440000ull, "defaults: reserve");
        check(s.policy == 1 && s.chunks == 0 && s.host_gb == 150 && s.host_bytes == 161061273600ull, "defaults: policy chunks host");
        check(s.routecache == NULL && s.lookahead == 0 && s.vqueue == 0 && s.vq_cap == 256 && s.vq_from == 0, "defaults: route cache fields at their initial values");
        check(s.layout == 0 && s.prefill == 0, "defaults: layout prefill");
        /* cards: K3GPU_CARDS=20, devices 16 -> 16 -> capped at 8 */
        setenv("K3GPU_CARDS", "20", 1);
        ec_settings_from_env(&s, 16);
        check(s.cards == 8, "cards capped at 8");
        /* route cache on, policy 2, queue 3 (nonzero -> 1), in-flight -1 -> (size_t)-1, from 92, look-ahead 10, reserve -1 -> 2^64 - 2^20 */
        setenv("K3GPU_ROUTECACHE", "/x/y", 1); setenv("K3GPU_POLICY", "2", 1); setenv("K3GPU_VQUEUE", "3", 1); setenv("K3GPU_VQ_INFLIGHT", "-1", 1);
        setenv("K3GPU_VQ_FROM", "92", 1); setenv("K3GPU_LOOKAHEAD", "10", 1); setenv("K3GPU_RESERVE_MB", "-1", 1); setenv("K3GPU_PREFILL", "7", 1); setenv("K3GPU_LAYOUT", "1", 1);
        ec_settings_from_env(&s, 8);
        check(s.routecache && !strcmp(s.routecache, "/x/y"), "route cache path");
        check(s.policy == 2 && s.vqueue == 1 && s.vq_cap == UINT64_MAX && s.vq_from == 92 && s.lookahead == 10, "route cache fields");
        check(s.reserve == 18446744073708503040ull, "reserve of -1 MB");
        check(s.prefill == 1 && s.layout == 1, "prefill 7 -> 1, layout 1");
        /* policy 1: the queue stays off */
        setenv("K3GPU_POLICY", "1", 1);
        ec_settings_from_env(&s, 8);
        check(s.vqueue == 0, "queue off unless policy 2");
        /* an empty K3GPU_ROUTECACHE counts as unset: the four fields keep their initial values although K3GPU_VQUEUE etc. are set */
        setenv("K3GPU_ROUTECACHE", "", 1); setenv("K3GPU_POLICY", "2", 1);
        ec_settings_from_env(&s, 8);
        check(s.routecache == NULL && s.lookahead == 0 && s.vqueue == 0 && s.vq_cap == 256 && s.vq_from == 0, "empty route cache = unset");
        clear_env();
    }

    /* pend_cap: 1000 * 6 / 10 = 600; 7 * 6 / 10 = 42 / 10 = 4 */
    check(ec_pend_cap(1000) == 600 && ec_pend_cap(7) == 4 && ec_pend_cap(0) == 0, "pend_cap");

    /* card_of, layout 0, 8 cards: L=0 -> 0; 1 -> 8/93 = 0; 12 -> 96/93 = 1; 50 -> 400/93 = 4; 92 -> 736/93 = 7 (651 <= 736 < 744), min(7,7) = 7. 3 cards: 92 -> 276/93 = 2; 31 -> 93/93 = 1; 30 -> 90/93 = 0.
       layout 1, 8 cards: 0 -> 0; 1 -> 0; 8 -> 7; 9 -> 8 % 8 = 0; 92 -> 91 % 8 = 3. 3 cards: 4 -> 3 % 3 = 0; 5 -> 1. layout 2 behaves as 0. */
    check(ec_card_of(0, 8, 0) == 0 && ec_card_of(0, 8, 1) == 0 && ec_card_of(0, 8, 12) == 1 && ec_card_of(0, 8, 50) == 4 && ec_card_of(0, 8, 92) == 7, "card_of layout 0, 8 cards");
    check(ec_card_of(0, 3, 92) == 2 && ec_card_of(0, 3, 31) == 1 && ec_card_of(0, 3, 30) == 0, "card_of layout 0, 3 cards");
    check(ec_card_of(1, 8, 0) == 0 && ec_card_of(1, 8, 1) == 0 && ec_card_of(1, 8, 8) == 7 && ec_card_of(1, 8, 9) == 0 && ec_card_of(1, 8, 92) == 3, "card_of layout 1, 8 cards");
    check(ec_card_of(1, 3, 4) == 0 && ec_card_of(1, 3, 5) == 1, "card_of layout 1, 3 cards");
    check(ec_card_of(2, 8, 50) == 4, "card_of layout 2 = layout 0");

    /* selectors (children, stderr captured) */
    rc = run_child(child_sel_copyq_unset_built, err, sizeof err);
    check(rc == 10, "copyq unset built: mode cpp"); check_str(err, "K3GPU: copy queue cpp\n", "copyq unset built: stderr");
    setenv("K3GPU_SLOTCACHE", "rust", 1);
    rc = run_child(child_sel_slot_rust_built, err, sizeof err);
    check(rc == 11, "slotcache rust built: mode rust"); check_str(err, "K3GPU: slot cache rust (/tmp/x.so)\n", "slotcache rust built: only the announce line");
    unsetenv("K3GPU_SLOTCACHE");
    setenv("K3GPU_ROUTETABLE", "Rust", 1);
    rc = run_child(child_sel_rt_bad, err, sizeof err);
    check(rc == 1, "routetable Rust: exit 1"); check_str(err, "K3GPU_ROUTETABLE must be cpp or rust, not Rust\n", "routetable Rust: message");
    unsetenv("K3GPU_ROUTETABLE");
    setenv("K3GPU_COPYQ", "rust", 1);
    rc = run_child(child_sel_copyq_notbuilt_rust, err, sizeof err);
    check(rc == 1, "copyq rust, not built: exit 1");
    check_str(err, "K3GPU_COPYQ=rust: this build has no copyq.h, only the C++ queue (build with -I src/rust/copyq/include)\n", "copyq rust, not built: message");
    setenv("K3GPU_COPYQ", "", 1);
    rc = run_child(child_sel_copyq_notbuilt_empty, err, sizeof err);
    check(rc == 10, "copyq empty, not built: cpp"); check_str(err, "K3GPU: copy queue cpp\n", "copyq empty, not built: line");
    unsetenv("K3GPU_COPYQ");

    /* trace */
    rc = run_child(child_trace_unset, err, sizeof err);
    check(rc == 10 && err[0] == 0, "trace unset: NULL, silent");
    setenv("K3GPU_TRACE", "/nonexistent_dir_engcfg/x", 1);
    rc = run_child(child_trace_bad, err, sizeof err);
    check(rc == 1, "trace unwritable: exit 1"); check_str(err, "cannot write /nonexistent_dir_engcfg/x\n", "trace unwritable: message");
    unsetenv("K3GPU_TRACE");

    /* prefill */
    rc = run_child(child_prefill0, err, sizeof err);
    check(rc == 1, "prefill, empty table: exit 1"); check_str(err, "K3GPU_PREFILL needs K3GPU_ROUTECACHE\n", "prefill, empty table: message");
    rc = run_child(child_prefill3, err, sizeof err);
    check(rc == 10, "prefill, 3 calls: continues");
    check_str(err, "K3GPU PREFILL ON: timing only. No drive is read, expert bytes are 4 records per layer, routes are taken from the route cache; outputs are not the model's.\n", "prefill ON line");

    /* messages */
    rc = run_child(child_msgs, err, sizeof err);
    check(rc == 10, "messages: exit 0");
    check_str(err, "K3GPU route cache: 5612 calls from /r.trace, look-ahead 92 calls, at most 1234 experts read ahead, card queues on (at most 256 in flight)\n"
                   "K3GPU: 8 cards, host expert cache 150 GB, 16 drive readers\n", "route cache line + start line");

    /* force_routes */
    rc = run_child(child_force, err, sizeof err);
    check(rc == 10, "force_routes: dealing result");
    check_str(err, "K3GPU prefill: call 0 (layer 8): does not match the route cache; the computed routes are used from here\n", "force_routes: one message only (first failing call)");
    rc = run_child(child_force_first_no_counts, err, sizeof err);
    check(rc == 10, "force_routes no counts: exit 0");
    check_str(err, "K3GPU prefill: call 4 (layer 3): the route cache has no counts; the computed routes are used from here\n", "force_routes: no counts message, once");
    rc = run_child(child_force_nofit, err, sizeof err);
    check(rc == 10, "force_routes nofit: exit 0");
    check_str(err, "K3GPU prefill: call 2 (layer 3): counts do not fit this call; the computed routes are used from here\n", "force_routes: do not fit message");
    rc = run_child(child_force_end, err, sizeof err);
    check(rc == 10, "force_routes end: exit 0");
    check_str(err, "K3GPU prefill: call 5 (layer 3): does not match the route cache; the computed routes are used from here\n", "force_routes: pos == size message");

    /* summary: half-rounding 1.25 -> "1.2"; 2 * 17547264 / 1e9 = 0.0351 -> "0.0"; 0.0005 -> "0.001" (the double is slightly above the half); 1000 * 0.0005 / 3 = 0.1667 -> "0.2" */
    rc = run_child(child_summary_small, err, sizeof err);
    check(rc == 10, "summary small: exit 0");
    check_str(err,
        "K3GPU copy queue cpp: jobs sent 1 drive->RAM, 2 drive->card, 3 RAM->card\n"
        "K3GPU slot cache cpp: 4 acquire, 5 try_acquire, 6 find\n"
        "K3GPU route table cpp: 0 calls, 7 next_use calls\n"
        "K3GPU experts: 10 needed, 5 already on the card (50.0%), 3 from the host cache (30.0%), 2 from the drive (20.0%, 0.0 GB); drive reads 1.2 s, host to card 0.5 s\n"
        "K3GPU card 0: 100 expert slots, 10 needed, 5 on the card, 3 host, 2 drive, 1 written back to RAM\n"
        "K3GPU prompt-pass layer calls: 3, 0.001 s in all, 0.2 ms each\n", "summary small");
    rc = run_child(child_summary_zero, err, sizeof err);
    check(rc == 10, "summary zero: exit 0");
    check_str(err,
        "K3GPU copy queue rust: jobs sent 0 drive->RAM, 0 drive->card, 0 RAM->card\n"
        "K3GPU slot cache rust: 0 acquire, 0 try_acquire, 0 find\n"
        "K3GPU route table rust: 5612 calls, 0 next_use calls\n", "summary with no ready card: three lines only");

    if (g_failed) { printf("FAIL (%d checks, %d failed)\n", g_checks, g_failed); return 1; }
    printf("PASS (%d checks, 0 failed)\n", g_checks);
    return 0;
}
