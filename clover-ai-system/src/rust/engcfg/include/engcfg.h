#ifndef ENGCFG_H
#define ENGCFG_H
/* Contract of engcfg: the small pure-CPU parts of the GPU backend's start-up and exit code, with everything that touches CUDA left in C++ and passed in as an argument.
   Behaviour reference: src/code/gpu/backend-gpu.cu AS COMMITTED AT 4d8d891 (line numbers below are those of that blob; the working tree may differ) and decode.cuh at the same commit:
     env_int (50)                              -> ec_env_int
     configure() (487-524), the settings lines  -> ec_settings_from_env, ec_pend_cap, ec_open_trace
     configure() messages (512, 517, 519, 523)  -> ec_print_route_cache, ec_prefill_start, ec_print_start_line
     copyq_select / slotcache_select / routetable_select, the variable parsing only (183-190, 224-228, 243-249, 258-264, 441-449, 456-462) -> ec_selector_*
     card_of (526)                              -> ec_card_of
     force_routes (374-387)                     -> ec_force_routes
     print_summary (465-485)                    -> ec_print_summary
   The C++ reference (ref/) and the Rust crate implement exactly this header; neither may change it.

   Conventions
   - Every message is written to the process's standard error (file descriptor 2, unbuffered, write errors ignored) as the exact bytes of the original fprintf(stderr, ...), one write per original line.
     Strings that the original prints with %s (paths, selector values) are printed as raw bytes up to their NUL, never converted to UTF-8.
   - "exit(1)" means: the message is written, then the process is ended with the C library's exit(1) (atexit handlers and stdio flushing run, as in the original).
   - Booleans are int: 0 is false, any other value is true.
   - "Environment" means the process environment as read by getenv (nothing is cached; each call reads it again). No locale is set by the library; atoi/printf follow the C locale like the original.
   - Not thread-safe except where stated (the original runs all of this on the main thread). The only process-global state is the "warned" flag of ec_force_routes.
   - Values that the C++ passes in and that the original takes from engine globals are named after the global in the comments.

   Order of the original configure() (the C++ adapter must call the functions in this order to keep the same output order and the same first fatal error):
       have = cudaGetDeviceCount                       ec_settings_from_env(&s, have)         (no output)
       g_cards, g_readers, g_verbose, g_reserve = s.cards, s.readers, s.verbose, s.reserve
       slotcache_select()    [ec_selector_mode(EC_SLOTCACHE,..); rust: dlopen; ec_selector_announce_rust]
       slotcache_set_policy(s.policy)
       routetable_select()   [same with EC_ROUTETABLE]
       g_chunks = s.chunks
       g_trace = ec_open_trace()
       g_host.init(s.host_bytes)                       (C++/CUDA)
       if (s.routecache) { rc_load_sel(s.routecache); g_lookahead = s.lookahead; g_pend_cap = ec_pend_cap(g_host.cap_slots);
                           g_route_hook = ...; g_rc_ok = !g_rc.empty(); if (s.policy == 2) slotcache_set_next_use(...);
                           g_vq = s.vqueue; g_vq_cap = s.vq_cap; g_vq_from = s.vq_from;
                           ec_print_route_cache(&s, g_rc.size(), g_pend_cap); }
       g_layout = s.layout; g_prefill = s.prefill
       if (s.prefill) { ec_prefill_start(g_rc.size()); g_force_routes = ...; }      (the original assigns g_force_routes between the check and the PREFILL ON line; no output depends on that)
       copyq_select()        [same with EC_COPYQ]
       atexit(print_summary) ec_print_start_line(&s)
*/
#include <stddef.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif

#define EC_MAX_CARDS 8   /* MAX_CARDS */

/* ---- env_int (50): `const char *v = getenv(name); return v && *v ? atoi(v) : dflt;`. An unset or empty variable gives dflt. Otherwise glibc atoi, i.e. (int)strtol(v, NULL, 10):
   leading C-locale whitespace (space \t \n \v \f \r), one optional + or -, decimal digits, stops at the first other byte; no digits gives 0 (not dflt); a value that does not fit
   a long saturates at LONG_MAX / LONG_MIN and the (int) cast keeps the low 32 bits (LONG_MAX -> -1, LONG_MIN -> 0, 4294967297 -> 1, 2147483648 -> INT_MIN); "0x10" gives 0; "+5" gives 5. */
int ec_env_int(const char *name, int dflt);

/* ---- the settings of configure(). One struct, filled by ec_settings_from_env; no output, no exit, no file or CUDA access. `devices` = what cudaGetDeviceCount returned.
   Defaults and derivations (variable -> field):
     K3GPU_CARDS 8        cards      = min(min(env_int, devices), EC_MAX_CARDS)         (may be 0 or negative; see ec_card_of)
     K3GPU_READERS 16     readers
     K3GPU_VERBOSE 0      verbose
     K3GPU_RESERVE_MB 2500 reserve   = ((size_t)(int)env_int) << 20                      (a negative int sign-extends to 64 bits, then the shift wraps modulo 2^64)
     K3GPU_POLICY 1       policy      (the argument of slotcache_set_policy)
     K3GPU_CHUNKS 0       chunks
     K3GPU_HOST_GB 150    host_gb     host_bytes = ((size_t)(int)env_int) << 30           (the argument of g_host.init)
     K3GPU_LAYOUT 0       layout
     K3GPU_PREFILL 0      prefill     = env_int != 0
     K3GPU_ROUTECACHE     routecache  = the variable's value if it is set and not empty, else NULL (a pointer into the environment, not a copy: valid until the environment is changed;
                                         the caller does not free it). Only when routecache != NULL are the next four read, as in the original:
     K3GPU_LOOKAHEAD 92   lookahead
     K3GPU_VQUEUE 0       vqueue      = (env_int != 0) && policy == 2
     K3GPU_VQ_INFLIGHT 256 vq_cap     = (size_t)(int)env_int                              (sign-extended)
     K3GPU_VQ_FROM 0      vq_from     = (long)env_int
   When routecache == NULL the four route-cache fields hold the initial values of the engine globals the original leaves untouched: lookahead 0, vqueue 0, vq_cap 256, vq_from 0
   (K3GPU_LOOKAHEAD, K3GPU_VQUEUE, K3GPU_VQ_INFLIGHT, K3GPU_VQ_FROM are then not looked at at all). K3GPU_TRACE and the three selector variables are not part of the struct
   (ec_open_trace and ec_selector_* read them when called). `out` must be non-NULL. */
typedef struct ec_settings {
    int cards, readers, verbose;
    size_t reserve;
    int policy, chunks;
    int host_gb;
    size_t host_bytes;
    const char *routecache;
    int lookahead;
    int vqueue;
    size_t vq_cap;
    long vq_from;
    int layout;
    int prefill;
} ec_settings;
void ec_settings_from_env(ec_settings *out, int devices);

/* g_pend_cap = g_host.cap_slots * 6 / 10, in size_t arithmetic (the multiplication wraps modulo 2^64). cap_slots is known only after g_host.init, hence a function of its own. */
size_t ec_pend_cap(size_t cap_slots);

/* ---- K3GPU_TRACE. Reads the variable: unset or empty -> returns NULL and does nothing. Otherwise fopen(value, "w") and returns the stream (g_trace). If fopen fails:
   writes "cannot write <value>\n" and exit(1). The returned FILE belongs to the caller. */
FILE *ec_open_trace(void);

/* ---- messages of configure(). All print exactly the original line.
   ec_print_route_cache: "K3GPU route cache: %zu calls from %s, look-ahead %d calls, at most %zu experts read ahead, card queues %s (at most %zu in flight)\n"
       with calls = g_rc.size(), s->routecache, s->lookahead, pend_cap = g_pend_cap, "on" if s->vqueue else "off", s->vq_cap.
       s->routecache must be non-NULL (the original only prints inside `if (rc && *rc)`); with NULL, glibc's printf prints "(null)" and so do both implementations (glibc behaviour, not a promise).
   ec_prefill_start (the body of `if (g_prefill)` without the assignment): if calls == 0 (g_rc.empty()): writes "K3GPU_PREFILL needs K3GPU_ROUTECACHE\n" and exit(1);
       otherwise writes "K3GPU PREFILL ON: timing only. No drive is read, expert bytes are 4 records per layer, routes are taken from the route cache; outputs are not the model's.\n".
       Note the condition is an empty parsed table, not an unset K3GPU_ROUTECACHE: a route-cache file with no lines also stops the run.
   ec_print_start_line: "K3GPU: %d cards, host expert cache %d GB, %d drive readers\n" with s->cards, s->host_gb, s->readers. */
void ec_print_route_cache(const ec_settings *s, size_t calls, size_t pend_cap);
void ec_prefill_start(size_t calls);
void ec_print_start_line(const ec_settings *s);

/* ---- implementation selectors (K3GPU_COPYQ, K3GPU_SLOTCACHE, K3GPU_ROUTETABLE): the parsing of the variable only; dlopen/dlsym, the symbol checks and their fatal messages stay in C++.
   component: EC_COPYQ, EC_SLOTCACHE or EC_ROUTETABLE (anything else is outside the contract). Names below: NAME = COPYQ / SLOTCACHE / ROUTETABLE, label = "copy queue" / "slot cache" /
   "route table", H = copyq.h / slotcache.h / routetable.h, WHAT = "the C++ queue" / "the C++ cache" / "the C++ parser", DIR = copyq / slotcache / routetable.
   ec_selector_mode(component, built): `built` is 1 if the engine was compiled with the Rust contract header (K3GPU_<NAME>_BUILT), else 0. m = the variable's value (NULL if unset).
     built != 0: m NULL, empty or exactly "cpp"          -> writes "K3GPU: <label> cpp\n", returns EC_MODE_CPP
                 m exactly "rust"                        -> writes nothing, returns EC_MODE_RUST (the caller then loads the library, see below)
                 anything else (case and whitespace matter) -> writes "K3GPU_<NAME> must be cpp or rust, not <m>\n" and exit(1)
     built == 0: m NULL, empty or exactly "cpp"          -> writes "K3GPU: <label> cpp\n", returns EC_MODE_CPP
                 anything else, including "rust"         -> writes "K3GPU_<NAME>=<m>: this build has no <H>, only <WHAT> (build with -I src/rust/<DIR>/include)\n" and exit(1)
   ec_selector_lib(component): the value of K3GPU_<NAME>_LIB if it is set and not empty (pointer into the environment), else the static string "libcopyq.so" / "libslotcache.so" /
     "libroutetable.so" (never freed, never NULL).
   ec_selector_announce_rust(component, lib): after the caller loaded `lib` successfully: writes "K3GPU: <label> rust (<lib>)\n" (lib = the string given, raw bytes; NULL prints "(null)" like glibc). */
enum { EC_COPYQ = 0, EC_SLOTCACHE = 1, EC_ROUTETABLE = 2 };
enum { EC_MODE_CPP = 0, EC_MODE_RUST = 1 };
int ec_selector_mode(int component, int built);
const char *ec_selector_lib(int component);
void ec_selector_announce_rust(int component, const char *lib);

/* ---- card_of (526): layout == 1: layer 0 -> 0, else (layer - 1) % cards (C remainder: the sign follows the dividend);
   any other layout value (0, 2, -1, ...): layer 0 -> 0, else min(cards - 1, (layer * cards) / 93) (C division, truncating toward zero). All int arithmetic wraps modulo 2^32 where the
   original would overflow (undefined behaviour there; x86-64 gcc wraps). Intended inputs: cards 1..EC_MAX_CARDS, layer 0..92. Outside the contract: layout == 1 with cards == 0, or
   cards == -1 with layer - 1 == INT_MIN: the original divides by zero / traps (SIGFPE); the Rust library aborts (SIGABRT) in exactly those two cases. Layout != 1 with cards <= 0 is defined
   arithmetic (cards 0: layer 0 -> 0, else -1) and gives the same results in both. */
int ec_card_of(int layout, int cards, int layer);

/* ---- force_routes (374-387), K3GPU_PREFILL. Replaces the router's choice of the call by the route cache's: the experts of g_rc[g_rc_pos], each dealt out to `count` tokens round-robin
   over n tokens (token index s % n, s counting across experts), written to sel[token * top + k] in dealing order.
   g_rc is a C++ std::vector<CallSet> the C ABI cannot take, so the caller passes its size and only the entry the function reads, g_rc[g_rc_pos]:
     pos = g_rc_pos, size = g_rc.size(); cur describes g_rc[pos] and is read only if pos < size (it may be NULL otherwise); top = TOP (16); L = the layer of this call; n = the number of
     tokens (rows); sel = w.h_selected.data(), room for n * top ints (the engine's buffer).
   `experts` / `counts` may be NULL when the matching length is 0. Steps and the single message (written at most once per process, by the first call that takes one of the four exits;
   later such calls are silent; the flag is process-global):
     1. pos >= size, or cur->layer != L                      -> warn "does not match the route cache"; sel untouched
     2. n_counts != n_experts                                -> warn "the route cache has no counts"; sel untouched
     3. total = sum of counts (long); fits = no count > n; if !fits or total != (long)n * top -> warn "counts do not fit this call"; sel untouched
     4. otherwise: fill[0..n) = 0; s = 0; for each expert i, for k in 0 .. counts[i]-1 (a count <= 0 deals nothing): t = (int)(s % n); sel[(size_t)t * top + fill[t]++] = experts[i]; s++.
   warn text: "K3GPU prefill: call %zu (layer %d): <reason>; the computed routes are used from here\n" with pos and L.
   The write in step 4 is NOT bounds-checked, as in the original: with counts that are not all >= 0 (a negative count passes steps 1-3 if the others compensate) one token can receive more than
   top experts and the write runs past sel; the Rust library reproduces the same writes. Outside the contract (the original: std::length_error -> terminate, SIGABRT; the Rust library aborts
   too): n < 0 when step 4 is reached. n == 0 is fine (step 3 then only passes with all counts 0 and nothing is written). top <= 0 is outside the contract (only top >= 1 is tested). */
typedef struct ec_call {
    int layer;
    const int *experts; size_t n_experts;
    const int *counts; size_t n_counts;
} ec_call;
void ec_force_routes(size_t pos, size_t size, const ec_call *cur, int L, int n, int *sel, int top);

/* ---- print_summary (465-485), the atexit handler. The C++ adapter fills one ec_summary from its globals at exit and calls ec_print_summary; the output is byte for byte the original's.
   Field -> original: cards = g_cards; expert_raw = EXPERT_RAW; card[c] = g_pool[c] (ready, slots, st.*; only entries c < cards are read, and only the ready ones are counted);
   copyq_rust = g_copyq_rust, cq_jobs[k] = g_cq_jobs[k].load(); slotcache_rust = g_sc.on, sc_calls = g_sc_calls; routetable_rust = g_rt.on, rc_calls = g_rc.size(),
   rt_nu_calls = g_rt_nu_calls; rc_pos = g_rc_pos; rc_ok = g_rc_ok; pf_planned, pf_skipped, pf_used, pf_waited, t_pwait, wb_skipped = the globals of that name (without g_); vq = g_vq;
   vq_ram, vq_drive, vq_wb, vq_nowb, vq_waited, t_vwait, t_plan = the globals of that name; n_calls, t_calls = g_n_calls, g_t_calls.
   Output, in order (formats are those of the original, doubles with the C library's printf: %.1f %.3f %.0f; "need" = sum of `needed` over ready cards c < cards):
     1. three always-present lines: copy queue, slot cache, route table (cpp/rust by the flags; counts in %lu / %zu)
     2. if need == 0: nothing more.   Otherwise:
     3. the "K3GPU experts:" line (percentages 100.0 * x / need, GB = from_drive * (double)expert_raw / 1e9, summed read/copy seconds in card order)
     4. one "K3GPU card %d:" line per ready card
     5. if rc_calls != 0: the "K3GPU route cache:" line;  if vq != 0: the "K3GPU card queues:" line (vq_ram + vq_drive wraps modulo 2^64)
     6. for each of the three call classes with n_calls[i] != 0 (true for NaN and +-inf, false for +0 and -0): the "K3GPU <name>: ..." line.
   cards must be <= EC_MAX_CARDS (the original indexes arrays of that size); cards < 0 reads no card. cards > EC_MAX_CARDS: outside the contract (Rust aborts, the original reads out of bounds). */
typedef struct ec_card_stats {
    int ready, slots;
    unsigned long needed, hit_card, hit_host, from_drive, written_back;
    double t_read, t_copy;
} ec_card_stats;
typedef struct ec_summary {
    int cards;
    int expert_raw;
    ec_card_stats card[EC_MAX_CARDS];
    int copyq_rust; unsigned long cq_jobs[3];
    int slotcache_rust; unsigned long sc_calls[3];
    int routetable_rust; size_t rc_calls; unsigned long rt_nu_calls;
    size_t rc_pos; int rc_ok;
    unsigned long pf_planned, pf_skipped, pf_used, pf_waited; double t_pwait; unsigned long wb_skipped;
    int vq; unsigned long vq_ram, vq_drive, vq_wb, vq_nowb, vq_waited; double t_vwait, t_plan;
    double n_calls[3], t_calls[3];
} ec_summary;
void ec_print_summary(const ec_summary *s);

#ifdef __cplusplus
}
#endif
#endif
