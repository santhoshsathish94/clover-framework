// Hand probes: runs one hand-made input against the reference and a candidate library (dlopen) and prints both outcomes side by side.
//   hand REF.so CAND.so force NAME        NAME in: u8 u16 total32 ntop32 s32 morecounts neg-compensated
//   hand REF.so CAND.so summary NEEDED HIT_CARD HIT_HOST FROM_DRIVE EXPERT_RAW N_CALLS1 T_CALLS1      (one ready card, cards = 1)
// Each library runs in its own forked child (stderr to a file, exit status/signal, FNV hash of the whole sel buffer and of the words that differ from the prefill pattern).
// Build: g++ -O2 -std=c++17 -I../../include -o hand hand.cpp -ldl     (from verify/mutants)
#include "engcfg.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <vector>
using std::string;
using std::vector;

struct Lib { void *h; void (*force)(size_t, size_t, const ec_call *, int, int, int *, int); void (*summary)(const ec_summary *); };
static Lib load(const char *p)
{
    Lib l; l.h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    if (!l.h) { fprintf(stderr, "dlopen %s: %s\n", p, dlerror()); exit(2); }
    *(void **)&l.force = dlsym(l.h, "ec_force_routes"); *(void **)&l.summary = dlsym(l.h, "ec_print_summary");
    if (!l.force || !l.summary) { fprintf(stderr, "missing symbol in %s\n", p); exit(2); }
    return l;
}
static string slurp(const char *p) { string s; FILE *f = fopen(p, "rb"); if (!f) return s; char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n); fclose(f); return s; }
static string esc(const string &s, size_t max = 400) { string o; for (unsigned char c : s) { if (o.size() > max) { o += "..."; break; } if (c == '\n') o += "\\n"; else if (c >= 32 && c < 127) o += (char)c; else { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; } } return o; }

struct Scn { int n, top; size_t pos, size; int L; vector<int> experts, counts; size_t words; };
static const int PAT = 0x5A5A5A5A;

static string run_force(const Lib &lib, const Scn &s, const char *tag)
{
    const string ep = string("/tmp/hand-") + tag + ".err", rp = string("/tmp/hand-") + tag + ".res";
    fflush(NULL);
    const pid_t pid = fork();
    if (pid == 0) {
        alarm(900);
        int efd = open(ep.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); dup2(efd, 2); close(efd);
        ec_call cur; cur.layer = s.L; cur.experts = s.experts.data(); cur.n_experts = s.experts.size(); cur.counts = s.counts.data(); cur.n_counts = s.counts.size();
        int *sel = (int *)malloc(s.words * 4);
        if (!sel) _exit(77);
        for (size_t i = 0; i < s.words; i++) sel[i] = PAT;
        lib.force(s.pos, s.size, &cur, s.L, s.n, sel, s.top);
        uint64_t h = 1469598103934665603ull, nd = 0, first = (uint64_t)-1;
        for (size_t i = 0; i < s.words; i++) { h = (h ^ (uint32_t)sel[i]) * 1099511628211ull; if (sel[i] != PAT) { if (first == (uint64_t)-1) first = i; nd++; } }
        FILE *r = fopen(rp.c_str(), "wb"); fprintf(r, "sel_hash=%016llx words_changed=%llu first_changed=%lld sel[0..3]=%d,%d,%d,%d", (unsigned long long)h, (unsigned long long)nd, (long long)first, sel[0], sel[1 < s.words ? 1 : 0], sel[2 < s.words ? 2 : 0], sel[3 < s.words ? 3 : 0]); fclose(r);
        _exit(0);
    }
    int st = 0; waitpid(pid, &st, 0);
    string o = WIFEXITED(st) ? "exit=" + std::to_string(WEXITSTATUS(st)) : "signal=" + std::to_string(WTERMSIG(st));
    o += " stderr=[" + esc(slurp(ep.c_str())) + "] " + slurp(rp.c_str());
    unlink(ep.c_str()); unlink(rp.c_str());
    return o;
}

static void do_force(const char *refp, const char *candp, const string &name)
{
    Scn s; s.pos = 0; s.size = 1; s.L = 3;
    if (name == "u8" || name == "u16") {   // n = 1, top = 1: one token receives every deal; a negative count compensates, so the sum is 1
        const int m = name == "u8" ? 300 : 70000;
        s.n = 1; s.top = 1;
        for (int i = 0; i < m; i++) { s.experts.push_back(1000 + i); s.counts.push_back(1); }
        s.experts.push_back(5); s.counts.push_back(-(m - 1));
        s.words = (size_t)m + 1024;
    } else if (name == "total32") {   // true total = 2^32 + n*top: rejected by the original; a 32-bit total wraps to n*top and the call is accepted
        s.n = 200; s.top = 16;
        const int64_t want = (1ll << 32) + 200ll * 16;
        const int64_t E = want / 200;
        for (int64_t i = 0; i < E; i++) { s.experts.push_back((int)(i % 1000)); s.counts.push_back(200); }
        s.experts.push_back(7); s.counts.push_back((int)(want - E * 200));
        s.words = 3200 + (size_t)(E + 2) + 1024;
    } else if (name == "ntop32") {   // n * top = 2^32 (wraps to 0 in 32 bits), all counts 0 (total 0): the original rejects, a 32-bit n*top accepts and deals nothing
        s.n = 4096; s.top = 1 << 20;
        s.experts = { 1, 2 }; s.counts = { 0, 0 };
        s.words = 4096;
    } else if (name == "s32") {   // 3 * 2^30 deals: the dealing counter passes 2^31
        s.n = 1 << 30; s.top = 2;
        s.experts = { 11, 12, 13, 14 }; s.counts = { 1 << 30, 1 << 30, 1 << 30, -(1 << 30) };
        s.words = (size_t)1 << 31; s.words += 1024;
    } else if (name == "morecounts") {   // one more count than experts
        s.n = 1; s.top = 1; s.experts = { 7 }; s.counts = { 1, 0 }; s.words = 64;
    } else if (name == "neg-compensated") {   // a negative count compensating a raise: in contract, writes run past n*top (reference behaviour)
        s.n = 2; s.top = 2; s.experts = { 1, 2, 3, 4, 5 }; s.counts = { 2, 2, 2, 1, -3 }; s.words = 4 + 512;
    } else { fprintf(stderr, "unknown force scenario %s\n", name.c_str()); exit(2); }
    Lib R = load(refp), C = load(candp);
    printf("scenario %s: n=%d top=%d experts=%zu counts=%zu sel words=%zu\n", name.c_str(), s.n, s.top, s.experts.size(), s.counts.size(), s.words);
    const string a = run_force(R, s, "ref"), b = run_force(C, s, "cand");
    printf("  reference: %s\n  candidate: %s\n  => %s\n", a.c_str(), b.c_str(), a == b ? "SAME" : "DIFFERENT");
}

static string run_summary(const Lib &lib, const ec_summary &s, const char *tag)
{
    const string ep = string("/tmp/hand-") + tag + ".err";
    fflush(NULL);
    const pid_t pid = fork();
    if (pid == 0) { int efd = open(ep.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); dup2(efd, 2); close(efd); lib.summary(&s); _exit(0); }
    int st = 0; waitpid(pid, &st, 0);
    const string o = slurp(ep.c_str()); unlink(ep.c_str()); return o;
}
static void do_summary(const char *refp, const char *candp, char **a)
{
    ec_summary s; memset(&s, 0, sizeof s);
    s.cards = 1; s.expert_raw = (int)atol(a[4]);
    s.card[0].ready = 1; s.card[0].needed = strtoul(a[0], 0, 10); s.card[0].hit_card = strtoul(a[1], 0, 10); s.card[0].hit_host = strtoul(a[2], 0, 10); s.card[0].from_drive = strtoul(a[3], 0, 10);
    s.n_calls[1] = atof(a[5]); s.t_calls[1] = atof(a[6]);
    Lib R = load(refp), C = load(candp);
    const string x = run_summary(R, s, "sref"), y = run_summary(C, s, "scand");
    printf("reference:\n%s\ncandidate:\n%s\n=> %s\n", x.c_str(), y.c_str(), x == y ? "SAME" : "DIFFERENT");
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: hand REF CAND force NAME | summary NEEDED HIT_CARD HIT_HOST FROM_DRIVE EXPERT_RAW N_CALLS1 T_CALLS1\n"); return 2; }
    if (!strcmp(argv[3], "force")) do_force(argv[1], argv[2], argv[4]);
    else if (!strcmp(argv[3], "summary") && argc == 11) do_summary(argv[1], argv[2], argv + 4);
    else return 2;
    return 0;
}
