// S9 (fatal paths) and Q5 (three questions) run each library in its own child process (the fatal path exits the process); the parent compares exit status, stdout transcript and stderr
// byte for byte. The orchestrator for the random scenarios (S1..S7), the engine replay (S8) and main() are at the end.
static const char *FATAL_MSG_TXT = "expert cache: every slot is in use by one call (too few slots)\n";
static int child_nu(int key) { return 100 + key * 3 + (key & 1); }

struct T {   // transcript helper bound to one library and one cache
    Api &a; C *c;
    T(Api &a_) : a(a_), c(a_.nw()) {}
    static unsigned char *S(int n) { return (unsigned char *)(SLOT_BASE + 64 * (uintptr_t)n); }
    void dump(const char *tag)
    {
        std::vector<sc_entry> e(a.size(c) + 8); std::vector<unsigned char *> s(a.snap_spare(c, nullptr, 0) + 8); std::vector<int> nu(a.snap_next(c, nullptr, nullptr, 0) + 8), nk(nu.size());
        const size_t ne = a.snap_lru(c, e.data(), e.size()), ns = a.snap_spare(c, s.data(), s.size()), nn = a.snap_next(c, nu.data(), nk.data(), nu.size());
        printf("  [%s] size=%zu lru(mru first):", tag, a.size(c));
        for (size_t i = 0; i < ne; i++) printf(" (%d #%d h=%.9g/%08x nu=%d)", e[i].key, slotno(e[i].slot), e[i].heat, fbits(e[i].heat), e[i].nu);
        printf("\n      spare(last=next):"); for (size_t i = 0; i < ns; i++) printf(" #%d", slotno(s[i]));
        printf("\n      next(set):"); for (size_t i = 0; i < nn; i++) printf(" (%d,%d)", nu[i], nk[i]);
        printf("\n");
    }
    void spare(int n) { printf("add_spare #%d\n", n); a.add_spare(c, S(n)); }
    void policy(int p) { printf("set_policy %d\n", p); a.set_policy(p); }
    void cb(bool on) { printf("set_next_use %s\n", on ? "child_nu(key)=100+3*key+(key&1)" : "NULL"); a.set_next_use(on ? child_nu : nullptr); }
    static std::string pstr(const std::vector<int> &p) { std::string s = "{"; for (size_t i = 0; i < p.size(); i++) s += sfmt("%s%d", i ? "," : "", p[i]); return s + "}"; }
    void acq(int key, const std::vector<int> &prot, float h0)
    {
        printf("acquire key=%d prot=%s heat0=%.9g\n", key, pstr(prot).c_str(), h0); fflush(stdout);
        int vk = 0x12345678; float vh = bitsf(0xDEADBEEFu);
        unsigned char *p = a.acquire(c, key, prot.empty() ? nullptr : prot.data(), prot.size(), h0, &vk, &vh);
        printf("  -> slot #%d vkey=%d vheat=", slotno(p), vk);
        if (fbits(vh) == 0xDEADBEEFu) printf("(unwritten)\n"); else printf("%.9g/%08x\n", vh, fbits(vh));
    }
    void tryacq(int key, int cand, const std::vector<int> &prot, float h0)
    {
        printf("try_acquire key=%d cand_nu=%d prot=%s heat0=%.9g\n", key, cand, pstr(prot).c_str(), h0); fflush(stdout);
        int vk = 0x12345678; float vh = bitsf(0xDEADBEEFu);
        unsigned char *p = a.try_acquire(c, key, cand, prot.empty() ? nullptr : prot.data(), prot.size(), h0, &vk, &vh);
        printf("  -> slot %s#%d vkey=%d vheat=", p ? "" : "NULL ", slotno(p), vk);
        if (fbits(vh) == 0xDEADBEEFu) printf("(unwritten)\n"); else printf("%.9g/%08x\n", vh, fbits(vh));
    }
    void touch(int key, float w) { printf("touch key=%d w=%.9g\n", key, w); a.touch(c, key, w); }
    void find(int key) { printf("find key=%d -> slot #%d\n", key, slotno(a.find(c, key))); }
    void worth(int key) { printf("worth key=%d -> %d\n", key, a.worth(c, key)); }
    void remove(int key) { unsigned char *p = a.remove(c, key); printf("remove key=%d -> slot %s#%d\n", key, p ? "" : "NULL ", slotno(p)); }
};

typedef void (*CaseFn)(Api &);
struct CaseDef { const char *name; bool fatal; CaseFn fn; };

static void fill3(T &t) { t.spare(3); t.spare(2); t.spare(1); t.acq(1, {}, 1); t.acq(2, {}, 2); t.acq(3, {}, 3); }

static void f_all_prot(Api &a, int pol, int nkeys, bool cbset, bool dups)
{
    T t(a); t.policy(pol); t.cb(false);
    for (int i = nkeys; i >= 1; i--) t.spare(i);
    for (int k = 1; k <= nkeys; k++) t.acq(1000 + k, {}, (float)k);
    if (cbset) t.cb(true);
    std::vector<int> p; for (int k = 1; k <= nkeys; k++) p.push_back(1000 + k);
    if (dups) { p.push_back(1001); p.push_back(1001); p.push_back(7); p.push_back(8); }
    t.dump("before");
    t.acq(5000, p, 1.0f);
    t.dump("after (reached only if the call returned)");
}
static void c_f1(Api &a) { f_all_prot(a, 0, 3, false, false); }
static void c_f2(Api &a) { f_all_prot(a, 1, 3, false, false); }
static void c_f3(Api &a) { f_all_prot(a, 1, 60, false, true); }
static void c_f3b(Api &a) { f_all_prot(a, 0, 60, false, true); }
static void c_f4(Api &a) { f_all_prot(a, 2, 5, true, true); }
static void c_f8(Api &a) { f_all_prot(a, 2, 5, false, false); }
static void c_f5(Api &a)   // 5a: entries stored while belady was off, no spare, nothing protected
{
    T t(a); t.policy(1); t.cb(false); fill3(t); t.dump("stored while belady off");
    t.policy(2); t.cb(true); t.dump("belady on"); t.acq(9, {}, 1.0f); t.dump("after");
}
static void c_f5b(Api &a)  // same with a protected key outside the held set and a duplicate
{
    T t(a); t.policy(2); t.cb(false); fill3(t); t.cb(true); t.acq(9, {77, 77, 1}, 1.0f); t.dump("after");
}
static void c_f6(Api &a) { T t(a); t.policy(2); t.cb(true); t.acq(1, {}, 1.0f); t.dump("after"); }
static void c_f7(Api &a) { T t(a); t.policy(0); t.cb(false); t.acq(1, {}, 1.0f); t.dump("after"); }
static void c_f7b(Api &a) { T t(a); t.policy(1); t.cb(false); t.acq(1, {}, 1.0f); t.dump("after"); }
static void c_f9(Api &a)   // not fatal: entries stored while belady off, then touched under belady so they are in the ordered set
{
    T t(a); t.policy(1); t.cb(false); fill3(t); t.policy(2); t.cb(true);
    t.touch(2, 1.0f); t.dump("after touch(2) under belady"); t.acq(9, {}, 1.0f); t.dump("after");
    t.touch(3, 1.0f); t.touch(1, 1.0f); t.dump("after touches"); t.acq(10, {1}, 1.0f); t.dump("after second acquire (prot {1})");
}
static void c_f10(Api &a)  // not fatal: belady on, a spare exists, entries not in the ordered set
{
    T t(a); t.policy(1); t.cb(false); fill3(t); t.spare(4); t.policy(2); t.cb(true); t.dump("before"); t.acq(9, {1, 2, 3}, 1.0f); t.dump("after");
}
static void c_f11(Api &a)  // belady, the set holds only protected keys but unprotected entries are held (stored while off)
{
    T t(a); t.policy(1); t.cb(false); fill3(t); t.policy(2); t.cb(true); t.touch(1, 1.0f); t.dump("only key 1 in the set"); t.acq(9, {1}, 1.0f); t.dump("after");
}
static void c_q5a4(Api &a)  // belady entries then belady off: heat rule removes the victim and its stale pair
{
    T t(a); t.policy(2); t.cb(true); t.spare(3); t.spare(2); t.spare(1); t.acq(1, {}, 5.0f); t.acq(2, {}, 1.0f); t.acq(3, {}, 9.0f); t.dump("belady, 3 entries");
    t.cb(false); t.dump("function unset"); t.acq(9, {}, 1.0f); t.dump("after acquire without function (heat rule)"); t.cb(true); t.touch(9, 1.0f); t.worth(9); t.dump("after touch/worth with function set again");
}
static void q5b_one(Api &a, int pol, bool cbset)
{
    T t(a); t.policy(pol); t.cb(cbset); t.spare(2); t.spare(1);
    t.acq(1, {}, 2.0f); t.dump("one entry, one spare");
    t.tryacq(5, -1000000000, {1, 1}, 3.0f); t.dump("try_acquire with a spare (cand_nu very small, key 1 protected)");
    t.tryacq(6, -1000000000, {}, 3.0f); t.dump("try_acquire with no spare (cand_nu very small)");
    t.tryacq(7, INT_MAX, {}, 3.0f); t.dump("try_acquire with no spare (cand_nu INT_MAX)");
}
static void c_q5b0(Api &a) { q5b_one(a, 0, false); }
static void c_q5b1(Api &a) { q5b_one(a, 1, false); }
static void c_q5b2n(Api &a) { q5b_one(a, 2, false); }
static void c_q5b2c(Api &a) { q5b_one(a, 2, true); }
static void c_q5b0c(Api &a) { q5b_one(a, 0, true); }
static void q5c_one(Api &a, int pol, bool cbset)
{
    T t(a); t.policy(pol); t.cb(cbset); t.spare(3); t.spare(2); t.spare(1);
    t.acq(1, {}, 2.0f); t.acq(2, {}, 100.0f); t.acq(3, {}, 0.5f); t.dump("heat0 2, 100, 0.5");
    t.touch(1, 1.5f); t.touch(1, 1.5f); t.touch(2, 0.0f); t.touch(3, 0.1f); t.touch(99, 4.0f); t.dump("after touches (1:2.0*.8+1.5, twice; 2; 3; absent 99)");
    t.find(2); t.dump("after find(2)");
    t.acq(4, {}, 1.0f); t.dump("after acquire (victim by the policy)");
}
static void c_q5c0(Api &a) { q5c_one(a, 0, false); }
static void c_q5c1(Api &a) { q5c_one(a, 1, false); }
static void c_q5c0cb(Api &a) { q5c_one(a, 0, true); }
static void c_q5c2c(Api &a) { q5c_one(a, 2, true); }
static void c_q5c2n(Api &a) { q5c_one(a, 2, false); }

static const CaseDef S9_CASES[] = {
    { "S9-f1-policy0-all-protected", true, c_f1 }, { "S9-f2-policy1-all-protected", true, c_f2 }, { "S9-f3-policy1-60-entries-all-protected-dups-absent", true, c_f3 },
    { "S9-f3b-policy0-60-entries-all-protected-dups-absent", true, c_f3b },
    { "S9-f4-belady-all-in-set-protected", true, c_f4 }, { "S9-f5-belady-set-empty-entries-held-no-spare", true, c_f5 },
    { "S9-f5b-belady-entries-held-prot-outside", true, c_f5b }, { "S9-f6-belady-empty-cache-no-spare", true, c_f6 },
    { "S9-f7-policy0-empty-cache-no-spare", true, c_f7 }, { "S9-f7b-policy1-empty-cache-no-spare", true, c_f7b },
    { "S9-f8-policy2-no-function-all-protected", true, c_f8 },
    { "S9-f11-belady-set-only-protected-unprotected-held", true, c_f11 },
    { "S9-f9-belady-touched-entry-is-victim (not fatal)", false, c_f9 }, { "S9-f10-belady-spare-exists (not fatal)", false, c_f10 },
};
static const CaseDef Q5_CASES[] = {
    { "Q5a-1-belady-acquire-entries-from-before-no-spare", true, c_f5 }, { "Q5a-2-belady-entries-from-before-only-touched-is-victim", false, c_f9 },
    { "Q5a-3-belady-then-off-heat-rule-erases-stale-pair", false, c_q5a4 },
    { "Q5b-1-try_acquire-spare-policy0", false, c_q5b0 }, { "Q5b-2-try_acquire-spare-policy1", false, c_q5b1 },
    { "Q5b-3-try_acquire-spare-policy2-no-function", false, c_q5b2n }, { "Q5b-4-try_acquire-spare-policy2-function", false, c_q5b2c },
    { "Q5b-5-try_acquire-spare-policy0-function-set", false, c_q5b0c },
    { "Q5c-1-touch-policy0", false, c_q5c0 }, { "Q5c-2-touch-policy1", false, c_q5c1 }, { "Q5c-3-touch-policy0-function-set", false, c_q5c0cb },
    { "Q5c-4-touch-policy2-function (belady touch)", false, c_q5c2c }, { "Q5c-5-touch-policy2-no-function", false, c_q5c2n },
};

static int child_main(const char *lib, const char *casename)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    Api a; if (!load_api(a, lib)) return 3;
    for (const CaseDef &c : S9_CASES) if (!strcmp(c.name, casename)) { c.fn(a); printf("RETURNED (no exit)\n"); return 0; }
    for (const CaseDef &c : Q5_CASES) if (!strcmp(c.name, casename)) { c.fn(a); printf("RETURNED (no exit)\n"); return 0; }
    fprintf(stderr, "unknown case %s\n", casename); return 4;
}

static std::string slurp(const std::string &p)
{
    std::string s; FILE *f = fopen(p.c_str(), "rb"); if (!f) return s; char b[65536]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n); fclose(f); return s;
}
static std::string safe(const char *n) { std::string s = n; for (char &c : s) if (!isalnum((unsigned char)c) && c != '-') c = '_'; return s; }

struct CaseOut { int status = -1; std::string out, err; };
static CaseOut run_child(const char *self, const char *lib, const char *name, const std::string &dir, int idx)
{
    const std::string base = dir + "/" + safe(name) + ".lib" + std::to_string(idx);
    const pid_t pid = fork();
    if (pid == 0) {
        int fo = open((base + ".out").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644), fe = open((base + ".err").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        dup2(fo, 1); dup2(fe, 2); close(fo); close(fe);
        _exit(child_main(lib, name));
    }
    int st = 0; waitpid(pid, &st, 0);
    CaseOut r; r.status = WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st); r.out = slurp(base + ".out"); r.err = slurp(base + ".err");
    (void)self; return r;
}

static void run_cases(const char *group, const CaseDef *cs, size_t n, const char *self, const char *lib0, const char *lib1, const std::string &outdir, FILE *summary)
{
    const std::string dir = outdir + "/cases"; mkdir(dir.c_str(), 0755);
    size_t pass = 0; std::string detail;
    for (size_t i = 0; i < n; i++) {
        CaseOut r0 = run_child(self, lib0, cs[i].name, dir, 0), r1 = run_child(self, lib1, cs[i].name, dir, 1);
        const bool agree = r0.status == r1.status && r0.out == r1.out && r0.err == r1.err;
        const bool expect = cs[i].fatal ? (r0.status == 1 && r0.err == FATAL_MSG_TXT) : (r0.status == 0 && r0.err.empty());
        const bool ok = agree && expect;
        if (ok) pass++;
        detail += sfmt("  %s %-62s ref: exit %d stderr %zuB%s | rust: exit %d stderr %zuB%s | agree=%d contract-expectation=%d\n", ok ? "PASS" : "FAIL", cs[i].name, r0.status, r0.err.size(), r0.err == FATAL_MSG_TXT ? " (= contract message)" : "",
                       r1.status, r1.err.size(), r1.err == FATAL_MSG_TXT ? " (= contract message)" : "", agree, expect);
    }
    const std::string line = sfmt("%-4s cases=%zu pass=%zu  %s\n", group, n, pass, pass == n ? "PASS" : "FAIL");
    fputs(line.c_str(), stdout); fputs(detail.c_str(), stdout); fputs(line.c_str(), summary); fputs(detail.c_str(), summary); fflush(stdout); fflush(summary);
}

// ------------------------------------------------------------------------------------------------ orchestration of the random scenarios and the engine replay
static const Cfg CFGS[] = {
    // name, what, pol, cbm, style, nc, sizeset, frac, full_ro, all_each, nseeds, nops
    { "S1", "policy 0, random mix, prot 0..32, integral/fractional heat (callback set in half the seeds, unused)", 0, 3, 2, 1, 0, 0, true, false, 64, 40000 },
    { "S2", "policy 1, same mix, fractional+integral heat0 and w", 1, 3, 2, 1, 0, 2, true, false, 64, 40000 },
    { "S3", "policy 2 with a callback (engine lists or hash with ties)", 2, 1, 2, 1, 0, 2, true, false, 64, 40000 },
    { "S4", "policy 2, callback unset and set again at random times", 2, 2, 2, 1, 0, 2, true, false, 64, 40000 },
    { "S5", "policy switched among 0/1/2 mid-run, callback flipping", 3, 2, 2, 1, 0, 2, true, false, 64, 40000 },
    { "S6", "small caches 3-8 slots (key space below and above)", 4, 4, 2, 1, 1, 2, true, false, 64, 30000 },
    { "S6", "large caches 2000-20000 slots (fill phase then mix)", 4, 4, 2, 1, 2, 2, true, false, 24, 60000 },
    { "S7", "nine caches sharing policy and callback", 3, 2, 2, 9, 0, 2, true, true, 64, 40000 },
    { "S10", "policy values outside 0..2 (3, 7, -1, INT_MAX, INT_MIN) mixed with 0/1/2, callback flipping", 5, 2, 2, 1, 0, 2, true, false, 64, 40000 },
    { "S11", "large prot sets (33..3000 keys, a tenth with out-of-range keys), policy switching 0/1/2", 3, 4, 2, 1, 0, 2, true, false, 64, 20000 },
    { "S11", "large prot sets, large caches (2000-20000 slots)", 3, 4, 2, 1, 2, 2, true, false, 24, 30000 },
};
static const int NCFG = (int)(sizeof CFGS / sizeof CFGS[0]);
static const EngCfg ENGS[] = {
    { "S8a-p0-plan", 0, false, 0, 1000, 9000, -1 }, { "S8a-p1-plan", 1, false, 0, 1000, 9000, -1 }, { "S8a-p2-plan", 2, false, 0, 1000, 9000, -1 },
    { "S8a-p2-vq", 2, true, 0, 1000, 9000, -1 }, { "S8a-p2-vq-from92", 2, true, 92, 1000, 9000, -1 },
    { "S8b-p0-plan", 0, false, 0, 100, 400, -1 }, { "S8b-p1-plan", 1, false, 0, 100, 400, -1 }, { "S8b-p2-plan", 2, false, 0, 100, 400, -1 },
    { "S8b-p2-vq", 2, true, 0, 100, 400, -1 },
    { "S8c-p2-plan-mismatch@2000", 2, false, 0, 1000, 9000, 2000 }, { "S8c-p2-vq-small-mismatch@1500", 2, true, 0, 100, 400, 1500 },
};
static const int NENG = (int)(sizeof ENGS / sizeof ENGS[0]);

struct Unit { int kind, idx; uint64_t seed; };

static void worker(int wid, const std::vector<Unit> &units, int jobs, Api *api, const std::vector<CallSet> &rc, double opsmul, const std::string &outdir)
{
    FILE *res = fopen((outdir + "/worker" + std::to_string(wid) + ".units").c_str(), "w");
    setvbuf(res, nullptr, _IOLBF, 0);
    for (size_t u = (size_t)wid; u < units.size(); u += (size_t)jobs) {
        const Unit &un = units[u];
        World w; w.a = api; w.seed = un.seed;
        const char *group = un.kind == 0 ? CFGS[un.idx].name : "S8";
        std::string name = un.kind == 0 ? std::string(CFGS[un.idx].name) + (CFGS[un.idx].sizeset == 1 ? "s" : CFGS[un.idx].sizeset == 2 ? "L" : "") : std::string(ENGS[un.idx].name);
        w.scen = name;
        fprintf(res, "BEGIN %s %s %llu\n", group, name.c_str(), (unsigned long long)un.seed);
        bool ok = true;
        try {
            if (un.kind == 0) rand_run(w, CFGS[un.idx], un.seed, std::max<long>(200, (long)(CFGS[un.idx].nops * opsmul)));
            else eng_unit(w, rc, ENGS[un.idx], un.seed);
        } catch (Mismatch &m) {
            ok = false;
            FILE *ff = fopen((outdir + "/fail-" + safe(name.c_str()) + "-seed" + std::to_string(un.seed) + ".txt").c_str(), "w");
            if (ff) { fputs(m.text.c_str(), ff); fclose(ff); }
        }
        fprintf(res, "UNIT %s %s %llu %llu %s", group, name.c_str(), (unsigned long long)un.seed, (unsigned long long)w.opidx, ok ? "PASS" : "FAIL");
        for (int i = 0; i < NCNT; i++) fprintf(res, " %llu", (unsigned long long)w.cnt[i]);
        fputc('\n', res);
        w.free_all();
    }
    fclose(res);
}

int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "--child")) return child_main(argv[2], argv[3]);
    if (argc < 3) { fprintf(stderr, "usage: %s LIB_REFERENCE LIB_CANDIDATE [--only S1,S2,..,S9,Q5] [--jobs N] [--opsmul F] [--seedmul F] [--out DIR] [--trace FILE]\n", argv[0]); return 2; }
    char r0[4096], r1[4096];
    if (!realpath(argv[1], r0) || !realpath(argv[2], r1)) { fprintf(stderr, "cannot resolve the library paths\n"); return 2; }
    std::string outdir = "out", trace = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/k3gpu/results/p1d.trace", only; int jobs = 32; double opsmul = 1.0, seedmul = 1.0;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--only" && i + 1 < argc) only = argv[++i]; else if (a == "--jobs" && i + 1 < argc) jobs = atoi(argv[++i]);
        else if (a == "--opsmul" && i + 1 < argc) opsmul = atof(argv[++i]); else if (a == "--seedmul" && i + 1 < argc) seedmul = atof(argv[++i]);
        else if (a == "--out" && i + 1 < argc) outdir = argv[++i]; else if (a == "--trace" && i + 1 < argc) trace = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    auto want = [&](const char *g) { return only.empty() || ("," + only + ",").find(std::string(",") + g + ",") != std::string::npos; };
    mkdir(outdir.c_str(), 0755);
    FILE *summary = fopen((outdir + "/summary.txt").c_str(), "w");
    fprintf(summary, "reference: %s\ncandidate: %s\nopsmul %.3f seedmul %.3f jobs %d trace %s\n", r0, r1, opsmul, seedmul, jobs, trace.c_str());
    printf("reference: %s\ncandidate: %s\n", r0, r1); fflush(stdout);

    // units: heavy ones first so that the round-robin spreads them over the workers
    std::vector<Unit> units;
    if (want("S8")) for (int i = 0; i < NENG; i++) units.push_back(Unit{ 1, i, (uint64_t)(100 + i) });
    for (int c = 0; c < NCFG; c++) {
        if (!want(CFGS[c].name)) continue;
        const int ns = std::max(1, (int)(CFGS[c].nseeds * seedmul));
        for (int s = 1; s <= ns; s++) units.push_back(Unit{ 0, c, (uint64_t)s });
    }
    std::stable_sort(units.begin(), units.end(), [](const Unit &a, const Unit &b) {
        auto cost = [](const Unit &u) { return u.kind == 1 ? 1e12 : (CFGS[u.idx].sizeset == 2 ? 1e10 : CFGS[u.idx].nc > 1 ? 1e7 : 1e6); };
        return cost(a) > cost(b);
    });
    std::vector<CallSet> rc;
    if (want("S8")) { if (!load_trace(trace.c_str(), rc) || rc.empty()) { fprintf(stderr, "cannot read the trace %s\n", trace.c_str()); return 2; } }
    if (jobs > (int)units.size()) jobs = (int)std::max<size_t>(1, units.size());

    const double t0 = (double)time(nullptr);
    if (!units.empty()) {
        std::vector<pid_t> pids;
        for (int wid = 0; wid < jobs; wid++) {
            const pid_t pid = fork();
            if (pid == 0) {
                static Api api[2]; if (!load_api(api[0], r0) || !load_api(api[1], r1)) _exit(3);
                worker(wid, units, jobs, api, rc, opsmul, outdir); _exit(0);
            }
            pids.push_back(pid);
        }
        for (size_t i = 0; i < pids.size(); i++) { int st = 0; waitpid(pids[i], &st, 0); if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) { fprintf(summary, "worker %zu ended abnormally: status %d\n", i, st); printf("worker %zu ended abnormally: status %d\n", i, st); } }
    }
    // merge
    struct Agg { size_t units = 0, pass = 0, started = 0; unsigned long long ops = 0; unsigned long long cnt[NCNT] = {}; std::vector<std::string> failed, seeds; };
    std::vector<std::string> groups; std::unordered_map<std::string, Agg> agg;
    auto G = [&](const std::string &g) -> Agg & { if (!agg.count(g)) groups.push_back(g); return agg[g]; };
    FILE *all = fopen((outdir + "/units.txt").c_str(), "w");
    for (int wid = 0; wid < jobs && !units.empty(); wid++) {
        std::string s = slurp(outdir + "/worker" + std::to_string(wid) + ".units"); fputs(s.c_str(), all);
        size_t p = 0;
        while (p < s.size()) {
            size_t e = s.find('\n', p); if (e == std::string::npos) e = s.size(); const std::string ln = s.substr(p, e - p); p = e + 1;
            char tag[16], g[64], nm[128], st[16]; unsigned long long sd, ops, c[NCNT];
            if (!strncmp(ln.c_str(), "BEGIN", 5)) { if (sscanf(ln.c_str(), "BEGIN %63s %127s %llu", g, nm, &sd) == 3) G(g).started++; continue; }
            if (sscanf(ln.c_str(), "%15s %63s %127s %llu %llu %15s", tag, g, nm, &sd, &ops, st) != 6) continue;
            Agg &A = G(g); A.units++; A.ops += ops; if (!strcmp(st, "PASS")) A.pass++; else A.failed.push_back(std::string(nm) + " seed " + std::to_string(sd));
            const char *q = strstr(ln.c_str(), st); q += strlen(st);
            for (int i = 0; i < NCNT; i++) { while (*q == ' ') q++; c[i] = strtoull(q, (char **)&q, 10); A.cnt[i] += c[i]; }
        }
    }
    fclose(all);
    for (const std::string &g : groups) {
        Agg &A = agg[g]; const bool ok = A.units == A.started && A.pass == A.units;
        const std::string line = sfmt("%-4s units=%zu (seeds/variants) ops=%llu  %s%s\n", g.c_str(), A.units, A.ops, ok ? "PASS" : "FAIL", A.units != A.started ? "  (a unit did not finish: worker died)" : "");
        const std::string cov = sfmt("       coverage: acquire=%llu (spare %llu, victims %llu: belady %llu, heat-window %llu, lru-only %llu) try_acquire=%llu (refused %llu) find-hit=%llu touch-held=%llu remove-hit=%llu worth true/false=%llu/%llu callback-calls=%llu skipped-would-exit=%llu vq-issued=%llu\n",
                                     A.cnt[0], A.cnt[1], A.cnt[2], A.cnt[3], A.cnt[4], A.cnt[5], A.cnt[6], A.cnt[7], A.cnt[8], A.cnt[9], A.cnt[10], A.cnt[11], A.cnt[12], A.cnt[13], A.cnt[14], A.cnt[15]);
        fputs(line.c_str(), stdout); fputs(cov.c_str(), stdout); fputs(line.c_str(), summary); fputs(cov.c_str(), summary);
        for (size_t i = 0; i < A.failed.size() && i < 10; i++) { std::string f = "       FAIL " + A.failed[i] + "\n"; fputs(f.c_str(), stdout); fputs(f.c_str(), summary); }
    }
    fflush(stdout);
    if (want("S9")) run_cases("S9", S9_CASES, sizeof S9_CASES / sizeof S9_CASES[0], argv[0], r0, r1, outdir, summary);
    if (want("Q5")) run_cases("Q5", Q5_CASES, sizeof Q5_CASES / sizeof Q5_CASES[0], argv[0], r0, r1, outdir, summary);
    const std::string tm = sfmt("wall %.0f s\n", (double)time(nullptr) - t0); fputs(tm.c_str(), stdout); fputs(tm.c_str(), summary);
    fclose(summary);
    return 0;
}
