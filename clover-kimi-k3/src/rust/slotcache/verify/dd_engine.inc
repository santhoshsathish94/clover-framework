// S8: engine replay. The SlotCache call sequence of pool_ensure (decode.cuh 427-505) and of the planners rc_plan / rc_plan_vq / vq_issue / vq_sweep (backend-gpu.cu 206-310),
// driven over the real trace; every call goes to both libraries through World::step (return values, out-parameters, log, full state of the touched cache after each).
// Asynchronous completion (reading threads) is simulated: each pending read / copy completes with probability 0.3 before each MoE call (seeded).
// Not modelled: drive reads, CUDA streams, copies, the Handle / fd checks (every layer is assumed to have a handle), the real timing of the completions.
struct CallSet { int layer; std::vector<int> experts, counts; };
enum { EXPERTS_N = 896, N_LAYERS_N = 92, CARDS_N = 8 };
static inline int ekey_(int layer, int e) { return layer * EXPERTS_N + e; }
static inline int card_of_(int L) { return L == 0 ? 0 : std::min(CARDS_N - 1, (L * CARDS_N) / 93); }

static bool load_trace(const char *path, std::vector<CallSet> &rc)   // rc_load (backend-gpu.cu 338-358)
{
    FILE *f = fopen(path, "r"); if (!f) return false;
    char *line = nullptr; size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        char *p = line; char *end;
        CallSet cs; cs.layer = (int)strtol(p, &end, 10); p = end; strtol(p, &end, 10); p = end;
        for (;;) {
            while (*p == ' ') p++;
            if (*p < '0' || *p > '9') break;
            cs.experts.push_back((int)strtol(p, &end, 10)); p = end;
            if (*p == ':') { cs.counts.push_back((int)strtol(p + 1, &end, 10)); p = end; }
        }
        rc.push_back(std::move(cs));
    }
    free(line); fclose(f);
    return true;
}

struct EngCfg { const char *name; int policy; bool vq; long vq_from; int pool_slots; int host_slots; long mismatch_at; };

struct Eng {
    World &w; const std::vector<CallSet> &rc; EngCfg cf;
    struct Pend { int state; unsigned char *slot; };
    struct Rel { int state; unsigned char *slot; };
    std::unordered_map<int, Pend> pend; std::unordered_set<int> pend_keys;
    std::vector<Rel> rel; std::vector<int> wb_keys; std::unordered_map<int, int> arr;
    size_t rc_planned = 0, pend_cap = 0, vq_cap = 256; bool rc_ok = true; int lookahead = 92; size_t cap_host = 0;
    std::vector<uint32_t> dptr;                 // the driver's own copy of g_use_ptr (rc_nu for cand_nu)
    Rng R;
    uint64_t skipped = 0, written_back = 0, wb_skipped = 0, from_host = 0, pool_hits = 0, tried = 0, issued = 0, planned = 0;

    Eng(World &w_, const std::vector<CallSet> &rc_, EngCfg c, uint64_t seed) : w(w_), rc(rc_), cf(c), R(seed) {}

    static const int HOST = 0;
    static int P(int card) { return 1 + card; }

    std::vector<int> keys_of(const std::unordered_set<int> &s) { std::vector<int> v(s.begin(), s.end()); std::sort(v.begin(), v.end()); return v; }
    unsigned char *find(int ci, int key) { return (unsigned char *)(uintptr_t)w.step(mkop(K_FIND, ci, key)).slot; }
    void touch(int ci, int key, float h) { Op o = mkop(K_TOUCH, ci, key); o.f = h; w.step(o); }
    bool contains(int ci, int key) { return w.step(mkop(K_CONTAINS, ci, key)).ival != 0; }
    bool worth(int ci, int key) { return w.step(mkop(K_WORTH, ci, key)).ival != 0; }
    unsigned char *remove(int ci, int key) { return (unsigned char *)(uintptr_t)w.step(mkop(K_REMOVE, ci, key)).slot; }
    void release(int ci, unsigned char *s) { Op o = mkop(K_RELEASE, ci); o.slot = s; w.step(o); }
    unsigned char *acquire(int ci, int key, const std::vector<int> &prot, float heat0, int *vk, float *vh)
    {
        *vk = -1;
        if (!w.acq_ok(w.d[ci], prot)) { skipped++; return nullptr; }      // the engine would exit here ("too few slots"); not a legal sequence for the comparison
        Op o = mkop(K_ACQ, ci, key); o.f = heat0; o.prot = prot; const Res r = w.step(o);
        *vk = (int)(uint32_t)r.vkey; if (*vk >= 0) *vh = bitsf((uint32_t)r.vheat);
        return (unsigned char *)(uintptr_t)r.slot;
    }
    unsigned char *try_acquire(int ci, int key, int cand, const std::vector<int> &prot, float heat0, int *vk, float *vh)
    {
        *vk = -1;
        if (!w.try_ok(w.d[ci], prot, cand)) { skipped++; return nullptr; }
        Op o = mkop(K_TRY, ci, key); o.cand = cand; o.f = heat0; o.prot = prot; const Res r = w.step(o);
        *vk = (int)(uint32_t)r.vkey; if (*vk >= 0) *vh = bitsf((uint32_t)r.vheat);
        return (unsigned char *)(uintptr_t)r.slot;
    }
    int drv_nu(int key)   // rc_nu of the engine, on the driver's own pointer state
    {
        if (key < 0 || (size_t)key >= g_model.uses.size() || g_model.uses[key].empty()) return 1000000000;
        const std::vector<int> &l = g_model.uses[key]; uint32_t &p = dptr[key];
        while (p < l.size() && l[p] <= g_model.cur) p++;
        return p < l.size() ? l[p] : 1000000000;
    }
    void pend_take(int key) { auto f = pend.find(key); if (f == pend.end()) return; pend.erase(f); pend_keys.erase(key); }
    void arr_wait(int key) { auto f = arr.find(key); if (f == arr.end()) return; arr.erase(f); }

    void advance()
    {
        for (auto &kv : pend) if (kv.second.state == 0 && R.p(0.3)) kv.second.state = 1;
        for (auto &r : rel) if (r.state == 0 && R.p(0.3)) r.state = 1;
        for (auto &kv : arr) if (kv.second == 0 && R.p(0.3)) kv.second = 1;
    }

    void pool_ensure(int card, int layer, const std::vector<int> &experts, const int *counts)
    {
        const int pi = P(card); const bool vq = cf.vq;
        std::vector<int> prot; for (int e : experts) prot.push_back(ekey_(layer, e));
        struct Need { int e; unsigned char *dst, *host; }; struct Victim { int key; float heat; unsigned char *src; };
        std::vector<Need> need; std::vector<Victim> victims;
        for (int e : experts) {
            const int key = ekey_(layer, e);
            unsigned char *p = find(pi, key);
            if (p) { if (vq) arr_wait(key); touch(pi, key, (float)counts[e]); pool_hits++; continue; }
            int vk; float vh = 0;
            p = acquire(pi, key, prot, (float)counts[e], &vk, &vh);
            if (!p) continue;
            if (vk >= 0) { if (vq) arr_wait(vk); victims.push_back(Victim{ vk, vh, p }); }
            need.push_back(Need{ e, p, nullptr });
        }
        if (need.empty()) return;
        std::vector<unsigned char *> held;
        for (auto &n : need) {
            const int key = ekey_(layer, n.e);
            if (find(HOST, key)) { pend_take(key); n.host = remove(HOST, key); held.push_back(n.host); from_host++; }
        }
        for (auto &v : victims) {
            if (!cap_host) break;
            if (!worth(HOST, v.key)) { wb_skipped++; continue; }
            int dk; float dh = 0;
            if (acquire(HOST, v.key, keys_of(pend_keys), v.heat, &dk, &dh)) written_back++;
        }
        for (unsigned char *h : held) release(HOST, h);
    }

    void rc_plan(size_t pos)
    {
        const size_t last = std::min(rc.size(), pos + (size_t)lookahead + 1);
        for (size_t i = std::max(rc_planned, pos); i < last; i++) {
            const CallSet &cs = rc[i]; const int pi = P(card_of_(cs.layer));
            for (int e : cs.experts) {
                const int key = ekey_(cs.layer, e);
                if (pend.count(key)) continue;
                if (contains(pi, key) || contains(HOST, key)) continue;
                if (pend.size() >= pend_cap) { rc_planned = i; return; }
                if (!worth(HOST, key)) { rc_planned = i; return; }
                int vk; float vh = 0;
                unsigned char *slot = acquire(HOST, key, keys_of(pend_keys), 1.0f, &vk, &vh);
                if (!slot) { rc_planned = i; return; }
                pend[key] = Pend{ 0, slot }; pend_keys.insert(key); planned++;
            }
            rc_planned = i + 1;
        }
    }

    void vq_sweep()
    {
        for (size_t i = 0; i < rel.size();) { if (rel[i].state == 1) { release(HOST, rel[i].slot); rel[i] = rel.back(); rel.pop_back(); } else i++; }
        for (size_t i = 0; i < wb_keys.size();) {
            auto f = pend.find(wb_keys[i]);
            if (f == pend.end()) { wb_keys[i] = wb_keys.back(); wb_keys.pop_back(); }
            else if (f->second.state == 1) { pend_keys.erase(f->first); pend.erase(f); wb_keys[i] = wb_keys.back(); wb_keys.pop_back(); }
            else i++;
        }
    }

    bool vq_issue(int card, int key, int cand_nu, const std::vector<int> &prot)
    {
        const int pi = P(card);
        const bool in_ram = contains(HOST, key);
        if (in_ram) { auto pf = pend.find(key); if (pf != pend.end() && pf->second.state == 0) return false; }
        int vk; float vh = 0;
        tried++;
        unsigned char *dst = try_acquire(pi, key, cand_nu, prot, 1.0f, &vk, &vh);
        if (!dst) return false;
        issued++;
        if (in_ram) { pend_take(key); unsigned char *hs = remove(HOST, key); rel.push_back(Rel{ 0, hs }); }
        if (vk >= 0) {
            arr_wait(vk);
            if (cap_host && pend.size() < cap_host * 8 / 10 && worth(HOST, vk)) {
                int dk; float dh = 0;
                unsigned char *h = acquire(HOST, vk, keys_of(pend_keys), vh, &dk, &dh);
                if (h) { pend[vk] = Pend{ 0, h }; pend_keys.insert(vk); wb_keys.push_back(vk); }
            }
        }
        arr[key] = 0;
        return true;
    }

    void rc_plan_vq(size_t pos, const std::vector<int> &cur)
    {
        vq_sweep();
        std::vector<int> prot; for (int e : cur) prot.push_back(ekey_(rc[pos].layer, e));
        size_t out = 0; for (auto &kv : arr) if (kv.second == 0) out++;
        bool full[CARDS_N] = {};
        const size_t last = std::min(rc.size(), pos + (size_t)lookahead + 1);
        for (size_t i = pos; i < last; i++) {
            const CallSet &cs = rc[i]; const int card = card_of_(cs.layer); const int pi = P(card);
            for (int e : cs.experts) {
                const int key = ekey_(cs.layer, e);
                if (contains(pi, key)) continue;
                if (!full[card] && out < vq_cap) {
                    if (vq_issue(card, key, i == pos ? (int)pos : drv_nu(key), prot)) { out++; continue; }
                    full[card] = true;
                }
                if (pend.count(key) || contains(HOST, key)) continue;
                if (pend.size() >= pend_cap || !worth(HOST, key)) continue;
                int vk; float vh = 0;
                unsigned char *slot = acquire(HOST, key, keys_of(pend_keys), 1.0f, &vk, &vh);
                if (!slot) continue;
                pend[key] = Pend{ 0, slot }; pend_keys.insert(key); planned++;
            }
        }
    }

    void run()
    {
        // caches: 0 = host, 1..8 = card pools; slots added in descending address order as the engine does
        for (int c = 0; c < 1 + CARDS_N; c++) {
            Dual &dd = w.add_cache(); const int n = c == 0 ? cf.host_slots : cf.pool_slots;
            std::vector<unsigned char *> s; for (int i = 0; i < n; i++) s.push_back(w.fresh());
            dd.total = (size_t)n;
            for (int i = n; i-- > 0;) { Op o = mkop(K_ADD, c); o.slot = s[(size_t)i]; w.step(o); }
        }
        cap_host = (size_t)cf.host_slots; pend_cap = cap_host * 6 / 10;
        { Op o = mkop(K_POLICY, -1); o.ival = cf.policy; w.step(o); }
        { Op o = mkop(K_SETNU, -1); o.ival = cf.policy == 2 ? 1 : 0; w.step(o); }
        g_model.mode = 0; dptr.assign(g_model.uses.size(), 0);
        for (size_t pos = 0; pos < rc.size(); pos++) {
            const CallSet &cs = rc[pos];
            advance();
            g_model.cur = (long)pos;
            if (rc_ok) {
                if (cf.mismatch_at >= 0 && (long)pos == cf.mismatch_at) { rc_ok = false; Op o = mkop(K_SETNU, -1); o.ival = 0; w.step(o); }
                else if (cf.vq && (long)pos >= cf.vq_from) rc_plan_vq(pos, cs.experts);
                else rc_plan(pos);
            }
            std::vector<int> cnt(EXPERTS_N, 0);
            for (size_t i = 0; i < cs.experts.size(); i++) cnt[(size_t)cs.experts[i]] = cs.counts[i];
            const int card = card_of_(cs.layer);
            for (size_t gi = 0; gi < cs.experts.size();) {
                const size_t gj = std::min(cs.experts.size(), gi + (size_t)cf.pool_slots);
                const std::vector<int> grp(cs.experts.begin() + (long)gi, cs.experts.begin() + (long)gj);
                pool_ensure(card, cs.layer, grp, cnt.data());
                gi = gj;
            }
            w.cmp_all();
        }
    }
};

static void eng_unit(World &w, const std::vector<CallSet> &rc, const EngCfg &cf, uint64_t seed)
{
    const size_t NK = (size_t)(N_LAYERS_N + 1) * EXPERTS_N;   // ekey = layer * EXPERTS + e, layers 1..92
    model_reset(NK);
    g_model.mode = 0; g_model.uses.assign(NK, std::vector<int>());
    for (size_t i = 0; i < rc.size(); i++) for (int e : rc[i].experts) g_model.uses[(size_t)ekey_(rc[i].layer, e)].push_back((int)i);
    w.full_ro = true; w.all_each_op = false;
    Eng E(w, rc, cf, seed);
    E.run();
    w.cnt[14] = E.skipped; w.cnt[15] = E.issued;
}
