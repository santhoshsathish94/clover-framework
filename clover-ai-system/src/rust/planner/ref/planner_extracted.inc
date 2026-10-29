struct Pending { std::atomic<int> state; unsigned char *slot; Pending() : state(0), slot(nullptr) {} };   // 0 queued or being read, 1 read
static std::unordered_map<int, Pending *> g_pend;
static std::unordered_set<int> g_pend_keys;
static unsigned long g_wb_skipped = 0;   // policy 2: experts a card gave up that were not worth a RAM slot (needed later than everything held there)
static double g_t_pwait = 0; static unsigned long g_pf_used = 0, g_pf_waited = 0;
static void pend_take(int key)   // waits until the expert's bytes are in its RAM slot, then forgets the entry
{
    auto f = g_pend.find(key);
    if (f == g_pend.end()) return;
    Pending *p = f->second;
    if (p->state.load(std::memory_order_acquire) == 0) {
        const double t0 = wall_now();
        while (p->state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        g_t_pwait += wall_now() - t0; g_pf_waited++;
    }
    g_pf_used++; g_pend.erase(f); g_pend_keys.erase(key); delete p;
}
struct Arrival { std::atomic<int> state; Arrival() : state(0) {} };
static std::unordered_map<int, std::shared_ptr<Arrival>> g_arr;
static double g_t_vwait = 0; static unsigned long g_vq_waited = 0;
static void arr_wait(int key)
{
    auto f = g_arr.find(key);
    if (f == g_arr.end()) return;
    std::shared_ptr<Arrival> a = f->second; g_arr.erase(f);
    if (a->state.load(std::memory_order_acquire) == 0) {
        const double t0 = wall_now();
        while (a->state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        g_t_vwait += wall_now() - t0; g_vq_waited++;
    }
}
static size_t g_rc_pos = 0, g_rc_planned = 0, g_pend_cap = 0;
static int g_lookahead = 0; static bool g_rc_ok = false;
static unsigned long g_pf_planned = 0, g_pf_skipped = 0;
struct Rel { std::atomic<int> state; unsigned char *slot; Rel() : state(0), slot(nullptr) {} };
struct PfJob {
    int fd = -1; off_t off = 0; unsigned char *slot = nullptr; Pending *p = nullptr; int layer = 0;
    int kind = 0, dev = 0; unsigned char *dst = nullptr, *host_src = nullptr, *victim_host = nullptr; Pending *wb = nullptr;
    std::shared_ptr<Arrival> arr; std::shared_ptr<Rel> rel;
};
static void rc_plan(size_t pos)
{
    const size_t last = std::min(g_rc.size(), pos + (size_t)g_lookahead + 1);
    for (size_t i = std::max(g_rc_planned, pos); i < last; i++) {
        const CallSet &cs = g_rc[i]; Handle *hd = g_handles[cs.layer];
        if (!hd || hd->fd < 0) { g_rc_planned = i + 1; continue; }
        CardPool &pool = g_pool[hd->card];
        for (int e : cs.experts) {
            const int key = ekey(cs.layer, e);
            if (g_pend.count(key)) continue;
            if ((pool.ready && pool.c.contains(key)) || g_host.c.contains(key)) { g_pf_skipped++; continue; }
            if (g_pend.size() >= g_pend_cap) { g_rc_planned = i; return; }   // the RAM window is full; planning goes on at the next call
            if (!g_host.c.worth(key)) { g_rc_planned = i; return; }          // policy 2: RAM holds only experts needed sooner than this one
            int vk; float vh;
            unsigned char *slot = g_host.c.acquire(key, g_pend_keys, 1.0f, &vk, &vh);
            Pending *p = new Pending; p->slot = slot;
            g_pend[key] = p; g_pend_keys.insert(key); g_pf_planned++;
            { PfJob j; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.slot = slot; j.p = p; pf_submit(j); }
        }
        g_rc_planned = i + 1;
    }
}
static std::vector<std::shared_ptr<Rel>> g_rel;
static std::vector<int> g_wb_keys;
static size_t g_vq_cap = 256;
static long g_vq_from = 0;   // K3GPU_VQ_FROM: the card queues work from this call on; before it the experts are only read ahead into RAM (92 = decode starts after a 92-call prompt pass)
static double g_t_plan = 0;   // seconds spent in the planning of the hook (main thread, on the critical path)
static unsigned long g_vq_ram = 0, g_vq_drive = 0, g_vq_wb = 0, g_vq_nowb = 0;

static void vq_sweep()
{
    for (size_t i = 0; i < g_rel.size();) {
        if (g_rel[i]->state.load(std::memory_order_acquire) == 1) { g_host.c.release(g_rel[i]->slot); g_rel[i] = g_rel.back(); g_rel.pop_back(); } else i++;
    }
    for (size_t i = 0; i < g_wb_keys.size();) {
        auto f = g_pend.find(g_wb_keys[i]);
        if (f == g_pend.end()) { g_wb_keys[i] = g_wb_keys.back(); g_wb_keys.pop_back(); }
        else if (f->second->state.load(std::memory_order_acquire) == 1) { g_pend_keys.erase(f->first); delete f->second; g_pend.erase(f); g_wb_keys[i] = g_wb_keys.back(); g_wb_keys.pop_back(); }
        else i++;
    }
}

// false if the pool takes nothing more for this expert now
static bool vq_issue(CardPool &pool, Handle *hd, int key, int e, int cand_nu, const std::unordered_set<int> &prot)
{
    const bool in_ram = g_host.c.contains(key);
    if (in_ram) { auto pf = g_pend.find(key); if (pf != g_pend.end() && pf->second->state.load(std::memory_order_acquire) == 0) return false; }   // its read has not finished
    int vk; float vh;
    unsigned char *dst = pool.c.try_acquire(key, cand_nu, prot, 1.0f, &vk, &vh);
    if (!dst) return false;
    PfJob j; j.dev = pool.dev; j.dst = dst; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.layer = hd->L;
    if (in_ram) {
        pend_take(key);
        j.host_src = g_host.c.remove(key); j.kind = 2;
        j.rel = std::make_shared<Rel>(); j.rel->slot = j.host_src; g_rel.push_back(j.rel); g_vq_ram++;
    } else { j.kind = 1; g_vq_drive++; }
    if (vk >= 0) {
        arr_wait(vk);   // the held expert was itself still on its way
        if (!g_prefill && g_host.cap_slots && g_pend.size() < g_host.cap_slots * 8 / 10 && g_host.c.worth(vk)) {
            int dk; float dh;
            j.victim_host = g_host.c.acquire(vk, g_pend_keys, vh, &dk, &dh);
            j.wb = new Pending; j.wb->slot = j.victim_host;
            g_pend[vk] = j.wb; g_pend_keys.insert(vk); g_wb_keys.push_back(vk); g_vq_wb++;
        } else g_vq_nowb++;
    }
    j.arr = std::make_shared<Arrival>(); g_arr[key] = j.arr;
    pf_submit(j);
    return true;
}

static void rc_plan_vq(size_t pos, const std::vector<int> &cur)
{
    vq_sweep();
    std::unordered_set<int> prot;
    for (int e : cur) prot.insert(ekey(g_rc[pos].layer, e));
    size_t out = 0;
    for (auto &kv : g_arr) if (kv.second->state.load(std::memory_order_acquire) == 0) out++;
    bool full[MAX_CARDS] = {};
    const size_t last = std::min(g_rc.size(), pos + (size_t)g_lookahead + 1);
    for (size_t i = pos; i < last; i++) {
        const CallSet &cs = g_rc[i]; Handle *hd = g_handles[cs.layer];
        if (!hd || hd->fd < 0) continue;
        CardPool &pool = g_pool[hd->card];
        for (int e : cs.experts) {
            const int key = ekey(cs.layer, e);
            if (pool.ready && pool.c.contains(key)) continue;   // on the card or on its way
            if (pool.ready && !full[hd->card] && out < g_vq_cap) {
                if (vq_issue(pool, hd, key, e, i == pos ? (int)pos : rc_nu_sel(key), prot)) { out++; continue; }
                full[hd->card] = true;
            }
            if (g_pend.count(key) || g_host.c.contains(key)) continue;
            if (g_pend.size() >= g_pend_cap || !g_host.c.worth(key)) continue;
            int vk; float vh;
            unsigned char *slot = g_host.c.acquire(key, g_pend_keys, 1.0f, &vk, &vh);
            Pending *p = new Pending; p->slot = slot;
            g_pend[key] = p; g_pend_keys.insert(key); g_pf_planned++;
            PfJob j; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.slot = slot; j.p = p; pf_submit(j);
        }
    }
}
static void route_hook(int L, const std::vector<int> &uniq)
{
    rc_set_cur((long)g_rc_pos);
    if (g_rc_ok) {
        if (g_rc_pos >= g_rc.size() || g_rc[g_rc_pos].layer != L || g_rc[g_rc_pos].experts != uniq) {
            fprintf(stderr, "K3GPU route cache: call %zu (layer %d) does not match the cache; look-ahead stopped\n", g_rc_pos, L);
            g_rc_ok = false; slotcache_set_next_use(nullptr);
        } else { const double tp = wall_now(); if (g_vq && (long)g_rc_pos >= g_vq_from) rc_plan_vq(g_rc_pos, uniq); else rc_plan(g_rc_pos); g_t_plan += wall_now() - tp; }
    }
    g_rc_pos++;
}
