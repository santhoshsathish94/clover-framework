// Slots with replacement of the least recently used entry (policy 0), or (policy 1) of the entry with the lowest heat among the 48 least recently used ones.
// An entry's heat is decayed by 0.8 at every use and raised by the number of tokens that used the expert in that call, so after a prompt pass the experts
// that many prompt tokens chose stay and the ones chosen by a single token go first. Slots in use by the current call are named in `prot` and are not replaced.
static int g_policy = 1, g_chunks = 0;
// policy 2 needs the route cache: g_nu_fn(key) = the number of the next call (after the one being processed) that needs the expert, or a huge number. The entry
// replaced is then the one needed farthest in the future (Belady / MIN); an expert that would be needed later than everything already held is not admitted by worth().
static int (*g_nu_fn)(int key) = nullptr;
struct SlotCache {
    std::list<int> lru;                                   // keys, most recent first
    struct Ent { std::list<int>::iterator it; unsigned char *p; float heat; int nu; };
    std::unordered_map<int, Ent> map;
    std::set<std::pair<int, int>> by_next;                // (next use, key), only kept in policy 2
    std::vector<unsigned char *> spare;                   // slots never used or released
    bool belady() const { return g_policy == 2 && g_nu_fn; }
    unsigned char *find(int key)
    {
        auto f = map.find(key);
        if (f == map.end()) return nullptr;
        lru.splice(lru.begin(), lru, f->second.it);
        return f->second.p;
    }
    void touch(int key, float w)
    {
        auto f = map.find(key); if (f == map.end()) return;
        if (belady()) { Ent &e = f->second; by_next.erase({ e.nu, key }); e.nu = g_nu_fn(key); by_next.insert({ e.nu, key }); }
        else f->second.heat = f->second.heat * 0.8f + w;
    }
    // true if the expert deserves a slot: a free slot exists, or (policy 2) it is needed sooner than the farthest one held
    bool worth(int key) const
    {
        if (!belady() || !spare.empty() || by_next.empty()) return true;
        return g_nu_fn(key) < by_next.rbegin()->first;
    }
    // a slot for `key`; if an entry had to be replaced, *vkey and *vheat name it (its bytes are still in the slot), else *vkey = -1
    unsigned char *acquire(int key, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        unsigned char *p; *vkey = -1;
        if (!spare.empty()) { p = spare.back(); spare.pop_back(); }
        else if (belady()) {
            auto it = by_next.end(); int victim = -1;
            while (it != by_next.begin()) { --it; if (!prot.count(it->second)) { victim = it->second; break; } }
            if (victim < 0) { fprintf(stderr, "expert cache: every slot is in use by one call (too few slots)\n"); exit(1); }
            Ent &v = map[victim]; p = v.p; *vkey = victim; *vheat = v.heat;
            by_next.erase({ v.nu, victim }); lru.erase(v.it); map.erase(victim);
        } else {
            auto it = lru.end(), best = lru.end(); float best_heat = 0; int seen = 0;
            while (it != lru.begin() && seen < (g_policy ? 48 : 1)) {
                --it;
                if (prot.count(*it)) continue;
                const float h = map[*it].heat;
                if (best == lru.end() || h < best_heat) { best = it; best_heat = h; }
                seen++;
            }
            if (best == lru.end()) { fprintf(stderr, "expert cache: every slot is in use by one call (too few slots)\n"); exit(1); }
            const int victim = *best;
            p = map[victim].p; *vkey = victim; *vheat = map[victim].heat;
            if (!by_next.empty()) by_next.erase({ map[victim].nu, victim });
            map.erase(victim); lru.erase(best);
        }
        lru.push_front(key);
        int nu = 0; if (belady()) { nu = g_nu_fn(key); by_next.insert({ nu, key }); }
        map[key] = Ent{ lru.begin(), p, heat0, nu };
        return p;
    }
    // read-ahead (policy 2): a slot for `key`, needed at call cand_nu, only if one is free or the held entry needed farthest ahead (not in prot) is needed later; else nullptr
    unsigned char *try_acquire(int key, int cand_nu, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        *vkey = -1;
        if (spare.empty()) {
            auto it = by_next.end(); int victim = -1;
            while (it != by_next.begin()) { --it; if (!prot.count(it->second)) { if (it->first > cand_nu) victim = it->second; break; } }
            if (victim < 0) return nullptr;
        }
        return acquire(key, prot, heat0, vkey, vheat);
    }
    // takes `key` out of the cache; its slot is returned to the caller, who releases it when its bytes are no longer needed
    unsigned char *remove(int key)
    {
        auto f = map.find(key);
        if (f == map.end()) return nullptr;
        unsigned char *p = f->second.p; by_next.erase({ f->second.nu, key }); lru.erase(f->second.it); map.erase(f);
        return p;
    }
    void release(unsigned char *p) { spare.push_back(p); }
};
