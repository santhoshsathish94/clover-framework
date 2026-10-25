// Reference implementation of ../include/slotcache.h: the engine's SlotCache text (extracted by extract.py, unchanged) behind thin extern "C" wrappers.
#include <cstdio>
#include <cstdlib>
#include <list>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "slotcache.h"

// decode.cuh lines 200-206 (g_policy, g_chunks, g_nu_fn and their comments) and 208-285 (struct SlotCache), byte for byte.
#include "slotcache_extracted.inc"

static SlotCache *S(sc_cache *c) { return reinterpret_cast<SlotCache *>(c); }
static const SlotCache *S(const sc_cache *c) { return reinterpret_cast<const SlotCache *>(c); }

extern "C" {

void sc_set_policy(int policy) { g_policy = policy; }
void sc_set_next_use(sc_next_use_fn fn) { g_nu_fn = fn; }

sc_cache *sc_new(void) { return reinterpret_cast<sc_cache *>(new SlotCache()); }
void sc_free(sc_cache *c) { delete S(c); }

void sc_add_spare(sc_cache *c, unsigned char *slot) { S(c)->spare.push_back(slot); }
void sc_release(sc_cache *c, unsigned char *slot) { S(c)->release(slot); }
int sc_contains(const sc_cache *c, int key) { return (int)S(c)->map.count(key); }
size_t sc_size(const sc_cache *c) { return S(c)->map.size(); }

unsigned char *sc_find(sc_cache *c, int key) { return S(c)->find(key); }
void sc_touch(sc_cache *c, int key, float w) { S(c)->touch(key, w); }
int sc_worth(const sc_cache *c, int key) { return S(c)->worth(key) ? 1 : 0; }

unsigned char *sc_acquire(sc_cache *c, int key, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat)
{
    std::unordered_set<int> p;
    for (size_t i = 0; i < nprot; i++) p.insert(prot[i]);
    return S(c)->acquire(key, p, heat0, vkey, vheat);
}

unsigned char *sc_try_acquire(sc_cache *c, int key, int cand_nu, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat)
{
    std::unordered_set<int> p;
    for (size_t i = 0; i < nprot; i++) p.insert(prot[i]);
    return S(c)->try_acquire(key, cand_nu, p, heat0, vkey, vheat);
}

unsigned char *sc_remove(sc_cache *c, int key) { return S(c)->remove(key); }

size_t sc_snapshot_lru(const sc_cache *c, sc_entry *out, size_t cap)
{
    const SlotCache *s = S(c);
    size_t n = 0;
    for (int key : s->lru) {
        if (n < cap) {
            const SlotCache::Ent &e = s->map.find(key)->second;
            out[n].key = key; out[n].slot = e.p; out[n].heat = e.heat; out[n].nu = e.nu;
        }
        n++;
    }
    return n;
}

size_t sc_snapshot_spare(const sc_cache *c, unsigned char **out, size_t cap)
{
    const SlotCache *s = S(c);
    for (size_t i = 0; i < s->spare.size() && i < cap; i++) out[i] = s->spare[i];
    return s->spare.size();
}

size_t sc_snapshot_next(const sc_cache *c, int *nu_out, int *key_out, size_t cap)
{
    const SlotCache *s = S(c);
    size_t n = 0;
    for (const std::pair<int, int> &p : s->by_next) {
        if (n < cap) { nu_out[n] = p.first; key_out[n] = p.second; }
        n++;
    }
    return n;
}

}  // extern "C"
