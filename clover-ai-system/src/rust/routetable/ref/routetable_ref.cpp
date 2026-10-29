// Reference implementation of ../include/routetable.h: the engine's route-table text (extracted by extract.py from backend-gpu.cu, unchanged) behind thin extern "C" wrappers.
#include <vector>
#include <unordered_map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "routetable.h"

// The ONLY adaptation: in the engine ekey (decode.cuh:201) is `layer * EXPERTS + e` with a compile-time EXPERTS; here the factor is the `experts` argument of rt_load.
static int g_experts_rt;
static inline int ekey(int layer, int e) { return layer * g_experts_rt + e; }

// backend-gpu.cu lines 52-66 (CallSet, g_rc, g_rc_pos/g_rc_planned/g_pend_cap, g_rc_cur, g_uses, g_use_ptr, rc_nu with its comment) and 369-387 (rc_load), byte for byte.
#include "routetable_extracted.inc"

struct rt_table { int dummy; };
static rt_table g_table;

static void clear_all()
{
    g_rc.clear(); g_uses.clear(); g_use_ptr.clear();
    g_rc_cur = -1;
}

extern "C" {

rt_table *rt_load(const char *path, int experts)
{
    g_experts_rt = experts;
    clear_all();
    rc_load(path);
    return &g_table;
}

void rt_free(rt_table *t)
{
    (void)t;
    clear_all();
}

size_t rt_calls(const rt_table *t) { (void)t; return g_rc.size(); }
int rt_layer(const rt_table *t, size_t i) { (void)t; return g_rc[i].layer; }

const int *rt_experts(const rt_table *t, size_t i, size_t *n)
{
    (void)t;
    const std::vector<int> &v = g_rc[i].experts;
    *n = v.size();
    return v.data();
}

const int *rt_counts(const rt_table *t, size_t i, size_t *n)
{
    (void)t;
    const std::vector<int> &v = g_rc[i].counts;
    *n = v.size();
    return v.data();
}

void rt_set_current(rt_table *t, long cur) { (void)t; g_rc_cur = cur; }
int rt_next_use(rt_table *t, int key) { (void)t; return rc_nu(key); }

}  // extern "C"
