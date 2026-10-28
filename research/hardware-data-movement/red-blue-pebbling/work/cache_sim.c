/*
 * Fully-associative word-granularity ideal cache simulator (LRU and Belady/OPT)
 * for n x n matrix multiplication C[i,j] += A[i,k] * B[k,j] (E6).
 *
 * Counts:
 *   - loads  (cache misses from slow memory)
 *   - stores (dirty writebacks of C to slow memory upon eviction + final cache flush)
 *   - total_io = loads + stores
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint64_t loads;
    uint64_t stores;
    uint64_t total_io;
} SimResult;

/* O(1) per-access fully-associative LRU simulator */
typedef struct {
    int S;
    int used;
    int head;
    int tail;
    int *prev;
    int *next;
    int *slot_addr;
    uint8_t *dirty;
    int *addr_to_slot;
    uint64_t loads;
    uint64_t stores;
} LRUCache;

static void lru_init(LRUCache *c, int S, int num_addrs) {
    c->S = S;
    c->used = 0;
    c->head = -1;
    c->tail = -1;
    c->prev = (int *)malloc(S * sizeof(int));
    c->next = (int *)malloc(S * sizeof(int));
    c->slot_addr = (int *)malloc(S * sizeof(int));
    c->dirty = (uint8_t *)calloc(S, 1);
    c->addr_to_slot = (int *)malloc(num_addrs * sizeof(int));
    memset(c->addr_to_slot, -1, num_addrs * sizeof(int));
    c->loads = 0;
    c->stores = 0;
}

static void lru_free(LRUCache *c) {
    free(c->prev);
    free(c->next);
    free(c->slot_addr);
    free(c->dirty);
    free(c->addr_to_slot);
}

static inline void lru_move_to_head(LRUCache *c, int s) {
    if (c->head == s) return;
    int p = c->prev[s];
    int nx = c->next[s];
    if (p != -1) c->next[p] = nx;
    if (nx != -1) c->prev[nx] = p;
    if (c->tail == s) c->tail = p;
    c->prev[s] = -1;
    c->next[s] = c->head;
    if (c->head != -1) c->prev[c->head] = s;
    c->head = s;
    if (c->tail == -1) c->tail = s;
}

static inline void lru_access(LRUCache *c, int addr, int is_write) {
    int s = c->addr_to_slot[addr];
    if (s != -1) {
        if (is_write) c->dirty[s] = 1;
        lru_move_to_head(c, s);
        return;
    }
    c->loads++;
    if (c->used < c->S) {
        s = c->used++;
        c->prev[s] = -1;
        c->next[s] = -1;
    } else {
        s = c->tail;
        if (c->dirty[s]) c->stores++;
        c->addr_to_slot[c->slot_addr[s]] = -1;
    }
    c->slot_addr[s] = addr;
    c->dirty[s] = (uint8_t)is_write;
    c->addr_to_slot[addr] = s;
    lru_move_to_head(c, s);
}

static SimResult lru_finish(LRUCache *c) {
    for (int s = 0; s < c->used; s++) {
        if (c->dirty[s]) c->stores++;
    }
    SimResult r = { c->loads, c->stores, c->loads + c->stores };
    return r;
}

/* Stream loop nest through LRU (and optionally record trace for OPT) */
/* order_mode: 0 = naive ijk, 1 = blocked (ii,jj,kk) + inner (i,j,k),
 *             2 = blocked (ii,jj,kk) + inner (i,k,j), 3 = blocked (ii,jj,kk) + inner (k,i,j) */
static SimResult run_lru_and_record(int n, int S, int b, int order_mode, uint32_t *trace_out) {
    int n2 = n * n;
    LRUCache lru;
    lru_init(&lru, S, 3 * n2);
    uint64_t idx = 0;

    if (order_mode == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    int a = i * n + k;
                    int bm = n2 + k * n + j;
                    int cm = 2 * n2 + i * n + j;
                    lru_access(&lru, a, 0);
                    lru_access(&lru, bm, 0);
                    lru_access(&lru, cm, 1);
                    if (trace_out) {
                        trace_out[idx++] = (uint32_t)(a << 1);
                        trace_out[idx++] = (uint32_t)(bm << 1);
                        trace_out[idx++] = (uint32_t)((cm << 1) | 1u);
                    }
                }
            }
        }
    } else {
        for (int ii = 0; ii < n; ii += b) {
            int i_end = (ii + b < n) ? (ii + b) : n;
            for (int jj = 0; jj < n; jj += b) {
                int j_end = (jj + b < n) ? (jj + b) : n;
                for (int kk = 0; kk < n; kk += b) {
                    int k_end = (kk + b < n) ? (kk + b) : n;
                    if (order_mode == 1) {
                        for (int i = ii; i < i_end; i++)
                            for (int j = jj; j < j_end; j++)
                                for (int k = kk; k < k_end; k++) {
                                    int a = i * n + k, bm = n2 + k * n + j, cm = 2 * n2 + i * n + j;
                                    lru_access(&lru, a, 0); lru_access(&lru, bm, 0); lru_access(&lru, cm, 1);
                                    if (trace_out) {
                                        trace_out[idx++] = (uint32_t)(a << 1);
                                        trace_out[idx++] = (uint32_t)(bm << 1);
                                        trace_out[idx++] = (uint32_t)((cm << 1) | 1u);
                                    }
                                }
                    } else if (order_mode == 2) {
                        for (int i = ii; i < i_end; i++)
                            for (int k = kk; k < k_end; k++)
                                for (int j = jj; j < j_end; j++) {
                                    int a = i * n + k, bm = n2 + k * n + j, cm = 2 * n2 + i * n + j;
                                    lru_access(&lru, a, 0); lru_access(&lru, bm, 0); lru_access(&lru, cm, 1);
                                    if (trace_out) {
                                        trace_out[idx++] = (uint32_t)(a << 1);
                                        trace_out[idx++] = (uint32_t)(bm << 1);
                                        trace_out[idx++] = (uint32_t)((cm << 1) | 1u);
                                    }
                                }
                    } else {
                        for (int k = kk; k < k_end; k++)
                            for (int i = ii; i < i_end; i++)
                                for (int j = jj; j < j_end; j++) {
                                    int a = i * n + k, bm = n2 + k * n + j, cm = 2 * n2 + i * n + j;
                                    lru_access(&lru, a, 0); lru_access(&lru, bm, 0); lru_access(&lru, cm, 1);
                                    if (trace_out) {
                                        trace_out[idx++] = (uint32_t)(a << 1);
                                        trace_out[idx++] = (uint32_t)(bm << 1);
                                        trace_out[idx++] = (uint32_t)((cm << 1) | 1u);
                                    }
                                }
                    }
                }
            }
        }
    }
    SimResult res = lru_finish(&lru);
    lru_free(&lru);
    return res;
}

/* O(T log S) Belady/OPT simulator using a binary max-heap over next_use */
static SimResult run_opt_from_trace(int n, int S, const uint32_t *trace, uint64_t T) {
    int num_addrs = 3 * n * n;
    uint32_t *next_use = (uint32_t *)malloc(T * sizeof(uint32_t));
    uint32_t *last_seen = (uint32_t *)malloc(num_addrs * sizeof(uint32_t));
    for (int i = 0; i < num_addrs; i++) last_seen[i] = 0xffffffffu;
    for (uint64_t t = T; t > 0; t--) {
        uint64_t i = t - 1;
        int addr = (int)(trace[i] >> 1);
        next_use[i] = last_seen[addr];
        last_seen[addr] = (uint32_t)i;
    }
    free(last_seen);

    int *addr_to_slot = (int *)malloc(num_addrs * sizeof(int));
    memset(addr_to_slot, -1, num_addrs * sizeof(int));
    int *slot_addr = (int *)malloc(S * sizeof(int));
    uint8_t *dirty = (uint8_t *)calloc(S, 1);
    uint32_t *slot_next = (uint32_t *)malloc(S * sizeof(uint32_t));

    /* Max-heap of slot indices ordered by slot_next[s] */
    int *heap = (int *)malloc(S * sizeof(int));
    int *slot_heap_pos = (int *)malloc(S * sizeof(int));
    int used = 0;
    uint64_t loads = 0, stores = 0;

    #define HEAP_SWAP(i, j) do { \
        int tmp = heap[i]; heap[i] = heap[j]; heap[j] = tmp; \
        slot_heap_pos[heap[i]] = (i); slot_heap_pos[heap[j]] = (j); \
    } while (0)

    #define SIFT_UP(pos) do { \
        int p = (pos); \
        while (p > 0) { \
            int par = (p - 1) >> 1; \
            if (slot_next[heap[p]] > slot_next[heap[par]]) { HEAP_SWAP(p, par); p = par; } \
            else break; \
        } \
    } while (0)

    #define SIFT_DOWN(pos, sz) do { \
        int p = (pos); \
        while (1) { \
            int l = 2 * p + 1, r = 2 * p + 2, best = p; \
            if (l < (sz) && slot_next[heap[l]] > slot_next[heap[best]]) best = l; \
            if (r < (sz) && slot_next[heap[r]] > slot_next[heap[best]]) best = r; \
            if (best != p) { HEAP_SWAP(p, best); p = best; } else break; \
        } \
    } while (0)

    for (uint64_t t = 0; t < T; t++) {
        int addr = (int)(trace[t] >> 1);
        int is_write = (int)(trace[t] & 1u);
        uint32_t nu = next_use[t];
        int s = addr_to_slot[addr];
        if (s != -1) {
            if (is_write) dirty[s] = 1;
            slot_next[s] = nu;
            int hp = slot_heap_pos[s];
            SIFT_UP(hp);
            SIFT_DOWN(hp, used);
        } else {
            loads++;
            if (used < S) {
                s = used++;
                heap[s] = s;
                slot_heap_pos[s] = s;
            } else {
                s = heap[0];
                if (dirty[s]) stores++;
                addr_to_slot[slot_addr[s]] = -1;
            }
            slot_addr[s] = addr;
            dirty[s] = (uint8_t)is_write;
            slot_next[s] = nu;
            addr_to_slot[addr] = s;
            int hp = slot_heap_pos[s];
            SIFT_UP(hp);
            SIFT_DOWN(hp, used);
        }
    }
    for (int s = 0; s < used; s++) {
        if (dirty[s]) stores++;
    }

    free(next_use); free(addr_to_slot); free(slot_addr);
    free(dirty); free(slot_next); free(heap); free(slot_heap_pos);
    SimResult res = { loads, stores, loads + stores };
    return res;
}

int main(int argc, char **argv) {
    printf("=== 1. HAND-COUNTABLE 2x2 CASE (S=4) AND OFF-BY-ONE CONTROLS (S=3, S=5) ===\n");
    for (int S = 2; S <= 6; S++) {
        uint32_t tr[24];
        SimResult lru = run_lru_and_record(2, S, 2, 0, tr);
        SimResult opt = run_opt_from_trace(2, S, tr, 24);
        printf("  n=2, S=%d: LRU loads=%llu stores=%llu total=%llu | OPT loads=%llu stores=%llu total=%llu\n",
               S, (unsigned long long)lru.loads, (unsigned long long)lru.stores, (unsigned long long)lru.total_io,
               (unsigned long long)opt.loads, (unsigned long long)opt.stores, (unsigned long long)opt.total_io);
    }

    printf("\n=== 2. MAIN SWEEP: n in {64,128,256,512}, S in {48,192,768,3072}, b=floor(sqrt(S/3)) ===\n");
    printf("Note: OPT run for n <= 256 (up to 50.3M accesses, 402 MB trace+next_use); skipped at n=512 (402.7M accesses).\n");
    int ns[4] = {64, 128, 256, 512};
    int ss[4] = {48, 192, 768, 3072};
    for (int ni = 0; ni < 4; ni++) {
        int n = ns[ni];
        uint64_t T = 3ULL * n * n * n;
        for (int si = 0; si < 4; si++) {
            int S = ss[si];
            int b = (int)floor(sqrt((double)S / 3.0));
            uint32_t *tr = (n <= 256) ? (uint32_t *)malloc(T * sizeof(uint32_t)) : NULL;
            SimResult naive_lru = run_lru_and_record(n, S, b, 0, tr);
            SimResult naive_opt = {0, 0, 0};
            if (tr) naive_opt = run_opt_from_trace(n, S, tr, T);
            SimResult blk_lru = run_lru_and_record(n, S, b, 1, tr);
            SimResult blk_opt = {0, 0, 0};
            if (tr) blk_opt = run_opt_from_trace(n, S, tr, T);
            if (tr) free(tr);
            double scale = ((double)n * n * n) / sqrt((double)S);
            printf("n=%3d S=%4d b=%2d | NaiveLRU=%10llu NaiveOPT=%10llu | BlkLRU=%10llu (c=%.3f) BlkOPT=%10llu (c=%.3f) | Naive/Blk(LRU)=%6.2fx\n",
                   n, S, b,
                   (unsigned long long)naive_lru.total_io, (unsigned long long)naive_opt.total_io,
                   (unsigned long long)blk_lru.total_io, (double)blk_lru.total_io / scale,
                   (unsigned long long)blk_opt.total_io, (double)blk_opt.total_io / scale,
                   (double)naive_lru.total_io / (double)blk_lru.total_io);
            fflush(stdout);
        }
    }

    printf("\n=== 3. TILE SIZE SENSITIVITY (b TOO SMALL AND b TOO LARGE) AT n=128, S=192 (sqrt(S/3)=8) ===\n");
    {
        int n = 128, S = 192;
        uint64_t T = 3ULL * n * n * n;
        int bs[7] = {2, 4, 6, 8, 12, 16, 32};
        uint32_t *tr = (uint32_t *)malloc(T * sizeof(uint32_t));
        for (int bi = 0; bi < 7; bi++) {
            int b = bs[bi];
            SimResult lru = run_lru_and_record(n, S, b, 1, tr);
            SimResult opt = run_opt_from_trace(n, S, tr, T);
            printf("  n=%d S=%d b=%2d (3b^2=%4d): BlkLRU=%8llu (loads=%8llu, stores=%6llu) | BlkOPT=%8llu (loads=%8llu, stores=%6llu) | LRU/OPT=%.3f\n",
                   n, S, b, 3*b*b,
                   (unsigned long long)lru.total_io, (unsigned long long)lru.loads, (unsigned long long)lru.stores,
                   (unsigned long long)opt.total_io, (unsigned long long)opt.loads, (unsigned long long)opt.stores,
                   (double)lru.total_io / (double)opt.total_io);
        }
        free(tr);
    }

    printf("\n=== 4. INTRA-TILE LOOP ORDER EXPERIMENT (EXPLAINING LRU VS OPT AT 3b^2=S) ===\n");
    {
        int n = 128, S = 192, b = 8;
        uint64_t T = 3ULL * n * n * n;
        uint32_t *tr = (uint32_t *)malloc(T * sizeof(uint32_t));
        const char *names[3] = {"inner_ijk", "inner_ikj", "inner_kij"};
        for (int mode = 1; mode <= 3; mode++) {
            SimResult lru = run_lru_and_record(n, S, b, mode, tr);
            SimResult opt = run_opt_from_trace(n, S, tr, T);
            printf("  n=%d S=%d b=%d %-10s: LRU total=%8llu (loads=%8llu, stores=%6llu) | OPT total=%8llu (loads=%8llu, stores=%6llu) | LRU/OPT=%.3f\n",
                   n, S, b, names[mode - 1],
                   (unsigned long long)lru.total_io, (unsigned long long)lru.loads, (unsigned long long)lru.stores,
                   (unsigned long long)opt.total_io, (unsigned long long)opt.loads, (unsigned long long)opt.stores,
                   (double)lru.total_io / (double)opt.total_io);
        }
        free(tr);
    }
    return 0;
}
