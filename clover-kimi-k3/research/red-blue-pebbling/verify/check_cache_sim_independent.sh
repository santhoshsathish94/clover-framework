cd /root/dm-study/pebbling
/root/ncm-cpu/venv/bin/python - <<'EOF'
# Coordinator's independent cache simulator (own code): word-granularity, fully associative, LRU and Belady/OPT.
# Access order per inner step: A[i,k], B[k,j], C[i,j] (C is read and written => dirty). Total I/O = misses + dirty write-backs (final flush included).
import heapq, sys
from collections import OrderedDict

def trace_naive(n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                yield (i, j, k)

def trace_blocked(n, b, order):
    for ii in range(0, n, b):
        for jj in range(0, n, b):
            for kk in range(0, n, b):
                rng = {"i": range(ii, min(ii + b, n)), "j": range(jj, min(jj + b, n)), "k": range(kk, min(kk + b, n))}
                o = order
                for x in rng[o[0]]:
                    for y in rng[o[1]]:
                        for z in rng[o[2]]:
                            d = {o[0]: x, o[1]: y, o[2]: z}
                            yield (d["i"], d["j"], d["k"])

def words(n, steps):
    n2 = n * n
    for i, j, k in steps:
        yield i * n + k
        yield n2 + k * n + j
        yield 2 * n2 + i * n + j

def lru(seq, S, n):
    c2 = 2 * n * n
    cache = OrderedDict(); dirty = set(); loads = stores = 0
    for w in seq:
        if w in cache:
            cache.move_to_end(w)
        else:
            loads += 1
            if len(cache) >= S:
                v, _ = cache.popitem(last=False)
                if v in dirty:
                    stores += 1; dirty.discard(v)
            cache[w] = 1
        if w >= c2:
            dirty.add(w)
    stores += len(dirty)
    return loads, stores

def opt(seq, S, n):
    c2 = 2 * n * n
    nxt = [0] * len(seq); last = {}
    INF = 1 << 60
    for t in range(len(seq) - 1, -1, -1):
        nxt[t] = last.get(seq[t], INF); last[seq[t]] = t
    cache = {}; heap = []; dirty = set(); loads = stores = 0
    for t, w in enumerate(seq):
        if w not in cache:
            loads += 1
            if len(cache) >= S:
                while True:
                    nu, v = heapq.heappop(heap)
                    if v in cache and cache[v] == -nu:
                        break
                del cache[v]
                if v in dirty:
                    stores += 1; dirty.discard(v)
        cache[w] = nxt[t]
        heapq.heappush(heap, (-nxt[t], w))
        if w >= c2:
            dirty.add(w)
    stores += len(dirty)
    return loads, stores

print("== 2x2 naive, hand-countable (study: S=4 LRU 20+4=24, OPT 15+4=19; S=3 OPT 17+4; S=5 LRU 18+4)")
seq = list(words(2, trace_naive(2)))
print("accesses", len(seq))
for S in (3, 4, 5):
    print("S=%d LRU %s  OPT %s" % (S, lru(seq, S, 2), opt(seq, S, 2)))
sys.stdout.flush()

print("== n=128, S=192, b=8 (study: ijk LRU 1,040,896 = 782,592+258,304, OPT 542,720; ikj LRU 948,736, OPT 540,928; kij LRU 557,056, OPT 528,128)")
for order in ("ijk", "ikj", "kij"):
    seq = list(words(128, trace_blocked(128, 8, order)))
    l = lru(seq, 192, 128); o = opt(seq, 192, 128)
    print("%s LRU loads=%d stores=%d total=%d | OPT loads=%d stores=%d total=%d | LRU/OPT=%.3f" % (order, l[0], l[1], sum(l), o[0], o[1], sum(o), sum(l) / sum(o)))
    sys.stdout.flush()
EOF
echo DONE-CACHE
