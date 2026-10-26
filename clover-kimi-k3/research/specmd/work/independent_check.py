"""
Independent check of the SpecMD study's global-cache hit rates (written by me, separately from sim_cache.py).
Reads the recorded OLMoE traces and replays the request streams with small, deliberately simple caches:
  LRU (OrderedDict), FIFO (OrderedDict), LFU (frequency over the whole history, ties by recency), and my own reading of
  Least-Stale: evict, in this order, (1) entries last used in an earlier pass and in a layer < current layer ("stale-left",
  smallest layer first), (2) other stale entries (farthest layer first), (3) entries used in the current pass (oldest first).
Streams (same definition as the study): decode = for prompt, token, layer, expert in top-k rank order; prompt = for prompt,
layer, token, expert. A "pass" is one token (decode) or one layer sweep of one prompt (prompt-style).
Prints hit rates next to the numbers the study reported in E6 so they can be compared by eye.
usage: python independent_check.py /path/to/traces
"""
import sys
from collections import OrderedDict
import numpy as np

REPORTED = {  # E6, global pool, no prefetch  (policy, C) -> hit %
    ("decode", "LRU", 10): 0.00, ("decode", "LRU", 51): 0.00, ("decode", "LRU", 256): 51.04,
    ("decode", "FIFO", 256): 46.15, ("decode", "LFU", 10): 5.01, ("decode", "LFU", 51): 17.64, ("decode", "LFU", 256): 48.94,
    ("decode", "LS", 256): 53.86, ("decode", "LS", 51): 0.00,
    ("prompt", "LRU", 10): 36.81, ("prompt", "LRU", 51): 92.88, ("prompt", "LRU", 256): 96.98,
    ("prompt", "FIFO", 10): 32.69, ("prompt", "FIFO", 51): 90.46, ("prompt", "FIFO", 256): 96.98,
    ("prompt", "LFU", 51): 18.14, ("prompt", "LFU", 256): 49.96,
}


def stream(idx, style):
    P, T, L, K = idx.shape
    if style == "decode":
        for p in range(P):
            for t in range(T):
                pass_id = p * T + t
                for l in range(L):
                    for e in idx[p, t, l]:
                        yield pass_id, l, int(e)
    else:
        for p in range(P):
            pass_id = p                       # one forward pass per prompt
            for l in range(L):
                for t in range(T):
                    for e in idx[p, t, l]:
                        yield pass_id, l, int(e)


def run(idx, style, policy, C):
    hits = n = 0
    if policy in ("LRU", "FIFO"):
        od = OrderedDict()
        for pass_id, l, e in stream(idx, style):
            k = (l, e)
            n += 1
            if k in od:
                hits += 1
                if policy == "LRU":
                    od.move_to_end(k)
            else:
                if len(od) >= C:
                    od.popitem(last=False)
                od[k] = True
        return 100.0 * hits / n
    freq = {}
    cache = {}                       # key -> [last_use_step, pass_id, layer]
    step = 0
    for pass_id, l, e in stream(idx, style):
        k = (l, e)
        step += 1
        n += 1
        freq[k] = freq.get(k, 0) + 1
        if k in cache:
            hits += 1
            cache[k][0] = step
            cache[k][1] = pass_id
            continue
        if len(cache) >= C:
            if policy == "LFU":
                v = min(cache, key=lambda x: (freq[x], cache[x][0]))
            elif policy == "LS2":      # the study's stated ordering rules, re-implemented by me
                def key2(x):
                    last, pid, lay, ins = cache[x]
                    left = lay < l
                    return (0 if pid < pass_id else 1, 0 if left else 1, pid, lay if left else -lay, ins)
                v = min(cache, key=key2)
            else:  # LS (my reading)
                def key(x):
                    last, pid, lay = cache[x][:3]
                    stale = pid < pass_id
                    if stale and lay < l:
                        return (0, lay, last)
                    if stale:
                        return (1, -lay, last)
                    return (2, last, 0)
                v = min(cache, key=key)
            del cache[v]
        cache[k] = [step, pass_id, l, step]
    return 100.0 * hits / n


if __name__ == "__main__":
    d = sys.argv[1]
    for style in ("decode", "prompt"):
        idx = np.load(f"{d}/{style}_traces.npz")["topk_indices"]
        print(f"== {style}: idx shape {idx.shape}")
        for policy in ("LRU", "FIFO", "LFU", "LS", "LS2"):
            for C in (10, 51, 256):
                got = run(idx, style, policy, C)
                rep = REPORTED.get((style, policy, C))
                print(f"  {policy:5} C={C:3d}  independent={got:6.2f}%   study E6={'%.2f%%' % rep if rep is not None else 'n/a'}", flush=True)
