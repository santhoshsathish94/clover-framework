"""
E8 (SpecMD study): the baseline repository's eviction scheme, replayed on the recorded OLMoE traces.

SOURCE READ (dvmazur/mixtral-offloading @ ce54518, src/expert_cache.py and src/build_model.py):
  * eviction_group = layer index; every expert of a layer belongs to that layer's group (build_model.py line 233).
  * a group holds `main_infos` (on-GPU experts, an OrderedDict from least to most recently used) and `offloaded_infos`.
  * the on-GPU experts of a group are fixed at start: experts with index >= offload_per_layer are placed on the GPU in
    ascending order (build_model.py: do_offload = expert_idx < offload_per_layer), so each layer keeps (num_experts - offload_per_layer).
  * load_experts(*uids): all uids must belong to one group; first `mark_used` for EVERY uid in call order (a uid in main_infos
    is a hit and moves to the MRU end; otherwise a miss), then each offloaded uid is swapped in, evicting the group's
    LRU (first entry of main_infos); the loaded expert goes to the MRU end.
  * _swap asserts the two experts are in the same group, so slots never move between layers.
This file reimplements exactly that, nothing else (no prefetch, no timing).

usage: python e8_baseline_group_lru.py /path/to/traces
"""
import sys
from collections import OrderedDict
import numpy as np

NE = 64     # OLMoE experts per layer


class Group:
    def __init__(self, m, init="high"):
        order = range(NE - m, NE) if init == "high" else range(m)
        self.main = OrderedDict((e, True) for e in order)
        self.m = m
        self.hits = self.misses = 0

    def call(self, uids):
        for u in uids:                              # mark_used for every uid first
            if u in self.main:
                self.main.move_to_end(u)
                self.hits += 1
            else:
                self.misses += 1
        todo = [u for u in uids if u not in self.main]
        for u in todo:                              # then swap each missing one in, evicting the LRU
            if u in self.main:                      # (cannot happen; kept for safety)
                continue
            victim = next(iter(self.main))
            del self.main[victim]
            self.main[u] = True


def replay(idx, style, m, init="high", order="rank"):
    P, T, L, K = idx.shape
    groups = [Group(m, init) for _ in range(L)]
    if style == "decode":
        calls = ((l, idx[p, t, l]) for p in range(P) for t in range(T) for l in range(L))
    else:
        calls = ((l, idx[p, t, l]) for p in range(P) for l in range(L) for t in range(T))
    for l, row in calls:
        uids = [int(x) for x in row]
        if order == "sorted":
            uids.sort()
        groups[l].call(uids)
    h = sum(g.hits for g in groups)
    n = h + sum(g.misses for g in groups)
    return 100.0 * h / n


if __name__ == "__main__":
    d = sys.argv[1]
    print("slots per layer m (total C = 16*m); hit rate % under the baseline's group-LRU semantics")
    for style in ("decode", "prompt"):
        idx = np.load(f"{d}/{style}_traces.npz")["topk_indices"]
        print(f"== {style}")
        for m in (1, 2, 3, 4, 8, 16, 32):
            print(f"  m={m:2d} (C={16*m:3d}): rank-order/high-init={replay(idx, style, m):6.2f}   rank/low-init={replay(idx, style, m, 'low'):6.2f}   sorted-order/high-init={replay(idx, style, m, 'high', 'sorted'):6.2f}", flush=True)
