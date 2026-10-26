import numpy as np


class SinglePoolCache:
    """Single cache pool of capacity C slots supporting SpecMD policies."""

    def __init__(self, capacity, policy="LRU", num_layers=16, static_top_set=None):
        self.capacity = int(capacity)
        self.policy = policy
        self.num_layers = int(num_layers)
        self.static_top_set = set(static_top_set) if static_top_set is not None else set()

        # key -> dict(last_step, ins_step, freq, pass_id, layer, expert, score, prefetched_unused)
        self.entries = {}
        self.hist_freq = {}
        self.hist_score_sum = {}
        self.hist_score_cnt = {}
        self.seen_ever = set()

        self.step = 0
        self.cur_pass_id = -1
        self.present_in_pass = set()
        self.inserted_in_pass = set()
        self.evicted_from_pass = set()
        self.evicted_from_intra = set()

        self.hits = 0
        self.misses = 0
        self.cold_misses = 0
        self.collision_misses_pass = 0
        self.collision_misses_intra = 0

        self.prefetch_requests = 0
        self.prefetch_loads = 0
        self.prefetch_already_cached = 0
        self.prefetch_useful = 0
        self.prefetch_wasted = 0

        self.layer_stats = {
            l: {"hits": 0, "misses": 0, "coll_pass": 0, "coll_intra": 0}
            for l in range(num_layers)
        }

    def begin_pass(self, pass_id):
        if pass_id != self.cur_pass_id:
            self.cur_pass_id = pass_id
            self.present_in_pass = set(self.entries.keys())
            self.inserted_in_pass = set()
            self.evicted_from_pass = set()
            self.evicted_from_intra = set()

    def _select_victim(self, cur_layer, cur_expert, next_use_map=None):
        if not self.entries:
            return None
        p = self.policy
        if p == "LRU":
            return min(self.entries.keys(), key=lambda k: self.entries[k]["last_step"])
        if p == "FIFO":
            return min(self.entries.keys(), key=lambda k: self.entries[k]["ins_step"])
        if p == "LFU":
            return min(
                self.entries.keys(),
                key=lambda k: (self.hist_freq.get(k, 0), self.entries[k]["last_step"]),
            )
        if p == "FLD":
            # Farthest Layer Distance (Appendix B: distance between cached expert's layer and current layer)
            return max(
                self.entries.keys(),
                key=lambda k: (abs(self.entries[k]["layer"] - cur_layer), -self.entries[k]["last_step"]),
            )
        if p == "SB":
            # Score-based (Appendix B: lowest historical gate score)
            return min(
                self.entries.keys(),
                key=lambda k: (
                    self.hist_score_sum.get(k, 0.0) / max(1, self.hist_score_cnt.get(k, 1)),
                    self.entries[k]["last_step"],
                ),
            )
        if p == "LS":
            # Least-Stale (Section 4.2, 3_methodology.tex L135-149):
            # 1. Stale queue (pass_id < cur_pass_id) evicted before current queue (pass_id == cur_pass_id)
            # 2. Within each queue, left layers (l_e < cur_layer) before current/right layers (l_e >= cur_layer).
            # 3. Older passes (smaller pass_id) evicted before newer passes; left layers ordered by layer FIFO (l_e),
            #    right layers (when no left layers exist, e.g. at cur_layer=0) ordered farthest layer first (-l_e).
            def ls_key(k):
                ent = self.entries[k]
                is_current = 0 if ent["pass_id"] < self.cur_pass_id else 1
                l_e = ent["layer"]
                is_left = l_e < cur_layer
                left_rank = 0 if is_left else 1
                layer_order = l_e if is_left else -l_e
                return (is_current, left_rank, ent["pass_id"], layer_order, ent["ins_step"])

            return min(self.entries.keys(), key=ls_key)
        if p == "Belady":
            # Evict item whose next use is farthest in the future
            return max(
                self.entries.keys(),
                key=lambda k: next_use_map.get(k, 10**12) if next_use_map is not None else 10**12,
            )
        raise ValueError(f"Unknown policy: {p}")

    def _evict_one(self, cur_layer, cur_expert, next_use_map=None):
        victim = self._select_victim(cur_layer, cur_expert, next_use_map)
        if victim is None:
            return
        ent = self.entries.pop(victim)
        if ent.get("prefetched_unused", False):
            self.prefetch_wasted += 1
        if victim in self.present_in_pass:
            self.evicted_from_pass.add(victim)
        if victim in self.inserted_in_pass:
            self.evicted_from_intra.add(victim)

    def access(self, key, pass_id, layer, expert, score=1.0, next_use_map=None):
        self.begin_pass(pass_id)
        self.step += 1

        self.hist_freq[key] = self.hist_freq.get(key, 0) + 1
        self.hist_score_sum[key] = self.hist_score_sum.get(key, 0.0) + float(score)
        self.hist_score_cnt[key] = self.hist_score_cnt.get(key, 0) + 1

        if self.policy == "StaticTopC":
            if key in self.static_top_set:
                self.hits += 1
                self.layer_stats[layer]["hits"] += 1
                return True
            else:
                self.misses += 1
                self.layer_stats[layer]["misses"] += 1
                if key not in self.seen_ever:
                    self.cold_misses += 1
                    self.seen_ever.add(key)
                return False

        # Deliberately broken negative control: inserts BEFORE checking hit
        if self.policy == "BrokenInsertFirst":
            if key not in self.entries and self.capacity > 0:
                if len(self.entries) >= self.capacity:
                    victim = min(self.entries.keys(), key=lambda k: self.entries[k]["last_step"])
                    self.entries.pop(victim)
                self.entries[key] = {
                    "last_step": self.step, "ins_step": self.step,
                    "pass_id": pass_id, "layer": layer, "expert": expert,
                    "prefetched_unused": False,
                }
            if key in self.entries:
                self.hits += 1
                return True

        if key in self.entries:
            self.hits += 1
            self.layer_stats[layer]["hits"] += 1
            ent = self.entries[key]
            ent["last_step"] = self.step
            ent["pass_id"] = pass_id
            if ent.get("prefetched_unused", False):
                self.prefetch_useful += 1
                ent["prefetched_unused"] = False
            self.present_in_pass.add(key)
            return True

        # Cache miss
        self.misses += 1
        self.layer_stats[layer]["misses"] += 1
        if key not in self.seen_ever:
            self.cold_misses += 1
            self.seen_ever.add(key)
        if key in self.evicted_from_pass:
            self.collision_misses_pass += 1
            self.layer_stats[layer]["coll_pass"] += 1
        if key in self.evicted_from_intra:
            self.collision_misses_intra += 1
            self.layer_stats[layer]["coll_intra"] += 1

        if self.capacity > 0:
            if len(self.entries) >= self.capacity:
                self._evict_one(layer, expert, next_use_map)
            self.entries[key] = {
                "last_step": self.step,
                "ins_step": self.step,
                "pass_id": pass_id,
                "layer": layer,
                "expert": expert,
                "prefetched_unused": False,
            }
            self.present_in_pass.add(key)
            self.inserted_in_pass.add(key)
        return False

    def access_group(self, keys, pass_id, layer, experts, scores):
        """Serve hits among a layer's k selected experts first, then load misses."""
        self.begin_pass(pass_id)
        miss_idx = []
        for idx, key in enumerate(keys):
            if key in self.entries or (self.policy == "StaticTopC" and key in self.static_top_set):
                self.access(key, pass_id, layer, experts[idx], scores[idx])
            else:
                miss_idx.append(idx)
        for idx in miss_idx:
            self.access(keys[idx], pass_id, layer, experts[idx], scores[idx])

    def prefetch(self, key, pass_id, cur_layer, target_layer, target_expert):
        self.begin_pass(pass_id)
        self.prefetch_requests += 1
        if self.capacity <= 0:
            return
        if key in self.entries:
            self.prefetch_already_cached += 1
            # Mark as current pass so Least-Stale protects it for upcoming target_layer
            self.entries[key]["pass_id"] = pass_id
            self.entries[key]["last_step"] = self.step
            self.present_in_pass.add(key)
            return

        self.step += 1
        if len(self.entries) >= self.capacity:
            self._evict_one(cur_layer, 64)
        self.entries[key] = {
            "last_step": self.step,
            "ins_step": self.step,
            "pass_id": pass_id,
            "layer": target_layer,
            "expert": target_expert,
            "prefetched_unused": True,
        }
        self.prefetch_loads += 1
        self.present_in_pass.add(key)
        self.inserted_in_pass.add(key)

    def summary(self):
        tot = self.hits + self.misses
        # Any prefetched item still unused at end of trace was never demanded
        still_unused = sum(1 for e in self.entries.values() if e.get("prefetched_unused", False))
        return {
            "policy": self.policy,
            "capacity": self.capacity,
            "total": tot,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / float(tot) if tot else 0.0,
            "miss_rate": self.misses / float(tot) if tot else 0.0,
            "cold_misses": self.cold_misses,
            "collision_misses_pass": self.collision_misses_pass,
            "collision_rate_pass": self.collision_misses_pass / float(tot) if tot else 0.0,
            "collision_misses_intra": self.collision_misses_intra,
            "collision_rate_intra": self.collision_misses_intra / float(tot) if tot else 0.0,
            "prefetch_requests": self.prefetch_requests,
            "prefetch_loads": self.prefetch_loads,
            "prefetch_already_cached": self.prefetch_already_cached,
            "prefetch_useful": self.prefetch_useful,
            "prefetch_wasted": self.prefetch_wasted + still_unused,
            "layer_stats": self.layer_stats,
        }
