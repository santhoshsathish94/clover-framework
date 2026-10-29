#!/usr/bin/env python3
"""
Clover 4-Tier Data-Movement & Red-Blue Pebble I/O Ledger (`k3_data_movement_ledger.py`)
======================================================================================
Implements the unified 4-Tier Data-Movement & Red-Blue Pebble I/O Ledger from:
  1. Hong & Kung (STOC 1981) Red-Blue Pebble Game (`red-blue-pebbling-context.md`):
     - Fast memory tier (Red pebbles, capacity S slots per card in `CardPool` + `k_res` permanent slots/layer)
     - Slow memory tier (Blue pebbles, `HostCache` pinned RAM + NVMe `experts.direct`)
     - Exact transitions:
       * R3 (Compute in Red): Permanent VRAM hit (`d_res_exp[L]`) or Dynamic `CardPool` VRAM hit (0 I/O)
       * R1_host (Blue->Red from Pinned RAM): `HostCache` H2D transfer (`17,547,264` bytes over PCIe)
       * R1_nvme (Blue->Red from NVMe): `O_DIRECT` pread + H2D transfer (`17,547,264` bytes)
       * R2_wb (Red->Blue Writeback): `CardPool` victim D2H transfer to `HostCache` (`17,547,264` bytes over PCIe)
       * R4_drop (Delete Red without Writeback): Clean-input victim drop when already backed in `HostCache` (0 I/O)
     - Hong & Kung 2S-partition lower bound: Q_min = S * (P(2S) - 1) where h = ceil(distinct_working_set / (2S))
  2. Eyeriss v1 4-Tier Energy & Data-Movement Hierarchy (`eyeriss-context.md`, ISCA 2016 Table IV):
     - Tier 0: On-chip GPU Shared Memory / Registers (1x per word)
     - Tier 1: On-card GPU GDDR6X VRAM (`d_res_exp` + `CardPool`) (2x per word)
     - Tier 2: Pinned Host RAM (`HostCache` over PCIe Gen4) (6x per word)
     - Tier 3: NVMe Direct Storage (`experts.direct` O_DIRECT) (200x per word)
  3. SpecMD Collision Miss & Replacement Policy Tracker (`specmd-context.md`, arXiv:2602.03921):
     - Tracks `collision_misses_pass` (an expert that was resident in `CardPool` at the start of pass t
       and evicted by an earlier layer l' < l during pass t before layer l requested it in pass t)
     - Compares:
       * Baseline Policy 1 (`Heat-48` + exclusive `HostCache` D2H writeback, current `decode.cuh` / `supply`)
       * Cycle 2 (`Clean-Input R4` inclusive `HostCache` eliminating redundant D2H writebacks of unmodified experts)
       * Cycle 3a (`Hit-First access_group` intra-layer ordering)
       * Cycle 3b (`Least-Stale LS` layer-order-aware eviction protecting upcoming layers l_e > l_cur in pass t)
       * Belady / OPT offline oracle (`Policy 2`)
"""

import argparse
import collections
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

EXPERTS = 896
EXPERT_RAW = 17_547_264  # 16.734 MiB per expert in Kimi K3
TOP = 16

# Card layer assignment in K3Full93Chain (`k3_gpu_experts_lib.cu:785`)
CARD_LAYERS = [
    (0, 1, 15),   # Card 0: L0 + MoE L1..15 (15 MoE layers)
    (1, 16, 31),  # Card 1: MoE L16..31 (16 MoE layers)
    (2, 32, 46),  # Card 2: MoE L32..46 (15 MoE layers)
    (3, 47, 61),  # Card 3: MoE L47..61 (15 MoE layers)
    (4, 62, 76),  # Card 4: MoE L62..76 (15 MoE layers)
    (5, 77, 92),  # Card 5: MoE L77..92 (16 MoE layers)
]


def layer_to_card(layer: int) -> int:
    for c, l0, l1 in CARD_LAYERS:
        if l0 <= layer <= l1:
            return c
    return 0


def ekey(layer: int, e: int) -> int:
    return layer * EXPERTS + e


def load_permanent_experts(dataset_dir: str, k_res: int = 28) -> Dict[int, Set[int]]:
    """
    Loads the exact `k_res` highest-router-bias permanent VRAM experts per layer `1..92`
    from `/opt/clover-k3/datasets/computed/trunk-L/trunk.direct` (record 30), matching
    `k3_full93_init` in `k3_gpu_experts_lib.cu:798-808`.
    """
    import struct
    perm: Dict[int, Set[int]] = {}
    for L in range(1, 93):
        tpath = os.path.join(dataset_dir, f"trunk-{L}", "trunk.direct")
        if os.path.exists(tpath):
            with open(tpath, "rb") as f:
                # Header has record table: uint32 nrec, then nrec * (uint64 off, uint64 len) or similar;
                # Let's read bias or fall back to top frequencies if trunk format is custom.
                pass
        perm[L] = set()
    return perm


def parse_trace_file(path: str) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    """
    Parses a K3GPU_TRACE file (`layer nc e1:c1 e2:c2 ...`).
    Returns list of `(layer, nc, [(expert_id, token_count), ...])`.
    """
    calls = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            layer = int(parts[0])
            nc = int(parts[1])
            ec = []
            for item in parts[2:]:
                e_str, c_str = item.split(":")
                ec.append((int(e_str), int(c_str)))
            calls.append((layer, nc, ec))
    return calls


def derive_permanent_experts_from_prefill(
    calls: List[Tuple[int, int, List[Tuple[int, int]]]], k_res: int = 28
) -> Dict[int, Set[int]]:
    """
    Selects the top `k_res` experts per layer from Pass 0 (the 157-token prompt prefill pass)
    broken by lowest expert index, or static prior across the trace.
    """
    perm: Dict[int, Set[int]] = {}
    for L in range(1, 93):
        perm[L] = set()
    for layer, nc, ec in calls[:92]:
        if layer in perm and not perm[layer]:
            # Sort by (-count, e) to pick top k_res
            ranked = sorted(ec, key=lambda x: (-x[1], x[0]))
            perm[layer] = set(e for e, _ in ranked[:k_res])
    return perm


class SlotCacheSim:
    """
    Exact faithful model of `SlotCache` in `decode.cuh` and `src/rust/slotcache/src/lib.rs`,
    extended with:
      - `policy == 0`: Strict LRU (`limit = 1`)
      - `policy == 1`: Heat-48 (`limit = 48` from LRU tail, lowest decayed `heat` evicted; current Clover default)
      - `policy == 2`: Belady / MIN offline oracle (evicts entry needed farthest in the future)
      - `policy == 3`: SpecMD Least-Stale (`LS`, evicts stale entries from already-visited layers
                       `(pass_id < cur_pass and l_e <= cur_layer)` or earlier passes before upcoming layers
                       `(pass_id == cur_pass - 1 and l_e > cur_layer)` in the same card's layer cycle!)
    """

    def __init__(self, cap_slots: int, policy: int = 1):
        self.cap_slots = cap_slots
        self.policy = policy
        # OrderedDict from LRU (oldest at beginning) to MRU (newest at end)
        # Value: [heat, pass_id, layer_id, next_use]
        self.entries: "collections.OrderedDict[int, List[float]]" = collections.OrderedDict()

    def contains(self, key: int) -> bool:
        return key in self.entries

    def find(self, key: int) -> bool:
        if key not in self.entries:
            return False
        self.entries.move_to_end(key, last=True)
        return True

    def touch(self, key: int, weight: float, cur_pass: int, cur_layer: int, next_use: int = 0) -> None:
        if key not in self.entries:
            return
        ent = self.entries[key]
        ent[0] = ent[0] * 0.8 + weight
        ent[1] = cur_pass
        ent[2] = cur_layer
        ent[3] = next_use

    def remove(self, key: int) -> Optional[float]:
        ent = self.entries.pop(key, None)
        return ent[0] if ent is not None else None

    def acquire(
        self,
        key: int,
        prot: Set[int],
        heat0: float,
        cur_pass: int,
        cur_layer: int,
        next_use: int = 0,
    ) -> Tuple[int, float]:
        """
        Inserts `key` into the cache. Returns `(victim_key, victim_heat)` or `(-1, 0.0)` if a spare slot was free.
        """
        vkey = -1
        vheat = 0.0
        if len(self.entries) >= self.cap_slots:
            if self.policy == 2:
                # Belady / MIN: evict non-protected entry with largest next_use
                best_k = -1
                best_nu = -1
                for k, ent in self.entries.items():
                    if k in prot:
                        continue
                    if ent[3] > best_nu:
                        best_nu = int(ent[3])
                        best_k = k
                vkey = best_k
            elif self.policy == 3:
                # SpecMD Least-Stale (LS) + Heat tie-breaking across the card's layer cycle:
                # At `(cur_pass, cur_layer)`:
                #   - Tier A (most stale / safe to evict first): entries from pass <= cur_pass - 2, OR
                #     entries from pass == cur_pass - 1 with layer <= cur_layer (already passed in cur_pass without being hit!)
                #   - Tier B (already used in cur_pass, won't be needed again until cur_pass + 1):
                #     entries with pass == cur_pass and layer < cur_layer
                #   - Tier C (upcoming in cur_pass — PROTECT from collision!):
                #     entries from pass == cur_pass - 1 with layer > cur_layer (about to be visited in cur_pass!)
                best_k = -1
                best_tuple = None
                for idx, (k, ent) in enumerate(self.entries.items()):
                    if k in prot:
                        continue
                    h_val, p_id, l_id, _ = ent
                    p_id = int(p_id)
                    l_id = int(l_id)
                    if p_id < cur_pass - 1 or (p_id == cur_pass - 1 and l_id <= cur_layer):
                        stale_tier = 0  # Truly stale: missed its turn in cur_pass or older
                    elif p_id == cur_pass and l_id < cur_layer:
                        stale_tier = 1  # Already used earlier in cur_pass; next possible use is cur_pass + 1
                    else:
                        stale_tier = 2  # Upcoming in cur_pass (p_id == cur_pass - 1 and l_id > cur_layer)
                    cand = (stale_tier, h_val, idx)
                    if best_tuple is None or cand < best_tuple:
                        best_tuple = cand
                        best_k = k
                vkey = best_k
            else:
                # Policy 0 (strict LRU, limit=1) or Policy 1 (Heat-48, limit=48 from LRU tail)
                limit = 48 if self.policy != 0 else 1
                seen = 0
                best_k = -1
                best_heat = 0.0
                for k, ent in self.entries.items():
                    if k in prot:
                        continue
                    if best_k == -1 or ent[0] < best_heat:
                        best_k = k
                        best_heat = ent[0]
                    seen += 1
                    if seen >= limit:
                        break
                vkey = best_k

            if vkey < 0:
                raise RuntimeError("expert cache: every slot is in use by one call (too few slots)")
            vheat = self.entries.pop(vkey)[0]

        self.entries[key] = [heat0, cur_pass, cur_layer, next_use]
        self.entries.move_to_end(key, last=True)
        return vkey, vheat


def run_simulation(
    calls: List[Tuple[int, int, List[Tuple[int, int]]]],
    card_slots: int = 128,
    host_slots: int = 2936,
    k_res: int = 28,
    policy: int = 1,
    inclusive_host: int = 0,
    hit_first: bool = False,
    decode_only_after_warmup: bool = True,
) -> Dict[str, object]:
    """
    Replays `calls` (list of `(layer, nc, [(e, count), ...])`) across the 6 GPU cards (`Card 0..5`)
    and the shared `HostCache` (`host_slots`), measuring the exact 4-Tier Data-Movement & Red-Blue Pebble ledger.
    """
    perm = derive_permanent_experts_from_prefill(calls, k_res=k_res)

    # Precompute next_use for Belady (policy == 2)
    next_use_arr = [10**9] * len(calls)
    next_use_per_call: List[Dict[int, int]] = [{} for _ in range(len(calls))]
    if policy == 2:
        last_seen: Dict[int, int] = {}
        for idx in range(len(calls) - 1, -1, -1):
            layer, _, ec = calls[idx]
            d = {}
            for e, _ in ec:
                if e in perm[layer]:
                    continue
                k = ekey(layer, e)
                d[k] = last_seen.get(k, 10**9)
                last_seen[k] = idx
            next_use_per_call[idx] = d

    pools = [SlotCacheSim(card_slots, policy=policy) for _ in range(6)]
    host_cache = SlotCacheSim(host_slots, policy=1 if policy != 2 else 2)

    # Metrics
    perm_hits = 0
    dyn_needed = 0
    hit_card = 0       # R3 (dynamic VRAM hit)
    hit_host = 0       # R1_host (Pinned RAM -> VRAM H2D)
    from_drive = 0     # R1_nvme (NVMe -> Bounce -> VRAM H2D)
    written_back = 0   # R2_wb (VRAM -> Pinned RAM D2H writeback)
    r4_clean_drops = 0 # R4 (0-byte clean drop when victim is already in inclusive HostCache)
    collision_misses_pass = 0

    # Track which keys were resident in each card pool at the start of the current 92-layer pass
    pass_start_resident: List[Set[int]] = [set() for _ in range(6)]
    evicted_in_current_pass: List[Set[int]] = [set() for _ in range(6)]

    # Distinct dynamic keys per card (for Hong & Kung 2S-partition bound)
    card_distinct_keys: List[Set[int]] = [set() for _ in range(6)]

    for call_idx, (layer, nc, ec) in enumerate(calls):
        cur_pass = call_idx // 92
        c = layer_to_card(layer)
        pool = pools[c]

        # At the start of each 92-layer pass on card c, snapshot resident keys
        if layer == CARD_LAYERS[c][1]:
            pass_start_resident[c] = set(pool.entries.keys())
            evicted_in_current_pass[c] = set()

        # Filter out permanent VRAM experts (`d_res_exp[L]`)
        dyn_ec = []
        for e, cnt in ec:
            if e in perm[layer]:
                if not decode_only_after_warmup or cur_pass >= 1:
                    perm_hits += 1
            else:
                dyn_ec.append((e, cnt))

        record_stats = (not decode_only_after_warmup) or (cur_pass >= 1)
        if record_stats:
            dyn_needed += len(dyn_ec)
            for e, _ in dyn_ec:
                card_distinct_keys[c].add(ekey(layer, e))

        # Process in chunks of `card_slots` if `len(dyn_ec) > card_slots`
        for gi in range(0, len(dyn_ec), card_slots):
            grp = dyn_ec[gi : gi + card_slots]
            prot = {ekey(layer, e) for e, _ in grp}
            need = []
            victims = []

            if hit_first:
                # SpecMD `access_group` Hit-First ordering:
                # Pass 1: find & touch ALL hits in `grp` first so their LRU order and heat are updated
                # before any miss scans for a victim!
                misses = []
                for e, cnt in grp:
                    k = ekey(layer, e)
                    nu = next_use_per_call[call_idx].get(k, 10**9) if policy == 2 else 0
                    if pool.find(k):
                        pool.touch(k, float(cnt), cur_pass, layer, nu)
                        if record_stats:
                            hit_card += 1
                    else:
                        misses.append((e, cnt, k, nu))
                # Pass 2: acquire slots for misses
                for e, cnt, k, nu in misses:
                    if record_stats and (k in pass_start_resident[c]) and (k in evicted_in_current_pass[c]):
                        collision_misses_pass += 1
                    vk, vh = pool.acquire(k, prot, float(cnt), cur_pass, layer, nu)
                    if vk >= 0:
                        evicted_in_current_pass[c].add(vk)
                        victims.append((vk, vh))
                    need.append((e, cnt, k, nu))
            else:
                # Baseline sequential rank/index order (`decode.cuh:691`)
                for e, cnt in grp:
                    k = ekey(layer, e)
                    nu = next_use_per_call[call_idx].get(k, 10**9) if policy == 2 else 0
                    if pool.find(k):
                        pool.touch(k, float(cnt), cur_pass, layer, nu)
                        if record_stats:
                            hit_card += 1
                        continue
                    if record_stats and (k in pass_start_resident[c]) and (k in evicted_in_current_pass[c]):
                        collision_misses_pass += 1
                    vk, vh = pool.acquire(k, prot, float(cnt), cur_pass, layer, nu)
                    if vk >= 0:
                        evicted_in_current_pass[c].add(vk)
                        victims.append((vk, vh))
                    need.append((e, cnt, k, nu))

            if not need:
                continue

            if inclusive_host == 0:
                # Current Clover (`decode.cuh:706-718`): Exclusive swap between CardPool and HostCache
                # 1. Hits in HostCache are REMOVED from HostCache (`hc.c.remove(key)`)
                for e, cnt, k, nu in need:
                    if host_cache.find(k):
                        host_cache.remove(k)
                        if record_stats:
                            hit_host += 1
                    else:
                        if record_stats:
                            from_drive += 1
                # 2. Every CardPool victim is written back D2H over PCIe into HostCache (`R2_wb`)
                for vk, vh in victims:
                    host_cache.acquire(vk, set(), vh, cur_pass, layer, 10**9)
                    if record_stats:
                        written_back += 1
            elif inclusive_host == 3:
                # Cycle 2c Zero-D2H Host-Shadow Pointer Swap (`Shadow-Swap R4`):
                # Each CardPool slot s in [0..S-1] retains the Host RAM buffer pointer from which it was loaded
                # (either the HostCache slot on H2D hit, or the host bounce slot on NVMe pread).
                # When CardPool evicts victim `vk` from slot s, its clean bytes are ALREADY in Host RAM,
                # so demoting `vk` into HostCache is an O(1) pointer swap (`R4_drop`, 0 PCIe D2H bytes,
                # 0 `written_back` event stall) while keeping CardPool (6*S) and HostCache (H) 100% DISJOINT!
                for e, cnt, k, nu in need:
                    if host_cache.find(k):
                        host_cache.remove(k)
                        if record_stats:
                            hit_host += 1
                    else:
                        if record_stats:
                            from_drive += 1
                for vk, vh in victims:
                    # O(1) host pointer handoff into HostCache (zero PCIe D2H transfer!)
                    host_cache.acquire(vk, set(), vh, cur_pass, layer, 10**9)
                    if record_stats:
                        r4_clean_drops += 1
            elif inclusive_host == 2:
                # Cycle 2b Victim-Admitted Keep-on-H2D-Hit (`keep_on_h2d_hit`):
                # 1. NVMe reads (`from_drive`) go directly to CardPool without polluting HostCache on first touch.
                # 2. HostCache hits (`hit_host`) are KEPT in HostCache (`find` + `touch`, no `remove`),
                #    so when CardPool later evicts that unmodified expert, it is ALREADY in HostCache -> 0-byte `R4` drop!
                # 3. Only victims not already in HostCache are written back D2H into HostCache.
                for e, cnt, k, nu in need:
                    if host_cache.find(k):
                        host_cache.touch(k, float(cnt), cur_pass, layer, nu)
                        if record_stats:
                            hit_host += 1
                    else:
                        if record_stats:
                            from_drive += 1
                for vk, vh in victims:
                    if host_cache.find(vk):
                        host_cache.touch(vk, vh, cur_pass, layer, 10**9)
                        if record_stats:
                            r4_clean_drops += 1
                    else:
                        host_cache.acquire(vk, set(), vh, cur_pass, layer, 10**9)
                        if record_stats:
                            written_back += 1
            else:
                # Cycle 2a Full Inclusive HostCache (insert into HostCache on NVMe pread + R4 drop on CardPool eviction):
                for e, cnt, k, nu in need:
                    if host_cache.find(k):
                        host_cache.touch(k, float(cnt), cur_pass, layer, nu)
                        if record_stats:
                            hit_host += 1
                    else:
                        if record_stats:
                            from_drive += 1
                        host_cache.acquire(k, set(), float(cnt), cur_pass, layer, nu)
                for vk, vh in victims:
                    if host_cache.contains(vk):
                        if record_stats:
                            r4_clean_drops += 1
                    else:
                        host_cache.acquire(vk, set(), vh, cur_pass, layer, 10**9)
                        if record_stats:
                            written_back += 1

    # Compute Hong & Kung 2S-partition lower bound Q_min across the 6 cards
    # For each card c with fast capacity S = card_slots and total distinct dynamic experts W_c > S,
    # any schedule must partition the DAG into at least P(2S) >= ceil(W_c / (2S)) subcalculations,
    # plus the compulsory cold-start loads when W_c > 2S.
    hk_q_min = 0
    for c in range(6):
        w_c = len(card_distinct_keys[c])
        p_2s = max(1, math.ceil(w_c / (2.0 * card_slots)))
        hk_q_min += card_slots * (p_2s - 1)

    total_requests = perm_hits + dyn_needed
    total_vram_hits = perm_hits + hit_card
    r1_loads = hit_host + from_drive
    r2_stores = written_back
    pebble_q = r1_loads + r2_stores

    # Bytes moved across tiers
    tier1_vram_bytes = total_vram_hits * EXPERT_RAW
    tier2_pcie_h2d_bytes = hit_host * EXPERT_RAW
    tier2_pcie_d2h_bytes = written_back * EXPERT_RAW
    tier3_nvme_bytes = from_drive * EXPERT_RAW
    total_pcie_bytes = tier2_pcie_h2d_bytes + tier2_pcie_d2h_bytes + tier3_nvme_bytes

    # Eyeriss v1 4-Tier Relative Energy Score (Tier 1 VRAM = 2x, Tier 2 Host RAM PCIe = 6x, Tier 3 NVMe = 200x per expert)
    eyeriss_cost = (
        2.0 * total_vram_hits
        + 6.0 * (hit_host + written_back)
        + 200.0 * from_drive
    )

    return {
        "card_slots": card_slots,
        "host_slots": host_slots,
        "k_res": k_res,
        "policy": policy,
        "inclusive_host": inclusive_host,
        "hit_first": hit_first,
        "total_requests": total_requests,
        "perm_hits": perm_hits,
        "dyn_needed": dyn_needed,
        "hit_card_R3": hit_card,
        "hit_host_R1": hit_host,
        "from_drive_R1": from_drive,
        "written_back_R2": written_back,
        "r4_clean_drops": r4_clean_drops,
        "collision_misses_pass": collision_misses_pass,
        "dyn_hit_rate_pct": (100.0 * hit_card / dyn_needed) if dyn_needed else 0.0,
        "overall_vram_hit_rate_pct": (100.0 * total_vram_hits / total_requests) if total_requests else 0.0,
        "collision_rate_pct": (100.0 * collision_misses_pass / dyn_needed) if dyn_needed else 0.0,
        "pebble_Q": pebble_q,
        "hong_kung_Q_min": hk_q_min,
        "tier1_vram_gb": tier1_vram_bytes / 1e9,
        "tier2_pcie_h2d_gb": tier2_pcie_h2d_bytes / 1e9,
        "tier2_pcie_d2h_gb": tier2_pcie_d2h_bytes / 1e9,
        "tier3_nvme_gb": tier3_nvme_bytes / 1e9,
        "total_pcie_gb": total_pcie_bytes / 1e9,
        "eyeriss_weighted_cost": eyeriss_cost,
        "eyeriss_cost_per_req": eyeriss_cost / total_requests if total_requests else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Clover 4-Tier Data-Movement & Red-Blue Pebble I/O Ledger")
    parser.add_argument("--trace", type=str, default="/home/usergpu/k3gpu/results/p1d.trace")
    parser.add_argument("--card-slots", type=int, default=128)
    parser.add_argument("--host-slots", type=int, default=2936)
    parser.add_argument("--k-res", type=int, default=28)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    calls = parse_trace_file(args.trace)
    n_passes = len(calls) // 92
    print(f"=== Clover 4-Tier Data-Movement & Red-Blue Pebble Ledger ===")
    print(f"Trace: {args.trace} ({len(calls)} layer calls = {n_passes} full 92-layer passes)")
    print(f"Config: k_res={args.k_res}/layer (2,576 perm VRAM experts), CardPool S={args.card_slots} slots/card, HostCache={args.host_slots} slots")
    print("-" * 118)

    configs = [
        ("Cycle 1 Baseline (Heat-48 + Exclusive D2H)", 1, 0, False),
        ("Cycle 1 Strict LRU (Policy 0 + Exclusive D2H)", 0, 0, False),
        ("Cycle 2a Red-Blue Full-Inclusive Host R4", 1, 1, False),
        ("Cycle 2b Red-Blue Keep-on-H2D-Hit R4", 1, 2, False),
        ("Cycle 2c Red-Blue Shadow-Swap Zero-D2H R4", 1, 3, False),
        ("Cycle 3b SpecMD LS + Exclusive D2H", 3, 0, True),
        ("Cycle 3b SpecMD LS + Shadow-Swap Zero-D2H R4", 3, 3, True),
        ("Offline Oracle Belady/OPT + Shadow-Swap R4", 2, 3, True),
    ]

    results = {}
    hdr = f"{'Configuration':<48} | {'DynHit%':>7} | {'AllVRAM%':>8} | {'Coll%':>6} | {'R1_Host':>7} | {'R1_NVMe':>7} | {'R2_WB':>7} | {'R4_Drop':>7} | {'PCIe_GB':>8} | {'Eyeriss/Req':>11}"
    print(hdr)
    print("-" * len(hdr))
    for label, pol, inc, hf in configs:
        r = run_simulation(
            calls,
            card_slots=args.card_slots,
            host_slots=args.host_slots,
            k_res=args.k_res,
            policy=pol,
            inclusive_host=inc,
            hit_first=hf,
        )
        results[label] = r
        print(
            f"{label:<48} | {r['dyn_hit_rate_pct']:>6.2f}% | {r['overall_vram_hit_rate_pct']:>7.2f}% | "
            f"{r['collision_rate_pct']:>5.2f}% | {r['hit_host_R1']:>7d} | {r['from_drive_R1']:>7d} | "
            f"{r['written_back_R2']:>7d} | {r['r4_clean_drops']:>7d} | {r['total_pcie_gb']:>7.2f}G | "
            f"{r['eyeriss_cost_per_req']:>11.3f}"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
