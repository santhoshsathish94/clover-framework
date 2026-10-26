import os
import json
import numpy as np
from sim_cache import SinglePoolCache

TRACE_DIR = "/root/dm-study/specmd/w2/traces"
POLICIES = ["LRU", "LFU", "FIFO", "FLD", "SB", "LS"]
CAPACITIES = [(10, "1%"), (51, "5%"), (256, "25%")]


def build_prompt_stream(indices, scores):
    """Layer-major per prompt (8 prompts x 16 layers x 256 tokens x 8 experts)."""
    P, T, L, K = indices.shape
    keys, passes, layers, exps, scs = [], [], [], [], []
    for p in range(P):
        for l in range(L):
            for t in range(T):
                for k in range(K):
                    e = int(indices[p, t, l, k])
                    keys.append(l * 64 + e)
                    passes.append(p)
                    layers.append(l)
                    exps.append(e)
                    scs.append(float(scores[p, t, l, k]))
    return keys, passes, layers, exps, scs


def build_decode_stream(indices, scores):
    """Token-major across all layers (2,048 tokens x 16 layers x 8 experts)."""
    P, T, L, K = indices.shape
    keys, passes, layers, exps, scs = [], [], [], [], []
    tok_id = 0
    for p in range(P):
        for t in range(T):
            for l in range(L):
                for k in range(K):
                    e = int(indices[p, t, l, k])
                    keys.append(l * 64 + e)
                    passes.append(tok_id)
                    layers.append(l)
                    exps.append(e)
                    scs.append(float(scores[p, t, l, k]))
            tok_id += 1
    return keys, passes, layers, exps, scs


def run_global(keys, passes, layers, exps, scs, cap, pol):
    c = SinglePoolCache(capacity=cap, policy=pol, num_layers=16)
    for i in range(len(keys)):
        c.access(keys[i], passes[i], layers[i], exps[i], scs[i])
    return c.summary()


def run_per_layer(keys, passes, layers, exps, scs, total_cap, pol, hits_first=False):
    base = total_cap // 16
    rem = total_cap % 16
    pools = [
        SinglePoolCache(capacity=base + (1 if l < rem else 0), policy=pol, num_layers=16)
        for l in range(16)
    ]
    if not hits_first:
        for i in range(len(keys)):
            l = layers[i]
            pools[l].access(keys[i], passes[i], l, exps[i], scs[i])
    else:
        for i in range(0, len(keys), 8):
            l = layers[i]
            pools[l].access_group(
                keys[i:i+8], passes[i], l, exps[i:i+8], scs[i:i+8]
            )
    hits = sum(p.hits for p in pools)
    misses = sum(p.misses for p in pools)
    tot = hits + misses
    return {
        "policy": f"PerLayer-{pol}",
        "capacity": total_cap,
        "hit_rate": hits / float(tot),
        "miss_rate": misses / float(tot),
    }


def main():
    p_data = np.load(os.path.join(TRACE_DIR, "prompt_traces.npz"))
    d_data = np.load(os.path.join(TRACE_DIR, "decode_traces.npz"))

    p_stream = build_prompt_stream(p_data["topk_indices"], p_data["topk_scores"])
    d_stream = build_decode_stream(d_data["topk_indices"], d_data["topk_scores"])

    summary = {"prompt_style_global": {}, "decode_style_global": {}, "decode_style_per_layer": {}}

    print("=== 1. PROMPT-STYLE GLOBAL CACHE (8 prompts x 16 layers x 256 tokens x 8 = 262,144 reqs) ===")
    for cap, label in CAPACITIES:
        summary["prompt_style_global"][cap] = {}
        print(f"\n--- Capacity C={cap} ({label} of 1,024 experts = {cap*12.0:.0f} MiB / {cap*12.0/1024:.3f} GiB bf16) ---")
        print(f"  {'Policy':6s} | {'HitRate':>8s} | {'MissRate':>8s} | {'CollIntra':>9s} | {'CollPass':>8s} | {'ColdMiss':>8s}")
        for pol in POLICIES:
            res = run_global(*p_stream, cap, pol)
            summary["prompt_style_global"][cap][pol] = {
                "hit_rate": round(res["hit_rate"], 5),
                "miss_rate": round(res["miss_rate"], 5),
                "collision_rate_intra": round(res["collision_rate_intra"], 5),
                "collision_rate_pass": round(res["collision_rate_pass"], 5),
                "cold_misses": res["cold_misses"],
                "layer_stats": res["layer_stats"],
            }
            print(
                f"  {pol:6s} | {res['hit_rate']*100:7.2f}% | {res['miss_rate']*100:7.2f}% | "
                f"{res['collision_rate_intra']*100:8.2f}% | {res['collision_rate_pass']*100:7.2f}% | {res['cold_misses']:8d}"
            )

    print("\n=== 2. PER-LAYER BREAKDOWN AT 5% CAPACITY (C=51) IN PROMPT-STYLE ===")
    print("  Layer |  LS Hit%  LS Coll% | LRU Hit% LRU Coll% | LFU Hit% LFU Coll% |  SB Hit%  SB Coll%")
    for l in range(16):
        row = []
        for pol in ["LS", "LRU", "LFU", "SB"]:
            st = summary["prompt_style_global"][51][pol]["layer_stats"][l]
            tot_l = st["hits"] + st["misses"]
            hr_l = st["hits"] / float(tot_l) * 100.0
            cr_l = st["coll_intra"] / float(tot_l) * 100.0
            row.append(f"{hr_l:7.2f}%  {cr_l:7.2f}%")
        print(f"  L{l:02d}   | " + " | ".join(row))

    print("\n=== 3. DECODE-STYLE GLOBAL CACHE (2,048 tokens x 16 layers x 8 = 262,144 reqs, no prefetch) ===")
    for cap, label in CAPACITIES:
        summary["decode_style_global"][cap] = {}
        print(f"\n--- Capacity C={cap} ({label} of 1,024 experts = {cap*12.0:.0f} MiB / {cap*12.0/1024:.3f} GiB bf16) ---")
        print(f"  {'Policy':6s} | {'HitRate':>8s} | {'MissRate':>8s} | {'CollPass':>8s} | {'CollIntra':>9s} | {'ColdMiss':>8s}")
        for pol in POLICIES:
            res = run_global(*d_stream, cap, pol)
            summary["decode_style_global"][cap][pol] = {
                "hit_rate": round(res["hit_rate"], 5),
                "miss_rate": round(res["miss_rate"], 5),
                "collision_rate_pass": round(res["collision_rate_pass"], 5),
                "collision_rate_intra": round(res["collision_rate_intra"], 5),
                "cold_misses": res["cold_misses"],
                "layer_stats": res["layer_stats"],
            }
            print(
                f"  {pol:6s} | {res['hit_rate']*100:7.2f}% | {res['miss_rate']*100:7.2f}% | "
                f"{res['collision_rate_pass']*100:7.2f}% | {res['collision_rate_intra']*100:8.2f}% | {res['cold_misses']:8d}"
            )

    print("\n=== 4. PER-LAYER BREAKDOWN AT 25% CAPACITY (C=256) IN DECODE-STYLE ===")
    print("  Layer |  LS Hit%  LS Coll% | LRU Hit% LRU Coll% | LFU Hit% LFU Coll% |  SB Hit%  SB Coll%")
    for l in range(16):
        row = []
        for pol in ["LS", "LRU", "LFU", "SB"]:
            st = summary["decode_style_global"][256][pol]["layer_stats"][l]
            tot_l = st["hits"] + st["misses"]
            hr_l = st["hits"] / float(tot_l) * 100.0
            cr_l = st["coll_pass"] / float(tot_l) * 100.0
            row.append(f"{hr_l:7.2f}%  {cr_l:7.2f}%")
        print(f"  L{l:02d}   | " + " | ".join(row))

    print("\n=== 5. DECODE-STYLE PER-LAYER PARTITIONED POOL VS GLOBAL POOL ===")
    for cap, label in CAPACITIES:
        summary["decode_style_per_layer"][cap] = {}
        row = []
        for pol in ["LRU", "LFU", "FIFO"]:
            r_seq = run_per_layer(*d_stream, cap, pol, hits_first=False)
            r_hf = run_per_layer(*d_stream, cap, pol, hits_first=True)
            summary["decode_style_per_layer"][cap][f"{pol}_seq"] = round(r_seq["hit_rate"], 5)
            summary["decode_style_per_layer"][cap][f"{pol}_hits_first"] = round(r_hf["hit_rate"], 5)
            row.append(f"{pol}(seq={r_seq['hit_rate']*100:5.2f}%, hits_first={r_hf['hit_rate']*100:5.2f}%)")
        print(f"  C={cap:3d} ({label:3s}): " + "  ".join(row))

    with open(os.path.join(TRACE_DIR, "e6_policy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved e6_policy_summary.json")


if __name__ == "__main__":
    main()
