import os
import json
import numpy as np
from sim_cache import SinglePoolCache

TRACE_DIR = "/root/dm-study/specmd/w2/traces"


def precompute_prefetches(pred_probs, mode, tau80):
    """Precompute prefetch lists once for (P, T, 15) using vectorized argsort."""
    P, T, L15, _ = pred_probs.shape
    if mode == "None":
        return None
    orders = np.argsort(-pred_probs, axis=-1)  # (P, T, 15, 64)
    if mode == "TopK_8":
        return orders[:, :, :, :8].tolist()
    if mode == "Overfetch_12":
        return orders[:, :, :, :12].tolist()
    if mode == "Score_Global80":
        out = [[[None] * L15 for _ in range(T)] for _ in range(P)]
        for p in range(P):
            for t in range(T):
                for l in range(L15):
                    row_ord = orders[p, t, l]
                    row_pr = pred_probs[p, t, l]
                    sel = [int(e) for e in row_ord if row_pr[e] >= tau80]
                    out[p][t][l] = sel if sel else [int(row_ord[0])]
        return out
    raise ValueError(mode)


def run_decode_with_prefetch(indices, scores, pref_table, cap, pol, pref_before_demand=False):
    """
    Simulate decode-style execution (2,048 tokens x 16 layers) with Gate Hook prefetching.
    If pref_before_demand=False (Algorithm 1 order): Phase 2 (demand misses at layer l) then Phase 3 (prefetch l+1).
    If pref_before_demand=True (concurrent/interleaved stress): prefetch l+1 issued before layer l demand misses.
    """
    P, T, L, K = indices.shape
    cache = SinglePoolCache(capacity=cap, policy=pol, num_layers=L)
    tok_id = 0
    for p in range(P):
        for t in range(T):
            for l in range(L):
                if pref_before_demand and l < L - 1 and pref_table is not None:
                    for pe in pref_table[p][t][l]:
                        cache.prefetch((l + 1) * 64 + pe, tok_id, l, l + 1, pe)

                # Demand accesses at layer l
                for k in range(K):
                    e = int(indices[p, t, l, k])
                    sc = float(scores[p, t, l, k])
                    cache.access(l * 64 + e, tok_id, l, e, sc)

                if (not pref_before_demand) and l < L - 1 and pref_table is not None:
                    for pe in pref_table[p][t][l]:
                        cache.prefetch((l + 1) * 64 + pe, tok_id, l, l + 1, pe)
            tok_id += 1
    return cache.summary()


def main():
    d_data = np.load(os.path.join(TRACE_DIR, "decode_traces.npz"))
    s_data = np.load(os.path.join(TRACE_DIR, "spec_predictions.npz"))
    indices = d_data["topk_indices"]
    scores = d_data["topk_scores"]
    pred_A = s_data["pred_A_prompt"].astype(np.float32)
    pred_B = s_data["pred_B_prompt"].astype(np.float32)
    tau80 = float(s_data["global_tau80"][0])

    pref_A = {m: precompute_prefetches(pred_A, m, tau80) for m in ["None", "TopK_8", "Overfetch_12", "Score_Global80"]}
    pref_B = {m: precompute_prefetches(pred_B, m, tau80) for m in ["TopK_8", "Overfetch_12", "Score_Global80"]}

    summary = {"phase2_then_phase3_varA": {}, "phase2_then_phase3_varB": {}, "concurrent_interleaved_varA": {}}

    print("=== 1. DECODE-STYLE WITH PREFETCH (Algorithm 1: Demand l -> Prefetch l+1, Variant A) ===")
    for cap, label in [(10, "1%"), (51, "5%"), (256, "25%")]:
        summary["phase2_then_phase3_varA"][cap] = {}
        print(f"\n--- Capacity C={cap} ({label} of 1,024 experts) ---")
        print(f"  {'Policy':6s} | {'PrefetchMode':14s} | {'HitRate':>7s} | {'CollPass':>8s} | {'CollIntra':>9s} | {'PrefLoad':>8s} | {'Useful':>7s} | {'Wasted':>7s}")
        for pref_mode in ["None", "TopK_8", "Overfetch_12", "Score_Global80"]:
            summary["phase2_then_phase3_varA"][cap][pref_mode] = {}
            pol_list = ["LS", "LRU", "LFU", "SB", "FLD"] if pref_mode in ["None", "TopK_8"] else ["LS", "LRU", "SB"]
            for pol in pol_list:
                res = run_decode_with_prefetch(indices, scores, pref_A[pref_mode], cap, pol, False)
                summary["phase2_then_phase3_varA"][cap][pref_mode][pol] = {
                    "hit_rate": round(res["hit_rate"], 5),
                    "coll_pass": round(res["collision_rate_pass"], 5),
                    "coll_intra": round(res["collision_rate_intra"], 5),
                    "pref_loads": res["prefetch_loads"],
                    "pref_useful": res["prefetch_useful"],
                    "pref_wasted": res["prefetch_wasted"],
                    "pref_cached": res["prefetch_already_cached"],
                }
                if pol in ["LS", "LRU", "SB"] or pref_mode == "TopK_8":
                    print(
                        f"  {pol:6s} | {pref_mode:14s} | {res['hit_rate']*100:6.2f}% | "
                        f"{res['collision_rate_pass']*100:7.2f}% | {res['collision_rate_intra']*100:8.2f}% | "
                        f"{res['prefetch_loads']:8d} | {res['prefetch_useful']:7d} | {res['prefetch_wasted']:7d}"
                    )

    print("\n=== 2. VARIANT B (Next-Layer RMSNorm) vs VARIANT A AT 5% (C=51) AND 25% (C=256) WITH LS ===")
    for cap in [51, 256]:
        summary["phase2_then_phase3_varB"][cap] = {}
        for pref_mode in ["TopK_8", "Overfetch_12", "Score_Global80"]:
            resA = summary["phase2_then_phase3_varA"][cap][pref_mode]["LS"]
            resB = run_decode_with_prefetch(indices, scores, pref_B[pref_mode], cap, "LS", False)
            summary["phase2_then_phase3_varB"][cap][pref_mode] = {
                "hit_rate": round(resB["hit_rate"], 5),
                "pref_loads": resB["prefetch_loads"],
                "pref_useful": resB["prefetch_useful"],
                "pref_wasted": resB["prefetch_wasted"],
            }
            print(
                f"  C={cap:3d} LS {pref_mode:14s}: VarA Hit={resA['hit_rate']*100:6.2f}% (useful={resA['pref_useful']:6d}, wasted={resA['pref_wasted']:6d}) "
                f"vs VarB Hit={resB['hit_rate']*100:6.2f}% (useful={resB['prefetch_useful']:6d}, wasted={resB['prefetch_wasted']:6d})"
            )

    print("\n=== 3. CONCURRENT / INTERLEAVED PREFETCH (Prefetch l+1 before Layer l Demand Misses) AT 5% (C=51) ===")
    for pref_mode in ["TopK_8", "Overfetch_12", "Score_Global80"]:
        summary["concurrent_interleaved_varA"][pref_mode] = {}
        for pol in ["LS", "LRU", "LFU", "SB", "FLD"]:
            res = run_decode_with_prefetch(indices, scores, pref_A[pref_mode], 51, pol, True)
            summary["concurrent_interleaved_varA"][pref_mode][pol] = {
                "hit_rate": round(res["hit_rate"], 5),
                "coll_intra": round(res["collision_rate_intra"], 5),
                "pref_useful": res["prefetch_useful"],
                "pref_wasted": res["prefetch_wasted"],
            }
            print(
                f"  C=51 {pol:4s} {pref_mode:14s}: Hit={res['hit_rate']*100:6.2f}% | CollIntra={res['collision_rate_intra']*100:6.2f}% | "
                f"Useful={res['prefetch_useful']:6d} | Wasted={res['prefetch_wasted']:6d}"
            )

    with open(os.path.join(TRACE_DIR, "e7_prefetch_sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved e7_prefetch_sim_summary.json")


if __name__ == "__main__":
    main()
