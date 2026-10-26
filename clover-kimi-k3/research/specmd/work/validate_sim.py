import os
import json
import numpy as np
from sim_cache import SinglePoolCache

TRACE_DIR = "/root/dm-study/specmd/w2/traces"


def run_trace_with_belady(keys, passes, layers, experts, scores, capacity, policy, num_layers=16, static_top=None):
    cache = SinglePoolCache(capacity=capacity, policy=policy, num_layers=num_layers, static_top_set=static_top)
    n = len(keys)
    if policy == "Belady":
        # Precompute next occurrence index for each step
        next_occ = [10**12] * n
        last_pos = {}
        for i in range(n - 1, -1, -1):
            k = keys[i]
            if k in last_pos:
                next_occ[i] = last_pos[k]
            last_pos[k] = i
        cur_next = {}
        for i in range(n):
            k = keys[i]
            cur_next[k] = next_occ[i]
            cache.access(k, passes[i], layers[i], experts[i], scores[i], next_use_map=cur_next)
    else:
        for i in range(n):
            cache.access(keys[i], passes[i], layers[i], experts[i], scores[i])
    return cache.summary()


def main():
    rng = np.random.default_rng(12345)
    results = {}

    # 1. IID Uniform(M=64), N=200,000 requests
    print("=== Test 1: IID Uniform(M=64), N=200,000 requests vs Closed-Form C/64 ===")
    M = 64
    N = 200000
    unif_keys = rng.integers(0, M, size=N).tolist()
    unif_passes = [i // 16 for i in range(N)]
    unif_layers = [i % 16 for i in range(N)]
    unif_scores = [1.0] * N

    t1 = {}
    for C in [8, 16, 32]:
        expected = C / float(M)
        t1[C] = {"expected": expected}
        row_str = [f"C={C:2d} (exact={expected*100:5.2f}%):"]
        for pol in ["LRU", "LFU", "FIFO", "LS"]:
            res = run_trace_with_belady(unif_keys, unif_passes, unif_layers, unif_keys, unif_scores, C, pol)
            hr = res["hit_rate"]
            err = abs(hr - expected)
            assert err < 0.003, f"Closed-form check failed for {pol} at C={C}: {hr} vs {expected}"
            t1[C][pol] = round(hr, 5)
            row_str.append(f"{pol}={hr*100:5.2f}% (|err|={err*100:.2f}%)")
        print("  " + "  ".join(row_str))
    results["test1_iid_uniform"] = t1

    # 2. IID Zipf(M=64, s=1.2), N=200,000 requests
    print("\n=== Test 2: IID Zipf(M=64, s=1.2), C=16, N=200,000 vs Exact Top-C Sum ===")
    ranks = np.arange(1, M + 1, dtype=np.float64)
    probs = (1.0 / (ranks ** 1.2))
    probs /= probs.sum()
    exact_top16 = float(probs[:16].sum())
    zipf_keys = rng.choice(M, size=N, p=probs).tolist()
    static_top16 = list(range(16))

    t2 = {"exact_top16_sum": round(exact_top16, 5)}
    for pol in ["StaticTopC", "LFU", "LRU", "FIFO", "Belady"]:
        res = run_trace_with_belady(
            zipf_keys, unif_passes, unif_layers, zipf_keys, unif_scores, 16, pol, static_top=static_top16
        )
        t2[pol] = round(res["hit_rate"], 5)
        print(f"  {pol:10s}: hit_rate={res['hit_rate']*100:6.2f}% (exact_top16={exact_top16*100:6.2f}%)")
    assert abs(t2["StaticTopC"] - exact_top16) < 0.003
    assert t2["Belady"] >= t2["StaticTopC"] >= t2["LFU"] >= t2["LRU"] >= t2["FIFO"]
    results["test2_iid_zipf"] = t2

    # 3. Cyclic sweep over C+1 = 17 items at capacity C=16 (500 cycles = 8,500 requests)
    print("\n=== Test 3: Cyclic Sweep over C+1=17 items at Capacity C=16 (8,500 requests) ===")
    C_cyc = 16
    M_cyc = 17
    cyc_keys = [i % M_cyc for i in range(M_cyc * 500)]
    cyc_passes = [i // 16 for i in range(len(cyc_keys))]
    cyc_layers = [i % 16 for i in range(len(cyc_keys))]
    cyc_scores = [1.0] * len(cyc_keys)

    t3 = {}
    for pol in ["LRU", "FIFO", "Belady"]:
        res = run_trace_with_belady(cyc_keys, cyc_passes, cyc_layers, cyc_keys, cyc_scores, C_cyc, pol)
        steady_hits = res["hits"]
        steady_total = res["total"] - C_cyc
        steady_hr = steady_hits / float(steady_total)
        t3[pol] = {"overall_hit_rate": round(res["hit_rate"], 5), "steady_state_hit_rate": round(steady_hr, 5)}
        print(f"  {pol:8s}: overall={res['hit_rate']*100:6.2f}%  steady_state={steady_hr*100:6.2f}% (hits={res['hits']}/{res['total']})")
    assert t3["LRU"]["overall_hit_rate"] == 0.0
    assert t3["FIFO"]["overall_hit_rate"] == 0.0
    assert abs(t3["Belady"]["steady_state_hit_rate"] - (15.0 / 16.0)) < 1e-3
    results["test3_cyclic_sweep"] = t3

    # 4. Belady optimality check on real OLMoE trace slice (Prompt 0, 256 tokens x 16 layers x 8 = 32,768 requests)
    print("\n=== Test 4: Belady Optimality on Real OLMoE Trace Slice (32,768 requests) ===")
    data = np.load(os.path.join(TRACE_DIR, "decode_traces.npz"))
    idx0 = data["topk_indices"][0]   # (256, 16, 8)
    sc0 = data["topk_scores"][0]     # (256, 16, 8)
    real_keys, real_passes, real_layers, real_exps, real_scores = [], [], [], [], []
    for t in range(256):
        for l in range(16):
            for k in range(8):
                e = int(idx0[t, l, k])
                real_keys.append(l * 64 + e)
                real_passes.append(t)
                real_layers.append(l)
                real_exps.append(e)
                real_scores.append(float(sc0[t, l, k]))

    t4 = {}
    for C in [10, 51, 256]:
        t4[C] = {}
        for pol in ["LRU", "LFU", "FIFO", "FLD", "SB", "LS", "Belady"]:
            res = run_trace_with_belady(real_keys, real_passes, real_layers, real_exps, real_scores, C, pol)
            t4[C][pol] = round(res["hit_rate"], 5)
        b_hr = t4[C]["Belady"]
        for pol in ["LRU", "LFU", "FIFO", "FLD", "SB", "LS"]:
            assert b_hr >= t4[C][pol] - 1e-9, f"Belady lost to {pol} at C={C}: {b_hr} < {t4[C][pol]}"
        print(f"  C={C:3d}: " + "  ".join(f"{p}={t4[C][p]*100:5.2f}%" for p in ["LRU", "LFU", "FIFO", "FLD", "SB", "LS", "Belady"]))
    results["test4_belady_real_slice"] = t4

    # 5. Negative controls (must fail the closed-form check |hr - 16/64| < 0.003)
    print("\n=== Test 5: Negative Controls (Deliberately Broken Simulator Variants) ===")
    expected_16 = 16.0 / 64.0
    res_broken_ins = run_trace_with_belady(
        unif_keys[:20000], unif_passes[:20000], unif_layers[:20000], unif_keys[:20000], unif_scores[:20000],
        16, "BrokenInsertFirst"
    )
    err_ins = abs(res_broken_ins["hit_rate"] - expected_16)
    failed_ins = err_ins >= 0.003

    res_broken_cap = run_trace_with_belady(
        unif_keys, unif_passes, unif_layers, unif_keys, unif_scores,
        17, "LRU"  # Off-by-one capacity C+1=17 instead of C=16
    )
    err_cap = abs(res_broken_cap["hit_rate"] - expected_16)
    failed_cap = err_cap >= 0.003

    print(f"  Negative Control A (BrokenInsertFirst at C=16): hit_rate={res_broken_ins['hit_rate']*100:.2f}% vs expected 25.00% -> caught={failed_ins}")
    print(f"  Negative Control B (OffByOneCapacity C+1=17):   hit_rate={res_broken_cap['hit_rate']*100:.2f}% vs expected 25.00% -> caught={failed_cap}")
    assert failed_ins and failed_cap, "Negative controls were not caught!"
    results["test5_negative_controls"] = {
        "BrokenInsertFirst_hit_rate": round(res_broken_ins["hit_rate"], 5),
        "BrokenInsertFirst_caught": failed_ins,
        "OffByOneCapacity_hit_rate": round(res_broken_cap["hit_rate"], 5),
        "OffByOneCapacity_caught": failed_cap,
    }

    with open(os.path.join(TRACE_DIR, "e5_validation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nALL 5 VALIDATION CHECKS PASSED. Saved e5_validation_summary.json")


if __name__ == "__main__":
    main()
