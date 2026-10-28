import os
import json
import numpy as np

TRACE_DIR = "/root/dm-study/specmd/w2/traces"


def gini_coeff(counts):
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1.0) / n)


def compute_stack_distances(stream):
    # stream: list/array of integer item IDs
    # returns list of stack distances (0-indexed: 0 means MRU, i.e., 1 distinct item in stack)
    # and cold_misses count
    stack = []
    pos_map = {}
    # Use an order-statistic / timestamp dict for fast exact stack distance (N <= 1024)
    last_seen = {}
    dists = []
    cold = 0
    for step, item in enumerate(stream):
        it = int(item)
        if it not in last_seen:
            cold += 1
        else:
            prev_step = last_seen[it]
            # count how many distinct items have last_seen > prev_step
            d = sum(1 for ts in last_seen.values() if ts > prev_step)
            dists.append(d)
        last_seen[it] = step
    return np.array(dists, dtype=np.int32), cold, len(stream)


def summarize_dists(dists, cold, total, label):
    hits_at = {}
    for cap in [8, 10, 16, 32, 51, 64, 120, 128, 256, 512]:
        # An access is an LRU hit at capacity `cap` iff it is not cold and its stack distance < cap
        h = int(np.sum(dists < cap))
        hits_at[cap] = round(h / float(total), 6)
    qs = np.percentile(dists, [0, 10, 25, 50, 75, 90, 100]).tolist() if dists.size > 0 else [0]*7
    print(f"=== Stack Reuse Distance: {label} (total={total}, reuses={dists.size}, cold={cold}) ===")
    print(f"  Quantiles [min, p10, p25, p50, p75, p90, max]: {[round(q, 1) for q in qs]}")
    print(f"  Fraction of all requests with stack_dist < C (exact LRU hit rate at capacity C):")
    print("   ", "  ".join(f"C={c}:{hits_at[c]*100:.2f}%" for c in [8, 10, 51, 120, 128, 256, 512]))
    return {
        "total_requests": total,
        "reuses": int(dists.size),
        "cold_misses": cold,
        "quantiles_p0_10_25_50_75_90_100": [round(q, 2) for q in qs],
        "lru_hit_rate_at_cap": hits_at,
    }


def main():
    data = np.load(os.path.join(TRACE_DIR, "prompt_traces.npz"))
    idx = data["topk_indices"]  # (P=8, T=256, L=16, K=8)
    P, T, L, K = idx.shape
    E = 64

    # 1. Window distinct experts for W in {1, 8, 64}
    print("=== 1. Distinct Experts Touched in Window W in {1, 8, 64} ===")
    win_stats = {}
    for W in [1, 8, 64]:
        per_layer_counts = []
        global_counts = []
        for p in range(P):
            for start in range(0, T - W + 1, W):
                window = idx[p, start:start+W]  # (W, L, K)
                tot_g = 0
                for l in range(L):
                    c = len(np.unique(window[:, l, :]))
                    per_layer_counts.append(c)
                    tot_g += c
                global_counts.append(tot_g)
        mean_l = float(np.mean(per_layer_counts))
        mean_g = float(np.mean(global_counts))
        unif_l = E * (1.0 - ((E - K) / float(E)) ** W)
        unif_g = L * unif_l
        win_stats[W] = {
            "per_layer_mean": round(mean_l, 3),
            "per_layer_pct": round(mean_l / E * 100, 2),
            "global_mean": round(mean_g, 2),
            "global_pct": round(mean_g / (L * E) * 100, 2),
            "iid_uniform_per_layer": round(unif_l, 3),
            "iid_uniform_global": round(unif_g, 2),
        }
        print(f"  W={W:2d}: per-layer={mean_l:5.2f}/64 ({mean_l/64*100:5.2f}%) [iid_unif={unif_l:5.2f}] | "
              f"global={mean_g:6.2f}/1024 ({mean_g/1024*100:5.2f}%) [iid_unif={unif_g:6.2f}]")

    # 2. Consecutive-token overlap in the same layer (and lags 1, 2, 4, 8, 16)
    print("\n=== 2. Same-Layer Token-to-Token Expert Overlap ===")
    lag_overlaps = {}
    for lag in [1, 2, 4, 8, 16]:
        shared = []
        for p in range(P):
            for t in range(lag, T):
                for l in range(L):
                    s_cur = set(idx[p, t, l].tolist())
                    s_prev = set(idx[p, t - lag, l].tolist())
                    shared.append(len(s_cur.intersection(s_prev)) / float(K))
        lag_overlaps[lag] = round(float(np.mean(shared)), 5)
        print(f"  Lag dt={lag:2d}: overlap={lag_overlaps[lag]*100:5.2f}% ({lag_overlaps[lag]*8:.3f}/8 experts)")

    # Shuffled-token control per layer (isolates popularity skew from temporal correlation)
    rng = np.random.default_rng(42)
    shuf_per_layer = []
    lag1_per_layer = {}
    for l in range(L):
        s_list = []
        for p in range(P):
            for t in range(1, T):
                s_cur = set(idx[p, t, l].tolist())
                s_prev = set(idx[p, t - 1, l].tolist())
                s_list.append(len(s_cur.intersection(s_prev)) / float(K))
        lag1_per_layer[l] = round(float(np.mean(s_list)), 5)
        # Shuffled across all P*T tokens in layer l
        flat_l = idx[:, :, l, :].reshape(-1, K).copy()
        rng.shuffle(flat_l, axis=0)
        sh_l = [len(set(flat_l[i].tolist()).intersection(set(flat_l[i-1].tolist()))) / float(K)
                for i in range(1, flat_l.shape[0])]
        shuf_per_layer.append(float(np.mean(sh_l)))
    shuf_mean = float(np.mean(shuf_per_layer))
    print(f"  Shuffled-token control (popularity skew only, no temporal locality): {shuf_mean*100:.2f}% ({shuf_mean*8:.3f}/8)")
    print(f"  IID uniform baseline (8/64): 12.50% (1.000/8)")
    print("  Per-layer lag-1 overlap:", " ".join(f"L{l}:{lag1_per_layer[l]*100:.1f}%" for l in range(L)))

    # 3. Popularity skew (Gini, top shares)
    print("\n=== 3. Expert Popularity Skew ===")
    layer_ginis = []
    all_counts = np.zeros((L, E), dtype=np.int64)
    for l in range(L):
        vals, cnts = np.unique(idx[:, :, l, :], return_counts=True)
        all_counts[l, vals] = cnts
        layer_ginis.append(round(gini_coeff(all_counts[l]), 4))
    flat_counts = np.sort(all_counts.reshape(-1))[::-1]
    total_act = int(flat_counts.sum())
    top_shares = {
        "top_1pct_10exp": round(float(flat_counts[:10].sum() / total_act), 5),
        "top_5pct_51exp": round(float(flat_counts[:51].sum() / total_act), 5),
        "top_12.5pct_128exp": round(float(flat_counts[:128].sum() / total_act), 5),
        "top_25pct_256exp": round(float(flat_counts[:256].sum() / total_act), 5),
    }
    print(f"  Global Gini across 1,024 experts: {gini_coeff(flat_counts):.4f} (mean per-layer Gini: {np.mean(layer_ginis):.4f})")
    print("  Per-layer Gini:", " ".join(f"L{l}:{layer_ginis[l]:.3f}" for l in range(L)))
    print(f"  Active experts seen at least once: {int(np.sum(flat_counts > 0))}/1024 (min count={flat_counts.min()}, max={flat_counts.max()}, mean={flat_counts.mean():.1f})")
    print("  Share of total activations by top experts:", {k: f"{v*100:.2f}%" for k, v in top_shares.items()})

    # 4. Reuse distance distributions in 3 stream orders
    print("\n=== 4. Reuse Distance Distributions ===")
    # (A) Within-layer stream (averaged across all 16 layers)
    within_dists = []
    within_cold = 0
    within_total = 0
    for l in range(L):
        s_l = idx[:, :, l, :].reshape(-1)
        d_l, c_l, tot_l = compute_stack_distances(s_l)
        within_dists.append(d_l)
        within_cold += c_l
        within_total += tot_l
    within_dists = np.concatenate(within_dists)
    st_within = summarize_dists(within_dists, within_cold, within_total, "Within-Layer Stream (per-layer token order, max dist 63)")

    # (B) Decode-style global stream: for p, for t, for l, for k -> global ID = l*64 + e
    decode_stream = []
    for p in range(P):
        for t in range(T):
            for l in range(L):
                for k in range(K):
                    decode_stream.append(l * E + int(idx[p, t, l, k]))
    d_dec, c_dec, tot_dec = compute_stack_distances(decode_stream)
    st_decode = summarize_dists(d_dec, c_dec, tot_dec, "Decode-Style Global Stream (token-major across 16 layers)")

    # (C) Prompt-style global stream: for p (256 tokens), for l=0..15, for t=0..255, for k -> global ID = l*64 + e
    prompt_stream = []
    for p in range(P):
        for l in range(L):
            for t in range(T):
                for k in range(K):
                    prompt_stream.append(l * E + int(idx[p, t, l, k]))
    d_prm, c_prm, tot_prm = compute_stack_distances(prompt_stream)
    st_prompt = summarize_dists(d_prm, c_prm, tot_prm, "Prompt-Style Global Stream (layer-major per 256-tok prompt)")

    summary = {
        "window_distinct": win_stats,
        "same_layer_lag_overlaps": lag_overlaps,
        "same_layer_lag1_per_layer": lag1_per_layer,
        "shuffled_token_control_overlap": round(shuf_mean, 5),
        "iid_uniform_overlap": 0.125,
        "popularity_skew": {
            "global_gini": round(gini_coeff(flat_counts), 4),
            "mean_layer_gini": round(float(np.mean(layer_ginis)), 4),
            "layer_ginis": layer_ginis,
            "min_count": int(flat_counts.min()),
            "max_count": int(flat_counts.max()),
            "top_shares": top_shares,
        },
        "reuse_distance": {
            "within_layer": st_within,
            "decode_global": st_decode,
            "prompt_global": st_prompt,
        },
    }
    with open(os.path.join(TRACE_DIR, "e4_access_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved e4_access_summary.json")


if __name__ == "__main__":
    main()
