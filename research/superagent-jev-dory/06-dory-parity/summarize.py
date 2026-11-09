import json
from pathlib import Path

r = json.loads(Path("/opt/clover-k3/superagent-jev-dory-study/dory-parity/results.json").read_text())
print("consistent_with_reference", r["consistent_with_reference"], "forward_copy_max_abs_diff", r["forward_copy_max_abs_diff"])
print("pr_head", r["pr_head"])
for k, v in r["per_k"].items():
    for part in ("primary", "secondary", "control_std_fp32_vs_bf16", "control_ref_f32_vs_bf16"):
        s = v[part]["summary"]
        print(f"K={k} {part:28} {s['verdict']:12} misses {s['top1_misses']}/{s['n']} median_kl {s['median_kl']:.2e} p90 {s['p90_kl']:.2e} max {s['max_kl']:.2e}")
    misses = [row["id"] for row in v["primary"]["rows"] if not row["top1_agree"]]
    print("   primary top1 misses:", misses, [round(row["dlogp_ref_top"], 4) for row in v["primary"]["rows"] if not row["top1_agree"]])
for name, s in r["mutants"].items():
    print(f"{name:24} {s['verdict']:12} misses {s['top1_misses']}/{s['n']} median_kl {s['median_kl']:.3f} p90 {s['p90_kl']:.3f} max {s['max_kl']:.3f}")
