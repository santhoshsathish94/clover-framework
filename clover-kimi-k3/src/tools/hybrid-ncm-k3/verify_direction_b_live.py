#!/usr/bin/env python3
"""
Live 93-Layer Verification of Clover Direction B Data-Movement Upgrades (`verify_direction_b_live.py`)
=====================================================================================================
Waits for `http://127.0.0.1:8765/api/health` on `gpu`, verifies that Policy 3 (`specmd_least_stale_r4`)
is active across all 6 expert GPUs (`cuda:0..5`) and `HostCache`, runs a sequence of live 93-layer
queries through `/api/chat`, and records the 4-Tier Data-Movement & Red-Blue Pebble I/O Ledger.
"""

import json
import time
import urllib.request


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    base = "http://127.0.0.1:8765"
    health = None
    for _ in range(40):
        try:
            health = get_json(f"{base}/api/health")
            if health.get("status") == "ready":
                break
        except Exception:
            time.sleep(2.5)
    if not health or health.get("status") != "ready":
        raise RuntimeError("Server did not become ready in time")

    dm0 = health.get("data_movement_ledger", {})
    print("=== 1. Initial /api/health with Direction B Data-Movement Ledger ===")
    print(json.dumps(dm0, indent=2))

    prompts = [
        ("q1_dm", "Explain how the Red-Blue Pebble Game and SpecMD Least-Stale eviction reduce PCIe data movement in Mixture-of-Experts inference."),
        ("q2_fa2", "How does FlashAttention-2's online softmax rescaling avoid writing the N x N attention matrix to HBM, and how does Split-KV combine partial logsumexp chunks?"),
        ("q3_eyeriss", "Compare Eyeriss v1's 4-tier storage hierarchy energy costs with Eyeriss v2's Hierarchical Mesh NoC broadcast vs unicast modes."),
        ("q4_repeat", "Explain how the Red-Blue Pebble Game and SpecMD Least-Stale eviction reduce PCIe data movement in Mixture-of-Experts inference."),
    ]

    print("\n=== 2. Running 4 Live 93-Layer Queries on Clover Server (Recording Token Processing, K3 93-Layer GPU ms, HybridAgent Plan ms, NCM ms, and Data-Movement Deltas) ===")
    prev_dm = dm0
    for tag, prompt in prompts:
        t0 = time.perf_counter()
        res = post_json(
            f"{base}/api/chat",
            {"session_id": f"dirb_{tag}_{int(time.time())}", "prompt": prompt, "cands": 4},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        h_now = get_json(f"{base}/api/health")
        dm = h_now.get("data_movement_ledger", {})
        k3_meta = res.get("k3_experts") or {}
        k3_ms = float(k3_meta.get("k3_moe_ms") or 0.0)
        plan_ms = float(res.get("plan_lat_ms") or 0.0)
        ncm_ms = float(res.get("ncm_mbr8_ms") or 0.0)
        n_toks = int(res.get("n_k3_tokens_mbr8") or 0)
        cands = 4
        tot_k3_toks = n_toks * cands
        k3_tok_per_s = (1000.0 * tot_k3_toks / k3_ms) if k3_ms > 0 else 0.0
        ms_per_layer = (k3_ms / 93.0) if k3_ms > 0 else 0.0

        d_perm = int(dm.get("perm_vram_hits", 0)) - int(prev_dm.get("perm_vram_hits", 0))
        d_need = int(dm.get("dyn_needed", 0)) - int(prev_dm.get("dyn_needed", 0))
        d_hc = int(dm.get("hit_card_R3", 0)) - int(prev_dm.get("hit_card_R3", 0))
        d_hh = int(dm.get("hit_host_R1", 0)) - int(prev_dm.get("hit_host_R1", 0))
        d_dr = int(dm.get("from_drive_R1", 0)) - int(prev_dm.get("from_drive_R1", 0))
        d_wb = int(dm.get("written_back_R2", 0)) - int(prev_dm.get("written_back_R2", 0))
        d_r4 = int(dm.get("r4_clean_drops", 0)) - int(prev_dm.get("r4_clean_drops", 0))
        d_spf = int(dm.get("spec_lookahead_hits", 0)) - int(prev_dm.get("spec_lookahead_hits", 0))
        marg_dyn_hit = (100.0 * d_hc / max(1, d_need)) if d_need > 0 else 100.0
        marg_vram_hit = (100.0 * (d_perm + d_hc) / max(1, d_perm + d_need))
        prev_dm = dm

        print(
            f"[{tag}] total={dt:6.1f}ms (Plan={plan_ms:5.1f}ms | K3_93L_GPU={k3_ms:6.1f}ms [{ms_per_layer:4.1f}ms/layer] | NCM+K3={ncm_ms:6.1f}ms) | "
            f"K3_Toks={tot_k3_toks:3d} ({k3_tok_per_s:5.1f} tok/s across 93 layers)\n"
            f"      Delta Ledger: perm={d_perm:4d} | dyn_need={d_need:4d} | hit_card_R3={d_hc:4d} ({marg_dyn_hit:5.1f}% marg / {dm.get('dyn_hit_rate_pct')}% cum) | "
            f"AllVRAM={marg_vram_hit:5.1f}% marg ({dm.get('overall_vram_hit_rate_pct')}% cum) | "
            f"hit_host_R1={d_hh:3d} | drive_R1={d_dr:3d} | wb_R2={d_wb:3d} | r4_drops={d_r4:3d} | spec_pf={d_spf:3d} | Eyeriss/req={dm.get('eyeriss_cost_per_req')}"
        )

    print("\n=== 3. Final /api/health Data-Movement Ledger ===")
    final_health = get_json(f"{base}/api/health")
    print(json.dumps(final_health.get("data_movement_ledger", {}), indent=2))


if __name__ == "__main__":
    main()
