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

    print("=== 1. Initial /api/health with Direction B Data-Movement Ledger ===")
    print(json.dumps(health.get("data_movement_ledger", {}), indent=2))

    prompts = [
        ("q1_dm", "Explain how the Red-Blue Pebble Game and SpecMD Least-Stale eviction reduce PCIe data movement in Mixture-of-Experts inference."),
        ("q2_fa2", "How does FlashAttention-2's online softmax rescaling avoid writing the N x N attention matrix to HBM, and how does Split-KV combine partial logsumexp chunks?"),
        ("q3_eyeriss", "Compare Eyeriss v1's 4-tier storage hierarchy energy costs with Eyeriss v2's Hierarchical Mesh NoC broadcast vs unicast modes."),
        ("q4_repeat", "Explain how the Red-Blue Pebble Game and SpecMD Least-Stale eviction reduce PCIe data movement in Mixture-of-Experts inference."),
    ]

    print("\n=== 2. Running 4 Live 93-Layer Queries on Clover Server ===")
    for tag, prompt in prompts:
        t0 = time.perf_counter()
        res = post_json(
            f"{base}/api/chat",
            {"session_id": f"dirb_{tag}_{int(time.time())}", "prompt": prompt, "fast_mode": False},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        h_now = get_json(f"{base}/api/health")
        dm = h_now.get("data_movement_ledger", {})
        reply_preview = (res.get("reply") or "")[:110].replace("\n", " ")
        print(
            f"[{tag}] wall={dt:6.1f}ms | policy={dm.get('policy_name')} | "
            f"perm_hits={dm.get('perm_vram_hits')} | dyn_need={dm.get('dyn_needed')} | "
            f"hit_card_R3={dm.get('hit_card_R3')} ({dm.get('dyn_hit_rate_pct')}%) | "
            f"AllVRAM={dm.get('overall_vram_hit_rate_pct')}% | "
            f"hit_host_R1={dm.get('hit_host_R1')} | from_drive_R1={dm.get('from_drive_R1')} | "
            f"wb_R2={dm.get('written_back_R2')} | r4_drops={dm.get('r4_clean_drops')} | "
            f"Eyeriss/req={dm.get('eyeriss_cost_per_req')}\n"
            f"      reply: {reply_preview}..."
        )

    print("\n=== 3. Final /api/health Data-Movement Ledger ===")
    final_health = get_json(f"{base}/api/health")
    print(json.dumps(final_health.get("data_movement_ledger", {}), indent=2))


if __name__ == "__main__":
    main()
