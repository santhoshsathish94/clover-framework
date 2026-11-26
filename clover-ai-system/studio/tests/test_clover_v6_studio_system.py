"""End-to-end verification of Clover AI Studio v6.0 4-Part System, Computed 0B Tools & Telemetry Ledger."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "clover"))

import clover_v6_system
import k3_host_engine
import local_tool_packs


def test_v6_computed_0b_tools():
    # 1. Exact A^B binary exponentiation in calculate(), including >4,300 digits (2^20000 = 6,021 digits)
    p_str = local_tool_packs.calculate("3 ^ 89")
    assert p_str == str(3 ** 89), p_str
    p_str2 = local_tool_packs.calculate("7 ** 61")
    assert p_str2 == str(7 ** 61), p_str2
    big_pow = clover_v6_system.exact_bigint_power("2", "20000")
    assert big_pow["digit_count"] == 6021 and "leading_64" in big_pow, big_pow

    # 2. 0B Periodogram + Harmonic OLS + Split-Conformal forecaster
    import math
    series = [10.0 + 0.25 * t + 4.0 * math.sin(2.0 * math.pi * t / 12.0) for t in range(48)]
    fc = clover_v6_system.forecast_series_0b(series, horizon=12)
    assert fc["engine"] == "clover-0b-periodogram-conformal"
    assert len(fc["p50"]) == 12
    assert fc["insample_mase"] < 0.05, fc
    for lo, mid, hi in zip(fc["p10"], fc["p50"], fc["p90"]):
        assert lo <= mid <= hi

    # 3. 0B Defeasible Epistemic Rule Resolver (filters rejected, honors unless & specificity)
    rules = [
        {"id": "r_general", "status": "accepted", "conditions": ["prod"], "unless": ["maintenance_window"], "action": "require_2fa"},
        {"id": "r_specific", "status": "accepted", "conditions": ["prod", "break_glass"], "action": "allow_hardware_key"},
        {"id": "r_rejected", "status": "rejected", "conditions": ["prod", "break_glass", "vip"], "action": "skip_all_auth"},
    ]
    res1 = clover_v6_system.resolve_defeasible_rules_0b(rules, ["prod", "break_glass", "vip"])
    assert res1["winner_id"] == "r_specific", res1
    assert "r_rejected" in res1["rejected_filtered"], res1

    res2 = clover_v6_system.resolve_defeasible_rules_0b(rules, ["prod", "maintenance_window"])
    assert res2["winner_id"] is None and "r_general" in res2["blocked_by_unless"], res2


def test_v6_4part_harness_and_telemetry():
    with tempfile.TemporaryDirectory() as tmp:
        v6 = clover_v6_system.CloverStudioV6(tmp)

        # 1. STOP_AND_ASK steering on destructive / underspecified risk
        r_stop = v6.solve_4part("Migrate our auth database and deploy it to production right now.", model="clover-35b-a3b")
        assert r_stop["status"] == "STOPPED_FOR_HUMAN", r_stop
        assert Path(r_stop["context_file"]).is_file()

        # 2. Tier-0 exact math tool routing
        r_math = v6.solve_4part("Compute 3^89 exactly.", model="clover-35b-a3b", generate_fn=lambda p, _c: f"Verified: {p}")
        assert r_math["status"] == "SOLVED", r_math
        assert "xcruncher_exact_math" in r_math["tools_used"], r_math
        assert str(3 ** 89) in r_math["answer"]

        # 3. Tier-0 2B Tool-Exposure Gate on single-function code + Cycle-2 syntax self-repair
        calls = []

        def _fake_2b(prompt_txt: str, cyc: int) -> str:
            calls.append((cyc, prompt_txt))
            if cyc == 1:
                return "```python\ndef add(a, b)\n    return a + b\n```"
            return "```python\ndef add(a, b):\n    return a + b\n```"

        r_code = v6.solve_4part("Write a Python function add(a, b) that returns their sum.", model="omni", generate_fn=_fake_2b, max_cycles=2)
        assert r_code["status"] == "SOLVED" and r_code["cycles_used"] == 2, r_code
        assert "xcruncher_exact_math" not in r_code["tools_used"], r_code

        # 4. Telemetry summary (`used_vs_unused`, `solver_cycles`, `fine_tune_signals`)
        ms = v6.metrics.summary()
        assert "clover-35b-a3b" in ms["used_vs_unused"]["models_used"], ms
        assert "omni" in ms["used_vs_unused"]["models_used"], ms
        assert "xcruncher_exact_math" in ms["used_vs_unused"]["tools_used"], ms
        assert ms["solver_cycles"]["solved_after_retry_count"] >= 1, ms
        assert len(ms["fine_tune_signals"]) >= 1, ms

        # 5. SPECS describe HG4-16R + 4-Part Steerable Harness
        assert "HG4-16R" in k3_host_engine.SPECS["clover-35b-a3b"]["weights"]
        assert any("4-Part" in x for x in k3_host_engine.SPECS["clover-35b-a3b"]["layers"])

        # 6. Security: SSH option injection & invalid targets raise ValueError
        for bad_target in ("ssh://-oProxyCommand=calc@example.com", "-oProxyCommand=calc@10.0.0.1", "user@gmail.com"):
            try:
                clover_v6_system.parse_target(bad_target)
                raise AssertionError(f"Expected ValueError for {bad_target}")
            except ValueError:
                pass

        # 7. CPU-only / Phone / Laptop rule:
        #    Small models run interactively; Big GPU models are skipped stating machine cannot run interactively!
        cpu_plan = v6.plan_system_setup(
            selected_models=["all"],
            servers=[],
            include_local=True,
            hardware_hint={"ram_gb": 16, "gpus": []},
            verify_live_completion=False,
        )
        assert set(cpu_plan["ready_models"]) == {"0b-computed-pack", "omni", "chronos-2", "vjepa-2.1"}, cpu_plan
        assert set(cpu_plan["skipped_models"]) == {
            "boltz-2", "goedel-prover-v2-32b", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"
        }, cpu_plan
        for p in cpu_plan["placements"]:
            if p["tier"] == "big_gpu":
                assert p["status"] == "skipped_not_interactive" and "cannot run" in p["reason"], p

        # 8. Single 24 GB GPU triple-model case (`boltz-2` + `goedel-32b` + `clover-35b-a3b` = 37.3 GB weights):
        #    Must NOT pin all three into 24 GB! Highest-resident gets `ready_pinned`, others get `ready_swap`.
        one_gpu_plan = v6.plan_system_setup(
            selected_models=["boltz-2", "goedel-prover-v2-32b", "clover-35b-a3b"],
            servers=[{"target": "http://10.0.0.10:8790", "gpus": [{"name": "RTX 3090", "vram_gb": 24.0}], "reachable": True}],
            include_local=False,
            verify_live_completion=False,
        )
        by_id = {p["model_id"]: p["status"] for p in one_gpu_plan["placements"]}
        assert by_id["goedel-prover-v2-32b"] == "ready_pinned", by_id
        assert by_id["clover-35b-a3b"] == "ready_swap", by_id
        assert by_id["boltz-2"] == "ready_swap", by_id

        # 9. Multi-GPU 3-machine setup (`local CPU` + `Server #1 4x24GB` + `Server #2 4x24GB`):
        #    Pins Nemotron 120B (TP=4 across 4x24GB) + Clover-35B + Goedel-32B + Boltz-2, skips ONLY Kimi 2.8T!
        applied_3m = v6.apply_system_setup(
            selected_models=["all"],
            servers=[
                {"target": "http://175.155.64.158:19529", "gpus": [{"name": "RTX 3090", "vram_gb": 24.0}] * 4, "reachable": True, "api_key": "k1"},
                {"target": "http://175.155.64.155:19497", "gpus": [{"name": "RTX 3090", "vram_gb": 24.0}] * 4, "reachable": True, "api_key": "k2"},
            ],
            include_local=True,
            hardware_hint={"ram_gb": 32, "gpus": []},
        )
        assert applied_3m["skipped_models"] == ["kimi-k3"], applied_3m["skipped_models"]
        chat_ids = [e["id"] for e in applied_3m["chat_entries"]]
        assert any("clover-35b-a3b" in cid for cid in chat_ids), chat_ids
        assert any("nemotron-3.5-super-vl" in cid for cid in chat_ids), chat_ids
        assert "goedel_lean4_prove" in applied_3m["mcp_routes"], applied_3m["mcp_routes"]
        assert "boltz2_fold_structure" in applied_3m["mcp_routes"], applied_3m["mcp_routes"]

        # 10. SSH host without HTTP gateway URL gets `needs_gateway` + safe `$CLOVER_GATEWAY_KEY` bootstrap (no secret leak)
        ssh_plan = v6.plan_system_setup(
            selected_models=["clover-35b-a3b"],
            servers=[{"target": "ssh://root@10.0.0.99:22", "gpus": [{"name": "RTX 3090", "vram_gb": 24.0}], "reachable": True, "api_key": "TOPSECRET123"}],
            include_local=False,
            api_key="TOPSECRET123",
            verify_live_completion=False,
        )
        assert ssh_plan["placements"][0]["status"] == "needs_gateway", ssh_plan["placements"]
        boot_cmd = clover_v6_system._build_bootstrap_cmd("root", "10.0.0.99", 22, "TOPSECRET123")
        assert "TOPSECRET123" not in boot_cmd and "CLOVER_GATEWAY_KEY" in boot_cmd, boot_cmd

        # 11. Removing a v6 setup placement updates `clover_v6_model_setup.json` to `removed_by_user`
        ent_35b = next(e for e in applied_3m["chat_entries"] if "clover-35b-a3b" in e["id"])
        v6.mark_placement_removed("clover-35b-a3b", ent_35b["url"])
        reloaded = v6.load_saved_setup()
        assert "clover-35b-a3b" not in reloaded["ready_models"], reloaded["ready_models"]
        v6.close()


if __name__ == "__main__":
    test_v6_computed_0b_tools()
    test_v6_4part_harness_and_telemetry()
    print("ALL CLOVER STUDIO V6 4-PART & TELEMETRY TESTS PASSED!")
