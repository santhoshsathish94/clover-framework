"""Measures the phone's fused C1 -> C2 -> C3 runtime (clover-framework s23_on_device_clover.py, laid out by
fused_probe_setup.ps1) on this laptop's CPU, on the natural regression's text-only conversations, with the regression's
own checks: C1 alone (our trained trunks on the C engine), C1 + C2 (HybridLink latent steps), C1 + C2 + C3 (NCM render).
The Studio's C1 (base LFM3-Omni-2B on llama.cpp, GPU) answers the same turns in the regression run it is compared with.
Usage: venv\\Scripts\\python.exe tests\\fused_probe.py [--out FILE.jsonl] [--configs 1,2,3]"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = Path.home() / "CloverAI-build" / "fused-probe"  # outside the install (fused_probe_setup.ps1)
sys.path.insert(0, str(HERE))
import natural_regression as NR  # noqa: E402  (checks)

CHATS = [
    ("heat-pump", [("Can you explain in two sentences what a heat pump does?", [NR.answered(), NR.has_any("heat")])]),
    ("shopping-math", [("I bought 3 notebooks at $4.50 each and a pen for $1.20. How much did I spend?", [NR.has_number(14.70)]),
                       ("And if I add two more notebooks?", [NR.has_number(23.70)]),
                       ("Split that between 3 friends - how much each?", [NR.has_number(7.90)])]),
    ("pens", [("A shop sells pens at 3 for $2. How much do 30 pens cost?", [NR.has_number(20)])]),
    ("percent", [("Hi! What's 15% of 80?", [NR.has_number(12)])]),
    ("photosynthesis", [("Without searching the web, explain what photosynthesis is in three sentences.",
                         [NR.has_any("light", "sun")])]),
    ("long-chat", [("I'm planning a 3-day trip to Jaipur in December with my parents.", [NR.answered()]),
                   ("What about food? We're vegetarian.", [NR.has_any("vegetarian", r"\bveg")]),
                   ("Remind me: how many days is the trip and who is coming with me?",
                    [NR.has_any(r"\b3\b", "three"), NR.has_any("parents")])]),
]
CONFIGS = {1: ("C1: trained trunks on the C engine", 1, False), 2: ("C1 + C2: HybridLink latent steps", 2, False),
           3: ("C1 + C2 + C3: NCM render", 3, True)}


def main():
    for s in (sys.stdout, sys.stderr):  # logs are written in the console's code page (see natural_regression.main)
        s.reconfigure(errors="backslashreplace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "CloverAI" / "verify" / "natural" / f"fused-probe-{int(time.time())}.jsonl"))
    ap.add_argument("--configs", default="1,2,3")
    ap.add_argument("--chats", default="", help="only these chats (comma-separated names)")
    a = ap.parse_args()
    only = {x for x in a.chats.split(",") if x}
    sys.path.insert(0, str(F / "code" / "clover-ai-system" / "mobile"))
    t0 = time.time()
    from s23_on_device_clover import CloverS23OnDeviceSystem
    system = CloverS23OnDeviceSystem(F / "base", load_neural_ncm=True)
    hn = system.hybrid_ncm
    head = {"setup_s": round(time.time() - t0, 1), "engine_threads": getattr(system.engine, "active_threads", None),
            "c2c3_loaded": hn is not None, "precomputed_trunks_active": getattr(hn, "precomputed_trunks_active", None),
            "stage3_hybridlink": getattr(hn, "has_stage3_hybridlink_2048", None)}
    print(json.dumps(head), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    totals = {}
    with open(a.out, "w", encoding="utf-8") as fo:
        fo.write(json.dumps({"head": head}) + "\n")
        for c in [int(x) for x in a.configs.split(",") if x.strip()]:
            label, cap, ncm = CONFIGS[c]
            for name, turns in CHATS:
                if only and name not in only:
                    continue
                hist = []
                for i, (prompt, checks) in enumerate(turns, 1):
                    t1 = time.time()
                    try:
                        res = system.run_cycle(prompt, max_active_layer=cap, use_ncm_renderer=ncm, history=list(hist),
                                               tools=[], allow_internet=False, allow_python=False,
                                               session_id=f"fused-{c}-{name}")
                        err = None
                    except Exception as exc:  # a crash is a result
                        res, err = {"reply": ""}, f"{type(exc).__name__}: {exc}"
                    secs = round(time.time() - t1, 1)
                    reply = str(res.get("reply") or "")
                    got = [(what, bool(ok)) for ok, what in (ck({"reply": reply}) for ck in checks)]
                    passed = not err and all(ok for _, ok in got)
                    totals.setdefault(label, [0, 0, 0.0])
                    totals[label][0] += passed
                    totals[label][1] += 1
                    totals[label][2] += secs
                    trace = res.get("layer_trace") if isinstance(res.get("layer_trace"), dict) else {}
                    l1, l2 = trace.get("layer1_lfm3_omni") or {}, trace.get("layer2_hybrid_agent") or {}
                    rec = {"config": label, "chat": name, "turn": i, "prompt": prompt, "ok": passed, "checks": got,
                           "seconds": secs, "error": err, "reply": reply[:1500],
                           "generated_tokens": res.get("generated_tokens"), "input_tokens": res.get("input_tokens"),
                           "max_active_layer": res.get("max_active_layer"),
                           # where the time goes: C1's prefill and decode, C2's latent step and plan
                           "c1_ms": {k: l1.get(k) for k in ("turn1_prefill_ms", "turn1_decode_ms", "turn1_generated_tokens",
                                                            "turn2_prefill_ms", "turn2_decode_ms", "turn2_generated_tokens")},
                           "c2_ms": {k: l2.get(k) for k in ("hybridlink_ms", "c2_sequence_plan_ms", "c2_sequence_plan_tokens")},
                           "ncm": trace.get("layer3_11level_ncm")}
                    fo.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fo.flush()
                    print(f"[{'PASS' if passed else 'FAIL'}] {label} | {name} {i} ({secs} s, {res.get('generated_tokens')} tok, "
                          f"C1 prefill {l1.get('turn1_prefill_ms')} ms decode {l1.get('turn1_decode_ms')} ms) "
                          f"{reply[:160]!r}" + (f" ERROR {err}" if err else ""), flush=True)
                    hist += [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
    print("\nconfiguration                          passed  turns  mean s")
    for label, (ok, n, s) in totals.items():
        print(f"{label:38} {ok:6} {n:6} {s / max(1, n):7.1f}")
    print("records:", a.out)


if __name__ == "__main__":
    main()
