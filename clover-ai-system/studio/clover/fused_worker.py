"""The phone's fused C1 -> C2 -> C3 runtime (clover-framework clover-ai-system/mobile/s23_on_device_clover.py) in its
own process, for the Studio's fused engine (fused_engine.py). Its own process because the phone runtime brings its own
versions of modules the Studio also has (clover_cycle, web_search, local_terminal, ...).
Protocol: one JSON request per line on stdin, one JSON answer per line on stdout; the first line out is {"ready": ...}.
Usage: python fused_worker.py <framework mobile folder> <runtime base folder>"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    code, base = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(code))
    # answers go to a copy of stdout; stdout itself (Python prints and the C engine's printf) goes to the log
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):  # the runtime prints while it loads; stdout carries only answers
        from s23_on_device_clover import CloverS23OnDeviceSystem
        system = CloverS23OnDeviceSystem(base, load_neural_ncm=True)
    hn = system.hybrid_ncm
    out.write(json.dumps({"ready": True, "setup_s": round(time.time() - t0, 1),
                          "threads": getattr(system.engine, "active_threads", None), "c2c3": hn is not None,
                          "stage3_hybridlink": getattr(hn, "has_stage3_hybridlink_2048", None)}) + "\n")
    out.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            t1 = time.time()
            with contextlib.redirect_stdout(sys.stderr):
                res = system.run_cycle(str(req.get("prompt") or ""), max_active_layer=int(req.get("layer", 3)),
                                       use_ncm_renderer=bool(req.get("ncm", True)), history=req.get("history") or [],
                                       tools=[], allow_internet=False, allow_python=False,
                                       max_new_tokens=int(req.get("max_new", 256)),
                                       session_id=str(req.get("sid") or "studio"))
            trace = res.get("layer_trace") if isinstance(res.get("layer_trace"), dict) else {}
            ans = {"reply": str(res.get("reply") or ""), "generated_tokens": res.get("generated_tokens"),
                   "input_tokens": res.get("input_tokens"), "seconds": round(time.time() - t1, 2),
                   "max_active_layer": res.get("max_active_layer"),
                   # run_cycle writes C3's record as layer3_11level_ncm (the first regression read layer3_ncm: null)
                   "c2": trace.get("layer2_hybrid_agent"), "c3": trace.get("layer3_11level_ncm"),
                   "c1": {k: (trace.get("layer1_lfm3_omni") or {}).get(k) for k in ("turn1_prefill_ms", "turn1_decode_ms",
                                                                                 "turn1_generated_tokens")}}
        except Exception as exc:  # noqa: BLE001 - reported to the Studio, the worker keeps serving
            ans = {"error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(ans, default=str) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
