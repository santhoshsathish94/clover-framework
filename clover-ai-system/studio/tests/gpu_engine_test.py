"""The Iris Xe engine (llama.cpp Vulkan, QAD Q4_0) inside LFM3Native: raw stream fields, a tool turn, determinism."""
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import winperf  # noqa: E402

winperf.disable_power_throttling()
import llamacpp_engine as L  # noqa: E402
import lfm3_native as N  # noqa: E402
import permissions as P  # noqa: E402

perms = P.PermissionStore(Path(tempfile.mkdtemp()) / "perms.json")
perms.update({"code": "allow", "internet": "deny", "files": "deny", "shell": "deny", "edit_files": "deny"})
t0 = time.perf_counter()
eng = N.LFM3Native(HERE.parent, perms, engine="gpu")
print(f"loaded in {time.perf_counter() - t0:.1f} s:", eng.describe())

ids = eng.tok("<|startoftext|><|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).input_ids
evs = list(eng.gpu.stream(ids, L.sampling_params(N.SAMPLING["greedy"], 24, 0)))
print("events:", len(evs), "first tokens:", evs[0][0], "final keys:", sorted((evs[-1][1] or {}).keys())[:20])
print("final stop_type:", (evs[-1][1] or {}).get("stop_type"), "timings:", (evs[-1][1] or {}).get("timings"))

for q, tools in (("Use Python to compute 17 * 23 and tell me the result.", True), ("What is the capital of Japan? One word.", False)):
    t1 = time.perf_counter()
    res = eng.run_turn([], q, tools=tools, sampling="greedy", max_new=1024, final_on_limit=True, force_answer=True)
    print(f"\nQ: {q}\n  {time.perf_counter() - t1:.1f} s tools={[c['tool'] for c in res['tool_calls']]} reply={res['reply'][:200]!r}")
    print("  generation:", [(g["prompt_tokens"], g["reused_tokens"], g["new_tokens"], g["prefill_s"], g["decode_s"], g["finish"]) for g in res["generation"]])

a = eng.run_turn([], "List three primary colors.", tools=False, sampling="greedy", max_new=512)["reply"]
b = eng.run_turn([], "List three primary colors.", tools=False, sampling="greedy", max_new=512)["reply"]
print("\ndeterministic (same greedy prompt twice):", a == b)
eng.close()
