"""NCM rendering vs LFM3 writing, on real step-1 answers: tokens per second, and how much of the text the NCM kept."""
import difflib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import winperf  # noqa: E402

winperf.disable_power_throttling()
import lfm3_clover  # noqa: E402
import lfm3_layers  # noqa: E402

base = HERE.parent
recs = [json.loads(ln) for ln in (base / "verify" / "layers" / "step1.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
picks = [r for r in recs if r.get("passed") and not r["raw_output"].startswith("(") and "```" not in r["raw_output"]]
picks = sorted(picks, key=lambda r: len(r["raw_output"]))[:: max(1, len(picks) // 5)][:5]
t0 = time.perf_counter()
ncm = lfm3_clover.Lfm3NCMRenderer(base / "data", base / "engine")
print(f"NCM loaded in {time.perf_counter() - t0:.1f} s")
for r in picks:
    reply = r["raw_output"]
    n_tok = len(ncm.tok.encode(reply))
    wins = lfm3_layers.answer_windows(reply)
    t1 = time.perf_counter()
    text, _, meta = ncm.render_plans_in_model_vocab(wins, cands=1)
    dt = time.perf_counter() - t1
    kept = difflib.SequenceMatcher(None, reply.split(), text.split()).ratio()
    print(f"{r['task_id']:34s} {n_tok:4d} tokens  {len(wins):2d} windows  NCM {dt:5.1f} s = {n_tok / dt:5.1f} tok/s "
          f"(LFM3 forward inside: {meta['c_engine_ms'] / 1000:.1f} s)  words kept {kept:.0%}")
    print("   in :", reply[:150].replace("\n", " "))
    print("   out:", text[:150].replace("\n", " "))
