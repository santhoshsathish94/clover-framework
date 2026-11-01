"""Does calling one C1 engine from two threads change its output? The framework runtime's turn and its S2 note writer
both call the same S23Lfm3NativeEngine. Two prompts run alone (twice each, to see whether the output repeats), then
both at once from two threads, three times. Prompts are the ones recorded in the observed overlapping turn.
Usage: python c1_concurrency.py <clover-ai-system> <fw-base> <framework.jsonl>"""
import json
import sys
import threading
import time
from pathlib import Path

root, base, trace = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / "mobile"))
import s23_on_device_clover as F  # noqa: E402

prompts = {}
for line in open(trace, encoding="utf-8"):
    e = json.loads(line)
    if e["phase"] == "call" and e["fn"].endswith("prefill_cached"):
        t = ((e.get("args") or {}).get("prompt_ids") or {}).get("text")
        if isinstance(t, str) and e["turn"] == 4 and e["thread"].startswith("Thread"):
            prompts.setdefault("answer", t)
        if isinstance(t, str) and e["thread"] == "s2-memory-index":
            prompts.setdefault("note", t)
assert set(prompts) == {"answer", "note"}, f"prompts not found in trace: {list(prompts)}"

tok = F.Lfm3PureBpeTokenizer(base / "tokenizer.json")
eng = F.S23Lfm3NativeEngine(base / "pruned", pin_s23_big_cores=True, use_4bit_ffn=True, madvise_active=True)
eng.configure_stop_mask(tok)
ids = {k: tok.encode(v, allow_special=True) for k, v in prompts.items()}
print("sys.path head:", sys.path[:4], flush=True)
print({k: len(v) for k, v in ids.items()}, "tokens", flush=True)


def run(name, out):
    t0 = time.perf_counter()
    h, _ = eng.prefill_cached(ids[name])
    gen = eng.decode_from_cache_with_hidden(h, 48, False)
    out[name] = (tok.decode(list(gen[0]), skip_special_tokens=False), round(time.perf_counter() - t0, 1))


alone = []
for rep in range(2):
    for name in ("answer", "note"):
        o = {}
        run(name, o)
        alone.append((name, rep, o[name]))
        print(f"ALONE   {name} #{rep}: {o[name][1]:6.1f} s  {o[name][0][:110]!r}", flush=True)
same = {n: len({a[2][0] for a in alone if a[0] == n}) == 1 for n in ("answer", "note")}
print("alone runs repeat exactly:", same, flush=True)
base_text = {n: next(a[2][0] for a in alone if a[0] == n) for n in ("answer", "note")}
for rep in range(3):
    o = {}
    ts = [threading.Thread(target=run, args=(n, o)) for n in ("answer", "note")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for n in ("answer", "note"):
        print(f"TOGETHER {n} #{rep}: {o[n][1]:6.1f} s  same as alone: {o[n][0] == base_text[n]}  {o[n][0][:110]!r}",
              flush=True)
