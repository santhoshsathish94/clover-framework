"""Which assistant opening lets C1 call a tool listed in the `<tools>` system block? Replays the exact turn-1 prompts
the after-A run gave C1 for the file and delete messages (from the trace), with the opening swapped.
Usage: python c1_tool_open.py <clover-ai-system> <fw-base> <framework.jsonl>"""
import json
import sys
from pathlib import Path

root, base, trace = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / "mobile"))
import s23_on_device_clover as F  # noqa: E402

OPEN = "<|im_start|>assistant\n"
prompts = {}
for line in open(trace, encoding="utf-8"):
    e = json.loads(line)
    if e["phase"] == "call" and e["fn"].endswith("prefill_cached"):
        t = ((e.get("args") or {}).get("prompt_ids") or {}).get("text")
        if isinstance(t, str) and "<tools>" in t and OPEN in t:
            last_user = t.rsplit("<|im_start|>user\n", 1)[-1]  # recall blocks quote earlier turns: the message itself
            key = "file" if "meeting-notes" in last_user else "delete" if "old report" in last_user else None
            if key:
                prompts.setdefault(key, t[: t.rindex(OPEN)])
assert set(prompts) == {"file", "delete"}, list(prompts)

tok = F.Lfm3PureBpeTokenizer(base / "tokenizer.json")
eng = F.S23Lfm3NativeEngine(base / "pruned", pin_s23_big_cores=True, use_4bit_ffn=True, madvise_active=True)
eng.configure_stop_mask(tok)
variants = [("answer (current)", "<think></think><answer>", False), ("empty think", "<think></think>", False),
            ("bare", "", False), ("think open", "<think>\n", True)]
for name, p in prompts.items():
    for label, opening, think in variants:
        ids = tok.encode(p + OPEN + opening, allow_special=True)
        h, ms = eng.prefill_cached(ids)
        gen = eng.decode_from_cache_with_hidden(h, 160, think)
        out = tok.decode(list(gen[0]), skip_special_tokens=False)
        call = F.CloverS23OnDeviceSystem._parse_lfm3_tool_call_xml(out)
        print(f"{name:6} | {label:16} | call={call and (call['function'], call['parameters'])} | {out[:230]!r}",
              flush=True)
