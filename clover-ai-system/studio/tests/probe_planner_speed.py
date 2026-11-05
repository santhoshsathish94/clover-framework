"""Planner speed probe on the laptop: bf16+int8-FFN vs Clover's fp32+int8-FFN loader; prefill + greedy KV decode."""
import json
import sys
import time
from pathlib import Path

import torch

base = Path(sys.argv[1])
mode = sys.argv[2]
sys.path.insert(0, str(base / "clover"))
import lfm3_agent as A  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

d = base / "data" / "lfm3" / "hf"
tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
t0 = time.perf_counter()
if mode == "fp32_int8":
    model, nq = A.load_planner_backbone(d, int8_ffn=True)
elif mode == "bf16":
    model, nq = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, trust_remote_code=True).eval(), 0
else:
    raise SystemExit(f"unknown mode {mode}")
load_s = time.perf_counter() - t0

text = ("You are Clover, a helpful assistant. " * 60) + "\nUser message:\nWhat is the capital of France and why is it famous?"
ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
rows = {"mode": mode, "load_s": round(load_s, 1), "int8_linears": nq, "threads": torch.get_num_threads(), "prompt_tokens": int(ids.shape[1])}
with torch.inference_mode():
    t1 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True, output_hidden_states=True, return_dict=True)
    rows["prefill_s"] = round(time.perf_counter() - t1, 2)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    steps = []
    gen = []
    for _ in range(32):
        nxt = int(logits.argmax(-1))
        gen.append(nxt)
        t2 = time.perf_counter()
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True, return_dict=True)
        steps.append(time.perf_counter() - t2)
        past, logits = out.past_key_values, out.logits[:, -1, :]
rows["decode_s_per_token_median"] = round(sorted(steps)[len(steps) // 2], 3)
rows["decode_s_first"] = round(steps[0], 3)
rows["text"] = tok.decode(gen)
try:
    import psutil
    rows["rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
except ImportError:
    pass
print(json.dumps(rows, ensure_ascii=False))
