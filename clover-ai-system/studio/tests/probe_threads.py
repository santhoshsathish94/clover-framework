"""Decode/prefill speed vs thread count for the fp32+int8 planner; run once per OMP wait policy (env set by caller)."""
import json
import os
import sys
import time
from pathlib import Path

import torch

base = Path(sys.argv[1])
sys.path.insert(0, str(base / "clover"))
import lfm3_agent as A  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

d = base / "data" / "lfm3" / "hf"
tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
model, _ = A.load_planner_backbone(d, int8_ffn=True)
text = ("You are Clover, a helpful assistant. " * 30) + "\nUser message:\nWhat is the capital of France?"
ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
res = {"OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"), "prompt_tokens": int(ids.shape[1]), "by_threads": {}}
with torch.inference_mode():
    for n in (int(x) for x in sys.argv[2].split(",")):
        torch.set_num_threads(n)
        model(input_ids=ids[:, :32], use_cache=False)
        t1 = time.perf_counter()
        out = model(input_ids=ids, use_cache=True, return_dict=True)
        pre = time.perf_counter() - t1
        past, logits = out.past_key_values, out.logits[:, -1, :]
        steps = []
        for _ in range(16):
            nxt = int(logits.argmax(-1))
            t2 = time.perf_counter()
            out = model(input_ids=torch.tensor([[nxt]]), past_key_values=past, use_cache=True, return_dict=True)
            steps.append(time.perf_counter() - t2)
            past, logits = out.past_key_values, out.logits[:, -1, :]
        res["by_threads"][n] = {"prefill_tok_s": round(ids.shape[1] / pre, 1), "decode_s_median": round(sorted(steps)[8], 3)}
        print(n, res["by_threads"][n], flush=True)
print(json.dumps(res))
