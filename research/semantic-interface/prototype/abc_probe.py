"""Probe for Experiment 8: what LFM3-Omni-2B S3 writes after its own generation prompt (greedy), per condition, so the
answer position is taken from the model's own output rather than assumed. Prints the continuation token by token."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi.minimal_pairs import generate  # noqa: E402
from run_abc import CONDITIONS, user_message  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

mdir = os.environ["LFM3_DIR"]
tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(mdir, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
rows = generate("seen")
picks = [rows[0], rows[1], rows[2], next(r for r in rows if r["phenomenon"] == "dist:role_reversal")]
for cond in CONDITIONS:
    for r in picks:
        text = tok.apply_chat_template([{"role": "user", "content": user_message(cond, r["a"], r["b"])}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False)
        new = out[0, ids.shape[1]:].tolist()
        print(cond, r["phenomenon"], "label", r["label"], "|", [tok.decode([t]) for t in new], flush=True)
