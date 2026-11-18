"""Shared helpers for the Qwen C1/C2/C3 pilot (k3 CPU). Run with /root/hybridagent-venv.

Data: databricks/databricks-dolly-15k (CC-BY-SA-3.0), no-context rows, unique instructions, split by sha1 of the instruction
(held-out = hash % 5 == 0), so the split does not depend on row order or on any model.
"""
import glob, hashlib, json, os, sys, time

os.environ.setdefault("HF_HOME", "/root/hf")
BASE = "/opt/clover-k3/qwen-pilot-20261118"
HA_SNAP = glob.glob("/root/hf/hub/models--togethercomputer--HybridAgent-8B-Agentic/snapshots/*")[0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dolly_rows(min_resp_tok=16, max_resp_tok=96, tok=None):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    seen, rows = set(), []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        ins, ctx, resp = r["instruction"].strip(), r["context"].strip(), r["response"].strip()
        if ctx or not ins or ins in seen:
            continue
        seen.add(ins)
        if tok is not None:
            n = len(tok(resp, add_special_tokens=False).input_ids)
            if not (min_resp_tok <= n <= max_resp_tok):
                continue
        h = int(hashlib.sha1(ins.encode("utf-8")).hexdigest(), 16)
        rows.append({"instruction": ins, "response": resp, "split": "held" if h % 5 == 0 else "train", "h": h})
    rows.sort(key=lambda r: r["h"])
    return rows


def build_ids(tok, instruction, response):
    """chat-template prompt ids + response ids; returns (ids, prefix_len)."""
    msgs = [{"role": "user", "content": instruction}]
    try:
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    except TypeError:
        pre = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(pre, "input_ids"):
        pre = pre["input_ids"]
    pre = list(pre)
    resp = tok(response, add_special_tokens=False).input_ids
    return pre + resp, len(pre)


def load_link(hidden=4096, path=None):
    import torch, torch.nn as nn

    class HybridLink(nn.Module):  # same as oss/HybridAgent/hybrid_agent/hybridlink.py
        def __init__(self, d):
            super().__init__()
            self.proj1 = nn.Linear(d, d); self.act = nn.GELU(); self.proj2 = nn.Linear(d, d)
            self.pre_ln = nn.LayerNorm(d); self.post_ln = nn.LayerNorm(d)

        def forward(self, h):
            return self.post_ln(h + self.proj2(self.act(self.proj1(self.pre_ln(h)))))

    m = HybridLink(hidden)
    if path:
        m.load_state_dict(torch.load(path, map_location="cpu"))
    return m
