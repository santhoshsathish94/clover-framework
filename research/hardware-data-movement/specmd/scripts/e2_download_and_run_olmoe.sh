#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/root/dm-study/specmd/w2/hf_cache"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export MALLOC_ARENA_MAX=2

MODEL_DIR="/root/dm-study/specmd/w2/model"
mkdir -p "$MODEL_DIR" "$HF_HOME"

echo "=== 1. Download allenai/OLMoE-1B-7B-0924 files (idempotent, size-checked, no images) ==="
/root/ncm-cpu/venv/bin/python - << 'PY'
import json, os, subprocess, concurrent.futures

model_dir = "/root/dm-study/specmd/w2/model"
with open("/root/dm-study/specmd/w2/model_tree.json") as f:
    items = json.load(f)

to_fetch = []
for it in items:
    fn = it["path"]
    if fn.endswith(".png") or fn.endswith(".jpg"):
        continue
    sz = it.get("size", 0)
    dst = os.path.join(model_dir, fn)
    if os.path.exists(dst) and os.path.getsize(dst) == sz:
        print(f"  [OK cached] {fn} ({sz} bytes)")
    else:
        to_fetch.append((fn, sz, dst))

def download_one(item):
    fn, sz, dst = item
    url = f"https://huggingface.co/allenai/OLMoE-1B-7B-0924/resolve/main/{fn}"
    tmp = dst + ".part"
    cmd = ["curl", "-sL", "--retry", "3", "-o", tmp, url]
    subprocess.run(cmd, check=True)
    actual = os.path.getsize(tmp)
    if actual != sz:
        raise RuntimeError(f"Size mismatch for {fn}: got {actual}, expected {sz}")
    os.replace(tmp, dst)
    return f"  [DOWNLOADED] {fn} ({actual} bytes)"

if to_fetch:
    print(f"Downloading {len(to_fetch)} files with 3 workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for msg in ex.map(download_one, to_fetch):
            print(msg, flush=True)
else:
    print("All files already present and size-verified.")
PY

echo ""
echo "=== 2. Disk usage check ==="
du -sh "$MODEL_DIR" /root/dm-study/specmd

echo ""
echo "=== 3. Load OLMoE-1B-7B-0924 in bf16 on CPU, inspect architecture, generate text ==="
/root/ncm-cpu/venv/bin/python - << 'PY'
import time, json, torch
from transformers import AutoTokenizer, OlmoeForCausalLM

def mem_status():
    info = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmPeak:", "VmSize:", "VmRSS:", "VmHWM:")):
                k, v = line.split(":", 1)
                info[k] = v.strip()
    return info

t0 = time.time()
model_dir = "/root/dm-study/specmd/w2/model"
with open(f"{model_dir}/config.json") as f:
    cfg = json.load(f)
print("Config summary:", {
    k: cfg.get(k) for k in [
        "architectures", "num_hidden_layers", "num_experts", "num_experts_per_tok",
        "hidden_size", "intermediate_size", "vocab_size", "norm_topk_prob"
    ]
})

torch.set_default_dtype(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = OlmoeForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
model.eval()
t_load = time.time() - t0
print(f"Loaded model in {t_load:.2f}s | Memory: {mem_status()}")

# Count expert vs non-expert parameters and bytes
total_params = 0
expert_params = 0
for name, p in model.named_parameters():
    n = p.numel()
    total_params += n
    if ".mlp.experts." in name:
        expert_params += n
non_expert_params = total_params - expert_params

num_layers = cfg["num_hidden_layers"]
num_experts = cfg["num_experts"]
experts_mod = model.model.layers[0].mlp.experts
layer0_expert_params = sum(p.numel() for p in experts_mod.parameters())
layer0_expert_bytes = sum(p.numel() * p.element_size() for p in experts_mod.parameters())
e0_params = layer0_expert_params // num_experts
e0_bytes = layer0_expert_bytes // num_experts
gate0 = model.model.layers[0].mlp.gate
print(f"Total params: {total_params:,} ({total_params*2/1e9:.3f} GB decimal, {total_params*2/(1024**3):.3f} GiB)")
print(f"Expert params: {expert_params:,} ({expert_params*2/1e9:.3f} GB decimal, {expert_params*2/(1024**3):.3f} GiB)")
print(f"Non-expert params: {non_expert_params:,} ({non_expert_params*2/1e9:.3f} GB decimal, {non_expert_params*2/(1024**3):.3f} GiB)")
print(f"Single expert params: {e0_params:,} | bytes (bf16): {e0_bytes:,} ({e0_bytes/(1024**2):.2f} MiB, {e0_bytes/1e6:.2f} MB)")
print(f"Experts module type: {type(experts_mod)}")
for pname, p in experts_mod.named_parameters():
    print(f"  experts.{pname}: shape={tuple(p.shape)}, dtype={p.dtype}")
print(f"Gate layer[0]: {type(gate0)}, params={[(n, tuple(p.shape), p.dtype) for n, p in gate0.named_parameters()]}")

# Generate short text deterministically
prompts = [
    "Mixture-of-Experts language models reduce computation per token by",
    "In computer architecture, a cache replacement policy determines"
]
for idx, pr in enumerate(prompts):
    inputs = tokenizer(pr, return_tensors="pt")
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False
        )
    dt = time.time() - t1
    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\nPrompt {idx+1} ({dt:.2f}s, {24/dt:.2f} tok/s): {gen_text!r}")

print("Final memory:", mem_status())
PY
