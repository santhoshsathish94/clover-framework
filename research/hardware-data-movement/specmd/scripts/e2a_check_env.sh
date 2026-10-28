#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/root/dm-study/specmd/w2/hf_cache"
mkdir -p "/root/dm-study/specmd/w2/model" "$HF_HOME"

echo "=== 1. Check Python & packages in /root/ncm-cpu/venv ==="
/root/ncm-cpu/venv/bin/python - << 'PY'
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
for pkg in ["transformers", "safetensors", "tokenizers", "huggingface_hub", "accelerate"]:
    try:
        m = __import__(pkg)
        print(f"{pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError as e:
        print(f"{pkg}: NOT INSTALLED ({e})")

try:
    from transformers import OlmoeForCausalLM, AutoTokenizer
    print("OlmoeForCausalLM import: OK")
except Exception as e:
    print("OlmoeForCausalLM import FAILED:", e)
PY

echo ""
echo "=== 2. Query HF API for allenai/OLMoE-1B-7B-0924 file list ==="
curl -sL "https://huggingface.co/api/models/allenai/OLMoE-1B-7B-0924/tree/main" -o /root/dm-study/specmd/w2/model_tree.json
/root/ncm-cpu/venv/bin/python - << 'PY'
import json
with open("/root/dm-study/specmd/w2/model_tree.json") as f:
    items = json.load(f)
total = 0
for it in items:
    sz = it.get("size", 0)
    total += sz
    print(f"  {it['path']:40s}  {sz/1e6:9.2f} MB")
print(f"Total repo size: {total/1e9:.3f} GB ({total} bytes)")
PY

echo ""
echo "=== 3. Check current disk usage of /root/dm-study/specmd ==="
du -sh /root/dm-study/specmd /root/dm-study/specmd/* 2>/dev/null || true
df -h /root/dm-study
