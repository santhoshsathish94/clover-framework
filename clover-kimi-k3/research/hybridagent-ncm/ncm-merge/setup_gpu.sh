#!/bin/bash
# Cycle 30 setup on the gpu box (creator permission 2026-10-22): everything under ~/plan2text, nothing else touched.
set -u
P=$HOME/plan2text
mkdir -p $P/data $P/out
cd $P
echo "== start $(date -Is)"
$HOME/ncm/venv/bin/python - <<'EOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download
for fn in ["TinyStories-train.txt", "TinyStories-valid.txt"]:
    q = hf_hub_download(repo_id="roneneldan/TinyStories", filename=fn, repo_type="dataset", local_dir=os.path.expanduser("~/plan2text/data"))
    print("data", q, os.path.getsize(q))
p = snapshot_download(repo_id="togethercomputer/HybridAgent-8B-Agentic", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*"])
print("model", p)
EOF
ls -la $P/data
du -sh ~/.cache/huggingface/hub/models--togethercomputer--HybridAgent-8B-Agentic
df -h ~ | tail -1
echo "== end $(date -Is)"
