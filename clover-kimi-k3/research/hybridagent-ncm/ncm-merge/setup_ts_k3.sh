#!/bin/bash
# Cycle 28 setup on k3 (CPU only): released TinyStories NCM checkpoints + TinyStories validation text, into /root/ncm-cpu. Does not touch other venvs.
set -u
D=/root/ncm-cpu
cd $D
echo "== start $(date -Is)"
. venv/bin/activate
export HF_HOME=/root/hf
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
p = snapshot_download(repo_id="hazyresearch/ncm-checkpoints", allow_patterns=["tinystories/*"], local_dir="/root/ncm-cpu/artifacts")
print("ckpt", p)
for fn in ["TinyStories-valid.txt", "TinyStoriesV2-GPT4-valid.txt"]:
    try:
        q = hf_hub_download(repo_id="roneneldan/TinyStories", filename=fn, repo_type="dataset", local_dir="/root/ncm-cpu/data")
        print("data", q)
    except Exception as e:
        print("data FAIL", fn, repr(e)[:200])
EOF
find artifacts/tinystories data -type f -exec ls -la {} \; | awk '{print $5, $9}'
echo "== end $(date -Is)"
touch $D/setup-ts.done
