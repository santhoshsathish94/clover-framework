#!/bin/bash
# QA gate setup on k3 (CPU only), into /root/ncm-cpu: released WMT14 NCM checkpoint (Qwen3 vocabulary), its English references,
# the Qwen3-0.6B tokenizer files, GSM8K test. XSum summaries are copied in separately by scp. Does not touch other venvs.
set -u
D=/root/ncm-cpu
cd $D
echo "== start $(date -Is)"
. venv/bin/activate
export HF_HOME=/root/hf HF_HUB_DISABLE_PROGRESS_BARS=1
python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
print("ckpt", snapshot_download("hazyresearch/ncm-checkpoints", allow_patterns=["wmt14/*"], local_dir="/root/ncm-cpu/artifacts"))
print("refs", hf_hub_download("hazyresearch/ncm-datasets", "wmt14/reference_translations.pickle", repo_type="dataset", local_dir="/root/ncm-cpu/data"))
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print("qwen3-0.6b tokenizer", len(t))
EOF
mkdir -p data/gsm8k
curl -fsSL -o data/gsm8k/test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
echo "gsm8k lines $(wc -l < data/gsm8k/test.jsonl)"
find artifacts/wmt14 data/wmt14 data/gsm8k data/xsum -type f -exec ls -la {} \; 2>/dev/null | awk '{print $5, $9}'
df -h / | tail -1
echo "== end $(date -Is)"
touch $D/setup-qa.done
