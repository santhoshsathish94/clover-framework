#!/bin/bash
# NCM (HazyResearch/ncm @ c702742) on k3 CPU, own venv, only the XSum checkpoint. Does not touch /root/hybridagent-venv.
set -u
D=/root/ncm-cpu
mkdir -p $D && cd $D
echo "== start $(date -Is)"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install --quiet numpy omegaconf hydra-core "transformers>=4.51" huggingface_hub regex
mkdir -p repo && tar -xf /root/ncm-c702742.tar -C repo
export HF_HOME=/root/hf
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "transformers", transformers.__version__, "threads", torch.get_num_threads())
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="hazyresearch/ncm-checkpoints", allow_patterns=["xsum/*"], local_dir="/root/ncm-cpu/artifacts")
print("ckpt", p)
from transformers import AutoTokenizer
print(len(AutoTokenizer.from_pretrained("gpt2")("hello world")["input_ids"]), "gpt2 tokenizer ok")
EOF
find artifacts -type f -exec ls -la {} \; | awk '{print $5, $9}'
echo "== end $(date -Is)"
touch $D/setup.done
