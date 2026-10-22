#!/bin/bash
# Set up HybridAgent (commit 1b41ceb) and the 8B agentic checkpoint on k3 (CPU only). Nothing outside /root/hybrid* and /root/hf is touched.
LOG=/root/hybrid-setup.log
exec >> "$LOG" 2>&1
set -x
echo "=== start $(date '+%F %T')"
df -h / | tail -1
cd /root
[ -d hybridagent ] || git clone --depth 1 https://github.com/togethercomputer/HybridAgent hybridagent
git -C hybridagent log -1 --format='%H %s'
python3 -m venv /root/hybridagent-venv || { echo "VENV FAILED"; exit 1; }
. /root/hybridagent-venv/bin/activate
python --version
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu || { echo "TORCH 2.11.0 CPU FAILED"; exit 2; }
grep -v -E '^(torch|deepspeed)==' hybridagent/requirements.txt | grep -v '^#' > /root/hybrid-req-cpu.txt
cat /root/hybrid-req-cpu.txt
pip install --no-cache-dir -r /root/hybrid-req-cpu.txt || { echo "REQUIREMENTS FAILED"; exit 3; }
pip install --no-cache-dir -e hybridagent --no-deps || { echo "EDITABLE INSTALL FAILED"; exit 4; }
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'threads', torch.get_num_threads())"
export HF_HOME=/root/hf
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="togethercomputer/HybridAgent-8B-Agentic")
print("MODEL", p)
PY
du -sh /root/hf
df -h / | tail -1
echo "=== done $(date '+%F %T')"
touch /root/hybrid-setup.done
