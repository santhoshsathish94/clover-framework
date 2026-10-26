#!/bin/bash
# Downloads the Clover Kimi K3 + HybridAgent + NCM checkpoints and evaluation datasets
# (`hybrid-ncm-k3/*`) from Hugging Face (`Santhoshsathish94/clover-kimi-k3`, dataset repo)
# into `$HYBRID_NCM_DIR` (`$ENGINE_DIR/src/datasets/hybrid-ncm-k3` by default),
# leaving existing `computed/` (`expert-1..92`, `trunk-0..92`) and `third-party/` untouched.
#
# Usage:
#   [HF_REPO=Santhoshsathish94/clover-kimi-k3] [HYBRID_NCM_DIR=/path/to/dir] ./download-hybrid-ncm-datasets.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

HF_REPO="${HF_REPO:-Santhoshsathish94/clover-kimi-k3}"
HF_REVISION="${HF_REVISION:-main}"
HYBRID_NCM_DIR="${HYBRID_NCM_DIR:-$DATASETS_DIR/hybrid-ncm-k3}"

hf_bin=$(command -v hf || command -v huggingface-cli || echo /root/.local/bin/hf)
if [ ! -x "$hf_bin" ] && [ -x "/opt/clover-k3/venv/bin/hf" ]; then
  hf_bin="/opt/clover-k3/venv/bin/hf"
fi

echo "== Downloading hybrid-ncm-k3 checkpoints & datasets from Hugging Face ($HF_REPO) -> $DATASETS_DIR =="
mkdir -p "$HYBRID_NCM_DIR"

if [ -x "$hf_bin" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf_bin" download "$HF_REPO" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --local-dir "$DATASETS_DIR" \
    --include 'hybrid-ncm-k3/*'
else
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    repo_type='dataset',
    revision='$HF_REVISION',
    local_dir='$DATASETS_DIR',
    allow_patterns=['hybrid-ncm-k3/*'],
)
"
fi

echo "== Verifying downloaded hybrid-ncm-k3 files in $HYBRID_NCM_DIR =="
ls -lh "$HYBRID_NCM_DIR"
