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
mkdir -p "$HYBRID_NCM_DIR" "$OPEN_WEIGHTS_DIR"

if [ -x "$hf_bin" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf_bin" download "$HF_REPO" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --local-dir "$DATASETS_DIR" \
    --include 'hybrid-ncm-k3/*' \
    --include 'open-weights/*'
else
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    repo_type='dataset',
    revision='$HF_REVISION',
    local_dir='$DATASETS_DIR',
    allow_patterns=['hybrid-ncm-k3/*', 'open-weights/*'],
)
"
fi

if [ -d "$DATASETS_DIR/open-weights" ]; then
  for f in model-00094-of-000096.safetensors tiktoken.model tokenizer_config.json; do
    if [ -f "$DATASETS_DIR/open-weights/$f" ] && [ ! -f "$OPEN_WEIGHTS_DIR/$f" ]; then
      ln -f "$DATASETS_DIR/open-weights/$f" "$OPEN_WEIGHTS_DIR/$f" 2>/dev/null || cp -f "$DATASETS_DIR/open-weights/$f" "$OPEN_WEIGHTS_DIR/$f"
    fi
  done
fi

# Create standardized hyphenated complete-English `checkpoint-*` and `dataset-*` hardlinks
for pair in \
  "e12_clover_k3_true_ha_ncm.pt:checkpoint-clover-true-hybrid-agent-ncm.pt" \
  "e11_clover_k3_native_ha_ncm.pt:checkpoint-clover-native-hybrid-agent-ncm.pt" \
  "e8_k3_direct.pt:checkpoint-next-concept-direct-decoder.pt" \
  "e6_last.pt:checkpoint-next-concept-generator-base.pt" \
  "ha_8b_int8_cpu.pt:checkpoint-hybrid-agent-8b-int8-cpu.pt" \
  "k3_vqvae_adapted.pt:checkpoint-concept-vector-quantizer-adapted.pt" \
  "k3_vqvae_init.pt:checkpoint-concept-vector-quantizer-initial.pt" \
  "k3_hybrid_bridges.pt:checkpoint-latent-representation-bridges.pt" \
  "staged_32win_inputs.pt:dataset-staged-32-window-inputs.pt" \
  "wmt14/tokenizer.pt:wmt14/checkpoint-wmt14-concept-tokenizer.pt" \
  "wmt14/config.yaml:wmt14/config-wmt14-concept-model.yaml" \
  "data/wmt14/reference_translations.pickle:data/wmt14/dataset-wmt14-reference-translations.pickle" \
  "data/xsum/passages.pickle:data/xsum/dataset-xsum-news-passages.pickle" \
  "data/xsum/reference_summaries.pickle:data/xsum/dataset-xsum-reference-summaries.pickle" \
  "data/sudoku/easy/puzzles.pickle:data/sudoku/easy/dataset-sudoku-easy-puzzles.pickle" \
  "data/sudoku/easy/solutions.pickle:data/sudoku/easy/dataset-sudoku-easy-solutions.pickle"
do
  src_rel="${pair%%:*}"
  dst_rel="${pair##*:}"
  if [ -f "$HYBRID_NCM_DIR/$src_rel" ] && [ ! -f "$HYBRID_NCM_DIR/$dst_rel" ]; then
    ln -f "$HYBRID_NCM_DIR/$src_rel" "$HYBRID_NCM_DIR/$dst_rel" 2>/dev/null || cp -f "$HYBRID_NCM_DIR/$src_rel" "$HYBRID_NCM_DIR/$dst_rel"
  fi
done

echo "== Verifying downloaded hybrid-ncm-k3 files in $HYBRID_NCM_DIR =="
ls -lh "$HYBRID_NCM_DIR"
