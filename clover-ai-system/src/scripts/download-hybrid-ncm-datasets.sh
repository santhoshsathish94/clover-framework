#!/bin/bash
# Downloads Clover AI Studio precomputed trunks, experts, HybridAgent/NCM checkpoints, and evaluation datasets
# from Hugging Face (`Santhoshsathish94/clover-ai-studio`, dataset repo), where every model family
# (`kimi/`, `nemotron/`, `liquid-lfm3/`) follows the identical 5-subfolder layout:
#   <model>/{experts, trunks, hybridagent, ncm, bridges}/
#
# Usage:
#   [HF_REPO=Santhoshsathish94/clover-ai-studio] [MODEL_FAMILY=kimi|nemotron|liquid-lfm3|all] ./download-hybrid-ncm-datasets.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

HF_REPO="${HF_REPO:-Santhoshsathish94/clover-ai-studio}"
HF_REVISION="${HF_REVISION:-main}"
MODEL_FAMILY="${MODEL_FAMILY:-kimi}"
HYBRID_NCM_DIR="${HYBRID_NCM_DIR:-$DATASETS_DIR/hybrid-ncm-k3}"
MANIFEST_TSV="${HYBRID_NCM_MANIFEST:-$SCRIPTS_DIR/hybrid-ncm-dataset-files.tsv}"

if [ ! -f "$MANIFEST_TSV" ]; then
  echo "ERROR: Required dataset manifest $MANIFEST_TSV not found." >&2
  exit 1
fi

hf_bin=$(command -v hf || command -v huggingface-cli || echo /root/.local/bin/hf)
if [ ! -x "$hf_bin" ] && [ -x "/opt/clover-k3/venv/bin/hf" ]; then
  hf_bin="/opt/clover-k3/venv/bin/hf"
fi

mapfile -t REQUIRED_REL_PATHS < <(awk -F'\t' '!/^#/ && NF>=1 && $1!="" {gsub(/\r/, "", $1); print $1}' "$MANIFEST_TSV")

echo "== Downloading Clover AI Studio datasets (MODEL_FAMILY=$MODEL_FAMILY from $HF_REPO) -> $DATASETS_DIR =="
mkdir -p "$HYBRID_NCM_DIR" "$OPEN_WEIGHTS_DIR"

INCLUDE_ARGS=()
if [ "$MODEL_FAMILY" = "kimi" ] || [ "$MODEL_FAMILY" = "all" ]; then
  INCLUDE_ARGS+=(
    --include "kimi/hybridagent/**"
    --include "kimi/ncm/**"
    --include "kimi/bridges/**"
    --include "kimi/trunks/open-weights/*"
  )
fi
if [ "$MODEL_FAMILY" = "nemotron" ] || [ "$MODEL_FAMILY" = "all" ]; then
  INCLUDE_ARGS+=(--include "nemotron/**")
fi
if [ "$MODEL_FAMILY" = "liquid-lfm3" ] || [ "$MODEL_FAMILY" = "all" ]; then
  INCLUDE_ARGS+=(--include "liquid-lfm3/**")
fi

if [ -x "$hf_bin" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf_bin" download "$HF_REPO" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --local-dir "$DATASETS_DIR" \
    "${INCLUDE_ARGS[@]}"
else
  MANIFEST_TSV="$MANIFEST_TSV" HF_REPO="$HF_REPO" HF_REVISION="$HF_REVISION" DATASETS_DIR="$DATASETS_DIR" MODEL_FAMILY="$MODEL_FAMILY" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

family = os.environ.get("MODEL_FAMILY", "kimi")
allow = []
if family in ("kimi", "all"):
    allow.extend([
        "kimi/hybridagent/**",
        "kimi/ncm/**",
        "kimi/bridges/**",
        "kimi/trunks/open-weights/*",
    ])
if family in ("nemotron", "all"):
    allow.append("nemotron/**")
if family in ("liquid-lfm3", "all"):
    allow.append("liquid-lfm3/**")

snapshot_download(
    repo_id=os.environ["HF_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_REVISION"],
    local_dir=os.environ["DATASETS_DIR"],
    allow_patterns=allow,
)
PY
fi

for k_sub in "$DATASETS_DIR/kimi/hybridagent" "$DATASETS_DIR/kimi/ncm" "$DATASETS_DIR/kimi/bridges"; do
  if [ -d "$k_sub" ]; then
    cp -rn "$k_sub/." "$HYBRID_NCM_DIR/" 2>/dev/null || cp -r "$k_sub/." "$HYBRID_NCM_DIR/"
  fi
done

for ow_src in "$DATASETS_DIR/kimi/trunks/open-weights" "$DATASETS_DIR/kimi/open-weights" "$DATASETS_DIR/open-weights"; do
  if [ -d "$ow_src" ]; then
    for f in model-00094-of-000096.safetensors tiktoken.model tokenizer_config.json; do
      if [ -f "$ow_src/$f" ] && [ ! -f "$OPEN_WEIGHTS_DIR/$f" ]; then
        ln -f "$ow_src/$f" "$OPEN_WEIGHTS_DIR/$f" 2>/dev/null || cp -f "$ow_src/$f" "$OPEN_WEIGHTS_DIR/$f"
      fi
    done
  fi
done

if [ "$MODEL_FAMILY" = "kimi" ] || [ "$MODEL_FAMILY" = "all" ]; then
  echo "== Verifying downloaded hybrid-ncm-k3 files against $MANIFEST_TSV =="
  missing=0
  for rel_p in "${REQUIRED_REL_PATHS[@]}"; do
    if [ ! -f "$HYBRID_NCM_DIR/$rel_p" ]; then
      echo "MISSING: $HYBRID_NCM_DIR/$rel_p" >&2
      missing=$((missing + 1))
    fi
  done
  [ "$missing" -eq 0 ] || { echo "FAIL: $missing required files missing in $HYBRID_NCM_DIR" >&2; exit 1; }
  ls -lh "$HYBRID_NCM_DIR"
fi
