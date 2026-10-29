#!/bin/bash
# Downloads ONLY the required Clover Kimi K3 + HybridAgent + NCM checkpoints and evaluation datasets
# listed in `src/scripts/hybrid-ncm-dataset-files.tsv` from Hugging Face (`Santhoshsathish94/clover-kimi-k3`,
# dataset repo) into `$HYBRID_NCM_DIR` (`$ENGINE_DIR/src/datasets/hybrid-ncm-k3` by default),
# leaving existing `computed/` (`expert-1..92`, `trunk-0..92`) and any unused/archived HF files untouched.
#
# Usage:
#   [HF_REPO=Santhoshsathish94/clover-kimi-k3] [HYBRID_NCM_DIR=/path/to/dir] ./download-hybrid-ncm-datasets.sh
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

HF_REPO="${HF_REPO:-Santhoshsathish94/clover-kimi-k3}"
HF_REVISION="${HF_REVISION:-main}"
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

echo "== Downloading ${#REQUIRED_REL_PATHS[@]} required hybrid-ncm-k3 files (from $(basename "$MANIFEST_TSV")) -> $DATASETS_DIR =="
mkdir -p "$HYBRID_NCM_DIR" "$OPEN_WEIGHTS_DIR"

INCLUDE_ARGS=()
for rel_p in "${REQUIRED_REL_PATHS[@]}"; do
  INCLUDE_ARGS+=(--include "hybrid-ncm-k3/$rel_p")
done
INCLUDE_ARGS+=(--include "open-weights/*")

if [ -x "$hf_bin" ]; then
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf_bin" download "$HF_REPO" \
    --repo-type dataset \
    --revision "$HF_REVISION" \
    --local-dir "$DATASETS_DIR" \
    "${INCLUDE_ARGS[@]}"
else
  MANIFEST_TSV="$MANIFEST_TSV" HF_REPO="$HF_REPO" HF_REVISION="$HF_REVISION" DATASETS_DIR="$DATASETS_DIR" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

manifest = os.environ["MANIFEST_TSV"]
allow = ["open-weights/*"]
with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rel_p = line.split("\t")[0].strip()
        if rel_p:
            allow.append(f"hybrid-ncm-k3/{rel_p}")

snapshot_download(
    repo_id=os.environ["HF_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_REVISION"],
    local_dir=os.environ["DATASETS_DIR"],
    allow_patterns=allow,
)
PY
fi

if [ -d "$DATASETS_DIR/open-weights" ]; then
  for f in model-00094-of-000096.safetensors tiktoken.model tokenizer_config.json; do
    if [ -f "$DATASETS_DIR/open-weights/$f" ] && [ ! -f "$OPEN_WEIGHTS_DIR/$f" ]; then
      ln -f "$DATASETS_DIR/open-weights/$f" "$OPEN_WEIGHTS_DIR/$f" 2>/dev/null || cp -f "$DATASETS_DIR/open-weights/$f" "$OPEN_WEIGHTS_DIR/$f"
    fi
  done
fi

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
