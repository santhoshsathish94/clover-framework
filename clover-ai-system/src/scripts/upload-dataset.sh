#!/bin/bash
# Uploads the computed dataset to a Hugging Face dataset repo: exactly the files in dataset-files.tsv (expert-N, trunk-N with operator.bin), plus the recorded
# values of the pod's chain (engine-inputs/refs) if they are in work/inputs/refs. The files are hard-linked into a staging folder first (no copy), then sent with
# `hf upload-large-folder`, which can be stopped and run again: it resumes.
# usage: [HF_REPO=Santhoshsathish94/clover-kimi-k3] [HF_REPO_TYPE=dataset] [STAGING=folder] [HF_WORKERS=16] upload-dataset.sh [--dry-run]
# Needs `hf auth login` done on this machine by you (a token with write access); the script never reads or prints the token.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
HF_REPO="${HF_REPO:-Santhoshsathish94/clover-kimi-k3}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_WORKERS="${HF_WORKERS:-16}"
STAGING="${STAGING:-$DATASETS_DIR/upload-staging}"
LIST="${DATASET_LIST:-$SCRIPTS_DIR/dataset-files.tsv}"
hf=$(command -v hf || echo /root/.local/bin/hf); [ -x "$hf" ] || { echo "no hf command line found"; exit 1; }
[ -f "$LIST" ] || { echo "$LIST is missing"; exit 1; }
"$SCRIPTS_DIR/check-dataset-files.sh" > /dev/null 2>&1 || { echo "the dataset is not complete (run check-dataset-files.sh)"; exit 1; }

rm -rf "$STAGING"; mkdir -p "$STAGING"
n=0; bytes=0
while IFS=$'\t' read -r path size sum; do
  case "$path" in '#'*|'') continue;; esac
  mkdir -p "$STAGING/$(dirname "$path")"
  ln -f "$DATASET_DIR/$path" "$STAGING/$path" 2> /dev/null || ln -s "$(readlink -f "$DATASET_DIR/$path")" "$STAGING/$path" || { echo "cannot stage $path"; exit 1; }
  n=$((n + 1)); bytes=$((bytes + size))
done < "$LIST"
if [ -d "$INPUTS_DIR/refs" ]; then
  mkdir -p "$STAGING/engine-inputs/refs"
  for f in "$INPUTS_DIR"/refs/*; do ln -f "$f" "$STAGING/engine-inputs/refs/$(basename "$f")" 2> /dev/null || ln -s "$f" "$STAGING/engine-inputs/refs/$(basename "$f")"; bytes=$((bytes + $(stat -L -c %s "$f"))); n=$((n + 1)); done
fi
echo "staged $n files, $((bytes / 1073741824)) GiB, in $STAGING"
[ "${1:-}" = "--dry-run" ] && { echo "dry run: nothing uploaded"; exit 0; }
"$hf" auth whoami > /dev/null 2>&1 || { echo "not logged in to Hugging Face on this machine: run  $hf auth login  yourself (paste a write token), then run this again"; exit 1; }
echo "uploading to $HF_REPO_TYPE $HF_REPO as $("$hf" auth whoami 2>/dev/null | head -1)"
exec "$hf" upload-large-folder "$HF_REPO" "$STAGING" --repo-type "$HF_REPO_TYPE" --num-workers "$HF_WORKERS"
