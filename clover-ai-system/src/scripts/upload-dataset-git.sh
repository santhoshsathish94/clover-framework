#!/bin/bash
# Uploads the computed dataset to a Hugging Face dataset repo with git over ssh, using the ssh key registered at huggingface.co/settings/keys (no token).
# Only the files in dataset-files.tsv go up. git-lfs copies every file it adds, so the upload goes in batches: one expert layer (15.7 GB) at a time, the
# trunk folders ten at a time; each batch is committed and pushed, then its copies are deleted. Run it again after any interruption: batches already on the
# remote are skipped and an unpushed commit is pushed first.
# usage: [HF_REPO=Santhoshsathish94/clover-kimi-k3] [PUSH_DIR=folder for the clone] [MAX_BATCHES=n] upload-dataset-git.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
HF_REPO="${HF_REPO:-Santhoshsathish94/clover-kimi-k3}"
PUSH_DIR="${PUSH_DIR:-$WORK_DIR/hf-push}"
MAX_BATCHES="${MAX_BATCHES:-1000}"
LIST="${DATASET_LIST:-$SCRIPTS_DIR/dataset-files.tsv}"
REPO="$PUSH_DIR/repo"
hf=$(command -v hf || echo /root/.local/bin/hf)
say() { echo "$(date +%H:%M:%S) $*"; }
command -v git-lfs > /dev/null || { echo "git-lfs is missing (apt-get install git-lfs)"; exit 1; }
[ -x "$hf" ] || { echo "no hf command line (it provides the multipart transfer agent for files over 5 GB)"; exit 1; }
"$SCRIPTS_DIR/check-dataset-files.sh" > /dev/null 2>&1 || { echo "the dataset is not complete (run check-dataset-files.sh)"; exit 1; }
mkdir -p "$PUSH_DIR"
if [ ! -d "$REPO/.git" ]; then GIT_LFS_SKIP_SMUDGE=1 git clone -q "git@hf.co:datasets/$HF_REPO" "$REPO" || exit 1; fi
cd "$REPO" || exit 1
git config user.name "${GIT_AUTHOR_NAME:-Santhosh Narayanan}"; git config user.email "${GIT_AUTHOR_EMAIL:-santhoshsathish94@gmail.com}"
git lfs install --local > /dev/null; git lfs track "*.direct" "*.f32" "*.pt" "*.pickle" "*.safetensors" > /dev/null; git config lfs.customtransfer.multipart.path "$hf"; "$hf" lfs-enable-largefiles . > /dev/null
git config lfs.customtransfer.multipart.path "$hf"
git add .gitattributes; git diff --cached --quiet || git commit -q -m "track *.direct, *.f32, *.pt, *.pickle, and *.safetensors with LFS"

paths=$(awk -F'\t' '!/^#/ && NF>=3 {print $1}' "$LIST"; [ -d "$INPUTS_DIR/refs" ] && ls "$INPUTS_DIR/refs" | sed 's|^|engine-inputs/refs/|'; [ -d "$HYBRID_NCM_DIR" ] && find "$HYBRID_NCM_DIR" -type f -printf 'hybrid-ncm-k3/%P\n')
batch_files() {   # label -> the files of that batch
  case "$1" in
    expert-*) echo "$paths" | grep "^$1/";;
    refs) echo "$paths" | grep "^engine-inputs/";;
    hybrid-ncm-k3) echo "$paths" | grep "^hybrid-ncm-k3/";;
    trunks-*) from=${1#trunks-}; for n in $(seq "$from" $((from + 9))); do echo "$paths" | grep "^trunk-$n/"; done;;
  esac
}
push_pending() {
  local tries=0
  while [ "$(git rev-list --count origin/main..HEAD)" != 0 ]; do
    if git push -q origin main; then return 0; fi
    tries=$((tries + 1)); say "push failed (attempt $tries), waiting a minute"; [ $tries -ge 6 ] && return 1; sleep 60
  done
}
done_batches=0
for label in refs hybrid-ncm-k3 $(seq 1 92 | sed 's/^/expert-/') $(seq 0 10 90 | sed 's/^/trunks-/'); do
  [ $done_batches -ge "$MAX_BATCHES" ] && break
  git fetch -q origin main || { say "fetch failed"; exit 1; }
  push_pending || exit 1
  remote=$(git ls-tree -r --name-only origin/main)
  todo=$(batch_files "$label" | grep -vxF -f <(echo "$remote") || true)
  [ -z "$todo" ] && continue
  free_gb=$(df --output=avail -B1G "$PUSH_DIR" | tail -1 | tr -d ' ')
  [ "$free_gb" -ge 18 ] || { say "only $free_gb GB free, stopping (one batch needs up to 17 GB)"; exit 1; }
  bytes=0
  while IFS= read -r f; do
    mkdir -p "$(dirname "$f")"
    src="$DATASET_DIR/$f"
    [ -e "$src" ] || src="$INPUTS_DIR/${f#engine-inputs/}"
    [ -e "$src" ] || src="$HYBRID_NCM_DIR/${f#hybrid-ncm-k3/}"
    ln -f "$src" "$f" 2> /dev/null || cp "$src" "$f" || exit 1
    bytes=$((bytes + $(stat -L -c %s "$f")))
  done <<< "$todo"
  say "batch $label: $(echo "$todo" | wc -l) files, $((bytes / 1048576)) MiB"
  started=$(date +%s)
  echo "$todo" | xargs -d '\n' git add -- || exit 1
  git diff --cached --quiet || git commit -q -m "add $label" || exit 1
  push_pending || exit 1
  secs=$(( $(date +%s) - started )); [ $secs -gt 0 ] || secs=1
  rm -rf .git/lfs/objects/* .git/lfs/tmp/* 2> /dev/null
  say "batch $label pushed in ${secs}s ($((bytes / 1048576 / secs)) MiB/s)"
  done_batches=$((done_batches + 1))
done
say "finished a pass; files on the remote: $(git ls-tree -r --name-only origin/main | wc -l)"
