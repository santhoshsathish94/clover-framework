#!/bin/bash
# Keeps the git upload going: runs upload-dataset-git.sh, and when it stops before every listed file is on the remote (a network error, a reboot, a refused
# push) waits and runs it again, until nothing is missing. Only one keeper runs at a time (flock), so it is safe to start it from cron every few minutes.
# usage: [PUSH_DIR=folder of the clone] [HF_REPO=...] keep-uploading.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
PUSH_DIR="${PUSH_DIR:-$WORK_DIR/hf-push}"
LIST="${DATASET_LIST:-$SCRIPTS_DIR/dataset-files.tsv}"
LOG="${UPLOAD_LOG:-/tmp/upload-full.out}"
mkdir -p "$PUSH_DIR"
exec 9> "$PUSH_DIR/keeper.lock"
flock -n 9 || exit 0
[ -f "$PUSH_DIR/COMPLETE" ] && exit 0
missing() {   # number of listed files not yet in origin/main
  [ -d "$PUSH_DIR/repo/.git" ] || { echo 999999; return; }
  git -C "$PUSH_DIR/repo" fetch -q origin main 2> /dev/null || { echo 999999; return; }
  awk -F'\t' '!/^#/ && NF>=3 {print $1}' "$LIST" | grep -vxF -f <(git -C "$PUSH_DIR/repo" ls-tree -r --name-only origin/main) | wc -l
}
while true; do
  if pgrep -f "scripts/upload-dataset-git.sh" > /dev/null; then sleep 60; continue; fi   # one started by hand is still running: wait for it
  echo "$(date +%H:%M:%S) keeper: starting the upload script" >> "$LOG"
  PUSH_DIR="$PUSH_DIR" "$SCRIPTS_DIR/upload-dataset-git.sh" >> "$LOG" 2>&1
  left=$(missing)
  if [ "$left" = 0 ]; then echo "$(date +%H:%M:%S) keeper: every listed file is on the remote" >> "$LOG"; touch "$PUSH_DIR/COMPLETE"; exit 0; fi
  echo "$(date +%H:%M:%S) keeper: $left files still missing, trying again in 5 minutes" >> "$LOG"
  sleep 300
done
