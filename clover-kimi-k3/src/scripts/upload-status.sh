#!/bin/bash
# Shows how the git upload is doing: running or not, the last log lines, batches pushed, data sent, free disk, and a warning if the log has been quiet too long.
# usage: [PUSH_DIR=...] [UPLOAD_LOG=/tmp/upload-full.out] upload-status.sh
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
PUSH_DIR="${PUSH_DIR:-$WORK_DIR/hf-push}"
LOG="${UPLOAD_LOG:-/tmp/upload-full.out}"
LIST="${DATASET_LIST:-$SCRIPTS_DIR/dataset-files.tsv}"
now=$(date +%s)
if pgrep -f "scripts/upload-dataset-git.sh" > /dev/null; then echo "upload script: RUNNING"; else echo "upload script: not running"; fi
if pgrep -f "scripts/keep-uploading.sh" > /dev/null; then echo "keeper: running"; else echo "keeper: not running"; fi
[ -f "$PUSH_DIR/COMPLETE" ] && echo "COMPLETE: every listed file is on the remote"
echo "--- last log lines ($LOG):"; tail -5 "$LOG" 2> /dev/null
pushed=$(grep -c " pushed in " "$LOG" 2> /dev/null); echo "--- batches pushed in this log: $pushed"
if [ -d "$PUSH_DIR/repo/.git" ]; then
  git -C "$PUSH_DIR/repo" fetch -q origin main 2> /dev/null
  total=$(awk -F'\t' '!/^#/ && NF>=3' "$LIST" | wc -l)
  have=$(awk -F'\t' '!/^#/ && NF>=3 {print $1}' "$LIST" | grep -cxF -f <(git -C "$PUSH_DIR/repo" ls-tree -r --name-only origin/main))
  gib=$(awk -F'\t' '!/^#/ && NF>=3 {print $1 "\t" $2}' "$LIST" | grep -F -f <(git -C "$PUSH_DIR/repo" ls-tree -r --name-only origin/main) | awk -F'\t' '{s+=$2} END{printf "%.0f", s/1073741824}')
  echo "on the remote: $have of $total listed files, about $gib of 1400 GiB"
fi
echo "free disk: $(df -h --output=avail "$PUSH_DIR" | tail -1 | tr -d ' ')"
if [ -f "$LOG" ]; then
  age=$(( now - $(stat -c %Y "$LOG") ))
  echo "log last written ${age}s ago"
  if pgrep -f "scripts/upload-dataset-git.sh" > /dev/null && [ "$age" -gt 1500 ]; then echo "WARNING: running but the log has been quiet for over 25 minutes (a batch takes about 7)"; fi
fi
