#!/bin/bash
# Does the trunk from Hugging Face (into RAM, /dev/shm) and the experts from the drive (O_DIRECT, cycle-io) slow each other down?
#   hf-trunk-concurrent.sh FIRST_LAYER LAYERS [TRUNKS]    (TRUNKS: how many trunks to download, default LAYERS)
# Part 1: the trunk download alone. Part 2: the drive reads alone. Part 3: both at the same time. Nothing the download writes touches the drive.
set -u
FIRST=${1:-13}; K=${2:-12}; T=${3:-$K}
DS=/opt/clover-k3/datasets/computed
PY=/opt/clover-k3/venv/bin/python
CYC=$HOME/probe/cycle-io
export HF_HOME=/dev/shm/hf-probe HF_XET_CACHE=/dev/shm/hf-probe/xet HF_HUB_DISABLE_TELEMETRY=1
OUT=/dev/shm/trunk-probe
LAST=$((FIRST + T - 1))

download() {   # prints seconds and bytes
    rm -rf "$OUT" "$HF_HOME"; mkdir -p "$OUT"
    local pats=""; for L in $(seq "$FIRST" "$LAST"); do pats="$pats trunk-$L/*"; done
    local t0=$(date +%s.%N)
    $PY - $pats <<'EOF'
import sys, os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Santhoshsathish94/clover-kimi-k3", repo_type="dataset", allow_patterns=sys.argv[1:], local_dir=os.environ.get("OUT", "/dev/shm/trunk-probe"), max_workers=int(os.environ.get("HF_WORKERS", "8")))
EOF
    local t1=$(date +%s.%N)
    local b=$(du -sb --exclude=.cache "$OUT" | cut -f1)
    echo "download: $b bytes in $(echo "$t1 - $t0" | bc) s = $(echo "scale=3; $b / ($t1 - $t0) / 1000000000" | bc) GB/s"
}
export OUT

echo "== 1. the trunk download alone (trunk-$FIRST..$LAST)"; download
echo "== 2. the drive alone (experts $FIRST..$((FIRST + K - 1)), O_DIRECT, 8 slots, 2 readers)"
CYCLE_DIRECT=1 $CYC $DS $K 8 2 250 0 $FIRST | grep -E "all [0-9]+ layers done|waited for experts"
echo "== 3. both at the same time"
( download ) > /dev/shm/hf-both.txt 2>&1 &
DL=$!
CYCLE_DIRECT=1 $CYC $DS $K 8 2 250 0 $FIRST | grep -E "all [0-9]+ layers done|waited for experts"
wait $DL; cat /dev/shm/hf-both.txt
rm -rf "$OUT" "$HF_HOME" /dev/shm/hf-both.txt
