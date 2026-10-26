#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="/root/dm-study/specmd/w2/hf_cache"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export MALLOC_ARENA_MAX=2

W2="/root/dm-study/specmd/w2"
TRACE_DIR="$W2/traces"
mkdir -p "$TRACE_DIR"

echo "=== 1. Download WikiText-2 validation prose sample via curl ==="
WT_URL="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
if [ ! -s "$TRACE_DIR/wikitext2_valid.txt" ]; then
    curl -sL "$WT_URL" -o "$TRACE_DIR/wikitext2_valid.txt"
fi
ls -lh "$TRACE_DIR/wikitext2_valid.txt"
head -n 5 "$TRACE_DIR/wikitext2_valid.txt"

echo ""
echo "=== 2. Run record_routing.py on OLMoE-1B-7B-0924 ==="
/root/ncm-cpu/venv/bin/python "$W2/record_routing.py"

echo ""
echo "=== 3. Check saved trace files ==="
ls -lh "$TRACE_DIR"
