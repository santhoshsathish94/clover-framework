#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/vaswani-2017

echo "=== 1. Inspect arXiv:1706.03762 TeX sources (including unused parameter_attention.tex & sqrt_d_trick.tex!) ==="
cd "$BASE/work/tex"
echo "--- Files in tex/ vs \\input in ms.tex ---"
grep "\\input" ms.tex || true
echo "--- Full content of sqrt_d_trick.tex ---"
cat sqrt_d_trick.tex
echo "--- Full content of parameter_attention.tex (unreferenced file in arXiv tarball!) ---"
cat parameter_attention.tex
echo "--- Author comments (%) across all .tex files ---"
grep -n "%" *.tex | head -n 60 || true

echo "=== 2. Inspect cloned reference repositories ==="
cd "$BASE/repo"
for d in *; do
  if [ -d "$d" ]; then
    echo "=== $d (commit $(git -C "$d" rev-parse --short HEAD)) ==="
    find "$d" -maxdepth 3 -not -path '*/.*'
  fi
done

echo "--- harvardnlp/annotated-transformer SublayerConnection & Embeddings ---"
grep -n -C 12 "class SublayerConnection\|class Embeddings" harvardnlp-annotated-transformer/the_annotated_transformer.py || true

echo "--- jadore801120/attention-is-all-you-need-pytorch Models.py & SubLayers.py ---"
head -n 90 jadore801120-attention-is-all-you-need-pytorch/transformer/SubLayers.py jadore801120-attention-is-all-you-need-pytorch/transformer/Models.py || true

echo "--- hyunwoongko/transformer encoder_layer.py & token_embeddings.py ---"
cat hyunwoongko-transformer/models/blocks/encoder_layer.py hyunwoongko-transformer/models/embedding/token_embeddings.py || true
echo "FINISHED"
