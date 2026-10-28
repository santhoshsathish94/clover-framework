#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/vaswani-2017
mkdir -p "$BASE/repo" "$BASE/work" "$BASE/work/tex"
cd "$BASE/work"

echo "=== 1. Fetch Vaswani et al. (2017) arXiv:1706.03762 PDF & TeX source ==="
curl -skL --max-time 30 -A "Mozilla/5.0" "https://arxiv.org/pdf/1706.03762.pdf" -o vaswani2017.pdf
curl -skL --max-time 30 -A "Mozilla/5.0" "https://arxiv.org/e-print/1706.03762" -o vaswani2017_src.tar.gz || true
if [ -f vaswani2017_src.tar.gz ]; then
  tar -xzf vaswani2017_src.tar.gz -C "$BASE/work/tex" 2>/dev/null || true
  ls -la "$BASE/work/tex"
fi

$PY - << 'EOF'
import fitz
doc = fitz.open("vaswani2017.pdf")
print("Pages:", len(doc))
text = [f"--- PAGE {i+1} ---\n" + page.get_text("text") for i, page in enumerate(doc)]
full = "\n".join(text)
open("vaswani2017.txt", "w", encoding="utf-8").write(full)
print("Extracted lines:", len(full.splitlines()), "chars:", len(full))
print("\n".join(full.splitlines()[:50]))
EOF

echo "=== 2. Clone reference repositories ==="
cd "$BASE/repo"
for r in \
  "harvardnlp/annotated-transformer" \
  "jadore801120/attention-is-all-you-need-pytorch" \
  "hyunwoongko/transformer"; do
  name=$(echo "$r" | tr '/' '-')
  if [ ! -d "$name" ]; then
    echo "Cloning $r -> $name"
    git clone --depth 1 "https://github.com/$r.git" "$name"
  fi
done
echo "FINISHED"
