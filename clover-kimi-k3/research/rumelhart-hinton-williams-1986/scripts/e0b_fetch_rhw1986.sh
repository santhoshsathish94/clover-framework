#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib:/root/dm-study/eyeriss/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
for r in "CodeForge-00/rumelhart_backpropagation" "Kushagra514/Backpropagation-from-scratch-Rumelhart-Hinton-Williams-1986-" "karpathy/micrograd"; do
  target=$(echo "$r" | tr '/' '-')
  if [ ! -d "$target" ]; then
    git clone --depth 1 "https://github.com/$r.git" "$target"
  fi
done

for d in CodeForge-00-rumelhart_backpropagation Kushagra514-Backpropagation-from-scratch-Rumelhart-Hinton-Williams-1986- karpathy-micrograd; do
  echo ">>> Repo: $d ($(git -C "$d" rev-parse HEAD))"
  find "$d" -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} + 2>/dev/null || true
done

echo "=== Check OCR / embedded text sources for Rumelhart, Hinton & Williams (1986) ==="
cd "$BASE/work"
URLS=(
  "https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf"
  "https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapt8.pdf"
  "https://apps.dtic.mil/sti/tr/pdf/ADA164453.pdf"
  "https://www.nature.com/articles/323533a0.pdf"
)
for idx in "${!URLS[@]}"; do
  u="${URLS[$idx]}"
  echo "Trying $u"
  curl -sL --connect-timeout 10 --max-time 45 -A "Mozilla/5.0" "$u" -o "cand_${idx}.pdf" || true
  fsize=$(stat -c%s "cand_${idx}.pdf" 2>/dev/null || echo 0)
  head_bytes=$(head -c 4 "cand_${idx}.pdf" 2>/dev/null || true)
  echo "  cand_${idx}.pdf: size=$fsize head=$head_bytes"
  if [ "$fsize" -gt 30000 ] && [ "$head_bytes" = "%PDF" ]; then
    PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('cand_${idx}.pdf')
txt = '\n'.join(f'=== PAGE {i+1} ===\n' + p.get_text() for i, p in enumerate(doc))
print('  pages=', len(doc), 'text_len=', len(txt), 'sample=', repr(txt[:250]))
if len(txt) > 2000:
    open('cand_${idx}.txt', 'w', encoding='utf-8').write(txt)
"
  fi
done

echo "=== Check if rapidocr_onnxruntime is in pylib to OCR rhw1986.pdf (4 pages) ==="
PYTHONPATH="$PYLIB" $PY -c "
import fitz
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()
doc = fitz.open('rhw1986.pdf')
full_txt = []
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes('png')
    res, _ = ocr(img_bytes)
    lines = [r[1] for r in (res or [])]
    print(f'Page {i+1}: {len(lines)} OCR lines')
    full_txt.append(f'=== PAGE {i+1} ===\n' + '\n'.join(lines))
open('rhw1986_nature_ocr.txt', 'w', encoding='utf-8').write('\n'.join(full_txt))
"
wc -l -c rhw1986_nature_ocr.txt
head -n 60 rhw1986_nature_ocr.txt
