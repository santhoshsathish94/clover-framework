#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"

echo "=== 1. Read dmoews-logic-theorist/README.md and transcription.txt header ==="
cat "$BASE/repo/dmoews-logic-theorist/README.md"
echo "--- transcription.txt (first 120 lines) ---"
head -n 120 "$BASE/repo/dmoews-logic-theorist/transcription.txt"

echo "=== 2. Check if any pages of cand.pdf have text or if we can fetch IRE / CMU OCR PDF ==="
cd "$BASE/work"
PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('cand.pdf')
non_empty = [i for i, p in enumerate(doc) if p.get_text().strip()]
print('Pages with embedded text in cand.pdf:', non_empty)
"

which tesseract pdftotext || true

# Check additional URLs for IRE IT-2(3) 1956 or 1957 WJCC paper with embedded text:
URLS2=(
  "https://ColorsOfLogic.org/lt1956.pdf"
  "http://shelf1.library.cmu.edu/IMLS/MindModels/logictheorymachine.pdf"
  "https://iiif.library.cmu.edu/file/Simon_box00063_fld04804_bdl0001_doc0001/Simon_box00063_fld04804_bdl0001_doc0001.pdf"
  "https://iiif.library.cmu.edu/file/Newell_box00057_fld04339_bdl0001_doc0001/Newell_box00057_fld04339_bdl0001_doc0001.pdf"
)
for u in "${URLS2[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 10 --max-time 30 -A "Mozilla/5.0" "$u" -o cand2.pdf; then
    fsize=$(stat -c%s cand2.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 cand2.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
  fi
done
