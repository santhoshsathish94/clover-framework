#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/eyeriss/pylib:/root/dm-study/eyeriss-v2/pylib:/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. OCR rhw1986.pdf (4 pages) using /root/dm-study/eyeriss/pylib ==="
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
