#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/lecun-1998/work
cd "$BASE"

export OMP_THREAD_LIMIT=1
tesseract --version | head -n 2

$PY - << 'EOF'
import fitz, subprocess, concurrent.futures

doc = fitz.open("lecun1998.pdf")
print("Pre-rendering", len(doc), "pages...")
pngs = [doc[i].get_pixmap(dpi=150).tobytes("png") for i in range(len(doc))]

def ocr_page(idx_png):
    i, png_bytes = idx_png
    p = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "3"],
        input=png_bytes,
        capture_output=True,
        check=True,
    )
    return i, f"--- PAGE {i+1} ---\n" + p.stdout.decode("utf-8", errors="replace")

results = [None] * len(pngs)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    for i, txt in ex.map(ocr_page, enumerate(pngs)):
        results[i] = txt

full = "\n".join(results)
open("lecun1998.txt", "w", encoding="utf-8").write(full)
print("Saved OCR lecun1998.txt! lines=", len(full.splitlines()), "chars=", len(full))
print("\n".join(full.splitlines()[:60]))
EOF
echo "FINISHED"
