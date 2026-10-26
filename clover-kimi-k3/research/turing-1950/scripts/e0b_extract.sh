#!/usr/bin/env bash
set -euo pipefail

BASE="/home/usergpu/foundations-study"
mkdir -p "$BASE/pylib"
STUDY="turing-1950"
cd "$BASE/$STUDY/work"

echo "=== Check virtualenvs or install pypdf into $BASE/pylib ==="
find /home/usergpu -maxdepth 3 -name "pytest" -o -name "pip" 2>/dev/null || true
python3 -m pip install --target "$BASE/pylib" pypdf --quiet
PYTHONPATH="$BASE/pylib" python3 -c "
from pypdf import PdfReader
reader = PdfReader('turing1950.pdf')
print('Pages:', len(reader.pages))
with open('turing1950.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        f.write(f'=== PAGE {i+1} ===\n')
        f.write(page.extract_text() or '')
        f.write('\n')
"
wc -l -c turing1950.txt
head -n 45 turing1950.txt
