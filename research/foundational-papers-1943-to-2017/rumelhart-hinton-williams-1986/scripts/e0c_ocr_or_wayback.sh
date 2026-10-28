#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
cd "$BASE/work"

echo "=== 1. Find all pylib dirs under /root/dm-study ==="
find /root/dm-study -maxdepth 3 -name "*ocr*" -o -name "pylib" 2>/dev/null || true
ALL_PYLIB=$(find /root/dm-study -maxdepth 2 -name "pylib" | tr '\n' ':')
echo "ALL_PYLIB=$ALL_PYLIB"

echo "=== 2. Try Wayback Machine for PDPVolIChapt8.pdf and backprop_old.pdf ==="
for u in \
  "https://web.archive.org/web/2021id_/https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapt8.pdf" \
  "https://web.archive.org/web/2021id_/http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf" \
  "https://web.archive.org/web/2020id_/http://www.cs.toronto.edu/~hinton/absps/pdp8.pdf"
do
  echo "Trying $u"
  curl -sL --connect-timeout 15 --max-time 60 "$u" -o wb_cand.pdf || true
  fsize=$(stat -c%s wb_cand.pdf 2>/dev/null || echo 0)
  head_bytes=$(head -c 4 wb_cand.pdf 2>/dev/null || true)
  echo "  size=$fsize head=$head_bytes"
  if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
    PYTHONPATH="$ALL_PYLIB" $PY -c "
import fitz
doc = fitz.open('wb_cand.pdf')
txt = '\n'.join(f'=== PAGE {i+1} ===\n' + p.get_text() for i, p in enumerate(doc))
print('  pages=', len(doc), 'text_len=', len(txt), 'sample=', repr(txt[:250]))
if len(txt) > 1000:
    open('rhw1986_text.txt', 'w', encoding='utf-8').write(txt)
"
    if [ -f rhw1986_text.txt ]; then
      wc -l -c rhw1986_text.txt
      head -n 60 rhw1986_text.txt
      break
    fi
  fi
done
