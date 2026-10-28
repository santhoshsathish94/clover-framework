#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

URLS=(
  "https://web.archive.org/web/20220120042254id_/https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf"
  "https://web.archive.org/web/2021id_/https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf"
  "https://web.archive.org/web/2020id_/http://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf"
  "https://web.archive.org/web/2021id_/https://websites.umich.edu/~jhalber/courses/380/f18/readings/Rosenblatt1958.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o r1958_cand.pdf; then
    fsize=$(stat -c%s r1958_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 r1958_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv r1958_cand.pdf rosenblatt1958.pdf
      echo "$u" > rosenblatt1958_url.txt
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('rosenblatt1958.pdf')
print('Pages:', len(doc))
with open('rosenblatt1958.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n')
        f.write(page.get_text())
"
      wc -l -c rosenblatt1958.txt
      head -n 60 rosenblatt1958.txt
      break
    fi
  fi
done
