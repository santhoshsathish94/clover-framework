#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/work"
echo "=== 1. Download Rosenblatt (1958) primary paper ==="
URLS=(
  "https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf"
  "https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=7b6292a7fb2d3ca9e1f0ef1f7dc8bf78ce867963"
  "https://courses.cs.umbc.edu/671/fall12/notes/papers/rosenblatt1958.pdf"
  "https://home.csulb.edu/~cwallis/382/readings/482/rosenblatt.1958.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o r1958_cand.pdf; then
    fsize=$(stat -c%s r1958_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 r1958_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('r1958_cand.pdf')
print('  Pages:', len(doc))
txt = ''
for i, page in enumerate(doc):
    txt += f'=== PAGE {i+1} ===\n' + page.get_text()
if 'PERCEPTRON' in txt.upper() or 'ROSENBLATT' in txt.upper():
    with open('rosenblatt1958.txt', 'w', encoding='utf-8') as f:
        f.write(txt)
    print('  Extracted rosenblatt1958.txt:', len(txt), 'chars')
"
      if [ -f rosenblatt1958.txt ]; then
        mv r1958_cand.pdf rosenblatt1958.pdf
        echo "$u" > rosenblatt1958_url.txt
        wc -l -c rosenblatt1958.txt
        head -n 55 rosenblatt1958.txt
        break
      fi
    fi
  fi
done

echo "=== 2. Search GitHub for Rosenblatt Perceptron reference repos ==="
$PY -c "
import urllib.request, json
for q in ('Rosenblatt+perceptron+language:python', 'Mark+I+perceptron+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=6'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
