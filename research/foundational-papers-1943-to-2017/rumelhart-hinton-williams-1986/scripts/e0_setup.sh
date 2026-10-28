#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/work"
echo "=== 1. Download Rumelhart, Hinton & Williams (1986) Nature 323: 533-536 primary paper ==="
URLS=(
  "https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf"
  "https://web.archive.org/web/2023id_/https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf"
  "https://web.archive.org/web/2022id_/http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf"
  "https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf"
  "https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapt8.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o rhw1986_cand.pdf; then
    fsize=$(stat -c%s rhw1986_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 rhw1986_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 30000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv rhw1986_cand.pdf rhw1986.pdf
      echo "$u" > rhw1986_url.txt
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('rhw1986.pdf')
print('Pages:', len(doc))
with open('rhw1986.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n' + p.get_text())
"
      wc -l -c rhw1986.txt
      head -n 55 rhw1986.txt
      break
    fi
  fi
done

echo "=== 2. Search GitHub for Rumelhart Hinton Williams 1986 / Backpropagation reference repos ==="
$PY -c "
import urllib.request, json
for q in ('Rumelhart+Hinton+Williams+1986+language:python', 'karpathy/micrograd', 'family+trees+Hinton+1986+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=6'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
