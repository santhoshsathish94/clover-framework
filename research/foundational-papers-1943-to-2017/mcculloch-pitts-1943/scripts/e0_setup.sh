#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/mcculloch-pitts-1943"
BASE="/root/dm-study/$STUDY"
mkdir -p "$BASE/repo" "$BASE/work"

echo "=== 1. Check Python & PDF tools on k3 ==="
which pdftotext || true
PY="/root/ncm-cpu/venv/bin/python"
PYTHONPATH="/root/dm-study/pebbling/pylib:/root/dm-study/eyeriss/pylib" $PY -c "
for mod in ('fitz', 'pypdf', 'pdfminer'):
    try:
        __import__(mod)
        print('available:', mod)
    except Exception as e:
        print('missing:', mod)
"

echo "=== 2. Download McCulloch & Pitts (1943) primary paper ==="
cd "$BASE/work"
URLS=(
  "https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf"
  "https://home.csulb.edu/~cwallis/382/readings/482/mcculloch.pitts.1943.pdf"
  "https://jontalle.web.engr.illinois.edu/uploads/410-NS.F22/McCullochPitts-43.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 45 -A "Mozilla/5.0" "$u" -o mp1943_cand.pdf; then
    fsize=$(stat -c%s mp1943_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 mp1943_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv mp1943_cand.pdf mp1943.pdf
      echo "Downloaded valid PDF from $u ($fsize bytes)"
      echo "$u" > mp1943_url.txt
      break
    fi
  fi
done

if [ -f mp1943.pdf ]; then
  PYTHONPATH="/root/dm-study/pebbling/pylib:/root/dm-study/eyeriss/pylib" $PY -c "
import fitz
doc = fitz.open('mp1943.pdf')
print('Pages:', len(doc))
with open('mp1943.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n')
        f.write(page.get_text())
"
  wc -l -c mp1943.txt
  head -n 45 mp1943.txt
fi

echo "=== 3. Search GitHub for McCulloch-Pitts reference repos ==="
$PY -c "
import urllib.request, json
url = 'https://api.github.com/search/repositories?q=McCulloch+Pitts+neuron+language:python&sort=stars&order=desc&per_page=8'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    data = json.loads(r.read().decode())
for item in data.get('items', []):
    print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
