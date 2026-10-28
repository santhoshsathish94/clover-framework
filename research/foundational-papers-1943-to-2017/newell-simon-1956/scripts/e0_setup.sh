#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/work"
echo "=== 1. Download Newell & Simon (1956) primary paper ==="
URLS=(
  "https://shelf1.library.cmu.edu/IMLS/MindModels/logictheorymachine.pdf"
  "http://iiif.library.cmu.edu/file/Simon_box00064_fld04894_bdl0001_doc0001/Simon_box00064_fld04894_bdl0001_doc0001.pdf"
  "https://www.rand.org/content/dam/rand/pubs/papers/2008/P868.pdf"
  "https://stacks.stanford.edu/file/druid:zd226nb4978/zd226nb4978.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o lt1956_cand.pdf; then
    fsize=$(stat -c%s lt1956_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 lt1956_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv lt1956_cand.pdf lt1956.pdf
      echo "Downloaded valid PDF from $u ($fsize bytes)"
      echo "$u" > lt1956_url.txt
      break
    fi
  fi
done

if [ -f lt1956.pdf ]; then
  PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('lt1956.pdf')
print('Pages:', len(doc))
with open('lt1956.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n')
        f.write(page.get_text())
"
  wc -l -c lt1956.txt
  head -n 55 lt1956.txt
fi

echo "=== 2. Search GitHub for Logic Theorist / Principia Mathematica prover repos ==="
$PY -c "
import urllib.request, json
for q in ('Logic+Theorist+language:python', 'Logic+Theory+Machine', 'Principia+Mathematica+prover+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=6'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
