#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. Check MIT AI Lab memos & academic mirrors for Minsky & Papert (1969) ==="
$PY -c "
import urllib.request, json
# Search Archive.org for any public Minsky / Papert items (e.g. bitsavers MIT AI memos or book scans)
url = 'https://archive.org/advancedsearch.php?q=creator:(Minsky)+AND+(Papert+OR+Perceptron)&fl[]=identifier,title,year&rows=25&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for doc in d['response'].get('docs', []):
    print(' ', doc)
"

URLS=(
  "https://direct.mit.edu/books/oa-monograph/pdf/2462849/9780262343930.pdf"
  "https://courses.cs.washington.edu/courses/cse599s/14sp/scribes/lecture12/minsky_papert.pdf"
  "https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/MinskyPapert.pdf"
  "https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2015/pdfs/MinskyPapert.pdf"
  "https://web.archive.org/web/2021id_/https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/MinskyPapert.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 10 --max-time 45 -A "Mozilla/5.0" "$u" -o cand.pdf; then
    fsize=$(stat -c%s cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
  fi
done

echo "=== 2. Search GitHub for Minsky / Perceptron / XOR / Parity repos ==="
$PY -c "
import urllib.request, json
for q in ('Minsky+Perceptron', 'Perceptrons+Papert', 'perceptron+parity+order'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=8'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
