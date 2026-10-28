#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. Print full abstracts of AIM-140, AIM-156, AIM-167 from aim_index.txt ==="
$PY -c "
txt = open('aim_index.txt', errors='ignore').read()
for m in ('AIM-140', 'AIM-156', 'AIM-167', 'AIM-200', 'AIM-275'):
    idx = txt.find('\n' + m + '\n')
    if idx != -1:
        print('====================================')
        print(txt[idx+1:idx+1800])
"

echo "=== 2. Check Archive.org identifiers for AIM-140, AIM-156, AIM-167 ==="
$PY -c "
import urllib.request, json
url = 'https://archive.org/advancedsearch.php?q=(AIM-140+OR+AIM-156+OR+AIM-167+OR+Linearly+Unrecognizable+Patterns)&fl[]=identifier,title,year&rows=20&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for doc in d['response'].get('docs', []):
    print(' ', doc)
"

echo "=== 3. Download AIM-140, AIM-156, AIM-167 from bitsavers or MIT DSpace ==="
for m in AIM-140 AIM-156 AIM-167; do
  u="http://bitsavers.trailing-edge.com/pdf/mit/ai/aim/${m}.pdf"
  curl -sL --connect-timeout 10 --max-time 45 "$u" -o "${m}.pdf" || true
  fsize=$(stat -c%s "${m}.pdf" 2>/dev/null || echo 0)
  head_bytes=$(head -c 4 "${m}.pdf" 2>/dev/null || true)
  echo "  $m from bitsavers: size=$fsize head=$head_bytes"
done
