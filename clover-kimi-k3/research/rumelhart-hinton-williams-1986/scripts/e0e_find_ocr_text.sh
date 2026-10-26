#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rumelhart-hinton-williams-1986"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
cd "$BASE/work"

echo "=== 1. Check cached wheels in /root/.cache/pip ==="
find /root/.cache/pip -name "*.whl" 2>/dev/null || true

echo "=== 2. Search Archive.org for Rumelhart Hinton Williams 1986 ==="
$PY -c "
import urllib.request, json
for q in [
    'title:(Learning+representations+by+back-propagating+errors)',
    'title:(Learning+Internal+Representations+by+Error+Propagation)',
    'creator:(Rumelhart)+AND+(Hinton+OR+Backpropagation+OR+Propagation)'
]:
    url = f'https://archive.org/advancedsearch.php?q={q}&fl[]=identifier,title,year&rows=15&output=json'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read().decode())
    print('>>> Query:', q)
    for doc in d['response'].get('docs', []):
        print(' ', doc)
"
