#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. Check parent dir /pdf/mit/ai/ on bitsavers for index file ==="
curl -sL "http://bitsavers.trailing-edge.com/pdf/mit/ai/" > mit_ai_dir.html
cat mit_ai_dir.html

echo "=== 2. Check OpenAlex for Marvin Minsky & Seymour Papert works ==="
$PY -c "
import urllib.request, json
url = 'https://api.openalex.org/works?search=Minsky+Papert+Perceptrons&per-page=15'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for w in d.get('results', []):
    print(w.get('publication_year'), '|', w.get('title'), '|', w.get('open_access', {}).get('oa_url'))
"
