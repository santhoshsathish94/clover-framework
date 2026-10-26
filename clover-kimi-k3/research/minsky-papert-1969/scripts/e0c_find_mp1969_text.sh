#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
for r in "TaraAlcira/PerceptronsRevisited" "puppy-atom/2017-11-07_XOR_Alando"; do
  target=$(echo "$r" | tr '/' '-')
  if [ ! -d "$target" ]; then
    git clone --depth 1 "https://github.com/$r.git" "$target"
  fi
done

for d in TaraAlcira-PerceptronsRevisited puppy-atom-2017-11-07_XOR_Alando; do
  echo ">>> Repo: $d ($(git -C "$d" rev-parse HEAD))"
  find "$d" -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} + 2>/dev/null || true
done

echo "=== Check bitsavers MIT AI Lab Memos (1966-1970) for Minsky & Papert Perceptrons memos ==="
cd "$BASE/work"
curl -sL "http://bitsavers.trailing-edge.com/pdf/mit/ai/aim/" | grep -iE "minsky|papert|perceptron|AIM-1[0-9]{2}" | head -n 40 || true

echo "=== Check OpenAlex / Crossref / Wayback for Minsky & Papert (1969/1988/2017) Perceptrons chapters ==="
$PY -c "
import urllib.request, json
url = 'https://api.openalex.org/works?search=Perceptrons+Introduction+Computational+Geometry+Minsky+Papert&per-page=8'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for w in d.get('results', []):
    print(w.get('id'), w.get('title'), w.get('publication_year'), w.get('open_access'))
"
