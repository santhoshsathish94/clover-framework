#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/hopfield-1982"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
for r in "in-c0/1982-hopfield-Neural-networks-paper" "pauljmello/Hopfield-Networks" "1hubert/hopfield-network"; do
  target=$(echo "$r" | tr '/' '-')
  if [ ! -d "$target" ]; then
    git clone --depth 1 "https://github.com/$r.git" "$target"
  fi
done

echo "=== Search GitHub for top-starred Hopfield network repos ==="
$PY -c "
import urllib.request, json
url = 'https://api.github.com/search/repositories?q=hopfield+network+language:python&sort=stars&order=desc&per_page=8'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    data = json.loads(r.read().decode())
for item in data.get('items', []):
    print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"

echo "=== Extract Hopfield (1982) primary paper PDF ==="
cd "$BASE/work"
cp "$BASE/repo/in-c0-1982-hopfield-Neural-networks-paper/pnas00447-0135.pdf" hopfield1982.pdf
PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('hopfield1982.pdf')
print('Pages:', len(doc))
with open('hopfield1982.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n' + p.get_text())
"
wc -l -c hopfield1982.txt
head -n 60 hopfield1982.txt

echo "=== Inspect cloned Hopfield repos ==="
cd "$BASE/repo"
for d in in-c0-1982-hopfield-Neural-networks-paper pauljmello-Hopfield-Networks 1hubert-hopfield-network; do
  echo ">>> Repo: $d ($(git -C "$d" rev-parse HEAD))"
  find "$d" -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} + 2>/dev/null || true
done
