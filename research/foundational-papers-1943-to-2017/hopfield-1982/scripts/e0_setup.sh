#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/hopfield-1982"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/work"
echo "=== 1. Download Hopfield (1982) PNAS 79(8): 2554-2558 primary paper ==="
URLS=(
  "https://www.pnas.org/doi/pdf/10.1073/pnas.79.8.2554"
  "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC346238&blobtype=pdf"
  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/pdf/pnas00447-0135.pdf"
  "https://www.cs.toronto.edu/~hinton/csc321/readings/hopfield82.pdf"
  "https://web.archive.org/web/2022id_/https://www.pnas.org/doi/pdf/10.1073/pnas.79.8.2554"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o h1982_cand.pdf; then
    fsize=$(stat -c%s h1982_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 h1982_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv h1982_cand.pdf hopfield1982.pdf
      echo "$u" > hopfield1982_url.txt
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('hopfield1982.pdf')
print('Pages:', len(doc))
with open('hopfield1982.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n' + p.get_text())
"
      wc -l -c hopfield1982.txt
      head -n 55 hopfield1982.txt
      break
    fi
  fi
done

echo "=== 2. Search GitHub for Hopfield Network (1982) reference repos ==="
$PY -c "
import urllib.request, json
for q in ('hopfield+network+1982+language:python', 'hopfield+network+associative+memory+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=6'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
