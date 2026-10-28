#!/usr/bin/env bash
set -euo pipefail

STUDY="turing-1950"
BASE="/home/usergpu/foundations-study/$STUDY"
mkdir -p "$BASE/repo" "$BASE/work"

echo "=== 1. Check PDF extraction tools on gpu ==="
which pdftotext || true
python3 -c "
for mod in ('fitz', 'pypdf', 'pdfminer'):
    try:
        __import__(mod)
        print('available:', mod)
    except Exception as e:
        print('missing:', mod)
"

echo "=== 2. Download Turing (1950) primary paper ==="
cd "$BASE/work"
URLS=(
  "https://courses.cs.umbc.edu/471/papers/turing.pdf"
  "https://www.csee.umbc.edu/courses/471/papers/turing.pdf"
  "https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf"
  "https://www.ivan.research.att.com/turing.pdf"
  "https://www.cs.ox.ac.uk/activities/ieg/e-library/sources/t_article.pdf"
  "https://phil415.pbworks.com/f/TuringComputing.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 45 -A "Mozilla/5.0" "$u" -o turing1950_cand.pdf; then
    fsize=$(stat -c%s turing1950_cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 turing1950_cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv turing1950_cand.pdf turing1950.pdf
      echo "Downloaded valid PDF from $u ($fsize bytes)"
      echo "$u" > turing1950_url.txt
      break
    fi
  fi
done

if [ -f turing1950.pdf ] && which pdftotext >/dev/null 2>&1; then
  pdftotext turing1950.pdf turing1950.txt
  wc -l -c turing1950.txt
  head -n 45 turing1950.txt
fi

echo "=== 3. Search GitHub for Turing machine / Computing Machinery and Intelligence repos ==="
python3 -c "
import urllib.request, json
for q in ('universal+turing+machine+language:python', 'imitation+game+turing+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=5'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
