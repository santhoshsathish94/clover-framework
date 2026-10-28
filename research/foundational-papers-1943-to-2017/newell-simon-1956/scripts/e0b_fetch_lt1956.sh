#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
for r in "dmoews/logic-theorist" "duplys/logic-theorist" "jblattnernyc/logic-theorist-companion"; do
  name=$(basename "$r")
  target=$(echo "$r" | tr '/' '-')
  if [ ! -d "$target" ]; then
    git clone --depth 1 "https://github.com/$r.git" "$target"
  fi
done

echo "=== Check cloned repos for paper links or files ==="
for d in dmoews-logic-theorist duplys-logic-theorist jblattnernyc-logic-theorist-companion; do
  echo ">>> Repo: $d ($(git -C "$d" rev-parse HEAD))"
  find "$d" -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} + 2>/dev/null || true
  cat "$d/README"* 2>/dev/null | head -n 60 || true
done

echo "=== Try downloading Newell & Simon (1956) RAND P-868 / IRE IT-2(3) PDF ==="
cd "$BASE/work"
rm -f lt1956.pdf lt1956.txt
URLS=(
  "https://www.rand.org/content/dam/rand/pubs/papers/2005/P868.pdf"
  "https://www.rand.org/content/dam/rand/pubs/papers/2006/P868.pdf"
  "https://www.rand.org/content/dam/rand/pubs/reports/2007/R366.pdf"
  "https://iiif.library.cmu.edu/file/Newell_box00057_fld04340_bdl0001_doc0001/Newell_box00057_fld04340_bdl0001_doc0001.pdf"
  "http://bitsavers.trailing-edge.com/pdf/rand/ipl/P-868_The_Logic_Theory_Machine_Jul56.pdf"
  "http://bitsavers.informatik.uni-stuttgart.de/pdf/rand/ipl/P-868_The_Logic_Theory_Machine_Jul56.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 15 --max-time 60 -A "Mozilla/5.0" "$u" -o cand.pdf; then
    fsize=$(stat -c%s cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('cand.pdf')
print('  Pages:', len(doc))
txt = doc[0].get_text() + (doc[1].get_text() if len(doc)>1 else '')
print('  First 300 chars:', repr(txt[:300]))
if 'Logic' in txt or 'LOGIC' in txt or 'Newell' in txt or 'NEWELL' in txt:
    with open('lt1956.txt', 'w', encoding='utf-8') as f:
        for i, p in enumerate(doc):
            f.write(f'=== PAGE {i+1} ===\n')
            f.write(p.get_text())
    print('  MATCHED Logic Theory Machine!')
"
      if [ -f lt1956.txt ]; then
        mv cand.pdf lt1956.pdf
        echo "$u" > lt1956_url.txt
        wc -l -c lt1956.txt
        break
      fi
    fi
  fi
done
