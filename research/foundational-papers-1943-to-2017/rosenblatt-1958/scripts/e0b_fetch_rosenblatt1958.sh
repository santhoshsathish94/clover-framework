#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
for r in "vukrosic/rosenblatt-from-scratch" "kaisdukes/rosenblatt-perceptron" "Honghe/perceptron"; do
  target=$(echo "$r" | tr '/' '-')
  if [ ! -d "$target" ]; then
    git clone --depth 1 "https://github.com/$r.git" "$target"
  fi
done

for d in vukrosic-rosenblatt-from-scratch kaisdukes-rosenblatt-perceptron Honghe-perceptron; do
  echo ">>> Repo: $d ($(git -C "$d" rev-parse HEAD))"
  find "$d" -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} +
done

echo "=== Search Archive.org & academic mirrors for Rosenblatt (1958) Psychological Review paper ==="
cd "$BASE/work"
$PY -c "
import urllib.request, json
url = 'https://archive.org/advancedsearch.php?q=Rosenblatt+Perceptron+1958&fl[]=identifier,title,year&rows=15&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    data = json.loads(r.read().decode())
for doc in data.get('response', {}).get('docs', []):
    print(doc)
"

URLS=(
  "https://websites.umich.edu/~jhalber/courses/380/f18/readings/Rosenblatt1958.pdf"
  "https://www.cs.toronto.edu/~hinton/absps/perceptron.pdf"
  "https://isl.stanford.edu/~widrow/papers/j1958theperceptron.pdf"
  "https://bpb-us-e2.wpmucdn.com/websites.umich.edu/dist/a/3204/files/2018/09/Rosenblatt1958.pdf"
  "https://redwood.berkeley.edu/wp-content/uploads/2021/08/Rosenblatt1958.pdf"
  "https://www.imit.kth.se/courses/2G1522/Rosenblatt58.pdf"
  "https://cs.nyu.edu/~davise/papers/Rosenblatt1958.pdf"
  "https://apps.dtic.mil/sti/tr/pdf/AD0204076.pdf"
  "https://apps.dtic.mil/sti/tr/pdf/AD0256582.pdf"
)
for u in "${URLS[@]}"; do
  echo "Trying $u"
  if curl -sL --connect-timeout 10 --max-time 45 -A "Mozilla/5.0" "$u" -o cand.pdf; then
    fsize=$(stat -c%s cand.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 cand.pdf 2>/dev/null || true)
    echo "  size=$fsize head=$head_bytes"
    if [ "$fsize" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('cand.pdf')
print('  Pages:', len(doc))
txt = ''
for i, p in enumerate(doc):
    txt += f'=== PAGE {i+1} ===\n' + p.get_text()
print('  Text len:', len(txt), 'Sample:', repr(txt[:250]))
if len(txt) > 1000:
    with open('rosenblatt1958.txt', 'w', encoding='utf-8') as f:
        f.write(txt)
"
      if [ -f rosenblatt1958.txt ]; then
        mv cand.pdf rosenblatt1958.pdf
        echo "$u" > rosenblatt1958_url.txt
        wc -l -c rosenblatt1958.txt
        break
      fi
    fi
  fi
done
