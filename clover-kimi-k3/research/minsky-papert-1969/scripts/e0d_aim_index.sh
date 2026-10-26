#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

echo "=== 1. Check OpenAlex for Minsky & Papert ==="
# Search DSpace MIT API or web for Minsky Papert Perceptrons memos:
$PY -c "
import urllib.request, json
url = 'https://api.openalex.org/works?filter=authorships.author.id:A5026126119&per-page=50'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode())
    for w in d.get('results', []):
        if 1965 <= (w.get('publication_year') or 0) <= 1972:
            print(w.get('publication_year'), w.get('title'), w.get('open_access', {}).get('oa_url'))
except Exception as e:
    print('OpenAlex err:', e)
"

echo "=== 2. Download Block (1970) 22-page review of Minsky & Papert (1969) and Anthony (1992) AND/OR Theorem ==="
for pair in \
  "block1970_review.pdf|https://www.sciencedirect.com/science/article/pii/S0019995870904092/pdf?md5=4ef320b9f6e5fb14c61bc9fc6e2505d9&pid=1-s2.0-S0019995870904092-main.pdf" \
  "and_or_theorem.pdf|https://www.cambridge.org/core/services/aop-cambridge-core/content/view/0097D818A6306FD3E6AE4C0AB263DAD5/S1446788700035333a.pdf/div-class-title-the-and-or-theorem-for-perceptrons-div.pdf" \
  "ext_group_invariance.pdf|https://repository.arizona.edu/bitstream/handle/10150/637027/azu_etd_mr_2019_0246_sip1_m.pdf"
do
  out="${pair%%|*}"
  u="${pair#*|}"
  echo "Trying $out from $u"
  curl -sL --connect-timeout 10 --max-time 45 -A "Mozilla/5.0" "$u" -o "$out" || true
  fsize=$(stat -c%s "$out" 2>/dev/null || echo 0)
  head_bytes=$(head -c 4 "$out" 2>/dev/null || true)
  echo "  $out: size=$fsize head=$head_bytes"
  if [ "$fsize" -gt 20000 ] && [ "$head_bytes" = "%PDF" ]; then
    PYTHONPATH="$PYLIB" $PY -c "
import fitz
doc = fitz.open('$out')
print('  Pages:', len(doc))
with open('${out%.pdf}.txt', 'w', encoding='utf-8') as f:
    for i, p in enumerate(doc):
        f.write(f'=== PAGE {i+1} ===\n' + p.get_text())
"
    wc -l -c "${out%.pdf}.txt"
  fi
done
