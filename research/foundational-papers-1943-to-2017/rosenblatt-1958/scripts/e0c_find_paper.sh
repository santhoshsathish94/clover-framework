#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"

echo "=== 1. Check vukrosic-rosenblatt-from-scratch and kaisdukes README ==="
cat "$BASE/repo/vukrosic-rosenblatt-from-scratch/README.md"
cat "$BASE/repo/kaisdukes-rosenblatt-perceptron/README.md" | head -n 90

echo "=== 2. Query OpenAlex, Semantic Scholar, and Archive.org for DOI 10.1037/h0042519 ==="
cd "$BASE/work"
$PY -c "
import urllib.request, json
for api in [
    'https://api.openalex.org/works/https://doi.org/10.1037/h0042519',
    'https://api.semanticscholar.org/graph/v1/paper/DOI:10.1037/h0042519?fields=title,year,openAccessPdf,externalIds,url',
    'https://archive.org/advancedsearch.php?q=title:(Perceptron)+AND+creator:(Rosenblatt)&fl[]=identifier,title,year&rows=15&output=json'
]:
    print('>>> API:', api)
    try:
        req = urllib.request.Request(api, headers={'User-Agent': 'CloverStudy/1.0'})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        if 'locations' in d:
            print('  best_oa_location:', d.get('best_oa_location'))
            for loc in d.get('locations', []):
                if loc.get('pdf_url'):
                    print('  pdf_url:', loc.get('pdf_url'))
        elif 'response' in d:
            for doc in d['response'].get('docs', []):
                print(' ', doc)
        else:
            print(' ', d)
    except Exception as e:
        print('  err:', e)
"
