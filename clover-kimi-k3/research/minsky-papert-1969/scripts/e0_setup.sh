#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/work"
echo "=== 1. Search Archive.org for Minsky & Papert (1969) Perceptrons ==="
$PY -c "
import urllib.request, json
url = 'https://archive.org/advancedsearch.php?q=title:(Perceptrons)+AND+creator:(Minsky)&fl[]=identifier,title,year&rows=15&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for doc in d['response'].get('docs', []):
    print(' ', doc)
"

echo "=== 2. Check Archive.org files for any matching identifier ==="
$PY -c "
import urllib.request, json
for ident in ('perceptronsintro0000mins', 'perceptronsintro00mins', 'perceptrons00mins'):
    try:
        url = f'https://archive.org/metadata/{ident}/files'
        req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode())
        print('>>> ident:', ident)
        for f in d.get('result', []):
            if 'txt' in f['name'] or 'pdf' in f['name'] or 'djvu' in f['name']:
                print('  ', f['name'], f.get('size'), f.get('private'))
    except Exception as e:
        print(ident, e)
"

echo "=== 3. Search GitHub for Minsky-Papert Perceptrons / Group Invariance / Predicate Order repos ==="
$PY -c "
import urllib.request, json
for q in ('Minsky+Papert+Perceptrons+language:python', 'Minsky+Papert+parity+connectedness', 'perceptron+xor+minsky+language:python'):
    print('--- Query:', q)
    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=6'
    req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read().decode())
    for item in data.get('items', []):
        print(f\"{item['full_name']} | stars={item['stargazers_count']} | {item['html_url']} | {item.get('description','')}\")
"
