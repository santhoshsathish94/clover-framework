#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/minsky-papert-1969"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

$PY -c "
import urllib.request, json
url = 'https://archive.org/advancedsearch.php?q=collection:(bitsavers)+AND+(AIM-140+OR+AIM-156+OR+AIM-167+OR+AIM-275)&fl[]=identifier,title&rows=20&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
for doc in d['response'].get('docs', []):
    ident = doc['identifier']
    print('Found bitsavers archive.org ident:', ident, doc.get('title'))
    # Fetch file list
    furl = f'https://archive.org/metadata/{ident}/files'
    with urllib.request.urlopen(urllib.request.Request(furl, headers={'User-Agent': 'CloverStudy/1.0'}), timeout=15) as r2:
        fd = json.loads(r2.read().decode())
    for f in fd.get('result', []):
        if f['name'].endswith('_djvu.txt'):
            turl = f'https://archive.org/download/{ident}/{f[\"name\"]}'
            print('  Downloading:', turl)
            urllib.request.urlretrieve(turl, f['name'])
"
ls -lh *_djvu.txt 2>/dev/null || true
for f in *_djvu.txt; do
  [ -f "$f" ] || continue
  echo "=== $f ==="
  wc -l -c "$f"
  head -n 45 "$f"
done
