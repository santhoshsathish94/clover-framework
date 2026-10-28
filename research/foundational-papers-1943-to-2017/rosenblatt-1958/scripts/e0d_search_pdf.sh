#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/rosenblatt-1958"
BASE="/root/dm-study/$STUDY"
PY="/root/ncm-cpu/venv/bin/python"
PYLIB="/root/dm-study/pebbling/pylib"
cd "$BASE/work"

$PY -c "
import urllib.request, urllib.parse, json, re

# 1. Search Archive.org for creator:Rosenblatt
url = 'https://archive.org/advancedsearch.php?q=creator:(Rosenblatt)&fl[]=identifier,title,year&rows=30&output=json'
req = urllib.request.Request(url, headers={'User-Agent': 'CloverStudy/1.0'})
with urllib.request.urlopen(req, timeout=20) as r:
    d = json.loads(r.read().decode())
print('=== Archive.org creator:(Rosenblatt) ===')
for doc in d['response']['docs']:
    print(' ', doc)

# 2. Search DuckDuckGo HTML for 'The Perceptron: A Probabilistic Model for Information Storage' filetype:pdf
ddg = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote('\"The Perceptron: A Probabilistic Model\" Rosenblatt pdf')
req2 = urllib.request.Request(ddg, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req2, timeout=20) as r:
    html = r.read().decode('utf-8', errors='ignore')
links = re.findall(r'uddg=([^\"&]+)', html)
print('=== DuckDuckGo links ===')
for l in links[:15]:
    print(' ', urllib.parse.unquote(l))
"
