#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/sutton-1988
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/work"

echo "=== 1. Fetch Sutton (1988) PDF ==="
for url in \
  "http://incompleteideas.net/papers/sutton-88-with-erratum.pdf" \
  "https://incompleteideas.net/papers/sutton-88-with-erratum.pdf" \
  "https://link.springer.com/content/pdf/10.1007/BF00115009.pdf"; do
  echo "Trying $url"
  if curl -sL --max-time 30 -A "Mozilla/5.0" "$url" -o candidate.pdf; then
    sz=$(stat -c%s candidate.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 candidate.pdf 2>/dev/null || echo "")
    echo "  size=$sz head=$head_bytes"
    if [ "$sz" -gt 50000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv candidate.pdf sutton1988.pdf
      break
    fi
  fi
done

if [ -f sutton1988.pdf ]; then
  $PY - << 'EOF'
import fitz
doc = fitz.open("sutton1988.pdf")
print("Pages:", len(doc))
text = []
for i, page in enumerate(doc):
    text.append(f"--- PAGE {i+1} ---\n" + page.get_text("text"))
full = "\n".join(text)
open("sutton1988.txt", "w", encoding="utf-8").write(full)
print("Extracted lines:", len(full.splitlines()), "chars:", len(full))
print("\n".join(full.splitlines()[:50]))
EOF
fi

echo "=== 2. Search GitHub for Sutton 1988 TD(lambda) random walk repos ==="
$PY - << 'EOF'
import urllib.request, json
queries = [
    "sutton+1988+temporal+differences",
    "sutton+1988+random+walk+td",
    "td+lambda+sutton+1988",
]
seen = set()
for q in queries:
    url = f"https://api.github.com/search/repositories?q={q}&per_page=8"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"Query {q}: total={data.get('total_count')}")
        for item in data.get("items", []):
            fn = item["full_name"]
            if fn not in seen:
                seen.add(fn)
                print(f"  {fn} (stars={item['stargazers_count']}): {item.get('description')}")
EOF
echo "FINISHED"
