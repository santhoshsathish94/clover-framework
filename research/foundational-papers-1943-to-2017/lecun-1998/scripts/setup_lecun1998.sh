#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/lecun-1998
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/work"

echo "=== 1. Fetch LeCun, Bottou, Bengio & Haffner (1998) PDF ==="
for url in \
  "http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf" \
  "https://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf" \
  "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf" \
  "http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"; do
  echo "Trying $url"
  if curl -sL --max-time 45 -A "Mozilla/5.0" "$url" -o candidate.pdf; then
    sz=$(stat -c%s candidate.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 candidate.pdf 2>/dev/null || echo "")
    echo "  size=$sz head=$head_bytes"
    if [ "$sz" -gt 200000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv candidate.pdf lecun1998.pdf
      break
    fi
  fi
done

if [ -f lecun1998.pdf ]; then
  $PY - << 'EOF'
import fitz
doc = fitz.open("lecun1998.pdf")
print("Pages:", len(doc))
text = []
for i, page in enumerate(doc):
    text.append(f"--- PAGE {i+1} ---\n" + page.get_text("text"))
full = "\n".join(text)
open("lecun1998.txt", "w", encoding="utf-8").write(full)
print("Extracted lines:", len(full.splitlines()), "chars:", len(full))
print("\n".join(full.splitlines()[:55]))
EOF
fi

echo "=== 2. Search GitHub for LeCun 1998 LeNet-5 (authentic C3 table, RBF, GTN) repos ==="
$PY - << 'EOF'
import urllib.request, json
queries = [
    "lenet-5+1998+rbf",
    "lecun+1998+lenet5",
    "lenet5+c3+table+rbf",
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
