#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dm-study/pebbling/pylib:${PYTHONPATH:-}
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/krizhevsky-2012
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/work"

echo "=== 1. Fetch Krizhevsky, Sutskever & Hinton (2012) NIPS PDF ==="
for url in \
  "https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" \
  "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" \
  "https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf"; do
  echo "Trying $url"
  if curl -skL --max-time 30 -A "Mozilla/5.0" "$url" -o candidate.pdf; then
    sz=$(stat -c%s candidate.pdf 2>/dev/null || echo 0)
    head_bytes=$(head -c 4 candidate.pdf 2>/dev/null || echo "")
    echo "  size=$sz head=$head_bytes"
    if [ "$sz" -gt 100000 ] && [ "$head_bytes" = "%PDF" ]; then
      mv candidate.pdf krizhevsky2012.pdf
      break
    fi
  fi
done

if [ -f krizhevsky2012.pdf ]; then
  $PY - << 'EOF'
import fitz
doc = fitz.open("krizhevsky2012.pdf")
print("Pages:", len(doc))
text = []
for i, page in enumerate(doc):
    text.append(f"--- PAGE {i+1} ---\n" + page.get_text("text"))
full = "\n".join(text)
open("krizhevsky2012.txt", "w", encoding="utf-8").write(full)
print("Extracted lines:", len(full.splitlines()), "chars:", len(full))
print("\n".join(full.splitlines()[:55]))
EOF
fi

echo "=== 2. Search GitHub for Krizhevsky 2012 AlexNet (LRN, 2-stream groups, PCA color aug) ==="
$PY - << 'EOF'
import urllib.request, json
for q in ["dansuh17+alexnet-pytorch", "alexnet+krizhevsky+2012+lrn", "alexnet+local+response+normalization+pca"]:
    url = f"https://api.github.com/search/repositories?q={q}&per_page=6"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"Query {q}: total={data.get('total_count')}")
        for it in data.get("items", []):
            print(f"  {it['full_name']} (stars={it['stargazers_count']}): {it.get('description')}")
EOF
echo "FINISHED"
