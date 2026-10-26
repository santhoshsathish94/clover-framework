#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/krizhevsky-2012
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/repo"

$PY - << 'EOF'
import urllib.request, json, subprocess, os

queries = ["alexnet+pytorch+lrn", "alexnet+2012+imagenet", "cuda-convnet"]
repos = []
seen = set()
for q in queries:
    url = f"https://api.github.com/search/repositories?q={q}&per_page=5"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read().decode())
        for it in d.get("items", []):
            fn = it["full_name"]
            if fn not in seen and it.get("size", 0) < 50000:
                seen.add(fn)
                repos.append((fn, it["stargazers_count"], it.get("description")))
                print(f"Found: {fn} (stars={it['stargazers_count']}): {it.get('description')}")

for fn, _, _ in repos[:4]:
    dname = fn.replace("/", "-")
    if not os.path.exists(dname):
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{fn}.git", dname], check=False)
EOF

for d in *; do
  if [ -d "$d" ]; then
    echo "=== $d (commit $(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo none)) ==="
    find "$d" -maxdepth 3 -not -path '*/.*'
    for f in $(find "$d" -maxdepth 3 -name "*.py" | sort | head -n 6); do
      echo "--- $f ---"
      head -n 80 "$f"
    done
  fi
done
echo "FINISHED"
