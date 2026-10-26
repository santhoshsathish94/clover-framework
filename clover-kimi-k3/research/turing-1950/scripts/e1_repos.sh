#!/usr/bin/env bash
set -euo pipefail

STUDY="turing-1950"
BASE="/home/usergpu/foundations-study/$STUDY"
mkdir -p "$BASE/repo" "$BASE/work"

cd "$BASE/repo"
if [ ! -d "vukrosic-turing" ]; then
  git clone --depth 1 https://github.com/vukrosic/turing-from-scratch.git vukrosic-turing
fi
if [ ! -d "simonbreiter-utm" ]; then
  git clone --depth 1 https://github.com/simonbreiter/universal-turing-machine.git simonbreiter-utm
fi

echo "=== Repo 1: vukrosic/turing-from-scratch ==="
cd "$BASE/repo/vukrosic-turing"
echo "Commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
find . -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} +

echo "=== Repo 2: simonbreiter/universal-turing-machine ==="
cd "$BASE/repo/simonbreiter-utm"
echo "Commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
find . -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} +
for f in $(find . -maxdepth 3 -name "*.py" -o -name "*.md" | sort); do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done
