#!/usr/bin/env bash
set -euo pipefail
PY=/root/ncm-cpu/venv/bin/python

BASE=/root/dm-study/foundations/sutton-1988
mkdir -p "$BASE/repo" "$BASE/work"
cd "$BASE/repo"

for r in \
  "MrShininnnnn/Temporal-Differences-Learning" \
  "vjp23/TD_Lambda" \
  "shashir/td-learning" \
  "robosangli/sutton-1988-reproduction"; do
  name=$(echo "$r" | tr '/' '-')
  if [ ! -d "$name" ]; then
    echo "Cloning $r -> $name"
    git clone --depth 1 "https://github.com/$r.git" "$name"
  fi
done

echo "=== 1. Inspect repo files ==="
for d in *; do
  if [ -d "$d" ]; then
    echo "--- $d (commit $(git -C "$d" rev-parse --short HEAD)) ---"
    find "$d" -maxdepth 3 -not -path '*/.*'
  fi
done

echo "=== 2. Run robosangli/sutton-1988-reproduction ==="
rm -rf "$BASE/work/robosangli_run"
cp -r "$BASE/repo/robosangli-sutton-1988-reproduction" "$BASE/work/robosangli_run"
cd "$BASE/work/robosangli_run"
for f in $(find . -maxdepth 2 -name "*.py" | sort); do
  echo ">>> Head of $f:"
  head -n 35 "$f"
done

echo "=== 3. Run shashir/td-learning ==="
rm -rf "$BASE/work/shashir_run"
cp -r "$BASE/repo/shashir-td-learning" "$BASE/work/shashir_run"
cd "$BASE/work/shashir_run"
for f in $(find . -maxdepth 2 -name "*.py" | sort); do
  echo ">>> Head of $f:"
  head -n 45 "$f"
done

echo "=== 4. Inspect MrShininnnnn and vjp23 ==="
for rdir in "MrShininnnnn-Temporal-Differences-Learning" "vjp23-TD_Lambda"; do
  cd "$BASE/repo/$rdir"
  echo ">>> $rdir README & Python files:"
  head -n 40 README* 2>/dev/null || true
  for f in $(find . -maxdepth 2 -name "*.py" | sort); do
    echo "--- $f ---"
    head -n 45 "$f"
  done
done
echo "FINISHED"
