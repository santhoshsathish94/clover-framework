#!/usr/bin/env bash
set -euo pipefail

STUDY="mcculloch-pitts-1943"
BASE="/home/usergpu/foundations-study/$STUDY"
PY="python3"
mkdir -p "$BASE/repo" "$BASE/work"

echo "=== Host & GPU check ==="
hostname
df -h /home/usergpu | tail -n 1
$PY --version

cd "$BASE/repo"
if [ ! -d "kaisdukes-mp" ]; then
  git clone --depth 1 https://github.com/kaisdukes/mcculloch-pitts-neuron.git kaisdukes-mp
fi
if [ ! -d "codeforge-mp" ]; then
  git clone --depth 1 https://github.com/CodeForge-00/mcculloch_pitts_neuron.git codeforge-mp
fi

echo "=== Repo 1: kaisdukes/mcculloch-pitts-neuron ==="
cd "$BASE/repo/kaisdukes-mp"
echo "Commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
find . -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} +
for f in $(find . -maxdepth 3 -name "*.py" -o -name "*.md" | sort); do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done

echo "=== Run Repo 1 files (in a work copy) ==="
rm -rf "$BASE/work/kaisdukes_run"
cp -r "$BASE/repo/kaisdukes-mp" "$BASE/work/kaisdukes_run"
cd "$BASE/work/kaisdukes_run"
for pyf in $(find . -maxdepth 2 -name "*.py" | sort); do
  echo ">>> Running $pyf"
  MPLBACKEND=Agg $PY "$pyf" || echo "Exit code: $?"
done

echo "=== Repo 2: CodeForge-00/mcculloch_pitts_neuron ==="
cd "$BASE/repo/codeforge-mp"
echo "Commit: $(git rev-parse HEAD) ($(git log -1 --format=%ci))"
find . -maxdepth 3 -not -path '*/.*' -type f -exec wc -l {} +
for f in $(find . -maxdepth 3 -name "*.py" -o -name "*.md" | sort); do
  echo "--- FILE: $f ---"
  cat "$f"
  echo ""
done

echo "=== Run Repo 2 files (in a work copy) ==="
rm -rf "$BASE/work/codeforge_run"
cp -r "$BASE/repo/codeforge-mp" "$BASE/work/codeforge_run"
cd "$BASE/work/codeforge_run"
for pyf in $(find . -maxdepth 2 -name "*.py" | sort); do
  echo ">>> Running $pyf"
  MPLBACKEND=Agg $PY "$pyf" || echo "Exit code: $?"
done
