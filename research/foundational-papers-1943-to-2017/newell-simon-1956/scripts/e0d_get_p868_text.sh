#!/usr/bin/env bash
set -euo pipefail

STUDY="foundations/newell-simon-1956"
BASE="/root/dm-study/$STUDY"
mkdir -p "$BASE/work" "$BASE/repo"

cd "$BASE/work"
echo "=== Download P-868 full text from archive.org ==="
curl -sL "https://archive.org/download/bitsavers_randiplP86ineJul56_3534001/P-868_The_Logic_Theory_Machine_Jul56_djvu.txt" -o p868_1956_full.txt
wc -l -c p868_1956_full.txt
head -n 80 p868_1956_full.txt

cd "$BASE/repo"
if [ ! -d "dmoews-ipl-v-lt" ]; then
  git clone --depth 1 https://github.com/dmoews/ipl-v-logic-theorist.git dmoews-ipl-v-lt
fi
echo ">>> dmoews-ipl-v-lt ($(git -C dmoews-ipl-v-lt rev-parse HEAD))"
find dmoews-ipl-v-lt -maxdepth 2 -not -path '*/.*' -type f -exec wc -l {} +
