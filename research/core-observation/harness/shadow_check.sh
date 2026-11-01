#!/usr/bin/env bash
# Module names that exist both beside a runtime script and in a folder sys.path puts ahead of it.
# Usage: shadow_check.sh <runtime folder> <folder searched first>...
set -u
top=$1; shift
for d in "$@"; do
  [ -d "$d" ] || continue
  for f in "$d"/*.py; do
    b=$(basename "$f")
    if [ -f "$top/$b" ]; then
      if cmp -s "$f" "$top/$b"; then s=same; else s=DIFFERENT; fi
      echo "$b: $d shadows $top ($s)"
    fi
  done
done
echo "checked: $*"
