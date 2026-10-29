#!/bin/bash
# Downloads the public DSpark draft checkpoint for Kimi K3 (config, weights) into $DSPARK_DIR, resuming if interrupted.
# Lists the files first and stops without downloading if the total is over 30 GB. Progress: fetch.log; file list: files.txt (in the same folder).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
REPO=Inferact/Kimi-K3-DSpark
mkdir -p "$DSPARK_DIR" && cd "$DSPARK_DIR"
curl -sL -m 60 "https://huggingface.co/api/models/$REPO/tree/main?recursive=1" -o tree.json
python3 - <<'EOF' > files.txt
import json
for e in json.load(open("tree.json")):
    if e.get("type") == "file":
        print(e["path"], e.get("size", 0))
EOF
total=$(awk '{s+=$2} END{print s+0}' files.txt)
echo "files: $(wc -l < files.txt), total bytes: $total" > fetch.log
if [ "$total" -eq 0 ]; then echo "no files listed (gated or missing?): $(head -c 300 tree.json)" >> fetch.log; exit 1; fi
if [ "$total" -gt 30000000000 ]; then echo "over 30 GB, not downloading" >> fetch.log; exit 1; fi
while read -r path size; do
  mkdir -p "$(dirname "$path")"
  curl -sL -C - --retry 5 --retry-delay 5 -o "$path" "https://huggingface.co/$REPO/resolve/main/$path"
  echo "$(date +%H:%M:%S) $path expected $size got $(stat -c %s "$path")" >> fetch.log
done < files.txt
echo "all done $(date)" >> fetch.log
