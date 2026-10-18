#!/bin/bash
# One-time: builds the engine folder on the host and fills it from the older scattered places. Copies, never deletes (old places stay until removed by hand).
# usage: set-up-host-folder.sh STAGE_DIR [move-dataset]
#   STAGE_DIR       a copy of this repo folder's src, scripts, tools, prompts, docs (scp -r)
#   move-dataset    also moves the 1.4 TB dataset into the engine folder (same filesystem, so a rename) and leaves a symlink at the old path
set -eu
STAGE=${1:?stage dir}
ENGINE=${ENGINE_DIR:-/opt/clover-k3/clover-kimi-k3}
OLD_WORK=${OLD_WORK:-/tmp/one-layer}
OLD_DATASET=/opt/clover-k3/clover-intelegence/dataset
LOG=$ENGINE/set-up.log
mkdir -p "$ENGINE"/{src,scripts,tools,prompts,docs,models,work/inputs,work/results,work/build}
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) set up $ENGINE from $STAGE"
for d in src scripts tools prompts docs; do cp -r "$STAGE/$d/." "$ENGINE/$d/"; done
find "$ENGINE/src" "$ENGINE/scripts" "$ENGINE/tools" "$ENGINE/prompts" "$ENGINE/docs" -type f \( -name '*.c' -o -name '*.h' -o -name '*.sh' -o -name '*.py' -o -name '*.txt' -o -name '*.tsv' -o -name '*.md' \) -exec sed -i 's/\r$//' {} +
chmod +x "$ENGINE"/scripts/*.sh
# recorded inputs: tokenizer files (links), the text, the 1000 recorded ids, the pod's own chain values
cp -a "$OLD_WORK/tokfiles" "$ENGINE/work/inputs/"
cp "$OLD_WORK/text.txt" "$OLD_WORK/tok-1000.txt" "$ENGINE/work/inputs/"
cp -r "$OLD_WORK/refs" "$ENGINE/work/inputs/"
# earlier results kept as references for the checks
R=$ENGINE/work/results
[ -d "$OLD_WORK/p34" ]            && cp -r "$OLD_WORK/p34"            "$R/recorded-34-exact"
[ -d "$OLD_WORK/p34_4bit_trunk" ] && cp -r "$OLD_WORK/p34_4bit_trunk" "$R/recorded-34-use_4bit_trunk"
[ -d "$OLD_WORK/p34q4" ]          && cp -r "$OLD_WORK/p34q4"          "$R/recorded-34-decode_only_4bit_trunk_old"
[ -d "$OLD_WORK/p34a8" ]          && cp -r "$OLD_WORK/p34a8"          "$R/recorded-34-experts_8bit_activations_old"
# the DSpark checkpoint
if [ -d /tmp/dspark ] && [ ! -e "$ENGINE/models/dspark" ]; then mv /tmp/dspark "$ENGINE/models/dspark"; fi
if [ "${2:-}" = "move-dataset" ]; then
  if [ -L "$OLD_DATASET" ]; then echo "dataset already moved: $(readlink "$OLD_DATASET")"
  else
    [ "$(stat -c %d "$OLD_DATASET")" = "$(stat -c %d "$ENGINE")" ] || { echo "dataset and engine folder are on different filesystems, not moving"; exit 1; }
    if pgrep -x whole-model > /dev/null; then echo "a run is using the model, not moving"; exit 1; fi
    mv "$OLD_DATASET" "$ENGINE/dataset"
    ln -s "$ENGINE/dataset" "$OLD_DATASET"
    echo "dataset moved to $ENGINE/dataset, symlink left at $OLD_DATASET"
  fi
fi
echo "== done $(date)"; du -sh --exclude=dataset "$ENGINE" 2>/dev/null | tail -1
