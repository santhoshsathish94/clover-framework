#!/bin/bash
# The 34 recorded prompts (prompts/recorded-34-prompts.tsv) under one arithmetic mode, one process per prompt, one prompt at a time.
# Output goes to work/results/<dir>/N.{txt,ids,out,log}: exact -> recorded-34-exact, use_4bit_trunk -> recorded-34-use_4bit_trunk (the layout compare-outputs.py reads).
# usage: run-34-recorded-prompts.sh MODE [FIRST LAST]      (uses the built program; do not rebuild while it runs)
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
cd "$RESULTS_DIR"
MODE=${1:?mode exact or use_4bit_trunk}
case $MODE in
  exact)          DIR=recorded-34-exact;          E="LB_EXPK=1" ;;
  use_4bit_trunk) DIR=recorded-34-use_4bit_trunk; E="LB_EXPK=1 USE_4BIT_TRUNK=1 CLOVER_LOCK_TRUNK=0 PF_CACHE=16" ;;
  *) echo "mode must be exact or use_4bit_trunk"; exit 2 ;;
esac
mkdir -p $DIR
ulimit -l unlimited
for i in $(seq ${2:-1} ${3:-34}); do
  awk -F'\t' -v n=$i '$1==n{printf "%s", $4}' "$ENGINE_DIR/prompts/recorded-34-prompts.tsv" > $DIR/$i.txt
  env -i PATH="$PATH" LB_IO=4 LB_SPEC=0 LB_THP=1 $E PF_DECODE=11 PF_SAVE_IDS=$DIR/$i.ids \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores "$BUILD_DIR/whole-model" "$POD_DIR" "$DATASET_DIR" "$OPEN_WEIGHTS_DIR" $DIR/$i.txt - > $DIR/$i.log 2>&1
  "$BUILD_DIR/ids-to-text" "$OPEN_WEIGHTS_DIR" $DIR/$i.ids $DIR/$i.out > /dev/null 2>&1
  echo "prompt $i done $(date +%H:%M:%S)"
done
echo "ALL DONE"
