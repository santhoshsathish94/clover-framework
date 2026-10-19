#!/bin/bash
# Proves the engine needs, from the dataset, only what scripts/dataset-files.tsv lists, and that the original open weights are what it reads for the embedding table,
# the output head, the final normalisation tensors and the tokenizer:
#  1. a scratch dataset folder holding only links to expert-* and trunk-* (nothing else of the dataset is reachable through it);
#  2. one recorded prompt run under strace: every file opened is resolved and listed (dataset files not in the list are reported; open weights files opened are shown);
#  3. scripts/verify-setup.sh on that scratch dataset.
# Run it detached (it takes about 40 minutes); the result is work/results/check-open-weights-run.result.
# usage: check-open-weights-run.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
OUT="$RESULTS_DIR/check-open-weights-run.result"
SUBSET=/tmp/dataset-subset
REAL_DATASET=$(cd "$DATASET_DIR" && pwd -P)
{
  echo "== check-open-weights-run $(date)"
  rm -rf "$SUBSET"; mkdir -p "$SUBSET"
  for d in "$REAL_DATASET"/expert-* "$REAL_DATASET"/trunk-*; do ln -s "$d" "$SUBSET/$(basename "$d")"; done
  [ -d "$REAL_DATASET/engine-inputs" ] && ln -s "$REAL_DATASET/engine-inputs" "$SUBSET/engine-inputs"
  echo "scratch dataset $SUBSET: $(ls "$SUBSET" | wc -l) links, nothing else of $REAL_DATASET is reachable through it"
  export DATASET_DIR="$SUBSET"
  cd "$RESULTS_DIR"
  awk -F'\t' '$1==1{printf "%s", $4}' "$PROMPTS_DIR/recorded-34-prompts.tsv" > ow-prompt-1.txt
  echo "-- recorded prompt 1 under strace"
  PF_DECODE=11 PF_CACHE=16 PF_SAVE_IDS=ow-1.ids BUILD=0 strace -f --seccomp-bpf -e trace=openat -o /tmp/ow-strace.txt "$SCRIPTS_DIR/run-model.sh" text ow-prompt-1.txt > ow-1.log 2>&1
  want=$(awk -F'\t' '!/^#/ && $1==1{print $2}' "$PROMPTS_DIR/expected-exact-ids.tsv"); got=$(tr -d '\n\r ' < ow-1.ids 2>/dev/null)
  [ -n "$want" ] && [ "$got" = "$want" ] && echo "PASS  prompt 1 ids equal the recorded ones ($got)" || echo "FAIL  prompt 1: wanted $want, got ${got:-nothing}"
  grep -v ' = -1 ' /tmp/ow-strace.txt | sed -nE 's/^[0-9]+ +openat\([^,]*, "([^"]+)".*/\1/p' | sort -u > /tmp/ow-opened-all.txt
  while read -r p; do r=$(readlink -f "$p" 2>/dev/null) && [ -f "$r" ] && echo "$r"; done < /tmp/ow-opened-all.txt | sort -u > /tmp/ow-opened-real.txt
  grep "^$REAL_DATASET/" /tmp/ow-opened-real.txt | sed "s|^$REAL_DATASET/||" | sort -u > /tmp/ow-opened-ds.txt
  awk -F'\t' '!/^#/ && NF>=3 {print $1}' "$SCRIPTS_DIR/dataset-files.tsv" | sort -u > /tmp/ow-listed.txt
  echo "dataset files opened in this run: $(wc -l < /tmp/ow-opened-ds.txt); of those not in scripts/dataset-files.tsv: $(comm -23 /tmp/ow-opened-ds.txt /tmp/ow-listed.txt | wc -l)"
  comm -23 /tmp/ow-opened-ds.txt /tmp/ow-listed.txt | head -10
  echo "files opened under the open weights folder: $(grep -c "^$(readlink -f "$OPEN_WEIGHTS_DIR")/" /tmp/ow-opened-real.txt)"
  grep "^$(readlink -f "$OPEN_WEIGHTS_DIR")/" /tmp/ow-opened-real.txt | sed 's|.*/||'
  echo "other regular files opened outside system folders and the dataset:"
  grep -vE "^($REAL_DATASET|$(readlink -f "$OPEN_WEIGHTS_DIR"))/|^/(usr|lib|lib64|etc|proc|sys|dev|run|var)/" /tmp/ow-opened-real.txt | head -30
  echo "-- verify-setup on the scratch dataset"
  "$SCRIPTS_DIR/verify-setup.sh"; echo "verify-setup exit status $?"
  echo "done $(date)"
} > "$OUT" 2>&1
