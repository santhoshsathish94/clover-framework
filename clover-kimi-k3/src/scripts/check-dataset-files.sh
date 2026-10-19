#!/bin/bash
# Checks that every file the engine opens is where it should be with the recorded size, and optionally the sha256 of each:
#   scripts/dataset-files.tsv       the precomputed dataset (expert-N, trunk-N with its operator.bin), in DATASET_DIR
#   scripts/open-weights-files.tsv  the original open weights files (moonshotai/Kimi-K3), in OPEN_WEIGHTS_DIR
# Each list: path, bytes, sha256.
# usage: [DATASET_DIR=...] [DATASET_LIST=file.tsv instead of scripts/dataset-files.tsv] [OPEN_WEIGHTS_DIR=...] [CHECK_HASHES=1] check-dataset-files.sh        exit status 0 = everything present and equal
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
rc=0

# check LIST DIR NAME
check() {
  local LIST="$1" DIR="$2" NAME="$3"
  [ -f "$LIST" ] || { echo "FAIL  $LIST is missing"; rc=1; return; }
  local missing=0 wrong=0 n=0 total=0 path size sum have
  while IFS=$'\t' read -r path size sum; do
    case "$path" in '#'*|'') continue;; esac
    n=$((n + 1)); total=$((total + size))
    if [ ! -e "$DIR/$path" ]; then missing=$((missing + 1)); [ $missing -le 5 ] && echo "      missing: $path"
    else have=$(stat -L -c %s "$DIR/$path"); [ "$have" = "$size" ] || { wrong=$((wrong + 1)); [ $wrong -le 5 ] && echo "      size $have, expected $size: $path"; }; fi
  done < "$LIST"
  local ok=1
  if [ $missing = 0 ] && [ $wrong = 0 ]; then echo "PASS  $NAME: $n files, $((total / 1073741824)) GB, all present with the recorded sizes"
  else echo "FAIL  $NAME: $missing missing and $wrong with a different size, of $n files in $DIR"; rc=1; ok=0; fi
  if [ "${CHECK_HASHES:-0}" = 1 ] && [ $ok = 1 ]; then
    echo "      hashing every $NAME file (CHECK_HASHES=1) ..."
    local sums; sums=$(mktemp); awk -F'\t' '!/^#/ && NF>=3 && $3!="" {print $3 "  " $1}' "$LIST" > "$sums"
    split -n l/8 "$sums" "$sums.part."
    local pids=() p hrc=0
    for p in "$sums".part.*; do ( cd "$DIR" && sha256sum -c --quiet "$p" ) & pids+=($!); done
    for p in "${pids[@]}"; do wait "$p" || hrc=1; done
    rm -f "$sums" "$sums".part.*
    [ $hrc = 0 ] && echo "PASS  $NAME: sha256 of every file matches" || { echo "FAIL  $NAME: sha256 mismatch (lines above)"; rc=1; }
  fi
}

check "${DATASET_LIST:-$SCRIPTS_DIR/dataset-files.tsv}" "$DATASET_DIR" "dataset"
check "$SCRIPTS_DIR/open-weights-files.tsv" "$OPEN_WEIGHTS_DIR" "open weights"
exit $rc