#!/bin/bash
# Checks that every file the engine opens is in the dataset with the recorded size (scripts/dataset-files.tsv: path, bytes, sha256), and optionally the sha256 of each.
# usage: [DATASET_DIR=...] [CHECK_HASHES=1] check-dataset-files.sh        exit status 0 = everything present and equal
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
LIST="$SCRIPTS_DIR/dataset-files.tsv"
[ -f "$LIST" ] || { echo "FAIL  $LIST is missing"; exit 1; }
missing=0; wrong=0; n=0; total=0
while IFS=$'\t' read -r path size sum; do
  case "$path" in '#'*|'') continue;; esac
  n=$((n + 1)); total=$((total + size))
  if [ ! -e "$DATASET_DIR/$path" ]; then missing=$((missing + 1)); [ $missing -le 5 ] && echo "      missing: $path"
  else have=$(stat -L -c %s "$DATASET_DIR/$path"); [ "$have" = "$size" ] || { wrong=$((wrong + 1)); [ $wrong -le 5 ] && echo "      size $have, expected $size: $path"; }; fi
done < "$LIST"
rc=0
if [ $missing = 0 ] && [ $wrong = 0 ]; then echo "PASS  $n files, $((total / 1073741824)) GB, all present with the recorded sizes"
else echo "FAIL  $missing missing and $wrong with a different size, of $n files in $DATASET_DIR"; rc=1; fi
if [ "${CHECK_HASHES:-0}" = 1 ] && [ $rc = 0 ]; then
  echo "      hashing every file (CHECK_HASHES=1) ..."
  sums=$(mktemp); awk -F'\t' '!/^#/ && NF>=3 && $3!="" {print $3 "  " $1}' "$LIST" > "$sums"
  split -n l/8 "$sums" "$sums.part."
  pids=(); for p in "$sums".part.*; do ( cd "$DATASET_DIR" && sha256sum -c --quiet "$p" ) & pids+=($!); done
  hrc=0; for p in "${pids[@]}"; do wait "$p" || hrc=1; done
  rm -f "$sums" "$sums".part.*
  [ $hrc = 0 ] && echo "PASS  sha256 of every file matches" || { echo "FAIL  sha256 mismatch (lines above)"; rc=1; }
fi
exit $rc
