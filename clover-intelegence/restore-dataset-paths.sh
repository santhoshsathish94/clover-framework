#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
DATA="$ROOT/dataset"
CODE="$ROOT/code/distrubuted-hosting"
RECORD="$ROOT/DATASET-ORIGINAL-PATHS-AX102"
ACTION=${1:-plan}
SELECTED=${2:-all}
PACKAGES=(normalization client server)
for layer in $(seq 1 92); do PACKAGES+=("tansformers/transformer-$layer"); done
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
absent() { [[ ! -e "$1" && ! -L "$1" ]] || fail "Occupied: $1"; }
identity() { stat -c '%d:%i:%s:%y:%a:%u:%g' -- "$1"; }

units() {
    printf 'normalization\tnormalization/leaves.json\tleaves.json\n'
    for entry in inputs outputs tiktoken.model vocabulary.bin; do printf 'client\tclient/%s\t%s\n' "$entry" "$entry"; done
    printf 'server\tserver/trunk-0\ttrunk-0\nserver\tserver/trunk-0-qkv\toperators/trunk-0-qkv\n'
    local layer group
    for layer in $(seq 1 92); do
        group="tansformers/transformer-$layer"
        printf '%s\t%s/trunk-%s\ttrunk-%s\n' "$group" "$group" "$layer" "$layer"
        printf '%s\t%s/root-%s\troot-%s\n' "$group" "$group" "$layer" "$layer"
        printf '%s\t%s/operators/qkv-all/layer-%s\toperators/qkv-all/layer-%s\n' "$group" "$group" "$layer" "$layer"
    done
}

check_unit() {
    local number=$1 base=$2 relative expected hash path
    while IFS=$'\t' read -r relative expected hash; do
        path="$base"
        [[ "$relative" == . ]] || path="$base/$relative"
        [[ -e "$path" && ! -L "$path" ]] || fail "Missing real entry: $path"
        [[ $(identity "$path") == "$expected" ]] || fail "Changed identity: $path"
        if [[ "$hash" != - ]]; then
            [[ $(sha256sum "$path" | cut -d ' ' -f 1) == "$hash" ]] || fail "Changed hash: $path"
        fi
    done < "$RECORD/unit-$number.tsv"
    [[ $(find "$base" -printf '.\n' | wc -l) -eq $(wc -l < "$RECORD/unit-$number.tsv") ]] || fail "Entry count changed: $base"
}

inactive() {
    local path=$1 output status
    local files=()
    mapfile -d '' -t files < <(find "$path" -type f -print0)
    [[ ${#files[@]} -gt 0 ]] || fail "Empty payload: $path"
    if output=$(fuser "${files[@]}" 2>&1); then fail "Active reader: $output"; else
        status=$?
        [[ $status -eq 1 && -z "$output" ]] || fail "Reader check failed: $output"
    fi
}

check_bin() {
    local group=$1 path="$CODE/$1/bin/dataset"
    if [[ "$group" == server ]]; then
        [[ -d "$path" && ! -L "$path" && $(find "$path" -mindepth 1 -maxdepth 1 | wc -l) -eq 2 ]] || fail 'Server mapping directory differs'
        [[ -L "$path/trunk-0" && "$path/trunk-0" -ef "$DATA/trunk-0" ]] || fail 'Server trunk link differs'
        [[ -L "$path/trunk-0-qkv" && "$path/trunk-0-qkv" -ef "$DATA/operators/trunk-0-qkv" ]] || fail 'Server operator link differs'
    else
        [[ -L "$path" && "$path" -ef "$DATA" ]] || fail "Bin root link differs: $group"
    fi
}

case "$ACTION" in
plan)
    absent "$RECORD"
    while IFS=$'\t' read -r group source target; do
        [[ -e "$DATA/$source" && ! -L "$DATA/$source" ]] || fail "Source not real: $source"
        [[ -z $(find "$DATA/$source" -type l -print -quit) ]] || fail "Nested payload link: $source"
        if [[ -e "$DATA/$target" || -L "$DATA/$target" ]]; then
            [[ -L "$DATA/$target" && "$DATA/$target" -ef "$DATA/$source" ]] || fail "Wrong existing destination: $target"
        fi
        absent "$DATA/$target.mapping-backup"
        [[ $(stat -c %d "$DATA/$source") == "$(stat -c %d "$DATA")" ]] || fail 'Cross-filesystem move'
        inactive "$DATA/$source"
    done < <(units)
    for group in "${PACKAGES[@]}"; do
        [[ -L "$CODE/$group/bin/dataset" && "$CODE/$group/bin/dataset" -ef "$DATA/$group" ]] || fail "Unexpected bin link: $group"
        absent "$CODE/$group/bin/dataset.mapping-backup"
    done
    mkdir "$RECORD"
    : > "$RECORD/plan.tsv"; : > "$RECORD/bin-links.tsv"; : > "$RECORD/external.tsv"
    declare -A moving=()
    number=0
    while IFS=$'\t' read -r group source target; do
        number=$((number + 1)); alias_text=-
        [[ ! -L "$DATA/$target" ]] || alias_text=$(readlink "$DATA/$target")
        printf '%s\t%s\t%s\t%s\t%s\n' "$number" "$group" "$source" "$target" "$alias_text" >> "$RECORD/plan.tsv"
        : > "$RECORD/unit-$number.tsv"
        while IFS= read -r -d '' path; do
            relative=${path#"$DATA/$source/"}; [[ "$path" != "$DATA/$source" ]] || relative=.
            hash=-
            if [[ -f "$path" ]]; then
                moving["$path"]=1
                if [[ $(stat -c %s "$path") -le 1048576 ]]; then hash=$(sha256sum "$path" | cut -d ' ' -f 1); fi
            fi
            printf '%s\t%s\t%s\n' "$relative" "$(identity "$path")" "$hash" >> "$RECORD/unit-$number.tsv"
        done < <(find "$DATA/$source" -print0 | sort -z)
    done < <(units)
    : > "$RECORD/unchanged.tsv"
    while IFS= read -r -d '' path; do
        [[ ${moving["$path"]:-0} == 0 ]] || continue
        printf '%s\t%s\n' "$path" "$(identity "$path")" >> "$RECORD/unchanged.tsv"
    done < <(find "$DATA" "$CODE" -type f -print0)
    for group in "${PACKAGES[@]}"; do printf '%s\t%s\n' "$group" "$(readlink "$CODE/$group/bin/dataset")" >> "$RECORD/bin-links.tsv"; done
    while IFS= read -r -d '' path; do
        resolved=$(readlink -f "$path") || fail "Broken external alias: $path"
        [[ "$resolved" == "$DATA/"* ]] || continue
        printf '%s\t%s\t%s\n' "$path" "$(readlink "$path")" "$(stat -Lc '%d:%i' "$path")" >> "$RECORD/external.tsv"
    done < <(find /opt/clover-k3/clover-data -type l -print0)
    : > "$RECORD/journal.tsv"; : > "$RECORD/completed.txt"
    printf 'READY\n' > "$RECORD/state.txt"
    sync -f "$RECORD/state.txt"
    printf 'PASS: %s original-path moves planned; no payload moved\n' "$number"
    ;;
move)
    [[ $(cat "$RECORD/state.txt") == READY ]] || fail 'No completed plan'
    for group in "${PACKAGES[@]}"; do
        [[ "$SELECTED" == all || "$SELECTED" == "$group" ]] || continue
        if grep -Fxq "$group" "$RECORD/completed.txt"; then check_bin "$group"; continue; fi
        while IFS=$'\t' read -r number owner source target alias_text; do
            [[ "$owner" == "$group" ]] || continue
            check_unit "$number" "$DATA/$source"
            inactive "$DATA/$source"
            absent "$DATA/$target.mapping-backup"
            if [[ "$alias_text" != - ]]; then
                [[ -L "$DATA/$target" && $(readlink "$DATA/$target") == "$alias_text" ]] || fail "Alias changed: $target"
            else absent "$DATA/$target"; fi
            printf '%s\tSTARTED\n' "$number" >> "$RECORD/journal.tsv"; sync -f "$RECORD/journal.tsv"
            if [[ "$alias_text" != - ]]; then mv -T "$DATA/$target" "$DATA/$target.mapping-backup"; fi
            if ! mv -T "$DATA/$source" "$DATA/$target"; then
                [[ "$alias_text" == - ]] || mv -T "$DATA/$target.mapping-backup" "$DATA/$target"
                fail "Rename failed: $source"
            fi
            check_unit "$number" "$DATA/$target"
            [[ "$alias_text" == - ]] || rm -- "$DATA/$target.mapping-backup"
            printf '%s\tMOVED_AND_VERIFIED\n' "$number" >> "$RECORD/journal.tsv"; sync -f "$RECORD/journal.tsv"
        done < "$RECORD/plan.tsv"
        path="$CODE/$group/bin/dataset"
        old=$(awk -F '\t' -v owner="$group" '$1==owner {print $2}' "$RECORD/bin-links.tsv")
        [[ -L "$path" && $(readlink "$path") == "$old" ]] || fail "Bin link changed: $path"
        mv -T "$path" "$path.mapping-backup"
        if [[ "$group" == server ]]; then
            mkdir "$path"
            ln -s "$(realpath --relative-to="$path" "$DATA/trunk-0")" "$path/trunk-0"
            ln -s "$(realpath --relative-to="$path" "$DATA/operators/trunk-0-qkv")" "$path/trunk-0-qkv"
        else ln -s "$(realpath --relative-to="$(dirname "$path")" "$DATA")" "$path"; fi
        check_bin "$group"
        rm -- "$path.mapping-backup"
        printf '%s\n' "$group" >> "$RECORD/completed.txt"; sync -f "$RECORD/completed.txt"
        printf 'PASS: %s real central payloads and bin-only mapping\n' "$group"
    done
    ;;
verify)
    [[ $(wc -l < "$RECORD/completed.txt") -eq 95 ]] || fail 'Incomplete packages'
    while IFS=$'\t' read -r number group source target alias_text; do
        absent "$DATA/$source"; check_unit "$number" "$DATA/$target"
    done < "$RECORD/plan.tsv"
    for group in "${PACKAGES[@]}"; do check_bin "$group"; done
    [[ -z $(find "$DATA" -type l -print -quit) ]] || fail 'Dataset still contains symlinks'
    [[ -z $(find "$CODE" -path '*/bin/dataset/*' -type f -print -quit) ]] || fail 'Real data files remain in bins'
    while IFS=$'\t' read -r path expected; do [[ $(identity "$path") == "$expected" ]] || fail "Unrelated file changed: $path"; done < "$RECORD/unchanged.tsv"
    while IFS=$'\t' read -r path text expected; do
        [[ -L "$path" && $(readlink "$path") == "$text" && $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "External alias changed: $path"
    done < "$RECORD/external.tsv"
    printf 'PASS: 283 original real dataset paths; no dataset symlinks; 95 bin mappings; external aliases and all payload identities preserved\n'
    ;;
*) fail 'Use plan, move [package|all], or verify' ;;
esac