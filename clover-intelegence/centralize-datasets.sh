#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
RECORD="$ROOT/DATASET-CENTRALIZATION-AX102"
ACTION=${1:-plan}
SELECTED=${2:-all}
GROUPS_TO_MOVE=(client server normalization)
for layer in $(seq 1 92); do GROUPS_TO_MOVE+=("tansformers/transformer-$layer"); done

fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
absent() { [[ ! -e "$1" && ! -L "$1" ]] || fail "Destination exists: $1"; }
identity() { stat -c '%d:%i:%s:%y:%a:%u:%g' -- "$1"; }
manifest() { printf '%s/%s.tsv' "$RECORD" "${1//\//-}"; }

inactive() {
    local group=$1 output status
    local files=()
    mapfile -d '' -t files < <(find "$ROOT/code/$group/bin/dataset" -type f -print0)
    [[ ${#files[@]} -gt 0 ]] || fail "No dataset files found: $group"
    if output=$(fuser "${files[@]}" 2>&1); then
        fail "Active dataset reader in $group: $output"
    else
        status=$?
        [[ $status -eq 1 && -z "$output" ]] || fail "Cannot establish reader status for $group: $output"
    fi
}

check_group() {
    local group=$1 side=$2 base kind relative expected hash path actual_count planned_count
    base="$ROOT/code/$group/bin/dataset"
    if [[ "$side" == target ]]; then base="$ROOT/dataset/$group"; fi
    [[ -d "$base" && ! -L "$base" ]] || fail "Not a real dataset directory: $base"
    while IFS=$'\t' read -r kind relative expected hash; do
        path="$base/$relative"
        [[ ! -L "$path" && -e "$path" ]] || fail "Missing or linked payload: $path"
        [[ $(identity "$path") == "$expected" ]] || fail "Identity changed: $path"
        if [[ "$kind" == f ]]; then
            [[ -f "$path" ]] || fail "Not a file: $path"
        else
            [[ -d "$path" ]] || fail "Not a directory: $path"
        fi
        if [[ "$hash" != - ]]; then
            [[ $(sha256sum -- "$path" | cut -d ' ' -f 1) == "$hash" ]] || fail "Hash changed: $path"
        fi
    done < "$(manifest "$group")"
    actual_count=$(find "$base" -printf '.\n' | wc -l)
    planned_count=$(wc -l < "$(manifest "$group")")
    [[ "$actual_count" -eq "$planned_count" ]] || fail "Dataset entry count changed: $group"
    if [[ "$side" == target ]]; then
        path="$ROOT/code/$group/bin/dataset"
        [[ -L "$path" && "$path" -ef "$base" ]] || fail "Runtime link mismatch: $path"
        [[ $(readlink "$path") == "$(realpath --relative-to="$(dirname "$path")" "$base")" ]] || fail "Runtime link text changed: $path"
    fi
}

check_aliases() {
    local path text expected
    while IFS=$'\t' read -r path text expected; do
        [[ -L "$path" && $(readlink "$path") == "$text" ]] || fail "Existing alias changed: $path"
        [[ $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "Existing alias resolves differently: $path"
    done < "$RECORD/aliases.tsv"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    [[ -d "$ROOT/dataset" && ! -L "$ROOT/dataset" ]] || fail 'Central dataset must be a real directory'
    absent "$RECORD"
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/code/$group/bin/dataset"
        [[ -d "$source" && ! -L "$source" ]] || fail "Missing real source: $source"
        absent "$ROOT/dataset/$group"
        absent "$source.centralizing"
        [[ $(stat -c %d "$source") == "$(stat -c %d "$ROOT/dataset")" ]] || fail "Different filesystems: $source"
        [[ -z $(find "$source" ! -type f ! -type d -print -quit) ]] || fail "Unexpected payload type: $source"
        inactive "$group"
    done
    mkdir -- "$RECORD"
    printf '%s\n' "$ROOT" > "$RECORD/root.txt"
    : > "$RECORD/aliases.tsv"
    for location in "$ROOT/dataset" /opt/clover-k3/clover-data; do
        while IFS= read -r -d '' path; do
            resolved=$(readlink -f "$path") || fail "Broken existing alias: $path"
            [[ "$resolved" == "$ROOT/"* ]] || continue
            printf '%s\t%s\t%s\n' "$path" "$(readlink "$path")" "$(stat -Lc '%d:%i' "$path")" >> "$RECORD/aliases.tsv"
        done < <(find "$location" -type l -print0)
    done
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/code/$group/bin/dataset"
        : > "$(manifest "$group")"
        while IFS= read -r -d '' path; do
            relative=${path#"$source/"}
            if [[ "$path" == "$source" ]]; then relative=.; fi
            hash=-; kind=d
            if [[ -f "$path" ]]; then
                kind=f
                if [[ $(stat -c %s "$path") -le 1048576 ]]; then hash=$(sha256sum "$path" | cut -d ' ' -f 1); fi
            fi
            printf '%s\t%s\t%s\t%s\n' "$kind" "$relative" "$(identity "$path")" "$hash" >> "$(manifest "$group")"
        done < <(find "$source" -print0 | sort -z)
    done
    find "$ROOT/code" -type f ! -path '*/bin/dataset/*' -print0 | sort -z | xargs -0 stat -c '%n|%d:%i:%s:%y' > "$RECORD/code-before.txt"
    find "$ROOT/dataset" -type f -print0 | sort -z | xargs -0 stat -c '%n|%d:%i:%s:%y' > "$RECORD/shared-before.txt"
    : > "$RECORD/journal.tsv"
    printf 'READY\n' > "$RECORD/state.txt"
    sync -f "$RECORD/state.txt"
    printf 'PASS: planned %s dataset directories; no payload moved\n' "${#GROUPS_TO_MOVE[@]}"
    ;;
move)
    [[ $(cat "$RECORD/root.txt") == "$ROOT" && $(cat "$RECORD/state.txt") == READY ]] || fail 'Missing completed plan'
    for group in "${GROUPS_TO_MOVE[@]}"; do
        [[ "$SELECTED" == all || "$SELECTED" == "$group" ]] || continue
        if grep -Fxq "$group" "$RECORD/completed.txt" 2>/dev/null; then
            check_group "$group" target
            continue
        fi
        source="$ROOT/code/$group/bin/dataset"
        target="$ROOT/dataset/$group"
        absent "$target"
        absent "$source.centralizing"
        check_group "$group" source
        inactive "$group"
        mkdir -p -- "$(dirname "$target")"
        link=$(realpath -m --relative-to="$(dirname "$source")" "$target")
        ln -s -- "$link" "$source.centralizing"
        printf '%s\tSTARTED\n' "$group" >> "$RECORD/journal.tsv"
        sync -f "$RECORD/journal.tsv"
        if ! mv -T -- "$source" "$target"; then
            rm -- "$source.centralizing"
            fail "Rename failed: $group"
        fi
        if ! mv -T -- "$source.centralizing" "$source"; then
            mv -T -- "$target" "$source"
            rm -- "$source.centralizing"
            fail "Link installation failed; restored $group"
        fi
        check_group "$group" target
        check_aliases
        printf '%s\tMOVED_AND_VERIFIED\n' "$group" >> "$RECORD/journal.tsv"
        printf '%s\n' "$group" >> "$RECORD/completed.txt"
        sync -f "$RECORD/completed.txt"
        printf 'PASS: %s moved; original identities, small hashes and runtime link retained\n' "$group"
    done
    ;;
verify)
    [[ $(wc -l < "$RECORD/completed.txt") -eq ${#GROUPS_TO_MOVE[@]} ]] || fail 'Incomplete move count'
    for group in "${GROUPS_TO_MOVE[@]}"; do check_group "$group" target; done
    check_aliases
    [[ -z $(find "$ROOT/code" -type d -path '*/bin/dataset' -print -quit) ]] || fail 'Real datasets remain inside code bins'
    find "$ROOT/code" -type f -print0 | sort -z | xargs -0 stat -c '%n|%d:%i:%s:%y' > "$RECORD/code-after.txt"
    cmp "$RECORD/code-before.txt" "$RECORD/code-after.txt"
    while IFS='|' read -r path expected; do
        [[ $(stat -c '%d:%i:%s:%y' "$path") == "$expected" ]] || fail "Shared file changed: $path"
    done < "$RECORD/shared-before.txt"
    printf 'PASS: 95 central datasets and bin links; all original aliases, shared data, code and binaries unchanged\n'
    ;;
*) fail 'Use plan, move [group|all], or verify' ;;
esac