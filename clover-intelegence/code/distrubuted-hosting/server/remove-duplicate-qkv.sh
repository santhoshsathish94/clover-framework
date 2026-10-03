#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
SERVER="$ROOT/code/server"
CANONICAL="$SERVER/bin/dataset/trunk-0-qkv"
DUPLICATE="$ROOT/dataset/operators/qkv-all/layer-0"
LEGACY=/opt/clover-k3/clover-data/operators/qkv-all/layer-0
PROVENANCE="$CANONICAL/qkv-all-layer-0-manifest.json"
PLAN="$SERVER/QKV-DEDUP-PLAN.tsv"
JOURNAL="$SERVER/QKV-DEDUP-JOURNAL.tsv"
EXPECTED=4925ea619a4fd268b441e8008e4c1cd714f681b328b03cccd52c69b6a9b6f654
ACTION=${1:-plan}

check_file() {
    local selected=$1 override=${2:-} kind path identity hash found=0
    while IFS=$'\t' read -r kind path identity hash; do
        [[ "$kind" == "$selected" ]] || continue
        if [[ -n "$override" ]]; then path=$override; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -c '%d:%i:%s:%h:%b:%y' -- "$path")" = "$identity"
        test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$hash"
        found=$((found+1))
    done < "$PLAN"
    test "$found" -eq 1
}

check_legacy() {
    local kind path identity target found=0
    while IFS=$'\t' read -r kind path identity target; do
        [[ "$kind" == legacy ]] || continue
        test "$path" = "$LEGACY"
        test -L "$path"
        test "$(readlink -- "$path")" = "$DUPLICATE"
        test "$target" = "$DUPLICATE"
        test "$(stat -c '%d:%i:%s:%h:%b:%y' -- "$path")" = "$identity"
        found=$((found+1))
    done < "$PLAN"
    test "$found" -eq 1
}

inactive() {
    local path
    for path in "$CANONICAL/qkv.bin" "$DUPLICATE/operator.bin" "$DUPLICATE/manifest.json"; do
        if fuser -- "$path" >/dev/null 2>&1; then printf 'Active file: %s\n' "$path" >&2; exit 1; fi
    done
}

record() {
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$(stat -c '%d:%i:%s:%h:%b:%y' -- "$2")" \
        "$(sha256sum -- "$2" | cut -d ' ' -f 1)" >> "$PLAN"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$DUPLICATE"
    test ! -L "$DUPLICATE"
    test "$(find "$DUPLICATE" -mindepth 1 | wc -l)" -eq 2
    for path in "$PLAN" "$JOURNAL" "$PROVENANCE"; do test ! -e "$path"; test ! -L "$path"; done
    for path in "$CANONICAL/qkv.bin" "$CANONICAL/manifest.json" "$DUPLICATE/operator.bin" "$DUPLICATE/manifest.json"; do
        test -f "$path"
        test ! -L "$path"
    done
    test "$(stat -c '%s' "$CANONICAL/qkv.bin")" -eq 265008216
    test "$(sha256sum "$CANONICAL/qkv.bin" | cut -d ' ' -f 1)" = "$EXPECTED"
    test "$(stat -c '%d' "$DUPLICATE/manifest.json")" = "$(stat -c '%d' "$CANONICAL")"
    cmp "$CANONICAL/qkv.bin" "$DUPLICATE/operator.bin"
    test -L "$LEGACY"
    test "$(readlink -- "$LEGACY")" = "$DUPLICATE"
    inactive
    (set -o noclobber; : > "$PLAN")
    record canonical "$CANONICAL/qkv.bin"
    record canonical-manifest "$CANONICAL/manifest.json"
    record duplicate "$DUPLICATE/operator.bin"
    record provenance "$DUPLICATE/manifest.json"
    printf 'legacy\t%s\t%s\t%s\n' "$LEGACY" "$(stat -c '%d:%i:%s:%h:%b:%y' "$LEGACY")" "$DUPLICATE" >> "$PLAN"
    sync -f "$PLAN"
    printf 'PASS: duplicate identity and inactive files checked; provenance and alias recorded; nothing deleted\n'
    ;;
remove)
    test ! -e "$JOURNAL"
    test ! -L "$JOURNAL"
    test ! -e "$PROVENANCE"
    test ! -L "$PROVENANCE"
    test ! -L "$DUPLICATE"
    test "$(find "$DUPLICATE" -mindepth 1 | wc -l)" -eq 2
    for kind in canonical canonical-manifest duplicate provenance; do check_file "$kind"; done
    check_legacy
    inactive
    cmp "$CANONICAL/qkv.bin" "$DUPLICATE/operator.bin"
    (set -o noclobber; : > "$JOURNAL")
    mv -T -- "$DUPLICATE/manifest.json" "$PROVENANCE"
    check_file provenance "$PROVENANCE"
    printf 'PROVENANCE_PRESERVED\t%s\n' "$PROVENANCE" >> "$JOURNAL"
    sync -f "$JOURNAL"
    check_file canonical
    check_file duplicate
    unlink -- "$DUPLICATE/operator.bin"
    printf 'DUPLICATE_REMOVED\t%s/operator.bin\n' "$DUPLICATE" >> "$JOURNAL"
    sync -f "$JOURNAL"
    rmdir -- "$DUPLICATE"
    check_legacy
    unlink -- "$LEGACY"
    printf 'DIRECTORY_AND_LEGACY_REMOVED\t%s\t%s\n' "$DUPLICATE" "$LEGACY" >> "$JOURNAL"
    sync -f "$JOURNAL"
    check_file canonical
    check_file canonical-manifest
    printf 'PASS: duplicate payload and old folder/link removed; canonical payload and both provenance records retained\n'
    ;;
verify)
    check_file canonical
    check_file canonical-manifest
    check_file provenance "$PROVENANCE"
    for path in "$DUPLICATE" "$LEGACY"; do test ! -e "$path"; test ! -L "$path"; done
    test "$(wc -l < "$JOURNAL")" -eq 3
    printf 'PASS: one canonical layer-0 operator payload retained; redundant folder and legacy alias absent\n'
    ;;
*) printf 'Use plan, remove or verify\n' >&2; exit 2 ;;
esac