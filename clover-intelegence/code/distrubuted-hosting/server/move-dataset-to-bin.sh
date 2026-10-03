#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
SERVER="$ROOT/code/server"
SOURCE="$SERVER/dataset"
TARGET="$SERVER/bin/dataset"
PLAN="$SERVER/BIN-DATASET-PLAN.tsv"
JOURNAL="$SERVER/BIN-DATASET-JOURNAL.tsv"
ACTION=${1:-plan}

check_files() {
    local side=$1 name identity hash path count=0
    while IFS=$'\t' read -r name identity hash; do
        path="$SOURCE/$name"
        if [[ "$side" == target ]]; then path="$TARGET/$name"; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -Lc '%d:%i:%s:%y' -- "$path")" = "$identity"
        test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$hash"
        if [[ "$side" == source ]] && fuser -- "$path" >/dev/null 2>&1; then
            printf 'Active file, stopping: %s\n' "$path" >&2; exit 1
        fi
        count=$((count + 1))
    done < "$PLAN"
    test "$count" -eq 9
    printf 'PASS: %s, nine dataset file identities and SHA256 hashes\n' "$side"
}

alias_path() {
    case "$1" in
    trunk-0) printf '%s/dataset/trunk-0' "$ROOT" ;;
    trunk-0-qkv) printf '%s/dataset/operators/trunk-0-qkv' "$ROOT" ;;
    *) exit 2 ;;
    esac
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$SOURCE"
    test ! -L "$SOURCE"
    test -d "$SERVER/bin"
    for path in "$TARGET" "$PLAN" "$JOURNAL"; do test ! -e "$path"; test ! -L "$path"; done
    test -z "$(find "$SOURCE" -type l -print -quit)"
    test "$(find "$SOURCE" -type f | wc -l)" -eq 9
    for name in trunk-0 trunk-0-qkv; do
        path=$(alias_path "$name")
        test -L "$path"
        test "$(readlink -- "$path")" = "$SOURCE/$name"
        test ! -e "$path.bin-move"
        test ! -L "$path.bin-move"
    done
    (set -o noclobber; : > "$PLAN")
    while IFS= read -r -d '' file; do
        test "$(stat -c '%d' -- "$file")" = "$(stat -c '%d' -- "$SERVER/bin")"
        printf '%s\t%s\t%s\n' "${file#"$SOURCE/"}" "$(stat -Lc '%d:%i:%s:%y' -- "$file")" \
            "$(sha256sum -- "$file" | cut -d ' ' -f 1)" >> "$PLAN"
    done < <(find "$SOURCE" -type f -print0 | sort -z)
    check_files source
    sync -f "$PLAN"
    ;;
move)
    test ! -e "$TARGET"
    test ! -L "$TARGET"
    test ! -L "$SOURCE"
    for path in "$JOURNAL"; do test ! -e "$path"; test ! -L "$path"; done
    for name in trunk-0 trunk-0-qkv; do
        path=$(alias_path "$name")
        test -L "$path"
        test "$(readlink -- "$path")" = "$SOURCE/$name"
        test ! -e "$path.bin-move"
        test ! -L "$path.bin-move"
    done
    check_files source
    identity=$(stat -c '%d:%i:%s:%y' -- "$SOURCE")
    mv -T -- "$SOURCE" "$TARGET"
    test "$(stat -c '%d:%i:%s:%y' -- "$TARGET")" = "$identity"
    check_files target
    (set -o noclobber; printf '%s\t%s\tMOVED_AND_VERIFIED\n' "$SOURCE" "$TARGET" > "$JOURNAL")
    sync -f "$JOURNAL"
    for name in trunk-0 trunk-0-qkv; do
        path=$(alias_path "$name")
        test -L "$path"
        test "$(readlink -- "$path")" = "$SOURCE/$name"
        ln -s -- "$TARGET/$name" "$path.bin-move"
        mv -T -- "$path.bin-move" "$path"
        test "$path" -ef "$TARGET/$name"
        printf '%s\t%s\tALIAS_UPDATED\n' "$path" "$TARGET/$name" >> "$JOURNAL"
        sync -f "$JOURNAL"
    done
    ;;
verify)
    test -d "$TARGET"
    test ! -L "$TARGET"
    test ! -e "$SOURCE"
    test ! -L "$SOURCE"
    test -z "$(find "$TARGET" -type l -print -quit)"
    check_files target
    for name in trunk-0 trunk-0-qkv; do
        path=$(alias_path "$name")
        test "$(readlink -- "$path")" = "$TARGET/$name"
        test "$path" -ef "$TARGET/$name"
        test ! -e "$path.bin-move"
        test ! -L "$path.bin-move"
    done
    test /opt/clover-k3/clover-data/trunk-0/index.bin -ef "$TARGET/trunk-0/index.bin"
    test /opt/clover-k3/clover-data/operators/trunk-0-qkv/qkv.bin -ef "$TARGET/trunk-0-qkv/qkv.bin"
    test "$(wc -l < "$JOURNAL")" -eq 3
    printf 'PASS: server datasets inside bin only; existing aliases updated\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac