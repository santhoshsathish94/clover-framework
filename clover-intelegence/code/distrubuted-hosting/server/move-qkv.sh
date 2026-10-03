#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
SOURCE="$ROOT/dataset/operators/trunk-0-qkv"
TARGET="$ROOT/code/server/dataset/trunk-0-qkv"
PLAN="$ROOT/code/server/QKV-MOVE-PLAN.tsv"
JOURNAL="$ROOT/code/server/QKV-MOVE-JOURNAL.tsv"
ACTION=${1:-plan}

check_files() {
    local side=$1 path name identity hash count=0
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
        if [[ "$side" == target ]]; then test "$SOURCE/$name" -ef "$path"; fi
        count=$((count + 1))
    done < "$PLAN"
    test "$count" -eq 2
    printf 'PASS: %s QKV payload and manifest identities/hashes\n' "$side"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$SOURCE"
    test ! -L "$SOURCE"
    test ! -e "$TARGET"
    test ! -L "$TARGET"
    test ! -e "$PLAN"
    test ! -e "$JOURNAL"
    test "$(find "$SOURCE" -mindepth 1 | wc -l)" -eq 2
    test "$(stat -c '%d' "$SOURCE")" = "$(stat -c '%d' "$ROOT/code/server")"
    (set -o noclobber; : > "$PLAN")
    for name in qkv.bin manifest.json; do
        test -f "$SOURCE/$name"
        test ! -L "$SOURCE/$name"
        printf '%s\t%s\t%s\n' "$name" "$(stat -Lc '%d:%i:%s:%y' -- "$SOURCE/$name")" \
            "$(sha256sum -- "$SOURCE/$name" | cut -d ' ' -f 1)" >> "$PLAN"
    done
    check_files source
    sync -f "$PLAN"
    ;;
move)
    test ! -e "$TARGET"
    test ! -L "$TARGET"
    test ! -L "$SOURCE"
    test ! -e "$JOURNAL"
    check_files source
    mkdir -p -- "$(dirname -- "$TARGET")"
    mv -T -- "$SOURCE" "$TARGET"
    if ! ln -s -- "$TARGET" "$SOURCE"; then
        if [[ ! -e "$SOURCE" && ! -L "$SOURCE" ]]; then mv -T -- "$TARGET" "$SOURCE"; fi
        exit 1
    fi
    check_files target
    (set -o noclobber; printf '%s\t%s\tMOVED_AND_VERIFIED\n' "$SOURCE" "$TARGET" > "$JOURNAL")
    sync -f "$JOURNAL"
    ;;
verify)
    test -d "$TARGET"
    test ! -L "$TARGET"
    test -L "$SOURCE"
    test "$(readlink -- "$SOURCE")" = "$TARGET"
    check_files target
    test /opt/clover-k3/clover-data/operators/trunk-0-qkv/qkv.bin -ef "$TARGET/qkv.bin"
    printf 'PASS: server owns real QKV files; both historical paths still resolve\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac