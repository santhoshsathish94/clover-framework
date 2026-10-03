#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
SOURCE="$ROOT/dataset/trunk-0"
TARGET="$ROOT/code/server/dataset/trunk-0"
PLAN="$ROOT/code/server/TRUNK0-MOVE-PLAN.tsv"
JOURNAL="$ROOT/code/server/TRUNK0-MOVE-JOURNAL.tsv"
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
    test "$count" -eq 6
    printf 'PASS: %s, six trunk-0 file identities and SHA256 hashes\n' "$side"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$SOURCE"
    test ! -L "$SOURCE"
    test ! -e "$TARGET"
    test ! -L "$TARGET"
    for record in "$PLAN" "$JOURNAL"; do test ! -e "$record"; test ! -L "$record"; done
    test "$(find "$SOURCE" -mindepth 1 | wc -l)" -eq 6
    test "$(stat -c '%d' "$SOURCE")" = "$(stat -c '%d' "$ROOT/code/server/dataset")"
    (set -o noclobber; : > "$PLAN")
    for name in common.bin kda.bin dense.bin index.bin build-values.json manifest.json; do
        test -f "$SOURCE/$name"
        test ! -L "$SOURCE/$name"
        printf '%s\t%s\t%s\n' "$name" "$(stat -Lc '%d:%i:%s:%y' -- "$SOURCE/$name")" \
            "$(sha256sum -- "$SOURCE/$name" | cut -d ' ' -f 1)" >> "$PLAN"
    done
    check_files source
    sync -f "$PLAN"
    ;;
move)
    test -d "$SOURCE"
    test ! -L "$SOURCE"
    test ! -e "$TARGET"
    test ! -L "$TARGET"
    test ! -e "$JOURNAL"
    test ! -L "$JOURNAL"
    check_files source
    directory_identity=$(stat -c '%d:%i:%s:%y' -- "$SOURCE")
    mv -T -- "$SOURCE" "$TARGET"
    if ! ln -s -- "$TARGET" "$SOURCE"; then
        if [[ ! -e "$SOURCE" && ! -L "$SOURCE" ]]; then mv -T -- "$TARGET" "$SOURCE"; fi
        exit 1
    fi
    test "$(stat -c '%d:%i:%s:%y' -- "$TARGET")" = "$directory_identity"
    check_files target
    (set -o noclobber; printf '%s\t%s\t%s\tMOVED_AND_VERIFIED\n' \
        "$SOURCE" "$TARGET" "$directory_identity" > "$JOURNAL")
    sync -f "$JOURNAL"
    ;;
verify)
    test -d "$TARGET"
    test ! -L "$TARGET"
    test -L "$SOURCE"
    test "$(readlink -- "$SOURCE")" = "$TARGET"
    check_files target
    test -z "$(find "$TARGET" -type l -print -quit)"
    for name in common.bin kda.bin dense.bin index.bin build-values.json manifest.json; do
        test "/opt/clover-k3/clover-data/trunk-0/$name" -ef "$TARGET/$name"
    done
    printf 'PASS: server owns trunk-0, both old paths resolve, no internal links\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac