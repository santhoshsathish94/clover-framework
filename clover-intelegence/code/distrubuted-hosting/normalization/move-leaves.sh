#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
PACKAGE="$ROOT/code/normalization"
SOURCE="$ROOT/dataset/leaves.json"
TARGET="$PACKAGE/bin/dataset/leaves.json"
PLAN="$PACKAGE/MOVE-PLAN.tsv"
JOURNAL="$PACKAGE/MOVE-JOURNAL.tsv"
ACTION=${1:-plan}
EXPECTED=a6a886747d400a545f4e16c2831ef0db7e8eab90575184a54d3bffe64f5b5f2a

check_file() {
    local path=$1 source target identity hash
    IFS=$'\t' read -r source target identity hash < "$PLAN"
    test "$source" = "$SOURCE"
    test "$target" = "$TARGET"
    test "$hash" = "$EXPECTED"
    test -f "$path"
    test ! -L "$path"
    test "$(stat -c '%d:%i:%s:%y' -- "$path")" = "$identity"
    test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$hash"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -f "$SOURCE"
    test ! -L "$SOURCE"
    for path in "$TARGET" "$PLAN" "$JOURNAL"; do test ! -e "$path"; test ! -L "$path"; done
    test "$(stat -c '%d' "$SOURCE")" = "$(stat -c '%d' "$PACKAGE/bin")"
    test "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" = "$EXPECTED"
    if fuser -- "$SOURCE" >/dev/null 2>&1; then printf 'Leaves file active; stopping\n' >&2; exit 1; fi
    (set -o noclobber; printf '%s\t%s\t%s\t%s\n' "$SOURCE" "$TARGET" \
        "$(stat -c '%d:%i:%s:%y' "$SOURCE")" "$EXPECTED" > "$PLAN")
    sync -f "$PLAN"
    printf 'PASS: leaves identity/hash recorded; nothing moved\n'
    ;;
move)
    for path in "$TARGET" "$JOURNAL"; do test ! -e "$path"; test ! -L "$path"; done
    check_file "$SOURCE"
    if fuser -- "$SOURCE" >/dev/null 2>&1; then printf 'Leaves file active; stopping\n' >&2; exit 1; fi
    mkdir -p -- "$(dirname "$TARGET")"
    mv -T -- "$SOURCE" "$TARGET"
    if ! ln -s -- "$TARGET" "$SOURCE"; then
        if [[ ! -e "$SOURCE" && ! -L "$SOURCE" ]]; then mv -T -- "$TARGET" "$SOURCE"; fi
        exit 1
    fi
    check_file "$TARGET"
    test "$SOURCE" -ef "$TARGET"
    (set -o noclobber; printf '%s\t%s\tMOVED_AND_VERIFIED\n' "$SOURCE" "$TARGET" > "$JOURNAL")
    sync -f "$JOURNAL"
    ;;
verify)
    check_file "$TARGET"
    test -L "$SOURCE"
    test "$(readlink -- "$SOURCE")" = "$TARGET"
    test "$SOURCE" -ef "$TARGET"
    test /opt/clover-k3/clover-data/leaves.json -ef "$TARGET"
    test "$(wc -l < "$JOURNAL")" -eq 1
    printf 'PASS: normalization owns unchanged leaves; both old aliases resolve\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac