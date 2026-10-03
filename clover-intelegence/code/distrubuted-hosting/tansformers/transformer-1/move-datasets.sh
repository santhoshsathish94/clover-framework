#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
PACKAGE="$ROOT/code/tansformers/transformer-1"
PLAN="$PACKAGE/MOVE-PLAN.tsv"
JOURNAL="$PACKAGE/MOVE-JOURNAL.tsv"
ACTION=${1:-plan}
GROUPS_TO_MOVE=(trunk-1 root-1 operators/qkv-all/layer-1)

check_group() {
    local selected=$1 side=$2 group source target identity hash path count=0
    while IFS=$'\t' read -r group source target identity hash; do
        [[ "$group" == "$selected" ]] || continue
        path=$source
        if [[ "$side" == target ]]; then path=$target; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -Lc '%d:%i:%s:%y' -- "$path")" = "$identity"
        test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$hash"
        if [[ "$side" == source ]] && fuser -- "$path" >/dev/null 2>&1; then
            printf 'Active file, stopping: %s\n' "$path" >&2; exit 1
        fi
        if [[ "$side" == target ]]; then test "$source" -ef "$target"; fi
        count=$((count + 1))
    done < "$PLAN"
    test "$count" -gt 0
    printf 'PASS: %s %s, %u file identities and SHA256 hashes\n' "$selected" "$side" "$count"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    for path in "$PLAN" "$JOURNAL"; do test ! -e "$path"; test ! -L "$path"; done
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/dataset/$group"
        target="$PACKAGE/bin/dataset/$group"
        test -d "$source"
        test ! -L "$source"
        test ! -e "$target"
        test ! -L "$target"
        test -z "$(find "$source" -type l -print -quit)"
        test "$(stat -c '%d' "$source")" = "$(stat -c '%d' "$PACKAGE/bin")"
    done
    (set -o noclobber; : > "$PLAN")
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/dataset/$group"
        target="$PACKAGE/bin/dataset/$group"
        while IFS= read -r -d '' file; do
            test "$(stat -c '%d' "$file")" = "$(stat -c '%d' "$PACKAGE/bin")"
            printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$file" "$target/${file#"$source/"}" \
                "$(stat -Lc '%d:%i:%s:%y' -- "$file")" "$(sha256sum -- "$file" | cut -d ' ' -f 1)" >> "$PLAN"
        done < <(find "$source" -type f -print0 | sort -z)
        check_group "$group" source
    done
    sync -f "$PLAN"
    printf 'PASS: three complete datasets inventoried; nothing moved\n'
    ;;
move)
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/dataset/$group"
        target="$PACKAGE/bin/dataset/$group"
        test ! -L "$source"
        test ! -e "$target"
        test ! -L "$target"
        check_group "$group" source
        mkdir -p -- "$(dirname -- "$target")"
        identity=$(stat -c '%d:%i:%s:%y' -- "$source")
        mv -T -- "$source" "$target"
        if ! ln -s -- "$target" "$source"; then
            if [[ ! -e "$source" && ! -L "$source" ]]; then mv -T -- "$target" "$source"; fi
            exit 1
        fi
        test "$(stat -c '%d:%i:%s:%y' -- "$target")" = "$identity"
        check_group "$group" target
        printf '%s\tMOVED_AND_VERIFIED\n' "$group" >> "$JOURNAL"
        sync -f "$JOURNAL"
    done
    ;;
verify)
    for group in "${GROUPS_TO_MOVE[@]}"; do
        source="$ROOT/dataset/$group"
        target="$PACKAGE/bin/dataset/$group"
        test -d "$target"
        test ! -L "$target"
        test -L "$source"
        test "$(readlink -- "$source")" = "$target"
        test -z "$(find "$target" -type l -print -quit)"
        check_group "$group" target
        test "/opt/clover-k3/clover-data/$group" -ef "$target"
    done
    test "$(wc -l < "$JOURNAL")" -eq 3
    printf 'PASS: all three real datasets inside transformer bin; historical aliases preserved\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac