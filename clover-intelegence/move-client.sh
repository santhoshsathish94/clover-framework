#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
PACKAGE="$ROOT/code/client"
PLAN="$ROOT/CLIENT-MOVE-PLAN.tsv"
JOURNAL="$ROOT/CLIENT-MOVE-JOURNAL.tsv"
ACTION=${1:-plan}

record_file() {
    local group=$1 source=$2 target=$3
    test -f "$source"
    test ! -L "$source"
    test "$(stat -Lc '%d' -- "$source")" = "$(stat -Lc '%d' -- "$ROOT")"
    if fuser -- "$source" >/dev/null 2>&1; then
        printf 'Active file, stopping: %s\n' "$source" >&2; exit 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$source" "$target" \
        "$(stat -Lc '%d:%i:%s:%y' -- "$source")" \
        "$(sha256sum -- "$source" | cut -d ' ' -f 1)" >> "$PLAN"
}

check_group() {
    local selected=$1 side=$2 checked=0
    local group source target expected_identity expected_hash
    while IFS=$'\t' read -r group source target expected_identity expected_hash; do
        [[ "$group" == "$selected" ]] || continue
        local path=$source
        if [[ "$side" == target ]]; then path=$target; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -Lc '%d:%i:%s:%y' -- "$path")" = "$expected_identity"
        test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$expected_hash"
        if [[ "$side" == source ]] && fuser -- "$path" >/dev/null 2>&1; then
            printf 'Active file, stopping: %s\n' "$path" >&2; exit 1
        fi
        if [[ "$side" == target && "$group" != code ]]; then test "$source" -ef "$target"; fi
        checked=$((checked + 1))
    done < "$PLAN"
    test "$checked" -gt 0
    printf 'PASS: %s %s, %s file identities and SHA256 hashes\n' "$selected" "$side" "$checked"
}

move_asset() {
    local group=$1 source=$2 target=$3
    test -d "$PACKAGE"
    test ! -e "$target"
    test ! -L "$target"
    test ! -L "$source"
    check_group "$group" source
    mkdir -p -- "$(dirname -- "$target")"
    mv -T -- "$source" "$target"
    if ! ln -s -- "$target" "$source"; then
        if [[ ! -e "$source" && ! -L "$source" ]]; then mv -T -- "$target" "$source"; fi
        exit 1
    fi
    check_group "$group" target
    printf '%s\tMOVED_AND_VERIFIED\n' "$group" >> "$JOURNAL"
    sync -f "$JOURNAL"
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$ROOT/code"
    test ! -L "$ROOT/code"
    test -f "$ROOT/code/client"
    test ! -e "$ROOT/.client-package-code-stage"
    test ! -e "$PLAN"
    test ! -e "$JOURNAL"
    (set -o noclobber; : > "$PLAN")
    for group in code inputs outputs; do
        source="$ROOT/code"
        target="$PACKAGE"
        if [[ "$group" != code ]]; then source="$ROOT/dataset/$group"; target="$PACKAGE/dataset/$group"; fi
        test -d "$source"
        test ! -L "$source"
        if [[ -n "$(find "$source" -type l -print -quit)" ]]; then
            printf 'Nested symlink requires review: %s\n' "$source" >&2; exit 1
        fi
        while IFS= read -r -d '' file; do
            record_file "$group" "$file" "$target/${file#"$source/"}"
        done < <(find "$source" -type f -print0 | sort -z)
    done
    record_file vocabulary "$ROOT/dataset/tiktoken.model" "$PACKAGE/dataset/tiktoken.model"
    record_file vocabulary "$ROOT/dataset/vocabulary.bin" "$PACKAGE/dataset/vocabulary.bin"
    record_file config "$ROOT/configs/tokenizer_config.json" "$PACKAGE/configs/tokenizer_config.json"
    sync -f "$PLAN"
    printf 'PASS: complete client file plan recorded, no files moved\n'
    ;;
code)
    check_group code source
    stage="$ROOT/.client-package-code-stage"
    test ! -e "$stage"
    mv -T -- "$ROOT/code" "$stage"
    if ! mkdir -- "$ROOT/code"; then mv -T -- "$stage" "$ROOT/code"; exit 1; fi
    if ! mv -T -- "$stage" "$PACKAGE"; then rmdir "$ROOT/code"; mv -T -- "$stage" "$ROOT/code"; exit 1; fi
    check_group code target
    printf 'code\tMOVED_AND_VERIFIED\n' >> "$JOURNAL"
    sync -f "$JOURNAL"
    ;;
inputs|outputs)
    move_asset "$ACTION" "$ROOT/dataset/$ACTION" "$PACKAGE/dataset/$ACTION"
    ;;
vocabulary)
    test -d "$PACKAGE"
    check_group vocabulary source
    for name in tiktoken.model vocabulary.bin; do
        source="$ROOT/dataset/$name"
        target="$PACKAGE/dataset/$name"
        test ! -e "$target"
        test ! -L "$target"
        mv -T -- "$source" "$target"
        if ! ln -s -- "$target" "$source"; then
            if [[ ! -e "$source" && ! -L "$source" ]]; then mv -T -- "$target" "$source"; fi
            exit 1
        fi
    done
    check_group vocabulary target
    printf 'vocabulary\tMOVED_AND_VERIFIED\n' >> "$JOURNAL"
    sync -f "$JOURNAL"
    ;;
config)
    move_asset config "$ROOT/configs/tokenizer_config.json" "$PACKAGE/configs/tokenizer_config.json"
    ;;
verify)
    for group in code inputs outputs vocabulary config; do check_group "$group" target; done
    test ! -e "$ROOT/.client-package-code-stage"
    test -z "$(find "$PACKAGE" -type l -print -quit)"
    printf 'PASS: self-contained client package, no internal symlinks, old data aliases preserved\n'
    ;;
*) printf 'Use plan, code, inputs, outputs, vocabulary, config or verify\n' >&2; exit 2 ;;
esac