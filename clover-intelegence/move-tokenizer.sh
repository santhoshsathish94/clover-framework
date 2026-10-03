#!/usr/bin/env bash
set -euo pipefail
HOME_DIR=/opt/clover-k3/clover-intelegence
action=${1:-plan}
names=(tiktoken.model vocabulary.bin tokenizer_config.json)
sources=(/opt/clover-k3/direct-equation-20261001-a/tiktoken.model /opt/clover-k3/direct-equation-20261001-a/data/vocabulary.bin /root/k3model/tokenizer_config.json)
targets=("$HOME_DIR/dataset/tiktoken.model" "$HOME_DIR/dataset/vocabulary.bin" "$HOME_DIR/configs/tokenizer_config.json")
plan="$HOME_DIR/TOKENIZER-MOVE-PLAN.tsv"
journal="$HOME_DIR/TOKENIZER-MOVE-JOURNAL.tsv"

check_inactive() {
    command -v fuser >/dev/null
    if fuser -- "$1" >/dev/null 2>&1; then
        printf 'File has active users: %s\n' "$1" >&2
        exit 1
    fi
}

verify_one() {
    local name=$1 source=$2 target=$3 expected_identity=$4 expected_hash=$5
    test -L "$source"
    test ! -L "$target"
    test "$(readlink -f -- "$source")" = "$target"
    test "$(stat -Lc '%d:%i:%s:%y' -- "$target")" = "$expected_identity"
    test "$(sha256sum -- "$target" | cut -d ' ' -f 1)" = "$expected_hash"
    test "$source" -ef "$target"
    printf 'PASS: %s identity/hash/old-path alias\n' "$name"
}

case "$action" in
plan)
    test ! -e "$plan"
    for index in 0 1 2; do
        test -f "${sources[index]}"
        test ! -L "${sources[index]}"
        test ! -e "${targets[index]}"
        test ! -L "${targets[index]}"
        test "$(stat -Lc '%d' -- "${sources[index]}")" = "$(stat -Lc '%d' -- "$HOME_DIR")"
        check_inactive "${sources[index]}"
    done
    mkdir -p -- "$HOME_DIR/configs"
    (set -o noclobber; : > "$plan")
    for index in 0 1 2; do
        printf '%s\t%s\t%s\t%s\t%s\n' "${names[index]}" "${sources[index]}" "${targets[index]}" \
            "$(stat -Lc '%d:%i:%s:%y' -- "${sources[index]}")" \
            "$(sha256sum -- "${sources[index]}" | cut -d ' ' -f 1)" >> "$plan"
    done
    printf 'PASS: three tokenizer assets planned; no payload moved\n'
    ;;
move)
    selected=${2:?asset name required}
    found=0
    while IFS=$'\t' read -r name source target expected_identity expected_hash; do
        if [[ "$name" != "$selected" ]]; then
            continue
        fi
        found=1
        if [[ -L "$source" ]]; then
            verify_one "$name" "$source" "$target" "$expected_identity" "$expected_hash"
            continue
        fi
        test ! -e "$target"
        test ! -L "$target"
        test "$(stat -Lc '%d:%i:%s:%y' -- "$source")" = "$expected_identity"
        test "$(sha256sum -- "$source" | cut -d ' ' -f 1)" = "$expected_hash"
        check_inactive "$source"
        mv -T -- "$source" "$target"
        if ! ln -s -- "$target" "$source"; then
            if [[ ! -e "$source" && ! -L "$source" ]]; then mv -T -- "$target" "$source"; fi
            exit 1
        fi
        verify_one "$name" "$source" "$target" "$expected_identity" "$expected_hash"
        printf '%s\t%s\t%s\tMOVED_AND_VERIFIED\n' "$name" "$target" "$expected_hash" >> "$journal"
        sync -f "$journal"
    done < "$plan"
    test "$found" = 1
    ;;
verify)
    while IFS=$'\t' read -r name source target expected_identity expected_hash; do
        verify_one "$name" "$source" "$target" "$expected_identity" "$expected_hash"
    done < "$plan"
    ;;
*) printf 'Use plan, move <asset>, or verify\n' >&2; exit 2 ;;
esac