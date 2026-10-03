#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/clover-k3/clover-intelegence
CLIENT="$ROOT/code/client"
BIN="$CLIENT/bin"
PLAN="$ROOT/CLIENT-BIN-MOVE-PLAN.tsv"
ALIASES="$ROOT/CLIENT-BIN-ALIASES.tsv"
JOURNAL="$ROOT/CLIENT-BIN-MOVE-JOURNAL.tsv"
ACTION=${1:-plan}

check_files() {
    local side=$1 count=0 group source target identity hash path
    while IFS=$'\t' read -r group source target identity hash; do
        path=$source
        if [[ "$side" == target ]]; then path=$target; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -Lc '%d:%i:%s:%y' -- "$path")" = "$identity"
        test "$(sha256sum -- "$path" | cut -d ' ' -f 1)" = "$hash"
        if [[ "$side" == source ]] && fuser -- "$path" >/dev/null 2>&1; then
            printf 'Active file, stopping: %s\n' "$path" >&2; exit 1
        fi
        if [[ "$side" == target ]]; then test ! -e "$source"; test ! -L "$source"; fi
        count=$((count + 1))
    done < "$PLAN"
    test "$count" -eq 27
    printf 'PASS: %s, all %s original file identities and SHA256 hashes\n' "$side" "$count"
}

check_aliases() {
    local count=0 source target identity resolved
    while IFS=$'\t' read -r source target identity resolved; do
        test -L "$source"
        test "$(readlink -- "$source")" = "$target"
        test "$(stat -c '%d:%i:%s:%y' -- "$source")" = "$identity"
        count=$((count + 1))
    done < "$ALIASES"
    test "$count" -eq 12
}

case "$ACTION" in
plan)
    command -v fuser >/dev/null
    test -d "$CLIENT"
    test ! -e "$BIN"
    test ! -L "$BIN"
    for record in "$PLAN" "$ALIASES" "$JOURNAL"; do test ! -e "$record"; test ! -L "$record"; done
    (set -o noclobber; : > "$PLAN")
    (set -o noclobber; : > "$ALIASES")
    for group in client test-client test-client-sanitized dataset configs; do
        source="$CLIENT/$group"
        test -e "$source"
        test ! -L "$source"
        test "$(stat -Lc '%d' -- "$source")" = "$(stat -Lc '%d' -- "$CLIENT")"
        test -z "$(find "$source" -type l -print -quit)"
        while IFS= read -r -d '' file; do
            test "$(stat -Lc '%d' -- "$file")" = "$(stat -Lc '%d' -- "$CLIENT")"
            if fuser -- "$file" >/dev/null 2>&1; then printf 'Active file: %s\n' "$file" >&2; exit 1; fi
            printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$file" "$BIN/${file#"$CLIENT/"}" \
                "$(stat -Lc '%d:%i:%s:%y' -- "$file")" \
                "$(sha256sum -- "$file" | cut -d ' ' -f 1)" >> "$PLAN"
        done < <(find "$source" -type f -print0 | sort -z)
    done
    for source in \
        "$ROOT/dataset/inputs" "$ROOT/dataset/outputs" \
        "$ROOT/dataset/tiktoken.model" "$ROOT/dataset/vocabulary.bin" \
        "$ROOT/configs/tokenizer_config.json" \
        /opt/clover-k3/clover-data/inputs /opt/clover-k3/clover-data/outputs \
        /opt/clover-k3/direct-equation-20261001-a/tiktoken.model \
        /opt/clover-k3/direct-equation-20261001-a/data/vocabulary.bin \
        /root/k3model/tokenizer_config.json /opt/clover-k3/seed /opt/clover-k3/fruit; do
        test -L "$source"
        resolved=$(readlink -e -- "$source")
        case "$resolved" in "$CLIENT/dataset/"*|"$CLIENT/configs/"*) ;; *) exit 1 ;; esac
        printf '%s\t%s\t%s\t%s\n' "$source" "$(readlink -- "$source")" \
            "$(stat -c '%d:%i:%s:%y' -- "$source")" "$resolved" >> "$ALIASES"
    done
    check_files source
    check_aliases
    sync -f "$PLAN"
    sync -f "$ALIASES"
    printf 'PASS: 27 files and 12 aliases recorded; no runtime files moved\n'
    ;;
move)
    test ! -e "$BIN"
    test ! -L "$BIN"
    test ! -e "$JOURNAL"
    test ! -L "$JOURNAL"
    check_files source
    check_aliases
    mkdir -- "$BIN"
    (set -o noclobber; : > "$JOURNAL")
    for group in client test-client test-client-sanitized dataset configs; do
        test ! -e "$BIN/$group"
        test ! -L "$BIN/$group"
        mv -T -- "$CLIENT/$group" "$BIN/$group"
        printf '%s\tMOVED\n' "$group" >> "$JOURNAL"
        sync -f "$JOURNAL"
    done
    check_files target
    (
        cd "$BIN"
        ./test-client dataset/tiktoken.model dataset/vocabulary.bin configs/tokenizer_config.json
        test "$(./client dataset/tiktoken.model dataset/vocabulary.bin configs/tokenizer_config.json word-to-id ' Paris')" = 17374
    )
    check_aliases
    while IFS=$'\t' read -r source target identity resolved; do
        test -L "$source"
        test "$(readlink -- "$source")" = "$target"
        test "$(stat -c '%d:%i:%s:%y' -- "$source")" = "$identity"
        unlink -- "$source"
        printf '%s\tALIAS_REMOVED\n' "$source" >> "$JOURNAL"
        sync -f "$JOURNAL"
    done < "$ALIASES"
    printf 'PASS: runtime moved and tested; all 12 approved aliases removed\n'
    ;;
verify)
    check_files target
    while IFS=$'\t' read -r source target identity resolved; do test ! -e "$source"; test ! -L "$source"; done < "$ALIASES"
    test -z "$(find "$BIN" -type l -print -quit)"
    test -f "$CLIENT/client.c"
    test -f "$CLIENT/test-client.c"
    test "$(wc -l < "$JOURNAL")" -eq 17
    printf 'PASS: runtime files only in bin, source retained, aliases absent, journal complete\n'
    ;;
*) printf 'Use plan, move or verify\n' >&2; exit 2 ;;
esac