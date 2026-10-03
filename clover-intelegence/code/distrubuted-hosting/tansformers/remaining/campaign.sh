#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=/opt/clover-k3/clover-intelegence
PACKAGES="$ROOT/code/tansformers"
WORK="$PACKAGES/remaining"
cd "$WORK"
phase=${1:?preflight, validate, move or run}

preflight() {
    command -v fuser >/dev/null
    for layer in $(seq 2 92); do
        package="$PACKAGES/transformer-$layer"
        mkdir -p "$package/bin"
        gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math \
            "$package/transformer-$layer.c" -lm -o "$package/bin/transformer-$layer"
        gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math \
            -fPIC -shared -DTRANSFORMER_NO_MAIN "$package/transformer-$layer.c" -lm -o "$package/bin/test-layer.so"
        "$package/bin/transformer-$layer" --inspect "$ROOT/dataset"
        plan="$package/MOVE-PLAN.tsv"
        test ! -e "$plan"
        test ! -L "$plan"
        (set -o noclobber; : > "$plan")
        for group in "trunk-$layer" "root-$layer" "operators/qkv-all/layer-$layer"; do
            source="$ROOT/dataset/$group"
            target="$package/bin/dataset/$group"
            test -d "$source"
            test ! -L "$source"
            test ! -e "$target"
            test ! -L "$target"
            test -z "$(find "$source" -type l -print -quit)"
            while IFS= read -r -d '' file; do
                test "$(stat -c '%d' "$file")" = "$(stat -c '%d' "$package/bin")"
                if fuser -- "$file" >/dev/null 2>&1; then printf 'Active file: %s\n' "$file" >&2; exit 1; fi
                printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$file" "$target/${file#"$source/"}" \
                    "$(stat -Lc '%d:%i:%s:%y' "$file")" "$(sha256sum "$file" | cut -d ' ' -f 1)" >> "$plan"
            done < <(find "$source" -type f -print0 | sort -z)
        done
        sync -f "$plan"
        printf 'PREFLIGHT\t%s\tPASS\n' "$layer"
    done
    sha256sum "$WORK"/reference-all.inc "$WORK"/test-reference.c "$WORK"/test-reference.sh "$WORK"/test-controls.c \
        "$PACKAGES"/transformer-{2..92}/transformer-*.c "$PACKAGES"/transformer-{2..92}/{root.h,decode.h} \
        "$PACKAGES"/transformer-{2..92}/bin/{transformer-*,test-layer.so} > "$WORK/SOURCE-LOCK.sha256"
}

validate() {
    sha256sum --check --status SOURCE-LOCK.sha256
    for layer in 2 3 12 92; do
        package="$PACKAGES/transformer-$layer"
        gcc -std=c11 -O1 -g -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math \
            -fsanitize=address,undefined -fno-omit-frame-pointer \
            "-DCANDIDATE_SOURCE=\"$package/transformer-$layer.c\"" test-controls.c -lm -o "$package/bin/test-controls"
        ASAN_OPTIONS=detect_leaks=1 "$package/bin/test-controls" "$ROOT/dataset"
    done
    test ! -e validation.log
    bash test-reference.sh all | tee validation.log
    for layer in $(seq 2 92); do for prompt in france japan; do
        test "$(grep -c "^VERIFIED$(printf '\t')$layer$(printf '\t')$prompt$(printf '\t')5$(printf '\t')" validation.log)" -eq 1
    done; done
    sha256sum --check --status SOURCE-LOCK.sha256
    printf 'PASS: all 182 layer/case comparisons completed\n'
}

check_group() {
    local plan=$1 selected=$2 side=$3 group source target identity hash path checked=0
    while IFS=$'\t' read -r group source target identity hash; do
        [[ "$group" == "$selected" ]] || continue
        path=$source
        if [[ "$side" == target ]]; then path=$target; fi
        test -f "$path"
        test ! -L "$path"
        test "$(stat -Lc '%d:%i:%s:%y' "$path")" = "$identity"
        test "$(sha256sum "$path" | cut -d ' ' -f 1)" = "$hash"
        if [[ "$side" == source ]] && fuser -- "$path" >/dev/null 2>&1; then printf 'Active file: %s\n' "$path" >&2; exit 1; fi
        if [[ "$side" == target ]]; then test "$source" -ef "$target"; fi
        checked=$((checked+1))
    done < "$plan"
    test "$checked" -gt 0
}

move() {
    sha256sum --check --status SOURCE-LOCK.sha256
    for layer in $(seq 2 92); do
        for prompt in france japan; do
            test "$(grep -c "^VERIFIED$(printf '\t')$layer$(printf '\t')$prompt$(printf '\t')5$(printf '\t')" validation.log)" -eq 1
        done
        package="$PACKAGES/transformer-$layer"
        journal="$package/MOVE-JOURNAL.tsv"
        test ! -e "$journal"
        for group in "trunk-$layer" "root-$layer" "operators/qkv-all/layer-$layer"; do
            source="$ROOT/dataset/$group"
            target="$package/bin/dataset/$group"
            test ! -L "$source"
            test ! -e "$target"
            test ! -L "$target"
            check_group "$package/MOVE-PLAN.tsv" "$group" source
            identity=$(stat -c '%d:%i:%s:%y' "$source")
            mkdir -p "$(dirname "$target")"
            mv -T "$source" "$target"
            if ! ln -s "$target" "$source"; then
                if [[ ! -e "$source" && ! -L "$source" ]]; then mv -T "$target" "$source"; fi
                exit 1
            fi
            test "$(stat -c '%d:%i:%s:%y' "$target")" = "$identity"
            check_group "$package/MOVE-PLAN.tsv" "$group" target
            test -z "$(find "$target" -type l -print -quit)"
            test "/opt/clover-k3/clover-data/$group" -ef "$target"
            printf '%s\tMOVED_AND_VERIFIED\n' "$group" >> "$journal"
            sync -f "$journal"
        done
        (cd "$package/bin"; "./transformer-$layer" --inspect)
        printf 'PACKAGED\t%s\tPASS\n' "$layer"
    done
    printf 'COMPLETE: 91 transformer packages, 182 layer/case comparisons, 273 verified directory moves\n'
}

case "$phase" in
preflight) preflight ;;
validate) validate ;;
move) move ;;
run) preflight; validate; move ;;
*) exit 2 ;;
esac