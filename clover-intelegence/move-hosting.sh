#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE="$ROOT/code"
TARGET="$SOURCE/distrubuted-hosting"
RECORD="$ROOT/HOSTING-MOVE-AX102"
ACTION=${1:-plan}
SELECTED=${2:-all}
PACKAGES=(client normalization pipeline server tansformers)
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
absent() { [[ ! -e "$1" && ! -L "$1" ]] || fail "Occupied path: $1"; }
identity() { stat -c '%d:%i:%s:%y:%a:%u:%g' -- "$1"; }

check_files() {
    local selected=$1 base=$2 relative expected hash
    while IFS=$'\t' read -r relative expected hash; do
        [[ "$relative" == "$selected/"* ]] || continue
        [[ -f "$base/$relative" && ! -L "$base/$relative" ]] || fail "Missing file: $relative"
        [[ $(identity "$base/$relative") == "$expected" ]] || fail "File identity changed: $relative"
        [[ $(sha256sum "$base/$relative" | cut -d ' ' -f 1) == "$hash" ]] || fail "File bytes changed: $relative"
    done < "$RECORD/files.tsv"
}

inactive() {
    local selected=$1 relative expected hash output status
    local files=()
    while IFS=$'\t' read -r relative expected hash; do
        [[ "$relative" == "$selected/"* ]] && files+=("$SOURCE/$relative")
    done < "$RECORD/files.tsv"
    if output=$(fuser "${files[@]}" 2>&1); then
        fail "Active reader in $selected: $output"
    else
        status=$?
        [[ $status -eq 1 && -z "$output" ]] || fail "Cannot establish reader status: $output"
    fi
}

check_aliases() {
    local path old destination expected
    while IFS=$'\t' read -r path old destination expected; do
        [[ -L "$path" && $(readlink "$path") == "$destination" ]] || fail "Alias text mismatch: $path"
        [[ $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "Alias identity mismatch: $path"
    done < "$RECORD/aliases.tsv"
}

case "$ACTION" in
plan)
    absent "$RECORD"; absent "$TARGET"
    [[ $(find "$SOURCE" -mindepth 1 -maxdepth 1 | wc -l) -eq 5 ]] || fail 'Unexpected code entries'
    for package in "${PACKAGES[@]}"; do
        [[ -d "$SOURCE/$package" && ! -L "$SOURCE/$package" ]] || fail "Invalid package: $package"
    done
    [[ $(stat -c %d "$SOURCE") == "$(stat -c %d "$ROOT")" ]] || fail 'Cross-device move'
    mkdir "$RECORD"
    : > "$RECORD/files.tsv"; : > "$RECORD/links.tsv"; : > "$RECORD/aliases.tsv"
    while IFS= read -r -d '' path; do
        printf '%s\t%s\t%s\n' "${path#"$SOURCE/"}" "$(identity "$path")" "$(sha256sum "$path" | cut -d ' ' -f 1)" >> "$RECORD/files.tsv"
    done < <(find "$SOURCE" -type f -print0 | sort -z)
    while IFS= read -r -d '' path; do
        destination=$(readlink -f "$path")
        [[ "$path" == */bin/dataset && "$destination" == "$ROOT/dataset/"* ]] || fail "Unexpected code link: $path"
        absent "$path.hosting-link"
        printf '%s\t%s\t%s\t%s\n' "${path#"$SOURCE/"}" "$(readlink "$path")" "$destination" "$(stat -Lc '%d:%i' "$path")" >> "$RECORD/links.tsv"
    done < <(find "$SOURCE" -type l -print0 | sort -z)
    [[ $(wc -l < "$RECORD/links.tsv") -eq 95 ]] || fail 'Expected 95 runtime data links'
    while IFS= read -r -d '' path; do
        old=$(readlink "$path")
        [[ "$old" == "$SOURCE/"* ]] || continue
        destination=$(readlink -f "$path")
        [[ "$destination" == "$ROOT/dataset/"* ]] || fail "Alias not backed by central data: $path"
        absent "$path.hosting-link"
        printf '%s\t%s\t%s\t%s\n' "$path" "$old" "$destination" "$(stat -Lc '%d:%i' "$path")" >> "$RECORD/aliases.tsv"
    done < <(find "$ROOT/dataset" /opt/clover-k3/clover-data -type l -print0)
    for package in "${PACKAGES[@]}"; do inactive "$package"; done
    : > "$RECORD/journal.tsv"; : > "$RECORD/completed.txt"
    printf 'READY\n' > "$RECORD/state.txt"
    sync -f "$RECORD/state.txt"
    printf 'PASS: five packages, %s files hashed, 95 links, %s old-code aliases; nothing moved\n' "$(wc -l < "$RECORD/files.tsv")" "$(wc -l < "$RECORD/aliases.tsv")"
    ;;
move)
    [[ $(cat "$RECORD/state.txt") == READY ]] || fail 'Missing complete plan'
    while IFS=$'\t' read -r path old destination expected; do
        [[ $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "Alias changed: $path"
        if [[ $(readlink "$path") == "$destination" ]]; then continue; fi
        [[ $(readlink "$path") == "$old" ]] || fail "Unexpected alias: $path"
        absent "$path.hosting-link"
        ln -s "$destination" "$path.hosting-link"
        mv -T "$path.hosting-link" "$path"
        printf '%s\tALIAS_CENTRALIZED\n' "$path" >> "$RECORD/journal.tsv"
    done < "$RECORD/aliases.tsv"
    check_aliases
    mkdir -p "$TARGET"
    for package in "${PACKAGES[@]}"; do
        [[ "$SELECTED" == all || "$SELECTED" == "$package" ]] || continue
        if grep -Fxq "$package" "$RECORD/completed.txt"; then check_files "$package" "$TARGET"; continue; fi
        absent "$TARGET/$package"
        check_files "$package" "$SOURCE"
        inactive "$package"
        printf '%s\tSTARTED\n' "$package" >> "$RECORD/journal.tsv"
        sync -f "$RECORD/journal.tsv"
        directory_identity=$(stat -c '%d:%i' "$SOURCE/$package")
        mv -T "$SOURCE/$package" "$TARGET/$package"
        [[ $(stat -c '%d:%i' "$TARGET/$package") == "$directory_identity" ]] || fail "Package identity changed: $package"
        while IFS=$'\t' read -r relative old destination expected; do
            [[ "$relative" == "$package/"* ]] || continue
            path="$TARGET/$relative"
            [[ -L "$path" && $(readlink "$path") == "$old" ]] || fail "Moved link changed: $path"
            absent "$path.hosting-link"
            ln -s "$(realpath --relative-to="$(dirname "$path")" "$destination")" "$path.hosting-link"
            mv -T "$path.hosting-link" "$path"
            [[ $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "Moved link identity mismatch: $path"
        done < "$RECORD/links.tsv"
        check_files "$package" "$TARGET"
        printf '%s\tMOVED_AND_VERIFIED\n' "$package" >> "$RECORD/journal.tsv"
        printf '%s\n' "$package" >> "$RECORD/completed.txt"
        sync -f "$RECORD/completed.txt"
        printf 'PASS: %s moved with unchanged file hashes and valid dataset links\n' "$package"
    done
    ;;
verify)
    [[ $(wc -l < "$RECORD/completed.txt") -eq 5 ]] || fail 'Not all packages moved'
    [[ $(find "$SOURCE" -mindepth 1 -maxdepth 1 | wc -l) -eq 1 ]] || fail 'Old entries remain in code'
    for package in "${PACKAGES[@]}"; do check_files "$package" "$TARGET"; done
    [[ $(find "$TARGET" -type f | wc -l) -eq $(wc -l < "$RECORD/files.tsv") ]] || fail 'File count mismatch'
    while IFS=$'\t' read -r relative old destination expected; do
        path="$TARGET/$relative"
        [[ -L "$path" && $(readlink -f "$path") == "$destination" ]] || fail "Link target mismatch: $relative"
        [[ $(stat -Lc '%d:%i' "$path") == "$expected" ]] || fail "Dataset identity mismatch: $relative"
    done < "$RECORD/links.tsv"
    check_aliases
    printf 'PASS: only distrubuted-hosting under code; all five packages, file hashes, 95 data links and legacy aliases verified\n'
    ;;
*) fail 'Use plan, move [package|all], or verify' ;;
esac