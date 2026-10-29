#!/bin/bash
# Builds each mutant crate and runs the unmodified driver (verify/run.sh, full settings) against its .so: once as shipped (stops at first failure) and once with --keep-going.
# Usage: run_all.sh [names...]   (default: all). Logs: verify/out/mut/<name>/ and <name>-kg/; summary: verify/out/mut/summary.txt
export PATH=$HOME/.cargo/bin:$PATH
V=$HOME/rustmig/routetable/verify
W=$V/work/mut
O=$V/out/mut
mkdir -p "$O"
NAMES="$@"
[ -z "$NAMES" ] && NAMES="M0 M1 M2 M3 M4 M5 M6 M7 M8 M8L M9 M10 M11 M12 M13 M14 M15 M16 M17 M18 M19"
for n in $NAMES; do
  mkdir -p "$O/$n"
  ( cd "$W/$n" && cargo build --release --offline > "$O/$n/build.log" 2>&1; echo $? > "$O/$n/build.exit" )
  so="$W/$n/target/release/libroutetable.so"
  echo "$n build.exit=$(cat $O/$n/build.exit) so_md5=$(md5sum < $so | cut -d' ' -f1)" >> "$O/summary.txt"
  [ "$(cat $O/$n/build.exit)" = "0" ] || continue
  for mode in "" "-kg"; do
    d="$O/$n$mode"
    rm -rf "$d"
    extra=""; [ -n "$mode" ] && extra="--keep-going"
    start=$(date +%s)
    ( [ "$n" = "M8L" ] && ulimit -v 4194304; cd "$V"; OUT="$d" timeout 1500 ./run.sh "$so" $extra > "$O/$n$mode.runout" 2>&1; echo "timeout-or-run-exit $?" >> "$O/$n$mode.runout" )
    echo "$n$mode done in $(( $(date +%s) - start )) s: $(tail -1 $d/stdout.txt)  $(grep -E '^overall' $d/stdout.txt)" >> "$O/summary.txt"
  done
done
echo ALLDONE >> "$O/summary.txt"
