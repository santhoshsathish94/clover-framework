#!/bin/bash
# Builds the probe crates made by mk_probes.py and runs the unmodified driver on each (full run with --keep-going, nice 10, one at a time).
# Usage: run_probes.sh [P41 P42 ...] (default: every directory under verify/work/probe). Result: verify/out/probe/summary.txt (units failing, event counts by tag).
export PATH=$HOME/.cargo/bin:$PATH
V=$HOME/rustmig/engcfg/verify
W=$V/work/probe
O=$V/out/probe
mkdir -p "$O"
NAMES="$@"
[ -z "$NAMES" ] && NAMES=$(cd "$W" && ls -d P* | sort -V)
for n in $NAMES; do
  rm -f "$O/$n.ev"
  ( cd "$W/$n" && nice -n 10 cargo build --release --offline > "$O/$n.build.log" 2>&1; echo $? > "$O/$n.build.exit" )
  [ "$(cat $O/$n.build.exit)" = "0" ] || { echo "$n BUILD FAILED" >> "$O/summary.txt"; continue; }
  d="$O/run-$n"; rm -rf "$d"
  ( ulimit -v 12582912; cd "$V"; OUT="$d" timeout 1500 ./run.sh "$W/$n/target/release/libengcfg.so" --keep-going > "$O/$n.runout" 2>&1 )
  find "$d" -type f -size +5M -delete 2>/dev/null
  nfail=$(grep -c -E '^UNIT .* (FAIL|CRASH)' "$d/units.log" 2>/dev/null); nunit=$(grep -c '^UNIT' "$d/units.log" 2>/dev/null)
  ev=$( [ -f "$O/$n.ev" ] && sort "$O/$n.ev" | uniq -c | tr '\n' ';' || echo none )
  echo "$n units=$nunit failing=$nfail events: $ev" >> "$O/summary.txt"
  gzip -f "$d/units.log" 2>/dev/null
  rm -rf "$W/$n/target/release/deps" "$W/$n/target/release/build"
done
echo PROBES-DONE >> "$O/summary.txt"
