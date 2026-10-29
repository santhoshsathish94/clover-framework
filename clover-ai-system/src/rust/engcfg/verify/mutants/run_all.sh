#!/bin/bash
# Builds each mutant crate and runs the UNMODIFIED driver (verify/run.sh, default full settings) against its .so:
# once as shipped (the driver stops scheduling at the first failure) and once with --keep-going (every unit runs; counts of failing units per scenario).
# Usage: run_all.sh [names...]   (default: every directory under verify/work/mut/M*). Logs: verify/out/mut/<name>/ and <name>-kg/; summary: verify/out/mut/summary.txt
# Disk rules: after each run raw files > 5 MB are deleted; df / du are logged after every 5 mutants; the script stops when free space on / is below 40 GB.
# Env VLIMIT_KB: `ulimit -v` of each driver run in KB (default 12582912 = 12 GB as in the first campaign; the driver-v2 campaign used 3145728 = 3 GB).
export PATH=$HOME/.cargo/bin:$PATH
V=$HOME/rustmig/engcfg/verify
W=$V/work/mut
O=$V/out/mut
mkdir -p "$O"
NAMES="$@"
[ -z "$NAMES" ] && NAMES=$(cd "$W" && ls -d M[0-9]* | sort)
count=0
for n in $NAMES; do
  mkdir -p "$O/$n"
  ( cd "$W/$n" && nice -n 10 cargo build --release --offline > "$O/$n/build.log" 2>&1; echo $? > "$O/$n/build.exit" )
  so="$W/$n/target/release/libengcfg.so"
  echo "$n build.exit=$(cat $O/$n/build.exit) warnings=$(grep -c '^warning' $O/$n/build.log) so_md5=$(md5sum < $so 2>/dev/null | cut -d' ' -f1)" >> "$O/summary.txt"
  if [ "$(cat $O/$n/build.exit)" = "0" ]; then
    for mode in "" "-kg"; do
      d="$O/$n$mode"
      rm -rf "$d"
      extra=""; [ -n "$mode" ] && extra="--keep-going"
      start=$(date +%s)
      ( ulimit -v ${VLIMIT_KB:-12582912}; cd "$V"; OUT="$d" timeout 1200 ./run.sh "$so" $extra > "$O/$n$mode.runout" 2>&1; echo "run.sh exit $?" >> "$O/$n$mode.runout" )
      el=$(( $(date +%s) - start ))
      find "$d" -type f -size +5M -delete 2>/dev/null
      nunit=$(grep -c '^UNIT' "$d/units.log" 2>/dev/null)
      nfail=$(grep -c -E '^UNIT .* (FAIL|CRASH)' "$d/units.log" 2>/dev/null)
      byscen=$(grep -E '^UNIT .* (FAIL|CRASH)' "$d/units.log" 2>/dev/null | awk '{c[$2]++} END {for (s in c) printf "%s:%d ", s, c[s]}')
      echo "$n$mode done in ${el}s: units_logged=$nunit failing=$nfail by_scenario=[$byscen] | $(grep -E '^TOTAL' $d/stdout.txt) | $(tail -1 $d/stdout.txt)" >> "$O/summary.txt"
      gzip -f "$d/units.log" 2>/dev/null
    done
  fi
  rm -rf "$W/$n/target/release/deps" "$W/$n/target/release/build" "$W/$n/target/release/incremental" 2>/dev/null
  count=$((count + 1))
  if [ $((count % 5)) -eq 0 ]; then
    free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    echo "DISK after $count mutants: free=${free}G; out/mut $(du -xsh $O | cut -f1); work/mut $(du -xsh $W | cut -f1)" >> "$O/summary.txt"
    if [ "$free" -lt 40 ]; then echo "STOP: free space below 40 GB" >> "$O/summary.txt"; exit 1; fi
  fi
done
echo ALLDONE >> "$O/summary.txt"
