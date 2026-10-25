#!/bin/bash
# Runs the driver (default counts, MULT scales them) against probe builds (work/mut/<name>/target/release/libpoolinit.so, built by mk_probes.py) with PI_PROBE set;
# one run at a time, nice -n 10, same virtual-memory cap as run_all.sh. Output: out/probe/<name>/ (units.log, summary.txt, .res; raw logs deleted), out/probe/<name>.events, <name>.stdout.
# usage (box, ~/rustmig/poolinit/verify): mutants/run_probes.sh NAME...        env MULT (default 1), JOBS (default 8)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
ulimit -v ${VMEM_KB:-3000000}
[ -f "$PWD/mutants/nodump.so" ] && export LD_PRELOAD=$PWD/mutants/nodump.so
mkdir -p out/probe
MULT=${MULT:-1}
N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))"
for n in "$@"; do
  so=$PWD/work/mut/$n/target/release/libpoolinit.so
  [ -f "$so" ] || { echo "$n NO SO"; continue; }
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt 40 ]; then echo "$n STOP free=${free}G"; exit 3; fi
  rm -rf out/probe/$n out/probe/$n.events
  PI_PROBE=$PWD/out/probe/$n.events RUST=$so JOBS=${JOBS:-8} timeout 1500 ./run.sh probe/$n --keep-raw 0 --keep-going --n "$N" > out/probe/$n.stdout 2>&1
  echo "$n rc=$? UNIT=$(grep -c '^UNIT' out/probe/$n/units.log) PASS=$(grep -c '^UNIT .* PASS ' out/probe/$n/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' out/probe/$n/units.log) events=$(wc -l < out/probe/$n.events 2>/dev/null) md5=$(md5sum $so | cut -c1-32)"
  find out/probe/$n -type f \( -name '*.log' -o -name '*.log.gz' -o -name '*.trace' -o -name '*.err' \) ! -name units.log -delete
done
echo "$(du -xsh out/probe | cut -f1) out/probe; $(df -h / | tail -1)"
