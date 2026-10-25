#!/bin/bash
# Driver v2: a control run of the UNMODIFIED crate (target/release/libpoolinit.so against ref/libpoolinit_ref.so), with a recount of units.log (not of the summary).
# usage (box, ~/rustmig/poolinit/verify): v2/ctl.sh NAME [MULT] [--noA]     MULT scales the default counts (H 600, HX 400, P 600, PX 800, PR 200, M 400); scenario A (60 units) is added unless --noA
# env JOBS (default 16), RUST (default the crate's target/release/libpoolinit.so). Output out/ctl-NAME/ (units.log, summary.txt, .res; raw logs of seeds 1-2 kept), md5 of both libraries printed.
cd "$(dirname "$0")/.."
NAME=${1:?name}; MULT=${2:-1}
SCN="H,HX,P,PX,PR,M,A"; N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT)),A=60"
if [ "$3" = "--noA" ]; then SCN="H,HX,P,PX,PR,M"; N="H=$((600*MULT)),HX=$((400*MULT)),P=$((600*MULT)),PX=$((800*MULT)),PR=$((200*MULT)),M=$((400*MULT))"; fi
ulimit -v ${VMEM_KB:-3000000}
RUSTLIB=${RUST:-$HOME/rustmig/poolinit/target/release/libpoolinit.so}
md5sum "$RUSTLIB" ../ref/libpoolinit_ref.so pi_driver
rm -rf out/ctl-$NAME
JOBS=${JOBS:-16} RUST=$RUSTLIB ./run.sh ctl-$NAME --keep-raw 3 --keep-going --scn $SCN --n "$N" | grep -v '^\['
O=out/ctl-$NAME
echo "recount: UNIT=$(grep -c '^UNIT' $O/units.log) PASS=$(grep -c '^UNIT .* PASS ' $O/units.log) FAIL=$(grep -c '^UNIT .* FAIL ' $O/units.log) distinct=$(grep '^UNIT' $O/units.log | awk '{print $2,$3}' | sort -u | wc -l) res=$(find $O -name '*.res' | wc -l) VIOLATION-lines=$(grep -l VIOLATION $O/*/u*.log 2>/dev/null | wc -l)"
find $O -type f \( -name '*.log' -o -name '*.err' -o -name '*.res' \) ! -name units.log ! -name 'u1.*' ! -name 'u2.*' -delete
echo "$(du -xsh $O | cut -f1) $O; $(df -h / | tail -1)"
