#!/bin/bash
# The driver-v2 campaign, in order (one run at a time; progress in out/mut2/progress.txt). usage (box, ~/rustmig/loader/verify): nohup mutants/v2_campaign.sh [STAGES] > /dev/null 2>&1 &
# STAGES (default: all) = any of: ctl (unmodified crate: run.sh N=200 three times, N=1000 once, summaries kept, raw outputs deleted), all (every mutant once, tag v2a), rep (racy and gap mutants, tags v2b v2c),
# soak (N=1000 of the mutants no run caught before, plus M00, tag v2s).  A stage is skipped if its mark file out/mut2/stage-NAME.done exists.
cd "$(dirname "$0")/.."
STAGES=${1:-"ctl all rep soak"}
mkdir -p out/mut2
P=out/mut2/progress.txt
ALL="M00 $(python3 mutants/mk_mutants.py --list | awk '{print $1}' | grep -v '^M00$' | tr '\n' ' ')"
RACY="M02 M05 M05w M06 M07 M09 M10 M11 M11w M13 M14 M15 M16 M17 M22 M23 M25 M26 M33 M46 M82 M83 M84"
GAP="M27 M51 M58 M59 M87 M92"
UNC="M05 M06 M17 M27 M50 M51 M58 M59 M75 M76 M82 M83 M84 M87 M92"
KEEP="M05 M14 M25 M27 M33 M46 M51 M58 M59 M82 M83 M84 M87 M92 M11"
for st in $STAGES; do
    [ -f out/mut2/stage-$st.done ] && { echo "$(date -u +%T) stage $st skipped (done)" >> $P; continue; }
    echo "$(date -u +%T) STAGE $st start" >> $P
    case $st in
    ctl)
        for t in a b c; do
            rm -rf out-v2$t; ./run.sh 200 out-v2$t > out/mut2/ctl-$t.txt 2>&1
            ./stats.sh out-v2$t > out/mut2/ctl-$t.stats.txt 2>&1
            cp out-v2$t/units.txt out/mut2/ctl-$t.units.txt; gzip -f out/mut2/ctl-$t.units.txt
            rm -rf out-v2$t
            echo "$(date -u +%T) ctl $t: $(head -3 out/mut2/ctl-$t.txt | tr '\n' ' ' | cut -c1-200) free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)G" >> $P
        done
        rm -rf out-v2s; ./run.sh 1000 out-v2s > out/mut2/ctl-soak.txt 2>&1
        ./stats.sh out-v2s > out/mut2/ctl-soak.stats.txt 2>&1
        cp out-v2s/units.txt out/mut2/ctl-soak.units.txt; gzip -f out/mut2/ctl-soak.units.txt
        rm -rf out-v2s
        echo "$(date -u +%T) ctl soak: $(head -3 out/mut2/ctl-soak.txt | tr '\n' ' ' | cut -c1-200)" >> $P
        ;;
    all)  N=200 TAG=v2a MAXFAIL=100 KEEP="$KEEP" mutants/run_mut2.sh $ALL ;;
    rep)
        N=200 TAG=v2b MAXFAIL=100 KEEP="$KEEP" mutants/run_mut2.sh $RACY $GAP
        N=200 TAG=v2c MAXFAIL=100 KEEP="$KEEP" mutants/run_mut2.sh $RACY $GAP
        ;;
    soak) N=1000 TAG=v2s MAXFAIL=100 KEEP="" mutants/run_mut2.sh M00 $UNC ;;
    esac
    touch out/mut2/stage-$st.done
    echo "$(date -u +%T) STAGE $st done; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)G; out/mut2=$(du -sh out/mut2 | cut -f1)" >> $P
done
echo "$(date -u +%T) CAMPAIGN DONE" >> $P
