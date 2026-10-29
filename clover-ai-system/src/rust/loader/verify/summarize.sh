#!/bin/bash
# Summary lines of a run from the raw unit lines: ./summarize.sh [outdir]
OUT=${1:-out}
cd "$(dirname "$0")"
U=$OUT/units.txt
echo "SUMMARY units=$(wc -l < $U) ref_pass=$(grep -c ' ref=PASS ' $U) rust_pass=$(grep -c ' rust=PASS ' $U) canon_same=$(grep -c 'canon=SAME' $U) ref_rc0=$(grep -c ' ref_rc=0 ' $U) rust_rc0=$(grep -c ' rust_rc=0 ' $U)"
for s in fifo barrier wait stress race init2 open shm natural real fatal book hotwait race2 big fd0 stack guard; do
  n=$(grep -c "^UNIT $s " $U)
  [ "$n" = 0 ] && continue
  echo "SCENARIO $s units=$n ref_pass=$(grep "^UNIT $s " $U | grep -c ' ref=PASS ') rust_pass=$(grep "^UNIT $s " $U | grep -c ' rust=PASS ') canon_same=$(grep "^UNIT $s " $U | grep -c 'canon=SAME')"
done
echo "CHECKS ref=$(cat $OUT/*.ref.txt | grep '^RESULT' | sed 's/.*checks=\([0-9]*\).*/\1/' | awk '{s+=$1} END{print s}') rust=$(cat $OUT/*.rust.txt | grep '^RESULT' | sed 's/.*checks=\([0-9]*\).*/\1/' | awk '{s+=$1} END{print s}')"
echo "O_DIRECT natural answers (layer lines, rust runs): /tmp accepted=$(cat $OUT/open.*.rust.txt | grep -c 'open=m0:direct-ok') refused=$(cat $OUT/open.*.rust.txt | grep -c 'open=m0:direct-refused'); /dev/shm accepted=$(cat $OUT/shm.*.rust.txt | grep -c 'open=m0:direct-ok') refused=$(cat $OUT/shm.*.rust.txt | grep -c 'open=m0:direct-refused')"
echo "race (informational) ref violations: $(cat $OUT/race.*.ref.txt | grep '^N race trials' | awk '{t+=$4; v+=$6} END{print v" of "t" first-submit trials"}')  rust: $(cat $OUT/race.*.rust.txt | grep '^N race trials' | awk '{t+=$4; v+=$6} END{print v" of "t}')"
echo "race2 (informational) ref violations: $(cat $OUT/race2.*.ref.txt 2>/dev/null | grep '^N race2 children' | awk '{t+=$6; v+=$8} END{print v+0" of "t+0" first-submit trials"}')  rust: $(cat $OUT/race2.*.rust.txt 2>/dev/null | grep '^N race2 children' | awk '{t+=$6; v+=$8} END{print v+0" of "t+0}')"
echo "hotwait (rust): $(cat $OUT/hotwait.*.rust.txt 2>/dev/null | grep -h '^N hotwait' | awk '{u++; w+=$4; r+=$6; j+=$8; q+=$12; s+=$14; sig+=$18} END{print u" units, waiters "w", rounds "r", jobs "j", quiesce checks "q", stuck "s", chaos signals "sig}')"
echo "BEGIN_LINES=$(wc -l < $OUT/jobs.txt) UNIT_LINES=$(wc -l < $U)"
echo "NOT-OK UNITS:"; grep -v ' ref=PASS ' $U; grep -v ' rust=PASS ' $U; grep -v 'canon=SAME' $U
echo "F-lines: ref=$(cat $OUT/*.ref.txt | grep -c '^F ') rust=$(cat $OUT/*.rust.txt | grep -c '^F ')"
echo "stderr files: $(ls $OUT/*.err 2>/dev/null | wc -l)"
echo "outdir bytes: $(du -sb $OUT | cut -f1)"
