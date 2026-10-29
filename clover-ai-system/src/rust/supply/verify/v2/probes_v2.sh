#!/bin/bash
# Re-runs the independent verifier's PROBES (a mutant plus counters of how often the changed path / the state where original and mutant differ is reached) on driver v2, for the 8 mutants it
# called equivalent (M16 M26 M41 M57 M90 M92 M97; M81 is byte-identical to the control and has no probe) and for the ones the v2 driver now catches (M42 M43 M52 M62 M67 M74 M76 M93).
# The author's mk_mutants-style probe scripts are used through copies in which only the output directory is changed (mutants/ -> v2/logs/probes-v2/), so nothing under mutants/ is overwritten.
# Output: v2/logs/probes-v2/ (probe-build.out, probes.txt, probe-<P>.events.txt, probe-summary.txt, P*.diff). Run it when no other run is using out/mut.
cd "$(dirname "$0")"
OUT=logs/probes-v2; mkdir -p $OUT
sed 's#DIFFS = os.path.join(SRC, "verify/mutants")#DIFFS = os.path.join(SRC, "verify/v2/logs/probes-v2")#' ../mutants/mk_probes.py > _mk_probes.py
sed 's#mutants/logs#v2/logs/probes-v2#g' ../mutants/run_probes.sh > _run_probes.sh
sed 's#cd "$(dirname "$0")/logs"#cd "$(dirname "$0")/logs/probes-v2"#' ../mutants/probe_summary.sh > _probe_summary.sh
chmod +x _run_probes.sh _probe_summary.sh
PROBES="P16 P26 P41 P42 P43 P52 P57 P62 P67 P74 P76 P90 P92 P93 P97"
python3 _mk_probes.py $PROBES > $OUT/probe-build.out 2>&1
( ulimit -v 3145728; ./_run_probes.sh $PROBES )
./_probe_summary.sh > /dev/null
rm -f _mk_probes.py _run_probes.sh _probe_summary.sh
cat $OUT/probes.txt | grep -v 'tag:' | head -40
