#!/usr/bin/env bash
# Settle whether the prefetch touch helps or hurts, on the fixed harness, at both lane
# counts, then put the tree back.
#
# The earlier verdict came from n=2 with one arm always first. The null A/B since then
# put numbers on what this box can resolve: 0.405 s of spread at four lanes and a 0.117 s
# position effect, so anything under about 0.5 s is not claimable.
#
# One lane is the positive control. The ad-hoc runs put the effect there at 3.1 s, far
# above the floor, so if the harness cannot see it the harness is wrong. Four lanes is
# the open question and the regime the pipeline actually runs in.
#
# Shipped as a file rather than inlined into an ssh command: the previous attempt was
# inlined and PowerShell expanded $C to empty before bash saw it, so the patch never
# applied and the run silently compared an env var against itself.
set -u
C=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
HARNESS=/tmp/ab-pipeline.sh

python3 /tmp/port-prefetch-touch.py "$C/common/live-root.h" || exit 1
grep -c CLOVER_PREFETCH_TOUCH "$C/common/live-root.h"
cd "$C/pipeline" || exit 1
bash build.sh 2>&1 | tail -1

AB_OUT=/tmp/ab-t1.tsv bash "$HARNESS" 1 4 \
    'touch|CLOVER_PREFETCH_TOUCH=1' 'plain|CLOVER_PREFETCH_TOUCH=0' > /tmp/ab-t1.log 2>&1
AB_OUT=/tmp/ab-t4.tsv bash "$HARNESS" 4 4 \
    'touch|CLOVER_PREFETCH_TOUCH=1' 'plain|CLOVER_PREFETCH_TOUCH=0' > /tmp/ab-t4.log 2>&1

cp /tmp/live-root-before.h "$C/common/live-root.h"
cd "$C/pipeline" || exit 1
bash build.sh 2>&1 | tail -1
echo "marks after revert: $(grep -c CLOVER_PREFETCH_TOUCH "$C/common/live-root.h")"
touch /tmp/settle.done
