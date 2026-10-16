#!/bin/sh
# What did this session actually save in the default configuration?
#
# The expert cache gives 1.30x but was turned back off by direction, since its
# 38.5 GB is the RAM long context needs. So the deployed default may be exactly
# as fast as it was before. One measurement said 3.5943 s against 3.7463, about
# 4 percent, but the same configuration also measured 3.697 earlier, so the
# spread is comparable to the effect.
#
# Alternating A/B, three each. Alternating because an earlier A/B in this
# project compared the baseline against itself and produced a textbook null,
# and because position in a run sequence has measurably moved results before.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OLD=/tmp/clover-one.BACKUP-2026-10-15
NEW=$S/bin/clover-one

[ -x "$OLD" ] || { echo "no backup of the pre-session binary at $OLD"; exit 1; }
echo "old $(md5sum $OLD | cut -c1-12)   new $(md5sum $NEW | cut -c1-12)"

for round in 1 2 3; do
    for arm in old new; do
        [ "$arm" = old ] && BIN=$OLD || BIN=$NEW
        BIN="$BIN" OUT=/tmp/ab-$arm-$round CACHE=0 sh /tmp/ta2.sh > /dev/null 2>&1
        s=$(python3 - "/tmp/ab-$arm-$round/steps.jsonl" <<'PY'
import json,sys
rows=[json.loads(l[10:]) for l in open(sys.argv[1]) if l.startswith("STEP_JSON ")]
t=[r.get("timed_seconds",0) for r in rows][1:]
print("%.4f" % (sum(t)/len(t)) if t else "nan")
PY
)
        echo "round $round $arm  $s s/token"
    done
done
