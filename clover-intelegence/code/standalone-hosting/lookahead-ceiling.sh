#!/bin/sh
# What is the ceiling for better prefetch? Readers can only run one layer ahead, because
# root_pull_end() drains at every layer boundary. If reads were kept busy continuously the
# request could not beat the read window, so that window is the floor.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
printf '{"input_ids":[1008,10484,318,15383,387],"max_new_tokens":2}\n' \
  | "$S/run.sh" >/tmp/ceil.out 2>/tmp/ceil.err
python3 - <<'EOF'
import json, re
lines = open("/tmp/ceil.err").read().split("\n")
steps = [l for l in lines if "STEP_JSON" in l]
stats = [json.loads(l[len("OPSTAT_JSON "):]) for l in lines if l.startswith("OPSTAT_JSON")]
prev_b = prev_s = 0.0
print("step            total   readGB  readwin   stall     ops   idle-readers  ceiling")
for i, (l, st) in enumerate(zip(steps, stats)):
    g = lambda k: float(re.search(r'"' + k + r'":([\d.]+)', l).group(1))
    tot = g("seconds")
    b = g("expert_read_bytes"); db = b - prev_b; prev_b = b
    w = g("expert_read_seconds"); dw = w - prev_s; prev_s = w
    stall = g("expert_stall_seconds")
    ops = sum(v["s"] for v in st.values())
    label = "prompt pass" if i == 0 else "output token"
    # readers idle whenever the request is running but no read window is open
    idle = tot - dw
    print("%-13s %7.2f %8.1f %8.2f %7.2f %7.2f %12.2f %8.2f" %
          (label, tot, db/1e9, dw, stall, ops, idle, max(dw, ops)))
EOF
