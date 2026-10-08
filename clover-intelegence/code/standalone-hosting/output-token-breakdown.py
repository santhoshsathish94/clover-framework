"""Output-token generation costs nearly as much as the whole prompt pass; this shows why."""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fix.err"
lines = open(path).read().split("\n")
steps = [l for l in lines if "STEP_JSON" in l][:6]
stats = [json.loads(l[len("OPSTAT_JSON "):]) for l in lines if l.startswith("OPSTAT_JSON")][:6]

print("FRANCE: 5 prompt tokens, then 5 generated output tokens")
print("step           total  readGB  stall      Q      X   rest  unacc   hits  Qcalls")
prev_bytes = prev_stall = 0.0
for index, (line, stat) in enumerate(zip(steps, stats)):
    get = lambda key: float(re.search(r'"' + key + r'":([\d.]+)', line).group(1))
    seconds = get("seconds")
    read = get("expert_read_bytes") - prev_bytes
    prev_bytes += read
    stall = get("expert_stall_seconds") - prev_stall
    prev_stall += stall
    q = stat["Q   int8 projection"]
    x = stat["X   mxfp4 expert proj"]["s"]
    ops = sum(v["s"] for v in stat.values())
    label = "prompt pass(5)" if index == 0 else "output token %d" % index
    print("%-14s %6.2f %7.1f %6.2f %6.2f %6.2f %6.2f %6.2f %6d %7d" % (
        label, seconds, read / 1e9, stall, q["s"], x, ops - q["s"] - x,
        seconds - stall - ops, int(get("expert_result_hits")), q["calls"]))

for index, label in ((0, "prompt pass, 5 positions"), (1, "output token, 1 position")):
    q = stats[index]["Q   int8 projection"]
    print("%s : Q moved %6.2f GB -> %5.1f GB/s, %6.1f GFLOP/s, %.2f FLOP/byte" % (
        label, q["wgb"], q["wgb"] / q["s"], q["gflop"] / q["s"],
        q["gflop"] / q["wgb"]))
