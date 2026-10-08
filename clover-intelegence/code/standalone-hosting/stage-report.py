#!/usr/bin/env python3
"""Per-stage and per-layer breakdown of a stage profile.

Splits cost into what a weight sweep pays once and what each position pays, by
comparing the prompt pass (5 positions) against a decode step (1 position).
Stages whose time barely moves are fixed per sweep: those are what batching buys.
"""
import json, re, sys
from collections import defaultdict

steps = defaultdict(lambda: defaultdict(float))     # step -> stage -> ms
layers = defaultdict(lambda: defaultdict(float))    # step -> layer -> ms
positions, totals = {}, {}

for line in open(sys.argv[1], errors="ignore"):
    if line.startswith("STAGE_JSON"):
        r = json.loads(line[len("STAGE_JSON "):])
        steps[r["step"]][r["stage"]] += r["ms"]
        positions[r["step"]] = r["positions"]
        if r["stage"] == "total:layer":
            layers[r["step"]][r["layer"]] += r["ms"]
    elif line.startswith("POSITION_JSON"):
        r = json.loads(line[len("POSITION_JSON "):])
        totals[r["step"]] = r["total_seconds"]

prompt, decode = 0, 1
npos = positions[prompt]

print("step totals")
for s in sorted(totals):
    print("  step %d  %d position(s)  %.3f s" % (s, positions[s], totals[s]))

print("\nwhere a decode step spends its time (ms, summed over all 93 layers)")
print("  %-38s %9s %9s %7s  %s" % ("stage", "1 pos", "%d pos" % npos, "ratio", "scaling"))
rows = sorted(steps[decode].items(), key=lambda kv: -kv[1])
shown = 0
for name, one in rows:
    if name.startswith("total:"):
        continue
    many = steps[prompt].get(name, 0.0)
    ratio = many / one if one > 0.05 else float("nan")
    if one < 10:
        continue
    kind = ""
    if ratio == ratio:
        kind = "fixed per sweep" if ratio < 1.6 else (
               "per position" if ratio > npos * 0.7 else "partly shared")
    print("  %-38s %9.1f %9.1f %7.2f  %s" % (name, one, many, ratio, kind))
    shown += 1
    if shown >= 22:
        break

print("\nfixed vs per-position, decode step")
fixed = sum(v for k, v in steps[decode].items()
            if not k.startswith(("total:", "op:", "worker:", "detail:"))
            and steps[prompt].get(k, 0) / v < 1.6) if steps[decode] else 0
print("  decode step total      %8.1f ms" % (totals[decode] * 1000))
print("  expert read volume     %8.2f GB" % steps[decode].get("detail:expert-read-gb", 0))
print("  expert stall           %8.1f ms" % steps[decode].get("detail:expert-stall", 0))

print("\nper-layer total, decode step (ms)")
per = layers[decode]
if per:
    ordered = sorted(per.items())
    mla = [v for k, v in ordered if (k % 4 == 3 and k <= 91) or k == 92]
    kda = [v for k, v in ordered if k >= 1 and not ((k % 4 == 3 and k <= 91) or k == 92)]
    print("  layer 0 (dense)        %8.1f" % per.get(0, 0))
    if mla: print("  MLA layers  n=%-3d      %8.1f mean   %8.1f min   %8.1f max" % (len(mla), sum(mla)/len(mla), min(mla), max(mla)))
    if kda: print("  KDA layers  n=%-3d      %8.1f mean   %8.1f min   %8.1f max" % (len(kda), sum(kda)/len(kda), min(kda), max(kda)))
    print("  sum of all layers      %8.1f" % sum(per.values()))
    top = sorted(ordered, key=lambda kv: -kv[1])[:8]
    print("  most expensive layers: " + ", ".join("L%d=%.0f" % (k, v) for k, v in top))
