#!/usr/bin/env python3
"""Decompose a run's STEP_JSON telemetry. expert_read_bytes and
expert_stall_seconds are cumulative over the process, so they are differenced."""
import re, sys

Q = '"Q   int8 projection"'
X = '"X   mxfp4 expert proj"'
B = '"B   bf16 lm_head"'

def field(line, key):
    m = re.search(re.escape(key) + r':([0-9.eE+-]+)', line)
    return float(m.group(1)) if m else float('nan')

for path in sys.argv[1:]:
    rows, prev_rb, prev_st = [], 0.0, 0.0
    for line in open(path, errors="ignore"):
        if not line.startswith("STEP_JSON"):
            continue
        rb = field(line, '"expert_read_bytes"')
        st = field(line, '"expert_stall_seconds"')
        rows.append((field(line, '"seconds"'), (rb - prev_rb) / 1e9, st - prev_st,
                     field(line, Q), field(line, X), field(line, B)))
        prev_rb, prev_st = rb, st
    print("=== %s : %d evaluations" % (path, len(rows)))
    print("   #   secs   readGB  stall      Q      X      B")
    for i, r in enumerate(rows):
        print("  %2d  %5.2f  %7.1f  %5.2f  %5.2f  %5.2f  %5.2f" % ((i,) + r))
    if rows:
        print("  total  %.1f s   %.1f GB   Q %.1f   X %.1f   B %.1f" % (
            sum(r[0] for r in rows), sum(r[1] for r in rows),
            sum(r[3] for r in rows), sum(r[4] for r in rows), sum(r[5] for r in rows)))
