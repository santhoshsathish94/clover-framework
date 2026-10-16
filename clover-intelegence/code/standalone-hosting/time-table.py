#!/usr/bin/env python3
"""Step 1: a time table for one token that adds up to the wall clock.

A decode step measured 3.763 s while its fourteen operator buckets summed to
2.520 s. Everything proposed so far has been aimed at the 2.52 s, which means a
third of the token has never been looked at. Reading costs CPU as well -
syscalls, copies out of page cache, waiting on the device - and none of it
lands in an operator bucket.

This reads the STEP_JSON and OPSTAT_JSON lines from a run and reports, per
decode step: wall, the sum of operators, and what is left over, with the
engine's own expert read and stall counters placed against the gap. Counters
that accumulate across the run are differenced so each step stands alone, and
whether they are per-step or cumulative is detected rather than assumed.

usage: time-table.py [dir]
"""
import json
import os
import sys

OPS_ORDER = None


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        for tag in ("STEP_JSON ", "OPSTAT_JSON "):
            if line.startswith(tag):
                out.append(json.loads(line[len(tag):]))
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/timing"
    steps = load(os.path.join(d, "steps.jsonl"))
    if not steps:
        print("no steps found in %s" % d)
        return 1

    # detect whether the io counters accumulate across steps
    rb = [s.get("expert_read_bytes", 0) for s in steps]
    cumulative = len(rb) > 1 and all(rb[i] <= rb[i + 1] for i in range(len(rb) - 1))
    print("expert_read_bytes across steps: %s" % rb)
    print("counters look %s\n" % ("CUMULATIVE - differencing" if cumulative else "per-step"))

    print("%-4s %9s %9s %9s %7s %9s %9s %9s"
          % ("step", "wall s", "ops s", "gap s", "gap %", "readGB", "read s", "stall s"))

    prev = None
    rows = []
    for s in steps:
        ops = s.get("operators", {})
        ops_sum = sum(ops.values())
        wall = s.get("timed_seconds", s.get("seconds", 0.0))

        def delta(key):
            v = s.get(key, 0.0)
            if cumulative and prev is not None:
                return v - prev.get(key, 0.0)
            return v

        rb_s = delta("expert_read_bytes")
        rs_s = delta("expert_read_seconds")
        st_s = delta("expert_stall_seconds")
        gap = wall - ops_sum
        rows.append((s.get("position"), wall, ops_sum, gap, rb_s, rs_s, st_s, ops))
        print("%-4s %9.4f %9.4f %9.4f %6.1f%% %9.2f %9.4f %9.4f"
              % (s.get("position"), wall, ops_sum, gap, 100.0 * gap / wall if wall else 0,
                 rb_s / 1e9, rs_s, st_s))
        prev = s

    # steady state: ignore the first step, which carries prefill effects
    tail = rows[1:] if len(rows) > 1 else rows
    n = float(len(tail))
    awall = sum(r[1] for r in tail) / n
    aops = sum(r[2] for r in tail) / n
    agap = sum(r[3] for r in tail) / n
    arb = sum(r[4] for r in tail) / n
    ars = sum(r[5] for r in tail) / n
    ast = sum(r[6] for r in tail) / n

    print("\nsteady state over %d steps (first excluded)" % len(tail))
    print("  wall            %8.4f s" % awall)
    print("  operators       %8.4f s   %5.1f%%" % (aops, 100.0 * aops / awall))
    print("  unaccounted     %8.4f s   %5.1f%%" % (agap, 100.0 * agap / awall))
    print("  expert bytes    %8.2f GB" % (arb / 1e9))
    print("  expert read     %8.4f s  (thread-summed)" % ars)
    print("  expert stall    %8.4f s  (thread-summed)" % ast)
    if ars > 0:
        print("  implied read rate %6.2f GB/s if read seconds were wall" % (arb / 1e9 / ars))

    # operator breakdown, steady state
    agg = {}
    for r in tail:
        for k, v in r[7].items():
            agg[k] = agg.get(k, 0.0) + v
    print("\n  %-26s %9s %8s" % ("operator", "s/token", "% wall"))
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        v /= n
        if v < 1e-4:
            continue
        print("  %-26s %9.4f %7.1f%%" % (k, v, 100.0 * v / awall))
    print("  %-26s %9.4f %7.1f%%" % ("UNACCOUNTED", agap, 100.0 * agap / awall))

    # process totals for the CPU question
    tp = os.path.join(d, "time.txt")
    if os.path.exists(tp):
        want = ("User time", "System time", "Elapsed", "Maximum resident",
                "Major", "Minor", "File system inputs", "Voluntary", "Involuntary")
        print("\nprocess totals")
        for line in open(tp):
            t = line.strip()
            if any(t.startswith(w) for w in want):
                print("  %s" % t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
