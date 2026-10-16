#!/usr/bin/env python3
"""Step 2: what does a longer context actually cost per token?

One run feeds each token back, so context sweeps from the seed length upward
and the slope is measurable directly rather than inferred. The question is not
only whether tokens get slower but which part gets slower: KDA layers hold
fixed-size recurrent state, while the 24 MLA layers attend over every stored
position, so growth confined to attention means something very different from
growth spread across every operator.

Reports binned averages so a hundred rows become a readable trend, then fits a
straight line per operator against context length and ranks by slope. A slope
indistinguishable from zero is the result worth having here - it would mean
context is close to free in this range.

usage: context-trend.py [dir] [seed-length]
"""
import json
import os
import sys


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("STEP_JSON "):
            out.append(json.loads(line[len("STEP_JSON "):]))
    return out


def fit(xs, ys):
    """least squares slope and intercept"""
    n = float(len(xs))
    if n < 2:
        return 0.0, ys[0] if ys else 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctx"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    steps = load(os.path.join(d, "steps.jsonl"))
    if not steps:
        print("no steps in %s" % d)
        return 1

    # the lane path passes position 0 to evaluate_tokens, so the position field
    # is unusable in this build; require it to be strictly increasing or fall back
    raw = [s.get("position", 0) for s in steps]
    increasing = all(raw[i] < raw[i + 1] for i in range(len(raw) - 1))
    if increasing:
        pos = raw
    else:
        pos = [seed + i for i in range(len(steps))]
        print("position field is not increasing (%s...) - using seed %d plus step order"
              % (raw[:4], seed))
    wall = [s.get("timed_seconds", s.get("seconds", 0.0)) for s in steps]
    print("%d steps, context %d to %d" % (len(steps), min(pos), max(pos)))

    # cumulative io counters -> per step
    rb = [s.get("expert_read_bytes", 0) for s in steps]
    cum = len(rb) > 1 and all(rb[i] <= rb[i + 1] for i in range(len(rb) - 1))
    drb = [rb[0]] + [rb[i] - rb[i - 1] for i in range(1, len(rb))] if cum else rb

    print("\nbinned trend (10 bins)")
    print("  %-14s %8s %9s %9s %9s %9s"
          % ("context", "steps", "wall s", "ops s", "gap s", "readGB"))
    nb = 10
    per = max(1, len(steps) // nb)
    for b in range(0, len(steps), per):
        chunk = list(range(b, min(b + per, len(steps))))
        if not chunk:
            continue
        w = sum(wall[i] for i in chunk) / len(chunk)
        o = sum(sum(steps[i].get("operators", {}).values()) for i in chunk) / len(chunk)
        g = w - o
        r = sum(drb[i] for i in chunk) / len(chunk) / 1e9
        print("  %-14s %8d %9.4f %9.4f %9.4f %9.2f"
              % ("%d-%d" % (pos[chunk[0]], pos[chunk[-1]]), len(chunk), w, o, g, r))

    # drop the first few steps: cache warming, not context
    warm = max(10, len(steps) // 10)
    xs = pos[warm:]
    print("\nslopes fitted over context %d..%d (first %d steps dropped as cache warming)"
          % (xs[0], xs[-1], warm))

    sw, iw = fit(xs, wall[warm:])
    print("  wall          %+.6f s per token of context  (intercept %.4f s)" % (sw, iw))
    print("  over 100 tokens of context that is %+.3f s" % (sw * 100))

    names = set()
    for s in steps:
        names.update(s.get("operators", {}))
    rows = []
    for nm in names:
        ys = [s.get("operators", {}).get(nm, 0.0) for s in steps[warm:]]
        sl, ic = fit(xs, ys)
        rows.append((sl, nm, ic, sum(ys) / len(ys)))
    rows.sort(key=lambda r: -abs(r[0]))
    print("\n  %-26s %14s %10s %10s" % ("operator", "slope s/token", "mean s", "per 100"))
    for sl, nm, ic, mean in rows[:8]:
        print("  %-26s %+14.7f %10.4f %+10.4f" % (nm, sl, mean, sl * 100))

    gaps = [wall[i] - sum(steps[i].get("operators", {}).values()) for i in range(warm, len(steps))]
    sg, _ = fit(xs, gaps)
    print("  %-26s %+14.7f %10.4f %+10.4f"
          % ("UNACCOUNTED (io wait)", sg, sum(gaps) / len(gaps), sg * 100))

    print()
    first = sum(wall[warm:warm + 10]) / 10.0
    last = sum(wall[-10:]) / 10.0
    print("first 10 after warming: %.4f s   last 10: %.4f s   growth %.1f%%"
          % (first, last, 100.0 * (last - first) / first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
