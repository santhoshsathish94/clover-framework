"""Analyse the human ratings. Usage: python analyze_rating.py human/ratings.csv
Written BEFORE any rating exists. Primary question: does guidance scale 1 (s1) change fluency or plan scores relative to no guidance (s0)?
Positive control: s4 should score lower on fluency than s0 (its Gen. PPL was 20.4 vs 13.0). Anchor: true stories.
Method: only plans rated for all four kinds count; per-plan paired differences, mean with a bootstrap 95% interval (2,000 resamples over plans), exact two-sided sign test on non-ties.
Repeats (same item shown twice) give the rater's test-retest agreement; the first rating of a repeated item is used in the comparison."""
import csv, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
key = json.load(open(os.path.join(HERE, "human", "key.json")))
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
rat = {r["id"]: (int(r["fluency"]), int(r["plan"])) for r in rows}
seen = {}  # ids are in presentation order, so the lower id is the first showing
first, again = {}, {}
for iid in sorted(rat):
    k = key[iid]; t = (k["plan_idx"], k["system"])
    (again if t in seen else first)[t] = rat[iid]
    seen[t] = True
plans = sorted({p for p, _ in first})
full = [p for p in plans if all((p, s) in first for s in ("true", "s0", "s1", "s4"))]
print(f"rated items {len(rat)}; plans with all four kinds rated: {len(full)}; repeats rated: {len(again)}")


def sign_p(w, l):
    n = w + l
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(min(w, l) + 1)) / 2 ** n)


def cmp(a, b, j, label):
    d = [first[(p, a)][j] - first[(p, b)][j] for p in full]
    if not d: return
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    print(f"  {label:34s} mean diff {sum(d)/len(d):+.2f}  95% CI {bs[50]:+.2f} to {bs[1949]:+.2f}  higher {w} / lower {l} / equal {len(d)-w-l}  sign p={sign_p(w, l):.3f}")


for j, name in ((0, "FLUENCY"), (1, "PLAN-FOLLOWING")):
    print(name)
    for s in ("true", "s0", "s1", "s4"):
        v = [first[(p, s)][j] for p in full]
        if v: print(f"  mean {s:5s} {sum(v)/len(v):.2f}")
    cmp("s1", "s0", j, "PRIMARY s1 - s0")
    cmp("s0", "s4", j, "control  s0 - s4")
    cmp("true", "s0", j, "anchor   true - s0")
if again:
    print("TEST-RETEST (repeated items)")
    for j, name in ((0, "fluency"), (1, "plan")):
        diffs = [abs(first[t][j] - again[t][j]) for t in again]
        print(f"  {name}: exact agreement {sum(x == 0 for x in diffs)}/{len(diffs)}, within 1 point {sum(x <= 1 for x in diffs)}/{len(diffs)}")
