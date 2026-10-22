"""Multi-rater analysis: human (first rated items only) and three model raters on all 84 items. Usage: python analyze_multi.py
Same comparisons as analyze_rating.py (primary s1 - s0; control s0 - s4; anchor true - s0), per rater and for the mean of the three model raters,
plus agreement between raters. Per-plan paired differences, bootstrap 95% interval over plans, exact sign test; only plans with all four kinds rated."""
import csv, json, math, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(HERE, "human")
key = json.load(open(os.path.join(H, "key.json")))
FILES = {"human": "ratings.csv", "gemini4argon": "ratings_gemini4argon.csv", "gpt6astra": "ratings_gpt6astra.csv", "opus55": "ratings_claude_opus55.csv"}
R = {}
for name, f in FILES.items():
    R[name] = {r["id"]: (int(r["fluency"]), int(r["plan"])) for r in csv.DictReader(open(os.path.join(H, f), encoding="utf-8"))}
models = [m for m in R if m != "human"]
R["mean3"] = {i: tuple(sum(R[m][i][j] for m in models) / 3 for j in (0, 1)) for i in R["opus55"]}


def first_showing(rat):
    seen, first = set(), {}
    for iid in sorted(rat):
        t = (key[iid]["plan_idx"], key[iid]["system"])
        if t not in seen:
            first[t] = rat[iid]
            seen.add(t)
    return first


def sign_p(w, l):
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, x) for x in range(min(w, l) + 1)) / 2 ** n)


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return float("nan") if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


for name in ["human"] + models + ["mean3"]:
    first = first_showing(R[name])
    full = sorted({p for p, _ in first if all((p, s) in first for s in ("true", "s0", "s1", "s4"))})
    print(f"\n== {name}: items rated {len(R[name])}, plans with all four kinds {len(full)}")
    for j, label in ((0, "fluency"), (1, "plan")):
        means = {s: sum(first[(p, s)][j] for p in full) / len(full) for s in ("true", "s0", "s1", "s4")} if full else {}
        print(f"  {label:8s} means " + "  ".join(f"{s} {m:.2f}" for s, m in means.items()))
        for a, b, tag in (("s1", "s0", "PRIMARY s1-s0"), ("s0", "s4", "control s0-s4"), ("true", "s0", "anchor true-s0")):
            d = [first[(p, a)][j] - first[(p, b)][j] for p in full]
            if not d: continue
            rng = random.Random(0); bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
            w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
            print(f"     {tag:16s} {sum(d)/len(d):+.2f}  95% CI {bs[50]:+.2f} to {bs[1949]:+.2f}  higher {w} lower {l} equal {len(d)-w-l}  sign p={sign_p(w, l):.3f}")

print("\n== agreement with the human on the items the human rated (n=%d)" % len(R["human"]))
common = sorted(R["human"])
for m in models + ["mean3"]:
    for j, label in ((0, "fluency"), (1, "plan")):
        h = [R["human"][i][j] for i in common]; x = [R[m][i][j] for i in common]
        sp = pearson(ranks(h), ranks(x))
        print(f"  {m:13s} {label:8s} Spearman {sp:.2f}  mean abs diff {sum(abs(a-b) for a,b in zip(h,x))/len(h):.2f}  mean diff (model-human) {sum(b-a for a,b in zip(h,x))/len(h):+.2f}  exact {sum(a==round(b) for a,b in zip(h,x))}/{len(h)}")
print("\n== agreement between model raters on all 84 items (Spearman)")
for i, a in enumerate(models):
    for b in models[i + 1:]:
        for j, label in ((0, "fluency"), (1, "plan")):
            print(f"  {a} vs {b} {label:8s} {pearson(ranks([R[a][k][j] for k in sorted(R[a])]), ranks([R[b][k][j] for k in sorted(R[a])])):.2f}")
print("\n== test-retest on repeated items (same plan and kind shown twice)")
rep = {}
for iid, v in key.items(): rep.setdefault((v["plan_idx"], v["system"]), []).append(iid)
rep = {t: sorted(v) for t, v in rep.items() if len(v) == 2}
for name in ["human"] + models:
    pairs = [(a, b) for a, b in rep.values() if a in R[name] and b in R[name]]
    if pairs:
        print(f"  {name:13s} pairs {len(pairs)}  fluency exact {sum(R[name][a][0]==R[name][b][0] for a,b in pairs)}  plan exact {sum(R[name][a][1]==R[name][b][1] for a,b in pairs)}")
