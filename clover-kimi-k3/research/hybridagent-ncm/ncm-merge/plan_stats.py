import json, glob, re, statistics as st, sys
pat = sys.argv[1] if len(sys.argv) > 1 else "gpu-out/plans_valid.*.jsonl"
rows = [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f, encoding="utf-8")]
print("n", len(rows), "bad", sum(r["bad"] for r in rows), "capped", sum(r["capped"] for r in rows))
wc = [len(r["plan"].split()) for r in rows]; print("words mean", round(st.mean(wc), 1), "min", min(wc), "max", max(wc))
STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some day".split())
def pw(t): return [w for w in dict.fromkeys(re.findall(r"[a-z]+", t.lower())) if len(w) >= 4 and w not in STOP]
def ret(p, s):
    w = pw(p)
    if not w: return None
    tw = set(x[:5] for x in re.findall(r"[a-z]+", s.lower())); return sum(1 for x in w if x[:5] in tw) / len(w)
v = [ret(r["plan"], r["story"]) for r in rows]; v = [x for x in v if x is not None]
print("plan words found in the story window (mean)", round(st.mean(v), 3), " share of plans <0.5:", round(sum(x < 0.5 for x in v) / len(v), 3))
for r in sorted(rows, key=lambda r: ret(r["plan"], r["story"]) or 0)[:3]:
    print("LOW", round(ret(r["plan"], r["story"]), 2), "|", r["plan"], "|", r["story"][:160].replace("\n", " "))
for r in rows[:3]:
    print("SAMPLE |", r["plan"])
print("end with . or ;:", sum(r["plan"].rstrip()[-1:] in ".;" for r in rows))
