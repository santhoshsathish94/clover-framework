"""Merge plan shards and filter. Train: drop `bad`, leaked instructions, and plans whose content words are found in the story window < MINRET.
Valid: drop only `bad` and leaks (no retention filter, so the evaluation is not selected for easy plans).
Usage: python build_pairs.py PREFIX OUT.jsonl MINRET   (PREFIX = out/plans_train ; MINRET 0 disables the retention filter)"""
import glob, json, re, sys
prefix, out, minret = sys.argv[1], sys.argv[2], float(sys.argv[3])
STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some day".split())
LEAK = re.compile(r"story ends|plan should|write the|\.\.\.|the user|as an ai", re.I)
def pw(t): return [w for w in dict.fromkeys(re.findall(r"[a-z]+", t.lower())) if len(w) >= 4 and w not in STOP]
def ret(p, s):
    w = pw(p)
    if not w: return None
    tw = set(x[:5] for x in re.findall(r"[a-z]+", s.lower())); return sum(1 for x in w if x[:5] in tw) / len(w)
rows = [json.loads(l) for f in sorted(glob.glob(prefix + ".*.jsonl")) for l in open(f, encoding="utf-8")]
cnt = {"read": len(rows), "bad": 0, "leak": 0, "lowret": 0, "kept": 0}
with open(out, "w", encoding="utf-8") as o:
    for r in rows:
        if r["bad"]: cnt["bad"] += 1; continue
        if LEAK.search(r["plan"]): cnt["leak"] += 1; continue
        x = ret(r["plan"], r["story"])
        if minret > 0 and (x is None or x < minret): cnt["lowret"] += 1; continue
        o.write(json.dumps({"ids": r["ids"], "plan": r["plan"]}, ensure_ascii=False) + "\n"); cnt["kept"] += 1
print(cnt)
