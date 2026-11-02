"""Summarise morph_v2 result files (4c / 4f): one line per criterion, model and component columns."""
import json
import statistics
import sys


def mean(xs):
    return round(statistics.mean(xs), 4) if xs else None


for path in sys.argv[1:]:
    r = json.load(open(path))
    print("==", path.split("\\")[-1], r.get("feature_set"), r.get("seeds"))
    c1 = [r[f"c1_fold{i}"]["table"]["all"] for i in range(5) if f"c1_fold{i}" in r]
    if c1:
        ms = [t["model"] for t in c1]
        print(f"  C1 model {mean(ms)} (range {round(max(ms) - min(ms), 4)}, {len(ms)} folds), component "
              f"{mean([t['component'] for t in c1])}")
    for k in ("c2_novel_endings", "c2_novel_endings_k4"):
        if k in r:
            t = r[k]["table"]
            print(f"  {k}: model {t['all']['model']} component {t['all']['component']} | "
                  + " ".join(f"{ft} {t[ft]['model']}" for ft in ("past", "past_participle", "gerund", "third_singular")))
    if "c5_learned_not_trivial" in r:
        c5 = r["c5_learned_not_trivial"]
        print("  C5 curve", {k: v["model"] for k, v in c5["curve"].items()}, "shuffled", c5["shuffled_label_control"],
              "majority", c5["majority_edit_baseline"], "rules", c5["rule_baseline"])
    c6 = [r[f"c6_fold{i}"]["table"]["all"]["component"] for i in range(5) if f"c6_fold{i}" in r]
    if c6:
        print("  C6 generation component", mean(c6))
    for cls in ("noun", "adj"):
        v = [r[f"c7_{cls}_fold{i}"]["table"]["all"] for i in range(5) if f"c7_{cls}_fold{i}" in r]
        if v:
            print(f"  C7 {cls}: model {mean([t['model'] for t in v])} component {mean([t['component'] for t in v])}")
    if "c3_c4_full" in r:
        c3, c4 = r["c3_c4_full"]["c3_nonce"], r["c3_c4_full"]["c4_ewt"]
        print("  C3 nonce", c3["_all"], "min class", min((v["analysis"], k) for k, v in c3.items() if k != "_all"))
        print("  C4", {s: v["all"]["acc"] for s, v in c4["systems"].items()})
