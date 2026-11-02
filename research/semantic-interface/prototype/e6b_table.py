"""Experiment 6b table: per COGS gen case, main-event role-set exact match for CSI v0.1 (sm, trf), S2 (learned linker,
trf) and S1 (LFM3 + LoRA, per seed and mean), plus S1's full-LF exact match. Missing inputs are skipped.
Usage: python e6b_table.py <results dir> [<S1 run dir> ...]"""
import glob
import json
import os
import sys

STRUCTURAL = {"cp_recursion", "pp_recursion", "obj_pp_to_subj_pp"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def s1_cases(run_dir, summary=None):
    by, tot = {}, [0, 0, 0]
    if summary is not None:
        files = {k: v for k, v in summary.items() if k.startswith("eval_gen_shard")}
        shards = [files[k] for k in sorted(files)]
        test = summary.get("eval_test_shard0of1.json")
    else:
        shards = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run_dir, "eval_gen_shard*of3.json")))]
        test = load(os.path.join(run_dir, "eval_test_shard0of1.json"))
    if len(shards) != 3:
        return None, None
    for r in shards:
        for case, v in r["by_case"].items():
            b = by.setdefault(case, [0, 0, 0])
            b[0] += v["n"]
            b[1] += v["lf_exact"] * v["n"]
            b[2] += v["role_exact"] * v["n"]
    for v in by.values():
        for i in range(3):
            tot[i] += v[i]
    out = {c: {"lf": v[1] / v[0], "role": v[2] / v[0]} for c, v in by.items()}
    out["_gen"] = {"lf": tot[1] / tot[0], "role": tot[2] / tot[0]}
    if test:
        out["_test"] = {"lf": test["lf_exact"], "role": test["role_exact"]}
    return out, len(shards)


def main():
    d = sys.argv[1]
    cols = {}
    for tag, f in (("CSI-sm", "e6a_cogs.json"), ("CSI-trf", "e6a_cogs_en_core_web_trf.json")):
        r = load(os.path.join(d, f))
        if r:
            g = r["splits"]["gen"]
            cols[tag] = {c: v["exact_match"] for c, v in g["by_case"].items()}
            cols[tag]["_gen"] = g["exact_match"]
            cols[tag]["_test"] = r["splits"]["test"]["exact_match"]
    r = load(os.path.join(d, "e6b_s2_trf.json"))
    if r:
        cols["S2"] = {c: v["role_exact_6a"] for c, v in r["gen"]["by_case"].items()}
        cols["S2"]["_gen"], cols["S2"]["_test"] = r["gen"]["role_exact_6a"], r["test"]["role_exact_6a"]
    r = load(os.path.join(d, "e6b_s2b_trf.json"))
    if r:
        cols["S2b(post-hoc)"] = {c: v["role_exact_6a"] for c, v in r["gen"]["by_case"].items()}
        cols["S2b(post-hoc)"]["_gen"], cols["S2b(post-hoc)"]["_test"] = r["gen"]["role_exact_6a"], r["test"]["role_exact_6a"]
    s1 = []
    summary = load(os.path.join(d, "e6b_s1_summary.json")) or {}
    runs = [(k, None, v) for k, v in summary.items()] + [(os.path.basename(p), p, None) for p in sys.argv[2:]]
    for name, path, summ in runs:
        o, _ = s1_cases(path, summ)
        if o:
            s1.append(o)
            cols[f"S1-{name}"] = {c: v["role"] for c, v in o.items()}
    if s1:
        cases = set.intersection(*(set(o) for o in s1))
        cols["S1-mean"] = {c: sum(o[c]["role"] for o in s1) / len(s1) for c in cases}
        cols["S1-mean-LF"] = {c: sum(o[c]["lf"] for o in s1) / len(s1) for c in cases}
    cases = sorted({c for v in cols.values() for c in v if not c.startswith("_")})
    names = list(cols)
    print("case".ljust(56) + "".join(n[:12].rjust(13) for n in names))
    for c in ["_test", "_gen"] + cases:
        mark = " *" if c in STRUCTURAL else ""
        print((c + mark).ljust(56) + "".join((f"{cols[n][c]:.3f}" if c in cols[n] else "-").rjust(13) for n in names))
    for n in names:
        vals = [cols[n][c] for c in STRUCTURAL if c in cols[n]]
        if vals:
            print(f"structural mean {n}: {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
