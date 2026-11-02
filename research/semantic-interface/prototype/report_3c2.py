"""Experiment 3c2 report: criteria K1-K4 (EXPERIMENTS.md), consistency checks and the pre-registered breakdowns.
Usage: python report_3c2.py <lexicon> <threshold> <csi run> <csi dev record run> <bge run> <e3c_v01> <e3c_v02> <out>
       [--gpu closeness json] [--audit e3c2_audit.json (adds the reported sensitivity analysis)]"""
import argparse
import collections
import json

import numpy as np

from threshold_3c2 import INF, decide, margin, scores, v01, v02


def per_ph(rows, pred):
    by = collections.defaultdict(list)
    for r, p in zip(rows, pred):
        by[r["phenomenon"]].append(int((p if p != -1 else 1 - r["label"]) == r["label"]))
    return {k: [round(float(np.mean(v)), 4), len(v)] for k, v in sorted(by.items())}


def cluster_ci(rows, pa, pb, n_boot=10000, seed=3111):
    """Balanced-accuracy difference a - b with a paired bootstrap that resamples whole items."""
    items = sorted({r["item"] for r in rows})
    ix = {it: i for i, it in enumerate(items)}
    cnt = np.zeros((2, len(items), 4))  # per system and item: TP, P, TN, N
    for r, *ps in zip(rows, pa, pb):
        for k, p in enumerate(ps):
            s = p if p != -1 else 1 - r["label"]
            cnt[k, ix[r["item"]]] += [r["label"] * s, r["label"], (1 - r["label"]) * (1 - s), 1 - r["label"]]
    draw = np.random.default_rng(seed).integers(0, len(items), size=(n_boot, len(items)))
    tot = cnt[:, draw].sum(axis=2)
    ba = (tot[..., 0] / tot[..., 1] + tot[..., 2] / tot[..., 3]) / 2
    full = cnt.sum(axis=1)
    point = (full[:, 0] / full[:, 1] + full[:, 2] / full[:, 3]) / 2
    d = ba[0] - ba[1]
    return [round(float(point[0] - point[1]), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def substituted(row):
    """The lexical evidence for the slot where the two sentences differ by a word (None if there is none)."""
    return next((e for e in row["lexical"] if e["slot"] in set(row["differences"]) | {x["slot"] for x in row["lexical"] if x["accepted"]}), None)


def breakdown(rows, preds, key):
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for i, r in enumerate(rows):
        k = key(r)
        for name, pred in preds.items():
            p = pred[i] if pred[i] != -1 else 1 - r["label"]
            cell = out[k][f"{name}:{'synonym' if r['label'] else 'control'}"]
            cell[0] += int(p == r["label"])
            cell[1] += 1
    return {str(k): {c: [round(a / n, 4), n] for c, (a, n) in sorted(v.items())} for k, v in sorted(out.items(), key=str)}


def main():
    ap = argparse.ArgumentParser()
    for name in ("lexicon", "threshold", "csi", "record", "bge", "v01_3c", "v02_3c", "out"):
        ap.add_argument(name)
    ap.add_argument("--gpu")
    ap.add_argument("--audit")
    a = ap.parse_args()
    lex, thr, csi, rec, bge = (json.load(open(p)) for p in (a.lexicon, a.threshold, a.csi, a.record, a.bge))
    old01, old02 = json.load(open(a.v01_3c)), json.load(open(a.v02_3c))
    tau, tau_bge, tau_e1 = thr["csi"]["tau"], thr["bge"]["tau"], bge["threshold_from_e1_seen"]
    assert float(csi["CSI_CLOSENESS"]) == tau, "the CSI run must use the dev threshold"
    shared = {it["id"]: it["shared"] for it in lex["items"]}
    rows = csi["test"]["rows"]
    m = [margin(r) for r in rows]
    sys_ = {"v0.3": [r["pred"] for r in rows], "v0.2": [v02(r) for r in rows], "v0.1": [v01(r) for r in rows],
            "bge_dev_tau": [int(r["cos"] >= tau_bge) for r in bge["test"]["rows"]],
            "bge_e1_tau": [int(r["cos"] >= tau_e1) for r in bge["test"]["rows"]]}
    assert [r["phenomenon"] for r in rows] == [r["phenomenon"] for r in bge["test"]["rows"]]
    checks = {"test_online_equals_offline": sys_["v0.3"] == decide(m, tau),
              "dev_online_equals_offline": [r["pred"] for r in csi["dev"]["rows"]] == decide([margin(r) for r in csi["dev"]["rows"]], tau),
              "dev_record_margins_equal": [margin(r) for r in rec["dev"]["rows"]] == [margin(r) for r in csi["dev"]["rows"]],
              "p3c_v02_reproduces_3c": [int((p if p != -1 else 1 - r["label"]) == r["label"]) for r, p in
                                        zip(csi["p3c"]["rows"], map(v02, csi["p3c"]["rows"]))] == old02["P"]["correct"],
              "p3c_v01_reproduces_3c": [int((p if p != -1 else 1 - r["label"]) == r["label"]) for r, p in
                                        zip(csi["p3c"]["rows"], map(v01, csi["p3c"]["rows"]))] == old01["P"]["correct"]}
    res = {"tau": tau, "tau_bge_dev": tau_bge, "tau_bge_e1": tau_e1, "checks": checks, "flagged": csi["flagged"],
           "test_items": len({r["item"] for r in rows}), "test_pairs": len(rows)}
    res["test"] = {k: {"balanced_accuracy": round(scores(rows, p)[0], 4), "per_phenomenon": per_ph(rows, p)}
                   for k, p in sys_.items()}
    ci = {k: cluster_ci(rows, sys_["v0.3"], sys_[k]) for k in ("v0.2", "v0.1", "bge_dev_tau")}
    ph = res["test"]["v0.3"]["per_phenomenon"]
    reg = {}
    for s in ("heldout3", "nonce2"):
        rr = csi[s]["rows"]
        p3, p1, pa = [r["pred"] for r in rr], [v01(r) for r in rr], decide([margin(r, align=False) for r in rr], tau)
        reg[s] = {"v0.3": round(scores(rr, p3)[0], 4), "v0.1": round(scores(rr, p1)[0], 4),
                  "v0.3_without_role_condition": round(scores(rr, pa)[0], 4),
                  "false_equivalence_v0.3": {k: v for k, v in scores(rr, p3)[1].items() if v > 0},
                  "false_equivalence_v0.1": {k: v for k, v in scores(rr, p1)[1].items() if v > 0},
                  "false_equivalence_without_role_condition": {k: v for k, v in scores(rr, pa)[1].items() if v > 0}}
    res["regression"] = reg
    checks["regression_v01_reproduces_3c"] = all(reg[s]["v0.1"] == old01[s]["balanced_accuracy"] for s in reg)
    res["criteria"] = {
        "K1": {"pass": res["test"]["v0.3"]["balanced_accuracy"] >= 0.90 and all(c[1] > 0 for c in ci.values()),
               "balanced_accuracy": res["test"]["v0.3"]["balanced_accuracy"], "difference_ci": ci},
        "K2": {"pass": all(v[0] >= 0.80 for k, v in ph.items() if k.startswith("equiv"))},
        "K3": {"pass": all(v[0] >= 0.95 for k, v in ph.items() if k.startswith("dist"))},
        "K4": {"pass": all(reg[s]["v0.3"] >= reg[s]["v0.1"] - 0.005 for s in reg)}}

    def sub_field(r, f):
        e = substituted(r)
        return None if e is None else (e.get("sense_a") or {}).get(f), None if e is None else (e.get("sense_b") or {}).get(f)

    def usage(r):
        ua, ub = sub_field(r, "usage_trained")
        return "n/a" if ua is None or ub is None else "both trained" if ua and ub else "at least one untrained"

    def poly(r):
        na, nb = sub_field(r, "n_meanings")
        if na is None or nb is None:
            return "n/a"
        n = max(na, nb)
        return "1" if n <= 1 else "2-3" if n <= 3 else "4-7" if n <= 7 else "8+"

    def first_sense(r):
        sa, sb = sub_field(r, "synset")
        return "n/a" if sa is None else "both chose the shared first sense" if sa == sb == shared[r["item"]] else "not both"

    def uncertain(r):
        pa, pb = sub_field(r, "p")
        return "n/a" if pa is None or pb is None else "top p < 0.5 for either" if min(pa, pb) < 0.5 else "both top p >= 0.5"

    def identical(r):
        sa, sb = sub_field(r, "synset")
        return "n/a" if sa is None else "same chosen meaning" if sa == sb else "different chosen meanings"

    show = {k: sys_[k] for k in ("v0.3", "v0.2", "bge_dev_tau")}
    syn_rows = [i for i, r in enumerate(rows) if r["label"] == 1]
    res["breakdowns"] = {
        "usage_trained": breakdown(rows, show, usage), "polysemy_max": breakdown(rows, show, poly),
        "dictionary_uncertain": breakdown(rows, show, uncertain),
        "part_of_speech": breakdown(rows, show, lambda r: "verb" if r["class"].startswith("verb") else "noun"),
        "synonyms_first_sense": breakdown([rows[i] for i in syn_rows], {k: [v[i] for i in syn_rows] for k, v in show.items()}, first_sense),
        "synonyms_chosen_meanings": breakdown([rows[i] for i in syn_rows], {k: [v[i] for i in syn_rows] for k, v in show.items()}, identical)}
    fin = [(x, r) for x, r in zip(m, rows) if x is not None and abs(x) != INF]
    syn_c = [x for x, r in fin if r["label"] == 1]
    res["closeness_auc_test"] = {"synonym_vs_related": auc(syn_c, [x for x, r in fin if "related" in r["phenomenon"]]),
                                 "synonym_vs_random": auc(syn_c, [x for x, r in fin if "random" in r["phenomenon"]]),
                                 "pairs_with_closeness": len(fin)}
    vals = sorted({x for x, _ in fin})
    best = max(((round(scores(rows, decide(m, t))[0], 4), round(t, 6)) for t in [(x + y) / 2 for x, y in zip(vals, vals[1:])]),
               default=None)
    res["best_tau_on_test"] = {"balanced_accuracy": best[0], "tau": best[1]} if best else None
    res["test_without_role_condition"] = round(scores(rows, decide([margin(r, align=False) for r in rows], tau))[0], 4)
    p3 = csi["p3c"]["rows"]
    res["p3c_examined"] = {"v0.3": round(scores(p3, [r["pred"] for r in p3])[0], 4), "v0.2": round(scores(p3, list(map(v02, p3)))[0], 4),
                           "per_phenomenon_v0.3": per_ph(p3, [r["pred"] for r in p3])}
    layer = collections.Counter()
    errors = []
    for r, p, x in zip(rows, sys_["v0.3"], m):
        if (p if p != -1 else 1 - r["label"]) == r["label"]:
            continue
        e = substituted(r)
        if r["label"] == 0:
            why = "control merged: no difference found (parser or lemma)" if e is None else "control merged: closeness >= tau"
        elif x is None or e is None or set(r["differences"]) - {e["slot"]}:
            why = "other difference (parser or structure)"
        elif not e["covered"]:
            why = "word not covered by the dictionary"
        elif not e["aligned"]:
            why = "role condition"
        else:
            why = "synonym: closeness below tau"
        layer[why] += 1
        errors.append({"a": r["a"], "b": r["b"], "label": r["label"], "phenomenon": r["phenomenon"], "why": why,
                       "closeness": e and e.get("closeness"), "chosen": e and [(e["sense_a"] or {}).get("synset"), (e["sense_b"] or {}).get("synset")],
                       "shared": shared[r["item"]]})
    res["errors_by_layer"] = dict(layer)
    res["errors"] = errors
    if a.audit:
        doubt = set(json.load(open(a.audit))["sensitivity_exclude"])
        keep = [i for i, r in enumerate(rows) if r["item"] not in doubt]
        kr = [rows[i] for i in keep]
        res["sensitivity_undoubted_items"] = {
            "items": len({r["item"] for r in kr}), "pairs": len(kr),
            **{k: {"balanced_accuracy": round(scores(kr, [p[i] for i in keep])[0], 4),
                   "per_phenomenon": per_ph(kr, [p[i] for i in keep])} for k, p in sys_.items()}}
    if a.gpu:
        g = json.load(open(a.gpu))
        k3 = {(e["sense_a"]["synset"], e["sense_b"]["synset"]): e["closeness"] for r in rows for e in r["lexical"]
              if e.get("closeness") is not None}
        diffs = [abs(k3[(x["a"], x["b"])] - x["closeness"]) for x in g["pairs"] if (x["a"], x["b"]) in k3]
        res["gpu_recheck"] = {"pairs": len(diffs), "max_abs_difference": round(max(diffs), 6) if diffs else None,
                              "pass": bool(diffs) and max(diffs) <= 0.001 and len(diffs) == len(k3)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=1))


if __name__ == "__main__":
    main()
