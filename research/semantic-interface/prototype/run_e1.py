"""Experiment 1 (E6+E7) runner: CSI v0.1 vs embedding-cosine baselines on identical generated minimal pairs.
Baseline thresholds are chosen on the SEEN lexicon (maximising balanced accuracy) and applied unchanged to UNSEEN.
Usage: python run_e1.py <out json> [--baselines name ...]
  baseline names: clover_hash clover_indra_hash bge_small lfm3_frozen_l4 core:<research/meaning-core run dir>
  (baselines need research/meaning-core/harness and clover-ai-system/mobile on sys.path; see --mc / --mobile)"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, compare  # noqa: E402
from csi.analyze import PARSER  # noqa: E402
from csi.minimal_pairs import generate  # noqa: E402


def balanced_acc(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else float("nan")
    return float((tpr + tnr) / 2)


def auc(scores, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, scores))


def per_phenomenon(rows, pred):
    by = collections.defaultdict(list)
    for r, p in zip(rows, pred):
        by[r["phenomenon"]].append(int(p == r["label"]))
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(by.items())}


def run_csi(rows):
    pred, errors, unsupported = [], [], 0
    for r in rows:
        c = compare(r["a"], r["b"])
        if c["equivalent"] is None:
            unsupported += 1
            p = -1
        else:
            p = int(c["equivalent"])
        pred.append(p)
        if p != r["label"]:
            errors.append({"a": r["a"], "b": r["b"], "label": r["label"], "phenomenon": r["phenomenon"],
                           "pred": p, "differences": c["differences"], "notes": c["notes"]})
    y = [r["label"] for r in rows]
    scored = [p if p != -1 else 1 - yy for p, yy in zip(pred, y)]  # an unsupported pair counts as an error
    return {"balanced_accuracy": round(balanced_acc(y, scored), 4),
            "coverage": round(1 - unsupported / len(rows), 4),
            "per_phenomenon_accuracy": per_phenomenon(rows, pred), "n_errors": len(errors), "errors": errors}


def make_encoder(name, mc, mobile):
    sys.path.insert(0, mc)
    sys.path.insert(0, mobile)
    if name in ("clover_hash", "clover_indra_hash", "bge_small"):
        import meaning_probes_lib as mp
        return mp.make(name)
    import meaning_core as mcore
    if name == "lfm3_frozen_l4":
        return mcore.baseline_encoder(os.environ["LFM3_DIR"], 4)
    if name.startswith("core:"):
        return mcore.load_core(os.environ["LFM3_DIR"], name[5:])
    raise SystemExit(f"unknown baseline {name}")


def cosines(fn, rows):
    texts = sorted({t for r in rows for t in (r["a"], r["b"])})
    v = np.asarray(fn(texts), dtype=np.float64)
    v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    idx = {t: i for i, t in enumerate(texts)}
    return np.array([v[idx[r["a"]]] @ v[idx[r["b"]]] for r in rows])


def run_baseline(fn, seen, unseen, test_name="unseen"):
    cs, cu = cosines(fn, seen), cosines(fn, unseen)
    ys, yu = np.array([r["label"] for r in seen]), np.array([r["label"] for r in unseen])
    cands = np.unique(cs)
    best = max(cands, key=lambda t: balanced_acc(ys, (cs >= t).astype(int)))
    pu = (cu >= best).astype(int)
    return {"threshold_from_seen": round(float(best), 4),
            "seen": {"balanced_accuracy": round(balanced_acc(ys, (cs >= best).astype(int)), 4), "auc": round(auc(cs, ys), 4)},
            test_name: {"balanced_accuracy": round(balanced_acc(yu, pu), 4), "auc": round(auc(cu, yu), 4),
                       "per_phenomenon_accuracy": per_phenomenon(unseen, pu),
                       "mean_cos_by_phenomenon": {k: round(float(np.mean([c for c, r in zip(cu, unseen) if r["phenomenon"] == k])), 4)
                                                  for k in sorted({r["phenomenon"] for r in unseen})}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--mc", default="")
    ap.add_argument("--mobile", default="")
    ap.add_argument("--lexicons", nargs="*", default=["seen", "unseen"])
    ap.add_argument("--test-lexicon", default="unseen", help="baselines: threshold from seen, applied to this lexicon")
    a = ap.parse_args()
    data = {lex: generate(lex) for lex in dict.fromkeys(["seen", a.test_lexicon] + a.lexicons)}
    res = {"experiment": "E1 = E6+E7 minimal pairs", "csi_version": CSI_VERSION, "parser": PARSER,
           "test_lexicon": a.test_lexicon,
           "sizes": {k: len(v) for k, v in data.items()},
           "label_balance": {k: round(float(np.mean([r["label"] for r in v])), 4) for k, v in data.items()},
           "csi": {}, "baselines": {}}
    for lex in a.lexicons:
        t0 = time.time()
        res["csi"][lex] = run_csi(data[lex])
        res["csi"][lex]["seconds"] = round(time.time() - t0, 1)
        print("CSI", lex, {k: v for k, v in res["csi"][lex].items() if k != "errors"}, flush=True)
    for name in a.baselines:
        t0 = time.time()
        fn = make_encoder(name, a.mc, a.mobile)
        res["baselines"][name] = run_baseline(fn, data["seen"], data[a.test_lexicon], a.test_lexicon)
        res["baselines"][name]["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps({k: v for k, v in res["baselines"][name].items() if k != a.test_lexicon}),
              f"{a.test_lexicon}:", res["baselines"][name][a.test_lexicon]["balanced_accuracy"],
              res["baselines"][name][a.test_lexicon]["auc"], flush=True)
        del fn
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
