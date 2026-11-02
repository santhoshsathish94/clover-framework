"""Experiment 3c report: criteria L1-L3 (EXPERIMENTS.md) from the CSI v0.1, CSI v0.2-lexical and bge-small runs.
Usage: python report_3c.py <v01 json> <v02 json> <bge json> <out json>"""
import json
import sys

import numpy as np


def paired_ci(x, y, n_boot=10000, seed=3107):
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    d = x[idx].mean(1) - y[idx].mean(1)
    return [round(float(x.mean() - y.mean()), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    v01, v02, bge = (json.load(open(p)) for p in sys.argv[1:4])
    P = v02["P"]
    out = {"P": {"v0.2": P["balanced_accuracy"], "v0.1": v01["P"]["balanced_accuracy"], "bge": bge["P"]["balanced_accuracy"],
                 "per_phenomenon_v0.2": P["per_phenomenon_accuracy"]},
           "A": {"v0.2": v02["A"]["balanced_accuracy"], "v0.1": v01["A"]["balanced_accuracy"], "bge": bge["A"]["balanced_accuracy"],
                 "per_phenomenon_v0.2": v02["A"]["per_phenomenon_accuracy"]},
           "regression": {s: {"v0.1": v01[s]["balanced_accuracy"], "v0.2": v02[s]["balanced_accuracy"]} for s in ("heldout3", "nonce2")}}
    # correctness lists are per pair; balanced accuracy differences are approximated by the per-pair accuracy difference
    vs01, vsbge = paired_ci(P["correct"], v01["P"]["correct"]), paired_ci(P["correct"], bge["P"]["correct"])
    l1 = P["balanced_accuracy"] >= 0.90 and vs01[1] > 0 and vsbge[1] > 0
    ph = P["per_phenomenon_accuracy"]
    l2 = all(v >= 0.80 for k, v in ph.items() if k.startswith("equiv")) and all(v >= 0.95 for k, v in ph.items() if k.startswith("dist"))
    l3 = all(v02[s]["balanced_accuracy"] >= v01[s]["balanced_accuracy"] - 0.005 for s in ("heldout3", "nonce2"))
    out["criteria"] = {"L1": {"pass": l1, "v0.2_minus_v0.1": vs01, "v0.2_minus_bge": vsbge},
                       "L2": {"pass": l2}, "L3": {"pass": l3}}
    json.dump(out, open(sys.argv[4], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
