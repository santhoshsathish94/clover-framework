"""Experiment 1N runner: CSI v0.1 compare() on PAWS labeled_final test (natural Wikipedia pairs). Measures coverage
and the scope gap of a main-proposition-only comparison; see EXPERIMENTS.md (pre-registered expectation <= 0.65).
Usage: python run_paws.py <out json>"""
import collections
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, compare  # noqa: E402
from run_e1 import balanced_acc  # noqa: E402


def main():
    from datasets import load_dataset
    paws = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    t0, rows = time.time(), []
    for a, b, y in zip(paws["sentence1"], paws["sentence2"], paws["label"]):
        c = compare(a, b)
        rows.append({"a": a, "b": b, "label": int(y), "equivalent": c["equivalent"],
                     "differences": c["differences"], "notes": c["notes"]})
    y = np.array([r["label"] for r in rows])
    covered = np.array([r["equivalent"] is not None for r in rows])
    pred = np.array([int(bool(r["equivalent"])) for r in rows])
    as_error = np.where(covered, pred, 1 - y)
    conf = collections.Counter((int(r["label"]), r["equivalent"]) for r in rows)
    unsupported = collections.Counter(n for r in rows if r["equivalent"] is None for n in r["notes"])
    diff_types = collections.Counter(d["type"] for r in rows if r["equivalent"] is False for d in r["differences"])
    errors = [r for r, c, p in zip(rows, covered, pred) if c and p != r["label"]]
    res = {"experiment": "E1N PAWS labeled_final test", "csi_version": CSI_VERSION, "n": len(rows),
           "label_balance": round(float(y.mean()), 4), "coverage": round(float(covered.mean()), 4),
           "balanced_accuracy_covered": round(balanced_acc(y[covered], pred[covered]), 4),
           "balanced_accuracy_unsupported_as_error": round(balanced_acc(y, as_error), 4),
           "confusion_label_vs_equivalent": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items(), key=str)},
           "unsupported_reasons_top": dict(unsupported.most_common(8)),
           "difference_types_when_not_equivalent": dict(diff_types.most_common()),
           "error_sample_seed7": random.Random(7).sample(errors, min(50, len(errors))),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != "error_sample_seed7"}, indent=1))
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
