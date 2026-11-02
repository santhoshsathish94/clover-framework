"""Freeze check F1 for Dictionary v1.1: re-score the saved checkpoint with code written separately from
dict_meaning.py (no shared functions), on CPU in float32, and compare item by item with the recorded results.
Usage: python check_senses.py <checkpoint dir> <data dir> <recorded run json> <out json> [--tokenizer name-or-dir]"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer, BertModel

SPLITS = ("test_n2", "test_z", "test_s", "test_l")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("data")
    ap.add_argument("recorded")
    ap.add_argument("out")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--predictions", default=None, help="write the predicted synset of every re-scored item (jsonl)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.checkpoint)
    ctx = BertModel.from_pretrained(f"{a.checkpoint}/context", torch_dtype=torch.float32).eval()
    gls = BertModel.from_pretrained(f"{a.checkpoint}/gloss", torch_dtype=torch.float32).eval()
    rows = [json.loads(line) for line in open(f"{a.data}/items.jsonl", encoding="utf-8")]
    definitions = {k: v["definition"] for k, v in json.load(open(f"{a.data}/synsets.json", encoding="utf-8")).items()}
    recorded = json.load(open(a.recorded))["results"]
    rows = [r for r in rows if r["split"] in SPLITS]
    t0 = time.time()
    wanted = sorted({c for r in rows for c in r["candidates"]})
    gvec = {}
    for i in range(0, len(wanted), 128):
        ids = wanted[i:i + 128]
        enc = tok([definitions[s] for s in ids], padding=True, truncation=True, max_length=64, return_tensors="pt")
        cls = gls(**enc).last_hidden_state[:, 0]
        gvec.update(zip(ids, cls))
    t_gloss = time.time() - t0
    out = {"checkpoint": a.checkpoint, "transformers": __import__("transformers").__version__,
           "torch": torch.__version__, "device": "cpu", "dtype": "float32", "splits": {}}
    pred_file = open(a.predictions, "w", encoding="utf-8") if a.predictions else None
    for split in SPLITS:
        sel = [r for r in rows if r["split"] == split]
        mine = []
        for i in range(0, len(sel), 64):
            batch = sel[i:i + 64]
            enc = tok([r["text"] for r in batch], padding=True, truncation=True, max_length=128,
                      return_offsets_mapping=True, return_tensors="pt")
            spans = enc.pop("offset_mapping")
            hid = ctx(**enc).last_hidden_state
            for k, r in enumerate(batch):
                s, e = spans[k, :, 0], spans[k, :, 1]
                pick = (s < r["end"]) & (e > r["start"]) & (e > s)
                vec = hid[k][pick].mean(0) if pick.any() else hid[k, 0]
                scores = [float(gvec[c] @ vec) for c in r["candidates"]]
                best = r["candidates"][max(range(len(scores)), key=scores.__getitem__)]
                mine.append(int(best == r["synset"]))
                if pred_file:
                    pred_file.write(json.dumps({"id": r["id"], "split": split, "predicted": best}) + "\n")
        rec = recorded[split]["correct"]
        assert len(rec) == len(mine), (split, len(rec), len(mine))
        out["splits"][split] = {"n": len(mine), "accuracy_recheck": round(sum(mine) / len(mine), 4),
                                "accuracy_recorded": recorded[split]["acc"],
                                "difference": round(sum(mine) / len(mine) - recorded[split]["acc"], 4),
                                "item_agreement": round(sum(x == y for x, y in zip(mine, rec)) / len(mine), 4)}
        print(split, json.dumps(out["splits"][split]), flush=True)
    out["seconds"] = {"glosses": round(t_gloss), "total": round(time.time() - t0)}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
