"""Freeze check F4 for Dictionary v1.1: the integration contract implemented by csi/senses.py (ARCHITECTURE.md 2b).
Thresholds set before the check first ran: agreement with the independent F1 scorer >= 0.99 on 200 test-N2 items
sampled across documents (seed 3103); every other check must hold exactly. Run twice (--digest-out, then --digest-in)
to show that a fresh process gives identical results.
Usage: python senses_contract_check.py <checkpoint> <oewn> <data dir> <f1 predictions jsonl> <out json>
                                       [--tokenizer dir] [--digest-out file | --digest-in file]"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi.senses import Senses  # noqa: E402

UPOS_OF = {"n": "NOUN", "v": "VERB", "a": "ADJ"}
WEIGHTS_SHA = {"context/model.safetensors": "e78597e79e7caa50", "gloss/model.safetensors": "0d913fb27fe37703"}


def run(s, items):
    return [s.choose(it["text"], it["start"], it["end"], it["lemma"], UPOS_OF[it["pos"]]) for it in items]


def main():
    ap = argparse.ArgumentParser()
    for name in ("checkpoint", "oewn", "data", "f1_predictions", "out"):
        ap.add_argument(name)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--digest-out", default=None)
    ap.add_argument("--digest-in", default=None)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(os.path.join(a.data, "items.jsonl"), encoding="utf-8")]
    test = [it for it in items if it["split"] == "test_n2"]
    sample = random.Random(3103).sample(test, 200)
    probes = random.Random(3104).sample(test, 30)
    s = Senses(a.checkpoint, a.oewn, tokenizer=a.tokenizer)
    cold = run(s, probes)  # first queries of this process: empty definition cache
    checks = {}
    r = s.choose("The bank raised its rates.", 4, 8, "bank", "NOUN")
    checks["schema"] = ({"lemma", "upos", "span", "scores_calibrated", "usage_trained", "provenance", "status",
                         "senses", "windowed"} <= set(r) and r["status"] == "chosen" and r["scores_calibrated"] is False
                        and all({"synset", "definition", "score", "p"} <= set(x) for x in r["senses"])
                        and abs(sum(x["p"] for x in r["senses"]) - 1) < 1e-3
                        and r["senses"] == sorted(r["senses"], key=lambda x: -x["score"]))
    pv = r["provenance"]
    checks["provenance_pins_weights_code_tokenizer_inventory"] = (
        all(pv["checkpoint_sha256"][k] == v for k, v in WEIGHTS_SHA.items()) and pv["code_sha256"] is not None
        and pv["tokenizer_sha256"] is not None and pv["inventory_sha256"] == "9ca6d1dcb75f822f")
    checks["out_of_scope"] = (s.choose("She ran quickly.", 8, 15, "quickly", "ADV")["status"] == "out_of_scope"
                              and s.choose("We ate ice cream.", 7, 16, "ice cream", "NOUN")["status"] == "out_of_scope")
    checks["invalid_span"] = all(s.choose("The bank closed.", st, en, "bank", "NOUN")["status"] == "invalid_span"
                                 for st, en in ((4, 4), (8, 4), (-1, 3), (4, 99), (3, 4)))
    r = s.choose("The blicket fell.", 4, 11, "blicket", "NOUN")
    checks["not_in_dictionary"] = r["status"] == "not_in_dictionary" and r["senses"] == [] and not s.meanings("blicket", "NOUN")
    lemma = next(l for (l, p), syns in sorted(s.inventory.items()) if p == "n" and len(syns) == 1 and len(l) > 4)
    before = s.model_calls
    r = s.choose(f"The {lemma} was there.", 4, 4 + len(lemma), lemma, "NOUN")
    checks["single_meaning_without_model_call"] = (r["status"] == "single_meaning" and s.model_calls == before
                                                   and r["senses"][0]["p"] == 1.0)
    long_text = "The committee met again. " * 60 + "He sat on the bank of the river."
    st = long_text.rindex("bank")
    r = s.choose(long_text, st, st + 4, "bank", "NOUN")
    checks["target_beyond_token_limit_is_windowed"] = r["status"] == "chosen" and r["windowed"] is True
    warm = run(s, list(reversed(probes)))[::-1]  # after many other queries, in the opposite order
    checks["independent_of_query_history"] = warm == cold
    same = sum(s.meanings(it["lemma"], UPOS_OF[it["pos"]]) == it["candidates"] for it in sample)
    preds = {}
    for line in open(a.f1_predictions, encoding="utf-8"):
        p = json.loads(line)
        preds[p["id"]] = p["predicted"]
    agree = sum(r["senses"][0]["synset"] == preds[it["id"]] for r, it in zip(run(s, sample), sample))
    checks["candidates_identical_200"] = same == len(sample)
    checks["agreement_with_f1_scorer_200"] = round(agree / len(sample), 4)
    checks["documents_in_sample"] = len({it["id"].split(":")[0] for it in sample})
    digest = hashlib.sha256(json.dumps(cold, sort_keys=True).encode()).hexdigest()
    if a.digest_out:
        open(a.digest_out, "w").write(digest)
    if a.digest_in:
        checks["identical_in_a_fresh_process"] = open(a.digest_in).read().strip() == digest
    passed = (all(v is True for v in checks.values() if isinstance(v, bool))
              and agree / len(sample) >= 0.99)
    out = {"check": "F4 integration contract", "single_meaning_example": lemma, "probe_digest": digest,
           "checks": checks, "pass": passed, "provenance": s.provenance}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
