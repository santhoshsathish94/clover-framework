"""Where does each model put probability at the content-word positions of the held-out test split? Mass on the
original word, on set members that are WordNet synonyms (synonym, near_synonym, inflection), on members that are
antonyms, and on members that are only related (hypernyms, co-hyponyms like knife/spoon, derivations, or no
relation): what the concept loss actually teaches.
Usage: python concept_mass.py <synonyms_test.jsonl> <out.json> <model>[@adapter] ...   (4-bit NF4, as in training)"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from concept_sets import relation

test_path, out_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [json.loads(l) for l in open(test_path, encoding="utf-8")]
results = {}
for spec in specs:
    name, _, adapter = spec.partition("@")
    tok = AutoTokenizer.from_pretrained(name)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": "cuda:0"})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tot = {"orig": 0.0, "synonym": 0.0, "antonym": 0.0, "related": 0.0, "set_total": 0.0}
    n = 0
    for r in rows:
        enc = tok(r["input_sequence"], truncation=True, max_length=256, return_tensors="pt").to("cuda:0")
        ids = enc["input_ids"][0]
        with torch.inference_mode():
            probs = model(**enc).logits[0].float().softmax(-1)
        for t in r["content_word_responses"]:
            pos = int(t["position"])
            if pos <= 0 or pos >= len(ids) or not t["synonyms"]:
                continue
            p = probs[pos - 1]
            orig = int(ids[pos])
            seen = {orig}
            syn = ant = rel = 0.0
            for s in t["synonyms"][:10]:
                tid = tok.encode(" " + s, add_special_tokens=False)
                if not tid or tid[0] in seen:
                    continue
                seen.add(tid[0])
                kind = relation(t["word"], s)
                if kind in ("synonym", "near_synonym", "inflection", "same"):
                    syn += float(p[tid[0]])
                elif kind == "antonym":
                    ant += float(p[tid[0]])
                else:
                    rel += float(p[tid[0]])
            tot["orig"] += float(p[orig]); tot["synonym"] += syn; tot["antonym"] += ant; tot["related"] += rel
            tot["set_total"] += float(p[orig]) + syn + ant + rel
            n += 1
    results[spec] = {k: round(v / n, 4) for k, v in tot.items()} | {"positions": n}
    print(spec, results[spec], flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(results, open(out_path, "w"), indent=1)
