"""B4 summary: markdown table from results/b4_test.json (core_eval.py output), one row per encoder.
Usage: python b4_table.py <b4_test.json>"""
import json
import sys

r = json.load(open(sys.argv[1]))
LABEL = {"clover_hash": "Clover hash (runtime fallback)", "clover_indra_hash": "Clover Indra-hash (runtime today)",
         "bge_small": "bge-small-en-v1.5", "lfm3_frozen_l4": "LFM3 S3 frozen, state 4 mean"}
print("| Encoder | STS (9 tasks) | PAWS AUC | SNLI rank | CondaQA AUC | Server: paraphrases / contradiction |")
print("|---|---|---|---|---|---|")
for k, v in r.items():
    name = LABEL.get(k, "Meaning core " + k.split("/")[-1] if k.startswith("core:") else k)
    s = v["server_cos_to_first"]
    print(f"| {name} | {v['sts_mean']:.2f} | {v['paws_test_auc']:.3f} | {v['snli_test_rank']:.3f} | "
          f"{v['condaqa_test_auc']:.3f} | {s[1]:.3f}, {s[2]:.3f} / {s[3]:.3f} |")
