import json, sys
import ncm_stage2 as S
import torch
CFG, tk, net, src, rep = S.load_models(torch.device("cpu"), 64)
pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
rows = S.read_pairs(sys.argv[1])
def dec(r):
    r = list(r)
    if eos in r:
        r = r[: r.index(eos)]
    return tk.pre_tokenizer.decode([t for t in r if t not in (eos, pad_id)], skip_special_tokens=True)
json.dump([dec(r["ids"]) for r in rows], open(sys.argv[2], "w", encoding="utf-8"))
print(len(rows))