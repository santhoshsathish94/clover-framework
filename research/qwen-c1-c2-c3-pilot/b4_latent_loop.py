"""B4 - the paper's HybridLink on a FROZEN backbone, in the latent loop: after the prompt (and the <latent> anchor for HybridAgent-8B),
feed R(h) back as the next input embedding K times, then continue in text. Compared with plain greedy text (control) and with K random
vectors of the link output's norm. Run on HybridAgent-8B (the model the link was trained with, backbone trained too) and on the frozen
Qwen3-8B-Base (link only; the paper calls link activation alone insufficient).
Metrics per variant over the held-out questions: content-word overlap with the model's own plain answer, content-word overlap with the human
Dolly answer, share of degenerate outputs (repeated tokens / empty), and the texts for reading.
Usage: python b4_latent_loop.py --model ha|base --n 16 --out results/b4_ha.json
"""
import argparse, glob, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["ha", "base"], required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--max-new", type=int, default=48)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
snap = HA_SNAP if a.model == "ha" else glob.glob("/root/hf/hub/models--Qwen--Qwen3-8B-Base/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).eval()
link = load_link(4096, HA_SNAP + "/adapter.pt").to(torch.bfloat16).eval()
emb = model.get_input_embeddings()
anchors = a.model == "ha"
open_id = tok.convert_tokens_to_ids("<latent>") if anchors else None
close_id = tok.convert_tokens_to_ids("</latent>") if anchors else None
eos = tok.eos_token_id
rows = [r for r in dolly_rows(tok=tok) if r["split"] == "held"][: a.n]
STOP = set("the and was were that with this they them then there their have had has her his she him for not but are you all can from out one very when what said would could about into just like over some also than been will your more most only such these those which while where who okay need think let user".split())
cw = lambda t: set(w[:5] for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 4 and w not in STOP)


@torch.no_grad()
def run(prompt_ids, K, mode):
    st = {}
    def feed(e):
        o = model(inputs_embeds=e[:, None, :], past_key_values=st["pkv"], use_cache=True, output_hidden_states=True)
        st["pkv"], st["h"], st["lg"] = o.past_key_values, o.hidden_states[-1][:, -1], o.logits[:, -1]
    o = model(torch.tensor([prompt_ids]), use_cache=True, output_hidden_states=True)
    st["pkv"], st["h"], st["lg"] = o.past_key_values, o.hidden_states[-1][:, -1], o.logits[:, -1]
    if K > 0:
        if anchors:
            feed(emb(torch.tensor([open_id])))
        for _ in range(K):
            z = link(st["h"])
            if mode == "random":
                r = torch.randn_like(z.float()); z = (r / r.norm() * z.float().norm()).to(z.dtype)
            feed(z)
        if anchors:
            feed(emb(torch.tensor([close_id])))
    gen = []
    for _ in range(a.max_new):
        nid = int(st["lg"].argmax(-1))
        if nid == eos or nid == tok.convert_tokens_to_ids("<|im_end|>"):
            break
        gen.append(nid)
        feed(emb(torch.tensor([nid])))
    return tok.decode(gen, skip_special_tokens=True).strip()


def prompt_ids_of(q):
    if anchors:
        m = [{"role": "user", "content": q + "\n\nAnswer in at most two short sentences."}]
        try:
            ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        except TypeError:
            ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
    return tok(f"Question: {q}\nAnswer in at most two short sentences.\nAnswer:", add_special_tokens=False).input_ids


variants = [("text_only", 0, "link"), ("latent_K4", 4, "link"), ("random_K4", 4, "random"), ("latent_K16", 16, "link")]
res = {"model": a.model, "n": len(rows), "max_new": a.max_new, "variants": {}}
outs = {v[0]: [] for v in variants}
t0 = time.time()
for i, r in enumerate(rows):
    pid = prompt_ids_of(r["instruction"])
    for name, K, mode in variants:
        outs[name].append(run(pid, K, mode))
    log(f"{i + 1}/{len(rows)} {time.time() - t0:.0f}s")


def degenerate(t):
    w = t.split()
    return len(w) < 3 or len(set(w)) / len(w) < 0.4


for name, _, _ in variants:
    ov_self = [len(cw(o) & cw(c)) / max(len(cw(c)), 1) for o, c in zip(outs[name], outs["text_only"])]
    ov_hum = [len(cw(o) & cw(r["response"])) / max(len(cw(r["response"])), 1) for o, r in zip(outs[name], rows)]
    res["variants"][name] = {"overlap_with_own_plain_answer": round(sum(ov_self) / len(rows), 4),
                             "overlap_with_human_answer": round(sum(ov_hum) / len(rows), 4),
                             "degenerate_share": round(sum(degenerate(o) for o in outs[name]) / len(rows), 4),
                             "texts": outs[name]}
res["questions"] = [r["instruction"] for r in rows]
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: {x: y for x, y in v.items() if x != "texts"} for k, v in res["variants"].items()}, indent=1))
