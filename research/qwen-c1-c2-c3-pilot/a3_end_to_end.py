"""A3 - end to end in the matched Qwen family (C1 = C2 = HybridAgent-8B in text mode, C3 = WMT14 NCM, optionally fine-tuned in A2).

For held-out Dolly questions: HybridAgent-8B writes its own short answer (greedy, <= --max-new tokens); NOTE it usually opens with chain-of-thought ("Okay, so the user is asking..."),
so 'its answer' is its text-mode thinking/plan prefix, the same form the Studio route used; the plan is the grounded plan
of that answer (its content words and numbers, as in A2); C3 renders the plan. Compared: the model's own answer (C1 alone), C3's
render of the plan, and controls (shuffled plan, blank plan). Metrics: content-word retention of the plan in the render and exact number
recall against the model's answer; the texts are stored for reading.
Usage: python a3_end_to_end.py --n 40 --ckpt runs/a2/last.pt --out results/a3.json     (--ckpt omitted = released generator)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2_c3_wmt14 as A2
from common import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new", type=int, default=60)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--answers", default=BASE + "/results/a3_answers.json")
ap.add_argument("--out", default=BASE + "/results/a3.json")
a = ap.parse_args()
torch.set_num_threads(24)

if os.path.exists(a.answers):
    ans = json.load(open(a.answers))
    log("reusing answers", a.answers, len(ans))
else:
    tok = AutoTokenizer.from_pretrained(HA_SNAP); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(HA_SNAP, dtype=torch.bfloat16).eval()
    qs = [r["instruction"] for r in dolly_rows() if r["split"] == "held"]
    # same held-out questions as A2's eval (first n of the 12-63-token-response filter) so the human answer is also available
    ncm_cfg, tk0, _, _, _ = A2.load_models(torch.device("cpu"))
    pairs = A2.make_pairs(tk0, "held", a.n)
    ans = []
    t0 = time.time()
    for i in range(0, len(pairs), a.bs):
        chunk = pairs[i:i + a.bs]
        prompts = []
        for p in chunk:
            m = [{"role": "user", "content": p["q"] + "\n\nAnswer in at most two short sentences."}]
            try:
                prompts.append(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False))
            except TypeError:
                prompts.append(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False))
        enc = tok(prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False)
        for p, o in zip(chunk, out):
            txt = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
            ans.append({"q": p["q"], "human": p["text"], "model_answer": txt})
        log(f"{len(ans)}/{len(pairs)} answers, {time.time() - t0:.0f}s")
    json.dump(ans, open(a.answers, "w"), indent=1, ensure_ascii=False)
    del model

dev = torch.device("cpu")
CFG, tk, net, src, rep = A2.load_models(dev, a.ckpt)
net.eval()
pad = tk.pre_tokenizer.pad_token_id
plans = [A2.plan_of(x["model_answer"]) for x in ans]
pids = A2.plan_ids_of(tk, plans)
res = {"ckpt": a.ckpt or "released WMT14 generator (zero-shot)", "n": len(ans), "settings": []}
for temp, top_k, top_p in ((0.6, 50, 0.9), (1.0, 0, 1.0)):
    m = A2.generate_texts(net, tk, pids, temp, top_k, top_p, 16, 42)
    s = A2.generate_texts(net, tk, torch.roll(pids, 1, 0), temp, top_k, top_p, 16, 42)
    b = A2.generate_texts(net, tk, torch.full_like(pids, pad), temp, top_k, top_p, 16, 42)
    ret = lambda texts: round(sum(x for x in (A2.retention(p, t) for p, t in zip(plans, texts)) if x is not None) / len(plans), 4)
    nr = lambda texts: [x for x in (A2.number_recall(t["model_answer"], g) for t, g in zip(ans, texts)) if x is not None]
    mean = lambda v: round(sum(v) / len(v), 4) if v else None
    res["settings"].append({"temp": temp, "top_k": top_k, "top_p": top_p,
        "retention_matched": ret(m), "retention_shuffled(control)": ret(s), "retention_blank(control)": ret(b),
        "retention_model_answer(ceiling)": ret([x["model_answer"] for x in ans]),
        "number_recall_matched": mean(nr(m)), "number_recall_shuffled(control)": mean(nr(s)), "number_recall_blank(control)": mean(nr(b)),
        "items": [{"q": x["q"], "model_answer": x["model_answer"], "plan": p, "c3_render": r, "shuffled_render": sr}
                  for x, p, r, sr in zip(ans, plans, m, s)]})
    print(json.dumps({k: v for k, v in res["settings"][-1].items() if k != "items"}), flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
log("wrote", a.out)
