"""Level-0 logits of the null branch in guided generation vs a plain null-plan run (Cycle 37). Usage: cfg_l0.py CKPT PAIRS OUT_JSON
Level-0 logits depend only on the prefix, not on sampled codes, so they can be compared exactly:
  a) generate(null plans, cfg 0)          -> logits La
  b) generate(real plans, cfg 0)          -> logits Lc
  c) generate(real plans, cfg 2)          -> batch [real; null] -> Lb[:B] should equal Lc, Lb[B:] should equal La
Differences are reported against the typical logit magnitude."""
import json, sys
import ncm_stage2 as S
import torch

ck, pairs, outp = sys.argv[1:4]
dev = torch.device("cuda:0")
CFG, tk, net, src, rep = S.load_models(dev, 64, ckpt=ck)
net.eval()
pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
rows = S.read_pairs(pairs)[:48]
plans = S.encode_plans(tk, [r["plan"] for r in rows], 64, pad_id).to(dev)
null = torch.full_like(plans, pad_id)
null[:, 0] = eos
cap = []
orig = net._sample_level


def spy(logits, *args, **kw):
    cap.append(logits.detach().float().cpu())
    return orig(logits, *args, **kw)


net._sample_level = spy


def l0(pl, cfg):
    cap.clear()
    with torch.no_grad():
        net.generate(None, pl, temperature=0, top_k=0, top_p=1.0, cfg_scale=cfg, cfg_null_eos=eos)
    return cap[0]


La, Lc, Lb = l0(null, 0.0), l0(plans, 0.0), l0(plans, 2.0)
B = len(plans)
V = rep and 4096


def stat(x, y):
    d = (x - y).abs()
    fin = torch.isfinite(x) & torch.isfinite(y)
    return {"max_abs_diff": float(d[fin].max()), "mean_abs_logit": float(x[fin].abs().mean()), "argmax_equal": f"{int((x.argmax(-1) == y.argmax(-1)).sum())}/{x.shape[0] * x.shape[1]}"}


res = {"ckpt": ck, "shapes": [list(La.shape), list(Lc.shape), list(Lb.shape)],
       "cond_branch_vs_plain_real": stat(Lb[:B], Lc), "null_branch_vs_plain_null": stat(Lb[B:], La),
       "control_real_vs_null_plain": stat(Lc, La)}
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
