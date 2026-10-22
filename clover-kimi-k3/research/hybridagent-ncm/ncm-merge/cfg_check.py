"""Paired checks of the classifier-free-guidance null prefix (Cycle 37). Usage: cfg_check.py CKPT PAIRS OUT_JSON
1. which ids the net uses as prefix pad vs the ids I train with
2. teacher-forced validation CE per level with the true plan, another story's plan, and the null prefix [EOS, PAD, ...] (the trainer's plan-dropout form)
3. generate() with temperature 0 and null plans: cfg_scale 0 vs 2 must give identical codes (cond branch == null branch -> guidance cancels), tests the doubled-batch path
4. same with real plans: guidance must change the codes (control: the check can fail)"""
import json, sys
import ncm_stage2 as S
import torch

ck, pairs, outp = sys.argv[1:4]
dev = torch.device("cuda:0")
CFG, tk, net, src, rep = S.load_models(dev, 64, ckpt=ck)
net.eval()
pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
res = {"ckpt": ck, "l0_head": rep["l0_head"], "net.prefix_pad_id": net.prefix_pad_id, "pad_id": pad_id, "eos": eos}
rows = S.read_pairs(pairs)
idx = S.encode_stories(tk, [r["ids"] for r in rows], dev).long().to(dev)
plans = S.encode_plans(tk, [r["plan"] for r in rows], 64, pad_id).to(dev)
null = torch.full_like(plans, pad_id)
null[:, 0] = eos
ce = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for name, pl in (("true", plans), ("other", torch.roll(plans, 1, 0)), ("null", null)):
        tot, n = 0, 0
        for i in range(0, len(idx), 64):
            _, pls = S.step_loss(net, tk, pl[i:i + 64], idx[i:i + 64], pad_id, None, 0.0)
            tot = tot + pls * len(pl[i:i + 64])
            n += len(pl[i:i + 64])
        ce[name] = (tot / n).tolist()
res["val_ce_level0"] = {k: round(v[0], 4) for k, v in ce.items()}
res["val_ce_mean12"] = {k: round(sum(v) / 12, 4) for k, v in ce.items()}


@torch.no_grad()
def codes(pl, cfg):
    return torch.cat([net.generate(None, pl[i:i + 48], temperature=0, top_k=0, top_p=1.0, cfg_scale=cfg, cfg_null_eos=eos) for i in range(0, len(pl), 48)])


N = 96
c0, c2 = codes(null[:N], 0.0), codes(null[:N], 2.0)
res["null_plan_temp0_cfg0_vs_cfg2_rows_identical"] = f"{int((c0 == c2).all(1).sum())}/{N}"
res["null_plan_temp0_codes_differing"] = int((c0 != c2).sum())
p0, p2 = codes(plans[:N], 0.0), codes(plans[:N], 2.0)
res["real_plan_temp0_cfg0_vs_cfg2_rows_identical"] = f"{int((p0 == p2).all(1).sum())}/{N}"
res["real_plan_level0_code_changed"] = f"{int((p0[:, 0] != p2[:, 0]).sum())}/{N}"
res["real_plan_vs_null_plan_level0_code_differs"] = f"{int((p0[:, 0] != c0[:, 0]).sum())}/{N}"
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
