"""Diagnostic: does the level-0 head of run1 depend on the plan? (val CE for level 0 was 7.8679 with the true plan AND with another story's plan.)
CPU only. Prints: head parameters, level-0 logits for different plans, and the hidden state the head reads."""
import json, os, sys
import ncm_stage2 as S
import torch
dev = torch.device("cpu")
ck = sys.argv[1]
CFG, tk, net, src, rep = S.load_models(dev, 64, ckpt=ck)
net.eval()
pad_id = tk.pre_tokenizer.pad_token_id
rows = S.read_pairs(os.path.expanduser("~/plan2text/data/pairs_valid.jsonl"))[:8]
plan_ids = S.encode_plans(tk, [r["plan"] for r in rows], 64, pad_id)
idx = S.encode_stories(tk, [r["ids"] for r in rows], dev).long()
vq = tk.vqvae.vq
h = net.lm_head_l0
print("lm_head_l0: proj weight norm", float(h.proj.weight.norm()), "log_inv_T", float(h.log_inv_T), "d_scale", float(h.d_scale))
cap = {}
hook = net.transformer.ln_f.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o.detach()))
with torch.no_grad():
    ncm_in = vq.idx_to_ncm_input(idx)
    pref = tk.vqvae.encoder.wte(plan_ids)
    mask = plan_ids == pad_id
    logits = net(torch.cat([pref, ncm_in], 1), prefix_ids=None, prefix_pad_mask=mask)
    hid = cap["h"]
    print("non-pad plan tokens per row:", (~mask).sum(1).tolist())
    l0 = logits[:, 0, :]
    print("level-0 logits, max abs difference between rows 0 and 1:", float((l0[0] - l0[1]).abs().max()), " between row 0 and mean:", float((l0 - l0.mean(0)).abs().max()))
    sl = net._level0_slice(hid, mask)[:, 0, :]
    print("hidden state read by the head: max abs difference row0 vs row1:", float((sl[0] - sl[1]).abs().max()), " norm", float(sl[0].norm()))
    z = h.proj(sl)
    print("proj output z: max abs diff row0 vs row1:", float((z[0] - z[1]).abs().max()), " norm", float(z[0].norm()))
    lg2 = net(torch.cat([pref.roll(1, 0), ncm_in], 1), prefix_ids=None, prefix_pad_mask=mask.roll(1, 0))
    print("level-0 logits with rolled plans vs matched, max abs diff:", float((lg2[:, 0, :] - logits[:, 0, :]).abs().max()))
    print("level-1 logits with rolled plans vs matched, max abs diff:", float((lg2[:, 1:9, :] - logits[:, 1:9, :]).abs().max()))
    sd0 = net.state_dict()
    print("lm_head_l0 keys:", [k for k in sd0 if "lm_head_l0" in k])
hook.remove()
