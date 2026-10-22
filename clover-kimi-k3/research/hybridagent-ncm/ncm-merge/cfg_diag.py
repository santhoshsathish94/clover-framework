"""Why does temperature-0 generation with null plans differ between cfg_scale 0 and 2 (Cycle 37 open item)? Usage: cfg_diag.py CKPT OUT_JSON
Spy on every level's logits and returned codes. Both runs use the same null prefix, so the guided combination (1+t)*cond - t*null equals cond up to rounding.
For each row: first level and position where the codes differ, the top1-top2 logit margin of the unguided run there, and the largest cond-vs-unguided logit difference at that level."""
import json, sys
import ncm_stage2 as S
import torch

ck, outp = sys.argv[1:3]
if len(sys.argv) > 3 and sys.argv[3] == "highest":
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda:0")
CFG, tk, net, src, rep = S.load_models(dev, 64, ckpt=ck)
net.eval()
pad_id, eos = tk.pre_tokenizer.pad_token_id, tk.pre_tokenizer.eos_token_id
B = 48
null = torch.full((B, 64), pad_id, device=dev)
null[:, 0] = eos
cap = []
orig = net._sample_level


def spy(logits, *a, **k):
    out = orig(logits, *a, **k)
    cap.append((logits.detach().float().cpu(), out.detach().cpu()))
    return out


net._sample_level = spy


def run(cfg):
    cap.clear()
    with torch.no_grad():
        net.generate(None, null, temperature=0, top_k=0, top_p=1.0, cfg_scale=cfg, cfg_null_eos=eos)
    return list(cap)


A, G = run(0.0), run(2.0)
res = {"ckpt": ck, "levels_captured": [len(A), len(G)], "rows": B, "matmul_precision": torch.get_float32_matmul_precision(),
       "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "codes_identical_all_rows": all(bool((A[k][1] == G[k][1][:B]).all()) for k in range(len(A))),
       "differing_codes_total": int(sum(int((A[k][1] != G[k][1][:B]).sum()) for k in range(len(A))))}
res["max_abs_logit_per_level"] = [round(float(A[k][0][torch.isfinite(A[k][0])].abs().max()), 1) for k in range(len(A))]
first, margins, diffs, ratios, same_first_level_noise = [], [], [], [], []
for b in range(B):
    hit = None
    for k in range(len(A)):
        pa, pg = A[k][1][b], G[k][1][b]
        d = (pa != pg).nonzero()
        if len(d):
            j = int(d[0])
            la, lc = A[k][0][b, j], G[k][0][b, j]
            fin = torch.isfinite(la)
            top = la[fin].topk(2).values
            hit = (k, j, float(top[0] - top[1]), float((la[fin] - lc[fin]).abs().max()), float(la[fin].abs().max()))
            break
    if hit:
        first.append(hit[0]); margins.append(hit[2]); diffs.append(hit[3]); ratios.append(hit[2] / max(hit[3], 1e-12))
res["rows_with_a_difference"] = len(first)
res["first_differing_level_histogram"] = {int(k): first.count(k) for k in sorted(set(first))}
if first:
    srt = lambda x: sorted(x)
    res["margin_at_first_difference_min_median_max"] = [srt(margins)[0], srt(margins)[len(margins) // 2], srt(margins)[-1]]
    res["cond_vs_unguided_logit_diff_at_that_level_min_median_max"] = [srt(diffs)[0], srt(diffs)[len(diffs) // 2], srt(diffs)[-1]]
    res["margin_over_diff_min"] = min(ratios)
    res["rows_where_margin_smaller_than_noise"] = sum(m < d for m, d in zip(margins, diffs))
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps(res, indent=1))
