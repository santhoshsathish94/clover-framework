"""Cycle 28 gate (k3 CPU): released TinyStories NCM tokenizer + generator, before any plan-conditioned training.
Same loading path as scripts/eval_shard.py (repo not modified). Checks:
 (1) tokenizer: exact token reconstruction of held-out TinyStories openings (128 BPE tokens), full hierarchy and cumulative per level;
 (2) generator: teacher-forced cross-entropy per level on the same openings with clean inputs, against a control whose inputs come from a DIFFERENT story;
 (3) generator: unconditional samples (level 0 from the stored unigram), to read.
Input: TinyStories-valid.txt (stories separated by <|endoftext|>). Only stories with >= 128 BPE tokens are used (first 128 tokens)."""
import argparse, json, os, sys, time
ROOT = "/root/ncm-cpu/repo"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch, torch.nn.functional as F
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--nsamples", type=int, default=6)
ap.add_argument("--out", default="/root/ncm-cpu/gate-ts.json")
a = ap.parse_args()
ck = "/root/ncm-cpu/artifacts/tinystories"
CFG = OmegaConf.load("config/eval/tinystories.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
print("generator ckpt keys:", [k for k in sd.keys()], flush=True)
net.load_state_dict(sd["model"], strict=True)
net.eval()
vq = tk.vqvae.vq
levels = list(vq.levels)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
print("levels", levels, "sum", sum(levels), "EOS", EOS, "PAD", PAD, "prefix_len", net.prefix_len, "prefix_pad_id", net.prefix_pad_id, flush=True)

txt = open("/root/ncm-cpu/data/TinyStories-valid.txt", encoding="utf-8").read()
stories = [s.strip() for s in txt.split("<|endoftext|>") if s.strip()]
win = []
for s in stories:
    ids = tk.pre_tokenizer.encode(s)
    if len(ids) >= 128:
        win.append(ids[:128])
    if len(win) >= a.n:
        break
X = torch.tensor(win)
print("stories read", len(stories), "windows", X.shape, flush=True)

res = {"n": len(win)}
with torch.no_grad():
    # (1) tokenizer reconstruction
    t0 = time.perf_counter()
    idx = torch.cat([tk.vqvae.encode(X[i:i + 32]) for i in range(0, len(X), 32)])
    assert idx.shape[1] == sum(levels), idx.shape
    def cum_full(ib):
        # full-length cumulative reconstruction after each level (idx_to_zhats aliases one tensor, its z_levels are downsampled)
        B = ib.shape[0]
        z = torch.zeros(B, levels[-1], tk.vqvae.embed_dim)
        outs, off = [], 0
        for k, l in enumerate(levels):
            e = vq.embeddings[k](ib[:, off:off + l])
            q = vq._up_to_full(e, k, levels[-1]) if k != len(levels) - 1 else e
            z = z + vq._apply_quant_resi(q, k)
            outs.append(z.clone())
            off += l
        return outs
    hits = [0] * len(levels)
    for i in range(0, len(X), 32):
        outs = cum_full(idx[i:i + 32])
        for k in range(len(levels)):
            hits[k] += (tk.vqvae.decoder(outs[k]).argmax(-1) == X[i:i + 32]).sum().item()
        # cross-check with the repo's own full decode
        if i == 0:
            ref = tk.vqvae.decode(idx[:32]).argmax(-1)
            res["cum_full_last_equals_repo_decode"] = bool((tk.vqvae.decoder(outs[-1]).argmax(-1) == ref).all())
    acc = [h / X.numel() for h in hits]
    res["recon_acc_cumulative_per_level"] = [round(v, 5) for v in acc]
    print("recon acc per cumulative level", res["recon_acc_cumulative_per_level"], f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # (2) teacher-forced CE per level, clean vs shuffled-story inputs
    def tf_ce(ncm_in, targets_idx, bs=16):
        tot = torch.zeros(len(levels) - 1); cnt = torch.zeros(len(levels) - 1)
        for i in range(0, len(ncm_in), bs):
            xin = ncm_in[i:i + bs]; tg = targets_idx[i:i + bs]
            B = xin.shape[0]
            pref = tk.vqvae.encoder.wte(torch.full((B, 1), EOS))
            logits = net(torch.cat([pref, xin], dim=1), prefix_ids=None, prefix_pad_mask=None)
            off = 0
            for k, l in enumerate(levels[1:]):
                lg = logits[:, off:off + l, :vq.vqvae_vocab_sizes[k + 1]]
                tot[k] += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg[:, off:off + l].reshape(-1), reduction="sum").item()
                cnt[k] += B * l
                off += l
        return (tot / cnt)
    ncm_in = vq.idx_to_ncm_input(idx)
    tgt = idx[:, levels[0]:]
    t0 = time.perf_counter()
    ce_clean = tf_ce(ncm_in, tgt)
    perm = torch.roll(torch.arange(len(idx)), 1)
    ce_wrong = tf_ce(ncm_in[perm], tgt)
    res["ce_clean_per_level"] = [round(v, 3) for v in ce_clean.tolist()]
    res["ce_other_story_inputs_per_level"] = [round(v, 3) for v in ce_wrong.tolist()]
    res["ce_uniform_reference"] = round(float(torch.log(torch.tensor(4096.0))), 3)
    print("CE clean       ", res["ce_clean_per_level"], flush=True)
    print("CE other-story ", res["ce_other_story_inputs_per_level"], f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # (3) unconditional samples
    l0p = sd["inference"]["level0_unigram"].float()
    print("level0_unigram", tuple(l0p.shape), "nonzero", int((l0p > 0).sum()), flush=True)
    torch.manual_seed(42)
    B = a.nsamples
    l0 = torch.multinomial(l0p, B, replacement=True).view(B, 1)
    t0 = time.perf_counter()
    g = net.generate(l0, torch.full((B, 1), EOS), temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
    _, zls = vq.idx_to_zhats(g, return_zlevels=True)
    ids = tk.vqvae.decoder(zls[-1]).argmax(-1).tolist()
    def dec(r):
        r = list(r)
        if EOS in r: r = r[:r.index(EOS)]
        return tk.pre_tokenizer.decode([t for t in r if t not in (EOS, PAD)], skip_special_tokens=True)
    res["samples"] = [dec(r) for r in ids]
    res["generate_s_B%d" % B] = round(time.perf_counter() - t0, 2)
    for s in res["samples"]:
        print("SAMPLE:", s[:300].replace("\n", " "), flush=True)
    res["true_examples"] = [tk.pre_tokenizer.decode(w) for w in win[:2]]
json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
