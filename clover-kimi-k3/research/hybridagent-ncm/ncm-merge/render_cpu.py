"""CPU copy of scratch-agents/ncm/k3ncm/render.py: renders text with the released XSum NCM (prefix 512, 9 levels, 48 output tokens).
Same config, checkpoint, loading path and shipped decoding (frequency / adjacency penalties) as scripts/eval_shard.py; the repo is not modified.
Per item: one shipped-setting generation (top_k=1) with the text after every level, and --nsamples sampled generations (temperature 1.0, top_k 50, top_p 0.9).
Input json: list of {"id", "variant", "text"}. Differences from render.py: device is the CPU, fp32, fewer samples by default."""
import argparse, json, os, sys, time

ROOT = "/root/ncm-cpu/repo"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--nsamples", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

CFG = OmegaConf.load("config/eval/xsum.yaml")
G = CFG.generation
ck = "/root/ncm-cpu/artifacts/xsum"
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))

freq, mu = float(G.frequency_penalty), float(G.frequency_deficit)
ADJ = [float(v) for v in str(G.adjacency_penalty).split(",")]
path, ref = str(G.frequency_allowance).split(":", 1)
obj = torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
for k in ref.split("."):
    obj = obj[k]
ALLOW = obj.double()


def pick_tokens(logits):
    B, T, V = logits.shape
    lg = logits.double()
    allow = (ALLOW[:V] * T)[None, :]
    al, aw, ad = ADJ[0], int(ADJ[1]), ADJ[2]
    cnt = torch.zeros(B, V, dtype=torch.float64)
    out = torch.empty(B, T, dtype=torch.long)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        excess = (cnt - allow).clamp_min(0.0)
        step -= freq * excess
        if mu > 0:
            step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64))
    return out


def dec(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def gen(x, temp, top_k, top_p):
    t0 = time.perf_counter()
    with torch.no_grad():
        g = net.generate(None, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=0.0, cfg_null_eos=EOS)
        t1 = time.perf_counter()
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        logits = tk.vqvae.decoder(zl[-1])
        ids = pick_tokens(logits)
    t2 = time.perf_counter()
    return g, logits, ids, (t1 - t0) * 1000, (t2 - t1) * 1000


items = json.load(open(a.inputs, encoding="utf-8"))
res = []
print("threads", torch.get_num_threads(), "items", len(items), flush=True)
gen(torch.full((1, P), PAD), 1.0, 1, 0.0)
for it in items:
    ids = src_tok(it["text"])["input_ids"]
    x = torch.tensor([(ids[:P] + [PAD] * P)[:P]])
    g, logits, out_ids, gen_ms, dec_ms = gen(x, 1.0, 1, 0.0)
    ship = dec(out_ids[0].tolist())
    plain = dec(logits.argmax(-1)[0].tolist())
    with torch.no_grad():
        lv = tk.decode_multiscales(g[:1])[0]
    samples, sgen_ms = [], 0.0
    if a.nsamples:
        torch.manual_seed(1234)
        sg, _, sids, sgen_ms, _ = gen(x.repeat(a.nsamples, 1), 1.0, 50, 0.9)
        samples = [dec(r) for r in sids.tolist()]
    res.append({"id": it["id"], "variant": it["variant"], "src_tokens": len(ids), "src_truncated": len(ids) > P,
                "shipped_text": ship, "plain_argmax_text": plain, "levels": [str(s) for s in lv],
                "generate_ms_B1": round(gen_ms, 1), "decode_ms_B1": round(dec_ms, 1),
                "generate_ms_B%d" % a.nsamples: round(sgen_ms, 1), "samples": samples})
    print(it["id"], it["variant"], f"src {len(ids)} tok, {gen_ms:.0f} ms ->", ship, flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", len(res), "items to", a.out)
