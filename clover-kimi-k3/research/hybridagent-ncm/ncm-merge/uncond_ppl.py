"""Reference for the plan-conditioned model: the RELEASED unconditional TinyStories generator (stage 1), same Gen. PPL / repeat / entropy code as quality_eval.py.
Samples N stories (level 0 from the stored unigram, as in the paper's evaluation), 3 seeds, optional temperature and adjacency penalty."""
import json, math, os, re, sys, time
ROOT = os.path.expanduser("~/ncm/repo"); ART = os.path.expanduser("~/ncm/artifacts/tinystories")
sys.path.insert(0, ROOT); os.chdir(ROOT)
import torch
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.gen_ppl import generation_perplexity
N = int(sys.argv[1]); temp = float(sys.argv[2]); adj = sys.argv[3] if len(sys.argv) > 3 else ""; out = sys.argv[4] if len(sys.argv) > 4 else "/tmp/uncond.json"
dev = torch.device("cuda:0")
CFG = OmegaConf.load("config/eval/tinystories.yaml")
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ART, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    CFG.tokenizer.pre_tokenizer[key] = os.path.join(ART, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer); tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(os.path.join(ART, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True); net.eval()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
l0p = sd["inference"]["level0_unigram"].float()
ADJ = [float(x) for x in adj.split(",")] if adj else None
def dec(r):
    r = list(r)
    if EOS in r: r = r[: r.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in r if t not in (EOS, PAD)], skip_special_tokens=True)
def pick(logits):
    if not ADJ: return logits.argmax(-1)
    al, aw, ad = ADJ[0], int(ADJ[1]), ADJ[2]; B, T, V = logits.shape; lg = logits.double()
    o = torch.empty(B, T, dtype=torch.long, device=lg.device); hist = []
    for t in range(T):
        s = lg[:, t, :].clone()
        for j, h in enumerate(reversed(hist[-aw:])): s.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = s.argmax(-1); o[:, t] = p; hist.append(p)
    return o
@torch.no_grad()
def gen(seed):
    torch.manual_seed(seed); outs = []
    for i in range(0, N, 64):
        B = min(64, N - i)
        l0 = torch.multinomial(l0p, B, replacement=True).view(B, 1).to(dev)
        g = net.generate(l0, torch.full((B, 1), EOS, device=dev), temperature=temp, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=EOS)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        outs += [dec(r) for r in pick(tk.vqvae.decoder(zl[-1])).tolist()]
    return outs
gtok = AutoTokenizer.from_pretrained("gpt2-large"); gm = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.float16).to(dev).eval()
words = lambda t: re.findall(r"[a-z']+", t.lower())
def ppl(texts):
    r = generation_perplexity(gm, gtok, [gtok.eos_token] * len(texts), texts, device="cuda:0", batch_size=32)
    return math.exp(float(r.nll_sum.sum() / r.token_count.sum()))
def rep(texts):
    n = d = 0
    for t in texts:
        w = words(t); n += sum(1 for i in range(1, len(w)) if w[i] == w[i - 1]); d += max(len(w) - 1, 0)
    return n / max(d, 1)
def ent(texts):
    hs = []
    for t in texts:
        ids = gtok(t)["input_ids"]
        if len(ids) < 2: continue
        c = {}
        for i in ids: c[i] = c.get(i, 0) + 1
        hs.append(-sum(v / len(ids) * math.log(v / len(ids)) for v in c.values()))
    return sum(hs) / len(hs)
runs = []
texts0 = None
for s in range(3):
    t = gen(42 + s); texts0 = texts0 or t
    runs.append({"gen_ppl": ppl(t), "adj_repeat_rate": rep(t), "entropy": ent(t)})
res = {"released_unconditional": True, "N": N, "temp": temp, "adj": adj, "mean": {k: round(sum(r[k] for r in runs) / 3, 4) for k in runs[0]}, "runs": runs, "examples": texts0[:4]}
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print("UNCOND", json.dumps({"temp": temp, "adj": adj, "mean": res["mean"]}))
