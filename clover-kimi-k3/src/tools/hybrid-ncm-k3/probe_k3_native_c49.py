"""Cycle 49: Probe & Test the 3 Possibilities for Native Clover-Kimi-K3 + NCM + HybridAgent Unification:
1. Possibility 1 (K3-Native NCM Tokenizer & VQ-VAE, vocab=163,840):
   - Builds the exact 95,367 shared byte-piece correspondence between Qwen3/WMT14 (151,680) and Kimi K3 (163,840).
   - Solves the ridge-regularized least-squares map between Kimi K3's 7,168-d `embed_tokens.weight` / `lm_head.weight`
     and WMT14's 384-d `vqvae.encoder.wte` and `vqvae.decoder.output_layer`.
   - Grafts all 163,840 Kimi K3 tokens into `vqvae.encoder.wte` and `vqvae.decoder.output_layer` so the VQ-VAE operates
     100% natively on Kimi K3 token IDs (0..163,839), and tests zero-shot vs 500-step adapted reconstruction on
     Kimi K3's actual `p1..p5.ids` and `chat-github.ids`!
2. Possibility 2 (K3-Native HybridAgent / HybridLink in d=7,168 vs Qwen->K3 Bridge):
   - Tests Option 2B: least-squares + MLP bridge from Qwen3 HybridAgent (d=4096) `embed_tokens` / `R(h)` into
     Kimi K3's 7,168-d embedding space (`embed_tokens.weight [163840, 7168]`), measuring nearest-token cosine readout
     directly in Kimi K3's 163,840-token vocabulary!
   - Tests Option 2A: K3-Native `HybridLink(d=7168)` on Kimi K3's 5-layer dense MLA model (`kimi-k3-dspark`) and
     recorded 93-layer Kimi K3 hidden states (`refs/embed.f32`, `refs/final.snaps.f32`, `refs/final.resid.f32`).
"""
import base64, json, math, os, sys, time
from clover_paths import setup_clover_sys_path, find_wmt14_art_dir, find_open_weights_dir

ROOT = setup_clover_sys_path()
ART_WMT = find_wmt14_art_dir()
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoTokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer


def build_shared_vocab_maps():
    k3_dir = find_open_weights_dir()
    k3_bpe = {}
    id_to_bytes_k3 = {}
    with open(os.path.join(k3_dir, "tiktoken.model"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rk = line.split()
            raw = base64.b64decode(b64)
            rk = int(rk)
            k3_bpe[raw] = rk
            id_to_bytes_k3[rk] = raw

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b); cs.append(2**8 + n); n += 1
    u2b = dict(zip([chr(c) for c in cs], bs))

    hf_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    q_ids_shared, k3_ids_shared = [], []
    q_to_k3 = {}
    k3_to_q = {}
    for tok_str, qid in hf_tok.get_vocab().items():
        try:
            raw = bytes([u2b[c] for c in tok_str])
        except KeyError:
            continue
        if raw in k3_bpe and qid < 151680:
            kid = k3_bpe[raw]
            q_ids_shared.append(qid)
            k3_ids_shared.append(kid)
            q_to_k3[qid] = kid
            k3_to_q[kid] = qid

    # For Kimi K3 tokens not in single-token Qwen3, find their sub-piece decomposition in Qwen3!
    k3_unmatched_decomp = {}
    for raw, kid in k3_bpe.items():
        if kid not in k3_to_q:
            try:
                txt = raw.decode("utf-8")
                sub_q = hf_tok.encode(txt, add_special_tokens=False)
                if sub_q:
                    k3_unmatched_decomp[kid] = sub_q
            except Exception:
                pass

    return (
        torch.tensor(q_ids_shared, dtype=torch.long),
        torch.tensor(k3_ids_shared, dtype=torch.long),
        q_to_k3,
        k3_to_q,
        k3_unmatched_decomp,
        id_to_bytes_k3,
    )


def main():
    dev = torch.device("cuda:0")
    t0 = time.perf_counter()
    q_shared, k3_shared, q_to_k3, k3_to_q, k3_decomp, id_to_bytes_k3 = build_shared_vocab_maps()
    print(f"Built vocab maps in {time.perf_counter()-t0:.1f}s: shared={len(q_shared)}, unmatched_with_utf8_decomp={len(k3_decomp)}", flush=True)

    # Load WMT14 VQ-VAE
    cfg = OmegaConf.load("config/eval/wmt14.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ART_WMT, "tokenizer.pt")
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    vqvae = tk.vqvae.to(dev).eval()

    wmt_enc_w = vqvae.encoder.wte.weight.detach().float()         # [151680, 384]
    wmt_dec_w = vqvae.decoder.output_layer.weight.detach().float() # [151680, 384]
    wmt_dec_b = vqvae.decoder.output_layer.bias.detach().float() if vqvae.decoder.output_layer.bias is not None else None

    # Load Kimi K3 embed_tokens.weight and lm_head.weight [163840, 7168]
    st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        k3_emb_cpu = f.get_tensor("language_model.model.embed_tokens.weight")  # [163840, 7168] bf16
        k3_head_cpu = f.get_tensor("language_model.lm_head.weight")            # [163840, 7168] bf16

    # Check how K3's 68,217 unmatched tokens can be initialized in the 163,840 K3-Native VQ-VAE:
    # Method 1 (Exact copy for 95,367 shared + sub-token mean for 68,217 multi-piece K3 tokens + ridge map from K3 7168-d embeddings):
    V_K3 = 163840
    d_vq = 384
    k3_vq_enc = torch.zeros(V_K3, d_vq, device=dev)
    k3_vq_dec = torch.zeros(V_K3, d_vq, device=dev)
    k3_vq_dec_b = torch.zeros(V_K3, device=dev)

    # Fit ridge regression from K3 7168-d embed_tokens -> WMT 384-d enc_w on the 95,367 shared tokens (on GPU in float32!)
    qs_dev = q_shared.to(dev)
    ks_dev = k3_shared.to(dev)
    X_k3_shared = k3_emb_cpu[k3_shared].to(dev, dtype=torch.float32)  # [95367, 7168]
    Y_enc_shared = wmt_enc_w[qs_dev]                                   # [95367, 384]
    Y_dec_shared = wmt_dec_w[qs_dev]                                   # [95367, 384]

    # Ridge solve: (X^T X + lam I) W = X^T Y
    XtX = X_k3_shared.t() @ X_k3_shared
    XtX.diagonal().add_(10.0)
    W_k3_to_enc = torch.linalg.solve(XtX, X_k3_shared.t() @ Y_enc_shared)  # [7168, 384]
    pred_enc = X_k3_shared @ W_k3_to_enc
    cos_enc = F.cosine_similarity(pred_enc, Y_enc_shared, dim=-1).mean().item()
    del XtX, X_k3_shared
    torch.cuda.empty_cache()

    X_head_shared = k3_head_cpu[k3_shared].to(dev, dtype=torch.float32)
    HtH = X_head_shared.t() @ X_head_shared
    HtH.diagonal().add_(10.0)
    W_k3_to_dec = torch.linalg.solve(HtH, X_head_shared.t() @ Y_dec_shared)  # [7168, 384]
    pred_dec = X_head_shared @ W_k3_to_dec
    cos_dec = F.cosine_similarity(pred_dec, Y_dec_shared, dim=-1).mean().item()
    del HtH, X_head_shared
    torch.cuda.empty_cache()
    print(f"Linear 7168->384 projection cosine on 95,367 shared tokens: enc={cos_enc:.4f}, dec={cos_dec:.4f}", flush=True)

    # Initialize all 163,840 rows of k3_vq_enc and k3_vq_dec:
    # 1. Start with the 7168->384 linear projection for all 163,840 Kimi K3 tokens
    for i in range(0, V_K3, 16384):
        sl = slice(i, min(i + 16384, V_K3))
        k3_vq_enc[sl] = k3_emb_cpu[sl].to(dev, dtype=torch.float32) @ W_k3_to_enc
        k3_vq_dec[sl] = k3_head_cpu[sl].to(dev, dtype=torch.float32) @ W_k3_to_dec
    if wmt_dec_b is not None:
        k3_vq_dec_b.fill_(wmt_dec_b.mean().item())

    # 2. For unmatched K3 tokens that decompose into Qwen3 sub-tokens, blend 50% sub-token mean + 50% K3-7168 projection
    decomp_kids = list(k3_decomp.keys())
    for kid in decomp_kids:
        sub_q = k3_decomp[kid]
        k3_vq_enc[kid] = 0.5 * k3_vq_enc[kid] + 0.5 * wmt_enc_w[sub_q].mean(0)
        k3_vq_dec[kid] = 0.5 * k3_vq_dec[kid] + 0.5 * wmt_dec_w[sub_q].mean(0)
        if wmt_dec_b is not None:
            k3_vq_dec_b[kid] = wmt_dec_b[sub_q].mean()

    # 3. For the 95,367 exact byte-piece matches, copy the exact WMT14 encoder/decoder rows!
    k3_vq_enc[ks_dev] = Y_enc_shared
    k3_vq_dec[ks_dev] = Y_dec_shared
    if wmt_dec_b is not None:
        k3_vq_dec_b[ks_dev] = wmt_dec_b[qs_dev]

    # Map special EOS and PAD tokens to Kimi K3 special IDs (163586 = <|im_end|>/eos, 163585 = pad)
    K3_EOS = 163586
    K3_PAD = 163585
    q_eos = tk.pre_tokenizer.eos_token_id
    q_pad = tk.pre_tokenizer.pad_token_id
    k3_vq_enc[K3_EOS] = wmt_enc_w[q_eos]
    k3_vq_dec[K3_EOS] = wmt_dec_w[q_eos]
    k3_vq_enc[K3_PAD] = wmt_enc_w[q_pad]
    k3_vq_dec[K3_PAD] = wmt_dec_w[q_pad]
    if wmt_dec_b is not None:
        k3_vq_dec_b[K3_EOS] = wmt_dec_b[q_eos]
        k3_vq_dec_b[K3_PAD] = wmt_dec_b[q_pad]

    # Surgical graft into vqvae!
    vqvae.in_vocab_size = V_K3
    vqvae.encoder.wte = nn.Embedding(V_K3, d_vq, device=dev)
    vqvae.encoder.wte.weight.data.copy_(k3_vq_enc)
    vqvae.decoder.output_layer = nn.Linear(d_vq, V_K3, bias=(wmt_dec_b is not None), device=dev)
    vqvae.decoder.output_layer.weight.data.copy_(k3_vq_dec)
    if wmt_dec_b is not None:
        vqvae.decoder.output_layer.bias.data.copy_(k3_vq_dec_b)

    # Test zero-shot K3-Native VQ-VAE reconstruction directly on Kimi K3's actual generated .ids files (p1b, p2, p3, p4)!
    res_dir = "/home/usergpu/k3gpu/results"
    L = 64
    for fn in ("p1b.ids", "p2.ids", "p3.ids", "p4.ids"):
        p = os.path.join(res_dir, fn)
        ids = [int(x) for x in open(p).read().replace(",", " ").split() if x.strip()]
        base_ids = [i for i in ids if i < 163584]
        chunks = [base_ids[i:i + (L - 1)] for i in range(0, len(base_ids), L - 1)]
        padded = [c + [K3_EOS] + [K3_PAD] * (L - 1 - len(c)) for c in chunks]
        inp = torch.tensor(padded, dtype=torch.long, device=dev)
        with torch.no_grad():
            idx = vqvae.encode(inp)
            pred = vqvae.decode(idx).argmax(-1)
        hit, tot = 0, 0
        rec_ids = []
        for ci, c in enumerate(chunks):
            nc = len(c)
            pr = pred[ci, :nc].tolist()
            rec_ids.extend(pr)
            hit += sum(int(a == b) for a, b in zip(c, pr))
            tot += nc
        orig_txt = b"".join(id_to_bytes_k3[i] for i in base_ids).decode("utf-8", errors="replace")
        rec_txt = b"".join(id_to_bytes_k3.get(i, b"?") for i in rec_ids).decode("utf-8", errors="replace")
        print(f"[K3-Native VQ-VAE zero-shot on {fn}] exact_k3_id_acc={hit}/{tot} ({100.0*hit/tot:.2f}%)", flush=True)
        print(f"   ORIG: {orig_txt[:140]}", flush=True)
        print(f"   REC : {rec_txt[:140]}", flush=True)


if __name__ == "__main__":
    main()
