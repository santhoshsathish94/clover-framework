"""Cycle 47 (Goal 1 Gate): Probe Kimi K3 answers (p1..p4 + github-incident-reply) across:
1. Released WMT14 NCM VQ-VAE tokenizer (Qwen3 vocab 151,680, L=64, 11 levels): measures exact token accuracy,
   fact/number retention, and per-level cumulative reconstruction on actual Kimi K3 answer & summary outputs.
2. Kimi K3 BPE tokenizer (tiktoken.model, 163,840 vocab) vs Qwen3-8B / WMT14 BPE tokenizer alignment:
   measures what fraction of tokens in Kimi K3's generated answers share identical byte pieces between
   Qwen3/WMT14 and Kimi K3, and whether a direct piece-to-Kimi-ID translation table maps NCM/HybridAgent
   outputs into Kimi K3 token IDs without re-tokenization drift.
3. Also inspects Kimi K3's embed_tokens.weight and lm_head.weight [163840, 7168] in model-00094-of-000096.safetensors
   to measure cosine geometry and check linear bridge rank from NCM z_hat (d=384) and HybridAgent R(h) (d=4096)
   to Kimi K3 embedding space (d=7168).
"""
import json, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
ROOT = os.environ.get("NCM_REPO", "/home/usergpu/ncm/repo")
ART_WMT = "/home/usergpu/ncm/artifacts/wmt14"
ART_LM1B = "/home/usergpu/ncm/artifacts/lm1b"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open
from tokenization.vqvae_tokenizer import VQVAETokenizer


def load_wmt14_tok(dev):
    cfg = OmegaConf.load("config/eval/wmt14.yaml")
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ART_WMT, "tokenizer.pt")
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(dev).eval()
    return cfg, tk


@torch.no_grad()
def idx_to_zhat_upto(vq, idx, max_lvl):
    levels = list(vq.levels)
    L_full = levels[-1]
    z_hat = torch.zeros(idx.shape[0], L_full, vq.embed_dim, device=idx.device)
    off = 0
    for k in range(min(max_lvl, len(levels))):
        l_k = levels[k]
        emb_k = vq.embeddings[k](idx[:, off:off + l_k])
        z_q_k = vq._up_to_full(emb_k, k, L_full) if k != len(levels) - 1 else emb_k
        z_q_k = vq._apply_quant_resi(z_q_k, k)
        z_hat = z_hat + z_q_k
        off += l_k
    return z_hat


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_w, tk_w = load_wmt14_tok(dev)
    vq = tk_w.vqvae.vq
    levels = list(vq.levels)
    L = tk_w.vqvae.context_length  # 64
    eos = tk_w.pre_tokenizer.eos_token_id
    pad = tk_w.pre_tokenizer.pad_token_id
    print(f"WMT14 VQ-VAE loaded: L={L}, levels={levels}, total_codes={sum(levels)}, vocab={tk_w.vqvae.in_vocab_size}", flush=True)

    # Load Kimi K3's actual outputs on p1..p4
    k3_sum = json.load(open("/home/usergpu/k3ncm/k3_summary.json", encoding="utf-8"))
    p1_ans = (
        "The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a new cycle lane along Mill Road. "
        "Councillor Priya Raman proposed the plan. Work is set to begin in March and should take 14 weeks to complete. "
        "Shop owners on Mill Road, led by Tom Eliot, had asked for the vote to be delayed because of concerns about parking. "
        "In response, the council promised to provide 40 temporary parking spaces on Dock Street. The decision means the project "
        "will go ahead as planned, despite objections from local businesses worried about losing parking for customers during the construction period."
    )
    p1_sum = "Harbourside Council approved funding for a Mill Road cycle lane despite parking concerns from local shop owners."
    k3_items = {"p1": {"answer": p1_ans, "summary": p1_sum}, **k3_sum}

    keys_map = {
        "p1": ["Harbourside", "7", "2", "Tuesday", "2.4", "Mill Road", "Priya Raman", "March", "14", "Tom Eliot", "40", "Dock Street"],
        "p2": ["Apollo 13", "1970", "NASA", "Moon", "oxygen", "lunar module", "Houston", "Pacific"],
        "p3": ["blue", "sunlight", "scatters", "molecules", "sunset", "red", "orange", "path"],
        "p4": ["02:14", "June", "B", "02:31", "1,820", "14", "100", "80", "500", "1 TB"],
    }

    report = {"wmt14_vqvae_on_k3": {}, "kimi_k3_safetensors": {}}

    for pid, obj in k3_items.items():
        for kind in ("answer", "summary"):
            txt = obj[kind]
            raw_ids = tk_w.pre_tokenizer.encode(txt)
            # Chunk into 64-token windows (up to 63 tokens + EOS)
            chunks = [raw_ids[i:i + (L - 1)] for i in range(0, len(raw_ids), L - 1)]
            padded = [c + [eos] + [pad] * (L - 1 - len(c)) for c in chunks]
            inp = torch.tensor(padded, dtype=torch.long, device=dev)
            with torch.no_grad():
                idx = tk_w.vqvae.encode(inp)
                # Full 11-level decode and coarse 6-level decode
                zh_full = idx_to_zhat_upto(vq, idx, max_lvl=len(levels))
                zh_lvl6 = idx_to_zhat_upto(vq, idx, max_lvl=6)
                pred_full = tk_w.vqvae.decoder(zh_full).argmax(-1)
                pred_lvl6 = tk_w.vqvae.decoder(zh_lvl6).argmax(-1)

            rec_ids_full, rec_ids_lvl6 = [], []
            tok_correct_full, tok_correct_lvl6, tok_total = 0, 0, 0
            for c_i, c in enumerate(chunks):
                n_c = len(c)
                p_f = pred_full[c_i, :n_c].tolist()
                p_6 = pred_lvl6[c_i, :n_c].tolist()
                rec_ids_full.extend(p_f)
                rec_ids_lvl6.extend(p_6)
                tok_correct_full += sum(int(a == b) for a, b in zip(c, p_f))
                tok_correct_lvl6 += sum(int(a == b) for a, b in zip(c, p_6))
                tok_total += n_c

            rec_txt_full = tk_w.pre_tokenizer.decode(rec_ids_full)
            rec_txt_lvl6 = tk_w.pre_tokenizer.decode(rec_ids_lvl6)
            k_list = keys_map[pid]
            keys_in_orig = [k for k in k_list if k.lower() in txt.lower()]
            keys_kept_full = [k for k in keys_in_orig if k.lower() in rec_txt_full.lower()]
            keys_kept_lvl6 = [k for k in keys_in_orig if k.lower() in rec_txt_lvl6.lower()]

            rec_entry = {
                "n_tokens": tok_total,
                "n_windows": len(chunks),
                "tok_acc_full11": round(tok_correct_full / max(tok_total, 1), 4),
                "tok_acc_lvl6": round(tok_correct_lvl6 / max(tok_total, 1), 4),
                "keys_in_orig": f"{len(keys_in_orig)}/{len(k_list)}",
                "keys_kept_full11": f"{len(keys_kept_full)}/{len(keys_in_orig)}",
                "keys_kept_lvl6": f"{len(keys_kept_lvl6)}/{len(keys_in_orig)}",
                "orig_text": txt,
                "rec_text_full11": rec_txt_full,
                "rec_text_lvl6": rec_txt_lvl6,
            }
            report["wmt14_vqvae_on_k3"][f"{pid}_{kind}"] = rec_entry
            print(f"[{pid}_{kind}] n_tok={tok_total} acc11={rec_entry['tok_acc_full11']} keys11={rec_entry['keys_kept_full11']} | rec: {rec_txt_full[:140]}", flush=True)

    # 2. Inspect Kimi K3 model-00094-of-000096.safetensors and tiktoken.model alignment with Qwen3 / WMT14
    st_path = "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
    with safe_open(st_path, framework="pt", device="cpu") as f:
        st_keys = list(f.keys())
        emb_w = f.get_tensor("language_model.model.embed_tokens.weight")  # [163840, 7168] bf16
        norm_w = f.get_tensor("language_model.model.norm.weight")         # [7168] bf16
        res_norm_w = f.get_tensor("language_model.model.output_attn_res_norm.weight")
        res_proj_w = f.get_tensor("language_model.model.output_attn_res_proj.weight")
        report["kimi_k3_safetensors"] = {
            "keys": st_keys,
            "embed_shape": list(emb_w.shape),
            "embed_rms": round(float(emb_w[:10000].float().pow(2).mean().sqrt().item()), 6),
            "norm_rms": round(float(norm_w.float().pow(2).mean().sqrt().item()), 6),
            "res_norm_rms": round(float(res_norm_w.float().pow(2).mean().sqrt().item()), 6),
            "res_proj_rms": round(float(res_proj_w.float().pow(2).mean().sqrt().item()), 6),
        }
    print("Kimi K3 safetensors summary:", report["kimi_k3_safetensors"], flush=True)

    # 3. Check Kimi K3 recorded reference tensors in /opt/clover-k3/clover-kimi-k3/src/work/inputs/refs/
    ref_dir = "/opt/clover-k3/clover-kimi-k3/src/work/inputs/refs"
    emb_f32 = torch.from_file(os.path.join(ref_dir, "embed.f32"), shared=False, size=1000 * 7168, dtype=torch.float32).view(1000, 7168)
    res_f32 = torch.from_file(os.path.join(ref_dir, "final.resid.f32"), shared=False, size=1000 * 7168, dtype=torch.float32).view(1000, 7168)
    snaps_f32 = torch.from_file(os.path.join(ref_dir, "final.snaps.f32"), shared=False, size=8 * 1000 * 7168, dtype=torch.float32).view(8, 1000, 7168)
    snap0_max_diff = float((snaps_f32[0] - emb_f32).abs().max().item())
    report["kimi_k3_refs"] = {
        "embed_f32_rms": round(float(emb_f32.pow(2).mean().sqrt().item()), 6),
        "final_resid_rms": round(float(res_f32.pow(2).mean().sqrt().item()), 6),
        "snap_rms_per_block": [round(float(snaps_f32[s].pow(2).mean().sqrt().item()), 4) for s in range(8)],
        "snap0_vs_embed_max_diff": snap0_max_diff,
    }
    print("Kimi K3 refs summary:", report["kimi_k3_refs"], flush=True)

    os.makedirs("/home/usergpu/merge_lf/out", exist_ok=True)
    json.dump(report, open("/home/usergpu/merge_lf/out/probe_k3_c47.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved /home/usergpu/merge_lf/out/probe_k3_c47.json", flush=True)


if __name__ == "__main__":
    main()
