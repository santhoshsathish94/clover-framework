"""Cycle 58: Training `K3ExpertBridge` + `NextConceptTransformer (NCM)` + `K3LexicalPointerDecoder` (`e10_k3_agent_experts.pt`)
conditioned on real Kimi K3 GPU MoE Expert activations (`[N, 48, 7168]` from `Layer 0 -> Layer 1 -> Layer 12 -> Layer 48 -> Layer 92`,
3,584 Kimi K3 experts in VRAM) + `HybridAgent-8B-Agentic` continuous `R(h)` (`[N, 48, 4096]`).
"""
import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, "/home/usergpu/hybridagent")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import (
    load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, K3_EOS, K3_PAD
)
from train_wmt14_k3 import corrupt_codes
from train_combo_prefix import ComboBridge, pack_combo_prefix
from train_e7_clean_chat import K3LexicalPointerDecoder
from k3_agent_engine import K3GPUExpertsRunner, K3ExpertBridge


def main():
    t0 = time.perf_counter()
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    # 2. Load the 4 shards of clubbed HybridAgent pairs + R(h) vectors from `/tmp/e8_8gpu_shard_0..3.pt`
    shard_paths = [f"/tmp/e8_8gpu_shard_{r_idx}.pt" for r_idx in range(4)]
    pairs_all = []
    rh_list = []
    msk_list = []
    for sp in shard_paths:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        n_gen = len(sd["gen_windows"])
        # Take 350 general pairs + all clubbed generated windows per shard (`~2,150` balanced windows total)
        sel_pairs = sd["pairs"][:350] + (sd["gen_windows"] * 3)
        sel_rh = torch.cat([sd["rh"][:350], sd["rh"][-n_gen:].repeat(3, 1, 1)], dim=0)
        sel_msk = torch.cat([sd["msk"][:350], sd["msk"][-n_gen:].repeat(3, 1)], dim=0)
        pairs_all.extend(sel_pairs)
        rh_list.append(sel_rh)
        msk_list.append(sel_msk)

    rh_all = torch.cat(rh_list, dim=0)
    msk_all = torch.cat(msk_list, dim=0)

    pids_list, tids_list = [], []
    for pl_txt, tg_txt in pairs_all:
        t_ids = k3_tok.encode(tg_txt)[:L_WMT - 1]
        p_ids = k3_tok.encode(pl_txt)[:P_BPE]
        if not t_ids:
            t_ids = [K3_EOS]
        if not p_ids:
            p_ids = t_ids[:P_BPE]
        pids_list.append(p_ids + [K3_PAD] * (P_BPE - len(p_ids)))
        tids_list.append(t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids)))

    all_pids = torch.tensor(pids_list, dtype=torch.long)
    all_tids = torch.tensor(tids_list, dtype=torch.long)
    N_tot = len(all_pids)
    k3_cache_path = "/tmp/e10_k3_moe_cache.pt"
    if os.path.exists(k3_cache_path):
        print(f"[Step 1] Loading cached Kimi K3 GPU MoE Expert states from {k3_cache_path}...", flush=True)
        all_h_k3 = torch.load(k3_cache_path, map_location="cpu", weights_only=True)
    else:
        k3_runner = K3GPUExpertsRunner.get_instance(max_n=512)
        print(f"[Step 1] Extracting real Kimi K3 GPU MoE Expert states (L0->L1->L12->L48->L92, 3,584 experts across GPUs 0,1,5,4) for {N_tot} windows...", flush=True)
        uniq_pids, inv_idx = torch.unique(all_pids, dim=0, return_inverse=True)
        print(f"  Unique plan windows = {len(uniq_pids)} (from {N_tot} total rows)", flush=True)

        k3_moe_chunks = []
        total_expert_ms = 0.0
        batch_w = 10  # 10 * 48 = 480 tokens per C/CUDA call (<= 512 max_n)
        for i in range(0, len(uniq_pids), batch_w):
            h_chunk, meta = k3_runner.forward_k3_ids(uniq_pids[i:i + batch_w])
            k3_moe_chunks.append(h_chunk.to(torch.bfloat16))
            total_expert_ms += meta["k3_moe_ms"]
            if (i // batch_w) % 25 == 0:
                print(f"  K3 GPU Experts: {min(i + batch_w, len(uniq_pids))}/{len(uniq_pids)} unique windows ({total_expert_ms/1000.0:.1f}s) | last unique experts={meta['unique_experts_per_layer']}", flush=True)

        uniq_h_k3 = torch.cat(k3_moe_chunks, dim=0)  # [N_uniq, 48, 7168] bfloat16
        all_h_k3 = uniq_h_k3[inv_idx]               # [N_tot, 48, 7168] bfloat16
        torch.save(all_h_k3, k3_cache_path)
        print(f"[Step 1 Complete] Extracted & cached {len(uniq_pids)} unique Kimi K3 MoE expert windows in {total_expert_ms/1000.0:.2f}s.", flush=True)

    # 3. Train `K3ExpertBridge` + `NextConceptTransformer (NCM)` + `K3LexicalPointerDecoder` across GPUs [2, 3, 6, 7]
    #    (while GPUs [0, 1, 5, 4] hold the 62.9 GB of Kimi K3 MoE experts -> all 8 GPUs active!)
    train_devs = [2, 3, 6, 7]
    dev = torch.device(f"cuda:{train_devs[0]}")
    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)

    e8_sd = torch.load(os.path.join(out_dir, "e8_k3_direct.pt"), map_location="cpu", weights_only=True)
    vqvae.load_state_dict(e8_sd["vqvae"])
    net.load_state_dict(e8_sd["net"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()
    for p in cbridge.parameters():
        p.requires_grad_(False)

    k3_bridge = K3ExpertBridge(d_k3=7168, d_vq=D_VQ).to(dev)
    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
    ptr_dec.load_state_dict(e8_sd["ptr_dec"])

    idx_chunks, zrh_chunks = [], []
    with torch.no_grad():
        for i in range(0, N_tot, 256):
            idx_chunks.append(vqvae.encode(all_tids[i:i + 256].to(dev)).cpu().to(torch.int16))
            zrh_chunks.append(cbridge(rh_all[i:i + 256].to(dev, dtype=torch.float32)).cpu().to(torch.bfloat16))
    idx_all = torch.cat(idx_chunks, dim=0)
    z_rh_all = torch.cat(zrh_chunks, dim=0)

    for name, p in net.named_parameters():
        p.requires_grad_(any(k in name for k in ("blocks.8", "blocks.9", "blocks.10", "blocks.11", "ln_f", "lm_heads")))

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)
    w_lvl = (ls ** (-0.7))[lid]

    class AgentK3StepModule(nn.Module):
        def __init__(self, k3b, net_m, vq_m, ptr_m, w_lvl_t):
            super().__init__()
            self.k3_bridge = k3b
            self.net = net_m
            self.vqvae = vq_m
            self.ptr_dec = ptr_m
            self.levels = list(vq_m.vq.levels)
            self.vsizes = list(vq_m.vq.vqvae_vocab_sizes)
            self.register_buffer("w_lvl_buf", w_lvl_t)

        def forward(self, b_hk3, wte_bpe, b_zrh, b_mrh, b_pids, ncm_in, b_idx, b_tids):
            pad_bpe = (b_pids == K3_PAD)
            z_k3_exp = self.k3_bridge(b_hk3).masked_fill(pad_bpe.unsqueeze(-1), 0.0)
            z_bpe = wte_bpe + z_k3_exp
            z_pref, pmask = pack_combo_prefix(b_zrh, b_mrh, z_bpe, pad_bpe)

            lg_ncm = self.net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
            B_cur, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_idx.reshape(-1), reduction="none").view(B_cur, T_c)
            loss_ncm = (ce_ncm * self.w_lvl_buf[None]).sum() / (self.w_lvl_buf.sum() * B_cur)

            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(self.levels):
                    V_k = self.vsizes[k]
                    pred_codes.append(lg_ncm[:, off:off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                pred_idx = torch.cat(pred_codes, dim=1)
                z_hats, _ = self.vqvae.vq.idx_to_zhats(pred_idx, return_zlevels=True)
                z_hat_top = z_hats[-1]

            lg_tok = self.ptr_dec(self.vqvae.decoder, z_hat_top, b_zrh, b_mrh, z_bpe, b_pids)
            loss_tok = F.cross_entropy(lg_tok.float().reshape(-1, V_K3), b_tids.reshape(-1), ignore_index=K3_PAD)
            return loss_ncm.unsqueeze(0), loss_tok.unsqueeze(0)

    step_mod = AgentK3StepModule(k3_bridge, net, vqvae, ptr_dec, w_lvl).to(dev)
    dp_mod = nn.DataParallel(step_mod, device_ids=train_devs)

    opt = torch.optim.AdamW([
        {"params": k3_bridge.parameters(), "lr": 8.0e-4},
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 4.0e-5},
        {"params": ptr_dec.parameters(), "lr": 1.2e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    net.train(); ptr_dec.train(); k3_bridge.train()
    steps = 160
    bs = len(train_devs) * 24  # 96 clubbed windows per step across GPUs [2, 3, 6, 7]
    g = torch.Generator().manual_seed(42)
    t_tr = time.perf_counter()
    print(f"[Step 2] Training K3ExpertBridge + NCM + K3LexicalPointerDecoder across GPUs {train_devs} ({steps} steps, bs={bs})...", flush=True)

    for step in range(1, steps + 1):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / float(steps)))
        opt.param_groups[0]["lr"] = 8.0e-4 * mult
        opt.param_groups[1]["lr"] = 4.0e-5 * mult
        opt.param_groups[2]["lr"] = 1.2e-4 * mult

        sel = torch.randint(0, N_tot, (bs,), generator=g)
        b_idx = idx_all[sel].long().to(dev, non_blocking=True)
        b_pids = all_pids[sel].to(dev, non_blocking=True)
        b_tids = all_tids[sel].to(dev, non_blocking=True)
        b_zrh = z_rh_all[sel].to(dev, dtype=torch.float32, non_blocking=True)
        b_mrh = msk_all[sel].to(dev, non_blocking=True)
        b_hk3 = all_h_k3[sel].to(dev, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.35, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            wte_bpe = vqvae.encoder.wte(b_pids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss_ncm_v, loss_tok_v = dp_mod(b_hk3, wte_bpe, b_zrh, b_mrh, b_pids, ncm_in, b_idx, b_tids)
            loss_ncm = loss_ncm_v.mean()
            loss_tok = loss_tok_v.mean()
            loss = loss_ncm + 1.5 * loss_tok

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(k3_bridge.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(ptr_dec.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % 40 == 0 or step == 1:
            print(
                f"  [Agent+K3Experts 8-GPU] step {step}/{steps} | loss_ncm={loss_ncm.item():.4f} | loss_tok={loss_tok.item():.4f} "
                f"| tanh(alpha_k3)={math.tanh(k3_bridge.alpha.item()):.4f} | elapsed={time.perf_counter() - t_tr:.1f}s",
                flush=True,
            )

    net.eval(); ptr_dec.eval(); k3_bridge.eval()
    save_path = os.path.join(out_dir, "e10_k3_agent_experts.pt")
    torch.save({
        "vqvae": vqvae.state_dict(),
        "net": net.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
        "k3_bridge": k3_bridge.state_dict(),
        "restricted_mask": e8_sd["restricted_mask"],
    }, save_path)
    print(f"[Saved] {save_path} in {time.perf_counter() - t0:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
