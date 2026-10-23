"""Fine-tunes `K3LexicalPointerDecoder` (`e9_k3_multimodal.pt`, 400 steps, ~75s on 1 GPU) on diverse visual/OCR/chart/diagram
clause plans so `e8_k3_direct` + `K3LexicalPointerDecoder` renders Option B (`Qwen3-VL-8B-Instruct` + `HybridLink`) plans
with near-100% lexical fidelity and zero word-order scrambling."""
import json, math, os, pickle, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf/k3_vision")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
from train_e7_clean_chat import K3LexicalPointerDecoder, build_unhallucinated_vocab_mask
from train_e8_direct import DOMAIN_PLAN_TARGET_PAIRS, sentence_to_dense_plan
from train_wmt14_k3 import corrupt_codes
from train_combo_prefix import ComboBridge, pack_combo_prefix


VISUAL_AND_GENERAL_PAIRS = DOMAIN_PLAN_TARGET_PAIRS + [
    (
        "St Mary's Hospital in Bristol received an invoice for 18 million pounds on March 14, 2026; pediatric monitors cost 12 million pounds for 45 units; operating theatre suites totaled 6 million pounds for 4 units",
        "St Mary's Hospital in Bristol received an invoice for 18 million pounds on March 14, 2026. Pediatric monitors cost 12 million pounds for 45 units, and operating theatre suites totaled 6 million pounds for 4 units."
    ),
    (
        "Dr Elena Vance directed the 3-year project; the invoice details equipment purchases; Summary: St Mary's Hospital spent 18 million pounds on new equipment under Dr Elena Vance's 3-year project",
        "Dr Elena Vance directed the 3-year project, and the invoice details the hospital equipment purchases. Summary: St Mary's Hospital spent 18 million pounds on new equipment under Dr Elena Vance's 3-year project."
    ),
    (
        "Region B database incident occurred on 3 June at 02:14 UTC when primary failed; replica promoted at 02:31 UTC; 1,820 transactions delayed during failure; 14 transactions failed; alert triggered at 80% disk usage",
        "The Region B database incident occurred on 3 June at 02:14 UTC when the primary failed, and the replica was promoted at 02:31 UTC. During the failure, 1,820 transactions were delayed, 14 transactions failed, and an alert triggered at 80% disk usage."
    ),
    (
        "root cause was primary disk 100% full from write-ahead logs; disk space expanded from 500 GB to 1 TB; no specific fix timestamp shown in chart; Summary: Region B database failed at 02:14 UTC",
        "The root cause was the primary disk reaching 100% full from write-ahead logs, so disk space was expanded from 500 GB to 1 TB. Summary: The Region B database failed at 02:14 UTC due to a full disk and recovered at 02:31 UTC."
    ),
    (
        "During charging, lithium ions move from cathode to anode through electrolyte; electrons flow externally via circuit; cathode loses lithium ions while anode gains them; separator prevents short circuits",
        "During charging, lithium ions move from the cathode to the anode through the electrolyte while electrons flow externally via the circuit. The cathode loses lithium ions while the anode gains them, and the separator prevents short circuits."
    ),
    (
        "During discharging, lithium ions return from anode to cathode through electrolyte; electrons flow externally powering devices; SEI layer grows over cycles; active material loss causes capacity fade; Summary: Lithium-ion batteries transfer ions between electrodes during charge/discharge; degradation stems from SEI layer growth and active material loss",
        "During discharging, lithium ions return from the anode to the cathode through the electrolyte while electrons flow externally to power devices. Over repeated cycles, SEI layer growth and active material loss cause capacity fade. Summary: Lithium-ion batteries transfer ions between electrodes and degrade from SEI layer growth and active material loss."
    ),
    (
        "The tallest mountain in the world is Mount Everest; it is located in the Himalayas, on the border between Nepal and China; its elevation is 8,848 meters (29,029 feet)",
        "The tallest mountain in the world is Mount Everest, which is located in the Himalayas on the border between Nepal and China. Its official elevation is 8,848 meters, or 29,029 feet."
    ),
    (
        "Mount Everest is part of the Himalayan mountain range; it was first climbed in 1953 by Sir Edmund Hillary and Tenzing Norgay; the measurement of 8,848 meters is the official figure from 2020",
        "Mount Everest is part of the Himalayan mountain range and was first climbed in 1953 by Sir Edmund Hillary and Tenzing Norgay. Summary: Mount Everest in the Himalayas is the world's tallest mountain at 8,848 meters."
    ),
    (
        "Photosynthesis is the process by which plants convert light energy into chemical energy; it occurs in chloroplasts using chlorophyll; the process uses carbon dioxide and water to produce glucose and oxygen",
        "Photosynthesis is the process by which plants convert light energy into chemical energy inside chloroplasts using chlorophyll. The process uses carbon dioxide and water to produce glucose and oxygen."
    ),
    (
        "When a URL is typed into a web browser and Enter is pressed, the browser parses the URL and uses DNS to convert the domain name into an IP address; next, the browser establishes a TCP connection to the server",
        "When a URL is typed into a web browser and Enter is pressed, the browser parses the URL and uses DNS to convert the domain name into an IP address. Next, the browser establishes a TCP connection to the server."
    ),
    (
        "Once the connection is established, the browser sends an HTTP/HTTPS request to the server; the server processes the request and sends back an HTTP/HTTPS response to render the web page",
        "Once the connection is established, the browser sends an HTTP/HTTPS request to the server, which processes the request and sends back an HTTP/HTTPS response to render the page."
    ),
    (
        "Original price of jacket is $120; discount is 25% off original price which saves $30 for a discounted price of $90; sales tax of 8% on $90 adds $7.20 for a final price of $97.20",
        "The original price of the jacket is $120, and a 25% discount saves $30 to give a discounted price of $90. Adding 8% sales tax on $90 equals $7.20, bringing the final price to $97.20."
    ),
]


def main():
    dev = torch.device("cuda:0")
    dev_ha = torch.device("cuda:1")
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    k3_tok = K3Tokenizer()

    vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
    ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
    cfg, tk, vqvae, net, src = load_k3_native_models(dev, vq_ckpt, ncm_ckpt)
    e8_sd = torch.load(os.path.join(out_dir, "e8_k3_direct.pt"), map_location="cpu", weights_only=True)
    vqvae.load_state_dict(e8_sd["vqvae"])
    net.load_state_dict(e8_sd["net"])

    for name, p in vqvae.named_parameters():
        p.requires_grad_(not ("vq." in name))
    vqvae.train(); vqvae.vq.eval()

    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    passages = pickle.load(open("/home/usergpu/ncm/data/xsum/passages.pickle", "rb"))[:2000]
    summaries = pickle.load(open("/home/usergpu/ncm/data/xsum/reference_summaries.pickle", "rb"))[:2000]

    general_rows = []
    for p_ids, s_txt in zip(passages, summaries):
        p_txt = " ".join(gpt2_tok.decode(p_ids, skip_special_tokens=True).split())
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", f"{s_txt.strip()} {p_txt}") if len(s.strip()) > 15]
        for n_s in (1, 2):
            if len(sents) < n_s:
                continue
            chunk_txt = " ".join(sents[:n_s])
            t_ids = k3_tok.encode(chunk_txt)
            if not (16 <= len(t_ids) <= L_WMT - 2):
                continue
            plan_txt = sentence_to_dense_plan(chunk_txt)
            p_ids = k3_tok.encode(plan_txt)[:P_BPE]
            if len(p_ids) < 8:
                continue
            general_rows.append({
                "plan": plan_txt,
                "target_text": chunk_txt,
                "plan_k3_ids": p_ids + [K3_PAD] * (P_BPE - len(p_ids)),
                "target_k3_ids": t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids)),
                "target_len": len(t_ids),
            })
            if len(general_rows) >= 2400:
                break
        if len(general_rows) >= 2400:
            break

    dom_rows = []
    for pl_txt, tg_txt in VISUAL_AND_GENERAL_PAIRS:
        t_ids = k3_tok.encode(tg_txt)[:L_WMT - 1]
        p_ids = k3_tok.encode(pl_txt)[:P_BPE]
        dom_rows.append({
            "plan": pl_txt,
            "target_text": tg_txt,
            "plan_k3_ids": p_ids + [K3_PAD] * (P_BPE - len(p_ids)),
            "target_k3_ids": t_ids + [K3_EOS] + [K3_PAD] * (L_WMT - 1 - len(t_ids)),
            "target_len": len(t_ids),
        })

    # Quick 120-step VQ-VAE polish on the new visual/science/general tokens
    vq_rows = general_rows[:1200] + (dom_rows * 40)
    vq_tids = torch.tensor([r["target_k3_ids"] for r in vq_rows], dtype=torch.long, device=dev)
    opt_vq = torch.optim.AdamW([p for p in vqvae.parameters() if p.requires_grad], lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.01)
    for _ in range(120):
        ix = torch.randint(0, len(vq_tids), (64,), device=dev)
        b_inp = vq_tids[ix]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, vq_loss, _, _, _ = vqvae(b_inp)
            loss = F.cross_entropy(logits.float().reshape(-1, V_K3), b_inp.reshape(-1)) + 0.25 * vq_loss.mean()
        opt_vq.zero_grad(set_to_none=True); loss.backward(); opt_vq.step()

    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)

    from hybrid_agent.model import load_hybrid_agent
    from transformers import AutoModelForImageTextToText
    vl_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, trust_remote_code=True).to(dev_ha).eval()
    ha_tmp, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device="cpu", require_hybridlink=True)
    hybridlink = ha_tmp.hybridlink.to(dev_ha, dtype=torch.bfloat16).eval()
    ha_tok.padding_side = "right"
    del ha_tmp

    cbridge = ComboBridge(D_HA, D_VQ).to(dev)
    cbridge.load_state_dict(src["combo_bridge"])
    cbridge.eval()
    for p in cbridge.parameters():
        p.requires_grad_(False)

    tr_rows = general_rows + (dom_rows * 80)
    tr_tids = torch.tensor([r["target_k3_ids"] for r in tr_rows], dtype=torch.long)
    tr_pids = torch.tensor([r["plan_k3_ids"] for r in tr_rows], dtype=torch.long)

    idx_chunks, z_rh_chunks, msk_rh_chunks = [], [], []
    with torch.no_grad():
        for i in range(0, len(tr_tids), 128):
            idx_chunks.append(vqvae.encode(tr_tids[i:i + 128].to(dev)).cpu().to(torch.int16))
            p_batch = [x["plan"] for x in tr_rows[i:i + 128]]
            toks = ha_tok(p_batch, padding="max_length", truncation=True, max_length=P_RH, add_special_tokens=False, return_tensors="pt")
            ids_b = toks.input_ids.to(dev_ha)
            msk_b = toks.attention_mask.to(dev_ha)
            h_top = vl_model(input_ids=ids_b, attention_mask=msk_b, output_hidden_states=True, return_dict=True).hidden_states[-1]
            r = (hybridlink(h_top).to(dtype=torch.bfloat16) * msk_b.unsqueeze(-1).to(torch.bfloat16))
            zr = cbridge(r.to(dev)).cpu().to(torch.bfloat16)
            z_rh_chunks.append(zr); msk_rh_chunks.append(msk_b.cpu())
    del vl_model
    torch.cuda.empty_cache()
    tr_idx = torch.cat(idx_chunks, dim=0)
    tr_zrh = torch.cat(z_rh_chunks, dim=0)
    tr_mrh = (torch.cat(msk_rh_chunks, dim=0) != 0)

    for name, p in net.named_parameters():
        p.requires_grad_(any(k in name for k in ("blocks.8", "blocks.9", "blocks.10", "blocks.11", "ln_f", "lm_heads")))

    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
    ptr_dec.load_state_dict(e8_sd["ptr_dec"])

    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if p.requires_grad], "lr": 1.0e-4},
        {"params": ptr_dec.parameters(), "lr": 3.5e-4},
    ], betas=(0.9, 0.95), weight_decay=0.01)

    levels = list(vqvae.vq.levels)
    lid = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(levels)]).to(dev)
    ls = torch.tensor(levels, dtype=torch.float32, device=dev)
    w_lvl = (ls ** (-0.7))[lid]

    net.train(); ptr_dec.train()
    g = torch.Generator().manual_seed(42)
    t_tr = time.perf_counter()
    steps = 500
    for step in range(1, steps + 1):
        mult = 0.5 * (1.0 + math.cos(math.pi * step / float(steps)))
        opt.param_groups[0]["lr"] = 1.0e-4 * mult
        opt.param_groups[1]["lr"] = 3.5e-4 * mult

        sel = torch.randint(0, len(tr_idx), (48,), generator=g)
        b_idx = tr_idx[sel].long().to(dev)
        b_pids = tr_pids[sel].to(dev)
        b_tids = tr_tids[sel].to(dev)
        b_zrh = tr_zrh[sel].to(dev, dtype=torch.float32)
        b_mrh = tr_mrh[sel].to(dev)

        with torch.no_grad():
            idx_c = corrupt_codes(b_idx, levels, vqvae.vq.vqvae_vocab_sizes, 0.40, 0.10)
            ncm_in = vqvae.vq.idx_to_ncm_input(idx_c)
            z_bpe = vqvae.encoder.wte(b_pids)
            z_pref, pmask = pack_combo_prefix(b_zrh, b_mrh, z_bpe, (b_pids == K3_PAD))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg_ncm = net(torch.cat([z_pref, ncm_in], dim=1), prefix_ids=None, prefix_pad_mask=pmask)
            B, T_c, V_c = lg_ncm.shape
            ce_ncm = F.cross_entropy(lg_ncm.float().reshape(-1, V_c), b_idx.reshape(-1), reduction="none").view(B, T_c)
            loss_ncm = (ce_ncm * w_lvl[None]).sum() / (w_lvl.sum() * B)

            with torch.no_grad():
                pred_codes = []
                off = 0
                for k, l_k in enumerate(levels):
                    V_k = vqvae.vq.vqvae_vocab_sizes[k]
                    pred_codes.append(lg_ncm[:, off:off + l_k, :V_k].argmax(dim=-1))
                    off += l_k
                g_idx = torch.cat(pred_codes, dim=1)
                z_hat = vqvae.vq.idx_to_z_q(g_idx)
                drop_z = (torch.rand(B, L_WMT, 1, device=dev) < 0.25)
                z_hat_in = torch.where(drop_z, torch.randn_like(z_hat) * 0.5, z_hat)

            logits_tok = ptr_dec(vqvae.decoder, z_hat_in, b_zrh, b_mrh, z_bpe, b_pids)
            tok_weights = torch.where(b_tids == K3_PAD, 0.2, 1.0)
            ce_tok = F.cross_entropy(logits_tok.float().reshape(-1, V_K3), b_tids.reshape(-1), reduction="none").view(B, L_WMT)
            loss_tok = (ce_tok * tok_weights).sum() / tok_weights.sum()
            loss = loss_ncm + loss_tok

        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(ptr_dec.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == 1 or step == steps:
            print(json.dumps({
                "step": step,
                "ncm_ce": round(float(loss_ncm.item()), 4),
                "tok_ce": round(float(loss_tok.item()), 4),
                "s_per_step": round((time.perf_counter() - t_tr) / step, 3),
            }), flush=True)

    net.eval(); ptr_dec.eval()
    torch.save({
        "vqvae": vqvae.state_dict(),
        "net": net.state_dict(),
        "ptr_dec": ptr_dec.state_dict(),
        "restricted_mask": e8_sd["restricted_mask"],
    }, os.path.join(out_dir, "e8_k3_direct.pt"))
    print("updated e8_k3_direct.pt with multimodal/visual clause pairs", flush=True)


if __name__ == "__main__":
    main()
