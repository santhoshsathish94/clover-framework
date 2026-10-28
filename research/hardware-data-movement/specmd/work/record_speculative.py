import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

MODEL_DIR = "/root/dm-study/specmd/w2/model"
TRACE_DIR = "/root/dm-study/specmd/w2/traces"


def eval_strategies(pred_probs, true_topk, global_tau80):
    """
    pred_probs: (N_tok, 15, 64) predicted probabilities for target layers 1..15
    true_topk:  (N_tok, 15, 8)  true top-8 indices for target layers 1..15
    """
    N, L15, M = pred_probs.shape
    strategies = {
        "TopK_1.0x_8": {"prec": [], "rec": [], "cnt": []},
        "Overfetch_1.5x_12": {"prec": [], "rec": [], "cnt": []},
        "Overfetch_2.0x_16": {"prec": [], "rec": [], "cnt": []},
        "Score_Global80": {"prec": [], "rec": [], "cnt": []},
        "Score_PerTok80": {"prec": [], "rec": [], "cnt": []},
    }
    per_layer_top8 = np.zeros(L15, dtype=np.float64)
    per_layer_ov12_rec = np.zeros(L15, dtype=np.float64)
    per_layer_sc80_rec = np.zeros(L15, dtype=np.float64)
    per_layer_sc80_prec = np.zeros(L15, dtype=np.float64)
    per_layer_sc80_cnt = np.zeros(L15, dtype=np.float64)

    # Sort indices descending by predicted probability: (N, 15, 64)
    sorted_idx = np.argsort(-pred_probs, axis=-1)
    pertok_tau80 = np.percentile(pred_probs, 80.0, axis=-1, keepdims=True)

    for l in range(L15):
        t_sets = [set(true_topk[i, l].tolist()) for i in range(N)]
        # Top-8, Top-12, Top-16
        for name, k_cnt in [("TopK_1.0x_8", 8), ("Overfetch_1.5x_12", 12), ("Overfetch_2.0x_16", 16)]:
            p_idx = sorted_idx[:, l, :k_cnt]
            hits = [len(t_sets[i].intersection(p_idx[i].tolist())) for i in range(N)]
            prec = np.mean([h / float(k_cnt) for h in hits])
            rec = np.mean([h / 8.0 for h in hits])
            strategies[name]["prec"].append(float(prec))
            strategies[name]["rec"].append(float(rec))
            strategies[name]["cnt"].append(float(k_cnt))
            if k_cnt == 8:
                per_layer_top8[l] = float(rec)
            elif k_cnt == 12:
                per_layer_ov12_rec[l] = float(rec)

        # Score_Global80 (adaptive count per token/layer)
        mask_g = pred_probs[:, l, :] >= global_tau80
        g_precs, g_recs, g_cnts = [], [], []
        for i in range(N):
            sel = np.nonzero(mask_g[i])[0].tolist()
            if not sel:
                sel = [int(sorted_idx[i, l, 0])]
            h = len(t_sets[i].intersection(sel))
            g_precs.append(h / float(len(sel)))
            g_recs.append(h / 8.0)
            g_cnts.append(len(sel))
        per_layer_sc80_prec[l] = float(np.mean(g_precs))
        per_layer_sc80_rec[l] = float(np.mean(g_recs))
        per_layer_sc80_cnt[l] = float(np.mean(g_cnts))
        strategies["Score_Global80"]["prec"].append(per_layer_sc80_prec[l])
        strategies["Score_Global80"]["rec"].append(per_layer_sc80_rec[l])
        strategies["Score_Global80"]["cnt"].append(per_layer_sc80_cnt[l])

        # Score_PerTok80
        mask_p = pred_probs[:, l, :] >= pertok_tau80[:, l, :]
        p_precs, p_recs, p_cnts = [], [], []
        for i in range(N):
            sel = np.nonzero(mask_p[i])[0].tolist()
            h = len(t_sets[i].intersection(sel))
            p_precs.append(h / float(len(sel)))
            p_recs.append(h / 8.0)
            p_cnts.append(len(sel))
        strategies["Score_PerTok80"]["prec"].append(float(np.mean(p_precs)))
        strategies["Score_PerTok80"]["rec"].append(float(np.mean(p_recs)))
        strategies["Score_PerTok80"]["cnt"].append(float(np.mean(p_cnts)))

    out = {}
    for name, d in strategies.items():
        out[name] = {
            "mean_precision": round(float(np.mean(d["prec"])), 5),
            "mean_recall": round(float(np.mean(d["rec"])), 5),
            "mean_count": round(float(np.mean(d["cnt"])), 3),
        }
    return out, per_layer_top8, per_layer_ov12_rec, per_layer_sc80_prec, per_layer_sc80_rec, per_layer_sc80_cnt


def main():
    torch.set_num_threads(6)
    p_data = np.load(os.path.join(TRACE_DIR, "prompt_traces.npz"))
    d_data = np.load(os.path.join(TRACE_DIR, "decode_traces.npz"))
    input_ids_all = torch.tensor(p_data["input_ids"], dtype=torch.long)  # (8, 256)

    print("Loading OLMoE-1B-7B-0924 in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    layers_mod = model.model.layers

    cur_r = {}
    cur_h = {}
    recording = [False]
    hooks = []

    def make_norm_hook(l_idx):
        def hook(module, args, output):
            if recording[0]:
                cur_r[l_idx] = args[0].detach()
                cur_h[l_idx] = output.detach()
        return hook

    for l in range(16):
        hooks.append(layers_mod[l].post_attention_layernorm.register_forward_hook(make_norm_hook(l)))

    # 1. Prompt-style across all 8 prompts (2,048 tokens)
    pred_A_prompt = np.zeros((8, 256, 15, 64), dtype=np.float32)
    pred_B_prompt = np.zeros((8, 256, 15, 64), dtype=np.float32)
    cos_h_adj = np.zeros((8, 256, 15), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for p in range(8):
            recording[0] = True
            _ = model(input_ids_all[p:p+1], use_cache=False)
            recording[0] = False
            for l in range(15):
                h_l = cur_h[l].view(-1, 2048)
                h_next_true = cur_h[l + 1].view(-1, 2048)
                r_l = cur_r[l].view(-1, 2048)
                w_next = layers_mod[l + 1].mlp.gate.weight

                # Variant A: gate_{l+1}(h_l)
                logits_A = F.linear(h_l, w_next)
                probs_A = F.softmax(logits_A, dim=-1, dtype=torch.float32)
                pred_A_prompt[p, :, l, :] = probs_A.numpy()

                # Variant B: gate_{l+1}(RMSNorm_{l+1}(r_l))
                h_l_norm_next = layers_mod[l + 1].post_attention_layernorm(r_l)
                logits_B = F.linear(h_l_norm_next, w_next)
                probs_B = F.softmax(logits_B, dim=-1, dtype=torch.float32)
                pred_B_prompt[p, :, l, :] = probs_B.numpy()

                cos = F.cosine_similarity(h_l.float(), h_next_true.float(), dim=-1)
                cos_h_adj[p, :, l] = cos.numpy()
    print(f"Prompt-style speculative pass (2,048 tokens) done in {time.time() - t0:.2f}s")

    # 2. Decode-style across 2 prompts (512 tokens)
    pred_A_decode = np.zeros((2, 256, 15, 64), dtype=np.float32)
    t1 = time.time()
    with torch.no_grad():
        for p in range(2):
            past_kv = None
            for t in range(256):
                recording[0] = True
                out = model(input_ids_all[p:p+1, t:t+1], past_key_values=past_kv, use_cache=True)
                recording[0] = False
                past_kv = out.past_key_values
                for l in range(15):
                    h_l = cur_h[l].view(-1, 2048)
                    w_next = layers_mod[l + 1].mlp.gate.weight
                    probs_A = F.softmax(F.linear(h_l, w_next), dim=-1, dtype=torch.float32)
                    pred_A_decode[p, t, l, :] = probs_A[0].numpy()
    print(f"Decode-style speculative pass (512 tokens) done in {time.time() - t1:.2f}s")

    for h in hooks:
        h.remove()

    global_tau80 = float(np.percentile( p_data["router_probs"], 80.0 ))
    print(f"Global 80th-percentile router softmax score tau_80 = {global_tau80:.6f} (uniform=0.015625)")

    true_prompt = p_data["topk_indices"][:, :, 1:, :].reshape(-1, 15, 8)
    out_A_p, l_top8_A, l_ov12_A, l_sc_prec_A, l_sc_rec_A, l_sc_cnt_A = eval_strategies(
        pred_A_prompt.reshape(-1, 15, 64), true_prompt, global_tau80
    )
    out_B_p, l_top8_B, _, _, _, _ = eval_strategies(
        pred_B_prompt.reshape(-1, 15, 64), true_prompt, global_tau80
    )
    true_decode_512 = d_data["topk_indices"][:2, :, 1:, :].reshape(-1, 15, 8)
    out_A_d, _, _, _, _, _ = eval_strategies(
        pred_A_decode.reshape(-1, 15, 64), true_decode_512, global_tau80
    )

    print("\n=== CROSS-LAYER PREDICTION ACCURACY (OVERALL ACROSS 15 LAYER TRANSITIONS) ===")
    print("  Variant A (Direct Gate Hook: gate_{l+1}(h_l)) - Prompt-style (2,048 tokens):")
    for k, v in out_A_p.items():
        print(f"    {k:20s}: Precision={v['mean_precision']*100:6.2f}%  Recall={v['mean_recall']*100:6.2f}%  AvgPrefetchCnt={v['mean_count']:5.2f}")
    print("  Variant B (Residual + Next-Layer RMSNorm: gate_{l+1}(RMSNorm_{l+1}(r_l))) - Prompt-style:")
    for k, v in out_B_p.items():
        print(f"    {k:20s}: Precision={v['mean_precision']*100:6.2f}%  Recall={v['mean_recall']*100:6.2f}%  AvgPrefetchCnt={v['mean_count']:5.2f}")
    print("  Variant A - Decode-style (512 tokens):")
    for k, v in out_A_d.items():
        print(f"    {k:20s}: Precision={v['mean_precision']*100:6.2f}%  Recall={v['mean_recall']*100:6.2f}%  AvgPrefetchCnt={v['mean_count']:5.2f}")

    print("\n=== PER-LAYER PREDICTION BREAKDOWN (l -> l+1, Prompt-style, 2,048 tokens) ===")
    print("  Trans   | CosSim(h_l,h_{l+1}) | VarA Top8 Rec | VarB Top8 Rec | Ov1.5x(12) Rec | Score80 Prec / Rec / AvgCnt")
    for l in range(15):
        cs = float(np.mean(cos_h_adj[:, :, l]))
        print(
            f"  L{l:02d}->L{l+1:02d} |       {cs:6.4f}        |    {l_top8_A[l]*100:6.2f}%    |    {l_top8_B[l]*100:6.2f}%    |     {l_ov12_A[l]*100:6.2f}%    |  {l_sc_prec_A[l]*100:5.2f}% / {l_sc_rec_A[l]*100:5.2f}% / {l_sc_cnt_A[l]:5.2f}"
        )

    np.savez_compressed(
        os.path.join(TRACE_DIR, "spec_predictions.npz"),
        pred_A_prompt=pred_A_prompt.astype(np.float16),
        pred_B_prompt=pred_B_prompt.astype(np.float16),
        global_tau80=np.array([global_tau80], dtype=np.float32),
    )
    with open(os.path.join(TRACE_DIR, "e7_prediction_summary.json"), "w") as f:
        json.dump({
            "global_tau80": global_tau80,
            "variant_A_prompt": out_A_p,
            "variant_B_prompt": out_B_p,
            "variant_A_decode_512": out_A_d,
            "per_layer_cos_sim": [round(float(np.mean(cos_h_adj[:, :, l])), 5) for l in range(15)],
            "per_layer_top8_rec_A": [round(float(x), 5) for x in l_top8_A],
            "per_layer_top8_rec_B": [round(float(x), 5) for x in l_top8_B],
            "per_layer_ov12_rec_A": [round(float(x), 5) for x in l_ov12_A],
            "per_layer_sc80_prec_A": [round(float(x), 5) for x in l_sc_prec_A],
            "per_layer_sc80_rec_A": [round(float(x), 5) for x in l_sc_rec_A],
            "per_layer_sc80_cnt_A": [round(float(x), 3) for x in l_sc_cnt_A],
        }, f, indent=2)
    print("\nSaved spec_predictions.npz and e7_prediction_summary.json")


if __name__ == "__main__":
    main()
