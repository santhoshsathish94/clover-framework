import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, OlmoeForCausalLM

MODEL_DIR = "/root/dm-study/specmd/w2/model"
TRACE_DIR = "/root/dm-study/specmd/w2/traces"
os.makedirs(TRACE_DIR, exist_ok=True)


def load_wikitext_prompts(tokenizer, num_prompts=8, seq_len=256):
    txt_path = os.path.join(TRACE_DIR, "wikitext2_valid.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if len(ln.strip()) > 80 and not ln.strip().startswith("=")]
    full_text = "\n\n".join(lines[:400])
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    needed = num_prompts * seq_len
    if enc.numel() < needed:
        raise RuntimeError(f"Not enough tokens: {enc.numel()} < {needed}")
    prompts = enc[:needed].reshape(num_prompts, seq_len).clone()
    return prompts


def run_prompt_style(model, prompts):
    num_prompts, seq_len = prompts.shape
    num_layers = len(model.model.layers)
    num_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok

    all_logits = np.zeros((num_prompts, seq_len, num_layers, num_experts), dtype=np.float32)
    all_probs = np.zeros((num_prompts, seq_len, num_layers, num_experts), dtype=np.float32)
    all_idx = np.zeros((num_prompts, seq_len, num_layers, top_k), dtype=np.uint8)
    all_scores = np.zeros((num_prompts, seq_len, num_layers, top_k), dtype=np.float32)

    cur_p = [0]
    hooks = []

    def make_hook(l_idx):
        def hook(module, inputs, output):
            logits, scores, indices = output
            probs = F.softmax(logits, dtype=torch.float32, dim=-1)
            p = cur_p[0]
            all_logits[p, :, l_idx, :] = logits.detach().float().cpu().numpy()
            all_probs[p, :, l_idx, :] = probs.detach().cpu().numpy()
            all_idx[p, :, l_idx, :] = indices.detach().cpu().numpy().astype(np.uint8)
            all_scores[p, :, l_idx, :] = scores.detach().float().cpu().numpy()
        return hook

    for l in range(num_layers):
        hooks.append(model.model.layers[l].mlp.gate.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        for p in range(num_prompts):
            cur_p[0] = p
            _ = model(prompts[p:p+1], use_cache=False)

    for h in hooks:
        h.remove()
    return all_logits, all_probs, all_idx, all_scores


def run_decode_style(model, prompts):
    num_prompts, seq_len = prompts.shape
    num_layers = len(model.model.layers)
    num_experts = model.config.num_experts
    top_k = model.config.num_experts_per_tok

    all_logits = np.zeros((num_prompts, seq_len, num_layers, num_experts), dtype=np.float32)
    all_probs = np.zeros((num_prompts, seq_len, num_layers, num_experts), dtype=np.float32)
    all_idx = np.zeros((num_prompts, seq_len, num_layers, top_k), dtype=np.uint8)
    all_scores = np.zeros((num_prompts, seq_len, num_layers, top_k), dtype=np.float32)

    cur_pt = [0, 0]
    hooks = []

    def make_hook(l_idx):
        def hook(module, inputs, output):
            logits, scores, indices = output
            probs = F.softmax(logits, dtype=torch.float32, dim=-1)
            p, t = cur_pt
            all_logits[p, t, l_idx, :] = logits[0].detach().float().cpu().numpy()
            all_probs[p, t, l_idx, :] = probs[0].detach().cpu().numpy()
            all_idx[p, t, l_idx, :] = indices[0].detach().cpu().numpy().astype(np.uint8)
            all_scores[p, t, l_idx, :] = scores[0].detach().float().cpu().numpy()
        return hook

    for l in range(num_layers):
        hooks.append(model.model.layers[l].mlp.gate.register_forward_hook(make_hook(l)))

    with torch.no_grad():
        for p in range(num_prompts):
            pkv = None
            for t in range(seq_len):
                cur_pt[0] = p
                cur_pt[1] = t
                out = model(prompts[p:p+1, t:t+1], past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values

    for h in hooks:
        h.remove()
    return all_logits, all_probs, all_idx, all_scores


def compare_traces(p_probs, p_idx, d_probs, d_idx, label="bf16"):
    abs_diff = np.abs(p_probs - d_probs)
    top1_match = float(np.mean(p_idx[..., 0] == d_idx[..., 0]))
    exact_ordered = float(np.mean(np.all(p_idx == d_idx, axis=-1)))
    # Set overlap (number of shared experts out of 8)
    P, T, L, K = p_idx.shape
    overlaps = np.zeros((P, T, L), dtype=np.float32)
    for p in range(P):
        for t in range(T):
            for l in range(L):
                s1 = set(p_idx[p, t, l].tolist())
                s2 = set(d_idx[p, t, l].tolist())
                overlaps[p, t, l] = len(s1.intersection(s2)) / float(K)
    print(f"=== Comparison ({label}): Prompt-style vs Decode-style ({P}x{T} tokens, {L} layers) ===")
    print(f"  Prob abs diff: mean={abs_diff.mean():.6e}, median={np.median(abs_diff):.6e}, max={abs_diff.max():.6e}")
    print(f"  Top-1 expert match:          {top1_match*100:.2f}%")
    print(f"  Top-8 set overlap (recall):  {overlaps.mean()*100:.2f}% (mean {overlaps.mean()*8:.3f} / 8 experts)")
    print(f"  Top-8 exact rank-order match:{exact_ordered*100:.2f}%")
    print("  Per-layer Top-1 match & Top-8 set overlap:")
    per_layer = {}
    for l in range(L):
        t1 = float(np.mean(p_idx[:, :, l, 0] == d_idx[:, :, l, 0]))
        ov = float(overlaps[:, :, l].mean())
        mad = float(abs_diff[:, :, l, :].mean())
        per_layer[l] = {"top1_match": round(t1, 5), "top8_set_overlap": round(ov, 5), "prob_mad": round(mad, 7)}
        print(f"    Layer {l:2d}: top1={t1*100:6.2f}%  top8_overlap={ov*100:6.2f}%  prob_mad={mad:.6e}")
    return {
        "prob_mad": float(abs_diff.mean()),
        "prob_max_diff": float(abs_diff.max()),
        "top1_match": top1_match,
        "top8_set_overlap": float(overlaps.mean()),
        "top8_exact_ordered": exact_ordered,
        "per_layer": per_layer,
    }


def main():
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = load_wikitext_prompts(tokenizer, num_prompts=8, seq_len=256)
    print(f"Prepared prompts shape={tuple(prompts.shape)} ({prompts.numel()} tokens total)")
    print("Sample decoded prompt[0][:120]:", repr(tokenizer.decode(prompts[0, :30])))

    model = OlmoeForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()

    t0 = time.time()
    p_logits, p_probs, p_idx, p_scores = run_prompt_style(model, prompts)
    t_prompt = time.time() - t0
    print(f"Prompt-style recording finished in {t_prompt:.2f}s ({prompts.numel()/t_prompt:.1f} tok/s)")

    t1 = time.time()
    d_logits, d_probs, d_idx, d_scores = run_decode_style(model, prompts)
    t_decode = time.time() - t1
    print(f"Decode-style recording finished in {t_decode:.2f}s ({prompts.numel()/t_decode:.1f} tok/s)")

    # Integrity checks on recorded traces
    print("\n=== Trace Integrity & Range Checks ===")
    print(f"  Prompt topk_indices shape={p_idx.shape}, min={p_idx.min()}, max={p_idx.max()}")
    print(f"  Decode topk_indices shape={d_idx.shape}, min={d_idx.min()}, max={d_idx.max()}")
    p_sum = p_probs.sum(axis=-1)
    d_sum = d_probs.sum(axis=-1)
    print(f"  Prompt router_probs sum: min={p_sum.min():.6f}, max={p_sum.max():.6f}")
    print(f"  Decode router_probs sum: min={d_sum.min():.6f}, max={d_sum.max():.6f}")
    # Check distinct experts per (token, layer) is always 8
    distinct_ok = all(len(set(row.tolist())) == 8 for row in p_idx.reshape(-1, 8))
    print(f"  All 32,768 (token, layer) decisions have 8 distinct experts: {distinct_ok}")

    bf16_cmp = compare_traces(p_probs, p_idx, d_probs, d_idx, label="bfloat16 (8x256=2048 tokens)")

    np.savez(
        os.path.join(TRACE_DIR, "prompt_traces.npz"),
        input_ids=prompts.numpy(),
        router_logits=p_logits.astype(np.float16),
        router_probs=p_probs.astype(np.float32),
        topk_indices=p_idx,
        topk_scores=p_scores.astype(np.float32),
    )
    np.savez(
        os.path.join(TRACE_DIR, "decode_traces.npz"),
        input_ids=prompts.numpy(),
        router_logits=d_logits.astype(np.float16),
        router_probs=d_probs.astype(np.float32),
        topk_indices=d_idx,
        topk_scores=d_scores.astype(np.float32),
    )

    # Control: float32 check on first 64 tokens of prompt 0 (layers 0..3) to isolate bfloat16 rounding
    print("\n=== Control: float32 Prompt-style vs Decode-style on Prompt 0 (64 tokens, layers 0..3) ===")
    import gc
    del model.model.layers[4:]
    model.config.num_hidden_layers = 4
    gc.collect()
    model.float()
    sub_prompts = prompts[:1, :64]
    _, fp32_p_probs, fp32_p_idx, _ = run_prompt_style(model, sub_prompts)
    _, fp32_d_probs, fp32_d_idx, _ = run_decode_style(model, sub_prompts)
    fp32_cmp = compare_traces(fp32_p_probs, fp32_p_idx, fp32_d_probs, fp32_d_idx, label="float32 control (1x64 tokens, 4 layers)")

    summary = {
        "num_prompts": 8,
        "seq_len": 256,
        "total_tokens": 2048,
        "num_layers": 16,
        "num_experts_per_layer": 64,
        "top_k": 8,
        "total_router_decisions": 32768,
        "total_expert_activations": 262144,
        "expert_bytes_bf16": 12582912,
        "prompt_style_sec": round(t_prompt, 2),
        "decode_style_sec": round(t_decode, 2),
        "bf16_comparison": bf16_cmp,
        "fp32_control_comparison": fp32_cmp,
    }
    with open(os.path.join(TRACE_DIR, "e3_trace_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved traces and summary to", TRACE_DIR)


if __name__ == "__main__":
    main()
