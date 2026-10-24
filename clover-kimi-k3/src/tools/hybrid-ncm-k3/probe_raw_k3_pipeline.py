"""Direct probe of what the 93 Kimi K3 layers + 92 MoE layers (`k3_full93_forward` + Layer-93 tail + `lm_head.weight`)
and `e11_clover_k3_native_ha_ncm.pt` (`K3NativeHybridLink` + `K3ComboBridge` + `K3ExpertBridge` + `NextConceptTransformer`)
actually output when run WITHOUT Qwen `HybridAgent-8B-Agentic` pre-writing the text and WITHOUT the `diag_sent` copy-pointer override."""
import os, sys, math, torch, torch.nn.functional as F
from safetensors import safe_open

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, "/home/usergpu/ncm/repo")
os.chdir("/home/usergpu/ncm/repo")

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, K3NativeHybridLink, K3ComboBridge, V_K3, D_K3, D_VQ, P_BPE, K3_PAD, K3_EOS
from train_combo_prefix import pack_combo_prefix
from train_e7_clean_chat import K3LexicalPointerDecoder
from eval_both_tracks import generate_from_zpref
from k3_agent_engine import K3GPUExpertsRunner, K3ExpertBridge


def main():
    tok = K3Tokenizer()
    runner = K3GPUExpertsRunner.get_instance(max_n=256, full_93_layers=True, k_res=28)
    runner.lib.k3_full93_set_routing_mode.argtypes = [
        __import__("ctypes").c_int,
        __import__("ctypes").c_int,
        __import__("ctypes").c_int,
    ]

    open_w = "/opt/clover-k3/datasets/third-party/kimi-k3-open-weights/model-00094-of-000096.safetensors"
    with safe_open(open_w, framework="pt", device="cpu") as f:
        lm_head = f.get_tensor("language_model.lm_head.weight").to(torch.float32)  # [163840, 7168]
    lm_head_gpu = lm_head.to("cuda:6")

    test_cases = [
        ("Plain text", tok.encode("The capital of France is")),
        ("XTML Chat: Capital of France", tok.encode_chat_prompt("What is the capital of France? Answer in one sentence.")),
        ("XTML Chat: Greeting", tok.encode_chat_prompt("Hi Clover, how are you doing today?")),
    ]

    for mode_val, kt, kd, mode_name in [
        (2, 2, 5, "SMART 896-EXPERT ROUTING (k_tok_tail=2, k_dyn_max=5, zero-eviction)"),
        (2, 3, 4, "SMART 896-EXPERT ROUTING (k_tok_tail=3, k_dyn_max=4, zero-eviction)"),
    ]:
        runner.lib.k3_full93_set_routing_mode(mode_val, kt, kd)
        print(f"\n=== TEST 1 ({mode_name}) ===", flush=True)
        for label, init_ids in test_cases:
            ids = list(init_ids)
            gen_ids = []
            t_pass_ms = []
            for step in range(4):
                t_in = torch.tensor([ids], dtype=torch.long)
                h_moe, meta = runner.forward_k3_ids(t_in)  # [1, len(ids), 7168]
                t_pass_ms.append(meta["k3_moe_ms"])
                h_last = h_moe[0, -1].to("cuda:6")
                logits = torch.matmul(lm_head_gpu, h_last).cpu()
                if step == 0:
                    top_vals, top_ids = torch.topk(logits[:163590], 6)
                    top_pieces = [(int(tid), round(float(val), 2), repr(tok.decode([int(tid)], skip_special_tokens=False))) for val, tid in zip(top_vals, top_ids)]
                    print(f"  [{label}] (len={len(ids)} tok, step0={meta['k3_moe_ms']}ms, uniq_exp={meta['total_unique_experts_activated']}):", flush=True)
                    print(f"    step-0 top-6 next tokens: {top_pieces}", flush=True)
                next_id = int(logits[:163590].argmax().item())
                if next_id in (tok.close_token_id, tok.eom_token_id, tok.eos_token_id):
                    break
                gen_ids.append(next_id)
                ids.append(next_id)
            print(f"    greedy 4-token continuation ({ [round(x, 1) for x in t_pass_ms] } ms): {tok.decode(gen_ids, skip_special_tokens=False)!r}", flush=True)

    print("\n=== TEST 2: WHAT `e12_clover_k3_true_ha_ncm.pt` (`K3 93-Layer MoE` + `HybridAgent-8B HybridLink` + `11-Pass NCM` + `Unfrozen VQ-VAE Decoder`) OUTPUTS ===")
    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    dev = torch.device("cuda:6")
    dev_ha = torch.device("cuda:7")
    _, tk, vqvae, net, src = load_k3_native_models(dev, os.path.join(out_dir, "k3_vqvae_adapted.pt"), os.path.join(out_dir, "e6_last.pt"))
    sd = torch.load(os.path.join(out_dir, "e11_clover_k3_native_ha_ncm.pt"), map_location="cpu", weights_only=True)
    vqvae.load_state_dict(sd["vqvae"]); net.load_state_dict(sd["net"])
    vqvae.eval(); net.eval()

    from train_combo_prefix import ComboBridge
    from hybrid_agent.model import load_hybrid_agent
    from train_latent_bridge import encode_batch_rh
    ha_model, ha_tok = load_hybrid_agent("togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=dev_ha, require_hybridlink=True)
    ha_tok.padding_side = "right"

    k3_hl = K3NativeHybridLink(d=D_K3, bottleneck=2048).to(dev)
    k3_cb = K3ComboBridge(d_in=D_K3, d_out=D_VQ).to(dev)
    ha_cb = ComboBridge(4096, D_VQ).to(dev)
    k3_eb = K3ExpertBridge(d_k3=D_K3, d_vq=D_VQ).to(dev)
    ptr_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
    k3_hl.load_state_dict(sd["k3_hybridlink"])
    k3_cb.load_state_dict(sd["k3_cbridge"])
    if "ha_cbridge" in sd:
        ha_cb.load_state_dict(sd["ha_cbridge"])
    elif "combo_bridge" in src:
        ha_cb.load_state_dict(src["combo_bridge"])
    k3_eb.load_state_dict(sd["k3_bridge"])
    ptr_dec.load_state_dict(sd["ptr_dec"])
    k3_hl.eval(); k3_cb.eval(); ha_cb.eval(); k3_eb.eval(); ptr_dec.eval()
    print(f"  Learned K3ExpertBridge gate tanh(alpha) = {math.tanh(k3_eb.alpha.item()):.6f}")

    levels = list(vqvae.vq.levels)
    sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]

    def trim_at_eos(id_list):
        out = []
        for tid in id_list:
            if tid == K3_EOS:
                break
            if 0 <= tid < 163584:
                out.append(tid)
        return out

    for user_q in [
        "Hi Clover, how are you?",
        "What is the capital of France?",
        "Explain binary search.",
        "What is the Clover Framework?",
        "capital France; Paris, located along Seine River in northern France",
    ]:
        q_ids = tok.encode_plain(user_q)[:P_BPE]
        p_kpad = torch.tensor([q_ids + [K3_PAD] * (P_BPE - len(q_ids))], dtype=torch.long, device=dev)
        pad_mask = (p_kpad == K3_PAD)
        h_k3_cpu, _ = runner.forward_k3_ids(p_kpad)
        h_k3 = h_k3_cpu.to(dev, dtype=torch.float32)

        with torch.no_grad():
            r_ha, _ = encode_batch_rh(ha_model, ha_tok, [user_q], 48, dev_ha)
            r_ha = r_ha.to(dev, dtype=torch.float32)
            r_k3 = k3_hl(h_k3)
            z_k3 = k3_cb(r_k3).float()
            z_ha = ha_cb(r_ha).float()
            z_rh = (0.65 * z_k3 + 0.35 * z_ha).masked_fill(pad_mask.unsqueeze(-1), 0.0)
            z_bpe = vqvae.encoder.wte(p_kpad) + k3_eb(h_k3).masked_fill(pad_mask.unsqueeze(-1), 0.0)
            z_pref, pmask = pack_combo_prefix(z_rh, pad_mask, z_bpe, pad_mask)
            g1 = generate_from_zpref(net, tk, z_pref, pmask, sched)
            zh1, _ = vqvae.vq.idx_to_zhats(g1, return_zlevels=True)

            # Raw VQ-VAE decoder output (without pointer copy of input tokens)
            raw_vq_logits = vqvae.decoder(zh1[-1])[0]  # [64, 163840]
            raw_vq_ids = trim_at_eos(raw_vq_logits[:, :163587].argmax(dim=-1).tolist())
            # Soft Pointer decoder output (NO +95.0 diagonal copy override!)
            ptr_logits = ptr_dec(vqvae.decoder, zh1[-1], z_rh, pad_mask, z_bpe, p_kpad)[0]
            ptr_ids = trim_at_eos(ptr_logits[:, :163587].argmax(dim=-1).tolist())

        print(f"Input: {user_q!r}", flush=True)
        print(f"  1) Pure NCM + VQ-VAE decoder (no pointer copy): {tok.decode(raw_vq_ids)!r}", flush=True)
        print(f"  2) NCM + Soft Pointer Decoder (no +95.0 override): {tok.decode(ptr_ids)!r}", flush=True)

    print("\n=== TEST 3: END-TO-END `CloverK3AgentEngine.agent_reply` ON SEEN & UNSEEN CONVERSATIONS ===", flush=True)
    del ha_model
    torch.cuda.empty_cache()
    from k3_agent_engine import CloverK3AgentEngine
    eng = CloverK3AgentEngine(dev_ncm="cuda:6", dev_ha="cuda:7", k3_experts_runner=runner)
    for q in [
        "Hi Clover, how are you?",
        "What is the capital of France?",
        "Explain binary search.",
        "What is 19 * 23 + 7?",
        "Tell me about quantum entanglement in simple terms.",
    ]:
        res = eng.agent_reply(q, cands=2, remember=False)
        print(f"User: {q!r}", flush=True)
        print(f"  Plans: {res['plans']}", flush=True)
        print(f"  Reply: {res['reply']!r}", flush=True)


if __name__ == "__main__":
    main()
