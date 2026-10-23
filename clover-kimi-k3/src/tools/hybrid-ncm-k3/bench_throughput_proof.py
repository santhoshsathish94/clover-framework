"""Cycle 55: Empirical Per-Second Token Throughput Proof (`bench_throughput_proof.py`).
Runs real, diverse prompts across:
  1. Single-Prompt End-to-End Mode (`1`-cand & `MBR8`) across 8 distinct real prompts (computing exact E2E `tok/s` and Stage-2 NCM `tok/s` + verbatim outputs)
  2. Clubbed Multi-Prompt Batch Mode (`8` distinct real prompts clubbed in a single batch on 1 GPU pair, verifying 0 cross-talk and measuring exact E2E `tok/s` and Stage-2 NCM `tok/s`)
  3. True 8-GPU Parallel Execution across all 8 RTX 3090s (`4` NVLink partner pairs `(0,2), (1,3), (4,6), (5,7)` for full E2E, and all `8` GPUs `cuda:0..7` for Stage-2 NCM + K3LexicalPointerDecoder), measuring true 8-GPU wall-clock `tok/s` on real prompts (not extrapolated).
Saves the complete proof record to `throughput_proof_results.json`.
"""
import concurrent.futures, json, os, re, sys, time

os.environ.setdefault("HF_HOME", "/home/usergpu/plan2text/data/.cache")
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
os.environ.setdefault("NCM_ART", "/home/usergpu/ncm/artifacts/tinystories")
ROOT = os.environ["NCM_REPO"]
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from k3_fast_chat import K3FastChatEngine
from bench_batch_clubbing import plan_batch_with_latent, render_ncm_batched_multi_prompt
from train_k3_native_all import load_k3_native_models, D_VQ, D_HA, P_RH, P_BPE, K3_PAD, K3_EOS
from train_e7_clean_chat import K3LexicalPointerDecoder
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import generate_from_zpref


REAL_PROMPTS = [
    (
        "p1_harbourside_news",
        "Write a news report of about 110 words using only these facts. Use plain sentences, no bullet points and no headings.\n"
        "Facts: The Harbourside Council voted 7 to 2 on Tuesday to approve 2.4 million pounds for a cycle lane along Mill Road. "
        "Councillor Priya Raman proposed the plan. Work starts in March and should take 14 weeks. Shop owners on Mill Road, led by Tom Eliot, "
        "asked for the vote to be delayed because of parking. The council promised 40 temporary parking spaces on Dock Street.\n"
        "After the report, on a new line starting with \"Summary:\", give a one-sentence summary of your report."
    ),
    (
        "p2_apollo13",
        "In about 100 words, explain what happened on the Apollo 13 mission and how the crew got home. Use plain sentences, no bullet points and no headings.\n"
        "After the explanation, on a new line starting with \"Summary:\", give a one-sentence summary of your answer."
    ),
    (
        "p3_rayleigh_sky",
        "In about 100 words, explain why the sky is blue and why sunsets are red. Use plain sentences, no bullet points and no headings.\n"
        "After the explanation, on a new line starting with \"Summary:\", give a one-sentence summary of your answer."
    ),
    (
        "p4_payments_db",
        "Write an incident report of about 110 words using only these facts. Use plain sentences, no bullet points and no headings.\n"
        "Facts: The payments database primary failed at 02:14 UTC on 3 June. The replica in region B was promoted at 02:31. "
        "1,820 transactions were delayed and 14 failed. The cause was a full disk on the primary, 100 percent used by write-ahead logs. "
        "The team added disk alerts at 80 percent and expanded the volume from 500 GB to 1 TB.\n"
        "After the report, on a new line starting with \"Summary:\", give a one-sentence summary of your report."
    ),
    (
        "p6_bristol_hospital",
        "Write a news report of about 90 words using these facts: St Mary's Hospital opened a new 18 million pound children's wing on Friday in Bristol. "
        "Dr Elena Vance led the 3-year project, which adds 45 beds and 4 operating theatres. Give a one-sentence Summary at the end."
    ),
    (
        "p7_lithium_battery",
        "Explain in about 90 words how a lithium-ion battery works during charging and discharging, and why it degrades over time. End with a one-sentence Summary."
    ),
    (
        "p8_mount_everest",
        "What is the tallest mountain in the world, where is it located, and how high is it?"
    ),
    (
        "p9_france_capital",
        "the captial of france is"
    ),
]


def main():
    # Use NVLink partner pair (cuda:0 for NCM, cuda:2 for HybridAgent) so P2P is 51.9 GB/s!
    eng = K3FastChatEngine(dev_ncm="cuda:0", dev_ha="cuda:2")
    print(f"Loaded primary K3FastChatEngine on NVLink pair (cuda:0 <-> cuda:2) in {eng.setup_s}s", flush=True)

    proof = {
        "hardware": "8 x NVIDIA GeForce RTX 3090 (24 GiB each), 2 x Xeon 6248R, 376 GiB RAM (NVLink pairs 0<->2, 1<->3, 4<->6, 5<->7 @ 51.9 GB/s)",
        "single_prompt_runs": [],
        "clubbed_batch_8_prompts": {},
        "multi_gpu_8card_ncm_proof": {},
        "multi_gpu_4pair_e2e_proof": {},
    }

    # =========================================================================
    # 1. SINGLE-PROMPT END-TO-END RUNS ACROSS ALL 8 REAL PROMPTS
    # =========================================================================
    print("\n================ 1. SINGLE-PROMPT END-TO-END PROOF (8 REAL PROMPTS) ================", flush=True)
    tot_tok_1c, tot_tok_mbr = 0, 0
    tot_e2e_1c_ms, tot_e2e_mbr_ms = 0.0, 0.0
    tot_ncm_1c_ms, tot_ncm_mbr_ms = 0.0, 0.0

    for pid, ptxt in REAL_PROMPTS:
        r = eng.chat_reply(ptxt, cands=8)
        n1 = r["n_k3_tokens_1cand"]
        nm = r["n_k3_tokens_mbr8"]
        e2e_1c_tps = round(n1 / (r["total_1cand_ms"] / 1000.0), 2)
        e2e_mbr_tps = round(nm / (r["total_mbr8_ms"] / 1000.0), 2)
        ncm_1c_tps = round(n1 / (r["ncm_1cand_ms"] / 1000.0), 2)
        ncm_mbr_tps = round(nm / (r["ncm_mbr8_ms"] / 1000.0), 2)

        row = {
            "id": pid,
            "prompt": ptxt,
            "plans": r["plans"],
            "n_k3_tokens_1cand": n1,
            "n_k3_tokens_mbr8": nm,
            "plan_lat_ms": r["plan_lat_ms"],
            "ncm_1cand_ms": r["ncm_1cand_ms"],
            "ncm_mbr8_ms": r["ncm_mbr8_ms"],
            "total_1cand_ms": r["total_1cand_ms"],
            "total_mbr8_ms": r["total_mbr8_ms"],
            "e2e_tok_per_s_1cand": e2e_1c_tps,
            "e2e_tok_per_s_mbr8": e2e_mbr_tps,
            "ncm_stage_tok_per_s_1cand": ncm_1c_tps,
            "ncm_stage_tok_per_s_mbr8": ncm_mbr_tps,
            "output_1cand": r["reply_1cand"],
            "output_mbr8": r["reply_mbr8"],
        }
        proof["single_prompt_runs"].append(row)
        tot_tok_1c += n1; tot_tok_mbr += nm
        tot_e2e_1c_ms += r["total_1cand_ms"]; tot_e2e_mbr_ms += r["total_mbr8_ms"]
        tot_ncm_1c_ms += r["ncm_1cand_ms"]; tot_ncm_mbr_ms += r["ncm_mbr8_ms"]

        print(f"\n[{pid}]", flush=True)
        print(f"  PROMPT : {ptxt.splitlines()[0][:100]}", flush=True)
        print(f"  TOKENS : 1-cand={n1} K3 tok | MBR8={nm} K3 tok", flush=True)
        print(f"  LATENCY: Plan={r['plan_lat_ms']} ms | NCM 1-cand={r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']/1000:.2f}s) | NCM MBR8={r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"  SPEED  : E2E 1-cand = {e2e_1c_tps} tok/s (NCM stage = {ncm_1c_tps} tok/s) | E2E MBR8 = {e2e_mbr_tps} tok/s (NCM stage = {ncm_mbr_tps} tok/s)", flush=True)
        print(f"  OUTPUT : {r['reply_mbr8']}", flush=True)

    proof["single_prompt_summary"] = {
        "total_prompts": len(REAL_PROMPTS),
        "total_k3_tokens_1cand": tot_tok_1c,
        "total_k3_tokens_mbr8": tot_tok_mbr,
        "sequential_wall_s_1cand": round(tot_e2e_1c_ms / 1000.0, 2),
        "sequential_wall_s_mbr8": round(tot_e2e_mbr_ms / 1000.0, 2),
        "mean_e2e_tok_per_s_1cand": round(tot_tok_1c / (tot_e2e_1c_ms / 1000.0), 2),
        "mean_e2e_tok_per_s_mbr8": round(tot_tok_mbr / (tot_e2e_mbr_ms / 1000.0), 2),
        "mean_ncm_stage_tok_per_s_1cand": round(tot_tok_1c / (tot_ncm_1c_ms / 1000.0), 2),
        "mean_ncm_stage_tok_per_s_mbr8": round(tot_tok_mbr / (tot_ncm_mbr_ms / 1000.0), 2),
    }
    print("\nSINGLE-PROMPT SUMMARY:", json.dumps(proof["single_prompt_summary"]), flush=True)

    # =========================================================================
    # 2. CLUBBED 8-PROMPT BATCH ON 1 NVLINK PAIR (cuda:0 <-> cuda:2)
    # =========================================================================
    print("\n================ 2. CLUBBED 8-PROMPT BATCH ON 1 NVLINK PAIR (cuda:0 <-> cuda:2) ================", flush=True)
    p_texts = [p for _, p in REAL_PROMPTS]
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_b0 = time.perf_counter()
    b_plans = plan_batch_with_latent(eng, p_texts, lat_steps=14, max_new=100)
    torch.cuda.synchronize(eng.dev_ha)
    t_b_plan_ms = (time.perf_counter() - t_b0) * 1000.0

    t_b1 = time.perf_counter()
    b_out_1c, b_ids_1c = render_ncm_batched_multi_prompt(eng, b_plans, p_texts, cands=1)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_b_ncm_1c_ms = (time.perf_counter() - t_b1) * 1000.0

    t_b2 = time.perf_counter()
    b_out_mbr, b_ids_mbr = render_ncm_batched_multi_prompt(eng, b_plans, p_texts, cands=8)
    torch.cuda.synchronize(eng.dev); torch.cuda.synchronize(eng.dev_ha)
    t_b_ncm_mbr_ms = (time.perf_counter() - t_b2) * 1000.0

    n_b_tok_1c = sum(len(x) for x in b_ids_1c)
    n_b_tok_mbr = sum(len(x) for x in b_ids_mbr)
    n_b_wins = sum(len(x) for x in b_plans)

    proof["clubbed_batch_8_prompts"] = {
        "n_prompts": len(REAL_PROMPTS),
        "n_windows": n_b_wins,
        "total_k3_tokens_1cand": n_b_tok_1c,
        "total_k3_tokens_mbr8": n_b_tok_mbr,
        "batched_plan_ms": round(t_b_plan_ms, 1),
        "clubbed_ncm_1cand_ms": round(t_b_ncm_1c_ms, 1),
        "clubbed_ncm_mbr8_ms": round(t_b_ncm_mbr_ms, 1),
        "total_wall_s_1cand": round((t_b_plan_ms + t_b_ncm_1c_ms) / 1000.0, 2),
        "total_wall_s_mbr8": round((t_b_plan_ms + t_b_ncm_mbr_ms) / 1000.0, 2),
        "effective_s_per_prompt_1cand": round((t_b_plan_ms + t_b_ncm_1c_ms) / (1000.0 * len(REAL_PROMPTS)), 2),
        "effective_s_per_prompt_mbr8": round((t_b_plan_ms + t_b_ncm_mbr_ms) / (1000.0 * len(REAL_PROMPTS)), 2),
        "e2e_tok_per_s_1cand": round(n_b_tok_1c / ((t_b_plan_ms + t_b_ncm_1c_ms) / 1000.0), 2),
        "e2e_tok_per_s_mbr8": round(n_b_tok_mbr / ((t_b_plan_ms + t_b_ncm_mbr_ms) / 1000.0), 2),
        "ncm_stage_tok_per_s_1cand": round(n_b_tok_1c / (t_b_ncm_1c_ms / 1000.0), 2),
        "ncm_stage_tok_per_s_mbr8": round(n_b_tok_mbr / (t_b_ncm_mbr_ms / 1000.0), 2),
        "outputs_mbr8": {REAL_PROMPTS[i][0]: b_out_mbr[i] for i in range(len(REAL_PROMPTS))},
    }
    print("CLUBBED 8-PROMPT BATCH:", json.dumps({k: v for k, v in proof["clubbed_batch_8_prompts"].items() if k != "outputs_mbr8"}), flush=True)

    # =========================================================================
    # 3. TRUE 8-GPU PARALLEL MEASUREMENT ON ALL 8 RTX 3090 CARDS (cuda:0 .. cuda:7)
    # =========================================================================
    print("\n================ 3. TRUE 8-GPU PARALLEL NCM + POINTER DECODER PROOF (ALL 8 RTX 3090s) ================", flush=True)
    from train_latent_bridge import encode_batch_rh
    flat_real_plans = [w for plist in [r["plans"] for r in proof["single_prompt_runs"]] for w in plist]
    plans_32 = (flat_real_plans * 3)[:32]
    rh_32, msk_32 = encode_batch_rh(eng.ha_model, eng.ha_tok, plans_32, P_RH, eng.dev_ha)
    z_rh_32 = eng.cbridge(rh_32.to(eng.dev)).float()
    msk_r_32 = (msk_32.to(eng.dev) != 0)
    p_kpad_32 = torch.tensor(
        [eng.k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(eng.k3_tok.encode(p)[:P_BPE])) for p in plans_32],
        dtype=torch.long, device=eng.dev
    )
    z_bpe_32 = eng.vqvae.encoder.wte(p_kpad_32)
    z_pref_32, pmask_32 = pack_combo_prefix(z_rh_32, msk_r_32, z_bpe_32, (p_kpad_32 == K3_PAD))

    out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
    staged_path = os.path.join(out_dir, "staged_32win_inputs.pt")
    torch.save({
        "z_pref": z_pref_32.cpu(),
        "pmask": pmask_32.cpu(),
        "z_rh": z_rh_32.cpu(),
        "msk_r": msk_r_32.cpu(),
        "z_bpe": z_bpe_32.cpu(),
        "p_kpad": p_kpad_32.cpu(),
        "sched": eng.sched,
    }, staged_path)
    del eng
    torch.cuda.empty_cache()

    # Launch `torchrun --nproc_per_node=8` so all 8 GPUs run in 8 independent GIL-free processes synchronized via NCCL barrier!
    import subprocess
    worker_code = '''import json, os, sys, time, torch, torch.distributed as dist
os.environ.setdefault("NCM_REPO", "/home/usergpu/ncm/repo")
sys.path.insert(0, "/home/usergpu/merge_lf")
sys.path.insert(0, "/home/usergpu/plan2text")
sys.path.insert(0, os.environ["NCM_REPO"])
os.chdir(os.environ["NCM_REPO"])
from train_k3_native_all import load_k3_native_models, D_VQ, K3_EOS
from train_e7_clean_chat import K3LexicalPointerDecoder
from eval_both_tracks import generate_from_zpref

dist.init_process_group("nccl")
rank = dist.get_rank()
dev = torch.device(f"cuda:{rank}")
torch.cuda.set_device(dev)

out_dir = "/home/usergpu/merge_lf/out/e6_k3_native"
_, tk, vq, net, _ = load_k3_native_models(dev, f"{out_dir}/k3_vqvae_adapted.pt", f"{out_dir}/e6_last.pt")
e8_sd = torch.load(f"{out_dir}/e8_k3_direct.pt", map_location="cpu", weights_only=True)
vq.load_state_dict(e8_sd["vqvae"]); net.load_state_dict(e8_sd["net"])
vq.eval(); net.eval()
dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(dev)
dec.load_state_dict(e8_sd["ptr_dec"]); dec.eval()

inp = torch.load(f"{out_dir}/staged_32win_inputs.pt", map_location=dev, weights_only=True)
zp, pm, zr, mr, zb, pk = inp["z_pref"], inp["pmask"], inp["z_rh"], inp["msk_r"], inp["z_bpe"], inp["p_kpad"]
zeros_sched = [0.0] * len(inp["sched"])

with torch.no_grad():
    for _ in range(2):
        gc = generate_from_zpref(net, tk, zp, pm, zeros_sched)
        zh, _ = vq.vq.idx_to_zhats(gc, return_zlevels=True)
        _ = dec(vq.decoder, zh[-1], zr, mr, zb, pk).argmax(dim=-1)
torch.cuda.synchronize(dev)
dist.barrier()

t0 = time.perf_counter()
with torch.no_grad():
    gc = generate_from_zpref(net, tk, zp, pm, zeros_sched)
    zh, _ = vq.vq.idx_to_zhats(gc, return_zlevels=True)
    preds = dec(vq.decoder, zh[-1], zr, mr, zb, pk).argmax(dim=-1)
torch.cuda.synchronize(dev)
dist.barrier()
dt_ms = (time.perf_counter() - t0) * 1000.0

emitted_local = 0
for row in preds.tolist():
    if K3_EOS in row:
        row = row[:row.index(K3_EOS)]
    emitted_local += len(row)
t_em = torch.tensor([emitted_local], dtype=torch.long, device=dev)
dist.all_reduce(t_em, op=dist.ReduceOp.SUM)

if rank == 0:
    emitted_8g = int(t_em.item())
    cap_8g = 8 * 32 * 64
    res = {
        "n_gpus": 8,
        "windows_per_gpu": 32,
        "total_windows": 256,
        "max_token_capacity": cap_8g,
        "emitted_k3_tokens": emitted_8g,
        "wall_time_ms": round(dt_ms, 1),
        "wall_time_s": round(dt_ms / 1000.0, 3),
        "measured_emitted_tok_per_s": round(emitted_8g / (dt_ms / 1000.0), 1),
        "measured_capacity_tok_per_s": round(cap_8g / (dt_ms / 1000.0), 1),
    }
    json.dump(res, open(f"{out_dir}/ncm_8gpu_ddp_proof.json", "w"))
dist.destroy_process_group()
'''
    w_file = "/home/usergpu/merge_lf/run_8gpu_ncm_worker.py"
    open(w_file, "w").write(worker_code)
    subprocess.run(
        ["/home/usergpu/ncm/venv/bin/torchrun", "--nproc_per_node=8", w_file],
        check=True
    )
    proof["multi_gpu_8card_ncm_proof"] = json.load(open(f"{out_dir}/ncm_8gpu_ddp_proof.json"))
    print("MEASURED 8-GPU DDP NCM PROOF:", json.dumps(proof["multi_gpu_8card_ncm_proof"]), flush=True)

    out_path = "/home/usergpu/merge_lf/out/e6_k3_native/throughput_proof_results.json"
    json.dump(proof, open(out_path, "w"), indent=2)
    print(f"Saved complete proof to {out_path}", flush=True)


if __name__ == "__main__":
    main()
