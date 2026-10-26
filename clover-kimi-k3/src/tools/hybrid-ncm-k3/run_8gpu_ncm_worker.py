import json, os, sys, time, torch, torch.distributed as dist
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
