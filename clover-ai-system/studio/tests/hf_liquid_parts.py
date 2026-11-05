"""Lists the liquid-lfm3 precomputed parts (Santhoshsathish94/clover-ai-studio) and compares them with this laptop's
model files by SHA-256. Read-only.  python tests\\hf_liquid_parts.py"""
import hashlib
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from safetensors import safe_open

REPO = "Santhoshsathish94/clover-ai-studio"
D = Path(__file__).resolve().parents[1] / "data"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


api = HfApi()
files = [e for e in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo="liquid-lfm3", recursive=True)
         if isinstance(e, RepoFile)]
print("liquid-lfm3 files:", len(files))
groups = defaultdict(list)
for e in files:
    groups[e.path.split("/")[1]].append(e)
for g, es in sorted(groups.items()):
    print(f"  {g:<12} {len(es):4} files {sum(e.size for e in es):>14,} bytes")
hf = {e.path: (e.lfs.sha256 if e.lfs else "git:" + e.blob_id, e.size) for e in files}
print("--- top-level files and global-0")
for p, (h, s) in sorted(hf.items()):
    if p.count("/") <= 2 or "/global-0/" in p:
        print(f"  {h}  {s:>12,}  {p}")
print("--- trunk-0 and plan-trunk-0")
for p, (h, s) in sorted(hf.items()):
    if "/trunk-0/" in p or "/plan-trunk-0/" in p:
        print(f"  {s:>12,}  {p}")
print("--- commits (newest first)")
for c in list(api.list_repo_commits(REPO, repo_type="dataset"))[:15]:
    print(f"  {c.created_at:%Y-%m-%d %H:%M}  {c.title}")

local = {
    "trained Omni NCM core": D / "omni/trained/omni2b-ncm/lfm3_ncm_core_bf16.pt",
    "trained Omni vqvae_wte": D / "omni/trained/omni2b-ncm/vqvae_wte.bf16.bin",
    "trained Omni vqvae_dec": D / "omni/trained/omni2b-ncm/vqvae_dec.bf16.bin",
    "trained Omni HybridLink adapter": D / "omni/trained/omni2b-s3/adapter.pt",
    "LFM3-1.7B NCM bridges": D / "ncm/lfm3_hybrid_ncm_bridges.pt",
    "LFM3-1.7B embed_tokens (laptop trunk)": D / "lfm3/pruned/global-0/embed_tokens.bf16.bin",
}
hashes = {name: sha(p) for name, p in local.items() if p.is_file()}
with safe_open(str(D / "omni/trained/omni2b-s3/model.safetensors"), "pt") as f:
    k = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
    t = f.get_tensor(k).to(torch.bfloat16).contiguous()
    hashes[f"trained Omni S3 {k} (bf16 bytes)"] = hashlib.sha256(t.view(torch.int16).numpy().tobytes()).hexdigest()
by_hash = defaultdict(list)
for p, (h, s) in hf.items():
    by_hash[h].append(p)
print("--- laptop files found in liquid-lfm3 (same SHA-256)")
for name, h in hashes.items():
    print(f"  {name}: {h[:16]}... -> {', '.join(by_hash.get(h, [])) or 'NOT in liquid-lfm3'}")
