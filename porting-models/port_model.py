#!/usr/bin/env python3
"""
Clover Universal Model Porting CLI (`porting-models/port_model.py` / `./porting-models/port`).

Provides a single unified command-line tool to port any open-weight foundation model into
Clover AI Studio's 7-layer neuro-symbolic stack (`Layer-1 Precomputed Trunks & Direct Experts` ->
`Layer-2 HybridAgent 3-Stage HybridSFT` -> `Layer-3 11-Level Next-Concept Model (NCM) & Bridges` ->
`Layers 4-7 Memory Tree, Ontology GraphRAG, Generative Compilation, and SpecMD + R4 SlotCache`).

Usage:
  ./porting-models/port inspect           --config porting-models/configs/kimi-k3.1-pro.json
  ./porting-models/port prepare-trunks    --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro
  ./porting-models/port prune-trunks      --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro
  ./porting-models/port train-hybridagent --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro --gpus 8
  ./porting-models/port train-ncm         --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro
  ./porting-models/port verify            --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro
  ./porting-models/port all               --config porting-models/configs/kimi-k3.1-pro.json --work-dir /dev/shm/clover-port/kimi-k3.1-pro --gpus 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_port_config(config_path: str | Path) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        alt = SCRIPT_DIR / config_path
        if alt.exists():
            p = alt
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_inspect(cfg: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    from kimi_k31_gpu_backbone import enable_ipv4_only

    enable_ipv4_only()
    repo_id = cfg["hf_repo_id"]
    drafter_repo = cfg.get("drafter_hf_repo_id")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Clover Port] Inspecting primary source repository: {repo_id}", flush=True)
    hf_cfg_path = hf_hub_download(repo_id, "config.json")
    hf_cfg = json.load(open(hf_cfg_path, encoding="utf-8"))

    idx_info: dict[str, Any] = {}
    try:
        idx_path = hf_hub_download(repo_id, "model.safetensors.index.json")
        idx_data = json.load(open(idx_path, encoding="utf-8"))
        wm = idx_data.get("weight_map", {})
        shards = sorted(set(wm.values()))
        indexer_tensors = [k for k in wm if ".indexer." in k]
        idx_info = {
            "total_parameter_bytes": idx_data.get("metadata", {}).get("total_size", 0),
            "total_tensors": len(wm),
            "total_shards": len(shards),
            "dsa_indexer_tensors_count": len(indexer_tensors),
        }
    except Exception as e:
        idx_info = {"note": str(e)}

    drafter_info: dict[str, Any] = {}
    if drafter_repo:
        try:
            d_cfg_path = hf_hub_download(drafter_repo, "config.json")
            d_cfg = json.load(open(d_cfg_path, encoding="utf-8"))
            drafter_info = {
                "drafter_repo": drafter_repo,
                "architectures": d_cfg.get("architectures"),
                "num_hidden_layers": d_cfg.get("num_hidden_layers"),
                "target_layer_ids": d_cfg.get("dflash_config", {}).get("target_layer_ids"),
            }
        except Exception as e:
            drafter_info = {"drafter_repo": drafter_repo, "error": str(e)}

    report = {
        "model_key": cfg["model_key"],
        "display_name": cfg.get("display_name"),
        "hf_repo_id": repo_id,
        "model_type": hf_cfg.get("model_type"),
        "architectures": hf_cfg.get("architectures"),
        "hidden_size": cfg["hidden_size"],
        "vocab_size": cfg["vocab_size"],
        "total_layers": cfg["total_layers"],
        "index_summary": idx_info,
        "drafter_summary": drafter_info,
    }
    out_p = work_dir / "inspect_report.json"
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def cmd_prepare_trunks(cfg: dict[str, Any], work_dir: Path, cached_experts: int = 32) -> dict[str, Any]:
    model_key = cfg["model_key"]
    if model_key.startswith("kimi-k3"):
        from kimi_k31_gpu_backbone import prepare_kimi_k31_datasets

        res = prepare_kimi_k31_datasets(
            work_dir=work_dir,
            repo_id=cfg["hf_repo_id"],
            drafter_repo_id=cfg.get("drafter_hf_repo_id") or "",
            num_cached_experts_per_stage=cached_experts,
            extract_all_24_dsa_indexers=True,
        )
        print(json.dumps(res, indent=2), flush=True)
        return res
    raise ValueError(f"Unsupported model_key for prepare_trunks: {model_key}")


def cmd_prune_trunks(cfg: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    model_key = cfg["model_key"]
    if model_key.startswith("kimi-k3"):
        from kimi_k31_gpu_backbone import verify_kimi_k31_stages

        res = verify_kimi_k31_stages(work_dir / "pruned")
        (work_dir / "prune_and_stage_verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=2), flush=True)
        return res
    raise ValueError(f"Unsupported model_key for prune_trunks: {model_key}")


def cmd_train_hybridagent(cfg: dict[str, Any], work_dir: Path, gpus: int = 8) -> None:
    t_defaults = cfg.get("training_defaults", {})
    s1 = int(t_defaults.get("hybridsft_stage1_steps", 40))
    s2 = int(t_defaults.get("hybridsft_stage2_steps", 60))
    s3 = int(t_defaults.get("hybridsft_stage3_steps", 60))
    script = SCRIPT_DIR / "train_kimi_k31_hybridagent_gpu.py"

    if gpus > 1:
        torchrun_bin = "/venv/main/bin/torchrun" if os.path.exists("/venv/main/bin/torchrun") else "torchrun"
        cmd = [
            torchrun_bin,
            f"--nproc_per_node={gpus}",
            str(script),
            "--work-dir",
            str(work_dir),
            "--stage1-steps",
            str(s1),
            "--stage2-steps",
            str(s2),
            "--stage3-steps",
            str(s3),
        ]
    else:
        cmd = [
            sys.executable,
            str(script),
            "--work-dir",
            str(work_dir),
            "--stage1-steps",
            str(s1),
            "--stage2-steps",
            str(s2),
            "--stage3-steps",
            str(s3),
        ]
    print(f"[Clover Port] Launching 3-Stage HybridSFT: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def cmd_train_ncm(cfg: dict[str, Any], work_dir: Path) -> None:
    t_defaults = cfg.get("training_defaults", {})
    n_win = int(t_defaults.get("ncm_windows", 1024))
    vq_steps = int(t_defaults.get("vqvae_steps", 400))
    ncm_steps = int(t_defaults.get("ncm_bridge_steps", 500))
    script = SCRIPT_DIR / "train_kimi_k31_ncm_gpu.py"

    cmd = [
        sys.executable,
        str(script),
        "--work-dir",
        str(work_dir),
        "--num-windows",
        str(n_win),
        "--vqvae-steps",
        str(vq_steps),
        "--ncm-steps",
        str(ncm_steps),
    ]
    print(f"[Clover Port] Launching 4-Phase GPU NCM Trainer: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def cmd_verify(cfg: dict[str, Any], work_dir: Path) -> dict[str, Any]:
    from kimi_k31_gpu_backbone import verify_kimi_k31_stages

    stage_ver = verify_kimi_k31_stages(work_dir / "pruned")
    ha_sum_p = work_dir / "hybridagent-8gpu" / "kimi_k31_hybridagent_training_summary.json"
    ncm_sum_p = work_dir / "hybrid-ncm" / "kimi_k31_hybrid_ncm_gpu_training_summary.json"

    report = {
        "model_key": cfg["model_key"],
        "hf_repo_id": cfg["hf_repo_id"],
        "drafter_hf_repo_id": cfg.get("drafter_hf_repo_id"),
        "stage_verification": stage_ver,
        "hybridagent_summary": json.loads(ha_sum_p.read_text(encoding="utf-8")) if ha_sum_p.exists() else None,
        "ncm_summary": json.loads(ncm_sum_p.read_text(encoding="utf-8")) if ncm_sum_p.exists() else None,
        "studio_checkpoint_exists": (work_dir / "hybrid-ncm" / "checkpoint-clover-true-hybrid-agent-ncm.pt").exists(),
    }
    (work_dir / "port_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clover Universal Model Porting CLI (`./porting-models/port`)")
    parser.add_argument(
        "command",
        choices=["inspect", "prepare-trunks", "prune-trunks", "train-hybridagent", "train-ncm", "verify", "all"],
        help="Porting stage to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "configs" / "kimi-k3.1-pro.json"),
        help="Path to model porting JSON config",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/dev/shm/clover-port/kimi-k3.1-pro"),
        help="Working directory for precomputed trunks, datasets, and trained checkpoints (use /dev/shm on GPU servers)",
    )
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs for distributed HybridSFT")
    parser.add_argument("--cached-experts", type=int, default=32, help="Number of MXFP4 experts to cache per representative stage")
    args = parser.parse_args()

    cfg = load_port_config(args.config)
    t0 = time.perf_counter()

    if args.command == "inspect":
        cmd_inspect(cfg, args.work_dir)
    elif args.command == "prepare-trunks":
        cmd_prepare_trunks(cfg, args.work_dir, cached_experts=args.cached_experts)
    elif args.command == "prune-trunks":
        cmd_prune_trunks(cfg, args.work_dir)
    elif args.command == "train-hybridagent":
        cmd_train_hybridagent(cfg, args.work_dir, gpus=args.gpus)
    elif args.command == "train-ncm":
        cmd_train_ncm(cfg, args.work_dir)
    elif args.command == "verify":
        cmd_verify(cfg, args.work_dir)
    elif args.command == "all":
        cmd_inspect(cfg, args.work_dir)
        cmd_prepare_trunks(cfg, args.work_dir, cached_experts=args.cached_experts)
        cmd_prune_trunks(cfg, args.work_dir)
        cmd_train_hybridagent(cfg, args.work_dir, gpus=args.gpus)
        cmd_train_ncm(cfg, args.work_dir)
        cmd_verify(cfg, args.work_dir)

    print(f"[Clover Port] Completed '{args.command}' in {time.perf_counter() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
