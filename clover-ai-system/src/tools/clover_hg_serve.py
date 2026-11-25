#!/usr/bin/env python3
"""One-command launcher (`clover_hg_serve.py`) for Clover's Hadamard-Grouped (`HG`) models.

Automatically:
  1. Downloads the requested `HG4-16R-OP` / `K31H` checkpoint from Hugging Face (if not cached locally).
  2. Compiles the zero-dependency C11 SIMD engine (`lib*_hg4_engine.so`) and — when NVIDIA `nvcc` is
     present — compiles `libhg_cuda_accel.so` (`RTX 3090 sm_86`, `RTX 4090 sm_89`, `A100 sm_80`,
     `H100/H200 sm_90`, `Blackwell sm_100/sm_120`) for automatic single- or multi-GPU VRAM offload.
  3. Starts Clover's OpenAI-compatible `/v1/chat/completions` & `/v1/models` HTTP server.

Quickstart:
  python3 clover-ai-system/src/tools/clover_hg_serve.py --model lfm3-omni-2b --port 8000
  python3 clover-ai-system/src/tools/clover_hg_serve.py --model clover-35b-a3b --gpus auto --port 8000
  python3 clover-ai-system/src/tools/clover_hg_serve.py --model nemotron-3.5-super-vl --gpus 0,1,2,3 --port 8000
  python3 clover-ai-system/src/tools/clover_hg_serve.py --model kimi-k3 --gpus auto --port 8000
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "lfm3-omni-2b": {
        "hf_repos": ["Santhoshsathish94/LFM3-Omni-2B-HG4-16R"],
        "env_var": "CLOVER_OMNI_HG4_DIR",
        "c_rel": "models/lfm3-1.7b/lfm3_hg4_engine.c",
        "so_name": "liblfm3_hg4_engine.so",
        "size_note": "0.83 GiB HG4-16R (Phone / Laptop / 1 GB VRAM)",
    },
    "clover-35b-a3b": {
        "hf_repos": ["Santhoshsathish94/Clover-35B-A3B-HG4-16R-OP"],
        "env_var": "CLOVER35B_HG4_DIR",
        "c_rel": "models/clover-35b-a3b/clover35b_hg4_engine.c",
        "so_name": "libclover35b_hg4_engine.so",
        "size_note": "17.2 GiB HG4-16R-OP (32 GB RAM or 1x 24 GB RTX 3090/4090)",
    },
    "nemotron-3.5-super-vl": {
        "hf_repos": ["Santhoshsathish94/Nemotron-3.5-Super-VL-120B-HG4-16R-OP"],
        "env_var": "CLOVER_NEMO_HG4_DIR",
        "c_rel": "models/nemotron-3.5-super-vl/nemotron_hg4_engine.c",
        "so_name": "libnemotron_hg4_engine.so",
        "size_note": "60.2 GiB HG4-16R-OP (128 GB RAM or 4x 24 GB RTX 3090/4090)",
    },
    "kimi-k3": {
        "hf_repos": [
            "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Hot-HG4-OP",
            "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part1",
            "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part2",
            "Santhoshsathish94/Kimi-K3.1-Pro-K31H-Cold-HG2-Part3",
        ],
        "env_var": "CLOVER_KIMI_HG4_DIR",
        "c_rel": "models/kimi-k3/kimi_k31_hg4_engine.c",
        "so_name": "libkimi_k31_hg4_engine.so",
        "size_note": "792 GB K31H (128 GB RAM + 1 TB NVMe, or 8x H200 / MI300X)",
    },
}

ALIASES = {
    "2b": "lfm3-omni-2b",
    "lfm3": "lfm3-omni-2b",
    "35b": "clover-35b-a3b",
    "clover-35b": "clover-35b-a3b",
    "120b": "nemotron-3.5-super-vl",
    "nemotron": "nemotron-3.5-super-vl",
    "nemotron-120b": "nemotron-3.5-super-vl",
    "kimi": "kimi-k3",
    "kimi-k3.1-pro": "kimi-k3",
}


def resolve_src_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here.parent / "models" / "hg_cuda_hook.h").is_file():
        return here.parent
    for cand in [Path.cwd() / "clover-ai-system" / "src", Path("/app/clover-ai-system/src")]:
        if (cand / "models" / "hg_cuda_hook.h").is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate `clover-ai-system/src`. Run inside a clone of "
        "`https://github.com/santhoshsathish94/clover-framework.git`."
    )


def detect_nvcc_arch_flags() -> list[str]:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                text=True,
                timeout=8,
            )
            caps = sorted({ln.strip().replace(".", "") for ln in out.splitlines() if ln.strip()})
            if caps:
                flags: list[str] = []
                for c in caps:
                    flags.extend(["-gencode", f"arch=compute_{c},code=sm_{c}"])
                return flags
        except Exception:
            pass
    # Universal multi-arch fatbin fallback (Ampere 3090/A100 + Ada 4090 + Hopper H100/H200)
    return [
        "-gencode", "arch=compute_80,code=sm_80",
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_89,code=sm_89",
        "-gencode", "arch=compute_90,code=sm_90",
    ]


def ensure_compiled(src_root: Path, lib_dir: Path, models: list[str], gpus: str, force_rebuild: bool = False) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("C compiler (`gcc` or `clang`) not found. Install `build-essential` / `gcc`.")

    for mid in models:
        spec = MODEL_REGISTRY[mid]
        c_src = src_root / str(spec["c_rel"])
        so_out = lib_dir / str(spec["so_name"])
        if force_rebuild or not so_out.is_file() or so_out.stat().st_mtime < c_src.stat().st_mtime:
            cmd = [cc, "-O3", "-fopenmp", "-ffp-contract=off", "-shared", "-fPIC", str(c_src), "-o", str(so_out), "-lm", "-ldl"]
            print(f"[clover-hg] Compiling CPU SIMD engine `{so_out.name}` ...", flush=True)
            subprocess.check_call(cmd)

    nvcc = shutil.which("nvcc")
    cuda_so = lib_dir / "libhg_cuda_accel.so"
    cu_src = src_root / "tools" / "hg_cuda_accel.cu"
    if gpus not in ("none", "off", "cpu") and nvcc and cu_src.is_file():
        if force_rebuild or not cuda_so.is_file() or cuda_so.stat().st_mtime < cu_src.stat().st_mtime:
            arch_flags = detect_nvcc_arch_flags()
            cmd = [nvcc, "-O3", "--use_fast_math", "-Xcompiler", "-fPIC", "-shared", *arch_flags, str(cu_src), "-o", str(cuda_so)]
            print(f"[clover-hg] Compiling CUDA VRAM accelerator `{cuda_so.name}` ({' '.join(arch_flags)}) ...", flush=True)
            subprocess.check_call(cmd)
    elif gpus not in ("none", "off", "cpu") and cuda_so.is_file():
        print(f"[clover-hg] Using prebuilt CUDA accelerator `{cuda_so}`.", flush=True)
    else:
        print("[clover-hg] Running in pure C11 CPU mode (`CLOVER_CUDA_GPUS=none`).", flush=True)


def ensure_weights(mid: str, cache_dir: Path, custom_dir: Path | None = None) -> Path:
    spec = MODEL_REGISTRY[mid]
    env_key = str(spec["env_var"])
    if custom_dir and (custom_dir / "meta.bin").is_file():
        return custom_dir
    env_existing = os.environ.get(env_key, "").strip()
    if env_existing and (Path(env_existing) / "meta.bin").is_file():
        return Path(env_existing)

    # Check standard k3 / container pre-mounted paths first
    default_dirs = {
        "lfm3-omni-2b": [Path("/opt/clover-k3/lfm3-omni-hg4"), Path("/models/lfm3-omni-2b")],
        "clover-35b-a3b": [Path("/opt/clover-k3/clover35b-full-hg4"), Path("/models/clover-35b-a3b")],
        "nemotron-3.5-super-vl": [Path("/opt/clover-k3/nemotron-full-hg4"), Path("/models/nemotron-3.5-super-vl")],
        "kimi-k3": [Path("/opt/clover-k3/kimi-k31-hg4"), Path("/models/kimi-k3")],
    }
    for d in default_dirs.get(mid, []):
        if (d / "meta.bin").is_file():
            return d

    target = custom_dir if custom_dir else (cache_dir / mid)
    if (target / "meta.bin").is_file():
        return target

    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install `huggingface_hub` (`pip install huggingface_hub tokenizers numpy`) to auto-download HG weights.") from exc

    for repo_id in spec["hf_repos"]:  # type: ignore[union-attr]
        print(f"[clover-hg] Downloading `{repo_id}` -> `{target}` ...", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(target), max_workers=8)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model",
        default="lfm3-omni-2b",
        help="Model ID (or comma-separated list / 'all'): lfm3-omni-2b, clover-35b-a3b, nemotron-3.5-super-vl, kimi-k3",
    )
    ap.add_argument("--model-dir", type=Path, default=None, help="Optional local path to an already-downloaded HG model folder")
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CLOVER_HG_CACHE", Path.home() / ".cache" / "clover-hg")))
    ap.add_argument("--gpus", default=os.environ.get("CLOVER_CUDA_GPUS", "auto"), help="'auto' (all NVIDIA GPUs), '0,1,2,3', or 'none' for pure CPU")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("CLOVER_GATEWAY_API_KEY", ""))
    ap.add_argument("--rebuild", action="store_true", help="Force recompiling C11 / CUDA shared libraries")
    ap.add_argument("--build-only", action="store_true", help="Compile shared libraries (and optionally verify) without starting HTTP server")
    args = ap.parse_args()

    raw_models = list(MODEL_REGISTRY.keys()) if args.model.strip().lower() == "all" else [m.strip().lower() for m in args.model.split(",") if m.strip()]
    models: list[str] = []
    for rm in raw_models:
        mid = ALIASES.get(rm, rm)
        if mid not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model `{rm}`. Choose from: {', '.join(MODEL_REGISTRY)}")
        if mid not in models:
            models.append(mid)

    src_root = resolve_src_root()
    lib_dir = args.cache_dir / "bin"
    ensure_compiled(src_root, lib_dir, models, args.gpus, force_rebuild=args.rebuild)

    os.environ["CLOVER_NATIVE_LIB_DIR"] = str(lib_dir)
    os.environ["CLOVER_CUDA_GPUS"] = args.gpus
    os.environ["CLOVER_ACTIVE_MODELS"] = ",".join(models)
    os.environ["CLOVER_KIMI_PREFER_NATIVE"] = "1"

    if args.build_only:
        print(f"[clover-hg] Build complete in `{lib_dir}` for models: {', '.join(models)}", flush=True)
        return

    for mid in models:
        mdir = ensure_weights(mid, args.cache_dir, args.model_dir if len(models) == 1 else None)
        env_key = str(MODEL_REGISTRY[mid]["env_var"])
        os.environ[env_key] = str(mdir)
        if mid == "kimi-k3":
            os.environ["CLOVER_KIMI_COLD_DIR"] = str(mdir)
        print(f"[clover-hg] Ready `{mid}` -> `{mdir}` ({MODEL_REGISTRY[mid]['size_note']})", flush=True)

    gw_dir = src_root / "tools" / "hg4-fusion-gateway"
    sys.path.insert(0, str(gw_dir))
    import gateway  # type: ignore[import-not-found]

    gw_argv = ["gateway.py", "--mode", "real", "--host", args.host, "--port", str(args.port)]
    if args.api_key:
        gw_argv.extend(["--api-key", args.api_key])
    print(f"[clover-hg] Serving OpenAI-compatible API at http://{args.host}:{args.port}/v1 (models: {', '.join(models)})", flush=True)
    gateway.main(gw_argv[1:])


if __name__ == "__main__":
    main()
