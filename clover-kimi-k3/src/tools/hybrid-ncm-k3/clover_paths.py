"""Unified Path, Dataset, and Device Discovery (`clover_paths.py`) for Clover Kimi K3.

Enables running the full 7-layer Clover system (`CloverK3AgentEngine`, `K3FastChatEngine`,
`K3MultimodalChatEngine`, `k3_chat_server.py`, and all 6 benchmark suites) on ANY GPU server
or CPU server without hardcoded machine-specific paths:
  - Automatically discovers `ENGINE_DIR` (`clover-kimi-k3/`), `DATASET_DIR` (`computed/`),
    `OPEN_WEIGHTS_DIR` (`kimi-k3-open-weights/`), `CKPT_DIR` (`hybrid-ncm-k3/`), and `WMT14_DIR`.
  - Automatically configures `sys.path` so `hybrid_agent`, `k3_vision`, `models`, `tokenization`,
    `utils`, `evaluation`, and `ncm_stage2` import cleanly from `clover-kimi-k3/src/tools/hybrid-ncm-k3`.
  - Automatically selects CUDA devices on GPU servers and `"cpu"` + `libk3_cpu_experts.so` on CPU servers.
"""
from __future__ import annotations

import os
import subprocess
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.environ.get("ENGINE_DIR") or os.path.abspath(os.path.join(TOOLS_DIR, "../../.."))
VISION_DIR = os.path.join(TOOLS_DIR, "k3_vision")


def setup_clover_sys_path() -> str:
    """Configures `HF_HOME`, `NCM_REPO`, `NCM_ART`, and `sys.path` portably across `gpu`, `k3`, and fresh servers."""
    if "HF_HOME" not in os.environ:
        if os.path.isdir("/home/usergpu/plan2text/data/.cache"):
            os.environ["HF_HOME"] = "/home/usergpu/plan2text/data/.cache"
        elif os.path.isdir("/root/hf"):
            os.environ["HF_HOME"] = "/root/hf"
    default_repo = "/home/usergpu/ncm/repo" if os.path.isdir("/home/usergpu/ncm/repo") else TOOLS_DIR
    os.environ.setdefault("NCM_REPO", default_repo)
    if not os.path.isdir(os.environ["NCM_REPO"]):
        os.environ["NCM_REPO"] = TOOLS_DIR
    default_art = (
        "/home/usergpu/ncm/artifacts/tinystories"
        if os.path.isdir("/home/usergpu/ncm/artifacts/tinystories")
        else ("/root/ncm-cpu/artifacts/tinystories" if os.path.isdir("/root/ncm-cpu/artifacts/tinystories") else TOOLS_DIR)
    )
    os.environ.setdefault("NCM_ART", default_art)
    for p in (VISION_DIR, TOOLS_DIR, "/home/usergpu/merge_lf/k3_vision", "/home/usergpu/merge_lf", "/home/usergpu/plan2text", "/home/usergpu/ncm/repo"):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return os.environ["NCM_REPO"]


def find_dataset_dir() -> str:
    """Finds the `computed` Kimi K3 dataset directory containing `trunk-0..92` and `expert-1..92`."""
    for cand in (
        os.environ.get("CLOVER_DATASET_DIR"),
        os.environ.get("DATASET_DIR"),
        os.path.join(ENGINE_DIR, "src/datasets/computed"),
        "/opt/clover-k3/clover-kimi-k3/src/datasets/computed",
        "/opt/clover-k3/datasets/computed",
    ):
        if cand and os.path.isdir(cand) and (
            os.path.isdir(os.path.join(cand, "trunk-0")) or os.path.isdir(os.path.join(cand, "expert-1"))
        ):
            return cand
    return os.path.join(ENGINE_DIR, "src/datasets/computed")


def find_open_weights_dir() -> str:
    """Finds the `kimi-k3-open-weights` directory containing `model-00094-of-000096.safetensors` and `tiktoken.model`."""
    for cand in (
        os.environ.get("CLOVER_OPEN_WEIGHTS_DIR"),
        os.environ.get("OPEN_WEIGHTS_DIR"),
        os.path.join(ENGINE_DIR, "src/datasets/third-party/kimi-k3-open-weights"),
        "/opt/clover-k3/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights",
        "/opt/clover-k3/datasets/third-party/kimi-k3-open-weights",
    ):
        if cand and os.path.isdir(cand) and (
            os.path.exists(os.path.join(cand, "tiktoken.model"))
            or os.path.exists(os.path.join(cand, "model-00094-of-000096.safetensors"))
        ):
            return cand
    return os.path.join(ENGINE_DIR, "src/datasets/third-party/kimi-k3-open-weights")


def find_open_weights_shard94() -> str:
    return os.path.join(find_open_weights_dir(), "model-00094-of-000096.safetensors")


def find_hybrid_ncm_ckpt_dir() -> str:
    """Finds the directory containing the trained Clover K3 + HybridAgent + NCM checkpoints
    (`e12_clover_k3_true_ha_ncm.pt`, `e11_clover_k3_native_ha_ncm.pt`, `k3_vqvae_adapted.pt`, `e6_last.pt`, etc.)."""
    ds_dir = find_dataset_dir()
    for cand in (
        os.environ.get("CLOVER_CKPT_DIR"),
        os.path.join(ENGINE_DIR, "src/datasets/hybrid-ncm-k3"),
        os.path.join(ds_dir, "hybrid-ncm-k3"),
        "/opt/clover-k3/clover-kimi-k3/src/datasets/hybrid-ncm-k3",
        "/opt/clover-k3/datasets/hybrid-ncm-k3",
        "/opt/clover-k3/datasets/computed/hybrid-ncm-k3",
        "/home/usergpu/merge_lf/out/e6_k3_native",
        os.path.join(TOOLS_DIR, "out/e6_k3_native"),
    ):
        if cand and os.path.isdir(cand) and any(
            os.path.exists(os.path.join(cand, fn))
            for fn in (
                "e12_clover_k3_true_ha_ncm.pt",
                "e11_clover_k3_native_ha_ncm.pt",
                "k3_vqvae_adapted.pt",
                "e8_k3_direct.pt",
            )
        ):
            return cand
    return "/home/usergpu/merge_lf/out/e6_k3_native" if os.path.isdir("/home/usergpu/merge_lf/out/e6_k3_native") else os.path.join(ENGINE_DIR, "src/datasets/hybrid-ncm-k3")


def find_wmt14_art_dir() -> str:
    """Finds the directory containing `wmt14/tokenizer.pt` and `wmt14/config.yaml`."""
    ckpt_dir = find_hybrid_ncm_ckpt_dir()
    for cand in (
        os.environ.get("CLOVER_WMT14_DIR"),
        os.path.join(ckpt_dir, "wmt14"),
        os.path.join(ENGINE_DIR, "src/datasets/hybrid-ncm-k3/wmt14"),
        "/opt/clover-k3/clover-kimi-k3/src/datasets/hybrid-ncm-k3/wmt14",
        "/root/ncm-cpu/artifacts/wmt14",
        "/home/usergpu/ncm/artifacts/wmt14",
    ):
        if cand and os.path.isdir(cand) and os.path.exists(os.path.join(cand, "tokenizer.pt")):
            return cand
    return os.path.join(ckpt_dir, "wmt14")


def find_ncm_eval_data_dir() -> str:
    """Finds the directory containing `wmt14/reference_translations.pickle`, `xsum/passages.pickle`, and `sudoku/easy/puzzles.pickle`."""
    ckpt_dir = find_hybrid_ncm_ckpt_dir()
    for cand in (
        os.environ.get("CLOVER_NCM_DATA_DIR"),
        os.path.join(ckpt_dir, "data"),
        os.path.join(ENGINE_DIR, "src/datasets/hybrid-ncm-k3/data"),
        "/opt/clover-k3/clover-kimi-k3/src/datasets/hybrid-ncm-k3/data",
        "/root/ncm-cpu/data",
        "/home/usergpu/ncm/data",
    ):
        if cand and os.path.isdir(cand):
            return cand
    return os.path.join(ckpt_dir, "data")


def find_wmt14_config_yaml() -> str:
    """Finds `config/eval/wmt14.yaml` relative to `TOOLS_DIR` or `/home/usergpu/ncm/repo`."""
    for cand in (
        os.path.join(TOOLS_DIR, "config/eval/wmt14.yaml"),
        "/home/usergpu/ncm/repo/config/eval/wmt14.yaml",
    ):
        if os.path.exists(cand):
            return cand
    return os.path.join(TOOLS_DIR, "config/eval/wmt14.yaml")


def resolve_device(requested_dev: str = "cuda:0") -> str:
    """Returns `requested_dev` if CUDA is available and the requested GPU index exists;
    otherwise automatically falls back to `'cuda:0'` (on single-GPU machines) or `'cpu'` (on CPU servers like `k3`)."""
    if os.environ.get("CLOVER_FORCE_CPU", "0") == "1":
        return "cpu"
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        n_gpus = torch.cuda.device_count()
        if requested_dev.startswith("cuda:"):
            idx = int(requested_dev.split(":")[1])
            if idx >= n_gpus:
                return f"cuda:{max(0, n_gpus - 1)}"
        return requested_dev
    except Exception:
        return "cpu"


def cuda_sync(*devs) -> None:
    """Synchronizes any CUDA devices in `devs`; safe no-op on CPU servers."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for d in devs:
            if d is None:
                continue
            d_str = str(d)
            if d_str.startswith("cuda"):
                torch.cuda.synchronize(d)
    except Exception:
        pass


def ensure_cpu_experts_lib() -> str:
    """Compiles and returns the path to `libk3_cpu_experts.so` for CPU servers (`k3`),
    linking `k3_cpu_experts_lib.c` with `layer0-batch.c`, `layers-1-to-92-batch.c` (`lbk_*`, `lbm_*`),
    `cpu-memory.c`, and optional Rust `libslotcache.so` / `libsupply.so`."""
    so_path = os.environ.get("CLOVER_CPU_EXPERTS_SO") or os.path.join(TOOLS_DIR, "libk3_cpu_experts.so")
    c_src = os.path.join(TOOLS_DIR, "k3_cpu_experts_lib.c")
    if os.path.exists(so_path) and os.path.exists(c_src):
        if os.path.getmtime(so_path) >= os.path.getmtime(c_src):
            return so_path

    # Locate `src/code/common` and `src/code/cpu`
    code_dir = os.path.join(ENGINE_DIR, "src/code")
    if not os.path.isdir(code_dir) and os.path.isdir("/opt/clover-k3/clover-kimi-k3/src/code"):
        code_dir = "/opt/clover-k3/clover-kimi-k3/src/code"
    cmn_dir = os.path.join(code_dir, "common")
    cpu_dir = os.path.join(code_dir, "cpu")
    sc_inc = os.path.join(ENGINE_DIR, "src/rust/slotcache/include")
    if not os.path.isdir(sc_inc) and os.path.isdir("/opt/clover-k3/clover-kimi-k3/src/rust/slotcache/include"):
        sc_inc = "/opt/clover-k3/clover-kimi-k3/src/rust/slotcache/include"

    build_dir = os.path.join(TOOLS_DIR, ".build_cpu_experts")
    os.makedirs(build_dir, exist_ok=True)

    # Also build Rust `libslotcache.so` if `cargo` is available and `libslotcache.so` is not yet built
    sc_crate = os.path.join(ENGINE_DIR, "src/rust/slotcache")
    sc_so = os.path.join(TOOLS_DIR, "libslotcache.so")
    if not os.path.exists(sc_so) and os.path.isdir(sc_crate):
        cargo_bin = "cargo"
        for cb in ("/root/.cargo/bin/cargo", os.path.expanduser("~/.cargo/bin/cargo")):
            if os.path.exists(cb):
                cargo_bin = cb
                break
        try:
            subprocess.run([cargo_bin, "build", "--release"], cwd=sc_crate, check=True)
            built_sc = os.path.join(sc_crate, "target/release/libslotcache.so")
            if os.path.exists(built_sc):
                import shutil
                shutil.copy2(built_sc, sc_so)
        except Exception:
            pass

    build_script = f"""
set -e
cd "{build_dir}"
CMN="{cmn_dir}"
CPU="{cpu_dir}"
INC="-I$CMN -I$CPU -I{sc_inc}"
F="-std=gnu11 -march=native -ffp-contract=off -fno-fast-math -fopenmp -fPIC -Wall -Wno-unused-function"
sed -E 's/^#define TRANSFORMER_LAYER 2$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\\(TRANSFORMER_LAYER - 1\\)/(2 - 1)/; s/"trunk-2\\/operator.bin"/lb_name(0)/; s/"trunk-2"/lb_name(1)/; s/"expert-2"/lb_name(2)/; s/expert_store_open\\(expert_store_directory, 2\\)/expert_store_open(expert_store_directory, lb_layer_no)/' "$CPU/layer-kda.c" > template-kda.c
sed -E 's/^#define TRANSFORMER_LAYER 3$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\\(TRANSFORMER_LAYER - 1\\)/(3 - 1)/; s/"trunk-3\\/operator.bin"/lb_name(0)/; s/"trunk-3"/lb_name(1)/; s/"expert-3"/lb_name(2)/; s/expert_store_open\\(expert_store_directory, 3\\)/expert_store_open(expert_store_directory, lb_layer_no)/' "$CPU/layer-mla.c" > template-mla.c
gcc $F -DPIPELINE_ENTRY -O2 -funroll-loops $INC -Dmain=layer0_main_unused -c "$CPU/layer0-batch.c" -o layer0.o
gcc $F -DLB_ENTRY -DLB_PFX=lbk -DLB_TEMPLATE='"template-kda.c"' -O3 $INC -I. -Dmain=kda_main_unused -c "$CPU/layers-1-to-92-batch.c" -o layers-kda.o
gcc $F -DLB_ENTRY -DLB_PFX=lbm -DLB_TEMPLATE='"template-mla.c"' -O3 $INC -I. -Dmain=mla_main_unused -c "$CPU/layers-1-to-92-batch.c" -o layers-mla.o
gcc $F -O2 $INC -c "$CPU/cpu-memory.c" -o cpu-memory.o
objcopy --keep-global-symbol=stage2_setup --keep-global-symbol=stage2_run --keep-global-symbol=stage2_run_from --keep-global-symbol=stage2_step --keep-global-symbol=stage2_stepk --keep-global-symbol=stage2_keep_positions --keep-global-symbol=stage2_use_position layer0.o
for p in lbk:layers-kda.o lbm:layers-mla.o; do
  pre=${{p%%:*}}; obj=${{p##*:}}
  objcopy --keep-global-symbol=${{pre}}_set_buffer --keep-global-symbol=${{pre}}_init --keep-global-symbol=${{pre}}_open --keep-global-symbol=${{pre}}_run --keep-global-symbol=${{pre}}_run_from --keep-global-symbol=${{pre}}_step --keep-global-symbol=${{pre}}_stepk --keep-global-symbol=${{pre}}_keep_positions --keep-global-symbol=${{pre}}_use_position --keep-global-symbol=${{pre}}_set_scratch --keep-global-symbol=${{pre}}_distinct --keep-global-symbol=${{pre}}_route_log --keep-global-symbol=${{pre}}_routes --keep-global-symbol=${{pre}}_hits --keep-global-symbol=${{pre}}_set_slots --keep-global-symbol=${{pre}}_cache_stats --keep-global-symbol=${{pre}}_guess_stats --keep-global-symbol=${{pre}}_laps "$obj"
done
gcc $F -O3 $INC -c "{c_src}" -o k3_cpu_experts_lib.o
gcc -shared -fPIC -fopenmp k3_cpu_experts_lib.o layer0.o layers-kda.o layers-mla.o cpu-memory.o -lm -lpthread -ldl -o "{so_path}"
"""
    subprocess.run(["bash", "-c", build_script], check=True)
    return so_path
