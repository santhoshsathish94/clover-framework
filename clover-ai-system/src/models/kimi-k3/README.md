# Kimi K3 Model Backend (`clover-ai-system/src/models/kimi-k3/`)

This folder registers **MoonshotAI's `Kimi K3`** (`moonshotai/Kimi-K3`, `2.78T` total / `104B` active parameters, `93` layers, `D_model = 7168`, `V = 163,840`) as a switchable Layer-1 backbone inside `clover-ai-system/`:

- **CPU & GPU Precomputed Engine (`K3TRK001`, `K3MAPS01` / `experts.direct`, `operator.bin`):**
  - Shared driver & boundary: [../../code/common/whole-model.c](../../code/common/whole-model.c), [../../code/common/backend.h](../../code/common/backend.h)
  - AVX-512 / OpenMP CPU layers: [../../code/cpu/layer0-batch.c](../../code/cpu/layer0-batch.c), [../../code/cpu/layers-1-to-92-batch.c](../../code/cpu/layers-1-to-92-batch.c), [../../code/cpu/trunk-4bit.h](../../code/cpu/trunk-4bit.h), [../../code/cpu/dspark-draft.c](../../code/cpu/dspark-draft.c)
  - CUDA GPU kernels & Rust host-path crates: [../../code/gpu/backend-gpu.cu](../../code/gpu/backend-gpu.cu), [../../tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu](../../tools/hybrid-ncm-k3/k3_gpu_experts_lib.cu), [../../tools/hybrid-ncm-k3/k3_cpu_experts_lib.c](../../tools/hybrid-ncm-k3/k3_cpu_experts_lib.c)
  - Trunk pruner: [../../tools/prune-trunk.py](../../tools/prune-trunk.py)
- **Switchable Layer-1 Registration:** `"kimi-k3"` in [../switchable_model_backend.py](../switchable_model_backend.py) and `class-kimi-k3-foundation-model-backend` in [../../tools/hybrid-ncm-k3/clover-object-model.py](../../tools/hybrid-ncm-k3/clover-object-model.py).
