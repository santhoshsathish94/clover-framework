# NVIDIA Nemotron 3.5 Super VL (`120B-A12B`) Model Backend (`clover-ai-system/src/models/nemotron-3.5-super-vl/`)

This folder implements the precomputed trunk (`NEMOTRK1`), composed operator (`NEMO_OP1` / `operator.bin`), trunk pruner, direct routed-expert store (`experts.direct` + `SpecMD Least-Stale + R4 Clean-Drop` slot cache), 4-bit planar group-64 quantization, and native 2-layer Multi-Token Prediction (`MTP`) speculative drafter for **NVIDIA's `Nemotron 3.5 Super VL`** (`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` & `NVFP4`, `124.45B` total / `12.77B` active parameters, `88` single-mixer backbone layers, `D_model = 4096`, `V = 131,072`, `1M` context window):

- **Precomputed Dataset & Operator Builder:** [prepare_nemotron_datasets.py](prepare_nemotron_datasets.py)
- **Trunk Pruner (`69.4%` Mamba-2 trunk reduction, `52.9%` GQA trunk reduction):** [prune_nemotron_trunk.py](prune_nemotron_trunk.py)
- **Precomputed Runtime Engine & SpecMD R4 Expert Slot Cache:** [nemotron_precomputed_engine.py](nemotron_precomputed_engine.py)
- **Native C/OpenMP Engine (`libnemotron_cpu_engine.so`):** [nemotron_cpu_engine.c](nemotron_cpu_engine.c)
- **Native `HybridAgent` (`NemotronNativeHybridLink` $d=4096$) & Native 11-Level `NCM` (`131,072`-Vocab VQ-VAE + Bridges):** [nemotron_hybrid_ncm.py](nemotron_hybrid_ncm.py)
- **Multimodal Vision (`C-RADIO`), Video (`EVS`), Audio/Voice Conversation & Image/Video Generation Engine:** [nemotron_multimodal.py](nemotron_multimodal.py)
- **End-to-End Verification & 7-Layer Clover Switching Suite:** [verify_nemotron_clover_integration.py](verify_nemotron_clover_integration.py)
- **Switchable Layer-1 Registration:** `"nemotron-3.5-super-vl"` in [../switchable_model_backend.py](../switchable_model_backend.py) and `class-nemotron-super-vl-foundation-model-backend` in [../../tools/hybrid-ncm-k3/clover-object-model.py](../../tools/hybrid-ncm-k3/clover-object-model.py).
