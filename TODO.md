# Clover Framework & Clover AI Studio — Master `TODO.md`

Following the canonical Clover Cycle (**`Context → Direction → Execution → Outcome → Growth`** in [AGENTS.md](AGENTS.md)).

---

## 1. Multi-Cycle `AGENTS.md` Clover Cycles & Official Yellow Clover Flower Logo
- [x] **Context:** Diagnosed why `OmniNative.run_turn`, `LFM3Native.run_turn`, `s23_on_device_clover.py`, and `AutonomousCloverEngine` previously stopped after Cycle 1 unless a human manually typed *"try again"*, and why Android 8+ adaptive icons & Store assets showed a green 5-leaf icon (`#10B981` in `ic_launcher_foreground.xml` and `build_msix.ps1`).
- [x] **Execution:**
  - Implemented `verify_turn_outcome(...)` + `AUTO_CYCLE_FIX` in [clover-ai-system/studio/clover/clover_cycle.py](clover-ai-system/studio/clover/clover_cycle.py) and [clover-ai-system/mobile/clover_cycle.py](clover-ai-system/mobile/clover_cycle.py) (checking compiler syntax/AST, sandboxed Python execution, unrecovered tool failures, arithmetic equations, and explicit format/word/sentence/bullet/JSON constraints).
  - Integrated automatic multi-cycle Outcome verification & `LRN-*` Growth recording (`max_clover_cycles = 3`) into [clover-ai-system/studio/clover/omni_native.py](clover-ai-system/studio/clover/omni_native.py), [clover-ai-system/studio/clover/lfm3_native.py](clover-ai-system/studio/clover/lfm3_native.py), [clover-ai-system/mobile/s23_on_device_clover.py](clover-ai-system/mobile/s23_on_device_clover.py), [clover-ai-system/studio/clover/autonomous_clover.py](clover-ai-system/studio/clover/autonomous_clover.py), and live UI badges in [clover-ai-system/studio/clover/web/app.html](clover-ai-system/studio/clover/web/app.html).
  - Replaced green adaptive icon vectors and Store assets with the official 5-petal yellow Clover flower (`#FFF078` → `#F8CF1E` → `#D89606`, stroke `#C68600`, dark navy `#0E1117`) and rebuilt Android `versionCode 14` (`clover-ai-studio-beta.aab`, `clover-ai-studio-beta.apk`) and Windows installers (`CloverAIStudio-Setup.exe`, `.msix`, `.msixupload`).
- [x] **Outcome & Growth:** Verified `ALL CLOVER CYCLE TESTS PASSED` in [clover-ai-system/studio/clover/test_clover_cycles.py](clover-ai-system/studio/clover/test_clover_cycles.py) across the 8x RTX 3090 GPU server, `k3`, and the Windows laptop (`laptop`), and verified live multi-cycle turns on the Windows laptop (`127.0.0.1:8765`).

---

## 2. Upload All Computed Datasets (Especially JEPA) to Hugging Face & Speed Up Installer Downloads
- [x] **Context:**
  - `setup_fetch.py` previously downloaded `141–189` dataset files sequentially on 1 TCP stream and waited `60s × 8` on blocked/slow `huggingface.co` connections before falling back to `HF_MIRROR`.
  - The 8x RTX 3090 trained JEPA artifacts (`jepa/jepa-trunk-0/{index,common,operator,ffn}.bin`, `real_clover_jepa_dataset.npz`, `jepa_outcome_adapter.npz`, `jepa_collective_growth.json`, and training/flow reports) needed to be published into `Santhoshsathish94/clover-ai-studio` (`liquid-lfm3/jepa/`) and included in `manifest.json` / `manifest-android.json`.
- [x] **Execution (Installer Speed & Mirror):**
  - Upgraded [clover-ai-system/studio/clover/setup_fetch.py](clover-ai-system/studio/clover/setup_fetch.py) with sticky `HF_MIRROR` fallback, 8-worker concurrent small-file downloads, 6-stream parallel HTTP `Range` slice downloads (`16 MB` chunks) for large files (`>= 12 MB`), and a `.clover-dataset.json` fast verification stamp (subsequent runs verify `141` files in `< 1 s`).
  - Upgraded [clover-ai-system/studio/clover/server.py](clover-ai-system/studio/clover/server.py) `_serve_hf_mirror` to support bounded `Range: bytes=start-end` requests and serve `jepa/` alongside `hybridagent/`, `bridges/`, `ncm/`, and `trunks/`.
- [x] **Execution (Hugging Face Upload):**
  - Uploaded all 12 computed `liquid-lfm3/jepa/` artifacts (`12,352,379` bytes: `jepa-trunk-0/{common,ffn,index,operator}.bin`, `real_clover_jepa_dataset.npz`, `jepa_outcome_adapter.npz`, `jepa_collective_growth.json`, and training/flow reports) and updated `README.md` to Hugging Face dataset `Santhoshsathish94/clover-ai-studio` (`dc432f32` → `a783d937`), and pinned `a783d9378a9e50c7a0f21bd7e21c0decd334a065` in [clover-ai-system/studio/clover/launch/manifest.json](clover-ai-system/studio/clover/launch/manifest.json) and [clover-ai-system/studio/clover/launch/manifest-android.json](clover-ai-system/studio/clover/launch/manifest-android.json).
  - Verified on the Windows laptop (`laptop`): `setup_fetch.py` downloaded all 12 new `jepa/` files in `10 s`, verified all `141` files on subsequent runs in `1 s`, and loaded `weights_source = "clvrtrk1:C:\\Users\\santh\\CloverAI\\data\\omni\\precomputed\\jepa\\jepa-trunk-0"`.

---

## 3. Port Open Model `Clover-35B-A3B` (`Diffuspace/Clover-35B-A3B-Preview`) to Clover
- [x] **Context (Primary Source Inspection of `Diffuspace/Clover-35B-A3B-Preview` & `Diffuspace/Clover-35B-A3B-Preview-GGUF`):**
  - **License:** `Apache-2.0` (`sha = 4b0402b65e523d0183a14c431955438714cb16ff`).
  - **Architecture:** `CloverMoeModelLM` (`model_type = "clover_moe"`), `35B` total parameters (`69,322,701,696` bytes across `28` safetensor shards, `813` tensors) with `~3B` active parameters per token.
  - **Dimensions:** `num_hidden_layers = 40`, `hidden_size = 2048` (`D_model = 2048`, matching Clover's 2048-d native bridge width), `vocab_size = 248,320`, `max_position_embeddings = 131,072` (`128K` context), `zero_centered = True` `CloverRMSNorm` (`weight = 1 + w`, `eps = 1e-6`).
  - **40-Layer 3:1 Hybrid Mixer Schedule:**
    - `30` **Bidirectional Gated DeltaNet (`BD-GDN`)** `linear_attention` layers (`layers 0, 1, 2, 4, 5, 6, ...`): `16` key heads $\times$ `128` (`2048`), `32` value heads $\times$ `128` (`4096`), `channels = 8192`, `conv1d` kernel width `4` + bidirectional preview `conv_right_weight [8192, 1, 3]` (`bd_gdn_conv_right_weight.safetensors`) and `bd_softmax_{logit_scale, history_bias, history_c} [32]` (`bd_gdn_softmax_scalars.safetensors`).
    - `10` **Gated Full-Attention (`Gated GQA`)** `full_attention` layers (`layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39` 0-indexed, interval `4`): `16` query heads $\times$ `256` with fused sigmoid output gate (`q_proj [8192, 2048]` $\to$ `q [4096]` + `gate [4096]`), `2` KV heads $\times$ `256` (`k_proj`, `v_proj [512, 2048]`), `q_norm` & `k_norm [256]`, `partial_rotary_factor = 0.25` (`64` rotary dims, `rope_theta = 10,000,000`).
  - **256-Expert MoE with Block Coreset Selection + Shared Expert (all 40 layers):**
    - `256` routed SwiGLU experts (`num_experts_per_tok = 8`, `moe_intermediate_size = 512`, `gate_up_proj [256, 1024, 2048]`, `down_proj [256, 2048, 512]`), with block-level expert coreset filtering (`coreset_torch`, `moe_block_coreset_top_p = 0.9`).
    - `1` shared SwiGLU expert (`shared_expert_intermediate_size = 512`) gated by `sigmoid(shared_expert_gate(x))`.
  - **Block Diffusion & Self-Speculative Decoding:**
    - Block size `32`, `<|mask|>` token ID `151669`, `LowConfidence` parallel block denoising (`threshold = 0.95`) + GGUF self-speculative decoding (`Diffuspace/Clover-35B-A3B-Preview-GGUF`: `Q4_0` `19.59 GB`, `Q4_K_M` `23.00 GB`, `Q8_0` `37.86 GB`).
- [x] **Execution:**
  - [x] **Step 3.1 ([porting-models/configs/clover-35b-a3b.json](porting-models/configs/clover-35b-a3b.json) & [porting-models/clover_35b_gpu_backbone.py](porting-models/clover_35b_gpu_backbone.py)):**
        Built the IPv4 HTTP Range safetensors streamer, `CLV35TRK` / `CLV35OP1` / `CLV35BD1` trunk/operator/BD-GDN packer, 4-bit group-64 direct expert packer (`experts.4bit.direct` + `experts.direct`), and GPU-resident differentiable stage model (`Clover35BGPUStageModel`).
  - [x] **Step 3.2 (8x RTX 3090 Stage Precomputation, Pruning & Bit-Exact Verification):**
        Streamed representative BD-GDN (`layer 0`) and Gated GQA (`layer 3`) shards + all 30 BD-GDN preview weights (`bd_gdn_conv_right_weight.safetensors` & `bd_gdn_softmax_scalars.safetensors`) + global embeddings/norm/lm_head on the 8x RTX 3090 GPU server (`51.94 s`), pruned `attention.bin` completely from `trunk-3`, and verified `causal_commit_pruned_vs_raw_cos = 1.0` (`max_abs_diff = 0.0`), `block_diffusion_preview_pruned_vs_raw_cos = 1.0` (`max_abs_diff = 0.0`), and `moe_4bit_vs_bf16_cos = 0.997095` ([porting-models/reports/clover_35b_prune_and_stage_verification.json](porting-models/reports/clover_35b_prune_and_stage_verification.json)).
  - [x] **Step 3.3 (8x RTX 3090 3-Stage `HybridAgent` + 4-Phase `11-Level NCM` + `JEPA` Training for `Clover-35B-A3B`):**
        Trained `Clover35BNativeHybridLink` (`2048 -> 512 -> 2048` & `2048 -> 1024 -> 2048`), `HybridLink2048`, `QwenToClover35BHybridBridge`, `Clover35BComboBridge(2048 -> 384)`, `Clover35BExpertBridge(2048 -> 384)`, `[248320, 384]` `vqvae_wte`/`vqvae_dec`, and `jepa-trunk-0` across all 8 RTX 3090 GPUs (`ntp_loss: 7.7253 -> 0.0032`, `token_accuracy: 100%`, `eval_cos_2048: 0.9812`, `ncm_surface_token_acc: 0.35% -> 98.43%`, [porting-models/reports/clover_35b_port_verification_report.json](porting-models/reports/clover_35b_port_verification_report.json)).
  - [x] **Step 3.4 ([clover-ai-system/src/models/clover-35b-a3b/](clover-ai-system/src/models/clover-35b-a3b/) & Studio Integration):**
        Created [clover-ai-system/src/models/clover-35b-a3b/clover_35b_hybrid_ncm.py](clover-ai-system/src/models/clover-35b-a3b/clover_35b_hybrid_ncm.py), [clover-ai-system/src/models/clover-35b-a3b/verify_clover_35b_clover_integration.py](clover-ai-system/src/models/clover-35b-a3b/verify_clover_35b_clover_integration.py), and [clover-ai-system/src/models/clover-35b-a3b/README.md](clover-ai-system/src/models/clover-35b-a3b/README.md), and registered `"clover-35b-a3b"` in [clover-ai-system/src/models/switchable_model_backend.py](clover-ai-system/src/models/switchable_model_backend.py), [clover-ai-system/studio/clover/k3_host_engine.py](clover-ai-system/studio/clover/k3_host_engine.py), and [clover-ai-system/studio/clover/server.py](clover-ai-system/studio/clover/server.py).
  - [x] **Step 3.5 (Hugging Face Upload of `clover-35b/` & `liquid-lfm3/jepa/`):**
        Uploaded all `300` files (`~4.04 GB`) of `clover-35b/` (`experts/`, `trunks/`, `hybridagent/`, `ncm/`, `bridges/`, `jepa/`), all `12` files (`12.35 MB`) of `liquid-lfm3/jepa/`, and the updated dataset card `README.md` to Hugging Face [`Santhoshsathish94/clover-ai-studio`](https://huggingface.co/datasets/Santhoshsathish94/clover-ai-studio) (commits `dc432f32`, `ce9a0f1d`, `a783d937`).
  - [x] **Step 3.6 (End-to-End Verification across GPU Server, `k3`, and Windows Laptop):**
        Verified `ALL_CLOVER_35B_INTEGRATION_CHECKS_PASSED` on the 8x RTX 3090 GPU server, verified `K3_MODELS` and `test_clover_cycles.py` on `k3`, and verified fast `setup_fetch.py` + live Studio + `test_clover_cycles.py` on the Windows laptop (`laptop`).
