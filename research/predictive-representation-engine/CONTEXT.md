# CONTEXT — Clover Reusable Predictive Representation Engine (`RPRE`)

Offered, not imposed. Last verified: 2026-11-07. Direction owner: Santhosh.  
Branch: `research/jepa-predictive-engine` | Worktree: `C:\personal\oss\clover-jepa-wt`.

## 1. Purpose & Guiding Architectural Question

This directory contains the source-level audit, unified implementation (`rpre/`), unit/integration test suite (`tests/test_rpre.py`), and multi-seed experimental verification (`scripts/run_all_jepa_experiments.py`, `results/jepa_investigation_results.json`) answering the guiding architectural question across the four core JEPA papers and repositories:

> **Can we build a reusable predictive representation engine that processes different kinds of information more efficiently, learns their underlying relationships, and improves through observed outcomes?**

## 2. Verified Upstream Sources Read in Full

1. **`I-JEPA`** (`facebookresearch/ijepa` @ `52c1ae95d05f743e000e8f10a1f3a79b10cff048`)
   - `src/masks/multiblock.py`, `src/masks/utils.py`, `src/models/vision_transformer.py`, `src/train.py`.
2. **`Audio-JEPA`** (`LudovicTuncay/Audio-JEPA` @ `ddd97ee9b88c00572f59bf2a00eb42120e6a4dea`)
   - `model.py`, `loss.py` (identified and fixed asymmetric centering bug at lines 42–46), `masks.py`, `train_fabric.py`.
3. **`EB-JEPA`** (`facebookresearch/eb_jepa` @ `966e61e9285b3a876f49b9774e9720d9a99a7925`)
   - `eb_jepa/jepa.py`, `eb_jepa/losses.py`, `eb_jepa/planning.py`, `examples/ac_video_jepa/main.py`, `examples/image_jepa/main.py`.
4. **`MJEPA`** (`facebookresearch/MJEPA` @ `82f3472219f2f58672999ed654314b34ad590c07`)
   - `model_mjepa.py`, `criterion.py`, `train_mjepa.py`, `extract_features_m.py`.

## 3. Directory Layout

- [`STAGE1-FOUR-PAPERS-AND-REPOS-AUDIT.md`](STAGE1-FOUR-PAPERS-AND-REPOS-AUDIT.md) — Line-referenced source audit of all four repositories.
- [`STAGE2-REUSABLE-PREDICTIVE-ENGINE-ARCHITECTURE.md`](STAGE2-REUSABLE-PREDICTIVE-ENGINE-ARCHITECTURE.md) — Unified mathematical and systems architecture of `RPRE`.
- [`STAGE3-EXPERIMENTS-AND-RESULTS.md`](STAGE3-EXPERIMENTS-AND-RESULTS.md) — Complete multi-seed experimental results (`E1`–`E5`) and ablations.
- [`STAGE4-CLOVER-AI-SYSTEM-JEPA-INTEGRATION.md`](STAGE4-CLOVER-AI-SYSTEM-JEPA-INTEGRATION.md) — Integration into the live Clover AI System (`clover/jepa_engine.py`, `models/jepa-trunk-0`), 5-flow benchmarks, and GPU training pipeline.
- [`STAGE5-AUTONOMOUS-CLOVER-CYCLE-AND-SELF-TRAINING.md`](STAGE5-AUTONOMOUS-CLOVER-CYCLE-AND-SELF-TRAINING.md) — End-to-end autonomous Clover Cycle problem solving, JEPA epistemic surprise problem discovery, and verified self-training (`clover/autonomous_clover.py`).
- [`rpre/`](rpre/) — Reusable PyTorch package:
  - [`rpre/masks.py`](rpre/masks.py) — `apply_masks`, `sample_multiblock_2d_masks`, `sample_audio_tf_masks`, `sample_random_masks`.
  - [`rpre/losses.py`](rpre/losses.py) — `CenteredNormMSELoss`, `HingeStdLoss`, `CovarianceLoss`, `VCLoss`, `TemporalSimilarityLoss`, `InverseDynamicsLoss`, `VC_IDM_Sim_Regularizer`, `BCSLoss`, `MJEPA9PathLoss`.
  - [`rpre/engine.py`](rpre/engine.py) — `VisualPatchAdapter`, `AudioSpectrogramAdapter`, `StateVectorAdapter`, `SharedHierarchicalEncoder`, `LatentTokenPredictor`, `CrossModalPredictor`, `ActionConditionedPredictor`, `OutcomeResidualAdapter`, `ReusablePredictiveEngine`.
  - [`rpre/planner.py`](rpre/planner.py) — `ReprTargetDistObjective`, `CEMPlanner`, `MPPIPlanner`, `CloverCycleAgent`.
- [`tests/test_rpre.py`](tests/test_rpre.py) — 15 unit and integration checks covering all modules and findings.
- [`scripts/run_all_jepa_experiments.py`](scripts/run_all_jepa_experiments.py) — Reproducible multi-seed (`[101, 202, 303]`) experiment suite (`E1`–`E5`).
- [`scripts/prepare_real_jepa_dataset.py`](scripts/prepare_real_jepa_dataset.py) — Extracts real visual, 16 kHz audio WAV, and `LFM3-Omni` (`2048-d`) / `VQ-VAE` (`384-d`) embeddings for GPU training.
- [`scripts/train_clover_jepa_trunks.py`](scripts/train_clover_jepa_trunks.py) — 8x RTX 3090 GPU / CPU PyTorch trainer & `CLVRTRK1` binary trunk exporter (`models/jepa-trunk-0`).
- [`scripts/run_autonomous_clover_experiment.py`](scripts/run_autonomous_clover_experiment.py) — Live `LFM3-Omni-2B` + `C1–C3` + `JEPA/RPRE` + `ETH` autonomous problem solving, problem synthesis, and self-training experiment.
- [`results/jepa_investigation_results.json`](results/jepa_investigation_results.json), [`results/jepa_trunk_training_report.json`](results/jepa_trunk_training_report.json), [`results/clover_jepa_flow_integration_results.json`](results/clover_jepa_flow_integration_results.json), [`results/autonomous_clover_self_training_results.json`](results/autonomous_clover_self_training_results.json) — Complete JSON outputs.

## 4. Verification & GPU Training Commands

```bash
python tests/test_rpre.py
python scripts/run_all_jepa_experiments.py
python scripts/prepare_real_jepa_dataset.py
python scripts/train_clover_jepa_trunks.py --device auto --scale full --epochs 240 --batch-size 256
python ../../clover-ai-system/studio/clover/test_jepa_clover_flows.py
python scripts/run_autonomous_clover_experiment.py
```

## 5. Stage 6 — Live S23 Phone Verification, System Senses (`S3`), Debug Logging & Continuous 8x RTX 3090 Collective Self-Training

- **Intended Outcome**: Merge `research/jepa-predictive-engine` to `main`, run continuous multi-GPU self-training on the 8x RTX 3090 server (`136.36.169.232:40456`) syncing verified updates to `https://api.cloverframework.com`, replace in-app window self-capture with OS System Senses (`Screenshot`, `Screen Recorder`, `Camera`, `Voice Recorder / Microphone`), expose JEPA architecture selection + Autonomous Clover Cycle runner + structured Debug Logs in the UI, and verify everything live on the physical Samsung Galaxy S23 (`SM-S911B`).
- **What Reality Showed on the S23 & What Was Fixed**:
  1. **In-Process Android CSI Import (`csi_child.py`)**: `importlib.import_module("csi.analyze")` returned the `analyze` function exported by `csi/__init__.py` rather than the submodule; fixed by resolving `sys.modules[analyze.__module__]`.
  2. **Adreno OpenCL Context Allocation on 6.9 GB Phone (`hardware.py`, `omni_native.py`, `llamacpp_engine.py`)**: `ctx=16384` + `-b 2048` exceeded Adreno OpenCL allocation on a 6.9 GB device when WebView + CSI were active; fixed by selecting `ctx=4096`, `ubatch=256` on Android devices under 8.5 GB RAM so `LFM3-Omni-2B` loads reliably on `QUALCOMM Adreno(TM)` GPU alongside CSI and C2/C3.
  3. **Android Subprocess Fork in `autonomous_clover.py` & `generative-code-compiler.py`**: Calling `subprocess.run([sys.executable, "-c", ...])` inside Android Chaquopy forked `app_process64` while holding ~2.5 GB of Adreno OpenCL memory, triggering Android `lmkd` (`SIGKILL`). Fixed by using an in-process sandbox on Android (`hasattr(sys, "getandroidapilevel")`).
  4. **System Senses (`device_actions.py`, `MainActivity.java`, `web/app.html`)**: Removed the dedicated self-window screenshot button and integrated a **System Senses (`S3`)** menu under `+` (`System Screenshot`, `System Screen Recorder`, `System Camera`, `System Microphone`) that talks to OS system apps via Android intents / Windows URI schemes (`ms-screenclip:`, `MediaStore`) and analyzes captured images, videos, and audio with `I-JEPA`, `Audio-JEPA`, and `LFM3-Omni-2B`.
  5. **Continuous 8x RTX 3090 Self-Training & Account-Agnostic OIDC Sync (`scripts/run_gpu_collective_self_trainer.py`)**: Running detached on `136.36.169.232:40456` across all 8 RTX 3090 GPUs (`val_loss: 1.7387 -> 0.1450`, `novel_heldout_c1_acc: 1.0`, `cycle_updates: 63,274+`), contributing via 3 independent pseudonymous OIDC worker identities (`contrib_*`) and publishing Ed25519-signed releases (`v1.0.321+`) on `https://api.cloverframework.com` that any user account can verify and sync.
