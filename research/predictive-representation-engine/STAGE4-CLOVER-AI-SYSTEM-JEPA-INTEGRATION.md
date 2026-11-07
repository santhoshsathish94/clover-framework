# Stage 4 — Clover AI System Integration, Selectable JEPA Interfaces & 5-Flow Benchmark

Offered, not imposed. Date: 2026-11-07. Direction owner: Santhosh.  
Worktree: `C:\personal\oss\clover-jepa-wt` (branch `research/jepa-predictive-engine`).  
Raw artifacts:
- [`results/jepa_trunk_training_report.json`](results/jepa_trunk_training_report.json)
- [`results/clover_jepa_flow_integration_results.json`](results/clover_jepa_flow_integration_results.json)

---

## 1. Unified Selectable JEPA Interface (`clover/jepa_engine.py`)

We established a single unified runtime and selector interface in [`clover-ai-system/studio/clover/jepa_engine.py`](../../clover-ai-system/studio/clover/jepa_engine.py) (mirrored in [`clover-ai-system/mobile/jepa_engine.py`](../../clover-ai-system/mobile/jepa_engine.py)) and exposed `jepa_arch` in [`clover-ai-system/studio/clover/server.py`](../../clover-ai-system/studio/clover/server.py) (`Settings.DEFAULTS["jepa_arch"] = "auto"`):

```python
JEPA_ARCHITECTURES = ("auto", "clover_rpre", "ijepa", "audio_jepa", "eb_jepa", "mjepa", "none")
```

### How Clover Chooses the Architecture (`choose_architecture`)
- When `jepa_arch == "auto"` (default), Clover selects the specialized JEPA architecture for each active flow in the turn:
  - **Flow 1 (`visual_s3`)** $\to$ **`ijepa`** (or **`mjepa`** when paired with audio)
  - **Flow 2 (`audio_s3`)** $\to$ **`audio_jepa`** (or **`mjepa`** when paired with visual)
  - **Flow 3 (`multimodal_s2`)** $\to$ **`mjepa`**
  - **Flow 4 (`c2_plan`)** $\to$ **`eb_jepa`** (or **`clover_rpre`** once outcome residuals are active)
  - **Flow 5 (`outcome_growth`)** $\to$ **`clover_rpre`**
- When `jepa_arch` is explicitly set to `"ijepa"`, `"audio_jepa"`, `"eb_jepa"`, `"mjepa"`, `"clover_rpre"`, or `"none"` (via `/api/settings` or `LayerKit.jepa_arch`), Clover forces that architecture across the pipeline so any flow can be benchmarked A/B.

---

## 2. How Each Clover AI System Flow Uses & Improves with JEPA

Executed via [`clover-ai-system/studio/clover/test_jepa_clover_flows.py`](../../clover-ai-system/studio/clover/test_jepa_clover_flows.py) using the trained `CLVRTRK1` binary trunk (`models/jepa-trunk-0`, 102 float32 tensors).

### Flow 1 — `S3` Visual & UI Inspection ([`image_inspector.py`](../../clover-ai-system/studio/clover/image_inspector.py) + `inspect_visual_latent`)
- **Integration**: `image_inspector.inspect_image` calls `jepa_engine.for_base().inspect_visual_latent(im, elements=elements, configured_arch=jepa_arch)`. It scores every detected UI/visual region `[e1..eN]` by `jepa_salience` (combining `I-JEPA` multi-block masked latent prediction surprise with standardized patch contrast) and classifies the visual layout (`ui_controls`, `tabular_chart`, `document_receipt`, `code_terminal`, `diagram_scene`, `natural_photo`).
- **Measured Improvement (`flow1_visual_s3`)**:

| Architecture (`jepa_arch`) | Top-1 Focal UI Control Precision | Scene Classification Accuracy | Encoder Self-Attention FLOPs Ratio | Mean Turn Latency (`ms`) |
|---|---:|---:|---:|---:|
| **`none`** (Pixel/Area Baseline) | $25.00\%$ | $0.00\%$ | $1.0000\times$ | $97.21\text{ ms}$ |
| **`ijepa`** | **$100.00\%$ ($+75.0\text{ pp}$)** | **$100.00\%$** | **$0.5625\times$ ($1.78\times$ reduction)** | $69.72\text{ ms}$ |
| **`mjepa`** | **$100.00\%$** | **$100.00\%$** | **$0.5625\times$** | $68.80\text{ ms}$ |
| **`clover_rpre` / `auto`** | **$100.00\%$** | **$100.00\%$** | **$0.5625\times$** | $69.90\text{ ms}$ |

---

### Flow 2 — `S3/C1` Voice & Audio Stream Inspection ([`omni_native.py`](../../clover-ai-system/studio/clover/omni_native.py) + `inspect_audio_latent`)
- **Integration**: On voice/audio turns (`has_new_audio` in `OmniNative.run_turn`), Clover extracts a $16 \times 16$ 2D log-mel spectrogram, runs `Audio-JEPA` time-frequency band masking with symmetric `CenteredNormMSELoss`, emits a `jepa_audio` layer telemetry event (`acoustic_regime`, `snr_db`, `tf_latent_cos`), and stores the 64-d audio latent cue `kit.audio_cue_64`.
- **Measured Improvement (`flow2_audio_s3`)**:

| Architecture (`jepa_arch`) | Acoustic Regime Accuracy | Masked Time-Frequency Latent Cosine (`tf_latent_cos`) | Symmetric `CenteredNormMSELoss` | Mean Latency (`ms`) |
|---|---:|---:|---:|---:|
| **`none`** | $0.00\%$ | $0.0000$ | — | $0.00\text{ ms}$ |
| **`audio_jepa` / `auto`** | **$100.00\%$** | **$0.9950$** | **$0.0100$** | **$2.09\text{ ms}$** |
| **`mjepa` / `clover_rpre`** | **$100.00\%$** | **$0.9941$** | **$0.0119$** | **$2.11\text{ ms}$** |

---

### Flow 3 — `S2` Multimodal Cross-Modal Grounding & Missing-Modality Recall (`fuse_and_retrieve_multimodal`)
- **Integration**: When a turn has a text/concept query and only one sensory modality (e.g., audio cue without image, or image without audio), `fuse_and_retrieve_multimodal` runs `MJEPA`'s 4-level hierarchical trunk and 6 cross-modal predictors (`a2v`, `v2a`, `s2v`, `s2a`, `v2s`, `a2s`) to impute the missing modality in latent space and rank candidate memories.
- **Measured Improvement (`flow3_multimodal_s2`)**:

| Architecture (`jepa_arch`) | Missing-Modality Recall@1 | Cross-Modal Audio-Visual Cosine (`cross_modal_av_cos`) |
|---|---:|---:|
| **`none`** (Unimodal Query Only) | $90.00\%$ | $0.0000$ |
| **`mjepa`** | **$100.00\%$ ($+10.0\text{ pp}$)** | **$0.1731$** |
| **`clover_rpre` / `auto`** | **$100.00\%$ ($+10.0\text{ pp}$)** | **$0.1682$** |

---

### Flow 4 — Core 2 (`C2`) Energy-Based Latent Trajectory & Tool Planning ([`omni_c2_c3.py`](../../clover-ai-system/studio/clover/omni_c2_c3.py) + [`lfm3_layers.py`](../../clover-ai-system/studio/clover/lfm3_layers.py))
- **Integration**: Inside `OmniC2C3Runtime.c2_plan`, after `HybridLink2048 + Lfm3NativeHybridLink` produces the initial 2048-d plan `r_primary`, Clover calls `jepa_engine.plan_c2_latent(...)`. It uses `EB-JEPA`'s `InverseDynamicsMLP` (`IDM`) to warm-start an action sequence and runs elitist `CEMPlanner` rollouts over horizon $T=4$ to minimize latent energy distance to the goal state and rank `offered_tools`.
- **Measured Improvement (`flow4_c2_latent_planning`)**:

| Architecture (`jepa_arch`) | Mean Latent Energy Before | Mean Latent Energy After | Latent Energy Reduction (`%`) | Latent `vc_std` (Non-Collapse) | Mean Latency (`ms`) |
|---|---:|---:|---:|---:|---:|
| **`none`** (Open-Loop `HybridLink` Only) | $1.2500$ | $1.2500$ | $0.00\%$ | $0.0000$ | $0.00\text{ ms}$ |
| **`eb_jepa` / `auto`** | $0.4150$ | **$0.3831$** | **$10.99\%–11.07\%$** | **$1.0000$** | **$1.26\text{ ms}$** |
| **`clover_rpre`** | $0.4557$ | **$0.3948$** | **$14.82\%$** | **$1.0000$** | **$1.75\text{ ms}$** |

---

### Flow 5 — Closed-Loop `Outcome → Growth` Adaptation ([`omni_native.py`](../../clover-ai-system/studio/clover/omni_native.py) `retry_turn` + `record_and_adapt_outcome`)
- **Integration**: When the user triggers a retry cycle (`retry_turn` in `omni_native.py`) or an outcome is recorded via `LayerKit.jepa_record_outcome`, `CloverJEPARuntime.record_and_adapt_outcome` updates the residual transition weights (`residual.fc1`, `residual.fc2`) subject to the **Baseline-Prototype Anomaly Gate** and **Held-Out Non-Regression Rollback Gate**, and persists the adapted weights to `<base>/growth/jepa_outcome_adapter.npz`.
- **Measured Improvement (`flow5_outcome_growth`)**:
  - Shifted-regime prediction MSE drops by **$17.67\times$ (`0.06610 → 0.00374`)** across Cycles 1–5.
  - Held-out Baseline Regime 0 MSE remains **`0.00000` (`0.0%` regression)**.

---

## 3. GPU Training Pipeline (Ready When You Switch On the GPU)

The pilot `CLVRTRK1` binary trunk (`clover-ai-system/studio/clover/models/jepa-trunk-0/`, 102 tensors, `7.25 MB`) has already been trained on `k3` (`loss: 2.9954 -> 0.1225`, `eval_visual_scene_acc = 100%`, `eval_acoustic_regime_acc = 100%`) and is active in the worktree.

Whenever you switch on the GPU server (`gpu` or Vast.ai), run the following single command to train the full-scale trunk on CUDA and hot-swap `models/jepa-trunk-0/` with zero runtime code changes:

```bash
python research/predictive-representation-engine/scripts/train_clover_jepa_trunks.py \
  --device cuda \
  --scale full \
  --epochs 120 \
  --batch-size 256
```
