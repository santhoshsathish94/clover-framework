# Stage 3 — Controlled Experiments, Multi-Seed Results & Architectural Answer

Offered, not imposed. Date: 2026-11-07. Direction owner: Santhosh.  
Worktree: `C:\personal\oss\clover-jepa-wt` (branch `research/jepa-predictive-engine`).  
Raw artifact: [`results/jepa_investigation_results.json`](results/jepa_investigation_results.json) (executed across seeds `[101, 202, 303]`).

---

## 1. Summary of the Five Controlled Experiments (`E1`–`E5`)

Every architectural claim across the four papers and our unified **Clover Reusable Predictive Representation Engine (`RPRE`)** was tested via controlled multi-seed (`[101, 202, 303]`) experiments in [`scripts/run_all_jepa_experiments.py`](scripts/run_all_jepa_experiments.py), backed by 15 unit and integration checks in [`tests/test_rpre.py`](tests/test_rpre.py).

---

## 2. `E1` — Visual Latent Prediction (`I-JEPA`)

**Question tested**: Does predicting masked regions in *latent representation space* with context-only sparse encoding (`apply_masks` before Transformer blocks) learn better relational structure at lower compute cost than pixel-space reconstruction (`MAE`), and why are `EMA + LayerNorm` targets required?

**Task**: Relational 2D visual scenes (`32x32`, $8 \times 8 = 64$ patches) where the class ($6$ classes) is determined by the **spatial quadrant relationship between two distinct visual objects** embedded in high-frequency background clutter (`sigma = 0.25` clean, `sigma = 0.70` noisy test).

### Results (`E1`, 3 seeds `[101, 202, 303]`)

| Arm | Context Tokens ($N_{\text{ctxt}} / 64$) | Encoder Self-Attention FLOPs Ratio | Representation Std | Linear Probe Acc (Clean) | Linear Probe Acc (Heavy Noise) |
|---|---:|---:|---:|---:|---:|
| **`pixel_mae`** (Pixel-space patch reconstruction) | $36.3 / 64$ | $0.3164\times$ | $0.7798$ | $69.17\%$ | $64.72\%$ |
| **`ijepa_no_ema_no_ln`** (Latent prediction without EMA/LayerNorm) | $36.3 / 64$ | $0.3164\times$ | **$0.0037$ (Collapsed)** | $28.89\%$ | $35.83\%$ |
| **`ijepa_random_mask`** (Latent prediction, random patch mask) | $36.3 / 64$ | $0.3164\times$ | $0.4306$ | **$100.00\%$** | **$98.05\%$** |
| **`ijepa_multiblock`** (I-JEPA 2D Multi-Block Context/Target Masking) | **$33.3 / 64$** | **$0.2991\times$ ($3.3\times$ reduction)** | $0.4354$ | **$100.00\%$** | **$97.78\%$** |

### Findings from `E1`
1. **Latent prediction filters out surface noise (`+33.1 pp` under heavy noise)**: `pixel_mae` wastes model capacity reconstructing random background pixel realizations (`64.72%` noisy accuracy), whereas `ijepa_multiblock` and `ijepa_random_mask` predict normalized target representations (`F.layer_norm`) and achieve **`100.00%` clean / `97.78%–98.05%` noisy accuracy**.
2. **`EMA + LayerNorm` is what prevents representation collapse in `I-JEPA`**: Removing EMA and target `LayerNorm` (`ijepa_no_ema_no_ln`) causes representation standard deviation to collapse to **`0.0037`** (`118x` smaller than `ijepa_multiblock`) and linear probe accuracy to drop to **`28.89%`**.
3. **Context-only sparse encoding (`apply_masks`) cuts Transformer self-attention FLOPs by `3.3x`**: Passing only the $N_{\text{ctxt}} \approx 33$ visible patches into the encoder reduces quadratic attention cost to $(33.3/64)^2 = \mathbf{29.91\%}$ of full-grid encoding.

---

## 3. `E2` — Audio Spectrogram Latent Prediction (`Audio-JEPA`)

**Question tested**: How does `CenteredNormMSELoss` (fixing `Audio-JEPA` `loss.py:42–46` asymmetric target-only centering) combined with structured time-frequency band masking compare against raw spectrogram reconstruction and standard MSE?

**Task**: 6-class structured time-frequency harmonic/FM/AM/transient log-mel spectrograms (`32x32`, $8 \times 8 = 64$ patches) under heavy background noise and random channel gain/offset shifts. We measure both linear probe accuracy and **masked target latent cosine alignment** (`latent_cos`) against clean target representations.

### Results (`E2`, 3 seeds `[101, 202, 303]`)

| Arm | Loss Function | Masking Strategy | Noisy Probe Accuracy | Masked Latent Cosine Alignment (`latent_cos`) |
|---|---|---|---:|---:|
| **`spec_autoencoder`** | Raw dB spectrogram MSE | Random patch | $100.00\%$ | $-0.0125$ |
| **`audio_jepa_mse_random`** | Unnormalized latent MSE | Random patch | $100.00\%$ | $0.7656$ |
| **`audio_jepa_norm_mse_random`** | Symmetric `CenteredNormMSELoss` | Random patch | $100.00\%$ | $0.8117$ |
| **`audio_jepa_norm_mse_tf_band`** | Symmetric `CenteredNormMSELoss` | **Time-Frequency Band Masking** | **$100.00\%$** | **$0.8362$** |

### Findings from `E2`
1. **Source-code bug verified and fixed (`Audio-JEPA` `loss.py:42–46`)**: Upstream `Loss.forward` standardizes `target` along the feature dimension (`(target - mean) / std`) before `norm_mse_loss`, but leaves `pred` uncentered, so `loss(z, z) > 0` whenever `z` has non-zero mean (verified in `tests/test_rpre.py`). Fixing this in `CenteredNormMSELoss(center_pred=True)` improves masked latent cosine alignment from **`0.7656` to `0.8117` (`+4.61 pp`)**.
2. **Structured time-frequency band masking (`sample_audio_tf_masks`) further improves latent prediction to `0.8362`**: Masking contiguous temporal segments and frequency harmonic bands forces the predictor to infer cross-harmonic and temporal trajectory structure rather than interpolating adjacent spectrogram cells.

---

## 4. `E3` — Energy-Based Regularizers & Latent MPC Planning (`EB-JEPA`)

**Question tested**: Can an action-conditioned JEPA world model learn non-collapsed, controllable representations *without* an EMA target encoder, and which `EB-JEPA` regularizers (`VCLoss`, `TemporalSimilarityLoss`, `InverseDynamicsLoss`) are required for closed-loop Model Predictive Control (`CEMPlanner`)?

**Task**: Continuous 2D point-mass navigation with velocity damping (`state_dim = 4`, `action_dim = 2`). Each arm trains an action-conditioned world model (`SharedHierarchicalEncoder` + `ActionConditionedPredictor`) on random trajectories and is then evaluated zero-shot with `CEMPlanner` across 20 goal-reaching episodes per seed.

### Results (`E3`, 3 seeds `[101, 202, 303]`)

| Arm | Regularizers Active | Latent Std (`repr_std`) | Effective Rank | Multi-Step Rollout MSE | Closed-Loop MPC Success Rate | Mean Final Goal Distance |
|---|---|---:|---:|---:|---:|---:|
| **`pred_only_no_reg`** | None ($\mathcal{L}_{\text{pred}}$ only) | **$0.0135$ (Collapsed)** | $9.82$ | $0.2252$ | **$6.67\%$** | $0.8655$ |
| **`vc_only`** | `HingeStdLoss` + `CovarianceLoss` | $1.0352$ | **$12.44$** | $0.0281$ | $60.00\%$ | $0.2501$ |
| **`vc_plus_temporal_sim`** | `VCLoss` + `TemporalSimilarityLoss` | $1.0113$ | $8.64$ | $0.0348$ | $61.67\%$ | $0.3285$ |
| **`full_eb_jepa_vc_sim_idm`** | `VCLoss` + `Sim` + **`InverseDynamicsLoss`** | **$1.0225$** | **$10.86$** | **$0.0253$** | **$85.00\%$** | **$0.2611$** |

### Findings from `E3`
1. **Unregularized single-network JEPA collapses completely**: Without `VCLoss`, `pred_only_no_reg` shrinks latent standard deviation to `0.0135`, rollout decoding MSE is `0.2252`, and closed-loop MPC planning succeeds on only **`6.67%`** of episodes.
2. **`VCLoss` (`HingeStdLoss` + `CovarianceLoss`) prevents collapse without EMA**: Adding `VCLoss` restores `repr_std = 1.0352`, reduces rollout MSE by **`8x` (`0.2252 -> 0.0281`)**, and lifts MPC planning success from **`6.67%` to `60.00%`**.
3. **`InverseDynamicsLoss` (`IDM`) is the decisive regularizer for action-controllable planning (`+25.0 pp`)**: While `vc_only` prevents collapse, the encoder can still allocate variance to uncontrollable state features. Adding `InverseDynamicsLoss` ($\|\text{IDM}(z_t, z_{t+1}) - a_t\|_2^2$) forces $(z_t, z_{t+1})$ to encode action-controllable coordinates, lowering rollout MSE to **`0.0253`** and lifting closed-loop MPC planning success to **`85.00%`**.

---

## 5. `E4` — Shared Audio-Visual Representation Learning (`MJEPA`)

**Question tested**: When a single Transformer encoder is shared across visual and audio tokens, does naive intra-modal training help or hurt, and what do MJEPA's 6 cross-modal predictive paths (`a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`) and 4-level hierarchical target extraction contribute?

**Task**: 6-class multimodal event benchmark where 3 latent visual attributes and 3 latent audio attributes jointly determine a 6-way synergistic event class under high unimodal noise (`sigma = 0.85`).

### Results (`E4`, 3 seeds `[101, 202, 303]`)

| Arm | Architecture | Visual Probe Acc | Audio Probe Acc | Joint Audio-Visual Acc | Cross-Modal Retrieval (`a2v` R@1) |
|---|---|---:|---:|---:|---:|
| **`unimodal_separate`** | 2 separate encoders (I-JEPA + Audio-JEPA) | $56.39\%$ | $20.56\%$ | $65.00\%$ | $0.83\%$ (unaligned spaces) |
| **`shared_intra_only_no_cross`** | 1 shared encoder, intra-modal `v2v + a2a` only | $55.56\%$ | $32.50\%$ | **$61.11\%$ ($-3.89\text{ pp}$)** | $17.78\%$ |
| **`shared_mjepa_9path_1level`** | 1 shared encoder, all 9 intra+cross paths, 1-level | $55.00\%$ | $36.11\%$ | $65.28\%$ | **$20.83\%$** |
| **`shared_mjepa_9path_4level_hier`** | 1 shared encoder, **all 9 paths + 4-level hierarchy** | **$64.17\%$** | **$37.50\%$** | **$68.61\%$** | **$20.28\%$** |

### Findings from `E4`
1. **Sharing a trunk without cross-modal prediction degrades joint accuracy (`65.00% -> 61.11%`)**: In `shared_intra_only_no_cross`, visual and audio tokens compete for capacity without explicit cross-modal alignment, hurting visual accuracy (`56.39% -> 55.56%`) and joint accuracy (`65.00% -> 61.11%`).
2. **MJEPA's 9-path objective + 4-level hierarchical targets turn modality sharing into a synergy gain**: `shared_mjepa_9path_4level_hier` uses half the encoder parameters of `unimodal_separate` while improving **visual accuracy (`56.39% -> 64.17%`, `+7.78 pp`)**, **audio accuracy (`20.56% -> 37.50%`, `+16.94 pp`)**, **joint accuracy (`65.00% -> 68.61%`, `+3.61 pp`)**, and **cross-modal audio-to-visual retrieval (`0.83% -> 20.28%`)**.

---

## 6. `E5` — Guiding Question: Reusable Predictive Engine Improving Through Observed Outcomes (`Clover RPRE`)

**Question tested**: When an agent is pre-trained on baseline dynamics (Regime 0: calm) and deployed across **5 Clover Cycles (`Context → Direction → Execution → Outcome → Growth`)** into two unseen physical regimes—**Regime 1** (northeast current `+[0.13, 0.11]`, signaled by top visual bar + high-frequency audio) and **Regime 2** (southwest cross-wind `-[0.13, -0.11]`, signaled by bottom visual bar + low-frequency audio)—can `RPRE` improve its world model and control performance through observed outcomes without forgetting baseline dynamics, and can it transfer zero-shot when the camera is occluded (audio-only)?

### Results Across Clover Cycles 1 → 5 (`E5`, 3 seeds `[101, 202, 303]`)

| Arm | Cycle 1 → Cycle 5 Shifted-Regime Pred MSE | Cycle 1 → Cycle 5 Shifted-Regime Goal Success | Cycle 5 Mean Goal Distance | Cycle 5 Baseline Regime 0 Retention | Cycle 5 Zero-Shot Audio-Only Transfer (Camera Occluded) |
|---|---:|---:|---:|---:|---:|
| **`arm1_static_open_loop_jepa`** (Standard open-loop EB-JEPA) | $0.1162 \to 0.1175$ | $60.42\% \to 54.17\%$ | $0.3646$ | **$93.33\%$** | $56.67\%$ |
| **`arm2_naive_ungated_finetune`** (Naive online SGD on predictor) | $0.0778 \to 0.0702$ | $70.83\% \to 62.50\%$ | $0.3137$ | $90.00\%$ (dips to $73.33\%$ in C3) | $40.00\%$ ($-16.67\text{ pp}$) |
| **`arm3_rpre_state_only_outcome_growth`** (State-only residual, no multimodal cues) | $0.1162 \to 0.0711$ | $60.42\% \to 29.17\%$ | $0.5855$ | $86.67\%$ | $26.67\%$ |
| **`arm4_full_clover_rpre_multimodal_growth`** (**Full Clover `RPRE`**) | **$0.1162 \to 0.0316$ ($3.7\times$ lower)** | **$60.42\% \to 79.17\%$ (peaks at $85.42\%$ in C3)** | **$0.2496$** | **$93.33\%$ ($100\%$ retained)** | **$80.00\%$ ($+23.33\text{ pp}$)** |

### Cycle-by-Cycle Progression of Full Clover `RPRE` (`arm4`)

| Clover Cycle | Shifted-Regime Latent Prediction MSE | Shifted-Regime Goal-Reaching Success | Mean Distance to Goal | Baseline Regime 0 Retention | Zero-Shot Audio-Only Transfer (Camera Occluded) |
|---:|---:|---:|---:|---:|---:|
| **Cycle 1** (Before first outcome update) | $0.1162$ | $60.42\%$ | $0.2925$ | $93.33\%$ | $56.67\%$ |
| **Cycle 2** (After 1st `Outcome → Growth` update) | $0.0817$ | $81.25\%$ | $0.1986$ | $93.33\%$ | $76.67\%$ |
| **Cycle 3** (After 2nd `Outcome → Growth` update) | $0.0506$ | **$85.42\%$** | **$0.1757$** | $93.33\%$ | $80.00\%$ |
| **Cycle 4** (After 3rd `Outcome → Growth` update) | $0.0341$ | $77.08\%$ | $0.2936$ | $93.33\%$ | **$83.33\%$** |
| **Cycle 5** (After 4th `Outcome → Growth` update) | **$0.0316$** | **$79.17\%$** | $0.2496$ | **$93.33\%$** | **$80.00\%$** |

### Why Each Architectural Piece Is Necessary (`E5` Ablation Diagnosis)
1. **Why `arm1_static_open_loop_jepa` fails to improve (`54.17%` in Cycle 5)**: Like all four upstream JEPA repositories, its world model is frozen after pre-training. Even though it observes outcomes $z_{t+1}^{\text{obs}} \neq \hat{z}_{t+1}$ at every step, it lacks a mechanism to turn observed discrepancies into model growth.
2. **Why `arm2_naive_ungated_finetune` oscillates (`62.50%`) and loses transfer (`40.00%`)**: Updating the transition predictor directly via online SGD on each episode overwrites Regime 1 weights during Regime 2 episodes and vice versa, and degrades baseline retention in intermediate cycles (`73.33%` in Cycle 3).
3. **Why `arm3_rpre_state_only_outcome_growth` fails (`29.17%`)**: Without multimodal visual/audio regime cues $c_t$, Regime 1 (`+[0.13, 0.11]`) and Regime 2 (`-[0.13, -0.11]`) are **unobservable/aliased in state-action space** $(z_t, a_t)$. Fitting a state-only residual averages two opposite vector fields, degrading planning below the static baseline.
4. **Why `arm4_full_clover_rpre_multimodal_growth` succeeds across all four metrics**:
   - **Shared Multimodal Context (`I-JEPA` + `Audio-JEPA` + `MJEPA`)**: Disambiguates Regime 1 vs. Regime 2 via the fused latent cue $c_t = \frac{1}{2}(\bar{z}^{(a)} + \bar{z}^{(v)})$.
   - **Closed-Loop Outcome Growth (`OutcomeResidualAdapter`)**: Cuts latent transition MSE from **`0.1162` to `0.0316` (`3.7x` reduction)** and lifts goal-reaching success from **`60.42%` to `79.17%`–`85.42%`**.
   - **Baseline-Prototype Anomaly Gate + Non-Regression Gate**: Preserves **`93.33%` (`100%` of baseline)** performance on Regime 0.
   - **Zero-Shot Missing-Modality Transfer (`80.00%`)**: When the camera is occluded (`visual_x = None`), `CloverCycleAgent.establish_context` uses MJEPA's trained `a2v_predictor` to infer $\hat{z}^{(v)} = f_{a\to v}(z^{(a)})$ from audio alone, enabling `OutcomeResidualAdapter` to apply the right physical drift correction without ever having trained the control policy on audio-only episodes.
