# Stage 1 — Primary Source & Repository Audit of the Four Core JEPA Papers

Offered, not imposed. Date: 2026-11-07. Direction owner: Santhosh.  
Worktree: `C:\personal\oss\clover-jepa-wt` (branch `research/jepa-predictive-engine`).

---

## 1. Audited Repositories & Provenance

All four upstream repositories were cloned and inspected line-by-line:

| # | Paper | Repository | Commit SHA | Commit Date | Core Focus |
|---|---|---|---|---|---|
| **1** | **I-JEPA**: *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (Assran et al., CVPR 2023, `arXiv:2301.08243`) | [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) | `52c1ae95d05f743e000e8f10a1f3a79b10cff048` | `2023-06-13` | Visual latent prediction with disjoint multi-block semantic masking and context-only sparse encoding |
| **2** | **Audio-JEPA**: *Joint-Embedding Predictive Architecture for Audio Representation Learning* (Tuncay et al., ICME 2025, `hal-05128180`) | [`LudovicTuncay/Audio-JEPA`](https://github.com/LudovicTuncay/Audio-JEPA) | `ddd97ee9b88c00572f59bf2a00eb42120e6a4dea` | `2026-07-16` | Time-frequency log-mel spectrogram latent prediction with centered L2-normalized cosine/MSE loss (`norm_mse`) |
| **3** | **EB-JEPA**: *Energy-Based Joint-Embedding Predictive Architectures* (Terver, Balestriero, ..., Rabbat, LeCun, Bar, 2026, `arXiv:2602.03604`) | [`facebookresearch/eb_jepa`](https://github.com/facebookresearch/eb_jepa) | `966e61e9285b3a876f49b9774e9720d9a99a7925` | `2026-02-04` | Energy-based anti-collapse regularization (`VCLoss`, `BCS` Epps-Pulley, `TemporalSimilarityLoss`, `InverseDynamicsLoss`), multi-step unrolling, and `CEM`/`MPPI` latent planning |
| **4** | **MJEPA**: *A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual Learning* (Teotia, Bardes, Rabbat, Chopra, Muckley, Ballas, 2026, `arXiv:2606.25225`) | [`facebookresearch/MJEPA`](https://github.com/facebookresearch/MJEPA) | `82f3472219f2f58672999ed654314b34ad590c07` | `2026-08-13` | Single shared Transformer trunk for audio & video with modality embeddings, 4-level hierarchical target extraction, and 3 intra-modal + 6 cross-modal predictive objectives |

---

## 2. Paper-by-Paper Code Audit & Mathematical Mechanisms

### 2.1 I-JEPA (`facebookresearch/ijepa`) — Visual Latent Prediction

- **Architecture (`src/models/vision_transformer.py`)**:
  - **Context Encoder $E_\theta$ (`VisionTransformer`, lines `325–430`)**: Patchifies images via `PatchEmbed` (`Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`), adds fixed 2D sine-cosine positional embeddings (`get_2d_sincos_pos_embed`), and **immediately drops masked tokens before the Transformer blocks** via `x = apply_masks(x, masks)` (`line 420`). Self-attention complexity is reduced from $\mathcal{O}(N^2)$ to $\mathcal{O}(N_{\text{ctxt}}^2)$.
  - **Target Encoder $E_{\bar\theta}$ (`src/train.py:165, 255–263`)**: Deep copy of $E_\theta$ with `requires_grad = False`. Runs on the full unmasked image under `torch.no_grad()`, normalizes over the feature dimension with `h = F.layer_norm(h, (h.size(-1),))` (`line 258`), and gathers target blocks `h = apply_masks(h, masks_pred)` (`line 261`). Updated via EMA momentum schedule $m_t \in [0.996, 1.0]$ (`lines 209, 286–288`).
  - **Latent Predictor $P_\phi$ (`VisionTransformerPredictor`, lines `229–322`)**: Projects context tokens from `embed_dim` ($D$) to a narrower bottleneck `predictor_embed_dim` ($D_p = 384$, `line 249`), adds `predictor_pos_embed` at context indices `masks_x`, instantiates shared learnable `mask_token` vectors conditioned on target positional embeddings at `masks` (`lines 303–312`), runs `depth` Transformer blocks over `[x_ctxt, pred_tokens]`, slices only the target predictions `x = x[:, N_ctxt:]`, and projects back to `embed_dim` (`lines 319–321`).
- **Multi-Block Semantic Masking (`src/masks/multiblock.py:21–169`)**:
  - Samples `npred=4` rectangular target blocks (`pred_mask_scale=(0.15, 0.2)`, `aspect_ratio=(0.75, 1.5)`) and records their binary complements `masks_C` (`lines 103–105`).
  - When `allow_overlap=False` (default), `constrain_mask` multiplies the sampled context block by `acceptable_regions = masks_C` (`lines 75–79, 139–152`) so the context encoder never sees any patch inside the target blocks.
- **Loss (`src/train.py:268–271`)**:
  - $\mathcal{L}_{\text{I-JEPA}} = \text{SmoothL1}(P_\phi(E_\theta(x, M_{\text{enc}}), M_{\text{enc}}, M_{\text{pred}}), \text{sg}(\text{LN}(E_{\bar\theta}(x))[M_{\text{pred}}]))$.

---

### 2.2 Audio-JEPA (`LudovicTuncay/Audio-JEPA`) — Audio Latent Prediction

- **Spectrogram Tokenization (`src/data/components/mel_spec.py:6–66`, `configs/model/jepa.yaml`)**:
  - Converts zero-mean 32 kHz waveforms into 2D Kaldi log-mel filterbank spectrograms of shape `[1, T=1024, F=128]` (`f_min=20`, `window_type='hanning'`), then patchifies with 2D patches `patch_size=(16, 16)` into a $64 \times 8 = 512$ time-frequency token grid.
  - Reuses the exact same `VisionTransformer` (`depth=12, embed_dim=768`) and `VisionTransformerPredictor` (`depth=6`) structure as I-JEPA (`configs/model/jepa.yaml:3–19`).
- **Loss Function & Codebase Finding (`src/models/components/loss.py:12–47`)**:
  - Defines `norm_mse_loss(pred, target)` as $2 - 2\langle \frac{\text{pred}}{\|\text{pred}\|_2}, \frac{\text{target}}{\|\text{target}\|_2}\rangle$.
  - **Codebase Audit Finding (`loss.py:42–46`)**: When `norm_pix_loss: true` (`configs/model/jepa.yaml:24`), `Loss.forward` standardizes `target` (`target = (target - mean) / (var + 1e-6).sqrt()`) before `norm_mse_loss`, **but leaves `pred` uncentered**. Consequently, if `pred` has a non-zero mean across features, `Loss(z, z) != 0` unless `pred` is also centered (verified in `tests/test_rpre.py`).
  - **Anisotropic Time-Frequency Structure (`src/masks/components/random_block.py` vs `multiblock.py`)**: Audio spectrograms have orthogonal physical axes (Time $T$ carries formant/rhythm progressions; Frequency $F$ carries simultaneous harmonic overtones). Structured time-band + frequency-band masking (`sample_audio_tf_masks`) forces the predictor to infer both missing temporal frames and missing harmonic bands.

---

### 2.3 EB-JEPA (`facebookresearch/eb_jepa`) — Generalized Learning & Latent Planning

- **Energy-Based Anti-Collapse Regularization (`eb_jepa/losses.py`)**:
  - Eliminates the requirement for an EMA target encoder by pairing the prediction energy $E_{\text{pred}} = \|P_\phi(z_t, a_t) - z_{t+1}\|_2^2$ (`SquareLossSeq`, `lines 16–26`) with explicit information-preserving regularizers on the state representations $z$:
    1. **`HingeStdLoss` (`lines 49–78`)**: $\mathcal{L}_{\text{std}}(z) = \frac{1}{D}\sum_{d=1}^D \max(0, \gamma - \sqrt{\text{Var}(z_{\cdot, d}) + 10^{-4}})$, preventing constant collapse.
    2. **`CovarianceLoss` (`lines 81–106`)**: $\mathcal{L}_{\text{cov}}(z) = \frac{1}{D(D-1)}\sum_{i \neq j} \text{Cov}(z)_{i,j}^2$, preventing dimensional collapse (feature redundancy).
    3. **`TemporalSimilarityLoss` (`lines 109–125`)**: $\mathcal{L}_{\text{sim\_t}}(z) = \frac{1}{T-1}\sum_{t=1}^{T-1}\|z_{t+1} - z_t\|_2^2$, enforcing smooth latent trajectories.
    4. **`InverseDynamicsLoss` (`lines 128–156`)**: $\mathcal{L}_{\text{idm}}(z, a) = \frac{1}{T-1}\sum_{t=1}^{T-1}\|\text{IDM}(z_t, z_{t+1}) - a_t\|_2^2$, forcing $z_t$ to retain controllable state variables that are affected by actions $a_t$.
    5. **`BCS` (`Batched Characteristic Slicing`, `lines 345–384`)**: Projects $z$ onto $S=256$ random unit vectors $A \in \mathbb{R}^{D \times S}$ and computes the **Epps-Pulley test statistic** (`epps_pulley`, `lines 356–369`) comparing the empirical characteristic function $\hat{\phi}(t) = \frac{1}{N}\sum_{n=1}^N e^{i t (z_n A)}$ against the standard Gaussian characteristic function $\phi(t) = e^{-\frac{1}{2}t^2}$ via trapezoidal integration over $t \in [-3, 3]$.
- **Unified Parallel & Autoregressive Latent Unrolling (`eb_jepa/jepa.py:49–197`)**:
  - `JEPA.unroll` unifies teacher-forced parallel training (`unroll_mode="parallel"`) and open-loop multi-step rollouts (`unroll_mode="autoregressive"`, `lines 159–186`): at each step $i \in \{0, \dots, H-1\}$, takes the sliding window `predicted_states[:, :, -effective_ctxt_window:]` and corresponding actions, predicts `pred_step`, and appends it to `predicted_states`.
- **Latent Model-Predictive Control (`eb_jepa/planning.py:347–575`)**:
  - `GCAgent` encodes the current observation $z_0 = E_\theta(x_0)$ and goal observation $z_g = E_\theta(x_g)$ (`set_goal`, `lines 393–409`), sets `ReprTargetDistMPCObjective` (`lines 446–474`), and runs `CEMPlanner.plan` (`lines 514–575`) over `n_iters` iterations of `num_samples` action trajectories, selecting the top `num_elites` trajectories to update $(\mu_a, \sigma_a)$ and executing the first action in the environment (`receding-horizon MPC`).
- **Key Architectural Limitation Identified in `eb_jepa/planning.py:258–295`**:
  - During `main_eval`, `GCAgent` steps in the real environment (`obs, reward, done, truncated, info = env.step(a)`), observes the true next state, and logs whether the episode succeeded or failed—**but never feeds the observed outcome discrepancy $(E_\theta(x_{t+1}^{\text{obs}}) - \hat{z}_{t+1})$ back into the predictor!** In any environment with unmodeled drift, friction, or regime shift, static open-loop `GCAgent` repeats the exact same prediction error on every episode.

---

### 2.4 MJEPA (`facebookresearch/MJEPA`) — Shared Audio-Visual Representation Learning

- **Single Shared Encoder Across Modalities (`src/models/mjepa_encoder.py:21–385`)**:
  - Uses separate patch tokenizers (`video_patch_embed` and `audio_patch_embed`), separate positional embeddings (`video_pos_embed` and `audio_pos_embed`), and learnable modality type tokens (`audio_mod_embed`, `video_mod_embed`, `lines 201–206`), feeding into **one shared `nn.ModuleList` of Transformer `Block`s** (`lines 127–148`).
  - Can run unimodally (`audio_x` only or `video_x` only) or jointly by concatenating tokens along the sequence dimension (`x = torch.cat([audio_x, video_x], dim=1)`, `line 330`) and splitting them back at output (`lines 364–366`).
- **4-Level Hierarchical Target Representation (`src/models/mjepa_encoder.py:165–198`, `src/models/mjepa_predictor.py:62–66, 173–179`)**:
  - Extracts intermediate representations from 4 depths (e.g., layers `[2, 5, 8, 11]` for depth 12), applies a dedicated `LayerNorm` per level (`self.norms_block`), and concatenates them along the feature dimension (`4 * embed_dim`, `mjepa_encoder.py:356`).
  - In `train.py:893–907`, each of the 4 hierarchical slices is independently `F.layer_norm`-normalized before computing the prediction loss (`normalize_and_concat`).
- **3 Intra-Modal + 6 Cross-Modal Predictive Objectives (`app/mjepa/train.py:1148–1208`, `src/audio_models/a2v_predictors.py:16–77`)**:
  - **Intra-modal (token-level masked prediction via shared `VisionTransformerPredictor`)**:
    1. `v2v_loss`: masked video tokens from visible video context.
    2. `a2a_loss`: masked audio tokens from visible audio context.
    3. `av2av_loss`: masked audio + video tokens jointly from visible audio + video context.
  - **Cross-modal (pooled latent prediction via `A2V2_MLP_Predictor` with mean or `AttentivePooler`)**:
    4. `a2v_loss`: audio context $\to$ pooled video target.
    5. `v2a_loss`: video context $\to$ pooled audio target.
    6. `av2a_loss`: joint AV context $\to$ pooled audio target.
    7. `av2v_loss`: joint AV context $\to$ pooled video target.
    8. `a2av_loss`: audio context $\to$ pooled joint AV target.
    9. `v2av_loss`: video context $\to$ pooled joint AV target.
- **Core Scientific Insight of MJEPA (`README.md:23`)**:
  - Sharing one encoder across modalities *without* cross-modal prediction degrades below unimodal encoders because audio and visual tokens compete for capacity in separate subspaces. Adding the 6 cross-modal prediction paths aligns the shared latent space so each modality actively improves the other.
