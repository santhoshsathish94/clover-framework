# C2 (HybridAgent) Weight-Space Improvement Study

## Correction Before Cross-Model Work

The previous "zero degradation on any metric" conclusion is withdrawn. Raw
g0.92 results show worse mean entry CE and exit cosine loss on both study splits,
a worse forced-rollout exit rank, and failure to meet the frozen backbone-loss
significance requirement. Small QA score equality does not override those facts.
All physical candidates were restored; no accepted replacement checkpoint exists.

The split reconstruction assumes seed/order rather than proving membership in
the actual Omni training run. Do not call those data verified unseen training
holdouts. The statement that AdamW movement is bounded by the sum of learning
rates is also withdrawn; that is not a valid general optimizer bound. Recorded
norms and loss changes remain observations, but their proposed causal explanation
was not established. The QA scorer's answer-occurrence fallback limits its counts.
Preserve raw results and frozen code; do not repair a failed gate after exposure.

## Direction and Boundaries

On 2026-11-12 the human directed:
> "okay only look at c2 if you can improve it not degrade it"

Boundaries in force:
- Work exclusively on **C2** (`LFM3-Omni-2B-HybridSFT-S3`: the 286-tensor LFM3 language backbone in `model.safetensors` and the 8-tensor `HybridLink` in `adapter.pt`, plus its precomputed export counterpart `hybridagent/plan-trunk-0`).
- **Do not degrade C2**: any candidate change must improve C2 on its target metrics while strictly preserving or improving all guarded C2 metrics (text/backbone loss, latent prediction loss, latent exit readout, and downstream generation accuracy) on held-out data. If any guarded metric degrades, reject and revert the change.
- **Separate copies only**: all work stays in the isolated worktree `C:\personal\oss\clover-astra-weightspace-wt` (`research/astra-lfm3-weightspace/c2_improvement`) and isolated K3 directory `/opt/clover-k3/astra-weightspace-20261111/c2_improvement`. Original checkpoints (`/opt/clover-k3/models/LFM3-Omni-2B-HybridSFT-S3`, `/opt/clover-k3/astra-weightspace-20261111/original`, `/opt/clover-k3/s23-omni2b`) are mounted read-only or inaccessible during runs.
- **No conventional gradient training** and no unapproved deployment.

## Context: What C2 Is and What Prior Cycles Showed

1. **Architecture and Training Contract (`HybridSFTModel`)**:
   - C2 consists of a 24-layer LFM3-Omni-2B causal LM backbone (`d = 2048`, vocab `128,000` with `<latent>` at `125033` and `</latent>` at `125034`) paired with `HybridLink(2048)`:
     $$R(h) = \text{LayerNorm}_{\text{post}}\!\left(h + W_2\,\text{GELU}(W_1\,\text{LayerNorm}_{\text{pre}}(h) + b_1) + b_2\right)$$
   - In Stage 3 (`self_conditioned`), the model runs two passes on hybrid trajectories:
     - Pass 1 runs the backbone on teacher token embeddings $e(x_t)$.
     - At latent positions (`latent_mask == 1`, i.e., `<latent>` and latent thought tokens), $R(h_t)$ replaces $e(x_{t+1})$ as the input embedding to Pass 2.
     - Pass 2 computes `backbone_loss` (cross-entropy on text tokens and `</latent>` exit anchors with weight `2.0`, conditioned on Pass 2's injected $R(h_t)$ vectors) and `hybridlink_loss`:
       $$\mathcal{L}_{\text{link}} = (1 - \cos(R(h_t), e(x_{t+1}))) + 0.2\,\text{MSE}(R(h_t), e(x_{t+1}))$$
       plus an equal-weight term on the exit anchor (`</latent>`).
   - The checkpoint's own `train_log.jsonl` records on its 32-trajectory evaluation split (`eval-32`, `n = 32`):
     - Before Stage 3: `loss = 4.2174`, `backbone_loss = 2.4834`, `hybridlink_loss = 1.7339`, `cos_loss = 0.6826`
     - After Stage 3 (80 steps): `loss = 1.8743`, `backbone_loss = 0.7709`, `hybridlink_loss = 1.1035`, `cos_loss = 0.6389`

2. **Why `post_ln` Scale Was Uncalibrated and Why Naive Post-LN Rescaling Is Subtle**:
   - In `adapter.pt`, `post_ln.weight` was initialized to `1.0` (` RMS = 1.0`), so $R(h_t)$ has RMS $\approx 0.997$, whereas LFM3 token embeddings $e(x)$ have RMS $\approx 0.0234$.
   - Across 120 Stage-2 steps ($\text{lr} \le 2\times 10^{-4}$) and 80 Stage-3 steps ($\text{lr} \le 8\times 10^{-5}$), AdamW could move `post_ln.weight` by at most $\sum \text{lr}_t < 0.0304$, so `post_ln.weight` remained at RMS $1.000$.
   - Crucially, however, during Stage 3 (`self_conditioned`), the backbone's 24 layers were updated for 80 steps *while Pass 2 was receiving unscaled $R(h_t)$ vectors of RMS $\approx 1.0$* at latent positions!
   - Furthermore, in `HybridSFTModel.forward`, `hybridlink_loss` has two parts:
     - `cos_loss` ($1 - \cos(R(h_t), e(x_{t+1}))$), which is invariant to positive scalar multiplication of $R(h_t)$, but sensitive to the direction of $R(h_t)$ (including `post_ln.bias` and low-cosine dimensions).
     - `mse_loss` ($\frac{1}{d}\|R(h_t) - e(x_{t+1})\|_2^2$), where $\|R(h_t)\|_{\text{RMS}} \approx 1.0$ vs $\|e(x_{t+1})\|_{\text{RMS}} \approx 0.0234$ creates $\text{MSE} \approx 1.0^2 = 1.0$, contributing $0.2 \times (1.0 + 1.0) \approx 0.40$ to `hybridlink_loss`!
   - If we scale down $R(h_t)$ to RMS $0.0234$, `mse_loss` drops by $\sim 99.9\%$, **but** we must check what happens to Pass 2 `backbone_loss` on actual self-conditioned trajectories! Does scaling $R(h_t)$ improve Pass 2 `backbone_loss` AND `hybridlink_loss` simultaneously, or is there an optimal calibration regime (or affine / subspace projection) where neither `backbone_loss` nor `cos_loss` degrades?

3. **Cycle 1 Plan (Diagnosis Before Any Edit)**:
   - Compare all 286 tensors of `LFM3-Omni-2B-HybridSFT-S3/model.safetensors` against `original/model.safetensors` (exact Frobenius norms, max abs diffs, and special token rows `<think>`, `</think>`, `<latent>`, `</latent>`).
   - Inspect all 8 tensors of `adapter.pt` and all 12 tensors of precomputed `hybridagent/plan-trunk-0`.
   - Load the untouched 32-trajectory evaluation split (`eval-32-s1234.jsonl`) and a disjoint discovery split from `train.jsonl` (`order[2432:2456]`, unseen during the 2,400-trajectory training run).
   - Verify whether `eval-32-s1234.jsonl` reproduces `train_log.jsonl`, and measure how `backbone_loss`, `hybridlink_loss`, `cos_loss`, and `mse_loss` behave on the discovery split.

## Cycle 1 & 2 Outcomes: Weight Diagnosis and Discovery Findings

1. **Exact Weight-Space Comparison (`WEIGHT_DIAGNOSIS.json`)**:
   - All 286 tensors in `LFM3-Omni-2B-HybridSFT-S3/model.safetensors` (`model.*`) map 1-to-1 to `model.language_model.*` in `original/model.safetensors`.
   - 62 of the 286 tensors (`embedding_norm.weight`, all 24 `ffn_norm.weight`, all 24 `operator_norm.weight`, and 13 `q_norm.weight`) are bit-identical (`diff_fro = 0.0`).
   - In `adapter.pt` (`HybridLink`), `pre_ln.weight` and `post_ln.weight` are identically `1.0` across all 2,048 coordinates (`min = 1.0, max = 1.0, std = 0.0`).
   - In `model.embed_tokens.weight`, the #1 and #4 most changed rows out of all 128,000 tokens are `125034:</latent>` (`l2_diff = 0.02053`) and `125033:<latent>` (`l2_diff = 0.01920`). Because those two rows started in the unused low-norm tail (`orig_rms = 0.00388`, 6x smaller than normal vocabulary rows `0.0225`) and 160 steps of AdamW at `lr = 1e-5` could only move them by `~0.020`, they remain at `c2_rms = 0.00383` and `0.00389`, causing high cross-entropy on `<latent>` entry (`entry_ce = 7.72`) and `</latent>` exit (`exit_ce = 6.58`, which carries weight `2.0` in `loss_mask`).

2. **Why Naive `post_ln` Rescaling (`0.0225`) Degrades Self-Conditioned C2 (`DISCOVERY_EXPLORATION.json`)**:
   - On the discovery trajectories (3,498 latent positions), shrinking `post_ln` by `gamma_rms = 0.02253` or `gamma_ls = 0.00744` improves single-step Pass-1 `hybridlink_loss` (`1.1929 -> 0.8003`), **but degrades the 2-pass self-conditioned C2 model**:
     - Baseline (`gamma = 1.0`): `loss = 1.7177`, `backbone_loss = 0.7181`, `hybridlink_loss = 0.9996`, `cos_loss = 0.5941`, `mse_loss = 0.8628`.
     - `gamma_rms = 0.02253`: `loss = 1.8179` (worse), `backbone_loss = 0.7834` (worse), `hybridlink_loss = 1.0345` (worse), `cos_loss = 0.7454` (worse).
   - Cause: during Stage 3 (`self_conditioned`), the 24-layer LFM3 backbone was trained on multi-step latent spans where Pass 2 receives $R(h_t)$ at unit RMS (`~1.0`). Collapsing $R(h_t)$ by 44x throws Pass 2 out of distribution across multi-step latent spans.

3. **Two-Mechanism Synergy on Split Discovery (`MECHANISM_PROBE.json`, `COMBO_DISCOVERY.json`)**:
   - Splitting the discovery trajectories into `fit_samples` (first 5) and `val_samples` (remaining 12 unseen discovery trajectories) revealed why single-component edits had a small tradeoff and how their combination eliminates it:
     - **Component A (`anchor1.5_only`)**: Extrapolating only the 2 special anchor rows `125033` (`<latent>`) and `125034` (`</latent>`) of `model.embed_tokens.weight` by `1.5x` along their exact HybridSFT task vector $\Delta e_a = e_{\text{C2}}[a] - e_{\text{orig}}[a]$:
       $$e_{\text{new}}[a] = e_{\text{orig}}[a] + 1.5\,(e_{\text{C2}}[a] - e_{\text{orig}}[a]), \quad a \in \{125033, 125034\}$$
       strongly reduces `entry_ce` (`-0.573`) and `exit_ce` (`-0.240`) and lowers `backbone_loss` (`-0.00314`).
     - **Component B (`g0.92_only`)**: Gently scaling `post_ln.weight` and `post_ln.bias` in `adapter.pt` by $\gamma = 0.92$ reduces `hybridlink_loss` (`-0.04947`), `cos_loss` (`-0.00171`), and `mse_loss` (`-0.11830`), while slightly raising `exit_ce` (`+0.0899`) when used without Component A.
     - **Combined (`raw_anchor1.5_g0.92`)**: Combining the 2-row anchor task-vector extrapolation ($1.5\times$) with the gentle `post_ln` calibration ($0.92\times$) improved **every single metric** on the 12 unseen discovery validation trajectories (`val_samples`):
       - `d_loss = -0.06752` (`12 better, 0 worse`)
       - `d_backbone_loss = -0.01010` (`12 better, 0 worse`)
       - `d_text_ce = -0.00059` (improved)
       - `d_entry_ce = -0.61899` (improved)
       - `d_exit_ce = -0.35801` (improved)
       - `d_hybridlink_loss = -0.05742` (`11 better, 0 worse`)
       - `d_cos_loss = -0.00127` (improved)
       - `d_mse_loss = -0.13558` (improved)

## Cycle 3 Plan: Frozen Held-Out Validation and Non-Degradation Gate

Before touching `eval32` or `heldout32`, we freeze the primary candidate (`combined_anchor1.5_g0.92`) alongside a byte-exact `noop_control` and the two single-component ablations (`anchor1.5_only` and `g0.92_only`) on physical checkpoint copies:
- **Held-out Split 1 (`eval32`)**: All 31 valid trajectories (`max_length = 8192`) of the checkpoint's official evaluation split (`eval-32-s1234.jsonl`).
- **Held-out Split 2 (`heldout32`)**: All 32 valid trajectories (`max_length = 8192`) of the unseen held-out split (`heldout32.jsonl`).
- **Downstream QA & Latent Rollout Check**: All 24 diagnostic math/factual/logic questions run through `HybridGenerationEngine` (greedy `temperature = 0.0`, `max_response_tokens = 96`, zero tools) plus a forced latent-span rollout check (`<latent>` injected after `<think>\n`, verifying `HybridLink` rollout and `</latent>` exit readout + final answer accuracy).
- **Strict Non-Degradation & Improvement Gate**:
  1. On both `eval32` and `heldout32`, the candidate must reduce mean `loss`, `backbone_loss`, `hybridlink_loss`, `cos_loss`, and `mse_loss`, with one-sided exact paired sign test $p \le 0.01$ on `loss`, `backbone_loss`, and `hybridlink_loss`.
  2. Ordinary text cross-entropy (`text_ce`) must not degrade (mean delta $\le 0.0$).
  3. Downstream QA accuracy on the 24 questions must be $\ge$ baseline C2 with zero newly broken questions, and forced-latent rollout accuracy must be $\ge$ baseline C2.

## Cycle 3 Outcome: Held-Out Validation, QA Rollout, Audit, and Reversal

The frozen validation (`validate_c2.py`) and independent audit (`verify.py`) completed on K3 (`25m 10.7s` and `8m 09.1s`):

1. **`noop_control` Parity**: Bit-identical (`0.0` diff) to `original` on all 31 `eval32` and 32 `heldout32` trajectories.
2. **Primary Combined Candidate (`combined_anchor1.5_g0.92`)**:
   - Passed **11 of 12** predeclared gates:
     - `eval32`: `loss` `1.85110 -> 1.78561` (`30/1`, $p=1.49\times 10^{-8}$), `backbone_loss` `0.77376 -> 0.76629` (`30/1`, $p=1.49\times 10^{-8}$), `hybridlink_loss` `1.07734 -> 1.01932` (`29/0`, $p=1.86\times 10^{-9}$), `cos_loss` `0.59091 -> 0.58928` (`29/0`, $p=1.86\times 10^{-9}$), `mse_loss` `0.91219 -> 0.77403` (`29/0`).
     - `heldout32`: `loss` `1.82452 -> 1.76088` (`30/2`, $p=1.23\times 10^{-7}$), `backbone_loss` `0.79094 -> 0.78297` (`29/3`, $p=1.28\times 10^{-6}$), `hybridlink_loss` `1.03358 -> 0.97790` (`29/0`, $p=1.86\times 10^{-9}$), `cos_loss` `0.57642 -> 0.57490` (`29/0`, $p=1.86\times 10^{-9}$), `mse_loss` `0.88391 -> 0.75006` (`29/0`).
     - Natural QA: `19/24` (`0` broken, `0` fixed).
   - **Failed `qa_latent4_non_degraded`**: scored `21/24` vs `22/24` (`2` broken, `1` fixed; `question-21` derived `"HELLO"` inside `<think>` and hit the 48-token cap right after `</think>The word`). Under Clover's strict gate discipline, `combined_anchor1.5_g0.92` is **rejected**.
3. **Single-Component Adapter Calibration (`g0.92_only`: `adapter.pt` `post_ln` $\times 0.92$)**:
  - **Recorded results, not zero degradation (see correction above)**:
     - Natural QA: **19/24** (`0` broken, `0` fixed; bit-identical to `original` on all 24 questions).
     - 4-Step Forced-Latent Rollout QA: **22/24** (`0` broken, `0` fixed).
     - `eval32`: `loss` `1.85110 -> 1.79395` (`29/0`, $p=1.86\times 10^{-9}$), `hybridlink_loss` `1.07734 -> 1.02178` (`29/0`, $p=1.86\times 10^{-9}$), `cos_loss` `0.59091 -> 0.58929` (`29/0`, $p=1.86\times 10^{-9}$), `mse_loss` `0.91219 -> 0.77403` (`29/0`), `backbone_loss` `0.77376 -> 0.77217` (`-0.00159`, `17/12`, $p=0.229$), `text_ce` `0.63764 -> 0.63714` (`-0.00050`, `21/8`, $p=0.0121$).
     - `heldout32`: `loss` `1.82452 -> 1.76863` (`29/0`, $p=1.86\times 10^{-9}$), `hybridlink_loss` `1.03358 -> 0.97933` (`29/0`, $p=1.86\times 10^{-9}$), `cos_loss` `0.57642 -> 0.57484` (`29/0`, $p=1.86\times 10^{-9}$), `mse_loss` `0.88391 -> 0.75005` (`29/0`), `backbone_loss` `0.79094 -> 0.78931` (`-0.00163`, `18/11`, $p=0.132$), `text_ce` `0.64735 -> 0.64683` (`-0.00051`, `18/11`, $p=0.132$).
4. **Reversal and Handoff**:
   - All three physical copies (`anchor1.5_backbone`, `g0.92_adapter`, `noop_adapter`) were audited and restored byte-exactly (`VERIFICATION.json`).
   - Evidence archive (excluding model weights) SHA-256: `8bd73abb188f6e13b054817fc23fcd9eab0c33741de2dc63f467ebaab4292ded`, verified before local extraction.


