# C2 (HybridAgent) Weight-Space Improvement & Non-Degradation Study

## Outcome Summary

- **Corrected conclusion (2026-11-12): no strictly non-degrading C2 upgrade was established.** The earlier recommendation overstated the evidence. Keep the original for end-to-end use; all trial files were restored.
- **Single-Component `g0.92_only` (`adapter.pt` `post_ln` scaled by `0.92`)**: improves several mean losses and preserves the recorded small QA scores, but is an unaccepted research candidate, not a verified non-degrading upgrade.
  - On **Held-Out `eval32`** (31 trajectories) and **Unseen Held-Out `heldout32`** (32 trajectories), `g0.92_only` improves total `loss`, `hybridlink_loss`, `cos_loss`, `mse_loss`, and `exit_mse_loss` on **58 out of 58 latent trajectories (0 worse; $p = 1.86\times 10^{-9}$ per split)**, while also reducing mean `backbone_loss` (`-0.00159` and `-0.00163`), mean `text_ce` (`-0.00050` and `-0.00051`), and mean `exit_ce` (`-0.1109` and `-0.0613`).
  - On **Natural QA (24 questions)**: **19/24** (`0` broken, `0` fixed; bit-identical to original C2).
  - On **4-Step Forced-Latent Rollout QA (24 questions)**: **22/24** (`0` broken, `0` fixed; zero degradation).
  - Under the predeclared joint gate (which additionally required $p \le 0.01$ on `backbone_loss` simultaneously with $p \le 0.01$ on `hybridlink_loss`), `g0.92_only` achieved $p = 1.86\times 10^{-9}$ on `loss` and `hybridlink_loss` and reduced mean `backbone_loss` with $p = 0.229$ (`17/12`) on `eval32` and $p = 0.132$ (`18/11`) on `heldout32`.
   - Mean `entry_ce` worsened by 0.004960 and 0.005873; mean `exit_cos_loss` worsened by 0.001451 and 0.000986. The forced-rollout median exit-anchor rank worsened from 3,565 to 9,605. Mean improvements therefore do not imply no degradation on every metric or example.
   - Dataset provenance is limited: these splits were reconstructed with an assumed seed and ordering. Their membership was not verified against the Omni training manifest; "official" and "unseen" labels below must not be treated as established provenance. The archived QA scorer also accepts some answer-token occurrences rather than requiring an unambiguous completed answer, so the QA counts are limited instrument results.
- **Combined `combined_anchor1.5_g0.92` (`anchor1.5` + `g0.92`)**: Passed **11 of 12** predeclared gates—improving `loss`, `backbone_loss`, and `hybridlink_loss` with $p \le 1.28\times 10^{-6}$ on both `eval32` and `heldout32`, and preserving natural QA (**19/24**, `0` broken)—but **failed the 4-step forced-latent QA non-degradation gate** (**22/24 $\to$ 21/24**, where `question-21` solved `"HELLO"` inside `<think>` and hit the 48-token cap after `</think>The word`). Because the gate is not weakened post-hoc, `combined_anchor1.5_g0.92` is rejected.
- **All physical trial copies (`anchor1.5_backbone`, `g0.92_adapter`, `noop_adapter`) were audited and restored byte-exactly** (`VERIFICATION.json`). No original model or live service was modified.

---

## 1. What Weight Observation Revealed (`WEIGHT_DIAGNOSIS.json`)

1. **Exact 286-Tensor Backbone Mapping**:
   - `LFM3-Omni-2B-HybridSFT-S3/model.safetensors` (`model.*`) maps 1-to-1 to `model.language_model.*` in `original/model.safetensors`.
   - 62 of the 286 tensors (`embedding_norm.weight`, all 24 `ffn_norm.weight`, all 24 `operator_norm.weight`, and 13 `q_norm.weight`) are bit-identical (`diff_fro = 0.0`).
2. **Why `post_ln` in `adapter.pt` Was Uncalibrated And Why Aggressive Shrinking (`0.0225x`) Degraded C2**:
   - In `adapter.pt` (`HybridLink`), `post_ln.weight` is identically `1.0` across all 2,048 coordinates (`min = 1.0, max = 1.0, std = 0.0`).
   - During Stage 3 (`self_conditioned`), the 24-layer LFM3 backbone was trained on multi-step `<latent>...</latent>` spans where Pass 2 received $R(h_t)$ at unit RMS (`~1.0`).
   - Shrinking `post_ln` by `44x` (`gamma = 0.02253`) improved single-step Pass-1 MSE, **but degraded the 2-pass self-conditioned C2 model** on discovery trajectories (`loss: 1.7177 -> 1.8179`, `backbone_loss: 0.7181 -> 0.7834`, `cos_loss: 0.5941 -> 0.7454`) because multi-step latent spans collapse when $R(h_t)$ is scaled down by 44x.
   - Conversely, gentle calibration (`gamma = 0.92`) keeps Pass 2 in the unit-RMS regime expected by the 24-layer backbone while reducing latent MSE and improving cosine alignment and downstream text cross-entropy.
3. **Under-Trained Special Anchor Rows (`125033:<latent>` and `125034:</latent>`)**:
   - Rows `125034` (`</latent>`) and `125033` (`<latent>`) are the #1 and #4 most changed rows in `model.embed_tokens.weight` (`l2_diff = 0.02053` and `0.01920`), but because they started in the low-norm unused tail (`orig_rms = 0.00388`, 6x smaller than ordinary vocabulary tokens `0.0225`), 160 steps of AdamW at `lr = 1e-5` left them at `c2_rms = 0.00383` and `0.00389` (`entry_ce = 9.43`, `exit_ce = 8.60`).

---

## 2. Held-Out Trajectory Results (`eval32` and `heldout32`)

All 5 arms were evaluated on the full-length (`max_length = 8192`) reconstructed study split (`eval32`, 31 trajectories, 29 with latent spans) and reserved study split (`heldout32`, 32 trajectories, 29 with latent spans). Membership relative to the original Omni training run remains unverified. `noop_control` matched `original` bit-for-bit (`0.0` diff on all trajectories).

### Reconstructed Study Split: `eval32` ($n = 31$)

| Arm | Total `loss` | `backbone_loss` | `text_ce` | `entry_ce` | `exit_ce` | `hybridlink_loss` | `cos_loss` | `mse_loss` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `original` / `noop_control` | 1.85110 | 0.77376 | 0.63764 | 9.42778 | 8.59695 | 1.07734 | 0.59091 | 0.91219 |
| `anchor1.5_only` | 1.84238 (30/1, $p=1.5\text{e-}8$) | 0.76746 (30/1, $p=1.5\text{e-}8$) | 0.63820 | 8.79764 (29/0) | 8.30614 (29/0) | 1.07492 (29/0, $p=1.9\text{e-}9$) | 0.59090 | 0.91220 |
| **`g0.92_only`** | **1.79395** (29/0, $p=1.9\text{e-}9$) | **0.77217** (17/12, $p=0.229$) | **0.63714** (21/8, $p=0.012$) | 9.43274 | **8.48602** | **1.02178** (29/0, $p=1.9\text{e-}9$) | **0.58929** (29/0, $p=1.9\text{e-}9$) | **0.77403** (29/0) |
| `combined_anchor1.5_g0.92` | **1.78561** (30/1, $p=1.5\text{e-}8$) | **0.76629** (30/1, $p=1.5\text{e-}8$) | 0.63792 | **8.80058** (29/0) | **8.21604** (29/0) | **1.01932** (29/0, $p=1.9\text{e-}9$) | **0.58928** (29/0, $p=1.9\text{e-}9$) | **0.77403** (29/0) |

### Reserved Study Split: `heldout32` ($n = 32$)

| Arm | Total `loss` | `backbone_loss` | `text_ce` | `entry_ce` | `exit_ce` | `hybridlink_loss` | `cos_loss` | `mse_loss` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `original` / `noop_control` | 1.82452 | 0.79094 | 0.64735 | 9.15531 | 8.14523 | 1.03358 | 0.57642 | 0.88391 |
| `anchor1.5_only` | 1.81647 (30/2, $p=1.2\text{e-}7$) | 0.78455 (30/2, $p=1.2\text{e-}7$) | 0.64775 | 8.54624 (29/0) | 7.87977 (29/0) | 1.03192 (29/0, $p=1.9\text{e-}9$) | 0.57647 | 0.88391 |
| **`g0.92_only`** | **1.76863** (29/0, $p=1.9\text{e-}9$) | **0.78931** (18/11, $p=0.132$) | **0.64683** (18/11, $p=0.132$) | 9.16118 | **8.08390** | **1.02178** $\to$ **0.97933** (29/0, $p=1.9\text{e-}9$) | **0.57484** (29/0, $p=1.9\text{e-}9$) | **0.75005** (29/0) |
| `combined_anchor1.5_g0.92` | **1.76088** (30/2, $p=1.2\text{e-}7$) | **0.78297** (29/3, $p=1.3\text{e-}6$) | 0.64738 | **8.54978** (29/0) | **7.81553** (28/1) | **0.97790** (29/0, $p=1.9\text{e-}9$) | **0.57490** (29/0, $p=1.9\text{e-}9$) | **0.75006** (29/0) |

---

## 3. Downstream Question-Answering & 4-Step Latent Rollout Check (24 Questions)

| Arm | Natural Greedy QA (/24) | Natural Broken / Fixed | 4-Step Latent Rollout QA (/24) | Latent4 Broken / Fixed |
|---|---:|---:|---:|---:|
| `original` | 19/24 | — | 22/24 | — |
| **`g0.92_only`** | **19/24** (bit-identical) | **0 broken / 0 fixed** | **22/24** | **0 broken / 0 fixed** |
| `anchor1.5_only` | 19/24 | 0 broken / 0 fixed | 22/24 | 1 broken / 1 fixed |
| `combined_anchor1.5_g0.92` | 19/24 | 0 broken / 0 fixed | 21/24 | 2 broken / 1 fixed |

---

## 4. Recommendation and Reproducibility

- **Do not promote `g0.92_only` as strictly non-degrading.** Retain it only as a reproducible research specification until fresh, provenance-checked validation meets the original gates, including entry/exit behavior and task-level correctness. No passing replacement dataset was produced by this study.
- **Reject the tested anchor and combined edits for deployment.** Their recorded rollout regressions remain evidence against adoption. These experiments do not establish that retraining is the only possible remedy.
- **Evidence**:
  - Code: [diagnose.py](diagnose.py), [explore_discovery.py](explore_discovery.py), [search_discovery.py](search_discovery.py), [probe_mechanisms.py](probe_mechanisms.py), [test_combo_discovery.py](test_combo_discovery.py), [validate_c2.py](validate_c2.py), [verify.py](verify.py).
  - Records: [CONTEXT.md](CONTEXT.md), `WEIGHT_DIAGNOSIS.json`, `DISCOVERY_EXPLORATION.json`, `DISCOVERY_SEARCH.json`, `MECHANISM_PROBE.json`, `COMBO_DISCOVERY.json`, `frozen/`, `results/`, `copies/*/PATCH_AUDIT.json`, `copies/*/REVERSAL.json`, and `VERIFICATION.json`.
  - Archive SHA-256 (excluding model weights): `8bd73abb188f6e13b054817fc23fcd9eab0c33741de2dc63f467ebaab4292ded`.
