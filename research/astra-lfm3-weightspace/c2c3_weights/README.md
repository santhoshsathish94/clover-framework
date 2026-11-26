# C2 HybridAgent and C3 NCM Weight Tests

## Recommendation

**Investigate C2 adapter gain calibration further; do not adopt it yet. Reject
the tested C3 bridge-scale change.** Neither passed its full predefined gate.
All six trial checkpoints were restored byte-exactly. Original models and the
installed application were not changed; no gradient fine-tuning or deployment ran.

This study inspected the actual Omni HybridSFT-S3 language model and adapter,
plus the raw NCM checkpoint, not just the exported inference-only pipeline.
The inventories cover 286 C2 backbone tensors, 8 adapter tensors and 362 C3
tensors. Statistical samples are labeled as samples; matrix row norms are
computed for all rows. Outliers were not treated as defects by themselves.

## C2 Result

The adapter output had roughly 43 times the token-input RMS on the loading
smoke. Its final LayerNorm weights have RMS 1.0. We changed only
`post_ln.weight` and `post_ln.bias` in separate copies, using gains fitted on
20 discovery texts. Discovery selected the RMS-matching gain **0.02350780**
over the least-squares gain **0.00836702**.

Held-out evaluation used 20 different texts, with two continuation positions
per text. Lower negative log-likelihood (NLL) is better:

| Feedback input | Mean held-out NLL |
|---|---:|
| Original adapter | 11.4983 |
| Calibrated adapter | 5.3439 |
| Zero-vector control | 8.0657 |
| Actual next-token embedding | 0.0885 |

The calibrated adapter improved 19/20 texts versus the original
(one-sided paired sign p=0.00002003), but only 14/20 versus zero input
(p=0.057659). The gate required p<=0.05 against both. **It failed that gate.**

This is a substantial feedback-calibration signal, not demonstrated planning
improvement. The continuations restate supplied notes; they do not solve new
reasoning tasks. In a separate bounded natural HybridAgent probe, original and
calibrated adapters both answered `43` and `Rome` correctly, with identical
outputs and zero latent steps on both prompts. No tool was permitted or used.
The adapter change cannot affect a turn in which the adapter is unused.

Next useful C2 work: establish representative natural latent usage, then test
the fixed gain on a new task-level validation set with the zero-input control
retained. Do not waive the failed gate or infer universal gain from two prompts.

## C3 Result

The faithful raw NCM path used its 48+48 packed prefix, padding mask, 64-token
target, all 11 levels and native generation implementation. The Omni vocabulary
was loaded from actual 128,000-row tensors; inherited Qwen/WMT14 config labels
were not used to load a different model. Exactly 22 absent training-only EMA
buffers were initialized; all inference state remained strictly checked.

We changed only `lfm3_combo_bridge.scale`, testing 0.5x, 1x and 2x copies.
Discovery selected 2x. On the 20 held-out texts:

| Measure | Original bridge | Doubled bridge |
|---|---:|---:|
| Mean code NLL, lower is better | 1.29713 | 1.31910 |
| Generated nonpad-token matches | 118/231 | 124/231 |
| VQ encode/decode reconstruction matches | 171/231 | 171/231 |

Code NLL improved on 5 texts and worsened on 15 (one-sided p=0.994091).
**Reject doubling the bridge:** the modest token-count increase does not
compensate for worse held-out likelihood under the predefined joint gate.

Next useful C3 work: diagnose VQ reconstruction and generalization before
further bridge-gain changes. The 171/231 reconstruction result is a diagnostic
reference, not a mathematical upper bound on all generated code sequences.
Generated token matches are positional accuracy, not a human semantic rating.

## Evidence and Limits

- Copies and tests stayed in the dedicated research worktree and K3 directory
  `/opt/clover-k3/astra-weightspace-20261111/c2c3_weights`.
- Before/after tensor hashes verify exactly the declared edits; no-op files
  remained byte-identical. All six copies have byte-exact reversal records.
- The independent auditor recomputed discovery selection and held-out scores,
  checked changed tensor values, and verified 18 source/copy provenance records.
- Weight trials completed in 504.075 seconds on K3 CPU; the natural planner
  usage probe completed in 18.139 seconds. These are run timings, not speed gains.
- The text sets are disjoint within this calibration, but similar-domain short
  notes and not guaranteed absent from the models' original training data.
- No broader planning, multimodal, quantized-device or production improvement
  is claimed. C1 weights and all earlier studies were left unchanged.

See [CONTEXT.md](CONTEXT.md), the frozen protocol, raw results, patch/reversal
audits and `VERIFICATION.json` for the reproducible record. Restored trial paths
contain the original weights, not the rejected candidate weights.

The raw K3 C3 checkpoint is byte-identical to the archived Omni-trained artifact
(SHA-256 `9a03a682b1b614a239f9e6c92276a96ad1e188e2d8bfe4f90c945cd752d16255`).
The evidence-only archive was verified before local extraction:
`9cebb5dc05d2a0e73c5d5a729faaa08a56c6b337e0cd578df9f95d9cc1742530`.