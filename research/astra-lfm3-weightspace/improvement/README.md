# LFM3-Omni initial-to-final improvement attempt

## Neuron-count follow-up

The requested larger edited-neuron experiment is complete; see
[neuron_count/RESULTS.md](neuron_count/RESULTS.md). Counts64/128/256/512/1024
were compared with fixed strength and matched total perturbation norm. No wider
edit beat the64-neuron tuning control, so no fresh confirmation was opened and
all nine trial copies were restored. This did not enlarge the model architecture.

## Earlier intervention result

**Outcome: no verified improved checkpoint. Retain the original.**

Eight documented single-component trials were completed, followed by untouched
final validation and multimodal regression. The selected patch failed the
predefined reliability gate and was reversed exactly. No model was deployed,
no original weight was overwritten, and no commit or push was made.

## Initial, candidate and retained final

| Measure | Initial original | Tested candidate | Retained final |
|---|---:|---:|---:|
| Arithmetic choices | 134/288 | 137/288 | 134/288 |
| Arithmetic free response | 61/72 | 61/72 | 61/72 |
| Direct number selection | 96/96 | 96/96 | 96/96 |

The candidate fixed18 and broke15 arithmetic-choice responses. Across72 underlying
questions,12 improved and13 worsened; one-sided exact sign p=0.654981. Four
rotations of a question were not counted as four independent successes.
The +1.04 percentage-point total is not a validated, consistent gain.

Unrelated/multimodal regression introduced no newly broken previously correct
cases, and ASR word error remained0.07. All four speech-code streams changed,
so waveform validity would not establish speech-quality preservation. The
independent speech observer was acquired but not used to rescue the failed
primary gate. No broad or quantized-runtime improvement is claimed.

## What was tested

Discovery isolated arithmetic-to-label sensitivity: addition/subtraction free
responses12/12 correct, but many rotated-choice versions wrong. Direct-selection
controls passed24/24 and matched numeric-binding references96/96.

Directions used only these discovery states. The initial grid tested one MLP
output tensor in layer22 or23,64 selected columns, strengths0.5/1/2. A documented
conditional layer22 follow-up tested4 and6. Strength4 tuned best (79/144 versus
74/144); strength6 regressed (65/144). Final data never selected the candidate.

The tested final patch changed129166 BF16 entries in
`model.language_model.layers.22.feed_forward.w2.weight`. All other tensors,
including embeddings, final norm and output head, remained unchanged. No
gradients, optimizer or conventional training were used.

## Evidence

- [RESULTS.md](RESULTS.md): initial/candidate results and decision.
- [OUTCOME.json](OUTCOME.json): final artifact decision and hashes.
- [FINAL_COMPARISON.json](FINAL_COMPARISON.json): grouped comparison and regression gate.
- [TRIALS.json](TRIALS.json) and [FOLLOWUP_TRIALS.json](FOLLOWUP_TRIALS.json): all eight trials.
- [SELECTION.json](SELECTION.json): candidate frozen before final evaluation.
- [PROTOCOL.md](PROTOCOL.md) and [DOSE_FOLLOWUP.md](DOSE_FOLLOWUP.md): gates and adaptive follow-up.
- [frozen/MANIFEST.json](frozen/MANIFEST.json): frozen cases and evaluation-source hashes.
- [proposals/HYPOTHESIS.md](proposals/HYPOTHESIS.md) and [proposals/evidence.json](proposals/evidence.json): derivation and columns.
- [CONTEXT.md](CONTEXT.md): observed cycles, corrections, limitations and growth.

Raw responses are under `diagnostic/`, `diagnostic-reference/`, and `runs/`.
Copy audits and exact reversal records are under `copies/`; large model/data
files are Git-ignored. The parent `reports/improvement-regression/` contains the
regression rerun. Total new responses:3286, including discovery and regression.

## Model artifacts and verification

K3 study root: `/opt/clover-k3/astra-weightspace-20261111`.
Retained final model: `original/`.
The restored `improvement/copies/final-candidate/` contains ORIGINAL weights,
not the rejected candidate; its audit preserves the tested hash.

Original/retained SHA-256:
`76c0ec2c3e27832fb19f886d5eb0afe37f03c5c32c4a796a7ce08cef119d2763`.
Rejected candidate SHA-256 when tested:
`0e428fe5808efa47e7f588078b8d348b8363001354a500e97df5a61110fa6078`.

Reconstruction uses the preserved parent proposal/directions and DOSE_FOLLOWUP
on a fresh research copy. Never overwrite existing copy names or originals.
The guarded tools require read-only original mounts and verified source hashes.
Use the existing isolated K3 venv to verify saved evidence:

```bash
/opt/clover-k3/astra-weightspace-20261111/venv/bin/python -B \
  /opt/clover-k3/astra-weightspace-20261111/improvement/verify_results.py \
  --root /opt/clover-k3/astra-weightspace-20261111
```

Verified:2964 tuning/final records, frozen sources/cases, grouped recomputation,
all eight single-tensor audits and reversals. Corrupted-file and changed-count
controls verify failure detection. Evidence archive SHA-256:
`eb476cd8803c30fabcccef3c28d0ac2f5eb57d753f762b8aa4ca9307950f6f63`.

## What remains unsolved

The requested better-model outcome was not achieved. The tested association did
not transfer as a robust correction; this does not prove that weight-space
improvement is impossible. Further work needs a different causal hypothesis and
fresh final data. Do not tune against this exposed validation set or relabel a
tuning gain as an improved final version.