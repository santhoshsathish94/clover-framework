# Astra: LFM3-Omni-2B weight-space pilot

## Latest continuation

The subsequent continuous improvement study is documented in
[improvement/README.md](improvement/README.md). Eight weight-edit trials reached
fresh final validation:134/288 ->137/288 arithmetic-choice answers, but12
underlying questions improved and13 worsened (p=0.655). The candidate was rejected
and reversed; the retained final checkpoint remains the original. This did not
achieve a verified model improvement.

## Earlier pilot

**Recommendation: retain the original checkpoint; reject the tested intervention.**

The original model and waveform codec were inventoried and analyzed read-only on
K3. A discovery-selected coordinate of the final shared RMS gain was attenuated
by 5%, 10% and 20%, separately, in physical copies. The frozen 112-case suite was
run on the original, a no-op copy, and each intervention: 560 case evaluations.
Held-out arithmetic remained **13/32** in every arm, with zero fixed/broken
validation answers. Every experimental copy was restored byte-for-byte.

This is a bounded exploratory pilot, not an exhaustive search over interventions
or a general multimodal benchmark. No model weights were committed or deployed.
The source code, reports, fixtures, and raw traces remain uncommitted in the
dedicated worktree; no commit, merge or push was performed.

## Deliverables

1. [reports/ARCHITECTURE.md](reports/ARCHITECTURE.md) and
   [reports/inventory.json](reports/inventory.json): all 868 stored tensors, shapes,
   dtypes, counts, offsets, architectural components and checkpoint hashes.
2. [reports/WEIGHTS.md](reports/WEIGHTS.md): complete distribution/norm survey,
   qualified spectra and correlations. Raw records:
   [reports/weight-survey-v2/tensors.jsonl](reports/weight-survey-v2/tensors.jsonl).
3. [reports/ACTIVATIONS.md](reports/ACTIVATIONS.md) and
   [reports/activation-analysis.json](reports/activation-analysis.json): observed
   success/failure contrasts, mathematical sensitivity and cohort confounds.
4. [reports/HYPOTHESES.md](reports/HYPOTHESES.md): ranked, pre-intervention H1-H3
   proposals. Only H1 was tested; H2/H3 were not substituted after validation.
5. [patch.py](patch.py) and exact byte specifications:
   [reports/patches/H1-alpha-05.json](reports/patches/H1-alpha-05.json),
   [reports/patches/H1-alpha-10.json](reports/patches/H1-alpha-10.json),
   [reports/patches/H1-alpha-20.json](reports/patches/H1-alpha-20.json).
6. [reports/RESULTS.md](reports/RESULTS.md) and
   [reports/COMPARISON.json](reports/COMPARISON.json): computed before/after results,
   exact paired tests, controls and the final recommendation.
7. [CONTEXT.md](CONTEXT.md): Clover record of direction, evidence, instrument
   failures, outcomes, limits and the next research question.

## Isolation and Identity

- Branch: `research/astra-lfm3-weightspace-20261111`.
- Worktree: `C:\personal\oss\clover-astra-weightspace-wt`.
- Base commit: `2ef6786b7634b5100f08a2d6b43754a89f782217`.
- K3 study: `/opt/clover-k3/astra-weightspace-20261111`.
- Original: `LiquidAI/LFM3-Omni-2B`, revision
  `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.
- Main checkpoint: 665 tensors, 2,166,928,384 stored BF16 parameters. Separate
  codec: 203 tensors, 73,559,090 stored elements, including buffers.
- [SEALED.json](SEALED.json) records every original-file digest;
  [FROZEN.json](FROZEN.json) records inference/scoring source, packages, settings
  and material identities. [upstream.json](upstream.json) pins acquisition.
- K3 jobs used a private venv, 8-CPU quota, 24 GiB limit, no swap, nice 15,
  network-disabled inference and read-only original/fixture mounts. No deployed
  Clover service, other checkout or original checkpoint was modified.

## Evidence and Limits

The no-op matched all 112 cases and 1,912 activation arrays exactly. Every trial
kept 1,680 upstream arrays identical while changing downstream arrays. Only one
BF16 element in one tensor changed, and all other checkpoint files were verified
unchanged. Per-copy audits and reversal proofs are under `copies/`.

The primary edit changed letter logits in 99 cases and all four speech-code
streams, but no task-score improvement was established. Statistical weight
outliers are not defects. The discovery contrast is confounded by answer label:
all eight A-labeled discovery answers passed, versus one of eight C-labeled ones.

Vision/video tests are controlled generated fixtures, not natural-image or
real-world video benchmarks. ASR is six pinned public LibriSpeech utterances
(7% word error in every arm). Speech was checked for valid nonempty waveform
output, not independently rated intelligibility or naturalness. Audio controls
failed 0/2 in every arm. These limits rule out a broad capability claim.

## Verify the Saved Handoff

From this directory, with standard Python only:

```powershell
python verify_handoff.py --self-test
python verify_handoff.py
```

This recomputes artifact hashes and paired validation results. Model execution
and original-weight integrity were verified on K3, not by this local checker.
Large NPZ/WAV/NPY evidence and downloaded reference source are available locally
but ignored by Git; keep the K3 evidence or the hash-verified archive for replay.

## Reproduce on K3

Use a **fresh** research directory. Do not reuse existing output or copy names;
the scripts deliberately refuse to overwrite them.

1. Copy the Python scripts, [PROTOCOL.md](PROTOCOL.md), [constraints.txt](constraints.txt),
   and saved [upstream.json](upstream.json) into the fresh directory. The saved
   upstream metadata forces the same original revision instead of a future `main`.
2. Use an independent copy of the study venv, or reproduce the package versions
   in [FROZEN.json](FROZEN.json). CPU torchvision requires the PyTorch CPU index.
3. Run `prepare.py --root "$ROOT"`; it verifies upstream Git/LFS hashes and seals
   its new original snapshot. Copy the frozen `fixtures/` directory after setup
   to reproduce exactly the same materials. Do not regenerate and silently reseal
   altered fixtures or scoring.
4. Run inventory and survey, then the text/media pilots, before sealing the new
   benchmark. All model-analysis commands must run in the sandbox below.
5. Run `benchmark.py --root "$ROOT" --seal`, then its baseline; run
   `analysis.py --root "$ROOT" --activations` to document the candidate before
   `experiment.py --root "$ROOT"`. Run `finalize.py` only after all reversals.

Representative sandbox invocation, matching the measured settings:

```bash
systemd-run --unit=astra-fresh-baseline \
  --property=Nice=15 --property=CPUQuota=800% \
  --property=MemoryMax=24G --property=MemorySwapMax=0 \
  --property=RuntimeMaxSec=1800 --property=ProtectSystem=strict \
  --property="ReadWritePaths=$ROOT" \
  --property="ReadOnlyPaths=$ROOT/original $ROOT/fixtures" \
  --property=NoNewPrivileges=true --property=CapabilityBoundingSet= \
  --property=PrivateNetwork=true --property=PrivateTmp=true \
  --property="StandardOutput=append:$ROOT/runs/baseline.log" \
  --property=StandardError=inherit \
  --setenv=OMP_NUM_THREADS=8 --setenv=OPENBLAS_NUM_THREADS=8 \
  "$ROOT/venv/bin/python" -B "$ROOT/benchmark.py" --root "$ROOT" --name baseline
```

The original-directory write guard will refuse an unprotected invocation.
Preserve old results when a runtime defect is found; change/fix the instrument
before a new seal and new comparison. Never adjust a held-out criterion to make
an intervention pass. A new coordinate requires a new preregistered question and
fresh validation data, not reuse of this set to rescue H1.