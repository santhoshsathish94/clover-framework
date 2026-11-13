# Sequential HybridAgent Weight-Space Study

## Result

Nemotron was completed and audited first, then Kimi K3. **Neither produced an
accepted improvement.** All edited trial checkpoints were restored byte-exactly;
original models and production services were not changed. Gradient training was
not performed. Do not load restored trial paths as if they were improved models.

The earlier LFM3 g0.92 "zero-degradation" claim was overstated. Its raw records
show entry/exit regressions and unmet significance requirements. That correction
is part of this handoff; the old claim is not a basis for promoting these models.

## Nemotron

The 23-tensor C2 checkpoint was tested using its saved three-stage training graph
(Mamba, MoE, attention), not the complete 88-layer model. Four gain changes were
compared on eight frozen diagnostic trajectories; sixteen reserved cases stayed
unopened because no candidate qualified.

| Change | Mean NTP loss | Cases With Worse NTP /8 |
|---|---:|---:|
| Original | 5.595701 | 0 |
| Standard HybridLink 0.92x | 5.595140 | 4 |
| Standard HybridLink 1.08x | 5.601898 | 7 |
| Native HybridLink 0.92x | 5.597499 | 5 |
| Native HybridLink 1.08x | 5.595040 | 5 |

Tiny average reductions did not satisfy per-example non-degradation. The audit
checked 18 source/copy records and all 40 evaluations. All five trial files,
including the no-op, are restored. The current workstation loader may prefer
weights in its bridge checkpoint over the standalone C2 file; verify actual
loaded tensor hashes before any future end-to-end comparison.

## Kimi K3

The native 7,168-dimensional HybridLink was isolated from the real combined
checkpoint; 135-byte LFS pointers were not used as weights. Existing staged data
contained NCM outputs, not C2 inputs, so new states were captured through the
existing CPU engine. The helper explicitly enabled all layers: the library's
default is only four MoE layers despite its `full93` API name.

A private read-only view joined the existing split dataset roots. Twelve
eight-token windows reported all 92 MoE layers active and finite states; the
first window reproduced the separate smoke capture hash exactly. This verifies
execution of that existing runtime, not independent official-model parity.

| Native Output Gain | Discovery Mean MSE | Mean Cosine | Windows Failing Zero Control /4 |
|---|---:|---:|---:|
| Original | 0.000695309 | 0.240348 | 3 |
| 0.92x | 0.000645878 | 0.240348 | 3 |
| 1.08x | 0.000750358 | 0.240348 | 4 |
| Zero vector | 0.000481276 | 0 | Control |

The smaller gain reduced magnitude error but did not beat the zero-vector
control consistently. No candidate reached the eight reserved validation windows.
All three trial files were restored. This is next-embedding prediction, not a
demonstrated planning or end-to-end improvement; the separate Qwen planner was
not loaded.

## Branch and Datasets

Branch: `research/astra-lfm3-weightspace-20261111`.

K3 study root: `/opt/clover-k3/hybridagent-weightspace-20261112`.

Nemotron baseline:
`nemotron/baseline/nemotron_hybridagent_final.pt`

Kimi baseline:
`kimi/baseline/checkpoint-clover-true-hybrid-agent-ncm.pt`

Paths above are relative to the K3 study root. Exact absolute paths, checksums,
component keys, evidence paths and limitations are in [HANDOFF.json](HANDOFF.json).
Both `candidate` fields are deliberately `null`: no improved dataset is ready.
Models and large backbone inputs remain on K3, not in Git.

Evidence archive SHA-256 (verified before local extraction):
`31cc59409adabeca25fa2dfc082de61c60a882631d0be63b679327fed6b77d4f`.

Run the read-only handoff check on K3:

Tested environment: Python 3.12.3, torch 2.11.0+cpu, transformers 5.20.2,
numpy 2.5.4 and safetensors 0.8.0. The existing private venv is reused; no
shared Python environment was modified.

```bash
/opt/clover-k3/astra-weightspace-20261111/venv/bin/python -B \
  /opt/clover-k3/hybridagent-weightspace-20261112/handoff.py
```

The branch supplies the research code, fixed source snapshots and audited
reports. Use a fresh study directory for a rerun: scripts reject existing output
files. Do not overwrite frozen cases, checkpoints or results to obtain a pass.

## Verification Boundaries

- Nemotron data was procedural arithmetic on a reduced stage model, not the
  original training holdout or full-model task validation.
- Kimi windows were reserved within this calibration; original training-set
  membership is unknown. Large expert payloads were mounted read-only rather
  than duplicated or independently content-hashed.
- Correct-position guards, zero controls and fixed source/case hashes prevented
  average-only gains from becoming accepted candidates.
- Baseline checksums and reversal records were independently checked after each
  model. The two model investigations were not run concurrently.
- Neither result is a speed, memory, broad quality, or deployment improvement.

Next work should obtain representative task-level validation and exact loader
contracts before expanding candidate families. See [CONTEXT.md](CONTEXT.md),
[nemotron/CONTEXT.md](nemotron/CONTEXT.md) and [kimi/CONTEXT.md](kimi/CONTEXT.md).