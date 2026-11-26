# C1/C2/C3 Composite Test

## Outcome

One namespaced checkpoint and custom loader work. The tested original Omni
capabilities are preserved, and the combined entry point matches the explicit
separate-stage pipeline. The strict inference-only C1 -> C2 -> C3 path is **not
an improved question-answering model**: it echoes the input on this diagnostic
set. No installed model was replaced and nothing was deployed.

| Check | Observed result |
|---|---:|
| Tensor round-trip | 1,174 / 1,174 byte-exact |
| Native multimodal parity | 16 / 16 cases |
| Native activation parity | 296 arrays exact |
| Separate/composite pipeline parity | 30 / 30 cases |
| C1/C2 state parity | 177 arrays exact |
| Native Omni diagnostic answers | 24 / 24 correct |
| Strict pipeline diagnostic answers | 0 / 24 correct |
| Original/copy file hashes after testing | 197 / 197 unchanged |

The native parity cases cover text, image, video, speech recognition, nonspeech
audio classification and speech output. Generated speech codes and waveform
hashes match exactly on all four speech cases. This establishes parity on those
cases, not a new general speech-quality claim.

## What The Ablations Show

| Intervention | Changed NCM states /30 | Changed final text /30 |
|---|---:|---:|
| Zero C1 states | 28 | 0 |
| Replace C2 plan with C1 states | 23 | 0 |
| Bypass NCM with zero latent output | 29 | 0 |

All 24 questions were echoed, as were six formatting controls. None of the 29
active pointer windows took its explicit fallback. The frozen decoder limits
its vocabulary to input text plus connectors and word variants. On ordinary
lines it adds a +65 copy bonus and a strong positional alignment bias. Those
rules are consistent with the observed insensitivity to changed neural states.
They were not removed or retuned to improve the reported scores.

Examples: native Omni answers `41` to `What is 14 plus 27?` and `Paris` to the
France-capital question; both pipeline versions return the questions themselves.

## Scope

This assembles the original full LFM3-Omni checkpoint, native speech codec, and
existing K3 C2/bridge/C3 exports. It does not average three models or substitute
the separately fine-tuned C2 transformer into C1. Original and trained token
embeddings are not identical, so lossless whole-backbone sharing is not justified.
All 125,033 shared tokenizer IDs agree; the trained tokenizer adds two latent
markers. Both embedding tables are retained.

The test used native BF16 CPU Omni and the frozen NumPy C2/C3 implementation on
K3, with JEPA, tools and memory excluded equally from both pipeline arms. The
24-question set is diagnostic, not a held-out improvement benchmark. No training,
new weight edit, quantized-device test, speed/memory benefit claim or deployment
is included. Paired activation equality was checked in-process; one matched
copy of each array is archived rather than both.

## Artifacts

- Dedicated worktree: `C:\personal\oss\clover-astra-weightspace-wt`.
- K3 experiment: `/opt/clover-k3/astra-weightspace-20261111/composite`.
- Unified weight file: `package/model.safetensors`, 5,598,321,144 bytes.
- Input provenance: `INPUTS.json`; tensor hashes and compatibility: `package/`.
- Frozen cases/source hashes: `frozen/`; raw outputs and activations: `results/`.
- Independent audit: `VERIFICATION.json`; cycle record: [CONTEXT.md](CONTEXT.md).
- Loader: [model.py](model.py); fixed reference computation: [reference_c2c3.py](reference_c2c3.py).

The checkpoint-only smoke passed with the original input paths inaccessible.
It also passed when launched from the final package directory. The evidence
archive SHA-256 is
`cfe9fcb72abb7933dc1f7d7b3c63c46b42e9b17bd88e50061ed33d42edd043fa`;
the local transfer was verified before extraction, with model weights excluded.
The prototype is retained for research, not promoted as an upgrade. The next
useful experiment should address the plan-to-answer contract and the decoder's
copy constraints before any retraining or backbone substitution is proposed.