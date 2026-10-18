# clover-kimi-k3

A C engine that runs all 93 layers of Kimi K3 in one process on a CPU host (Ryzen 9 7950X3D, 124 GB RAM, NVMe RAID1): words in, next token and text out.
The prompt is processed in batches, decode keeps recently used experts in RAM, and two options are available: a 4-bit trunk for decode and speculative decoding.

## The final version

`scripts/run-model.sh` with no options. Measured on the host, recorded 1,000-token prompt (745 words):

| | result |
|---|---|
| Whole request, words to next token | 297.5 s and 301.0 s in two measurements (layers 2-92: 296.5 s, layer 0: 1.1 s, layer 1: 3.3 s, head: 0.06 s) |
| Against the pod's own chain | embeddings 1000 of 1000 rows, layer 92 output 1000 of 1000 rows, 8 snapshots 8000 of 8000 rows bit-identical |
| Decode, exact, 16 cached experts per layer | 3.1 - 3.5 s per token |
| Decode, `USE_4BIT_TRUNK=1` | 2.5 - 2.8 s per token (not bit-identical; 8 of 34 recorded prompts give the same 12 tokens as exact, the rest are fluent but differ) |
| Speculative decoding | same tokens as plain decoding; 2.5x faster on a repeating prompt, 21% slower on ordinary code (10% of draft tokens kept) |

## Folder map

| folder | what is in it |
|---|---|
| `src/` | the program, six files: |
| | `words-to-embeddings.c` text to token ids to embedding vectors (stage 1) |
| | `layer0.c` layer 0, the "server" layer: KDA attention plus a dense MLP (stage 2) |
| | `layers-1-to-92.c` layer 1 and, built twice (KDA / MLA), layers 2-92: batched trunk, router, experts, expert cache, state checkpoints (stage 3, `lbk_*`, `lbm_*`) |
| | `trunk-4bit.h` the 4-bit trunk: packing and kernels, shared by `layer0.c` and `layers-1-to-92.c` |
| | `whole-model.c` the driver: setup, prompt pass, decode, speculative decoding, chat mode, final head through the deployed pod library |
| | `ids-to-text.c` small tool: token ids to text |
| `scripts/` | `settings.sh` (all paths, one place) and `run-model.sh` (build and run), then the checks: `check-one-prompt.sh`, `check-1000-token-prompt.sh`, `check-speculative-decoding.sh`, `compare-exact-and-4bit-trunk.sh`, `compare-chat-answers.sh`, `run-34-recorded-prompts.sh`, `measure-multi-token-pass.sh`, `verify-engine-folder.sh`; setup: `set-up-host-folder.sh`, `download-dspark-checkpoint.sh` |
| `tools/` | `compare-outputs.py`, `compare-34-prompts-with-recorded.py`, `expert-route-analysis.py` |
| `prompts/` | the 34 recorded prompts and their recorded results, and three small prompts used by the checks |
| `docs/` | `current-status.md` (the dated log of everything measured, including every negative result) and `layer0-verdict.md` |
| `models/dspark/` | (host only) the Kimi K3 DSpark draft checkpoint, 7.1 GB, not yet used by the engine |
| `work/` | (host only) `build/` the compiled program, `inputs/` tokenizer files, recorded prompt and the pod's chain values, `results/` every log and output |
| `dataset/` | (host only) the model: experts `root-N` (15.7 GB per layer), `trunk-N`, operators, tokenizer files. 1.4 TB. |

On the host the engine lives in `/opt/clover-k3/clover-kimi-k3`. The old path `/opt/clover-k3/clover-intelegence/dataset` is a link to the dataset there.
The deployed pod sources the build reuses (`POD_DIR`, read only) and the C tokenizer (`TOKENIZER_DIR`) stay where they are.

## Running

```
scripts/run-model.sh recorded                          # the 1000-token prompt, compared with the pod's chain
scripts/run-model.sh text prompts/code-is-prime.txt    # any text file
PF_DECODE=48 PF_CACHE=16 scripts/run-model.sh text prompts/code-is-prime.txt                      # 48 more tokens, plain
PF_DECODE=48 PF_CACHE=16 USE_4BIT_TRUNK=1 scripts/run-model.sh text prompts/code-is-prime.txt     # 4-bit trunk in decode
PF_DECODE=48 PF_CACHE=16 SPECULATIVE_DRAFT_TOKENS=7 scripts/run-model.sh text prompts/repeating-numbers.txt
```

Settings (environment): `USE_4BIT_TRUNK` (0/1), `PF_DECODE` (new tokens), `PF_CACHE` (experts kept per layer), `SPECULATIVE_DRAFT_TOKENS` (0-15), `PF_CHAT` (1, low, high, max) with `PF_SYSTEM`,
`PF_SAVE_IDS` (write generated ids), `PF_VERBOSE`, `BUILD=0` (skip build), `CLOVER_LOCK_TRUNK` (0 = do not lock the trunk files in RAM; needed with the 4-bit trunk).
Long runs on the host are started detached (`setsid nohup ... &`) and their log read afterwards.

## Not in this folder

Everything else tried in the month before the final version (micro-benchmarks of expert kernels and lookup tables, the one-layer harnesses, the layer 0 probes, the first three-file
pipeline, the 8-bit-activation evaluation scripts, float dumps) is in `clover-kimi-k3-archive-20261018.zip` next to the repository, 53 files. Their results, all of them negative
or superseded, are recorded in `docs/current-status.md`.
