# clover-kimi-k3

A C engine that runs all 93 layers of Kimi K3 in one process on a CPU host (Ryzen 9 7950X3D, 124 GB RAM, NVMe RAID1): words in, next token and text out.
The prompt is processed in batches, decode keeps recently used experts in RAM, and two options are available: a 4-bit trunk for decode and speculative decoding.

## The final version

`src/scripts/run-model.sh` with no options. Measured on the host, recorded 1,000-token prompt (745 words):

| | result |
|---|---|
| Whole request, words to next token | 297.5 s and 301.0 s in two measurements (layers 2-92: 296.5 s, layer 0: 1.1 s, layer 1: 3.3 s, head: 0.06 s) |
| Against the pod's own chain | embeddings 1000 of 1000 rows, layer 92 output 1000 of 1000 rows, 8 snapshots 8000 of 8000 rows bit-identical |
| Decode, exact, 16 cached experts per layer | 3.1 - 3.5 s per token |
| Decode, `USE_4BIT_TRUNK=1` | 2.5 - 2.8 s per token (not bit-identical; 8 of 34 recorded prompts give the same 12 tokens as exact, the rest are fluent but differ) |
| Speculative decoding, prompt lookup (`SPECULATIVE_DRAFTER=` empty) | same tokens as plain decoding; 2.5x faster on a repeating prompt, 21% slower on ordinary code (10% of draft tokens kept) |
| Speculative decoding, DSpark draft model (`SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8`; the width, 0 to 8 draft tokens, is chosen per pass from how many were right recently), 48 new tokens | same ids as plain decoding on all 3 prompts, both trunks. Exact trunk, s per token plain -> DSpark: code 3.463 -> 2.028, repeating numbers 3.132 -> 0.902, prose 3.212 -> 2.972. With `USE_4BIT_TRUNK=1` (against 4-bit plain): code 2.823 -> 1.924, repeating 2.510 -> 1.214, prose 2.609 -> 2.314. A fixed width of 8 was faster on code (1.793) and 1.5x slower than plain on prose (4.766). One run each |
| Prompt pass with 8-bit activations (experimental, not bit-identical) | experts only 222.0 s; experts and trunk 203.0 s; the 4-bit trunk is not used here (it makes the prompt pass slower). Quality of the trunk part not yet checked |

## Folder map

```
clover-kimi-k3/
├── README.md
├── docs/                              current-status.md (the dated log of everything measured, every negative result), layer0-verdict.md
└── src/
    ├── code/                          all the code: common/ (shared by every machine), cpu/ (the workstation engine), gpu/ (a GPU server: probes only so far), third-party/ (see below)
    │   └── third-party/kimi-k3-in-c/  the tokenizer headers (Apache-2.0, with licence and notice, github.com/FareedKhan-dev/kimi-k3-in-c at commit a2ad8e5, unmodified)
    ├── scripts/                       run-model.sh (build and run), settings.sh (every path, one place), checks, chat, new-machine setup, the two file lists
    ├── tools/                         compare-outputs.py, compare-34-prompts-with-recorded.py, expert-route-analysis.py, prune-trunk.py, check-open-weights.c
    ├── prompts/                       the 34 recorded prompts and their recorded results, expected-exact-ids.tsv, the recorded 1000-token prompt, small prompts
    ├── datasets/                      host only, never committed
    │   ├── third-party/               made by others
    │   │   ├── kimi-k3-open-weights/  three original files of moonshotai/Kimi-K3 under their own names (4.4 GiB)
    │   │   └── kimi-k3-dspark/        the DSpark draft checkpoint, Inferact/Kimi-K3-DSpark (7.1 GB)
    │   └── computed/                  computed for this engine
    │       ├── expert-1 ... expert-92 the experts of each layer (15.7 GB each)
    │       └── trunk-0 ... trunk-92   the other weights of each layer: only the records the engine reads (33.25 GiB for layers 1-92), and operator.bin, the Q/K/V weights (18.3 GiB in all)
    └── work/                          host only: build/ (compiled program), inputs/ (recorded prompt, the pod's chain values), results/ (every log and output)
```

The code in `src/code/`, by what it does. It is split by kind of machine: `common/` has no vector instructions and no OpenMP kernels and includes nothing from `cpu/`; `cpu/` is the workstation engine (AVX-512, OpenMP, huge pages). A different kind of machine gets its own folder beside `cpu/` and shares `common/`; `gpu/` is the first, and holds probes only, no engine yet (see [its README](src/code/gpu/README.md)).

| file | what it is |
|---|---|
| `common/whole-model.c` | the driver: setup, prompt pass, decode, speculative decoding, chat mode, the final stage loaded as a library |
| `common/backend.h` | the boundary: everything the driver calls from a backend (layer 0, the two layer kinds, DSpark, expert memory), written down once; a backend defines these |
| `cpu/cpu-memory.c` | the CPU backend's expert memory: anonymous memory on huge pages, touched at once |
| `common/words-to-embeddings.c` | stage 1: text to token ids to embedding vectors |
| `cpu/layer0-batch.c` | stage 2: layer 0 for a batch of tokens, built on `layer0.c` |
| `cpu/layers-1-to-92-batch.c` | stage 3: layers 1-92 for a batch of tokens, built twice (KDA layers 1-92 except the MLA ones, from `layer-kda.c`; MLA layers, from `layer-mla.c`): batched trunk, router, experts, expert cache, state checkpoints (`lbk_*`, `lbm_*`) |
| `cpu/layer0.c`, `cpu/layer-kda.c`, `cpu/layer-mla.c` | one token through one layer: layer 0, the source of the KDA layers 1-92 (written for layer 2; the layer number is a run-time value), the source of the MLA layers (written for layer 3); the batch files are built on these |
| `cpu/dspark-draft.c` | the DSpark draft model (5 dense MLA layers, Markov head), drafts 8 tokens from the target's hidden states after layers 2, 23, 47, 71, 89 |
| `common/ids-to-text.c` | small tool: token ids to text |
| `cpu/trunk-4bit.h` | the 4-bit trunk: packing and kernels, shared by `layer0-batch.c` and `layers-1-to-92-batch.c` |
| `common/final-stage.c` (with `final-normalization.c`, `token-ends.h`) | the last stage, built as `pipeline-stage.so`: final normalisation, vocabulary head, detokenise |
| `common/`: `embedding-and-head-tables.h`, `safetensors.h`, `sha256.h`, `inflate-simple.h` | the embedding table and output head read from the open weights, the safetensors reader, hashing, the decoder for the table containers |
| `cpu/expert-reader.h`; `common/`: `expert-metadata.h`, `expert-validation.h`, `inflate-fast.h` | reading and checking the experts (`expert-N`) |
| `common/`: `stage-routing.h`, `stage-trace.h`, `generation-request.h`, `deliver-answer.h`, `json-reader.h` | small shared pieces of the earlier pod design |

Most of these files were written for an earlier design with one pod per layer; they are the same code except for reading the open weights directly, for trunk records that may be pruned, and for the names (`root` is now `expert`). Their old names and places are in `git log --follow`.

On the host the engine lives in `/opt/clover-k3/clover-kimi-k3`. Nothing outside this folder is needed to build or run it. The data under `src/datasets` can be linked in from a bigger filesystem (`DATASET_DIR`, `OPEN_WEIGHTS_DIR`, `DSPARK_DIR`).

## Setting up a new machine

```
curl -fsSL https://raw.githubusercontent.com/santhoshsathish94/clover-framework/main/clover-kimi-k3/src/scripts/set-up-new-machine.sh -o set-up-new-machine.sh
sudo bash set-up-new-machine.sh                  # every step; or name steps: preflight packages clone download open-weights check-dataset inputs build dspark verify
```

It checks the machine (Linux x86_64, AVX-512 advised, 124 GB RAM was what it was measured on, about 1.4 TB of local NVMe with O_DIRECT, memory-lock limit), installs the tools, clones only this folder from GitHub into `/opt/clover-k3/clover-framework`, downloads the precomputed dataset from Hugging Face (`HF_REPO`, default `Santhoshsathish94/clover-kimi-k3`, type `dataset`, resumable, `HF_TOKEN` for a private repo) and, from the original `moonshotai/Kimi-K3` at a pinned commit, the three open-weights files the engine reads (4.4 GiB), checks every file against `src/scripts/dataset-files.tsv` and `src/scripts/open-weights-files.tsv`, builds, fetches the DSpark checkpoint and runs `src/scripts/verify-setup.sh`: recorded prompts must give the recorded 12 token ids, DSpark speculative decoding must equal plain decoding, the recorded 1000-token prompt must match the pod's chain bit for bit (when `engine-inputs/refs` is in the dataset repo), and a chat must remember a name from the turn before. Every step can be re-run; the report is `/opt/clover-k3/set-up-new-machine.report`.

The same steps run as a GitHub Actions workflow, `.github/workflows/set-up-clover-kimi-k3.yml` (start it from the Actions tab), on the new machine registered as a self-hosted runner with the label `clover-k3` (its service needs `LimitMEMLOCK=infinity`).

What the Hugging Face repo must contain (its root is the dataset folder): `expert-1` to `expert-92`, `trunk-0` to `trunk-92` (each with its `operator.bin`; `trunk-1` to `trunk-92` pruned with `tools/prune-trunk.py`), and optionally `engine-inputs/refs/` with `embed.f32`, `final.resid.f32`, `final.snaps.f32` (286 MB, the pod's chain for the recorded prompt). Nothing that is in the open weights is in it: the embedding table, the output head, the normalisation tensors and the tokenizer files come from `moonshotai/Kimi-K3`, the DSpark draft model from `Inferact/Kimi-K3-DSpark`.

Publishing that dataset to the repository (the maintainer's side): `src/scripts/upload-dataset-git.sh` pushes it over ssh with git-lfs, in batches small enough for a nearly full disk (one expert layer, or ten trunks, at a time), and can be run again; `src/scripts/keep-uploading.sh` starts it again until every listed file is on the remote and then writes a `COMPLETE` marker; `src/scripts/upload-status.sh` shows how far it is. `src/scripts/upload-dataset.sh` is the same upload with `hf upload-large-folder` for someone who has a token. How it was done and measured is in `docs/current-status.md`.

## Running

```
src/scripts/run-model.sh recorded                          # the 1000-token prompt, compared with the pod's chain
src/scripts/run-model.sh text src/prompts/code-is-prime.txt    # any text file
PF_DECODE=48 PF_CACHE=16 src/scripts/run-model.sh text src/prompts/code-is-prime.txt                      # 48 more tokens, plain
PF_DECODE=48 PF_CACHE=16 USE_4BIT_TRUNK=1 src/scripts/run-model.sh text src/prompts/code-is-prime.txt     # 4-bit trunk in decode
PF_DECODE=48 PF_CACHE=16 SPECULATIVE_DRAFT_TOKENS=7 src/scripts/run-model.sh text src/prompts/repeating-numbers.txt
PF_DECODE=48 PF_CACHE=16 SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8 src/scripts/run-model.sh text src/prompts/code-is-prime.txt   # DSpark draft model
src/scripts/check-dspark-drafter.sh 48                     # plain against DSpark on three prompts, ids must be identical
```

### Chat with several turns

One process keeps the model loaded and the conversation in memory, so a new turn only runs its own tokens through the prompt pass (a short follow-up costs about 30 s plus its reply).

```
src/scripts/chat-start.sh mychat "Answer briefly."     # starts it detached; session-mychat/ready appears when the model is loaded
src/scripts/chat-say.sh mychat "What is a KDA layer?"  # sends one message and waits (WAIT=seconds, 60 by default) for the reply
src/scripts/chat-stop.sh mychat
```

Settings: `PF_CONTEXT` positions the layers hold (8192, about 19 GB of attention cache), `PF_DECODE` longest reply (2500), `PF_CHUNK` tokens per prompt pass (1024), `PF_REPLY_RESERVE` room kept for a reply (1024), `PF_INPUT_MAX` longest message before it is summarised first (half the context), `PF_SUMMARY_TOKENS` longest summary (320). When a new message would leave less than `PF_REPLY_RESERVE` positions for the reply, the oldest turns are replaced by a summary the model writes (kept in `session-NAME/conversation-summary-NNNN.txt`) and the state is rebuilt, which costs a prompt pass over the kept text. `src/scripts/check-chat-session.sh` runs these paths with a small window; its measured results are in `docs/current-status.md`.

Settings (environment): `USE_4BIT_TRUNK` (0/1), `ACTIVATION_8BIT_EXPERT` and `ACTIVATION_8BIT_TRUNK` (0/1, experimental, off by default, prompt pass), `PF_DECODE` (new tokens), `PF_CACHE` (experts kept per layer), `SPECULATIVE_DRAFT_TOKENS` (0-15), `PF_CHAT` (1, low, high, max) with `PF_SYSTEM`,
`PF_SAVE_IDS` (write generated ids), `PF_VERBOSE`, `BUILD=0` (skip build), `CLOVER_LOCK_TRUNK` (0 = do not lock the trunk files in RAM; needed with the 4-bit trunk).
Long runs on the host are started detached (`setsid nohup ... &`) and their log read afterwards.

## Not in this folder

Everything else tried in the month before the final version (micro-benchmarks of expert kernels and lookup tables, the one-layer harnesses, the layer 0 probes, the first three-file
pipeline, the 8-bit-activation evaluation scripts, float dumps) is in `clover-kimi-k3-archive-20261018.zip` next to the repository, 53 files. Their results, all of them negative
or superseded, are recorded in `docs/current-status.md`.
