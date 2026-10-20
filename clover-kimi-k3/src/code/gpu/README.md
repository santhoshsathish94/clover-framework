# gpu: the engine for a GPU server (started: layers 0 and 1)

The dated record of this work is `../../../docs/clover-kimi-k3-gpu-context.md`. What runs here so far: layer 0 and layer 1 for a window of tokens on one card, with everything the layer needs in video memory, compared with the CPU engine's rows. Nothing else of the model runs on a GPU yet. The probes below read the machine.

## Layers on one card

| file | what it does |
|---|---|
| `kernels.cuh` | the kernels both programs share: normalisation, the 8-bit weight matrix product, the convolution, the recurrent state, the aggregate, the activation |
| `layer0.cu` | layer 0 (attention and the dense MLP). `layer0-gpu TRUNK0_DIR base.embed base.layer0 [REPEATS]` |
| `layer1.cu` | layer 1 (attention, router, 16 of 896 experts per token, shared expert) with all 896 experts on the card, 16.9 GiB of a 24 GiB card for 1000 tokens. `layer1-gpu DATASET_DIR base.embed base.layer0 base.layer1 [REPEATS]` |

Build: `nvcc -O3 --fmad=false -arch=sm_86 -o layer1-gpu layer1.cu`; optional flags: `-DFAST_ROUTER` (float product plus exact double for the candidates, `-lcublas`), `-DEXPERT_V2` (register-tiled expert kernel), `-DINT8_PLANES` (the trunk matrices on the integer tensor cores, `-lcublas`; not the same sums as the float kernel). The first two give the same output bits as without them (the RTX 3090 is sm_86; the server has CUDA 12.6 in `/usr/local/cuda-12.6`). The input and reference rows are written by the CPU engine: `PF_LAST_LAYER=1 PF_SAVE_ROWS=base scripts/run-model.sh recorded` writes `base.embed`, `base.layer0` and `base.layer1` into `src/work/results`. The programs read the dataset files directly and include nothing from `../cpu/`.

## What it may use

- It may include `../common/`. It must never include anything from `../cpu/`.
- The datasets stay where they are, in `src/datasets`, and are shared. If a GPU engine turns out to need a file the CPU engine does not (a different layout of the experts, say), that file gets its own list (`dataset-files-gpu.tsv`) next to the verified CPU list, and the CPU list is not changed.
- The recorded prompts, reference values and compare tools in `src/prompts`, `src/tools` and `src/work/inputs` are shared.

## What it will be checked against

The CPU engine is checked bit for bit, which depends on a fixed summation order and no fused multiply-add. A GPU will not match bit for bit. The gate for this folder is not decided. The first candidate: the same token ids as `src/prompts/expected-exact-ids.tsv` on the recorded prompts, with the per-layer difference from the CPU chain reported, not asserted, until we have seen the figures.

## The machine, as the provider lists it (not yet read from the machine)

Ubuntu 24.04 Server; 8 x RTX 3090 (the listing says "VRAM per card: 192 GB"; NVIDIA gives 24 GB per RTX 3090, so 192 GB is the total); 2 x Intel, 24 cores at 4.0 GHz; 384 GB RAM; "2000 NVMe" (no unit); 10 Gbps port; rented by the week. `probes/machine-readout.sh` reads the real figures and prints them next to the listing.

## An idea, not a decision (the creator, 2026-10-20)

Divide the trunk by 8 and fix each part in one card; move only the experts, in a cycle over the 8 cards, so that no card waits for data: each card processes one layer of 896 experts at a time and, when it finishes, fetches the layer 8 ahead.

Two things to check against the machine's real figures before choosing anything. These are arithmetic on the repository's numbers, not measurements.

- **Memory per card.** One layer's experts are 14.64 GiB. The trunk is 33.25 GiB, 18.3 GiB more with `operator.bin`; an eighth is 4.2 to 6.4 GiB per card. With 24 GiB per card, one layer plus a trunk part leaves about 3 to 5 GiB, so the next layer cannot be fetched whole into the same card while the current one is used. It could wait in RAM (384 GB holds the experts of about 24 to 26 layers) and be streamed in pieces, or each card could hold an eighth of every layer.
- **What a token needs.** A token uses 16 of a layer's 896 experts, 281 MB of the 14.64 GiB. A pass over a long prompt touches many more of a layer's experts (how many, for 1,000 tokens, has not been measured here), so moving whole layers may fit the prompt pass. One token at a time it would move about 98% bytes that are not used, and one token passes through the cards one after another, so only one card works at a time unless several requests or draft windows are in flight.

Neither point rules the idea out. They are the first things to measure.

## Probes

| file | what it does |
|---|---|
| `probes/machine-readout.sh` | reads the machine: CPUs, RAM, GPUs and the links between them, NVMe, network, limits; prints a summary beside the provider's listing. Read only, needs no root. |
| `probes/nvme-read-rate.c` | reads one big file with O_DIRECT for 1, 4 and 16 MiB blocks and 1 to 32 threads, sequential or random, and prints GB/s. Reads only. On the CPU host (two NVMe drives in RAID 1) it gave 3.8 GB/s with one thread and 14.4 GB/s from 4 threads at 4 MiB, the same as the figures recorded earlier (3.94 and 14.08 to 14.44). |
| `probes/sgemm-ceiling.cu` | cuBLAS on the shapes of layer 0's weight matrices (float, and TF32 as a speed reference) with 1000 tokens: what the card does with NVIDIA's own kernel, to read our matrix kernel against. Build: `nvcc -O3 -arch=sm_86 -o sgemm-ceiling sgemm-ceiling.cu -lcublas`. |
| `probes/multi-read.cpp` | files read into RAM as one queue of 64 MiB chunks shared by workers for the local drive (O_DIRECT), Hugging Face (curl ranges) and a second server (ssh stream), with what each delivered by the time the drive finished. Build: `g++ -O2 -std=c++14 -pthread -o multi-read multi-read.cpp`. The second-server stream needs a restricted key that is not set up any more. |
| `probes/ram-copy.c`, `probes/io-legs.cu`, `probes/cycle-io.cu` | what each leg of the data path can move, by thread count and NUMA placement (memory to memory, file to memory, host to card, card to host, card to card), and the cycle's data path on eight cards with a stand-in for the computing. Build: `gcc -O2 -pthread -o ram-copy ram-copy.c`; `nvcc -O3 -std=c++14 -arch=sm_86 -o io-legs io-legs.cu -lpthread` (the same for `cycle-io`). |
| `probes/expert-supply.cu` | how fast one layer's 896 experts (15.72 GB) reach the cards: from the NVMe with O_DIRECT, from the page cache, or from pinned memory over the bus, for 1 to 8 cards at once. Build: `nvcc -O3 -std=c++14 -arch=sm_86 -o expert-supply expert-supply.cu -lpthread`. |
| `probes/gemm-simt-bench.cu`, `probes/gemm-imma-bench.cu`, `probes/bench-common.cuh` | the matrix product benches: the float kernel v1 against v2 (bit-identical), and the integer-digit product on the integer tensor cores against the float kernel, on the real weights of layer 0 and the real embedding rows. Build with `nvcc -O3 --fmad=false -arch=sm_86` (the second also `-lcublas`). |

## First hour on the GPU server

The stop rule (the end date, and what we want from the first day) is the creator's to set. It is not set yet.

1. Connect with an SSH key. The dataset repository is public, so no token is typed or needed.
2. `git clone --depth 1 https://github.com/santhoshsathish94/clover-framework`, then `bash clover-framework/clover-kimi-k3/src/code/gpu/probes/machine-readout.sh /path/on/the/biggest/disk`. Keep the report.
3. Read it against the listing. Stop and decide before going on if: there are not 8 GPUs, or they are not RTX 3090s; the data disk has less than about 1,412 GiB free (the dataset 1,399.8 GiB, the open weights 4.4 GiB, the DSpark checkpoint 7.1 GB); the RAM differs from the listing; or the CPU flags show no `avx512` (then the CPU engine cannot run there, and the baseline is the workstation).
4. Measure the disk first, because the download writes to it and would spoil the figures: `head -c 16G /dev/urandom > /path/on/the/data/disk/scratch.bin` (about half a minute), `gcc -O2 -pthread -o nvme-read-rate .../probes/nvme-read-rate.c`, `./nvme-read-rate /path/on/the/data/disk/scratch.bin`, then delete the scratch file.
5. Start the download in the background (it resumes, and its preflight refuses a disk that is too small): fetch `set-up-new-machine.sh` as in the README, then `sudo nohup bash set-up-new-machine.sh preflight packages clone download open-weights check-dataset > ~/set-up.log 2>&1 &`. Do not run the `build`, `dspark` or `verify` steps yet; whether to run the CPU engine there is decided from step 3.
6. While it downloads: the CUDA version and driver are in the report (`nvcc` missing means no toolkit; installing it is a decision, not an automatic step). Nothing that writes to the data disk.
7. Write what was read into `docs/clover-kimi-k3-cpu-context.md` before the next step.

Next, once the machine is seen and the CUDA version is known: transfer rates (NVMe to RAM to card, card to card), then one layer's expert read and projection on one card.
