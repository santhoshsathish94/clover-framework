# gpu: the engine for a GPU server (not started)

Nothing in this folder runs the model yet. It holds the first probes, so that the engine for a GPU server is built against what the machine really is.

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
