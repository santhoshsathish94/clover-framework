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

Next, once the machine is seen and the CUDA version is known: transfer rates (NVMe to RAM to card, card to card), then one layer's expert read and projection on one card.
