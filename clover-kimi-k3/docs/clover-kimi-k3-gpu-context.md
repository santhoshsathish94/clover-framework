# clover-kimi-k3 GPU context

The dated record of the GPU work, the way `clover-kimi-k3-cpu-context.md` is the record of the CPU engine. Each entry says what was run, on what, what was observed and what it does not cover. Written after every cycle.

## Direction (the creator's, 2026-10-20)

Run the layers on the GPUs with everything a layer needs held in video memory (the layer's trunk files and, for layer 1, all 896 experts), and see how much faster than the CPU engine it is. Start with layer 0 and layer 1. Go one step at a time. The pipeline idea (one layer's experts per card, the next layer's prefetched) was recorded earlier as an idea, not a decision, and is not what this first step tests.

## Where this runs

Rented server `gpu` (141.101.132.38, user `usergpu`, passwordless sudo by the creator's choice): 8 x RTX 3090 24 GiB (23.6 GiB usable as reported by CUDA), 2 x Xeon Gold 6248R (96 threads), 376 GiB RAM, one NVMe (3.25 GB/s sequential best), driver 580.178.04, CUDA toolkit 12.6, `gcc` 13.3. Rented from 20.10.26 00:06 to 27.10.26 01:20. NVLink only between pairs 0-2, 1-3, 4-6, 5-7. All cards on CPU socket 0. Details and measurements are in `clover-kimi-k3-cpu-context.md` (entries of 2026-10-20) because they were taken before this file existed.

## How a GPU result is judged (first candidate, not final)

The CPU engine is bit-identical to the pod's recorded chain end to end. A GPU layer is judged against the CPU engine's rows for the same input, saved by `PF_SAVE_ROWS=base` with `PF_LAST_LAYER=L` (`base.embed`, `base.layer0`, `base.layerL`). The report gives bit-identical share, maximum absolute difference, relative RMS difference and the lowest cosine of a position. Matrix sums on the GPU run in a different order and (for the experts) in float instead of double, so bit-identity is not expected from the matrix products; it is not claimed. No tolerance is fixed yet; the creator sets it.

## 2026-10-20 — layer 0 on one GPU

- Code: `src/code/gpu/layer0.cu`, built on the server with `nvcc -O3 --fmad=false -arch=sm_86` (no fused multiply-add, as the CPU engine is built). Follows `src/code/cpu/layer0.c`: norms, convolution and the recurrent state in the CPU's order; the 8-bit weight matrices through one tiled kernel (128 x 128 x 8 tile, 8 x 8 per thread, the code as a signed integer, row scale applied after the sum, as on the CPU); all 1000 positions of the recorded prompt at once, the recurrence in order over positions with one block per head.
- Weights on the card: 1.44 GB (`trunk-0`: common, kda, dense, operator), copied from the files as they are, 1.33 s including reading them (probably from the page cache; not separated).
- Input: the CPU engine's layer 0 input rows for the recorded 1000-token prompt (bit-identical to the pod's embeddings, 1000 of 1000 rows). Reference: the CPU engine's layer 0 output on the same rows.
- Time on card 0, three runs, layer 0 for 1000 tokens: 228.0, 234.3 and 233.4 ms (stages in the last run: input norm 0.1, q/k/v matrices 47.5, convolution and l2 2.7, decay and beta 4.4, recurrence 14.0, gate and output matrix 32.7, aggregate 0.5, dense three matrices 131.4 ms). Does not include copying the input in or the output out, or loading the weights. The CPU engine on the same server took 1,949 to 2,609 ms for layer 0 in four runs, while the download was running; the GPU runs were taken with the download still running too (it does not use the card). About 8 to 11 times faster than those CPU runs; the matrix kernel is a first, untuned one.
- Against the CPU engine's output (1000 x 7168 values): bit-identical 1.18%, no non-finite values, maximum absolute difference 1.58e-06, relative RMS difference 2.51e-06, lowest cosine of a position 1.000000000, worst position 536 (relative RMS 3.24e-06). One input, one prompt, layer 0 only; not compared with the pod directly (the CPU engine is the reference).
- Not covered: layer 1, any other prompt, other cards, the tolerance the creator wants.
