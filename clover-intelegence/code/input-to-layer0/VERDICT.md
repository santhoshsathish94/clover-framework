# Input to vector, and layer 0: final verdict (2026-10-17)

Host `k3` (Ryzen 9 7950X3D, 16 cores, 124 GB), deployed tree `/opt/clover-k3/clover-intelegence/code/distrubuted-hosting`, working copies in `/tmp/one-layer`. Rule that governed all work: results stay bit-identical to the original arithmetic. Every figure is one run unless a count is given; run-to-run noise was about 3% where repeated. `clover-one.c` and the deployed tree were not changed.

## 1. Text to token id to vector

- Tokenizer (C, from `tiktoken.model`): the 34 recorded prompts encode to exactly the recorded ids (416 tokens) and decode back to the same text. A 66,758-byte real text became 12,641 tokens in 4.2 ms and decodes back exactly.
- The "rank" in `tiktoken.model` is the token's fixed id (line number minus 1). It never changes with repetition. A repeated word is not merged: "the the the the" is 4 tokens (2108, 276, 276, 276), each its own position. Case and a leading space change the id; a long word can be several tokens.
- Embedding table: 163,840 rows x 7,168 bf16 values = 2.35 GB. A lookup takes 0.0047 ms warm, 0.075 ms cold.
- All 1,000 looked-up vectors are bit-identical to the S0 the pod chain saved. The same id gives the same vector (3,441 of 3,441 position pairs). The 1,000 tokens (492 distinct ids) received exactly 492 distinct vectors. The vector carries no position.

## 2. What layer 0 does to a token

| # | step | depends on |
|---|---|---|
| 1 | normalise the vector | this token only |
| 2 | project to raw Q, K, V (3 x 12,288), beta (96), decay (via 128), output gate (12,288) | this token only |
| 3 | 3-tap convolution, silu, L2 on Q and K | this and the previous 3 tokens |
| 4 | recurrent memory update and read (96 heads x 128 x 128 state) | all earlier tokens |
| 5 | per-head norm, times the gate, output projection back to 7,168 | step 4 |
| 6 | mix the original embedding and the attention result (softmax of two scores) | step 5 and the embedding |
| 7 | dense MLP 7,168 to 33,792 to 7,168, added on | step 6 |

- Source: official `modeling_kimi_linear.py` (`KimiDeltaAttention`, `_apply_attn_res`) and the tech report, equation 1. Our `server.c` follows it.
- Tokens affect each other only forward, through steps 3 and 4. Tested against the recorded 1,000-token output (bit-identical positions): first 500 tokens only gives 500 of 500 unchanged; replacing token 0 gives 0 of 1,000 unchanged; replacing token 500 gives 500 of 1,000 unchanged (I counted matches, not which positions).
- The same id at different positions gives different layer-0 outputs (id 1101, 25 times: cosine with the first occurrence 0.98, then about 0.61). The layer-0 output is nearly unrelated to its input (cosine -0.11 to 0.17).
- The original embedding is used again after layer 0: in step 6 and as the saved copy S0 that layers 1 to 92 mix in.
- A projection is a dot product of the 7,168-number input with a weight row of 8-bit codes, times one scale per row; there is no bias (every matrix stores exactly rows x 4 scale bytes + rows x columns code bytes). Done as 448 blocks of 16 columns with 16 lane totals folded in a fixed order.

## 3. What can be looked up or reused

- Final vectors cannot be tabulated: 0 of 1,000 vectors from the attention output, aggregate, dense output or layer-0 output matched any of the 163,840 embedding rows exactly (control: the embeddings matched 1,000 of 1,000); nearest-row cosine 0.06 to 0.24 over a 100-vector sample.
- Steps 1 and 2 depend on the token only: bit-identical for 508 of 508 repeats in each of 7 groups. Reusing them once per distinct id in a window keeps the output bit-identical.
- A per-id table of those results would be 246,144 bytes per id: 40.3 GB for all ids, about 476 MB for the 1,933 ids in the 12,641-token text (the tokenizer step counted 1,940; not reconciled). Not built: once batched, the token-only work is 0.42 ms/token at most.

## 4. Speed (layer 0 only)

| 1,000 real tokens (492 distinct ids) | ms/token | 1,000 tokens |
|---|---|---|
| one token at a time (server's own function) | 25.5 | 25.5 s |
| batched, old kernel | 2.72 | 2.7 s |
| batched, 4-row x 4-token kernel, window 1,000 | 1.39 | 1.4 s |
| same, plus vectorised recurrent update, built with -O2 | 1.31 | 1.3 s |
| same, built with -O3 | 1.16 | 1.2 s |
| same, built with -O2 -funroll-loops (final) | 1.09 | 1.1 s |

| 10,000 real tokens (1,735 distinct ids) | ms/token | 10,000 tokens |
|---|---|---|
| one token at a time | 25.60 | 256.0 s |
| batched, window 32 | 1.266 | 12.7 s |
| batched, window 1,000 | 1.069 | 10.7 s |
| batched, window 10,000 | 0.990 | 9.9 s |

- Output equal to the reference at every position in every run (1,000 of 1,000 against the recording; 10,000 of 10,000 against the server's own one-at-a-time output).
- Kernel: this host's raw multiply-add peak is 80 GMAC/s per thread, 1,250 GMAC/s on 16 threads. The old kernel (1 row x 4 tokens, AVX2) ran at 341-487; the AVX-512 kernel taking 4 weight rows x 4 tokens per pass ran at 962-1,071 in isolation and 760-860 inside the layer. All 9 variants tested were bit-identical.
- Weight tile (tokens sharing one pass over the weights): best at 16 to 64 (1.371 / 1.418 / 1.441 ms/token at 16 / 32 / 64), worse at 8 (1.709) and above 128 (1.508 at 128, 2.247 at 1,000). Rule: tile 32, window = prompt length (a window beyond about 1,000 only adds sharing of the token-only work). Buffers cost 877,448 bytes per window position.
- Threads: 8 threads 5.02 ms/token, 16 threads 2.64, 32 threads 2.79 (old kernel, window 1,000).
- Build flags matter (1,000 tokens, window 1,000, ms/token, output equal to the recording at 1,000 of 1,000 positions every time): -O2 1.279; -O2 plus inlining flag 1.275; -O2 plus the two vectorising flags 1.288; -O3 1.160; -O3 plus -funroll-loops 1.158; -O2 plus -funroll-loops 1.092; -O2 plus loop peeling and unswitching 1.092. So the gain comes from the compiler unrolling the kernels' loops, not from vectorising or inlining, and -O2 with unrolling beats -O3. The run scripts build with `-O2 -funroll-loops` (`L0_OPT` overrides). At 10,000 tokens the output is equal to the server's own output at 10,000 of 10,000 positions in all three windows. Floating-point flags are the same in every build (`-ffp-contract=off -fno-fast-math`), so the arithmetic does not change; this was checked by the outputs, not by reading the assembly.
- Where the time went before the -O3 build (1,000 tokens, window 1,000, 1.31 ms/token): matrices 1.215 ms (dense gate+up 0.575, dense down 0.316, raw Q/K/V 0.154, attention output 0.112, output gate 0.052); not matrices about 0.07 ms (dense activation 0.035, convolution 0.024, recurrent state 0.013). With -O3 the dense gate+up is 0.504, dense down 0.287, raw Q/K/V 0.141, convolution 0.021, recurrent state 0.012. Estimated headroom is the matrices only.
- Recurrent state update: the 96 heads were already parallel; each head's update is now also vectorised across its 128 independent value columns (16 per instruction, multiply and add kept separate, same order per column). 0.102 to 0.013 ms/token, whole layer 1.393 to 1.313 ms/token on 1,000 tokens. Checked three ways: output equal to the recording at 1,000 of 1,000 positions; output equal to the server's own output at 10,000 of 10,000 positions in windows of 32, 1,000 and 10,000; and scalar against vectorised runs of 2,000 tokens in one program give identical memory state (96 x 128 x 128), identical convolution history and identical outputs. Switch: env `L0_VEC=0/1`.

## 5. Decisions and rejected options

- Kept bit-identical. Column-order summation changes the last bits in 11,938 of 12,288 outputs (RMS error against exact 1.2e-6, against 1.6e-7 for the server's 16-lane order) and ran between 16% slower and 14% faster than the row kernel, so it was dropped. Double-precision accumulation rounded once at the end is 6.7x more accurate but differs from the original in 86% of outputs; not used. A faster, differently rounded version would need your decision plus an end-to-end check that expert routing and next tokens do not change; that check was not run.
- An early benchmark of the column kernel looked 4x slower only because the compiler did not unroll it; found by reading the assembly. Check assembly before trusting a kernel benchmark.
- Software prefetch did not help layer 0 (1,000 tokens, window 1,000, ms/token, output equal to the recording each time): none 1.093; the next row group's weights prefetched into L2 while the current one is computed 1.127; into L1 1.132; the activations prefetched 8 steps ahead 1.121; both 1.139. Each variant was 2.6-4.2% slower. Switches `L0_PF=0/1/2` and `-DPFX=128` (off by default). I did not find out why; the likely reason is that the hardware prefetchers already follow these straight row streams and the extra instructions cost slots.

## 6. Not covered

Other kinds of text; the 24 attention layers; running any of this inside the engine (`clover-one.c`); loading weights from disk. The whole-model time is an estimate only (about 8 minutes per 1,000-token window if layers 1 to 92 ran like batched layer 1 and expert loading were not overlapped), not measured.

## 7. Files (copy to the host `/tmp/one-layer` to run; they read data files already there)

| file | purpose |
|---|---|
| `input-stage.c`, `input-stage-run.sh` | tokenizer and embedding checks (section 1) |
| `tok-show.c` | ids and pieces for chosen strings, repeated-word demo |
| `layer0-observe.c`, `layer0-run.sh [N] [SHOW] [DUMP]` | layer 0 stage by stage with per-position observations and dumps of intermediate vectors |
| `vec-match.c`, `vec-match-run.sh` | hash and nearest-row test against the embedding table |
| `projection-show.c` | one projection with real numbers, summation-order comparison |
| `proj-bench.c`, `proj-bench-run.sh` | raw multiply-add peak and projection kernel variants |
| `layer0-batch.c`, `layer0-batch-run.sh N W1,W2,.. DEDUP [THREADS]` | the batched layer 0 (final version); `window:tile` entries, env `L0_WIDE=0/1`, `L0_VEC=0/1` |
| `layer0-10k-run.sh` | the 10,000-token run (reference `-` = the server's own output) |
