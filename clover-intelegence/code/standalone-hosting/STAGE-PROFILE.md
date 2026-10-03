# Stage-By-Stage Profile

AX102, 2026-10-02, 16 OpenMP threads with close/core affinity. One process per run; three incremental positions: two input positions and one generated continuation. Identical prior token assertions passed. No model activation arrays were saved.

[Workflow](../../../clover-one/workflow.md), [stage definitions](../../../research/k3/model/k3-stages.md), and the shared local /ai/ website were read against actual runtime source. The website is a recorded example, not live timing evidence. Its stage-count claims do not establish runtime speed.

Raw timing-only evidence: [before](stage-profile-before.json), [after](stage-profile-after-corrected.json). All 93 layer totals exist for all three positions (279 per run). Nested projection/operator/detail rows and parallel worker sums must NOT be added to the disjoint stage wall times. Clocks and report bookkeeping add instrumentation overhead; these are observations, not an uninstrumented controlled benchmark.

## Input Boundary

The native tokenizer was loaded once. Repeat means cover 1,000 calls; OS caches were not flushed. Every embedding coordinate matched the original central shard. The standalone service receives IDs; it does not tokenize text itself.

| Operation | First call / setup (ms) | Warm mean (ms) |
|---|---:|---:|
| Tokenizer load | 24.541058 | once |
| Text to IDs | 0.006271 | 0.000371 |
| Seed metadata open | 1.641398 | once |
| Embedding row, input position 0 | 1.769608 | 0.002263 |
| Embedding row, input position 1 | 1.603948 | 0.002309 |

Cold reader use includes compressed-block read, decode and integrity checks. A cached row is BF16 widening and copying. The fraction-of-a-millisecond expectation holds for the measured warm path, not metadata setup or first compressed-block access.

## Whole Positions

| Position | Role | Before (s) | After (s) |
|---:|---|---:|---:|
| 0 | first input, no head | 17.572909 | 16.962130 |
| 1 | last input, first head fill | 32.630372 | 32.361147 |
| 2 | continuation, warm head | 17.023243 | 16.687541 |

Startup: before 5.205087 s; after 5.242405 s. Startup is not charged to an embedding lookup. First-use head fill is shown separately below, not presented as steady token cost.

## Stage Totals At Continuation

All values below are wall milliseconds, summed across layers where applicable. Rows are disjoint; total layer rows are omitted from this table.

| Stage | Before (ms) | After (ms) |
|---|---:|---:|
| binding-and-stored-folds | 0.473343 | 0.452745 |
| pre-attention-aggregation | 1.459211 | 1.485430 |
| snapshot-push | 0.013013 | 0.015024 |
| pre-attention-normalization | 1.030616 | 1.031099 |
| attention | 986.342369 | 1018.025705 |
| attention-residual | 0.311264 | 0.276364 |
| pre-mlp-aggregation-and-normalization | 2.390652 | 2.402749 |
| moe-scratch-and-parameter-bind | 0.221969 | 0.227787 |
| router-and-top16 | 54.623718 | 54.333485 |
| prefetch-submit-and-latent-projection | 55.416938 | 56.085251 |
| experts-mix-normalize-up | 15530.133561 | 15140.517317 |
| shared-expert-and-cleanup | 293.908199 | 314.354385 |
| dense-mlp | 16.587660 | 16.361325 |
| mlp-residual-cache-save-and-unbind | 26.612914 | 26.347752 |
| tail: final-aggregation-and-normalization | 0.026459 | 0.052358 |
| tail: head-fill-and-projection | 51.367868 | 50.909445 |
| tail: argmax-and-cleanup | 0.096600 | 0.100998 |

## Why Time Is Spent

- Binding reads the existing prepared vectors, including fold records 37/38 and KDA decay-base record 39. They are not regenerated per token. Copies and guards cost microseconds here.
- Layer 0 pre-attention aggregation is a guarded copy, not a computed aggregate. Later layers score and mix current residual/snapshot values; those coefficients depend on the current state.
- Attention and dense/shared projections use stored coefficients but still read and multiply their matrices. Their individual Q/K/V/G/O and MLP projection timings are recorded below.
- Routed experts dominate. Their stage includes bounded read-ahead waits, decompression, validating the same decoded-BF16 checksums, gate/up/down projections, input-dependent activation and mixing. The original expert and QKV metadata distinguishes fixed parameters from recorded input-specific outputs.
- Head first use fills the verified BF16 byte cache. Repeated head projections avoid that format work and take about 51 ms in this profile; filling is not a per-token requirement.
- The tested larger Huffman tables did not reduce decode CPU time. That experiment was rejected. Four-literal decoding from one bounded bit window reduced the alternating decoder probe from about 1.378 to 1.04 summed worker-seconds; original outputs/checks are preserved.

This cycle does not establish five-second tokens. It isolates the actual remaining cost instead of treating every stage as slow. Full 128-input/128-output model execution remains unmeasured.

## Every Layer

Continuation-position wall milliseconds. Expert total includes mix/normalization/up; dense/shared is a separate disjoint stage. Worker columns below are NOT part of this wall table.

| Layer | Type | Binding/folds | Pre-attn norm | Attention | Router/top16 | Experts/mix | Dense/shared | Total |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | KDA | 0.005 | 0.013 | 12.151 | 0.000 | 0.000 | 16.361 | 28.913 |
| 1 | KDA | 0.005 | 0.012 | 11.846 | 0.584 | 169.236 | 3.351 | 186.074 |
| 2 | KDA | 0.005 | 0.011 | 12.607 | 0.598 | 169.887 | 3.402 | 187.513 |
| 3 | MLA | 0.005 | 0.011 | 6.017 | 0.632 | 165.800 | 3.361 | 176.516 |
| 4 | KDA | 0.004 | 0.011 | 12.610 | 0.603 | 164.290 | 3.349 | 181.905 |
| 5 | KDA | 0.005 | 0.011 | 12.563 | 0.572 | 157.852 | 3.330 | 175.367 |
| 6 | KDA | 0.005 | 0.011 | 12.551 | 0.583 | 168.286 | 3.415 | 185.892 |
| 7 | MLA | 0.005 | 0.011 | 5.832 | 0.616 | 163.165 | 3.329 | 173.640 |
| 8 | KDA | 0.004 | 0.011 | 12.612 | 0.605 | 167.515 | 3.362 | 185.177 |
| 9 | KDA | 0.005 | 0.011 | 12.468 | 0.573 | 156.598 | 3.355 | 174.005 |
| 10 | KDA | 0.005 | 0.010 | 12.456 | 0.604 | 164.071 | 3.379 | 181.549 |
| 11 | MLA | 0.005 | 0.010 | 5.839 | 0.623 | 150.825 | 3.468 | 161.443 |
| 12 | KDA | 0.004 | 0.011 | 12.502 | 0.582 | 158.482 | 3.328 | 175.930 |
| 13 | KDA | 0.008 | 0.011 | 12.501 | 0.574 | 167.955 | 3.377 | 185.427 |
| 14 | KDA | 0.004 | 0.011 | 12.422 | 0.577 | 162.532 | 3.336 | 179.883 |
| 15 | MLA | 0.004 | 0.010 | 5.845 | 0.616 | 163.139 | 3.352 | 173.654 |
| 16 | KDA | 0.004 | 0.011 | 12.498 | 0.581 | 156.990 | 3.316 | 174.425 |
| 17 | KDA | 0.005 | 0.011 | 12.431 | 0.571 | 157.869 | 3.308 | 175.194 |
| 18 | KDA | 0.005 | 0.011 | 12.377 | 0.571 | 161.016 | 3.336 | 178.315 |
| 19 | MLA | 0.005 | 0.011 | 5.814 | 0.618 | 155.067 | 3.297 | 165.500 |
| 20 | KDA | 0.004 | 0.011 | 12.463 | 0.583 | 155.934 | 3.307 | 173.332 |
| 21 | KDA | 0.005 | 0.011 | 12.468 | 0.570 | 172.555 | 3.383 | 190.005 |
| 22 | KDA | 0.005 | 0.011 | 12.425 | 0.563 | 158.251 | 3.319 | 175.574 |
| 23 | MLA | 0.005 | 0.010 | 5.774 | 0.621 | 162.835 | 3.339 | 173.278 |
| 24 | KDA | 0.004 | 0.011 | 12.480 | 0.582 | 156.821 | 3.316 | 174.247 |
| 25 | KDA | 0.005 | 0.011 | 12.458 | 0.573 | 169.337 | 3.369 | 186.748 |
| 26 | KDA | 0.005 | 0.011 | 12.454 | 0.570 | 160.720 | 3.312 | 178.068 |
| 27 | MLA | 0.005 | 0.010 | 5.851 | 0.621 | 164.768 | 3.348 | 175.288 |
| 28 | KDA | 0.004 | 0.011 | 12.479 | 0.582 | 171.278 | 4.271 | 189.669 |
| 29 | KDA | 0.006 | 0.011 | 15.471 | 0.587 | 162.625 | 3.174 | 183.070 |
| 30 | KDA | 0.005 | 0.011 | 12.285 | 0.569 | 167.276 | 3.330 | 184.500 |
| 31 | MLA | 0.006 | 0.011 | 5.895 | 0.614 | 163.059 | 3.287 | 173.556 |
| 32 | KDA | 0.004 | 0.011 | 12.517 | 0.583 | 166.415 | 3.089 | 183.689 |
| 33 | KDA | 0.006 | 0.012 | 12.321 | 0.574 | 170.750 | 3.919 | 188.681 |
| 34 | KDA | 0.006 | 0.013 | 15.427 | 0.587 | 165.913 | 3.315 | 186.496 |
| 35 | MLA | 0.005 | 0.011 | 5.849 | 0.616 | 171.057 | 3.326 | 181.549 |
| 36 | KDA | 0.004 | 0.011 | 12.495 | 0.585 | 165.262 | 3.333 | 182.726 |
| 37 | KDA | 0.005 | 0.011 | 12.511 | 0.579 | 169.030 | 3.364 | 186.508 |
| 38 | KDA | 0.005 | 0.011 | 12.493 | 0.572 | 157.676 | 3.308 | 175.064 |
| 39 | MLA | 0.004 | 0.011 | 5.840 | 0.613 | 159.745 | 3.335 | 170.245 |
| 40 | KDA | 0.005 | 0.016 | 12.397 | 0.587 | 160.191 | 3.291 | 177.510 |
| 41 | KDA | 0.005 | 0.010 | 12.440 | 0.570 | 164.400 | 3.362 | 181.827 |
| 42 | KDA | 0.004 | 0.011 | 12.480 | 0.575 | 176.198 | 3.356 | 193.649 |
| 43 | MLA | 0.005 | 0.011 | 5.832 | 0.623 | 168.561 | 3.303 | 179.033 |
| 44 | KDA | 0.004 | 0.011 | 12.466 | 0.583 | 171.384 | 3.351 | 188.830 |
| 45 | KDA | 0.005 | 0.011 | 12.503 | 0.572 | 165.054 | 3.275 | 182.439 |
| 46 | KDA | 0.005 | 0.011 | 12.517 | 0.579 | 166.190 | 3.273 | 183.568 |
| 47 | MLA | 0.005 | 0.011 | 5.770 | 0.623 | 168.133 | 3.296 | 178.538 |
| 48 | KDA | 0.004 | 0.011 | 12.449 | 0.582 | 160.995 | 3.230 | 178.299 |
| 49 | KDA | 0.004 | 0.011 | 12.471 | 0.584 | 163.714 | 3.287 | 181.083 |
| 50 | KDA | 0.004 | 0.011 | 12.460 | 0.579 | 169.162 | 4.670 | 187.893 |
| 51 | MLA | 0.005 | 0.011 | 6.051 | 0.626 | 160.141 | 3.240 | 170.771 |
| 52 | KDA | 0.004 | 0.010 | 12.495 | 0.586 | 163.052 | 4.472 | 181.655 |
| 53 | KDA | 0.004 | 0.011 | 12.434 | 0.581 | 159.722 | 3.253 | 177.007 |
| 54 | KDA | 0.004 | 0.011 | 12.461 | 0.585 | 170.007 | 3.306 | 187.378 |
| 55 | MLA | 0.004 | 0.010 | 5.783 | 0.623 | 171.164 | 3.364 | 181.641 |
| 56 | KDA | 0.005 | 0.011 | 12.486 | 0.585 | 159.408 | 3.304 | 176.829 |
| 57 | KDA | 0.005 | 0.011 | 12.541 | 0.577 | 158.156 | 3.300 | 175.604 |
| 58 | KDA | 0.005 | 0.011 | 12.501 | 0.571 | 165.225 | 3.336 | 182.660 |
| 59 | MLA | 0.005 | 0.011 | 5.874 | 0.621 | 156.302 | 3.314 | 166.822 |
| 60 | KDA | 0.004 | 0.011 | 12.514 | 0.583 | 165.394 | 3.332 | 182.877 |
| 61 | KDA | 0.005 | 0.011 | 12.515 | 0.571 | 153.067 | 3.293 | 170.469 |
| 62 | KDA | 0.005 | 0.011 | 12.517 | 0.571 | 169.471 | 4.011 | 187.620 |
| 63 | MLA | 0.005 | 0.011 | 7.119 | 0.638 | 158.780 | 3.276 | 170.701 |
| 64 | KDA | 0.004 | 0.011 | 12.492 | 0.584 | 170.723 | 3.983 | 188.866 |
| 65 | KDA | 0.006 | 0.012 | 15.380 | 0.596 | 153.784 | 3.270 | 174.285 |
| 66 | KDA | 0.004 | 0.011 | 12.490 | 0.576 | 156.576 | 3.290 | 173.954 |
| 67 | MLA | 0.005 | 0.011 | 5.917 | 0.619 | 166.298 | 3.358 | 176.909 |
| 68 | KDA | 0.005 | 0.011 | 12.458 | 0.579 | 162.176 | 3.277 | 179.551 |
| 69 | KDA | 0.004 | 0.011 | 12.460 | 0.570 | 164.179 | 3.305 | 181.546 |
| 70 | KDA | 0.006 | 0.011 | 12.448 | 0.575 | 161.069 | 3.289 | 178.407 |
| 71 | MLA | 0.005 | 0.011 | 5.778 | 0.618 | 166.535 | 3.340 | 176.985 |
| 72 | KDA | 0.004 | 0.012 | 12.408 | 0.584 | 169.507 | 3.291 | 186.837 |
| 73 | KDA | 0.006 | 0.011 | 12.479 | 0.577 | 174.203 | 4.129 | 192.429 |
| 74 | KDA | 0.005 | 0.012 | 15.430 | 0.602 | 171.278 | 3.331 | 191.886 |
| 75 | MLA | 0.005 | 0.010 | 5.761 | 0.619 | 172.576 | 3.327 | 183.006 |
| 76 | KDA | 0.005 | 0.011 | 12.424 | 0.583 | 168.222 | 3.340 | 185.633 |
| 77 | KDA | 0.004 | 0.012 | 12.454 | 0.581 | 173.419 | 4.076 | 191.590 |
| 78 | KDA | 0.005 | 0.011 | 15.489 | 0.605 | 167.117 | 3.311 | 187.798 |
| 79 | MLA | 0.004 | 0.011 | 5.889 | 0.615 | 163.111 | 3.300 | 173.639 |
| 80 | KDA | 0.004 | 0.012 | 12.495 | 0.593 | 165.213 | 3.307 | 182.666 |
| 81 | KDA | 0.004 | 0.011 | 12.447 | 0.572 | 156.086 | 3.310 | 173.456 |
| 82 | KDA | 0.004 | 0.011 | 12.446 | 0.572 | 163.455 | 3.337 | 180.846 |
| 83 | MLA | 0.005 | 0.010 | 5.852 | 0.619 | 174.905 | 3.361 | 185.465 |
| 84 | KDA | 0.005 | 0.012 | 12.411 | 0.579 | 160.035 | 3.304 | 177.399 |
| 85 | KDA | 0.006 | 0.011 | 12.469 | 0.572 | 166.541 | 3.378 | 184.001 |
| 86 | KDA | 0.006 | 0.011 | 12.456 | 0.574 | 155.143 | 3.330 | 172.547 |
| 87 | MLA | 0.006 | 0.011 | 5.825 | 0.615 | 173.089 | 4.161 | 184.412 |
| 88 | KDA | 0.006 | 0.012 | 13.758 | 0.579 | 156.398 | 3.501 | 175.313 |
| 89 | KDA | 0.006 | 0.012 | 12.433 | 0.574 | 175.069 | 4.185 | 193.328 |
| 90 | KDA | 0.006 | 0.011 | 13.220 | 0.573 | 169.986 | 3.350 | 188.180 |
| 91 | MLA | 0.005 | 0.011 | 5.852 | 0.623 | 174.092 | 3.312 | 184.602 |
| 92 | MLA | 0.006 | 0.011 | 5.630 | 0.603 | 173.979 | 3.314 | 184.243 |

## Every Layer Worker Costs

Summed parallel worker milliseconds, not wall milliseconds. Read is the block-access wrapper; asynchronous disk reading occurs in the separate pipeline thread. Its exposed wait is wall time. Neither is a measurement of physical disk bytes.

| Layer | Decode worker ms | CRC worker ms | Math worker ms | Read-ahead wait ms |
|---:|---:|---:|---:|---:|
| 1 | 1052.172 | 591.975 | 388.364 | 7.086 |
| 2 | 1058.661 | 599.334 | 388.953 | 6.429 |
| 3 | 1059.690 | 599.300 | 382.947 | 7.185 |
| 4 | 1060.340 | 600.613 | 388.730 | 2.213 |
| 5 | 1060.856 | 599.448 | 377.735 | 6.815 |
| 6 | 1065.741 | 601.886 | 384.219 | 6.409 |
| 7 | 1061.531 | 599.246 | 383.941 | 7.151 |
| 8 | 1063.288 | 605.920 | 387.505 | 2.385 |
| 9 | 1058.839 | 598.033 | 378.534 | 7.094 |
| 10 | 1061.011 | 602.710 | 383.340 | 2.287 |
| 11 | 1057.427 | 598.234 | 378.143 | 2.226 |
| 12 | 1041.897 | 600.921 | 378.422 | 7.454 |
| 13 | 1063.128 | 605.155 | 384.775 | 6.769 |
| 14 | 1058.429 | 601.276 | 382.823 | 2.041 |
| 15 | 1057.930 | 604.794 | 383.803 | 2.314 |
| 16 | 1051.813 | 600.882 | 381.745 | 2.257 |
| 17 | 1054.082 | 603.477 | 378.547 | 7.317 |
| 18 | 1060.981 | 602.997 | 382.855 | 2.303 |
| 19 | 1055.636 | 599.764 | 381.175 | 2.372 |
| 20 | 1057.144 | 601.251 | 380.815 | 2.298 |
| 21 | 1057.753 | 605.653 | 387.714 | 10.334 |
| 22 | 1055.500 | 602.470 | 380.735 | 2.121 |
| 23 | 1064.061 | 604.881 | 386.322 | 2.251 |
| 24 | 1065.795 | 602.056 | 384.110 | 2.427 |
| 25 | 1075.273 | 606.131 | 384.316 | 7.301 |
| 26 | 1067.766 | 602.836 | 380.650 | 2.069 |
| 27 | 1071.820 | 607.419 | 384.830 | 2.304 |
| 28 | 1076.131 | 605.652 | 385.420 | 7.393 |
| 29 | 1069.221 | 603.998 | 382.105 | 2.271 |
| 30 | 1073.552 | 606.369 | 388.668 | 2.236 |
| 31 | 1068.309 | 607.158 | 384.704 | 2.493 |
| 32 | 1073.733 | 606.405 | 390.080 | 2.336 |
| 33 | 1074.581 | 605.988 | 382.272 | 7.225 |
| 34 | 1070.507 | 605.515 | 383.839 | 2.464 |
| 35 | 1073.923 | 607.774 | 386.084 | 7.182 |
| 36 | 1073.672 | 605.341 | 380.750 | 7.076 |
| 37 | 1078.980 | 610.964 | 384.896 | 6.880 |
| 38 | 1067.155 | 601.850 | 381.503 | 2.240 |
| 39 | 1067.959 | 604.668 | 384.439 | 2.091 |
| 40 | 1065.997 | 603.986 | 381.459 | 2.238 |
| 41 | 1068.949 | 608.656 | 383.760 | 2.216 |
| 42 | 1069.635 | 623.966 | 396.336 | 7.074 |
| 43 | 1070.711 | 608.318 | 383.687 | 3.594 |
| 44 | 1076.815 | 610.958 | 385.677 | 6.429 |
| 45 | 1069.105 | 603.204 | 379.840 | 7.532 |
| 46 | 1071.390 | 608.462 | 386.396 | 2.063 |
| 47 | 1073.238 | 608.457 | 386.441 | 2.249 |
| 48 | 1069.094 | 605.663 | 380.304 | 1.747 |
| 49 | 1070.769 | 607.457 | 383.990 | 2.252 |
| 50 | 1073.680 | 608.959 | 386.044 | 2.268 |
| 51 | 1069.699 | 605.327 | 378.884 | 2.245 |
| 52 | 1069.055 | 608.130 | 380.922 | 2.333 |
| 53 | 1066.614 | 606.316 | 377.345 | 2.392 |
| 54 | 1075.160 | 607.632 | 383.843 | 6.178 |
| 55 | 1084.360 | 615.177 | 390.859 | 7.203 |
| 56 | 1068.069 | 604.965 | 382.046 | 3.062 |
| 57 | 1067.425 | 610.743 | 377.069 | 2.542 |
| 58 | 1074.707 | 608.138 | 384.350 | 2.206 |
| 59 | 1069.300 | 604.003 | 379.412 | 2.101 |
| 60 | 1069.024 | 608.200 | 386.925 | 2.080 |
| 61 | 1058.538 | 603.762 | 379.686 | 2.234 |
| 62 | 1066.027 | 610.645 | 382.450 | 7.356 |
| 63 | 1059.182 | 605.878 | 378.819 | 7.211 |
| 64 | 1066.744 | 608.060 | 385.009 | 7.318 |
| 65 | 1056.796 | 605.473 | 380.460 | 2.138 |
| 66 | 1058.819 | 604.115 | 377.751 | 2.025 |
| 67 | 1065.458 | 608.344 | 383.181 | 7.146 |
| 68 | 1058.329 | 605.119 | 378.988 | 5.287 |
| 69 | 1060.400 | 607.362 | 383.398 | 4.492 |
| 70 | 1058.265 | 605.264 | 382.012 | 1.963 |
| 71 | 1064.621 | 612.154 | 384.698 | 7.328 |
| 72 | 1063.457 | 605.860 | 380.541 | 6.894 |
| 73 | 1070.059 | 611.825 | 387.920 | 6.777 |
| 74 | 1066.183 | 607.951 | 385.099 | 6.772 |
| 75 | 1068.049 | 608.593 | 385.302 | 7.109 |
| 76 | 1063.592 | 604.906 | 380.984 | 7.117 |
| 77 | 1069.679 | 609.101 | 388.923 | 6.862 |
| 78 | 1064.293 | 607.593 | 384.826 | 2.401 |
| 79 | 1062.265 | 606.765 | 381.176 | 1.963 |
| 80 | 1066.374 | 613.103 | 387.479 | 1.869 |
| 81 | 1065.304 | 603.973 | 378.027 | 1.784 |
| 82 | 1073.733 | 608.351 | 384.706 | 2.518 |
| 83 | 1080.189 | 609.918 | 386.963 | 7.278 |
| 84 | 1061.827 | 605.195 | 382.196 | 2.133 |
| 85 | 1065.737 | 606.732 | 382.536 | 7.277 |
| 86 | 1061.551 | 603.265 | 379.020 | 2.564 |
| 87 | 1075.013 | 609.085 | 386.157 | 7.345 |
| 88 | 1065.742 | 603.102 | 378.256 | 2.223 |
| 89 | 1079.291 | 608.551 | 386.744 | 7.355 |
| 90 | 1074.925 | 606.476 | 381.446 | 7.269 |
| 91 | 1081.168 | 612.305 | 388.034 | 7.180 |
| 92 | 1073.273 | 610.109 | 387.417 | 7.234 |

## Detailed Layer Stages

Execution-order wall stages and nested projection details for the continuation position. Only rows listed in the disjoint stage table above can be summed.

### Layer 0

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004979 | undefined |
| pre-attention-aggregation | 0.001393 | undefined |
| snapshot-push | 0.000681 | undefined |
| pre-attention-normalization | 0.013215 | undefined |
| Q | 2.418900 | undefined |
| K | 2.469656 | undefined |
| V | 2.264622 | undefined |
| B | 0.023324 | undefined |
| FA | 0.027902 | undefined |
| FB | 0.048240 | undefined |
| G | 1.965774 | undefined |
| O | 2.037408 | undefined |
| attention | 12.151468 | undefined |
| attention-residual | 0.001794 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024085 | undefined |
| MGATE | 5.488897 | undefined |
| MUP | 5.355488 | undefined |
| MDOWN | 5.478158 | undefined |
| dense-mlp | 16.361325 | undefined |
| mlp-residual-cache-save-and-unbind | 0.353631 | undefined |
| detail:read-ahead-wait | 0.000000 | undefined |
| total:layer | 28.913421 | undefined |

### Layer 1

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004569 | undefined |
| pre-attention-aggregation | 0.015910 | undefined |
| snapshot-push | 0.000110 | undefined |
| pre-attention-normalization | 0.012273 | undefined |
| Q | 2.402379 | undefined |
| K | 2.472189 | undefined |
| V | 2.295239 | undefined |
| B | 0.022041 | undefined |
| FA | 0.027962 | undefined |
| FB | 0.048130 | undefined |
| G | 1.977696 | undefined |
| O | 1.998455 | undefined |
| attention | 11.845717 | undefined |
| attention-residual | 0.002625 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024486 | undefined |
| moe-scratch-and-parameter-bind | 0.003156 | undefined |
| router-and-top16 | 0.584492 | undefined |
| EDOWN | 0.589141 | undefined |
| prefetch-submit-and-latent-projection | 0.589762 | undefined |
| detail:expert-gate | 54.986665 | undefined |
| detail:expert-up | 53.553494 | undefined |
| detail:expert-activation | 0.123199 | undefined |
| detail:expert-down | 52.844744 | undefined |
| EUP | 0.587007 | undefined |
| experts-mix-normalize-up | 169.235975 | undefined |
| SH1 | 1.097110 | undefined |
| SH3 | 1.136223 | undefined |
| SH2 | 1.105536 | undefined |
| shared-expert-and-cleanup | 3.351032 | undefined |
| mlp-residual-cache-save-and-unbind | 0.402643 | undefined |
| detail:read-ahead-wait | 7.085814 | undefined |
| total:layer | 186.074441 | undefined |

### Layer 2

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004919 | undefined |
| pre-attention-aggregation | 0.015079 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010580 | undefined |
| Q | 2.622401 | undefined |
| K | 2.635215 | undefined |
| V | 2.435201 | undefined |
| B | 0.026079 | undefined |
| FA | 0.027411 | undefined |
| FB | 0.047508 | undefined |
| G | 2.117787 | undefined |
| O | 2.068555 | undefined |
| attention | 12.606908 | undefined |
| attention-residual | 0.002504 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025337 | undefined |
| moe-scratch-and-parameter-bind | 0.002455 | undefined |
| router-and-top16 | 0.597587 | undefined |
| EDOWN | 0.588330 | undefined |
| prefetch-submit-and-latent-projection | 0.588921 | undefined |
| detail:expert-gate | 54.454999 | undefined |
| detail:expert-up | 53.177181 | undefined |
| detail:expert-activation | 0.121778 | undefined |
| detail:expert-down | 55.059161 | undefined |
| EUP | 0.588550 | undefined |
| experts-mix-normalize-up | 169.887201 | undefined |
| SH1 | 1.118670 | undefined |
| SH3 | 1.157783 | undefined |
| SH2 | 1.113661 | undefined |
| shared-expert-and-cleanup | 3.402098 | undefined |
| mlp-residual-cache-save-and-unbind | 0.367287 | undefined |
| detail:read-ahead-wait | 6.428573 | undefined |
| total:layer | 187.512508 | undefined |

### Layer 3

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005130 | undefined |
| pre-attention-aggregation | 0.014898 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011351 | undefined |
| QA | 0.273982 | undefined |
| QB | 0.716669 | undefined |
| KA | 0.109995 | undefined |
| KB | 0.317152 | undefined |
| G | 2.271104 | undefined |
| O | 2.218276 | undefined |
| attention | 6.017425 | undefined |
| attention-residual | 0.001653 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025588 | undefined |
| moe-scratch-and-parameter-bind | 0.001994 | undefined |
| router-and-top16 | 0.631691 | undefined |
| EDOWN | 0.635398 | undefined |
| prefetch-submit-and-latent-projection | 0.635888 | undefined |
| detail:expert-gate | 53.580916 | undefined |
| detail:expert-up | 51.459515 | undefined |
| detail:expert-activation | 0.125246 | undefined |
| detail:expert-down | 52.804538 | undefined |
| EUP | 0.590243 | undefined |
| experts-mix-normalize-up | 165.800063 | undefined |
| SH1 | 1.091820 | undefined |
| SH3 | 1.141262 | undefined |
| SH2 | 1.115915 | undefined |
| shared-expert-and-cleanup | 3.361411 | undefined |
| mlp-residual-cache-save-and-unbind | 0.007174 | undefined |
| detail:read-ahead-wait | 7.185436 | undefined |
| total:layer | 176.515959 | undefined |

### Layer 4

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004078 | undefined |
| pre-attention-aggregation | 0.014347 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010510 | undefined |
| Q | 2.603255 | undefined |
| K | 2.569030 | undefined |
| V | 2.420423 | undefined |
| B | 0.021611 | undefined |
| FA | 0.032661 | undefined |
| FB | 0.054703 | undefined |
| G | 2.114842 | undefined |
| O | 2.092731 | undefined |
| attention | 12.610486 | undefined |
| attention-residual | 0.001643 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025097 | undefined |
| moe-scratch-and-parameter-bind | 0.002405 | undefined |
| router-and-top16 | 0.602666 | undefined |
| EDOWN | 0.597406 | undefined |
| prefetch-submit-and-latent-projection | 0.598178 | undefined |
| detail:expert-gate | 53.786782 | undefined |
| detail:expert-up | 51.884128 | undefined |
| detail:expert-activation | 0.125405 | undefined |
| detail:expert-down | 55.629688 | undefined |
| EUP | 0.592257 | undefined |
| experts-mix-normalize-up | 164.290402 | undefined |
| SH1 | 1.092381 | undefined |
| SH3 | 1.129941 | undefined |
| SH2 | 1.113621 | undefined |
| shared-expert-and-cleanup | 3.349059 | undefined |
| mlp-residual-cache-save-and-unbind | 0.393656 | undefined |
| detail:read-ahead-wait | 2.213356 | undefined |
| total:layer | 181.904849 | undefined |

### Layer 5

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004538 | undefined |
| pre-attention-aggregation | 0.015619 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010610 | undefined |
| Q | 2.628402 | undefined |
| K | 2.587315 | undefined |
| V | 2.429300 | undefined |
| B | 0.030738 | undefined |
| FA | 0.031960 | undefined |
| FB | 0.060704 | undefined |
| G | 2.120463 | undefined |
| O | 2.040073 | undefined |
| attention | 12.562967 | undefined |
| attention-residual | 0.002966 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025177 | undefined |
| moe-scratch-and-parameter-bind | 0.002545 | undefined |
| router-and-top16 | 0.572109 | undefined |
| EDOWN | 0.585895 | undefined |
| prefetch-submit-and-latent-projection | 0.586637 | undefined |
| detail:expert-gate | 52.360429 | undefined |
| detail:expert-up | 48.368056 | undefined |
| detail:expert-activation | 0.106529 | undefined |
| detail:expert-down | 49.545738 | undefined |
| EUP | 0.597827 | undefined |
| experts-mix-normalize-up | 157.852201 | undefined |
| SH1 | 1.082462 | undefined |
| SH3 | 1.120855 | undefined |
| SH2 | 1.113801 | undefined |
| shared-expert-and-cleanup | 3.329702 | undefined |
| mlp-residual-cache-save-and-unbind | 0.399807 | undefined |
| detail:read-ahead-wait | 6.814645 | undefined |
| total:layer | 175.367292 | undefined |

### Layer 6

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005300 | undefined |
| pre-attention-aggregation | 0.015108 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.011241 | undefined |
| Q | 2.622591 | undefined |
| K | 2.580472 | undefined |
| V | 2.424771 | undefined |
| B | 0.022151 | undefined |
| FA | 0.033262 | undefined |
| FB | 0.058479 | undefined |
| G | 2.128948 | undefined |
| O | 2.036586 | undefined |
| attention | 12.551275 | undefined |
| attention-residual | 0.002685 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025287 | undefined |
| moe-scratch-and-parameter-bind | 0.003026 | undefined |
| router-and-top16 | 0.582910 | undefined |
| EDOWN | 0.594461 | undefined |
| prefetch-submit-and-latent-projection | 0.595152 | undefined |
| detail:expert-gate | 53.752707 | undefined |
| detail:expert-up | 51.829533 | undefined |
| detail:expert-activation | 0.123830 | undefined |
| detail:expert-down | 55.502989 | undefined |
| EUP | 0.602456 | undefined |
| experts-mix-normalize-up | 168.285508 | undefined |
| SH1 | 1.114102 | undefined |
| SH3 | 1.136674 | undefined |
| SH2 | 1.150890 | undefined |
| shared-expert-and-cleanup | 3.414822 | undefined |
| mlp-residual-cache-save-and-unbind | 0.396771 | undefined |
| detail:read-ahead-wait | 6.408936 | undefined |
| total:layer | 185.891619 | undefined |

### Layer 7

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005130 | undefined |
| pre-attention-aggregation | 0.015228 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010911 | undefined |
| QA | 0.270826 | undefined |
| QB | 0.696813 | undefined |
| KA | 0.105277 | undefined |
| KB | 0.311833 | undefined |
| G | 2.166279 | undefined |
| O | 2.179904 | undefined |
| attention | 5.832299 | undefined |
| attention-residual | 0.002615 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025337 | undefined |
| moe-scratch-and-parameter-bind | 0.002805 | undefined |
| router-and-top16 | 0.616101 | undefined |
| EDOWN | 0.626621 | undefined |
| prefetch-submit-and-latent-projection | 0.627152 | undefined |
| detail:expert-gate | 53.061687 | undefined |
| detail:expert-up | 49.805747 | undefined |
| detail:expert-activation | 0.121778 | undefined |
| detail:expert-down | 52.366587 | undefined |
| EUP | 0.595843 | undefined |
| experts-mix-normalize-up | 163.164729 | undefined |
| SH1 | 1.087413 | undefined |
| SH3 | 1.115545 | undefined |
| SH2 | 1.113090 | undefined |
| shared-expert-and-cleanup | 3.329001 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006031 | undefined |
| detail:read-ahead-wait | 7.151033 | undefined |
| total:layer | 173.639935 | undefined |

### Layer 8

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004058 | undefined |
| pre-attention-aggregation | 0.014267 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011061 | undefined |
| Q | 2.581965 | undefined |
| K | 2.597243 | undefined |
| V | 2.417307 | undefined |
| B | 0.025507 | undefined |
| FA | 0.029024 | undefined |
| FB | 0.059091 | undefined |
| G | 2.112558 | undefined |
| O | 2.062855 | undefined |
| attention | 12.611968 | undefined |
| attention-residual | 0.002575 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026490 | undefined |
| moe-scratch-and-parameter-bind | 0.002274 | undefined |
| router-and-top16 | 0.604580 | undefined |
| EDOWN | 0.612505 | undefined |
| prefetch-submit-and-latent-projection | 0.613336 | undefined |
| detail:expert-gate | 54.588550 | undefined |
| detail:expert-up | 53.764742 | undefined |
| detail:expert-activation | 0.121347 | undefined |
| detail:expert-down | 55.991232 | undefined |
| EUP | 0.598568 | undefined |
| experts-mix-normalize-up | 167.515299 | undefined |
| SH1 | 1.083856 | undefined |
| SH3 | 1.140561 | undefined |
| SH2 | 1.124371 | undefined |
| shared-expert-and-cleanup | 3.361632 | undefined |
| mlp-residual-cache-save-and-unbind | 0.405808 | undefined |
| detail:read-ahead-wait | 2.384868 | undefined |
| total:layer | 185.177004 | undefined |

### Layer 9

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004669 | undefined |
| pre-attention-aggregation | 0.015008 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010640 | undefined |
| Q | 2.608595 | undefined |
| K | 2.586844 | undefined |
| V | 2.366752 | undefined |
| B | 0.029966 | undefined |
| FA | 0.026931 | undefined |
| FB | 0.064360 | undefined |
| G | 2.113800 | undefined |
| O | 2.033961 | undefined |
| attention | 12.467598 | undefined |
| attention-residual | 0.003797 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025278 | undefined |
| moe-scratch-and-parameter-bind | 0.002555 | undefined |
| router-and-top16 | 0.573030 | undefined |
| EDOWN | 0.571659 | undefined |
| prefetch-submit-and-latent-projection | 0.572650 | undefined |
| detail:expert-gate | 51.366006 | undefined |
| detail:expert-up | 46.695850 | undefined |
| detail:expert-activation | 0.111206 | undefined |
| detail:expert-down | 50.662062 | undefined |
| EUP | 0.604460 | undefined |
| experts-mix-normalize-up | 156.598328 | undefined |
| SH1 | 1.093383 | undefined |
| SH3 | 1.117619 | undefined |
| SH2 | 1.131264 | undefined |
| shared-expert-and-cleanup | 3.355331 | undefined |
| mlp-residual-cache-save-and-unbind | 0.372095 | undefined |
| detail:read-ahead-wait | 7.094426 | undefined |
| total:layer | 174.004535 | undefined |

### Layer 10

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005230 | undefined |
| pre-attention-aggregation | 0.014798 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010459 | undefined |
| Q | 2.597915 | undefined |
| K | 2.577977 | undefined |
| V | 2.381570 | undefined |
| B | 0.025197 | undefined |
| FA | 0.035497 | undefined |
| FB | 0.059221 | undefined |
| G | 2.111385 | undefined |
| O | 2.032889 | undefined |
| attention | 12.455766 | undefined |
| attention-residual | 0.002675 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026119 | undefined |
| moe-scratch-and-parameter-bind | 0.002665 | undefined |
| router-and-top16 | 0.604440 | undefined |
| EDOWN | 0.591826 | undefined |
| prefetch-submit-and-latent-projection | 0.592828 | undefined |
| detail:expert-gate | 54.316081 | undefined |
| detail:expert-up | 51.250614 | undefined |
| detail:expert-activation | 0.122730 | undefined |
| detail:expert-down | 55.410346 | undefined |
| EUP | 0.611483 | undefined |
| experts-mix-normalize-up | 164.071273 | undefined |
| SH1 | 1.105526 | undefined |
| SH3 | 1.149979 | undefined |
| SH2 | 1.110115 | undefined |
| shared-expert-and-cleanup | 3.379285 | undefined |
| mlp-residual-cache-save-and-unbind | 0.379379 | undefined |
| detail:read-ahead-wait | 2.287274 | undefined |
| total:layer | 181.548854 | undefined |

### Layer 11

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004939 | undefined |
| pre-attention-aggregation | 0.015338 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.009909 | undefined |
| QA | 0.270786 | undefined |
| QB | 0.695660 | undefined |
| KA | 0.106179 | undefined |
| KB | 0.327472 | undefined |
| G | 2.167671 | undefined |
| O | 2.170365 | undefined |
| attention | 5.839482 | undefined |
| attention-residual | 0.002204 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025447 | undefined |
| moe-scratch-and-parameter-bind | 0.001964 | undefined |
| router-and-top16 | 0.622834 | undefined |
| EDOWN | 0.618806 | undefined |
| prefetch-submit-and-latent-projection | 0.619457 | undefined |
| detail:expert-gate | 51.153693 | undefined |
| detail:expert-up | 45.880658 | undefined |
| detail:expert-activation | 0.110333 | undefined |
| detail:expert-down | 50.561287 | undefined |
| EUP | 0.826826 | undefined |
| experts-mix-normalize-up | 150.824689 | undefined |
| SH1 | 1.220571 | undefined |
| SH3 | 1.111998 | undefined |
| SH2 | 1.121636 | undefined |
| shared-expert-and-cleanup | 3.467580 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005500 | undefined |
| detail:read-ahead-wait | 2.225539 | undefined |
| total:layer | 161.443072 | undefined |

### Layer 12

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.003827 | undefined |
| pre-attention-aggregation | 0.014567 | undefined |
| snapshot-push | 0.001433 | undefined |
| pre-attention-normalization | 0.010970 | undefined |
| Q | 2.623562 | undefined |
| K | 2.526151 | undefined |
| V | 2.361313 | undefined |
| B | 0.021670 | undefined |
| FA | 0.037971 | undefined |
| FB | 0.050224 | undefined |
| G | 2.091919 | undefined |
| O | 2.068025 | undefined |
| attention | 12.502383 | undefined |
| attention-residual | 0.001823 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025398 | undefined |
| moe-scratch-and-parameter-bind | 0.002254 | undefined |
| router-and-top16 | 0.581748 | undefined |
| EDOWN | 0.584412 | undefined |
| prefetch-submit-and-latent-projection | 0.585244 | undefined |
| detail:expert-gate | 51.371731 | undefined |
| detail:expert-up | 47.669696 | undefined |
| detail:expert-activation | 0.113759 | undefined |
| detail:expert-down | 51.199918 | undefined |
| EUP | 0.599601 | undefined |
| experts-mix-normalize-up | 158.481858 | undefined |
| SH1 | 1.075690 | undefined |
| SH3 | 1.127246 | undefined |
| SH2 | 1.110996 | undefined |
| shared-expert-and-cleanup | 3.327748 | undefined |
| mlp-residual-cache-save-and-unbind | 0.386392 | undefined |
| detail:read-ahead-wait | 7.454389 | undefined |
| total:layer | 175.929964 | undefined |

### Layer 13

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.008306 | undefined |
| pre-attention-aggregation | 0.017653 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010589 | undefined |
| Q | 2.603235 | undefined |
| K | 2.563450 | undefined |
| V | 2.392811 | undefined |
| B | 0.033853 | undefined |
| FA | 0.030607 | undefined |
| FB | 0.057607 | undefined |
| G | 2.110524 | undefined |
| O | 2.041065 | undefined |
| attention | 12.500830 | undefined |
| attention-residual | 0.003707 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025137 | undefined |
| moe-scratch-and-parameter-bind | 0.002315 | undefined |
| router-and-top16 | 0.574193 | undefined |
| EDOWN | 0.572590 | undefined |
| prefetch-submit-and-latent-projection | 0.573512 | undefined |
| detail:expert-gate | 54.371344 | undefined |
| detail:expert-up | 51.511901 | undefined |
| detail:expert-activation | 0.118462 | undefined |
| detail:expert-down | 54.492340 | undefined |
| EUP | 0.619998 | undefined |
| experts-mix-normalize-up | 167.955472 | undefined |
| SH1 | 1.094845 | undefined |
| SH3 | 1.133799 | undefined |
| SH2 | 1.134540 | undefined |
| shared-expert-and-cleanup | 3.377111 | undefined |
| mlp-residual-cache-save-and-unbind | 0.373137 | undefined |
| detail:read-ahead-wait | 6.769251 | undefined |
| total:layer | 185.426910 | undefined |

### Layer 14

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004408 | undefined |
| pre-attention-aggregation | 0.015749 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010570 | undefined |
| Q | 2.616330 | undefined |
| K | 2.551888 | undefined |
| V | 2.381400 | undefined |
| B | 0.021981 | undefined |
| FA | 0.034705 | undefined |
| FB | 0.051727 | undefined |
| G | 2.100425 | undefined |
| O | 2.013562 | undefined |
| attention | 12.421682 | undefined |
| attention-residual | 0.002685 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025568 | undefined |
| moe-scratch-and-parameter-bind | 0.002786 | undefined |
| router-and-top16 | 0.576828 | undefined |
| EDOWN | 0.582098 | undefined |
| prefetch-submit-and-latent-projection | 0.582909 | undefined |
| detail:expert-gate | 53.614339 | undefined |
| detail:expert-up | 52.655463 | undefined |
| detail:expert-activation | 0.131405 | undefined |
| detail:expert-down | 53.390121 | undefined |
| EUP | 0.618185 | undefined |
| experts-mix-normalize-up | 162.531645 | undefined |
| SH1 | 1.092652 | undefined |
| SH3 | 1.123450 | undefined |
| SH2 | 1.106418 | undefined |
| shared-expert-and-cleanup | 3.335924 | undefined |
| mlp-residual-cache-save-and-unbind | 0.367517 | undefined |
| detail:read-ahead-wait | 2.041144 | undefined |
| total:layer | 179.882890 | undefined |

### Layer 15

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004207 | undefined |
| pre-attention-aggregation | 0.015459 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010400 | undefined |
| QA | 0.268612 | undefined |
| QB | 0.695039 | undefined |
| KA | 0.106289 | undefined |
| KB | 0.321281 | undefined |
| G | 2.165106 | undefined |
| O | 2.183430 | undefined |
| attention | 5.844963 | undefined |
| attention-residual | 0.002765 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025899 | undefined |
| moe-scratch-and-parameter-bind | 0.002043 | undefined |
| router-and-top16 | 0.616011 | undefined |
| EDOWN | 0.628996 | undefined |
| prefetch-submit-and-latent-projection | 0.629737 | undefined |
| detail:expert-gate | 53.665094 | undefined |
| detail:expert-up | 51.473168 | undefined |
| detail:expert-activation | 0.115855 | undefined |
| detail:expert-down | 54.870519 | undefined |
| EUP | 0.623085 | undefined |
| experts-mix-normalize-up | 163.139282 | undefined |
| SH1 | 1.095928 | undefined |
| SH3 | 1.115494 | undefined |
| SH2 | 1.126896 | undefined |
| shared-expert-and-cleanup | 3.351623 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006813 | undefined |
| detail:read-ahead-wait | 2.314444 | undefined |
| total:layer | 173.653850 | undefined |

### Layer 16

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004158 | undefined |
| pre-attention-aggregation | 0.017533 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011341 | undefined |
| Q | 2.586112 | undefined |
| K | 2.517675 | undefined |
| V | 2.359730 | undefined |
| B | 0.023243 | undefined |
| FA | 0.037730 | undefined |
| FB | 0.062747 | undefined |
| G | 2.118118 | undefined |
| O | 2.064698 | undefined |
| attention | 12.497665 | undefined |
| attention-residual | 0.002995 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024486 | undefined |
| moe-scratch-and-parameter-bind | 0.002405 | undefined |
| router-and-top16 | 0.580895 | undefined |
| EDOWN | 0.586536 | undefined |
| prefetch-submit-and-latent-projection | 0.587298 | undefined |
| detail:expert-gate | 52.251624 | undefined |
| detail:expert-up | 49.504989 | undefined |
| detail:expert-activation | 0.121017 | undefined |
| detail:expert-down | 52.174118 | undefined |
| EUP | 0.601935 | undefined |
| experts-mix-normalize-up | 156.990441 | undefined |
| SH1 | 1.085999 | undefined |
| SH3 | 1.103392 | undefined |
| SH2 | 1.112869 | undefined |
| shared-expert-and-cleanup | 3.315997 | undefined |
| mlp-residual-cache-save-and-unbind | 0.384549 | undefined |
| detail:read-ahead-wait | 2.257116 | undefined |
| total:layer | 174.425172 | undefined |

### Layer 17

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004859 | undefined |
| pre-attention-aggregation | 0.015098 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010529 | undefined |
| Q | 2.600038 | undefined |
| K | 2.538644 | undefined |
| V | 2.368105 | undefined |
| B | 0.022001 | undefined |
| FA | 0.032872 | undefined |
| FB | 0.055494 | undefined |
| G | 2.111566 | undefined |
| O | 2.046284 | undefined |
| attention | 12.430910 | undefined |
| attention-residual | 0.002905 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025287 | undefined |
| moe-scratch-and-parameter-bind | 0.002185 | undefined |
| router-and-top16 | 0.570957 | undefined |
| EDOWN | 0.579964 | undefined |
| prefetch-submit-and-latent-projection | 0.580785 | undefined |
| detail:expert-gate | 51.587912 | undefined |
| detail:expert-up | 47.197909 | undefined |
| detail:expert-activation | 0.109986 | undefined |
| detail:expert-down | 50.978802 | undefined |
| EUP | 0.599811 | undefined |
| experts-mix-normalize-up | 157.869243 | undefined |
| SH1 | 1.075780 | undefined |
| SH3 | 1.113050 | undefined |
| SH2 | 1.105115 | undefined |
| shared-expert-and-cleanup | 3.307661 | undefined |
| mlp-residual-cache-save-and-unbind | 0.368148 | undefined |
| detail:read-ahead-wait | 7.317227 | undefined |
| total:layer | 175.194249 | undefined |

### Layer 18

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004969 | undefined |
| pre-attention-aggregation | 0.015228 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010730 | undefined |
| Q | 2.583417 | undefined |
| K | 2.519658 | undefined |
| V | 2.341505 | undefined |
| B | 0.025247 | undefined |
| FA | 0.037000 | undefined |
| FB | 0.051226 | undefined |
| G | 2.098782 | undefined |
| O | 2.055622 | undefined |
| attention | 12.377159 | undefined |
| attention-residual | 0.003176 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025307 | undefined |
| moe-scratch-and-parameter-bind | 0.002825 | undefined |
| router-and-top16 | 0.571488 | undefined |
| EDOWN | 0.577149 | undefined |
| prefetch-submit-and-latent-projection | 0.578060 | undefined |
| detail:expert-gate | 53.361008 | undefined |
| detail:expert-up | 50.471266 | undefined |
| detail:expert-activation | 0.113040 | undefined |
| detail:expert-down | 54.087666 | undefined |
| EUP | 0.599680 | undefined |
| experts-mix-normalize-up | 161.015533 | undefined |
| SH1 | 1.081410 | undefined |
| SH3 | 1.116717 | undefined |
| SH2 | 1.123469 | undefined |
| shared-expert-and-cleanup | 3.335814 | undefined |
| mlp-residual-cache-save-and-unbind | 0.368629 | undefined |
| detail:read-ahead-wait | 2.302952 | undefined |
| total:layer | 178.314570 | undefined |

### Layer 19

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004609 | undefined |
| pre-attention-aggregation | 0.014667 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010519 | undefined |
| QA | 0.277228 | undefined |
| QB | 0.695499 | undefined |
| KA | 0.109855 | undefined |
| KB | 0.310510 | undefined |
| G | 2.146831 | undefined |
| O | 2.176898 | undefined |
| attention | 5.814025 | undefined |
| attention-residual | 0.002825 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025808 | undefined |
| moe-scratch-and-parameter-bind | 0.001984 | undefined |
| router-and-top16 | 0.618456 | undefined |
| EDOWN | 0.631641 | undefined |
| prefetch-submit-and-latent-projection | 0.632392 | undefined |
| detail:expert-gate | 52.139484 | undefined |
| detail:expert-up | 48.149355 | undefined |
| detail:expert-activation | 0.114011 | undefined |
| detail:expert-down | 51.618570 | undefined |
| EUP | 0.593599 | undefined |
| experts-mix-normalize-up | 155.066595 | undefined |
| SH1 | 1.068337 | undefined |
| SH3 | 1.110555 | undefined |
| SH2 | 1.104093 | undefined |
| shared-expert-and-cleanup | 3.296601 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005981 | undefined |
| detail:read-ahead-wait | 2.371574 | undefined |
| total:layer | 165.500153 | undefined |

### Layer 20

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004408 | undefined |
| pre-attention-aggregation | 0.014838 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011241 | undefined |
| Q | 2.575092 | undefined |
| K | 2.520099 | undefined |
| V | 2.353688 | undefined |
| B | 0.023164 | undefined |
| FA | 0.035957 | undefined |
| FB | 0.050875 | undefined |
| G | 2.102769 | undefined |
| O | 2.080708 | undefined |
| attention | 12.462720 | undefined |
| attention-residual | 0.002755 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024756 | undefined |
| moe-scratch-and-parameter-bind | 0.002454 | undefined |
| router-and-top16 | 0.583020 | undefined |
| EDOWN | 0.590474 | undefined |
| prefetch-submit-and-latent-projection | 0.591605 | undefined |
| detail:expert-gate | 52.010400 | undefined |
| detail:expert-up | 49.213006 | undefined |
| detail:expert-activation | 0.121957 | undefined |
| detail:expert-down | 51.613953 | undefined |
| EUP | 0.591365 | undefined |
| experts-mix-normalize-up | 155.933957 | undefined |
| SH1 | 1.069809 | undefined |
| SH3 | 1.111026 | undefined |
| SH2 | 1.111507 | undefined |
| shared-expert-and-cleanup | 3.306759 | undefined |
| mlp-residual-cache-save-and-unbind | 0.387354 | undefined |
| detail:read-ahead-wait | 2.298456 | undefined |
| total:layer | 173.331919 | undefined |

### Layer 21

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004970 | undefined |
| pre-attention-aggregation | 0.015078 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010560 | undefined |
| Q | 2.596743 | undefined |
| K | 2.542792 | undefined |
| V | 2.373746 | undefined |
| B | 0.022081 | undefined |
| FA | 0.036749 | undefined |
| FB | 0.054702 | undefined |
| G | 2.120683 | undefined |
| O | 2.057064 | undefined |
| attention | 12.468200 | undefined |
| attention-residual | 0.003106 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025066 | undefined |
| moe-scratch-and-parameter-bind | 0.002605 | undefined |
| router-and-top16 | 0.569945 | undefined |
| EDOWN | 0.581827 | undefined |
| prefetch-submit-and-latent-projection | 0.582809 | undefined |
| detail:expert-gate | 53.549902 | undefined |
| detail:expert-up | 51.932940 | undefined |
| detail:expert-activation | 0.127817 | undefined |
| detail:expert-down | 55.922865 | undefined |
| EUP | 0.593308 | undefined |
| experts-mix-normalize-up | 172.555037 | undefined |
| SH1 | 1.105646 | undefined |
| SH3 | 1.144860 | undefined |
| SH2 | 1.118420 | undefined |
| shared-expert-and-cleanup | 3.382992 | undefined |
| mlp-residual-cache-save-and-unbind | 0.378517 | undefined |
| detail:read-ahead-wait | 10.333642 | undefined |
| total:layer | 190.005287 | undefined |

### Layer 22

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005159 | undefined |
| pre-attention-aggregation | 0.015068 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010961 | undefined |
| Q | 2.618894 | undefined |
| K | 2.555646 | undefined |
| V | 2.348679 | undefined |
| B | 0.032241 | undefined |
| FA | 0.033973 | undefined |
| FB | 0.051577 | undefined |
| G | 2.091488 | undefined |
| O | 2.025596 | undefined |
| attention | 12.425200 | undefined |
| attention-residual | 0.003276 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025087 | undefined |
| moe-scratch-and-parameter-bind | 0.002404 | undefined |
| router-and-top16 | 0.563193 | undefined |
| EDOWN | 0.575285 | undefined |
| prefetch-submit-and-latent-projection | 0.576297 | undefined |
| detail:expert-gate | 52.900638 | undefined |
| detail:expert-up | 50.274019 | undefined |
| detail:expert-activation | 0.121176 | undefined |
| detail:expert-down | 52.148121 | undefined |
| EUP | 0.593459 | undefined |
| experts-mix-normalize-up | 158.250876 | undefined |
| SH1 | 1.083375 | undefined |
| SH3 | 1.110706 | undefined |
| SH2 | 1.105546 | undefined |
| shared-expert-and-cleanup | 3.318682 | undefined |
| mlp-residual-cache-save-and-unbind | 0.370442 | undefined |
| detail:read-ahead-wait | 2.121314 | undefined |
| total:layer | 175.573878 | undefined |

### Layer 23

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005040 | undefined |
| pre-attention-aggregation | 0.015088 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010229 | undefined |
| QA | 0.269473 | undefined |
| QB | 0.688958 | undefined |
| KA | 0.105147 | undefined |
| KB | 0.316001 | undefined |
| G | 2.122195 | undefined |
| O | 2.165988 | undefined |
| attention | 5.774320 | undefined |
| attention-residual | 0.002535 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024866 | undefined |
| moe-scratch-and-parameter-bind | 0.002255 | undefined |
| router-and-top16 | 0.620600 | undefined |
| EDOWN | 0.635898 | undefined |
| prefetch-submit-and-latent-projection | 0.636860 | undefined |
| detail:expert-gate | 53.094039 | undefined |
| detail:expert-up | 51.293632 | undefined |
| detail:expert-activation | 0.123593 | undefined |
| detail:expert-down | 55.382949 | undefined |
| EUP | 0.598008 | undefined |
| experts-mix-normalize-up | 162.834612 | undefined |
| SH1 | 1.092712 | undefined |
| SH3 | 1.115735 | undefined |
| SH2 | 1.115515 | undefined |
| shared-expert-and-cleanup | 3.339090 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006112 | undefined |
| detail:read-ahead-wait | 2.251005 | undefined |
| total:layer | 173.278269 | undefined |

### Layer 24

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004378 | undefined |
| pre-attention-aggregation | 0.014617 | undefined |
| snapshot-push | 0.001844 | undefined |
| pre-attention-normalization | 0.011151 | undefined |
| Q | 2.566065 | undefined |
| K | 2.525850 | undefined |
| V | 2.363377 | undefined |
| B | 0.021941 | undefined |
| FA | 0.030146 | undefined |
| FB | 0.050644 | undefined |
| G | 2.116976 | undefined |
| O | 2.062865 | undefined |
| attention | 12.479670 | undefined |
| attention-residual | 0.002915 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024816 | undefined |
| moe-scratch-and-parameter-bind | 0.002515 | undefined |
| router-and-top16 | 0.581607 | undefined |
| EDOWN | 0.586617 | undefined |
| prefetch-submit-and-latent-projection | 0.587678 | undefined |
| detail:expert-gate | 52.167859 | undefined |
| detail:expert-up | 49.308535 | undefined |
| detail:expert-activation | 0.117628 | undefined |
| detail:expert-down | 52.114696 | undefined |
| EUP | 0.591435 | undefined |
| experts-mix-normalize-up | 156.820844 | undefined |
| SH1 | 1.074308 | undefined |
| SH3 | 1.113541 | undefined |
| SH2 | 1.113221 | undefined |
| shared-expert-and-cleanup | 3.315837 | undefined |
| mlp-residual-cache-save-and-unbind | 0.392403 | undefined |
| detail:read-ahead-wait | 2.427237 | undefined |
| total:layer | 174.247359 | undefined |

### Layer 25

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004969 | undefined |
| pre-attention-aggregation | 0.015278 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011110 | undefined |
| Q | 2.593446 | undefined |
| K | 2.530659 | undefined |
| V | 2.374978 | undefined |
| B | 0.026680 | undefined |
| FA | 0.036288 | undefined |
| FB | 0.050254 | undefined |
| G | 2.124721 | undefined |
| O | 2.046304 | undefined |
| attention | 12.458341 | undefined |
| attention-residual | 0.003427 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025578 | undefined |
| moe-scratch-and-parameter-bind | 0.002304 | undefined |
| router-and-top16 | 0.573221 | undefined |
| EDOWN | 0.575505 | undefined |
| prefetch-submit-and-latent-projection | 0.576598 | undefined |
| detail:expert-gate | 54.202842 | undefined |
| detail:expert-up | 51.691145 | undefined |
| detail:expert-activation | 0.126648 | undefined |
| detail:expert-down | 55.318784 | undefined |
| EUP | 0.594871 | undefined |
| experts-mix-normalize-up | 169.337404 | undefined |
| SH1 | 1.105416 | undefined |
| SH3 | 1.137496 | undefined |
| SH2 | 1.111457 | undefined |
| shared-expert-and-cleanup | 3.368775 | undefined |
| mlp-residual-cache-save-and-unbind | 0.363961 | undefined |
| detail:read-ahead-wait | 7.300892 | undefined |
| total:layer | 186.748360 | undefined |

### Layer 26

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004979 | undefined |
| pre-attention-aggregation | 0.015178 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010881 | undefined |
| Q | 2.597003 | undefined |
| K | 2.565183 | undefined |
| V | 2.369048 | undefined |
| B | 0.021460 | undefined |
| FA | 0.036749 | undefined |
| FB | 0.050254 | undefined |
| G | 2.107128 | undefined |
| O | 2.038369 | undefined |
| attention | 12.453783 | undefined |
| attention-residual | 0.003386 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025017 | undefined |
| moe-scratch-and-parameter-bind | 0.002525 | undefined |
| router-and-top16 | 0.569594 | undefined |
| EDOWN | 0.577589 | undefined |
| prefetch-submit-and-latent-projection | 0.578982 | undefined |
| detail:expert-gate | 53.673514 | undefined |
| detail:expert-up | 52.146357 | undefined |
| detail:expert-activation | 0.127387 | undefined |
| detail:expert-down | 52.007659 | undefined |
| EUP | 0.592066 | undefined |
| experts-mix-normalize-up | 160.720151 | undefined |
| SH1 | 1.076512 | undefined |
| SH3 | 1.118851 | undefined |
| SH2 | 1.102250 | undefined |
| shared-expert-and-cleanup | 3.312150 | undefined |
| mlp-residual-cache-save-and-unbind | 0.363890 | undefined |
| detail:read-ahead-wait | 2.069044 | undefined |
| total:layer | 178.068169 | undefined |

### Layer 27

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004529 | undefined |
| pre-attention-aggregation | 0.015128 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010380 | undefined |
| QA | 0.284181 | undefined |
| QB | 0.700099 | undefined |
| KA | 0.109865 | undefined |
| KB | 0.328735 | undefined |
| G | 2.159375 | undefined |
| O | 2.165697 | undefined |
| attention | 5.850923 | undefined |
| attention-residual | 0.002534 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024997 | undefined |
| moe-scratch-and-parameter-bind | 0.002335 | undefined |
| router-and-top16 | 0.621431 | undefined |
| EDOWN | 0.624367 | undefined |
| prefetch-submit-and-latent-projection | 0.625299 | undefined |
| detail:expert-gate | 54.194995 | undefined |
| detail:expert-up | 52.030244 | undefined |
| detail:expert-activation | 0.122262 | undefined |
| detail:expert-down | 55.424822 | undefined |
| EUP | 0.591215 | undefined |
| experts-mix-normalize-up | 164.768495 | undefined |
| SH1 | 1.075810 | undefined |
| SH3 | 1.122878 | undefined |
| SH2 | 1.134620 | undefined |
| shared-expert-and-cleanup | 3.348076 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006262 | undefined |
| detail:read-ahead-wait | 2.303816 | undefined |
| total:layer | 175.288034 | undefined |

### Layer 28

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.003887 | undefined |
| pre-attention-aggregation | 0.014868 | undefined |
| snapshot-push | 0.000030 | undefined |
| pre-attention-normalization | 0.010830 | undefined |
| Q | 2.610007 | undefined |
| K | 2.520470 | undefined |
| V | 2.354620 | undefined |
| B | 0.031900 | undefined |
| FA | 0.027321 | undefined |
| FB | 0.050193 | undefined |
| G | 2.107738 | undefined |
| O | 2.058357 | undefined |
| attention | 12.479050 | undefined |
| attention-residual | 0.002906 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025528 | undefined |
| moe-scratch-and-parameter-bind | 0.002304 | undefined |
| router-and-top16 | 0.581687 | undefined |
| EDOWN | 0.588389 | undefined |
| prefetch-submit-and-latent-projection | 0.589552 | undefined |
| detail:expert-gate | 54.457938 | undefined |
| detail:expert-up | 52.934583 | undefined |
| detail:expert-activation | 0.128873 | undefined |
| detail:expert-down | 55.442999 | undefined |
| EUP | 0.815283 | undefined |
| experts-mix-normalize-up | 171.278391 | undefined |
| SH1 | 1.408683 | undefined |
| SH3 | 1.416948 | undefined |
| SH2 | 1.427748 | undefined |
| shared-expert-and-cleanup | 4.271493 | undefined |
| mlp-residual-cache-save-and-unbind | 0.400378 | undefined |
| detail:read-ahead-wait | 7.392595 | undefined |
| total:layer | 189.668978 | undefined |

### Layer 29

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005561 | undefined |
| pre-attention-aggregation | 0.016872 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010991 | undefined |
| Q | 3.165966 | undefined |
| K | 3.138054 | undefined |
| V | 3.060740 | undefined |
| B | 0.034034 | undefined |
| FA | 0.031459 | undefined |
| FB | 0.051657 | undefined |
| G | 2.696529 | undefined |
| O | 2.630867 | undefined |
| attention | 15.470720 | undefined |
| attention-residual | 0.002725 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025708 | undefined |
| moe-scratch-and-parameter-bind | 0.002545 | undefined |
| router-and-top16 | 0.587277 | undefined |
| EDOWN | 0.767324 | undefined |
| prefetch-submit-and-latent-projection | 0.768556 | undefined |
| detail:expert-gate | 53.780499 | undefined |
| detail:expert-up | 52.033578 | undefined |
| detail:expert-activation | 0.124602 | undefined |
| detail:expert-down | 53.687015 | undefined |
| EUP | 0.620209 | undefined |
| experts-mix-normalize-up | 162.624680 | undefined |
| SH1 | 1.074007 | undefined |
| SH3 | 1.051685 | undefined |
| SH2 | 1.033371 | undefined |
| shared-expert-and-cleanup | 3.174091 | undefined |
| mlp-residual-cache-save-and-unbind | 0.371284 | undefined |
| detail:read-ahead-wait | 2.270612 | undefined |
| total:layer | 183.069526 | undefined |

### Layer 30

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005370 | undefined |
| pre-attention-aggregation | 0.015239 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011482 | undefined |
| Q | 2.593106 | undefined |
| K | 2.516252 | undefined |
| V | 2.316439 | undefined |
| B | 0.030717 | undefined |
| FA | 0.026980 | undefined |
| FB | 0.051076 | undefined |
| G | 2.056894 | undefined |
| O | 2.042888 | undefined |
| attention | 12.285408 | undefined |
| attention-residual | 0.003356 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024726 | undefined |
| moe-scratch-and-parameter-bind | 0.002765 | undefined |
| router-and-top16 | 0.569424 | undefined |
| EDOWN | 0.571879 | undefined |
| prefetch-submit-and-latent-projection | 0.573141 | undefined |
| detail:expert-gate | 54.408756 | undefined |
| detail:expert-up | 53.728993 | undefined |
| detail:expert-activation | 0.134896 | undefined |
| detail:expert-down | 56.032717 | undefined |
| EUP | 0.615140 | undefined |
| experts-mix-normalize-up | 167.275962 | undefined |
| SH1 | 1.108321 | undefined |
| SH3 | 1.116376 | undefined |
| SH2 | 1.090758 | undefined |
| shared-expert-and-cleanup | 3.330203 | undefined |
| mlp-residual-cache-save-and-unbind | 0.394116 | undefined |
| detail:read-ahead-wait | 2.235840 | undefined |
| total:layer | 184.499888 | undefined |

### Layer 31

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005700 | undefined |
| pre-attention-aggregation | 0.015609 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010730 | undefined |
| QA | 0.264044 | undefined |
| QB | 0.698355 | undefined |
| KA | 0.103454 | undefined |
| KB | 0.314598 | undefined |
| G | 2.242190 | undefined |
| O | 2.165937 | undefined |
| attention | 5.894585 | undefined |
| attention-residual | 0.002555 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025096 | undefined |
| moe-scratch-and-parameter-bind | 0.002374 | undefined |
| router-and-top16 | 0.614178 | undefined |
| EDOWN | 0.623054 | undefined |
| prefetch-submit-and-latent-projection | 0.624027 | undefined |
| detail:expert-gate | 53.700880 | undefined |
| detail:expert-up | 52.600998 | undefined |
| detail:expert-activation | 0.128692 | undefined |
| detail:expert-down | 53.423774 | undefined |
| EUP | 0.595903 | undefined |
| experts-mix-normalize-up | 163.059071 | undefined |
| SH1 | 1.065742 | undefined |
| SH3 | 1.107519 | undefined |
| SH2 | 1.098603 | undefined |
| shared-expert-and-cleanup | 3.287103 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005691 | undefined |
| detail:read-ahead-wait | 2.493141 | undefined |
| total:layer | 173.555536 | undefined |

### Layer 32

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004338 | undefined |
| pre-attention-aggregation | 0.014727 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011351 | undefined |
| Q | 2.565634 | undefined |
| K | 2.533384 | undefined |
| V | 2.364228 | undefined |
| B | 0.023243 | undefined |
| FA | 0.029054 | undefined |
| FB | 0.051315 | undefined |
| G | 2.118509 | undefined |
| O | 2.075368 | undefined |
| attention | 12.517191 | undefined |
| attention-residual | 0.002554 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024866 | undefined |
| moe-scratch-and-parameter-bind | 0.002505 | undefined |
| router-and-top16 | 0.582980 | undefined |
| EDOWN | 0.593058 | undefined |
| prefetch-submit-and-latent-projection | 0.594221 | undefined |
| detail:expert-gate | 53.707665 | undefined |
| detail:expert-up | 52.997355 | undefined |
| detail:expert-activation | 0.127911 | undefined |
| detail:expert-down | 56.513739 | undefined |
| EUP | 0.618446 | undefined |
| experts-mix-normalize-up | 166.414632 | undefined |
| SH1 | 1.043309 | undefined |
| SH3 | 1.021699 | undefined |
| SH2 | 1.008946 | undefined |
| shared-expert-and-cleanup | 3.088672 | undefined |
| mlp-residual-cache-save-and-unbind | 0.421978 | undefined |
| detail:read-ahead-wait | 2.335653 | undefined |
| total:layer | 183.689274 | undefined |

### Layer 33

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005911 | undefined |
| pre-attention-aggregation | 0.016671 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.012203 | undefined |
| Q | 2.578899 | undefined |
| K | 2.450119 | undefined |
| V | 2.286954 | undefined |
| B | 0.025617 | undefined |
| FA | 0.033663 | undefined |
| FB | 0.051536 | undefined |
| G | 2.036095 | undefined |
| O | 2.048138 | undefined |
| attention | 12.320864 | undefined |
| attention-residual | 0.003246 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024757 | undefined |
| moe-scratch-and-parameter-bind | 0.002394 | undefined |
| router-and-top16 | 0.573812 | undefined |
| EDOWN | 0.580004 | undefined |
| prefetch-submit-and-latent-projection | 0.581497 | undefined |
| detail:expert-gate | 54.315501 | undefined |
| detail:expert-up | 53.462956 | undefined |
| detail:expert-activation | 0.132698 | undefined |
| detail:expert-down | 54.699789 | undefined |
| EUP | 0.789725 | undefined |
| experts-mix-normalize-up | 170.749603 | undefined |
| SH1 | 1.324395 | undefined |
| SH3 | 1.297325 | undefined |
| SH2 | 1.279281 | undefined |
| shared-expert-and-cleanup | 3.919274 | undefined |
| mlp-residual-cache-save-and-unbind | 0.460290 | undefined |
| detail:read-ahead-wait | 7.224564 | undefined |
| total:layer | 188.681102 | undefined |

### Layer 34

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005841 | undefined |
| pre-attention-aggregation | 0.017282 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.012984 | undefined |
| Q | 3.070197 | undefined |
| K | 3.091988 | undefined |
| V | 2.969930 | undefined |
| B | 0.027241 | undefined |
| FA | 0.032792 | undefined |
| FB | 0.058850 | undefined |
| G | 2.719773 | undefined |
| O | 2.639713 | undefined |
| attention | 15.427239 | undefined |
| attention-residual | 0.003517 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025818 | undefined |
| moe-scratch-and-parameter-bind | 0.002575 | undefined |
| router-and-top16 | 0.587097 | undefined |
| EDOWN | 0.800165 | undefined |
| prefetch-submit-and-latent-projection | 0.801397 | undefined |
| detail:expert-gate | 53.980533 | undefined |
| detail:expert-up | 53.475382 | undefined |
| detail:expert-activation | 0.133240 | undefined |
| detail:expert-down | 55.110253 | undefined |
| EUP | 0.626401 | undefined |
| experts-mix-normalize-up | 165.912694 | undefined |
| SH1 | 1.101378 | undefined |
| SH3 | 1.157603 | undefined |
| SH2 | 1.038831 | undefined |
| shared-expert-and-cleanup | 3.314905 | undefined |
| mlp-residual-cache-save-and-unbind | 0.374260 | undefined |
| detail:read-ahead-wait | 2.463731 | undefined |
| total:layer | 186.495657 | undefined |

### Layer 35

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005130 | undefined |
| pre-attention-aggregation | 0.015459 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010690 | undefined |
| QA | 0.267389 | undefined |
| QB | 0.684229 | undefined |
| KA | 0.105617 | undefined |
| KB | 0.321531 | undefined |
| G | 2.170195 | undefined |
| O | 2.193208 | undefined |
| attention | 5.849481 | undefined |
| attention-residual | 0.002675 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025318 | undefined |
| moe-scratch-and-parameter-bind | 0.001793 | undefined |
| router-and-top16 | 0.616391 | undefined |
| EDOWN | 0.623455 | undefined |
| prefetch-submit-and-latent-projection | 0.624437 | undefined |
| detail:expert-gate | 53.952903 | undefined |
| detail:expert-up | 52.803363 | undefined |
| detail:expert-activation | 0.135643 | undefined |
| detail:expert-down | 56.265222 | undefined |
| EUP | 0.592427 | undefined |
| experts-mix-normalize-up | 171.056597 | undefined |
| SH1 | 1.083625 | undefined |
| SH3 | 1.119773 | undefined |
| SH2 | 1.107429 | undefined |
| shared-expert-and-cleanup | 3.326166 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005661 | undefined |
| detail:read-ahead-wait | 7.181959 | undefined |
| total:layer | 181.549495 | undefined |

### Layer 36

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004008 | undefined |
| pre-attention-aggregation | 0.014968 | undefined |
| snapshot-push | 0.001653 | undefined |
| pre-attention-normalization | 0.011221 | undefined |
| Q | 2.585442 | undefined |
| K | 2.539856 | undefined |
| V | 2.351244 | undefined |
| B | 0.024997 | undefined |
| FA | 0.031459 | undefined |
| FB | 0.050514 | undefined |
| G | 2.100495 | undefined |
| O | 2.065149 | undefined |
| attention | 12.495461 | undefined |
| attention-residual | 0.002074 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025117 | undefined |
| moe-scratch-and-parameter-bind | 0.002815 | undefined |
| router-and-top16 | 0.584713 | undefined |
| EDOWN | 0.589562 | undefined |
| prefetch-submit-and-latent-projection | 0.590904 | undefined |
| detail:expert-gate | 53.474810 | undefined |
| detail:expert-up | 51.348647 | undefined |
| detail:expert-activation | 0.130212 | undefined |
| detail:expert-down | 52.522408 | undefined |
| EUP | 0.588510 | undefined |
| experts-mix-normalize-up | 165.261748 | undefined |
| SH1 | 1.090598 | undefined |
| SH3 | 1.121486 | undefined |
| SH2 | 1.106187 | undefined |
| shared-expert-and-cleanup | 3.333009 | undefined |
| mlp-residual-cache-save-and-unbind | 0.388156 | undefined |
| detail:read-ahead-wait | 7.075982 | undefined |
| total:layer | 182.725963 | undefined |

### Layer 37

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005000 | undefined |
| pre-attention-aggregation | 0.015379 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011060 | undefined |
| Q | 2.624114 | undefined |
| K | 2.563270 | undefined |
| V | 2.355581 | undefined |
| B | 0.021760 | undefined |
| FA | 0.032611 | undefined |
| FB | 0.048600 | undefined |
| G | 2.099724 | undefined |
| O | 2.063927 | undefined |
| attention | 12.511450 | undefined |
| attention-residual | 0.003095 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025498 | undefined |
| moe-scratch-and-parameter-bind | 0.002565 | undefined |
| router-and-top16 | 0.578822 | undefined |
| EDOWN | 0.574944 | undefined |
| prefetch-submit-and-latent-projection | 0.576256 | undefined |
| detail:expert-gate | 53.657311 | undefined |
| detail:expert-up | 52.072468 | undefined |
| detail:expert-activation | 0.126296 | undefined |
| detail:expert-down | 55.580394 | undefined |
| EUP | 0.588529 | undefined |
| experts-mix-normalize-up | 169.030119 | undefined |
| SH1 | 1.108482 | undefined |
| SH3 | 1.135281 | undefined |
| SH2 | 1.105075 | undefined |
| shared-expert-and-cleanup | 3.364157 | undefined |
| mlp-residual-cache-save-and-unbind | 0.374490 | undefined |
| detail:read-ahead-wait | 6.880076 | undefined |
| total:layer | 186.508291 | undefined |

### Layer 38

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005029 | undefined |
| pre-attention-aggregation | 0.015699 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.011191 | undefined |
| Q | 2.592504 | undefined |
| K | 2.569842 | undefined |
| V | 2.382643 | undefined |
| B | 0.025478 | undefined |
| FA | 0.036939 | undefined |
| FB | 0.048431 | undefined |
| G | 2.102559 | undefined |
| O | 2.051384 | undefined |
| attention | 12.492595 | undefined |
| attention-residual | 0.003106 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025558 | undefined |
| moe-scratch-and-parameter-bind | 0.002705 | undefined |
| router-and-top16 | 0.572099 | undefined |
| EDOWN | 0.579032 | undefined |
| prefetch-submit-and-latent-projection | 0.580344 | undefined |
| detail:expert-gate | 52.676879 | undefined |
| detail:expert-up | 49.789241 | undefined |
| detail:expert-activation | 0.117682 | undefined |
| detail:expert-down | 52.144573 | undefined |
| EUP | 0.590123 | undefined |
| experts-mix-normalize-up | 157.675531 | undefined |
| SH1 | 1.064118 | undefined |
| SH3 | 1.116376 | undefined |
| SH2 | 1.112109 | undefined |
| shared-expert-and-cleanup | 3.307782 | undefined |
| mlp-residual-cache-save-and-unbind | 0.360955 | undefined |
| detail:read-ahead-wait | 2.240198 | undefined |
| total:layer | 175.063665 | undefined |

### Layer 39

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004378 | undefined |
| pre-attention-aggregation | 0.015259 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.010850 | undefined |
| QA | 0.274002 | undefined |
| QB | 0.699798 | undefined |
| KA | 0.109244 | undefined |
| KB | 0.328444 | undefined |
| G | 2.155027 | undefined |
| O | 2.172360 | undefined |
| attention | 5.839913 | undefined |
| attention-residual | 0.002465 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025798 | undefined |
| moe-scratch-and-parameter-bind | 0.001864 | undefined |
| router-and-top16 | 0.612866 | undefined |
| EDOWN | 0.633374 | undefined |
| prefetch-submit-and-latent-projection | 0.634446 | undefined |
| detail:expert-gate | 52.896829 | undefined |
| detail:expert-up | 49.459990 | undefined |
| detail:expert-activation | 0.112742 | undefined |
| detail:expert-down | 54.492591 | undefined |
| EUP | 0.583700 | undefined |
| experts-mix-normalize-up | 159.745069 | undefined |
| SH1 | 1.072985 | undefined |
| SH3 | 1.124451 | undefined |
| SH2 | 1.122448 | undefined |
| shared-expert-and-cleanup | 3.334942 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006322 | undefined |
| detail:read-ahead-wait | 2.090736 | undefined |
| total:layer | 170.244830 | undefined |

### Layer 40

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004519 | undefined |
| pre-attention-aggregation | 0.015158 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.015819 | undefined |
| Q | 2.584861 | undefined |
| K | 2.481177 | undefined |
| V | 2.326277 | undefined |
| B | 0.024085 | undefined |
| FA | 0.027792 | undefined |
| FB | 0.056435 | undefined |
| G | 2.105244 | undefined |
| O | 2.068024 | undefined |
| attention | 12.396967 | undefined |
| attention-residual | 0.002645 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024656 | undefined |
| moe-scratch-and-parameter-bind | 0.002394 | undefined |
| router-and-top16 | 0.586966 | undefined |
| EDOWN | 0.588940 | undefined |
| prefetch-submit-and-latent-projection | 0.590243 | undefined |
| detail:expert-gate | 53.396816 | undefined |
| detail:expert-up | 51.191816 | undefined |
| detail:expert-activation | 0.122450 | undefined |
| detail:expert-down | 52.523362 | undefined |
| EUP | 0.593920 | undefined |
| experts-mix-normalize-up | 160.190952 | undefined |
| SH1 | 1.066724 | undefined |
| SH3 | 1.102971 | undefined |
| SH2 | 1.105606 | undefined |
| shared-expert-and-cleanup | 3.290509 | undefined |
| mlp-residual-cache-save-and-unbind | 0.377746 | undefined |
| detail:read-ahead-wait | 2.238112 | undefined |
| total:layer | 177.510237 | undefined |

### Layer 41

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004579 | undefined |
| pre-attention-aggregation | 0.015329 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010499 | undefined |
| Q | 2.599748 | undefined |
| K | 2.535498 | undefined |
| V | 2.351184 | undefined |
| B | 0.026810 | undefined |
| FA | 0.029816 | undefined |
| FB | 0.051155 | undefined |
| G | 2.134519 | undefined |
| O | 2.049520 | undefined |
| attention | 12.440187 | undefined |
| attention-residual | 0.003006 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027121 | undefined |
| moe-scratch-and-parameter-bind | 0.002424 | undefined |
| router-and-top16 | 0.569564 | undefined |
| EDOWN | 0.584002 | undefined |
| prefetch-submit-and-latent-projection | 0.585364 | undefined |
| detail:expert-gate | 53.561239 | undefined |
| detail:expert-up | 51.922630 | undefined |
| detail:expert-activation | 0.127598 | undefined |
| detail:expert-down | 55.847546 | undefined |
| EUP | 0.595994 | undefined |
| experts-mix-normalize-up | 164.400378 | undefined |
| SH1 | 1.077534 | undefined |
| SH3 | 1.142384 | undefined |
| SH2 | 1.126916 | undefined |
| shared-expert-and-cleanup | 3.362193 | undefined |
| mlp-residual-cache-save-and-unbind | 0.394657 | undefined |
| detail:read-ahead-wait | 2.215748 | undefined |
| total:layer | 181.827304 | undefined |

### Layer 42

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004458 | undefined |
| pre-attention-aggregation | 0.015780 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010700 | undefined |
| Q | 2.619264 | undefined |
| K | 2.553672 | undefined |
| V | 2.342237 | undefined |
| B | 0.033843 | undefined |
| FA | 0.027371 | undefined |
| FB | 0.050605 | undefined |
| G | 2.100225 | undefined |
| O | 2.058156 | undefined |
| attention | 12.479701 | undefined |
| attention-residual | 0.003317 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025227 | undefined |
| moe-scratch-and-parameter-bind | 0.002695 | undefined |
| router-and-top16 | 0.574614 | undefined |
| EDOWN | 0.580715 | undefined |
| prefetch-submit-and-latent-projection | 0.582058 | undefined |
| detail:expert-gate | 54.913026 | undefined |
| detail:expert-up | 54.887664 | undefined |
| detail:expert-activation | 0.129830 | undefined |
| detail:expert-down | 58.454204 | undefined |
| EUP | 0.602085 | undefined |
| experts-mix-normalize-up | 176.197984 | undefined |
| SH1 | 1.105025 | undefined |
| SH3 | 1.131154 | undefined |
| SH2 | 1.104114 | undefined |
| shared-expert-and-cleanup | 3.355871 | undefined |
| mlp-residual-cache-save-and-unbind | 0.385130 | undefined |
| detail:read-ahead-wait | 7.073767 | undefined |
| total:layer | 193.649095 | undefined |

### Layer 43

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005099 | undefined |
| pre-attention-aggregation | 0.015879 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010579 | undefined |
| QA | 0.272590 | undefined |
| QB | 0.689949 | undefined |
| KA | 0.106288 | undefined |
| KB | 0.314407 | undefined |
| G | 2.154817 | undefined |
| O | 2.186716 | undefined |
| attention | 5.831998 | undefined |
| attention-residual | 0.002795 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026239 | undefined |
| moe-scratch-and-parameter-bind | 0.002094 | undefined |
| router-and-top16 | 0.623335 | undefined |
| EDOWN | 0.631430 | undefined |
| prefetch-submit-and-latent-projection | 0.632832 | undefined |
| detail:expert-gate | 55.545317 | undefined |
| detail:expert-up | 54.016042 | undefined |
| detail:expert-activation | 0.121916 | undefined |
| detail:expert-down | 54.550631 | undefined |
| EUP | 0.582959 | undefined |
| experts-mix-normalize-up | 168.561173 | undefined |
| SH1 | 1.075330 | undefined |
| SH3 | 1.107239 | undefined |
| SH2 | 1.105015 | undefined |
| shared-expert-and-cleanup | 3.303093 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006272 | undefined |
| detail:read-ahead-wait | 3.594137 | undefined |
| total:layer | 179.033152 | undefined |

### Layer 44

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004187 | undefined |
| pre-attention-aggregation | 0.015760 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011421 | undefined |
| Q | 2.584139 | undefined |
| K | 2.526591 | undefined |
| V | 2.332098 | undefined |
| B | 0.029515 | undefined |
| FA | 0.034044 | undefined |
| FB | 0.049072 | undefined |
| G | 2.090106 | undefined |
| O | 2.081189 | undefined |
| attention | 12.466486 | undefined |
| attention-residual | 0.003166 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025467 | undefined |
| moe-scratch-and-parameter-bind | 0.002765 | undefined |
| router-and-top16 | 0.582729 | undefined |
| EDOWN | 0.582809 | undefined |
| prefetch-submit-and-latent-projection | 0.584402 | undefined |
| detail:expert-gate | 55.633341 | undefined |
| detail:expert-up | 54.744413 | undefined |
| detail:expert-activation | 0.120144 | undefined |
| detail:expert-down | 53.710740 | undefined |
| EUP | 0.611222 | undefined |
| experts-mix-normalize-up | 171.383909 | undefined |
| SH1 | 1.092120 | undefined |
| SH3 | 1.120534 | undefined |
| SH2 | 1.123169 | undefined |
| shared-expert-and-cleanup | 3.351442 | undefined |
| mlp-residual-cache-save-and-unbind | 0.386312 | undefined |
| detail:read-ahead-wait | 6.429096 | undefined |
| total:layer | 188.830120 | undefined |

### Layer 45

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004899 | undefined |
| pre-attention-aggregation | 0.015328 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010750 | undefined |
| Q | 2.620447 | undefined |
| K | 2.555475 | undefined |
| V | 2.384646 | undefined |
| B | 0.022182 | undefined |
| FA | 0.039715 | undefined |
| FB | 0.051707 | undefined |
| G | 2.116495 | undefined |
| O | 2.046133 | undefined |
| attention | 12.503085 | undefined |
| attention-residual | 0.003086 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024355 | undefined |
| moe-scratch-and-parameter-bind | 0.002465 | undefined |
| router-and-top16 | 0.572189 | undefined |
| EDOWN | 0.578000 | undefined |
| prefetch-submit-and-latent-projection | 0.579513 | undefined |
| detail:expert-gate | 54.466704 | undefined |
| detail:expert-up | 52.012387 | undefined |
| detail:expert-activation | 0.112711 | undefined |
| detail:expert-down | 50.215279 | undefined |
| EUP | 0.586065 | undefined |
| experts-mix-normalize-up | 165.053900 | undefined |
| SH1 | 1.065401 | undefined |
| SH3 | 1.099174 | undefined |
| SH2 | 1.094766 | undefined |
| shared-expert-and-cleanup | 3.274990 | undefined |
| mlp-residual-cache-save-and-unbind | 0.381804 | undefined |
| detail:read-ahead-wait | 7.532426 | undefined |
| total:layer | 182.439127 | undefined |

### Layer 46

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005140 | undefined |
| pre-attention-aggregation | 0.015479 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011401 | undefined |
| Q | 2.607022 | undefined |
| K | 2.523346 | undefined |
| V | 2.362715 | undefined |
| B | 0.027060 | undefined |
| FA | 0.044694 | undefined |
| FB | 0.050484 | undefined |
| G | 2.119581 | undefined |
| O | 2.100535 | undefined |
| attention | 12.516991 | undefined |
| attention-residual | 0.003687 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025357 | undefined |
| moe-scratch-and-parameter-bind | 0.002435 | undefined |
| router-and-top16 | 0.579373 | undefined |
| EDOWN | 0.577870 | undefined |
| prefetch-submit-and-latent-projection | 0.579343 | undefined |
| detail:expert-gate | 55.521584 | undefined |
| detail:expert-up | 54.079859 | undefined |
| detail:expert-activation | 0.114393 | undefined |
| detail:expert-down | 53.682303 | undefined |
| EUP | 0.594000 | undefined |
| experts-mix-normalize-up | 166.190363 | undefined |
| SH1 | 1.062155 | undefined |
| SH3 | 1.100206 | undefined |
| SH2 | 1.095367 | undefined |
| shared-expert-and-cleanup | 3.273307 | undefined |
| mlp-residual-cache-save-and-unbind | 0.351958 | undefined |
| detail:read-ahead-wait | 2.063305 | undefined |
| total:layer | 183.567526 | undefined |

### Layer 47

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004919 | undefined |
| pre-attention-aggregation | 0.015348 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011201 | undefined |
| QA | 0.277398 | undefined |
| QB | 0.698896 | undefined |
| KA | 0.113101 | undefined |
| KB | 0.312434 | undefined |
| G | 2.107688 | undefined |
| O | 2.155228 | undefined |
| attention | 5.769501 | undefined |
| attention-residual | 0.002475 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025387 | undefined |
| moe-scratch-and-parameter-bind | 0.002104 | undefined |
| router-and-top16 | 0.622634 | undefined |
| EDOWN | 0.634466 | undefined |
| prefetch-submit-and-latent-projection | 0.635778 | undefined |
| detail:expert-gate | 55.976063 | undefined |
| detail:expert-up | 55.322101 | undefined |
| detail:expert-activation | 0.119465 | undefined |
| detail:expert-down | 53.730389 | undefined |
| EUP | 0.601093 | undefined |
| experts-mix-normalize-up | 168.133074 | undefined |
| SH1 | 1.066594 | undefined |
| SH3 | 1.100346 | undefined |
| SH2 | 1.112699 | undefined |
| shared-expert-and-cleanup | 3.296200 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006111 | undefined |
| detail:read-ahead-wait | 2.248751 | undefined |
| total:layer | 178.537686 | undefined |

### Layer 48

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004348 | undefined |
| pre-attention-aggregation | 0.015228 | undefined |
| snapshot-push | 0.001312 | undefined |
| pre-attention-normalization | 0.011131 | undefined |
| Q | 2.577937 | undefined |
| K | 2.479042 | undefined |
| V | 2.325837 | undefined |
| B | 0.023503 | undefined |
| FA | 0.036177 | undefined |
| FB | 0.050565 | undefined |
| G | 2.119712 | undefined |
| O | 2.113740 | undefined |
| attention | 12.448563 | undefined |
| attention-residual | 0.001984 | undefined |
| pre-mlp-aggregation-and-normalization | 0.031859 | undefined |
| moe-scratch-and-parameter-bind | 0.002605 | undefined |
| router-and-top16 | 0.582047 | undefined |
| EDOWN | 0.587868 | undefined |
| prefetch-submit-and-latent-projection | 0.589312 | undefined |
| detail:expert-gate | 55.415256 | undefined |
| detail:expert-up | 53.735567 | undefined |
| detail:expert-activation | 0.110136 | undefined |
| detail:expert-down | 49.262607 | undefined |
| EUP | 0.585945 | undefined |
| experts-mix-normalize-up | 160.994934 | undefined |
| SH1 | 1.038510 | undefined |
| SH3 | 1.083735 | undefined |
| SH2 | 1.091901 | undefined |
| shared-expert-and-cleanup | 3.230166 | undefined |
| mlp-residual-cache-save-and-unbind | 0.372206 | undefined |
| detail:read-ahead-wait | 1.747294 | undefined |
| total:layer | 178.299111 | undefined |

### Layer 49

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004288 | undefined |
| pre-attention-aggregation | 0.016190 | undefined |
| snapshot-push | 0.000030 | undefined |
| pre-attention-normalization | 0.010941 | undefined |
| Q | 2.595250 | undefined |
| K | 2.502577 | undefined |
| V | 2.342688 | undefined |
| B | 0.029315 | undefined |
| FA | 0.035556 | undefined |
| FB | 0.050494 | undefined |
| G | 2.132976 | undefined |
| O | 2.088753 | undefined |
| attention | 12.470624 | undefined |
| attention-residual | 0.003586 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025358 | undefined |
| moe-scratch-and-parameter-bind | 0.002485 | undefined |
| router-and-top16 | 0.583891 | undefined |
| EDOWN | 0.586626 | undefined |
| prefetch-submit-and-latent-projection | 0.588089 | undefined |
| detail:expert-gate | 54.607124 | undefined |
| detail:expert-up | 52.049719 | undefined |
| detail:expert-activation | 0.111568 | undefined |
| detail:expert-down | 53.967681 | undefined |
| EUP | 0.591936 | undefined |
| experts-mix-normalize-up | 163.714276 | undefined |
| SH1 | 1.059449 | undefined |
| SH3 | 1.109934 | undefined |
| SH2 | 1.101568 | undefined |
| shared-expert-and-cleanup | 3.286832 | undefined |
| mlp-residual-cache-save-and-unbind | 0.362918 | undefined |
| detail:read-ahead-wait | 2.252259 | undefined |
| total:layer | 181.082842 | undefined |

### Layer 50

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004468 | undefined |
| pre-attention-aggregation | 0.015579 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011011 | undefined |
| Q | 2.589139 | undefined |
| K | 2.499471 | undefined |
| V | 2.349280 | undefined |
| B | 0.022482 | undefined |
| FA | 0.038722 | undefined |
| FB | 0.068458 | undefined |
| G | 2.149036 | undefined |
| O | 2.065560 | undefined |
| attention | 12.459694 | undefined |
| attention-residual | 0.003587 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025427 | undefined |
| moe-scratch-and-parameter-bind | 0.002124 | undefined |
| router-and-top16 | 0.579313 | undefined |
| EDOWN | 0.584853 | undefined |
| prefetch-submit-and-latent-projection | 0.586426 | undefined |
| detail:expert-gate | 56.717420 | undefined |
| detail:expert-up | 55.203769 | undefined |
| detail:expert-activation | 0.104863 | undefined |
| detail:expert-down | 53.823221 | undefined |
| EUP | 0.913447 | undefined |
| experts-mix-normalize-up | 169.161565 | undefined |
| SH1 | 1.567479 | undefined |
| SH3 | 1.554836 | undefined |
| SH2 | 1.529378 | undefined |
| shared-expert-and-cleanup | 4.670197 | undefined |
| mlp-residual-cache-save-and-unbind | 0.359622 | undefined |
| detail:read-ahead-wait | 2.267558 | undefined |
| total:layer | 187.892748 | undefined |

### Layer 51

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004508 | undefined |
| pre-attention-aggregation | 0.017182 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010870 | undefined |
| QA | 0.399306 | undefined |
| QB | 0.740934 | undefined |
| KA | 0.105377 | undefined |
| KB | 0.316822 | undefined |
| G | 2.126955 | undefined |
| O | 2.254302 | undefined |
| attention | 6.051018 | undefined |
| attention-residual | 0.002364 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025558 | undefined |
| moe-scratch-and-parameter-bind | 0.002535 | undefined |
| router-and-top16 | 0.626490 | undefined |
| EDOWN | 0.627853 | undefined |
| prefetch-submit-and-latent-projection | 0.629256 | undefined |
| detail:expert-gate | 55.259775 | undefined |
| detail:expert-up | 52.699320 | undefined |
| detail:expert-activation | 0.103274 | undefined |
| detail:expert-down | 49.106505 | undefined |
| EUP | 0.594882 | undefined |
| experts-mix-normalize-up | 160.141079 | undefined |
| SH1 | 1.042829 | undefined |
| SH3 | 1.079187 | undefined |
| SH2 | 1.101729 | undefined |
| shared-expert-and-cleanup | 3.240084 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006532 | undefined |
| detail:read-ahead-wait | 2.245054 | undefined |
| total:layer | 170.771123 | undefined |

### Layer 52

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.003837 | undefined |
| pre-attention-aggregation | 0.015669 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.009989 | undefined |
| Q | 2.588107 | undefined |
| K | 2.462852 | undefined |
| V | 2.334413 | undefined |
| B | 0.022301 | undefined |
| FA | 0.037259 | undefined |
| FB | 0.062056 | undefined |
| G | 2.140731 | undefined |
| O | 2.125622 | undefined |
| attention | 12.494889 | undefined |
| attention-residual | 0.003066 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025658 | undefined |
| moe-scratch-and-parameter-bind | 0.002264 | undefined |
| router-and-top16 | 0.586306 | undefined |
| EDOWN | 0.594982 | undefined |
| prefetch-submit-and-latent-projection | 0.596755 | undefined |
| detail:expert-gate | 54.801247 | undefined |
| detail:expert-up | 50.967896 | undefined |
| detail:expert-activation | 0.102529 | undefined |
| detail:expert-down | 53.828098 | undefined |
| EUP | 0.891626 | undefined |
| experts-mix-normalize-up | 163.052129 | undefined |
| SH1 | 1.548373 | undefined |
| SH3 | 1.574672 | undefined |
| SH2 | 1.329805 | undefined |
| shared-expert-and-cleanup | 4.472147 | undefined |
| mlp-residual-cache-save-and-unbind | 0.378327 | undefined |
| detail:read-ahead-wait | 2.332968 | undefined |
| total:layer | 181.655132 | undefined |

### Layer 53

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004458 | undefined |
| pre-attention-aggregation | 0.016310 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011061 | undefined |
| Q | 2.589790 | undefined |
| K | 2.466729 | undefined |
| V | 2.358197 | undefined |
| B | 0.034094 | undefined |
| FA | 0.039394 | undefined |
| FB | 0.050334 | undefined |
| G | 2.132094 | undefined |
| O | 2.085427 | undefined |
| attention | 12.433535 | undefined |
| attention-residual | 0.003286 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025438 | undefined |
| moe-scratch-and-parameter-bind | 0.002945 | undefined |
| router-and-top16 | 0.580665 | undefined |
| EDOWN | 0.576838 | undefined |
| prefetch-submit-and-latent-projection | 0.578340 | undefined |
| detail:expert-gate | 55.190042 | undefined |
| detail:expert-up | 52.338024 | undefined |
| detail:expert-activation | 0.099636 | undefined |
| detail:expert-down | 48.990620 | undefined |
| EUP | 0.579593 | undefined |
| experts-mix-normalize-up | 159.722497 | undefined |
| SH1 | 1.050714 | undefined |
| SH3 | 1.080329 | undefined |
| SH2 | 1.105035 | undefined |
| shared-expert-and-cleanup | 3.252738 | undefined |
| mlp-residual-cache-save-and-unbind | 0.361486 | undefined |
| detail:read-ahead-wait | 2.391810 | undefined |
| total:layer | 177.007316 | undefined |

### Layer 54

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004177 | undefined |
| pre-attention-aggregation | 0.015589 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010810 | undefined |
| Q | 2.598245 | undefined |
| K | 2.490033 | undefined |
| V | 2.368135 | undefined |
| B | 0.022121 | undefined |
| FA | 0.040225 | undefined |
| FB | 0.050184 | undefined |
| G | 2.124671 | undefined |
| O | 2.085127 | undefined |
| attention | 12.460886 | undefined |
| attention-residual | 0.003106 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024967 | undefined |
| moe-scratch-and-parameter-bind | 0.002534 | undefined |
| router-and-top16 | 0.585384 | undefined |
| EDOWN | 0.586035 | undefined |
| prefetch-submit-and-latent-projection | 0.587659 | undefined |
| detail:expert-gate | 54.662840 | undefined |
| detail:expert-up | 53.048863 | undefined |
| detail:expert-activation | 0.124503 | undefined |
| detail:expert-down | 55.249958 | undefined |
| EUP | 0.590022 | undefined |
| experts-mix-normalize-up | 170.007015 | undefined |
| SH1 | 1.074337 | undefined |
| SH3 | 1.107590 | undefined |
| SH2 | 1.107941 | undefined |
| shared-expert-and-cleanup | 3.305657 | undefined |
| mlp-residual-cache-save-and-unbind | 0.356026 | undefined |
| detail:read-ahead-wait | 6.177612 | undefined |
| total:layer | 187.378367 | undefined |

### Layer 55

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004399 | undefined |
| pre-attention-aggregation | 0.015438 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010499 | undefined |
| QA | 0.273942 | undefined |
| QB | 0.707682 | undefined |
| KA | 0.108633 | undefined |
| KB | 0.329105 | undefined |
| G | 2.098041 | undefined |
| O | 2.161820 | undefined |
| attention | 5.782536 | undefined |
| attention-residual | 0.002495 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025899 | undefined |
| moe-scratch-and-parameter-bind | 0.002514 | undefined |
| router-and-top16 | 0.623124 | undefined |
| EDOWN | 0.623976 | undefined |
| prefetch-submit-and-latent-projection | 0.625469 | undefined |
| detail:expert-gate | 53.554529 | undefined |
| detail:expert-up | 52.594144 | undefined |
| detail:expert-activation | 0.132670 | undefined |
| detail:expert-down | 56.924204 | undefined |
| EUP | 0.593359 | undefined |
| experts-mix-normalize-up | 171.163687 | undefined |
| SH1 | 1.099314 | undefined |
| SH3 | 1.131795 | undefined |
| SH2 | 1.116857 | undefined |
| shared-expert-and-cleanup | 3.364287 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006041 | undefined |
| detail:read-ahead-wait | 7.202701 | undefined |
| total:layer | 181.641075 | undefined |

### Layer 56

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004779 | undefined |
| pre-attention-aggregation | 0.015979 | undefined |
| snapshot-push | 0.000030 | undefined |
| pre-attention-normalization | 0.011231 | undefined |
| Q | 2.567478 | undefined |
| K | 2.489642 | undefined |
| V | 2.379386 | undefined |
| B | 0.026219 | undefined |
| FA | 0.040666 | undefined |
| FB | 0.048751 | undefined |
| G | 2.116676 | undefined |
| O | 2.094644 | undefined |
| attention | 12.485652 | undefined |
| attention-residual | 0.003055 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025487 | undefined |
| moe-scratch-and-parameter-bind | 0.002765 | undefined |
| router-and-top16 | 0.584672 | undefined |
| EDOWN | 0.586586 | undefined |
| prefetch-submit-and-latent-projection | 0.588199 | undefined |
| detail:expert-gate | 52.501492 | undefined |
| detail:expert-up | 50.136593 | undefined |
| detail:expert-activation | 0.124072 | undefined |
| detail:expert-down | 52.831880 | undefined |
| EUP | 0.593098 | undefined |
| experts-mix-normalize-up | 159.407738 | undefined |
| SH1 | 1.072094 | undefined |
| SH3 | 1.108372 | undefined |
| SH2 | 1.106908 | undefined |
| shared-expert-and-cleanup | 3.303604 | undefined |
| mlp-residual-cache-save-and-unbind | 0.380912 | undefined |
| detail:read-ahead-wait | 3.062451 | undefined |
| total:layer | 176.829494 | undefined |

### Layer 57

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004919 | undefined |
| pre-attention-aggregation | 0.016581 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011311 | undefined |
| Q | 2.604948 | undefined |
| K | 2.563550 | undefined |
| V | 2.401808 | undefined |
| B | 0.022652 | undefined |
| FA | 0.030878 | undefined |
| FB | 0.050174 | undefined |
| G | 2.137314 | undefined |
| O | 2.065520 | undefined |
| attention | 12.541095 | undefined |
| attention-residual | 0.003156 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025538 | undefined |
| moe-scratch-and-parameter-bind | 0.002755 | undefined |
| router-and-top16 | 0.577049 | undefined |
| EDOWN | 0.572210 | undefined |
| prefetch-submit-and-latent-projection | 0.573952 | undefined |
| detail:expert-gate | 53.026933 | undefined |
| detail:expert-up | 49.853881 | undefined |
| detail:expert-activation | 0.114924 | undefined |
| detail:expert-down | 51.895260 | undefined |
| EUP | 0.578070 | undefined |
| experts-mix-normalize-up | 158.155688 | undefined |
| SH1 | 1.051355 | undefined |
| SH3 | 1.114763 | undefined |
| SH2 | 1.117047 | undefined |
| shared-expert-and-cleanup | 3.299536 | undefined |
| mlp-residual-cache-save-and-unbind | 0.376774 | undefined |
| detail:read-ahead-wait | 2.541929 | undefined |
| total:layer | 175.603884 | undefined |

### Layer 58

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004558 | undefined |
| pre-attention-aggregation | 0.015349 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010741 | undefined |
| Q | 2.614044 | undefined |
| K | 2.557379 | undefined |
| V | 2.384897 | undefined |
| B | 0.029866 | undefined |
| FA | 0.036649 | undefined |
| FB | 0.052067 | undefined |
| G | 2.106717 | undefined |
| O | 2.058597 | undefined |
| attention | 12.501311 | undefined |
| attention-residual | 0.003086 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025157 | undefined |
| moe-scratch-and-parameter-bind | 0.002475 | undefined |
| router-and-top16 | 0.570717 | undefined |
| EDOWN | 0.589292 | undefined |
| prefetch-submit-and-latent-projection | 0.591044 | undefined |
| detail:expert-gate | 54.207369 | undefined |
| detail:expert-up | 52.980306 | undefined |
| detail:expert-activation | 0.127570 | undefined |
| detail:expert-down | 54.955639 | undefined |
| EUP | 0.589341 | undefined |
| experts-mix-normalize-up | 165.225359 | undefined |
| SH1 | 1.085589 | undefined |
| SH3 | 1.120584 | undefined |
| SH2 | 1.113341 | undefined |
| shared-expert-and-cleanup | 3.336175 | undefined |
| mlp-residual-cache-save-and-unbind | 0.358700 | undefined |
| detail:read-ahead-wait | 2.206425 | undefined |
| total:layer | 182.660401 | undefined |

### Layer 59

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004959 | undefined |
| pre-attention-aggregation | 0.015519 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.011201 | undefined |
| QA | 0.270956 | undefined |
| QB | 0.713733 | undefined |
| KA | 0.107000 | undefined |
| KB | 0.316651 | undefined |
| G | 2.154276 | undefined |
| O | 2.201363 | undefined |
| attention | 5.874347 | undefined |
| attention-residual | 0.002564 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025126 | undefined |
| moe-scratch-and-parameter-bind | 0.002344 | undefined |
| router-and-top16 | 0.621011 | undefined |
| EDOWN | 0.625930 | undefined |
| prefetch-submit-and-latent-projection | 0.627443 | undefined |
| detail:expert-gate | 52.275979 | undefined |
| detail:expert-up | 49.078595 | undefined |
| detail:expert-activation | 0.123300 | undefined |
| detail:expert-down | 51.989604 | undefined |
| EUP | 0.583961 | undefined |
| experts-mix-normalize-up | 156.301944 | undefined |
| SH1 | 1.064970 | undefined |
| SH3 | 1.114572 | undefined |
| SH2 | 1.116486 | undefined |
| shared-expert-and-cleanup | 3.313572 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005681 | undefined |
| detail:read-ahead-wait | 2.100946 | undefined |
| total:layer | 166.821562 | undefined |

### Layer 60

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004148 | undefined |
| pre-attention-aggregation | 0.015689 | undefined |
| snapshot-push | 0.001593 | undefined |
| pre-attention-normalization | 0.010970 | undefined |
| Q | 2.580893 | undefined |
| K | 2.515741 | undefined |
| V | 2.367143 | undefined |
| B | 0.026039 | undefined |
| FA | 0.028814 | undefined |
| FB | 0.050184 | undefined |
| G | 2.114873 | undefined |
| O | 2.106297 | undefined |
| attention | 12.513865 | undefined |
| attention-residual | 0.001904 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026049 | undefined |
| moe-scratch-and-parameter-bind | 0.003046 | undefined |
| router-and-top16 | 0.583460 | undefined |
| EDOWN | 0.593409 | undefined |
| prefetch-submit-and-latent-projection | 0.595012 | undefined |
| detail:expert-gate | 53.823539 | undefined |
| detail:expert-up | 52.797913 | undefined |
| detail:expert-activation | 0.126928 | undefined |
| detail:expert-down | 55.802488 | undefined |
| EUP | 0.595183 | undefined |
| experts-mix-normalize-up | 165.394486 | undefined |
| SH1 | 1.082894 | undefined |
| SH3 | 1.120033 | undefined |
| SH2 | 1.112809 | undefined |
| shared-expert-and-cleanup | 3.332137 | undefined |
| mlp-residual-cache-save-and-unbind | 0.378066 | undefined |
| detail:read-ahead-wait | 2.080296 | undefined |
| total:layer | 182.876785 | undefined |

### Layer 61

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004618 | undefined |
| pre-attention-aggregation | 0.016651 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010630 | undefined |
| Q | 2.618473 | undefined |
| K | 2.573679 | undefined |
| V | 2.394614 | undefined |
| B | 0.029776 | undefined |
| FA | 0.027421 | undefined |
| FB | 0.050975 | undefined |
| G | 2.122928 | undefined |
| O | 2.038069 | undefined |
| attention | 12.514967 | undefined |
| attention-residual | 0.003597 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025658 | undefined |
| moe-scratch-and-parameter-bind | 0.003026 | undefined |
| router-and-top16 | 0.571228 | undefined |
| EDOWN | 0.581757 | undefined |
| prefetch-submit-and-latent-projection | 0.583651 | undefined |
| detail:expert-gate | 51.220294 | undefined |
| detail:expert-up | 47.467995 | undefined |
| detail:expert-activation | 0.114161 | undefined |
| detail:expert-down | 51.293702 | undefined |
| EUP | 0.588570 | undefined |
| experts-mix-normalize-up | 153.066989 | undefined |
| SH1 | 1.068868 | undefined |
| SH3 | 1.097040 | undefined |
| SH2 | 1.109954 | undefined |
| shared-expert-and-cleanup | 3.293184 | undefined |
| mlp-residual-cache-save-and-unbind | 0.358349 | undefined |
| detail:read-ahead-wait | 2.233983 | undefined |
| total:layer | 170.469389 | undefined |

### Layer 62

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004739 | undefined |
| pre-attention-aggregation | 0.017272 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010961 | undefined |
| Q | 2.600951 | undefined |
| K | 2.547321 | undefined |
| V | 2.396238 | undefined |
| B | 0.024495 | undefined |
| FA | 0.028513 | undefined |
| FB | 0.049032 | undefined |
| G | 2.142273 | undefined |
| O | 2.057445 | undefined |
| attention | 12.516730 | undefined |
| attention-residual | 0.003085 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025678 | undefined |
| moe-scratch-and-parameter-bind | 0.002985 | undefined |
| router-and-top16 | 0.571068 | undefined |
| EDOWN | 0.580104 | undefined |
| prefetch-submit-and-latent-projection | 0.581817 | undefined |
| detail:expert-gate | 54.295161 | undefined |
| detail:expert-up | 51.487183 | undefined |
| detail:expert-activation | 0.120785 | undefined |
| detail:expert-down | 55.275995 | undefined |
| EUP | 0.773145 | undefined |
| experts-mix-normalize-up | 169.470682 | undefined |
| SH1 | 1.330366 | undefined |
| SH3 | 1.330286 | undefined |
| SH2 | 1.331057 | undefined |
| shared-expert-and-cleanup | 4.011146 | undefined |
| mlp-residual-cache-save-and-unbind | 0.387414 | undefined |
| detail:read-ahead-wait | 7.355686 | undefined |
| total:layer | 187.620430 | undefined |

### Layer 63

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004999 | undefined |
| pre-attention-aggregation | 0.018334 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011231 | undefined |
| QA | 0.323555 | undefined |
| QB | 0.800456 | undefined |
| KA | 0.123121 | undefined |
| KB | 0.341458 | undefined |
| G | 2.705836 | undefined |
| O | 2.719532 | undefined |
| attention | 7.119304 | undefined |
| attention-residual | 0.002384 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026579 | undefined |
| moe-scratch-and-parameter-bind | 0.002404 | undefined |
| router-and-top16 | 0.637811 | undefined |
| EDOWN | 0.797019 | undefined |
| prefetch-submit-and-latent-projection | 0.798632 | undefined |
| detail:expert-gate | 51.286429 | undefined |
| detail:expert-up | 47.937752 | undefined |
| detail:expert-activation | 0.114605 | undefined |
| detail:expert-down | 51.494955 | undefined |
| EUP | 0.584442 | undefined |
| experts-mix-normalize-up | 158.780196 | undefined |
| SH1 | 1.061634 | undefined |
| SH3 | 1.093273 | undefined |
| SH2 | 1.104905 | undefined |
| shared-expert-and-cleanup | 3.276263 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005621 | undefined |
| detail:read-ahead-wait | 7.211415 | undefined |
| total:layer | 170.700591 | undefined |

### Layer 64

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004368 | undefined |
| pre-attention-aggregation | 0.016070 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010991 | undefined |
| Q | 2.580032 | undefined |
| K | 2.490124 | undefined |
| V | 2.353378 | undefined |
| B | 0.024075 | undefined |
| FA | 0.034084 | undefined |
| FB | 0.051527 | undefined |
| G | 2.125051 | undefined |
| O | 2.097720 | undefined |
| attention | 12.491673 | undefined |
| attention-residual | 0.002765 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025157 | undefined |
| moe-scratch-and-parameter-bind | 0.002485 | undefined |
| router-and-top16 | 0.583631 | undefined |
| EDOWN | 0.591345 | undefined |
| prefetch-submit-and-latent-projection | 0.593139 | undefined |
| detail:expert-gate | 53.986435 | undefined |
| detail:expert-up | 52.563336 | undefined |
| detail:expert-activation | 0.139419 | undefined |
| detail:expert-down | 55.725705 | undefined |
| EUP | 0.814722 | undefined |
| experts-mix-normalize-up | 170.723293 | undefined |
| SH1 | 1.348029 | undefined |
| SH3 | 1.309788 | undefined |
| SH2 | 1.305980 | undefined |
| shared-expert-and-cleanup | 3.983314 | undefined |
| mlp-residual-cache-save-and-unbind | 0.411990 | undefined |
| detail:read-ahead-wait | 7.318346 | undefined |
| total:layer | 188.865946 | undefined |

### Layer 65

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005621 | undefined |
| pre-attention-aggregation | 0.019206 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.012042 | undefined |
| Q | 3.023320 | undefined |
| K | 3.054528 | undefined |
| V | 2.988535 | undefined |
| B | 0.025938 | undefined |
| FA | 0.038122 | undefined |
| FB | 0.069831 | undefined |
| G | 2.717709 | undefined |
| O | 2.681441 | undefined |
| attention | 15.380401 | undefined |
| attention-residual | 0.003807 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026840 | undefined |
| moe-scratch-and-parameter-bind | 0.002435 | undefined |
| router-and-top16 | 0.595563 | undefined |
| EDOWN | 0.779146 | undefined |
| prefetch-submit-and-latent-projection | 0.780889 | undefined |
| detail:expert-gate | 51.507612 | undefined |
| detail:expert-up | 48.036956 | undefined |
| detail:expert-activation | 0.112442 | undefined |
| detail:expert-down | 51.247588 | undefined |
| EUP | 0.586115 | undefined |
| experts-mix-normalize-up | 153.783778 | undefined |
| SH1 | 1.063137 | undefined |
| SH3 | 1.088324 | undefined |
| SH2 | 1.101829 | undefined |
| shared-expert-and-cleanup | 3.269660 | undefined |
| mlp-residual-cache-save-and-unbind | 0.387364 | undefined |
| detail:read-ahead-wait | 2.137597 | undefined |
| total:layer | 174.285310 | undefined |

### Layer 66

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004349 | undefined |
| pre-attention-aggregation | 0.015198 | undefined |
| snapshot-push | 0.000030 | undefined |
| pre-attention-normalization | 0.011292 | undefined |
| Q | 2.588287 | undefined |
| K | 2.518887 | undefined |
| V | 2.389525 | undefined |
| B | 0.024426 | undefined |
| FA | 0.034715 | undefined |
| FB | 0.050515 | undefined |
| G | 2.141041 | undefined |
| O | 2.066111 | undefined |
| attention | 12.490080 | undefined |
| attention-residual | 0.003797 | undefined |
| pre-mlp-aggregation-and-normalization | 0.024956 | undefined |
| moe-scratch-and-parameter-bind | 0.002114 | undefined |
| router-and-top16 | 0.575696 | undefined |
| EDOWN | 0.581937 | undefined |
| prefetch-submit-and-latent-projection | 0.583700 | undefined |
| detail:expert-gate | 51.909215 | undefined |
| detail:expert-up | 49.462782 | undefined |
| detail:expert-activation | 0.121228 | undefined |
| detail:expert-down | 52.307497 | undefined |
| EUP | 0.586556 | undefined |
| experts-mix-normalize-up | 156.576136 | undefined |
| SH1 | 1.063878 | undefined |
| SH3 | 1.106748 | undefined |
| SH2 | 1.102571 | undefined |
| shared-expert-and-cleanup | 3.290188 | undefined |
| mlp-residual-cache-save-and-unbind | 0.359121 | undefined |
| detail:read-ahead-wait | 2.024883 | undefined |
| total:layer | 173.954372 | undefined |

### Layer 67

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004789 | undefined |
| pre-attention-aggregation | 0.016711 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011010 | undefined |
| QA | 0.392553 | undefined |
| QB | 0.692324 | undefined |
| KA | 0.109805 | undefined |
| KB | 0.312054 | undefined |
| G | 2.130652 | undefined |
| O | 2.170286 | undefined |
| attention | 5.917107 | undefined |
| attention-residual | 0.002935 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025978 | undefined |
| moe-scratch-and-parameter-bind | 0.002064 | undefined |
| router-and-top16 | 0.619177 | undefined |
| EDOWN | 0.628104 | undefined |
| prefetch-submit-and-latent-projection | 0.629827 | undefined |
| detail:expert-gate | 52.859401 | undefined |
| detail:expert-up | 50.394944 | undefined |
| detail:expert-activation | 0.120263 | undefined |
| detail:expert-down | 55.012886 | undefined |
| EUP | 0.598949 | undefined |
| experts-mix-normalize-up | 166.298255 | undefined |
| SH1 | 1.096820 | undefined |
| SH3 | 1.125032 | undefined |
| SH2 | 1.118610 | undefined |
| shared-expert-and-cleanup | 3.357685 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006122 | undefined |
| detail:read-ahead-wait | 7.146293 | undefined |
| total:layer | 176.909303 | undefined |

### Layer 68

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004719 | undefined |
| pre-attention-aggregation | 0.015699 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011231 | undefined |
| Q | 2.579050 | undefined |
| K | 2.517193 | undefined |
| V | 2.375329 | undefined |
| B | 0.021690 | undefined |
| FA | 0.030697 | undefined |
| FB | 0.048010 | undefined |
| G | 2.091088 | undefined |
| O | 2.064719 | undefined |
| attention | 12.457970 | undefined |
| attention-residual | 0.003657 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026078 | undefined |
| moe-scratch-and-parameter-bind | 0.002555 | undefined |
| router-and-top16 | 0.579303 | undefined |
| EDOWN | 0.584853 | undefined |
| prefetch-submit-and-latent-projection | 0.586957 | undefined |
| detail:expert-gate | 52.411321 | undefined |
| detail:expert-up | 50.295870 | undefined |
| detail:expert-activation | 0.124883 | undefined |
| detail:expert-down | 53.260228 | undefined |
| EUP | 0.588850 | undefined |
| experts-mix-normalize-up | 162.175901 | undefined |
| SH1 | 1.069639 | undefined |
| SH3 | 1.096700 | undefined |
| SH2 | 1.094094 | undefined |
| shared-expert-and-cleanup | 3.277425 | undefined |
| mlp-residual-cache-save-and-unbind | 0.391562 | undefined |
| detail:read-ahead-wait | 5.286628 | undefined |
| total:layer | 179.551130 | undefined |

### Layer 69

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004328 | undefined |
| pre-attention-aggregation | 0.015880 | undefined |
| snapshot-push | 0.000401 | undefined |
| pre-attention-normalization | 0.011011 | undefined |
| Q | 2.579490 | undefined |
| K | 2.525540 | undefined |
| V | 2.361352 | undefined |
| B | 0.033022 | undefined |
| FA | 0.036388 | undefined |
| FB | 0.048611 | undefined |
| G | 2.122917 | undefined |
| O | 2.065860 | undefined |
| attention | 12.460134 | undefined |
| attention-residual | 0.003296 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025648 | undefined |
| moe-scratch-and-parameter-bind | 0.002504 | undefined |
| router-and-top16 | 0.570186 | undefined |
| EDOWN | 0.579613 | undefined |
| prefetch-submit-and-latent-projection | 0.581577 | undefined |
| detail:expert-gate | 52.769901 | undefined |
| detail:expert-up | 50.755073 | undefined |
| detail:expert-activation | 0.125455 | undefined |
| detail:expert-down | 55.254263 | undefined |
| EUP | 0.604029 | undefined |
| experts-mix-normalize-up | 164.179004 | undefined |
| SH1 | 1.080559 | undefined |
| SH3 | 1.105516 | undefined |
| SH2 | 1.101428 | undefined |
| shared-expert-and-cleanup | 3.304575 | undefined |
| mlp-residual-cache-save-and-unbind | 0.368919 | undefined |
| detail:read-ahead-wait | 4.492016 | undefined |
| total:layer | 181.545588 | undefined |

### Layer 70

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005621 | undefined |
| pre-attention-aggregation | 0.016501 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010660 | undefined |
| Q | 2.603184 | undefined |
| K | 2.506184 | undefined |
| V | 2.367203 | undefined |
| B | 0.022472 | undefined |
| FA | 0.029845 | undefined |
| FB | 0.048601 | undefined |
| G | 2.123097 | undefined |
| O | 2.058467 | undefined |
| attention | 12.448352 | undefined |
| attention-residual | 0.003567 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026209 | undefined |
| moe-scratch-and-parameter-bind | 0.002675 | undefined |
| router-and-top16 | 0.575295 | undefined |
| EDOWN | 0.575175 | undefined |
| prefetch-submit-and-latent-projection | 0.577099 | undefined |
| detail:expert-gate | 53.326743 | undefined |
| detail:expert-up | 51.855875 | undefined |
| detail:expert-activation | 0.132795 | undefined |
| detail:expert-down | 53.005831 | undefined |
| EUP | 0.596354 | undefined |
| experts-mix-normalize-up | 161.069363 | undefined |
| SH1 | 1.068717 | undefined |
| SH3 | 1.100005 | undefined |
| SH2 | 1.102520 | undefined |
| shared-expert-and-cleanup | 3.288686 | undefined |
| mlp-residual-cache-save-and-unbind | 0.363980 | undefined |
| detail:read-ahead-wait | 1.963059 | undefined |
| total:layer | 178.406512 | undefined |

### Layer 71

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004869 | undefined |
| pre-attention-aggregation | 0.016060 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010740 | undefined |
| QA | 0.279823 | undefined |
| QB | 0.685671 | undefined |
| KA | 0.105888 | undefined |
| KB | 0.334846 | undefined |
| G | 2.112297 | undefined |
| O | 2.151571 | undefined |
| attention | 5.778338 | undefined |
| attention-residual | 0.002726 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025197 | undefined |
| moe-scratch-and-parameter-bind | 0.002274 | undefined |
| router-and-top16 | 0.618335 | undefined |
| EDOWN | 0.623936 | undefined |
| prefetch-submit-and-latent-projection | 0.625800 | undefined |
| detail:expert-gate | 52.748881 | undefined |
| detail:expert-up | 50.391545 | undefined |
| detail:expert-activation | 0.120593 | undefined |
| detail:expert-down | 55.159527 | undefined |
| EUP | 0.617254 | undefined |
| experts-mix-normalize-up | 166.535448 | undefined |
| SH1 | 1.089456 | undefined |
| SH3 | 1.106237 | undefined |
| SH2 | 1.127237 | undefined |
| shared-expert-and-cleanup | 3.340322 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005911 | undefined |
| detail:read-ahead-wait | 7.327843 | undefined |
| total:layer | 176.984594 | undefined |

### Layer 72

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004368 | undefined |
| pre-attention-aggregation | 0.015579 | undefined |
| snapshot-push | 0.002084 | undefined |
| pre-attention-normalization | 0.011622 | undefined |
| Q | 2.559523 | undefined |
| K | 2.460588 | undefined |
| V | 2.330996 | undefined |
| B | 0.026049 | undefined |
| FA | 0.030728 | undefined |
| FB | 0.052548 | undefined |
| G | 2.112517 | undefined |
| O | 2.105855 | undefined |
| attention | 12.407556 | undefined |
| attention-residual | 0.001893 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025748 | undefined |
| moe-scratch-and-parameter-bind | 0.002424 | undefined |
| router-and-top16 | 0.584222 | undefined |
| EDOWN | 0.585554 | undefined |
| prefetch-submit-and-latent-projection | 0.587548 | undefined |
| detail:expert-gate | 53.975885 | undefined |
| detail:expert-up | 54.040988 | undefined |
| detail:expert-activation | 0.134893 | undefined |
| detail:expert-down | 53.668414 | undefined |
| EUP | 0.584713 | undefined |
| experts-mix-normalize-up | 169.506620 | undefined |
| SH1 | 1.071232 | undefined |
| SH3 | 1.106167 | undefined |
| SH2 | 1.096178 | undefined |
| shared-expert-and-cleanup | 3.290749 | undefined |
| mlp-residual-cache-save-and-unbind | 0.377816 | undefined |
| detail:read-ahead-wait | 6.894187 | undefined |
| total:layer | 186.837346 | undefined |

### Layer 73

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005861 | undefined |
| pre-attention-aggregation | 0.016702 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010890 | undefined |
| Q | 2.581995 | undefined |
| K | 2.513397 | undefined |
| V | 2.363737 | undefined |
| B | 0.031980 | undefined |
| FA | 0.027231 | undefined |
| FB | 0.051517 | undefined |
| G | 2.123839 | undefined |
| O | 2.084565 | undefined |
| attention | 12.479100 | undefined |
| attention-residual | 0.003467 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025368 | undefined |
| moe-scratch-and-parameter-bind | 0.002575 | undefined |
| router-and-top16 | 0.577299 | undefined |
| EDOWN | 0.584462 | undefined |
| prefetch-submit-and-latent-projection | 0.586526 | undefined |
| detail:expert-gate | 54.410107 | undefined |
| detail:expert-up | 54.436516 | undefined |
| detail:expert-activation | 0.136694 | undefined |
| detail:expert-down | 57.429490 | undefined |
| EUP | 0.802980 | undefined |
| experts-mix-normalize-up | 174.202526 | undefined |
| SH1 | 1.354231 | undefined |
| SH3 | 1.377114 | undefined |
| SH2 | 1.377474 | undefined |
| shared-expert-and-cleanup | 4.128746 | undefined |
| mlp-residual-cache-save-and-unbind | 0.370963 | undefined |
| detail:read-ahead-wait | 6.776745 | undefined |
| total:layer | 192.429246 | undefined |

### Layer 74

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004869 | undefined |
| pre-attention-aggregation | 0.017563 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011562 | undefined |
| Q | 3.108189 | undefined |
| K | 3.127324 | undefined |
| V | 2.993334 | undefined |
| B | 0.029766 | undefined |
| FA | 0.032671 | undefined |
| FB | 0.074449 | undefined |
| G | 2.712468 | undefined |
| O | 2.675219 | undefined |
| attention | 15.429844 | undefined |
| attention-residual | 0.003125 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027041 | undefined |
| moe-scratch-and-parameter-bind | 0.002474 | undefined |
| router-and-top16 | 0.601694 | undefined |
| EDOWN | 0.784927 | undefined |
| prefetch-submit-and-latent-projection | 0.787081 | undefined |
| detail:expert-gate | 54.218489 | undefined |
| detail:expert-up | 54.132340 | undefined |
| detail:expert-activation | 0.135633 | undefined |
| detail:expert-down | 55.199133 | undefined |
| EUP | 0.611593 | undefined |
| experts-mix-normalize-up | 171.277880 | undefined |
| SH1 | 1.091670 | undefined |
| SH3 | 1.108251 | undefined |
| SH2 | 1.114482 | undefined |
| shared-expert-and-cleanup | 3.330804 | undefined |
| mlp-residual-cache-save-and-unbind | 0.372035 | undefined |
| detail:read-ahead-wait | 6.771825 | undefined |
| total:layer | 191.885520 | undefined |

### Layer 75

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005380 | undefined |
| pre-attention-aggregation | 0.017162 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.010490 | undefined |
| QA | 0.266157 | undefined |
| QB | 0.698665 | undefined |
| KA | 0.108643 | undefined |
| KB | 0.309498 | undefined |
| G | 2.108470 | undefined |
| O | 2.162081 | undefined |
| attention | 5.761336 | undefined |
| attention-residual | 0.002865 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026921 | undefined |
| moe-scratch-and-parameter-bind | 0.002345 | undefined |
| router-and-top16 | 0.619448 | undefined |
| EDOWN | 0.629066 | undefined |
| prefetch-submit-and-latent-projection | 0.630949 | undefined |
| detail:expert-gate | 54.430094 | undefined |
| detail:expert-up | 54.192322 | undefined |
| detail:expert-activation | 0.133131 | undefined |
| detail:expert-down | 55.892896 | undefined |
| EUP | 0.612504 | undefined |
| experts-mix-normalize-up | 172.575735 | undefined |
| SH1 | 1.086971 | undefined |
| SH3 | 1.115475 | undefined |
| SH2 | 1.107931 | undefined |
| shared-expert-and-cleanup | 3.327438 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006061 | undefined |
| detail:read-ahead-wait | 7.109295 | undefined |
| total:layer | 183.005766 | undefined |

### Layer 76

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005060 | undefined |
| pre-attention-aggregation | 0.016221 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010510 | undefined |
| Q | 2.569221 | undefined |
| K | 2.487859 | undefined |
| V | 2.343409 | undefined |
| B | 0.021720 | undefined |
| FA | 0.037470 | undefined |
| FB | 0.053961 | undefined |
| G | 2.116515 | undefined |
| O | 2.094133 | undefined |
| attention | 12.423746 | undefined |
| attention-residual | 0.003126 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027491 | undefined |
| moe-scratch-and-parameter-bind | 0.002705 | undefined |
| router-and-top16 | 0.583460 | undefined |
| EDOWN | 0.587978 | undefined |
| prefetch-submit-and-latent-projection | 0.589943 | undefined |
| detail:expert-gate | 54.078188 | undefined |
| detail:expert-up | 53.502171 | undefined |
| detail:expert-activation | 0.134740 | undefined |
| detail:expert-down | 52.576801 | undefined |
| EUP | 0.600412 | undefined |
| experts-mix-normalize-up | 168.221869 | undefined |
| SH1 | 1.090377 | undefined |
| SH3 | 1.125002 | undefined |
| SH2 | 1.107550 | undefined |
| shared-expert-and-cleanup | 3.339781 | undefined |
| mlp-residual-cache-save-and-unbind | 0.389117 | undefined |
| detail:read-ahead-wait | 7.116839 | undefined |
| total:layer | 185.633136 | undefined |

### Layer 77

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004398 | undefined |
| pre-attention-aggregation | 0.017503 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011762 | undefined |
| Q | 2.612011 | undefined |
| K | 2.567248 | undefined |
| V | 2.376852 | undefined |
| B | 0.029535 | undefined |
| FA | 0.040015 | undefined |
| FB | 0.055254 | undefined |
| G | 2.103140 | undefined |
| O | 2.013884 | undefined |
| attention | 12.453743 | undefined |
| attention-residual | 0.002425 | undefined |
| pre-mlp-aggregation-and-normalization | 0.029486 | undefined |
| moe-scratch-and-parameter-bind | 0.002725 | undefined |
| router-and-top16 | 0.580725 | undefined |
| EDOWN | 0.593830 | undefined |
| prefetch-submit-and-latent-projection | 0.595853 | undefined |
| detail:expert-gate | 54.581987 | undefined |
| detail:expert-up | 54.393856 | undefined |
| detail:expert-activation | 0.134192 | undefined |
| detail:expert-down | 56.480165 | undefined |
| EUP | 0.764399 | undefined |
| experts-mix-normalize-up | 173.419492 | undefined |
| SH1 | 1.330346 | undefined |
| SH3 | 1.352938 | undefined |
| SH2 | 1.373287 | undefined |
| shared-expert-and-cleanup | 4.075707 | undefined |
| mlp-residual-cache-save-and-unbind | 0.375442 | undefined |
| detail:read-ahead-wait | 6.861753 | undefined |
| total:layer | 191.589937 | undefined |

### Layer 78

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004689 | undefined |
| pre-attention-aggregation | 0.019166 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011161 | undefined |
| Q | 3.165205 | undefined |
| K | 3.152421 | undefined |
| V | 3.052073 | undefined |
| B | 0.027582 | undefined |
| FA | 0.028643 | undefined |
| FB | 0.054723 | undefined |
| G | 2.714733 | undefined |
| O | 2.655222 | undefined |
| attention | 15.488824 | undefined |
| attention-residual | 0.002244 | undefined |
| pre-mlp-aggregation-and-normalization | 0.029605 | undefined |
| moe-scratch-and-parameter-bind | 0.002214 | undefined |
| router-and-top16 | 0.604880 | undefined |
| EDOWN | 0.802169 | undefined |
| prefetch-submit-and-latent-projection | 0.804263 | undefined |
| detail:expert-gate | 54.180446 | undefined |
| detail:expert-up | 54.180880 | undefined |
| detail:expert-activation | 0.135916 | undefined |
| detail:expert-down | 55.404235 | undefined |
| EUP | 0.602355 | undefined |
| experts-mix-normalize-up | 167.116544 | undefined |
| SH1 | 1.078044 | undefined |
| SH3 | 1.113110 | undefined |
| SH2 | 1.102971 | undefined |
| shared-expert-and-cleanup | 3.310546 | undefined |
| mlp-residual-cache-save-and-unbind | 0.382765 | undefined |
| detail:read-ahead-wait | 2.400508 | undefined |
| total:layer | 187.797510 | undefined |

### Layer 79

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004398 | undefined |
| pre-attention-aggregation | 0.017703 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011141 | undefined |
| QA | 0.270014 | undefined |
| QB | 0.707903 | undefined |
| KA | 0.106399 | undefined |
| KB | 0.320368 | undefined |
| G | 2.179743 | undefined |
| O | 2.193870 | undefined |
| attention | 5.888734 | undefined |
| attention-residual | 0.002645 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027392 | undefined |
| moe-scratch-and-parameter-bind | 0.002114 | undefined |
| router-and-top16 | 0.615160 | undefined |
| EDOWN | 0.630619 | undefined |
| prefetch-submit-and-latent-projection | 0.632562 | undefined |
| detail:expert-gate | 54.265760 | undefined |
| detail:expert-up | 53.548787 | undefined |
| detail:expert-activation | 0.131418 | undefined |
| detail:expert-down | 52.393207 | undefined |
| EUP | 0.602606 | undefined |
| experts-mix-normalize-up | 163.110628 | undefined |
| SH1 | 1.067695 | undefined |
| SH3 | 1.109143 | undefined |
| SH2 | 1.104945 | undefined |
| shared-expert-and-cleanup | 3.299816 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005921 | undefined |
| detail:read-ahead-wait | 1.962678 | undefined |
| total:layer | 173.638692 | undefined |

### Layer 80

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.003978 | undefined |
| pre-attention-aggregation | 0.017062 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.012132 | undefined |
| Q | 2.585050 | undefined |
| K | 2.528264 | undefined |
| V | 2.360181 | undefined |
| B | 0.023424 | undefined |
| FA | 0.035586 | undefined |
| FB | 0.050764 | undefined |
| G | 2.119611 | undefined |
| O | 2.071331 | undefined |
| attention | 12.495481 | undefined |
| attention-residual | 0.003146 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025878 | undefined |
| moe-scratch-and-parameter-bind | 0.002705 | undefined |
| router-and-top16 | 0.592998 | undefined |
| EDOWN | 0.587257 | undefined |
| prefetch-submit-and-latent-projection | 0.589462 | undefined |
| detail:expert-gate | 53.418876 | undefined |
| detail:expert-up | 52.711762 | undefined |
| detail:expert-activation | 0.136064 | undefined |
| detail:expert-down | 56.257850 | undefined |
| EUP | 0.604690 | undefined |
| experts-mix-normalize-up | 165.213277 | undefined |
| SH1 | 1.073376 | undefined |
| SH3 | 1.108932 | undefined |
| SH2 | 1.105145 | undefined |
| shared-expert-and-cleanup | 3.306528 | undefined |
| mlp-residual-cache-save-and-unbind | 0.381183 | undefined |
| detail:read-ahead-wait | 1.869363 | undefined |
| total:layer | 182.666361 | undefined |

### Layer 81

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004499 | undefined |
| pre-attention-aggregation | 0.017132 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011151 | undefined |
| Q | 2.603756 | undefined |
| K | 2.544355 | undefined |
| V | 2.365591 | undefined |
| B | 0.021570 | undefined |
| FA | 0.034525 | undefined |
| FB | 0.052769 | undefined |
| G | 2.107288 | undefined |
| O | 2.039632 | undefined |
| attention | 12.447100 | undefined |
| attention-residual | 0.003747 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026219 | undefined |
| moe-scratch-and-parameter-bind | 0.002395 | undefined |
| router-and-top16 | 0.571908 | undefined |
| EDOWN | 0.582028 | undefined |
| prefetch-submit-and-latent-projection | 0.584282 | undefined |
| detail:expert-gate | 52.373511 | undefined |
| detail:expert-up | 48.978084 | undefined |
| detail:expert-activation | 0.119403 | undefined |
| detail:expert-down | 52.037804 | undefined |
| EUP | 0.600812 | undefined |
| experts-mix-normalize-up | 156.086211 | undefined |
| SH1 | 1.078044 | undefined |
| SH3 | 1.109593 | undefined |
| SH2 | 1.104925 | undefined |
| shared-expert-and-cleanup | 3.310065 | undefined |
| mlp-residual-cache-save-and-unbind | 0.369069 | undefined |
| detail:read-ahead-wait | 1.783761 | undefined |
| total:layer | 173.455961 | undefined |

### Layer 82

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004498 | undefined |
| pre-attention-aggregation | 0.017863 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011441 | undefined |
| Q | 2.603304 | undefined |
| K | 2.539876 | undefined |
| V | 2.354550 | undefined |
| B | 0.024376 | undefined |
| FA | 0.031599 | undefined |
| FB | 0.051466 | undefined |
| G | 2.102549 | undefined |
| O | 2.061522 | undefined |
| attention | 12.446269 | undefined |
| attention-residual | 0.003397 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026059 | undefined |
| moe-scratch-and-parameter-bind | 0.002595 | undefined |
| router-and-top16 | 0.571648 | undefined |
| EDOWN | 0.578220 | undefined |
| prefetch-submit-and-latent-projection | 0.580355 | undefined |
| detail:expert-gate | 53.293430 | undefined |
| detail:expert-up | 51.213925 | undefined |
| detail:expert-activation | 0.136562 | undefined |
| detail:expert-down | 55.482814 | undefined |
| EUP | 0.607706 | undefined |
| experts-mix-normalize-up | 163.454840 | undefined |
| SH1 | 1.081381 | undefined |
| SH3 | 1.120384 | undefined |
| SH2 | 1.116777 | undefined |
| shared-expert-and-cleanup | 3.336765 | undefined |
| mlp-residual-cache-save-and-unbind | 0.368338 | undefined |
| detail:read-ahead-wait | 2.518248 | undefined |
| total:layer | 180.846060 | undefined |

### Layer 83

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.004949 | undefined |
| pre-attention-aggregation | 0.016611 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010200 | undefined |
| QA | 0.278139 | undefined |
| QB | 0.693055 | undefined |
| KA | 0.111358 | undefined |
| KB | 0.316972 | undefined |
| G | 2.143345 | undefined |
| O | 2.197536 | undefined |
| attention | 5.851765 | undefined |
| attention-residual | 0.002354 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027952 | undefined |
| moe-scratch-and-parameter-bind | 0.002314 | undefined |
| router-and-top16 | 0.618566 | undefined |
| EDOWN | 0.633694 | undefined |
| prefetch-submit-and-latent-projection | 0.635758 | undefined |
| detail:expert-gate | 55.089070 | undefined |
| detail:expert-up | 54.523300 | undefined |
| detail:expert-activation | 0.146143 | undefined |
| detail:expert-down | 57.032367 | undefined |
| EUP | 0.616421 | undefined |
| experts-mix-normalize-up | 174.904767 | undefined |
| SH1 | 1.114974 | undefined |
| SH3 | 1.121465 | undefined |
| SH2 | 1.105125 | undefined |
| shared-expert-and-cleanup | 3.360781 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006863 | undefined |
| detail:read-ahead-wait | 7.277741 | undefined |
| total:layer | 185.464651 | undefined |

### Layer 84

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005440 | undefined |
| pre-attention-aggregation | 0.017232 | undefined |
| snapshot-push | 0.002205 | undefined |
| pre-attention-normalization | 0.011712 | undefined |
| Q | 2.561267 | undefined |
| K | 2.523416 | undefined |
| V | 2.343429 | undefined |
| B | 0.024035 | undefined |
| FA | 0.033011 | undefined |
| FB | 0.048742 | undefined |
| G | 2.069216 | undefined |
| O | 2.063587 | undefined |
| attention | 12.411403 | undefined |
| attention-residual | 0.002063 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027050 | undefined |
| moe-scratch-and-parameter-bind | 0.002414 | undefined |
| router-and-top16 | 0.579353 | undefined |
| EDOWN | 0.579594 | undefined |
| prefetch-submit-and-latent-projection | 0.581867 | undefined |
| detail:expert-gate | 52.656240 | undefined |
| detail:expert-up | 51.371711 | undefined |
| detail:expert-activation | 0.127860 | undefined |
| detail:expert-down | 52.927696 | undefined |
| EUP | 0.601474 | undefined |
| experts-mix-normalize-up | 160.034710 | undefined |
| SH1 | 1.077193 | undefined |
| SH3 | 1.104444 | undefined |
| SH2 | 1.104043 | undefined |
| shared-expert-and-cleanup | 3.303544 | undefined |
| mlp-residual-cache-save-and-unbind | 0.397763 | undefined |
| detail:read-ahead-wait | 2.133197 | undefined |
| total:layer | 177.398988 | undefined |

### Layer 85

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.006142 | undefined |
| pre-attention-aggregation | 0.017803 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.010690 | undefined |
| Q | 2.605278 | undefined |
| K | 2.557669 | undefined |
| V | 2.373145 | undefined |
| B | 0.021741 | undefined |
| FA | 0.030196 | undefined |
| FB | 0.050995 | undefined |
| G | 2.108861 | undefined |
| O | 2.043439 | undefined |
| attention | 12.469462 | undefined |
| attention-residual | 0.004198 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026810 | undefined |
| moe-scratch-and-parameter-bind | 0.002745 | undefined |
| router-and-top16 | 0.571968 | undefined |
| EDOWN | 0.577109 | undefined |
| prefetch-submit-and-latent-projection | 0.579613 | undefined |
| detail:expert-gate | 52.787676 | undefined |
| detail:expert-up | 50.192817 | undefined |
| detail:expert-activation | 0.119613 | undefined |
| detail:expert-down | 55.338231 | undefined |
| EUP | 0.630188 | undefined |
| experts-mix-normalize-up | 166.541379 | undefined |
| SH1 | 1.111266 | undefined |
| SH3 | 1.134901 | undefined |
| SH2 | 1.114052 | undefined |
| shared-expert-and-cleanup | 3.378163 | undefined |
| mlp-residual-cache-save-and-unbind | 0.369431 | undefined |
| detail:read-ahead-wait | 7.277291 | undefined |
| total:layer | 184.000885 | undefined |

### Layer 86

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005550 | undefined |
| pre-attention-aggregation | 0.018044 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011201 | undefined |
| Q | 2.628552 | undefined |
| K | 2.546890 | undefined |
| V | 2.338661 | undefined |
| B | 0.037209 | undefined |
| FA | 0.030477 | undefined |
| FB | 0.053160 | undefined |
| G | 2.101287 | undefined |
| O | 2.051364 | undefined |
| attention | 12.455886 | undefined |
| attention-residual | 0.003536 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027321 | undefined |
| moe-scratch-and-parameter-bind | 0.002725 | undefined |
| router-and-top16 | 0.573793 | undefined |
| EDOWN | 0.581397 | undefined |
| prefetch-submit-and-latent-projection | 0.583520 | undefined |
| detail:expert-gate | 51.962040 | undefined |
| detail:expert-up | 48.197910 | undefined |
| detail:expert-activation | 0.118331 | undefined |
| detail:expert-down | 51.506464 | undefined |
| EUP | 0.602165 | undefined |
| experts-mix-normalize-up | 155.143459 | undefined |
| SH1 | 1.084917 | undefined |
| SH3 | 1.110415 | undefined |
| SH2 | 1.117228 | undefined |
| shared-expert-and-cleanup | 3.330033 | undefined |
| mlp-residual-cache-save-and-unbind | 0.368950 | undefined |
| detail:read-ahead-wait | 2.563630 | undefined |
| total:layer | 172.546881 | undefined |

### Layer 87

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005881 | undefined |
| pre-attention-aggregation | 0.018424 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010660 | undefined |
| QA | 0.271146 | undefined |
| QB | 0.702493 | undefined |
| KA | 0.108183 | undefined |
| KB | 0.314688 | undefined |
| G | 2.146872 | undefined |
| O | 2.169173 | undefined |
| attention | 5.825466 | undefined |
| attention-residual | 0.004208 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026219 | undefined |
| moe-scratch-and-parameter-bind | 0.002625 | undefined |
| router-and-top16 | 0.615109 | undefined |
| EDOWN | 0.621892 | undefined |
| prefetch-submit-and-latent-projection | 0.624207 | undefined |
| detail:expert-gate | 54.643825 | undefined |
| detail:expert-up | 53.381514 | undefined |
| detail:expert-activation | 0.138561 | undefined |
| detail:expert-down | 56.565938 | undefined |
| EUP | 0.791609 | undefined |
| experts-mix-normalize-up | 173.088704 | undefined |
| SH1 | 1.378526 | undefined |
| SH3 | 1.384007 | undefined |
| SH2 | 1.377093 | undefined |
| shared-expert-and-cleanup | 4.161276 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006252 | undefined |
| detail:read-ahead-wait | 7.345447 | undefined |
| total:layer | 184.411904 | undefined |

### Layer 88

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.006041 | undefined |
| pre-attention-aggregation | 0.019196 | undefined |
| snapshot-push | 0.000021 | undefined |
| pre-attention-normalization | 0.011962 | undefined |
| Q | 3.077101 | undefined |
| K | 3.069947 | undefined |
| V | 2.659990 | undefined |
| B | 0.021300 | undefined |
| FA | 0.031338 | undefined |
| FB | 0.062667 | undefined |
| G | 2.051875 | undefined |
| O | 2.051614 | undefined |
| attention | 13.758020 | undefined |
| attention-residual | 0.003587 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027101 | undefined |
| moe-scratch-and-parameter-bind | 0.002385 | undefined |
| router-and-top16 | 0.578541 | undefined |
| EDOWN | 0.586315 | undefined |
| prefetch-submit-and-latent-projection | 0.588570 | undefined |
| detail:expert-gate | 52.417634 | undefined |
| detail:expert-up | 48.917174 | undefined |
| detail:expert-activation | 0.127732 | undefined |
| detail:expert-down | 51.905937 | undefined |
| EUP | 0.603939 | undefined |
| experts-mix-normalize-up | 156.397874 | undefined |
| SH1 | 1.244305 | undefined |
| SH3 | 1.137034 | undefined |
| SH2 | 1.101709 | undefined |
| shared-expert-and-cleanup | 3.501443 | undefined |
| mlp-residual-cache-save-and-unbind | 0.395319 | undefined |
| detail:read-ahead-wait | 2.222752 | undefined |
| total:layer | 175.313271 | undefined |

### Layer 89

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005831 | undefined |
| pre-attention-aggregation | 0.018605 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011562 | undefined |
| Q | 2.625516 | undefined |
| K | 2.550597 | undefined |
| V | 2.329123 | undefined |
| B | 0.027301 | undefined |
| FA | 0.037550 | undefined |
| FB | 0.053370 | undefined |
| G | 2.066893 | undefined |
| O | 2.034041 | undefined |
| attention | 12.433204 | undefined |
| attention-residual | 0.004449 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027001 | undefined |
| moe-scratch-and-parameter-bind | 0.002755 | undefined |
| router-and-top16 | 0.573611 | undefined |
| EDOWN | 0.577830 | undefined |
| prefetch-submit-and-latent-projection | 0.580144 | undefined |
| detail:expert-gate | 54.934978 | undefined |
| detail:expert-up | 54.550712 | undefined |
| detail:expert-activation | 0.145661 | undefined |
| detail:expert-down | 57.054950 | undefined |
| EUP | 0.790638 | undefined |
| experts-mix-normalize-up | 175.069155 | undefined |
| SH1 | 1.426376 | undefined |
| SH3 | 1.370811 | undefined |
| SH2 | 1.366634 | undefined |
| shared-expert-and-cleanup | 4.185001 | undefined |
| mlp-residual-cache-save-and-unbind | 0.393024 | undefined |
| detail:read-ahead-wait | 7.355124 | undefined |
| total:layer | 193.327845 | undefined |

### Layer 90

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005610 | undefined |
| pre-attention-aggregation | 0.019086 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.011121 | undefined |
| Q | 3.169723 | undefined |
| K | 2.810141 | undefined |
| V | 2.354179 | undefined |
| B | 0.021941 | undefined |
| FA | 0.032661 | undefined |
| FB | 0.053180 | undefined |
| G | 2.093873 | undefined |
| O | 2.021377 | undefined |
| attention | 13.219604 | undefined |
| attention-residual | 0.003977 | undefined |
| pre-mlp-aggregation-and-normalization | 0.026740 | undefined |
| moe-scratch-and-parameter-bind | 0.002786 | undefined |
| router-and-top16 | 0.572951 | undefined |
| EDOWN | 0.579724 | undefined |
| prefetch-submit-and-latent-projection | 0.581958 | undefined |
| detail:expert-gate | 54.108904 | undefined |
| detail:expert-up | 53.556152 | undefined |
| detail:expert-activation | 0.135209 | undefined |
| detail:expert-down | 54.079670 | undefined |
| EUP | 0.606884 | undefined |
| experts-mix-normalize-up | 169.986296 | undefined |
| SH1 | 1.082734 | undefined |
| SH3 | 1.122437 | undefined |
| SH2 | 1.126665 | undefined |
| shared-expert-and-cleanup | 3.349800 | undefined |
| mlp-residual-cache-save-and-unbind | 0.376594 | undefined |
| detail:read-ahead-wait | 7.268904 | undefined |
| total:layer | 188.180095 | undefined |

### Layer 91

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005209 | undefined |
| pre-attention-aggregation | 0.018464 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010990 | undefined |
| QA | 0.278761 | undefined |
| QB | 0.687736 | undefined |
| KA | 0.106870 | undefined |
| KB | 0.327983 | undefined |
| G | 2.166288 | undefined |
| O | 2.171869 | undefined |
| attention | 5.851575 | undefined |
| attention-residual | 0.003997 | undefined |
| pre-mlp-aggregation-and-normalization | 0.027742 | undefined |
| moe-scratch-and-parameter-bind | 0.002214 | undefined |
| router-and-top16 | 0.622914 | undefined |
| EDOWN | 0.622473 | undefined |
| prefetch-submit-and-latent-projection | 0.624577 | undefined |
| detail:expert-gate | 54.870538 | undefined |
| detail:expert-up | 54.275651 | undefined |
| detail:expert-activation | 0.150910 | undefined |
| detail:expert-down | 56.775300 | undefined |
| EUP | 0.601825 | undefined |
| experts-mix-normalize-up | 174.091728 | undefined |
| SH1 | 1.075580 | undefined |
| SH3 | 1.108291 | undefined |
| SH2 | 1.109744 | undefined |
| shared-expert-and-cleanup | 3.312160 | undefined |
| mlp-residual-cache-save-and-unbind | 0.006252 | undefined |
| detail:read-ahead-wait | 7.179693 | undefined |
| total:layer | 184.601527 | undefined |

### Layer 92

| Measurement | ms | Calls |
|---|---:|---:|
| binding-and-stored-folds | 0.005961 | undefined |
| pre-attention-aggregation | 0.017112 | undefined |
| snapshot-push | 0.000020 | undefined |
| pre-attention-normalization | 0.010700 | undefined |
| QA | 0.274012 | undefined |
| QB | 0.709045 | undefined |
| KA | 0.107180 | undefined |
| KB | 0.313246 | undefined |
| G | 2.044781 | undefined |
| O | 2.053518 | undefined |
| attention | 5.629510 | undefined |
| attention-residual | 0.004058 | undefined |
| pre-mlp-aggregation-and-normalization | 0.025708 | undefined |
| moe-scratch-and-parameter-bind | 0.002023 | undefined |
| router-and-top16 | 0.603438 | undefined |
| EDOWN | 0.619588 | undefined |
| prefetch-submit-and-latent-projection | 0.621832 | undefined |
| detail:expert-gate | 54.908279 | undefined |
| detail:expert-up | 54.188404 | undefined |
| detail:expert-activation | 0.140174 | undefined |
| detail:expert-down | 56.670071 | undefined |
| EUP | 0.608847 | undefined |
| experts-mix-normalize-up | 173.979238 | undefined |
| SH1 | 1.082433 | undefined |
| SH3 | 1.111306 | undefined |
| SH2 | 1.101598 | undefined |
| shared-expert-and-cleanup | 3.314464 | undefined |
| mlp-residual-cache-save-and-unbind | 0.005490 | undefined |
| detail:read-ahead-wait | 7.234378 | undefined |
| total:layer | 184.243319 | undefined |

