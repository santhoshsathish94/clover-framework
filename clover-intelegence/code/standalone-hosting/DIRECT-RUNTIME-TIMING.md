# Current Runtime Timing

AX102 2026-10-03, direct expert format, default config, 16 threads; 2 input positions and 1 continuation; timing only.

Source: [timing data](stage-profile-direct-20261003.json). Output assertions passed: IDs 418 then 276 (Rain falls -> on the). All 279 layer totals are present, with 35-50 recorded measurements per layer, depending on layer type. These are boundaries and nested measurements, not a claim that every scalar operation has a separate timer.

Startup: **14.225284 s**. First output (two input positions): **33.930719 s**. Next output (one continuation position): **10.925254 s**. This measures the current source with profiling enabled; it is not a retroactive decomposition of the earlier 45.028-second run.

## Elapsed-Time Breakdown

| Work | First output, two input positions (s) | Next output (s) |
|---|---:|---:|
| Routed experts, mixing, latent norm and up projection | 14.704353 | 6.830807 |
| Vocabulary head preparation and projection | 15.892511 | 0.051031 |
| Attention across all 93 layers | 2.268479 | 2.828333 |
| Shared experts | 0.685523 | 0.796524 |
| Residual/state save, unbind and next-layer prediction | 0.058866 | 0.026238 |
| Latent down projection | 0.129293 | 0.177410 |
| Live router and top-16 | 0.117997 | 0.155040 |
| Layer-0 dense MLP | 0.038790 | 0.033750 |
| Other boundaries and unassigned timer overhead | 0.034907 | 0.026121 |
| Total | 33.930719 | 10.925254 |

First head use prepares the retained BF16 table and scores it. The head timer does not separate reading, decompression, layout conversion, checksums and scoring. The next output reuses that head table, not a cached answer. Dataset preparation is not run.

## Inside Routed Experts

Gate/up/down wall timers are nested inside routed-expert processing above; do not add them again. Each includes its block decoding, validation and multiplication.

| Nested measurement | First output (s) | Next output (s) |
|---|---:|---:|
| detail:expert-gate | 4.917797 | 2.254950 |
| detail:expert-up | 4.651364 | 2.199623 |
| detail:expert-activation | 0.019157 | 0.010602 |
| detail:expert-down | 4.872598 | 2.169172 |
| detail:read-ahead-wait | 0.000000 | 0.000000 |

Worker times below sum concurrent threads. They locate CPU work but are **not elapsed seconds**, and must not be added to the wall-time table. The read wrapper accesses already-prefetched bytes; it is not total disk I/O time.

| Worker phase | First output (summed worker s) | Next output (summed worker s) |
|---|---:|---:|
| read | 0.000000 | 0.000000 |
| decode | 0.000000 | 0.000000 |
| crc | 0.000000 | 0.000000 |
| math | 185.367781 | 82.552469 |

## Every Layer

Elapsed milliseconds. Layer 0 is dense; all remaining layers have routed and shared experts.

| Layer | Attention | First input | Second input | Continuation |
|---:|---|---:|---:|---:|
| 0 | KDA | 35.088953 | 37.518072 | 54.921839 |
| 1 | KDA | 93.190294 | 95.200389 | 121.832210 |
| 2 | KDA | 95.271823 | 95.773019 | 92.368728 |
| 3 | MLA | 81.017218 | 85.557852 | 80.772931 |
| 4 | KDA | 84.012495 | 83.020462 | 84.540953 |
| 5 | KDA | 95.746750 | 95.864230 | 75.841107 |
| 6 | KDA | 89.866153 | 90.526316 | 91.154489 |
| 7 | MLA | 93.533095 | 84.188643 | 87.140670 |
| 8 | KDA | 100.178411 | 99.359570 | 90.643083 |
| 9 | KDA | 86.981633 | 89.761658 | 70.553467 |
| 10 | KDA | 90.557274 | 86.479925 | 79.127037 |
| 11 | MLA | 87.448616 | 79.694257 | 64.909922 |
| 12 | KDA | 94.662775 | 90.470912 | 78.462815 |
| 13 | KDA | 95.068613 | 89.706213 | 85.739391 |
| 14 | KDA | 95.055820 | 95.248289 | 92.592356 |
| 15 | MLA | 89.953796 | 86.826272 | 78.328003 |
| 16 | KDA | 87.886765 | 85.441105 | 77.166974 |
| 17 | KDA | 94.248311 | 89.910164 | 80.191766 |
| 18 | KDA | 88.530797 | 88.681950 | 73.813678 |
| 19 | MLA | 90.317236 | 85.663590 | 64.666767 |
| 20 | KDA | 93.795244 | 84.775641 | 85.490226 |
| 21 | KDA | 94.518465 | 97.480460 | 92.586916 |
| 22 | KDA | 93.490474 | 88.554090 | 92.427869 |
| 23 | MLA | 77.513282 | 90.472656 | 84.742639 |
| 24 | KDA | 81.497135 | 75.789530 | 78.888370 |
| 25 | KDA | 89.868727 | 92.747847 | 96.878915 |
| 26 | KDA | 86.371953 | 144.575795 | 93.505903 |
| 27 | MLA | 85.612836 | 90.581008 | 86.737156 |
| 28 | KDA | 97.593130 | 97.636691 | 101.304605 |
| 29 | KDA | 89.098908 | 94.346794 | 87.809569 |
| 30 | KDA | 102.931555 | 95.456076 | 104.859987 |
| 31 | MLA | 59.913516 | 84.652551 | 87.555174 |
| 32 | KDA | 50.917428 | 94.701106 | 92.200855 |
| 33 | KDA | 89.894766 | 249.568401 | 100.485103 |
| 34 | KDA | 62.434587 | 95.451419 | 99.027970 |
| 35 | MLA | 48.937599 | 88.841308 | 93.763765 |
| 36 | KDA | 90.492042 | 92.681222 | 117.674100 |
| 37 | KDA | 99.830551 | 91.741877 | 102.132250 |
| 38 | KDA | 105.572379 | 91.454189 | 90.721931 |
| 39 | MLA | 52.492120 | 85.396892 | 65.401761 |
| 40 | KDA | 45.053802 | 89.553638 | 87.016999 |
| 41 | KDA | 93.549094 | 93.279520 | 109.045939 |
| 42 | KDA | 66.584411 | 71.657069 | 114.177858 |
| 43 | MLA | 87.720153 | 77.897359 | 92.734702 |
| 44 | KDA | 86.540478 | 90.780030 | 112.099945 |
| 45 | KDA | 79.488923 | 79.689397 | 110.198282 |
| 46 | KDA | 67.320817 | 80.099854 | 112.349331 |
| 47 | MLA | 84.376766 | 76.555502 | 103.553266 |
| 48 | KDA | 101.090364 | 97.690903 | 111.848505 |
| 49 | KDA | 82.381828 | 82.795401 | 105.653169 |
| 50 | KDA | 95.288023 | 83.703989 | 112.417308 |
| 51 | MLA | 85.976946 | 83.414768 | 110.785750 |
| 52 | KDA | 98.158376 | 90.781282 | 104.021741 |
| 53 | KDA | 94.405104 | 90.648003 | 119.603265 |
| 54 | KDA | 93.262578 | 89.519484 | 124.689989 |
| 55 | MLA | 89.169290 | 82.376047 | 102.790550 |
| 56 | KDA | 101.946564 | 89.876842 | 101.058073 |
| 57 | KDA | 100.168822 | 98.456333 | 87.287454 |
| 58 | KDA | 94.042506 | 91.558915 | 96.959706 |
| 59 | MLA | 95.110611 | 91.175749 | 80.583818 |
| 60 | KDA | 97.865438 | 95.036884 | 100.797627 |
| 61 | KDA | 99.139640 | 95.956853 | 92.298958 |
| 62 | KDA | 98.006052 | 97.658361 | 103.382156 |
| 63 | MLA | 93.910889 | 90.792213 | 76.204185 |
| 64 | KDA | 100.843904 | 99.580272 | 100.241598 |
| 65 | KDA | 100.204910 | 100.095005 | 87.732225 |
| 66 | KDA | 103.975115 | 98.782503 | 98.897436 |
| 67 | MLA | 90.949347 | 87.008312 | 141.687729 |
| 68 | KDA | 99.396710 | 98.936730 | 149.301565 |
| 69 | KDA | 97.497471 | 100.586954 | 114.845736 |
| 70 | KDA | 103.000233 | 99.465107 | 175.445624 |
| 71 | MLA | 95.224854 | 93.935075 | 101.171866 |
| 72 | KDA | 101.509318 | 100.988244 | 132.796847 |
| 73 | KDA | 101.808807 | 103.334047 | 142.298760 |
| 74 | KDA | 94.610778 | 95.958286 | 142.222909 |
| 75 | MLA | 77.126539 | 77.807431 | 106.007932 |
| 76 | KDA | 97.703086 | 102.313289 | 223.388083 |
| 77 | KDA | 97.474639 | 102.951271 | 548.497219 |
| 78 | KDA | 90.731830 | 103.893493 | 504.146131 |
| 79 | MLA | 77.288932 | 95.741189 | 291.276630 |
| 80 | KDA | 87.226460 | 219.897382 | 174.347382 |
| 81 | KDA | 83.191992 | 353.324787 | 148.727382 |
| 82 | KDA | 93.504060 | 607.124564 | 140.058986 |
| 83 | MLA | 91.797071 | 81.920326 | 105.051595 |
| 84 | KDA | 102.925303 | 92.534047 | 142.685243 |
| 85 | KDA | 98.560968 | 95.342144 | 126.447934 |
| 86 | KDA | 91.966026 | 87.668275 | 135.107254 |
| 87 | MLA | 89.410662 | 77.910644 | 126.528154 |
| 88 | KDA | 91.057819 | 95.541347 | 151.854655 |
| 89 | KDA | 101.087620 | 90.015210 | 151.217034 |
| 90 | KDA | 98.447998 | 168.094309 | 160.702888 |
| 91 | MLA | 87.142142 | 182.404417 | 149.846894 |
| 92 | MLA | 88.703189 | 156.236934 | 147.820398 |

## Every Recorded Measurement

Milliseconds. Boundary rows are disjoint within their layer; projections, operator aggregates, details and worker rows are nested or parallel and cannot be summed with them.

### Input

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| request-reset | boundary | 2.864041 / 1 | 0.154408 / 1 | 0.190707 / 1 |
| embedding | boundary | 1.976012 / 1 | 1.803229 / 1 | 1.745070 / 1 |
| scratch-setup | boundary | 0.009848 / 1 | 0.000250 / 1 | 0.000440 / 1 |

### Layer 0

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.016691 / 1 | 0.006573 / 1 | 0.007354 / 1 |
| pre-attention-aggregation | boundary | 0.012653 / 1 | 0.001222 / 1 | 0.001393 / 1 |
| snapshot-push | boundary | 0.001523 / 1 | 0.000912 / 1 | 0.000721 / 1 |
| pre-attention-normalization | boundary | 2.726835 / 1 | 0.021801 / 1 | 0.040195 / 1 |
| Q | nested | 3.582634 / 1 | 3.107276 / 1 | 5.790830 / 1 |
| K | nested | 2.890140 / 1 | 3.024842 / 1 | 3.252327 / 1 |
| V | nested | 2.365139 / 1 | 2.925637 / 1 | 2.585361 / 1 |
| B | nested | 0.022051 / 1 | 0.026008 / 1 | 0.027692 / 1 |
| FA | nested | 0.033864 / 1 | 0.034193 / 1 | 0.032481 / 1 |
| FB | nested | 0.066214 / 1 | 0.052237 / 1 | 0.120345 / 1 |
| G | nested | 2.182648 / 1 | 2.570203 / 1 | 3.570752 / 1 |
| O | nested | 2.192427 / 1 | 2.715223 / 1 | 4.425479 / 1 |
| attention | boundary | 14.579752 / 1 | 15.568459 / 1 | 20.732047 / 1 |
| attention-residual | boundary | 0.002064 / 1 | 0.001663 / 1 | 0.001723 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.041107 / 1 | 0.026129 / 1 | 0.025628 / 1 |
| MGATE | nested | 5.789678 / 1 | 7.375280 / 1 | 12.939216 / 1 |
| MUP | nested | 5.824643 / 1 | 7.127307 / 1 | 9.590620 / 1 |
| MDOWN | nested | 5.671046 / 1 | 6.878913 / 1 | 11.166404 / 1 |
| dense-mlp | boundary | 17.328428 / 1 | 21.461250 / 1 | 33.749880 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000491 / 1 | 0.000320 / 1 | 0.000511 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.378497 / 1 | 0.428772 / 1 | 0.362057 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| op:Q   int8 projection | nested | 30.619130 / 1 | 35.835577 / 1 | 53.500177 / 1 |
| op:N   rmsnorm | nested | 2.742073 / 1 | 0.038261 / 1 | 0.059612 / 1 |
| op:L   l2 per-head | nested | 0.005820 / 1 | 0.005280 / 1 | 0.006161 / 1 |
| op:SiTU + sigma | nested | 0.036939 / 1 | 0.052408 / 1 | 0.050886 / 1 |
| op:C   shortconv | nested | 0.098364 / 1 | 0.063098 / 1 | 0.064080 / 1 |
| op:AR  snapshot aggregate | nested | 0.023564 / 1 | 0.016601 / 1 | 0.014998 / 1 |
| op:D   kda delta-rule | nested | 0.211465 / 1 | 0.144059 / 1 | 0.142968 / 1 |
| op:alpha / beta / gate | nested | 0.084198 / 1 | 0.082785 / 1 | 0.094757 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 35.088953 / 1 | 37.518072 / 1 | 54.921839 / 1 |

### Layer 1

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004999 / 1 | 0.005370 / 1 | 0.114314 / 1 |
| pre-attention-aggregation | boundary | 0.015519 / 1 | 0.018294 / 1 | 0.017552 / 1 |
| snapshot-push | boundary | 0.000040 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012784 / 1 | 0.011382 / 1 | 0.012434 / 1 |
| Q | nested | 2.450468 / 1 | 2.462691 / 1 | 4.366769 / 1 |
| K | nested | 2.360340 / 1 | 2.280511 / 1 | 3.239853 / 1 |
| V | nested | 2.136252 / 1 | 2.203387 / 1 | 2.720373 / 1 |
| B | nested | 0.019397 / 1 | 0.021029 / 1 | 0.436976 / 1 |
| FA | nested | 0.028102 / 1 | 0.029004 / 1 | 0.240429 / 1 |
| FB | nested | 0.046587 / 1 | 0.046357 / 1 | 0.377445 / 1 |
| G | nested | 2.004966 / 1 | 1.906131 / 1 | 8.229025 / 1 |
| O | nested | 2.011017 / 1 | 2.161659 / 1 | 7.491879 / 1 |
| attention | boundary | 11.946702 / 1 | 12.193713 / 1 | 27.763896 / 1 |
| attention-residual | boundary | 0.003036 / 1 | 0.002695 / 1 | 0.001994 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025017 / 1 | 0.025808 / 1 | 0.026359 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.002475 / 1 | 0.407902 / 1 |
| router-and-top16 | boundary | 0.564925 / 1 | 0.591405 / 1 | 2.631477 / 1 |
| EDOWN | nested | 0.595903 / 1 | 0.582829 / 1 | 2.405113 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.596274 / 1 | 0.583220 / 1 | 2.405624 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.098744 / 1 | 0.033623 / 1 | 0.011050 / 1 |
| SH1 | nested | 1.019395 / 1 | 0.969592 / 1 | 3.875081 / 1 |
| SH3 | nested | 1.018893 / 1 | 0.969011 / 1 | 3.996938 / 1 |
| SH2 | nested | 1.006220 / 1 | 0.982676 / 1 | 4.111082 / 1 |
| shared-expert-during-read | boundary | 3.054507 / 1 | 2.931377 / 1 | 11.996024 / 1 |
| detail:expert-gate | nested | 25.684121 / 16 | 25.915820 / 16 | 26.107102 / 16 |
| detail:expert-up | nested | 24.281770 / 16 | 24.363554 / 16 | 23.740389 / 16 |
| detail:expert-activation | nested | 0.100774 / 16 | 0.098826 / 16 | 0.110397 / 16 |
| detail:expert-down | nested | 25.322884 / 16 | 26.812557 / 16 | 23.259557 / 16 |
| EUP | nested | 0.693306 / 1 | 0.790898 / 1 | 2.491675 / 1 |
| experts-mix-normalize-up | boundary | 76.492875 / 1 | 78.342050 / 1 | 76.052201 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002534 / 1 | 0.002405 / 1 | 0.002966 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000140 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.367436 / 1 | 0.454609 / 1 | 0.385671 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1048.760188 / 1 | 1039.587002 / 1 | 940.333043 / 1 |
| op:Q   int8 projection | nested | 15.389442 / 1 | 15.404030 / 1 | 43.980804 / 1 |
| op:X   mxfp4 expert proj | nested | 75.401728 / 1 | 77.203632 / 1 | 73.229387 / 1 |
| op:N   rmsnorm | nested | 0.034786 / 1 | 0.031959 / 1 | 0.024146 / 1 |
| op:L   l2 per-head | nested | 0.006633 / 1 | 0.004709 / 1 | 0.005981 / 1 |
| op:SiTU + sigma | nested | 0.105535 / 1 | 0.106658 / 1 | 0.121037 / 1 |
| op:C   shortconv | nested | 0.056566 / 1 | 0.056035 / 1 | 0.036609 / 1 |
| op:AR  snapshot aggregate | nested | 0.030497 / 1 | 0.033773 / 1 | 0.031860 / 1 |
| op:D   kda delta-rule | nested | 0.203090 / 1 | 0.138639 / 1 | 0.162033 / 1 |
| op:router dot product | nested | 0.562090 / 1 | 0.588550 / 1 | 2.627971 / 1 |
| op:top-k selection | nested | 0.002404 / 1 | 0.002444 / 1 | 0.002645 / 1 |
| op:alpha / beta / gate | nested | 0.074930 / 1 | 0.088766 / 1 | 0.058379 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.190294 / 1 | 95.200389 / 1 | 121.832210 / 1 |

### Layer 2

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005079 / 1 | 0.004839 / 1 | 0.006422 / 1 |
| pre-attention-aggregation | boundary | 0.018926 / 1 | 0.017012 / 1 | 0.015609 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010840 / 1 | 0.011271 / 1 | 0.011412 / 1 |
| Q | nested | 2.758765 / 1 | 3.169812 / 1 | 4.978773 / 1 |
| K | nested | 2.714492 / 1 | 2.835008 / 1 | 3.566303 / 1 |
| V | nested | 2.564431 / 1 | 2.798860 / 1 | 2.822714 / 1 |
| B | nested | 0.022872 / 1 | 0.028383 / 1 | 0.126556 / 1 |
| FA | nested | 0.029946 / 1 | 0.032891 / 1 | 0.038772 / 1 |
| FB | nested | 0.055593 / 1 | 0.050664 / 1 | 0.065653 / 1 |
| G | nested | 2.370840 / 1 | 2.704112 / 1 | 4.286069 / 1 |
| O | nested | 2.302702 / 1 | 2.654499 / 1 | 3.246767 / 1 |
| attention | boundary | 13.816846 / 1 | 15.322700 / 1 | 19.731618 / 1 |
| attention-residual | boundary | 0.003316 / 1 | 0.002995 / 1 | 0.002234 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025007 / 1 | 0.025367 / 1 | 0.026419 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002635 / 1 | 0.002405 / 1 | 0.003277 / 1 |
| router-and-top16 | boundary | 0.601214 / 1 | 0.602115 / 1 | 0.741806 / 1 |
| EDOWN | nested | 0.661686 / 1 | 0.760311 / 1 | 0.935468 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.662227 / 1 | 0.760772 / 1 | 0.935979 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.091662 / 1 | 0.023073 / 1 | 0.015269 / 1 |
| SH1 | nested | 1.153456 / 1 | 1.349472 / 1 | 1.637660 / 1 |
| SH3 | nested | 1.137766 / 1 | 1.351876 / 1 | 1.633783 / 1 |
| SH2 | nested | 1.142104 / 1 | 1.356595 / 1 | 1.807096 / 1 |
| shared-expert-during-read | boundary | 3.445067 / 1 | 4.070636 / 1 | 5.091363 / 1 |
| detail:expert-gate | nested | 25.546385 / 16 | 25.096741 / 16 | 21.470168 / 16 |
| detail:expert-up | nested | 24.294969 / 16 | 24.234420 / 16 | 20.575374 / 16 |
| detail:expert-activation | nested | 0.098063 / 16 | 0.099416 / 16 | 0.095539 / 16 |
| detail:expert-down | nested | 25.186822 / 16 | 24.134264 / 16 | 22.021116 / 16 |
| EUP | nested | 0.616522 / 1 | 0.586536 / 1 | 0.919428 / 1 |
| experts-mix-normalize-up | boundary | 76.160263 / 1 | 74.500181 / 1 | 65.418643 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002474 / 1 | 0.002114 / 1 | 0.002796 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000170 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.423541 / 1 | 0.425294 / 1 | 0.363609 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1031.767123 / 1 | 996.318210 / 1 | 880.031024 / 1 |
| op:Q   int8 projection | nested | 17.529631 / 1 | 19.677506 / 1 | 26.063349 / 1 |
| op:X   mxfp4 expert proj | nested | 75.140477 / 1 | 73.579309 / 1 | 64.176410 / 1 |
| op:N   rmsnorm | nested | 0.039914 / 1 | 0.036187 / 1 | 0.023684 / 1 |
| op:L   l2 per-head | nested | 0.004960 / 1 | 0.006021 / 1 | 0.005780 / 1 |
| op:SiTU + sigma | nested | 0.107251 / 1 | 0.109582 / 1 | 0.105918 / 1 |
| op:C   shortconv | nested | 0.079147 / 1 | 0.054212 / 1 | 0.041137 / 1 |
| op:AR  snapshot aggregate | nested | 0.034133 / 1 | 0.032180 / 1 | 0.031159 / 1 |
| op:D   kda delta-rule | nested | 0.198982 / 1 | 0.128260 / 1 | 0.184465 / 1 |
| op:router dot product | nested | 0.598238 / 1 | 0.598769 / 1 | 0.738640 / 1 |
| op:top-k selection | nested | 0.002635 / 1 | 0.002565 / 1 | 0.002705 / 1 |
| op:alpha / beta / gate | nested | 0.075471 / 1 | 0.072495 / 1 | 0.056415 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.271823 / 1 | 95.773019 / 1 | 92.368728 / 1 |

### Layer 3

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005199 / 1 | 0.005199 / 1 | 0.006892 / 1 |
| pre-attention-aggregation | boundary | 0.016921 / 1 | 0.017873 / 1 | 0.015118 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011331 / 1 | 0.010610 / 1 | 0.011131 / 1 |
| QA | nested | 0.310951 / 1 | 0.296524 / 1 | 0.447486 / 1 |
| QB | nested | 0.712622 / 1 | 0.690921 / 1 | 1.146913 / 1 |
| KA | nested | 0.104996 / 1 | 0.101980 / 1 | 0.295923 / 1 |
| KB | nested | 0.319437 / 1 | 0.303838 / 1 | 0.622043 / 1 |
| G | nested | 2.021908 / 1 | 1.951186 / 1 | 3.075848 / 1 |
| O | nested | 1.982855 / 1 | 1.936038 / 1 | 3.134457 / 1 |
| attention | boundary | 5.603861 / 1 | 5.374542 / 1 | 8.821443 / 1 |
| attention-residual | boundary | 0.002895 / 1 | 0.002134 / 1 | 0.001553 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023825 / 1 | 0.024576 / 1 | 0.025218 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002665 / 1 | 0.002395 / 1 | 0.003126 / 1 |
| router-and-top16 | boundary | 0.572150 / 1 | 0.569434 / 1 | 0.693936 / 1 |
| EDOWN | nested | 0.576227 / 1 | 0.567330 / 1 | 1.039662 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.576918 / 1 | 0.567720 / 1 | 1.040063 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.083165 / 1 | 0.031799 / 1 | 0.015419 / 1 |
| SH1 | nested | 1.012882 / 1 | 0.963451 / 1 | 1.635306 / 1 |
| SH3 | nested | 1.020717 / 1 | 0.959663 / 1 | 1.871948 / 1 |
| SH2 | nested | 0.988257 / 1 | 0.967057 / 1 | 1.739460 / 1 |
| shared-expert-during-read | boundary | 3.031694 / 1 | 2.900540 / 1 | 5.257013 / 1 |
| detail:expert-gate | nested | 24.162515 / 16 | 24.852906 / 16 | 21.306842 / 16 |
| detail:expert-up | nested | 23.027907 / 16 | 24.195428 / 16 | 21.022369 / 16 |
| detail:expert-activation | nested | 0.105344 / 16 | 0.105898 / 16 | 0.100387 / 16 |
| detail:expert-down | nested | 22.775174 / 16 | 25.637568 / 16 | 21.097581 / 16 |
| EUP | nested | 0.592918 / 1 | 0.889001 / 1 | 1.022250 / 1 |
| experts-mix-normalize-up | boundary | 71.073439 / 1 | 76.039538 / 1 | 64.871100 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003116 / 1 | 0.001974 / 1 | 0.002224 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000241 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007665 / 1 | 0.007104 / 1 | 0.006473 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 943.161132 / 1 | 915.530165 / 1 | 853.415939 / 1 |
| op:Q   int8 projection | nested | 9.642319 / 1 | 9.625776 / 1 | 16.029763 / 1 |
| op:X   mxfp4 expert proj | nested | 70.085633 / 1 | 74.807032 / 1 | 63.541575 / 1 |
| op:N   rmsnorm | nested | 0.026441 / 1 | 0.024607 / 1 | 0.023032 / 1 |
| op:SiTU + sigma | nested | 0.112879 / 1 | 0.114084 / 1 | 0.108293 / 1 |
| op:AR  snapshot aggregate | nested | 0.030698 / 1 | 0.032391 / 1 | 0.029305 / 1 |
| op:SA  softmax attention | nested | 0.014457 / 1 | 0.009507 / 1 | 0.011812 / 1 |
| op:router dot product | nested | 0.569113 / 1 | 0.566568 / 1 | 0.690951 / 1 |
| op:top-k selection | nested | 0.002595 / 1 | 0.002375 / 1 | 0.002645 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 81.017218 / 1 | 85.557852 / 1 | 80.772931 / 1 |

### Layer 4

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004749 / 1 | 0.003958 / 1 | 0.006292 / 1 |
| pre-attention-aggregation | boundary | 0.014327 / 1 | 0.014527 / 1 | 0.014648 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010780 / 1 | 0.009978 / 1 | 0.011421 / 1 |
| Q | nested | 2.524527 / 1 | 2.869872 / 1 | 4.631103 / 1 |
| K | nested | 2.361773 / 1 | 2.364738 / 1 | 3.546677 / 1 |
| V | nested | 2.264732 / 1 | 2.261505 / 1 | 2.871596 / 1 |
| B | nested | 0.021180 / 1 | 0.020708 / 1 | 0.027732 / 1 |
| FA | nested | 0.026489 / 1 | 0.027050 / 1 | 0.199162 / 1 |
| FB | nested | 0.045905 / 1 | 0.044964 / 1 | 0.062266 / 1 |
| G | nested | 1.974720 / 1 | 1.927171 / 1 | 4.228059 / 1 |
| O | nested | 2.052636 / 1 | 2.041194 / 1 | 3.370157 / 1 |
| attention | boundary | 12.408084 / 1 | 12.652951 / 1 | 19.706561 / 1 |
| attention-residual | boundary | 0.003026 / 1 | 0.001844 / 1 | 0.002004 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025487 / 1 | 0.024306 / 1 | 0.026340 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002464 / 1 | 0.002424 / 1 | 0.003426 / 1 |
| router-and-top16 | boundary | 0.582237 / 1 | 0.585073 / 1 | 0.690320 / 1 |
| EDOWN | nested | 0.596535 / 1 | 0.579002 / 1 | 0.967558 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.597146 / 1 | 0.579493 / 1 | 0.967948 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.096009 / 1 | 0.032280 / 1 | 0.015169 / 1 |
| SH1 | nested | 1.155519 / 1 | 0.983177 / 1 | 1.743558 / 1 |
| SH3 | nested | 1.011881 / 1 | 0.993376 / 1 | 1.554916 / 1 |
| SH2 | nested | 0.992404 / 1 | 0.989409 / 1 | 1.585132 / 1 |
| shared-expert-during-read | boundary | 3.172307 / 1 | 2.976331 / 1 | 4.895908 / 1 |
| detail:expert-gate | nested | 22.190093 / 16 | 21.767041 / 16 | 19.153435 / 16 |
| detail:expert-up | nested | 21.510294 / 16 | 21.134332 / 16 | 18.633115 / 16 |
| detail:expert-activation | nested | 0.100529 / 16 | 0.097333 / 16 | 0.104486 / 16 |
| detail:expert-down | nested | 21.743927 / 16 | 21.578180 / 16 | 18.526218 / 16 |
| EUP | nested | 0.692183 / 1 | 0.755972 / 1 | 1.053378 / 1 |
| experts-mix-normalize-up | boundary | 66.658029 / 1 | 65.697564 / 1 | 57.824413 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002635 / 1 | 0.002655 / 1 | 0.002645 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000150 / 1 | 0.000371 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.432358 / 1 | 0.434592 / 1 | 0.371053 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 878.397525 / 1 | 877.995637 / 1 | 738.920449 / 1 |
| op:Q   int8 projection | nested | 15.718979 / 1 | 15.856678 / 1 | 25.839540 / 1 |
| op:X   mxfp4 expert proj | nested | 65.560958 / 1 | 64.592739 / 1 | 56.432711 / 1 |
| op:N   rmsnorm | nested | 0.033402 / 1 | 0.030197 / 1 | 0.023665 / 1 |
| op:L   l2 per-head | nested | 0.005871 / 1 | 0.005280 / 1 | 0.005791 / 1 |
| op:SiTU + sigma | nested | 0.107711 / 1 | 0.104626 / 1 | 0.113982 / 1 |
| op:C   shortconv | nested | 0.051926 / 1 | 0.065421 / 1 | 0.036909 / 1 |
| op:AR  snapshot aggregate | nested | 0.030006 / 1 | 0.029285 / 1 | 0.029454 / 1 |
| op:D   kda delta-rule | nested | 0.148427 / 1 | 0.136355 / 1 | 0.151122 / 1 |
| op:router dot product | nested | 0.579473 / 1 | 0.582278 / 1 | 0.687424 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002364 / 1 | 0.002495 / 1 |
| op:alpha / beta / gate | nested | 0.073247 / 1 | 0.079037 / 1 | 0.056245 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 84.012495 / 1 | 83.020462 / 1 | 84.540953 / 1 |

### Layer 5

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005300 / 1 | 0.004809 / 1 | 0.006111 / 1 |
| pre-attention-aggregation | boundary | 0.017021 / 1 | 0.017343 / 1 | 0.015589 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012062 / 1 | 0.011512 / 1 | 0.011592 / 1 |
| Q | nested | 2.784693 / 1 | 2.981480 / 1 | 4.822040 / 1 |
| K | nested | 2.718238 / 1 | 2.894679 / 1 | 3.180964 / 1 |
| V | nested | 2.609696 / 1 | 2.817084 / 1 | 2.562187 / 1 |
| B | nested | 0.023373 / 1 | 0.025537 / 1 | 0.026991 / 1 |
| FA | nested | 0.029325 / 1 | 0.032490 / 1 | 0.130564 / 1 |
| FB | nested | 0.050755 / 1 | 0.049061 / 1 | 0.191238 / 1 |
| G | nested | 2.296932 / 1 | 2.469714 / 1 | 3.325855 / 1 |
| O | nested | 2.322450 / 1 | 2.585430 / 1 | 2.952517 / 1 |
| attention | boundary | 13.951848 / 1 | 14.944344 / 1 | 17.916607 / 1 |
| attention-residual | boundary | 0.002945 / 1 | 0.002525 / 1 | 0.002004 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025087 / 1 | 0.025187 / 1 | 0.026981 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002605 / 1 | 0.002625 / 1 | 0.003637 / 1 |
| router-and-top16 | boundary | 0.608637 / 1 | 0.605512 / 1 | 0.690741 / 1 |
| EDOWN | nested | 0.685481 / 1 | 0.763457 / 1 | 0.937802 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.685862 / 1 | 0.763858 / 1 | 0.938303 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.084127 / 1 | 0.018023 / 1 | 0.007464 / 1 |
| SH1 | nested | 1.142885 / 1 | 1.314296 / 1 | 1.672755 / 1 |
| SH3 | nested | 1.140581 / 1 | 1.304889 / 1 | 1.672756 / 1 |
| SH2 | nested | 1.150379 / 1 | 1.302253 / 1 | 1.500905 / 1 |
| shared-expert-during-read | boundary | 3.446450 / 1 | 3.934361 / 1 | 4.856664 / 1 |
| detail:expert-gate | nested | 25.620950 / 16 | 24.279667 / 16 | 16.365209 / 16 |
| detail:expert-up | nested | 24.437409 / 16 | 24.291510 / 16 | 15.905420 / 16 |
| detail:expert-activation | nested | 0.098673 / 16 | 0.100218 / 16 | 0.096740 / 16 |
| detail:expert-down | nested | 25.229419 / 16 | 25.501170 / 16 | 17.319337 / 16 |
| EUP | nested | 0.699557 / 1 | 0.580354 / 1 | 0.955215 / 1 |
| experts-mix-normalize-up | boundary | 76.484429 / 1 | 75.124157 / 1 | 50.982569 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002635 / 1 | 0.002445 / 1 | 0.002214 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000161 / 1 | 0.000290 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.414555 / 1 | 0.404726 / 1 | 0.377836 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1037.359590 / 1 | 944.706814 / 1 | 687.206170 / 1 |
| op:Q   int8 projection | nested | 17.652393 / 1 | 19.119081 / 1 | 23.929986 / 1 |
| op:X   mxfp4 expert proj | nested | 75.404461 / 1 | 74.190462 / 1 | 49.703281 / 1 |
| op:N   rmsnorm | nested | 0.029994 / 1 | 0.033624 / 1 | 0.022672 / 1 |
| op:L   l2 per-head | nested | 0.005240 / 1 | 0.005160 / 1 | 0.005380 / 1 |
| op:SiTU + sigma | nested | 0.108151 / 1 | 0.110258 / 1 | 0.104395 / 1 |
| op:C   shortconv | nested | 0.059000 / 1 | 0.057397 / 1 | 0.036118 / 1 |
| op:AR  snapshot aggregate | nested | 0.032310 / 1 | 0.032771 / 1 | 0.031309 / 1 |
| op:D   kda delta-rule | nested | 0.144620 / 1 | 0.132498 / 1 | 0.218428 / 1 |
| op:router dot product | nested | 0.605842 / 1 | 0.601984 / 1 | 0.688086 / 1 |
| op:top-k selection | nested | 0.002364 / 1 | 0.003115 / 1 | 0.002224 / 1 |
| op:alpha / beta / gate | nested | 0.076012 / 1 | 0.073998 / 1 | 0.056926 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.746750 / 1 | 95.864230 / 1 | 75.841107 / 1 |

### Layer 6

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004959 / 1 | 0.004428 / 1 | 0.006873 / 1 |
| pre-attention-aggregation | boundary | 0.015749 / 1 | 0.015970 / 1 | 0.015910 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.011020 / 1 | 0.011873 / 1 |
| Q | nested | 2.816452 / 1 | 2.451831 / 1 | 5.226456 / 1 |
| K | nested | 2.700476 / 1 | 2.326156 / 1 | 3.265131 / 1 |
| V | nested | 2.631857 / 1 | 2.276173 / 1 | 2.611620 / 1 |
| B | nested | 0.023714 / 1 | 0.020588 / 1 | 0.027311 / 1 |
| FA | nested | 0.028082 / 1 | 0.028302 / 1 | 0.032721 / 1 |
| FB | nested | 0.061324 / 1 | 0.047749 / 1 | 0.056565 / 1 |
| G | nested | 2.226580 / 1 | 1.902575 / 1 | 2.964419 / 1 |
| O | nested | 2.364659 / 1 | 2.030604 / 1 | 2.963476 / 1 |
| attention | boundary | 13.978619 / 1 | 12.143299 / 1 | 17.827240 / 1 |
| attention-residual | boundary | 0.003437 / 1 | 0.003667 / 1 | 0.002675 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024486 / 1 | 0.023674 / 1 | 0.025218 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001934 / 1 | 0.002394 / 1 | 0.003096 / 1 |
| router-and-top16 | boundary | 0.587027 / 1 | 0.559416 / 1 | 0.939676 / 1 |
| EDOWN | nested | 0.645577 / 1 | 0.558243 / 1 | 0.893189 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.645997 / 1 | 0.558674 / 1 | 0.893630 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.087002 / 1 | 0.022232 / 1 | 0.012343 / 1 |
| SH1 | nested | 1.132195 / 1 | 0.964633 / 1 | 1.927471 / 1 |
| SH3 | nested | 1.109783 / 1 | 0.959233 / 1 | 2.335955 / 1 |
| SH2 | nested | 1.127858 / 1 | 0.957639 / 1 | 1.824779 / 1 |
| shared-expert-during-read | boundary | 3.381899 / 1 | 2.891633 / 1 | 6.098324 / 1 |
| detail:expert-gate | nested | 23.968335 / 16 | 24.530682 / 16 | 21.160408 / 16 |
| detail:expert-up | nested | 22.561185 / 16 | 22.796294 / 16 | 20.690622 / 16 |
| detail:expert-activation | nested | 0.097022 / 16 | 0.103987 / 16 | 0.096811 / 16 |
| detail:expert-down | nested | 23.073842 / 16 | 25.512451 / 16 | 21.431343 / 16 |
| EUP | nested | 0.596084 / 1 | 0.589712 / 1 | 1.227985 / 1 |
| experts-mix-normalize-up | boundary | 70.736259 / 1 | 73.898959 / 1 | 64.943034 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003487 / 1 | 0.003216 / 1 | 0.002414 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000231 / 1 | 0.000331 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.381022 / 1 | 0.384579 / 1 | 0.369110 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 951.918921 / 1 | 892.027185 / 1 | 872.593823 / 1 |
| op:Q   int8 projection | nested | 17.462649 / 1 | 15.111985 / 1 | 25.355164 / 1 |
| op:X   mxfp4 expert proj | nested | 69.719310 / 1 | 72.961982 / 1 | 63.398126 / 1 |
| op:N   rmsnorm | nested | 0.032582 / 1 | 0.034835 / 1 | 0.024265 / 1 |
| op:L   l2 per-head | nested | 0.007624 / 1 | 0.004849 / 1 | 0.005791 / 1 |
| op:SiTU + sigma | nested | 0.106508 / 1 | 0.111369 / 1 | 0.104066 / 1 |
| op:C   shortconv | nested | 0.054121 / 1 | 0.051477 / 1 | 0.030997 / 1 |
| op:AR  snapshot aggregate | nested | 0.030147 / 1 | 0.030006 / 1 | 0.030618 / 1 |
| op:D   kda delta-rule | nested | 0.136555 / 1 | 0.123240 / 1 | 0.106078 / 1 |
| op:router dot product | nested | 0.583671 / 1 | 0.556740 / 1 | 0.936981 / 1 |
| op:top-k selection | nested | 0.002575 / 1 | 0.002374 / 1 | 0.002294 / 1 |
| op:alpha / beta / gate | nested | 0.075931 / 1 | 0.074088 / 1 | 0.057147 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.866153 / 1 | 90.526316 / 1 | 91.154489 / 1 |

### Layer 7

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004489 / 1 | 0.004007 / 1 | 0.005801 / 1 |
| pre-attention-aggregation | boundary | 0.015028 / 1 | 0.015078 / 1 | 0.014998 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011752 / 1 | 0.011241 / 1 | 0.011341 / 1 |
| QA | nested | 0.302565 / 1 | 0.296804 / 1 | 0.411909 / 1 |
| QB | nested | 0.763867 / 1 | 0.684519 / 1 | 1.148235 / 1 |
| KA | nested | 0.106039 / 1 | 0.101209 / 1 | 0.216555 / 1 |
| KB | nested | 0.335036 / 1 | 0.296995 / 1 | 0.568683 / 1 |
| G | nested | 1.961836 / 1 | 1.909778 / 1 | 3.726232 / 1 |
| O | nested | 1.963949 / 1 | 1.907434 / 1 | 3.298082 / 1 |
| attention | boundary | 5.529803 / 1 | 5.293872 / 1 | 9.473812 / 1 |
| attention-residual | boundary | 0.002645 / 1 | 0.002575 / 1 | 0.001914 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024486 / 1 | 0.024055 / 1 | 0.026760 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002615 / 1 | 0.002224 / 1 | 0.002975 / 1 |
| router-and-top16 | boundary | 0.570395 / 1 | 0.560658 / 1 | 0.735053 / 1 |
| EDOWN | nested | 0.576367 / 1 | 0.575646 / 1 | 0.988807 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.576878 / 1 | 0.576086 / 1 | 0.989348 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078396 / 1 | 0.032200 / 1 | 0.012293 / 1 |
| SH1 | nested | 0.980151 / 1 | 0.958952 / 1 | 1.813388 / 1 |
| SH3 | nested | 1.002283 / 1 | 0.952790 / 1 | 1.574071 / 1 |
| SH2 | nested | 0.972978 / 1 | 0.957870 / 1 | 1.663017 / 1 |
| shared-expert-during-read | boundary | 2.965541 / 1 | 2.879681 / 1 | 5.064202 / 1 |
| detail:expert-gate | nested | 28.275572 / 16 | 23.573285 / 16 | 23.962414 / 16 |
| detail:expert-up | nested | 27.115978 / 16 | 23.247204 / 16 | 23.413429 / 16 |
| detail:expert-activation | nested | 0.105666 / 16 | 0.102848 / 16 | 0.117740 / 16 |
| detail:expert-down | nested | 27.220910 / 16 | 26.665976 / 16 | 22.000656 / 16 |
| EUP | nested | 0.598238 / 1 | 0.793794 / 1 | 0.951307 / 1 |
| experts-mix-normalize-up | boundary | 83.737832 / 1 | 74.775797 / 1 | 70.783697 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003927 / 1 | 0.003226 / 1 | 0.006302 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000561 / 1 | 0.000151 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005911 / 1 | 0.004979 / 1 | 0.008817 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1010.771986 / 1 | 919.718644 / 1 | 798.076380 / 1 |
| op:Q   int8 projection | nested | 9.562017 / 1 | 9.434637 / 1 | 16.358163 / 1 |
| op:X   mxfp4 expert proj | nested | 82.737443 / 1 | 73.609889 / 1 | 69.514964 / 1 |
| op:N   rmsnorm | nested | 0.024907 / 1 | 0.023293 / 1 | 0.023375 / 1 |
| op:SiTU + sigma | nested | 0.112382 / 1 | 0.110314 / 1 | 0.128891 / 1 |
| op:AR  snapshot aggregate | nested | 0.028734 / 1 | 0.029073 / 1 | 0.029545 / 1 |
| op:SA  softmax attention | nested | 0.006852 / 1 | 0.009788 / 1 | 0.012714 / 1 |
| op:router dot product | nested | 0.567761 / 1 | 0.558023 / 1 | 0.732318 / 1 |
| op:top-k selection | nested | 0.002224 / 1 | 0.002224 / 1 | 0.002415 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.533095 / 1 | 84.188643 / 1 | 87.140670 / 1 |

### Layer 8

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003557 / 1 | 0.003787 / 1 | 0.007113 / 1 |
| pre-attention-aggregation | boundary | 0.013736 / 1 | 0.013776 / 1 | 0.014818 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010790 / 1 | 0.010851 / 1 | 0.114825 / 1 |
| Q | nested | 2.519818 / 1 | 3.233772 / 1 | 4.774581 / 1 |
| K | nested | 2.392601 / 1 | 3.220577 / 1 | 3.733215 / 1 |
| V | nested | 2.318472 / 1 | 3.110743 / 1 | 2.890801 / 1 |
| B | nested | 0.021741 / 1 | 0.024776 / 1 | 0.032030 / 1 |
| FA | nested | 0.026430 / 1 | 0.034064 / 1 | 0.034304 / 1 |
| FB | nested | 0.064892 / 1 | 0.057006 / 1 | 0.062166 / 1 |
| G | nested | 2.276914 / 1 | 2.683454 / 1 | 3.050390 / 1 |
| O | nested | 2.446071 / 1 | 2.806714 / 1 | 3.351161 / 1 |
| attention | boundary | 13.092974 / 1 | 16.144125 / 1 | 18.650729 / 1 |
| attention-residual | boundary | 0.003467 / 1 | 0.003016 / 1 | 0.002565 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024085 / 1 | 0.023494 / 1 | 0.026400 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002014 / 1 | 0.002194 / 1 | 0.003066 / 1 |
| router-and-top16 | boundary | 0.634275 / 1 | 0.586707 / 1 | 0.839289 / 1 |
| EDOWN | nested | 0.691371 / 1 | 0.824120 / 1 | 0.956457 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.691862 / 1 | 0.824651 / 1 | 0.957119 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.080280 / 1 | 0.022402 / 1 | 0.007163 / 1 |
| SH1 | nested | 1.444629 / 1 | 1.425754 / 1 | 1.668217 / 1 |
| SH3 | nested | 1.168383 / 1 | 1.420745 / 1 | 1.706198 / 1 |
| SH2 | nested | 1.178412 / 1 | 1.423520 / 1 | 1.682184 / 1 |
| shared-expert-during-read | boundary | 3.804018 / 1 | 4.282762 / 1 | 5.069672 / 1 |
| detail:expert-gate | nested | 27.376182 / 16 | 26.223578 / 16 | 21.491757 / 16 |
| detail:expert-up | nested | 26.689518 / 16 | 24.757898 / 16 | 20.693668 / 16 |
| detail:expert-activation | nested | 0.111530 / 16 | 0.097586 / 16 | 0.108454 / 16 |
| detail:expert-down | nested | 26.114315 / 16 | 25.010141 / 16 | 20.986803 / 16 |
| EUP | nested | 0.697584 / 1 | 0.593119 / 1 | 0.950576 / 1 |
| experts-mix-normalize-up | boundary | 81.411024 / 1 | 77.033225 / 1 | 64.568704 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002826 / 1 | 0.002455 / 1 | 0.002534 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000180 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.399767 / 1 | 0.402292 / 1 | 0.375391 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1077.281162 / 1 | 1023.249455 / 1 | 815.273554 / 1 |
| op:Q   int8 projection | nested | 17.245826 / 1 | 20.856449 / 1 | 24.890497 / 1 |
| op:X   mxfp4 expert proj | nested | 80.313606 / 1 | 76.111393 / 1 | 63.301958 / 1 |
| op:N   rmsnorm | nested | 0.032141 / 1 | 0.031056 / 1 | 0.024366 / 1 |
| op:L   l2 per-head | nested | 0.005570 / 1 | 0.005621 / 1 | 0.005640 / 1 |
| op:SiTU + sigma | nested | 0.121559 / 1 | 0.107544 / 1 | 0.118911 / 1 |
| op:C   shortconv | nested | 0.052509 / 1 | 0.046898 / 1 | 0.037741 / 1 |
| op:AR  snapshot aggregate | nested | 0.027903 / 1 | 0.028061 / 1 | 0.030297 / 1 |
| op:D   kda delta-rule | nested | 0.127218 / 1 | 0.112290 / 1 | 0.148277 / 1 |
| op:router dot product | nested | 0.631921 / 1 | 0.583700 / 1 | 0.836453 / 1 |
| op:top-k selection | nested | 0.002034 / 1 | 0.002634 / 1 | 0.002344 / 1 |
| op:alpha / beta / gate | nested | 0.074089 / 1 | 0.074609 / 1 | 0.056836 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 100.178411 / 1 | 99.359570 / 1 | 90.643083 / 1 |

### Layer 9

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005180 / 1 | 0.005280 / 1 | 0.006412 / 1 |
| pre-attention-aggregation | boundary | 0.015970 / 1 | 0.016981 / 1 | 0.015128 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010660 / 1 | 0.010961 / 1 | 0.109735 / 1 |
| Q | nested | 2.802507 / 1 | 2.509308 / 1 | 4.858598 / 1 |
| K | nested | 2.536239 / 1 | 2.391498 / 1 | 3.134386 / 1 |
| V | nested | 2.373705 / 1 | 2.253010 / 1 | 2.586673 / 1 |
| B | nested | 0.022452 / 1 | 0.023514 / 1 | 0.026950 / 1 |
| FA | nested | 0.026189 / 1 | 0.024877 / 1 | 0.032381 / 1 |
| FB | nested | 0.049913 / 1 | 0.046898 / 1 | 0.055023 / 1 |
| G | nested | 2.043298 / 1 | 1.926059 / 1 | 2.931146 / 1 |
| O | nested | 2.138836 / 1 | 1.992963 / 1 | 3.866504 / 1 |
| attention | boundary | 13.022433 / 1 | 12.235322 / 1 | 18.133952 / 1 |
| attention-residual | boundary | 0.003497 / 1 | 0.002424 / 1 | 0.002044 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024576 / 1 | 0.024656 / 1 | 0.025748 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002204 / 1 | 0.002295 / 1 | 0.002906 / 1 |
| router-and-top16 | boundary | 0.572570 / 1 | 0.584232 / 1 | 0.720046 / 1 |
| EDOWN | nested | 0.611523 / 1 | 0.594511 / 1 | 0.935639 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.611974 / 1 | 0.595072 / 1 | 0.936630 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078958 / 1 | 0.022623 / 1 | 0.014698 / 1 |
| SH1 | nested | 1.005729 / 1 | 0.995029 / 1 | 1.633172 / 1 |
| SH3 | nested | 1.024304 / 1 | 0.991583 / 1 | 1.773554 / 1 |
| SH2 | nested | 1.021429 / 1 | 0.978478 / 1 | 1.623994 / 1 |
| shared-expert-during-read | boundary | 3.061621 / 1 | 2.975249 / 1 | 5.041039 / 1 |
| detail:expert-gate | nested | 23.205176 / 16 | 24.233799 / 16 | 14.362719 / 16 |
| detail:expert-up | nested | 22.030093 / 16 | 23.652746 / 16 | 13.992505 / 16 |
| detail:expert-activation | nested | 0.096330 / 16 | 0.107873 / 16 | 0.095768 / 16 |
| detail:expert-down | nested | 22.718917 / 16 | 23.908833 / 16 | 15.470615 / 16 |
| EUP | nested | 0.693175 / 1 | 0.583741 / 1 | 0.924477 / 1 |
| experts-mix-normalize-up | boundary | 69.166586 / 1 | 72.855438 / 1 | 45.173124 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002865 / 1 | 0.002334 / 1 | 0.002344 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000290 / 1 | 0.000150 / 1 | 0.000501 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.398725 / 1 | 0.425135 / 1 | 0.365683 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 935.170631 / 1 | 852.272060 / 1 | 619.993302 / 1 |
| op:Q   int8 projection | nested | 16.347768 / 1 | 15.309897 / 1 | 24.380644 / 1 |
| op:X   mxfp4 expert proj | nested | 68.072682 / 1 | 71.925380 / 1 | 43.942054 / 1 |
| op:N   rmsnorm | nested | 0.034886 / 1 | 0.034013 / 1 | 0.024326 / 1 |
| op:L   l2 per-head | nested | 0.005851 / 1 | 0.005480 / 1 | 0.005470 / 1 |
| op:SiTU + sigma | nested | 0.104004 / 1 | 0.115127 / 1 | 0.103323 / 1 |
| op:C   shortconv | nested | 0.062366 / 1 | 0.057366 / 1 | 0.035627 / 1 |
| op:AR  snapshot aggregate | nested | 0.030496 / 1 | 0.032050 / 1 | 0.030266 / 1 |
| op:D   kda delta-rule | nested | 0.118732 / 1 | 0.120796 / 1 | 0.220642 / 1 |
| op:router dot product | nested | 0.569554 / 1 | 0.580966 / 1 | 0.717290 / 1 |
| op:top-k selection | nested | 0.002635 / 1 | 0.002835 / 1 | 0.002294 / 1 |
| op:alpha / beta / gate | nested | 0.075831 / 1 | 0.076954 / 1 | 0.056615 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 86.981633 / 1 | 89.761658 / 1 | 70.553467 / 1 |

### Layer 10

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004618 / 1 | 0.005310 / 1 | 0.006212 / 1 |
| pre-attention-aggregation | boundary | 0.015990 / 1 | 0.016340 / 1 | 0.015128 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011482 / 1 | 0.011301 / 1 | 0.011842 / 1 |
| Q | nested | 2.761190 / 1 | 2.485394 / 1 | 5.336602 / 1 |
| K | nested | 2.669778 / 1 | 2.368245 / 1 | 3.347645 / 1 |
| V | nested | 2.586323 / 1 | 2.258280 / 1 | 2.640233 / 1 |
| B | nested | 0.025127 / 1 | 0.020839 / 1 | 0.133099 / 1 |
| FA | nested | 0.033002 / 1 | 0.028213 / 1 | 0.142105 / 1 |
| FB | nested | 0.060874 / 1 | 0.047719 / 1 | 0.054481 / 1 |
| G | nested | 2.252779 / 1 | 1.927942 / 1 | 3.215007 / 1 |
| O | nested | 2.338048 / 1 | 2.011879 / 1 | 3.138534 / 1 |
| attention | boundary | 13.785729 / 1 | 12.246782 / 1 | 18.642302 / 1 |
| attention-residual | boundary | 0.004489 / 1 | 0.002845 / 1 | 0.002825 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025207 / 1 | 0.025257 / 1 | 0.025869 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002204 / 1 | 0.002274 / 1 | 0.003196 / 1 |
| router-and-top16 | boundary | 0.602415 / 1 | 0.598168 / 1 | 0.669180 / 1 |
| EDOWN | nested | 0.662799 / 1 | 0.594931 / 1 | 1.122257 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.663310 / 1 | 0.595503 / 1 | 1.122838 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.075962 / 1 | 0.027932 / 1 | 0.006983 / 1 |
| SH1 | nested | 1.150559 / 1 | 0.990521 / 1 | 1.611100 / 1 |
| SH3 | nested | 1.167661 / 1 | 0.986303 / 1 | 1.588388 / 1 |
| SH2 | nested | 1.179975 / 1 | 0.972798 / 1 | 1.545928 / 1 |
| shared-expert-during-read | boundary | 3.511180 / 1 | 2.960542 / 1 | 4.755976 / 1 |
| detail:expert-gate | nested | 23.883147 / 16 | 22.841340 / 16 | 17.675084 / 16 |
| detail:expert-up | nested | 22.894216 / 16 | 21.827593 / 16 | 17.144863 / 16 |
| detail:expert-activation | nested | 0.099820 / 16 | 0.102613 / 16 | 0.109781 / 16 |
| detail:expert-down | nested | 23.541405 / 16 | 23.612668 / 16 | 17.197234 / 16 |
| EUP | nested | 0.593419 / 1 | 0.790407 / 1 | 1.020947 / 1 |
| experts-mix-normalize-up | boundary | 71.446896 / 1 | 69.548619 / 1 | 53.486978 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002705 / 1 | 0.002524 / 1 | 0.002575 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000331 / 1 | 0.000421 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.400549 / 1 | 0.432578 / 1 | 0.371194 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 957.910285 / 1 | 876.020029 / 1 | 695.650955 / 1 |
| op:Q   int8 projection | nested | 17.479910 / 1 | 15.481979 / 1 | 24.894545 / 1 |
| op:X   mxfp4 expert proj | nested | 70.442310 / 1 | 68.408700 / 1 | 52.149421 / 1 |
| op:N   rmsnorm | nested | 0.032301 / 1 | 0.032772 / 1 | 0.023313 / 1 |
| op:L   l2 per-head | nested | 0.005020 / 1 | 0.005540 / 1 | 0.005751 / 1 |
| op:SiTU + sigma | nested | 0.109819 / 1 | 0.110878 / 1 | 0.117497 / 1 |
| op:C   shortconv | nested | 0.052258 / 1 | 0.057978 / 1 | 0.035917 / 1 |
| op:AR  snapshot aggregate | nested | 0.031168 / 1 | 0.030918 / 1 | 0.030187 / 1 |
| op:D   kda delta-rule | nested | 0.118581 / 1 | 0.128891 / 1 | 0.217046 / 1 |
| op:router dot product | nested | 0.599721 / 1 | 0.594952 / 1 | 0.666475 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002545 / 1 | 0.002305 / 1 |
| op:alpha / beta / gate | nested | 0.076012 / 1 | 0.071494 / 1 | 0.057357 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 90.557274 / 1 | 86.479925 / 1 | 79.127037 / 1 |

### Layer 11

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005220 / 1 | 0.005450 / 1 | 0.010609 / 1 |
| pre-attention-aggregation | boundary | 0.015899 / 1 | 0.016781 / 1 | 0.015699 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010520 / 1 | 0.011070 / 1 | 0.010961 / 1 |
| QA | nested | 0.295813 / 1 | 0.362378 / 1 | 0.591215 / 1 |
| QB | nested | 0.727279 / 1 | 0.885665 / 1 | 1.337610 / 1 |
| KA | nested | 0.110898 / 1 | 0.119964 / 1 | 0.226583 / 1 |
| KB | nested | 0.324767 / 1 | 0.320429 / 1 | 0.473765 / 1 |
| G | nested | 1.987854 / 1 | 2.652636 / 1 | 3.607250 / 1 |
| O | nested | 1.957859 / 1 | 2.701869 / 1 | 3.247337 / 1 |
| attention | boundary | 5.507251 / 1 | 7.138929 / 1 | 9.583156 / 1 |
| attention-residual | boundary | 0.002264 / 1 | 0.002415 / 1 | 0.002355 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024045 / 1 | 0.026239 / 1 | 0.032761 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002445 / 1 | 0.002304 / 1 | 0.002615 / 1 |
| router-and-top16 | boundary | 0.583150 / 1 | 0.581937 / 1 | 0.774768 / 1 |
| EDOWN | nested | 0.588309 / 1 | 0.755862 / 1 | 1.001973 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.588871 / 1 | 0.756564 / 1 | 1.002544 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.064841 / 1 | 0.017794 / 1 | 0.006212 / 1 |
| SH1 | nested | 0.998746 / 1 | 1.283538 / 1 | 1.658108 / 1 |
| SH3 | nested | 1.003014 / 1 | 1.324024 / 1 | 1.551399 / 1 |
| SH2 | nested | 0.978598 / 1 | 1.306531 / 1 | 1.807487 / 1 |
| shared-expert-during-read | boundary | 2.990878 / 1 | 3.927058 / 1 | 5.030800 / 1 |
| detail:expert-gate | nested | 28.050291 / 16 | 21.946005 / 16 | 16.199438 / 16 |
| detail:expert-up | nested | 24.089287 / 16 | 21.805597 / 16 | 16.134347 / 16 |
| detail:expert-activation | nested | 0.105086 / 16 | 0.102131 / 16 | 0.117698 / 16 |
| detail:expert-down | nested | 24.388289 / 16 | 22.364780 / 16 | 14.597346 / 16 |
| EUP | nested | 0.607205 / 1 | 0.597286 / 1 | 1.016659 / 1 |
| experts-mix-normalize-up | boundary | 77.639537 / 1 | 67.194742 / 1 | 48.422515 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002354 / 1 | 0.002304 / 1 | 0.002775 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000131 / 1 | 0.000231 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007304 / 1 | 0.006923 / 1 | 0.007894 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 992.510894 / 1 | 800.778611 / 1 | 558.353802 / 1 |
| op:Q   int8 projection | nested | 9.579119 / 1 | 12.308920 / 1 | 16.517903 / 1 |
| op:X   mxfp4 expert proj | nested | 76.656992 / 1 | 66.241649 / 1 | 47.071359 / 1 |
| op:N   rmsnorm | nested | 0.023342 / 1 | 0.023833 / 1 | 0.024213 / 1 |
| op:SiTU + sigma | nested | 0.112901 / 1 | 0.111961 / 1 | 0.128621 / 1 |
| op:AR  snapshot aggregate | nested | 0.029445 / 1 | 0.032050 / 1 | 0.037371 / 1 |
| op:SA  softmax attention | nested | 0.006412 / 1 | 0.009027 / 1 | 0.011852 / 1 |
| op:router dot product | nested | 0.580524 / 1 | 0.579001 / 1 | 0.772173 / 1 |
| op:top-k selection | nested | 0.002194 / 1 | 0.002404 / 1 | 0.002275 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 87.448616 / 1 | 79.694257 / 1 | 64.909922 / 1 |

### Layer 12

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004108 / 1 | 0.004157 / 1 | 0.005871 / 1 |
| pre-attention-aggregation | boundary | 0.014267 / 1 | 0.014477 / 1 | 0.014908 / 1 |
| snapshot-push | boundary | 0.001393 / 1 | 0.001243 / 1 | 0.000992 / 1 |
| pre-attention-normalization | boundary | 0.010199 / 1 | 0.009969 / 1 | 0.011101 / 1 |
| Q | nested | 2.529596 / 1 | 2.509058 / 1 | 4.411273 / 1 |
| K | nested | 2.371802 / 1 | 2.345673 / 1 | 3.024421 / 1 |
| V | nested | 2.288315 / 1 | 2.210971 / 1 | 2.533203 / 1 |
| B | nested | 0.022402 / 1 | 0.022572 / 1 | 0.026690 / 1 |
| FA | nested | 0.028093 / 1 | 0.026559 / 1 | 0.032310 / 1 |
| FB | nested | 0.061144 / 1 | 0.047228 / 1 | 0.145792 / 1 |
| G | nested | 1.990049 / 1 | 1.919526 / 1 | 3.055178 / 1 |
| O | nested | 2.057033 / 1 | 2.026487 / 1 | 3.215789 / 1 |
| attention | boundary | 12.440204 / 1 | 12.200276 / 1 | 17.201751 / 1 |
| attention-residual | boundary | 0.002405 / 1 | 0.002024 / 1 | 0.001773 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024706 / 1 | 0.025207 / 1 | 0.026880 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002836 / 1 | 0.002535 / 1 | 0.002875 / 1 |
| router-and-top16 | boundary | 0.590644 / 1 | 0.594171 / 1 | 0.711900 / 1 |
| EDOWN | nested | 0.608897 / 1 | 0.591836 / 1 | 1.190184 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609468 / 1 | 0.592417 / 1 | 1.190926 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085199 / 1 | 0.026830 / 1 | 0.019346 / 1 |
| SH1 | nested | 1.043500 / 1 | 1.008896 / 1 | 1.820972 / 1 |
| SH3 | nested | 1.059539 / 1 | 0.992595 / 1 | 1.576426 / 1 |
| SH2 | nested | 1.001341 / 1 | 0.983617 / 1 | 1.739129 / 1 |
| shared-expert-during-read | boundary | 3.114960 / 1 | 2.995568 / 1 | 5.147047 / 1 |
| detail:expert-gate | nested | 25.516437 / 16 | 24.718644 / 16 | 17.678773 / 16 |
| detail:expert-up | nested | 24.215657 / 16 | 24.358551 / 16 | 17.012520 / 16 |
| detail:expert-activation | nested | 0.100439 / 16 | 0.105828 / 16 | 0.099277 / 16 |
| detail:expert-down | nested | 26.292860 / 16 | 23.367563 / 16 | 17.317003 / 16 |
| EUP | nested | 0.784115 / 1 | 0.606554 / 1 | 1.286574 / 1 |
| experts-mix-normalize-up | boundary | 77.325390 / 1 | 73.549265 / 1 | 53.751502 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002174 / 1 | 0.002715 / 1 | 0.002495 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000290 / 1 | 0.000150 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427429 / 1 | 0.445803 / 1 | 0.369110 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 948.271462 / 1 | 888.053738 / 1 | 694.126897 / 1 |
| op:Q   int8 projection | nested | 15.844414 / 1 | 15.290069 / 1 | 24.056339 / 1 |
| op:X   mxfp4 expert proj | nested | 76.151809 / 1 | 72.575634 / 1 | 52.132257 / 1 |
| op:N   rmsnorm | nested | 0.032722 / 1 | 0.033113 / 1 | 0.022743 / 1 |
| op:L   l2 per-head | nested | 0.005159 / 1 | 0.004959 / 1 | 0.005871 / 1 |
| op:SiTU + sigma | nested | 0.107962 / 1 | 0.113123 / 1 | 0.106718 / 1 |
| op:C   shortconv | nested | 0.053538 / 1 | 0.056957 / 1 | 0.042409 / 1 |
| op:AR  snapshot aggregate | nested | 0.029626 / 1 | 0.030067 / 1 | 0.030417 / 1 |
| op:D   kda delta-rule | nested | 0.141664 / 1 | 0.141935 / 1 | 0.142877 / 1 |
| op:router dot product | nested | 0.587417 / 1 | 0.591535 / 1 | 0.709065 / 1 |
| op:top-k selection | nested | 0.002655 / 1 | 0.002344 / 1 | 0.002424 / 1 |
| op:alpha / beta / gate | nested | 0.072957 / 1 | 0.074660 / 1 | 0.057097 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.662775 / 1 | 90.470912 / 1 | 78.462815 / 1 |

### Layer 13

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005120 / 1 | 0.004929 / 1 | 0.093515 / 1 |
| pre-attention-aggregation | boundary | 0.019447 / 1 | 0.019667 / 1 | 0.016220 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.013305 / 1 | 0.011692 / 1 | 0.010981 / 1 |
| Q | nested | 3.082620 / 1 | 2.514248 / 1 | 4.374925 / 1 |
| K | nested | 2.869201 / 1 | 2.388452 / 1 | 3.070026 / 1 |
| V | nested | 2.273939 / 1 | 2.234465 / 1 | 2.561556 / 1 |
| B | nested | 0.023163 / 1 | 0.025838 / 1 | 0.027020 / 1 |
| FA | nested | 0.033021 / 1 | 0.025087 / 1 | 0.131325 / 1 |
| FB | nested | 0.062346 / 1 | 0.050845 / 1 | 0.126656 / 1 |
| G | nested | 1.983276 / 1 | 1.915329 / 1 | 3.252457 / 1 |
| O | nested | 2.027589 / 1 | 2.024463 / 1 | 3.385286 / 1 |
| attention | boundary | 13.428962 / 1 | 12.258435 / 1 | 17.654347 / 1 |
| attention-residual | boundary | 0.002424 / 1 | 0.002835 / 1 | 0.002445 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024956 / 1 | 0.024897 / 1 | 0.026209 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002394 / 1 | 0.002725 / 1 | 0.003266 / 1 |
| router-and-top16 | boundary | 0.588940 / 1 | 0.591876 / 1 | 0.825372 / 1 |
| EDOWN | nested | 0.618576 / 1 | 0.579262 / 1 | 0.914780 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.619347 / 1 | 0.579893 / 1 | 0.915921 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085299 / 1 | 0.023343 / 1 | 0.006191 / 1 |
| SH1 | nested | 1.013704 / 1 | 0.990420 / 1 | 1.723591 / 1 |
| SH3 | nested | 1.012522 / 1 | 0.993306 / 1 | 1.847542 / 1 |
| SH2 | nested | 1.018002 / 1 | 0.991944 / 1 | 1.677976 / 1 |
| shared-expert-during-read | boundary | 3.055238 / 1 | 2.986500 / 1 | 5.261952 / 1 |
| detail:expert-gate | nested | 25.769830 / 16 | 23.842178 / 16 | 20.455380 / 16 |
| detail:expert-up | nested | 24.238349 / 16 | 22.688399 / 16 | 19.978760 / 16 |
| detail:expert-activation | nested | 0.099511 / 16 | 0.098872 / 16 | 0.110716 / 16 |
| detail:expert-down | nested | 25.524734 / 16 | 24.982567 / 16 | 18.689250 / 16 |
| EUP | nested | 0.712612 / 1 | 0.797971 / 1 | 0.969581 / 1 |
| experts-mix-normalize-up | boundary | 76.792254 / 1 | 72.772233 / 1 | 60.551839 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002524 / 1 | 0.002284 / 1 | 0.002675 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000130 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.423762 / 1 | 0.420455 / 1 | 0.363339 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1023.758704 / 1 | 952.040034 / 1 | 741.041881 / 1 |
| op:Q   int8 projection | nested | 16.729158 / 1 | 15.530135 / 1 | 24.060516 / 1 |
| op:X   mxfp4 expert proj | nested | 75.659698 / 1 | 71.638764 / 1 | 59.261079 / 1 |
| op:N   rmsnorm | nested | 0.040886 / 1 | 0.034045 / 1 | 0.023945 / 1 |
| op:L   l2 per-head | nested | 0.005370 / 1 | 0.005440 / 1 | 0.005811 / 1 |
| op:SiTU + sigma | nested | 0.107062 / 1 | 0.106047 / 1 | 0.120615 / 1 |
| op:C   shortconv | nested | 0.055092 / 1 | 0.057378 / 1 | 0.035927 / 1 |
| op:AR  snapshot aggregate | nested | 0.034575 / 1 | 0.034595 / 1 | 0.030958 / 1 |
| op:D   kda delta-rule | nested | 0.135723 / 1 | 0.136164 / 1 | 0.215984 / 1 |
| op:router dot product | nested | 0.586205 / 1 | 0.588921 / 1 | 0.822366 / 1 |
| op:top-k selection | nested | 0.002495 / 1 | 0.002575 / 1 | 0.002475 / 1 |
| op:alpha / beta / gate | nested | 0.081553 / 1 | 0.082605 / 1 | 0.056796 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.068613 / 1 | 89.706213 / 1 | 85.739391 / 1 |

### Layer 14

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005310 / 1 | 0.005400 / 1 | 0.006112 / 1 |
| pre-attention-aggregation | boundary | 0.019717 / 1 | 0.020829 / 1 | 0.016361 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.016971 / 1 | 0.011882 / 1 | 0.011171 / 1 |
| Q | nested | 2.804961 / 1 | 3.097067 / 1 | 4.788948 / 1 |
| K | nested | 2.649861 / 1 | 3.018670 / 1 | 3.179450 / 1 |
| V | nested | 2.586563 / 1 | 3.047905 / 1 | 2.646655 / 1 |
| B | nested | 0.025618 / 1 | 0.027101 / 1 | 0.100047 / 1 |
| FA | nested | 0.028062 / 1 | 0.036278 / 1 | 0.131565 / 1 |
| FB | nested | 0.063138 / 1 | 0.057217 / 1 | 0.172512 / 1 |
| G | nested | 2.286232 / 1 | 2.731604 / 1 | 3.230115 / 1 |
| O | nested | 2.227933 / 1 | 2.675759 / 1 | 3.467239 / 1 |
| attention | boundary | 13.779276 / 1 | 15.747494 / 1 | 18.348233 / 1 |
| attention-residual | boundary | 0.002334 / 1 | 0.002685 / 1 | 0.001683 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024876 / 1 | 0.026730 / 1 | 0.026109 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002244 / 1 | 0.002485 / 1 | 0.003086 / 1 |
| router-and-top16 | boundary | 0.624136 / 1 | 0.599029 / 1 | 0.708694 / 1 |
| EDOWN | nested | 0.676995 / 1 | 0.798773 / 1 | 0.946088 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.677546 / 1 | 0.799654 / 1 | 0.946789 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.081382 / 1 | 0.034544 / 1 | 0.006802 / 1 |
| SH1 | nested | 1.140270 / 1 | 1.369248 / 1 | 1.696760 / 1 |
| SH3 | nested | 1.145791 / 1 | 1.304718 / 1 | 1.706168 / 1 |
| SH2 | nested | 1.081271 / 1 | 1.287806 / 1 | 1.587196 / 1 |
| shared-expert-during-read | boundary | 3.379845 / 1 | 3.974867 / 1 | 5.001365 / 1 |
| detail:expert-gate | nested | 25.598642 / 16 | 24.845724 / 16 | 22.190176 / 16 |
| detail:expert-up | nested | 24.199307 / 16 | 23.767286 / 16 | 20.944918 / 16 |
| detail:expert-activation | nested | 0.098794 / 16 | 0.101427 / 16 | 0.093315 / 16 |
| detail:expert-down | nested | 24.983793 / 16 | 23.720071 / 16 | 22.552958 / 16 |
| EUP | nested | 0.689839 / 1 | 0.758438 / 1 | 1.009646 / 1 |
| experts-mix-normalize-up | boundary | 76.007247 / 1 | 73.577848 / 1 | 67.136171 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002875 / 1 | 0.002665 / 1 | 0.002394 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000330 / 1 | 0.000221 / 1 | 0.000330 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.426867 / 1 | 0.437017 / 1 | 0.372376 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1025.449254 / 1 | 991.756061 / 1 | 915.991988 / 1 |
| op:Q   int8 projection | nested | 17.405069 / 1 | 20.208851 / 1 | 24.660356 / 1 |
| op:X   mxfp4 expert proj | nested | 74.908997 / 1 | 72.462105 / 1 | 65.809734 / 1 |
| op:N   rmsnorm | nested | 0.039553 / 1 | 0.034543 / 1 | 0.024236 / 1 |
| op:L   l2 per-head | nested | 0.005640 / 1 | 0.005330 / 1 | 0.005970 / 1 |
| op:SiTU + sigma | nested | 0.108031 / 1 | 0.111257 / 1 | 0.101410 / 1 |
| op:C   shortconv | nested | 0.079809 / 1 | 0.058450 / 1 | 0.036168 / 1 |
| op:AR  snapshot aggregate | nested | 0.034976 / 1 | 0.037309 / 1 | 0.031109 / 1 |
| op:D   kda delta-rule | nested | 0.138860 / 1 | 0.130453 / 1 | 0.214210 / 1 |
| op:router dot product | nested | 0.621120 / 1 | 0.596024 / 1 | 0.706009 / 1 |
| op:top-k selection | nested | 0.002535 / 1 | 0.002635 / 1 | 0.002074 / 1 |
| op:alpha / beta / gate | nested | 0.075712 / 1 | 0.072455 / 1 | 0.057437 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.055820 / 1 | 95.248289 / 1 | 92.592356 / 1 |

### Layer 15

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005720 / 1 | 0.005390 / 1 | 0.006091 / 1 |
| pre-attention-aggregation | boundary | 0.019807 / 1 | 0.021029 / 1 | 0.016040 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010870 / 1 | 0.010290 / 1 | 0.011231 / 1 |
| QA | nested | 0.298187 / 1 | 0.344143 / 1 | 0.664682 / 1 |
| QB | nested | 0.771702 / 1 | 0.797540 / 1 | 1.006601 / 1 |
| KA | nested | 0.116287 / 1 | 0.132317 / 1 | 0.222576 / 1 |
| KB | nested | 0.344824 / 1 | 0.333763 / 1 | 0.557171 / 1 |
| G | nested | 2.261585 / 1 | 2.643219 / 1 | 3.392499 / 1 |
| O | nested | 2.348959 / 1 | 2.737635 / 1 | 3.490552 / 1 |
| attention | boundary | 6.238997 / 1 | 7.085158 / 1 | 9.431422 / 1 |
| attention-residual | boundary | 0.002374 / 1 | 0.002294 / 1 | 0.001894 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025478 / 1 | 0.025788 / 1 | 0.026650 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002214 / 1 | 0.002154 / 1 | 0.002865 / 1 |
| router-and-top16 | boundary | 0.608657 / 1 | 0.595954 / 1 | 0.798662 / 1 |
| EDOWN | nested | 0.684529 / 1 | 0.793182 / 1 | 1.023973 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.685140 / 1 | 0.793843 / 1 | 1.024594 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085650 / 1 | 0.017553 / 1 | 0.016591 / 1 |
| SH1 | nested | 1.155309 / 1 | 1.329644 / 1 | 1.764257 / 1 |
| SH3 | nested | 1.139689 / 1 | 1.364800 / 1 | 1.799923 / 1 |
| SH2 | nested | 1.146613 / 1 | 1.360583 / 1 | 1.782801 / 1 |
| shared-expert-during-read | boundary | 3.455006 / 1 | 4.068341 / 1 | 5.361999 / 1 |
| detail:expert-gate | nested | 26.519322 / 16 | 24.452370 / 16 | 20.657949 / 16 |
| detail:expert-up | nested | 25.373891 / 16 | 22.941024 / 16 | 19.957408 / 16 |
| detail:expert-activation | nested | 0.101056 / 16 | 0.103343 / 16 | 0.110258 / 16 |
| detail:expert-down | nested | 25.780921 / 16 | 25.730467 / 16 | 19.309919 / 16 |
| EUP | nested | 0.593209 / 1 | 0.593539 / 1 | 1.220631 / 1 |
| experts-mix-normalize-up | boundary | 78.798873 / 1 | 74.184191 / 1 | 61.614444 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002615 / 1 | 0.002184 / 1 | 0.002374 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000160 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007474 / 1 | 0.007284 / 1 | 0.008165 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1056.449047 / 1 | 943.643681 / 1 | 763.197904 / 1 |
| op:Q   int8 projection | nested | 10.859321 / 1 | 12.429092 / 1 | 16.924191 / 1 |
| op:X   mxfp4 expert proj | nested | 77.805018 / 1 | 73.255807 / 1 | 60.063407 / 1 |
| op:N   rmsnorm | nested | 0.024485 / 1 | 0.023272 / 1 | 0.023135 / 1 |
| op:SiTU + sigma | nested | 0.110894 / 1 | 0.113451 / 1 | 0.122358 / 1 |
| op:AR  snapshot aggregate | nested | 0.034856 / 1 | 0.036477 / 1 | 0.031530 / 1 |
| op:SA  softmax attention | nested | 0.006573 / 1 | 0.010009 / 1 | 0.011281 / 1 |
| op:router dot product | nested | 0.606123 / 1 | 0.593118 / 1 | 0.795456 / 1 |
| op:top-k selection | nested | 0.002084 / 1 | 0.002404 / 1 | 0.002726 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.953796 / 1 | 86.826272 / 1 | 78.328003 / 1 |

### Layer 16

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004499 / 1 | 0.004248 / 1 | 0.006532 / 1 |
| pre-attention-aggregation | boundary | 0.024586 / 1 | 0.015088 / 1 | 0.015820 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010720 / 1 | 0.009778 / 1 | 0.011111 / 1 |
| Q | nested | 2.517483 / 1 | 2.490343 / 1 | 4.501592 / 1 |
| K | nested | 2.389204 / 1 | 2.347637 / 1 | 3.248890 / 1 |
| V | nested | 2.254372 / 1 | 2.220329 / 1 | 2.482899 / 1 |
| B | nested | 0.024335 / 1 | 0.021660 / 1 | 0.026660 / 1 |
| FA | nested | 0.025598 / 1 | 0.026199 / 1 | 0.121276 / 1 |
| FB | nested | 0.064009 / 1 | 0.049292 / 1 | 0.055834 / 1 |
| G | nested | 2.021066 / 1 | 1.912433 / 1 | 3.241456 / 1 |
| O | nested | 2.026306 / 1 | 2.041173 / 1 | 3.171636 / 1 |
| attention | boundary | 12.438691 / 1 | 12.201618 / 1 | 17.551464 / 1 |
| attention-residual | boundary | 0.002384 / 1 | 0.002434 / 1 | 0.002194 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024957 / 1 | 0.025227 / 1 | 0.026259 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002695 / 1 | 0.002685 / 1 | 0.003577 / 1 |
| router-and-top16 | boundary | 0.608637 / 1 | 0.584633 / 1 | 0.692133 / 1 |
| EDOWN | nested | 0.604700 / 1 | 0.581597 / 1 | 1.038310 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.605331 / 1 | 0.582278 / 1 | 1.038911 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059882 / 1 | 0.017623 / 1 | 0.012323 / 1 |
| SH1 | nested | 1.025546 / 1 | 0.984991 / 1 | 1.692753 / 1 |
| SH3 | nested | 1.016880 / 1 | 0.987645 / 1 | 1.552942 / 1 |
| SH2 | nested | 1.003816 / 1 | 0.982155 / 1 | 1.379498 / 1 |
| shared-expert-during-read | boundary | 3.057052 / 1 | 2.965662 / 1 | 4.635972 / 1 |
| detail:expert-gate | nested | 23.931687 / 16 | 23.943175 / 16 | 17.125270 / 16 |
| detail:expert-up | nested | 23.011717 / 16 | 20.782874 / 16 | 16.504257 / 16 |
| detail:expert-activation | nested | 0.105607 / 16 | 0.102881 / 16 | 0.097142 / 16 |
| detail:expert-down | nested | 22.524345 / 16 | 22.637440 / 16 | 17.741469 / 16 |
| EUP | nested | 0.603868 / 1 | 0.741084 / 1 | 0.985181 / 1 |
| experts-mix-normalize-up | boundary | 70.606987 / 1 | 68.578217 / 1 | 52.794925 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002594 / 1 | 0.003216 / 1 | 0.002695 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000141 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.432077 / 1 | 0.442957 / 1 | 0.367617 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 905.204878 / 1 | 863.332497 / 1 | 724.480986 / 1 |
| op:Q   int8 projection | nested | 15.575732 / 1 | 15.384796 / 1 | 23.496752 / 1 |
| op:X   mxfp4 expert proj | nested | 69.604204 / 1 | 67.495605 / 1 | 51.496891 / 1 |
| op:N   rmsnorm | nested | 0.032580 / 1 | 0.036769 / 1 | 0.023563 / 1 |
| op:L   l2 per-head | nested | 0.005370 / 1 | 0.005360 / 1 | 0.005269 / 1 |
| op:SiTU + sigma | nested | 0.113171 / 1 | 0.110623 / 1 | 0.104887 / 1 |
| op:C   shortconv | nested | 0.059712 / 1 | 0.053540 / 1 | 0.036329 / 1 |
| op:AR  snapshot aggregate | nested | 0.039694 / 1 | 0.030177 / 1 | 0.030978 / 1 |
| op:D   kda delta-rule | nested | 0.144620 / 1 | 0.136735 / 1 | 0.144790 / 1 |
| op:router dot product | nested | 0.605622 / 1 | 0.581858 / 1 | 0.689238 / 1 |
| op:top-k selection | nested | 0.002525 / 1 | 0.002354 / 1 | 0.002635 / 1 |
| op:alpha / beta / gate | nested | 0.074469 / 1 | 0.073257 / 1 | 0.057468 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 87.886765 / 1 | 85.441105 / 1 | 77.166974 / 1 |

### Layer 17

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005511 / 1 | 0.005770 / 1 | 0.006452 / 1 |
| pre-attention-aggregation | boundary | 0.020739 / 1 | 0.021571 / 1 | 0.016350 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012373 / 1 | 0.012013 / 1 | 0.011863 / 1 |
| Q | nested | 2.531460 / 1 | 3.029060 / 1 | 4.273314 / 1 |
| K | nested | 2.374967 / 1 | 2.966563 / 1 | 3.063985 / 1 |
| V | nested | 2.240026 / 1 | 2.938179 / 1 | 2.571766 / 1 |
| B | nested | 0.020428 / 1 | 0.024936 / 1 | 0.027492 / 1 |
| FA | nested | 0.026890 / 1 | 0.032321 / 1 | 0.100217 / 1 |
| FB | nested | 0.047208 / 1 | 0.058188 / 1 | 0.150711 / 1 |
| G | nested | 1.962467 / 1 | 1.962788 / 1 | 3.360679 / 1 |
| O | nested | 2.032067 / 1 | 2.031736 / 1 | 2.888217 / 1 |
| attention | boundary | 12.386143 / 1 | 14.167612 / 1 | 17.069564 / 1 |
| attention-residual | boundary | 0.002494 / 1 | 0.002615 / 1 | 0.002184 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025508 / 1 | 0.025417 / 1 | 0.025778 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002174 / 1 | 0.002084 / 1 | 0.003296 / 1 |
| router-and-top16 | boundary | 0.604149 / 1 | 0.590173 / 1 | 0.651999 / 1 |
| EDOWN | nested | 0.600582 / 1 | 0.583480 / 1 | 0.917985 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.601234 / 1 | 0.584151 / 1 | 0.918617 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.070813 / 1 | 0.049162 / 1 | 0.006142 / 1 |
| SH1 | nested | 1.002213 / 1 | 0.989159 / 1 | 1.726526 / 1 |
| SH3 | nested | 1.013935 / 1 | 1.010418 / 1 | 2.223615 / 1 |
| SH2 | nested | 0.983898 / 1 | 0.981204 / 1 | 2.072702 / 1 |
| shared-expert-during-read | boundary | 3.010666 / 1 | 2.992211 / 1 | 6.036449 / 1 |
| detail:expert-gate | nested | 26.025109 / 16 | 23.581592 / 16 | 18.651340 / 16 |
| detail:expert-up | nested | 25.307897 / 16 | 23.007120 / 16 | 18.234832 / 16 |
| detail:expert-activation | nested | 0.106337 / 16 | 0.098924 / 16 | 0.115843 / 16 |
| detail:expert-down | nested | 24.610093 / 16 | 23.135611 / 16 | 16.673503 / 16 |
| EUP | nested | 0.584011 / 1 | 0.765631 / 1 | 1.045554 / 1 |
| experts-mix-normalize-up | boundary | 77.063381 / 1 | 71.016031 / 1 | 55.072131 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002274 / 1 | 0.002986 / 1 | 0.002555 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000361 / 1 | 0.000151 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.435193 / 1 | 0.431977 / 1 | 0.362758 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1023.012799 / 1 | 941.737559 / 1 | 641.314830 / 1 |
| op:Q   int8 projection | nested | 15.418690 / 1 | 17.372198 / 1 | 24.420677 / 1 |
| op:X   mxfp4 expert proj | nested | 76.085232 / 1 | 69.855891 / 1 | 53.705507 / 1 |
| op:N   rmsnorm | nested | 0.039724 / 1 | 0.032311 / 1 | 0.023655 / 1 |
| op:L   l2 per-head | nested | 0.006743 / 1 | 0.005230 / 1 | 0.005831 / 1 |
| op:SiTU + sigma | nested | 0.113500 / 1 | 0.107000 / 1 | 0.126344 / 1 |
| op:C   shortconv | nested | 0.062878 / 1 | 0.056795 / 1 | 0.035587 / 1 |
| op:AR  snapshot aggregate | nested | 0.036027 / 1 | 0.037609 / 1 | 0.031580 / 1 |
| op:D   kda delta-rule | nested | 0.138239 / 1 | 0.132508 / 1 | 0.218558 / 1 |
| op:router dot product | nested | 0.601434 / 1 | 0.587348 / 1 | 0.649513 / 1 |
| op:top-k selection | nested | 0.002074 / 1 | 0.002284 / 1 | 0.002154 / 1 |
| op:alpha / beta / gate | nested | 0.079228 / 1 | 0.078827 / 1 | 0.056796 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.248311 / 1 | 89.910164 / 1 | 80.191766 / 1 |

### Layer 18

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005460 / 1 | 0.005581 / 1 | 0.006322 / 1 |
| pre-attention-aggregation | boundary | 0.020067 / 1 | 0.022322 / 1 | 0.017022 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010720 / 1 | 0.011521 / 1 | 0.011050 / 1 |
| Q | nested | 2.540888 / 1 | 3.030973 / 1 | 4.750336 / 1 |
| K | nested | 2.390897 / 1 | 3.023058 / 1 | 3.478960 / 1 |
| V | nested | 2.270292 / 1 | 2.929333 / 1 | 3.016186 / 1 |
| B | nested | 0.021461 / 1 | 0.028794 / 1 | 0.029175 / 1 |
| FA | nested | 0.027001 / 1 | 0.034264 / 1 | 0.038993 / 1 |
| FB | nested | 0.058038 / 1 | 0.058209 / 1 | 0.151503 / 1 |
| G | nested | 1.960573 / 1 | 2.683173 / 1 | 3.458913 / 1 |
| O | nested | 2.060349 / 1 | 2.653678 / 1 | 3.279948 / 1 |
| attention | boundary | 12.486371 / 1 | 15.636727 / 1 | 18.869467 / 1 |
| attention-residual | boundary | 0.002204 / 1 | 0.003036 / 1 | 0.002234 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025819 / 1 | 0.027682 / 1 | 0.028513 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002525 / 1 | 0.002725 / 1 | 0.003186 / 1 |
| router-and-top16 | boundary | 0.594691 / 1 | 0.595522 / 1 | 0.744470 / 1 |
| EDOWN | nested | 0.606583 / 1 | 0.763977 / 1 | 1.025436 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.607214 / 1 | 0.764889 / 1 | 1.026378 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.060473 / 1 | 0.035837 / 1 | 0.012905 / 1 |
| SH1 | nested | 1.022320 / 1 | 1.361554 / 1 | 1.935627 / 1 |
| SH3 | nested | 1.026458 / 1 | 1.360803 / 1 | 1.765479 / 1 |
| SH2 | nested | 0.996832 / 1 | 1.373065 / 1 | 1.883830 / 1 |
| shared-expert-during-read | boundary | 3.059196 / 1 | 4.110390 / 1 | 5.599092 / 1 |
| detail:expert-gate | nested | 23.921799 / 16 | 22.004605 / 16 | 15.513479 / 16 |
| detail:expert-up | nested | 22.707618 / 16 | 20.500734 / 16 | 14.776039 / 16 |
| detail:expert-activation | nested | 0.096331 / 16 | 0.110146 / 16 | 0.095566 / 16 |
| detail:expert-down | nested | 23.354577 / 16 | 23.409352 / 16 | 15.387802 / 16 |
| EUP | nested | 0.700028 / 1 | 0.594771 / 1 | 0.994518 / 1 |
| experts-mix-normalize-up | boundary | 71.201058 / 1 | 67.026147 / 1 | 47.115874 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002245 / 1 | 0.002345 / 1 | 0.002435 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000231 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.446855 / 1 | 0.429323 / 1 | 0.368589 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 974.361022 / 1 | 849.524005 / 1 | 628.958169 / 1 |
| op:Q   int8 projection | nested | 15.680148 / 1 | 19.894121 / 1 | 25.807061 / 1 |
| op:X   mxfp4 expert proj | nested | 70.113365 / 1 | 66.061573 / 1 | 45.803203 / 1 |
| op:N   rmsnorm | nested | 0.043110 / 1 | 0.037400 / 1 | 0.024396 / 1 |
| op:L   l2 per-head | nested | 0.005391 / 1 | 0.006131 / 1 | 0.006852 / 1 |
| op:SiTU + sigma | nested | 0.106298 / 1 | 0.120977 / 1 | 0.106006 / 1 |
| op:C   shortconv | nested | 0.058248 / 1 | 0.056706 / 1 | 0.043472 / 1 |
| op:AR  snapshot aggregate | nested | 0.036308 / 1 | 0.039554 / 1 | 0.034024 / 1 |
| op:D   kda delta-rule | nested | 0.134130 / 1 | 0.131385 / 1 | 0.229820 / 1 |
| op:router dot product | nested | 0.592086 / 1 | 0.591695 / 1 | 0.741806 / 1 |
| op:top-k selection | nested | 0.002204 / 1 | 0.003416 / 1 | 0.002334 / 1 |
| op:alpha / beta / gate | nested | 0.070942 / 1 | 0.106058 / 1 | 0.057989 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 88.530797 / 1 | 88.681950 / 1 | 73.813678 / 1 |

### Layer 19

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006262 / 1 | 0.005851 / 1 | 0.006161 / 1 |
| pre-attention-aggregation | boundary | 0.018375 / 1 | 0.017923 / 1 | 0.016360 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011482 / 1 | 0.011201 / 1 | 0.010901 / 1 |
| QA | nested | 0.313656 / 1 | 0.294801 / 1 | 0.529620 / 1 |
| QB | nested | 0.779156 / 1 | 0.680301 / 1 | 1.103281 / 1 |
| KA | nested | 0.109425 / 1 | 0.123050 / 1 | 0.223688 / 1 |
| KB | nested | 0.348622 / 1 | 0.306423 / 1 | 0.578651 / 1 |
| G | nested | 2.316328 / 1 | 1.956084 / 1 | 3.195341 / 1 |
| O | nested | 2.335884 / 1 | 1.944714 / 1 | 3.129176 / 1 |
| attention | boundary | 6.300543 / 1 | 5.409187 / 1 | 8.858753 / 1 |
| attention-residual | boundary | 0.002285 / 1 | 0.002455 / 1 | 0.002174 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.035867 / 1 | 0.024706 / 1 | 0.026139 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001934 / 1 | 0.002394 / 1 | 0.002986 / 1 |
| router-and-top16 | boundary | 0.618966 / 1 | 0.566358 / 1 | 0.679509 / 1 |
| EDOWN | nested | 0.679579 / 1 | 0.566478 / 1 | 0.946368 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.680521 / 1 | 0.567199 / 1 | 0.947050 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.053300 / 1 | 0.027712 / 1 | 0.012103 / 1 |
| SH1 | nested | 1.131755 / 1 | 0.963721 / 1 | 1.477150 / 1 |
| SH3 | nested | 1.148626 / 1 | 0.967708 / 1 | 1.550918 / 1 |
| SH2 | nested | 1.140221 / 1 | 0.970193 / 1 | 1.465809 / 1 |
| shared-expert-during-read | boundary | 3.434157 / 1 | 2.912723 / 1 | 4.504716 / 1 |
| detail:expert-gate | nested | 26.787813 / 16 | 25.471966 / 16 | 16.084883 / 16 |
| detail:expert-up | nested | 25.824186 / 16 | 23.363134 / 16 | 15.456550 / 16 |
| detail:expert-activation | nested | 0.111167 / 16 | 0.118572 / 16 | 0.096000 / 16 |
| detail:expert-down | nested | 25.277381 / 16 | 25.971316 / 16 | 16.568295 / 16 |
| EUP | nested | 0.716418 / 1 | 0.775238 / 1 | 1.021879 / 1 |
| experts-mix-normalize-up | boundary | 79.138808 / 1 | 76.101413 / 1 | 49.584177 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002615 / 1 | 0.002505 / 1 | 0.002425 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000300 / 1 | 0.000231 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006081 / 1 | 0.006602 / 1 | 0.007814 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1002.709031 / 1 | 945.054161 / 1 | 665.624631 / 1 |
| op:Q   int8 projection | nested | 11.018338 / 1 | 9.547538 / 1 | 15.220146 / 1 |
| op:X   mxfp4 expert proj | nested | 78.034355 / 1 | 74.958720 / 1 | 48.236698 / 1 |
| op:N   rmsnorm | nested | 0.024206 / 1 | 0.024617 / 1 | 0.023193 / 1 |
| op:SiTU + sigma | nested | 0.121805 / 1 | 0.126188 / 1 | 0.103414 / 1 |
| op:AR  snapshot aggregate | nested | 0.044453 / 1 | 0.032702 / 1 | 0.031238 / 1 |
| op:SA  softmax attention | nested | 0.006903 / 1 | 0.009699 / 1 | 0.011842 / 1 |
| op:router dot product | nested | 0.616231 / 1 | 0.563853 / 1 | 0.676895 / 1 |
| op:top-k selection | nested | 0.002355 / 1 | 0.002165 / 1 | 0.002164 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 90.317236 / 1 | 85.663590 / 1 | 64.666767 / 1 |

### Layer 20

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006643 / 1 | 0.004739 / 1 | 0.005821 / 1 |
| pre-attention-aggregation | boundary | 0.019065 / 1 | 0.015078 / 1 | 0.015509 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010790 / 1 | 0.010109 / 1 | 0.011151 / 1 |
| Q | nested | 2.876124 / 1 | 3.100844 / 1 | 4.505850 / 1 |
| K | nested | 2.825840 / 1 | 2.515070 / 1 | 3.448653 / 1 |
| V | nested | 2.732085 / 1 | 2.236248 / 1 | 2.955032 / 1 |
| B | nested | 0.023994 / 1 | 0.020709 / 1 | 0.030026 / 1 |
| FA | nested | 0.031659 / 1 | 0.030948 / 1 | 0.040465 / 1 |
| FB | nested | 0.050144 / 1 | 0.050174 / 1 | 0.167052 / 1 |
| G | nested | 2.364808 / 1 | 1.914788 / 1 | 3.347425 / 1 |
| O | nested | 2.390487 / 1 | 2.040733 / 1 | 3.436340 / 1 |
| attention | boundary | 14.420936 / 1 | 12.990202 / 1 | 18.696513 / 1 |
| attention-residual | boundary | 0.002565 / 1 | 0.002795 / 1 | 0.001774 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025788 / 1 | 0.025357 / 1 | 0.027141 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002274 / 1 | 0.002325 / 1 | 0.003396 / 1 |
| router-and-top16 | boundary | 0.659222 / 1 | 0.595012 / 1 | 0.739512 / 1 |
| EDOWN | nested | 0.720015 / 1 | 0.573191 / 1 | 1.191316 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.720626 / 1 | 0.573862 / 1 | 1.192047 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.091861 / 1 | 0.022842 / 1 | 0.006682 / 1 |
| SH1 | nested | 1.239516 / 1 | 0.979531 / 1 | 1.617753 / 1 |
| SH3 | nested | 1.180596 / 1 | 0.986884 / 1 | 1.984247 / 1 |
| SH2 | nested | 1.218677 / 1 | 0.984770 / 1 | 1.680790 / 1 |
| shared-expert-during-read | boundary | 3.652415 / 1 | 2.962475 / 1 | 5.296636 / 1 |
| detail:expert-gate | nested | 24.811078 / 16 | 21.744880 / 16 | 20.080049 / 16 |
| detail:expert-up | nested | 23.431543 / 16 | 21.486886 / 16 | 19.574754 / 16 |
| detail:expert-activation | nested | 0.099185 / 16 | 0.095379 / 16 | 0.119694 / 16 |
| detail:expert-down | nested | 24.365376 / 16 | 22.585420 / 16 | 18.008719 / 16 |
| EUP | nested | 0.585664 / 1 | 0.843286 / 1 | 0.960455 / 1 |
| experts-mix-normalize-up | boundary | 73.743649 / 1 | 67.129359 / 1 | 59.103513 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002565 / 1 | 0.002765 / 1 | 0.002344 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000141 / 1 | 0.000571 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429743 / 1 | 0.432639 / 1 | 0.381714 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 995.860851 / 1 | 848.607777 / 1 | 677.328953 / 1 |
| op:Q   int8 projection | nested | 18.238248 / 1 | 16.275742 / 1 | 25.363859 / 1 |
| op:X   mxfp4 expert proj | nested | 72.741705 / 1 | 65.946109 / 1 | 57.815395 / 1 |
| op:N   rmsnorm | nested | 0.034053 / 1 | 0.031849 / 1 | 0.024056 / 1 |
| op:L   l2 per-head | nested | 0.005240 / 1 | 0.005099 / 1 | 0.005941 / 1 |
| op:SiTU + sigma | nested | 0.109003 / 1 | 0.103153 / 1 | 0.130374 / 1 |
| op:C   shortconv | nested | 0.058117 / 1 | 0.058479 / 1 | 0.036567 / 1 |
| op:AR  snapshot aggregate | nested | 0.035566 / 1 | 0.030486 / 1 | 0.031558 / 1 |
| op:D   kda delta-rule | nested | 0.154399 / 1 | 0.142016 / 1 | 0.149259 / 1 |
| op:router dot product | nested | 0.655956 / 1 | 0.592237 / 1 | 0.736737 / 1 |
| op:top-k selection | nested | 0.002845 / 1 | 0.002395 / 1 | 0.002365 / 1 |
| op:alpha / beta / gate | nested | 0.074098 / 1 | 0.072726 / 1 | 0.057948 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.795244 / 1 | 84.775641 / 1 | 85.490226 / 1 |

### Layer 21

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005520 / 1 | 0.005370 / 1 | 0.005761 / 1 |
| pre-attention-aggregation | boundary | 0.018184 / 1 | 0.019396 / 1 | 0.016010 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012643 / 1 | 0.010931 / 1 | 0.011341 / 1 |
| Q | nested | 2.540066 / 1 | 3.393460 / 1 | 5.543338 / 1 |
| K | nested | 2.398491 / 1 | 3.237098 / 1 | 3.619793 / 1 |
| V | nested | 2.288526 / 1 | 3.184700 / 1 | 3.030713 / 1 |
| B | nested | 0.022913 / 1 | 0.028493 / 1 | 0.029546 / 1 |
| FA | nested | 0.029435 / 1 | 0.036137 / 1 | 0.039925 / 1 |
| FB | nested | 0.048520 / 1 | 0.065212 / 1 | 0.155480 / 1 |
| G | nested | 1.945245 / 1 | 2.684445 / 1 | 3.433997 / 1 |
| O | nested | 2.038739 / 1 | 2.750309 / 1 | 3.222651 / 1 |
| attention | boundary | 12.411070 / 1 | 16.480813 / 1 | 19.724995 / 1 |
| attention-residual | boundary | 0.002574 / 1 | 0.002465 / 1 | 0.001924 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025177 / 1 | 0.025467 / 1 | 0.026500 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.002424 / 1 | 0.003015 / 1 |
| router-and-top16 | boundary | 0.601224 / 1 | 0.595813 / 1 | 0.685992 / 1 |
| EDOWN | nested | 0.724163 / 1 | 0.801248 / 1 | 1.064900 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.724824 / 1 | 0.801989 / 1 | 1.065651 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059431 / 1 | 0.021891 / 1 | 0.006492 / 1 |
| SH1 | nested | 1.009035 / 1 | 1.358418 / 1 | 1.767722 / 1 |
| SH3 | nested | 1.027139 / 1 | 1.369099 / 1 | 1.542732 / 1 |
| SH2 | nested | 1.010608 / 1 | 1.362946 / 1 | 1.734892 / 1 |
| shared-expert-during-read | boundary | 3.057823 / 1 | 4.103738 / 1 | 5.056979 / 1 |
| detail:expert-gate | nested | 24.871363 / 16 | 24.703339 / 16 | 21.696390 / 16 |
| detail:expert-up | nested | 24.508757 / 16 | 24.063660 / 16 | 20.979109 / 16 |
| detail:expert-activation | nested | 0.102026 / 16 | 0.099488 / 16 | 0.103912 / 16 |
| detail:expert-down | nested | 26.424595 / 16 | 25.148716 / 16 | 21.398582 / 16 |
| EUP | nested | 0.829410 / 1 | 0.586075 / 1 | 1.070881 / 1 |
| experts-mix-normalize-up | boundary | 77.153970 / 1 | 74.984346 / 1 | 65.611082 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002504 / 1 | 0.002184 / 1 | 0.004920 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000150 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.434442 / 1 | 0.417100 / 1 | 0.359501 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 947.356155 / 1 | 982.351118 / 1 | 858.752288 / 1 |
| op:Q   int8 projection | nested | 15.910878 / 1 | 20.855988 / 1 | 26.254434 / 1 |
| op:X   mxfp4 expert proj | nested | 75.942408 / 1 | 74.051362 / 1 | 64.212657 / 1 |
| op:N   rmsnorm | nested | 0.036599 / 1 | 0.051236 / 1 | 0.024785 / 1 |
| op:L   l2 per-head | nested | 0.005250 / 1 | 0.005050 / 1 | 0.005269 / 1 |
| op:SiTU + sigma | nested | 0.109962 / 1 | 0.109589 / 1 | 0.111798 / 1 |
| op:C   shortconv | nested | 0.068036 / 1 | 0.056746 / 1 | 0.040606 / 1 |
| op:AR  snapshot aggregate | nested | 0.033482 / 1 | 0.034976 / 1 | 0.031859 / 1 |
| op:D   kda delta-rule | nested | 0.138148 / 1 | 0.138669 / 1 | 0.209983 / 1 |
| op:router dot product | nested | 0.598268 / 1 | 0.592578 / 1 | 0.682966 / 1 |
| op:top-k selection | nested | 0.002504 / 1 | 0.002545 / 1 | 0.002604 / 1 |
| op:alpha / beta / gate | nested | 0.076863 / 1 | 0.073187 / 1 | 0.060904 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.518465 / 1 | 97.480460 / 1 | 92.586916 / 1 |

### Layer 22

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004919 / 1 | 0.005260 / 1 | 0.006753 / 1 |
| pre-attention-aggregation | boundary | 0.020618 / 1 | 0.017293 / 1 | 0.016430 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011120 / 1 | 0.011271 / 1 | 0.011181 / 1 |
| Q | nested | 2.524548 / 1 | 2.518296 / 1 | 4.933939 / 1 |
| K | nested | 2.388322 / 1 | 2.380077 / 1 | 3.626757 / 1 |
| V | nested | 2.293996 / 1 | 2.255665 / 1 | 3.173069 / 1 |
| B | nested | 0.020468 / 1 | 0.020768 / 1 | 0.031860 / 1 |
| FA | nested | 0.027071 / 1 | 0.027432 / 1 | 0.036128 / 1 |
| FB | nested | 0.061334 / 1 | 0.046848 / 1 | 0.156492 / 1 |
| G | nested | 1.985109 / 1 | 1.928012 / 1 | 3.527271 / 1 |
| O | nested | 2.029362 / 1 | 2.023171 / 1 | 3.530847 / 1 |
| attention | boundary | 12.415308 / 1 | 12.236493 / 1 | 19.670974 / 1 |
| attention-residual | boundary | 0.002354 / 1 | 0.002375 / 1 | 0.001944 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025598 / 1 | 0.024846 / 1 | 0.028733 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002795 / 1 | 0.002384 / 1 | 0.003246 / 1 |
| router-and-top16 | boundary | 0.602115 / 1 | 0.586476 / 1 | 0.732419 / 1 |
| EDOWN | nested | 0.628023 / 1 | 0.586446 / 1 | 1.082422 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.629095 / 1 | 0.587177 / 1 | 1.083494 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052729 / 1 | 0.038832 / 1 | 0.005831 / 1 |
| SH1 | nested | 1.090227 / 1 | 0.985992 / 1 | 1.820001 / 1 |
| SH3 | nested | 1.209650 / 1 | 0.993507 / 1 | 1.791678 / 1 |
| SH2 | nested | 1.223446 / 1 | 0.985732 / 1 | 1.595521 / 1 |
| shared-expert-during-read | boundary | 3.536719 / 1 | 2.976942 / 1 | 5.222348 / 1 |
| detail:expert-gate | nested | 25.288368 / 16 | 25.975042 / 16 | 22.007230 / 16 |
| detail:expert-up | nested | 24.566343 / 16 | 21.781848 / 16 | 21.520383 / 16 |
| detail:expert-activation | nested | 0.109705 / 16 | 0.125713 / 16 | 0.118472 / 16 |
| detail:expert-down | nested | 24.664988 / 16 | 22.380668 / 16 | 19.930777 / 16 |
| EUP | nested | 0.734372 / 1 | 0.897507 / 1 | 1.326929 / 1 |
| experts-mix-normalize-up | boundary | 75.754926 / 1 | 71.612937 / 1 | 65.264064 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002675 / 1 | 0.003016 / 1 | 0.002705 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000331 / 1 | 0.000231 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.422800 / 1 | 0.441185 / 1 | 0.370502 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 957.079512 / 1 | 919.076805 / 1 | 798.837191 / 1 |
| op:Q   int8 projection | nested | 16.214386 / 1 | 15.647898 / 1 | 26.631052 / 1 |
| op:X   mxfp4 expert proj | nested | 74.665198 / 1 | 70.313237 / 1 | 63.612852 / 1 |
| op:N   rmsnorm | nested | 0.034274 / 1 | 0.032622 / 1 | 0.024345 / 1 |
| op:L   l2 per-head | nested | 0.005310 / 1 | 0.005250 / 1 | 0.006432 / 1 |
| op:SiTU + sigma | nested | 0.119932 / 1 | 0.132908 / 1 | 0.130212 / 1 |
| op:C   shortconv | nested | 0.063028 / 1 | 0.054582 / 1 | 0.043301 / 1 |
| op:AR  snapshot aggregate | nested | 0.036097 / 1 | 0.032681 / 1 | 0.034495 / 1 |
| op:D   kda delta-rule | nested | 0.131526 / 1 | 0.128641 / 1 | 0.223107 / 1 |
| op:router dot product | nested | 0.599320 / 1 | 0.583320 / 1 | 0.729393 / 1 |
| op:top-k selection | nested | 0.002485 / 1 | 0.002775 / 1 | 0.002455 / 1 |
| op:alpha / beta / gate | nested | 0.073938 / 1 | 0.074579 / 1 | 0.057888 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.490474 / 1 | 88.554090 / 1 | 92.427869 / 1 |

### Layer 23

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005851 / 1 | 0.006493 / 1 | 0.006141 / 1 |
| pre-attention-aggregation | boundary | 0.018805 / 1 | 0.022051 / 1 | 0.016481 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011211 / 1 | 0.012624 / 1 | 0.011602 / 1 |
| QA | nested | 0.330418 / 1 | 0.381132 / 1 | 0.618585 / 1 |
| QB | nested | 0.819080 / 1 | 0.887619 / 1 | 1.271576 / 1 |
| KA | nested | 0.113582 / 1 | 0.137427 / 1 | 0.223678 / 1 |
| KB | nested | 0.365383 / 1 | 0.377455 / 1 | 0.569905 / 1 |
| G | nested | 2.346715 / 1 | 2.876785 / 1 | 3.771847 / 1 |
| O | nested | 1.988355 / 1 | 3.015565 / 1 | 3.599535 / 1 |
| attention | boundary | 6.065062 / 1 | 7.804733 / 1 | 10.166636 / 1 |
| attention-residual | boundary | 0.002304 / 1 | 0.002674 / 1 | 0.002565 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024927 / 1 | 0.028704 / 1 | 0.029615 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002534 / 1 | 0.002525 / 1 | 0.003096 / 1 |
| router-and-top16 | boundary | 0.576828 / 1 | 0.591385 / 1 | 0.926872 / 1 |
| EDOWN | nested | 0.592357 / 1 | 0.855288 / 1 | 1.094675 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.593218 / 1 | 0.856289 / 1 | 1.095537 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.073587 / 1 | 0.036167 / 1 | 0.006562 / 1 |
| SH1 | nested | 0.981634 / 1 | 1.476910 / 1 | 1.855227 / 1 |
| SH3 | nested | 0.991413 / 1 | 1.562920 / 1 | 1.662747 / 1 |
| SH2 | nested | 0.994468 / 1 | 1.573049 / 1 | 1.837163 / 1 |
| shared-expert-during-read | boundary | 2.978946 / 1 | 4.628889 / 1 | 5.369683 / 1 |
| detail:expert-gate | nested | 22.096131 / 16 | 26.027363 / 16 | 22.662365 / 16 |
| detail:expert-up | nested | 21.457146 / 16 | 25.363542 / 16 | 22.157041 / 16 |
| detail:expert-activation | nested | 0.098134 / 16 | 0.118821 / 16 | 0.127657 / 16 |
| detail:expert-down | nested | 22.446097 / 16 | 23.857265 / 16 | 20.749422 / 16 |
| EUP | nested | 0.600782 / 1 | 0.696140 / 1 | 1.043439 / 1 |
| experts-mix-normalize-up | boundary | 67.143215 / 1 | 76.459933 / 1 | 67.089445 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002225 / 1 | 0.002715 / 1 | 0.002605 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000151 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007695 / 1 | 0.007404 / 1 | 0.007904 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 912.270482 / 1 | 959.080513 / 1 | 826.472844 / 1 |
| op:Q   int8 projection | nested | 10.122706 / 1 | 13.838877 / 1 | 17.546623 / 1 |
| op:X   mxfp4 expert proj | nested | 66.134080 / 1 | 75.404422 / 1 | 65.732780 / 1 |
| op:N   rmsnorm | nested | 0.024575 / 1 | 0.026088 / 1 | 0.025429 / 1 |
| op:SiTU + sigma | nested | 0.106300 / 1 | 0.130685 / 1 | 0.138629 / 1 |
| op:AR  snapshot aggregate | nested | 0.033544 / 1 | 0.039565 / 1 | 0.033983 / 1 |
| op:SA  softmax attention | nested | 0.007213 / 1 | 0.009638 / 1 | 0.012634 / 1 |
| op:router dot product | nested | 0.573452 / 1 | 0.588289 / 1 | 0.923686 / 1 |
| op:top-k selection | nested | 0.002876 / 1 | 0.002645 / 1 | 0.002685 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 77.513282 / 1 | 90.472656 / 1 | 84.742639 / 1 |

### Layer 24

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004298 / 1 | 0.004538 / 1 | 0.006181 / 1 |
| pre-attention-aggregation | boundary | 0.014978 / 1 | 0.015128 / 1 | 0.015609 / 1 |
| snapshot-push | boundary | 0.001282 / 1 | 0.001182 / 1 | 0.001122 / 1 |
| pre-attention-normalization | boundary | 0.010860 / 1 | 0.009979 / 1 | 0.010409 / 1 |
| Q | nested | 2.508086 / 1 | 2.542440 / 1 | 5.056408 / 1 |
| K | nested | 2.396107 / 1 | 2.384957 / 1 | 3.559310 / 1 |
| V | nested | 2.260815 / 1 | 2.254533 / 1 | 2.630855 / 1 |
| B | nested | 0.022512 / 1 | 0.021390 / 1 | 0.124272 / 1 |
| FA | nested | 0.027732 / 1 | 0.027541 / 1 | 0.126827 / 1 |
| FB | nested | 0.047408 / 1 | 0.047629 / 1 | 0.154278 / 1 |
| G | nested | 1.953040 / 1 | 1.941007 / 1 | 3.543310 / 1 |
| O | nested | 2.049399 / 1 | 2.023230 / 1 | 3.259180 / 1 |
| attention | boundary | 12.363962 / 1 | 12.362048 / 1 | 19.181741 / 1 |
| attention-residual | boundary | 0.001994 / 1 | 0.001453 / 1 | 0.001733 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025407 / 1 | 0.026509 / 1 | 0.026910 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002665 / 1 | 0.002485 / 1 | 0.003186 / 1 |
| router-and-top16 | boundary | 0.613857 / 1 | 0.590533 / 1 | 0.772694 / 1 |
| EDOWN | nested | 0.587007 / 1 | 0.592927 / 1 | 1.059760 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.587728 / 1 | 0.593649 / 1 | 1.060701 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.062837 / 1 | 0.028733 / 1 | 0.007003 / 1 |
| SH1 | nested | 1.001020 / 1 | 1.004357 / 1 | 1.708713 / 1 |
| SH3 | nested | 1.032288 / 1 | 0.998736 / 1 | 2.039581 / 1 |
| SH2 | nested | 1.019065 / 1 | 0.986754 / 1 | 1.874101 / 1 |
| shared-expert-during-read | boundary | 3.063584 / 1 | 3.001478 / 1 | 5.633837 / 1 |
| detail:expert-gate | nested | 21.914167 / 16 | 19.748944 / 16 | 16.846107 / 16 |
| detail:expert-up | nested | 21.276404 / 16 | 18.456355 / 16 | 16.253128 / 16 |
| detail:expert-activation | nested | 0.113792 / 16 | 0.105119 / 16 | 0.101873 / 16 |
| detail:expert-down | nested | 19.957661 / 16 | 19.432219 / 16 | 17.249171 / 16 |
| EUP | nested | 0.608177 / 1 | 0.588409 / 1 | 0.999367 / 1 |
| experts-mix-normalize-up | boundary | 64.301115 / 1 | 58.715448 / 1 | 51.786782 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002324 / 1 | 0.002375 / 1 | 0.003005 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000140 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.432158 / 1 | 0.426246 / 1 | 0.369521 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 798.825959 / 1 | 804.935782 / 1 | 711.995554 / 1 |
| op:Q   int8 projection | nested | 15.511280 / 1 | 15.412468 / 1 | 26.134139 / 1 |
| op:X   mxfp4 expert proj | nested | 63.300056 / 1 | 57.779873 / 1 | 50.486653 / 1 |
| op:N   rmsnorm | nested | 0.035718 / 1 | 0.032701 / 1 | 0.023113 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.006973 / 1 | 0.005270 / 1 |
| op:SiTU + sigma | nested | 0.121433 / 1 | 0.112423 / 1 | 0.109363 / 1 |
| op:C   shortconv | nested | 0.058409 / 1 | 0.052609 / 1 | 0.036218 / 1 |
| op:AR  snapshot aggregate | nested | 0.030877 / 1 | 0.031148 / 1 | 0.031509 / 1 |
| op:D   kda delta-rule | nested | 0.140883 / 1 | 0.141725 / 1 | 0.144260 / 1 |
| op:router dot product | nested | 0.610851 / 1 | 0.587438 / 1 | 0.769277 / 1 |
| op:top-k selection | nested | 0.002534 / 1 | 0.002515 / 1 | 0.002545 / 1 |
| op:alpha / beta / gate | nested | 0.075010 / 1 | 0.079809 / 1 | 0.056475 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 81.497135 / 1 | 75.789530 / 1 | 78.888370 / 1 |

### Layer 25

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004949 / 1 | 0.005460 / 1 | 0.006191 / 1 |
| pre-attention-aggregation | boundary | 0.020218 / 1 | 0.021140 / 1 | 0.016140 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011452 / 1 | 0.012183 / 1 | 0.011992 / 1 |
| Q | nested | 2.527693 / 1 | 2.499119 / 1 | 4.372940 / 1 |
| K | nested | 2.424410 / 1 | 2.384125 / 1 | 3.176505 / 1 |
| V | nested | 2.278998 / 1 | 2.244183 / 1 | 2.639903 / 1 |
| B | nested | 0.025979 / 1 | 0.020509 / 1 | 0.130874 / 1 |
| FA | nested | 0.028433 / 1 | 0.029264 / 1 | 0.033142 / 1 |
| FB | nested | 0.046457 / 1 | 0.048381 / 1 | 0.144139 / 1 |
| G | nested | 1.986552 / 1 | 1.918545 / 1 | 3.349739 / 1 |
| O | nested | 2.038269 / 1 | 2.048237 / 1 | 2.999665 / 1 |
| attention | boundary | 12.491621 / 1 | 12.260819 / 1 | 17.572453 / 1 |
| attention-residual | boundary | 0.002504 / 1 | 0.002545 / 1 | 0.002024 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025658 / 1 | 0.035246 / 1 | 0.026349 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002284 / 1 | 0.002524 / 1 | 0.102392 / 1 |
| router-and-top16 | boundary | 0.600472 / 1 | 0.594511 / 1 | 0.749179 / 1 |
| EDOWN | nested | 0.613596 / 1 | 0.580475 / 1 | 1.035255 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.614408 / 1 | 0.581206 / 1 | 1.036096 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.054702 / 1 | 0.012974 / 1 | 0.006753 / 1 |
| SH1 | nested | 1.009015 / 1 | 0.997684 / 1 | 1.775508 / 1 |
| SH3 | nested | 1.021198 / 1 | 1.001331 / 1 | 1.788733 / 1 |
| SH2 | nested | 1.024575 / 1 | 0.996101 / 1 | 1.545187 / 1 |
| shared-expert-during-read | boundary | 3.066369 / 1 | 3.006357 / 1 | 5.120698 / 1 |
| detail:expert-gate | nested | 24.340676 / 16 | 25.482765 / 16 | 24.310896 / 16 |
| detail:expert-up | nested | 23.345502 / 16 | 23.573087 / 16 | 23.621286 / 16 |
| detail:expert-activation | nested | 0.100667 / 16 | 0.114734 / 16 | 0.117349 / 16 |
| detail:expert-down | nested | 23.624404 / 16 | 25.524220 / 16 | 22.438496 / 16 |
| EUP | nested | 0.713232 / 1 | 0.715217 / 1 | 0.984610 / 1 |
| experts-mix-normalize-up | boundary | 72.544458 / 1 | 75.784051 / 1 | 71.814113 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002204 / 1 | 0.002985 / 1 | 0.002474 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000320 / 1 | 0.000150 / 1 | 0.000351 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.417360 / 1 | 0.417750 / 1 | 0.403504 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 990.111865 / 1 | 961.205271 / 1 | 905.994474 / 1 |
| op:Q   int8 projection | nested | 15.736604 / 1 | 15.481737 / 1 | 23.974188 / 1 |
| op:X   mxfp4 expert proj | nested | 71.450691 / 1 | 74.734199 / 1 | 70.529043 / 1 |
| op:N   rmsnorm | nested | 0.042559 / 1 | 0.035065 / 1 | 0.023924 / 1 |
| op:L   l2 per-head | nested | 0.004869 / 1 | 0.004779 / 1 | 0.005801 / 1 |
| op:SiTU + sigma | nested | 0.108524 / 1 | 0.122510 / 1 | 0.124963 / 1 |
| op:C   shortconv | nested | 0.052367 / 1 | 0.056176 / 1 | 0.036639 / 1 |
| op:AR  snapshot aggregate | nested | 0.036468 / 1 | 0.044764 / 1 | 0.031981 / 1 |
| op:D   kda delta-rule | nested | 0.135363 / 1 | 0.136315 / 1 | 0.216795 / 1 |
| op:router dot product | nested | 0.598078 / 1 | 0.591816 / 1 | 0.746605 / 1 |
| op:top-k selection | nested | 0.002134 / 1 | 0.002355 / 1 | 0.002094 / 1 |
| op:alpha / beta / gate | nested | 0.080280 / 1 | 0.072105 / 1 | 0.057026 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.868727 / 1 | 92.747847 / 1 | 96.878915 / 1 |

### Layer 26

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005170 / 1 | 0.005249 / 1 | 0.007294 / 1 |
| pre-attention-aggregation | boundary | 0.018164 / 1 | 0.017864 / 1 | 0.018484 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010870 / 1 | 0.014267 / 1 | 0.013325 / 1 |
| Q | nested | 2.789993 / 1 | 2.760398 / 1 | 4.325061 / 1 |
| K | nested | 2.666482 / 1 | 2.669248 / 1 | 3.330703 / 1 |
| V | nested | 2.685268 / 1 | 2.631777 / 1 | 2.938931 / 1 |
| B | nested | 0.026479 / 1 | 0.023093 / 1 | 0.031008 / 1 |
| FA | nested | 0.028032 / 1 | 0.028774 / 1 | 0.133309 / 1 |
| FB | nested | 0.061134 / 1 | 0.060503 / 1 | 0.151073 / 1 |
| G | nested | 2.329874 / 1 | 2.241127 / 1 | 3.717426 / 1 |
| O | nested | 2.484863 / 1 | 2.364828 / 1 | 3.304334 / 1 |
| attention | boundary | 14.172261 / 1 | 13.885735 / 1 | 18.849800 / 1 |
| attention-residual | boundary | 0.003376 / 1 | 0.003356 / 1 | 0.003246 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025047 / 1 | 0.024376 / 1 | 0.025848 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002234 / 1 | 0.002414 / 1 | 0.003036 / 1 |
| router-and-top16 | boundary | 0.594130 / 1 | 0.598909 / 1 | 0.859295 / 1 |
| EDOWN | nested | 0.686422 / 1 | 0.657438 / 1 | 1.101357 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.687134 / 1 | 0.658290 / 1 | 1.102891 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.057377 / 1 | 0.017814 / 1 | 0.013575 / 1 |
| SH1 | nested | 1.137495 / 1 | 1.137375 / 1 | 1.682734 / 1 |
| SH3 | nested | 1.128008 / 1 | 1.124271 / 1 | 1.682413 / 1 |
| SH2 | nested | 1.195233 / 1 | 1.133989 / 1 | 1.695989 / 1 |
| shared-expert-during-read | boundary | 3.474512 / 1 | 3.409330 / 1 | 5.076064 / 1 |
| detail:expert-gate | nested | 22.324933 / 16 | 42.939630 / 16 | 22.559672 / 16 |
| detail:expert-up | nested | 21.327671 / 16 | 41.707772 / 16 | 21.427005 / 16 |
| detail:expert-activation | nested | 0.098854 / 16 | 0.119071 / 16 | 0.101749 / 16 |
| detail:expert-down | nested | 22.147329 / 16 | 39.472156 / 16 | 21.710214 / 16 |
| EUP | nested | 0.592026 / 1 | 0.879533 / 1 | 1.026418 / 1 |
| experts-mix-normalize-up | boundary | 66.929966 / 1 | 125.534888 / 1 | 67.160607 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003026 / 1 | 0.002505 / 1 | 0.002695 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000211 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.381102 / 1 | 0.392954 / 1 | 0.361455 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 900.997673 / 1 | 1500.843167 / 1 | 896.012855 / 1 |
| op:Q   int8 projection | nested | 17.809757 / 1 | 17.710740 / 1 | 25.119216 / 1 |
| op:X   mxfp4 expert proj | nested | 65.938564 / 1 | 124.278312 / 1 | 65.837222 / 1 |
| op:N   rmsnorm | nested | 0.035116 / 1 | 0.037400 / 1 | 0.024937 / 1 |
| op:L   l2 per-head | nested | 0.005130 / 1 | 0.005570 / 1 | 0.006202 / 1 |
| op:SiTU + sigma | nested | 0.109124 / 1 | 0.129342 / 1 | 0.112520 / 1 |
| op:C   shortconv | nested | 0.056736 / 1 | 0.079899 / 1 | 0.035627 / 1 |
| op:AR  snapshot aggregate | nested | 0.033072 / 1 | 0.032771 / 1 | 0.033443 / 1 |
| op:D   kda delta-rule | nested | 0.137406 / 1 | 0.137797 / 1 | 0.126245 / 1 |
| op:router dot product | nested | 0.591395 / 1 | 0.590262 / 1 | 0.856080 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.008235 / 1 | 0.002835 / 1 |
| op:alpha / beta / gate | nested | 0.071303 / 1 | 0.076293 / 1 | 0.057928 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 86.371953 / 1 | 144.575795 / 1 | 93.505903 / 1 |

### Layer 27

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004008 / 1 | 0.004107 / 1 | 0.006823 / 1 |
| pre-attention-aggregation | boundary | 0.016420 / 1 | 0.016761 / 1 | 0.017442 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010600 / 1 | 0.010570 / 1 | 0.011071 / 1 |
| QA | nested | 0.297707 / 1 | 0.389688 / 1 | 0.434041 / 1 |
| QB | nested | 0.792972 / 1 | 0.936249 / 1 | 1.227122 / 1 |
| KA | nested | 0.109775 / 1 | 0.139881 / 1 | 0.154188 / 1 |
| KB | nested | 0.323464 / 1 | 0.372326 / 1 | 0.541221 / 1 |
| G | nested | 1.969110 / 1 | 2.721435 / 1 | 3.300387 / 1 |
| O | nested | 1.960253 / 1 | 2.943570 / 1 | 3.822442 / 1 |
| attention | boundary | 5.549809 / 1 | 7.601353 / 1 | 9.581222 / 1 |
| attention-residual | boundary | 0.003096 / 1 | 0.002565 / 1 | 0.001773 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024706 / 1 | 0.025998 / 1 | 0.026359 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002264 / 1 | 0.002034 / 1 | 0.002575 / 1 |
| router-and-top16 | boundary | 0.564465 / 1 | 0.588660 / 1 | 0.963350 / 1 |
| EDOWN | nested | 0.578691 / 1 | 0.841462 / 1 | 1.092151 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.579423 / 1 | 0.842204 / 1 | 1.092902 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078406 / 1 | 0.022572 / 1 | 0.006643 / 1 |
| SH1 | nested | 0.993076 / 1 | 1.427908 / 1 | 1.907714 / 1 |
| SH3 | nested | 0.968740 / 1 | 1.430022 / 1 | 2.056883 / 1 |
| SH2 | nested | 0.987415 / 1 | 1.521733 / 1 | 1.794282 / 1 |
| shared-expert-during-read | boundary | 2.960001 / 1 | 4.394170 / 1 | 5.774279 / 1 |
| detail:expert-gate | nested | 25.458930 / 16 | 25.029315 / 16 | 23.290867 / 16 |
| detail:expert-up | nested | 25.667531 / 16 | 24.721370 / 16 | 22.764324 / 16 |
| detail:expert-activation | nested | 0.114343 / 16 | 0.102623 / 16 | 0.116641 / 16 |
| detail:expert-down | nested | 23.407105 / 16 | 25.940889 / 16 | 21.551817 / 16 |
| EUP | nested | 0.680351 / 1 | 0.883110 / 1 | 1.149898 / 1 |
| experts-mix-normalize-up | boundary | 75.803727 / 1 | 77.054073 / 1 | 69.233861 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002775 / 1 | 0.002896 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000150 / 1 | 0.000281 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005480 / 1 | 0.005370 / 1 | 0.007854 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 983.112290 / 1 | 964.194734 / 1 | 858.874144 / 1 |
| op:Q   int8 projection | nested | 9.660290 / 1 | 13.606102 / 1 | 17.478809 / 1 |
| op:X   mxfp4 expert proj | nested | 74.688595 / 1 | 75.834051 / 1 | 67.764227 / 1 |
| op:N   rmsnorm | nested | 0.023014 / 1 | 0.022683 / 1 | 0.023182 / 1 |
| op:SiTU + sigma | nested | 0.121686 / 1 | 0.112291 / 1 | 0.128381 / 1 |
| op:AR  snapshot aggregate | nested | 0.030847 / 1 | 0.032040 / 1 | 0.032732 / 1 |
| op:SA  softmax attention | nested | 0.006973 / 1 | 0.009959 / 1 | 0.012534 / 1 |
| op:router dot product | nested | 0.561569 / 1 | 0.585644 / 1 | 0.959884 / 1 |
| op:top-k selection | nested | 0.002455 / 1 | 0.002395 / 1 | 0.002785 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 85.612836 / 1 | 90.581008 / 1 | 86.737156 / 1 |

### Layer 28

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.002995 / 1 | 0.003496 / 1 | 0.005430 / 1 |
| pre-attention-aggregation | boundary | 0.014678 / 1 | 0.015489 / 1 | 0.015349 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011332 / 1 | 0.010880 / 1 | 0.010980 / 1 |
| Q | nested | 2.822384 / 1 | 3.348466 / 1 | 4.754123 / 1 |
| K | nested | 2.721926 / 1 | 3.204948 / 1 | 3.519025 / 1 |
| V | nested | 2.670590 / 1 | 3.129808 / 1 | 2.952306 / 1 |
| B | nested | 0.025197 / 1 | 0.032301 / 1 | 0.092183 / 1 |
| FA | nested | 0.029335 / 1 | 0.033422 / 1 | 0.144460 / 1 |
| FB | nested | 0.062667 / 1 | 0.055373 / 1 | 0.066614 / 1 |
| G | nested | 2.353097 / 1 | 2.990518 / 1 | 3.516721 / 1 |
| O | nested | 2.404262 / 1 | 3.089452 / 1 | 3.557968 / 1 |
| attention | boundary | 14.141944 / 1 | 16.894387 / 1 | 19.318797 / 1 |
| attention-residual | boundary | 0.003727 / 1 | 0.003506 / 1 | 0.002084 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024896 / 1 | 0.025788 / 1 | 0.031499 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002174 / 1 | 0.002074 / 1 | 0.003386 / 1 |
| router-and-top16 | boundary | 0.587618 / 1 | 0.603608 / 1 | 0.772713 / 1 |
| EDOWN | nested | 0.700980 / 1 | 0.895924 / 1 | 1.076471 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.701781 / 1 | 0.896736 / 1 | 1.077493 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052699 / 1 | 0.012012 / 1 | 0.006482 / 1 |
| SH1 | nested | 1.182760 / 1 | 1.553402 / 1 | 1.916621 / 1 |
| SH3 | nested | 1.171499 / 1 | 1.548954 / 1 | 1.859134 / 1 |
| SH2 | nested | 1.196365 / 1 | 1.541651 / 1 | 1.725394 / 1 |
| shared-expert-during-read | boundary | 3.563979 / 1 | 4.657542 / 1 | 5.516407 / 1 |
| detail:expert-gate | nested | 26.477422 / 16 | 25.243486 / 16 | 25.247814 / 16 |
| detail:expert-up | nested | 25.262251 / 16 | 23.309303 / 16 | 24.494306 / 16 |
| detail:expert-activation | nested | 0.106336 / 16 | 0.102100 / 16 | 0.117840 / 16 |
| detail:expert-down | nested | 25.183237 / 16 | 24.466394 / 16 | 22.920680 / 16 |
| EUP | nested | 0.596194 / 1 | 0.591465 / 1 | 1.011009 / 1 |
| experts-mix-normalize-up | boundary | 78.048281 / 1 | 74.092921 / 1 | 74.158793 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002615 / 1 | 0.002425 / 1 | 0.002295 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000150 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.423702 / 1 | 0.407882 / 1 | 0.373548 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1024.299204 / 1 | 958.644155 / 1 | 931.028126 / 1 |
| op:Q   int8 projection | nested | 17.935663 / 1 | 22.013993 / 1 | 26.190445 / 1 |
| op:X   mxfp4 expert proj | nested | 77.072105 / 1 | 73.166629 / 1 | 72.821935 / 1 |
| op:N   rmsnorm | nested | 0.037019 / 1 | 0.032170 / 1 | 0.027132 / 1 |
| op:L   l2 per-head | nested | 0.005080 / 1 | 0.005480 / 1 | 0.006111 / 1 |
| op:SiTU + sigma | nested | 0.115916 / 1 | 0.111487 / 1 | 0.128961 / 1 |
| op:C   shortconv | nested | 0.053390 / 1 | 0.049052 / 1 | 0.037610 / 1 |
| op:AR  snapshot aggregate | nested | 0.029686 / 1 | 0.031599 / 1 | 0.032712 / 1 |
| op:D   kda delta-rule | nested | 0.118912 / 1 | 0.114674 / 1 | 0.148087 / 1 |
| op:router dot product | nested | 0.584833 / 1 | 0.600592 / 1 | 0.770019 / 1 |
| op:top-k selection | nested | 0.002425 / 1 | 0.002685 / 1 | 0.002254 / 1 |
| op:alpha / beta / gate | nested | 0.071824 / 1 | 0.073889 / 1 | 0.057698 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 97.593130 / 1 | 97.636691 / 1 | 101.304605 / 1 |

### Layer 29

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004739 / 1 | 0.004529 / 1 | 0.006252 / 1 |
| pre-attention-aggregation | boundary | 0.016982 / 1 | 0.015999 / 1 | 0.016101 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011011 / 1 | 0.011632 / 1 | 0.010820 / 1 |
| Q | nested | 2.506383 / 1 | 2.527362 / 1 | 5.481873 / 1 |
| K | nested | 2.420012 / 1 | 2.416766 / 1 | 3.363074 / 1 |
| V | nested | 2.268489 / 1 | 2.301972 / 1 | 2.716806 / 1 |
| B | nested | 0.020408 / 1 | 0.020689 / 1 | 0.156573 / 1 |
| FA | nested | 0.029094 / 1 | 0.027942 / 1 | 0.121497 / 1 |
| FB | nested | 0.063879 / 1 | 0.046166 / 1 | 0.054332 / 1 |
| G | nested | 1.982224 / 1 | 1.977746 / 1 | 3.250573 / 1 |
| O | nested | 2.064768 / 1 | 2.009285 / 1 | 3.319763 / 1 |
| attention | boundary | 12.369652 / 1 | 12.301725 / 1 | 19.216265 / 1 |
| attention-residual | boundary | 0.003437 / 1 | 0.002886 / 1 | 0.002635 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024295 / 1 | 0.025668 / 1 | 0.026820 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002114 / 1 | 0.002524 / 1 | 0.003196 / 1 |
| router-and-top16 | boundary | 0.614898 / 1 | 0.579924 / 1 | 0.791008 / 1 |
| EDOWN | nested | 0.616331 / 1 | 0.581176 / 1 | 1.133597 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.617483 / 1 | 0.582088 / 1 | 1.134680 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.075531 / 1 | 0.012263 / 1 | 0.006012 / 1 |
| SH1 | nested | 1.000489 / 1 | 0.999136 / 1 | 1.670922 / 1 |
| SH3 | nested | 0.996852 / 1 | 0.990461 / 1 | 1.696940 / 1 |
| SH2 | nested | 1.478192 / 1 | 0.984980 / 1 | 2.441722 / 1 |
| shared-expert-during-read | boundary | 3.487416 / 1 | 2.985860 / 1 | 5.821137 / 1 |
| detail:expert-gate | nested | 23.955141 / 16 | 25.556219 / 16 | 20.054503 / 16 |
| detail:expert-up | nested | 22.470376 / 16 | 25.750927 / 16 | 18.950305 / 16 |
| detail:expert-activation | nested | 0.098183 / 16 | 0.105008 / 16 | 0.096159 / 16 |
| detail:expert-down | nested | 23.769416 / 16 | 24.863645 / 16 | 19.951055 / 16 |
| EUP | nested | 0.710978 / 1 | 0.745573 / 1 | 0.969672 / 1 |
| experts-mix-normalize-up | boundary | 71.454941 / 1 | 77.396473 / 1 | 60.396689 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003146 / 1 | 0.002214 / 1 | 0.003988 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000140 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.403735 / 1 | 0.414745 / 1 | 0.364381 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 959.868648 / 1 | 1009.595140 / 1 | 824.645828 / 1 |
| op:Q   int8 projection | nested | 16.155998 / 1 | 15.627600 / 1 | 26.375643 / 1 |
| op:X   mxfp4 expert proj | nested | 70.337150 / 1 | 76.319351 / 1 | 59.093436 / 1 |
| op:N   rmsnorm | nested | 0.031870 / 1 | 0.034534 / 1 | 0.022923 / 1 |
| op:L   l2 per-head | nested | 0.005079 / 1 | 0.004719 / 1 | 0.005911 / 1 |
| op:SiTU + sigma | nested | 0.105448 / 1 | 0.112429 / 1 | 0.103422 / 1 |
| op:C   shortconv | nested | 0.047288 / 1 | 0.048812 / 1 | 0.035616 / 1 |
| op:AR  snapshot aggregate | nested | 0.031860 / 1 | 0.031749 / 1 | 0.032220 / 1 |
| op:D   kda delta-rule | nested | 0.114795 / 1 | 0.114675 / 1 | 0.213008 / 1 |
| op:router dot product | nested | 0.612284 / 1 | 0.576938 / 1 | 0.788183 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002494 / 1 | 0.002465 / 1 |
| op:alpha / beta / gate | nested | 0.072836 / 1 | 0.072325 / 1 | 0.056526 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.098908 / 1 | 94.346794 / 1 | 87.809569 / 1 |

### Layer 30

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005320 / 1 | 0.004929 / 1 | 0.006371 / 1 |
| pre-attention-aggregation | boundary | 0.017142 / 1 | 0.017502 / 1 | 0.016391 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011681 / 1 | 0.011592 / 1 | 0.010449 / 1 |
| Q | nested | 2.823335 / 1 | 2.535908 / 1 | 4.828581 / 1 |
| K | nested | 2.771599 / 1 | 2.393863 / 1 | 3.400604 / 1 |
| V | nested | 2.674337 / 1 | 2.257798 / 1 | 3.048787 / 1 |
| B | nested | 0.024796 / 1 | 0.022802 / 1 | 0.123882 / 1 |
| FA | nested | 0.027992 / 1 | 0.028453 / 1 | 0.164547 / 1 |
| FB | nested | 0.055504 / 1 | 0.046738 / 1 | 0.068058 / 1 |
| G | nested | 2.356644 / 1 | 1.934995 / 1 | 3.511861 / 1 |
| O | nested | 2.393432 / 1 | 2.019043 / 1 | 3.606519 / 1 |
| attention | boundary | 14.198500 / 1 | 12.328405 / 1 | 19.408925 / 1 |
| attention-residual | boundary | 0.002805 / 1 | 0.002455 / 1 | 0.002245 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025528 / 1 | 0.032441 / 1 | 0.028653 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002395 / 1 | 0.002554 / 1 | 0.003096 / 1 |
| router-and-top16 | boundary | 0.618636 / 1 | 0.592446 / 1 | 0.786359 / 1 |
| EDOWN | nested | 0.668759 / 1 | 0.593228 / 1 | 0.929727 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.669531 / 1 | 0.593980 / 1 | 0.930658 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078156 / 1 | 0.006121 / 1 | 0.013536 / 1 |
| SH1 | nested | 1.175877 / 1 | 0.991673 / 1 | 1.663127 / 1 |
| SH3 | nested | 1.168033 / 1 | 1.002884 / 1 | 1.689577 / 1 |
| SH2 | nested | 1.180696 / 1 | 0.979831 / 1 | 1.812006 / 1 |
| shared-expert-during-read | boundary | 3.539744 / 1 | 2.986320 / 1 | 5.179337 / 1 |
| detail:expert-gate | nested | 27.733178 / 16 | 25.960329 / 16 | 26.277540 / 16 |
| detail:expert-up | nested | 26.645369 / 16 | 25.634861 / 16 | 25.762596 / 16 |
| detail:expert-activation | nested | 0.102102 / 16 | 0.102893 / 16 | 0.116307 / 16 |
| detail:expert-down | nested | 27.790936 / 16 | 25.618715 / 16 | 24.421850 / 16 |
| EUP | nested | 0.594671 / 1 | 0.754440 / 1 | 1.146503 / 1 |
| experts-mix-normalize-up | boundary | 83.313860 / 1 | 78.442507 / 1 | 78.095028 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002434 / 1 | 0.002325 / 1 | 0.002775 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000220 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.434462 / 1 | 0.423421 / 1 | 0.366334 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1082.746080 / 1 | 1052.870511 / 1 | 987.742361 / 1 |
| op:Q   int8 projection | nested | 17.914233 / 1 | 15.560163 / 1 | 25.991954 / 1 |
| op:X   mxfp4 expert proj | nested | 82.317348 / 1 | 77.361719 / 1 | 76.622587 / 1 |
| op:N   rmsnorm | nested | 0.033613 / 1 | 0.037971 / 1 | 0.023443 / 1 |
| op:L   l2 per-head | nested | 0.005410 / 1 | 0.005079 / 1 | 0.006322 / 1 |
| op:SiTU + sigma | nested | 0.113445 / 1 | 0.110727 / 1 | 0.127180 / 1 |
| op:C   shortconv | nested | 0.057418 / 1 | 0.062607 / 1 | 0.042840 / 1 |
| op:AR  snapshot aggregate | nested | 0.032851 / 1 | 0.039924 / 1 | 0.034574 / 1 |
| op:D   kda delta-rule | nested | 0.127819 / 1 | 0.126386 / 1 | 0.224159 / 1 |
| op:router dot product | nested | 0.615740 / 1 | 0.589101 / 1 | 0.782823 / 1 |
| op:top-k selection | nested | 0.002454 / 1 | 0.002885 / 1 | 0.002865 / 1 |
| op:alpha / beta / gate | nested | 0.080641 / 1 | 0.073438 / 1 | 0.057818 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 102.931555 / 1 | 95.456076 / 1 | 104.859987 / 1 |

### Layer 31

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005220 / 1 | 0.005340 / 1 | 0.006392 / 1 |
| pre-attention-aggregation | boundary | 0.019095 / 1 | 0.019676 / 1 | 0.016871 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010931 / 1 | 0.010680 / 1 | 0.012744 / 1 |
| QA | nested | 0.298708 / 1 | 0.337120 / 1 | 0.552282 / 1 |
| QB | nested | 0.758236 / 1 | 0.830301 / 1 | 1.288267 / 1 |
| KA | nested | 0.100257 / 1 | 0.121267 / 1 | 0.216845 / 1 |
| KB | nested | 0.306353 / 1 | 0.353661 / 1 | 0.580274 / 1 |
| G | nested | 1.985700 / 1 | 2.496725 / 1 | 3.639089 / 1 |
| O | nested | 1.995238 / 1 | 2.538072 / 1 | 3.370808 / 1 |
| attention | boundary | 5.542455 / 1 | 6.786811 / 1 | 9.873658 / 1 |
| attention-residual | boundary | 0.002996 / 1 | 0.001693 / 1 | 0.002544 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.057828 / 1 | 0.025117 / 1 | 0.026239 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002184 / 1 | 0.002184 / 1 | 0.002956 / 1 |
| router-and-top16 | boundary | 0.578250 / 1 | 0.623866 / 1 | 0.759820 / 1 |
| EDOWN | nested | 0.587258 / 1 | 0.724994 / 1 | 1.007202 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.588159 / 1 | 0.726097 / 1 | 1.008144 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.068037 / 1 | 0.022883 / 1 | 0.006272 / 1 |
| SH1 | nested | 0.992575 / 1 | 1.208948 / 1 | 1.666193 / 1 |
| SH3 | nested | 1.003526 / 1 | 1.188531 / 1 | 2.089044 / 1 |
| SH2 | nested | 0.995921 / 1 | 1.198169 / 1 | 1.717869 / 1 |
| shared-expert-during-read | boundary | 3.003452 / 1 | 3.609764 / 1 | 5.493134 / 1 |
| detail:expert-gate | nested | 16.820530 / 16 | 23.928281 / 16 | 23.940733 / 16 |
| detail:expert-up | nested | 15.489090 / 16 | 23.871946 / 16 | 23.221229 / 16 |
| detail:expert-activation | nested | 0.098634 / 16 | 0.098564 / 16 | 0.117669 / 16 |
| detail:expert-down | nested | 16.567105 / 16 | 23.766154 / 16 | 21.686150 / 16 |
| EUP | nested | 0.604389 / 1 | 0.728481 / 1 | 0.981845 / 1 |
| experts-mix-normalize-up | boundary | 50.016866 / 1 | 72.800476 / 1 | 70.326763 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002474 / 1 | 0.002104 / 1 | 0.002675 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000241 / 1 | 0.000150 / 1 | 0.000491 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006763 / 1 | 0.007514 / 1 | 0.007333 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 688.630061 / 1 | 982.585440 / 1 | 876.378843 / 1 |
| op:Q   int8 projection | nested | 9.626718 / 1 | 11.725008 / 1 | 17.108186 / 1 |
| op:X   mxfp4 expert proj | nested | 49.019133 / 1 | 71.710783 / 1 | 69.009574 / 1 |
| op:N   rmsnorm | nested | 0.056797 / 1 | 0.024555 / 1 | 0.024426 / 1 |
| op:SiTU + sigma | nested | 0.105848 / 1 | 0.108753 / 1 | 0.128612 / 1 |
| op:AR  snapshot aggregate | nested | 0.033583 / 1 | 0.034224 / 1 | 0.032230 / 1 |
| op:SA  softmax attention | nested | 0.006653 / 1 | 0.009518 / 1 | 0.012743 / 1 |
| op:router dot product | nested | 0.573772 / 1 | 0.621150 / 1 | 0.756834 / 1 |
| op:top-k selection | nested | 0.003867 / 1 | 0.002375 / 1 | 0.002494 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 59.913516 / 1 | 84.652551 / 1 | 87.555174 / 1 |

### Layer 32

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004509 / 1 | 0.004358 / 1 | 0.006823 / 1 |
| pre-attention-aggregation | boundary | 0.014767 / 1 | 0.015559 / 1 | 0.015950 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010189 / 1 | 0.010419 / 1 | 0.010801 / 1 |
| Q | nested | 2.583598 / 1 | 2.871205 / 1 | 4.440647 / 1 |
| K | nested | 2.357955 / 1 | 2.868931 / 1 | 3.193467 / 1 |
| V | nested | 2.255785 / 1 | 2.751141 / 1 | 2.558581 / 1 |
| B | nested | 0.020939 / 1 | 0.029265 / 1 | 0.120545 / 1 |
| FA | nested | 0.026810 / 1 | 0.028463 / 1 | 0.142637 / 1 |
| FB | nested | 0.051256 / 1 | 0.053801 / 1 | 0.150671 / 1 |
| G | nested | 1.980360 / 1 | 2.385276 / 1 | 4.060277 / 1 |
| O | nested | 2.020225 / 1 | 2.476898 / 1 | 3.662283 / 1 |
| attention | boundary | 12.371425 / 1 | 14.583949 / 1 | 19.040436 / 1 |
| attention-residual | boundary | 0.002444 / 1 | 0.002214 / 1 | 0.001973 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026810 / 1 | 0.025187 / 1 | 0.028343 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001974 / 1 | 0.002535 / 1 | 0.003657 / 1 |
| router-and-top16 | boundary | 0.583230 / 1 | 0.643372 / 1 | 0.691352 / 1 |
| EDOWN | nested | 0.721207 / 1 | 0.724143 / 1 | 1.367175 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.722260 / 1 | 0.725336 / 1 | 1.368066 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.063669 / 1 | 0.027692 / 1 | 0.006553 / 1 |
| SH1 | nested | 1.266246 / 1 | 1.233255 / 1 | 2.550054 / 1 |
| SH3 | nested | 1.083875 / 1 | 1.220411 / 1 | 2.420823 / 1 |
| SH2 | nested | 1.276866 / 1 | 1.228336 / 1 | 2.041544 / 1 |
| shared-expert-during-read | boundary | 3.639300 / 1 | 3.696657 / 1 | 7.024715 / 1 |
| detail:expert-gate | nested | 9.987431 / 16 | 25.547433 / 16 | 21.244406 / 16 |
| detail:expert-up | nested | 10.095314 / 16 | 25.024908 / 16 | 20.093010 / 16 |
| detail:expert-activation | nested | 0.091533 / 16 | 0.115377 / 16 | 0.101440 / 16 |
| detail:expert-down | nested | 11.793235 / 16 | 22.731773 / 16 | 20.636297 / 16 |
| EUP | nested | 0.633724 / 1 | 0.712742 / 1 | 1.200173 / 1 |
| experts-mix-normalize-up | boundary | 33.025397 / 1 | 74.526901 / 1 | 63.613630 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002364 / 1 | 0.002224 / 1 | 0.003096 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000151 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.439802 / 1 | 0.425155 / 1 | 0.374901 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 449.910537 / 1 | 952.802919 / 1 | 838.060079 / 1 |
| op:Q   int8 projection | nested | 16.277183 / 1 | 18.582402 / 1 | 27.906792 / 1 |
| op:X   mxfp4 expert proj | nested | 32.011424 / 1 | 73.465308 / 1 | 62.119820 / 1 |
| op:N   rmsnorm | nested | 0.035185 / 1 | 0.029686 / 1 | 0.023132 / 1 |
| op:L   l2 per-head | nested | 0.004960 / 1 | 0.005490 / 1 | 0.005921 / 1 |
| op:SiTU + sigma | nested | 0.100147 / 1 | 0.125856 / 1 | 0.109446 / 1 |
| op:C   shortconv | nested | 0.054782 / 1 | 0.060122 / 1 | 0.036668 / 1 |
| op:AR  snapshot aggregate | nested | 0.031799 / 1 | 0.031289 / 1 | 0.032831 / 1 |
| op:D   kda delta-rule | nested | 0.138388 / 1 | 0.146213 / 1 | 0.144450 / 1 |
| op:router dot product | nested | 0.579864 / 1 | 0.640407 / 1 | 0.688246 / 1 |
| op:top-k selection | nested | 0.002976 / 1 | 0.002655 / 1 | 0.002564 / 1 |
| op:alpha / beta / gate | nested | 0.071503 / 1 | 0.074600 / 1 | 0.056636 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 50.917428 / 1 | 94.701106 / 1 | 92.200855 / 1 |

### Layer 33

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005520 / 1 | 0.005179 / 1 | 0.005571 / 1 |
| pre-attention-aggregation | boundary | 0.016822 / 1 | 0.020829 / 1 | 0.017773 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.009998 / 1 | 0.010600 / 1 | 0.011641 / 1 |
| Q | nested | 2.603003 / 1 | 2.803167 / 1 | 5.057049 / 1 |
| K | nested | 2.371020 / 1 | 2.704744 / 1 | 3.567176 / 1 |
| V | nested | 2.215469 / 1 | 2.577166 / 1 | 2.956124 / 1 |
| B | nested | 0.020979 / 1 | 0.023324 / 1 | 0.031960 / 1 |
| FA | nested | 0.028083 / 1 | 0.028383 / 1 | 0.039063 / 1 |
| FB | nested | 0.063499 / 1 | 0.053130 / 1 | 0.063789 / 1 |
| G | nested | 1.983806 / 1 | 2.235817 / 1 | 3.198736 / 1 |
| O | nested | 2.028430 / 1 | 2.309415 / 1 | 3.720491 / 1 |
| attention | boundary | 12.393556 / 1 | 15.066521 / 1 | 19.298849 / 1 |
| attention-residual | boundary | 0.002975 / 1 | 0.002224 / 1 | 0.002535 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024596 / 1 | 0.026429 / 1 | 0.027031 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002514 / 1 | 0.002435 / 1 | 0.002995 / 1 |
| router-and-top16 | boundary | 0.573412 / 1 | 0.631300 / 1 | 0.933675 / 1 |
| EDOWN | nested | 0.593519 / 1 | 0.678909 / 1 | 1.096499 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.594391 / 1 | 0.679900 / 1 | 1.097691 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.065362 / 1 | 0.012854 / 1 | 0.012834 / 1 |
| SH1 | nested | 0.995630 / 1 | 1.155169 / 1 | 1.922311 / 1 |
| SH3 | nested | 1.005418 / 1 | 1.146852 / 1 | 1.876776 / 1 |
| SH2 | nested | 0.986273 / 1 | 1.144628 / 1 | 1.776169 / 1 |
| shared-expert-during-read | boundary | 2.998673 / 1 | 3.460406 / 1 | 5.590606 / 1 |
| detail:expert-gate | nested | 24.488446 / 16 | 71.921314 / 16 | 24.654489 / 16 |
| detail:expert-up | nested | 23.186215 / 16 | 64.700992 / 16 | 23.939742 / 16 |
| detail:expert-activation | nested | 0.094878 / 16 | 0.114304 / 16 | 0.117509 / 16 |
| detail:expert-down | nested | 23.782438 / 16 | 89.724891 / 16 | 22.805299 / 16 |
| EUP | nested | 0.768325 / 1 | 0.730876 / 1 | 1.203548 / 1 |
| experts-mix-normalize-up | boundary | 72.753628 / 1 | 229.166281 / 1 | 73.103381 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002896 / 1 | 0.002825 / 1 | 0.002966 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000561 / 1 | 0.000151 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.439020 / 1 | 0.471310 / 1 | 0.367848 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 978.705521 / 1 | 2600.841129 / 1 | 926.940232 / 1 |
| op:Q   int8 projection | nested | 15.661792 / 1 | 17.590137 / 1 | 26.507498 / 1 |
| op:X   mxfp4 expert proj | nested | 71.599854 / 1 | 226.511040 / 1 | 71.563487 / 1 |
| op:N   rmsnorm | nested | 0.034034 / 1 | 0.034635 / 1 | 0.024976 / 1 |
| op:L   l2 per-head | nested | 0.004950 / 1 | 0.005841 / 1 | 0.006932 / 1 |
| op:SiTU + sigma | nested | 0.102533 / 1 | 0.123933 / 1 | 0.128529 / 1 |
| op:C   shortconv | nested | 0.054812 / 1 | 0.150871 / 1 | 0.043942 / 1 |
| op:AR  snapshot aggregate | nested | 0.031980 / 1 | 0.036938 / 1 | 0.034133 / 1 |
| op:D   kda delta-rule | nested | 0.131595 / 1 | 0.149019 / 1 | 0.225191 / 1 |
| op:router dot product | nested | 0.570145 / 1 | 0.628364 / 1 | 0.930969 / 1 |
| op:top-k selection | nested | 0.002926 / 1 | 0.002655 / 1 | 0.002174 / 1 |
| op:alpha / beta / gate | nested | 0.071724 / 1 | 0.073668 / 1 | 0.056656 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.894766 / 1 | 249.568401 / 1 | 100.485103 / 1 |

### Layer 34

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005390 / 1 | 0.005521 / 1 | 0.005490 / 1 |
| pre-attention-aggregation | boundary | 0.017613 / 1 | 0.024225 / 1 | 0.016000 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.013255 / 1 | 0.011321 / 1 |
| Q | nested | 3.263678 / 1 | 2.925907 / 1 | 4.771335 / 1 |
| K | nested | 2.908224 / 1 | 2.836129 / 1 | 3.533512 / 1 |
| V | nested | 2.763493 / 1 | 2.721274 / 1 | 2.974498 / 1 |
| B | nested | 0.029355 / 1 | 0.022592 / 1 | 0.117850 / 1 |
| FA | nested | 0.035777 / 1 | 0.029996 / 1 | 0.128590 / 1 |
| FB | nested | 0.064380 / 1 | 0.051296 / 1 | 0.062627 / 1 |
| G | nested | 2.678004 / 1 | 2.354640 / 1 | 3.520538 / 1 |
| O | nested | 2.706597 / 1 | 2.336356 / 1 | 3.614013 / 1 |
| attention | boundary | 15.519017 / 1 | 14.441844 / 1 | 19.416158 / 1 |
| attention-residual | boundary | 0.003126 / 1 | 0.002314 / 1 | 0.002465 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025478 / 1 | 0.026039 / 1 | 0.026510 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002635 / 1 | 0.002885 / 1 | 0.003707 / 1 |
| router-and-top16 | boundary | 0.596254 / 1 | 0.624416 / 1 | 0.776200 / 1 |
| EDOWN | nested | 0.791980 / 1 | 0.697052 / 1 | 1.033922 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.792901 / 1 | 0.697954 / 1 | 1.034873 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032190 / 1 | 0.006502 / 1 | 0.011662 / 1 |
| SH1 | nested | 1.382684 / 1 | 1.215812 / 1 | 1.885433 / 1 |
| SH3 | nested | 1.362015 / 1 | 1.161441 / 1 | 1.719503 / 1 |
| SH2 | nested | 1.443246 / 1 | 1.327901 / 1 | 1.774325 / 1 |
| shared-expert-during-read | boundary | 4.202072 / 1 | 3.719820 / 1 | 5.393668 / 1 |
| detail:expert-gate | nested | 12.693698 / 16 | 25.710170 / 16 | 24.263015 / 16 |
| detail:expert-up | nested | 12.652982 / 16 | 24.371487 / 16 | 23.563828 / 16 |
| detail:expert-activation | nested | 0.100678 / 16 | 0.103982 / 16 | 0.120177 / 16 |
| detail:expert-down | nested | 14.346084 / 16 | 24.289646 / 16 | 22.557838 / 16 |
| EUP | nested | 0.612645 / 1 | 0.605872 / 1 | 1.061714 / 1 |
| experts-mix-normalize-up | boundary | 40.793692 / 1 | 75.444095 / 1 | 71.950477 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002815 / 1 | 0.002765 / 1 | 0.002364 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000220 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.420877 / 1 | 0.430194 / 1 | 0.367447 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 404.323817 / 1 | 1006.416066 / 1 | 913.823457 / 1 |
| op:Q   int8 projection | nested | 20.040384 / 1 | 18.284305 / 1 | 26.196107 / 1 |
| op:X   mxfp4 expert proj | nested | 39.838336 / 1 | 74.525278 / 1 | 70.553628 / 1 |
| op:N   rmsnorm | nested | 0.034204 / 1 | 0.038604 / 1 | 0.024345 / 1 |
| op:L   l2 per-head | nested | 0.005290 / 1 | 0.005330 / 1 | 0.005560 / 1 |
| op:SiTU + sigma | nested | 0.110357 / 1 | 0.114242 / 1 | 0.130555 / 1 |
| op:C   shortconv | nested | 0.048822 / 1 | 0.061444 / 1 | 0.038372 / 1 |
| op:AR  snapshot aggregate | nested | 0.032922 / 1 | 0.040165 / 1 | 0.031810 / 1 |
| op:D   kda delta-rule | nested | 0.133249 / 1 | 0.152605 / 1 | 0.235911 / 1 |
| op:router dot product | nested | 0.592838 / 1 | 0.621331 / 1 | 0.773395 / 1 |
| op:top-k selection | nested | 0.002645 / 1 | 0.002755 / 1 | 0.002424 / 1 |
| op:alpha / beta / gate | nested | 0.074079 / 1 | 0.073938 / 1 | 0.057578 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 62.434587 / 1 | 95.451419 / 1 | 99.027970 / 1 |

### Layer 35

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005030 / 1 | 0.004900 / 1 | 0.006152 / 1 |
| pre-attention-aggregation | boundary | 0.016571 / 1 | 0.020428 / 1 | 0.017202 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010549 / 1 | 0.010369 / 1 | 0.010981 / 1 |
| QA | nested | 0.300051 / 1 | 0.293869 / 1 | 0.442336 / 1 |
| QB | nested | 0.747056 / 1 | 0.706941 / 1 | 1.143206 / 1 |
| KA | nested | 0.100517 / 1 | 0.103603 / 1 | 0.222396 / 1 |
| KB | nested | 0.445081 / 1 | 0.305882 / 1 | 0.686232 / 1 |
| G | nested | 1.964140 / 1 | 2.010086 / 1 | 3.413889 / 1 |
| O | nested | 2.135941 / 1 | 1.985490 / 1 | 3.507624 / 1 |
| attention | boundary | 5.790469 / 1 | 5.506980 / 1 | 9.529816 / 1 |
| attention-residual | boundary | 0.002705 / 1 | 0.002585 / 1 | 0.002194 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025688 / 1 | 0.025177 / 1 | 0.027381 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002494 / 1 | 0.002685 / 1 | 0.002895 / 1 |
| router-and-top16 | boundary | 0.577149 / 1 | 0.568362 / 1 | 0.964823 / 1 |
| EDOWN | nested | 0.567981 / 1 | 0.574964 / 1 | 1.058167 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.568823 / 1 | 0.575836 / 1 | 1.059350 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085259 / 1 | 0.012603 / 1 | 0.011592 / 1 |
| SH1 | nested | 0.967478 / 1 | 0.974100 / 1 | 1.695128 / 1 |
| SH3 | nested | 0.970464 / 1 | 0.963110 / 1 | 1.771189 / 1 |
| SH2 | nested | 0.979310 / 1 | 0.968971 / 1 | 1.588218 / 1 |
| shared-expert-during-read | boundary | 2.928842 / 1 | 2.917451 / 1 | 5.069523 / 1 |
| detail:expert-gate | nested | 12.242986 / 16 | 26.732561 / 16 | 25.389488 / 16 |
| detail:expert-up | nested | 12.174658 / 16 | 26.172875 / 16 | 26.834028 / 16 |
| detail:expert-activation | nested | 0.092530 / 16 | 0.114664 / 16 | 0.118690 / 16 |
| detail:expert-down | nested | 13.274633 / 16 | 25.180189 / 16 | 23.407947 / 16 |
| EUP | nested | 0.650035 / 1 | 0.581626 / 1 | 0.939886 / 1 |
| experts-mix-normalize-up | boundary | 38.899001 / 1 | 79.176128 / 1 | 77.041730 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003236 / 1 | 0.002285 / 1 | 0.002385 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000591 / 1 | 0.000130 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.012633 / 1 | 0.006382 / 1 | 0.007925 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 522.291324 / 1 | 1022.604542 / 1 | 972.981954 / 1 |
| op:Q   int8 projection | nested | 9.826803 / 1 | 9.467479 / 1 | 16.466588 / 1 |
| op:X   mxfp4 expert proj | nested | 37.830866 / 1 | 78.249840 / 1 | 75.798535 / 1 |
| op:N   rmsnorm | nested | 0.023714 / 1 | 0.023755 / 1 | 0.024245 / 1 |
| op:SiTU + sigma | nested | 0.099746 / 1 | 0.122119 / 1 | 0.129601 / 1 |
| op:AR  snapshot aggregate | nested | 0.031658 / 1 | 0.034916 / 1 | 0.033333 / 1 |
| op:SA  softmax attention | nested | 0.006783 / 1 | 0.009618 / 1 | 0.012002 / 1 |
| op:router dot product | nested | 0.574313 / 1 | 0.565226 / 1 | 0.961957 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002404 / 1 | 0.002605 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 48.937599 / 1 | 88.841308 / 1 | 93.763765 / 1 |

### Layer 36

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003677 / 1 | 0.003566 / 1 | 0.005971 / 1 |
| pre-attention-aggregation | boundary | 0.014637 / 1 | 0.015589 / 1 | 0.015750 / 1 |
| snapshot-push | boundary | 0.001172 / 1 | 0.001332 / 1 | 0.001322 / 1 |
| pre-attention-normalization | boundary | 0.010530 / 1 | 0.010069 / 1 | 0.010450 / 1 |
| Q | nested | 2.556367 / 1 | 2.473662 / 1 | 18.344326 / 1 |
| K | nested | 2.352215 / 1 | 2.342196 / 1 | 7.979780 / 1 |
| V | nested | 2.301190 / 1 | 2.381179 / 1 | 3.431482 / 1 |
| B | nested | 0.022042 / 1 | 0.025538 / 1 | 0.028313 / 1 |
| FA | nested | 0.027902 / 1 | 0.027501 / 1 | 0.146784 / 1 |
| FB | nested | 0.047389 / 1 | 0.049783 / 1 | 0.184364 / 1 |
| G | nested | 1.939053 / 1 | 1.904678 / 1 | 3.733766 / 1 |
| O | nested | 2.065219 / 1 | 2.091438 / 1 | 3.367242 / 1 |
| attention | boundary | 12.385412 / 1 | 12.394669 / 1 | 37.975035 / 1 |
| attention-residual | boundary | 0.002645 / 1 | 0.002334 / 1 | 0.002124 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024987 / 1 | 0.025338 / 1 | 0.027501 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002414 / 1 | 0.002334 / 1 | 0.002725 / 1 |
| router-and-top16 | boundary | 0.691182 / 1 | 0.876418 / 1 | 0.914168 / 1 |
| EDOWN | nested | 0.564825 / 1 | 0.566559 / 1 | 1.074397 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.565686 / 1 | 0.567491 / 1 | 1.076511 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.063027 / 1 | 0.012102 / 1 | 0.006112 / 1 |
| SH1 | nested | 1.443417 / 1 | 1.695368 / 1 | 1.861007 / 1 |
| SH3 | nested | 1.297234 / 1 | 1.292785 / 1 | 1.625277 / 1 |
| SH2 | nested | 1.529057 / 1 | 1.718300 / 1 | 1.883459 / 1 |
| shared-expert-during-read | boundary | 4.281049 / 1 | 4.718106 / 1 | 5.384551 / 1 |
| detail:expert-gate | nested | 23.812993 / 16 | 24.743138 / 16 | 24.289337 / 16 |
| detail:expert-up | nested | 22.792958 / 16 | 23.463729 / 16 | 23.695116 / 16 |
| detail:expert-activation | nested | 0.095027 / 16 | 0.100077 / 16 | 0.117691 / 16 |
| detail:expert-down | nested | 23.989032 / 16 | 23.777828 / 16 | 22.219715 / 16 |
| EUP | nested | 0.840600 / 1 | 1.138628 / 1 | 1.136654 / 1 |
| experts-mix-normalize-up | boundary | 71.982076 / 1 | 73.609999 / 1 | 71.857092 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003106 / 1 | 0.002815 / 1 | 0.002726 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000290 / 1 | 0.000150 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.450743 / 1 | 0.429112 / 1 | 0.381944 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 991.689218 / 1 | 975.289292 / 1 | 893.205680 / 1 |
| op:Q   int8 projection | nested | 16.985056 / 1 | 17.705952 / 1 | 44.795238 / 1 |
| op:X   mxfp4 expert proj | nested | 70.740054 / 1 | 72.134932 / 1 | 70.371861 / 1 |
| op:N   rmsnorm | nested | 0.033031 / 1 | 0.032511 / 1 | 0.022993 / 1 |
| op:L   l2 per-head | nested | 0.005290 / 1 | 0.004959 / 1 | 0.005571 / 1 |
| op:SiTU + sigma | nested | 0.102322 / 1 | 0.107421 / 1 | 0.128390 / 1 |
| op:C   shortconv | nested | 0.048531 / 1 | 0.051917 / 1 | 0.039634 / 1 |
| op:AR  snapshot aggregate | nested | 0.029905 / 1 | 0.030767 / 1 | 0.032381 / 1 |
| op:D   kda delta-rule | nested | 0.134892 / 1 | 0.136394 / 1 | 0.149068 / 1 |
| op:router dot product | nested | 0.688205 / 1 | 0.873342 / 1 | 0.911372 / 1 |
| op:top-k selection | nested | 0.002595 / 1 | 0.002405 / 1 | 0.002324 / 1 |
| op:alpha / beta / gate | nested | 0.075411 / 1 | 0.074038 / 1 | 0.057447 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 90.492042 / 1 | 92.681222 / 1 | 117.674100 / 1 |

### Layer 37

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004508 / 1 | 0.003848 / 1 | 0.006402 / 1 |
| pre-attention-aggregation | boundary | 0.016832 / 1 | 0.017021 / 1 | 0.016741 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011001 / 1 | 0.011191 / 1 | 0.011241 / 1 |
| Q | nested | 3.035562 / 1 | 3.322327 / 1 | 5.734475 / 1 |
| K | nested | 3.082379 / 1 | 2.875943 / 1 | 3.649008 / 1 |
| V | nested | 2.989306 / 1 | 2.939092 / 1 | 2.907833 / 1 |
| B | nested | 0.024867 / 1 | 0.028273 / 1 | 0.030316 / 1 |
| FA | nested | 0.033583 / 1 | 0.034625 / 1 | 0.141173 / 1 |
| FB | nested | 0.051166 / 1 | 0.062197 / 1 | 0.163696 / 1 |
| G | nested | 2.575753 / 1 | 2.565473 / 1 | 3.188768 / 1 |
| O | nested | 2.719511 / 1 | 2.385147 / 1 | 3.730170 / 1 |
| attention | boundary | 15.573639 / 1 | 15.303013 / 1 | 20.280864 / 1 |
| attention-residual | boundary | 0.003516 / 1 | 0.003757 / 1 | 0.001773 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024345 / 1 | 0.026920 / 1 | 0.029155 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002545 / 1 | 0.002054 / 1 | 0.104686 / 1 |
| router-and-top16 | boundary | 0.589351 / 1 | 0.654413 / 1 | 0.758016 / 1 |
| EDOWN | nested | 0.760451 / 1 | 0.826083 / 1 | 1.034633 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.761332 / 1 | 0.827035 / 1 | 1.035976 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.057267 / 1 | 0.027331 / 1 | 0.006342 / 1 |
| SH1 | nested | 1.513288 / 1 | 1.189643 / 1 | 1.791206 / 1 |
| SH3 | nested | 1.343430 / 1 | 1.351425 / 1 | 1.710055 / 1 |
| SH2 | nested | 1.882197 / 1 | 1.504992 / 1 | 1.696160 / 1 |
| shared-expert-during-read | boundary | 4.753041 / 1 | 4.061018 / 1 | 5.212410 / 1 |
| detail:expert-gate | nested | 25.858144 / 16 | 23.322559 / 16 | 25.326661 / 16 |
| detail:expert-up | nested | 24.868215 / 16 | 22.705347 / 16 | 24.508610 / 16 |
| detail:expert-activation | nested | 0.107500 / 16 | 0.108152 / 16 | 0.123210 / 16 |
| detail:expert-down | nested | 25.387256 / 16 | 23.234773 / 16 | 22.901522 / 16 |
| EUP | nested | 0.868242 / 1 | 0.591926 / 1 | 1.047087 / 1 |
| experts-mix-normalize-up | boundary | 77.523009 / 1 | 70.380975 / 1 | 74.280541 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002865 / 1 | 0.002565 / 1 | 0.002595 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000771 / 1 | 0.000151 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.497610 / 1 | 0.410678 / 1 | 0.374951 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 954.194517 / 1 | 945.709247 / 1 | 933.443836 / 1 |
| op:Q   int8 projection | nested | 20.877942 / 1 | 19.675264 / 1 | 26.823017 / 1 |
| op:X   mxfp4 expert proj | nested | 76.272704 / 1 | 69.422103 / 1 | 72.911503 / 1 |
| op:N   rmsnorm | nested | 0.034395 / 1 | 0.037761 / 1 | 0.024156 / 1 |
| op:L   l2 per-head | nested | 0.004969 / 1 | 0.005510 / 1 | 0.006051 / 1 |
| op:SiTU + sigma | nested | 0.117328 / 1 | 0.118430 / 1 | 0.134130 / 1 |
| op:C   shortconv | nested | 0.059241 / 1 | 0.058129 / 1 | 0.040305 / 1 |
| op:AR  snapshot aggregate | nested | 0.031789 / 1 | 0.034023 / 1 | 0.035056 / 1 |
| op:D   kda delta-rule | nested | 0.128089 / 1 | 0.137817 / 1 | 0.219841 / 1 |
| op:router dot product | nested | 0.586406 / 1 | 0.651558 / 1 | 0.755121 / 1 |
| op:top-k selection | nested | 0.002485 / 1 | 0.002545 / 1 | 0.002485 / 1 |
| op:alpha / beta / gate | nested | 0.073708 / 1 | 0.074259 / 1 | 0.057728 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 99.830551 / 1 | 91.741877 / 1 | 102.132250 / 1 |

### Layer 38

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005691 / 1 | 0.004959 / 1 | 0.006923 / 1 |
| pre-attention-aggregation | boundary | 0.019396 / 1 | 0.018184 / 1 | 0.017493 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012343 / 1 | 0.011441 / 1 | 0.010850 / 1 |
| Q | nested | 3.330232 / 1 | 2.513586 / 1 | 5.090131 / 1 |
| K | nested | 3.156829 / 1 | 2.417116 / 1 | 3.707688 / 1 |
| V | nested | 3.030854 / 1 | 2.300679 / 1 | 2.994105 / 1 |
| B | nested | 0.026860 / 1 | 0.023905 / 1 | 0.029155 / 1 |
| FA | nested | 0.033592 / 1 | 0.028013 / 1 | 0.037951 / 1 |
| FB | nested | 0.060262 / 1 | 0.065202 / 1 | 0.142517 / 1 |
| G | nested | 2.772339 / 1 | 1.960222 / 1 | 3.335162 / 1 |
| O | nested | 2.770146 / 1 | 2.044019 / 1 | 3.368184 / 1 |
| attention | boundary | 16.347745 / 1 | 12.436336 / 1 | 19.433861 / 1 |
| attention-residual | boundary | 0.002434 / 1 | 0.002464 / 1 | 0.002464 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027000 / 1 | 0.026058 / 1 | 0.032681 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.002535 / 1 | 0.003216 / 1 |
| router-and-top16 | boundary | 0.699708 / 1 | 0.581437 / 1 | 0.780288 / 1 |
| EDOWN | nested | 0.808341 / 1 | 0.601224 / 1 | 1.093784 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.809412 / 1 | 0.602185 / 1 | 1.094866 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041979 / 1 | 0.012915 / 1 | 0.012363 / 1 |
| SH1 | nested | 1.435422 / 1 | 1.009186 / 1 | 1.693694 / 1 |
| SH3 | nested | 1.405566 / 1 | 0.992605 / 1 | 1.793531 / 1 |
| SH2 | nested | 1.367175 / 1 | 0.993917 / 1 | 1.850939 / 1 |
| shared-expert-during-read | boundary | 4.222840 / 1 | 3.007569 / 1 | 5.353483 / 1 |
| detail:expert-gate | nested | 27.487811 / 16 | 25.182193 / 16 | 21.688595 / 16 |
| detail:expert-up | nested | 26.462554 / 16 | 24.673222 / 16 | 20.982502 / 16 |
| detail:expert-activation | nested | 0.104064 / 16 | 0.107794 / 16 | 0.119895 / 16 |
| detail:expert-down | nested | 27.824320 / 16 | 23.409572 / 16 | 19.425705 / 16 |
| EUP | nested | 0.648372 / 1 | 0.605752 / 1 | 0.999988 / 1 |
| experts-mix-normalize-up | boundary | 82.949438 / 1 | 74.367033 / 1 | 63.584205 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002364 / 1 | 0.002404 / 1 | 0.002545 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000220 / 1 | 0.000331 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.419423 / 1 | 0.368388 / 1 | 0.375742 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1037.701702 / 1 | 961.153764 / 1 | 771.689487 / 1 |
| op:Q   int8 projection | nested | 20.844325 / 1 | 15.553899 / 1 | 26.135193 / 1 |
| op:X   mxfp4 expert proj | nested | 81.938724 / 1 | 73.425495 / 1 | 62.267036 / 1 |
| op:N   rmsnorm | nested | 0.034485 / 1 | 0.032381 / 1 | 0.023555 / 1 |
| op:L   l2 per-head | nested | 0.005681 / 1 | 0.004809 / 1 | 0.006081 / 1 |
| op:SiTU + sigma | nested | 0.114694 / 1 | 0.115107 / 1 | 0.130235 / 1 |
| op:C   shortconv | nested | 0.062958 / 1 | 0.066184 / 1 | 0.041398 / 1 |
| op:AR  snapshot aggregate | nested | 0.036719 / 1 | 0.034155 / 1 | 0.035055 / 1 |
| op:D   kda delta-rule | nested | 0.180247 / 1 | 0.131536 / 1 | 0.215543 / 1 |
| op:router dot product | nested | 0.696462 / 1 | 0.577940 / 1 | 0.777793 / 1 |
| op:top-k selection | nested | 0.002495 / 1 | 0.002795 / 1 | 0.002134 / 1 |
| op:alpha / beta / gate | nested | 0.073567 / 1 | 0.073647 / 1 | 0.057398 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 105.572379 / 1 | 91.454189 / 1 | 90.721931 / 1 |

### Layer 39

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005300 / 1 | 0.005070 / 1 | 0.006341 / 1 |
| pre-attention-aggregation | boundary | 0.020118 / 1 | 0.016451 / 1 | 0.017272 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011822 / 1 | 0.009788 / 1 | 0.011201 / 1 |
| QA | nested | 0.303207 / 1 | 0.260857 / 1 | 0.500165 / 1 |
| QB | nested | 0.741956 / 1 | 0.696792 / 1 | 1.069949 / 1 |
| KA | nested | 0.107070 / 1 | 0.103343 / 1 | 0.231453 / 1 |
| KB | nested | 0.338442 / 1 | 0.309098 / 1 | 0.579543 / 1 |
| G | nested | 2.024483 / 1 | 2.099312 / 1 | 3.555683 / 1 |
| O | nested | 1.993865 / 1 | 2.095806 / 1 | 3.645552 / 1 |
| attention | boundary | 5.606475 / 1 | 5.661248 / 1 | 9.693262 / 1 |
| attention-residual | boundary | 0.002555 / 1 | 0.002595 / 1 | 0.005871 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025337 / 1 | 0.024275 / 1 | 0.030137 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.002044 / 1 | 0.002655 / 1 |
| router-and-top16 | boundary | 0.581978 / 1 | 0.617103 / 1 | 0.894652 / 1 |
| EDOWN | nested | 0.591165 / 1 | 0.609689 / 1 | 1.093003 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.592207 / 1 | 0.610671 / 1 | 1.093984 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.057247 / 1 | 0.022392 / 1 | 0.006132 / 1 |
| SH1 | nested | 1.002183 / 1 | 0.995961 / 1 | 1.794513 / 1 |
| SH3 | nested | 0.999478 / 1 | 1.009215 / 1 | 1.727668 / 1 |
| SH2 | nested | 1.018783 / 1 | 1.010007 / 1 | 1.766080 / 1 |
| shared-expert-during-read | boundary | 3.032035 / 1 | 3.026585 / 1 | 5.303039 / 1 |
| detail:expert-gate | nested | 14.047308 / 16 | 25.127210 / 16 | 16.123304 / 16 |
| detail:expert-up | nested | 13.387603 / 16 | 25.627425 / 16 | 15.498161 / 16 |
| detail:expert-activation | nested | 0.092292 / 16 | 0.103877 / 16 | 0.108634 / 16 |
| detail:expert-down | nested | 13.964502 / 16 | 23.430050 / 16 | 15.135942 / 16 |
| EUP | nested | 0.610551 / 1 | 0.705808 / 1 | 1.087201 / 1 |
| experts-mix-normalize-up | boundary | 42.534294 / 1 | 75.379855 / 1 | 48.315215 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002324 / 1 | 0.002454 / 1 | 0.002594 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000291 / 1 | 0.000331 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007634 / 1 | 0.006401 / 1 | 0.008295 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 591.692343 / 1 | 932.827858 / 1 | 581.580492 / 1 |
| op:Q   int8 projection | nested | 9.729980 / 1 | 9.894716 / 1 | 17.049156 / 1 |
| op:X   mxfp4 expert proj | nested | 41.541668 / 1 | 74.341636 / 1 | 46.916440 / 1 |
| op:N   rmsnorm | nested | 0.024948 / 1 | 0.022262 / 1 | 0.024416 / 1 |
| op:SiTU + sigma | nested | 0.100109 / 1 | 0.111442 / 1 | 0.119313 / 1 |
| op:AR  snapshot aggregate | nested | 0.035246 / 1 | 0.030797 / 1 | 0.035417 / 1 |
| op:SA  softmax attention | nested | 0.006683 / 1 | 0.008356 / 1 | 0.012503 / 1 |
| op:router dot product | nested | 0.578922 / 1 | 0.614347 / 1 | 0.890955 / 1 |
| op:top-k selection | nested | 0.002715 / 1 | 0.002404 / 1 | 0.003136 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 52.492120 / 1 | 85.396892 / 1 | 65.401761 / 1 |

### Layer 40

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004619 / 1 | 0.004047 / 1 | 0.006713 / 1 |
| pre-attention-aggregation | boundary | 0.015990 / 1 | 0.016221 / 1 | 0.016300 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.015139 / 1 | 0.010029 / 1 | 0.010219 / 1 |
| Q | nested | 2.601911 / 1 | 2.861496 / 1 | 5.541514 / 1 |
| K | nested | 2.447633 / 1 | 2.829036 / 1 | 3.370798 / 1 |
| V | nested | 2.258971 / 1 | 2.704954 / 1 | 2.481707 / 1 |
| B | nested | 0.025928 / 1 | 0.024226 / 1 | 0.025838 / 1 |
| FA | nested | 0.025789 / 1 | 0.028463 / 1 | 0.032481 / 1 |
| FB | nested | 0.061966 / 1 | 0.062246 / 1 | 0.140843 / 1 |
| G | nested | 1.923313 / 1 | 2.308824 / 1 | 3.174611 / 1 |
| O | nested | 2.037527 / 1 | 2.345503 / 1 | 3.137292 / 1 |
| attention | boundary | 12.450373 / 1 | 14.248814 / 1 | 18.725608 / 1 |
| attention-residual | boundary | 0.002435 / 1 | 0.002084 / 1 | 0.002224 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025628 / 1 | 0.026740 / 1 | 0.026310 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.002655 / 1 | 0.002915 / 1 |
| router-and-top16 | boundary | 0.588740 / 1 | 0.632832 / 1 | 0.741635 / 1 |
| EDOWN | nested | 0.595773 / 1 | 0.694538 / 1 | 0.946208 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.596745 / 1 | 0.695439 / 1 | 0.947230 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.049632 / 1 | 0.023343 / 1 | 0.006101 / 1 |
| SH1 | nested | 1.018904 / 1 | 1.196305 / 1 | 1.678746 / 1 |
| SH3 | nested | 1.012803 / 1 | 1.187168 / 1 | 1.572338 / 1 |
| SH2 | nested | 0.995870 / 1 | 1.184252 / 1 | 1.526111 / 1 |
| shared-expert-during-read | boundary | 3.039709 / 1 | 3.581982 / 1 | 4.789218 / 1 |
| detail:expert-gate | nested | 8.547261 / 16 | 23.612315 / 16 | 20.243376 / 16 |
| detail:expert-up | nested | 8.514177 / 16 | 22.964231 / 16 | 19.301493 / 16 |
| detail:expert-activation | nested | 0.091762 / 16 | 0.097864 / 16 | 0.095977 / 16 |
| detail:expert-down | nested | 9.648889 / 16 | 22.205802 / 16 | 20.275303 / 16 |
| EUP | nested | 0.605060 / 1 | 0.589461 / 1 | 1.050262 / 1 |
| experts-mix-normalize-up | boundary | 27.812166 / 1 | 69.860432 / 1 | 61.356492 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002385 / 1 | 0.002695 / 1 | 0.002675 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000351 / 1 | 0.000150 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.437027 / 1 | 0.435323 / 1 | 0.372145 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 390.444589 / 1 | 941.718469 / 1 | 834.902261 / 1 |
| op:Q   int8 projection | nested | 15.609914 / 1 | 18.015122 / 1 | 24.676958 / 1 |
| op:X   mxfp4 expert proj | nested | 26.852437 / 1 | 68.933650 / 1 | 59.968410 / 1 |
| op:N   rmsnorm | nested | 0.033924 / 1 | 0.040446 / 1 | 0.022652 / 1 |
| op:L   l2 per-head | nested | 0.005200 / 1 | 0.005090 / 1 | 0.005530 / 1 |
| op:SiTU + sigma | nested | 0.099615 / 1 | 0.107803 / 1 | 0.103442 / 1 |
| op:C   shortconv | nested | 0.050715 / 1 | 0.059001 / 1 | 0.037640 / 1 |
| op:AR  snapshot aggregate | nested | 0.032041 / 1 | 0.033372 / 1 | 0.032240 / 1 |
| op:D   kda delta-rule | nested | 0.141895 / 1 | 0.149760 / 1 | 0.144340 / 1 |
| op:router dot product | nested | 0.585574 / 1 | 0.630117 / 1 | 0.738880 / 1 |
| op:top-k selection | nested | 0.002785 / 1 | 0.002264 / 1 | 0.002344 / 1 |
| op:alpha / beta / gate | nested | 0.074058 / 1 | 0.071213 / 1 | 0.056676 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 45.053802 / 1 | 89.553638 / 1 | 87.016999 / 1 |

### Layer 41

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005039 / 1 | 0.005340 / 1 | 0.006081 / 1 |
| pre-attention-aggregation | boundary | 0.019135 / 1 | 0.023744 / 1 | 0.017172 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010800 / 1 | 0.012123 / 1 | 0.011121 / 1 |
| Q | nested | 3.135979 / 1 | 2.506824 / 1 | 5.075153 / 1 |
| K | nested | 2.952968 / 1 | 2.379326 / 1 | 3.337105 / 1 |
| V | nested | 2.672393 / 1 | 2.249724 / 1 | 2.830097 / 1 |
| B | nested | 0.023013 / 1 | 0.022842 / 1 | 0.028513 / 1 |
| FA | nested | 0.033372 / 1 | 0.029565 / 1 | 0.034835 / 1 |
| FB | nested | 0.054431 / 1 | 0.047299 / 1 | 0.149950 / 1 |
| G | nested | 2.636136 / 1 | 1.919286 / 1 | 3.383642 / 1 |
| O | nested | 2.624434 / 1 | 2.030975 / 1 | 3.382359 / 1 |
| attention | boundary | 15.205722 / 1 | 12.261921 / 1 | 18.885487 / 1 |
| attention-residual | boundary | 0.002916 / 1 | 0.002484 / 1 | 0.002033 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027010 / 1 | 0.026389 / 1 | 0.027922 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002424 / 1 | 0.002535 / 1 | 0.003176 / 1 |
| router-and-top16 | boundary | 0.590574 / 1 | 0.600542 / 1 | 0.726448 / 1 |
| EDOWN | nested | 0.771301 / 1 | 0.596555 / 1 | 1.060271 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.772192 / 1 | 0.597637 / 1 | 1.061584 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042980 / 1 | 0.018114 / 1 | 0.012623 / 1 |
| SH1 | nested | 1.293527 / 1 | 1.018022 / 1 | 1.689406 / 1 |
| SH3 | nested | 1.307312 / 1 | 1.000078 / 1 | 1.685029 / 1 |
| SH2 | nested | 1.345263 / 1 | 0.984069 / 1 | 1.934745 / 1 |
| shared-expert-during-read | boundary | 3.960510 / 1 | 3.013991 / 1 | 5.323757 / 1 |
| detail:expert-gate | nested | 23.734846 / 16 | 25.259134 / 16 | 25.988748 / 16 |
| detail:expert-up | nested | 23.163221 / 16 | 25.111609 / 16 | 27.111740 / 16 |
| detail:expert-activation | nested | 0.103053 / 16 | 0.108273 / 16 | 0.115237 / 16 |
| detail:expert-down | nested | 24.438894 / 16 | 24.811631 / 16 | 26.885342 / 16 |
| EUP | nested | 0.606974 / 1 | 0.597116 / 1 | 1.948020 / 1 |
| experts-mix-normalize-up | boundary | 72.458447 / 1 | 76.280738 / 1 | 82.587863 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002325 / 1 | 0.002305 / 1 | 0.002725 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000571 / 1 | 0.000150 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.438269 / 1 | 0.420866 / 1 | 0.367296 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 896.189313 / 1 | 996.554056 / 1 | 955.087341 / 1 |
| op:Q   int8 projection | nested | 19.455441 / 1 | 15.380127 / 1 | 26.537153 / 1 |
| op:X   mxfp4 expert proj | nested | 71.496237 / 1 | 75.347045 / 1 | 80.155567 / 1 |
| op:N   rmsnorm | nested | 0.038723 / 1 | 0.032851 / 1 | 0.023765 / 1 |
| op:L   l2 per-head | nested | 0.005240 / 1 | 0.004909 / 1 | 0.005881 / 1 |
| op:SiTU + sigma | nested | 0.113282 / 1 | 0.115985 / 1 | 0.125856 / 1 |
| op:C   shortconv | nested | 0.050795 / 1 | 0.063748 / 1 | 0.042740 / 1 |
| op:AR  snapshot aggregate | nested | 0.036138 / 1 | 0.040366 / 1 | 0.034054 / 1 |
| op:D   kda delta-rule | nested | 0.134171 / 1 | 0.136215 / 1 | 0.228918 / 1 |
| op:router dot product | nested | 0.587838 / 1 | 0.597336 / 1 | 0.723502 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002775 / 1 | 0.002625 / 1 |
| op:alpha / beta / gate | nested | 0.076403 / 1 | 0.073507 / 1 | 0.057237 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.549094 / 1 | 93.279520 / 1 | 109.045939 / 1 |

### Layer 42

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005230 / 1 | 0.005781 / 1 | 0.005631 / 1 |
| pre-attention-aggregation | boundary | 0.018495 / 1 | 0.025658 / 1 | 0.017282 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011692 / 1 | 0.011772 / 1 | 0.011371 / 1 |
| Q | nested | 2.479323 / 1 | 2.506854 / 1 | 5.730267 / 1 |
| K | nested | 2.400214 / 1 | 2.392069 / 1 | 3.303112 / 1 |
| V | nested | 2.267206 / 1 | 2.260494 / 1 | 2.789933 / 1 |
| B | nested | 0.022352 / 1 | 0.023093 / 1 | 0.027031 / 1 |
| FA | nested | 0.025488 / 1 | 0.032360 / 1 | 0.033933 / 1 |
| FB | nested | 0.064310 / 1 | 0.064660 / 1 | 0.156923 / 1 |
| G | nested | 1.974299 / 1 | 1.988054 / 1 | 5.080002 / 1 |
| O | nested | 2.040543 / 1 | 2.032157 / 1 | 4.475683 / 1 |
| attention | boundary | 12.336010 / 1 | 12.410428 / 1 | 22.255945 / 1 |
| attention-residual | boundary | 0.002775 / 1 | 0.002384 / 1 | 0.001954 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.030897 / 1 | 0.041748 / 1 | 0.029035 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003256 / 1 | 0.002645 / 1 | 0.003166 / 1 |
| router-and-top16 | boundary | 0.571908 / 1 | 0.592767 / 1 | 1.293646 / 1 |
| EDOWN | nested | 0.592888 / 1 | 0.609729 / 1 | 1.507757 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.593920 / 1 | 0.610801 / 1 | 1.508899 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.069640 / 1 | 0.012093 / 1 | 0.006492 / 1 |
| SH1 | nested | 1.014456 / 1 | 1.040654 / 1 | 2.273568 / 1 |
| SH3 | nested | 1.021909 / 1 | 1.022962 / 1 | 2.017770 / 1 |
| SH2 | nested | 1.004577 / 1 | 0.988157 / 1 | 2.205731 / 1 |
| shared-expert-during-read | boundary | 3.052414 / 1 | 3.063294 / 1 | 6.512308 / 1 |
| detail:expert-gate | nested | 16.362966 / 16 | 18.406612 / 16 | 28.582154 / 16 |
| detail:expert-up | nested | 15.451192 / 16 | 17.330122 / 16 | 26.661247 / 16 |
| detail:expert-activation | nested | 0.093456 / 16 | 0.101278 / 16 | 0.118015 / 16 |
| detail:expert-down | nested | 16.499689 / 16 | 17.630303 / 16 | 24.940490 / 16 |
| EUP | nested | 0.598058 / 1 | 0.665023 / 1 | 1.451472 / 1 |
| experts-mix-normalize-up | boundary | 49.444436 / 1 | 54.502667 / 1 | 82.143813 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002705 / 1 | 0.002745 / 1 | 0.002314 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000321 / 1 | 0.000471 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429683 / 1 | 0.360814 / 1 | 0.374009 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 690.851378 / 1 | 706.726235 / 1 | 1045.650613 / 1 |
| op:Q   int8 projection | nested | 15.504138 / 1 | 15.624924 / 1 | 31.051339 / 1 |
| op:X   mxfp4 expert proj | nested | 48.460407 / 1 | 53.522484 / 1 | 80.357627 / 1 |
| op:N   rmsnorm | nested | 0.033903 / 1 | 0.040616 / 1 | 0.025357 / 1 |
| op:L   l2 per-head | nested | 0.005009 / 1 | 0.005170 / 1 | 0.006031 / 1 |
| op:SiTU + sigma | nested | 0.100168 / 1 | 0.108672 / 1 | 0.128352 / 1 |
| op:C   shortconv | nested | 0.051586 / 1 | 0.064240 / 1 | 0.043301 / 1 |
| op:AR  snapshot aggregate | nested | 0.038812 / 1 | 0.057287 / 1 | 0.033953 / 1 |
| op:D   kda delta-rule | nested | 0.125936 / 1 | 0.133068 / 1 | 0.206004 / 1 |
| op:router dot product | nested | 0.568843 / 1 | 0.589912 / 1 | 1.290901 / 1 |
| op:top-k selection | nested | 0.002294 / 1 | 0.002444 / 1 | 0.002414 / 1 |
| op:alpha / beta / gate | nested | 0.074319 / 1 | 0.073708 / 1 | 0.057247 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 66.584411 / 1 | 71.657069 / 1 | 114.177858 / 1 |

### Layer 43

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005270 / 1 | 0.004619 / 1 | 0.172994 / 1 |
| pre-attention-aggregation | boundary | 0.019787 / 1 | 0.016360 / 1 | 0.019016 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011311 / 1 | 0.009888 / 1 | 0.010410 / 1 |
| QA | nested | 0.307254 / 1 | 0.286475 / 1 | 0.744881 / 1 |
| QB | nested | 0.738861 / 1 | 0.741906 / 1 | 1.480486 / 1 |
| KA | nested | 0.117559 / 1 | 0.119894 / 1 | 0.397062 / 1 |
| KB | nested | 0.384629 / 1 | 0.326340 / 1 | 0.774167 / 1 |
| G | nested | 2.614075 / 1 | 2.358397 / 1 | 4.527700 / 1 |
| O | nested | 2.713721 / 1 | 2.457802 / 1 | 4.309442 / 1 |
| attention | boundary | 6.972959 / 1 | 6.389278 / 1 | 12.432469 / 1 |
| attention-residual | boundary | 0.002655 / 1 | 0.001693 / 1 | 0.001903 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024707 / 1 | 0.025447 / 1 | 0.026429 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002094 / 1 | 0.002264 / 1 | 0.002755 / 1 |
| router-and-top16 | boundary | 0.582669 / 1 | 0.610411 / 1 | 1.234797 / 1 |
| EDOWN | nested | 0.779436 / 1 | 0.667397 / 1 | 1.423019 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.780639 / 1 | 0.668448 / 1 | 1.424041 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032751 / 1 | 0.027271 / 1 | 0.007754 / 1 |
| SH1 | nested | 1.371622 / 1 | 1.174394 / 1 | 2.266796 / 1 |
| SH3 | nested | 1.341276 / 1 | 1.143276 / 1 | 2.341325 / 1 |
| SH2 | nested | 1.308736 / 1 | 1.156952 / 1 | 2.248822 / 1 |
| shared-expert-during-read | boundary | 4.035460 / 1 | 3.488749 / 1 | 6.873423 / 1 |
| detail:expert-gate | nested | 24.804398 / 16 | 22.030384 / 16 | 23.801111 / 16 |
| detail:expert-up | nested | 23.894296 / 16 | 21.957727 / 16 | 23.294766 / 16 |
| detail:expert-activation | nested | 0.105286 / 16 | 0.103439 / 16 | 0.118924 / 16 |
| detail:expert-down | nested | 25.419085 / 16 | 21.528541 / 16 | 21.645446 / 16 |
| EUP | nested | 0.594801 / 1 | 0.586556 / 1 | 1.286394 / 1 |
| experts-mix-normalize-up | boundary | 75.228192 / 1 | 66.633322 / 1 | 70.507090 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002836 / 1 | 0.002444 / 1 | 0.002394 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000150 / 1 | 0.000220 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007985 / 1 | 0.006271 / 1 | 0.007444 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 921.305672 / 1 | 866.029014 / 1 | 871.834965 / 1 |
| op:Q   int8 projection | nested | 12.270759 / 1 | 11.018218 / 1 | 21.798368 / 1 |
| op:X   mxfp4 expert proj | nested | 74.280831 / 1 | 65.676045 / 1 | 68.916459 / 1 |
| op:N   rmsnorm | nested | 0.024185 / 1 | 0.022761 / 1 | 0.023485 / 1 |
| op:SiTU + sigma | nested | 0.114874 / 1 | 0.113338 / 1 | 0.130497 / 1 |
| op:AR  snapshot aggregate | nested | 0.033954 / 1 | 0.031929 / 1 | 0.034594 / 1 |
| op:SA  softmax attention | nested | 0.006301 / 1 | 0.009127 / 1 | 0.013936 / 1 |
| op:router dot product | nested | 0.579112 / 1 | 0.607244 / 1 | 1.232132 / 1 |
| op:top-k selection | nested | 0.002795 / 1 | 0.002745 / 1 | 0.002204 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 87.720153 / 1 | 77.897359 / 1 | 92.734702 / 1 |

### Layer 44

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004779 / 1 | 0.004178 / 1 | 0.007274 / 1 |
| pre-attention-aggregation | boundary | 0.015008 / 1 | 0.015700 / 1 | 0.016110 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010099 / 1 | 0.010580 / 1 | 0.010781 / 1 |
| Q | nested | 2.529837 / 1 | 2.448305 / 1 | 6.893730 / 1 |
| K | nested | 2.428648 / 1 | 2.466518 / 1 | 3.666271 / 1 |
| V | nested | 2.276634 / 1 | 2.285000 / 1 | 2.890330 / 1 |
| B | nested | 0.027281 / 1 | 0.024787 / 1 | 0.132988 / 1 |
| FA | nested | 0.032982 / 1 | 0.037300 / 1 | 0.102181 / 1 |
| FB | nested | 0.059381 / 1 | 0.197399 / 1 | 0.210043 / 1 |
| G | nested | 1.946247 / 1 | 1.948631 / 1 | 4.325402 / 1 |
| O | nested | 2.046514 / 1 | 2.030484 / 1 | 4.314902 / 1 |
| attention | boundary | 12.472855 / 1 | 12.457838 / 1 | 23.261954 / 1 |
| attention-residual | boundary | 0.002645 / 1 | 0.002294 / 1 | 0.002445 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025928 / 1 | 0.026640 / 1 | 0.031829 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002375 / 1 | 0.002615 / 1 | 0.105658 / 1 |
| router-and-top16 | boundary | 0.633504 / 1 | 0.603638 / 1 | 1.186727 / 1 |
| EDOWN | nested | 0.683998 / 1 | 0.681103 / 1 | 1.472100 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.685281 / 1 | 0.682324 / 1 | 1.473563 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.037310 / 1 | 0.017563 / 1 | 0.006122 / 1 |
| SH1 | nested | 1.245056 / 1 | 1.431405 / 1 | 2.133075 / 1 |
| SH3 | nested | 1.133368 / 1 | 1.637109 / 1 | 2.169484 / 1 |
| SH2 | nested | 0.990981 / 1 | 2.157702 / 1 | 2.502907 / 1 |
| shared-expert-during-read | boundary | 3.381728 / 1 | 5.238278 / 1 | 6.820895 / 1 |
| detail:expert-gate | nested | 23.075455 / 16 | 24.420850 / 16 | 26.447339 / 16 |
| detail:expert-up | nested | 21.734459 / 16 | 22.740509 / 16 | 26.114293 / 16 |
| detail:expert-activation | nested | 0.095228 / 16 | 0.111998 / 16 | 0.115205 / 16 |
| detail:expert-down | nested | 22.677567 / 16 | 22.803355 / 16 | 24.492142 / 16 |
| EUP | nested | 0.812919 / 1 | 0.875205 / 1 | 1.239946 / 1 |
| experts-mix-normalize-up | boundary | 68.812955 / 1 | 71.349605 / 1 | 78.791169 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002796 / 1 | 0.002605 / 1 | 0.002875 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000140 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.441445 / 1 | 0.354692 / 1 | 0.367887 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 910.499858 / 1 | 931.429339 / 1 | 999.240951 / 1 |
| op:Q   int8 projection | nested | 16.212454 / 1 | 18.219431 / 1 | 32.051107 / 1 |
| op:X   mxfp4 expert proj | nested | 67.641446 / 1 | 70.135848 / 1 | 77.226687 / 1 |
| op:N   rmsnorm | nested | 0.036057 / 1 | 0.035446 / 1 | 0.023093 / 1 |
| op:L   l2 per-head | nested | 0.005761 / 1 | 0.005390 / 1 | 0.005871 / 1 |
| op:SiTU + sigma | nested | 0.102895 / 1 | 0.119453 / 1 | 0.126065 / 1 |
| op:C   shortconv | nested | 0.059261 / 1 | 0.054543 / 1 | 0.036538 / 1 |
| op:AR  snapshot aggregate | nested | 0.031188 / 1 | 0.032801 / 1 | 0.033393 / 1 |
| op:D   kda delta-rule | nested | 0.142206 / 1 | 0.144039 / 1 | 0.145211 / 1 |
| op:router dot product | nested | 0.630638 / 1 | 0.600933 / 1 | 1.183972 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002385 / 1 | 0.002384 / 1 |
| op:alpha / beta / gate | nested | 0.073838 / 1 | 0.074970 / 1 | 0.057488 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 86.540478 / 1 | 90.780030 / 1 | 112.099945 / 1 |

### Layer 45

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005230 / 1 | 0.004859 / 1 | 0.006252 / 1 |
| pre-attention-aggregation | boundary | 0.018504 / 1 | 0.016721 / 1 | 0.017663 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011421 / 1 | 0.010059 / 1 | 0.011431 / 1 |
| Q | nested | 3.091206 / 1 | 2.843854 / 1 | 6.675813 / 1 |
| K | nested | 3.048867 / 1 | 2.852200 / 1 | 3.783039 / 1 |
| V | nested | 2.946234 / 1 | 2.699735 / 1 | 2.997901 / 1 |
| B | nested | 0.028103 / 1 | 0.024687 / 1 | 0.329746 / 1 |
| FA | nested | 0.033012 / 1 | 0.029104 / 1 | 0.194875 / 1 |
| FB | nested | 0.050244 / 1 | 0.050945 / 1 | 0.271077 / 1 |
| G | nested | 2.628031 / 1 | 2.134608 / 1 | 4.539432 / 1 |
| O | nested | 2.590901 / 1 | 2.026938 / 1 | 4.563356 / 1 |
| attention | boundary | 15.535167 / 1 | 13.606854 / 1 | 24.399570 / 1 |
| attention-residual | boundary | 0.002585 / 1 | 0.001813 / 1 | 0.001804 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026660 / 1 | 0.026559 / 1 | 0.026670 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002424 / 1 | 0.002735 / 1 | 0.153247 / 1 |
| router-and-top16 | boundary | 0.605040 / 1 | 0.605732 / 1 | 1.214910 / 1 |
| EDOWN | nested | 0.798762 / 1 | 0.606603 / 1 | 1.473844 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.799774 / 1 | 0.607886 / 1 | 1.475176 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043081 / 1 | 0.012394 / 1 | 0.012513 / 1 |
| SH1 | nested | 1.346316 / 1 | 1.032289 / 1 | 2.296240 / 1 |
| SH3 | nested | 1.410205 / 1 | 0.983318 / 1 | 2.455699 / 1 |
| SH2 | nested | 1.349702 / 1 | 0.968499 / 1 | 2.535767 / 1 |
| shared-expert-during-read | boundary | 4.120860 / 1 | 2.996118 / 1 | 7.303316 / 1 |
| detail:expert-gate | nested | 18.818499 / 16 | 20.235650 / 16 | 25.246994 / 16 |
| detail:expert-up | nested | 18.333727 / 16 | 19.403442 / 16 | 24.612048 / 16 |
| detail:expert-activation | nested | 0.099034 / 16 | 0.095819 / 16 | 0.115645 / 16 |
| detail:expert-down | nested | 19.590706 / 16 | 20.732850 / 16 | 23.290776 / 16 |
| EUP | nested | 0.609538 / 1 | 0.591896 / 1 | 1.511163 / 1 |
| experts-mix-normalize-up | boundary | 57.880990 / 1 | 61.431303 / 1 | 75.189459 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002274 / 1 | 0.002594 / 1 | 0.002805 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000481 / 1 | 0.000151 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.423411 / 1 | 0.352458 / 1 | 0.370813 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 707.181805 / 1 | 831.998117 / 1 | 953.225049 / 1 |
| op:Q   int8 projection | nested | 19.929526 / 1 | 16.843282 / 1 | 33.626279 / 1 |
| op:X   mxfp4 expert proj | nested | 56.898530 / 1 | 60.526191 / 1 | 73.326610 / 1 |
| op:N   rmsnorm | nested | 0.033773 / 1 | 0.032851 / 1 | 0.023793 / 1 |
| op:L   l2 per-head | nested | 0.005080 / 1 | 0.005270 / 1 | 0.005901 / 1 |
| op:SiTU + sigma | nested | 0.109523 / 1 | 0.103333 / 1 | 0.126375 / 1 |
| op:C   shortconv | nested | 0.065562 / 1 | 0.057727 / 1 | 0.043071 / 1 |
| op:AR  snapshot aggregate | nested | 0.035166 / 1 | 0.033582 / 1 | 0.034185 / 1 |
| op:D   kda delta-rule | nested | 0.135864 / 1 | 0.206055 / 1 | 0.222827 / 1 |
| op:router dot product | nested | 0.602425 / 1 | 0.602887 / 1 | 1.212084 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002494 / 1 | 0.002345 / 1 |
| op:alpha / beta / gate | nested | 0.073828 / 1 | 0.076102 / 1 | 0.057618 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 79.488923 / 1 | 79.689397 / 1 | 110.198282 / 1 |

### Layer 46

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005189 / 1 | 0.004338 / 1 | 0.006833 / 1 |
| pre-attention-aggregation | boundary | 0.022573 / 1 | 0.016971 / 1 | 0.019416 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011111 / 1 | 0.010459 / 1 | 0.011391 / 1 |
| Q | nested | 2.494741 / 1 | 2.467060 / 1 | 6.289712 / 1 |
| K | nested | 2.391679 / 1 | 2.507926 / 1 | 3.586601 / 1 |
| V | nested | 2.271073 / 1 | 2.310858 / 1 | 3.149835 / 1 |
| B | nested | 0.024315 / 1 | 0.023925 / 1 | 0.144430 / 1 |
| FA | nested | 0.025647 / 1 | 0.025718 / 1 | 0.137908 / 1 |
| FB | nested | 0.066424 / 1 | 0.045004 / 1 | 0.197189 / 1 |
| G | nested | 1.960894 / 1 | 1.994758 / 1 | 4.770483 / 1 |
| O | nested | 2.014554 / 1 | 2.006209 / 1 | 4.325181 / 1 |
| attention | boundary | 12.343955 / 1 | 12.303158 / 1 | 23.341243 / 1 |
| attention-residual | boundary | 0.002605 / 1 | 0.002305 / 1 | 0.002144 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025849 / 1 | 0.027050 / 1 | 0.028833 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002695 / 1 | 0.002735 / 1 | 0.140873 / 1 |
| router-and-top16 | boundary | 0.595904 / 1 | 0.599471 / 1 | 1.185986 / 1 |
| EDOWN | nested | 0.598368 / 1 | 0.583961 / 1 | 1.386691 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.599410 / 1 | 0.585224 / 1 | 1.388033 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.058850 / 1 | 0.013305 / 1 | 0.006172 / 1 |
| SH1 | nested | 1.004296 / 1 | 0.981003 / 1 | 2.342737 / 1 |
| SH3 | nested | 1.028582 / 1 | 1.006321 / 1 | 2.187057 / 1 |
| SH2 | nested | 0.995069 / 1 | 0.998286 / 1 | 2.355220 / 1 |
| shared-expert-during-read | boundary | 3.040481 / 1 | 2.997150 / 1 | 6.900815 / 1 |
| detail:expert-gate | nested | 16.561684 / 16 | 21.350559 / 16 | 26.501687 / 16 |
| detail:expert-up | nested | 15.617991 / 16 | 20.936359 / 16 | 25.722070 / 16 |
| detail:expert-activation | nested | 0.093514 / 16 | 0.116005 / 16 | 0.121889 / 16 |
| detail:expert-down | nested | 16.857676 / 16 | 19.803294 / 16 | 24.671328 / 16 |
| EUP | nested | 0.592798 / 1 | 0.590073 / 1 | 1.467441 / 1 |
| experts-mix-normalize-up | boundary | 50.167967 / 1 | 63.171965 / 1 | 78.899582 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002675 / 1 | 0.002464 / 1 | 0.002244 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000521 / 1 | 0.000230 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427389 / 1 | 0.351517 / 1 | 0.402041 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 688.789148 / 1 | 797.491829 / 1 | 1000.550020 / 1 |
| op:Q   int8 projection | nested | 15.466908 / 1 | 15.539746 / 1 | 32.338862 / 1 |
| op:X   mxfp4 expert proj | nested | 49.188536 / 1 | 62.266642 / 1 | 77.079422 / 1 |
| op:N   rmsnorm | nested | 0.032289 / 1 | 0.033363 / 1 | 0.024566 / 1 |
| op:L   l2 per-head | nested | 0.005210 / 1 | 0.005100 / 1 | 0.005711 / 1 |
| op:SiTU + sigma | nested | 0.101499 / 1 | 0.123040 / 1 | 0.132710 / 1 |
| op:C   shortconv | nested | 0.073627 / 1 | 0.063288 / 1 | 0.043332 / 1 |
| op:AR  snapshot aggregate | nested | 0.039033 / 1 | 0.034004 / 1 | 0.037470 / 1 |
| op:D   kda delta-rule | nested | 0.146814 / 1 | 0.203831 / 1 | 0.218599 / 1 |
| op:router dot product | nested | 0.592967 / 1 | 0.596855 / 1 | 1.182901 / 1 |
| op:top-k selection | nested | 0.002615 / 1 | 0.002294 / 1 | 0.002555 / 1 |
| op:alpha / beta / gate | nested | 0.072806 / 1 | 0.074609 / 1 | 0.057377 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 67.320817 / 1 | 80.099854 / 1 | 112.349331 / 1 |

### Layer 47

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005060 / 1 | 0.004749 / 1 | 0.087263 / 1 |
| pre-attention-aggregation | boundary | 0.022261 / 1 | 0.017092 / 1 | 0.026880 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011271 / 1 | 0.009868 / 1 | 0.012834 / 1 |
| QA | nested | 0.340616 / 1 | 0.264284 / 1 | 0.686263 / 1 |
| QB | nested | 0.835641 / 1 | 0.718533 / 1 | 1.469656 / 1 |
| KA | nested | 0.118963 / 1 | 0.100729 / 1 | 0.396541 / 1 |
| KB | nested | 0.346678 / 1 | 0.317553 / 1 | 0.814252 / 1 |
| G | nested | 2.468302 / 1 | 2.049148 / 1 | 3.558919 / 1 |
| O | nested | 2.608334 / 1 | 2.070509 / 1 | 4.360658 / 1 |
| attention | boundary | 6.825484 / 1 | 5.619670 / 1 | 11.409659 / 1 |
| attention-residual | boundary | 0.002745 / 1 | 0.001783 / 1 | 0.002405 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025237 / 1 | 0.026069 / 1 | 0.026359 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002324 / 1 | 0.002064 / 1 | 0.003336 / 1 |
| router-and-top16 | boundary | 0.583490 / 1 | 0.574704 / 1 | 1.201264 / 1 |
| EDOWN | nested | 0.735825 / 1 | 0.606623 / 1 | 1.405757 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.736777 / 1 | 0.607926 / 1 | 1.406989 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.053591 / 1 | 0.022382 / 1 | 0.006322 / 1 |
| SH1 | nested | 1.260395 / 1 | 0.976284 / 1 | 2.378474 / 1 |
| SH3 | nested | 1.242822 / 1 | 0.972126 / 1 | 2.568198 / 1 |
| SH2 | nested | 1.298556 / 1 | 1.009256 / 1 | 2.401367 / 1 |
| shared-expert-during-read | boundary | 3.816661 / 1 | 2.970090 / 1 | 7.363979 / 1 |
| detail:expert-gate | nested | 23.591789 / 16 | 22.143003 / 16 | 27.461380 / 16 |
| detail:expert-up | nested | 22.851488 / 16 | 21.373395 / 16 | 27.107619 / 16 |
| detail:expert-activation | nested | 0.102401 / 16 | 0.100617 / 16 | 0.119492 / 16 |
| detail:expert-down | nested | 24.695551 / 16 | 21.889241 / 16 | 25.633948 / 16 |
| EUP | nested | 0.600181 / 1 | 0.714926 / 1 | 1.238314 / 1 |
| experts-mix-normalize-up | boundary | 72.271316 / 1 | 66.677505 / 1 | 81.980107 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002535 / 1 | 0.002575 / 1 | 0.002825 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000380 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006773 / 1 | 0.007454 / 1 | 0.008386 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 880.733369 / 1 | 892.416577 / 1 | 1041.978368 / 1 |
| op:Q   int8 projection | nested | 11.854991 / 1 | 9.798838 / 1 | 21.276887 / 1 |
| op:X   mxfp4 expert proj | nested | 71.299851 / 1 | 65.566420 / 1 | 80.383123 / 1 |
| op:N   rmsnorm | nested | 0.024877 / 1 | 0.022902 / 1 | 0.025047 / 1 |
| op:SiTU + sigma | nested | 0.112961 / 1 | 0.108361 / 1 | 0.130443 / 1 |
| op:AR  snapshot aggregate | nested | 0.037059 / 1 | 0.032891 / 1 | 0.042731 / 1 |
| op:SA  softmax attention | nested | 0.006913 / 1 | 0.008977 / 1 | 0.012233 / 1 |
| op:router dot product | nested | 0.580735 / 1 | 0.572230 / 1 | 1.198820 / 1 |
| op:top-k selection | nested | 0.002404 / 1 | 0.002094 / 1 | 0.001984 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 84.376766 / 1 | 76.555502 / 1 | 103.553266 / 1 |

### Layer 48

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004298 / 1 | 0.004669 / 1 | 0.146263 / 1 |
| pre-attention-aggregation | boundary | 0.015770 / 1 | 0.016060 / 1 | 0.018305 / 1 |
| snapshot-push | boundary | 0.001633 / 1 | 0.001522 / 1 | 0.001663 / 1 |
| pre-attention-normalization | boundary | 0.010229 / 1 | 0.010660 / 1 | 0.010810 / 1 |
| Q | nested | 2.495282 / 1 | 2.845857 / 1 | 6.085160 / 1 |
| K | nested | 2.413991 / 1 | 2.792808 / 1 | 3.900959 / 1 |
| V | nested | 2.265633 / 1 | 2.640163 / 1 | 2.955372 / 1 |
| B | nested | 0.020529 / 1 | 0.021590 / 1 | 0.029926 / 1 |
| FA | nested | 0.033383 / 1 | 0.027040 / 1 | 0.124091 / 1 |
| FB | nested | 0.047940 / 1 | 0.049542 / 1 | 0.165469 / 1 |
| G | nested | 1.952488 / 1 | 2.169113 / 1 | 4.661860 / 1 |
| O | nested | 2.283327 / 1 | 2.262418 / 1 | 4.232849 / 1 |
| attention | boundary | 12.648393 / 1 | 13.927112 / 1 | 22.933010 / 1 |
| attention-residual | boundary | 0.001914 / 1 | 0.002104 / 1 | 0.002464 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027582 / 1 | 0.027582 / 1 | 0.029535 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002746 / 1 | 0.002214 / 1 | 0.119864 / 1 |
| router-and-top16 | boundary | 0.612915 / 1 | 0.627322 / 1 | 1.206825 / 1 |
| EDOWN | nested | 0.700829 / 1 | 0.671615 / 1 | 1.346957 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.701891 / 1 | 0.672707 / 1 | 1.348219 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042118 / 1 | 0.012193 / 1 | 0.006362 / 1 |
| SH1 | nested | 1.020126 / 1 | 1.182940 / 1 | 2.217313 / 1 |
| SH3 | nested | 0.993686 / 1 | 1.164085 / 1 | 2.454837 / 1 |
| SH2 | nested | 0.988767 / 1 | 1.173753 / 1 | 2.355461 / 1 |
| shared-expert-during-read | boundary | 3.015304 / 1 | 3.535766 / 1 | 7.043250 / 1 |
| detail:expert-gate | nested | 27.672605 / 16 | 26.846381 / 16 | 26.194396 / 16 |
| detail:expert-up | nested | 26.264665 / 16 | 25.026273 / 16 | 25.685694 / 16 |
| detail:expert-activation | nested | 0.109625 / 16 | 0.101417 / 16 | 0.115686 / 16 |
| detail:expert-down | nested | 28.260904 / 16 | 25.435314 / 16 | 24.759183 / 16 |
| EUP | nested | 0.792802 / 1 | 0.598719 / 1 | 1.469696 / 1 |
| experts-mix-normalize-up | boundary | 83.539241 / 1 | 78.416288 / 1 | 78.599571 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002454 / 1 | 0.002845 / 1 | 0.002856 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000320 / 1 | 0.000150 / 1 | 0.000370 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.451393 / 1 | 0.418913 / 1 | 0.366084 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1047.654969 / 1 | 1058.238404 / 1 | 996.697061 / 1 |
| op:Q   int8 projection | nested | 16.007209 / 1 | 17.598273 / 1 | 31.998210 / 1 |
| op:X   mxfp4 expert proj | nested | 82.371638 / 1 | 77.473649 / 1 | 76.821543 / 1 |
| op:N   rmsnorm | nested | 0.040385 / 1 | 0.034655 / 1 | 0.022262 / 1 |
| op:L   l2 per-head | nested | 0.005100 / 1 | 0.005190 / 1 | 0.006081 / 1 |
| op:SiTU + sigma | nested | 0.117530 / 1 | 0.111577 / 1 | 0.126407 / 1 |
| op:C   shortconv | nested | 0.057848 / 1 | 0.063509 / 1 | 0.039293 / 1 |
| op:AR  snapshot aggregate | nested | 0.032942 / 1 | 0.033322 / 1 | 0.037199 / 1 |
| op:D   kda delta-rule | nested | 0.143037 / 1 | 0.147034 / 1 | 0.148037 / 1 |
| op:router dot product | nested | 0.609760 / 1 | 0.624728 / 1 | 1.203999 / 1 |
| op:top-k selection | nested | 0.002655 / 1 | 0.002224 / 1 | 0.002375 / 1 |
| op:alpha / beta / gate | nested | 0.075471 / 1 | 0.074239 / 1 | 0.056676 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 101.090364 / 1 | 97.690903 / 1 | 111.848505 / 1 |

### Layer 49

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005160 / 1 | 0.011812 / 1 | 0.139470 / 1 |
| pre-attention-aggregation | boundary | 0.020619 / 1 | 0.025798 / 1 | 0.017362 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011631 / 1 | 0.012333 / 1 | 0.114695 / 1 |
| Q | nested | 2.528635 / 1 | 2.510661 / 1 | 5.845442 / 1 |
| K | nested | 2.426814 / 1 | 2.351564 / 1 | 3.361371 / 1 |
| V | nested | 2.253641 / 1 | 2.254342 / 1 | 3.178058 / 1 |
| B | nested | 0.022732 / 1 | 0.022011 / 1 | 0.126687 / 1 |
| FA | nested | 0.027090 / 1 | 0.026950 / 1 | 0.039815 / 1 |
| FB | nested | 0.059681 / 1 | 0.047499 / 1 | 0.280735 / 1 |
| G | nested | 1.931980 / 1 | 1.929946 / 1 | 4.861704 / 1 |
| O | nested | 2.043979 / 1 | 2.022549 / 1 | 4.415510 / 1 |
| attention | boundary | 12.370594 / 1 | 12.262832 / 1 | 22.758945 / 1 |
| attention-residual | boundary | 0.002274 / 1 | 0.002164 / 1 | 0.002705 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026851 / 1 | 0.027171 / 1 | 0.028333 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002304 / 1 | 0.002745 / 1 | 0.102692 / 1 |
| router-and-top16 | boundary | 0.612084 / 1 | 0.597607 / 1 | 1.215431 / 1 |
| EDOWN | nested | 0.617954 / 1 | 0.599910 / 1 | 1.273239 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.619177 / 1 | 0.600983 / 1 | 1.274521 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052428 / 1 | 0.012483 / 1 | 0.012404 / 1 |
| SH1 | nested | 1.010538 / 1 | 0.997223 / 1 | 2.425121 / 1 |
| SH3 | nested | 1.215802 / 1 | 1.004477 / 1 | 2.484453 / 1 |
| SH2 | nested | 0.997183 / 1 | 0.975603 / 1 | 2.393633 / 1 |
| shared-expert-during-read | boundary | 3.236087 / 1 | 2.989526 / 1 | 7.319506 / 1 |
| detail:expert-gate | nested | 21.406213 / 16 | 21.946577 / 16 | 24.484519 / 16 |
| detail:expert-up | nested | 20.393023 / 16 | 20.080089 / 16 | 24.366819 / 16 |
| detail:expert-activation | nested | 0.095880 / 16 | 0.115882 / 16 | 0.119734 / 16 |
| detail:expert-down | nested | 21.843228 / 16 | 22.668465 / 16 | 21.547962 / 16 |
| EUP | nested | 0.803562 / 1 | 0.601304 / 1 | 1.342498 / 1 |
| experts-mix-normalize-up | boundary | 64.970526 / 1 | 65.811928 / 1 | 72.281155 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002315 / 1 | 0.002525 / 1 | 0.002444 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000150 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.434682 / 1 | 0.422379 / 1 | 0.367797 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 874.493103 / 1 | 776.339786 / 1 | 862.360425 / 1 |
| op:Q   int8 projection | nested | 15.938071 / 1 | 15.342337 / 1 | 32.026460 / 1 |
| op:X   mxfp4 expert proj | nested | 63.803565 / 1 | 64.873092 / 1 | 70.586658 / 1 |
| op:N   rmsnorm | nested | 0.032812 / 1 | 0.033402 / 1 | 0.024677 / 1 |
| op:L   l2 per-head | nested | 0.005300 / 1 | 0.009718 / 1 | 0.005981 / 1 |
| op:SiTU + sigma | nested | 0.103404 / 1 | 0.123413 / 1 | 0.130465 / 1 |
| op:C   shortconv | nested | 0.053620 / 1 | 0.054762 / 1 | 0.042009 / 1 |
| op:AR  snapshot aggregate | nested | 0.038021 / 1 | 0.043171 / 1 | 0.034665 / 1 |
| op:D   kda delta-rule | nested | 0.136785 / 1 | 0.149690 / 1 | 0.221574 / 1 |
| op:router dot product | nested | 0.609329 / 1 | 0.594721 / 1 | 1.212385 / 1 |
| op:top-k selection | nested | 0.002484 / 1 | 0.002454 / 1 | 0.002465 / 1 |
| op:alpha / beta / gate | nested | 0.073217 / 1 | 0.075922 / 1 | 0.057918 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 82.381828 / 1 | 82.795401 / 1 | 105.653169 / 1 |

### Layer 50

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005681 / 1 | 0.005330 / 1 | 0.006562 / 1 |
| pre-attention-aggregation | boundary | 0.022492 / 1 | 0.026630 / 1 | 0.017253 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011632 / 1 | 0.011231 / 1 | 0.014027 / 1 |
| Q | nested | 3.245635 / 1 | 2.520970 / 1 | 6.765061 / 1 |
| K | nested | 3.159754 / 1 | 2.376039 / 1 | 3.572806 / 1 |
| V | nested | 3.088411 / 1 | 2.241468 / 1 | 2.921338 / 1 |
| B | nested | 0.027893 / 1 | 0.022172 / 1 | 0.148808 / 1 |
| FA | nested | 0.032180 / 1 | 0.027602 / 1 | 0.208990 / 1 |
| FB | nested | 0.061184 / 1 | 0.045285 / 1 | 0.177872 / 1 |
| G | nested | 2.794872 / 1 | 1.951457 / 1 | 4.516378 / 1 |
| O | nested | 2.988715 / 1 | 2.035814 / 1 | 5.229781 / 1 |
| attention | boundary | 16.468361 / 1 | 12.296686 / 1 | 24.192533 / 1 |
| attention-residual | boundary | 0.002624 / 1 | 0.002354 / 1 | 0.002295 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028143 / 1 | 0.027561 / 1 | 0.029846 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002234 / 1 | 0.002665 / 1 | 0.153306 / 1 |
| router-and-top16 | boundary | 0.603207 / 1 | 0.594561 / 1 | 1.101087 / 1 |
| EDOWN | nested | 0.870506 / 1 | 0.594461 / 1 | 1.339453 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.871678 / 1 | 0.595703 / 1 | 1.340825 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041988 / 1 | 0.006252 / 1 | 0.011832 / 1 |
| SH1 | nested | 1.541360 / 1 | 0.989729 / 1 | 2.146301 / 1 |
| SH3 | nested | 1.578339 / 1 | 0.993767 / 1 | 2.419321 / 1 |
| SH2 | nested | 1.621830 / 1 | 0.980833 / 1 | 2.312581 / 1 |
| shared-expert-during-read | boundary | 4.757309 / 1 | 2.976922 / 1 | 6.894232 / 1 |
| detail:expert-gate | nested | 24.144365 / 16 | 21.579459 / 16 | 26.504303 / 16 |
| detail:expert-up | nested | 22.674735 / 16 | 21.746475 / 16 | 25.459641 / 16 |
| detail:expert-activation | nested | 0.096711 / 16 | 0.096559 / 16 | 0.153986 / 16 |
| detail:expert-down | nested | 24.080341 / 16 | 22.189150 / 16 | 24.472846 / 16 |
| EUP | nested | 0.595312 / 1 | 0.700719 / 1 | 1.284610 / 1 |
| experts-mix-normalize-up | boundary | 72.032120 / 1 | 66.699306 / 1 | 78.269313 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002324 / 1 | 0.002605 / 1 | 0.002414 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000461 / 1 | 0.000220 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.425525 / 1 | 0.443469 / 1 | 0.369190 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 919.401044 / 1 | 896.897930 / 1 | 984.512172 / 1 |
| op:Q   int8 projection | nested | 21.604286 / 1 | 15.478912 / 1 | 33.041154 / 1 |
| op:X   mxfp4 expert proj | nested | 71.060035 / 1 | 65.674507 / 1 | 76.658105 / 1 |
| op:N   rmsnorm | nested | 0.033343 / 1 | 0.036708 / 1 | 0.023654 / 1 |
| op:L   l2 per-head | nested | 0.004959 / 1 | 0.004859 / 1 | 0.006071 / 1 |
| op:SiTU + sigma | nested | 0.107338 / 1 | 0.104294 / 1 | 0.164637 / 1 |
| op:C   shortconv | nested | 0.067286 / 1 | 0.056045 / 1 | 0.043601 / 1 |
| op:AR  snapshot aggregate | nested | 0.040826 / 1 | 0.043471 / 1 | 0.035125 / 1 |
| op:D   kda delta-rule | nested | 0.128280 / 1 | 0.125815 / 1 | 0.216304 / 1 |
| op:router dot product | nested | 0.600532 / 1 | 0.591696 / 1 | 1.097931 / 1 |
| op:top-k selection | nested | 0.002224 / 1 | 0.002495 / 1 | 0.002645 / 1 |
| op:alpha / beta / gate | nested | 0.073418 / 1 | 0.073868 / 1 | 0.058129 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.288023 / 1 | 83.703989 / 1 | 112.417308 / 1 |

### Layer 51

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004889 / 1 | 0.005019 / 1 | 0.006092 / 1 |
| pre-attention-aggregation | boundary | 0.020919 / 1 | 0.025537 / 1 | 0.018154 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011252 / 1 | 0.011421 / 1 | 0.013024 / 1 |
| QA | nested | 0.291284 / 1 | 0.312654 / 1 | 0.797730 / 1 |
| QB | nested | 0.710768 / 1 | 0.858183 / 1 | 1.735934 / 1 |
| KA | nested | 0.104876 / 1 | 0.134672 / 1 | 0.362909 / 1 |
| KB | nested | 0.348040 / 1 | 0.422620 / 1 | 0.807869 / 1 |
| G | nested | 1.968698 / 1 | 2.342687 / 1 | 5.488445 / 1 |
| O | nested | 1.971464 / 1 | 2.383434 / 1 | 4.593062 / 1 |
| attention | boundary | 5.492172 / 1 | 6.554707 / 1 | 13.892938 / 1 |
| attention-residual | boundary | 0.002394 / 1 | 0.002836 / 1 | 0.002685 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026600 / 1 | 0.027131 / 1 | 0.029105 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002144 / 1 | 0.002144 / 1 | 0.002875 / 1 |
| router-and-top16 | boundary | 0.566629 / 1 | 0.613777 / 1 | 1.277797 / 1 |
| EDOWN | nested | 0.572940 / 1 | 0.683417 / 1 | 1.485506 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.574153 / 1 | 0.684840 / 1 | 1.486788 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032410 / 1 | 0.012784 / 1 | 0.006522 / 1 |
| SH1 | nested | 0.984189 / 1 | 1.164396 / 1 | 2.423950 / 1 |
| SH3 | nested | 0.976224 / 1 | 1.118349 / 1 | 2.922470 / 1 |
| SH2 | nested | 0.976425 / 1 | 1.111066 / 1 | 2.935946 / 1 |
| shared-expert-during-read | boundary | 2.949070 / 1 | 3.408759 / 1 | 8.303485 / 1 |
| detail:expert-gate | nested | 25.727601 / 16 | 24.599451 / 16 | 27.460979 / 16 |
| detail:expert-up | nested | 24.984874 / 16 | 23.258858 / 16 | 27.003907 / 16 |
| detail:expert-activation | nested | 0.102750 / 16 | 0.099346 / 16 | 0.112058 / 16 |
| detail:expert-down | nested | 24.438363 / 16 | 23.109261 / 16 | 29.359899 / 16 |
| EUP | nested | 0.601925 / 1 | 0.585594 / 1 | 1.406999 / 1 |
| experts-mix-normalize-up | boundary | 76.266472 / 1 | 72.042699 / 1 | 85.721778 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002274 / 1 | 0.002465 / 1 | 0.002355 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000260 / 1 | 0.000210 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.011932 / 1 | 0.007454 / 1 | 0.008416 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 997.226743 / 1 | 939.325758 / 1 | 1001.859408 / 1 |
| op:Q   int8 projection | nested | 9.505612 / 1 | 11.115909 / 1 | 24.959088 / 1 |
| op:X   mxfp4 expert proj | nested | 75.318800 / 1 | 71.130494 / 1 | 84.001172 / 1 |
| op:N   rmsnorm | nested | 0.024445 / 1 | 0.023656 / 1 | 0.025147 / 1 |
| op:SiTU + sigma | nested | 0.110464 / 1 | 0.109584 / 1 | 0.127817 / 1 |
| op:AR  snapshot aggregate | nested | 0.036568 / 1 | 0.042319 / 1 | 0.035577 / 1 |
| op:SA  softmax attention | nested | 0.006953 / 1 | 0.009478 / 1 | 0.013005 / 1 |
| op:router dot product | nested | 0.563854 / 1 | 0.610481 / 1 | 1.274642 / 1 |
| op:top-k selection | nested | 0.002364 / 1 | 0.002805 / 1 | 0.002735 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 85.976946 / 1 | 83.414768 / 1 | 110.785750 / 1 |

### Layer 52

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004949 / 1 | 0.004408 / 1 | 0.005941 / 1 |
| pre-attention-aggregation | boundary | 0.016040 / 1 | 0.016681 / 1 | 0.018654 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.009919 / 1 | 0.011010 / 1 | 0.011051 / 1 |
| Q | nested | 2.546889 / 1 | 2.499240 / 1 | 7.205864 / 1 |
| K | nested | 2.409301 / 1 | 2.378735 / 1 | 4.018870 / 1 |
| V | nested | 2.250315 / 1 | 2.275872 / 1 | 3.517652 / 1 |
| B | nested | 0.024476 / 1 | 0.025647 / 1 | 0.148478 / 1 |
| FA | nested | 0.026610 / 1 | 0.026570 / 1 | 0.160089 / 1 |
| FB | nested | 0.049823 / 1 | 0.050755 / 1 | 0.180487 / 1 |
| G | nested | 1.944454 / 1 | 1.922011 / 1 | 5.178716 / 1 |
| O | nested | 2.055010 / 1 | 2.017179 / 1 | 5.418525 / 1 |
| attention | boundary | 12.427250 / 1 | 12.322093 / 1 | 26.549166 / 1 |
| attention-residual | boundary | 0.002164 / 1 | 0.002334 / 1 | 0.002856 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.032140 / 1 | 0.026579 / 1 | 0.028383 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003637 / 1 | 0.002685 / 1 | 0.161091 / 1 |
| router-and-top16 | boundary | 0.602295 / 1 | 0.594480 / 1 | 1.346385 / 1 |
| EDOWN | nested | 0.598178 / 1 | 0.588730 / 1 | 1.750171 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.599320 / 1 | 0.589912 / 1 | 1.751623 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059441 / 1 | 0.017693 / 1 | 0.006281 / 1 |
| SH1 | nested | 1.028942 / 1 | 1.123329 / 1 | 2.585321 / 1 |
| SH3 | nested | 1.012281 / 1 | 1.123489 / 1 | 2.462070 / 1 |
| SH2 | nested | 1.038680 / 1 | 1.060332 / 1 | 2.510351 / 1 |
| shared-expert-during-read | boundary | 3.092779 / 1 | 3.322498 / 1 | 7.573661 / 1 |
| detail:expert-gate | nested | 27.414504 / 16 | 25.087274 / 16 | 21.050232 / 16 |
| detail:expert-up | nested | 26.413335 / 16 | 23.682750 / 16 | 21.244474 / 16 |
| detail:expert-activation | nested | 0.101420 / 16 | 0.108371 / 16 | 0.097182 / 16 |
| detail:expert-down | nested | 25.782595 / 16 | 23.457439 / 16 | 21.951645 / 16 |
| EUP | nested | 0.706150 / 1 | 0.685681 / 1 | 1.464757 / 1 |
| experts-mix-normalize-up | boundary | 80.868030 / 1 | 73.417679 / 1 | 66.186908 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002434 / 1 | 0.002675 / 1 | 0.002524 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000130 / 1 | 0.000280 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.424212 / 1 | 0.437608 / 1 | 0.364090 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1064.363207 / 1 | 941.777990 / 1 | 845.566734 / 1 |
| op:Q   int8 projection | nested | 15.689497 / 1 | 15.776187 / 1 | 36.599405 / 1 |
| op:X   mxfp4 expert proj | nested | 79.779817 / 1 | 72.402953 / 1 | 64.407292 / 1 |
| op:N   rmsnorm | nested | 0.039073 / 1 | 0.033343 / 1 | 0.023785 / 1 |
| op:L   l2 per-head | nested | 0.005310 / 1 | 0.006522 / 1 | 0.005841 / 1 |
| op:SiTU + sigma | nested | 0.109443 / 1 | 0.118811 / 1 | 0.108023 / 1 |
| op:C   shortconv | nested | 0.067205 / 1 | 0.056065 / 1 | 0.037490 / 1 |
| op:AR  snapshot aggregate | nested | 0.032981 / 1 | 0.033573 / 1 | 0.036298 / 1 |
| op:D   kda delta-rule | nested | 0.141193 / 1 | 0.143759 / 1 | 0.144570 / 1 |
| op:router dot product | nested | 0.599119 / 1 | 0.591175 / 1 | 1.343460 / 1 |
| op:top-k selection | nested | 0.002815 / 1 | 0.002905 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.073618 / 1 | 0.081041 / 1 | 0.057368 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 98.158376 / 1 | 90.781282 / 1 | 104.021741 / 1 |

### Layer 53

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005931 / 1 | 0.005640 / 1 | 0.116057 / 1 |
| pre-attention-aggregation | boundary | 0.020137 / 1 | 0.027531 / 1 | 0.017593 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011301 / 1 | 0.012423 / 1 | 0.010781 / 1 |
| Q | nested | 2.838113 / 1 | 2.836760 / 1 | 6.658882 / 1 |
| K | nested | 2.683554 / 1 | 2.780184 / 1 | 3.186534 / 1 |
| V | nested | 2.579370 / 1 | 2.326497 / 1 | 2.634733 / 1 |
| B | nested | 0.023544 / 1 | 0.021210 / 1 | 0.162023 / 1 |
| FA | nested | 0.028984 / 1 | 0.026820 / 1 | 0.158587 / 1 |
| FB | nested | 0.065964 / 1 | 0.068698 / 1 | 0.227145 / 1 |
| G | nested | 2.323051 / 1 | 1.934955 / 1 | 5.534701 / 1 |
| O | nested | 2.347816 / 1 | 2.019443 / 1 | 4.653044 / 1 |
| attention | boundary | 14.014896 / 1 | 13.138148 / 1 | 23.877214 / 1 |
| attention-residual | boundary | 0.002194 / 1 | 0.002174 / 1 | 0.002004 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027371 / 1 | 0.026850 / 1 | 0.028693 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002585 / 1 | 0.002875 / 1 | 0.003466 / 1 |
| router-and-top16 | boundary | 0.629867 / 1 | 0.607165 / 1 | 1.312613 / 1 |
| EDOWN | nested | 0.682275 / 1 | 0.592327 / 1 | 1.705367 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.683427 / 1 | 0.593689 / 1 | 1.706709 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048460 / 1 | 0.022772 / 1 | 0.006442 / 1 |
| SH1 | nested | 1.185806 / 1 | 1.041856 / 1 | 2.594087 / 1 |
| SH3 | nested | 1.173653 / 1 | 1.016189 / 1 | 3.055960 / 1 |
| SH2 | nested | 1.178732 / 1 | 1.012933 / 1 | 3.076789 / 1 |
| shared-expert-during-read | boundary | 3.553129 / 1 | 3.083521 / 1 | 8.742615 / 1 |
| detail:expert-gate | nested | 24.739728 / 16 | 25.183365 / 16 | 26.157985 / 16 |
| detail:expert-up | nested | 23.623980 / 16 | 23.269236 / 16 | 25.670102 / 16 |
| detail:expert-activation | nested | 0.094629 / 16 | 0.102793 / 16 | 0.114683 / 16 |
| detail:expert-down | nested | 25.245771 / 16 | 22.995005 / 16 | 29.586241 / 16 |
| EUP | nested | 0.803141 / 1 | 0.701891 / 1 | 1.463895 / 1 |
| experts-mix-normalize-up | boundary | 74.944351 / 1 | 72.656637 / 1 | 83.391865 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.002425 / 1 | 0.002675 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000130 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.445332 / 1 | 0.453037 / 1 | 0.369741 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 993.435045 / 1 | 967.573739 / 1 | 952.774407 / 1 |
| op:Q   int8 projection | nested | 17.912579 / 1 | 16.378290 / 1 | 35.109984 / 1 |
| op:X   mxfp4 expert proj | nested | 73.771301 / 1 | 71.617314 / 1 | 81.596739 / 1 |
| op:N   rmsnorm | nested | 0.032590 / 1 | 0.038061 / 1 | 0.025227 / 1 |
| op:L   l2 per-head | nested | 0.005641 / 1 | 0.005120 / 1 | 0.005640 / 1 |
| op:SiTU + sigma | nested | 0.104617 / 1 | 0.110627 / 1 | 0.125222 / 1 |
| op:C   shortconv | nested | 0.056776 / 1 | 0.055854 / 1 | 0.037931 / 1 |
| op:AR  snapshot aggregate | nested | 0.037129 / 1 | 0.044393 / 1 | 0.035306 / 1 |
| op:D   kda delta-rule | nested | 0.143669 / 1 | 0.136966 / 1 | 0.235139 / 1 |
| op:router dot product | nested | 0.626901 / 1 | 0.604269 / 1 | 1.309597 / 1 |
| op:top-k selection | nested | 0.002505 / 1 | 0.002454 / 1 | 0.002525 / 1 |
| op:alpha / beta / gate | nested | 0.072605 / 1 | 0.074198 / 1 | 0.057057 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.405104 / 1 | 90.648003 / 1 | 119.603265 / 1 |

### Layer 54

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005450 / 1 | 0.005450 / 1 | 0.009939 / 1 |
| pre-attention-aggregation | boundary | 0.023003 / 1 | 0.027852 / 1 | 0.018475 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012152 / 1 | 0.012143 / 1 | 0.011080 / 1 |
| Q | nested | 3.282954 / 1 | 2.828635 / 1 | 5.414987 / 1 |
| K | nested | 3.381318 / 1 | 2.783140 / 1 | 3.794470 / 1 |
| V | nested | 3.194939 / 1 | 2.673265 / 1 | 3.460215 / 1 |
| B | nested | 0.030087 / 1 | 0.024266 / 1 | 0.172252 / 1 |
| FA | nested | 0.035707 / 1 | 0.030076 / 1 | 0.259354 / 1 |
| FB | nested | 0.065953 / 1 | 0.055314 / 1 | 0.322192 / 1 |
| G | nested | 2.750068 / 1 | 2.301461 / 1 | 5.370224 / 1 |
| O | nested | 2.716505 / 1 | 2.663767 / 1 | 5.877251 / 1 |
| attention | boundary | 16.549212 / 1 | 14.459487 / 1 | 25.341970 / 1 |
| attention-residual | boundary | 0.002504 / 1 | 0.001934 / 1 | 0.002455 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027281 / 1 | 0.027191 / 1 | 0.028424 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002234 / 1 | 0.002364 / 1 | 0.176981 / 1 |
| router-and-top16 | boundary | 0.607726 / 1 | 0.636089 / 1 | 1.373576 / 1 |
| EDOWN | nested | 0.801808 / 1 | 0.695339 / 1 | 1.529909 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.803060 / 1 | 0.696732 / 1 | 1.531341 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043261 / 1 | 0.017573 / 1 | 0.006342 / 1 |
| SH1 | nested | 1.357427 / 1 | 1.179604 / 1 | 2.508897 / 1 |
| SH3 | nested | 1.417799 / 1 | 1.167802 / 1 | 3.199728 / 1 |
| SH2 | nested | 1.409103 / 1 | 1.187670 / 1 | 2.673806 / 1 |
| shared-expert-during-read | boundary | 4.199126 / 1 | 3.550795 / 1 | 8.399204 / 1 |
| detail:expert-gate | nested | 23.734408 / 16 | 23.224424 / 16 | 28.277417 / 16 |
| detail:expert-up | nested | 22.125669 / 16 | 22.735530 / 16 | 26.693474 / 16 |
| detail:expert-activation | nested | 0.097008 / 16 | 0.103183 / 16 | 0.112319 / 16 |
| detail:expert-down | nested | 23.537768 / 16 | 22.576674 / 16 | 30.472145 / 16 |
| EUP | nested | 0.615239 / 1 | 0.600052 / 1 | 1.456952 / 1 |
| experts-mix-normalize-up | boundary | 70.540393 / 1 | 69.642665 / 1 | 87.402178 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002254 / 1 | 0.002825 / 1 | 0.002304 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000220 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429252 / 1 | 0.422640 / 1 | 0.372235 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 946.808139 / 1 | 927.613298 / 1 | 1014.858101 / 1 |
| op:Q   int8 projection | nested | 21.057395 / 1 | 18.188625 / 1 | 36.038355 / 1 |
| op:X   mxfp4 expert proj | nested | 69.563277 / 1 | 68.708970 / 1 | 85.622875 / 1 |
| op:N   rmsnorm | nested | 0.035006 / 1 | 0.033883 / 1 | 0.023664 / 1 |
| op:L   l2 per-head | nested | 0.005741 / 1 | 0.005160 / 1 | 0.005520 / 1 |
| op:SiTU + sigma | nested | 0.106839 / 1 | 0.113153 / 1 | 0.123700 / 1 |
| op:C   shortconv | nested | 0.053409 / 1 | 0.055053 / 1 | 0.035746 / 1 |
| op:AR  snapshot aggregate | nested | 0.040276 / 1 | 0.045134 / 1 | 0.036117 / 1 |
| op:D   kda delta-rule | nested | 0.133369 / 1 | 0.145321 / 1 | 0.242172 / 1 |
| op:router dot product | nested | 0.604580 / 1 | 0.632833 / 1 | 1.370180 / 1 |
| op:top-k selection | nested | 0.002846 / 1 | 0.002765 / 1 | 0.002194 / 1 |
| op:alpha / beta / gate | nested | 0.073738 / 1 | 0.072556 / 1 | 0.057328 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.262578 / 1 | 89.519484 / 1 | 124.689989 / 1 |

### Layer 55

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005310 / 1 | 0.005250 / 1 | 0.006051 / 1 |
| pre-attention-aggregation | boundary | 0.022402 / 1 | 0.021300 / 1 | 0.017683 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010750 / 1 | 0.011762 / 1 | 0.164106 / 1 |
| QA | nested | 0.299620 / 1 | 0.294019 / 1 | 0.866018 / 1 |
| QB | nested | 0.738510 / 1 | 0.701731 / 1 | 1.844026 / 1 |
| KA | nested | 0.102472 / 1 | 0.118161 / 1 | 0.440082 / 1 |
| KB | nested | 0.334946 / 1 | 0.305932 / 1 | 0.959543 / 1 |
| G | nested | 2.063996 / 1 | 1.983075 / 1 | 4.694482 / 1 |
| O | nested | 1.983887 / 1 | 1.962928 / 1 | 4.667962 / 1 |
| attention | boundary | 5.619590 / 1 | 5.466113 / 1 | 13.576006 / 1 |
| attention-residual | boundary | 0.002275 / 1 | 0.002294 / 1 | 0.001953 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025899 / 1 | 0.026810 / 1 | 0.028093 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002355 / 1 | 0.002454 / 1 | 0.003136 / 1 |
| router-and-top16 | boundary | 0.573893 / 1 | 0.571859 / 1 | 1.162582 / 1 |
| EDOWN | nested | 0.599100 / 1 | 0.575184 / 1 | 1.596513 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.600312 / 1 | 0.576488 / 1 | 1.598036 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048441 / 1 | 0.017212 / 1 | 0.005991 / 1 |
| SH1 | nested | 0.993236 / 1 | 0.962829 / 1 | 2.377512 / 1 |
| SH3 | nested | 0.971656 / 1 | 0.993507 / 1 | 2.436553 / 1 |
| SH2 | nested | 0.990150 / 1 | 0.981925 / 1 | 2.247379 / 1 |
| shared-expert-during-read | boundary | 2.971182 / 1 | 2.950793 / 1 | 7.077994 / 1 |
| detail:expert-gate | nested | 25.922377 / 16 | 24.550651 / 16 | 26.789196 / 16 |
| detail:expert-up | nested | 25.109766 / 16 | 23.977954 / 16 | 25.833899 / 16 |
| detail:expert-activation | nested | 0.103282 / 16 | 0.108712 / 16 | 0.123873 / 16 |
| detail:expert-down | nested | 27.063175 / 16 | 23.057041 / 16 | 24.334109 / 16 |
| EUP | nested | 0.622974 / 1 | 0.588039 / 1 | 1.657397 / 1 |
| experts-mix-normalize-up | boundary | 79.263221 / 1 | 72.699227 / 1 | 79.123009 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002835 / 1 | 0.002585 / 1 | 0.002274 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000281 / 1 | 0.000140 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007734 / 1 | 0.008045 / 1 | 0.007854 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 996.124366 / 1 | 945.536279 / 1 | 997.237478 / 1 |
| op:Q   int8 projection | nested | 9.699174 / 1 | 9.465898 / 1 | 23.786074 / 1 |
| op:X   mxfp4 expert proj | nested | 78.268623 / 1 | 71.762915 / 1 | 77.150574 / 1 |
| op:N   rmsnorm | nested | 0.023914 / 1 | 0.026137 / 1 | 0.024585 / 1 |
| op:SiTU + sigma | nested | 0.114383 / 1 | 0.116268 / 1 | 0.135453 / 1 |
| op:AR  snapshot aggregate | nested | 0.038312 / 1 | 0.037530 / 1 | 0.034333 / 1 |
| op:SA  softmax attention | nested | 0.006402 / 1 | 0.009137 / 1 | 0.013716 / 1 |
| op:router dot product | nested | 0.571147 / 1 | 0.569204 / 1 | 1.159967 / 1 |
| op:top-k selection | nested | 0.002274 / 1 | 0.002284 / 1 | 0.002185 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.169290 / 1 | 82.376047 / 1 | 102.790550 / 1 |

### Layer 56

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004057 / 1 | 0.004608 / 1 | 0.072896 / 1 |
| pre-attention-aggregation | boundary | 0.016391 / 1 | 0.016851 / 1 | 0.017323 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.009718 / 1 | 0.009869 / 1 | 0.011321 / 1 |
| Q | nested | 2.571976 / 1 | 2.502115 / 1 | 5.562894 / 1 |
| K | nested | 2.398471 / 1 | 2.371450 / 1 | 3.814197 / 1 |
| V | nested | 2.261846 / 1 | 2.253832 / 1 | 2.950683 / 1 |
| B | nested | 0.020658 / 1 | 0.020628 / 1 | 0.290994 / 1 |
| FA | nested | 0.033473 / 1 | 0.029395 / 1 | 0.165078 / 1 |
| FB | nested | 0.059972 / 1 | 0.047027 / 1 | 0.152555 / 1 |
| G | nested | 1.935607 / 1 | 1.957428 / 1 | 4.326524 / 1 |
| O | nested | 2.068595 / 1 | 2.112427 / 1 | 4.720280 / 1 |
| attention | boundary | 12.515184 / 1 | 12.420087 / 1 | 22.712738 / 1 |
| attention-residual | boundary | 0.002224 / 1 | 0.001503 / 1 | 0.003126 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026370 / 1 | 0.026239 / 1 | 0.026931 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002625 / 1 | 0.002555 / 1 | 0.002856 / 1 |
| router-and-top16 | boundary | 0.596204 / 1 | 0.604329 / 1 | 1.237753 / 1 |
| EDOWN | nested | 0.590484 / 1 | 0.601234 / 1 | 1.471099 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.592036 / 1 | 0.602536 / 1 | 1.472621 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027892 / 1 | 0.017192 / 1 | 0.006422 / 1 |
| SH1 | nested | 1.012492 / 1 | 1.009296 / 1 | 2.449376 / 1 |
| SH3 | nested | 1.021348 / 1 | 1.012181 / 1 | 2.604557 / 1 |
| SH2 | nested | 1.002984 / 1 | 0.988807 / 1 | 2.509950 / 1 |
| shared-expert-during-read | boundary | 3.050079 / 1 | 3.023319 / 1 | 7.580955 / 1 |
| detail:expert-gate | nested | 28.067813 / 16 | 24.942815 / 16 | 21.849837 / 16 |
| detail:expert-up | nested | 26.430487 / 16 | 22.582968 / 16 | 23.331584 / 16 |
| detail:expert-activation | nested | 0.104333 / 16 | 0.105035 / 16 | 0.128341 / 16 |
| detail:expert-down | nested | 28.830939 / 16 | 23.954640 / 16 | 20.690621 / 16 |
| EUP | nested | 0.811236 / 1 | 0.697904 / 1 | 1.138407 / 1 |
| experts-mix-normalize-up | boundary | 84.653092 / 1 | 72.695319 / 1 | 67.523035 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002505 / 1 | 0.002545 / 1 | 0.002805 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000141 / 1 | 0.000390 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.434282 / 1 | 0.435433 / 1 | 0.369590 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1076.710480 / 1 | 941.594017 / 1 | 797.394274 / 1 |
| op:Q   int8 projection | nested | 15.787660 / 1 | 15.602322 / 1 | 32.154972 / 1 |
| op:X   mxfp4 expert proj | nested | 83.504637 / 1 | 71.657903 / 1 | 66.069781 / 1 |
| op:N   rmsnorm | nested | 0.033051 / 1 | 0.039454 / 1 | 0.023826 / 1 |
| op:L   l2 per-head | nested | 0.005371 / 1 | 0.005600 / 1 | 0.006122 / 1 |
| op:SiTU + sigma | nested | 0.112521 / 1 | 0.112420 / 1 | 0.138947 / 1 |
| op:C   shortconv | nested | 0.074259 / 1 | 0.057157 / 1 | 0.038482 / 1 |
| op:AR  snapshot aggregate | nested | 0.033262 / 1 | 0.033363 / 1 | 0.033932 / 1 |
| op:D   kda delta-rule | nested | 0.142747 / 1 | 0.144429 / 1 | 0.147926 / 1 |
| op:router dot product | nested | 0.593369 / 1 | 0.600813 / 1 | 1.234677 / 1 |
| op:top-k selection | nested | 0.002364 / 1 | 0.002655 / 1 | 0.002675 / 1 |
| op:alpha / beta / gate | nested | 0.072285 / 1 | 0.071924 / 1 | 0.057587 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 101.946564 / 1 | 89.876842 / 1 | 101.058073 / 1 |

### Layer 57

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005620 / 1 | 0.005881 / 1 | 0.006442 / 1 |
| pre-attention-aggregation | boundary | 0.020358 / 1 | 0.026289 / 1 | 0.018715 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011582 / 1 | 0.012844 / 1 | 0.014788 / 1 |
| Q | nested | 2.585821 / 1 | 2.846769 / 1 | 4.981407 / 1 |
| K | nested | 2.374677 / 1 | 2.778000 / 1 | 3.468851 / 1 |
| V | nested | 2.253992 / 1 | 2.651144 / 1 | 2.979607 / 1 |
| B | nested | 0.022011 / 1 | 0.025167 / 1 | 0.140512 / 1 |
| FA | nested | 0.033162 / 1 | 0.028514 / 1 | 0.096791 / 1 |
| FB | nested | 0.047849 / 1 | 0.050304 / 1 | 0.161852 / 1 |
| G | nested | 1.943411 / 1 | 2.815260 / 1 | 4.065686 / 1 |
| O | nested | 2.236329 / 1 | 3.444576 / 1 | 4.096965 / 1 |
| attention | boundary | 12.604231 / 1 | 15.774905 / 1 | 20.642680 / 1 |
| attention-residual | boundary | 0.002715 / 1 | 0.002084 / 1 | 0.001954 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026209 / 1 | 0.026620 / 1 | 0.029305 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002745 / 1 | 0.002935 / 1 | 0.003536 / 1 |
| router-and-top16 | boundary | 0.591726 / 1 | 0.626110 / 1 | 1.073095 / 1 |
| EDOWN | nested | 0.594471 / 1 | 0.683137 / 1 | 1.508058 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595823 / 1 | 0.684459 / 1 | 1.509420 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.049743 / 1 | 0.038101 / 1 | 0.006352 / 1 |
| SH1 | nested | 1.010659 / 1 | 1.204701 / 1 | 2.248211 / 1 |
| SH3 | nested | 1.024875 / 1 | 1.160448 / 1 | 2.299466 / 1 |
| SH2 | nested | 0.985561 / 1 | 1.216263 / 1 | 2.403120 / 1 |
| shared-expert-during-read | boundary | 3.034250 / 1 | 3.597431 / 1 | 6.968230 / 1 |
| detail:expert-gate | nested | 27.046545 / 16 | 26.187039 / 16 | 18.639779 / 16 |
| detail:expert-up | nested | 26.341626 / 16 | 25.335557 / 16 | 17.293522 / 16 |
| detail:expert-activation | nested | 0.099849 / 16 | 0.107422 / 16 | 0.100026 / 16 |
| detail:expert-down | nested | 28.013771 / 16 | 24.433916 / 16 | 19.071953 / 16 |
| EUP | nested | 0.826274 / 1 | 0.710387 / 1 | 1.116826 / 1 |
| experts-mix-normalize-up | boundary | 82.770815 / 1 | 77.206849 / 1 | 56.618520 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002956 / 1 | 0.002344 / 1 | 0.002605 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000290 / 1 | 0.000140 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.435573 / 1 | 0.434802 / 1 | 0.376814 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1071.773943 / 1 | 998.063940 / 1 | 742.071757 / 1 |
| op:Q   int8 projection | nested | 15.937480 / 1 | 19.613207 / 1 | 29.565700 / 1 |
| op:X   mxfp4 expert proj | nested | 81.575752 / 1 | 76.134443 / 1 | 55.174062 / 1 |
| op:N   rmsnorm | nested | 0.031427 / 1 | 0.034414 / 1 | 0.027060 / 1 |
| op:L   l2 per-head | nested | 0.004839 / 1 | 0.004859 / 1 | 0.006933 / 1 |
| op:SiTU + sigma | nested | 0.107782 / 1 | 0.118492 / 1 | 0.111699 / 1 |
| op:C   shortconv | nested | 0.058329 / 1 | 0.054732 / 1 | 0.041818 / 1 |
| op:AR  snapshot aggregate | nested | 0.037150 / 1 | 0.043051 / 1 | 0.037080 / 1 |
| op:D   kda delta-rule | nested | 0.134672 / 1 | 0.148407 / 1 | 0.218178 / 1 |
| op:router dot product | nested | 0.588720 / 1 | 0.623435 / 1 | 1.070510 / 1 |
| op:top-k selection | nested | 0.002525 / 1 | 0.002335 / 1 | 0.002194 / 1 |
| op:alpha / beta / gate | nested | 0.073768 / 1 | 0.074379 / 1 | 0.062056 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 100.168822 / 1 | 98.456333 / 1 | 87.287454 / 1 |

### Layer 58

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005390 / 1 | 0.005270 / 1 | 0.006912 / 1 |
| pre-attention-aggregation | boundary | 0.021180 / 1 | 0.025117 / 1 | 0.017412 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011201 / 1 | 0.012183 / 1 | 0.010980 / 1 |
| Q | nested | 3.209718 / 1 | 2.865113 / 1 | 4.883014 / 1 |
| K | nested | 3.154274 / 1 | 2.777991 / 1 | 3.242429 / 1 |
| V | nested | 3.057252 / 1 | 2.670059 / 1 | 2.537581 / 1 |
| B | nested | 0.028613 / 1 | 0.024286 / 1 | 0.026489 / 1 |
| FA | nested | 0.035406 / 1 | 0.030007 / 1 | 0.138319 / 1 |
| FB | nested | 0.067366 / 1 | 0.066715 / 1 | 0.151483 / 1 |
| G | nested | 2.743025 / 1 | 2.378494 / 1 | 3.503376 / 1 |
| O | nested | 2.822995 / 1 | 2.204689 / 1 | 3.929933 / 1 |
| attention | boundary | 16.228091 / 1 | 14.162222 / 1 | 19.046427 / 1 |
| attention-residual | boundary | 0.002284 / 1 | 0.002033 / 1 | 0.002044 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.032391 / 1 | 0.027482 / 1 | 0.027452 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002625 / 1 | 0.002926 / 1 | 0.002876 / 1 |
| router-and-top16 | boundary | 0.602195 / 1 | 0.597797 / 1 | 1.289309 / 1 |
| EDOWN | nested | 0.811266 / 1 | 0.590474 / 1 | 1.512226 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.812598 / 1 | 0.591766 / 1 | 1.513719 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.033423 / 1 | 0.006402 / 1 | 0.023534 / 1 |
| SH1 | nested | 1.425433 / 1 | 1.445721 / 1 | 2.008342 / 1 |
| SH3 | nested | 1.358258 / 1 | 1.516414 / 1 | 1.913645 / 1 |
| SH2 | nested | 1.374378 / 1 | 1.570775 / 1 | 1.813358 / 1 |
| shared-expert-during-read | boundary | 4.173738 / 1 | 4.546045 / 1 | 5.748762 / 1 |
| detail:expert-gate | nested | 23.968596 / 16 | 23.031235 / 16 | 23.586962 / 16 |
| detail:expert-up | nested | 22.635814 / 16 | 23.071556 / 16 | 21.942811 / 16 |
| detail:expert-activation | nested | 0.096651 / 16 | 0.096691 / 16 | 0.118891 / 16 |
| detail:expert-down | nested | 23.714890 / 16 | 23.399481 / 16 | 21.676786 / 16 |
| EUP | nested | 0.831484 / 1 | 1.117268 / 1 | 1.139729 / 1 |
| experts-mix-normalize-up | boundary | 71.678559 / 1 | 71.124835 / 1 | 68.884178 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002905 / 1 | 0.002655 / 1 | 0.002174 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000220 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.421588 / 1 | 0.438069 / 1 | 0.368449 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 961.961759 / 1 | 944.924763 / 1 | 878.946675 / 1 |
| op:Q   int8 projection | nested | 20.917513 / 1 | 19.256393 / 1 | 26.797770 / 1 |
| op:X   mxfp4 expert proj | nested | 70.488036 / 1 | 69.675428 / 1 | 67.396939 / 1 |
| op:N   rmsnorm | nested | 0.038592 / 1 | 0.036808 / 1 | 0.024376 / 1 |
| op:L   l2 per-head | nested | 0.005690 / 1 | 0.005169 / 1 | 0.005471 / 1 |
| op:SiTU + sigma | nested | 0.107462 / 1 | 0.104335 / 1 | 0.126415 / 1 |
| op:C   shortconv | nested | 0.062777 / 1 | 0.058460 / 1 | 0.036388 / 1 |
| op:AR  snapshot aggregate | nested | 0.043490 / 1 | 0.042710 / 1 | 0.034144 / 1 |
| op:D   kda delta-rule | nested | 0.145953 / 1 | 0.143639 / 1 | 0.215302 / 1 |
| op:router dot product | nested | 0.599140 / 1 | 0.595101 / 1 | 1.286985 / 1 |
| op:top-k selection | nested | 0.002565 / 1 | 0.002244 / 1 | 0.002074 / 1 |
| op:alpha / beta / gate | nested | 0.074480 / 1 | 0.074560 / 1 | 0.057116 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.042506 / 1 | 91.558915 / 1 | 96.959706 / 1 |

### Layer 59

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005540 / 1 | 0.005130 / 1 | 0.010139 / 1 |
| pre-attention-aggregation | boundary | 0.021400 / 1 | 0.024035 / 1 | 0.018224 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011602 / 1 | 0.011792 / 1 | 0.011001 / 1 |
| QA | nested | 0.382274 / 1 | 0.315980 / 1 | 0.680472 / 1 |
| QB | nested | 0.873852 / 1 | 0.777613 / 1 | 1.746544 / 1 |
| KA | nested | 0.127989 / 1 | 0.114875 / 1 | 0.370983 / 1 |
| KB | nested | 0.320549 / 1 | 0.327081 / 1 | 0.855999 / 1 |
| G | nested | 2.803518 / 1 | 2.660951 / 1 | 3.562256 / 1 |
| O | nested | 2.828045 / 1 | 2.332438 / 1 | 4.089792 / 1 |
| attention | boundary | 7.434050 / 1 | 6.631320 / 1 | 11.415740 / 1 |
| attention-residual | boundary | 0.002134 / 1 | 0.002435 / 1 | 0.002214 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026499 / 1 | 0.026520 / 1 | 0.027632 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002324 / 1 | 0.002695 / 1 | 0.173935 / 1 |
| router-and-top16 | boundary | 0.583530 / 1 | 0.607575 / 1 | 1.184583 / 1 |
| EDOWN | nested | 0.823829 / 1 | 0.647880 / 1 | 1.533605 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.825152 / 1 | 0.649514 / 1 | 1.535199 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.054341 / 1 | 0.012834 / 1 | 0.012063 / 1 |
| SH1 | nested | 1.406228 / 1 | 1.100246 / 1 | 2.310257 / 1 |
| SH3 | nested | 1.406108 / 1 | 1.141373 / 1 | 2.283997 / 1 |
| SH2 | nested | 1.396670 / 1 | 1.181858 / 1 | 2.007461 / 1 |
| shared-expert-during-read | boundary | 4.224503 / 1 | 3.439527 / 1 | 6.619158 / 1 |
| detail:expert-gate | nested | 27.465831 / 16 | 27.175203 / 16 | 20.233146 / 16 |
| detail:expert-up | nested | 25.664905 / 16 | 25.833962 / 16 | 19.173854 / 16 |
| detail:expert-activation | nested | 0.098242 / 16 | 0.104267 / 16 | 0.116498 / 16 |
| detail:expert-down | nested | 27.625358 / 16 | 25.616171 / 16 | 18.494158 / 16 |
| EUP | nested | 0.589361 / 1 | 0.599340 / 1 | 1.135281 / 1 |
| experts-mix-normalize-up | boundary | 81.894869 / 1 | 79.738509 / 1 | 59.545559 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002745 / 1 | 0.002495 / 1 | 0.002504 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000270 / 1 | 0.000290 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007764 / 1 | 0.007023 / 1 | 0.008476 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1071.501813 / 1 | 1063.775257 / 1 | 725.921707 / 1 |
| op:Q   int8 projection | nested | 12.957029 / 1 | 11.198384 / 1 | 20.575194 / 1 |
| op:X   mxfp4 expert proj | nested | 80.931007 / 1 | 78.804001 / 1 | 58.088346 / 1 |
| op:N   rmsnorm | nested | 0.025087 / 1 | 0.024987 / 1 | 0.024547 / 1 |
| op:SiTU + sigma | nested | 0.108170 / 1 | 0.114997 / 1 | 0.128410 / 1 |
| op:AR  snapshot aggregate | nested | 0.037490 / 1 | 0.040005 / 1 | 0.034574 / 1 |
| op:SA  softmax attention | nested | 0.006652 / 1 | 0.009988 / 1 | 0.012183 / 1 |
| op:router dot product | nested | 0.579894 / 1 | 0.604559 / 1 | 1.181738 / 1 |
| op:top-k selection | nested | 0.002916 / 1 | 0.002574 / 1 | 0.002424 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.110611 / 1 | 91.175749 / 1 | 80.583818 / 1 |

### Layer 60

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003958 / 1 | 0.004258 / 1 | 0.006082 / 1 |
| pre-attention-aggregation | boundary | 0.016340 / 1 | 0.016721 / 1 | 0.017152 / 1 |
| snapshot-push | boundary | 0.001423 / 1 | 0.001643 / 1 | 0.001853 / 1 |
| pre-attention-normalization | boundary | 0.010791 / 1 | 0.010139 / 1 | 0.010851 / 1 |
| Q | nested | 2.538132 / 1 | 2.510160 / 1 | 4.953245 / 1 |
| K | nested | 2.415553 / 1 | 2.379827 / 1 | 3.095574 / 1 |
| V | nested | 2.237421 / 1 | 2.238312 / 1 | 2.579429 / 1 |
| B | nested | 0.020478 / 1 | 0.022432 / 1 | 0.130163 / 1 |
| FA | nested | 0.026540 / 1 | 0.026790 / 1 | 0.119072 / 1 |
| FB | nested | 0.046367 / 1 | 0.047939 / 1 | 0.156342 / 1 |
| G | nested | 1.948571 / 1 | 2.287705 / 1 | 3.420621 / 1 |
| O | nested | 2.093782 / 1 | 2.266696 / 1 | 3.702428 / 1 |
| attention | boundary | 12.420197 / 1 | 12.885586 / 1 | 18.867574 / 1 |
| attention-residual | boundary | 0.001894 / 1 | 0.002024 / 1 | 0.002004 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027461 / 1 | 0.027441 / 1 | 0.028293 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002274 / 1 | 0.002906 / 1 | 0.002865 / 1 |
| router-and-top16 | boundary | 0.620649 / 1 | 0.619518 / 1 | 0.861891 / 1 |
| EDOWN | nested | 0.607936 / 1 | 0.592557 / 1 | 1.197387 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609279 / 1 | 0.593860 / 1 | 1.199110 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.022572 / 1 | 0.012043 / 1 | 0.006753 / 1 |
| SH1 | nested | 1.022831 / 1 | 1.005188 / 1 | 1.730854 / 1 |
| SH3 | nested | 1.010869 / 1 | 1.012242 / 1 | 2.025805 / 1 |
| SH2 | nested | 1.001552 / 1 | 0.980321 / 1 | 1.956796 / 1 |
| shared-expert-during-read | boundary | 3.048445 / 1 | 3.011206 / 1 | 5.730077 / 1 |
| detail:expert-gate | nested | 26.481703 / 16 | 26.280386 / 16 | 24.322243 / 16 |
| detail:expert-up | nested | 25.399118 / 16 | 25.151456 / 16 | 23.614763 / 16 |
| detail:expert-activation | nested | 0.101018 / 16 | 0.110758 / 16 | 0.122217 / 16 |
| detail:expert-down | nested | 27.626901 / 16 | 24.860793 / 16 | 24.009971 / 16 |
| EUP | nested | 0.594692 / 1 | 0.588270 / 1 | 1.190244 / 1 |
| experts-mix-normalize-up | boundary | 80.626058 / 1 | 77.402915 / 1 | 73.662977 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002895 / 1 | 0.003156 / 1 | 0.002184 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000411 / 1 | 0.000130 / 1 | 0.000551 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.435304 / 1 | 0.427990 / 1 | 0.381963 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1007.971563 / 1 | 1015.338650 / 1 | 887.747849 / 1 |
| op:Q   int8 projection | nested | 15.563252 / 1 | 15.957037 / 1 | 26.255917 / 1 |
| op:X   mxfp4 expert proj | nested | 79.685419 / 1 | 76.480171 / 1 | 72.143859 / 1 |
| op:N   rmsnorm | nested | 0.035234 / 1 | 0.040225 / 1 | 0.023744 / 1 |
| op:L   l2 per-head | nested | 0.005410 / 1 | 0.005360 / 1 | 0.005320 / 1 |
| op:SiTU + sigma | nested | 0.108883 / 1 | 0.118824 / 1 | 0.133258 / 1 |
| op:C   shortconv | nested | 0.058650 / 1 | 0.056486 / 1 | 0.035766 / 1 |
| op:AR  snapshot aggregate | nested | 0.034153 / 1 | 0.034004 / 1 | 0.035065 / 1 |
| op:D   kda delta-rule | nested | 0.144360 / 1 | 0.144490 / 1 | 0.145802 / 1 |
| op:router dot product | nested | 0.617444 / 1 | 0.616071 / 1 | 0.859136 / 1 |
| op:top-k selection | nested | 0.002655 / 1 | 0.003046 / 1 | 0.002414 / 1 |
| op:alpha / beta / gate | nested | 0.075011 / 1 | 0.073788 / 1 | 0.056716 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 97.865438 / 1 | 95.036884 / 1 | 100.797627 / 1 |

### Layer 61

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005250 / 1 | 0.005160 / 1 | 0.006482 / 1 |
| pre-attention-aggregation | boundary | 0.023624 / 1 | 0.026881 / 1 | 0.018134 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011041 / 1 | 0.011191 / 1 | 0.011542 / 1 |
| Q | nested | 2.574370 / 1 | 2.504269 / 1 | 6.428211 / 1 |
| K | nested | 2.370470 / 1 | 2.382272 / 1 | 3.753734 / 1 |
| V | nested | 2.240947 / 1 | 2.234004 / 1 | 3.364526 / 1 |
| B | nested | 0.024616 / 1 | 0.024015 / 1 | 0.036197 / 1 |
| FA | nested | 0.026820 / 1 | 0.028012 / 1 | 0.123591 / 1 |
| FB | nested | 0.049021 / 1 | 0.052608 / 1 | 0.145061 / 1 |
| G | nested | 1.931369 / 1 | 1.914617 / 1 | 3.867376 / 1 |
| O | nested | 2.035604 / 1 | 2.015676 / 1 | 4.280798 / 1 |
| attention | boundary | 12.327393 / 1 | 12.252594 / 1 | 22.652155 / 1 |
| attention-residual | boundary | 0.002575 / 1 | 0.002144 / 1 | 0.001503 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026680 / 1 | 0.027462 / 1 | 0.028733 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002535 / 1 | 0.002765 / 1 | 0.003327 / 1 |
| router-and-top16 | boundary | 0.584853 / 1 | 0.594450 / 1 | 0.897537 / 1 |
| EDOWN | nested | 0.583611 / 1 | 0.585414 / 1 | 1.436995 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.584983 / 1 | 0.586707 / 1 | 1.438448 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027682 / 1 | 0.032311 / 1 | 0.006682 / 1 |
| SH1 | nested | 0.991182 / 1 | 1.003746 / 1 | 2.425201 / 1 |
| SH3 | nested | 0.989258 / 1 | 1.008244 / 1 | 1.982053 / 1 |
| SH2 | nested | 0.992274 / 1 | 0.982415 / 1 | 2.170125 / 1 |
| shared-expert-during-read | boundary | 2.986189 / 1 | 3.007680 / 1 | 6.593370 / 1 |
| detail:expert-gate | nested | 27.004607 / 16 | 26.404113 / 16 | 18.924730 / 16 |
| detail:expert-up | nested | 26.330448 / 16 | 25.902257 / 16 | 18.710076 / 16 |
| detail:expert-activation | nested | 0.113320 / 16 | 0.106822 / 16 | 0.112750 / 16 |
| detail:expert-down | nested | 27.652492 / 16 | 25.371426 / 16 | 20.931162 / 16 |
| EUP | nested | 0.594181 / 1 | 0.722871 / 1 | 1.138297 / 1 |
| experts-mix-normalize-up | boundary | 82.112715 / 1 | 78.958191 / 1 | 60.213057 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002525 / 1 | 0.002855 / 1 | 0.003066 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000140 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427098 / 1 | 0.429823 / 1 | 0.408854 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1047.298723 / 1 | 1047.868483 / 1 | 631.516715 / 1 |
| op:Q   int8 projection | nested | 15.402037 / 1 | 15.456732 / 1 | 31.150512 / 1 |
| op:X   mxfp4 expert proj | nested | 81.177194 / 1 | 77.861410 / 1 | 58.749051 / 1 |
| op:N   rmsnorm | nested | 0.035966 / 1 | 0.035816 / 1 | 0.025167 / 1 |
| op:L   l2 per-head | nested | 0.004979 / 1 | 0.004809 / 1 | 0.005740 / 1 |
| op:SiTU + sigma | nested | 0.121346 / 1 | 0.114374 / 1 | 0.123440 / 1 |
| op:C   shortconv | nested | 0.052909 / 1 | 0.054161 / 1 | 0.037280 / 1 |
| op:AR  snapshot aggregate | nested | 0.040526 / 1 | 0.044402 / 1 | 0.036077 / 1 |
| op:D   kda delta-rule | nested | 0.140042 / 1 | 0.138008 / 1 | 0.224390 / 1 |
| op:router dot product | nested | 0.582158 / 1 | 0.591635 / 1 | 0.894812 / 1 |
| op:top-k selection | nested | 0.002395 / 1 | 0.002325 / 1 | 0.002504 / 1 |
| op:alpha / beta / gate | nested | 0.074770 / 1 | 0.074639 / 1 | 0.057798 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 99.139640 / 1 | 95.956853 / 1 | 92.298958 / 1 |

### Layer 62

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004899 / 1 | 0.005289 / 1 | 0.008005 / 1 |
| pre-attention-aggregation | boundary | 0.025457 / 1 | 0.027321 / 1 | 0.023905 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010950 / 1 | 0.010519 / 1 | 0.012814 / 1 |
| Q | nested | 2.526601 / 1 | 2.836620 / 1 | 7.855788 / 1 |
| K | nested | 2.395015 / 1 | 2.761179 / 1 | 3.421953 / 1 |
| V | nested | 2.299957 / 1 | 2.693744 / 1 | 2.822974 / 1 |
| B | nested | 0.020378 / 1 | 0.024325 / 1 | 0.098023 / 1 |
| FA | nested | 0.027311 / 1 | 0.030948 / 1 | 0.128650 / 1 |
| FB | nested | 0.048270 / 1 | 0.054833 / 1 | 0.163636 / 1 |
| G | nested | 1.946968 / 1 | 2.289809 / 1 | 3.600096 / 1 |
| O | nested | 2.029331 / 1 | 2.274701 / 1 | 3.828904 / 1 |
| attention | boundary | 12.386594 / 1 | 14.111407 / 1 | 22.753424 / 1 |
| attention-residual | boundary | 0.002435 / 1 | 0.002455 / 1 | 0.002324 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026389 / 1 | 0.027201 / 1 | 0.029755 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002535 / 1 | 0.002785 / 1 | 0.003497 / 1 |
| router-and-top16 | boundary | 0.590634 / 1 | 0.632772 / 1 | 0.892076 / 1 |
| EDOWN | nested | 0.588199 / 1 | 0.670442 / 1 | 1.275443 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.589602 / 1 | 0.671995 / 1 | 1.277147 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.047549 / 1 | 0.009638 / 1 | 0.010850 / 1 |
| SH1 | nested | 0.994468 / 1 | 1.187138 / 1 | 2.429570 / 1 |
| SH3 | nested | 1.013835 / 1 | 1.181909 / 1 | 2.250285 / 1 |
| SH2 | nested | 1.002333 / 1 | 1.183912 / 1 | 2.029543 / 1 |
| shared-expert-during-read | boundary | 3.023981 / 1 | 3.569098 / 1 | 6.725416 / 1 |
| detail:expert-gate | nested | 26.569635 / 16 | 26.449069 / 16 | 23.192543 / 16 |
| detail:expert-up | nested | 25.429104 / 16 | 25.188424 / 16 | 21.715504 / 16 |
| detail:expert-activation | nested | 0.103281 / 16 | 0.103134 / 16 | 0.120325 / 16 |
| detail:expert-down | nested | 27.542662 / 16 | 25.226242 / 16 | 24.579078 / 16 |
| EUP | nested | 0.743600 / 1 | 0.727960 / 1 | 1.222023 / 1 |
| experts-mix-normalize-up | boundary | 80.843313 / 1 | 78.127069 / 1 | 71.236694 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002845 / 1 | 0.002475 / 1 | 0.002435 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000210 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.433230 / 1 | 0.442637 / 1 | 0.389057 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1039.321159 / 1 | 1042.576600 / 1 | 804.003282 / 1 |
| op:Q   int8 projection | nested | 15.634813 / 1 | 17.915885 / 1 | 31.125264 / 1 |
| op:X   mxfp4 expert proj | nested | 79.722068 / 1 | 77.043174 / 1 | 69.688691 / 1 |
| op:N   rmsnorm | nested | 0.035908 / 1 | 0.036029 / 1 | 0.025207 / 1 |
| op:L   l2 per-head | nested | 0.005851 / 1 | 0.005310 / 1 | 0.005510 / 1 |
| op:SiTU + sigma | nested | 0.111207 / 1 | 0.113844 / 1 | 0.130504 / 1 |
| op:C   shortconv | nested | 0.058809 / 1 | 0.062698 / 1 | 0.037289 / 1 |
| op:AR  snapshot aggregate | nested | 0.042199 / 1 | 0.044955 / 1 | 0.042419 / 1 |
| op:D   kda delta-rule | nested | 0.131747 / 1 | 0.143368 / 1 | 0.207758 / 1 |
| op:router dot product | nested | 0.587618 / 1 | 0.629566 / 1 | 0.889141 / 1 |
| op:top-k selection | nested | 0.002504 / 1 | 0.002735 / 1 | 0.002395 / 1 |
| op:alpha / beta / gate | nested | 0.074349 / 1 | 0.071794 / 1 | 0.056996 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 98.006052 / 1 | 97.658361 / 1 | 103.382156 / 1 |

### Layer 63

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006802 / 1 | 0.005691 / 1 | 0.006452 / 1 |
| pre-attention-aggregation | boundary | 0.024225 / 1 | 0.023203 / 1 | 0.019547 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.016611 / 1 | 0.010630 / 1 | 0.011020 / 1 |
| QA | nested | 0.345035 / 1 | 0.314598 / 1 | 0.466651 / 1 |
| QB | nested | 0.830512 / 1 | 0.776711 / 1 | 1.329675 / 1 |
| KA | nested | 0.124993 / 1 | 0.115105 / 1 | 0.204071 / 1 |
| KB | nested | 0.334495 / 1 | 0.338853 / 1 | 0.527145 / 1 |
| G | nested | 2.615186 / 1 | 2.362744 / 1 | 4.671900 / 1 |
| O | nested | 3.530958 / 1 | 2.735461 / 1 | 3.553750 / 1 |
| attention | boundary | 7.881747 / 1 | 6.744763 / 1 | 10.861133 / 1 |
| attention-residual | boundary | 0.002304 / 1 | 0.002023 / 1 | 0.001934 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027962 / 1 | 0.027150 / 1 | 0.028643 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002224 / 1 | 0.002375 / 1 | 0.002905 / 1 |
| router-and-top16 | boundary | 0.730806 / 1 | 0.704245 / 1 | 0.751003 / 1 |
| EDOWN | nested | 0.834910 / 1 | 0.825612 / 1 | 1.185825 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.836312 / 1 | 0.826995 / 1 | 1.187448 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032361 / 1 | 0.017122 / 1 | 0.009638 / 1 |
| SH1 | nested | 1.211184 / 1 | 1.630316 / 1 | 1.743999 / 1 |
| SH3 | nested | 0.978598 / 1 | 1.172100 / 1 | 1.791558 / 1 |
| SH2 | nested | 0.973479 / 1 | 1.166740 / 1 | 1.770118 / 1 |
| shared-expert-during-read | boundary | 3.175724 / 1 | 3.984696 / 1 | 5.325751 / 1 |
| detail:expert-gate | nested | 26.865334 / 16 | 26.263194 / 16 | 19.589673 / 16 |
| detail:expert-up | nested | 25.954083 / 16 | 24.967073 / 16 | 19.183472 / 16 |
| detail:expert-activation | nested | 0.098653 / 16 | 0.104935 / 16 | 0.133569 / 16 |
| detail:expert-down | nested | 27.048558 / 16 | 25.712384 / 16 | 17.649799 / 16 |
| EUP | nested | 0.745012 / 1 | 0.958271 / 1 | 1.013173 / 1 |
| experts-mix-normalize-up | boundary | 81.147712 / 1 | 78.417340 / 1 | 57.969063 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002465 / 1 | 0.002625 / 1 | 0.002184 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000281 / 1 | 0.000190 / 1 | 0.000511 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008365 / 1 | 0.007995 / 1 | 0.009137 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1059.941701 / 1 | 1061.600293 / 1 | 683.535481 / 1 |
| op:Q   int8 projection | nested | 12.523040 / 1 | 12.395300 / 1 | 18.256482 / 1 |
| op:X   mxfp4 expert proj | nested | 80.050239 / 1 | 77.131720 / 1 | 56.633268 / 1 |
| op:N   rmsnorm | nested | 0.030439 / 1 | 0.023752 / 1 | 0.023224 / 1 |
| op:SiTU + sigma | nested | 0.105958 / 1 | 0.114924 / 1 | 0.144860 / 1 |
| op:AR  snapshot aggregate | nested | 0.041358 / 1 | 0.040056 / 1 | 0.036979 / 1 |
| op:SA  softmax attention | nested | 0.006712 / 1 | 0.009157 / 1 | 0.012133 / 1 |
| op:router dot product | nested | 0.727790 / 1 | 0.701340 / 1 | 0.748158 / 1 |
| op:top-k selection | nested | 0.002505 / 1 | 0.002395 / 1 | 0.002444 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.910889 / 1 | 90.792213 / 1 | 76.204185 / 1 |

### Layer 64

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004669 / 1 | 0.004118 / 1 | 0.006402 / 1 |
| pre-attention-aggregation | boundary | 0.018354 / 1 | 0.017513 / 1 | 0.017724 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010410 / 1 | 0.014577 / 1 | 0.011311 / 1 |
| Q | nested | 3.075055 / 1 | 2.871024 / 1 | 4.949698 / 1 |
| K | nested | 2.978335 / 1 | 2.820650 / 1 | 3.629531 / 1 |
| V | nested | 2.927560 / 1 | 2.679336 / 1 | 2.923512 / 1 |
| B | nested | 0.027862 / 1 | 0.024426 / 1 | 0.031258 / 1 |
| FA | nested | 0.031268 / 1 | 0.029084 / 1 | 0.037230 / 1 |
| FB | nested | 0.052238 / 1 | 0.055333 / 1 | 0.061425 / 1 |
| G | nested | 2.669838 / 1 | 2.350021 / 1 | 3.005055 / 1 |
| O | nested | 2.612531 / 1 | 2.399804 / 1 | 3.436801 / 1 |
| attention | boundary | 15.494330 / 1 | 14.342789 / 1 | 18.802723 / 1 |
| attention-residual | boundary | 0.002565 / 1 | 0.002665 / 1 | 0.002245 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028493 / 1 | 0.028313 / 1 | 0.028944 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002324 / 1 | 0.002374 / 1 | 0.002855 / 1 |
| router-and-top16 | boundary | 0.743729 / 1 | 0.636430 / 1 | 0.780027 / 1 |
| EDOWN | nested | 0.792360 / 1 | 0.702472 / 1 | 0.944745 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.793693 / 1 | 0.703905 / 1 | 0.946178 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.034064 / 1 | 0.022572 / 1 | 0.007114 / 1 |
| SH1 | nested | 1.355212 / 1 | 1.365331 / 1 | 1.858002 / 1 |
| SH3 | nested | 1.361965 / 1 | 1.633252 / 1 | 1.790205 / 1 |
| SH2 | nested | 1.359510 / 1 | 2.043598 / 1 | 1.526763 / 1 |
| shared-expert-during-read | boundary | 4.092427 / 1 | 5.058953 / 1 | 5.191280 / 1 |
| detail:expert-gate | nested | 26.175919 / 16 | 26.175689 / 16 | 25.059102 / 16 |
| detail:expert-up | nested | 25.052311 / 16 | 25.245331 / 16 | 24.254457 / 16 |
| detail:expert-activation | nested | 0.098095 / 16 | 0.104566 / 16 | 0.121855 / 16 |
| detail:expert-down | nested | 26.506864 / 16 | 25.403944 / 16 | 22.989155 / 16 |
| EUP | nested | 0.900863 / 1 | 0.952129 / 1 | 1.205633 / 1 |
| experts-mix-normalize-up | boundary | 79.167061 / 1 | 78.304370 / 1 | 74.032497 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002745 / 1 | 0.002796 / 1 | 0.002745 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000130 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.433290 / 1 | 0.422830 / 1 | 0.390590 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1049.483720 / 1 | 1037.326736 / 1 | 923.560861 / 1 |
| op:Q   int8 projection | nested | 20.143005 / 1 | 19.924526 / 1 | 25.398046 / 1 |
| op:X   mxfp4 expert proj | nested | 77.912467 / 1 | 77.009872 / 1 | 72.503469 / 1 |
| op:N   rmsnorm | nested | 0.032471 / 1 | 0.033382 / 1 | 0.025046 / 1 |
| op:L   l2 per-head | nested | 0.004959 / 1 | 0.005310 / 1 | 0.005690 / 1 |
| op:SiTU + sigma | nested | 0.108293 / 1 | 0.115445 / 1 | 0.132486 / 1 |
| op:C   shortconv | nested | 0.058891 / 1 | 0.055104 / 1 | 0.037480 / 1 |
| op:AR  snapshot aggregate | nested | 0.036939 / 1 | 0.035586 / 1 | 0.035847 / 1 |
| op:D   kda delta-rule | nested | 0.143929 / 1 | 0.147606 / 1 | 0.147185 / 1 |
| op:router dot product | nested | 0.740614 / 1 | 0.633514 / 1 | 0.777443 / 1 |
| op:top-k selection | nested | 0.002725 / 1 | 0.002585 / 1 | 0.002334 / 1 |
| op:alpha / beta / gate | nested | 0.074129 / 1 | 0.073918 / 1 | 0.056807 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 100.843904 / 1 | 99.580272 / 1 | 100.241598 / 1 |

### Layer 65

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005580 / 1 | 0.106228 / 1 | 0.006392 / 1 |
| pre-attention-aggregation | boundary | 0.020688 / 1 | 0.024566 / 1 | 0.022442 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011311 / 1 | 0.012563 / 1 | 0.012212 / 1 |
| Q | nested | 3.485463 / 1 | 2.512414 / 1 | 9.114129 / 1 |
| K | nested | 3.310055 / 1 | 2.379175 / 1 | 3.467759 / 1 |
| V | nested | 3.345411 / 1 | 2.247169 / 1 | 3.101566 / 1 |
| B | nested | 0.029355 / 1 | 0.026540 / 1 | 0.030607 / 1 |
| FA | nested | 0.038091 / 1 | 0.035095 / 1 | 0.133941 / 1 |
| FB | nested | 0.059972 / 1 | 0.196667 / 1 | 0.140071 / 1 |
| G | nested | 3.046733 / 1 | 3.262406 / 1 | 3.471216 / 1 |
| O | nested | 2.906160 / 1 | 2.890421 / 1 | 3.761668 / 1 |
| attention | boundary | 17.327967 / 1 | 14.638783 / 1 | 23.905156 / 1 |
| attention-residual | boundary | 0.002585 / 1 | 0.002054 / 1 | 0.003096 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028202 / 1 | 0.028262 / 1 | 0.027161 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002415 / 1 | 0.002305 / 1 | 0.003526 / 1 |
| router-and-top16 | boundary | 0.591585 / 1 | 0.775308 / 1 | 0.929136 / 1 |
| EDOWN | nested | 0.771451 / 1 | 0.862151 / 1 | 1.249394 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.772814 / 1 | 0.863533 / 1 | 1.250998 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048501 / 1 | 0.021540 / 1 | 0.014918 / 1 |
| SH1 | nested | 1.298316 / 1 | 1.822636 / 1 | 2.245716 / 1 |
| SH3 | nested | 1.350133 / 1 | 0.996321 / 1 | 2.840557 / 1 |
| SH2 | nested | 1.289069 / 1 | 0.989729 / 1 | 2.439889 / 1 |
| shared-expert-during-read | boundary | 3.952845 / 1 | 3.822332 / 1 | 7.542923 / 1 |
| detail:expert-gate | nested | 25.551131 / 16 | 26.575916 / 16 | 18.060472 / 16 |
| detail:expert-up | nested | 24.332675 / 16 | 25.857704 / 16 | 17.749707 / 16 |
| detail:expert-activation | nested | 0.095025 / 16 | 0.109845 / 16 | 0.122988 / 16 |
| detail:expert-down | nested | 25.774079 / 16 | 25.311340 / 16 | 16.193087 / 16 |
| EUP | nested | 0.771752 / 1 | 1.066493 / 1 | 1.105355 / 1 |
| experts-mix-normalize-up | boundary | 76.978042 / 1 | 79.348210 / 1 | 53.631399 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002564 / 1 | 0.002404 / 1 | 0.002555 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000511 / 1 | 0.000140 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.444190 / 1 | 0.430905 / 1 | 0.364601 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1039.784641 / 1 | 1046.268791 / 1 | 630.568645 / 1 |
| op:Q   int8 projection | nested | 21.700366 / 1 | 19.285534 / 1 | 33.100243 / 1 |
| op:X   mxfp4 expert proj | nested | 75.833632 / 1 | 77.935802 / 1 | 52.201342 / 1 |
| op:N   rmsnorm | nested | 0.040826 / 1 | 0.037240 / 1 | 0.028562 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.005841 / 1 | 0.006593 / 1 |
| op:SiTU + sigma | nested | 0.104444 / 1 | 0.117209 / 1 | 0.134099 / 1 |
| op:C   shortconv | nested | 0.062126 / 1 | 0.051075 / 1 | 0.042899 / 1 |
| op:AR  snapshot aggregate | nested | 0.038733 / 1 | 0.042219 / 1 | 0.039114 / 1 |
| op:D   kda delta-rule | nested | 0.136475 / 1 | 0.134662 / 1 | 0.141705 / 1 |
| op:router dot product | nested | 0.588500 / 1 | 0.772103 / 1 | 0.925790 / 1 |
| op:top-k selection | nested | 0.002675 / 1 | 0.002805 / 1 | 0.002865 / 1 |
| op:alpha / beta / gate | nested | 0.074900 / 1 | 0.074559 / 1 | 0.056896 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 100.204910 / 1 | 100.095005 / 1 | 87.732225 / 1 |

### Layer 66

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005320 / 1 | 0.005731 / 1 | 0.008075 / 1 |
| pre-attention-aggregation | boundary | 0.026760 / 1 | 0.024566 / 1 | 0.018725 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011481 / 1 | 0.011221 / 1 | 0.011372 / 1 |
| Q | nested | 3.129638 / 1 | 2.519157 / 1 | 4.987730 / 1 |
| K | nested | 2.877136 / 1 | 2.373545 / 1 | 3.327216 / 1 |
| V | nested | 2.944031 / 1 | 2.238443 / 1 | 3.004654 / 1 |
| B | nested | 0.021931 / 1 | 0.023083 / 1 | 0.153918 / 1 |
| FA | nested | 0.031168 / 1 | 0.029175 / 1 | 0.136405 / 1 |
| FB | nested | 0.052027 / 1 | 0.049272 / 1 | 0.225321 / 1 |
| G | nested | 2.613934 / 1 | 1.927051 / 1 | 4.433113 / 1 |
| O | nested | 2.668285 / 1 | 2.182628 / 1 | 4.806241 / 1 |
| attention | boundary | 15.420794 / 1 | 12.439553 / 1 | 21.731184 / 1 |
| attention-residual | boundary | 0.002565 / 1 | 0.002365 / 1 | 0.002013 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028433 / 1 | 0.027501 / 1 | 0.028924 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002675 / 1 | 0.002464 / 1 | 0.178243 / 1 |
| router-and-top16 | boundary | 0.597897 / 1 | 0.594391 / 1 | 1.702351 / 1 |
| EDOWN | nested | 0.773636 / 1 | 0.586657 / 1 | 1.670060 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.774998 / 1 | 0.588219 / 1 | 1.671794 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.028063 / 1 | 0.006041 / 1 | 0.013946 / 1 |
| SH1 | nested | 1.361745 / 1 | 0.998625 / 1 | 2.683184 / 1 |
| SH3 | nested | 1.368417 / 1 | 0.999908 / 1 | 2.774484 / 1 |
| SH2 | nested | 1.347798 / 1 | 1.009115 / 1 | 2.901301 / 1 |
| shared-expert-during-read | boundary | 4.093749 / 1 | 3.020925 / 1 | 8.375910 / 1 |
| detail:expert-gate | nested | 27.506074 / 16 | 27.760149 / 16 | 20.729233 / 16 |
| detail:expert-up | nested | 26.112290 / 16 | 26.776411 / 16 | 21.065310 / 16 |
| detail:expert-activation | nested | 0.104716 / 16 | 0.107631 / 16 | 0.124401 / 16 |
| detail:expert-down | nested | 27.768483 / 16 | 25.823660 / 16 | 20.248813 / 16 |
| EUP | nested | 0.590614 / 1 | 0.737127 / 1 | 2.116345 / 1 |
| experts-mix-normalize-up | boundary | 82.527862 / 1 | 81.602963 / 1 | 64.760203 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002716 / 1 | 0.002765 / 1 | 0.002544 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000231 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.434962 / 1 | 0.436566 / 1 | 0.375351 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1083.984301 / 1 | 1076.895587 / 1 | 772.951760 / 1 |
| op:Q   int8 projection | nested | 19.778809 / 1 | 15.672183 / 1 | 33.218199 / 1 |
| op:X   mxfp4 expert proj | nested | 81.575130 / 1 | 80.548250 / 1 | 62.254481 / 1 |
| op:N   rmsnorm | nested | 0.036277 / 1 | 0.034004 / 1 | 0.026759 / 1 |
| op:L   l2 per-head | nested | 0.005700 / 1 | 0.005700 / 1 | 0.006212 / 1 |
| op:SiTU + sigma | nested | 0.114622 / 1 | 0.115325 / 1 | 0.135542 / 1 |
| op:C   shortconv | nested | 0.066014 / 1 | 0.054853 / 1 | 0.044482 / 1 |
| op:AR  snapshot aggregate | nested | 0.045385 / 1 | 0.042049 / 1 | 0.037310 / 1 |
| op:D   kda delta-rule | nested | 0.131956 / 1 | 0.144390 / 1 | 0.223147 / 1 |
| op:router dot product | nested | 0.595012 / 1 | 0.591415 / 1 | 1.699436 / 1 |
| op:top-k selection | nested | 0.002455 / 1 | 0.002535 / 1 | 0.002524 / 1 |
| op:alpha / beta / gate | nested | 0.073838 / 1 | 0.072025 / 1 | 0.057678 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 103.975115 / 1 | 98.782503 / 1 | 98.897436 / 1 |

### Layer 67

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005200 / 1 | 0.005230 / 1 | 0.174135 / 1 |
| pre-attention-aggregation | boundary | 0.030577 / 1 | 0.030477 / 1 | 0.018515 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012132 / 1 | 0.010850 / 1 | 0.010549 / 1 |
| QA | nested | 0.288469 / 1 | 0.293047 / 1 | 1.268239 / 1 |
| QB | nested | 0.697924 / 1 | 0.703174 / 1 | 1.880323 / 1 |
| KA | nested | 0.101240 / 1 | 0.100838 / 1 | 0.485157 / 1 |
| KB | nested | 0.301033 / 1 | 0.307004 / 1 | 3.362673 / 1 |
| G | nested | 1.960964 / 1 | 2.056462 / 1 | 9.112075 / 1 |
| O | nested | 1.954873 / 1 | 1.969610 / 1 | 12.456204 / 1 |
| attention | boundary | 5.405490 / 1 | 5.547014 / 1 | 28.829497 / 1 |
| attention-residual | boundary | 0.002194 / 1 | 0.002204 / 1 | 0.001633 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026630 / 1 | 0.025859 / 1 | 0.029475 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002184 / 1 | 0.002465 / 1 | 0.003437 / 1 |
| router-and-top16 | boundary | 0.582969 / 1 | 0.569454 / 1 | 5.652551 / 1 |
| EDOWN | nested | 0.567651 / 1 | 0.564555 / 1 | 4.568025 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.569113 / 1 | 0.566088 / 1 | 4.569658 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027822 / 1 | 0.006442 / 1 | 0.010430 / 1 |
| SH1 | nested | 0.973780 / 1 | 0.961967 / 1 | 6.111719 / 1 |
| SH3 | nested | 0.966586 / 1 | 0.958671 / 1 | 6.668500 / 1 |
| SH2 | nested | 0.985151 / 1 | 0.967378 / 1 | 6.735365 / 1 |
| shared-expert-during-read | boundary | 2.938831 / 1 | 2.901752 / 1 | 19.533868 / 1 |
| detail:expert-gate | nested | 26.779617 / 16 | 25.942663 / 16 | 24.991850 / 16 |
| detail:expert-up | nested | 25.592290 / 16 | 24.935853 / 16 | 25.760021 / 16 |
| detail:expert-activation | nested | 0.102110 / 16 | 0.107370 / 16 | 0.120324 / 16 |
| detail:expert-down | nested | 27.800197 / 16 | 25.217989 / 16 | 27.283847 / 16 |
| EUP | nested | 0.590203 / 1 | 0.720937 / 1 | 4.201530 / 1 |
| experts-mix-normalize-up | boundary | 81.315636 / 1 | 77.313778 / 1 | 82.824975 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002355 / 1 | 0.002815 / 1 | 0.003026 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000140 / 1 | 0.000351 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008405 / 1 | 0.008385 / 1 | 0.009277 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1032.163489 / 1 | 1036.618935 / 1 | 911.521670 / 1 |
| op:Q   int8 projection | nested | 9.386690 / 1 | 9.602433 / 1 | 56.847998 / 1 |
| op:X   mxfp4 expert proj | nested | 80.356442 / 1 | 76.290517 / 1 | 78.234149 / 1 |
| op:N   rmsnorm | nested | 0.024064 / 1 | 0.023923 / 1 | 0.025247 / 1 |
| op:SiTU + sigma | nested | 0.110047 / 1 | 0.114935 / 1 | 0.131535 / 1 |
| op:AR  snapshot aggregate | nested | 0.046608 / 1 | 0.046277 / 1 | 0.035817 / 1 |
| op:SA  softmax attention | nested | 0.007033 / 1 | 0.009367 / 1 | 0.012233 / 1 |
| op:router dot product | nested | 0.579893 / 1 | 0.566399 / 1 | 5.649356 / 1 |
| op:top-k selection | nested | 0.002624 / 1 | 0.002465 / 1 | 0.002785 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 90.949347 / 1 | 87.008312 / 1 | 141.687729 / 1 |

### Layer 68

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004418 / 1 | 0.004809 / 1 | 0.006072 / 1 |
| pre-attention-aggregation | boundary | 0.017543 / 1 | 0.017973 / 1 | 0.018455 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010911 / 1 | 0.010159 / 1 | 0.011561 / 1 |
| Q | nested | 2.500321 / 1 | 2.884420 / 1 | 26.423341 / 1 |
| K | nested | 2.351414 / 1 | 2.822143 / 1 | 5.276940 / 1 |
| V | nested | 2.240767 / 1 | 2.707759 / 1 | 2.965682 / 1 |
| B | nested | 0.021200 / 1 | 0.023604 / 1 | 0.198150 / 1 |
| FA | nested | 0.027602 / 1 | 0.029305 / 1 | 0.152064 / 1 |
| FB | nested | 0.050775 / 1 | 0.058068 / 1 | 0.264504 / 1 |
| G | nested | 1.922662 / 1 | 2.353237 / 1 | 14.061253 / 1 |
| O | nested | 2.043097 / 1 | 2.406116 / 1 | 10.919693 / 1 |
| attention | boundary | 12.299871 / 1 | 14.415104 / 1 | 61.131112 / 1 |
| attention-residual | boundary | 0.002614 / 1 | 0.002594 / 1 | 0.002254 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027161 / 1 | 0.027741 / 1 | 0.033753 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002415 / 1 | 0.002635 / 1 | 0.003336 / 1 |
| router-and-top16 | boundary | 0.600682 / 1 | 0.714215 / 1 | 2.685348 / 1 |
| EDOWN | nested | 0.812649 / 1 | 0.699838 / 1 | 3.159644 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.814141 / 1 | 0.701561 / 1 | 3.161407 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026850 / 1 | 0.006622 / 1 | 0.014728 / 1 |
| SH1 | nested | 1.150570 / 1 | 1.855597 / 1 | 2.635344 / 1 |
| SH3 | nested | 0.995670 / 1 | 2.203347 / 1 | 2.360631 / 1 |
| SH2 | nested | 0.988918 / 1 | 1.980360 / 1 | 2.151600 / 1 |
| shared-expert-during-read | boundary | 3.148844 / 1 | 6.055385 / 1 | 7.163866 / 1 |
| detail:expert-gate | nested | 26.941879 / 16 | 25.765023 / 16 | 24.672841 / 16 |
| detail:expert-up | nested | 25.759590 / 16 | 24.539402 / 16 | 23.940764 / 16 |
| detail:expert-activation | nested | 0.102311 / 16 | 0.100869 / 16 | 0.119031 / 16 |
| detail:expert-down | nested | 28.149637 / 16 | 24.736630 / 16 | 24.133622 / 16 |
| EUP | nested | 0.598037 / 1 | 0.967328 / 1 | 1.374618 / 1 |
| experts-mix-normalize-up | boundary | 81.989335 / 1 | 76.514836 / 1 | 74.671371 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003016 / 1 | 0.003737 / 1 | 0.002885 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000601 / 1 | 0.000140 / 1 | 0.000290 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.432538 / 1 | 0.443017 / 1 | 0.376925 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1025.182602 / 1 | 1024.494061 / 1 | 904.982718 / 1 |
| op:Q   int8 projection | nested | 15.702008 / 1 | 20.989479 / 1 | 71.941260 / 1 |
| op:X   mxfp4 expert proj | nested | 81.036174 / 1 | 75.225375 / 1 | 72.951245 / 1 |
| op:N   rmsnorm | nested | 0.035486 / 1 | 0.033262 / 1 | 0.024005 / 1 |
| op:L   l2 per-head | nested | 0.004870 / 1 | 0.004919 / 1 | 0.006062 / 1 |
| op:SiTU + sigma | nested | 0.110187 / 1 | 0.111360 / 1 | 0.129582 / 1 |
| op:C   shortconv | nested | 0.084607 / 1 | 0.055053 / 1 | 0.039623 / 1 |
| op:AR  snapshot aggregate | nested | 0.034794 / 1 | 0.035747 / 1 | 0.039885 / 1 |
| op:D   kda delta-rule | nested | 0.149519 / 1 | 0.145712 / 1 | 0.154799 / 1 |
| op:router dot product | nested | 0.597967 / 1 | 0.710858 / 1 | 2.682432 / 1 |
| op:top-k selection | nested | 0.002375 / 1 | 0.002845 / 1 | 0.002555 / 1 |
| op:alpha / beta / gate | nested | 0.074248 / 1 | 0.074349 / 1 | 0.056415 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 99.396710 / 1 | 98.936730 / 1 | 149.301565 / 1 |

### Layer 69

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005520 / 1 | 0.006583 / 1 | 0.006532 / 1 |
| pre-attention-aggregation | boundary | 0.021861 / 1 | 0.027191 / 1 | 0.021230 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010399 / 1 | 0.011421 / 1 | 0.011431 / 1 |
| Q | nested | 2.516412 / 1 | 2.878098 / 1 | 18.215896 / 1 |
| K | nested | 2.367955 / 1 | 2.790203 / 1 | 3.301088 / 1 |
| V | nested | 2.231911 / 1 | 2.707529 / 1 | 2.900249 / 1 |
| B | nested | 0.023173 / 1 | 0.164007 / 1 | 0.132057 / 1 |
| FA | nested | 0.026560 / 1 | 0.033833 / 1 | 0.162664 / 1 |
| FB | nested | 0.045655 / 1 | 0.176179 / 1 | 0.187129 / 1 |
| G | nested | 1.947058 / 1 | 3.892373 / 1 | 4.447861 / 1 |
| O | nested | 2.034682 / 1 | 4.181874 / 1 | 4.397777 / 1 |
| attention | boundary | 12.312977 / 1 | 18.063872 / 1 | 34.384346 / 1 |
| attention-residual | boundary | 0.002274 / 1 | 0.002375 / 1 | 0.002004 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027081 / 1 | 0.029084 / 1 | 0.036628 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002745 / 1 | 0.002385 / 1 | 0.201797 / 1 |
| router-and-top16 | boundary | 0.591235 / 1 | 0.701991 / 1 | 1.871206 / 1 |
| EDOWN | nested | 0.584432 / 1 | 0.667447 / 1 | 1.956175 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.586035 / 1 | 0.669060 / 1 | 1.957859 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027261 / 1 | 0.013285 / 1 | 0.007995 / 1 |
| SH1 | nested | 1.004908 / 1 | 1.191326 / 1 | 3.797735 / 1 |
| SH3 | nested | 1.018843 / 1 | 1.154627 / 1 | 3.947015 / 1 |
| SH2 | nested | 0.999717 / 1 | 1.164045 / 1 | 3.849783 / 1 |
| shared-expert-during-read | boundary | 3.037475 / 1 | 3.525918 / 1 | 11.611385 / 1 |
| detail:expert-gate | nested | 26.777594 / 16 | 25.883312 / 16 | 21.970960 / 16 |
| detail:expert-up | nested | 25.427068 / 16 | 24.854417 / 16 | 19.860609 / 16 |
| detail:expert-activation | nested | 0.099042 / 16 | 0.101553 / 16 | 0.123291 / 16 |
| detail:expert-down | nested | 26.897690 / 16 | 25.095409 / 16 | 20.217276 / 16 |
| EUP | nested | 0.787332 / 1 | 0.709606 / 1 | 1.734260 / 1 |
| experts-mix-normalize-up | boundary | 80.427416 / 1 | 77.078299 / 1 | 64.343364 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002886 / 1 | 0.002695 / 1 | 0.003146 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000281 / 1 | 0.000140 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.425014 / 1 | 0.435945 / 1 | 0.369280 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1043.637363 / 1 | 1036.148432 / 1 | 752.398874 / 1 |
| op:Q   int8 projection | nested | 15.587113 / 1 | 21.709243 / 1 | 49.027795 / 1 |
| op:X   mxfp4 expert proj | nested | 79.286882 / 1 | 76.019260 / 1 | 62.255742 / 1 |
| op:N   rmsnorm | nested | 0.037310 / 1 | 0.036558 / 1 | 0.027732 / 1 |
| op:L   l2 per-head | nested | 0.005089 / 1 | 0.005461 / 1 | 0.005891 / 1 |
| op:SiTU + sigma | nested | 0.106858 / 1 | 0.111629 / 1 | 0.134232 / 1 |
| op:C   shortconv | nested | 0.060662 / 1 | 0.058338 / 1 | 0.042970 / 1 |
| op:AR  snapshot aggregate | nested | 0.038562 / 1 | 0.045045 / 1 | 0.043140 / 1 |
| op:D   kda delta-rule | nested | 0.136375 / 1 | 0.147345 / 1 | 0.202709 / 1 |
| op:router dot product | nested | 0.588400 / 1 | 0.699236 / 1 | 1.868391 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002395 / 1 | 0.002414 / 1 |
| op:alpha / beta / gate | nested | 0.071534 / 1 | 0.075130 / 1 | 0.057648 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 97.497471 / 1 | 100.586954 / 1 | 114.845736 / 1 |

### Layer 70

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004629 / 1 | 0.005079 / 1 | 0.172993 / 1 |
| pre-attention-aggregation | boundary | 0.025598 / 1 | 0.028313 / 1 | 0.017794 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011252 / 1 | 0.011912 / 1 | 0.011341 / 1 |
| Q | nested | 3.129147 / 1 | 2.815080 / 1 | 17.589065 / 1 |
| K | nested | 3.057513 / 1 | 2.766379 / 1 | 3.234173 / 1 |
| V | nested | 3.005205 / 1 | 2.677042 / 1 | 2.907242 / 1 |
| B | nested | 0.023464 / 1 | 0.022041 / 1 | 0.281977 / 1 |
| FA | nested | 0.027611 / 1 | 0.029555 / 1 | 0.233406 / 1 |
| FB | nested | 0.062026 / 1 | 0.055684 / 1 | 0.241011 / 1 |
| G | nested | 2.600750 / 1 | 2.282224 / 1 | 8.507235 / 1 |
| O | nested | 2.617220 / 1 | 2.373325 / 1 | 7.567129 / 1 |
| attention | boundary | 15.611700 / 1 | 14.129310 / 1 | 41.202576 / 1 |
| attention-residual | boundary | 0.002435 / 1 | 0.002544 / 1 | 0.001973 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027902 / 1 | 0.028033 / 1 | 0.028813 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002495 / 1 | 0.002535 / 1 | 0.003486 / 1 |
| router-and-top16 | boundary | 0.597376 / 1 | 0.623535 / 1 | 4.612769 / 1 |
| EDOWN | nested | 0.779857 / 1 | 0.690941 / 1 | 4.128555 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.781269 / 1 | 0.692424 / 1 | 4.130268 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042179 / 1 | 0.022953 / 1 | 0.014807 / 1 |
| SH1 | nested | 1.312242 / 1 | 1.177660 / 1 | 13.388946 / 1 |
| SH3 | nested | 1.348490 / 1 | 1.160718 / 1 | 9.055580 / 1 |
| SH2 | nested | 1.354821 / 1 | 1.175727 / 1 | 5.184637 / 1 |
| shared-expert-during-read | boundary | 4.032405 / 1 | 3.530406 / 1 | 27.650484 / 1 |
| detail:expert-gate | nested | 26.830853 / 16 | 27.135232 / 16 | 31.219865 / 16 |
| detail:expert-up | nested | 25.974723 / 16 | 25.601366 / 16 | 30.411822 / 16 |
| detail:expert-activation | nested | 0.097695 / 16 | 0.101870 / 16 | 0.126748 / 16 |
| detail:expert-down | nested | 27.254354 / 16 | 26.081615 / 16 | 32.504102 / 16 |
| EUP | nested | 0.796989 / 1 | 0.583781 / 1 | 2.489301 / 1 |
| experts-mix-normalize-up | boundary | 81.409351 / 1 | 79.935177 / 1 | 97.179457 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002575 / 1 | 0.003006 / 1 | 0.002725 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000221 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427589 / 1 | 0.432688 / 1 | 0.398394 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1080.611685 / 1 | 1075.932575 / 1 | 1113.691993 / 1 |
| op:Q   int8 projection | nested | 20.113772 / 1 | 17.808374 / 1 | 74.806255 / 1 |
| op:X   mxfp4 expert proj | nested | 80.242670 / 1 | 79.006560 / 1 | 94.345303 / 1 |
| op:N   rmsnorm | nested | 0.037861 / 1 | 0.032511 / 1 | 0.024956 / 1 |
| op:L   l2 per-head | nested | 0.005200 / 1 | 0.005170 / 1 | 0.007183 / 1 |
| op:SiTU + sigma | nested | 0.108124 / 1 | 0.112080 / 1 | 0.141383 / 1 |
| op:C   shortconv | nested | 0.054151 / 1 | 0.067376 / 1 | 0.044333 / 1 |
| op:AR  snapshot aggregate | nested | 0.043411 / 1 | 0.045916 / 1 | 0.035596 / 1 |
| op:D   kda delta-rule | nested | 0.133008 / 1 | 0.142717 / 1 | 0.190266 / 1 |
| op:router dot product | nested | 0.594231 / 1 | 0.620429 / 1 | 4.609803 / 1 |
| op:top-k selection | nested | 0.002595 / 1 | 0.002575 / 1 | 0.002474 / 1 |
| op:alpha / beta / gate | nested | 0.074029 / 1 | 0.074499 / 1 | 0.064040 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 103.000233 / 1 | 99.465107 / 1 | 175.445624 / 1 |

### Layer 71

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005901 / 1 | 0.004959 / 1 | 0.007093 / 1 |
| pre-attention-aggregation | boundary | 0.026530 / 1 | 0.029896 / 1 | 0.021230 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010770 / 1 | 0.011502 / 1 | 0.011361 / 1 |
| QA | nested | 0.386192 / 1 | 0.291384 / 1 | 0.779467 / 1 |
| QB | nested | 0.891756 / 1 | 0.703835 / 1 | 2.162500 / 1 |
| KA | nested | 0.146093 / 1 | 0.102231 / 1 | 0.558003 / 1 |
| KB | nested | 0.329526 / 1 | 0.307354 / 1 | 0.896264 / 1 |
| G | nested | 2.777260 / 1 | 1.993054 / 1 | 7.433850 / 1 |
| O | nested | 2.838163 / 1 | 1.952287 / 1 | 7.368218 / 1 |
| attention | boundary | 7.470067 / 1 | 5.455012 / 1 | 19.310261 / 1 |
| attention-residual | boundary | 0.001954 / 1 | 0.002074 / 1 | 0.001884 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027592 / 1 | 0.026199 / 1 | 0.027732 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002434 / 1 | 0.002395 / 1 | 0.220031 / 1 |
| router-and-top16 | boundary | 0.585404 / 1 | 0.567440 / 1 | 2.808157 / 1 |
| EDOWN | nested | 0.775710 / 1 | 0.575625 / 1 | 2.193408 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.777172 / 1 | 0.577129 / 1 | 2.195012 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.017313 / 1 | 0.012012 / 1 | 0.006332 / 1 |
| SH1 | nested | 1.408311 / 1 | 0.967879 / 1 | 3.968134 / 1 |
| SH3 | nested | 1.312131 / 1 | 0.968951 / 1 | 4.073471 / 1 |
| SH2 | nested | 1.285282 / 1 | 0.981153 / 1 | 3.605447 / 1 |
| shared-expert-during-read | boundary | 4.020873 / 1 | 2.931277 / 1 | 11.663292 / 1 |
| detail:expert-gate | nested | 27.276224 / 16 | 28.080305 / 16 | 21.474345 / 16 |
| detail:expert-up | nested | 25.967289 / 16 | 27.135831 / 16 | 21.057042 / 16 |
| detail:expert-activation | nested | 0.097922 / 16 | 0.109975 / 16 | 0.123468 / 16 |
| detail:expert-down | nested | 27.714123 / 16 | 27.960014 / 16 | 19.519661 / 16 |
| EUP | nested | 0.773355 / 1 | 0.591546 / 1 | 2.257118 / 1 |
| experts-mix-normalize-up | boundary | 82.252125 / 1 | 84.285565 / 1 | 64.870929 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.002705 / 1 | 0.002455 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000130 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007544 / 1 | 0.008856 / 1 | 0.008626 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1075.993806 / 1 | 1074.185357 / 1 | 772.682199 / 1 |
| op:Q   int8 projection | nested | 12.922526 / 1 | 9.434135 / 1 | 35.294067 / 1 |
| op:X   mxfp4 expert proj | nested | 81.143254 / 1 | 83.373588 / 1 | 62.258581 / 1 |
| op:N   rmsnorm | nested | 0.024969 / 1 | 0.025327 / 1 | 0.023683 / 1 |
| op:SiTU + sigma | nested | 0.107429 / 1 | 0.117970 / 1 | 0.133848 / 1 |
| op:AR  snapshot aggregate | nested | 0.043271 / 1 | 0.045867 / 1 | 0.038373 / 1 |
| op:SA  softmax attention | nested | 0.007023 / 1 | 0.008906 / 1 | 0.011071 / 1 |
| op:router dot product | nested | 0.582729 / 1 | 0.564495 / 1 | 2.805101 / 1 |
| op:top-k selection | nested | 0.002295 / 1 | 0.002545 / 1 | 0.002645 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 95.224854 / 1 | 93.935075 / 1 | 101.171866 / 1 |

### Layer 72

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004439 / 1 | 0.004649 / 1 | 0.006092 / 1 |
| pre-attention-aggregation | boundary | 0.017612 / 1 | 0.017152 / 1 | 0.017693 / 1 |
| snapshot-push | boundary | 0.001593 / 1 | 0.001823 / 1 | 0.002735 / 1 |
| pre-attention-normalization | boundary | 0.009759 / 1 | 0.010369 / 1 | 0.134291 / 1 |
| Q | nested | 3.064867 / 1 | 2.537872 / 1 | 18.041831 / 1 |
| K | nested | 2.990628 / 1 | 2.381069 / 1 | 3.348456 / 1 |
| V | nested | 2.893737 / 1 | 2.234455 / 1 | 3.078021 / 1 |
| B | nested | 0.025949 / 1 | 0.022021 / 1 | 0.125324 / 1 |
| FA | nested | 0.034124 / 1 | 0.027311 / 1 | 0.130444 / 1 |
| FB | nested | 0.055954 / 1 | 0.054131 / 1 | 0.197980 / 1 |
| G | nested | 2.673646 / 1 | 1.956876 / 1 | 6.705629 / 1 |
| O | nested | 2.726784 / 1 | 2.018782 / 1 | 6.491298 / 1 |
| attention | boundary | 15.559582 / 1 | 12.313968 / 1 | 38.852655 / 1 |
| attention-residual | boundary | 0.002154 / 1 | 0.001964 / 1 | 0.001783 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028033 / 1 | 0.028594 / 1 | 0.028473 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002424 / 1 | 0.002825 / 1 | 0.165619 / 1 |
| router-and-top16 | boundary | 0.596134 / 1 | 0.599481 / 1 | 2.270021 / 1 |
| EDOWN | nested | 0.814992 / 1 | 0.598198 / 1 | 2.385728 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.816746 / 1 | 0.599811 / 1 | 2.387671 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032060 / 1 | 0.006482 / 1 | 0.006252 / 1 |
| SH1 | nested | 1.356464 / 1 | 1.007613 / 1 | 2.026687 / 1 |
| SH3 | nested | 1.429131 / 1 | 1.013263 / 1 | 2.452082 / 1 |
| SH2 | nested | 1.414172 / 1 | 0.991102 / 1 | 2.066782 / 1 |
| shared-expert-during-read | boundary | 4.219023 / 1 | 3.026194 / 1 | 6.562352 / 1 |
| detail:expert-gate | nested | 26.466013 / 16 | 27.357323 / 16 | 27.999899 / 16 |
| detail:expert-up | nested | 25.603388 / 16 | 26.767626 / 16 | 26.673849 / 16 |
| detail:expert-activation | nested | 0.097570 / 16 | 0.104345 / 16 | 0.118229 / 16 |
| detail:expert-down | nested | 26.555700 / 16 | 28.673225 / 16 | 25.421987 / 16 |
| EUP | nested | 0.595734 / 1 | 0.602846 / 1 | 1.274010 / 1 |
| experts-mix-normalize-up | boundary | 79.767664 / 1 | 83.911856 / 1 | 81.909926 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003146 / 1 | 0.002975 / 1 | 0.002796 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000141 / 1 | 0.000290 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.430895 / 1 | 0.440794 / 1 | 0.429392 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1043.409897 / 1 | 1039.009053 / 1 | 1049.933854 / 1 |
| op:Q   int8 projection | nested | 20.074529 / 1 | 15.444079 / 1 | 48.322470 / 1 |
| op:X   mxfp4 expert proj | nested | 78.809552 / 1 | 82.989172 / 1 | 80.300610 / 1 |
| op:N   rmsnorm | nested | 0.034955 / 1 | 0.030387 / 1 | 0.024928 / 1 |
| op:L   l2 per-head | nested | 0.005811 / 1 | 0.005069 / 1 | 0.005611 / 1 |
| op:SiTU + sigma | nested | 0.110776 / 1 | 0.111138 / 1 | 0.129049 / 1 |
| op:C   shortconv | nested | 0.065431 / 1 | 0.059160 / 1 | 0.038562 / 1 |
| op:AR  snapshot aggregate | nested | 0.035496 / 1 | 0.035416 / 1 | 0.035497 / 1 |
| op:D   kda delta-rule | nested | 0.144460 / 1 | 0.144691 / 1 | 0.148587 / 1 |
| op:router dot product | nested | 0.593149 / 1 | 0.596634 / 1 | 2.267316 / 1 |
| op:top-k selection | nested | 0.002675 / 1 | 0.002374 / 1 | 0.002324 / 1 |
| op:alpha / beta / gate | nested | 0.073437 / 1 | 0.072536 / 1 | 0.056786 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 101.509318 / 1 | 100.988244 / 1 | 132.796847 / 1 |

### Layer 73

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005030 / 1 | 0.005460 / 1 | 0.007454 / 1 |
| pre-attention-aggregation | boundary | 0.025157 / 1 | 0.036228 / 1 | 0.030026 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011301 / 1 | 0.013294 / 1 | 0.011942 / 1 |
| Q | nested | 2.557980 / 1 | 2.522083 / 1 | 22.256816 / 1 |
| K | nested | 2.369307 / 1 | 2.395757 / 1 | 5.332163 / 1 |
| V | nested | 2.242641 / 1 | 2.306700 / 1 | 2.914055 / 1 |
| B | nested | 0.024917 / 1 | 0.024246 / 1 | 0.134632 / 1 |
| FA | nested | 0.024536 / 1 | 0.024936 / 1 | 0.108653 / 1 |
| FB | nested | 0.049472 / 1 | 0.063529 / 1 | 0.166151 / 1 |
| G | nested | 1.939153 / 1 | 1.944002 / 1 | 3.773110 / 1 |
| O | nested | 2.058877 / 1 | 2.237070 / 1 | 4.184578 / 1 |
| attention | boundary | 12.377577 / 1 | 12.608609 / 1 | 39.668039 / 1 |
| attention-residual | boundary | 0.002435 / 1 | 0.002244 / 1 | 0.002204 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026980 / 1 | 0.028754 / 1 | 0.029856 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002374 / 1 | 0.002765 / 1 | 0.002996 / 1 |
| router-and-top16 | boundary | 0.592277 / 1 | 0.619558 / 1 | 1.145560 / 1 |
| EDOWN | nested | 0.589100 / 1 | 0.589431 / 1 | 1.328202 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590763 / 1 | 0.591145 / 1 | 1.329935 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027372 / 1 | 0.012483 / 1 | 0.006262 / 1 |
| SH1 | nested | 0.999097 / 1 | 0.996262 / 1 | 2.167430 / 1 |
| SH3 | nested | 1.033160 / 1 | 1.259574 / 1 | 2.046814 / 1 |
| SH2 | nested | 0.987175 / 1 | 1.143617 / 1 | 2.148936 / 1 |
| shared-expert-during-read | boundary | 3.033137 / 1 | 3.412737 / 1 | 6.380932 / 1 |
| detail:expert-gate | nested | 27.892046 / 16 | 28.392733 / 16 | 32.633322 / 16 |
| detail:expert-up | nested | 27.011723 / 16 | 27.251900 / 16 | 30.091643 / 16 |
| detail:expert-activation | nested | 0.109235 / 16 | 0.104003 / 16 | 0.121728 / 16 |
| detail:expert-down | nested | 28.620768 / 16 | 28.779488 / 16 | 28.576593 / 16 |
| EUP | nested | 0.595283 / 1 | 0.616692 / 1 | 1.457965 / 1 |
| experts-mix-normalize-up | boundary | 84.670896 / 1 | 85.556991 / 1 | 93.277455 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.002535 / 1 | 0.002835 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000331 / 1 | 0.000140 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.422770 / 1 | 0.424413 / 1 | 0.384408 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1076.270126 / 1 | 1071.278616 / 1 | 1127.176577 / 1 |
| op:Q   int8 projection | nested | 15.469192 / 1 | 16.122475 / 1 | 48.017700 / 1 |
| op:X   mxfp4 expert proj | nested | 83.727863 / 1 | 84.617626 / 1 | 91.513509 / 1 |
| op:N   rmsnorm | nested | 0.035415 / 1 | 0.036830 / 1 | 0.025387 / 1 |
| op:L   l2 per-head | nested | 0.005060 / 1 | 0.005250 / 1 | 0.005971 / 1 |
| op:SiTU + sigma | nested | 0.116929 / 1 | 0.111419 / 1 | 0.133488 / 1 |
| op:C   shortconv | nested | 0.054101 / 1 | 0.060583 / 1 | 0.039544 / 1 |
| op:AR  snapshot aggregate | nested | 0.042550 / 1 | 0.054552 / 1 | 0.048861 / 1 |
| op:D   kda delta-rule | nested | 0.138268 / 1 | 0.136124 / 1 | 0.124743 / 1 |
| op:router dot product | nested | 0.589592 / 1 | 0.616742 / 1 | 1.142615 / 1 |
| op:top-k selection | nested | 0.002395 / 1 | 0.002415 / 1 | 0.002585 / 1 |
| op:alpha / beta / gate | nested | 0.073437 / 1 | 0.073858 / 1 | 0.057577 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 101.808807 / 1 | 103.334047 / 1 | 142.298760 / 1 |

### Layer 74

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005480 / 1 | 0.005290 / 1 | 0.006081 / 1 |
| pre-attention-aggregation | boundary | 0.026700 / 1 | 0.034084 / 1 | 0.020909 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012603 / 1 | 0.012363 / 1 | 0.010930 / 1 |
| Q | nested | 2.507235 / 1 | 2.554633 / 1 | 24.700773 / 1 |
| K | nested | 2.398913 / 1 | 2.429379 / 1 | 4.090092 / 1 |
| V | nested | 2.228994 / 1 | 2.263820 / 1 | 3.007750 / 1 |
| B | nested | 0.023764 / 1 | 0.022883 / 1 | 0.253183 / 1 |
| FA | nested | 0.027100 / 1 | 0.027110 / 1 | 0.195295 / 1 |
| FB | nested | 0.045706 / 1 | 0.047318 / 1 | 0.363038 / 1 |
| G | nested | 1.950605 / 1 | 1.966455 / 1 | 4.678722 / 1 |
| O | nested | 2.029943 / 1 | 2.029782 / 1 | 4.160454 / 1 |
| attention | boundary | 12.280436 / 1 | 12.419476 / 1 | 42.092198 / 1 |
| attention-residual | boundary | 0.002514 / 1 | 0.002234 / 1 | 0.002284 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027611 / 1 | 0.027622 / 1 | 0.029786 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002425 / 1 | 0.002545 / 1 | 0.003617 / 1 |
| router-and-top16 | boundary | 0.592878 / 1 | 0.598769 / 1 | 0.906674 / 1 |
| EDOWN | nested | 0.589701 / 1 | 0.589031 / 1 | 1.076361 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.591315 / 1 | 0.590684 / 1 | 1.078285 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042219 / 1 | 0.022632 / 1 | 0.013445 / 1 |
| SH1 | nested | 1.003636 / 1 | 1.001912 / 1 | 2.189591 / 1 |
| SH3 | nested | 1.013434 / 1 | 1.008805 / 1 | 2.127124 / 1 |
| SH2 | nested | 1.009035 / 1 | 0.988047 / 1 | 2.691299 / 1 |
| shared-expert-during-read | boundary | 3.040171 / 1 | 3.012349 / 1 | 7.025667 / 1 |
| detail:expert-gate | nested | 25.371083 / 16 | 25.547554 / 16 | 29.528841 / 16 |
| detail:expert-up | nested | 24.364901 / 16 | 24.951864 / 16 | 28.325096 / 16 |
| detail:expert-activation | nested | 0.102402 / 16 | 0.102632 / 16 | 0.119974 / 16 |
| detail:expert-down | nested | 26.634156 / 16 | 26.929567 / 16 | 30.976419 / 16 |
| EUP | nested | 0.602967 / 1 | 0.816476 / 1 | 1.266667 / 1 |
| experts-mix-normalize-up | boundary | 77.530313 / 1 | 78.762545 / 1 | 90.636722 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002705 / 1 | 0.002395 / 1 | 0.002765 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000331 / 1 | 0.000220 / 1 | 0.000561 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.435363 / 1 | 0.447286 / 1 | 0.374540 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 962.912972 / 1 | 956.382320 / 1 | 1088.235204 / 1 |
| op:Q   int8 projection | nested | 15.429268 / 1 | 15.744157 / 1 | 50.798705 / 1 |
| op:X   mxfp4 expert proj | nested | 76.560861 / 1 | 77.622313 / 1 | 89.040649 / 1 |
| op:N   rmsnorm | nested | 0.038202 / 1 | 0.035277 / 1 | 0.025196 / 1 |
| op:L   l2 per-head | nested | 0.005140 / 1 | 0.005029 / 1 | 0.006042 / 1 |
| op:SiTU + sigma | nested | 0.109878 / 1 | 0.110257 / 1 | 0.125736 / 1 |
| op:C   shortconv | nested | 0.053441 / 1 | 0.058719 / 1 | 0.041758 / 1 |
| op:AR  snapshot aggregate | nested | 0.044333 / 1 | 0.051937 / 1 | 0.039093 / 1 |
| op:D   kda delta-rule | nested | 0.132548 / 1 | 0.134381 / 1 | 0.197879 / 1 |
| op:router dot product | nested | 0.590313 / 1 | 0.595974 / 1 | 0.903839 / 1 |
| op:top-k selection | nested | 0.002265 / 1 | 0.002374 / 1 | 0.002274 / 1 |
| op:alpha / beta / gate | nested | 0.070612 / 1 | 0.091541 / 1 | 0.061865 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 94.610778 / 1 | 95.958286 / 1 | 142.222909 / 1 |

### Layer 75

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006242 / 1 | 0.006312 / 1 | 0.007213 / 1 |
| pre-attention-aggregation | boundary | 0.026168 / 1 | 0.033272 / 1 | 0.021150 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011552 / 1 | 0.010850 / 1 | 0.011431 / 1 |
| QA | nested | 0.288659 / 1 | 0.354652 / 1 | 0.635157 / 1 |
| QB | nested | 0.697994 / 1 | 0.843365 / 1 | 1.320959 / 1 |
| KA | nested | 0.101560 / 1 | 0.137267 / 1 | 0.319808 / 1 |
| KB | nested | 0.306243 / 1 | 0.359311 / 1 | 0.656146 / 1 |
| G | nested | 1.977224 / 1 | 2.690988 / 1 | 4.768079 / 1 |
| O | nested | 1.931700 / 1 | 2.846840 / 1 | 5.092786 / 1 |
| attention | boundary | 5.405901 / 1 | 7.332921 / 1 | 12.903209 / 1 |
| attention-residual | boundary | 0.002294 / 1 | 0.002064 / 1 | 0.001754 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026780 / 1 | 0.027461 / 1 | 0.029625 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002184 / 1 | 0.002455 / 1 | 0.003376 / 1 |
| router-and-top16 | boundary | 0.566258 / 1 | 0.599761 / 1 | 0.895854 / 1 |
| EDOWN | nested | 0.567490 / 1 | 0.844579 / 1 | 1.340304 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.568933 / 1 | 0.846141 / 1 | 1.341997 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032901 / 1 | 0.006372 / 1 | 0.007353 / 1 |
| SH1 | nested | 0.962349 / 1 | 1.386050 / 1 | 2.415574 / 1 |
| SH3 | nested | 0.962088 / 1 | 1.432707 / 1 | 2.351484 / 1 |
| SH2 | nested | 0.969772 / 1 | 1.463063 / 1 | 2.037376 / 1 |
| shared-expert-during-read | boundary | 2.907283 / 1 | 4.297540 / 1 | 6.821726 / 1 |
| detail:expert-gate | nested | 21.992683 / 16 | 21.254023 / 16 | 28.303262 / 16 |
| detail:expert-up | nested | 21.275111 / 16 | 20.490174 / 16 | 27.444437 / 16 |
| detail:expert-activation | nested | 0.097474 / 16 | 0.095418 / 16 | 0.116228 / 16 |
| detail:expert-down | nested | 22.899966 / 16 | 21.795953 / 16 | 26.400568 / 16 |
| EUP | nested | 0.839870 / 1 | 0.580415 / 1 | 1.233935 / 1 |
| experts-mix-normalize-up | boundary | 67.539586 / 1 | 64.613889 / 1 | 83.931814 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.002975 / 1 | 0.002735 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000320 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008797 / 1 | 0.008466 / 1 | 0.009227 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 867.786227 / 1 | 868.798541 / 1 | 1077.496386 / 1 |
| op:Q   int8 projection | nested | 9.603696 / 1 | 12.937883 / 1 | 22.169815 / 1 |
| op:X   mxfp4 expert proj | nested | 66.353711 / 1 | 63.723566 / 1 | 82.355367 / 1 |
| op:N   rmsnorm | nested | 0.024345 / 1 | 0.023012 / 1 | 0.023895 / 1 |
| op:SiTU + sigma | nested | 0.105017 / 1 | 0.104865 / 1 | 0.126579 / 1 |
| op:AR  snapshot aggregate | nested | 0.043030 / 1 | 0.050474 / 1 | 0.039565 / 1 |
| op:SA  softmax attention | nested | 0.006773 / 1 | 0.009547 / 1 | 0.012633 / 1 |
| op:router dot product | nested | 0.563373 / 1 | 0.596855 / 1 | 0.893009 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002385 / 1 | 0.002605 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 77.126539 / 1 | 77.807431 / 1 | 106.007932 / 1 |

### Layer 76

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004349 / 1 | 0.005921 / 1 | 0.007264 / 1 |
| pre-attention-aggregation | boundary | 0.018003 / 1 | 0.017263 / 1 | 0.019065 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010029 / 1 | 0.009898 / 1 | 0.011381 / 1 |
| Q | nested | 3.145778 / 1 | 2.478711 / 1 | 34.727447 / 1 |
| K | nested | 3.056160 / 1 | 2.363987 / 1 | 3.629071 / 1 |
| V | nested | 3.069465 / 1 | 2.229526 / 1 | 2.884559 / 1 |
| B | nested | 0.030226 / 1 | 0.022823 / 1 | 0.031158 / 1 |
| FA | nested | 0.037821 / 1 | 0.026058 / 1 | 0.036648 / 1 |
| FB | nested | 0.050264 / 1 | 0.048201 / 1 | 0.147846 / 1 |
| G | nested | 2.781568 / 1 | 3.144195 / 1 | 6.347320 / 1 |
| O | nested | 2.847941 / 1 | 2.973246 / 1 | 11.603871 / 1 |
| attention | boundary | 16.121883 / 1 | 14.354751 / 1 | 60.195413 / 1 |
| attention-residual | boundary | 0.002204 / 1 | 0.002424 / 1 | 0.001853 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028964 / 1 | 0.027251 / 1 | 0.029576 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002404 / 1 | 0.003106 / 1 | 0.003767 / 1 |
| router-and-top16 | boundary | 0.603728 / 1 | 0.793021 / 1 | 2.952948 / 1 |
| EDOWN | nested | 0.832085 / 1 | 0.890544 / 1 | 4.012757 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.833628 / 1 | 0.892077 / 1 | 4.014651 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042980 / 1 | 0.006803 / 1 | 0.006793 / 1 |
| SH1 | nested | 1.417980 / 1 | 1.539717 / 1 | 4.811470 / 1 |
| SH3 | nested | 1.406027 / 1 | 1.686422 / 1 | 4.538591 / 1 |
| SH2 | nested | 1.422969 / 1 | 1.499111 / 1 | 4.972762 / 1 |
| shared-expert-during-read | boundary | 4.262675 / 1 | 4.738964 / 1 | 14.340084 / 1 |
| detail:expert-gate | nested | 24.912596 / 16 | 27.220692 / 16 | 46.138488 / 16 |
| detail:expert-up | nested | 23.935164 / 16 | 25.304712 / 16 | 40.462074 / 16 |
| detail:expert-activation | nested | 0.092793 / 16 | 0.104333 / 16 | 0.116088 / 16 |
| detail:expert-down | nested | 25.271317 / 16 | 26.906366 / 16 | 42.118476 / 16 |
| EUP | nested | 0.653341 / 1 | 1.052597 / 1 | 4.582562 / 1 |
| experts-mix-normalize-up | boundary | 75.322147 / 1 | 81.003252 / 1 | 141.341401 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002264 / 1 | 0.003106 / 1 | 0.002605 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000300 / 1 | 0.000130 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429593 / 1 | 0.437808 / 1 | 0.442246 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1000.602267 / 1 | 1009.666255 / 1 | 1366.540971 / 1 |
| op:Q   int8 projection | nested | 20.750114 / 1 | 19.953352 / 1 | 82.324101 / 1 |
| op:X   mxfp4 expert proj | nested | 74.302539 / 1 | 79.627270 / 1 | 128.926677 / 1 |
| op:N   rmsnorm | nested | 0.032130 / 1 | 0.033653 / 1 | 0.024636 / 1 |
| op:L   l2 per-head | nested | 0.005510 / 1 | 0.005390 / 1 | 0.005420 / 1 |
| op:SiTU + sigma | nested | 0.102772 / 1 | 0.111715 / 1 | 0.127119 / 1 |
| op:C   shortconv | nested | 0.060744 / 1 | 0.047109 / 1 | 0.038562 / 1 |
| op:AR  snapshot aggregate | nested | 0.037049 / 1 | 0.034885 / 1 | 0.036989 / 1 |
| op:D   kda delta-rule | nested | 0.146594 / 1 | 0.143969 / 1 | 0.157775 / 1 |
| op:router dot product | nested | 0.601184 / 1 | 0.790327 / 1 | 2.950072 / 1 |
| op:top-k selection | nested | 0.002144 / 1 | 0.002314 / 1 | 0.002324 / 1 |
| op:alpha / beta / gate | nested | 0.071334 / 1 | 0.072676 / 1 | 0.056716 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 97.703086 / 1 | 102.313289 / 1 | 223.388083 / 1 |

### Layer 77

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004699 / 1 | 0.007544 / 1 | 0.007013 / 1 |
| pre-attention-aggregation | boundary | 0.033473 / 1 | 0.028453 / 1 | 0.029225 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011211 / 1 | 0.011321 / 1 | 0.586686 / 1 |
| Q | nested | 2.515960 / 1 | 2.545557 / 1 | 192.533563 / 1 |
| K | nested | 2.375268 / 1 | 2.381711 / 1 | 3.074685 / 1 |
| V | nested | 2.228774 / 1 | 2.238813 / 1 | 2.791676 / 1 |
| B | nested | 0.023594 / 1 | 0.025918 / 1 | 1.681502 / 1 |
| FA | nested | 0.025167 / 1 | 0.032972 / 1 | 3.209156 / 1 |
| FB | nested | 0.047830 / 1 | 0.071914 / 1 | 0.237213 / 1 |
| G | nested | 1.932460 / 1 | 3.140247 / 1 | 18.599102 / 1 |
| O | nested | 2.022459 / 1 | 3.515829 / 1 | 20.132557 / 1 |
| attention | boundary | 12.241272 / 1 | 15.021437 / 1 | 243.230428 / 1 |
| attention-residual | boundary | 0.002235 / 1 | 0.002445 / 1 | 0.001593 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027732 / 1 | 0.027121 / 1 | 0.035035 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002514 / 1 | 0.002444 / 1 | 0.174836 / 1 |
| router-and-top16 | boundary | 0.589692 / 1 | 0.705057 / 1 | 12.496139 / 1 |
| EDOWN | nested | 0.585454 / 1 | 0.943172 / 1 | 9.860794 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.587138 / 1 | 0.944815 / 1 | 9.862958 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027561 / 1 | 0.021300 / 1 | 0.006312 / 1 |
| SH1 | nested | 0.998837 / 1 | 1.859825 / 1 | 12.688337 / 1 |
| SH3 | nested | 0.984310 / 1 | 1.914798 / 1 | 11.317547 / 1 |
| SH2 | nested | 0.976634 / 1 | 2.154045 / 1 | 8.205692 / 1 |
| shared-expert-during-read | boundary | 2.973005 / 1 | 5.943044 / 1 | 32.247664 / 1 |
| detail:expert-gate | nested | 26.549367 / 16 | 27.605981 / 16 | 70.555110 / 16 |
| detail:expert-up | nested | 25.722634 / 16 | 24.538079 / 16 | 92.257061 / 16 |
| detail:expert-activation | nested | 0.101057 / 16 | 0.101092 / 16 | 0.127091 / 16 |
| detail:expert-down | nested | 27.134900 / 16 | 26.099265 / 16 | 74.464924 / 16 |
| EUP | nested | 0.591215 / 1 | 1.006380 / 1 | 5.475300 / 1 |
| experts-mix-normalize-up | boundary | 80.528254 / 1 | 79.772933 / 1 | 249.317331 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002424 / 1 | 0.002986 / 1 | 0.003186 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000130 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.425184 / 1 | 0.440032 / 1 | 0.476460 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 984.563293 / 1 | 1081.203536 / 1 | 2620.653866 / 1 |
| op:Q   int8 projection | nested | 15.306458 / 1 | 21.828977 / 1 | 289.805000 / 1 |
| op:X   mxfp4 expert proj | nested | 79.601894 / 1 | 78.437618 / 1 | 237.495552 / 1 |
| op:N   rmsnorm | nested | 0.032932 / 1 | 0.031558 / 1 | 0.024786 / 1 |
| op:L   l2 per-head | nested | 0.007755 / 1 | 0.006051 / 1 | 0.005691 / 1 |
| op:SiTU + sigma | nested | 0.108433 / 1 | 0.109104 / 1 | 0.154240 / 1 |
| op:C   shortconv | nested | 0.057618 / 1 | 0.048130 / 1 | 0.045053 / 1 |
| op:AR  snapshot aggregate | nested | 0.051345 / 1 | 0.046177 / 1 | 0.052788 / 1 |
| op:D   kda delta-rule | nested | 0.135824 / 1 | 0.138680 / 1 | 0.157965 / 1 |
| op:router dot product | nested | 0.586686 / 1 | 0.702092 / 1 | 12.492943 / 1 |
| op:top-k selection | nested | 0.002545 / 1 | 0.002525 / 1 | 0.002585 / 1 |
| op:alpha / beta / gate | nested | 0.073437 / 1 | 0.073317 / 1 | 0.056425 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 97.474639 / 1 | 102.951271 / 1 | 548.497219 / 1 |

### Layer 78

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005400 / 1 | 0.010210 / 1 | 0.007064 / 1 |
| pre-attention-aggregation | boundary | 0.022543 / 1 | 0.031810 / 1 | 0.031839 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010890 / 1 | 0.011902 / 1 | 0.230721 / 1 |
| Q | nested | 2.515230 / 1 | 3.111664 / 1 | 173.123195 / 1 |
| K | nested | 2.395626 / 1 | 2.995296 / 1 | 3.070407 / 1 |
| V | nested | 2.239034 / 1 | 2.939803 / 1 | 2.812735 / 1 |
| B | nested | 0.022251 / 1 | 0.026550 / 1 | 1.780477 / 1 |
| FA | nested | 0.027291 / 1 | 0.164828 / 1 | 0.136525 / 1 |
| FB | nested | 0.044894 / 1 | 0.135273 / 1 | 3.723718 / 1 |
| G | nested | 1.927982 / 1 | 3.152621 / 1 | 20.261278 / 1 |
| O | nested | 2.026727 / 1 | 3.218544 / 1 | 29.920335 / 1 |
| attention | boundary | 12.302106 / 1 | 16.800502 / 1 | 235.659201 / 1 |
| attention-residual | boundary | 0.002435 / 1 | 0.002515 / 1 | 0.001963 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028583 / 1 | 0.029164 / 1 | 0.035716 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002485 / 1 | 0.002875 / 1 | 0.003466 / 1 |
| router-and-top16 | boundary | 0.591986 / 1 | 0.892959 / 1 | 10.235344 / 1 |
| EDOWN | nested | 0.585804 / 1 | 0.877960 / 1 | 12.629177 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.587357 / 1 | 0.879804 / 1 | 12.631051 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027482 / 1 | 0.012945 / 1 | 0.006292 / 1 |
| SH1 | nested | 0.996602 / 1 | 2.158383 / 1 | 7.757174 / 1 |
| SH3 | nested | 1.019445 / 1 | 1.818568 / 1 | 10.411173 / 1 |
| SH2 | nested | 0.985030 / 1 | 2.000588 / 1 | 8.255906 / 1 |
| shared-expert-during-read | boundary | 3.014783 / 1 | 5.994630 / 1 | 26.446053 / 1 |
| detail:expert-gate | nested | 26.179054 / 16 | 26.762997 / 16 | 72.438348 / 16 |
| detail:expert-up | nested | 22.608292 / 16 | 24.278203 / 16 | 65.774960 / 16 |
| detail:expert-activation | nested | 0.116727 / 16 | 0.099735 / 16 | 0.116656 / 16 |
| detail:expert-down | nested | 23.520076 / 16 | 25.838772 / 16 | 68.876254 / 16 |
| EUP | nested | 0.763907 / 1 | 1.435462 / 1 | 9.802246 / 1 |
| experts-mix-normalize-up | boundary | 73.685920 / 1 | 78.831073 / 1 | 218.430310 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003036 / 1 | 0.002415 / 1 | 0.002996 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000221 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427880 / 1 | 0.369591 / 1 | 0.403755 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 916.011195 / 1 | 1038.981576 / 1 | 2309.622480 / 1 |
| op:Q   int8 projection | nested | 15.548330 / 1 | 24.033494 / 1 | 283.682152 / 1 |
| op:X   mxfp4 expert proj | nested | 72.548915 / 1 | 77.070696 / 1 | 207.299213 / 1 |
| op:N   rmsnorm | nested | 0.035076 / 1 | 0.034914 / 1 | 0.026409 / 1 |
| op:L   l2 per-head | nested | 0.005019 / 1 | 0.005540 / 1 | 0.005119 / 1 |
| op:SiTU + sigma | nested | 0.123400 / 1 | 0.110587 / 1 | 0.131776 / 1 |
| op:C   shortconv | nested | 0.072917 / 1 | 0.055775 / 1 | 0.040725 / 1 |
| op:AR  snapshot aggregate | nested | 0.040586 / 1 | 0.050693 / 1 | 0.055084 / 1 |
| op:D   kda delta-rule | nested | 0.131766 / 1 | 0.133399 / 1 | 0.152826 / 1 |
| op:router dot product | nested | 0.589120 / 1 | 0.889963 / 1 | 10.232519 / 1 |
| op:top-k selection | nested | 0.002465 / 1 | 0.002544 / 1 | 0.002455 / 1 |
| op:alpha / beta / gate | nested | 0.073498 / 1 | 0.071904 / 1 | 0.058219 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 90.731830 / 1 | 103.893493 / 1 | 504.146131 / 1 |

### Layer 79

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005120 / 1 | 0.006021 / 1 | 0.006602 / 1 |
| pre-attention-aggregation | boundary | 0.025177 / 1 | 0.021079 / 1 | 0.023555 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011511 / 1 | 0.010179 / 1 | 2.102469 / 1 |
| QA | nested | 0.322282 / 1 | 0.368498 / 1 | 5.145034 / 1 |
| QB | nested | 0.983168 / 1 | 0.851221 / 1 | 12.417883 / 1 |
| KA | nested | 0.118962 / 1 | 0.139531 / 1 | 1.788111 / 1 |
| KB | nested | 0.409295 / 1 | 0.358670 / 1 | 4.548949 / 1 |
| G | nested | 2.421404 / 1 | 3.999002 / 1 | 21.930076 / 1 |
| O | nested | 2.493910 / 1 | 3.737383 / 1 | 36.893585 / 1 |
| attention | boundary | 6.851021 / 1 | 9.554201 / 1 | 83.000764 / 1 |
| attention-residual | boundary | 0.001703 / 1 | 0.002124 / 1 | 0.001152 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027842 / 1 | 0.028964 / 1 | 0.028383 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002144 / 1 | 0.002104 / 1 | 1.135822 / 1 |
| router-and-top16 | boundary | 0.569104 / 1 | 0.661476 / 1 | 12.652802 / 1 |
| EDOWN | nested | 0.579894 / 1 | 1.133608 / 1 | 11.938968 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.581547 / 1 | 1.135251 / 1 | 11.940861 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048050 / 1 | 0.006482 / 1 | 0.006752 / 1 |
| SH1 | nested | 0.976585 / 1 | 1.840249 / 1 | 18.976988 / 1 |
| SH3 | nested | 1.166429 / 1 | 1.610328 / 1 | 21.421205 / 1 |
| SH2 | nested | 1.259373 / 1 | 1.355533 / 1 | 12.253395 / 1 |
| shared-expert-during-read | boundary | 3.416364 / 1 | 4.822120 / 1 | 52.669070 / 1 |
| detail:expert-gate | nested | 22.280910 / 16 | 27.494704 / 16 | 46.617636 / 16 |
| detail:expert-up | nested | 21.219641 / 16 | 24.085874 / 16 | 41.829008 / 16 |
| detail:expert-activation | nested | 0.092744 / 16 | 0.106038 / 16 | 0.120665 / 16 |
| detail:expert-down | nested | 21.025597 / 16 | 26.137948 / 16 | 36.206460 / 16 |
| EUP | nested | 0.648281 / 1 | 1.243363 / 1 | 2.392500 / 1 |
| experts-mix-normalize-up | boundary | 65.719495 / 1 | 79.460950 / 1 | 127.675597 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002425 / 1 | 0.002896 / 1 | 0.003376 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000411 / 1 | 0.000130 / 1 | 0.000291 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008526 / 1 | 0.008917 / 1 | 0.008927 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 855.390374 / 1 | 1046.204574 / 1 | 1524.195665 / 1 |
| op:Q   int8 projection | nested | 11.378372 / 1 | 16.635933 / 1 | 149.704630 / 1 |
| op:X   mxfp4 expert proj | nested | 64.711502 / 1 | 77.917625 / 1 | 124.866328 / 1 |
| op:N   rmsnorm | nested | 0.024174 / 1 | 0.023003 / 1 | 0.025829 / 1 |
| op:SiTU + sigma | nested | 0.100068 / 1 | 0.115966 / 1 | 0.130525 / 1 |
| op:AR  snapshot aggregate | nested | 0.042430 / 1 | 0.039585 / 1 | 0.041647 / 1 |
| op:SA  softmax attention | nested | 0.006923 / 1 | 0.008747 / 1 | 0.021460 / 1 |
| op:router dot product | nested | 0.565036 / 1 | 0.658360 / 1 | 12.647551 / 1 |
| op:top-k selection | nested | 0.003717 / 1 | 0.002635 / 1 | 0.004428 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 77.288932 / 1 | 95.741189 / 1 | 291.276630 / 1 |

### Layer 80

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004489 / 1 | 0.005670 / 1 | 0.176680 / 1 |
| pre-attention-aggregation | boundary | 0.017633 / 1 | 0.019387 / 1 | 0.019827 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.009878 / 1 | 0.010580 / 1 | 0.203571 / 1 |
| Q | nested | 2.503428 / 1 | 3.010715 / 1 | 46.934666 / 1 |
| K | nested | 2.362674 / 1 | 3.027357 / 1 | 2.973937 / 1 |
| V | nested | 2.204018 / 1 | 2.883608 / 1 | 2.814309 / 1 |
| B | nested | 0.022562 / 1 | 0.027772 / 1 | 0.235370 / 1 |
| FA | nested | 0.026149 / 1 | 0.035195 / 1 | 0.208449 / 1 |
| FB | nested | 0.056656 / 1 | 0.073196 / 1 | 0.342710 / 1 |
| G | nested | 1.915349 / 1 | 3.123797 / 1 | 7.185455 / 1 |
| O | nested | 2.015667 / 1 | 3.105893 / 1 | 11.670546 / 1 |
| attention | boundary | 12.193052 / 1 | 16.292382 / 1 | 73.109282 / 1 |
| attention-residual | boundary | 0.001693 / 1 | 0.006552 / 1 | 0.002555 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028614 / 1 | 0.031990 / 1 | 0.029976 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002445 / 1 | 0.002625 / 1 | 0.336650 / 1 |
| router-and-top16 | boundary | 0.593359 / 1 | 0.650165 / 1 | 3.359006 / 1 |
| EDOWN | nested | 0.582408 / 1 | 0.892908 / 1 | 3.373223 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.584092 / 1 | 0.894611 / 1 | 3.374956 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.044483 / 1 | 0.006202 / 1 | 0.006251 / 1 |
| SH1 | nested | 1.016599 / 1 | 1.674028 / 1 | 5.318227 / 1 |
| SH3 | nested | 0.985842 / 1 | 1.514871 / 1 | 5.464140 / 1 |
| SH2 | nested | 0.982566 / 1 | 1.627060 / 1 | 5.860290 / 1 |
| shared-expert-during-read | boundary | 2.998793 / 1 | 4.832430 / 1 | 16.661331 / 1 |
| detail:expert-gate | nested | 25.067187 / 16 | 54.053194 / 16 | 25.380763 / 16 |
| detail:expert-up | nested | 21.092099 / 16 | 65.888009 / 16 | 25.056028 / 16 |
| detail:expert-activation | nested | 0.118319 / 16 | 0.128768 / 16 | 0.118693 / 16 |
| detail:expert-down | nested | 22.932741 / 16 | 72.499850 / 16 | 23.230999 / 16 |
| EUP | nested | 0.590133 / 1 | 2.033440 / 1 | 2.447984 / 1 |
| experts-mix-normalize-up | boundary | 70.308660 / 1 | 196.628354 / 1 | 76.661679 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002515 / 1 | 0.003025 / 1 | 0.002504 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000341 / 1 | 0.000141 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.418041 / 1 | 0.494224 / 1 | 0.382605 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 873.024444 / 1 | 1662.880760 / 1 | 950.878013 / 1 |
| op:Q   int8 projection | nested | 15.262345 / 1 | 23.028066 / 1 | 94.827513 / 1 |
| op:X   mxfp4 expert proj | nested | 69.341852 / 1 | 192.664326 / 1 | 73.880113 / 1 |
| op:N   rmsnorm | nested | 0.032179 / 1 | 0.036137 / 1 | 0.024996 / 1 |
| op:L   l2 per-head | nested | 0.005149 / 1 | 0.005541 / 1 | 0.006071 / 1 |
| op:SiTU + sigma | nested | 0.125021 / 1 | 0.139028 / 1 | 0.129686 / 1 |
| op:C   shortconv | nested | 0.054673 / 1 | 0.045314 / 1 | 0.040085 / 1 |
| op:AR  snapshot aggregate | nested | 0.035987 / 1 | 0.040845 / 1 | 0.038352 / 1 |
| op:D   kda delta-rule | nested | 0.145081 / 1 | 0.147015 / 1 | 0.153117 / 1 |
| op:router dot product | nested | 0.590413 / 1 | 0.647300 / 1 | 3.356060 / 1 |
| op:top-k selection | nested | 0.002695 / 1 | 0.002495 / 1 | 0.002344 / 1 |
| op:alpha / beta / gate | nested | 0.073007 / 1 | 0.072225 / 1 | 0.057267 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 87.226460 / 1 | 219.897382 / 1 | 174.347382 / 1 |

### Layer 81

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005089 / 1 | 0.006833 / 1 | 0.006192 / 1 |
| pre-attention-aggregation | boundary | 0.027662 / 1 | 0.033011 / 1 | 0.019246 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011572 / 1 | 0.011932 / 1 | 0.230000 / 1 |
| Q | nested | 2.511813 / 1 | 3.231408 / 1 | 48.510109 / 1 |
| K | nested | 2.349670 / 1 | 2.753645 / 1 | 2.908624 / 1 |
| V | nested | 2.229225 / 1 | 2.511563 / 1 | 2.757963 / 1 |
| B | nested | 0.022071 / 1 | 0.183502 / 1 | 0.165730 / 1 |
| FA | nested | 0.026910 / 1 | 0.042539 / 1 | 0.189464 / 1 |
| FB | nested | 0.046928 / 1 | 0.081742 / 1 | 0.339955 / 1 |
| G | nested | 1.961956 / 1 | 8.596242 / 1 | 6.509653 / 1 |
| O | nested | 2.036856 / 1 | 5.410559 / 1 | 8.654019 / 1 |
| attention | boundary | 12.240311 / 1 | 34.202286 / 1 | 70.777666 / 1 |
| attention-residual | boundary | 0.001874 / 1 | 0.002164 / 1 | 0.001743 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028083 / 1 | 0.031258 / 1 | 0.029315 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002615 / 1 | 0.003236 / 1 | 0.178383 / 1 |
| router-and-top16 | boundary | 0.589381 / 1 | 0.831183 / 1 | 3.169052 / 1 |
| EDOWN | nested | 0.600973 / 1 | 0.935949 / 1 | 3.460265 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.602686 / 1 | 0.937682 / 1 | 3.462279 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.017853 / 1 | 0.011592 / 1 | 0.018054 / 1 |
| SH1 | nested | 1.006340 / 1 | 2.446972 / 1 | 3.649459 / 1 |
| SH3 | nested | 0.993736 / 1 | 5.176232 / 1 | 3.585139 / 1 |
| SH2 | nested | 0.971875 / 1 | 3.275801 / 1 | 4.635602 / 1 |
| shared-expert-during-read | boundary | 2.986490 / 1 | 10.916547 / 1 | 11.887561 / 1 |
| detail:expert-gate | nested | 21.839106 / 16 | 121.409711 / 16 | 19.516636 / 16 |
| detail:expert-up | nested | 20.603167 / 16 | 89.275078 / 16 | 19.248315 / 16 |
| detail:expert-activation | nested | 0.115598 / 16 | 0.117099 / 16 | 0.119491 / 16 |
| detail:expert-down | nested | 22.290891 / 16 | 89.123653 / 16 | 17.840585 / 16 |
| EUP | nested | 0.887718 / 1 | 3.465134 / 1 | 1.382483 / 1 |
| experts-mix-normalize-up | boundary | 66.211814 / 1 | 305.862054 / 1 | 58.550830 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002955 / 1 | 0.002835 / 1 | 0.002395 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000301 / 1 | 0.000140 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.441285 / 1 | 0.452856 / 1 | 0.371564 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 813.807616 / 1 | 3280.468510 / 1 | 681.271552 / 1 |
| op:Q   int8 projection | nested | 15.644669 / 1 | 38.108913 / 1 | 86.746582 / 1 |
| op:X   mxfp4 expert proj | nested | 64.970127 / 1 | 300.026551 / 1 | 56.818894 / 1 |
| op:N   rmsnorm | nested | 0.033242 / 1 | 0.033151 / 1 | 0.025228 / 1 |
| op:L   l2 per-head | nested | 0.005280 / 1 | 0.015799 / 1 | 0.006122 / 1 |
| op:SiTU + sigma | nested | 0.122348 / 1 | 0.127838 / 1 | 0.130290 / 1 |
| op:C   shortconv | nested | 0.052308 / 1 | 6.041760 / 1 | 0.041347 / 1 |
| op:AR  snapshot aggregate | nested | 0.045715 / 1 | 0.053339 / 1 | 0.037629 / 1 |
| op:D   kda delta-rule | nested | 0.136956 / 1 | 0.142997 / 1 | 0.190175 / 1 |
| op:router dot product | nested | 0.585794 / 1 | 0.828248 / 1 | 3.166396 / 1 |
| op:top-k selection | nested | 0.002555 / 1 | 0.002485 / 1 | 0.002164 / 1 |
| op:alpha / beta / gate | nested | 0.072405 / 1 | 4.295255 / 1 | 0.059682 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 83.191992 / 1 | 353.324787 / 1 | 148.727382 / 1 |

### Layer 82

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005761 / 1 | 0.007655 / 1 | 0.006703 / 1 |
| pre-attention-aggregation | boundary | 0.035887 / 1 | 0.033342 / 1 | 0.020909 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012162 / 1 | 0.012573 / 1 | 0.223157 / 1 |
| Q | nested | 3.233732 / 1 | 3.806692 / 1 | 45.539469 / 1 |
| K | nested | 3.238621 / 1 | 3.283916 / 1 | 2.892795 / 1 |
| V | nested | 3.193457 / 1 | 3.008170 / 1 | 2.679978 / 1 |
| B | nested | 0.030828 / 1 | 0.151784 / 1 | 3.062672 / 1 |
| FA | nested | 0.039023 / 1 | 0.039404 / 1 | 0.134962 / 1 |
| FB | nested | 0.066985 / 1 | 1.600129 / 1 | 0.250949 / 1 |
| G | nested | 3.067873 / 1 | 6.291105 / 1 | 3.983924 / 1 |
| O | nested | 2.439598 / 1 | 5.612667 / 1 | 4.119407 / 1 |
| attention | boundary | 16.506142 / 1 | 25.707804 / 1 | 63.295716 / 1 |
| attention-residual | boundary | 0.002375 / 1 | 0.004108 / 1 | 0.002084 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028413 / 1 | 0.027522 / 1 | 0.029395 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002865 / 1 | 0.002705 / 1 | 0.002936 / 1 |
| router-and-top16 | boundary | 0.595122 / 1 | 3.947325 / 1 | 1.087803 / 1 |
| EDOWN | nested | 0.582709 / 1 | 1.253692 / 1 | 1.316019 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.584452 / 1 | 1.255886 / 1 | 1.318013 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.023284 / 1 | 0.011793 / 1 | 0.006903 / 1 |
| SH1 | nested | 0.996412 / 1 | 2.007120 / 1 | 2.123507 / 1 |
| SH3 | nested | 0.987385 / 1 | 1.728199 / 1 | 2.362855 / 1 |
| SH2 | nested | 0.981834 / 1 | 1.873541 / 1 | 2.211452 / 1 |
| shared-expert-during-read | boundary | 2.979367 / 1 | 5.625641 / 1 | 6.714996 / 1 |
| detail:expert-gate | nested | 23.488117 / 16 | 193.822188 / 16 | 22.696198 / 16 |
| detail:expert-up | nested | 22.859140 / 16 | 160.620518 / 16 | 21.701990 / 16 |
| detail:expert-activation | nested | 0.097600 / 16 | 0.167864 / 16 | 0.124212 / 16 |
| detail:expert-down | nested | 24.472595 / 16 | 200.080063 / 16 | 20.640216 / 16 |
| EUP | nested | 0.909450 / 1 | 1.487259 / 1 | 1.332029 / 1 |
| experts-mix-normalize-up | boundary | 72.261278 / 1 | 569.985851 / 1 | 66.933062 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003286 / 1 | 0.003106 / 1 | 0.003837 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000320 / 1 | 0.000281 / 1 | 0.000331 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.438409 / 1 | 0.479195 / 1 | 0.385290 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 964.756891 / 1 | 5428.923914 / 1 | 819.010289 / 1 |
| op:Q   int8 projection | nested | 19.765971 / 1 | 32.141787 / 1 | 72.008286 / 1 |
| op:X   mxfp4 expert proj | nested | 71.016211 / 1 | 554.791133 / 1 | 65.260057 / 1 |
| op:N   rmsnorm | nested | 0.033532 / 1 | 0.035465 / 1 | 0.023913 / 1 |
| op:L   l2 per-head | nested | 0.006642 / 1 | 0.005440 / 1 | 0.005961 / 1 |
| op:SiTU + sigma | nested | 0.104486 / 1 | 0.178033 / 1 | 0.135011 / 1 |
| op:C   shortconv | nested | 0.055273 / 1 | 0.066124 / 1 | 0.039093 / 1 |
| op:AR  snapshot aggregate | nested | 0.053761 / 1 | 0.050845 / 1 | 0.040285 / 1 |
| op:D   kda delta-rule | nested | 0.137516 / 1 | 0.143749 / 1 | 0.198281 / 1 |
| op:router dot product | nested | 0.591826 / 1 | 3.944600 / 1 | 1.085007 / 1 |
| op:top-k selection | nested | 0.002434 / 1 | 0.002284 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.107130 / 1 | 0.075652 / 1 | 0.058138 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 93.504060 / 1 | 607.124564 / 1 | 140.058986 / 1 |

### Layer 83

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006112 / 1 | 0.005951 / 1 | 0.112931 / 1 |
| pre-attention-aggregation | boundary | 0.030808 / 1 | 0.030336 / 1 | 0.026881 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.013736 / 1 | 0.011642 / 1 | 0.209151 / 1 |
| QA | nested | 0.374520 / 1 | 0.452566 / 1 | 1.633242 / 1 |
| QB | nested | 0.886516 / 1 | 0.940868 / 1 | 4.780632 / 1 |
| KA | nested | 0.141525 / 1 | 0.133380 / 1 | 1.044141 / 1 |
| KB | nested | 0.408884 / 1 | 0.359632 / 1 | 2.379937 / 1 |
| G | nested | 3.048235 / 1 | 3.752472 / 1 | 4.553168 / 1 |
| O | nested | 3.038848 / 1 | 4.276590 / 1 | 3.998582 / 1 |
| attention | boundary | 8.041615 / 1 | 10.027416 / 1 | 18.917948 / 1 |
| attention-residual | boundary | 0.002054 / 1 | 0.001362 / 1 | 0.001663 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.030016 / 1 | 0.028342 / 1 | 0.026961 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002495 / 1 | 0.002685 / 1 | 0.002725 / 1 |
| router-and-top16 | boundary | 0.602816 / 1 | 0.967247 / 1 | 1.100877 / 1 |
| EDOWN | nested | 0.881427 / 1 | 1.163174 / 1 | 1.713181 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.884232 / 1 | 1.164907 / 1 | 1.714884 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.034885 / 1 | 0.006132 / 1 | 0.006893 / 1 |
| SH1 | nested | 1.547041 / 1 | 2.326748 / 1 | 2.396899 / 1 |
| SH3 | nested | 1.533585 / 1 | 2.042757 / 1 | 2.771398 / 1 |
| SH2 | nested | 1.526462 / 1 | 2.169343 / 1 | 2.864853 / 1 |
| shared-expert-during-read | boundary | 4.627777 / 1 | 6.556470 / 1 | 8.051104 / 1 |
| detail:expert-gate | nested | 26.807681 / 16 | 20.764978 / 16 | 25.555701 / 16 |
| detail:expert-up | nested | 24.354147 / 16 | 19.253756 / 16 | 24.260660 / 16 |
| detail:expert-activation | nested | 0.111988 / 16 | 0.116107 / 16 | 0.129860 / 16 |
| detail:expert-down | nested | 25.156432 / 16 | 21.307973 / 16 | 22.983133 / 16 |
| EUP | nested | 0.586746 / 1 | 1.219498 / 1 | 1.457873 / 1 |
| experts-mix-normalize-up | boundary | 77.480600 / 1 | 63.085924 / 1 | 74.847210 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002945 / 1 | 0.002404 / 1 | 0.002454 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000311 / 1 | 0.000131 / 1 | 0.000451 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008025 / 1 | 0.008125 / 1 | 0.008927 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1022.689360 / 1 | 708.432219 / 1 | 950.220278 / 1 |
| op:Q   int8 projection | nested | 13.972217 / 1 | 18.835043 / 1 | 29.592403 / 1 |
| op:X   mxfp4 expert proj | nested | 76.538980 / 1 | 61.537972 / 1 | 73.025077 / 1 |
| op:N   rmsnorm | nested | 0.027292 / 1 | 0.024735 / 1 | 0.024967 / 1 |
| op:SiTU + sigma | nested | 0.123321 / 1 | 0.127258 / 1 | 0.141291 / 1 |
| op:AR  snapshot aggregate | nested | 0.050424 / 1 | 0.048180 / 1 | 0.043341 / 1 |
| op:SA  softmax attention | nested | 0.007304 / 1 | 0.010310 / 1 | 0.013044 / 1 |
| op:router dot product | nested | 0.599400 / 1 | 0.964091 / 1 | 1.098162 / 1 |
| op:top-k selection | nested | 0.002736 / 1 | 0.002805 / 1 | 0.002405 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 91.797071 / 1 | 81.920326 / 1 | 105.051595 / 1 |

### Layer 84

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004569 / 1 | 0.005931 / 1 | 0.006132 / 1 |
| pre-attention-aggregation | boundary | 0.017703 / 1 | 0.019126 / 1 | 0.020137 / 1 |
| snapshot-push | boundary | 0.001873 / 1 | 0.001854 / 1 | 0.002474 / 1 |
| pre-attention-normalization | boundary | 0.010149 / 1 | 0.010229 / 1 | 0.222926 / 1 |
| Q | nested | 2.510691 / 1 | 4.153110 / 1 | 39.769248 / 1 |
| K | nested | 2.361553 / 1 | 3.435660 / 1 | 3.137692 / 1 |
| V | nested | 2.249002 / 1 | 3.110702 / 1 | 2.888046 / 1 |
| B | nested | 0.020929 / 1 | 0.026670 / 1 | 0.342310 / 1 |
| FA | nested | 0.027010 / 1 | 0.134050 / 1 | 0.177632 / 1 |
| FB | nested | 0.048731 / 1 | 0.121307 / 1 | 0.208699 / 1 |
| G | nested | 1.947008 / 1 | 3.781987 / 1 | 6.193693 / 1 |
| O | nested | 2.023050 / 1 | 4.052762 / 1 | 7.065582 / 1 |
| attention | boundary | 12.256661 / 1 | 19.870187 / 1 | 60.677704 / 1 |
| attention-residual | boundary | 0.002164 / 1 | 0.002024 / 1 | 0.002124 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.029104 / 1 | 0.030327 / 1 | 0.030887 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002524 / 1 | 0.003086 / 1 | 0.003516 / 1 |
| router-and-top16 | boundary | 0.594792 / 1 | 0.764027 / 1 | 1.820471 / 1 |
| EDOWN | nested | 0.594671 / 1 | 1.273900 / 1 | 2.217063 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.596394 / 1 | 1.275614 / 1 | 2.219447 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026930 / 1 | 0.017543 / 1 | 0.012724 / 1 |
| SH1 | nested | 1.015016 / 1 | 2.274490 / 1 | 2.694775 / 1 |
| SH3 | nested | 1.000349 / 1 | 2.078935 / 1 | 2.926017 / 1 |
| SH2 | nested | 0.987495 / 1 | 1.997563 / 1 | 3.119659 / 1 |
| shared-expert-during-read | boundary | 3.016947 / 1 | 6.368149 / 1 | 8.758455 / 1 |
| detail:expert-gate | nested | 28.085870 / 16 | 21.477823 / 16 | 22.930075 / 16 |
| detail:expert-up | nested | 27.117550 / 16 | 19.695253 / 16 | 22.401534 / 16 |
| detail:expert-activation | nested | 0.112025 / 16 | 0.105107 / 16 | 0.115878 / 16 |
| detail:expert-down | nested | 29.372233 / 16 | 20.862502 / 16 | 20.971323 / 16 |
| EUP | nested | 0.782101 / 1 | 1.198128 / 1 | 1.656786 / 1 |
| experts-mix-normalize-up | boundary | 85.905031 / 1 | 63.768780 / 1 | 68.514036 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002685 / 1 | 0.002444 / 1 | 0.002184 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000320 / 1 | 0.000140 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.436546 / 1 | 0.374870 / 1 | 0.370222 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1037.039483 / 1 | 830.384516 / 1 | 838.889075 / 1 |
| op:Q   int8 projection | nested | 15.566194 / 1 | 27.637148 / 1 | 72.394597 / 1 |
| op:X   mxfp4 expert proj | nested | 84.787716 / 1 | 62.236466 / 1 | 66.517304 / 1 |
| op:N   rmsnorm | nested | 0.032001 / 1 | 0.031359 / 1 | 0.024586 / 1 |
| op:L   l2 per-head | nested | 0.005110 / 1 | 0.005090 / 1 | 0.005951 / 1 |
| op:SiTU + sigma | nested | 0.119499 / 1 | 0.115827 / 1 | 0.127250 / 1 |
| op:C   shortconv | nested | 0.051866 / 1 | 0.046016 / 1 | 0.037891 / 1 |
| op:AR  snapshot aggregate | nested | 0.036699 / 1 | 0.039225 / 1 | 0.040556 / 1 |
| op:D   kda delta-rule | nested | 0.145572 / 1 | 0.144209 / 1 | 0.148267 / 1 |
| op:router dot product | nested | 0.591264 / 1 | 0.760901 / 1 | 1.817346 / 1 |
| op:top-k selection | nested | 0.002985 / 1 | 0.002816 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.072345 / 1 | 0.074299 / 1 | 0.057969 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 102.925303 / 1 | 92.534047 / 1 | 142.685243 / 1 |

### Layer 85

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005951 / 1 | 0.005942 / 1 | 0.081242 / 1 |
| pre-attention-aggregation | boundary | 0.029786 / 1 | 0.020488 / 1 | 0.019947 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012443 / 1 | 0.010971 / 1 | 0.100318 / 1 |
| Q | nested | 3.155236 / 1 | 3.530436 / 1 | 32.276448 / 1 |
| K | nested | 2.832532 / 1 | 2.984276 / 1 | 3.223343 / 1 |
| V | nested | 2.814278 / 1 | 2.464324 / 1 | 2.877396 / 1 |
| B | nested | 0.023424 / 1 | 0.124232 / 1 | 0.289260 / 1 |
| FA | nested | 0.035195 / 1 | 0.032872 / 1 | 0.143808 / 1 |
| FB | nested | 0.055433 / 1 | 0.151704 / 1 | 0.237454 / 1 |
| G | nested | 2.779583 / 1 | 3.857067 / 1 | 5.240011 / 1 |
| O | nested | 2.836420 / 1 | 3.800310 / 1 | 4.900306 / 1 |
| attention | boundary | 15.610969 / 1 | 17.919692 / 1 | 50.019770 / 1 |
| attention-residual | boundary | 0.002695 / 1 | 0.002494 / 1 | 0.002454 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.036909 / 1 | 0.028934 / 1 | 0.031219 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002594 / 1 | 0.002605 / 1 | 0.379920 / 1 |
| router-and-top16 | boundary | 0.600131 / 1 | 0.851431 / 1 | 1.149147 / 1 |
| EDOWN | nested | 0.830542 / 1 | 1.007833 / 1 | 1.624456 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.832406 / 1 | 1.009486 / 1 | 1.626229 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027141 / 1 | 0.006532 / 1 | 0.006332 / 1 |
| SH1 | nested | 1.431054 / 1 | 1.782841 / 1 | 3.446650 / 1 |
| SH3 | nested | 1.438829 / 1 | 1.959521 / 1 | 2.774975 / 1 |
| SH2 | nested | 1.400447 / 1 | 2.066502 / 1 | 2.734570 / 1 |
| shared-expert-during-read | boundary | 4.286790 / 1 | 5.823321 / 1 | 8.973166 / 1 |
| detail:expert-gate | nested | 25.746348 / 16 | 23.569480 / 16 | 21.685799 / 16 |
| detail:expert-up | nested | 24.553597 / 16 | 20.974429 / 16 | 20.572219 / 16 |
| detail:expert-activation | nested | 0.098783 / 16 | 0.101169 / 16 | 0.120036 / 16 |
| detail:expert-down | nested | 25.248123 / 16 | 23.111953 / 16 | 19.271110 / 16 |
| EUP | nested | 0.584563 / 1 | 1.069479 / 1 | 1.590863 / 1 |
| experts-mix-normalize-up | boundary | 76.676268 / 1 | 69.261844 / 1 | 63.667591 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002515 / 1 | 0.002344 / 1 | 0.002404 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000151 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.414365 / 1 | 0.373958 / 1 | 0.365293 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1031.260836 / 1 | 894.221851 / 1 | 764.031394 / 1 |
| op:Q   int8 projection | nested | 20.215954 / 1 | 24.829524 / 1 | 61.357675 / 1 |
| op:X   mxfp4 expert proj | nested | 75.748606 / 1 | 67.857741 / 1 | 61.745300 / 1 |
| op:N   rmsnorm | nested | 0.042338 / 1 | 0.039352 / 1 | 0.024216 / 1 |
| op:L   l2 per-head | nested | 0.005019 / 1 | 0.005731 / 1 | 0.005750 / 1 |
| op:SiTU + sigma | nested | 0.109043 / 1 | 0.108792 / 1 | 0.130324 / 1 |
| op:C   shortconv | nested | 0.058219 / 1 | 0.050173 / 1 | 0.042410 / 1 |
| op:AR  snapshot aggregate | nested | 0.049984 / 1 | 0.038823 / 1 | 0.040485 / 1 |
| op:D   kda delta-rule | nested | 0.139842 / 1 | 0.225161 / 1 | 0.198551 / 1 |
| op:router dot product | nested | 0.597026 / 1 | 0.848485 / 1 | 1.146061 / 1 |
| op:top-k selection | nested | 0.002715 / 1 | 0.002545 / 1 | 0.002465 / 1 |
| op:alpha / beta / gate | nested | 0.073568 / 1 | 0.072415 / 1 | 0.056746 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 98.560968 / 1 | 95.342144 / 1 | 126.447934 / 1 |

### Layer 86

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005801 / 1 | 0.006061 / 1 | 0.087594 / 1 |
| pre-attention-aggregation | boundary | 0.033843 / 1 | 0.020859 / 1 | 0.020519 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011211 / 1 | 0.010239 / 1 | 0.208770 / 1 |
| Q | nested | 2.510541 / 1 | 4.021524 / 1 | 27.813518 / 1 |
| K | nested | 2.371461 / 1 | 3.591490 / 1 | 3.067291 / 1 |
| V | nested | 2.220920 / 1 | 3.356792 / 1 | 2.771168 / 1 |
| B | nested | 0.020399 / 1 | 0.028714 / 1 | 0.280885 / 1 |
| FA | nested | 0.027361 / 1 | 0.131486 / 1 | 0.139892 / 1 |
| FB | nested | 0.055444 / 1 | 0.157314 / 1 | 0.226744 / 1 |
| G | nested | 1.925317 / 1 | 3.993752 / 1 | 5.801429 / 1 |
| O | nested | 2.040473 / 1 | 4.603953 / 1 | 5.061207 / 1 |
| attention | boundary | 12.223208 / 1 | 20.851421 / 1 | 45.917245 / 1 |
| attention-residual | boundary | 0.002665 / 1 | 0.002485 / 1 | 0.002044 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028764 / 1 | 0.030137 / 1 | 0.030447 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002455 / 1 | 0.003016 / 1 | 0.169877 / 1 |
| router-and-top16 | boundary | 0.586696 / 1 | 0.954764 / 1 | 1.672946 / 1 |
| EDOWN | nested | 0.580064 / 1 | 1.497518 / 1 | 3.363965 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.581727 / 1 | 1.499442 / 1 | 3.366440 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.017213 / 1 | 0.011932 / 1 | 0.012814 / 1 |
| SH1 | nested | 0.985401 / 1 | 1.985770 / 1 | 4.517661 / 1 |
| SH3 | nested | 0.987475 / 1 | 2.222273 / 1 | 5.145925 / 1 |
| SH2 | nested | 0.983778 / 1 | 1.942199 / 1 | 5.739204 / 1 |
| shared-expert-during-read | boundary | 2.971192 / 1 | 6.167273 / 1 | 15.420082 / 1 |
| detail:expert-gate | nested | 25.477925 / 16 | 19.320208 / 16 | 20.933412 / 16 |
| detail:expert-up | nested | 23.458113 / 16 | 17.662350 / 16 | 20.691675 / 16 |
| detail:expert-activation | nested | 0.105027 / 16 | 0.104214 / 16 | 0.114523 / 16 |
| detail:expert-down | nested | 25.006244 / 16 | 19.045035 / 16 | 22.927530 / 16 |
| EUP | nested | 0.589392 / 1 | 1.180426 / 1 | 2.713210 / 1 |
| experts-mix-normalize-up | boundary | 75.073032 / 1 | 57.718846 / 1 | 67.804841 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.002665 / 1 | 0.002474 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000221 / 1 | 0.000320 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.404816 / 1 | 0.368759 / 1 | 0.370232 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 945.519385 / 1 | 742.879861 / 1 | 677.782748 / 1 |
| op:Q   int8 projection | nested | 15.296623 / 1 | 28.711447 / 1 | 66.640397 / 1 |
| op:X   mxfp4 expert proj | nested | 74.149147 / 1 | 56.226528 / 1 | 64.769007 / 1 |
| op:N   rmsnorm | nested | 0.031028 / 1 | 0.034214 / 1 | 0.025748 / 1 |
| op:L   l2 per-head | nested | 0.005380 / 1 | 0.005199 / 1 | 0.005991 / 1 |
| op:SiTU + sigma | nested | 0.113381 / 1 | 0.114635 / 1 | 0.125033 / 1 |
| op:C   shortconv | nested | 0.056645 / 1 | 0.053931 / 1 | 0.039834 / 1 |
| op:AR  snapshot aggregate | nested | 0.052357 / 1 | 0.040866 / 1 | 0.041448 / 1 |
| op:D   kda delta-rule | nested | 0.132989 / 1 | 0.222215 / 1 | 0.203901 / 1 |
| op:router dot product | nested | 0.583731 / 1 | 0.951659 / 1 | 1.670431 / 1 |
| op:top-k selection | nested | 0.002505 / 1 | 0.002815 / 1 | 0.002124 / 1 |
| op:alpha / beta / gate | nested | 0.072055 / 1 | 0.075841 / 1 | 0.057638 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 91.966026 / 1 | 87.668275 / 1 | 135.107254 / 1 |

### Layer 87

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005811 / 1 | 0.007034 / 1 | 0.007324 / 1 |
| pre-attention-aggregation | boundary | 0.032560 / 1 | 0.021059 / 1 | 0.020579 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011141 / 1 | 0.009988 / 1 | 0.200355 / 1 |
| QA | nested | 0.303637 / 1 | 0.467614 / 1 | 1.416136 / 1 |
| QB | nested | 0.700248 / 1 | 0.913928 / 1 | 3.374656 / 1 |
| KA | nested | 0.105808 / 1 | 0.225933 / 1 | 0.777923 / 1 |
| KB | nested | 0.305120 / 1 | 0.409335 / 1 | 1.696409 / 1 |
| G | nested | 1.982384 / 1 | 3.609154 / 1 | 6.373329 / 1 |
| O | nested | 1.943151 / 1 | 3.670548 / 1 | 6.001634 / 1 |
| attention | boundary | 5.443892 / 1 | 9.396787 / 1 | 20.067385 / 1 |
| attention-residual | boundary | 0.003105 / 1 | 0.003066 / 1 | 0.002825 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026980 / 1 | 0.028383 / 1 | 0.029635 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002174 / 1 | 0.002424 / 1 | 0.199463 / 1 |
| router-and-top16 | boundary | 0.574133 / 1 | 0.702543 / 1 | 2.517824 / 1 |
| EDOWN | nested | 0.567440 / 1 | 1.087902 / 1 | 2.003664 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.569243 / 1 | 1.089535 / 1 | 2.005477 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.037760 / 1 | 0.011882 / 1 | 0.012123 / 1 |
| SH1 | nested | 0.965854 / 1 | 1.893077 / 1 | 4.341131 / 1 |
| SH3 | nested | 0.965153 / 1 | 2.023732 / 1 | 6.849899 / 1 |
| SH2 | nested | 0.972637 / 1 | 1.718180 / 1 | 5.314670 / 1 |
| shared-expert-during-read | boundary | 2.918292 / 1 | 5.648985 / 1 | 16.523343 / 1 |
| detail:expert-gate | nested | 26.164326 / 16 | 20.090605 / 16 | 26.805445 / 16 |
| detail:expert-up | nested | 26.056507 / 16 | 19.106780 / 16 | 26.254817 / 16 |
| detail:expert-activation | nested | 0.105816 / 16 | 0.098594 / 16 | 0.116969 / 16 |
| detail:expert-down | nested | 26.346954 / 16 | 20.214740 / 16 | 28.966303 / 16 |
| EUP | nested | 0.628495 / 1 | 1.018563 / 1 | 2.367684 / 1 |
| experts-mix-normalize-up | boundary | 79.752876 / 1 | 60.960072 / 1 | 84.909591 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002765 / 1 | 0.002445 / 1 | 0.002665 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000141 / 1 | 0.000280 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.008356 / 1 | 0.007734 / 1 | 0.008757 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 986.124927 / 1 | 827.717688 / 1 | 946.094857 / 1 |
| op:Q   int8 projection | nested | 9.438717 / 1 | 17.036272 / 1 | 40.515650 / 1 |
| op:X   mxfp4 expert proj | nested | 78.774158 / 1 | 59.607943 / 1 | 82.241235 / 1 |
| op:N   rmsnorm | nested | 0.024757 / 1 | 0.023363 / 1 | 0.025347 / 1 |
| op:SiTU + sigma | nested | 0.114042 / 1 | 0.106138 / 1 | 0.127912 / 1 |
| op:AR  snapshot aggregate | nested | 0.049212 / 1 | 0.039012 / 1 | 0.039994 / 1 |
| op:SA  softmax attention | nested | 0.006813 / 1 | 0.008987 / 1 | 0.011611 / 1 |
| op:router dot product | nested | 0.571368 / 1 | 0.699487 / 1 | 2.514889 / 1 |
| op:top-k selection | nested | 0.002495 / 1 | 0.002765 / 1 | 0.002505 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 89.410662 / 1 | 77.910644 / 1 | 126.528154 / 1 |

### Layer 88

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005981 / 1 | 0.006683 / 1 | 0.177301 / 1 |
| pre-attention-aggregation | boundary | 0.018134 / 1 | 0.020118 / 1 | 0.019045 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010049 / 1 | 0.010370 / 1 | 0.199252 / 1 |
| Q | nested | 2.542130 / 1 | 4.033056 / 1 | 31.023817 / 1 |
| K | nested | 2.388613 / 1 | 3.580490 / 1 | 3.713288 / 1 |
| V | nested | 2.295659 / 1 | 2.954841 / 1 | 3.360469 / 1 |
| B | nested | 0.022222 / 1 | 0.028513 / 1 | 0.382945 / 1 |
| FA | nested | 0.027702 / 1 | 0.134321 / 1 | 0.275795 / 1 |
| FB | nested | 0.049192 / 1 | 0.170008 / 1 | 0.316311 / 1 |
| G | nested | 1.935517 / 1 | 3.602973 / 1 | 8.271295 / 1 |
| O | nested | 2.023711 / 1 | 3.883116 / 1 | 11.104418 / 1 |
| attention | boundary | 12.356808 / 1 | 19.428652 / 1 | 59.285272 / 1 |
| attention-residual | boundary | 0.002555 / 1 | 0.002305 / 1 | 0.002494 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027822 / 1 | 0.031850 / 1 | 0.030577 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002324 / 1 | 0.003246 / 1 | 0.003387 / 1 |
| router-and-top16 | boundary | 0.599020 / 1 | 0.866810 / 1 | 2.726194 / 1 |
| EDOWN | nested | 0.589631 / 1 | 1.078175 / 1 | 4.188436 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.591495 / 1 | 1.079988 / 1 | 4.190370 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027472 / 1 | 0.012082 / 1 | 0.006542 / 1 |
| SH1 | nested | 1.002433 / 1 | 1.741253 / 1 | 4.759784 / 1 |
| SH3 | nested | 1.015568 / 1 | 1.949894 / 1 | 5.133762 / 1 |
| SH2 | nested | 0.988417 / 1 | 1.914047 / 1 | 4.165784 / 1 |
| shared-expert-during-read | boundary | 3.029952 / 1 | 5.622365 / 1 | 14.077935 / 1 |
| detail:expert-gate | nested | 24.903984 / 16 | 23.115122 / 16 | 22.502786 / 16 |
| detail:expert-up | nested | 23.306217 / 16 | 21.072481 / 16 | 21.708903 / 16 |
| detail:expert-activation | nested | 0.097591 / 16 | 0.107349 / 16 | 0.122530 / 16 |
| detail:expert-down | nested | 24.625080 / 16 | 22.388680 / 16 | 24.010114 / 16 |
| EUP | nested | 0.583340 / 1 | 0.942712 / 1 | 1.978256 / 1 |
| experts-mix-normalize-up | boundary | 73.935697 / 1 | 68.063134 / 1 | 70.738352 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002174 / 1 | 0.002445 / 1 | 0.002986 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000230 / 1 | 0.000130 / 1 | 0.000481 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427780 / 1 | 0.371324 / 1 | 0.374189 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 990.750302 / 1 | 884.723768 / 1 | 739.321152 / 1 |
| op:Q   int8 projection | nested | 15.462695 / 1 | 26.011524 / 1 | 78.672208 / 1 |
| op:X   mxfp4 expert proj | nested | 73.035926 / 1 | 66.788013 / 1 | 68.441450 / 1 |
| op:N   rmsnorm | nested | 0.030258 / 1 | 0.037771 / 1 | 0.024365 / 1 |
| op:L   l2 per-head | nested | 0.005129 / 1 | 0.005971 / 1 | 0.005631 / 1 |
| op:SiTU + sigma | nested | 0.109002 / 1 | 0.118169 / 1 | 0.134172 / 1 |
| op:C   shortconv | nested | 0.054792 / 1 | 0.047698 / 1 | 0.038041 / 1 |
| op:AR  snapshot aggregate | nested | 0.036348 / 1 | 0.040926 / 1 | 0.038832 / 1 |
| op:D   kda delta-rule | nested | 0.142016 / 1 | 0.147025 / 1 | 0.145332 / 1 |
| op:router dot product | nested | 0.596384 / 1 | 0.863834 / 1 | 2.723198 / 1 |
| op:top-k selection | nested | 0.002234 / 1 | 0.002595 / 1 | 0.002585 / 1 |
| op:alpha / beta / gate | nested | 0.073607 / 1 | 0.073627 / 1 | 0.058038 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 91.057819 / 1 | 95.541347 / 1 | 151.854655 / 1 |

### Layer 89

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006202 / 1 | 0.006953 / 1 | 0.172883 / 1 |
| pre-attention-aggregation | boundary | 0.028424 / 1 | 0.019586 / 1 | 0.020318 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011993 / 1 | 0.010329 / 1 | 0.197118 / 1 |
| Q | nested | 2.512033 / 1 | 3.626406 / 1 | 26.539628 / 1 |
| K | nested | 2.722998 / 1 | 2.908575 / 1 | 3.691067 / 1 |
| V | nested | 2.811694 / 1 | 2.479382 / 1 | 3.443584 / 1 |
| B | nested | 0.022842 / 1 | 0.026550 / 1 | 0.190656 / 1 |
| FA | nested | 0.034304 / 1 | 0.033012 / 1 | 0.224349 / 1 |
| FB | nested | 0.070181 / 1 | 0.130294 / 1 | 0.306492 / 1 |
| G | nested | 2.668937 / 1 | 3.142662 / 1 | 8.179513 / 1 |
| O | nested | 2.742715 / 1 | 3.326606 / 1 | 9.170985 / 1 |
| attention | boundary | 14.631459 / 1 | 16.647806 / 1 | 52.503801 / 1 |
| attention-residual | boundary | 0.002665 / 1 | 0.002384 / 1 | 0.003266 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.030417 / 1 | 0.029184 / 1 | 0.030737 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003186 / 1 | 0.002835 / 1 | 0.281676 / 1 |
| router-and-top16 | boundary | 0.610972 / 1 | 0.884794 / 1 | 2.837251 / 1 |
| EDOWN | nested | 0.798452 / 1 | 1.114313 / 1 | 2.407198 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.800185 / 1 | 1.115946 / 1 | 2.409081 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043351 / 1 | 0.017382 / 1 | 0.006582 / 1 |
| SH1 | nested | 1.400216 / 1 | 1.565866 / 1 | 4.409409 / 1 |
| SH3 | nested | 1.475708 / 1 | 1.692392 / 1 | 5.065565 / 1 |
| SH2 | nested | 1.449489 / 1 | 1.670000 / 1 | 4.128294 / 1 |
| shared-expert-during-read | boundary | 4.343095 / 1 | 4.943397 / 1 | 13.621952 / 1 |
| detail:expert-gate | nested | 26.467401 / 16 | 21.461840 / 16 | 25.192203 / 16 |
| detail:expert-up | nested | 26.197559 / 16 | 20.645807 / 16 | 23.793089 / 16 |
| detail:expert-activation | nested | 0.106399 / 16 | 0.104489 / 16 | 0.130173 / 16 |
| detail:expert-down | nested | 26.314916 / 16 | 22.005495 / 16 | 26.975020 / 16 |
| EUP | nested | 0.585363 / 1 | 1.288668 / 1 | 2.270653 / 1 |
| experts-mix-normalize-up | boundary | 80.134099 / 1 | 65.934667 / 1 | 78.744511 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002615 / 1 | 0.002695 / 1 | 0.002746 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000291 / 1 | 0.000120 / 1 | 0.000300 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.417841 / 1 | 0.377465 / 1 | 0.363579 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1000.553053 / 1 | 883.686797 / 1 | 850.681932 / 1 |
| op:Q   int8 projection | nested | 19.293216 / 1 | 23.002963 / 1 | 70.025540 / 1 |
| op:X   mxfp4 expert proj | nested | 79.195283 / 1 | 64.319548 / 1 | 76.191380 / 1 |
| op:N   rmsnorm | nested | 0.032951 / 1 | 0.033031 / 1 | 0.025168 / 1 |
| op:L   l2 per-head | nested | 0.005220 / 1 | 0.005971 / 1 | 0.005290 / 1 |
| op:SiTU + sigma | nested | 0.117389 / 1 | 0.113215 / 1 | 0.141956 / 1 |
| op:C   shortconv | nested | 0.060454 / 1 | 0.057327 / 1 | 0.035635 / 1 |
| op:AR  snapshot aggregate | nested | 0.048450 / 1 | 0.038523 / 1 | 0.041308 / 1 |
| op:D   kda delta-rule | nested | 0.136776 / 1 | 0.225261 / 1 | 0.217987 / 1 |
| op:router dot product | nested | 0.607746 / 1 | 0.881918 / 1 | 2.834136 / 1 |
| op:top-k selection | nested | 0.002695 / 1 | 0.002565 / 1 | 0.002725 / 1 |
| op:alpha / beta / gate | nested | 0.073157 / 1 | 0.079890 / 1 | 0.058500 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 101.087620 / 1 | 90.015210 / 1 | 151.217034 / 1 |

### Layer 90

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005831 / 1 | 0.009979 / 1 | 0.006773 / 1 |
| pre-attention-aggregation | boundary | 0.033863 / 1 | 0.020018 / 1 | 0.020418 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012083 / 1 | 0.010720 / 1 | 0.198872 / 1 |
| Q | nested | 2.507245 / 1 | 3.261143 / 1 | 42.639851 / 1 |
| K | nested | 2.371100 / 1 | 3.156017 / 1 | 3.486314 / 1 |
| V | nested | 2.248290 / 1 | 2.688343 / 1 | 3.128986 / 1 |
| B | nested | 0.024415 / 1 | 0.029555 / 1 | 0.331871 / 1 |
| FA | nested | 0.027441 / 1 | 0.136916 / 1 | 0.171501 / 1 |
| FB | nested | 0.046357 / 1 | 0.168796 / 1 | 0.166451 / 1 |
| G | nested | 1.931498 / 1 | 3.540576 / 1 | 9.747443 / 1 |
| O | nested | 2.022479 / 1 | 3.782698 / 1 | 9.148643 / 1 |
| attention | boundary | 12.222698 / 1 | 17.682409 / 1 | 69.573165 / 1 |
| attention-residual | boundary | 0.002725 / 1 | 0.002365 / 1 | 0.002354 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027101 / 1 | 0.030327 / 1 | 0.032300 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003036 / 1 | 0.002806 / 1 | 0.003737 / 1 |
| router-and-top16 | boundary | 0.594281 / 1 | 0.844007 / 1 | 2.748004 / 1 |
| EDOWN | nested | 0.592076 / 1 | 1.075600 / 1 | 2.864121 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595042 / 1 | 1.077363 / 1 | 2.866576 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032391 / 1 | 0.011852 / 1 | 0.012002 / 1 |
| SH1 | nested | 0.990571 / 1 | 2.048016 / 1 | 4.978161 / 1 |
| SH3 | nested | 0.995961 / 1 | 1.877768 / 1 | 4.725880 / 1 |
| SH2 | nested | 0.981474 / 1 | 1.798521 / 1 | 5.350647 / 1 |
| shared-expert-during-read | boundary | 2.982633 / 1 | 5.741979 / 1 | 15.073404 / 1 |
| detail:expert-gate | nested | 26.660966 / 16 | 41.689817 / 16 | 22.036172 / 16 |
| detail:expert-up | nested | 25.537788 / 16 | 50.135297 / 16 | 20.969733 / 16 |
| detail:expert-activation | nested | 0.109131 / 16 | 0.125224 / 16 | 0.120867 / 16 |
| detail:expert-down | nested | 28.136340 / 16 | 46.752726 / 16 | 23.898834 / 16 |
| EUP | nested | 0.594000 / 1 | 2.269149 / 1 | 2.314955 / 1 |
| experts-mix-normalize-up | boundary | 81.499259 / 1 | 142.195598 / 1 | 69.765524 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002524 / 1 | 0.002965 / 1 | 0.002625 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000220 / 1 | 0.000310 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.412591 / 1 | 0.441144 / 1 | 0.373177 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 1000.103792 / 1 | 1417.727832 / 1 | 740.552958 / 1 |
| op:Q   int8 projection | nested | 15.331235 / 1 | 25.831286 / 1 | 89.053031 / 1 |
| op:X   mxfp4 expert proj | nested | 80.552460 / 1 | 138.803141 / 1 | 67.128737 / 1 |
| op:N   rmsnorm | nested | 0.031608 / 1 | 0.035427 / 1 | 0.026159 / 1 |
| op:L   l2 per-head | nested | 0.005631 / 1 | 0.005410 / 1 | 0.005781 / 1 |
| op:SiTU + sigma | nested | 0.117148 / 1 | 0.136546 / 1 | 0.132117 / 1 |
| op:C   shortconv | nested | 0.057356 / 1 | 0.049261 / 1 | 0.036459 / 1 |
| op:AR  snapshot aggregate | nested | 0.051487 / 1 | 0.040375 / 1 | 0.041267 / 1 |
| op:D   kda delta-rule | nested | 0.135443 / 1 | 0.198110 / 1 | 0.195916 / 1 |
| op:router dot product | nested | 0.591515 / 1 | 0.841493 / 1 | 2.745329 / 1 |
| op:top-k selection | nested | 0.002364 / 1 | 0.002194 / 1 | 0.002295 / 1 |
| op:alpha / beta / gate | nested | 0.072495 / 1 | 0.072155 / 1 | 0.057668 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 98.447998 / 1 | 168.094309 / 1 | 160.702888 / 1 |

### Layer 91

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005941 / 1 | 0.006091 / 1 | 0.129101 / 1 |
| pre-attention-aggregation | boundary | 0.039584 / 1 | 0.024125 / 1 | 0.019717 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011111 / 1 | 0.011211 / 1 | 0.011030 / 1 |
| QA | nested | 0.291905 / 1 | 0.447716 / 1 | 1.405286 / 1 |
| QB | nested | 0.730395 / 1 | 2.374317 / 1 | 3.269770 / 1 |
| KA | nested | 0.103374 / 1 | 0.289731 / 1 | 0.678728 / 1 |
| KB | nested | 0.307554 / 1 | 0.581998 / 1 | 1.588327 / 1 |
| G | nested | 1.970862 / 1 | 4.873226 / 1 | 15.418860 / 1 |
| O | nested | 1.947499 / 1 | 4.646922 / 1 | 25.760192 / 1 |
| attention | boundary | 5.457217 / 1 | 13.339354 / 1 | 48.621097 / 1 |
| attention-residual | boundary | 0.003346 / 1 | 0.003036 / 1 | 0.002745 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027742 / 1 | 0.028604 / 1 | 0.038863 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002495 / 1 | 0.100438 / 1 | 0.369591 / 1 |
| router-and-top16 | boundary | 0.567551 / 1 | 0.880014 / 1 | 2.952276 / 1 |
| EDOWN | nested | 0.567250 / 1 | 1.227874 / 1 | 5.025920 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.569014 / 1 | 1.229748 / 1 | 5.030769 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027111 / 1 | 0.006152 / 1 | 0.007624 / 1 |
| SH1 | nested | 0.958080 / 1 | 4.637685 / 1 | 5.422943 / 1 |
| SH3 | nested | 0.961837 / 1 | 2.850807 / 1 | 5.149131 / 1 |
| SH2 | nested | 0.966626 / 1 | 3.280880 / 1 | 4.612248 / 1 |
| shared-expert-during-read | boundary | 2.900820 / 1 | 10.787387 / 1 | 15.202625 / 1 |
| detail:expert-gate | nested | 25.826134 / 16 | 56.048457 / 16 | 24.751718 / 16 |
| detail:expert-up | nested | 23.984574 / 16 | 44.381715 / 16 | 23.779561 / 16 |
| detail:expert-activation | nested | 0.102242 / 16 | 0.121721 / 16 | 0.124062 / 16 |
| detail:expert-down | nested | 26.353390 / 16 | 48.703533 / 16 | 26.252663 / 16 |
| EUP | nested | 0.791088 / 1 | 2.390967 / 1 | 2.118849 / 1 |
| experts-mix-normalize-up | boundary | 77.500377 / 1 | 155.953144 / 1 | 77.428493 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002515 / 1 | 0.002845 / 1 | 0.002896 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000241 / 1 | 0.000321 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007454 / 1 | 0.009668 / 1 | 0.008556 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 989.636187 / 1 | 1472.235477 / 1 | 818.211470 / 1 |
| op:Q   int8 projection | nested | 9.595097 / 1 | 27.600531 / 1 | 70.448812 / 1 |
| op:X   mxfp4 expert proj | nested | 76.373572 / 1 | 149.361649 / 1 | 75.014120 / 1 |
| op:N   rmsnorm | nested | 0.024496 / 1 | 0.024335 / 1 | 0.025166 / 1 |
| op:SiTU + sigma | nested | 0.110207 / 1 | 0.132732 / 1 | 0.135393 / 1 |
| op:AR  snapshot aggregate | nested | 0.056505 / 1 | 0.042781 / 1 | 0.048210 / 1 |
| op:SA  softmax attention | nested | 0.007003 / 1 | 0.010449 / 1 | 0.011822 / 1 |
| op:router dot product | nested | 0.564785 / 1 | 0.876758 / 1 | 2.949401 / 1 |
| op:top-k selection | nested | 0.002405 / 1 | 0.002806 / 1 | 0.002305 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 87.142142 / 1 | 182.404417 / 1 | 149.846894 / 1 |

### Layer 92

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005490 / 1 | 0.006452 / 1 | 0.174216 / 1 |
| pre-attention-aggregation | boundary | 0.018554 / 1 | 0.023303 / 1 | 0.019456 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010550 / 1 | 0.011211 / 1 | 0.075241 / 1 |
| QA | nested | 0.339845 / 1 | 0.567600 / 1 | 1.347598 / 1 |
| QB | nested | 0.802269 / 1 | 1.450530 / 1 | 3.241196 / 1 |
| KA | nested | 0.125615 / 1 | 0.244447 / 1 | 0.574643 / 1 |
| KB | nested | 0.358490 / 1 | 0.508430 / 1 | 1.496877 / 1 |
| G | nested | 2.740250 / 1 | 5.251212 / 1 | 8.945685 / 1 |
| O | nested | 2.895711 / 1 | 6.119264 / 1 | 9.876554 / 1 |
| attention | boundary | 7.369360 / 1 | 14.253632 / 1 | 25.983650 / 1 |
| attention-residual | boundary | 0.002816 / 1 | 0.003877 / 1 | 0.003346 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.029275 / 1 | 0.029375 / 1 | 0.029074 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001974 / 1 | 0.101540 / 1 | 0.207928 / 1 |
| router-and-top16 | boundary | 0.583541 / 1 | 0.901966 / 1 | 4.142330 / 1 |
| EDOWN | nested | 0.824421 / 1 | 1.327421 / 1 | 3.555543 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.826164 / 1 | 1.329575 / 1 | 3.557517 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.023434 / 1 | 0.006332 / 1 | 0.006251 / 1 |
| SH1 | nested | 1.417248 / 1 | 3.054337 / 1 | 5.332273 / 1 |
| SH3 | nested | 1.501635 / 1 | 3.160486 / 1 | 4.741088 / 1 |
| SH2 | nested | 1.509631 / 1 | 3.262926 / 1 | 6.106781 / 1 |
| shared-expert-during-read | boundary | 4.445687 / 1 | 9.495452 / 1 | 16.199729 / 1 |
| detail:expert-gate | nested | 24.515164 / 16 | 45.056385 / 16 | 31.473318 / 16 |
| detail:expert-up | nested | 24.240284 / 16 | 38.697387 / 16 | 30.726321 / 16 |
| detail:expert-activation | nested | 0.102462 / 16 | 0.122301 / 16 | 0.128902 / 16 |
| detail:expert-down | nested | 25.155582 / 16 | 42.264079 / 16 | 30.395493 / 16 |
| EUP | nested | 0.903468 / 1 | 2.399584 / 1 | 4.134065 / 1 |
| experts-mix-normalize-up | boundary | 75.353126 / 1 | 130.037261 / 1 | 97.387355 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002124 / 1 | 0.003006 / 1 | 0.002905 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000100 / 1 | 0.000030 / 1 | 0.000111 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006853 / 1 | 0.008707 / 1 | 0.008466 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 981.915671 / 1 | 1392.867117 / 1 | 1099.462487 / 1 |
| op:Q   int8 projection | nested | 13.417290 / 1 | 27.344764 / 1 | 49.350891 / 1 |
| op:X   mxfp4 expert proj | nested | 74.121172 / 1 | 126.241860 / 1 | 92.837153 / 1 |
| op:N   rmsnorm | nested | 0.024156 / 1 | 0.025056 / 1 | 0.024986 / 1 |
| op:SiTU + sigma | nested | 0.112942 / 1 | 0.133380 / 1 | 0.140342 / 1 |
| op:AR  snapshot aggregate | nested | 0.036959 / 1 | 0.041377 / 1 | 0.037901 / 1 |
| op:SA  softmax attention | nested | 0.006502 / 1 | 0.008546 / 1 | 0.011411 / 1 |
| op:router dot product | nested | 0.580595 / 1 | 0.899160 / 1 | 4.139574 / 1 |
| op:top-k selection | nested | 0.002595 / 1 | 0.002094 / 1 | 0.002314 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 88.703189 / 1 | 156.236934 / 1 | 147.820398 / 1 |

### Output

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| final-aggregation-and-normalization | boundary | not executed | 0.050304 / 1 | 0.050885 / 1 |
| head-fill-and-projection | boundary | not executed | 15892.511291 / 1 | 51.030680 / 1 |
| argmax-and-cleanup | boundary | not executed | 0.181729 / 1 | 0.102913 / 1 |

