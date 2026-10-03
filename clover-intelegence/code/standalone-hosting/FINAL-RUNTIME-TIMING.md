# Current Runtime Timing

AX102 2026-10-03 final build; direct experts and tables, AVX2 float32 pair kernel, no page warm-up; timing only.

Source: [timing data](stage-profile-final-20261003.json). Output assertions passed: IDs 418 then 276 (Rain falls -> on the). All 279 layer totals are present, with 35-50 recorded measurements per layer, depending on layer type. These are boundaries and nested measurements, not a claim that every scalar operation has a separate timer.

Startup: **1.006694 s**. First output (two input positions): **4.726936 s**. Next output (one continuation position): **2.262324 s**. This measures the current source with profiling enabled; it is not a retroactive decomposition of the earlier 45.028-second run.

## Elapsed-Time Breakdown

| Work | First output, two input positions (s) | Next output (s) |
|---|---:|---:|
| Routed experts, mixing, latent norm and up projection | 1.732597 | 0.829760 |
| Vocabulary head preparation and projection | 0.054090 | 0.050937 |
| Attention across all 93 layers | 2.006439 | 0.942166 |
| Shared experts | 0.589574 | 0.279743 |
| Residual/state save, unbind and next-layer prediction | 0.054943 | 0.026139 |
| Latent down projection | 0.116020 | 0.054949 |
| Live router and top-16 | 0.108682 | 0.052654 |
| Layer-0 dense MLP | 0.035893 | 0.016350 |
| Other boundaries and unassigned timer overhead | 0.028697 | 0.009627 |
| Total | 4.726936 | 2.262324 |

First head use prepares the retained BF16 table and scores it. The head timer does not separate reading, decompression, layout conversion, checksums and scoring. The next output reuses that head table, not a cached answer. Dataset preparation is not run.

## Inside Routed Experts

Gate/up/down wall timers are nested inside routed-expert processing above; do not add them again. Each includes its block decoding, validation and multiplication.

| Nested measurement | First output (s) | Next output (s) |
|---|---:|---:|
| detail:expert-gate | 0.500180 | 0.240215 |
| detail:expert-up | 0.499452 | 0.239554 |
| detail:expert-activation | 0.017043 | 0.008619 |
| detail:expert-down | 0.531881 | 0.254219 |
| detail:read-ahead-wait | 0.000000 | 0.000000 |

Worker times below sum concurrent threads. They locate CPU work but are **not elapsed seconds**, and must not be added to the wall-time table. The read wrapper accesses already-prefetched bytes; it is not total disk I/O time.

| Worker phase | First output (summed worker s) | Next output (summed worker s) |
|---|---:|---:|
| read | 0.000000 | 0.000000 |
| decode | 0.000000 | 0.000000 |
| crc | 0.000000 | 0.000000 |
| math | 21.012403 | 10.030519 |

## Every Layer

Elapsed milliseconds. Layer 0 is dense; all remaining layers have routed and shared experts.

| Layer | Attention | First input | Second input | Continuation |
|---:|---|---:|---:|---:|
| 0 | KDA | 35.742725 | 29.423810 | 29.100085 |
| 1 | KDA | 27.877372 | 26.309663 | 25.815710 |
| 2 | KDA | 28.051768 | 26.393889 | 25.760827 |
| 3 | MLA | 21.359014 | 19.012730 | 18.825821 |
| 4 | KDA | 28.272029 | 26.731421 | 25.709381 |
| 5 | KDA | 27.922456 | 26.301678 | 25.296089 |
| 6 | KDA | 30.487057 | 26.136990 | 25.434578 |
| 7 | MLA | 20.899956 | 18.994405 | 18.590992 |
| 8 | KDA | 27.929999 | 26.728104 | 25.657684 |
| 9 | KDA | 27.369683 | 26.269658 | 25.090034 |
| 10 | KDA | 27.247715 | 26.229563 | 25.093981 |
| 11 | MLA | 20.143232 | 18.857160 | 18.015767 |
| 12 | KDA | 27.895095 | 26.233160 | 25.091056 |
| 13 | KDA | 27.189036 | 26.202452 | 25.013801 |
| 14 | KDA | 28.357238 | 26.386256 | 25.697499 |
| 15 | MLA | 22.754771 | 19.028069 | 18.439429 |
| 16 | KDA | 27.491009 | 26.310274 | 25.280109 |
| 17 | KDA | 27.285205 | 26.228020 | 24.891483 |
| 18 | KDA | 27.794226 | 26.335080 | 25.017870 |
| 19 | MLA | 20.111203 | 19.163812 | 18.288247 |
| 20 | KDA | 27.309721 | 26.433023 | 24.918343 |
| 21 | KDA | 27.250500 | 26.280188 | 25.223894 |
| 22 | KDA | 27.470671 | 26.419678 | 25.304415 |
| 23 | MLA | 20.269107 | 19.190232 | 18.681792 |
| 24 | KDA | 27.298440 | 25.826600 | 25.811552 |
| 25 | KDA | 27.330149 | 26.065796 | 25.757180 |
| 26 | KDA | 27.214913 | 25.751079 | 25.303643 |
| 27 | MLA | 20.228551 | 19.109652 | 20.285378 |
| 28 | KDA | 27.318427 | 26.137040 | 25.365629 |
| 29 | KDA | 27.378208 | 26.290086 | 25.378533 |
| 30 | KDA | 27.365295 | 26.077719 | 25.435029 |
| 31 | MLA | 20.171084 | 19.139898 | 19.379075 |
| 32 | KDA | 27.389100 | 26.136590 | 25.117315 |
| 33 | KDA | 27.430437 | 26.018208 | 25.520989 |
| 34 | KDA | 27.687416 | 26.328708 | 25.546627 |
| 35 | MLA | 20.266152 | 19.157070 | 18.974439 |
| 36 | KDA | 27.399228 | 26.066117 | 25.228393 |
| 37 | KDA | 27.400911 | 25.824726 | 25.310977 |
| 38 | KDA | 27.241594 | 25.926317 | 25.025142 |
| 39 | MLA | 20.205689 | 19.219216 | 18.299838 |
| 40 | KDA | 27.882100 | 26.279917 | 25.430249 |
| 41 | KDA | 27.747439 | 26.152780 | 25.664117 |
| 42 | KDA | 27.352541 | 25.587092 | 25.764724 |
| 43 | MLA | 20.067931 | 18.884731 | 18.687242 |
| 44 | KDA | 28.037561 | 26.016565 | 25.665670 |
| 45 | KDA | 31.834374 | 25.795653 | 25.733175 |
| 46 | KDA | 27.365846 | 25.600528 | 25.689344 |
| 47 | MLA | 20.198675 | 18.858442 | 19.150197 |
| 48 | KDA | 27.767547 | 26.366689 | 25.711605 |
| 49 | KDA | 27.364152 | 25.813055 | 25.344820 |
| 50 | KDA | 27.305984 | 25.803066 | 25.544473 |
| 51 | MLA | 20.174280 | 18.987333 | 18.983796 |
| 52 | KDA | 27.387105 | 26.046691 | 25.188718 |
| 53 | KDA | 27.678640 | 26.292801 | 25.706746 |
| 54 | KDA | 27.310432 | 25.739878 | 25.343207 |
| 55 | MLA | 20.185271 | 18.870023 | 18.824659 |
| 56 | KDA | 27.353623 | 26.037103 | 24.907142 |
| 57 | KDA | 27.401723 | 26.091224 | 25.024010 |
| 58 | KDA | 27.757066 | 26.284245 | 25.763572 |
| 59 | MLA | 20.569799 | 19.154395 | 18.486577 |
| 60 | KDA | 27.450203 | 25.790673 | 25.398861 |
| 61 | KDA | 29.979899 | 26.429306 | 30.009665 |
| 62 | KDA | 27.462276 | 26.078270 | 25.491745 |
| 63 | MLA | 20.254941 | 19.111956 | 18.306461 |
| 64 | KDA | 27.859858 | 26.399019 | 25.644530 |
| 65 | KDA | 28.051326 | 26.118235 | 24.888477 |
| 66 | KDA | 27.442039 | 26.126861 | 25.079434 |
| 67 | MLA | 20.165564 | 19.036174 | 18.516052 |
| 68 | KDA | 27.360245 | 25.862758 | 25.083822 |
| 69 | KDA | 27.812139 | 26.306596 | 25.204799 |
| 70 | KDA | 27.359144 | 26.221498 | 25.432905 |
| 71 | MLA | 20.678272 | 19.124850 | 18.543173 |
| 72 | KDA | 27.159340 | 26.101644 | 25.591460 |
| 73 | KDA | 27.192943 | 26.250442 | 25.728958 |
| 74 | KDA | 27.356528 | 25.965650 | 25.733536 |
| 75 | MLA | 20.105753 | 18.811363 | 19.173921 |
| 76 | KDA | 27.252083 | 26.026123 | 25.715412 |
| 77 | KDA | 27.180139 | 26.038897 | 25.341634 |
| 78 | KDA | 27.638946 | 26.226467 | 25.567556 |
| 79 | MLA | 20.282913 | 19.093922 | 18.906121 |
| 80 | KDA | 27.297959 | 25.948388 | 25.383332 |
| 81 | KDA | 27.418795 | 26.091294 | 25.102017 |
| 82 | KDA | 27.326102 | 25.912771 | 25.146219 |
| 83 | MLA | 20.193115 | 18.603035 | 18.996089 |
| 84 | KDA | 27.519242 | 25.643829 | 25.153733 |
| 85 | KDA | 27.527978 | 25.837942 | 25.049348 |
| 86 | KDA | 27.469029 | 25.443244 | 24.878328 |
| 87 | MLA | 20.281871 | 18.748106 | 18.926179 |
| 88 | KDA | 28.124523 | 26.037785 | 25.280159 |
| 89 | KDA | 27.478897 | 25.746430 | 25.207584 |
| 90 | KDA | 27.608328 | 25.873007 | 25.267265 |
| 91 | MLA | 20.202143 | 18.986431 | 18.668447 |
| 92 | MLA | 20.169631 | 18.897465 | 18.891143 |

## Every Recorded Measurement

Milliseconds. Boundary rows are disjoint within their layer; projections, operator aggregates, details and worker rows are nested or parallel and cannot be summed with them.

### Input

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| request-reset | boundary | 2.346224 / 1 | 0.189053 / 1 | 0.140352 / 1 |
| embedding | boundary | 0.106059 / 1 | 0.034635 / 1 | 0.040045 / 1 |
| scratch-setup | boundary | 0.005360 / 1 | 0.000350 / 1 | 0.000821 / 1 |

### Layer 0

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.023223 / 1 | 0.005911 / 1 | 0.004047 / 1 |
| pre-attention-aggregation | boundary | 0.011802 / 1 | 0.002494 / 1 | 0.001223 / 1 |
| snapshot-push | boundary | 0.000632 / 1 | 0.001002 / 1 | 0.000581 / 1 |
| pre-attention-normalization | boundary | 0.533286 / 1 | 0.020979 / 1 | 0.024295 / 1 |
| Q | nested | 3.823173 / 1 | 2.518585 / 1 | 2.481626 / 1 |
| K | nested | 2.931957 / 1 | 2.446831 / 1 | 2.543071 / 1 |
| V | nested | 2.381599 / 1 | 2.265733 / 1 | 2.340744 / 1 |
| B | nested | 0.110326 / 1 | 0.026450 / 1 | 0.023604 / 1 |
| FA | nested | 0.066494 / 1 | 0.028714 / 1 | 0.026129 / 1 |
| FB | nested | 0.082174 / 1 | 0.046868 / 1 | 0.046808 / 1 |
| G | nested | 2.438035 / 1 | 1.983516 / 1 | 1.976362 / 1 |
| O | nested | 2.386058 / 1 | 2.088192 / 1 | 2.009925 / 1 |
| attention | boundary | 15.332634 / 1 | 12.496446 / 1 | 12.345214 / 1 |
| attention-residual | boundary | 0.001733 / 1 | 0.001603 / 1 | 0.001984 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.041578 / 1 | 0.026369 / 1 | 0.024987 / 1 |
| MGATE | nested | 6.534919 / 1 | 5.572511 / 1 | 5.557673 / 1 |
| MUP | nested | 6.468013 / 1 | 5.243325 / 1 | 5.272440 / 1 |
| MDOWN | nested | 6.352377 / 1 | 5.552563 / 1 | 5.479768 / 1 |
| dense-mlp | boundary | 19.439978 / 1 | 16.452728 / 1 | 16.349785 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000391 / 1 | 0.000401 / 1 | 0.000481 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.356646 / 1 | 0.415206 / 1 | 0.347098 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| op:Q   int8 projection | nested | 33.573683 / 1 | 27.771875 / 1 | 27.756695 / 1 |
| op:N   rmsnorm | nested | 0.549267 / 1 | 0.047590 / 1 | 0.039935 / 1 |
| op:L   l2 per-head | nested | 0.005450 / 1 | 0.004990 / 1 | 0.005320 / 1 |
| op:SiTU + sigma | nested | 0.036208 / 1 | 0.037510 / 1 | 0.037610 / 1 |
| op:C   shortconv | nested | 0.085890 / 1 | 0.062838 / 1 | 0.059160 / 1 |
| op:AR  snapshot aggregate | nested | 0.024416 / 1 | 0.016411 / 1 | 0.015349 / 1 |
| op:D   kda delta-rule | nested | 0.189895 / 1 | 0.168765 / 1 | 0.148738 / 1 |
| op:alpha / beta / gate | nested | 0.076062 / 1 | 0.071664 / 1 | 0.073417 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 35.742725 / 1 | 29.423810 / 1 | 29.100085 / 1 |

### Layer 1

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010911 / 1 | 0.005150 / 1 | 0.004729 / 1 |
| pre-attention-aggregation | boundary | 0.014948 / 1 | 0.017853 / 1 | 0.015239 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000160 / 1 |
| pre-attention-normalization | boundary | 0.012664 / 1 | 0.014337 / 1 | 0.012604 / 1 |
| Q | nested | 2.516632 / 1 | 2.451080 / 1 | 2.370819 / 1 |
| K | nested | 2.369026 / 1 | 2.322459 / 1 | 2.440410 / 1 |
| V | nested | 2.145879 / 1 | 2.187507 / 1 | 2.268838 / 1 |
| B | nested | 0.066674 / 1 | 0.026469 / 1 | 0.026710 / 1 |
| FA | nested | 0.054812 / 1 | 0.027662 / 1 | 0.028894 / 1 |
| FB | nested | 0.077735 / 1 | 0.049653 / 1 | 0.050434 / 1 |
| G | nested | 2.266054 / 1 | 1.979959 / 1 | 2.022659 / 1 |
| O | nested | 2.299015 / 1 | 2.038728 / 1 | 2.041043 / 1 |
| attention | boundary | 12.691471 / 1 | 12.145841 / 1 | 11.854487 / 1 |
| attention-residual | boundary | 0.003506 / 1 | 0.003527 / 1 | 0.003747 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024886 / 1 | 0.025418 / 1 | 0.025117 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009778 / 1 | 0.002394 / 1 | 0.002565 / 1 |
| router-and-top16 | boundary | 0.602005 / 1 | 0.570626 / 1 | 0.568833 / 1 |
| EDOWN | nested | 0.694968 / 1 | 0.587898 / 1 | 0.577289 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.695359 / 1 | 0.588219 / 1 | 0.577980 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.096480 / 1 | 0.033132 / 1 | 0.006742 / 1 |
| SH1 | nested | 1.144378 / 1 | 1.027290 / 1 | 0.980402 / 1 |
| SH3 | nested | 1.176728 / 1 | 1.064759 / 1 | 1.032289 / 1 |
| SH2 | nested | 1.136193 / 1 | 0.995250 / 1 | 0.981052 / 1 |
| shared-expert-during-read | boundary | 3.466756 / 1 | 3.101434 / 1 | 3.004002 / 1 |
| detail:expert-gate | nested | 2.839485 / 16 | 2.709954 / 16 | 2.751931 / 16 |
| detail:expert-up | nested | 2.842481 / 16 | 2.733878 / 16 | 2.742141 / 16 |
| detail:expert-activation | nested | 0.088274 / 16 | 0.091219 / 16 | 0.089879 / 16 |
| detail:expert-down | nested | 3.066778 / 16 | 2.884007 / 16 | 2.915948 / 16 |
| EUP | nested | 0.672948 / 1 | 0.621311 / 1 | 0.614468 / 1 |
| experts-mix-normalize-up | boundary | 9.898402 / 1 | 9.383650 / 1 | 9.405833 / 1 |
| mlp-merge-and-cleanup | boundary | 0.005300 / 1 | 0.002915 / 1 | 0.002424 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000170 / 1 | 0.000240 / 1 | 0.000190 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.343121 / 1 | 0.413062 / 1 | 0.329575 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.949303 / 1 | 114.474167 / 1 | 114.364887 / 1 |
| op:Q   int8 projection | nested | 16.619480 / 1 | 15.378011 / 1 | 15.433319 / 1 |
| op:X   mxfp4 expert proj | nested | 8.846998 / 1 | 8.429339 / 1 | 8.509781 / 1 |
| op:N   rmsnorm | nested | 0.033963 / 1 | 0.034093 / 1 | 0.023606 / 1 |
| op:L   l2 per-head | nested | 0.005821 / 1 | 0.005831 / 1 | 0.006282 / 1 |
| op:SiTU + sigma | nested | 0.096139 / 1 | 0.102972 / 1 | 0.098293 / 1 |
| op:C   shortconv | nested | 0.072996 / 1 | 0.069400 / 1 | 0.036178 / 1 |
| op:AR  snapshot aggregate | nested | 0.029345 / 1 | 0.032470 / 1 | 0.030085 / 1 |
| op:D   kda delta-rule | nested | 0.181399 / 1 | 0.133349 / 1 | 0.232875 / 1 |
| op:router dot product | nested | 0.598838 / 1 | 0.566969 / 1 | 0.565857 / 1 |
| op:top-k selection | nested | 0.002695 / 1 | 0.002455 / 1 | 0.002726 / 1 |
| op:alpha / beta / gate | nested | 0.073287 / 1 | 0.075661 / 1 | 0.057627 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.877372 / 1 | 26.309663 / 1 | 25.815710 / 1 |

### Layer 2

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010941 / 1 | 0.005810 / 1 | 0.004278 / 1 |
| pre-attention-aggregation | boundary | 0.014317 / 1 | 0.015939 / 1 | 0.015278 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011752 / 1 | 0.011702 / 1 | 0.011130 / 1 |
| Q | nested | 2.519567 / 1 | 2.425882 / 1 | 2.413619 / 1 |
| K | nested | 2.312109 / 1 | 2.404492 / 1 | 2.487667 / 1 |
| V | nested | 2.108470 / 1 | 2.243952 / 1 | 2.271293 / 1 |
| B | nested | 0.100337 / 1 | 0.020809 / 1 | 0.021220 / 1 |
| FA | nested | 0.058048 / 1 | 0.028253 / 1 | 0.026970 / 1 |
| FB | nested | 0.072455 / 1 | 0.050174 / 1 | 0.048421 / 1 |
| G | nested | 2.344510 / 1 | 1.959491 / 1 | 1.983025 / 1 |
| O | nested | 2.367813 / 1 | 2.042997 / 1 | 2.047405 / 1 |
| attention | boundary | 12.801026 / 1 | 12.241279 / 1 | 11.924027 / 1 |
| attention-residual | boundary | 0.003637 / 1 | 0.003537 / 1 | 0.003908 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024817 / 1 | 0.026009 / 1 | 0.025347 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010309 / 1 | 0.002765 / 1 | 0.002504 / 1 |
| router-and-top16 | boundary | 0.598999 / 1 | 0.580364 / 1 | 0.574032 / 1 |
| EDOWN | nested | 0.664902 / 1 | 0.611613 / 1 | 0.605672 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.665353 / 1 | 0.612114 / 1 | 0.606173 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.089367 / 1 | 0.023584 / 1 | 0.015038 / 1 |
| SH1 | nested | 1.156461 / 1 | 1.040694 / 1 | 1.000879 / 1 |
| SH3 | nested | 1.162572 / 1 | 1.039522 / 1 | 1.002352 / 1 |
| SH2 | nested | 1.135902 / 1 | 1.012852 / 1 | 0.998606 / 1 |
| shared-expert-during-read | boundary | 3.464232 / 1 | 3.103058 / 1 | 3.011997 / 1 |
| detail:expert-gate | nested | 2.857075 / 16 | 2.735959 / 16 | 2.691417 / 16 |
| detail:expert-up | nested | 2.905879 / 16 | 2.766407 / 16 | 2.668736 / 16 |
| detail:expert-activation | nested | 0.087944 / 16 | 0.092893 / 16 | 0.090729 / 16 |
| detail:expert-down | nested | 3.091534 / 16 | 2.889739 / 16 | 2.837438 / 16 |
| EUP | nested | 0.668729 / 1 | 0.596454 / 1 | 0.615239 / 1 |
| experts-mix-normalize-up | boundary | 9.996897 / 1 | 9.399089 / 1 | 9.205558 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003236 / 1 | 0.002925 / 1 | 0.002956 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000150 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.354752 / 1 | 0.363639 / 1 | 0.356726 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 117.440654 / 1 | 114.463395 / 1 | 111.817955 / 1 |
| op:Q   int8 projection | nested | 16.670363 / 1 | 15.475703 / 1 | 15.520876 / 1 |
| op:X   mxfp4 expert proj | nested | 8.952916 / 1 | 8.496543 / 1 | 8.299478 / 1 |
| op:N   rmsnorm | nested | 0.036718 / 1 | 0.032871 / 1 | 0.024156 / 1 |
| op:L   l2 per-head | nested | 0.005621 / 1 | 0.005210 / 1 | 0.005580 / 1 |
| op:SiTU + sigma | nested | 0.095597 / 1 | 0.101050 / 1 | 0.098684 / 1 |
| op:C   shortconv | nested | 0.050713 / 1 | 0.067807 / 1 | 0.034665 / 1 |
| op:AR  snapshot aggregate | nested | 0.029304 / 1 | 0.031268 / 1 | 0.030207 / 1 |
| op:D   kda delta-rule | nested | 0.218768 / 1 | 0.150762 / 1 | 0.246150 / 1 |
| op:router dot product | nested | 0.596103 / 1 | 0.577539 / 1 | 0.570787 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002284 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.077575 / 1 | 0.073547 / 1 | 0.057707 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.051768 / 1 | 26.393889 / 1 | 25.760827 / 1 |

### Layer 3

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010790 / 1 | 0.004548 / 1 | 0.005290 / 1 |
| pre-attention-aggregation | boundary | 0.014057 / 1 | 0.016241 / 1 | 0.015158 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000261 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010109 / 1 | 0.010399 / 1 | 0.011611 / 1 |
| QA | nested | 0.301964 / 1 | 0.276055 / 1 | 0.305141 / 1 |
| QB | nested | 0.742667 / 1 | 0.661686 / 1 | 0.696541 / 1 |
| KA | nested | 0.128280 / 1 | 0.103233 / 1 | 0.104054 / 1 |
| KB | nested | 0.372576 / 1 | 0.305801 / 1 | 0.302274 / 1 |
| G | nested | 2.389093 / 1 | 2.076119 / 1 | 2.017730 / 1 |
| O | nested | 2.392711 / 1 | 2.090005 / 1 | 2.019733 / 1 |
| attention | boundary | 6.477381 / 1 | 5.607065 / 1 | 5.539960 / 1 |
| attention-residual | boundary | 0.002495 / 1 | 0.003065 / 1 | 0.003237 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024717 / 1 | 0.025016 / 1 | 0.026980 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009488 / 1 | 0.002154 / 1 | 0.006022 / 1 |
| router-and-top16 | boundary | 0.626551 / 1 | 0.590994 / 1 | 0.600973 / 1 |
| EDOWN | nested | 0.691262 / 1 | 0.584372 / 1 | 0.581396 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.691652 / 1 | 0.584723 / 1 | 0.581787 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.080050 / 1 | 0.032340 / 1 | 0.011812 / 1 |
| SH1 | nested | 1.180245 / 1 | 0.982786 / 1 | 0.992003 / 1 |
| SH3 | nested | 1.165928 / 1 | 0.997113 / 1 | 0.999657 / 1 |
| SH2 | nested | 1.133588 / 1 | 0.998976 / 1 | 0.996672 / 1 |
| shared-expert-during-read | boundary | 3.489039 / 1 | 2.988383 / 1 | 2.998472 / 1 |
| detail:expert-gate | nested | 2.849023 / 16 | 2.645593 / 16 | 2.633058 / 16 |
| detail:expert-up | nested | 2.874673 / 16 | 2.646782 / 16 | 2.649311 / 16 |
| detail:expert-activation | nested | 0.088236 / 16 | 0.090134 / 16 | 0.093414 / 16 |
| detail:expert-down | nested | 3.089142 / 16 | 2.809069 / 16 | 2.750366 / 16 |
| EUP | nested | 0.645075 / 1 | 0.605972 / 1 | 0.599280 / 1 |
| experts-mix-normalize-up | boundary | 9.911216 / 1 | 9.136499 / 1 | 9.013779 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003346 / 1 | 0.003286 / 1 | 0.003747 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000360 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005921 / 1 | 0.005841 / 1 | 0.004969 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.878719 / 1 | 111.354145 / 1 | 110.277774 / 1 |
| op:Q   int8 projection | nested | 11.142095 / 1 | 9.680855 / 1 | 9.613188 / 1 |
| op:X   mxfp4 expert proj | nested | 8.912600 / 1 | 8.203154 / 1 | 8.141100 / 1 |
| op:N   rmsnorm | nested | 0.024695 / 1 | 0.023774 / 1 | 0.024046 / 1 |
| op:SiTU + sigma | nested | 0.096101 / 1 | 0.097928 / 1 | 0.101531 / 1 |
| op:AR  snapshot aggregate | nested | 0.028083 / 1 | 0.030176 / 1 | 0.030918 / 1 |
| op:SA  softmax attention | nested | 0.012984 / 1 | 0.008426 / 1 | 0.012313 / 1 |
| op:router dot product | nested | 0.623435 / 1 | 0.588049 / 1 | 0.597937 / 1 |
| op:top-k selection | nested | 0.002554 / 1 | 0.002374 / 1 | 0.002685 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 21.359014 / 1 | 19.012730 / 1 | 18.825821 / 1 |

### Layer 4

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010911 / 1 | 0.004519 / 1 | 0.005149 / 1 |
| pre-attention-aggregation | boundary | 0.014628 / 1 | 0.014988 / 1 | 0.014948 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.011491 / 1 | 0.010820 / 1 |
| Q | nested | 2.722887 / 1 | 2.660911 / 1 | 2.665780 / 1 |
| K | nested | 2.311569 / 1 | 2.560334 / 1 | 2.437685 / 1 |
| V | nested | 2.173610 / 1 | 2.263509 / 1 | 2.193468 / 1 |
| B | nested | 0.088806 / 1 | 0.024636 / 1 | 0.028744 / 1 |
| FA | nested | 0.056656 / 1 | 0.030237 / 1 | 0.031740 / 1 |
| FB | nested | 0.075251 / 1 | 0.048571 / 1 | 0.049523 / 1 |
| G | nested | 2.269350 / 1 | 1.973658 / 1 | 2.001429 / 1 |
| O | nested | 2.307481 / 1 | 2.124710 / 1 | 2.132043 / 1 |
| attention | boundary | 13.086899 / 1 | 12.775057 / 1 | 12.259724 / 1 |
| attention-residual | boundary | 0.003336 / 1 | 0.004078 / 1 | 0.003446 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025167 / 1 | 0.025348 / 1 | 0.024205 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009928 / 1 | 0.002635 / 1 | 0.002284 / 1 |
| router-and-top16 | boundary | 0.609558 / 1 | 0.572510 / 1 | 0.570586 / 1 |
| EDOWN | nested | 0.672046 / 1 | 0.587548 / 1 | 0.584171 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.672757 / 1 | 0.588069 / 1 | 0.584672 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.096741 / 1 | 0.032431 / 1 | 0.014266 / 1 |
| SH1 | nested | 1.170988 / 1 | 1.025135 / 1 | 1.012541 / 1 |
| SH3 | nested | 1.144468 / 1 | 1.020407 / 1 | 1.004978 / 1 |
| SH2 | nested | 1.160358 / 1 | 1.023402 / 1 | 0.997253 / 1 |
| shared-expert-during-read | boundary | 3.491253 / 1 | 3.079273 / 1 | 3.025212 / 1 |
| detail:expert-gate | nested | 2.805583 / 16 | 2.650851 / 16 | 2.550673 / 16 |
| detail:expert-up | nested | 2.847480 / 16 | 2.673211 / 16 | 2.563850 / 16 |
| detail:expert-activation | nested | 0.089046 / 16 | 0.090858 / 16 | 0.094638 / 16 |
| detail:expert-down | nested | 3.034197 / 16 | 2.821057 / 16 | 2.673303 / 16 |
| EUP | nested | 0.656136 / 1 | 0.599259 / 1 | 0.603778 / 1 |
| experts-mix-normalize-up | boundary | 9.826699 / 1 | 9.185451 / 1 | 8.803957 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003116 / 1 | 0.002555 / 1 | 0.003887 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000170 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.406710 / 1 | 0.430374 / 1 | 0.383827 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.926595 / 1 | 111.642679 / 1 | 106.948044 / 1 |
| op:Q   int8 projection | nested | 16.807531 / 1 | 15.940351 / 1 | 15.741521 / 1 |
| op:X   mxfp4 expert proj | nested | 8.789273 / 1 | 8.248893 / 1 | 7.895710 / 1 |
| op:N   rmsnorm | nested | 0.034445 / 1 | 0.046818 / 1 | 0.023434 / 1 |
| op:L   l2 per-head | nested | 0.005721 / 1 | 0.005320 / 1 | 0.005190 / 1 |
| op:SiTU + sigma | nested | 0.099505 / 1 | 0.098489 / 1 | 0.102803 / 1 |
| op:C   shortconv | nested | 0.075882 / 1 | 0.069359 / 1 | 0.037680 / 1 |
| op:AR  snapshot aggregate | nested | 0.029975 / 1 | 0.030187 / 1 | 0.028823 / 1 |
| op:D   kda delta-rule | nested | 0.145742 / 1 | 0.142476 / 1 | 0.136685 / 1 |
| op:router dot product | nested | 0.606743 / 1 | 0.569585 / 1 | 0.567680 / 1 |
| op:top-k selection | nested | 0.002224 / 1 | 0.002264 / 1 | 0.002535 / 1 |
| op:alpha / beta / gate | nested | 0.074519 / 1 | 0.073267 / 1 | 0.057858 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.272029 / 1 | 26.731421 / 1 | 25.709381 / 1 |

### Layer 5

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012794 / 1 | 0.005240 / 1 | 0.005400 / 1 |
| pre-attention-aggregation | boundary | 0.016161 / 1 | 0.016461 / 1 | 0.016171 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012173 / 1 | 0.013034 / 1 | 0.013235 / 1 |
| Q | nested | 2.537631 / 1 | 2.457942 / 1 | 2.457231 / 1 |
| K | nested | 2.289036 / 1 | 2.322499 / 1 | 2.275691 / 1 |
| V | nested | 2.151289 / 1 | 2.181886 / 1 | 2.157621 / 1 |
| B | nested | 0.071574 / 1 | 0.020809 / 1 | 0.020949 / 1 |
| FA | nested | 0.058109 / 1 | 0.032200 / 1 | 0.033984 / 1 |
| FB | nested | 0.066474 / 1 | 0.056916 / 1 | 0.055995 / 1 |
| G | nested | 2.276303 / 1 | 2.022919 / 1 | 2.025093 / 1 |
| O | nested | 2.296270 / 1 | 2.049209 / 1 | 2.053026 / 1 |
| attention | boundary | 12.799172 / 1 | 12.216754 / 1 | 11.828018 / 1 |
| attention-residual | boundary | 0.003517 / 1 | 0.003427 / 1 | 0.003937 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024366 / 1 | 0.025688 / 1 | 0.025237 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009979 / 1 | 0.002615 / 1 | 0.002645 / 1 |
| router-and-top16 | boundary | 0.623094 / 1 | 0.581797 / 1 | 0.590183 / 1 |
| EDOWN | nested | 0.691983 / 1 | 0.625258 / 1 | 0.624928 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.692524 / 1 | 0.625739 / 1 | 0.625469 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.082094 / 1 | 0.018715 / 1 | 0.007033 / 1 |
| SH1 | nested | 1.153385 / 1 | 1.055792 / 1 | 1.050001 / 1 |
| SH3 | nested | 1.137755 / 1 | 1.029744 / 1 | 1.036576 / 1 |
| SH2 | nested | 1.131995 / 1 | 1.001421 / 1 | 0.999207 / 1 |
| shared-expert-during-read | boundary | 3.432904 / 1 | 3.097658 / 1 | 3.096004 / 1 |
| detail:expert-gate | nested | 2.809348 / 16 | 2.690805 / 16 | 2.503567 / 16 |
| detail:expert-up | nested | 2.800952 / 16 | 2.702595 / 16 | 2.522494 / 16 |
| detail:expert-activation | nested | 0.087764 / 16 | 0.091770 / 16 | 0.089209 / 16 |
| detail:expert-down | nested | 3.061861 / 16 | 2.865062 / 16 | 2.657666 / 16 |
| EUP | nested | 0.656196 / 1 | 0.597676 / 1 | 0.587177 / 1 |
| experts-mix-normalize-up | boundary | 9.791603 / 1 | 9.276761 / 1 | 8.670699 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003286 / 1 | 0.003527 / 1 | 0.003226 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000280 / 1 | 0.000160 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.416167 / 1 | 0.411569 / 1 | 0.406400 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.669211 / 1 | 114.726996 / 1 | 106.162617 / 1 |
| op:Q   int8 projection | nested | 16.515986 / 1 | 15.452710 / 1 | 15.375666 / 1 |
| op:X   mxfp4 expert proj | nested | 8.773730 / 1 | 8.363745 / 1 | 7.787358 / 1 |
| op:N   rmsnorm | nested | 0.033542 / 1 | 0.035356 / 1 | 0.024505 / 1 |
| op:L   l2 per-head | nested | 0.005550 / 1 | 0.005621 / 1 | 0.005701 / 1 |
| op:SiTU + sigma | nested | 0.095739 / 1 | 0.099936 / 1 | 0.097333 / 1 |
| op:C   shortconv | nested | 0.051786 / 1 | 0.058269 / 1 | 0.033403 / 1 |
| op:AR  snapshot aggregate | nested | 0.030417 / 1 | 0.031890 / 1 | 0.031378 / 1 |
| op:D   kda delta-rule | nested | 0.155872 / 1 | 0.160069 / 1 | 0.142777 / 1 |
| op:router dot product | nested | 0.620099 / 1 | 0.578811 / 1 | 0.587197 / 1 |
| op:top-k selection | nested | 0.002354 / 1 | 0.002535 / 1 | 0.002534 / 1 |
| op:alpha / beta / gate | nested | 0.071544 / 1 | 0.073568 / 1 | 0.059110 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.922456 / 1 | 26.301678 / 1 | 25.296089 / 1 |

### Layer 6

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012323 / 1 | 0.005490 / 1 | 0.005140 / 1 |
| pre-attention-aggregation | boundary | 0.020458 / 1 | 0.016120 / 1 | 0.015489 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011621 / 1 | 0.011011 / 1 | 0.011702 / 1 |
| Q | nested | 2.542570 / 1 | 2.479102 / 1 | 2.451139 / 1 |
| K | nested | 2.223444 / 1 | 2.305527 / 1 | 2.239844 / 1 |
| V | nested | 2.125541 / 1 | 2.208035 / 1 | 2.123327 / 1 |
| B | nested | 0.076924 / 1 | 0.021730 / 1 | 0.021490 / 1 |
| FA | nested | 0.058339 / 1 | 0.037059 / 1 | 0.034805 / 1 |
| FB | nested | 0.080310 / 1 | 0.047909 / 1 | 0.049202 / 1 |
| G | nested | 2.253570 / 1 | 2.010236 / 1 | 2.007000 / 1 |
| O | nested | 2.281463 / 1 | 2.113729 / 1 | 2.117856 / 1 |
| attention | boundary | 12.680460 / 1 | 12.301482 / 1 | 11.793363 / 1 |
| attention-residual | boundary | 0.004058 / 1 | 0.003606 / 1 | 0.003968 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024325 / 1 | 0.025698 / 1 | 0.024867 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.013064 / 1 | 0.002384 / 1 | 0.002194 / 1 |
| router-and-top16 | boundary | 0.616562 / 1 | 0.583750 / 1 | 0.571799 / 1 |
| EDOWN | nested | 0.659933 / 1 | 0.575034 / 1 | 0.577379 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.660444 / 1 | 0.575475 / 1 | 0.577890 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.088495 / 1 | 0.024055 / 1 | 0.016050 / 1 |
| SH1 | nested | 1.135451 / 1 | 1.003965 / 1 | 1.007362 / 1 |
| SH3 | nested | 1.142254 / 1 | 1.036446 / 1 | 1.016619 / 1 |
| SH2 | nested | 1.139819 / 1 | 0.997905 / 1 | 0.992855 / 1 |
| shared-expert-during-read | boundary | 3.428114 / 1 | 3.048355 / 1 | 3.027346 / 1 |
| detail:expert-gate | nested | 3.695053 / 16 | 2.686340 / 16 | 2.604064 / 16 |
| detail:expert-up | nested | 3.606999 / 16 | 2.614525 / 16 | 2.600769 / 16 |
| detail:expert-activation | nested | 0.104668 / 16 | 0.089536 / 16 | 0.090780 / 16 |
| detail:expert-down | nested | 4.057512 / 16 | 2.810728 / 16 | 2.775095 / 16 |
| EUP | nested | 0.658000 / 1 | 0.604560 / 1 | 0.594601 / 1 |
| experts-mix-normalize-up | boundary | 12.514079 / 1 | 9.132411 / 1 | 8.977753 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003046 / 1 | 0.003576 / 1 | 0.003026 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000531 / 1 | 0.000251 / 1 | 0.000311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.407211 / 1 | 0.401119 / 1 | 0.401400 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 119.401274 / 1 | 112.001648 / 1 | 110.464795 / 1 |
| op:Q   int8 projection | nested | 16.376006 / 1 | 15.439463 / 1 | 15.231467 / 1 |
| op:X   mxfp4 expert proj | nested | 11.479577 / 1 | 8.216031 / 1 | 8.085786 / 1 |
| op:N   rmsnorm | nested | 0.032230 / 1 | 0.032451 / 1 | 0.022763 / 1 |
| op:L   l2 per-head | nested | 0.005630 / 1 | 0.005480 / 1 | 0.005770 / 1 |
| op:SiTU + sigma | nested | 0.112953 / 1 | 0.097372 / 1 | 0.098834 / 1 |
| op:C   shortconv | nested | 0.052518 / 1 | 0.060824 / 1 | 0.033563 / 1 |
| op:AR  snapshot aggregate | nested | 0.034876 / 1 | 0.031578 / 1 | 0.030046 / 1 |
| op:D   kda delta-rule | nested | 0.135162 / 1 | 0.136616 / 1 | 0.131315 / 1 |
| op:router dot product | nested | 0.613326 / 1 | 0.580755 / 1 | 0.568973 / 1 |
| op:top-k selection | nested | 0.002765 / 1 | 0.002625 / 1 | 0.002544 / 1 |
| op:alpha / beta / gate | nested | 0.070552 / 1 | 0.073227 / 1 | 0.059161 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 30.487057 / 1 | 26.136990 / 1 | 25.434578 / 1 |

### Layer 7

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012163 / 1 | 0.005531 / 1 | 0.005640 / 1 |
| pre-attention-aggregation | boundary | 0.016060 / 1 | 0.015469 / 1 | 0.015749 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010950 / 1 | 0.010419 / 1 | 0.010840 / 1 |
| QA | nested | 0.308336 / 1 | 0.302224 / 1 | 0.304069 / 1 |
| QB | nested | 0.731767 / 1 | 0.690581 / 1 | 0.692454 / 1 |
| KA | nested | 0.128941 / 1 | 0.106489 / 1 | 0.105657 / 1 |
| KB | nested | 0.342159 / 1 | 0.303006 / 1 | 0.314378 / 1 |
| G | nested | 2.276543 / 1 | 1.958970 / 1 | 1.958348 / 1 |
| O | nested | 2.322088 / 1 | 2.083232 / 1 | 2.085366 / 1 |
| attention | boundary | 6.217286 / 1 | 5.538547 / 1 | 5.554758 / 1 |
| attention-residual | boundary | 0.002615 / 1 | 0.002665 / 1 | 0.002615 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023895 / 1 | 0.025307 / 1 | 0.025979 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009969 / 1 | 0.002124 / 1 | 0.002355 / 1 |
| router-and-top16 | boundary | 0.611853 / 1 | 0.574293 / 1 | 0.575725 / 1 |
| EDOWN | nested | 0.677165 / 1 | 0.582789 / 1 | 0.593809 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.677666 / 1 | 0.583220 / 1 | 0.594280 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.077776 / 1 | 0.032521 / 1 | 0.011631 / 1 |
| SH1 | nested | 1.150700 / 1 | 0.985782 / 1 | 0.987986 / 1 |
| SH3 | nested | 1.130372 / 1 | 0.963299 / 1 | 0.959333 / 1 |
| SH2 | nested | 1.151551 / 1 | 1.012932 / 1 | 1.022010 / 1 |
| shared-expert-during-read | boundary | 3.442361 / 1 | 2.972884 / 1 | 2.979706 / 1 |
| detail:expert-gate | nested | 2.823675 / 16 | 2.654680 / 16 | 2.544975 / 16 |
| detail:expert-up | nested | 2.800363 / 16 | 2.661410 / 16 | 2.529085 / 16 |
| detail:expert-activation | nested | 0.088123 / 16 | 0.093455 / 16 | 0.091652 / 16 |
| detail:expert-down | nested | 3.041562 / 16 | 2.852167 / 16 | 2.736723 / 16 |
| EUP | nested | 0.659693 / 1 | 0.617033 / 1 | 0.588369 / 1 |
| experts-mix-normalize-up | boundary | 9.785242 / 1 | 9.219423 / 1 | 8.800431 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003346 / 1 | 0.003406 / 1 | 0.003176 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005811 / 1 | 0.005520 / 1 | 0.005260 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.579807 / 1 | 112.593452 / 1 | 107.755425 / 1 |
| op:Q   int8 projection | nested | 10.877842 / 1 | 9.605024 / 1 | 9.610534 / 1 |
| op:X   mxfp4 expert proj | nested | 8.769694 / 1 | 8.280842 / 1 | 7.923192 / 1 |
| op:N   rmsnorm | nested | 0.023484 / 1 | 0.023324 / 1 | 0.023905 / 1 |
| op:SiTU + sigma | nested | 0.095668 / 1 | 0.101531 / 1 | 0.099675 / 1 |
| op:AR  snapshot aggregate | nested | 0.029746 / 1 | 0.030448 / 1 | 0.030397 / 1 |
| op:SA  softmax attention | nested | 0.006943 / 1 | 0.009067 / 1 | 0.011151 / 1 |
| op:router dot product | nested | 0.609088 / 1 | 0.571648 / 1 | 0.572981 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002284 / 1 | 0.002294 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.899956 / 1 | 18.994405 / 1 | 18.590992 / 1 |

### Layer 8

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011782 / 1 | 0.004799 / 1 | 0.003887 / 1 |
| pre-attention-aggregation | boundary | 0.014537 / 1 | 0.014417 / 1 | 0.014097 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000090 / 1 |
| pre-attention-normalization | boundary | 0.010690 / 1 | 0.010510 / 1 | 0.011021 / 1 |
| Q | nested | 2.659899 / 1 | 2.640252 / 1 | 2.648538 / 1 |
| K | nested | 2.278126 / 1 | 2.474182 / 1 | 2.290008 / 1 |
| V | nested | 2.117235 / 1 | 2.228323 / 1 | 2.113388 / 1 |
| B | nested | 0.069700 / 1 | 0.021640 / 1 | 0.020729 / 1 |
| FA | nested | 0.058248 / 1 | 0.035126 / 1 | 0.033242 / 1 |
| FB | nested | 0.073998 / 1 | 0.046336 / 1 | 0.046437 / 1 |
| G | nested | 2.201262 / 1 | 2.010696 / 1 | 2.017129 / 1 |
| O | nested | 2.287734 / 1 | 2.131933 / 1 | 2.137573 / 1 |
| attention | boundary | 12.853403 / 1 | 12.649893 / 1 | 12.057576 / 1 |
| attention-residual | boundary | 0.003145 / 1 | 0.003847 / 1 | 0.003226 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026419 / 1 | 0.025938 / 1 | 0.025407 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010409 / 1 | 0.002334 / 1 | 0.002204 / 1 |
| router-and-top16 | boundary | 0.597416 / 1 | 0.563183 / 1 | 0.563914 / 1 |
| EDOWN | nested | 0.655184 / 1 | 0.591846 / 1 | 0.594671 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.655725 / 1 | 0.592377 / 1 | 0.595222 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.082344 / 1 | 0.024216 / 1 | 0.006662 / 1 |
| SH1 | nested | 1.140741 / 1 | 1.034283 / 1 | 1.040904 / 1 |
| SH3 | nested | 1.119031 / 1 | 1.036747 / 1 | 1.018303 / 1 |
| SH2 | nested | 1.133788 / 1 | 0.991722 / 1 | 0.977286 / 1 |
| shared-expert-during-read | boundary | 3.404390 / 1 | 3.073512 / 1 | 3.046932 / 1 |
| detail:expert-gate | nested | 2.822994 / 16 | 2.719429 / 16 | 2.583888 / 16 |
| detail:expert-up | nested | 2.809768 / 16 | 2.677732 / 16 | 2.568347 / 16 |
| detail:expert-activation | nested | 0.093624 / 16 | 0.091232 / 16 | 0.092945 / 16 |
| detail:expert-down | nested | 3.076768 / 16 | 2.936586 / 16 | 2.764818 / 16 |
| EUP | nested | 0.645306 / 1 | 0.596454 / 1 | 0.601343 / 1 |
| experts-mix-normalize-up | boundary | 9.853409 / 1 | 9.355528 / 1 | 8.925915 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002705 / 1 | 0.003076 / 1 | 0.002405 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.399737 / 1 | 0.401069 / 1 | 0.395609 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.435117 / 1 | 114.422222 / 1 | 109.114583 / 1 |
| op:Q   int8 projection | nested | 16.438841 / 1 | 15.837809 / 1 | 15.537897 / 1 |
| op:X   mxfp4 expert proj | nested | 8.820027 / 1 | 8.442120 / 1 | 8.026736 / 1 |
| op:N   rmsnorm | nested | 0.031188 / 1 | 0.032411 / 1 | 0.023895 / 1 |
| op:L   l2 per-head | nested | 0.005079 / 1 | 0.005441 / 1 | 0.005630 / 1 |
| op:SiTU + sigma | nested | 0.102280 / 1 | 0.099367 / 1 | 0.100467 / 1 |
| op:C   shortconv | nested | 0.071344 / 1 | 0.065381 / 1 | 0.036429 / 1 |
| op:AR  snapshot aggregate | nested | 0.030428 / 1 | 0.029896 / 1 | 0.029094 / 1 |
| op:D   kda delta-rule | nested | 0.144951 / 1 | 0.139310 / 1 | 0.142867 / 1 |
| op:router dot product | nested | 0.594411 / 1 | 0.560046 / 1 | 0.561189 / 1 |
| op:top-k selection | nested | 0.002354 / 1 | 0.002315 / 1 | 0.002274 / 1 |
| op:alpha / beta / gate | nested | 0.071855 / 1 | 0.073437 / 1 | 0.065762 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.929999 / 1 | 26.728104 / 1 | 25.657684 / 1 |

### Layer 9

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012764 / 1 | 0.005250 / 1 | 0.005711 / 1 |
| pre-attention-aggregation | boundary | 0.015830 / 1 | 0.016902 / 1 | 0.015800 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011531 / 1 | 0.011160 / 1 | 0.011301 / 1 |
| Q | nested | 2.485704 / 1 | 2.477288 / 1 | 2.434418 / 1 |
| K | nested | 2.235136 / 1 | 2.294747 / 1 | 2.214457 / 1 |
| V | nested | 2.113999 / 1 | 2.186435 / 1 | 2.111114 / 1 |
| B | nested | 0.103613 / 1 | 0.026209 / 1 | 0.022983 / 1 |
| FA | nested | 0.065692 / 1 | 0.036077 / 1 | 0.027922 / 1 |
| FB | nested | 0.068658 / 1 | 0.059561 / 1 | 0.048390 / 1 |
| G | nested | 2.173761 / 1 | 1.992041 / 1 | 2.009194 / 1 |
| O | nested | 2.227201 / 1 | 2.092770 / 1 | 2.120521 / 1 |
| attention | boundary | 12.591263 / 1 | 12.260215 / 1 | 11.777062 / 1 |
| attention-residual | boundary | 0.004078 / 1 | 0.003316 / 1 | 0.004207 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024757 / 1 | 0.025979 / 1 | 0.024876 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010299 / 1 | 0.002354 / 1 | 0.002525 / 1 |
| router-and-top16 | boundary | 0.613295 / 1 | 0.567981 / 1 | 0.562521 / 1 |
| EDOWN | nested | 0.669360 / 1 | 0.581026 / 1 | 0.588609 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.669871 / 1 | 0.581497 / 1 | 0.589151 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.079548 / 1 | 0.023364 / 1 | 0.015699 / 1 |
| SH1 | nested | 1.104093 / 1 | 1.047928 / 1 | 1.035895 / 1 |
| SH3 | nested | 1.131353 / 1 | 1.075880 / 1 | 1.057976 / 1 |
| SH2 | nested | 1.108201 / 1 | 1.008424 / 1 | 1.018513 / 1 |
| shared-expert-during-read | boundary | 3.354557 / 1 | 3.142902 / 1 | 3.122684 / 1 |
| detail:expert-gate | nested | 2.784783 / 16 | 2.671681 / 16 | 2.472422 / 16 |
| detail:expert-up | nested | 2.740909 / 16 | 2.685086 / 16 | 2.464564 / 16 |
| detail:expert-activation | nested | 0.089157 / 16 | 0.092602 / 16 | 0.107532 / 16 |
| detail:expert-down | nested | 2.940968 / 16 | 2.855425 / 16 | 2.598585 / 16 |
| EUP | nested | 0.637110 / 1 | 0.600562 / 1 | 0.596404 / 1 |
| experts-mix-normalize-up | boundary | 9.594766 / 1 | 9.236797 / 1 | 8.570401 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003066 / 1 | 0.003347 / 1 | 0.002705 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000151 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.380671 / 1 | 0.385059 / 1 | 0.382074 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.539332 / 1 | 112.897966 / 1 | 103.529648 / 1 |
| op:Q   int8 projection | nested | 16.122349 / 1 | 15.473601 / 1 | 15.284485 / 1 |
| op:X   mxfp4 expert proj | nested | 8.573727 / 1 | 8.322728 / 1 | 7.661466 / 1 |
| op:N   rmsnorm | nested | 0.039703 / 1 | 0.034826 / 1 | 0.024405 / 1 |
| op:L   l2 per-head | nested | 0.006182 / 1 | 0.005510 / 1 | 0.005490 / 1 |
| op:SiTU + sigma | nested | 0.097563 / 1 | 0.100759 / 1 | 0.115396 / 1 |
| op:C   shortconv | nested | 0.056826 / 1 | 0.069409 / 1 | 0.036358 / 1 |
| op:AR  snapshot aggregate | nested | 0.030296 / 1 | 0.032431 / 1 | 0.030277 / 1 |
| op:D   kda delta-rule | nested | 0.147044 / 1 | 0.143959 / 1 | 0.139350 / 1 |
| op:router dot product | nested | 0.610190 / 1 | 0.564925 / 1 | 0.559906 / 1 |
| op:top-k selection | nested | 0.002645 / 1 | 0.002806 / 1 | 0.002184 / 1 |
| op:alpha / beta / gate | nested | 0.070983 / 1 | 0.074799 / 1 | 0.059060 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.369683 / 1 | 26.269658 / 1 | 25.090034 / 1 |

### Layer 10

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012764 / 1 | 0.005460 / 1 | 0.004608 / 1 |
| pre-attention-aggregation | boundary | 0.015529 / 1 | 0.021059 / 1 | 0.016050 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011431 / 1 | 0.012514 / 1 | 0.012063 / 1 |
| Q | nested | 2.478791 / 1 | 2.465777 / 1 | 2.453093 / 1 |
| K | nested | 2.218615 / 1 | 2.268939 / 1 | 2.199830 / 1 |
| V | nested | 2.109361 / 1 | 2.159694 / 1 | 2.111946 / 1 |
| B | nested | 0.091861 / 1 | 0.021130 / 1 | 0.021830 / 1 |
| FA | nested | 0.051766 / 1 | 0.031248 / 1 | 0.028253 / 1 |
| FB | nested | 0.075231 / 1 | 0.051506 / 1 | 0.049533 / 1 |
| G | nested | 2.202936 / 1 | 2.030683 / 1 | 2.041605 / 1 |
| O | nested | 2.236088 / 1 | 2.036846 / 1 | 2.032016 / 1 |
| attention | boundary | 12.551539 / 1 | 12.158595 / 1 | 11.691002 / 1 |
| attention-residual | boundary | 0.003347 / 1 | 0.003957 / 1 | 0.004058 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025498 / 1 | 0.025638 / 1 | 0.024346 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009838 / 1 | 0.002465 / 1 | 0.002485 / 1 |
| router-and-top16 | boundary | 0.608367 / 1 | 0.572059 / 1 | 0.573502 / 1 |
| EDOWN | nested | 0.661245 / 1 | 0.607896 / 1 | 0.600252 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.661757 / 1 | 0.608457 / 1 | 0.600813 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.076603 / 1 | 0.028503 / 1 | 0.007544 / 1 |
| SH1 | nested | 1.108641 / 1 | 1.003756 / 1 | 0.991192 / 1 |
| SH3 | nested | 1.087853 / 1 | 1.003635 / 1 | 0.995821 / 1 |
| SH2 | nested | 1.103883 / 1 | 1.033030 / 1 | 1.009245 / 1 |
| shared-expert-during-read | boundary | 3.311417 / 1 | 3.050840 / 1 | 3.007849 / 1 |
| detail:expert-gate | nested | 2.731310 / 16 | 2.705253 / 16 | 2.539123 / 16 |
| detail:expert-up | nested | 2.746329 / 16 | 2.708450 / 16 | 2.515680 / 16 |
| detail:expert-activation | nested | 0.091430 / 16 | 0.093659 / 16 | 0.091862 / 16 |
| detail:expert-down | nested | 2.944842 / 16 | 2.876584 / 16 | 2.661583 / 16 |
| EUP | nested | 0.634715 / 1 | 0.606213 / 1 | 0.602686 / 1 |
| experts-mix-normalize-up | boundary | 9.556193 / 1 | 9.329209 / 1 | 8.742122 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003166 / 1 | 0.003196 / 1 | 0.003236 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000250 / 1 | 0.000230 / 1 | 0.000231 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.396441 / 1 | 0.403625 / 1 | 0.400498 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.112336 / 1 | 113.047488 / 1 | 105.734485 / 1 |
| op:Q   int8 projection | nested | 16.059111 / 1 | 15.318339 / 1 | 15.135409 / 1 |
| op:X   mxfp4 expert proj | nested | 8.532419 / 1 | 8.402768 / 1 | 7.827552 / 1 |
| op:N   rmsnorm | nested | 0.033803 / 1 | 0.032659 / 1 | 0.024678 / 1 |
| op:L   l2 per-head | nested | 0.005110 / 1 | 0.006081 / 1 | 0.005922 / 1 |
| op:SiTU + sigma | nested | 0.099955 / 1 | 0.101313 / 1 | 0.100327 / 1 |
| op:C   shortconv | nested | 0.055463 / 1 | 0.060073 / 1 | 0.033543 / 1 |
| op:AR  snapshot aggregate | nested | 0.030617 / 1 | 0.036088 / 1 | 0.030437 / 1 |
| op:D   kda delta-rule | nested | 0.125083 / 1 | 0.127818 / 1 | 0.120104 / 1 |
| op:router dot product | nested | 0.605541 / 1 | 0.568903 / 1 | 0.570636 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002565 / 1 | 0.002224 / 1 |
| op:alpha / beta / gate | nested | 0.071073 / 1 | 0.074159 / 1 | 0.059321 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.247715 / 1 | 26.229563 / 1 | 25.093981 / 1 |

### Layer 11

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013014 / 1 | 0.005951 / 1 | 0.005209 / 1 |
| pre-attention-aggregation | boundary | 0.015930 / 1 | 0.016230 / 1 | 0.015749 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012393 / 1 | 0.011351 / 1 | 0.013495 / 1 |
| QA | nested | 0.325238 / 1 | 0.312524 / 1 | 0.310870 / 1 |
| QB | nested | 0.715216 / 1 | 0.671304 / 1 | 0.680922 / 1 |
| KA | nested | 0.123802 / 1 | 0.099916 / 1 | 0.104285 / 1 |
| KB | nested | 0.331890 / 1 | 0.314157 / 1 | 0.324406 / 1 |
| G | nested | 2.191434 / 1 | 1.975851 / 1 | 1.954001 / 1 |
| O | nested | 2.201102 / 1 | 1.969089 / 1 | 1.954001 / 1 |
| attention | boundary | 5.989370 / 1 | 5.439893 / 1 | 5.426599 / 1 |
| attention-residual | boundary | 0.002785 / 1 | 0.002685 / 1 | 0.002815 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023875 / 1 | 0.024686 / 1 | 0.025537 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.014988 / 1 | 0.002134 / 1 | 0.002224 / 1 |
| router-and-top16 | boundary | 0.604860 / 1 | 0.571368 / 1 | 0.576818 / 1 |
| EDOWN | nested | 0.640287 / 1 | 0.593259 / 1 | 0.587397 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.640737 / 1 | 0.593790 / 1 | 0.587958 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.064520 / 1 | 0.017012 / 1 | 0.006512 / 1 |
| SH1 | nested | 1.090297 / 1 | 1.001682 / 1 | 1.012352 / 1 |
| SH3 | nested | 1.083925 / 1 | 0.984991 / 1 | 0.976123 / 1 |
| SH2 | nested | 1.081050 / 1 | 0.981003 / 1 | 0.973799 / 1 |
| shared-expert-during-read | boundary | 3.265551 / 1 | 2.978074 / 1 | 2.973255 / 1 |
| detail:expert-gate | nested | 2.771619 / 16 | 2.678362 / 16 | 2.417368 / 16 |
| detail:expert-up | nested | 2.716786 / 16 | 2.665340 / 16 | 2.426054 / 16 |
| detail:expert-activation | nested | 0.091555 / 16 | 0.093485 / 16 | 0.094656 / 16 |
| detail:expert-down | nested | 2.900951 / 16 | 2.798490 / 16 | 2.505521 / 16 |
| EUP | nested | 0.629436 / 1 | 0.595733 / 1 | 0.590643 / 1 |
| experts-mix-normalize-up | boundary | 9.482786 / 1 | 9.180492 / 1 | 8.367272 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003366 / 1 | 0.003647 / 1 | 0.003246 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000150 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005481 / 1 | 0.005811 / 1 | 0.005411 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.175366 / 1 | 112.168837 / 1 | 100.550820 / 1 |
| op:Q   int8 projection | nested | 10.412404 / 1 | 9.498255 / 1 | 9.467447 / 1 |
| op:X   mxfp4 expert proj | nested | 8.499919 / 1 | 8.255113 / 1 | 7.463454 / 1 |
| op:N   rmsnorm | nested | 0.024186 / 1 | 0.024605 / 1 | 0.026230 / 1 |
| op:SiTU + sigma | nested | 0.099597 / 1 | 0.101670 / 1 | 0.103363 / 1 |
| op:AR  snapshot aggregate | nested | 0.029825 / 1 | 0.029776 / 1 | 0.030337 / 1 |
| op:SA  softmax attention | nested | 0.006672 / 1 | 0.009187 / 1 | 0.011782 / 1 |
| op:router dot product | nested | 0.601784 / 1 | 0.569033 / 1 | 0.573591 / 1 |
| op:top-k selection | nested | 0.002434 / 1 | 0.002104 / 1 | 0.002475 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.143232 / 1 | 18.857160 / 1 | 18.015767 / 1 |

### Layer 12

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011131 / 1 | 0.004488 / 1 | 0.003607 / 1 |
| pre-attention-aggregation | boundary | 0.014266 / 1 | 0.014487 / 1 | 0.014537 / 1 |
| snapshot-push | boundary | 0.001102 / 1 | 0.001253 / 1 | 0.001192 / 1 |
| pre-attention-normalization | boundary | 0.011432 / 1 | 0.010660 / 1 | 0.011100 / 1 |
| Q | nested | 2.473601 / 1 | 2.462620 / 1 | 2.436862 / 1 |
| K | nested | 2.238843 / 1 | 2.269239 / 1 | 2.182207 / 1 |
| V | nested | 2.119399 / 1 | 2.152491 / 1 | 2.096427 / 1 |
| B | nested | 0.092573 / 1 | 0.026389 / 1 | 0.026880 / 1 |
| FA | nested | 0.058299 / 1 | 0.034534 / 1 | 0.034415 / 1 |
| FB | nested | 0.070973 / 1 | 0.047709 / 1 | 0.050434 / 1 |
| G | nested | 2.206382 / 1 | 2.029541 / 1 | 2.021286 / 1 |
| O | nested | 2.765387 / 1 | 2.056462 / 1 | 2.059698 / 1 |
| attention | boundary | 13.091207 / 1 | 12.177691 / 1 | 11.655264 / 1 |
| attention-residual | boundary | 0.002194 / 1 | 0.002254 / 1 | 0.002174 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026129 / 1 | 0.026991 / 1 | 0.024907 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009909 / 1 | 0.001914 / 1 | 0.002355 / 1 |
| router-and-top16 | boundary | 0.621150 / 1 | 0.573452 / 1 | 0.572129 / 1 |
| EDOWN | nested | 0.772373 / 1 | 0.598528 / 1 | 0.603427 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.772884 / 1 | 0.599180 / 1 | 0.603958 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.077305 / 1 | 0.036418 / 1 | 0.019116 / 1 |
| SH1 | nested | 1.157853 / 1 | 1.053428 / 1 | 1.020656 / 1 |
| SH3 | nested | 1.114311 / 1 | 1.007262 / 1 | 1.016209 / 1 |
| SH2 | nested | 1.097069 / 1 | 1.010568 / 1 | 1.004136 / 1 |
| shared-expert-during-read | boundary | 3.383241 / 1 | 3.081747 / 1 | 3.051741 / 1 |
| detail:expert-gate | nested | 2.741091 / 16 | 2.663885 / 16 | 2.514205 / 16 |
| detail:expert-up | nested | 2.715255 / 16 | 2.692520 / 16 | 2.523842 / 16 |
| detail:expert-activation | nested | 0.094785 / 16 | 0.092903 / 16 | 0.095167 / 16 |
| detail:expert-down | nested | 2.895748 / 16 | 2.884598 / 16 | 2.668384 / 16 |
| EUP | nested | 0.634004 / 1 | 0.610902 / 1 | 0.589742 / 1 |
| experts-mix-normalize-up | boundary | 9.469321 / 1 | 9.296027 / 1 | 8.730371 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003516 / 1 | 0.003296 / 1 | 0.003296 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000310 / 1 | 0.000160 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.394076 / 1 | 0.399336 / 1 | 0.391451 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.077921 / 1 | 113.668132 / 1 | 105.801139 / 1 |
| op:Q   int8 projection | nested | 16.799464 / 1 | 15.357800 / 1 | 15.140416 / 1 |
| op:X   mxfp4 expert proj | nested | 8.467308 / 1 | 8.354166 / 1 | 7.822264 / 1 |
| op:N   rmsnorm | nested | 0.034203 / 1 | 0.038713 / 1 | 0.023715 / 1 |
| op:L   l2 per-head | nested | 0.005941 / 1 | 0.005891 / 1 | 0.005920 / 1 |
| op:SiTU + sigma | nested | 0.106296 / 1 | 0.100736 / 1 | 0.103092 / 1 |
| op:C   shortconv | nested | 0.079989 / 1 | 0.063319 / 1 | 0.036247 / 1 |
| op:AR  snapshot aggregate | nested | 0.030216 / 1 | 0.031048 / 1 | 0.029324 / 1 |
| op:D   kda delta-rule | nested | 0.142005 / 1 | 0.141314 / 1 | 0.134641 / 1 |
| op:router dot product | nested | 0.618015 / 1 | 0.570475 / 1 | 0.569413 / 1 |
| op:top-k selection | nested | 0.002775 / 1 | 0.002284 / 1 | 0.002234 / 1 |
| op:alpha / beta / gate | nested | 0.072936 / 1 | 0.073968 / 1 | 0.058359 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.895095 / 1 | 26.233160 / 1 | 25.091056 / 1 |

### Layer 13

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.018204 / 1 | 0.005540 / 1 | 0.004709 / 1 |
| pre-attention-aggregation | boundary | 0.018855 / 1 | 0.017021 / 1 | 0.016040 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011903 / 1 | 0.011742 / 1 | 0.012543 / 1 |
| Q | nested | 2.470686 / 1 | 2.464565 / 1 | 2.440209 / 1 |
| K | nested | 2.248951 / 1 | 2.257788 / 1 | 2.213576 / 1 |
| V | nested | 2.137293 / 1 | 2.157220 / 1 | 2.129809 / 1 |
| B | nested | 0.094687 / 1 | 0.020819 / 1 | 0.021290 / 1 |
| FA | nested | 0.059411 / 1 | 0.034474 / 1 | 0.027912 / 1 |
| FB | nested | 0.076643 / 1 | 0.047629 / 1 | 0.048600 / 1 |
| G | nested | 2.188648 / 1 | 1.976202 / 1 | 1.977745 / 1 |
| O | nested | 2.209939 / 1 | 2.078954 / 1 | 2.076690 / 1 |
| attention | boundary | 12.541761 / 1 | 12.122678 / 1 | 11.648662 / 1 |
| attention-residual | boundary | 0.003717 / 1 | 0.003337 / 1 | 0.003657 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025688 / 1 | 0.030267 / 1 | 0.024746 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009939 / 1 | 0.002304 / 1 | 0.002315 / 1 |
| router-and-top16 | boundary | 0.596254 / 1 | 0.560156 / 1 | 0.560978 / 1 |
| EDOWN | nested | 0.654262 / 1 | 0.586165 / 1 | 0.581226 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.654763 / 1 | 0.586786 / 1 | 0.581827 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085640 / 1 | 0.022632 / 1 | 0.006673 / 1 |
| SH1 | nested | 1.121295 / 1 | 1.008163 / 1 | 0.999417 / 1 |
| SH3 | nested | 1.094585 / 1 | 0.990250 / 1 | 0.989228 / 1 |
| SH2 | nested | 1.078415 / 1 | 1.007432 / 1 | 0.999377 / 1 |
| shared-expert-during-read | boundary | 3.304454 / 1 | 3.016455 / 1 | 2.998572 / 1 |
| detail:expert-gate | nested | 2.746463 / 16 | 2.727315 / 16 | 2.531598 / 16 |
| detail:expert-up | nested | 2.727856 / 16 | 2.737294 / 16 | 2.537938 / 16 |
| detail:expert-activation | nested | 0.091040 / 16 | 0.090360 / 16 | 0.094387 / 16 |
| detail:expert-down | nested | 2.897444 / 16 | 2.898936 / 16 | 2.681210 / 16 |
| EUP | nested | 0.632712 / 1 | 0.613606 / 1 | 0.590002 / 1 |
| experts-mix-normalize-up | boundary | 9.499357 / 1 | 9.413276 / 1 | 8.756930 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003496 / 1 | 0.002845 / 1 | 0.002795 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000250 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.410757 / 1 | 0.403183 / 1 | 0.389328 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.441234 / 1 | 115.385468 / 1 | 107.126723 / 1 |
| op:Q   int8 projection | nested | 16.066136 / 1 | 15.241274 / 1 | 15.093588 / 1 |
| op:X   mxfp4 expert proj | nested | 8.483690 / 1 | 8.474110 / 1 | 7.866766 / 1 |
| op:N   rmsnorm | nested | 0.033061 / 1 | 0.033162 / 1 | 0.024698 / 1 |
| op:L   l2 per-head | nested | 0.005751 / 1 | 0.005320 / 1 | 0.005941 / 1 |
| op:SiTU + sigma | nested | 0.098944 / 1 | 0.098374 / 1 | 0.102472 / 1 |
| op:C   shortconv | nested | 0.065813 / 1 | 0.056966 / 1 | 0.033361 / 1 |
| op:AR  snapshot aggregate | nested | 0.034194 / 1 | 0.037210 / 1 | 0.030608 / 1 |
| op:D   kda delta-rule | nested | 0.116637 / 1 | 0.118822 / 1 | 0.113312 / 1 |
| op:router dot product | nested | 0.592887 / 1 | 0.557582 / 1 | 0.558243 / 1 |
| op:top-k selection | nested | 0.002886 / 1 | 0.002164 / 1 | 0.002104 / 1 |
| op:alpha / beta / gate | nested | 0.072796 / 1 | 0.074429 / 1 | 0.059451 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.189036 / 1 | 26.202452 / 1 | 25.013801 / 1 |

### Layer 14

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012945 / 1 | 0.004879 / 1 | 0.004409 / 1 |
| pre-attention-aggregation | boundary | 0.016150 / 1 | 0.016601 / 1 | 0.016731 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011952 / 1 | 0.011332 / 1 | 0.011221 / 1 |
| Q | nested | 2.526000 / 1 | 2.528534 / 1 | 2.488879 / 1 |
| K | nested | 2.238532 / 1 | 2.264741 / 1 | 2.180885 / 1 |
| V | nested | 2.132895 / 1 | 2.144867 / 1 | 2.105153 / 1 |
| B | nested | 0.087734 / 1 | 0.025417 / 1 | 0.026409 / 1 |
| FA | nested | 0.061205 / 1 | 0.035857 / 1 | 0.037800 / 1 |
| FB | nested | 0.077555 / 1 | 0.051005 / 1 | 0.053019 / 1 |
| G | nested | 2.177017 / 1 | 2.012119 / 1 | 2.019903 / 1 |
| O | nested | 2.262006 / 1 | 2.182117 / 1 | 2.184120 / 1 |
| attention | boundary | 12.653790 / 1 | 12.306000 / 1 | 11.773225 / 1 |
| attention-residual | boundary | 0.003606 / 1 | 0.003737 / 1 | 0.003847 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025808 / 1 | 0.026048 / 1 | 0.026089 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010039 / 1 | 0.003206 / 1 | 0.002164 / 1 |
| router-and-top16 | boundary | 0.603688 / 1 | 0.561349 / 1 | 0.564174 / 1 |
| EDOWN | nested | 0.655274 / 1 | 0.608036 / 1 | 0.595021 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.655856 / 1 | 0.608587 / 1 | 0.595633 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.081242 / 1 | 0.029736 / 1 | 0.006853 / 1 |
| SH1 | nested | 1.149648 / 1 | 1.057115 / 1 | 1.036647 / 1 |
| SH3 | nested | 1.120684 / 1 | 1.029774 / 1 | 1.000559 / 1 |
| SH2 | nested | 1.081851 / 1 | 1.005558 / 1 | 0.999337 / 1 |
| shared-expert-during-read | boundary | 3.363163 / 1 | 3.103579 / 1 | 3.047203 / 1 |
| detail:expert-gate | nested | 3.118933 / 16 | 2.680305 / 16 | 2.641773 / 16 |
| detail:expert-up | nested | 3.062973 / 16 | 2.664259 / 16 | 2.674035 / 16 |
| detail:expert-activation | nested | 0.105960 / 16 | 0.090387 / 16 | 0.091240 / 16 |
| detail:expert-down | nested | 2.982683 / 16 | 2.895627 / 16 | 2.816621 / 16 |
| EUP | nested | 0.801557 / 1 | 0.588560 / 1 | 0.690240 / 1 |
| experts-mix-normalize-up | boundary | 10.490639 / 1 | 9.280959 / 1 | 9.244090 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003176 / 1 | 0.009056 / 1 | 0.003086 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000251 / 1 | 0.000260 / 1 | 0.000290 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.420917 / 1 | 0.416759 / 1 | 0.394377 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 117.981871 / 1 | 113.709010 / 1 | 111.614565 / 1 |
| op:Q   int8 projection | nested | 16.370244 / 1 | 15.531917 / 1 | 15.416010 / 1 |
| op:X   mxfp4 expert proj | nested | 9.292977 / 1 | 8.352262 / 1 | 8.246063 / 1 |
| op:N   rmsnorm | nested | 0.034465 / 1 | 0.034243 / 1 | 0.023645 / 1 |
| op:L   l2 per-head | nested | 0.005800 / 1 | 0.005210 / 1 | 0.006061 / 1 |
| op:SiTU + sigma | nested | 0.114617 / 1 | 0.098463 / 1 | 0.099307 / 1 |
| op:C   shortconv | nested | 0.052869 / 1 | 0.055814 / 1 | 0.032802 / 1 |
| op:AR  snapshot aggregate | nested | 0.031539 / 1 | 0.031879 / 1 | 0.033092 / 1 |
| op:D   kda delta-rule | nested | 0.133208 / 1 | 0.130153 / 1 | 0.108052 / 1 |
| op:router dot product | nested | 0.600592 / 1 | 0.558373 / 1 | 0.561329 / 1 |
| op:top-k selection | nested | 0.002384 / 1 | 0.002304 / 1 | 0.002094 / 1 |
| op:alpha / beta / gate | nested | 0.073216 / 1 | 0.078116 / 1 | 0.059081 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.357238 / 1 | 26.386256 / 1 | 25.697499 / 1 |

### Layer 15

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013114 / 1 | 0.005300 / 1 | 0.004398 / 1 |
| pre-attention-aggregation | boundary | 0.016181 / 1 | 0.017783 / 1 | 0.016411 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010870 / 1 | 0.010940 / 1 | 0.016371 / 1 |
| QA | nested | 0.355273 / 1 | 0.296224 / 1 | 0.297165 / 1 |
| QB | nested | 0.954553 / 1 | 0.686693 / 1 | 0.687123 / 1 |
| KA | nested | 0.142466 / 1 | 0.100949 / 1 | 0.100758 / 1 |
| KB | nested | 0.432698 / 1 | 0.301644 / 1 | 0.298317 / 1 |
| G | nested | 2.725482 / 1 | 1.967586 / 1 | 1.939153 / 1 |
| O | nested | 2.697700 / 1 | 1.970762 / 1 | 1.940916 / 1 |
| attention | boundary | 7.415244 / 1 | 5.421548 / 1 | 5.363360 / 1 |
| attention-residual | boundary | 0.003035 / 1 | 0.003056 / 1 | 0.003186 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025818 / 1 | 0.027141 / 1 | 0.025177 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.011051 / 1 | 0.002345 / 1 | 0.002144 / 1 |
| router-and-top16 | boundary | 0.609379 / 1 | 0.575125 / 1 | 0.568682 / 1 |
| EDOWN | nested | 0.786590 / 1 | 0.583540 / 1 | 0.583290 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.787171 / 1 | 0.584191 / 1 | 0.583841 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.084527 / 1 | 0.016781 / 1 | 0.017763 / 1 |
| SH1 | nested | 1.386070 / 1 | 1.008444 / 1 | 1.011961 / 1 |
| SH3 | nested | 1.354220 / 1 | 0.973068 / 1 | 0.983297 / 1 |
| SH2 | nested | 1.318293 / 1 | 1.019054 / 1 | 1.015067 / 1 |
| shared-expert-during-read | boundary | 4.071737 / 1 | 3.010945 / 1 | 3.020854 / 1 |
| detail:expert-gate | nested | 2.809909 / 16 | 2.707198 / 16 | 2.516782 / 16 |
| detail:expert-up | nested | 2.772030 / 16 | 2.696659 / 16 | 2.547830 / 16 |
| detail:expert-activation | nested | 0.092684 / 16 | 0.092982 / 16 | 0.090150 / 16 |
| detail:expert-down | nested | 2.971842 / 16 | 2.916400 / 16 | 2.729871 / 16 |
| EUP | nested | 0.634095 / 1 | 0.599299 / 1 | 0.594080 / 1 |
| experts-mix-normalize-up | boundary | 9.693430 / 1 | 9.340090 / 1 | 8.804279 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003436 / 1 | 0.003376 / 1 | 0.003266 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005129 / 1 | 0.005120 / 1 | 0.004999 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.708945 / 1 | 114.716702 / 1 | 107.010161 / 1 |
| op:Q   int8 projection | nested | 12.786058 / 1 | 9.506058 / 1 | 9.449512 / 1 |
| op:X   mxfp4 expert proj | nested | 8.668905 / 1 | 8.435852 / 1 | 7.907114 / 1 |
| op:N   rmsnorm | nested | 0.023525 / 1 | 0.024315 / 1 | 0.024625 / 1 |
| op:SiTU + sigma | nested | 0.103504 / 1 | 0.100617 / 1 | 0.098135 / 1 |
| op:AR  snapshot aggregate | nested | 0.031038 / 1 | 0.033523 / 1 | 0.030667 / 1 |
| op:SA  softmax attention | nested | 0.007114 / 1 | 0.009298 / 1 | 0.011853 / 1 |
| op:router dot product | nested | 0.606614 / 1 | 0.572359 / 1 | 0.566057 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002275 / 1 | 0.002345 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 22.754771 / 1 | 19.028069 / 1 | 18.439429 / 1 |

### Layer 16

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011481 / 1 | 0.004118 / 1 | 0.003606 / 1 |
| pre-attention-aggregation | boundary | 0.014106 / 1 | 0.015108 / 1 | 0.014347 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010920 / 1 | 0.010910 / 1 | 0.011050 / 1 |
| Q | nested | 2.655972 / 1 | 2.645051 / 1 | 2.622810 / 1 |
| K | nested | 2.353106 / 1 | 2.322268 / 1 | 2.227812 / 1 |
| V | nested | 2.186385 / 1 | 2.163352 / 1 | 2.126753 / 1 |
| B | nested | 0.094907 / 1 | 0.020428 / 1 | 0.020879 / 1 |
| FA | nested | 0.058429 / 1 | 0.033993 / 1 | 0.033312 / 1 |
| FB | nested | 0.070041 / 1 | 0.048681 / 1 | 0.049482 / 1 |
| G | nested | 2.153243 / 1 | 2.013361 / 1 | 2.018581 / 1 |
| O | nested | 2.224025 / 1 | 2.071430 / 1 | 2.087720 / 1 |
| attention | boundary | 12.874923 / 1 | 12.439449 / 1 | 11.889101 / 1 |
| attention-residual | boundary | 0.003987 / 1 | 0.003196 / 1 | 0.003707 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026049 / 1 | 0.026289 / 1 | 0.024657 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010340 / 1 | 0.002325 / 1 | 0.002244 / 1 |
| router-and-top16 | boundary | 0.610651 / 1 | 0.578852 / 1 | 0.574082 / 1 |
| EDOWN | nested | 0.669050 / 1 | 0.618916 / 1 | 0.586596 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.669641 / 1 | 0.619737 / 1 | 0.587197 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059411 / 1 | 0.018645 / 1 | 0.013645 / 1 |
| SH1 | nested | 1.135071 / 1 | 1.050733 / 1 | 1.009115 / 1 |
| SH3 | nested | 1.088984 / 1 | 0.978027 / 1 | 0.975493 / 1 |
| SH2 | nested | 1.090377 / 1 | 1.008244 / 1 | 1.020757 / 1 |
| shared-expert-during-read | boundary | 3.325453 / 1 | 3.048215 / 1 | 3.016075 / 1 |
| detail:expert-gate | nested | 2.732418 / 16 | 2.641613 / 16 | 2.540720 / 16 |
| detail:expert-up | nested | 2.733264 / 16 | 2.621577 / 16 | 2.497205 / 16 |
| detail:expert-activation | nested | 0.093984 / 16 | 0.092443 / 16 | 0.090550 / 16 |
| detail:expert-down | nested | 2.889548 / 16 | 2.841249 / 16 | 2.692761 / 16 |
| EUP | nested | 0.631650 / 1 | 0.596524 / 1 | 0.592327 / 1 |
| experts-mix-normalize-up | boundary | 9.462488 / 1 | 9.132050 / 1 | 8.751500 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003196 / 1 | 0.003366 / 1 | 0.002594 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000160 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.403624 / 1 | 0.403263 / 1 | 0.381503 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.158262 / 1 | 111.511792 / 1 | 106.756020 / 1 |
| op:Q   int8 projection | nested | 16.409588 / 1 | 15.569057 / 1 | 15.370056 / 1 |
| op:X   mxfp4 expert proj | nested | 8.472962 / 1 | 8.220828 / 1 | 7.845980 / 1 |
| op:N   rmsnorm | nested | 0.037240 / 1 | 0.034606 / 1 | 0.023794 / 1 |
| op:L   l2 per-head | nested | 0.005711 / 1 | 0.005791 / 1 | 0.005750 / 1 |
| op:SiTU + sigma | nested | 0.102551 / 1 | 0.100949 / 1 | 0.098693 / 1 |
| op:C   shortconv | nested | 0.063358 / 1 | 0.077546 / 1 | 0.039083 / 1 |
| op:AR  snapshot aggregate | nested | 0.029785 / 1 | 0.030417 / 1 | 0.028984 / 1 |
| op:D   kda delta-rule | nested | 0.146163 / 1 | 0.144841 / 1 | 0.118412 / 1 |
| op:router dot product | nested | 0.607545 / 1 | 0.576276 / 1 | 0.571007 / 1 |
| op:top-k selection | nested | 0.002635 / 1 | 0.002054 / 1 | 0.002705 / 1 |
| op:alpha / beta / gate | nested | 0.071914 / 1 | 0.073768 / 1 | 0.057968 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.491009 / 1 | 26.310274 / 1 | 25.280109 / 1 |

### Layer 17

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012914 / 1 | 0.005159 / 1 | 0.004859 / 1 |
| pre-attention-aggregation | boundary | 0.015849 / 1 | 0.015909 / 1 | 0.015108 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011621 / 1 | 0.011722 / 1 | 0.012023 / 1 |
| Q | nested | 2.448915 / 1 | 2.438035 / 1 | 2.422927 / 1 |
| K | nested | 2.275200 / 1 | 2.237350 / 1 | 2.194981 / 1 |
| V | nested | 2.160376 / 1 | 2.143615 / 1 | 2.126933 / 1 |
| B | nested | 0.067206 / 1 | 0.028383 / 1 | 0.027431 / 1 |
| FA | nested | 0.059722 / 1 | 0.035747 / 1 | 0.038522 / 1 |
| FB | nested | 0.066805 / 1 | 0.049543 / 1 | 0.049663 / 1 |
| G | nested | 2.178360 / 1 | 1.986682 / 1 | 1.992663 / 1 |
| O | nested | 2.236879 / 1 | 2.076339 / 1 | 2.094022 / 1 |
| attention | boundary | 12.615969 / 1 | 12.133418 / 1 | 11.710839 / 1 |
| attention-residual | boundary | 0.003426 / 1 | 0.003887 / 1 | 0.003717 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026560 / 1 | 0.026159 / 1 | 0.024857 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009678 / 1 | 0.002375 / 1 | 0.002375 / 1 |
| router-and-top16 | boundary | 0.616131 / 1 | 0.583130 / 1 | 0.575495 / 1 |
| EDOWN | nested | 0.660494 / 1 | 0.611953 / 1 | 0.591335 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.661145 / 1 | 0.612675 / 1 | 0.591916 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.070702 / 1 | 0.048480 / 1 | 0.006632 / 1 |
| SH1 | nested | 1.118480 / 1 | 1.021007 / 1 | 0.992374 / 1 |
| SH3 | nested | 1.094224 / 1 | 0.985591 / 1 | 0.984849 / 1 |
| SH2 | nested | 1.094885 / 1 | 1.017401 / 1 | 1.022881 / 1 |
| shared-expert-during-read | boundary | 3.318179 / 1 | 3.035501 / 1 | 3.011326 / 1 |
| detail:expert-gate | nested | 2.767088 / 16 | 2.690367 / 16 | 2.462762 / 16 |
| detail:expert-up | nested | 2.712548 / 16 | 2.692601 / 16 | 2.440178 / 16 |
| detail:expert-activation | nested | 0.088976 / 16 | 0.090260 / 16 | 0.093868 / 16 |
| detail:expert-down | nested | 2.915355 / 16 | 2.871265 / 16 | 2.620704 / 16 |
| EUP | nested | 0.632572 / 1 | 0.614918 / 1 | 0.613947 / 1 |
| experts-mix-normalize-up | boundary | 9.514615 / 1 | 9.346412 / 1 | 8.557177 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003496 / 1 | 0.003377 / 1 | 0.002976 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000170 / 1 | 0.000161 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.400117 / 1 | 0.394688 / 1 | 0.367085 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.311816 / 1 | 114.431437 / 1 | 103.015572 / 1 |
| op:Q   int8 projection | nested | 16.092194 / 1 | 15.244972 / 1 | 15.150825 / 1 |
| op:X   mxfp4 expert proj | nested | 8.511129 / 1 | 8.368684 / 1 | 7.642831 / 1 |
| op:N   rmsnorm | nested | 0.035565 / 1 | 0.042277 / 1 | 0.024987 / 1 |
| op:L   l2 per-head | nested | 0.005921 / 1 | 0.005270 / 1 | 0.006121 / 1 |
| op:SiTU + sigma | nested | 0.097273 / 1 | 0.098674 / 1 | 0.102382 / 1 |
| op:C   shortconv | nested | 0.062636 / 1 | 0.055444 / 1 | 0.032701 / 1 |
| op:AR  snapshot aggregate | nested | 0.031629 / 1 | 0.031349 / 1 | 0.029436 / 1 |
| op:D   kda delta-rule | nested | 0.164558 / 1 | 0.155400 / 1 | 0.136395 / 1 |
| op:router dot product | nested | 0.613526 / 1 | 0.580244 / 1 | 0.572750 / 1 |
| op:top-k selection | nested | 0.002013 / 1 | 0.002424 / 1 | 0.002264 / 1 |
| op:alpha / beta / gate | nested | 0.072826 / 1 | 0.075090 / 1 | 0.059321 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.285205 / 1 | 26.228020 / 1 | 24.891483 / 1 |

### Layer 18

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013385 / 1 | 0.005781 / 1 | 0.006092 / 1 |
| pre-attention-aggregation | boundary | 0.016200 / 1 | 0.015579 / 1 | 0.016942 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012663 / 1 | 0.012493 / 1 | 0.011191 / 1 |
| Q | nested | 2.486315 / 1 | 2.487046 / 1 | 2.457280 / 1 |
| K | nested | 2.269310 / 1 | 2.251155 / 1 | 2.172228 / 1 |
| V | nested | 2.158573 / 1 | 2.147182 / 1 | 2.093521 / 1 |
| B | nested | 0.080660 / 1 | 0.021530 / 1 | 0.020348 / 1 |
| FA | nested | 0.058249 / 1 | 0.032030 / 1 | 0.033823 / 1 |
| FB | nested | 0.072475 / 1 | 0.047769 / 1 | 0.049452 / 1 |
| G | nested | 2.193017 / 1 | 1.997000 / 1 | 2.007901 / 1 |
| O | nested | 2.241968 / 1 | 2.107868 / 1 | 2.103961 / 1 |
| attention | boundary | 12.621259 / 1 | 12.177310 / 1 | 11.647680 / 1 |
| attention-residual | boundary | 0.003857 / 1 | 0.003837 / 1 | 0.004087 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025207 / 1 | 0.025117 / 1 | 0.024436 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010209 / 1 | 0.002524 / 1 | 0.002765 / 1 |
| router-and-top16 | boundary | 0.598368 / 1 | 0.563623 / 1 | 0.561118 / 1 |
| EDOWN | nested | 0.648933 / 1 | 0.598668 / 1 | 0.591906 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.649504 / 1 | 0.599410 / 1 | 0.592547 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059311 / 1 | 0.027983 / 1 | 0.014477 / 1 |
| SH1 | nested | 1.120473 / 1 | 1.019345 / 1 | 0.998305 / 1 |
| SH3 | nested | 1.146242 / 1 | 1.089525 / 1 | 1.065140 / 1 |
| SH2 | nested | 1.109122 / 1 | 1.045764 / 1 | 1.041386 / 1 |
| shared-expert-during-read | boundary | 3.386366 / 1 | 3.165845 / 1 | 3.116172 / 1 |
| detail:expert-gate | nested | 2.893546 / 16 | 2.721845 / 16 | 2.495230 / 16 |
| detail:expert-up | nested | 2.880449 / 16 | 2.672633 / 16 | 2.443025 / 16 |
| detail:expert-activation | nested | 0.095326 / 16 | 0.090978 / 16 | 0.090449 / 16 |
| detail:expert-down | nested | 2.949408 / 16 | 2.881342 / 16 | 2.646923 / 16 |
| EUP | nested | 0.772844 / 1 | 0.592687 / 1 | 0.605310 / 1 |
| experts-mix-normalize-up | boundary | 9.968774 / 1 | 9.310354 / 1 | 8.615585 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003567 / 1 | 0.003406 / 1 | 0.003076 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000231 / 1 | 0.000361 / 1 | 0.000260 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.419784 / 1 | 0.416478 / 1 | 0.396631 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 117.424025 / 1 | 114.211594 / 1 | 104.273638 / 1 |
| op:Q   int8 projection | nested | 16.356640 / 1 | 15.435806 / 1 | 15.238871 / 1 |
| op:X   mxfp4 expert proj | nested | 8.844645 / 1 | 8.391898 / 1 | 7.701877 / 1 |
| op:N   rmsnorm | nested | 0.037621 / 1 | 0.048430 / 1 | 0.023583 / 1 |
| op:L   l2 per-head | nested | 0.005400 / 1 | 0.005401 / 1 | 0.004989 / 1 |
| op:SiTU + sigma | nested | 0.103322 / 1 | 0.099374 / 1 | 0.099002 / 1 |
| op:C   shortconv | nested | 0.060583 / 1 | 0.055575 / 1 | 0.032751 / 1 |
| op:AR  snapshot aggregate | nested | 0.031017 / 1 | 0.030566 / 1 | 0.031239 / 1 |
| op:D   kda delta-rule | nested | 0.134692 / 1 | 0.133399 / 1 | 0.113642 / 1 |
| op:router dot product | nested | 0.595843 / 1 | 0.560618 / 1 | 0.558443 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002314 / 1 | 0.002314 / 1 |
| op:alpha / beta / gate | nested | 0.070682 / 1 | 0.073888 / 1 | 0.065973 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.794226 / 1 | 26.335080 / 1 | 25.017870 / 1 |

### Layer 19

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013015 / 1 | 0.005480 / 1 | 0.005350 / 1 |
| pre-attention-aggregation | boundary | 0.015760 / 1 | 0.016090 / 1 | 0.016941 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010309 / 1 | 0.010600 / 1 | 0.010109 / 1 |
| QA | nested | 0.308817 / 1 | 0.283800 / 1 | 0.282167 / 1 |
| QB | nested | 0.719905 / 1 | 0.701210 / 1 | 0.691242 / 1 |
| KA | nested | 0.129191 / 1 | 0.102321 / 1 | 0.101440 / 1 |
| KB | nested | 0.344404 / 1 | 0.312053 / 1 | 0.301353 / 1 |
| G | nested | 2.158483 / 1 | 1.975070 / 1 | 1.975460 / 1 |
| O | nested | 2.203657 / 1 | 2.069587 / 1 | 2.023831 / 1 |
| attention | boundary | 5.967679 / 1 | 5.544047 / 1 | 5.471131 / 1 |
| attention-residual | boundary | 0.002655 / 1 | 0.003146 / 1 | 0.003196 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026640 / 1 | 0.025408 / 1 | 0.024305 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010109 / 1 | 0.002214 / 1 | 0.002144 / 1 |
| router-and-top16 | boundary | 0.603498 / 1 | 0.571308 / 1 | 0.570606 / 1 |
| EDOWN | nested | 0.639585 / 1 | 0.581126 / 1 | 0.583521 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.640206 / 1 | 0.581837 / 1 | 0.584152 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052448 / 1 | 0.027872 / 1 | 0.012584 / 1 |
| SH1 | nested | 1.092822 / 1 | 1.009506 / 1 | 1.008755 / 1 |
| SH3 | nested | 1.084115 / 1 | 0.967047 / 1 | 0.966977 / 1 |
| SH2 | nested | 1.083313 / 1 | 0.989058 / 1 | 0.993556 / 1 |
| shared-expert-during-read | boundary | 3.271492 / 1 | 2.976851 / 1 | 2.980489 / 1 |
| detail:expert-gate | nested | 2.766147 / 16 | 2.715724 / 16 | 2.483550 / 16 |
| detail:expert-up | nested | 2.730512 / 16 | 2.728017 / 16 | 2.479351 / 16 |
| detail:expert-activation | nested | 0.088978 / 16 | 0.091120 / 16 | 0.093256 / 16 |
| detail:expert-down | nested | 2.913851 / 16 | 2.892543 / 16 | 2.613883 / 16 |
| EUP | nested | 0.622673 / 1 | 0.614237 / 1 | 0.592076 / 1 |
| experts-mix-normalize-up | boundary | 9.482365 / 1 | 9.384733 / 1 | 8.594246 / 1 |
| mlp-merge-and-cleanup | boundary | 0.004408 / 1 | 0.003597 / 1 | 0.003486 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000140 / 1 | 0.000150 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005440 / 1 | 0.005541 / 1 | 0.004328 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.227009 / 1 | 114.564241 / 1 | 103.842973 / 1 |
| op:Q   int8 projection | nested | 10.385624 / 1 | 9.603690 / 1 | 9.519095 / 1 |
| op:X   mxfp4 expert proj | nested | 8.525958 / 1 | 8.453593 / 1 | 7.697090 / 1 |
| op:N   rmsnorm | nested | 0.024917 / 1 | 0.024687 / 1 | 0.022873 / 1 |
| op:SiTU + sigma | nested | 0.097515 / 1 | 0.099535 / 1 | 0.101719 / 1 |
| op:AR  snapshot aggregate | nested | 0.030738 / 1 | 0.030397 / 1 | 0.031008 / 1 |
| op:SA  softmax attention | nested | 0.006832 / 1 | 0.009147 / 1 | 0.011913 / 1 |
| op:router dot product | nested | 0.600752 / 1 | 0.568813 / 1 | 0.568292 / 1 |
| op:top-k selection | nested | 0.002325 / 1 | 0.002144 / 1 | 0.001993 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.111203 / 1 | 19.163812 / 1 | 18.288247 / 1 |

### Layer 20

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012583 / 1 | 0.004419 / 1 | 0.003216 / 1 |
| pre-attention-aggregation | boundary | 0.013926 / 1 | 0.016190 / 1 | 0.016381 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.010890 / 1 | 0.010740 / 1 |
| Q | nested | 2.501994 / 1 | 2.476297 / 1 | 2.444427 / 1 |
| K | nested | 2.291050 / 1 | 2.246477 / 1 | 2.169412 / 1 |
| V | nested | 2.160867 / 1 | 2.146239 / 1 | 2.097178 / 1 |
| B | nested | 0.068037 / 1 | 0.024656 / 1 | 0.024486 / 1 |
| FA | nested | 0.064370 / 1 | 0.031599 / 1 | 0.030978 / 1 |
| FB | nested | 0.069219 / 1 | 0.051015 / 1 | 0.050945 / 1 |
| G | nested | 2.163001 / 1 | 1.996680 / 1 | 1.996340 / 1 |
| O | nested | 2.219106 / 1 | 2.103500 / 1 | 2.101206 / 1 |
| attention | boundary | 12.605830 / 1 | 12.173192 / 1 | 11.600943 / 1 |
| attention-residual | boundary | 0.004018 / 1 | 0.003707 / 1 | 0.004077 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024846 / 1 | 0.026390 / 1 | 0.026319 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010519 / 1 | 0.002154 / 1 | 0.002004 / 1 |
| router-and-top16 | boundary | 0.602075 / 1 | 0.572570 / 1 | 0.563222 / 1 |
| EDOWN | nested | 0.644194 / 1 | 0.591916 / 1 | 0.591846 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.644795 / 1 | 0.592697 / 1 | 0.592477 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.101961 / 1 | 0.023233 / 1 | 0.006732 / 1 |
| SH1 | nested | 1.102840 / 1 | 1.028341 / 1 | 0.981133 / 1 |
| SH3 | nested | 1.109413 / 1 | 1.028912 / 1 | 1.019064 / 1 |
| SH2 | nested | 1.091279 / 1 | 1.042137 / 1 | 1.013643 / 1 |
| shared-expert-during-read | boundary | 3.314182 / 1 | 3.110851 / 1 | 3.024911 / 1 |
| detail:expert-gate | nested | 2.740020 / 16 | 2.763101 / 16 | 2.500020 / 16 |
| detail:expert-up | nested | 2.734647 / 16 | 2.806631 / 16 | 2.514379 / 16 |
| detail:expert-activation | nested | 0.089929 / 16 | 0.098413 / 16 | 0.092319 / 16 |
| detail:expert-down | nested | 2.904016 / 16 | 2.845899 / 16 | 2.617780 / 16 |
| EUP | nested | 0.645606 / 1 | 0.619227 / 1 | 0.611943 / 1 |
| experts-mix-normalize-up | boundary | 9.542557 / 1 | 9.480131 / 1 | 8.671400 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003326 / 1 | 0.003116 / 1 | 0.003346 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000140 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.411879 / 1 | 0.407612 / 1 | 0.387003 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.219271 / 1 | 114.196427 / 1 | 104.704043 / 1 |
| op:Q   int8 projection | nested | 16.129392 / 1 | 15.385332 / 1 | 15.130657 / 1 |
| op:X   mxfp4 expert proj | nested | 8.499268 / 1 | 8.541007 / 1 | 7.752192 / 1 |
| op:N   rmsnorm | nested | 0.038582 / 1 | 0.036037 / 1 | 0.023354 / 1 |
| op:L   l2 per-head | nested | 0.005901 / 1 | 0.005971 / 1 | 0.005951 / 1 |
| op:SiTU + sigma | nested | 0.098004 / 1 | 0.106919 / 1 | 0.100635 / 1 |
| op:C   shortconv | nested | 0.065393 / 1 | 0.069941 / 1 | 0.034805 / 1 |
| op:AR  snapshot aggregate | nested | 0.028734 / 1 | 0.031870 / 1 | 0.032752 / 1 |
| op:D   kda delta-rule | nested | 0.146544 / 1 | 0.139951 / 1 | 0.115045 / 1 |
| op:router dot product | nested | 0.598979 / 1 | 0.569925 / 1 | 0.560568 / 1 |
| op:top-k selection | nested | 0.002615 / 1 | 0.002023 / 1 | 0.002314 / 1 |
| op:alpha / beta / gate | nested | 0.073037 / 1 | 0.074740 / 1 | 0.058559 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.309721 / 1 | 26.433023 / 1 | 24.918343 / 1 |

### Layer 21

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014337 / 1 | 0.005921 / 1 | 0.005841 / 1 |
| pre-attention-aggregation | boundary | 0.016951 / 1 | 0.016721 / 1 | 0.017242 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.013716 / 1 | 0.021010 / 1 | 0.011872 / 1 |
| Q | nested | 2.541518 / 1 | 2.460717 / 1 | 2.419810 / 1 |
| K | nested | 2.297272 / 1 | 2.233623 / 1 | 2.169393 / 1 |
| V | nested | 2.157631 / 1 | 2.127455 / 1 | 2.099623 / 1 |
| B | nested | 0.068448 / 1 | 0.022151 / 1 | 0.021550 / 1 |
| FA | nested | 0.061214 / 1 | 0.029626 / 1 | 0.031388 / 1 |
| FB | nested | 0.068569 / 1 | 0.049362 / 1 | 0.049743 / 1 |
| G | nested | 2.161017 / 1 | 1.995408 / 1 | 1.983726 / 1 |
| O | nested | 2.212744 / 1 | 2.108940 / 1 | 2.083623 / 1 |
| attention | boundary | 12.665773 / 1 | 12.121516 / 1 | 11.571748 / 1 |
| attention-residual | boundary | 0.003617 / 1 | 0.003577 / 1 | 0.003737 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025037 / 1 | 0.025487 / 1 | 0.025097 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010028 / 1 | 0.002935 / 1 | 0.001874 / 1 |
| router-and-top16 | boundary | 0.603387 / 1 | 0.577128 / 1 | 0.571267 / 1 |
| EDOWN | nested | 0.648281 / 1 | 0.610211 / 1 | 0.583640 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.648893 / 1 | 0.611032 / 1 | 0.584352 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059802 / 1 | 0.023134 / 1 | 0.006142 / 1 |
| SH1 | nested | 1.133037 / 1 | 1.052938 / 1 | 1.017642 / 1 |
| SH3 | nested | 1.082272 / 1 | 0.975743 / 1 | 0.972707 / 1 |
| SH2 | nested | 1.087241 / 1 | 0.988557 / 1 | 1.002432 / 1 |
| shared-expert-during-read | boundary | 3.313640 / 1 | 3.028809 / 1 | 3.003582 / 1 |
| detail:expert-gate | nested | 2.726463 / 16 | 2.733245 / 16 | 2.611117 / 16 |
| detail:expert-up | nested | 2.723658 / 16 | 2.738796 / 16 | 2.591639 / 16 |
| detail:expert-activation | nested | 0.092354 / 16 | 0.094777 / 16 | 0.094736 / 16 |
| detail:expert-down | nested | 2.905375 / 16 | 2.899276 / 16 | 2.797438 / 16 |
| EUP | nested | 0.634725 / 1 | 0.615069 / 1 | 0.601834 / 1 |
| experts-mix-normalize-up | boundary | 9.462789 / 1 | 9.432161 / 1 | 9.022907 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002906 / 1 | 0.003356 / 1 | 0.002705 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000150 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.403454 / 1 | 0.401199 / 1 | 0.389157 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.493326 / 1 | 114.701509 / 1 | 110.141628 / 1 |
| op:Q   int8 projection | nested | 16.152436 / 1 | 15.268116 / 1 | 15.035518 / 1 |
| op:X   mxfp4 expert proj | nested | 8.479983 / 1 | 8.494249 / 1 | 8.123357 / 1 |
| op:N   rmsnorm | nested | 0.034074 / 1 | 0.042460 / 1 | 0.023714 / 1 |
| op:L   l2 per-head | nested | 0.005401 / 1 | 0.005781 / 1 | 0.005720 / 1 |
| op:SiTU + sigma | nested | 0.100811 / 1 | 0.103062 / 1 | 0.102400 / 1 |
| op:C   shortconv | nested | 0.063719 / 1 | 0.055123 / 1 | 0.031989 / 1 |
| op:AR  snapshot aggregate | nested | 0.031338 / 1 | 0.031559 / 1 | 0.032190 / 1 |
| op:D   kda delta-rule | nested | 0.131936 / 1 | 0.128049 / 1 | 0.111408 / 1 |
| op:router dot product | nested | 0.600823 / 1 | 0.574263 / 1 | 0.568241 / 1 |
| op:top-k selection | nested | 0.002204 / 1 | 0.002595 / 1 | 0.002756 / 1 |
| op:alpha / beta / gate | nested | 0.071173 / 1 | 0.074139 / 1 | 0.058850 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.250500 / 1 | 26.280188 / 1 | 25.223894 / 1 |

### Layer 22

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012794 / 1 | 0.005971 / 1 | 0.005320 / 1 |
| pre-attention-aggregation | boundary | 0.015299 / 1 | 0.017032 / 1 | 0.016641 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000110 / 1 |
| pre-attention-normalization | boundary | 0.011131 / 1 | 0.010720 / 1 | 0.011311 / 1 |
| Q | nested | 2.657755 / 1 | 2.658316 / 1 | 2.628601 / 1 |
| K | nested | 2.391748 / 1 | 2.321728 / 1 | 2.168601 / 1 |
| V | nested | 2.167630 / 1 | 2.141581 / 1 | 2.087761 / 1 |
| B | nested | 0.069280 / 1 | 0.029385 / 1 | 0.027571 / 1 |
| FA | nested | 0.057087 / 1 | 0.026449 / 1 | 0.025467 / 1 |
| FB | nested | 0.076122 / 1 | 0.050735 / 1 | 0.048751 / 1 |
| G | nested | 2.153022 / 1 | 1.965672 / 1 | 1.958088 / 1 |
| O | nested | 2.219397 / 1 | 2.119640 / 1 | 2.125231 / 1 |
| attention | boundary | 12.888308 / 1 | 12.454157 / 1 | 11.807840 / 1 |
| attention-residual | boundary | 0.003166 / 1 | 0.003546 / 1 | 0.003356 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026570 / 1 | 0.026099 / 1 | 0.025758 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010360 / 1 | 0.002264 / 1 | 0.002454 / 1 |
| router-and-top16 | boundary | 0.623134 / 1 | 0.578821 / 1 | 0.573191 / 1 |
| EDOWN | nested | 0.649443 / 1 | 0.601464 / 1 | 0.597727 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.650405 / 1 | 0.602376 / 1 | 0.598468 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052358 / 1 | 0.043482 / 1 | 0.006783 / 1 |
| SH1 | nested | 1.130161 / 1 | 1.008404 / 1 | 0.984048 / 1 |
| SH3 | nested | 1.098472 / 1 | 1.013404 / 1 | 0.994879 / 1 |
| SH2 | nested | 1.116506 / 1 | 1.018182 / 1 | 1.026789 / 1 |
| shared-expert-during-read | boundary | 3.356360 / 1 | 3.051821 / 1 | 3.016876 / 1 |
| detail:expert-gate | nested | 2.742000 / 16 | 2.662324 / 16 | 2.589206 / 16 |
| detail:expert-up | nested | 2.718277 / 16 | 2.667381 / 16 | 2.557147 / 16 |
| detail:expert-activation | nested | 0.091422 / 16 | 0.092572 / 16 | 0.091774 / 16 |
| detail:expert-down | nested | 2.883807 / 16 | 2.831880 / 16 | 2.700582 / 16 |
| EUP | nested | 0.629857 / 1 | 0.607505 / 1 | 0.607756 / 1 |
| experts-mix-normalize-up | boundary | 9.431280 / 1 | 9.238960 / 1 | 8.874790 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002444 / 1 | 0.002926 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000241 / 1 | 0.000250 / 1 | 0.000240 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.380821 / 1 | 0.375141 / 1 | 0.352699 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.153168 / 1 | 112.749191 / 1 | 108.020335 / 1 |
| op:Q   int8 projection | nested | 16.414998 / 1 | 15.560632 / 1 | 15.279677 / 1 |
| op:X   mxfp4 expert proj | nested | 8.464672 / 1 | 8.283684 / 1 | 7.967945 / 1 |
| op:N   rmsnorm | nested | 0.033723 / 1 | 0.039655 / 1 | 0.023535 / 1 |
| op:L   l2 per-head | nested | 0.006052 / 1 | 0.005611 / 1 | 0.006051 / 1 |
| op:SiTU + sigma | nested | 0.099957 / 1 | 0.101148 / 1 | 0.100060 / 1 |
| op:C   shortconv | nested | 0.056325 / 1 | 0.053690 / 1 | 0.034144 / 1 |
| op:AR  snapshot aggregate | nested | 0.031579 / 1 | 0.033111 / 1 | 0.031569 / 1 |
| op:D   kda delta-rule | nested | 0.161882 / 1 | 0.152455 / 1 | 0.134962 / 1 |
| op:router dot product | nested | 0.620178 / 1 | 0.575636 / 1 | 0.570115 / 1 |
| op:top-k selection | nested | 0.002565 / 1 | 0.002885 / 1 | 0.002415 / 1 |
| op:alpha / beta / gate | nested | 0.082664 / 1 | 0.073948 / 1 | 0.058119 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.470671 / 1 | 26.419678 / 1 | 25.304415 / 1 |

### Layer 23

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012032 / 1 | 0.004839 / 1 | 0.005069 / 1 |
| pre-attention-aggregation | boundary | 0.016200 / 1 | 0.016851 / 1 | 0.016701 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010640 / 1 | 0.010930 / 1 | 0.010469 / 1 |
| QA | nested | 0.312163 / 1 | 0.305591 / 1 | 0.303847 / 1 |
| QB | nested | 0.752095 / 1 | 0.739031 / 1 | 0.734081 / 1 |
| KA | nested | 0.125585 / 1 | 0.108082 / 1 | 0.110206 / 1 |
| KB | nested | 0.344964 / 1 | 0.307134 / 1 | 0.308346 / 1 |
| G | nested | 2.185042 / 1 | 2.042436 / 1 | 2.016757 / 1 |
| O | nested | 2.191184 / 1 | 2.024553 / 1 | 2.006779 / 1 |
| attention | boundary | 6.014156 / 1 | 5.626281 / 1 | 5.576238 / 1 |
| attention-residual | boundary | 0.002896 / 1 | 0.002905 / 1 | 0.003046 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025147 / 1 | 0.024306 / 1 | 0.028664 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010880 / 1 | 0.002455 / 1 | 0.002525 / 1 |
| router-and-top16 | boundary | 0.603428 / 1 | 0.574764 / 1 | 0.575776 / 1 |
| EDOWN | nested | 0.642070 / 1 | 0.603698 / 1 | 0.599750 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.642721 / 1 | 0.604439 / 1 | 0.600442 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.071934 / 1 | 0.027321 / 1 | 0.006071 / 1 |
| SH1 | nested | 1.089956 / 1 | 0.990210 / 1 | 0.993937 / 1 |
| SH3 | nested | 1.077634 / 1 | 0.969912 / 1 | 0.975923 / 1 |
| SH2 | nested | 1.092260 / 1 | 1.007923 / 1 | 0.997374 / 1 |
| shared-expert-during-read | boundary | 3.270751 / 1 | 2.979797 / 1 | 2.978825 / 1 |
| detail:expert-gate | nested | 2.756027 / 16 | 2.678374 / 16 | 2.570172 / 16 |
| detail:expert-up | nested | 2.748056 / 16 | 2.703482 / 16 | 2.578327 / 16 |
| detail:expert-activation | nested | 0.092162 / 16 | 0.094398 / 16 | 0.090978 / 16 |
| detail:expert-down | nested | 2.943830 / 16 | 2.876427 / 16 | 2.709162 / 16 |
| EUP | nested | 0.636580 / 1 | 0.596454 / 1 | 0.603076 / 1 |
| experts-mix-normalize-up | boundary | 9.572935 / 1 | 9.300025 / 1 | 8.863910 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003145 / 1 | 0.003076 / 1 | 0.002695 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000160 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006031 / 1 | 0.006111 / 1 | 0.005130 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.507535 / 1 | 113.346027 / 1 | 107.608927 / 1 |
| op:Q   int8 projection | nested | 10.448260 / 1 | 9.693781 / 1 | 9.648675 / 1 |
| op:X   mxfp4 expert proj | nested | 8.569951 / 1 | 8.382470 / 1 | 7.978145 / 1 |
| op:N   rmsnorm | nested | 0.024455 / 1 | 0.024525 / 1 | 0.024136 / 1 |
| op:SiTU + sigma | nested | 0.100537 / 1 | 0.102823 / 1 | 0.099484 / 1 |
| op:AR  snapshot aggregate | nested | 0.030588 / 1 | 0.030887 / 1 | 0.031449 / 1 |
| op:SA  softmax attention | nested | 0.006883 / 1 | 0.009778 / 1 | 0.011962 / 1 |
| op:router dot product | nested | 0.600202 / 1 | 0.571688 / 1 | 0.573011 / 1 |
| op:top-k selection | nested | 0.002665 / 1 | 0.002184 / 1 | 0.002284 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.269107 / 1 | 19.190232 / 1 | 18.681792 / 1 |

### Layer 24

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011241 / 1 | 0.004639 / 1 | 0.003587 / 1 |
| pre-attention-aggregation | boundary | 0.014357 / 1 | 0.016360 / 1 | 0.016070 / 1 |
| snapshot-push | boundary | 0.001302 / 1 | 0.001312 / 1 | 0.001313 / 1 |
| pre-attention-normalization | boundary | 0.010580 / 1 | 0.010941 / 1 | 0.010761 / 1 |
| Q | nested | 2.558891 / 1 | 2.466939 / 1 | 2.445408 / 1 |
| K | nested | 2.297001 / 1 | 2.215369 / 1 | 2.147763 / 1 |
| V | nested | 2.152672 / 1 | 2.111445 / 1 | 2.098050 / 1 |
| B | nested | 0.068728 / 1 | 0.027471 / 1 | 0.027651 / 1 |
| FA | nested | 0.063108 / 1 | 0.033012 / 1 | 0.033072 / 1 |
| FB | nested | 0.076663 / 1 | 0.048500 / 1 | 0.049543 / 1 |
| G | nested | 2.171196 / 1 | 1.983656 / 1 | 1.984187 / 1 |
| O | nested | 2.227061 / 1 | 2.092449 / 1 | 2.106786 / 1 |
| attention | boundary | 12.707600 / 1 | 12.093053 / 1 | 11.613276 / 1 |
| attention-residual | boundary | 0.002395 / 1 | 0.007754 / 1 | 0.002234 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025377 / 1 | 0.027762 / 1 | 0.024957 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009508 / 1 | 0.002194 / 1 | 0.002094 / 1 |
| router-and-top16 | boundary | 0.607766 / 1 | 0.569894 / 1 | 0.570175 / 1 |
| EDOWN | nested | 0.656096 / 1 | 0.607064 / 1 | 0.590162 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.656868 / 1 | 0.608036 / 1 | 0.590944 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.058670 / 1 | 0.029275 / 1 | 0.006823 / 1 |
| SH1 | nested | 1.120413 / 1 | 1.021558 / 1 | 0.996231 / 1 |
| SH3 | nested | 1.095477 / 1 | 1.039131 / 1 | 1.015558 / 1 |
| SH2 | nested | 1.099905 / 1 | 1.019865 / 1 | 1.022019 / 1 |
| shared-expert-during-read | boundary | 3.327366 / 1 | 3.091786 / 1 | 3.044908 / 1 |
| detail:expert-gate | nested | 2.727809 / 16 | 2.579808 / 16 | 2.837060 / 16 |
| detail:expert-up | nested | 2.732369 / 16 | 2.577915 / 16 | 2.787147 / 16 |
| detail:expert-activation | nested | 0.090348 / 16 | 0.091141 / 16 | 0.101150 / 16 |
| detail:expert-down | nested | 2.873618 / 16 | 2.726142 / 16 | 2.737334 / 16 |
| EUP | nested | 0.631269 / 1 | 0.612404 / 1 | 0.733020 / 1 |
| experts-mix-normalize-up | boundary | 9.441859 / 1 | 8.941204 / 1 | 9.513814 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002655 / 1 | 0.003366 / 1 | 0.002685 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000250 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.414113 / 1 | 0.412560 / 1 | 0.401460 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.511224 / 1 | 108.816455 / 1 | 107.791306 / 1 |
| op:Q   int8 projection | nested | 16.216946 / 1 | 15.277251 / 1 | 15.247624 / 1 |
| op:X   mxfp4 expert proj | nested | 8.458703 / 1 | 8.005726 / 1 | 8.493307 / 1 |
| op:N   rmsnorm | nested | 0.037620 / 1 | 0.033041 / 1 | 0.023382 / 1 |
| op:L   l2 per-head | nested | 0.004969 / 1 | 0.005771 / 1 | 0.005861 / 1 |
| op:SiTU + sigma | nested | 0.099114 / 1 | 0.098926 / 1 | 0.109113 / 1 |
| op:C   shortconv | nested | 0.072917 / 1 | 0.066104 / 1 | 0.035847 / 1 |
| op:AR  snapshot aggregate | nested | 0.030186 / 1 | 0.033452 / 1 | 0.031378 / 1 |
| op:D   kda delta-rule | nested | 0.147165 / 1 | 0.144149 / 1 | 0.127348 / 1 |
| op:router dot product | nested | 0.605091 / 1 | 0.566959 / 1 | 0.566949 / 1 |
| op:top-k selection | nested | 0.002224 / 1 | 0.002414 / 1 | 0.002715 / 1 |
| op:alpha / beta / gate | nested | 0.073518 / 1 | 0.074389 / 1 | 0.058279 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.298440 / 1 | 25.826600 / 1 | 25.811552 / 1 |

### Layer 25

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012423 / 1 | 0.005249 / 1 | 0.004809 / 1 |
| pre-attention-aggregation | boundary | 0.016792 / 1 | 0.015960 / 1 | 0.016701 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.012213 / 1 | 0.011241 / 1 | 0.012123 / 1 |
| Q | nested | 2.479312 / 1 | 2.442683 / 1 | 2.865253 / 1 |
| K | nested | 2.287755 / 1 | 2.216711 / 1 | 2.166267 / 1 |
| V | nested | 2.164143 / 1 | 2.120722 / 1 | 2.103880 / 1 |
| B | nested | 0.093395 / 1 | 0.030266 / 1 | 0.027131 / 1 |
| FA | nested | 0.059220 / 1 | 0.028443 / 1 | 0.032861 / 1 |
| FB | nested | 0.068418 / 1 | 0.047549 / 1 | 0.051947 / 1 |
| G | nested | 2.186215 / 1 | 2.036385 / 1 | 2.026746 / 1 |
| O | nested | 2.236208 / 1 | 2.071079 / 1 | 2.074796 / 1 |
| attention | boundary | 12.688846 / 1 | 12.129300 / 1 | 12.101228 / 1 |
| attention-residual | boundary | 0.003105 / 1 | 0.003537 / 1 | 0.003888 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025558 / 1 | 0.025497 / 1 | 0.024746 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010620 / 1 | 0.002675 / 1 | 0.002023 / 1 |
| router-and-top16 | boundary | 0.610240 / 1 | 0.576287 / 1 | 0.580855 / 1 |
| EDOWN | nested | 0.660203 / 1 | 0.588299 / 1 | 0.580715 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.660985 / 1 | 0.588990 / 1 | 0.581386 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.054733 / 1 | 0.012954 / 1 | 0.006112 / 1 |
| SH1 | nested | 1.129861 / 1 | 1.047897 / 1 | 1.039262 / 1 |
| SH3 | nested | 1.121956 / 1 | 1.011079 / 1 | 0.997213 / 1 |
| SH2 | nested | 1.107429 / 1 | 1.012893 / 1 | 0.990070 / 1 |
| shared-expert-during-read | boundary | 3.374445 / 1 | 3.082759 / 1 | 3.037415 / 1 |
| detail:expert-gate | nested | 2.697517 / 16 | 2.625234 / 16 | 2.589687 / 16 |
| detail:expert-up | nested | 2.719580 / 16 | 2.661151 / 16 | 2.614473 / 16 |
| detail:expert-activation | nested | 0.089136 / 16 | 0.091047 / 16 | 0.090180 / 16 |
| detail:expert-down | nested | 2.909497 / 16 | 2.872316 / 16 | 2.804938 / 16 |
| EUP | nested | 0.642902 / 1 | 0.614838 / 1 | 0.600202 / 1 |
| experts-mix-normalize-up | boundary | 9.436440 / 1 | 9.210227 / 1 | 9.006426 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003055 / 1 | 0.002976 / 1 | 0.003035 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000461 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.411639 / 1 | 0.391331 / 1 | 0.369921 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.526353 / 1 | 112.548317 / 1 | 110.658264 / 1 |
| op:Q   int8 projection | nested | 16.235442 / 1 | 15.267109 / 1 | 15.554720 / 1 |
| op:X   mxfp4 expert proj | nested | 8.447260 / 1 | 8.281182 / 1 | 8.131442 / 1 |
| op:N   rmsnorm | nested | 0.033903 / 1 | 0.033532 / 1 | 0.024005 / 1 |
| op:L   l2 per-head | nested | 0.005300 / 1 | 0.005561 / 1 | 0.005831 / 1 |
| op:SiTU + sigma | nested | 0.101268 / 1 | 0.098882 / 1 | 0.098074 / 1 |
| op:C   shortconv | nested | 0.054903 / 1 | 0.072434 / 1 | 0.033903 / 1 |
| op:AR  snapshot aggregate | nested | 0.031930 / 1 | 0.031399 / 1 | 0.031158 / 1 |
| op:D   kda delta-rule | nested | 0.162403 / 1 | 0.164587 / 1 | 0.148337 / 1 |
| op:router dot product | nested | 0.607495 / 1 | 0.573351 / 1 | 0.578030 / 1 |
| op:top-k selection | nested | 0.002114 / 1 | 0.002234 / 1 | 0.002374 / 1 |
| op:alpha / beta / gate | nested | 0.072837 / 1 | 0.075020 / 1 | 0.058229 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.330149 / 1 | 26.065796 / 1 | 25.757180 / 1 |

### Layer 26

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.021190 / 1 | 0.006001 / 1 | 0.005640 / 1 |
| pre-attention-aggregation | boundary | 0.016672 / 1 | 0.017252 / 1 | 0.016521 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.011972 / 1 | 0.012624 / 1 | 0.011662 / 1 |
| Q | nested | 2.454375 / 1 | 2.443314 / 1 | 2.437344 / 1 |
| K | nested | 2.289257 / 1 | 2.209017 / 1 | 2.172458 / 1 |
| V | nested | 2.156509 / 1 | 2.118468 / 1 | 2.096547 / 1 |
| B | nested | 0.076252 / 1 | 0.028373 / 1 | 0.028583 / 1 |
| FA | nested | 0.059271 / 1 | 0.026409 / 1 | 0.029455 / 1 |
| FB | nested | 0.075381 / 1 | 0.049783 / 1 | 0.052438 / 1 |
| G | nested | 2.174432 / 1 | 1.986742 / 1 | 2.000046 / 1 |
| O | nested | 2.201853 / 1 | 2.051112 / 1 | 2.054208 / 1 |
| attention | boundary | 12.563972 / 1 | 12.015768 / 1 | 11.619358 / 1 |
| attention-residual | boundary | 0.003276 / 1 | 0.003507 / 1 | 0.003466 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026049 / 1 | 0.025578 / 1 | 0.025357 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010459 / 1 | 0.002204 / 1 | 0.002024 / 1 |
| router-and-top16 | boundary | 0.628023 / 1 | 0.588289 / 1 | 0.586676 / 1 |
| EDOWN | nested | 0.670573 / 1 | 0.641518 / 1 | 0.625849 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.671345 / 1 | 0.642330 / 1 | 0.626630 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.064972 / 1 | 0.018034 / 1 | 0.013005 / 1 |
| SH1 | nested | 1.105185 / 1 | 1.003024 / 1 | 0.991924 / 1 |
| SH3 | nested | 1.106989 / 1 | 0.990841 / 1 | 0.978889 / 1 |
| SH2 | nested | 1.079036 / 1 | 0.999087 / 1 | 0.993766 / 1 |
| shared-expert-during-read | boundary | 3.303121 / 1 | 3.004052 / 1 | 2.975700 / 1 |
| detail:expert-gate | nested | 2.728539 / 16 | 2.632711 / 16 | 2.611741 / 16 |
| detail:expert-up | nested | 2.738478 / 16 | 2.604509 / 16 | 2.635933 / 16 |
| detail:expert-activation | nested | 0.093955 / 16 | 0.090275 / 16 | 0.091995 / 16 |
| detail:expert-down | nested | 2.912123 / 16 | 2.739970 / 16 | 2.784662 / 16 |
| EUP | nested | 0.632111 / 1 | 0.604079 / 1 | 0.602616 / 1 |
| experts-mix-normalize-up | boundary | 9.487174 / 1 | 9.020993 / 1 | 9.038677 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002575 / 1 | 0.002685 / 1 | 0.002584 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000250 / 1 | 0.000240 / 1 | 0.000481 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.397452 / 1 | 0.384578 / 1 | 0.369621 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.676648 / 1 | 109.887966 / 1 | 110.597031 / 1 |
| op:Q   int8 projection | nested | 16.079432 / 1 | 15.150165 / 1 | 15.062498 / 1 |
| op:X   mxfp4 expert proj | nested | 8.505633 / 1 | 8.100331 / 1 | 8.157603 / 1 |
| op:N   rmsnorm | nested | 0.034425 / 1 | 0.037810 / 1 | 0.023973 / 1 |
| op:L   l2 per-head | nested | 0.005721 / 1 | 0.005420 / 1 | 0.005500 / 1 |
| op:SiTU + sigma | nested | 0.102542 / 1 | 0.098320 / 1 | 0.100058 / 1 |
| op:C   shortconv | nested | 0.061555 / 1 | 0.069560 / 1 | 0.031719 / 1 |
| op:AR  snapshot aggregate | nested | 0.032681 / 1 | 0.032900 / 1 | 0.031259 / 1 |
| op:D   kda delta-rule | nested | 0.130815 / 1 | 0.123270 / 1 | 0.107732 / 1 |
| op:router dot product | nested | 0.625268 / 1 | 0.585574 / 1 | 0.583831 / 1 |
| op:top-k selection | nested | 0.002244 / 1 | 0.002355 / 1 | 0.002154 / 1 |
| op:alpha / beta / gate | nested | 0.073166 / 1 | 0.072796 / 1 | 0.058850 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.214913 / 1 | 25.751079 / 1 | 25.303643 / 1 |

### Layer 27

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012483 / 1 | 0.005450 / 1 | 0.005009 / 1 |
| pre-attention-aggregation | boundary | 0.017212 / 1 | 0.017373 / 1 | 0.023053 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000030 / 1 | 0.000101 / 1 |
| pre-attention-normalization | boundary | 0.011201 / 1 | 0.010610 / 1 | 0.011892 / 1 |
| QA | nested | 0.318375 / 1 | 0.311372 / 1 | 0.306743 / 1 |
| QB | nested | 0.746184 / 1 | 0.728080 / 1 | 0.729132 / 1 |
| KA | nested | 0.125695 / 1 | 0.101269 / 1 | 0.101319 / 1 |
| KB | nested | 0.348481 / 1 | 0.308426 / 1 | 0.305381 / 1 |
| G | nested | 2.165446 / 1 | 1.931529 / 1 | 1.925227 / 1 |
| O | nested | 2.185403 / 1 | 1.995087 / 1 | 1.960823 / 1 |
| attention | boundary | 5.991914 / 1 | 5.474117 / 1 | 5.425576 / 1 |
| attention-residual | boundary | 0.002755 / 1 | 0.002935 / 1 | 0.003246 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025538 / 1 | 0.024576 / 1 | 0.025447 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009718 / 1 | 0.002024 / 1 | 0.002124 / 1 |
| router-and-top16 | boundary | 0.600171 / 1 | 0.567090 / 1 | 0.576447 / 1 |
| EDOWN | nested | 0.640457 / 1 | 0.588219 / 1 | 0.588179 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.641248 / 1 | 0.588960 / 1 | 0.588910 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078657 / 1 | 0.022132 / 1 | 0.006071 / 1 |
| SH1 | nested | 1.114171 / 1 | 1.035315 / 1 | 1.045253 / 1 |
| SH3 | nested | 1.099885 / 1 | 0.985671 / 1 | 0.989299 / 1 |
| SH2 | nested | 1.079637 / 1 | 0.989549 / 1 | 1.006601 / 1 |
| shared-expert-during-read | boundary | 3.305155 / 1 | 3.022217 / 1 | 3.052002 / 1 |
| detail:expert-gate | nested | 2.718728 / 16 | 2.723640 / 16 | 3.068721 / 16 |
| detail:expert-up | nested | 2.728040 / 16 | 2.688474 / 16 | 3.077059 / 16 |
| detail:expert-activation | nested | 0.092444 / 16 | 0.093653 / 16 | 0.103434 / 16 |
| detail:expert-down | nested | 2.938609 / 16 | 2.908245 / 16 | 3.387868 / 16 |
| EUP | nested | 0.634936 / 1 | 0.589532 / 1 | 0.589051 / 1 |
| experts-mix-normalize-up | boundary | 9.517231 / 1 | 9.355348 / 1 | 10.550601 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003045 / 1 | 0.003266 / 1 | 0.002795 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000411 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005390 / 1 | 0.006723 / 1 | 0.004939 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.488099 / 1 | 114.744007 / 1 | 110.058210 / 1 |
| op:Q   int8 projection | nested | 10.457398 / 1 | 9.562677 / 1 | 9.545496 / 1 |
| op:X   mxfp4 expert proj | nested | 8.510889 / 1 | 8.446813 / 1 | 9.670346 / 1 |
| op:N   rmsnorm | nested | 0.025418 / 1 | 0.024576 / 1 | 0.024334 / 1 |
| op:SiTU + sigma | nested | 0.101008 / 1 | 0.101677 / 1 | 0.111139 / 1 |
| op:AR  snapshot aggregate | nested | 0.031429 / 1 | 0.031359 / 1 | 0.037862 / 1 |
| op:SA  softmax attention | nested | 0.006612 / 1 | 0.009228 / 1 | 0.012002 / 1 |
| op:router dot product | nested | 0.596995 / 1 | 0.564054 / 1 | 0.573732 / 1 |
| op:top-k selection | nested | 0.002625 / 1 | 0.002535 / 1 | 0.002295 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.228551 / 1 | 19.109652 / 1 | 20.285378 / 1 |

### Layer 28

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011201 / 1 | 0.004138 / 1 | 0.003577 / 1 |
| pre-attention-aggregation | boundary | 0.014797 / 1 | 0.014346 / 1 | 0.014567 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011161 / 1 | 0.011381 / 1 | 0.010570 / 1 |
| Q | nested | 2.477468 / 1 | 2.464715 / 1 | 2.449737 / 1 |
| K | nested | 2.313792 / 1 | 2.236348 / 1 | 2.142172 / 1 |
| V | nested | 2.163351 / 1 | 2.117566 / 1 | 2.104823 / 1 |
| B | nested | 0.085119 / 1 | 0.031299 / 1 | 0.031529 / 1 |
| FA | nested | 0.053220 / 1 | 0.026129 / 1 | 0.025568 / 1 |
| FB | nested | 0.075240 / 1 | 0.046737 / 1 | 0.051005 / 1 |
| G | nested | 2.155367 / 1 | 1.977976 / 1 | 1.969760 / 1 |
| O | nested | 2.244713 / 1 | 2.126112 / 1 | 2.137824 / 1 |
| attention | boundary | 12.630907 / 1 | 12.085459 / 1 | 11.582389 / 1 |
| attention-residual | boundary | 0.004038 / 1 | 0.003798 / 1 | 0.003998 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025387 / 1 | 0.026499 / 1 | 0.024315 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009608 / 1 | 0.002395 / 1 | 0.002064 / 1 |
| router-and-top16 | boundary | 0.600381 / 1 | 0.575926 / 1 | 0.571568 / 1 |
| EDOWN | nested | 0.670443 / 1 | 0.623585 / 1 | 0.618686 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.671284 / 1 | 0.624426 / 1 | 0.619427 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.055123 / 1 | 0.013114 / 1 | 0.006562 / 1 |
| SH1 | nested | 1.134951 / 1 | 1.010989 / 1 | 0.994969 / 1 |
| SH3 | nested | 1.111346 / 1 | 1.034502 / 1 | 1.012792 / 1 |
| SH2 | nested | 1.090928 / 1 | 1.020536 / 1 | 0.999467 / 1 |
| shared-expert-during-read | boundary | 3.348696 / 1 | 3.077500 / 1 | 3.018589 / 1 |
| detail:expert-gate | nested | 2.783813 / 16 | 2.689566 / 16 | 2.648017 / 16 |
| detail:expert-up | nested | 2.753352 / 16 | 2.698149 / 16 | 2.659390 / 16 |
| detail:expert-activation | nested | 0.091027 / 16 | 0.093186 / 16 | 0.090960 / 16 |
| detail:expert-down | nested | 2.886124 / 16 | 2.885207 / 16 | 2.792687 / 16 |
| EUP | nested | 0.630638 / 1 | 0.597386 / 1 | 0.611713 / 1 |
| experts-mix-normalize-up | boundary | 9.522941 / 1 | 9.297189 / 1 | 9.118155 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002605 / 1 | 0.002965 / 1 | 0.002615 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000271 / 1 | 0.000161 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.400729 / 1 | 0.390349 / 1 | 0.380190 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.291408 / 1 | 113.612054 / 1 | 111.447497 / 1 |
| op:Q   int8 projection | nested | 16.204995 / 1 | 15.311996 / 1 | 15.148210 / 1 |
| op:X   mxfp4 expert proj | nested | 8.548399 / 1 | 8.399943 / 1 | 8.224904 / 1 |
| op:N   rmsnorm | nested | 0.034445 / 1 | 0.030306 / 1 | 0.022492 / 1 |
| op:L   l2 per-head | nested | 0.005350 / 1 | 0.005801 / 1 | 0.006241 / 1 |
| op:SiTU + sigma | nested | 0.099512 / 1 | 0.100811 / 1 | 0.098866 / 1 |
| op:C   shortconv | nested | 0.069059 / 1 | 0.065091 / 1 | 0.034185 / 1 |
| op:AR  snapshot aggregate | nested | 0.030296 / 1 | 0.030606 / 1 | 0.029024 / 1 |
| op:D   kda delta-rule | nested | 0.145843 / 1 | 0.138218 / 1 | 0.105618 / 1 |
| op:router dot product | nested | 0.597025 / 1 | 0.572900 / 1 | 0.569244 / 1 |
| op:top-k selection | nested | 0.002604 / 1 | 0.002385 / 1 | 0.001934 / 1 |
| op:alpha / beta / gate | nested | 0.073217 / 1 | 0.074860 / 1 | 0.058540 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.318427 / 1 | 26.137040 / 1 | 25.365629 / 1 |

### Layer 29

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013175 / 1 | 0.005560 / 1 | 0.004960 / 1 |
| pre-attention-aggregation | boundary | 0.015849 / 1 | 0.016481 / 1 | 0.015418 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.011973 / 1 | 0.011311 / 1 | 0.011201 / 1 |
| Q | nested | 2.472990 / 1 | 2.457170 / 1 | 2.432775 / 1 |
| K | nested | 2.328019 / 1 | 2.231689 / 1 | 2.164935 / 1 |
| V | nested | 2.176246 / 1 | 2.139026 / 1 | 2.101997 / 1 |
| B | nested | 0.071324 / 1 | 0.024155 / 1 | 0.022803 / 1 |
| FA | nested | 0.058659 / 1 | 0.032450 / 1 | 0.032320 / 1 |
| FB | nested | 0.068849 / 1 | 0.048550 / 1 | 0.050675 / 1 |
| G | nested | 2.174673 / 1 | 1.981672 / 1 | 1.977985 / 1 |
| O | nested | 2.259952 / 1 | 2.206092 / 1 | 2.215328 / 1 |
| attention | boundary | 12.684698 / 1 | 12.212796 / 1 | 11.696772 / 1 |
| attention-residual | boundary | 0.003387 / 1 | 0.003567 / 1 | 0.004028 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025087 / 1 | 0.025337 / 1 | 0.025147 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009548 / 1 | 0.002274 / 1 | 0.002244 / 1 |
| router-and-top16 | boundary | 0.593619 / 1 | 0.562190 / 1 | 0.565757 / 1 |
| EDOWN | nested | 0.676063 / 1 | 0.631660 / 1 | 0.626841 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.676825 / 1 | 0.632622 / 1 | 0.627652 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078657 / 1 | 0.012794 / 1 | 0.006282 / 1 |
| SH1 | nested | 1.104573 / 1 | 0.979941 / 1 | 0.973068 / 1 |
| SH3 | nested | 1.117678 / 1 | 0.998876 / 1 | 0.989549 / 1 |
| SH2 | nested | 1.118730 / 1 | 1.050433 / 1 | 1.047807 / 1 |
| shared-expert-during-read | boundary | 3.352313 / 1 | 3.040971 / 1 | 3.022076 / 1 |
| detail:expert-gate | nested | 2.739549 / 16 | 2.719029 / 16 | 2.572655 / 16 |
| detail:expert-up | nested | 2.728268 / 16 | 2.710573 / 16 | 2.638261 / 16 |
| detail:expert-activation | nested | 0.093524 / 16 | 0.089478 / 16 | 0.090638 / 16 |
| detail:expert-down | nested | 2.885581 / 16 | 2.895368 / 16 | 2.770186 / 16 |
| EUP | nested | 0.635116 / 1 | 0.601063 / 1 | 0.598899 / 1 |
| experts-mix-normalize-up | boundary | 9.489399 / 1 | 9.348976 / 1 | 8.995365 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002825 / 1 | 0.003096 / 1 | 0.003166 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000170 / 1 | 0.000181 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.412741 / 1 | 0.404015 / 1 | 0.390449 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.539129 / 1 | 115.001068 / 1 | 109.130142 / 1 |
| op:Q   int8 projection | nested | 16.261180 / 1 | 15.381043 / 1 | 15.233291 / 1 |
| op:X   mxfp4 expert proj | nested | 8.482197 / 1 | 8.448852 / 1 | 8.106426 / 1 |
| op:N   rmsnorm | nested | 0.034595 / 1 | 0.032020 / 1 | 0.022743 / 1 |
| op:L   l2 per-head | nested | 0.009237 / 1 | 0.005821 / 1 | 0.009127 / 1 |
| op:SiTU + sigma | nested | 0.101669 / 1 | 0.097925 / 1 | 0.099156 / 1 |
| op:C   shortconv | nested | 0.060463 / 1 | 0.063928 / 1 | 0.032702 / 1 |
| op:AR  snapshot aggregate | nested | 0.031098 / 1 | 0.031690 / 1 | 0.030076 / 1 |
| op:D   kda delta-rule | nested | 0.130594 / 1 | 0.120736 / 1 | 0.103424 / 1 |
| op:router dot product | nested | 0.590764 / 1 | 0.559195 / 1 | 0.562992 / 1 |
| op:top-k selection | nested | 0.002335 / 1 | 0.002314 / 1 | 0.002244 / 1 |
| op:alpha / beta / gate | nested | 0.077735 / 1 | 0.071904 / 1 | 0.059792 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.378208 / 1 | 26.290086 / 1 | 25.378533 / 1 |

### Layer 30

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013916 / 1 | 0.005972 / 1 | 0.005150 / 1 |
| pre-attention-aggregation | boundary | 0.017182 / 1 | 0.016922 / 1 | 0.016020 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012062 / 1 | 0.012573 / 1 | 0.013395 / 1 |
| Q | nested | 2.494651 / 1 | 2.473711 / 1 | 2.456028 / 1 |
| K | nested | 2.326776 / 1 | 2.220308 / 1 | 2.176887 / 1 |
| V | nested | 2.182798 / 1 | 2.138465 / 1 | 2.100033 / 1 |
| B | nested | 0.068839 / 1 | 0.024727 / 1 | 0.021239 / 1 |
| FA | nested | 0.056776 / 1 | 0.027331 / 1 | 0.025347 / 1 |
| FB | nested | 0.076453 / 1 | 0.048851 / 1 | 0.049402 / 1 |
| G | nested | 2.191114 / 1 | 1.987854 / 1 | 1.988465 / 1 |
| O | nested | 2.209178 / 1 | 2.131281 / 1 | 2.117556 / 1 |
| attention | boundary | 12.618865 / 1 | 12.049802 / 1 | 11.578923 / 1 |
| attention-residual | boundary | 0.003627 / 1 | 0.003777 / 1 | 0.004058 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025838 / 1 | 0.024646 / 1 | 0.025077 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010149 / 1 | 0.002124 / 1 | 0.002164 / 1 |
| router-and-top16 | boundary | 0.602185 / 1 | 0.565647 / 1 | 0.569744 / 1 |
| EDOWN | nested | 0.665674 / 1 | 0.599811 / 1 | 0.603137 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.666445 / 1 | 0.600682 / 1 | 0.603898 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.074389 / 1 | 0.006773 / 1 | 0.012103 / 1 |
| SH1 | nested | 1.128699 / 1 | 1.014595 / 1 | 1.005348 / 1 |
| SH3 | nested | 1.108381 / 1 | 1.012611 / 1 | 0.986413 / 1 |
| SH2 | nested | 1.096228 / 1 | 1.019394 / 1 | 1.018342 / 1 |
| shared-expert-during-read | boundary | 3.344939 / 1 | 3.058094 / 1 | 3.021665 / 1 |
| detail:expert-gate | nested | 2.754558 / 16 | 2.717075 / 16 | 2.682166 / 16 |
| detail:expert-up | nested | 2.736278 / 16 | 2.722509 / 16 | 2.687411 / 16 |
| detail:expert-activation | nested | 0.090392 / 16 | 0.095178 / 16 | 0.092362 / 16 |
| detail:expert-down | nested | 2.924774 / 16 | 2.839865 / 16 | 2.791757 / 16 |
| EUP | nested | 0.633734 / 1 | 0.598087 / 1 | 0.604359 / 1 |
| experts-mix-normalize-up | boundary | 9.548428 / 1 | 9.310554 / 1 | 9.191592 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002926 / 1 | 0.003346 / 1 | 0.002885 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000260 / 1 | 0.000230 / 1 | 0.000240 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.413703 / 1 | 0.409245 / 1 | 0.380741 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.635670 / 1 | 114.441612 / 1 | 112.112700 / 1 |
| op:Q   int8 projection | nested | 16.237779 / 1 | 15.295293 / 1 | 15.150825 / 1 |
| op:X   mxfp4 expert proj | nested | 8.543006 / 1 | 8.410363 / 1 | 8.290423 / 1 |
| op:N   rmsnorm | nested | 0.040957 / 1 | 0.035767 / 1 | 0.025428 / 1 |
| op:L   l2 per-head | nested | 0.005530 / 1 | 0.005360 / 1 | 0.005360 / 1 |
| op:SiTU + sigma | nested | 0.099159 / 1 | 0.103143 / 1 | 0.100586 / 1 |
| op:C   shortconv | nested | 0.070973 / 1 | 0.046337 / 1 | 0.032340 / 1 |
| op:AR  snapshot aggregate | nested | 0.032431 / 1 | 0.031539 / 1 | 0.030146 / 1 |
| op:D   kda delta-rule | nested | 0.117049 / 1 | 0.112971 / 1 | 0.097683 / 1 |
| op:router dot product | nested | 0.599059 / 1 | 0.562251 / 1 | 0.566899 / 1 |
| op:top-k selection | nested | 0.002585 / 1 | 0.002595 / 1 | 0.002365 / 1 |
| op:alpha / beta / gate | nested | 0.071093 / 1 | 0.073868 / 1 | 0.058910 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.365295 / 1 | 26.077719 / 1 | 25.435029 / 1 |

### Layer 31

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012463 / 1 | 0.005090 / 1 | 0.004198 / 1 |
| pre-attention-aggregation | boundary | 0.015709 / 1 | 0.016331 / 1 | 0.015369 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010770 / 1 | 0.010209 / 1 | 0.010840 / 1 |
| QA | nested | 0.308947 / 1 | 0.296824 / 1 | 0.302125 / 1 |
| QB | nested | 0.731126 / 1 | 0.706240 / 1 | 0.712551 / 1 |
| KA | nested | 0.125424 / 1 | 0.108733 / 1 | 0.108182 / 1 |
| KB | nested | 0.346698 / 1 | 0.300942 / 1 | 0.307023 / 1 |
| G | nested | 2.186625 / 1 | 1.978206 / 1 | 1.968187 / 1 |
| O | nested | 2.173149 / 1 | 1.970812 / 1 | 1.964830 / 1 |
| attention | boundary | 5.974773 / 1 | 5.458728 / 1 | 5.463547 / 1 |
| attention-residual | boundary | 0.002736 / 1 | 0.003066 / 1 | 0.003015 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026068 / 1 | 0.024876 / 1 | 0.027011 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010169 / 1 | 0.002305 / 1 | 0.001943 / 1 |
| router-and-top16 | boundary | 0.624447 / 1 | 0.600242 / 1 | 0.599901 / 1 |
| EDOWN | nested | 0.639715 / 1 | 0.596113 / 1 | 0.731236 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.640486 / 1 | 0.596895 / 1 | 0.732338 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.067326 / 1 | 0.022131 / 1 | 0.006552 / 1 |
| SH1 | nested | 1.107850 / 1 | 0.996251 / 1 | 1.142585 / 1 |
| SH3 | nested | 1.090117 / 1 | 0.973559 / 1 | 1.291904 / 1 |
| SH2 | nested | 1.083344 / 1 | 1.029203 / 1 | 1.026828 / 1 |
| shared-expert-during-read | boundary | 3.292742 / 1 | 3.010344 / 1 | 3.475022 / 1 |
| detail:expert-gate | nested | 2.707247 / 16 | 2.706407 / 16 | 2.634545 / 16 |
| detail:expert-up | nested | 2.734957 / 16 | 2.713348 / 16 | 2.587415 / 16 |
| detail:expert-activation | nested | 0.089459 / 16 | 0.090901 / 16 | 0.090374 / 16 |
| detail:expert-down | nested | 2.920066 / 16 | 2.908581 / 16 | 2.782660 / 16 |
| EUP | nested | 0.629216 / 1 | 0.607425 / 1 | 0.599630 / 1 |
| experts-mix-normalize-up | boundary | 9.476774 / 1 | 9.373762 / 1 | 9.024250 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002996 / 1 | 0.002745 / 1 | 0.002905 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000171 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006111 / 1 | 0.005591 / 1 | 0.004619 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.579031 / 1 | 114.574186 / 1 | 109.967047 / 1 |
| op:Q   int8 projection | nested | 10.420960 / 1 | 9.563066 / 1 | 10.153738 / 1 |
| op:X   mxfp4 expert proj | nested | 8.488450 / 1 | 8.455168 / 1 | 8.131922 / 1 |
| op:N   rmsnorm | nested | 0.024284 / 1 | 0.022994 / 1 | 0.024707 / 1 |
| op:SiTU + sigma | nested | 0.097834 / 1 | 0.098775 / 1 | 0.100745 / 1 |
| op:AR  snapshot aggregate | nested | 0.030357 / 1 | 0.030647 / 1 | 0.030468 / 1 |
| op:SA  softmax attention | nested | 0.006632 / 1 | 0.009027 / 1 | 0.011612 / 1 |
| op:router dot product | nested | 0.621622 / 1 | 0.597496 / 1 | 0.597466 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002415 / 1 | 0.002124 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.171084 / 1 | 19.139898 / 1 | 19.379075 / 1 |

### Layer 32

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011512 / 1 | 0.003817 / 1 | 0.003917 / 1 |
| pre-attention-aggregation | boundary | 0.015098 / 1 | 0.014507 / 1 | 0.014818 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010870 / 1 | 0.010710 / 1 | 0.011161 / 1 |
| Q | nested | 2.472790 / 1 | 2.453303 / 1 | 2.428677 / 1 |
| K | nested | 2.322258 / 1 | 2.215108 / 1 | 2.157260 / 1 |
| V | nested | 2.171917 / 1 | 2.135049 / 1 | 2.107587 / 1 |
| B | nested | 0.086351 / 1 | 0.026980 / 1 | 0.028924 / 1 |
| FA | nested | 0.058379 / 1 | 0.034905 / 1 | 0.035777 / 1 |
| FB | nested | 0.071313 / 1 | 0.047368 / 1 | 0.051366 / 1 |
| G | nested | 2.181826 / 1 | 2.038078 / 1 | 2.026866 / 1 |
| O | nested | 2.204619 / 1 | 2.082561 / 1 | 2.092680 / 1 |
| attention | boundary | 12.674128 / 1 | 12.094916 / 1 | 11.597958 / 1 |
| attention-residual | boundary | 0.003076 / 1 | 0.003527 / 1 | 0.003857 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026139 / 1 | 0.026470 / 1 | 0.025007 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009277 / 1 | 0.002384 / 1 | 0.001873 / 1 |
| router-and-top16 | boundary | 0.614769 / 1 | 0.562431 / 1 | 0.561039 / 1 |
| EDOWN | nested | 0.649103 / 1 | 0.580485 / 1 | 0.570536 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.649904 / 1 | 0.581376 / 1 | 0.571337 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.065322 / 1 | 0.027983 / 1 | 0.005961 / 1 |
| SH1 | nested | 1.148556 / 1 | 1.027559 / 1 | 1.002443 / 1 |
| SH3 | nested | 1.142785 / 1 | 1.066913 / 1 | 1.052046 / 1 |
| SH2 | nested | 1.074768 / 1 | 0.993556 / 1 | 0.991412 / 1 |
| shared-expert-during-read | boundary | 3.377951 / 1 | 3.099751 / 1 | 3.058204 / 1 |
| detail:expert-gate | nested | 2.738186 / 16 | 2.719842 / 16 | 2.578187 / 16 |
| detail:expert-up | nested | 2.729607 / 16 | 2.694282 / 16 | 2.578878 / 16 |
| detail:expert-activation | nested | 0.090149 / 16 | 0.091351 / 16 | 0.091753 / 16 |
| detail:expert-down | nested | 2.908143 / 16 | 2.851925 / 16 | 2.714251 / 16 |
| EUP | nested | 0.633835 / 1 | 0.597576 / 1 | 0.612064 / 1 |
| experts-mix-normalize-up | boundary | 9.499297 / 1 | 9.307328 / 1 | 8.879238 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002655 / 1 | 0.003316 / 1 | 0.003156 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000270 / 1 | 0.000191 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.420255 / 1 | 0.388937 / 1 | 0.371133 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.248734 / 1 | 113.438602 / 1 | 107.808120 / 1 |
| op:Q   int8 projection | nested | 16.216987 / 1 | 15.297810 / 1 | 15.155745 / 1 |
| op:X   mxfp4 expert proj | nested | 8.503767 / 1 | 8.395763 / 1 | 8.001138 / 1 |
| op:N   rmsnorm | nested | 0.037049 / 1 | 0.032592 / 1 | 0.022994 / 1 |
| op:L   l2 per-head | nested | 0.005541 / 1 | 0.005360 / 1 | 0.005700 / 1 |
| op:SiTU + sigma | nested | 0.098744 / 1 | 0.099667 / 1 | 0.100800 / 1 |
| op:C   shortconv | nested | 0.067145 / 1 | 0.070962 / 1 | 0.033543 / 1 |
| op:AR  snapshot aggregate | nested | 0.030898 / 1 | 0.030156 / 1 | 0.030086 / 1 |
| op:D   kda delta-rule | nested | 0.151633 / 1 | 0.135884 / 1 | 0.108403 / 1 |
| op:router dot product | nested | 0.611011 / 1 | 0.559265 / 1 | 0.558353 / 1 |
| op:top-k selection | nested | 0.002785 / 1 | 0.002665 / 1 | 0.002334 / 1 |
| op:alpha / beta / gate | nested | 0.073888 / 1 | 0.073928 / 1 | 0.058369 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.389100 / 1 | 26.136590 / 1 | 25.117315 / 1 |

### Layer 33

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013555 / 1 | 0.006382 / 1 | 0.006001 / 1 |
| pre-attention-aggregation | boundary | 0.017353 / 1 | 0.016761 / 1 | 0.018454 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012212 / 1 | 0.011422 / 1 | 0.012313 / 1 |
| Q | nested | 2.489360 / 1 | 2.439487 / 1 | 2.420382 / 1 |
| K | nested | 2.329001 / 1 | 2.218905 / 1 | 2.282474 / 1 |
| V | nested | 2.178990 / 1 | 2.141721 / 1 | 2.125220 / 1 |
| B | nested | 0.093134 / 1 | 0.022071 / 1 | 0.020899 / 1 |
| FA | nested | 0.056365 / 1 | 0.030106 / 1 | 0.029525 / 1 |
| FB | nested | 0.078987 / 1 | 0.051326 / 1 | 0.051666 / 1 |
| G | nested | 2.173320 / 1 | 1.947839 / 1 | 1.947439 / 1 |
| O | nested | 2.231669 / 1 | 2.069837 / 1 | 2.053045 / 1 |
| attention | boundary | 12.743268 / 1 | 12.035586 / 1 | 11.638604 / 1 |
| attention-residual | boundary | 0.002555 / 1 | 0.003466 / 1 | 0.003877 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026360 / 1 | 0.025979 / 1 | 0.024496 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010048 / 1 | 0.002044 / 1 | 0.002334 / 1 |
| router-and-top16 | boundary | 0.623515 / 1 | 0.582488 / 1 | 0.576778 / 1 |
| EDOWN | nested | 0.667808 / 1 | 0.601504 / 1 | 0.590633 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.668670 / 1 | 0.602626 / 1 | 0.591455 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.063739 / 1 | 0.012403 / 1 | 0.011892 / 1 |
| SH1 | nested | 1.117658 / 1 | 1.014706 / 1 | 0.999167 / 1 |
| SH3 | nested | 1.108781 / 1 | 1.024174 / 1 | 1.022080 / 1 |
| SH2 | nested | 1.092160 / 1 | 0.997433 / 1 | 0.996632 / 1 |
| shared-expert-during-read | boundary | 3.330382 / 1 | 3.048175 / 1 | 3.029690 / 1 |
| detail:expert-gate | nested | 2.729371 / 16 | 2.699530 / 16 | 2.715291 / 16 |
| detail:expert-up | nested | 2.740249 / 16 | 2.703011 / 16 | 2.661514 / 16 |
| detail:expert-activation | nested | 0.090800 / 16 | 0.091683 / 16 | 0.090550 / 16 |
| detail:expert-down | nested | 2.911380 / 16 | 2.854473 / 16 | 2.833376 / 16 |
| EUP | nested | 0.636840 / 1 | 0.601273 / 1 | 0.602696 / 1 |
| experts-mix-normalize-up | boundary | 9.503595 / 1 | 9.276711 / 1 | 9.227660 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002404 / 1 | 0.003296 / 1 | 0.002705 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000170 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.401781 / 1 | 0.381963 / 1 | 0.358940 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.483014 / 1 | 113.470647 / 1 | 111.755119 / 1 |
| op:Q   int8 projection | nested | 16.252392 / 1 | 15.158741 / 1 | 15.140294 / 1 |
| op:X   mxfp4 expert proj | nested | 8.510489 / 1 | 8.387716 / 1 | 8.339600 / 1 |
| op:N   rmsnorm | nested | 0.041277 / 1 | 0.033341 / 1 | 0.024006 / 1 |
| op:L   l2 per-head | nested | 0.005491 / 1 | 0.005751 / 1 | 0.006062 / 1 |
| op:SiTU + sigma | nested | 0.099437 / 1 | 0.099956 / 1 | 0.099077 / 1 |
| op:C   shortconv | nested | 0.054212 / 1 | 0.080802 / 1 | 0.031630 / 1 |
| op:AR  snapshot aggregate | nested | 0.033644 / 1 | 0.032511 / 1 | 0.032961 / 1 |
| op:D   kda delta-rule | nested | 0.153116 / 1 | 0.149630 / 1 | 0.119704 / 1 |
| op:router dot product | nested | 0.620058 / 1 | 0.579443 / 1 | 0.573893 / 1 |
| op:top-k selection | nested | 0.002886 / 1 | 0.002305 / 1 | 0.002494 / 1 |
| op:alpha / beta / gate | nested | 0.074049 / 1 | 0.073207 / 1 | 0.058830 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.430437 / 1 | 26.018208 / 1 | 25.520989 / 1 |

### Layer 34

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014127 / 1 | 0.005870 / 1 | 0.005410 / 1 |
| pre-attention-aggregation | boundary | 0.017272 / 1 | 0.015930 / 1 | 0.015940 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011982 / 1 | 0.010921 / 1 | 0.011972 / 1 |
| Q | nested | 2.656032 / 1 | 2.641244 / 1 | 2.638890 / 1 |
| K | nested | 2.444938 / 1 | 2.265813 / 1 | 2.177999 / 1 |
| V | nested | 2.227492 / 1 | 2.116975 / 1 | 2.113799 / 1 |
| B | nested | 0.084478 / 1 | 0.021400 / 1 | 0.023234 / 1 |
| FA | nested | 0.056024 / 1 | 0.034465 / 1 | 0.032330 / 1 |
| FB | nested | 0.067677 / 1 | 0.052588 / 1 | 0.055744 / 1 |
| G | nested | 2.179371 / 1 | 1.943421 / 1 | 1.943330 / 1 |
| O | nested | 2.294006 / 1 | 2.192857 / 1 | 2.180143 / 1 |
| attention | boundary | 13.084946 / 1 | 12.319877 / 1 | 11.847895 / 1 |
| attention-residual | boundary | 0.003477 / 1 | 0.003266 / 1 | 0.003667 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026429 / 1 | 0.027291 / 1 | 0.025487 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009859 / 1 | 0.002495 / 1 | 0.002375 / 1 |
| router-and-top16 | boundary | 0.619858 / 1 | 0.578922 / 1 | 0.580344 / 1 |
| EDOWN | nested | 0.655906 / 1 | 0.574132 / 1 | 0.559345 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.656827 / 1 | 0.575064 / 1 | 0.560296 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032531 / 1 | 0.006272 / 1 | 0.012894 / 1 |
| SH1 | nested | 1.120754 / 1 | 1.035845 / 1 | 1.015137 / 1 |
| SH3 | nested | 1.112689 / 1 | 1.043960 / 1 | 1.030716 / 1 |
| SH2 | nested | 1.082542 / 1 | 0.994268 / 1 | 0.975152 / 1 |
| shared-expert-during-read | boundary | 3.327897 / 1 | 3.086506 / 1 | 3.032917 / 1 |
| detail:expert-gate | nested | 2.722198 / 16 | 2.694534 / 16 | 2.621972 / 16 |
| detail:expert-up | nested | 2.744083 / 16 | 2.686860 / 16 | 2.622888 / 16 |
| detail:expert-activation | nested | 0.096412 / 16 | 0.090919 / 16 | 0.091419 / 16 |
| detail:expert-down | nested | 2.880910 / 16 | 2.862301 / 16 | 2.800881 / 16 |
| EUP | nested | 0.628644 / 1 | 0.601614 / 1 | 0.583700 / 1 |
| experts-mix-normalize-up | boundary | 9.438173 / 1 | 9.278575 / 1 | 9.055638 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002735 / 1 | 0.002615 / 1 | 0.002665 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000371 / 1 | 0.000251 / 1 | 0.000251 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.432528 / 1 | 0.406689 / 1 | 0.380641 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.643009 / 1 | 114.430493 / 1 | 111.670558 / 1 |
| op:Q   int8 projection | nested | 16.608818 / 1 | 15.516469 / 1 | 15.328035 / 1 |
| op:X   mxfp4 expert proj | nested | 8.483391 / 1 | 8.373323 / 1 | 8.176788 / 1 |
| op:N   rmsnorm | nested | 0.035859 / 1 | 0.034485 / 1 | 0.023414 / 1 |
| op:L   l2 per-head | nested | 0.006111 / 1 | 0.005239 / 1 | 0.004980 / 1 |
| op:SiTU + sigma | nested | 0.104788 / 1 | 0.099374 / 1 | 0.099444 / 1 |
| op:C   shortconv | nested | 0.063738 / 1 | 0.064821 / 1 | 0.031198 / 1 |
| op:AR  snapshot aggregate | nested | 0.033152 / 1 | 0.032351 / 1 | 0.030888 / 1 |
| op:D   kda delta-rule | nested | 0.158106 / 1 | 0.148909 / 1 | 0.123851 / 1 |
| op:router dot product | nested | 0.616663 / 1 | 0.576077 / 1 | 0.577018 / 1 |
| op:top-k selection | nested | 0.002534 / 1 | 0.002424 / 1 | 0.002675 / 1 |
| op:alpha / beta / gate | nested | 0.072766 / 1 | 0.073237 / 1 | 0.066013 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.687416 / 1 | 26.328708 / 1 | 25.546627 / 1 |

### Layer 35

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012744 / 1 | 0.004799 / 1 | 0.005360 / 1 |
| pre-attention-aggregation | boundary | 0.018214 / 1 | 0.016871 / 1 | 0.016040 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012684 / 1 | 0.010330 / 1 | 0.011311 / 1 |
| QA | nested | 0.305521 / 1 | 0.299078 / 1 | 0.289380 / 1 |
| QB | nested | 0.731547 / 1 | 0.682084 / 1 | 0.677986 / 1 |
| KA | nested | 0.127077 / 1 | 0.101470 / 1 | 0.101840 / 1 |
| KB | nested | 0.336349 / 1 | 0.331920 / 1 | 0.300652 / 1 |
| G | nested | 2.183850 / 1 | 2.011609 / 1 | 2.021888 / 1 |
| O | nested | 2.161027 / 1 | 1.935927 / 1 | 1.946687 / 1 |
| attention | boundary | 5.948292 / 1 | 5.460210 / 1 | 5.437839 / 1 |
| attention-residual | boundary | 0.002826 / 1 | 0.002475 / 1 | 0.003206 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025207 / 1 | 0.026259 / 1 | 0.025147 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009578 / 1 | 0.002185 / 1 | 0.002124 / 1 |
| router-and-top16 | boundary | 0.615770 / 1 | 0.587086 / 1 | 0.585434 / 1 |
| EDOWN | nested | 0.635798 / 1 | 0.600281 / 1 | 0.596274 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.636620 / 1 | 0.601203 / 1 | 0.597166 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.083686 / 1 | 0.012052 / 1 | 0.011832 / 1 |
| SH1 | nested | 1.095707 / 1 | 0.991272 / 1 | 0.992604 / 1 |
| SH3 | nested | 1.098682 / 1 | 0.998626 / 1 | 1.008314 / 1 |
| SH2 | nested | 1.078495 / 1 | 0.981274 / 1 | 0.977376 / 1 |
| shared-expert-during-read | boundary | 3.284617 / 1 | 2.982893 / 1 | 2.990387 / 1 |
| detail:expert-gate | nested | 2.746890 / 16 | 2.735680 / 16 | 2.725663 / 16 |
| detail:expert-up | nested | 2.747631 / 16 | 2.727785 / 16 | 2.689124 / 16 |
| detail:expert-activation | nested | 0.091042 / 16 | 0.098112 / 16 | 0.094045 / 16 |
| detail:expert-down | nested | 2.964438 / 16 | 2.942819 / 16 | 2.861174 / 16 |
| EUP | nested | 0.635137 / 1 | 0.598578 / 1 | 0.598378 / 1 |
| experts-mix-normalize-up | boundary | 9.599413 / 1 | 9.431530 / 1 | 9.272894 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002325 / 1 | 0.004097 / 1 | 0.002805 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000170 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005941 / 1 | 0.006142 / 1 | 0.004589 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.573311 / 1 | 115.613629 / 1 | 113.659614 / 1 |
| op:Q   int8 projection | nested | 10.387870 / 1 | 9.530453 / 1 | 9.509987 / 1 |
| op:X   mxfp4 expert proj | nested | 8.589396 / 1 | 8.544523 / 1 | 8.410803 / 1 |
| op:N   rmsnorm | nested | 0.025826 / 1 | 0.023944 / 1 | 0.024296 / 1 |
| op:SiTU + sigma | nested | 0.099668 / 1 | 0.102281 / 1 | 0.102632 / 1 |
| op:AR  snapshot aggregate | nested | 0.032992 / 1 | 0.031369 / 1 | 0.030568 / 1 |
| op:SA  softmax attention | nested | 0.006712 / 1 | 0.008506 / 1 | 0.011873 / 1 |
| op:router dot product | nested | 0.613186 / 1 | 0.584522 / 1 | 0.582889 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002084 / 1 | 0.002184 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.266152 / 1 | 19.157070 / 1 | 18.974439 / 1 |

### Layer 36

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011422 / 1 | 0.003887 / 1 | 0.003687 / 1 |
| pre-attention-aggregation | boundary | 0.015609 / 1 | 0.014988 / 1 | 0.014207 / 1 |
| snapshot-push | boundary | 0.001212 / 1 | 0.001122 / 1 | 0.001352 / 1 |
| pre-attention-normalization | boundary | 0.010981 / 1 | 0.010640 / 1 | 0.014748 / 1 |
| Q | nested | 2.468592 / 1 | 2.452753 / 1 | 2.446059 / 1 |
| K | nested | 2.344631 / 1 | 2.223344 / 1 | 2.160486 / 1 |
| V | nested | 2.187146 / 1 | 2.109161 / 1 | 2.090796 / 1 |
| B | nested | 0.092092 / 1 | 0.021360 / 1 | 0.020658 / 1 |
| FA | nested | 0.052508 / 1 | 0.032611 / 1 | 0.034324 / 1 |
| FB | nested | 0.070902 / 1 | 0.056756 / 1 | 0.057016 / 1 |
| G | nested | 2.176496 / 1 | 2.035513 / 1 | 2.045311 / 1 |
| O | nested | 2.189801 / 1 | 2.063304 / 1 | 2.021948 / 1 |
| attention | boundary | 12.697752 / 1 | 12.046886 / 1 | 11.526063 / 1 |
| attention-residual | boundary | 0.002104 / 1 | 0.002605 / 1 | 0.002395 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027031 / 1 | 0.026269 / 1 | 0.025307 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009718 / 1 | 0.002445 / 1 | 0.002615 / 1 |
| router-and-top16 | boundary | 0.605551 / 1 | 0.566318 / 1 | 0.564875 / 1 |
| EDOWN | nested | 0.658099 / 1 | 0.584322 / 1 | 0.579813 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.659062 / 1 | 0.585204 / 1 | 0.580685 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.062897 / 1 | 0.012443 / 1 | 0.006142 / 1 |
| SH1 | nested | 1.123368 / 1 | 0.993626 / 1 | 0.985762 / 1 |
| SH3 | nested | 1.098482 / 1 | 1.011800 / 1 | 1.002543 / 1 |
| SH2 | nested | 1.105365 / 1 | 1.055562 / 1 | 1.031007 / 1 |
| shared-expert-during-read | boundary | 3.339599 / 1 | 3.072801 / 1 | 3.031173 / 1 |
| detail:expert-gate | nested | 2.727084 / 16 | 2.699061 / 16 | 2.630993 / 16 |
| detail:expert-up | nested | 2.728468 / 16 | 2.693452 / 16 | 2.607923 / 16 |
| detail:expert-activation | nested | 0.090736 / 16 | 0.090583 / 16 | 0.089848 / 16 |
| detail:expert-down | nested | 2.931999 / 16 | 2.854482 / 16 | 2.810263 / 16 |
| EUP | nested | 0.632923 / 1 | 0.595933 / 1 | 0.589371 / 1 |
| experts-mix-normalize-up | boundary | 9.506179 / 1 | 9.276290 / 1 | 9.051881 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002595 / 1 | 0.003116 / 1 | 0.002935 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000200 / 1 | 0.000160 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.438780 / 1 | 0.432428 / 1 | 0.391752 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.173156 / 1 | 112.705523 / 1 | 110.338286 / 1 |
| op:Q   int8 projection | nested | 16.198793 / 1 | 15.234484 / 1 | 15.062971 / 1 |
| op:X   mxfp4 expert proj | nested | 8.519658 / 1 | 8.378783 / 1 | 8.180430 / 1 |
| op:N   rmsnorm | nested | 0.036758 / 1 | 0.032682 / 1 | 0.026811 / 1 |
| op:L   l2 per-head | nested | 0.005851 / 1 | 0.005821 / 1 | 0.006412 / 1 |
| op:SiTU + sigma | nested | 0.099023 / 1 | 0.098144 / 1 | 0.097952 / 1 |
| op:C   shortconv | nested | 0.073948 / 1 | 0.076182 / 1 | 0.032099 / 1 |
| op:AR  snapshot aggregate | nested | 0.032041 / 1 | 0.031149 / 1 | 0.029696 / 1 |
| op:D   kda delta-rule | nested | 0.148067 / 1 | 0.131015 / 1 | 0.100889 / 1 |
| op:router dot product | nested | 0.602295 / 1 | 0.563462 / 1 | 0.562130 / 1 |
| op:top-k selection | nested | 0.002795 / 1 | 0.002164 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.072956 / 1 | 0.073387 / 1 | 0.058229 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.399228 / 1 | 26.066117 / 1 | 25.228393 / 1 |

### Layer 37

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012574 / 1 | 0.004980 / 1 | 0.004478 / 1 |
| pre-attention-aggregation | boundary | 0.021009 / 1 | 0.017132 / 1 | 0.016761 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010650 / 1 | 0.011261 / 1 | 0.011421 / 1 |
| Q | nested | 2.464364 / 1 | 2.458133 / 1 | 2.450368 / 1 |
| K | nested | 2.318712 / 1 | 2.191595 / 1 | 2.172860 / 1 |
| V | nested | 2.186404 / 1 | 2.097669 / 1 | 2.087360 / 1 |
| B | nested | 0.071834 / 1 | 0.024225 / 1 | 0.021230 / 1 |
| FA | nested | 0.056616 / 1 | 0.034856 / 1 | 0.035436 / 1 |
| FB | nested | 0.075451 / 1 | 0.048200 / 1 | 0.049673 / 1 |
| G | nested | 2.153934 / 1 | 1.973847 / 1 | 1.951276 / 1 |
| O | nested | 2.327428 / 1 | 2.152682 / 1 | 2.155878 / 1 |
| attention | boundary | 12.726686 / 1 | 12.009226 / 1 | 11.587789 / 1 |
| attention-residual | boundary | 0.003196 / 1 | 0.003547 / 1 | 0.003476 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026139 / 1 | 0.025467 / 1 | 0.025318 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010329 / 1 | 0.002154 / 1 | 0.002384 / 1 |
| router-and-top16 | boundary | 0.607104 / 1 | 0.573382 / 1 | 0.567911 / 1 |
| EDOWN | nested | 0.667456 / 1 | 0.607696 / 1 | 0.599340 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.668409 / 1 | 0.608657 / 1 | 0.600331 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059451 / 1 | 0.028333 / 1 | 0.006291 / 1 |
| SH1 | nested | 1.134429 / 1 | 0.997644 / 1 | 0.991743 / 1 |
| SH3 | nested | 1.119000 / 1 | 1.004677 / 1 | 0.991292 / 1 |
| SH2 | nested | 1.100435 / 1 | 1.003716 / 1 | 1.004005 / 1 |
| shared-expert-during-read | boundary | 3.365999 / 1 | 3.018389 / 1 | 2.999023 / 1 |
| detail:expert-gate | nested | 2.726603 / 16 | 2.649622 / 16 | 2.636615 / 16 |
| detail:expert-up | nested | 2.723588 / 16 | 2.654520 / 16 | 2.626095 / 16 |
| detail:expert-activation | nested | 0.091190 / 16 | 0.089759 / 16 | 0.089039 / 16 |
| detail:expert-down | nested | 2.915095 / 16 | 2.776405 / 16 | 2.834503 / 16 |
| EUP | nested | 0.627753 / 1 | 0.600362 / 1 | 0.597646 / 1 |
| experts-mix-normalize-up | boundary | 9.473779 / 1 | 9.133935 / 1 | 9.119227 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002956 / 1 | 0.002975 / 1 | 0.002866 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.403163 / 1 | 0.375191 / 1 | 0.353450 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.320940 / 1 | 111.345900 / 1 | 111.429727 / 1 |
| op:Q   int8 projection | nested | 16.302075 / 1 | 15.193436 / 1 | 15.106383 / 1 |
| op:X   mxfp4 expert proj | nested | 8.498537 / 1 | 8.211986 / 1 | 8.228283 / 1 |
| op:N   rmsnorm | nested | 0.033843 / 1 | 0.033173 / 1 | 0.023124 / 1 |
| op:L   l2 per-head | nested | 0.005981 / 1 | 0.005861 / 1 | 0.005430 / 1 |
| op:SiTU + sigma | nested | 0.099638 / 1 | 0.098605 / 1 | 0.097143 / 1 |
| op:C   shortconv | nested | 0.058820 / 1 | 0.057087 / 1 | 0.031519 / 1 |
| op:AR  snapshot aggregate | nested | 0.037280 / 1 | 0.032973 / 1 | 0.032210 / 1 |
| op:D   kda delta-rule | nested | 0.134251 / 1 | 0.115125 / 1 | 0.096099 / 1 |
| op:router dot product | nested | 0.603909 / 1 | 0.570485 / 1 | 0.565045 / 1 |
| op:top-k selection | nested | 0.002585 / 1 | 0.002284 / 1 | 0.002485 / 1 |
| op:alpha / beta / gate | nested | 0.074008 / 1 | 0.075652 / 1 | 0.059070 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.400911 / 1 | 25.824726 / 1 | 25.310977 / 1 |

### Layer 38

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013786 / 1 | 0.006012 / 1 | 0.004989 / 1 |
| pre-attention-aggregation | boundary | 0.019236 / 1 | 0.017613 / 1 | 0.016330 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011912 / 1 | 0.011822 / 1 | 0.011301 / 1 |
| Q | nested | 2.445509 / 1 | 2.434278 / 1 | 2.417627 / 1 |
| K | nested | 2.336084 / 1 | 2.191695 / 1 | 2.155637 / 1 |
| V | nested | 2.207434 / 1 | 2.101786 / 1 | 2.104762 / 1 |
| B | nested | 0.075722 / 1 | 0.021319 / 1 | 0.021140 / 1 |
| FA | nested | 0.059762 / 1 | 0.032901 / 1 | 0.031749 / 1 |
| FB | nested | 0.074870 / 1 | 0.050374 / 1 | 0.048059 / 1 |
| G | nested | 2.170926 / 1 | 1.961043 / 1 | 1.965171 / 1 |
| O | nested | 2.190071 / 1 | 2.049138 / 1 | 2.042445 / 1 |
| attention | boundary | 12.649502 / 1 | 11.909159 / 1 | 11.507859 / 1 |
| attention-residual | boundary | 0.002855 / 1 | 0.003487 / 1 | 0.003567 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026089 / 1 | 0.025968 / 1 | 0.027692 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010009 / 1 | 0.002174 / 1 | 0.002194 / 1 |
| router-and-top16 | boundary | 0.603668 / 1 | 0.576967 / 1 | 0.569384 / 1 |
| EDOWN | nested | 0.643893 / 1 | 0.580464 / 1 | 0.623846 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.644935 / 1 | 0.581416 / 1 | 0.624727 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043441 / 1 | 0.012473 / 1 | 0.014017 / 1 |
| SH1 | nested | 1.134490 / 1 | 0.995890 / 1 | 1.003154 / 1 |
| SH3 | nested | 1.122867 / 1 | 1.068947 / 1 | 1.046314 / 1 |
| SH2 | nested | 1.087662 / 1 | 1.012000 / 1 | 0.990771 / 1 |
| shared-expert-during-read | boundary | 3.357032 / 1 | 3.088731 / 1 | 3.052332 / 1 |
| detail:expert-gate | nested | 2.758304 / 16 | 2.702018 / 16 | 2.562540 / 16 |
| detail:expert-up | nested | 2.718018 / 16 | 2.687961 / 16 | 2.558132 / 16 |
| detail:expert-activation | nested | 0.090459 / 16 | 0.090480 / 16 | 0.089800 / 16 |
| detail:expert-down | nested | 2.882636 / 16 | 2.878888 / 16 | 2.708579 / 16 |
| EUP | nested | 0.637121 / 1 | 0.607896 / 1 | 0.590192 / 1 |
| experts-mix-normalize-up | boundary | 9.472797 / 1 | 9.312257 / 1 | 8.833734 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002475 / 1 | 0.002925 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000230 / 1 | 0.000240 / 1 | 0.000441 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.374179 / 1 | 0.365583 / 1 | 0.344404 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.428325 / 1 | 113.427058 / 1 | 107.258356 / 1 |
| op:Q   int8 projection | nested | 16.184576 / 1 | 15.106029 / 1 | 15.039026 / 1 |
| op:X   mxfp4 expert proj | nested | 8.494560 / 1 | 8.402356 / 1 | 7.961877 / 1 |
| op:N   rmsnorm | nested | 0.032791 / 1 | 0.035626 / 1 | 0.024526 / 1 |
| op:L   l2 per-head | nested | 0.005130 / 1 | 0.005891 / 1 | 0.005370 / 1 |
| op:SiTU + sigma | nested | 0.098856 / 1 | 0.098815 / 1 | 0.098316 / 1 |
| op:C   shortconv | nested | 0.066745 / 1 | 0.057257 / 1 | 0.030907 / 1 |
| op:AR  snapshot aggregate | nested | 0.035226 / 1 | 0.032621 / 1 | 0.033012 / 1 |
| op:D   kda delta-rule | nested | 0.161040 / 1 | 0.145382 / 1 | 0.124513 / 1 |
| op:router dot product | nested | 0.601113 / 1 | 0.574122 / 1 | 0.567100 / 1 |
| op:top-k selection | nested | 0.002144 / 1 | 0.002354 / 1 | 0.001944 / 1 |
| op:alpha / beta / gate | nested | 0.073177 / 1 | 0.073507 / 1 | 0.058730 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.241594 / 1 | 25.926317 / 1 | 25.025142 / 1 |

### Layer 39

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013876 / 1 | 0.005921 / 1 | 0.005550 / 1 |
| pre-attention-aggregation | boundary | 0.026630 / 1 | 0.018054 / 1 | 0.017383 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012554 / 1 | 0.010950 / 1 | 0.011010 / 1 |
| QA | nested | 0.342389 / 1 | 0.342289 / 1 | 0.342130 / 1 |
| QB | nested | 0.737688 / 1 | 0.697904 / 1 | 0.684158 / 1 |
| KA | nested | 0.124342 / 1 | 0.106679 / 1 | 0.098855 / 1 |
| KB | nested | 0.344674 / 1 | 0.300992 / 1 | 0.317023 / 1 |
| G | nested | 2.217422 / 1 | 2.039541 / 1 | 2.030042 / 1 |
| O | nested | 2.185293 / 1 | 2.047495 / 1 | 2.034952 / 1 |
| attention | boundary | 6.052898 / 1 | 5.633094 / 1 | 5.604290 / 1 |
| attention-residual | boundary | 0.002855 / 1 | 0.002655 / 1 | 0.003116 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025698 / 1 | 0.024866 / 1 | 0.024806 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009818 / 1 | 0.002395 / 1 | 0.002425 / 1 |
| router-and-top16 | boundary | 0.588650 / 1 | 0.566939 / 1 | 0.570425 / 1 |
| EDOWN | nested | 0.635417 / 1 | 0.590052 / 1 | 0.596755 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.636298 / 1 | 0.590934 / 1 | 0.597656 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.063319 / 1 | 0.022442 / 1 | 0.006182 / 1 |
| SH1 | nested | 1.099073 / 1 | 0.983968 / 1 | 0.984770 / 1 |
| SH3 | nested | 1.094895 / 1 | 0.998546 / 1 | 1.001191 / 1 |
| SH2 | nested | 1.078315 / 1 | 0.994728 / 1 | 0.991432 / 1 |
| shared-expert-during-read | boundary | 3.284075 / 1 | 2.988924 / 1 | 2.989285 / 1 |
| detail:expert-gate | nested | 2.724201 / 16 | 2.695707 / 16 | 2.452159 / 16 |
| detail:expert-up | nested | 2.733677 / 16 | 2.696648 / 16 | 2.435160 / 16 |
| detail:expert-activation | nested | 0.096020 / 16 | 0.089948 / 16 | 0.090466 / 16 |
| detail:expert-down | nested | 2.901060 / 16 | 2.908996 / 16 | 2.555796 / 16 |
| EUP | nested | 0.625368 / 1 | 0.596414 / 1 | 0.593769 / 1 |
| experts-mix-normalize-up | boundary | 9.470283 / 1 | 9.333828 / 1 | 8.450167 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002826 / 1 | 0.003316 / 1 | 0.002885 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000151 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006482 / 1 | 0.005461 / 1 | 0.005190 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.426335 / 1 | 114.547256 / 1 | 102.106663 / 1 |
| op:Q   int8 projection | nested | 10.483576 / 1 | 9.697116 / 1 | 9.673773 / 1 |
| op:X   mxfp4 expert proj | nested | 8.498249 / 1 | 8.435398 / 1 | 7.578338 / 1 |
| op:N   rmsnorm | nested | 0.025529 / 1 | 0.023916 / 1 | 0.023033 / 1 |
| op:SiTU + sigma | nested | 0.104476 / 1 | 0.098175 / 1 | 0.098914 / 1 |
| op:AR  snapshot aggregate | nested | 0.041487 / 1 | 0.032530 / 1 | 0.031760 / 1 |
| op:SA  softmax attention | nested | 0.006963 / 1 | 0.008827 / 1 | 0.012033 / 1 |
| op:router dot product | nested | 0.585573 / 1 | 0.564013 / 1 | 0.567610 / 1 |
| op:top-k selection | nested | 0.002785 / 1 | 0.002524 / 1 | 0.002254 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.205689 / 1 | 19.219216 / 1 | 18.299838 / 1 |

### Layer 40

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011602 / 1 | 0.003797 / 1 | 0.003446 / 1 |
| pre-attention-aggregation | boundary | 0.015690 / 1 | 0.014828 / 1 | 0.015088 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010680 / 1 | 0.010811 / 1 | 0.012814 / 1 |
| Q | nested | 2.753243 / 1 | 2.668385 / 1 | 2.641024 / 1 |
| K | nested | 2.501664 / 1 | 2.235757 / 1 | 2.159694 / 1 |
| V | nested | 2.300067 / 1 | 2.096376 / 1 | 2.092369 / 1 |
| B | nested | 0.081813 / 1 | 0.027020 / 1 | 0.025337 / 1 |
| FA | nested | 0.059632 / 1 | 0.036218 / 1 | 0.037981 / 1 |
| FB | nested | 0.068749 / 1 | 0.052438 / 1 | 0.047830 / 1 |
| G | nested | 2.177648 / 1 | 1.987443 / 1 | 1.990599 / 1 |
| O | nested | 2.234505 / 1 | 2.217423 / 1 | 2.205721 / 1 |
| attention | boundary | 13.255003 / 1 | 12.365862 / 1 | 11.860408 / 1 |
| attention-residual | boundary | 0.003086 / 1 | 0.003146 / 1 | 0.002685 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027913 / 1 | 0.026329 / 1 | 0.025938 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009628 / 1 | 0.002194 / 1 | 0.002525 / 1 |
| router-and-top16 | boundary | 0.621731 / 1 | 0.576246 / 1 | 0.570526 / 1 |
| EDOWN | nested | 0.660524 / 1 | 0.579292 / 1 | 0.571257 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.661446 / 1 | 0.580234 / 1 | 0.572340 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.047649 / 1 | 0.023384 / 1 | 0.006732 / 1 |
| SH1 | nested | 1.134920 / 1 | 0.997954 / 1 | 0.984198 / 1 |
| SH3 | nested | 1.120253 / 1 | 1.055692 / 1 | 1.032259 / 1 |
| SH2 | nested | 1.106227 / 1 | 1.034353 / 1 | 1.033901 / 1 |
| shared-expert-during-read | boundary | 3.373522 / 1 | 3.100172 / 1 | 3.064715 / 1 |
| detail:expert-gate | nested | 2.713910 / 16 | 2.626027 / 16 | 2.576722 / 16 |
| detail:expert-up | nested | 2.715131 / 16 | 2.630223 / 16 | 2.575966 / 16 |
| detail:expert-activation | nested | 0.089999 / 16 | 0.090517 / 16 | 0.089809 / 16 |
| detail:expert-down | nested | 2.869280 / 16 | 2.830719 / 16 | 2.752022 / 16 |
| EUP | nested | 0.631650 / 1 | 0.592547 / 1 | 0.594911 / 1 |
| experts-mix-normalize-up | boundary | 9.409629 / 1 | 9.135788 / 1 | 8.915286 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002214 / 1 | 0.002645 / 1 | 0.002525 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000260 / 1 | 0.000150 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.422630 / 1 | 0.425024 / 1 | 0.365994 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.150043 / 1 | 112.487826 / 1 | 109.175158 / 1 |
| op:Q   int8 projection | nested | 16.829231 / 1 | 15.578786 / 1 | 15.415440 / 1 |
| op:X   mxfp4 expert proj | nested | 8.433322 / 1 | 8.222952 / 1 | 8.039600 / 1 |
| op:N   rmsnorm | nested | 0.035086 / 1 | 0.031869 / 1 | 0.025468 / 1 |
| op:L   l2 per-head | nested | 0.006241 / 1 | 0.005550 / 1 | 0.006111 / 1 |
| op:SiTU + sigma | nested | 0.098576 / 1 | 0.098854 / 1 | 0.100368 / 1 |
| op:C   shortconv | nested | 0.073888 / 1 | 0.071473 / 1 | 0.032992 / 1 |
| op:AR  snapshot aggregate | nested | 0.033432 / 1 | 0.031178 / 1 | 0.031068 / 1 |
| op:D   kda delta-rule | nested | 0.147406 / 1 | 0.141604 / 1 | 0.110697 / 1 |
| op:router dot product | nested | 0.618576 / 1 | 0.573301 / 1 | 0.567721 / 1 |
| op:top-k selection | nested | 0.002695 / 1 | 0.002575 / 1 | 0.002174 / 1 |
| op:alpha / beta / gate | nested | 0.074269 / 1 | 0.084197 / 1 | 0.058540 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.882100 / 1 | 26.279917 / 1 | 25.430249 / 1 |

### Layer 41

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012714 / 1 | 0.004699 / 1 | 0.004929 / 1 |
| pre-attention-aggregation | boundary | 0.016491 / 1 | 0.016721 / 1 | 0.017162 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010520 / 1 | 0.010821 / 1 | 0.010810 / 1 |
| Q | nested | 2.726905 / 1 | 2.640603 / 1 | 2.631917 / 1 |
| K | nested | 2.493869 / 1 | 2.220619 / 1 | 2.169273 / 1 |
| V | nested | 2.263639 / 1 | 2.091136 / 1 | 2.083072 / 1 |
| B | nested | 0.093064 / 1 | 0.021360 / 1 | 0.020960 / 1 |
| FA | nested | 0.061575 / 1 | 0.027982 / 1 | 0.027151 / 1 |
| FB | nested | 0.079388 / 1 | 0.050454 / 1 | 0.048841 / 1 |
| G | nested | 2.153333 / 1 | 1.987453 / 1 | 1.983005 / 1 |
| O | nested | 2.220839 / 1 | 2.171938 / 1 | 2.168431 / 1 |
| attention | boundary | 13.169283 / 1 | 12.299859 / 1 | 11.820784 / 1 |
| attention-residual | boundary | 0.003016 / 1 | 0.003737 / 1 | 0.003106 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025708 / 1 | 0.027251 / 1 | 0.025708 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009778 / 1 | 0.002084 / 1 | 0.001974 / 1 |
| router-and-top16 | boundary | 0.606343 / 1 | 0.579232 / 1 | 0.578962 / 1 |
| EDOWN | nested | 0.643973 / 1 | 0.580694 / 1 | 0.581577 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.645045 / 1 | 0.581727 / 1 | 0.582609 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041758 / 1 | 0.019657 / 1 | 0.012834 / 1 |
| SH1 | nested | 1.121105 / 1 | 1.006190 / 1 | 0.985671 / 1 |
| SH3 | nested | 1.101238 / 1 | 0.983978 / 1 | 0.985772 / 1 |
| SH2 | nested | 1.082131 / 1 | 1.000179 / 1 | 0.979911 / 1 |
| shared-expert-during-read | boundary | 3.316556 / 1 | 3.002399 / 1 | 2.963236 / 1 |
| detail:expert-gate | nested | 2.723901 / 16 | 2.683226 / 16 | 2.732577 / 16 |
| detail:expert-up | nested | 2.722316 / 16 | 2.640994 / 16 | 2.687061 / 16 |
| detail:expert-activation | nested | 0.092394 / 16 | 0.088928 / 16 | 0.096939 / 16 |
| detail:expert-down | nested | 2.924764 / 16 | 2.842027 / 16 | 2.739217 / 16 |
| EUP | nested | 0.639104 / 1 | 0.596825 / 1 | 0.595713 / 1 |
| experts-mix-normalize-up | boundary | 9.484870 / 1 | 9.206540 / 1 | 9.259178 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002675 / 1 | 0.002865 / 1 | 0.002525 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000141 / 1 | 0.000171 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.392794 / 1 | 0.384648 / 1 | 0.370723 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.492673 / 1 | 113.204106 / 1 | 109.909360 / 1 |
| op:Q   int8 projection | nested | 16.678670 / 1 | 15.377528 / 1 | 15.259479 / 1 |
| op:X   mxfp4 expert proj | nested | 8.509208 / 1 | 8.300600 / 1 | 8.301218 / 1 |
| op:N   rmsnorm | nested | 0.038291 / 1 | 0.038521 / 1 | 0.023945 / 1 |
| op:L   l2 per-head | nested | 0.006552 / 1 | 0.005440 / 1 | 0.005470 / 1 |
| op:SiTU + sigma | nested | 0.100556 / 1 | 0.096992 / 1 | 0.104804 / 1 |
| op:C   shortconv | nested | 0.067286 / 1 | 0.067987 / 1 | 0.033232 / 1 |
| op:AR  snapshot aggregate | nested | 0.032390 / 1 | 0.033273 / 1 | 0.032169 / 1 |
| op:D   kda delta-rule | nested | 0.133830 / 1 | 0.124623 / 1 | 0.106329 / 1 |
| op:router dot product | nested | 0.603207 / 1 | 0.576367 / 1 | 0.575906 / 1 |
| op:top-k selection | nested | 0.002555 / 1 | 0.002595 / 1 | 0.002484 / 1 |
| op:alpha / beta / gate | nested | 0.074519 / 1 | 0.073567 / 1 | 0.058800 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.747439 / 1 | 26.152780 / 1 | 25.664117 / 1 |

### Layer 42

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012834 / 1 | 0.004468 / 1 | 0.004799 / 1 |
| pre-attention-aggregation | boundary | 0.017693 / 1 | 0.016791 / 1 | 0.016671 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012634 / 1 | 0.011411 / 1 | 0.011131 / 1 |
| Q | nested | 2.513876 / 1 | 2.493729 / 1 | 2.478179 / 1 |
| K | nested | 2.330955 / 1 | 2.188608 / 1 | 2.182617 / 1 |
| V | nested | 2.202525 / 1 | 2.101546 / 1 | 2.092440 / 1 |
| B | nested | 0.092223 / 1 | 0.030176 / 1 | 0.029846 / 1 |
| FA | nested | 0.058529 / 1 | 0.035467 / 1 | 0.033703 / 1 |
| FB | nested | 0.073046 / 1 | 0.049873 / 1 | 0.049833 / 1 |
| G | nested | 2.152191 / 1 | 1.977495 / 1 | 1.968558 / 1 |
| O | nested | 2.249302 / 1 | 2.208646 / 1 | 2.204969 / 1 |
| attention | boundary | 12.696660 / 1 | 12.114242 / 1 | 11.721328 / 1 |
| attention-residual | boundary | 0.003317 / 1 | 0.003536 / 1 | 0.003686 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026499 / 1 | 0.025347 / 1 | 0.024626 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010339 / 1 | 0.001984 / 1 | 0.002154 / 1 |
| router-and-top16 | boundary | 0.598107 / 1 | 0.568993 / 1 | 0.569544 / 1 |
| EDOWN | nested | 0.656527 / 1 | 0.592998 / 1 | 0.590724 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.658410 / 1 | 0.593999 / 1 | 0.591795 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.070011 / 1 | 0.012403 / 1 | 0.007293 / 1 |
| SH1 | nested | 1.131584 / 1 | 1.000820 / 1 | 0.995290 / 1 |
| SH3 | nested | 1.133317 / 1 | 1.091038 / 1 | 1.072163 / 1 |
| SH2 | nested | 1.082993 / 1 | 1.016359 / 1 | 1.000068 / 1 |
| shared-expert-during-read | boundary | 3.360258 / 1 | 3.123896 / 1 | 3.080225 / 1 |
| detail:expert-gate | nested | 2.708647 / 16 | 2.483009 / 16 | 2.716338 / 16 |
| detail:expert-up | nested | 2.700635 / 16 | 2.477879 / 16 | 2.685768 / 16 |
| detail:expert-activation | nested | 0.092361 / 16 | 0.089837 / 16 | 0.090069 / 16 |
| detail:expert-down | nested | 2.890710 / 16 | 2.665240 / 16 | 2.873930 / 16 |
| EUP | nested | 0.635006 / 1 | 0.598348 / 1 | 0.612675 / 1 |
| experts-mix-normalize-up | boundary | 9.441379 / 1 | 8.677682 / 1 | 9.325963 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002615 / 1 | 0.002816 / 1 | 0.002756 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000231 / 1 | 0.000231 / 1 | 0.000210 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.430615 / 1 | 0.419423 / 1 | 0.392593 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.218796 / 1 | 104.491948 / 1 | 113.825122 / 1 |
| op:Q   int8 projection | nested | 16.310392 / 1 | 15.383421 / 1 | 15.308860 / 1 |
| op:X   mxfp4 expert proj | nested | 8.439285 / 1 | 7.762885 / 1 | 8.412539 / 1 |
| op:N   rmsnorm | nested | 0.033111 / 1 | 0.033493 / 1 | 0.023325 / 1 |
| op:L   l2 per-head | nested | 0.005721 / 1 | 0.005740 / 1 | 0.005961 / 1 |
| op:SiTU + sigma | nested | 0.100777 / 1 | 0.101268 / 1 | 0.098506 / 1 |
| op:C   shortconv | nested | 0.051185 / 1 | 0.051626 / 1 | 0.033624 / 1 |
| op:AR  snapshot aggregate | nested | 0.034054 / 1 | 0.032190 / 1 | 0.031499 / 1 |
| op:D   kda delta-rule | nested | 0.128089 / 1 | 0.121236 / 1 | 0.100548 / 1 |
| op:router dot product | nested | 0.595423 / 1 | 0.566207 / 1 | 0.566899 / 1 |
| op:top-k selection | nested | 0.002244 / 1 | 0.002394 / 1 | 0.002184 / 1 |
| op:alpha / beta / gate | nested | 0.072796 / 1 | 0.073137 / 1 | 0.059141 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.352541 / 1 | 25.587092 / 1 | 25.764724 / 1 |

### Layer 43

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012292 / 1 | 0.004609 / 1 | 0.004940 / 1 |
| pre-attention-aggregation | boundary | 0.018334 / 1 | 0.016360 / 1 | 0.017042 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.011101 / 1 | 0.011020 / 1 | 0.010880 / 1 |
| QA | nested | 0.302585 / 1 | 0.289200 / 1 | 0.295733 / 1 |
| QB | nested | 0.694588 / 1 | 0.679820 / 1 | 0.666595 / 1 |
| KA | nested | 0.134190 / 1 | 0.108763 / 1 | 0.108483 / 1 |
| KB | nested | 0.344243 / 1 | 0.317223 / 1 | 0.332371 / 1 |
| G | nested | 2.167480 / 1 | 1.974790 / 1 | 1.965923 / 1 |
| O | nested | 2.185162 / 1 | 1.987222 / 1 | 1.977945 / 1 |
| attention | boundary | 5.926893 / 1 | 5.456764 / 1 | 5.457446 / 1 |
| attention-residual | boundary | 0.002755 / 1 | 0.003006 / 1 | 0.002796 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025377 / 1 | 0.025277 / 1 | 0.025177 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010560 / 1 | 0.002414 / 1 | 0.001784 / 1 |
| router-and-top16 | boundary | 0.611122 / 1 | 0.580084 / 1 | 0.575815 / 1 |
| EDOWN | nested | 0.646207 / 1 | 0.591274 / 1 | 0.590854 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.647209 / 1 | 0.592326 / 1 | 0.591886 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031759 / 1 | 0.026780 / 1 | 0.006312 / 1 |
| SH1 | nested | 1.092521 / 1 | 0.978097 / 1 | 0.980251 / 1 |
| SH3 | nested | 1.100797 / 1 | 1.015537 / 1 | 1.009837 / 1 |
| SH2 | nested | 1.087361 / 1 | 1.004857 / 1 | 1.003775 / 1 |
| shared-expert-during-read | boundary | 3.292321 / 1 | 3.010946 / 1 | 3.005956 / 1 |
| detail:expert-gate | nested | 2.726702 / 16 | 2.632929 / 16 | 2.603013 / 16 |
| detail:expert-up | nested | 2.725961 / 16 | 2.634675 / 16 | 2.599827 / 16 |
| detail:expert-activation | nested | 0.093884 / 16 | 0.091971 / 16 | 0.092761 / 16 |
| detail:expert-down | nested | 2.913342 / 16 | 2.834522 / 16 | 2.759693 / 16 |
| EUP | nested | 0.628063 / 1 | 0.585524 / 1 | 0.601564 / 1 |
| experts-mix-normalize-up | boundary | 9.458952 / 1 | 9.135768 / 1 | 8.968475 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002385 / 1 | 0.003156 / 1 | 0.003105 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000160 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.007013 / 1 | 0.006352 / 1 | 0.005610 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.557872 / 1 | 111.311591 / 1 | 109.902237 / 1 |
| op:Q   int8 projection | nested | 10.381885 / 1 | 9.530766 / 1 | 9.532009 / 1 |
| op:X   mxfp4 expert proj | nested | 8.508106 / 1 | 8.242288 / 1 | 8.102346 / 1 |
| op:N   rmsnorm | nested | 0.023984 / 1 | 0.024064 / 1 | 0.023704 / 1 |
| op:SiTU + sigma | nested | 0.101838 / 1 | 0.100547 / 1 | 0.100725 / 1 |
| op:AR  snapshot aggregate | nested | 0.033372 / 1 | 0.031098 / 1 | 0.031539 / 1 |
| op:SA  softmax attention | nested | 0.006803 / 1 | 0.008977 / 1 | 0.012092 / 1 |
| op:router dot product | nested | 0.608246 / 1 | 0.577299 / 1 | 0.573682 / 1 |
| op:top-k selection | nested | 0.002425 / 1 | 0.002435 / 1 | 0.001853 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.067931 / 1 | 18.884731 / 1 | 18.687242 / 1 |

### Layer 44

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011843 / 1 | 0.003737 / 1 | 0.003256 / 1 |
| pre-attention-aggregation | boundary | 0.015248 / 1 | 0.015148 / 1 | 0.014487 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.009628 / 1 | 0.010279 / 1 | 0.010670 / 1 |
| Q | nested | 2.609044 / 1 | 2.586943 / 1 | 2.579829 / 1 |
| K | nested | 2.425701 / 1 | 2.246747 / 1 | 2.210370 / 1 |
| V | nested | 2.219467 / 1 | 2.100264 / 1 | 2.086208 / 1 |
| B | nested | 0.087794 / 1 | 0.021279 / 1 | 0.021731 / 1 |
| FA | nested | 0.057547 / 1 | 0.037780 / 1 | 0.037841 / 1 |
| FB | nested | 0.076673 / 1 | 0.050354 / 1 | 0.053971 / 1 |
| G | nested | 2.174352 / 1 | 1.999906 / 1 | 2.012890 / 1 |
| O | nested | 2.218625 / 1 | 2.116383 / 1 | 2.093341 / 1 |
| attention | boundary | 12.970982 / 1 | 12.215101 / 1 | 11.759259 / 1 |
| attention-residual | boundary | 0.003306 / 1 | 0.003407 / 1 | 0.003557 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026309 / 1 | 0.025868 / 1 | 0.024556 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009588 / 1 | 0.002404 / 1 | 0.001954 / 1 |
| router-and-top16 | boundary | 0.607886 / 1 | 0.566739 / 1 | 0.571498 / 1 |
| EDOWN | nested | 0.648732 / 1 | 0.581226 / 1 | 0.578471 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.649694 / 1 | 0.582358 / 1 | 0.579563 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.038552 / 1 | 0.017803 / 1 | 0.006873 / 1 |
| SH1 | nested | 1.138207 / 1 | 1.016338 / 1 | 1.012582 / 1 |
| SH3 | nested | 1.093442 / 1 | 0.980252 / 1 | 0.971866 / 1 |
| SH2 | nested | 1.080759 / 1 | 1.005158 / 1 | 0.996632 / 1 |
| shared-expert-during-read | boundary | 3.325092 / 1 | 3.014152 / 1 | 2.993162 / 1 |
| detail:expert-gate | nested | 2.839496 / 16 | 2.676269 / 16 | 2.707591 / 16 |
| detail:expert-up | nested | 2.834696 / 16 | 2.645691 / 16 | 2.704104 / 16 |
| detail:expert-activation | nested | 0.096541 / 16 | 0.089497 / 16 | 0.091791 / 16 |
| detail:expert-down | nested | 3.032808 / 16 | 2.810510 / 16 | 2.879443 / 16 |
| EUP | nested | 0.784876 / 1 | 0.593979 / 1 | 0.584482 / 1 |
| experts-mix-normalize-up | boundary | 9.965588 / 1 | 9.171484 / 1 | 9.327115 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002335 / 1 | 0.003236 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000180 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.390279 / 1 | 0.373838 / 1 | 0.355774 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.853871 / 1 | 111.203565 / 1 | 113.877886 / 1 |
| op:Q   int8 projection | nested | 16.613325 / 1 | 15.334916 / 1 | 15.234384 / 1 |
| op:X   mxfp4 expert proj | nested | 8.852259 / 1 | 8.273306 / 1 | 8.431351 / 1 |
| op:N   rmsnorm | nested | 0.036668 / 1 | 0.038290 / 1 | 0.022934 / 1 |
| op:L   l2 per-head | nested | 0.005250 / 1 | 0.005410 / 1 | 0.006001 / 1 |
| op:SiTU + sigma | nested | 0.105548 / 1 | 0.097883 / 1 | 0.100007 / 1 |
| op:C   shortconv | nested | 0.080821 / 1 | 0.076944 / 1 | 0.033402 / 1 |
| op:AR  snapshot aggregate | nested | 0.031929 / 1 | 0.030898 / 1 | 0.029095 / 1 |
| op:D   kda delta-rule | nested | 0.146424 / 1 | 0.132026 / 1 | 0.105988 / 1 |
| op:router dot product | nested | 0.604930 / 1 | 0.564144 / 1 | 0.568703 / 1 |
| op:top-k selection | nested | 0.002425 / 1 | 0.002144 / 1 | 0.002114 / 1 |
| op:alpha / beta / gate | nested | 0.073678 / 1 | 0.073787 / 1 | 0.058229 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.037561 / 1 | 26.016565 / 1 | 25.665670 / 1 |

### Layer 45

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013214 / 1 | 0.006082 / 1 | 0.005470 / 1 |
| pre-attention-aggregation | boundary | 0.024686 / 1 | 0.016781 / 1 | 0.016821 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011271 / 1 | 0.011150 / 1 | 0.011632 / 1 |
| Q | nested | 3.423004 / 1 | 2.647726 / 1 | 2.658386 / 1 |
| K | nested | 2.949340 / 1 | 2.212222 / 1 | 2.178470 / 1 |
| V | nested | 2.824968 / 1 | 2.078483 / 1 | 2.092559 / 1 |
| B | nested | 0.073167 / 1 | 0.022342 / 1 | 0.023574 / 1 |
| FA | nested | 0.067536 / 1 | 0.029796 / 1 | 0.029114 / 1 |
| FB | nested | 0.075611 / 1 | 0.052318 / 1 | 0.054261 / 1 |
| G | nested | 3.023659 / 1 | 2.004235 / 1 | 2.004996 / 1 |
| O | nested | 3.059976 / 1 | 2.163532 / 1 | 2.160747 / 1 |
| attention | boundary | 16.604812 / 1 | 12.290351 / 1 | 11.926211 / 1 |
| attention-residual | boundary | 0.003186 / 1 | 0.003386 / 1 | 0.003266 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027471 / 1 | 0.027141 / 1 | 0.026270 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.012874 / 1 | 0.002565 / 1 | 0.002244 / 1 |
| router-and-top16 | boundary | 0.650165 / 1 | 0.573832 / 1 | 0.573181 / 1 |
| EDOWN | nested | 0.909009 / 1 | 0.595202 / 1 | 0.588519 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.910020 / 1 | 0.596354 / 1 | 0.589511 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.044102 / 1 | 0.013595 / 1 | 0.013596 / 1 |
| SH1 | nested | 1.337790 / 1 | 1.018653 / 1 | 1.008294 / 1 |
| SH3 | nested | 1.250035 / 1 | 0.992424 / 1 | 0.972036 / 1 |
| SH2 | nested | 1.091950 / 1 | 1.006610 / 1 | 1.001942 / 1 |
| shared-expert-during-read | boundary | 3.691837 / 1 | 3.029951 / 1 | 2.994004 / 1 |
| detail:expert-gate | nested | 2.707627 / 16 | 2.533145 / 16 | 2.635794 / 16 |
| detail:expert-up | nested | 2.733447 / 16 | 2.550846 / 16 | 2.664668 / 16 |
| detail:expert-activation | nested | 0.089998 / 16 | 0.091230 / 16 | 0.091790 / 16 |
| detail:expert-down | nested | 2.895348 / 16 | 2.703419 / 16 | 2.814921 / 16 |
| EUP | nested | 0.627623 / 1 | 0.600402 / 1 | 0.595072 / 1 |
| experts-mix-normalize-up | boundary | 9.438393 / 1 | 8.832241 / 1 | 9.195229 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002625 / 1 | 0.002755 / 1 | 0.002625 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000170 / 1 | 0.000150 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.388546 / 1 | 0.379078 / 1 | 0.362537 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.132890 / 1 | 107.746604 / 1 | 111.314349 / 1 |
| op:Q   int8 projection | nested | 20.711955 / 1 | 15.422261 / 1 | 15.366469 / 1 |
| op:X   mxfp4 expert proj | nested | 8.475634 / 1 | 7.929729 / 1 | 8.255954 / 1 |
| op:N   rmsnorm | nested | 0.039364 / 1 | 0.043781 / 1 | 0.025918 / 1 |
| op:L   l2 per-head | nested | 0.005430 / 1 | 0.005881 / 1 | 0.005981 / 1 |
| op:SiTU + sigma | nested | 0.098343 / 1 | 0.099474 / 1 | 0.099715 / 1 |
| op:C   shortconv | nested | 0.053801 / 1 | 0.058420 / 1 | 0.033834 / 1 |
| op:AR  snapshot aggregate | nested | 0.041878 / 1 | 0.033383 / 1 | 0.032791 / 1 |
| op:D   kda delta-rule | nested | 0.158937 / 1 | 0.145893 / 1 | 0.121397 / 1 |
| op:router dot product | nested | 0.647119 / 1 | 0.571157 / 1 | 0.570306 / 1 |
| op:top-k selection | nested | 0.002585 / 1 | 0.002325 / 1 | 0.002435 / 1 |
| op:alpha / beta / gate | nested | 0.072947 / 1 | 0.073768 / 1 | 0.058409 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 31.834374 / 1 | 25.795653 / 1 | 25.733175 / 1 |

### Layer 46

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012623 / 1 | 0.005861 / 1 | 0.004989 / 1 |
| pre-attention-aggregation | boundary | 0.023383 / 1 | 0.017212 / 1 | 0.024115 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011421 / 1 | 0.011011 / 1 | 0.012784 / 1 |
| Q | nested | 2.504969 / 1 | 2.473020 / 1 | 2.441301 / 1 |
| K | nested | 2.322489 / 1 | 2.186705 / 1 | 2.150297 / 1 |
| V | nested | 2.188037 / 1 | 2.083082 / 1 | 2.085516 / 1 |
| B | nested | 0.087694 / 1 | 0.020499 / 1 | 0.021410 / 1 |
| FA | nested | 0.055574 / 1 | 0.034895 / 1 | 0.033582 / 1 |
| FB | nested | 0.069249 / 1 | 0.048100 / 1 | 0.048841 / 1 |
| G | nested | 2.192095 / 1 | 2.078673 / 1 | 2.069115 / 1 |
| O | nested | 2.232521 / 1 | 2.162169 / 1 | 2.152321 / 1 |
| attention | boundary | 12.741304 / 1 | 12.199892 / 1 | 11.756084 / 1 |
| attention-residual | boundary | 0.003146 / 1 | 0.003627 / 1 | 0.003857 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026770 / 1 | 0.025798 / 1 | 0.025237 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009959 / 1 | 0.002184 / 1 | 0.002334 / 1 |
| router-and-top16 | boundary | 0.598117 / 1 | 0.558593 / 1 | 0.562681 / 1 |
| EDOWN | nested | 0.670893 / 1 | 0.605531 / 1 | 0.602986 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.671905 / 1 | 0.606763 / 1 | 0.604048 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.060132 / 1 | 0.012443 / 1 | 0.006402 / 1 |
| SH1 | nested | 1.144388 / 1 | 1.001221 / 1 | 1.002733 / 1 |
| SH3 | nested | 1.106056 / 1 | 1.003675 / 1 | 1.007883 / 1 |
| SH2 | nested | 1.103111 / 1 | 1.042387 / 1 | 1.044070 / 1 |
| shared-expert-during-read | boundary | 3.366108 / 1 | 3.059556 / 1 | 3.074413 / 1 |
| detail:expert-gate | nested | 2.698463 / 16 | 2.515850 / 16 | 2.637919 / 16 |
| detail:expert-up | nested | 2.701059 / 16 | 2.503466 / 16 | 2.646915 / 16 |
| detail:expert-activation | nested | 0.089539 / 16 | 0.089316 / 16 | 0.092574 / 16 |
| detail:expert-down | nested | 2.881282 / 16 | 2.648306 / 16 | 2.851416 / 16 |
| EUP | nested | 0.628154 / 1 | 0.594781 / 1 | 0.612033 / 1 |
| experts-mix-normalize-up | boundary | 9.417524 / 1 | 8.696818 / 1 | 9.224072 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002805 / 1 | 0.003206 / 1 | 0.002695 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000240 / 1 | 0.000451 / 1 | 0.000250 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.406319 / 1 | 0.385911 / 1 | 0.373588 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 114.989075 / 1 | 106.767883 / 1 | 112.701924 / 1 |
| op:Q   int8 projection | nested | 16.303327 / 1 | 15.332764 / 1 | 15.270157 / 1 |
| op:X   mxfp4 expert proj | nested | 8.420652 / 1 | 7.807186 / 1 | 8.278888 / 1 |
| op:N   rmsnorm | nested | 0.034714 / 1 | 0.033674 / 1 | 0.025879 / 1 |
| op:L   l2 per-head | nested | 0.005580 / 1 | 0.005360 / 1 | 0.005409 / 1 |
| op:SiTU + sigma | nested | 0.097944 / 1 | 0.097543 / 1 | 0.103956 / 1 |
| op:C   shortconv | nested | 0.055223 / 1 | 0.059672 / 1 | 0.032942 / 1 |
| op:AR  snapshot aggregate | nested | 0.039563 / 1 | 0.032732 / 1 | 0.038651 / 1 |
| op:D   kda delta-rule | nested | 0.140954 / 1 | 0.126176 / 1 | 0.113011 / 1 |
| op:router dot product | nested | 0.595132 / 1 | 0.556199 / 1 | 0.559666 / 1 |
| op:top-k selection | nested | 0.002575 / 1 | 0.002134 / 1 | 0.002675 / 1 |
| op:alpha / beta / gate | nested | 0.072776 / 1 | 0.071313 / 1 | 0.058780 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.365846 / 1 | 25.600528 / 1 | 25.689344 / 1 |

### Layer 47

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014978 / 1 | 0.006462 / 1 | 0.005781 / 1 |
| pre-attention-aggregation | boundary | 0.021069 / 1 | 0.016611 / 1 | 0.016641 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000110 / 1 |
| pre-attention-normalization | boundary | 0.012273 / 1 | 0.011271 / 1 | 0.011772 / 1 |
| QA | nested | 0.320499 / 1 | 0.310019 / 1 | 0.313365 / 1 |
| QB | nested | 0.721739 / 1 | 0.694518 / 1 | 0.686863 / 1 |
| KA | nested | 0.129292 / 1 | 0.098804 / 1 | 0.101490 / 1 |
| KB | nested | 0.342350 / 1 | 0.302856 / 1 | 0.322663 / 1 |
| G | nested | 2.191855 / 1 | 2.015065 / 1 | 2.009644 / 1 |
| O | nested | 2.163292 / 1 | 1.941818 / 1 | 1.935246 / 1 |
| attention | boundary | 5.971967 / 1 | 5.468967 / 1 | 5.468927 / 1 |
| attention-residual | boundary | 0.002765 / 1 | 0.002926 / 1 | 0.002785 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026098 / 1 | 0.025348 / 1 | 0.025828 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010409 / 1 | 0.002244 / 1 | 0.002084 / 1 |
| router-and-top16 | boundary | 0.614098 / 1 | 0.579844 / 1 | 0.581016 / 1 |
| EDOWN | nested | 0.649083 / 1 | 0.599420 / 1 | 0.600211 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.650205 / 1 | 0.600432 / 1 | 0.601273 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.053190 / 1 | 0.021921 / 1 | 0.005741 / 1 |
| SH1 | nested | 1.096699 / 1 | 0.997564 / 1 | 0.999497 / 1 |
| SH3 | nested | 1.081089 / 1 | 0.991112 / 1 | 0.998676 / 1 |
| SH2 | nested | 1.076141 / 1 | 0.973980 / 1 | 0.969602 / 1 |
| shared-expert-during-read | boundary | 3.265901 / 1 | 2.974608 / 1 | 2.980148 / 1 |
| detail:expert-gate | nested | 2.750007 / 16 | 2.664066 / 16 | 2.718447 / 16 |
| detail:expert-up | nested | 2.755857 / 16 | 2.642566 / 16 | 2.764943 / 16 |
| detail:expert-activation | nested | 0.090338 / 16 | 0.090917 / 16 | 0.100927 / 16 |
| detail:expert-down | nested | 2.916929 / 16 | 2.766899 / 16 | 2.870189 / 16 |
| EUP | nested | 0.632792 / 1 | 0.603247 / 1 | 0.602777 / 1 |
| experts-mix-normalize-up | boundary | 9.536336 / 1 | 9.128284 / 1 | 9.429036 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002395 / 1 | 0.003286 / 1 | 0.003126 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000141 / 1 | 0.000141 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006282 / 1 | 0.005540 / 1 | 0.005360 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.791689 / 1 | 109.723471 / 1 | 113.691666 / 1 |
| op:Q   int8 projection | nested | 10.403487 / 1 | 9.527121 / 1 | 9.538792 / 1 |
| op:X   mxfp4 expert proj | nested | 8.565240 / 1 | 8.218120 / 1 | 8.506130 / 1 |
| op:N   rmsnorm | nested | 0.024496 / 1 | 0.023474 / 1 | 0.023764 / 1 |
| op:SiTU + sigma | nested | 0.098604 / 1 | 0.098762 / 1 | 0.109102 / 1 |
| op:AR  snapshot aggregate | nested | 0.036508 / 1 | 0.030907 / 1 | 0.031179 / 1 |
| op:SA  softmax attention | nested | 0.007434 / 1 | 0.009017 / 1 | 0.012433 / 1 |
| op:router dot product | nested | 0.611282 / 1 | 0.577629 / 1 | 0.579061 / 1 |
| op:top-k selection | nested | 0.002304 / 1 | 0.001843 / 1 | 0.001693 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.198675 / 1 | 18.858442 / 1 | 19.150197 / 1 |

### Layer 48

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010901 / 1 | 0.003586 / 1 | 0.003386 / 1 |
| pre-attention-aggregation | boundary | 0.015419 / 1 | 0.014707 / 1 | 0.014938 / 1 |
| snapshot-push | boundary | 0.001523 / 1 | 0.001874 / 1 | 0.001853 / 1 |
| pre-attention-normalization | boundary | 0.010941 / 1 | 0.009999 / 1 | 0.010489 / 1 |
| Q | nested | 2.682612 / 1 | 2.676901 / 1 | 2.657234 / 1 |
| K | nested | 2.459434 / 1 | 2.219818 / 1 | 2.177097 / 1 |
| V | nested | 2.247179 / 1 | 2.093250 / 1 | 2.087921 / 1 |
| B | nested | 0.091852 / 1 | 0.028864 / 1 | 0.028834 / 1 |
| FA | nested | 0.061094 / 1 | 0.036338 / 1 | 0.035887 / 1 |
| FB | nested | 0.073307 / 1 | 0.047609 / 1 | 0.048110 / 1 |
| G | nested | 2.177748 / 1 | 2.009815 / 1 | 2.007851 / 1 |
| O | nested | 2.213025 / 1 | 2.053907 / 1 | 2.054999 / 1 |
| attention | boundary | 13.092991 / 1 | 12.218577 / 1 | 11.796358 / 1 |
| attention-residual | boundary | 0.002184 / 1 | 0.002334 / 1 | 0.002365 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027421 / 1 | 0.026359 / 1 | 0.025768 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010139 / 1 | 0.002014 / 1 | 0.002144 / 1 |
| router-and-top16 | boundary | 0.621972 / 1 | 0.577770 / 1 | 0.575635 / 1 |
| EDOWN | nested | 0.669310 / 1 | 0.606322 / 1 | 0.606413 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.670543 / 1 | 0.607505 / 1 | 0.607465 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041898 / 1 | 0.013004 / 1 | 0.006612 / 1 |
| SH1 | nested | 1.139759 / 1 | 0.998626 / 1 | 1.004256 / 1 |
| SH3 | nested | 1.110385 / 1 | 1.005979 / 1 | 1.010648 / 1 |
| SH2 | nested | 1.123569 / 1 | 1.066262 / 1 | 1.049100 / 1 |
| shared-expert-during-read | boundary | 3.386226 / 1 | 3.083801 / 1 | 3.076978 / 1 |
| detail:expert-gate | nested | 2.731122 / 16 | 2.723487 / 16 | 2.662464 / 16 |
| detail:expert-up | nested | 2.721126 / 16 | 2.707476 / 16 | 2.648441 / 16 |
| detail:expert-activation | nested | 0.090361 / 16 | 0.093945 / 16 | 0.090089 / 16 |
| detail:expert-down | nested | 2.872224 / 16 | 2.900408 / 16 | 2.854704 / 16 |
| EUP | nested | 0.627963 / 1 | 0.596514 / 1 | 0.593609 / 1 |
| experts-mix-normalize-up | boundary | 9.433835 / 1 | 9.382498 / 1 | 9.185490 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002775 / 1 | 0.002755 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000171 / 1 | 0.000151 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.427369 / 1 | 0.408754 / 1 | 0.388366 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.278453 / 1 | 115.050469 / 1 | 112.681688 / 1 |
| op:Q   int8 projection | nested | 16.675323 / 1 | 15.438333 / 1 | 15.360378 / 1 |
| op:X   mxfp4 expert proj | nested | 8.466597 / 1 | 8.478270 / 1 | 8.308441 / 1 |
| op:N   rmsnorm | nested | 0.042629 / 1 | 0.042691 / 1 | 0.023334 / 1 |
| op:L   l2 per-head | nested | 0.005540 / 1 | 0.005750 / 1 | 0.005761 / 1 |
| op:SiTU + sigma | nested | 0.098909 / 1 | 0.102480 / 1 | 0.099105 / 1 |
| op:C   shortconv | nested | 0.068990 / 1 | 0.064660 / 1 | 0.035456 / 1 |
| op:AR  snapshot aggregate | nested | 0.032912 / 1 | 0.031038 / 1 | 0.030737 / 1 |
| op:D   kda delta-rule | nested | 0.146073 / 1 | 0.131014 / 1 | 0.118131 / 1 |
| op:router dot product | nested | 0.619077 / 1 | 0.574764 / 1 | 0.573161 / 1 |
| op:top-k selection | nested | 0.002425 / 1 | 0.002354 / 1 | 0.002084 / 1 |
| op:alpha / beta / gate | nested | 0.080961 / 1 | 0.073818 / 1 | 0.058489 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.767547 / 1 | 26.366689 / 1 | 25.711605 / 1 |

### Layer 49

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012353 / 1 | 0.004429 / 1 | 0.004588 / 1 |
| pre-attention-aggregation | boundary | 0.017693 / 1 | 0.017172 / 1 | 0.016531 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010670 / 1 | 0.010700 / 1 | 0.011482 / 1 |
| Q | nested | 2.446941 / 1 | 2.427996 / 1 | 2.415633 / 1 |
| K | nested | 2.312330 / 1 | 2.175233 / 1 | 2.168030 / 1 |
| V | nested | 2.194730 / 1 | 2.108590 / 1 | 2.096176 / 1 |
| B | nested | 0.072966 / 1 | 0.021430 / 1 | 0.021290 / 1 |
| FA | nested | 0.056556 / 1 | 0.031429 / 1 | 0.034544 / 1 |
| FB | nested | 0.073608 / 1 | 0.048150 / 1 | 0.047178 / 1 |
| G | nested | 2.164063 / 1 | 1.951336 / 1 | 1.944573 / 1 |
| O | nested | 2.254021 / 1 | 2.161017 / 1 | 2.153122 / 1 |
| attention | boundary | 12.655944 / 1 | 11.989028 / 1 | 11.649053 / 1 |
| attention-residual | boundary | 0.003306 / 1 | 0.003597 / 1 | 0.003747 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026630 / 1 | 0.025868 / 1 | 0.025848 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009528 / 1 | 0.002234 / 1 | 0.002415 / 1 |
| router-and-top16 | boundary | 0.603789 / 1 | 0.573472 / 1 | 0.567781 / 1 |
| EDOWN | nested | 0.657739 / 1 | 0.586446 / 1 | 0.586185 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.658881 / 1 | 0.587508 / 1 | 0.587337 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052438 / 1 | 0.011902 / 1 | 0.013385 / 1 |
| SH1 | nested | 1.141482 / 1 | 1.039862 / 1 | 1.034473 / 1 |
| SH3 | nested | 1.113320 / 1 | 0.997264 / 1 | 0.998405 / 1 |
| SH2 | nested | 1.095878 / 1 | 1.004847 / 1 | 1.009355 / 1 |
| shared-expert-during-read | boundary | 3.364004 / 1 | 3.054356 / 1 | 3.054937 / 1 |
| detail:expert-gate | nested | 2.747884 / 16 | 2.678223 / 16 | 2.632279 / 16 |
| detail:expert-up | nested | 2.742543 / 16 | 2.647798 / 16 | 2.593235 / 16 |
| detail:expert-activation | nested | 0.090119 / 16 | 0.088971 / 16 | 0.091331 / 16 |
| detail:expert-down | nested | 2.931297 / 16 | 2.774976 / 16 | 2.735618 / 16 |
| EUP | nested | 0.633554 / 1 | 0.595593 / 1 | 0.590163 / 1 |
| experts-mix-normalize-up | boundary | 9.547537 / 1 | 9.156226 / 1 | 9.035260 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002765 / 1 | 0.002976 / 1 | 0.002855 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000350 / 1 | 0.000161 / 1 | 0.000411 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.381923 / 1 | 0.361436 / 1 | 0.357287 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.499283 / 1 | 110.184445 / 1 | 108.211361 / 1 |
| op:Q   int8 projection | nested | 16.215564 / 1 | 15.147520 / 1 | 15.097176 / 1 |
| op:X   mxfp4 expert proj | nested | 8.565261 / 1 | 8.243450 / 1 | 8.106797 / 1 |
| op:N   rmsnorm | nested | 0.031008 / 1 | 0.036307 / 1 | 0.024645 / 1 |
| op:L   l2 per-head | nested | 0.005530 / 1 | 0.006101 / 1 | 0.005590 / 1 |
| op:SiTU + sigma | nested | 0.099316 / 1 | 0.097255 / 1 | 0.099785 / 1 |
| op:C   shortconv | nested | 0.068308 / 1 | 0.064851 / 1 | 0.033562 / 1 |
| op:AR  snapshot aggregate | nested | 0.034285 / 1 | 0.033073 / 1 | 0.031889 / 1 |
| op:D   kda delta-rule | nested | 0.151132 / 1 | 0.133981 / 1 | 0.140903 / 1 |
| op:router dot product | nested | 0.600662 / 1 | 0.570887 / 1 | 0.564534 / 1 |
| op:top-k selection | nested | 0.002555 / 1 | 0.002354 / 1 | 0.002755 / 1 |
| op:alpha / beta / gate | nested | 0.072686 / 1 | 0.071734 / 1 | 0.058910 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.364152 / 1 | 25.813055 / 1 | 25.344820 / 1 |

### Layer 50

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014507 / 1 | 0.005951 / 1 | 0.005721 / 1 |
| pre-attention-aggregation | boundary | 0.023303 / 1 | 0.018916 / 1 | 0.019276 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000101 / 1 |
| pre-attention-normalization | boundary | 0.012243 / 1 | 0.011511 / 1 | 0.011201 / 1 |
| Q | nested | 2.473461 / 1 | 2.452041 / 1 | 2.447152 / 1 |
| K | nested | 2.314444 / 1 | 2.182096 / 1 | 2.185323 / 1 |
| V | nested | 2.183980 / 1 | 2.109331 / 1 | 2.097308 / 1 |
| B | nested | 0.073898 / 1 | 0.028053 / 1 | 0.025989 / 1 |
| FA | nested | 0.063148 / 1 | 0.032461 / 1 | 0.030106 / 1 |
| FB | nested | 0.075932 / 1 | 0.049533 / 1 | 0.049893 / 1 |
| G | nested | 2.187937 / 1 | 2.005457 / 1 | 2.009073 / 1 |
| O | nested | 2.228784 / 1 | 2.060189 / 1 | 2.046624 / 1 |
| attention | boundary | 12.684177 / 1 | 11.966276 / 1 | 11.623615 / 1 |
| attention-residual | boundary | 0.003467 / 1 | 0.003457 / 1 | 0.003848 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025928 / 1 | 0.025558 / 1 | 0.025438 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010801 / 1 | 0.002695 / 1 | 0.002244 / 1 |
| router-and-top16 | boundary | 0.616131 / 1 | 0.568903 / 1 | 0.571187 / 1 |
| EDOWN | nested | 0.658640 / 1 | 0.590373 / 1 | 0.596024 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.659742 / 1 | 0.591424 / 1 | 0.597146 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042490 / 1 | 0.006261 / 1 | 0.012714 / 1 |
| SH1 | nested | 1.131293 / 1 | 1.002453 / 1 | 1.025787 / 1 |
| SH3 | nested | 1.120584 / 1 | 1.036547 / 1 | 1.038971 / 1 |
| SH2 | nested | 1.087472 / 1 | 0.996241 / 1 | 1.004106 / 1 |
| shared-expert-during-read | boundary | 3.351642 / 1 | 3.047383 / 1 | 3.080926 / 1 |
| detail:expert-gate | nested | 2.713077 / 16 | 2.659189 / 16 | 2.678665 / 16 |
| detail:expert-up | nested | 2.734036 / 16 | 2.650472 / 16 | 2.657623 / 16 |
| detail:expert-activation | nested | 0.089610 / 16 | 0.089433 / 16 | 0.089476 / 16 |
| detail:expert-down | nested | 2.901100 / 16 | 2.818093 / 16 | 2.786005 / 16 |
| EUP | nested | 0.634465 / 1 | 0.602816 / 1 | 0.600702 / 1 |
| experts-mix-normalize-up | boundary | 9.459553 / 1 | 9.168018 / 1 | 9.198836 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002585 / 1 | 0.007163 / 1 | 0.002976 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000240 / 1 | 0.000240 / 1 | 0.000260 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.387604 / 1 | 0.367567 / 1 | 0.377516 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.478013 / 1 | 111.304856 / 1 | 111.327979 / 1 |
| op:Q   int8 projection | nested | 16.232185 / 1 | 15.145957 / 1 | 15.155393 / 1 |
| op:X   mxfp4 expert proj | nested | 8.491846 / 1 | 8.271542 / 1 | 8.265872 / 1 |
| op:N   rmsnorm | nested | 0.034295 / 1 | 0.033633 / 1 | 0.023944 / 1 |
| op:L   l2 per-head | nested | 0.005610 / 1 | 0.005721 / 1 | 0.006071 / 1 |
| op:SiTU + sigma | nested | 0.097922 / 1 | 0.097330 / 1 | 0.097281 / 1 |
| op:C   shortconv | nested | 0.050824 / 1 | 0.052117 / 1 | 0.034614 / 1 |
| op:AR  snapshot aggregate | nested | 0.039274 / 1 | 0.034655 / 1 | 0.034224 / 1 |
| op:D   kda delta-rule | nested | 0.138389 / 1 | 0.115716 / 1 | 0.115877 / 1 |
| op:router dot product | nested | 0.613156 / 1 | 0.566338 / 1 | 0.568261 / 1 |
| op:top-k selection | nested | 0.002525 / 1 | 0.002204 / 1 | 0.002485 / 1 |
| op:alpha / beta / gate | nested | 0.074569 / 1 | 0.072876 / 1 | 0.058910 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.305984 / 1 | 25.803066 / 1 | 25.544473 / 1 |

### Layer 51

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013816 / 1 | 0.005140 / 1 | 0.005641 / 1 |
| pre-attention-aggregation | boundary | 0.018905 / 1 | 0.018134 / 1 | 0.017823 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000090 / 1 |
| pre-attention-normalization | boundary | 0.011852 / 1 | 0.011181 / 1 | 0.011021 / 1 |
| QA | nested | 0.328684 / 1 | 0.318324 / 1 | 0.321100 / 1 |
| QB | nested | 0.739191 / 1 | 0.685691 / 1 | 0.702793 / 1 |
| KA | nested | 0.127448 / 1 | 0.104415 / 1 | 0.098974 / 1 |
| KB | nested | 0.345465 / 1 | 0.305461 / 1 | 0.309909 / 1 |
| G | nested | 2.190733 / 1 | 2.011628 / 1 | 2.022990 / 1 |
| O | nested | 2.201302 / 1 | 2.020455 / 1 | 2.017700 / 1 |
| attention | boundary | 6.035326 / 1 | 5.544077 / 1 | 5.583051 / 1 |
| attention-residual | boundary | 0.003026 / 1 | 0.002795 / 1 | 0.003076 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.032421 / 1 | 0.029735 / 1 | 0.025688 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.011952 / 1 | 0.001773 / 1 | 0.002324 / 1 |
| router-and-top16 | boundary | 0.602024 / 1 | 0.567761 / 1 | 0.572239 / 1 |
| EDOWN | nested | 0.640928 / 1 | 0.601563 / 1 | 0.601935 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.642170 / 1 | 0.602626 / 1 | 0.603047 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032611 / 1 | 0.011561 / 1 | 0.006482 / 1 |
| SH1 | nested | 1.090126 / 1 | 0.982385 / 1 | 0.972256 / 1 |
| SH3 | nested | 1.098032 / 1 | 0.975272 / 1 | 0.973108 / 1 |
| SH2 | nested | 1.074608 / 1 | 0.994919 / 1 | 0.998606 / 1 |
| shared-expert-during-read | boundary | 3.275299 / 1 | 2.964388 / 1 | 2.956453 / 1 |
| detail:expert-gate | nested | 2.725141 / 16 | 2.695557 / 16 | 2.636504 / 16 |
| detail:expert-up | nested | 2.735028 / 16 | 2.657574 / 16 | 2.652075 / 16 |
| detail:expert-activation | nested | 0.092024 / 16 | 0.092504 / 16 | 0.089992 / 16 |
| detail:expert-down | nested | 2.908795 / 16 | 2.809098 / 16 | 2.822112 / 16 |
| EUP | nested | 0.632933 / 1 | 0.608898 / 1 | 0.609309 / 1 |
| experts-mix-normalize-up | boundary | 9.471375 / 1 | 9.208604 / 1 | 9.176714 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002495 / 1 | 0.002966 / 1 | 0.003036 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000160 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.009047 / 1 | 0.005461 / 1 | 0.005581 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.825994 / 1 | 112.013844 / 1 | 111.029673 / 1 |
| op:Q   int8 projection | nested | 10.468127 / 1 | 9.607647 / 1 | 9.627369 / 1 |
| op:X   mxfp4 expert proj | nested | 8.520936 / 1 | 8.309737 / 1 | 8.255461 / 1 |
| op:N   rmsnorm | nested | 0.028584 / 1 | 0.023844 / 1 | 0.024044 / 1 |
| op:SiTU + sigma | nested | 0.100579 / 1 | 0.100279 / 1 | 0.098417 / 1 |
| op:AR  snapshot aggregate | nested | 0.036980 / 1 | 0.037049 / 1 | 0.032651 / 1 |
| op:SA  softmax attention | nested | 0.006692 / 1 | 0.009077 / 1 | 0.011732 / 1 |
| op:router dot product | nested | 0.598949 / 1 | 0.564886 / 1 | 0.569493 / 1 |
| op:top-k selection | nested | 0.002625 / 1 | 0.002425 / 1 | 0.002384 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.174280 / 1 | 18.987333 / 1 | 18.983796 / 1 |

### Layer 52

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010600 / 1 | 0.003917 / 1 | 0.003586 / 1 |
| pre-attention-aggregation | boundary | 0.015689 / 1 | 0.014658 / 1 | 0.015088 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010861 / 1 | 0.010901 / 1 | 0.010750 / 1 |
| Q | nested | 2.494951 / 1 | 2.449967 / 1 | 2.445098 / 1 |
| K | nested | 2.316919 / 1 | 2.173230 / 1 | 2.160015 / 1 |
| V | nested | 2.183679 / 1 | 2.094253 / 1 | 2.095084 / 1 |
| B | nested | 0.093735 / 1 | 0.021790 / 1 | 0.022131 / 1 |
| FA | nested | 0.062977 / 1 | 0.031318 / 1 | 0.032732 / 1 |
| FB | nested | 0.073457 / 1 | 0.045495 / 1 | 0.048541 / 1 |
| G | nested | 2.156068 / 1 | 1.950023 / 1 | 1.957107 / 1 |
| O | nested | 2.240336 / 1 | 2.180844 / 1 | 2.193798 / 1 |
| attention | boundary | 12.690549 / 1 | 11.956097 / 1 | 11.647109 / 1 |
| attention-residual | boundary | 0.002836 / 1 | 0.003727 / 1 | 0.003536 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026709 / 1 | 0.026259 / 1 | 0.025578 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010159 / 1 | 0.002315 / 1 | 0.002595 / 1 |
| router-and-top16 | boundary | 0.603727 / 1 | 0.567971 / 1 | 0.564855 / 1 |
| EDOWN | nested | 0.668439 / 1 | 0.600312 / 1 | 0.601924 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.669541 / 1 | 0.601384 / 1 | 0.603116 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.058789 / 1 | 0.016851 / 1 | 0.006191 / 1 |
| SH1 | nested | 1.140781 / 1 | 1.027890 / 1 | 1.035615 / 1 |
| SH3 | nested | 1.126044 / 1 | 1.021769 / 1 | 1.025636 / 1 |
| SH2 | nested | 1.074187 / 1 | 1.005830 / 1 | 0.994438 / 1 |
| shared-expert-during-read | boundary | 3.353826 / 1 | 3.068042 / 1 | 3.068462 / 1 |
| detail:expert-gate | nested | 2.705978 / 16 | 2.709422 / 16 | 2.535928 / 16 |
| detail:expert-up | nested | 2.704030 / 16 | 2.699163 / 16 | 2.515020 / 16 |
| detail:expert-activation | nested | 0.088954 / 16 | 0.090740 / 16 | 0.089560 / 16 |
| detail:expert-down | nested | 2.915815 / 16 | 2.894909 / 16 | 2.686940 / 16 |
| EUP | nested | 0.634275 / 1 | 0.603738 / 1 | 0.604951 / 1 |
| experts-mix-normalize-up | boundary | 9.461346 / 1 | 9.359095 / 1 | 8.817804 / 1 |
| mlp-merge-and-cleanup | boundary | 0.006362 / 1 | 0.002765 / 1 | 0.003146 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000141 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.452776 / 1 | 0.399908 / 1 | 0.404967 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.478617 / 1 | 114.771050 / 1 | 106.564996 / 1 |
| op:Q   int8 projection | nested | 16.263775 / 1 | 15.204787 / 1 | 15.215105 / 1 |
| op:X   mxfp4 expert proj | nested | 8.471066 / 1 | 8.449384 / 1 | 7.882407 / 1 |
| op:N   rmsnorm | nested | 0.041407 / 1 | 0.033763 / 1 | 0.023964 / 1 |
| op:L   l2 per-head | nested | 0.005260 / 1 | 0.005821 / 1 | 0.006012 / 1 |
| op:SiTU + sigma | nested | 0.097632 / 1 | 0.099095 / 1 | 0.097996 / 1 |
| op:C   shortconv | nested | 0.072886 / 1 | 0.065743 / 1 | 0.034795 / 1 |
| op:AR  snapshot aggregate | nested | 0.032240 / 1 | 0.030577 / 1 | 0.030428 / 1 |
| op:D   kda delta-rule | nested | 0.147345 / 1 | 0.122960 / 1 | 0.129762 / 1 |
| op:router dot product | nested | 0.600502 / 1 | 0.565186 / 1 | 0.561910 / 1 |
| op:top-k selection | nested | 0.002525 / 1 | 0.002374 / 1 | 0.002565 / 1 |
| op:alpha / beta / gate | nested | 0.071653 / 1 | 0.073828 / 1 | 0.058148 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.387105 / 1 | 26.046691 / 1 | 25.188718 / 1 |

### Layer 53

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013185 / 1 | 0.005510 / 1 | 0.004779 / 1 |
| pre-attention-aggregation | boundary | 0.017252 / 1 | 0.016280 / 1 | 0.016250 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.010820 / 1 | 0.010830 / 1 |
| Q | nested | 2.699413 / 1 | 2.692841 / 1 | 2.669648 / 1 |
| K | nested | 2.461188 / 1 | 2.237400 / 1 | 2.170014 / 1 |
| V | nested | 2.251927 / 1 | 2.122295 / 1 | 2.069466 / 1 |
| B | nested | 0.066404 / 1 | 0.020859 / 1 | 0.019877 / 1 |
| FA | nested | 0.059071 / 1 | 0.027812 / 1 | 0.028864 / 1 |
| FB | nested | 0.074379 / 1 | 0.049212 / 1 | 0.047018 / 1 |
| G | nested | 2.163221 / 1 | 1.963779 / 1 | 1.956865 / 1 |
| O | nested | 2.229215 / 1 | 2.181736 / 1 | 2.193097 / 1 |
| attention | boundary | 13.026757 / 1 | 12.281235 / 1 | 11.837506 / 1 |
| attention-residual | boundary | 0.003998 / 1 | 0.003106 / 1 | 0.003356 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026821 / 1 | 0.026409 / 1 | 0.026700 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010590 / 1 | 0.002615 / 1 | 0.002515 / 1 |
| router-and-top16 | boundary | 0.622613 / 1 | 0.584221 / 1 | 0.580956 / 1 |
| EDOWN | nested | 0.665313 / 1 | 0.603548 / 1 | 0.596785 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.666515 / 1 | 0.604750 / 1 | 0.597897 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048821 / 1 | 0.028273 / 1 | 0.006231 / 1 |
| SH1 | nested | 1.132616 / 1 | 0.990060 / 1 | 0.986242 / 1 |
| SH3 | nested | 1.108551 / 1 | 1.012432 / 1 | 1.004347 / 1 |
| SH2 | nested | 1.099695 / 1 | 1.050973 / 1 | 1.045944 / 1 |
| shared-expert-during-read | boundary | 3.354126 / 1 | 3.066319 / 1 | 3.049066 / 1 |
| detail:expert-gate | nested | 2.705216 / 16 | 2.663997 / 16 | 2.633863 / 16 |
| detail:expert-up | nested | 2.692241 / 16 | 2.670623 / 16 | 2.618572 / 16 |
| detail:expert-activation | nested | 0.089837 / 16 | 0.091175 / 16 | 0.092502 / 16 |
| detail:expert-down | nested | 2.904157 / 16 | 2.832134 / 16 | 2.830891 / 16 |
| EUP | nested | 0.635437 / 1 | 0.610330 / 1 | 0.589712 / 1 |
| experts-mix-normalize-up | boundary | 9.428154 / 1 | 9.234282 / 1 | 9.153861 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002595 / 1 | 0.002685 / 1 | 0.002595 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000250 / 1 | 0.000411 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.433650 / 1 | 0.414444 / 1 | 0.402301 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 114.938896 / 1 | 113.373805 / 1 | 110.991790 / 1 |
| op:Q   int8 projection | nested | 16.644458 / 1 | 15.561634 / 1 | 15.376016 / 1 |
| op:X   mxfp4 expert proj | nested | 8.449126 / 1 | 8.313923 / 1 | 8.233243 / 1 |
| op:N   rmsnorm | nested | 0.033211 / 1 | 0.034085 / 1 | 0.023273 / 1 |
| op:L   l2 per-head | nested | 0.005942 / 1 | 0.005801 / 1 | 0.005331 / 1 |
| op:SiTU + sigma | nested | 0.098453 / 1 | 0.099277 / 1 | 0.100809 / 1 |
| op:C   shortconv | nested | 0.065503 / 1 | 0.061145 / 1 | 0.033323 / 1 |
| op:AR  snapshot aggregate | nested | 0.033694 / 1 | 0.032420 / 1 | 0.032742 / 1 |
| op:D   kda delta-rule | nested | 0.121897 / 1 | 0.107331 / 1 | 0.105396 / 1 |
| op:router dot product | nested | 0.619487 / 1 | 0.581556 / 1 | 0.578340 / 1 |
| op:top-k selection | nested | 0.002655 / 1 | 0.002274 / 1 | 0.002083 / 1 |
| op:alpha / beta / gate | nested | 0.075300 / 1 | 0.073287 / 1 | 0.058289 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.678640 / 1 | 26.292801 / 1 | 25.706746 / 1 |

### Layer 54

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012824 / 1 | 0.005010 / 1 | 0.005199 / 1 |
| pre-attention-aggregation | boundary | 0.017243 / 1 | 0.016441 / 1 | 0.016420 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000090 / 1 |
| pre-attention-normalization | boundary | 0.011752 / 1 | 0.010169 / 1 | 0.011210 / 1 |
| Q | nested | 2.534745 / 1 | 2.444647 / 1 | 2.436392 / 1 |
| K | nested | 2.329262 / 1 | 2.193608 / 1 | 2.137814 / 1 |
| V | nested | 2.188659 / 1 | 2.109381 / 1 | 2.087140 / 1 |
| B | nested | 0.086592 / 1 | 0.022071 / 1 | 0.022352 / 1 |
| FA | nested | 0.072224 / 1 | 0.028864 / 1 | 0.033152 / 1 |
| FB | nested | 0.065703 / 1 | 0.050414 / 1 | 0.049813 / 1 |
| G | nested | 2.201884 / 1 | 2.047445 / 1 | 2.038829 / 1 |
| O | nested | 2.219126 / 1 | 2.131352 / 1 | 2.134669 / 1 |
| attention | boundary | 12.724883 / 1 | 12.017451 / 1 | 11.609259 / 1 |
| attention-residual | boundary | 0.002786 / 1 | 0.003807 / 1 | 0.003737 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026820 / 1 | 0.026129 / 1 | 0.026479 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010159 / 1 | 0.002084 / 1 | 0.002524 / 1 |
| router-and-top16 | boundary | 0.598077 / 1 | 0.568362 / 1 | 0.563743 / 1 |
| EDOWN | nested | 0.643122 / 1 | 0.570145 / 1 | 0.573742 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.644454 / 1 | 0.571237 / 1 | 0.574854 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042379 / 1 | 0.023734 / 1 | 0.006232 / 1 |
| SH1 | nested | 1.107178 / 1 | 0.990731 / 1 | 0.979079 / 1 |
| SH3 | nested | 1.133007 / 1 | 0.984238 / 1 | 0.984579 / 1 |
| SH2 | nested | 1.092000 / 1 | 0.989969 / 1 | 0.985010 / 1 |
| shared-expert-during-read | boundary | 3.345150 / 1 | 2.984145 / 1 | 2.961182 / 1 |
| detail:expert-gate | nested | 2.714070 / 16 | 2.613422 / 16 | 2.663438 / 16 |
| detail:expert-up | nested | 2.744748 / 16 | 2.678352 / 16 | 2.665679 / 16 |
| detail:expert-activation | nested | 0.087443 / 16 | 0.094757 / 16 | 0.092320 / 16 |
| detail:expert-down | nested | 2.896512 / 16 | 2.792359 / 16 | 2.798253 / 16 |
| EUP | nested | 0.631449 / 1 | 0.592918 / 1 | 0.598899 / 1 |
| experts-mix-normalize-up | boundary | 9.468289 / 1 | 9.143422 / 1 | 9.193355 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002755 / 1 | 0.003025 / 1 | 0.002454 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000251 / 1 | 0.000240 / 1 | 0.000511 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.388576 / 1 | 0.352789 / 1 | 0.354352 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.721172 / 1 | 111.171554 / 1 | 112.492552 / 1 |
| op:Q   int8 projection | nested | 16.303300 / 1 | 15.154051 / 1 | 15.059497 / 1 |
| op:X   mxfp4 expert proj | nested | 8.500801 / 1 | 8.236876 / 1 | 8.281011 / 1 |
| op:N   rmsnorm | nested | 0.038522 / 1 | 0.033011 / 1 | 0.024186 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.005490 / 1 | 0.006362 / 1 |
| op:SiTU + sigma | nested | 0.096089 / 1 | 0.105738 / 1 | 0.100694 / 1 |
| op:C   shortconv | nested | 0.050144 / 1 | 0.056456 / 1 | 0.032090 / 1 |
| op:AR  snapshot aggregate | nested | 0.033962 / 1 | 0.032492 / 1 | 0.032390 / 1 |
| op:D   kda delta-rule | nested | 0.137958 / 1 | 0.116909 / 1 | 0.111008 / 1 |
| op:router dot product | nested | 0.595011 / 1 | 0.565576 / 1 | 0.561439 / 1 |
| op:top-k selection | nested | 0.002275 / 1 | 0.002344 / 1 | 0.001844 / 1 |
| op:alpha / beta / gate | nested | 0.075311 / 1 | 0.074620 / 1 | 0.058980 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.310432 / 1 | 25.739878 / 1 | 25.343207 / 1 |

### Layer 55

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012644 / 1 | 0.004719 / 1 | 0.004448 / 1 |
| pre-attention-aggregation | boundary | 0.021129 / 1 | 0.016901 / 1 | 0.017312 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.010810 / 1 | 0.010800 / 1 | 0.010820 / 1 |
| QA | nested | 0.309568 / 1 | 0.303457 / 1 | 0.313876 / 1 |
| QB | nested | 0.735795 / 1 | 0.696811 / 1 | 0.692073 / 1 |
| KA | nested | 0.120205 / 1 | 0.104264 / 1 | 0.102501 / 1 |
| KB | nested | 0.349062 / 1 | 0.304559 / 1 | 0.318875 / 1 |
| G | nested | 2.222271 / 1 | 2.003413 / 1 | 2.005367 / 1 |
| O | nested | 2.172900 / 1 | 1.918083 / 1 | 1.905049 / 1 |
| attention | boundary | 6.011651 / 1 | 5.434342 / 1 | 5.448549 / 1 |
| attention-residual | boundary | 0.002354 / 1 | 0.002855 / 1 | 0.003276 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025968 / 1 | 0.029315 / 1 | 0.024826 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010560 / 1 | 0.002555 / 1 | 0.001944 / 1 |
| router-and-top16 | boundary | 0.607084 / 1 | 0.568252 / 1 | 0.569183 / 1 |
| EDOWN | nested | 0.635347 / 1 | 0.586225 / 1 | 0.576197 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.636600 / 1 | 0.587347 / 1 | 0.577379 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.047529 / 1 | 0.017322 / 1 | 0.006312 / 1 |
| SH1 | nested | 1.106848 / 1 | 1.006110 / 1 | 1.017651 / 1 |
| SH3 | nested | 1.080428 / 1 | 0.990721 / 1 | 0.990711 / 1 |
| SH2 | nested | 1.089235 / 1 | 1.033330 / 1 | 1.025366 / 1 |
| shared-expert-during-read | boundary | 3.289706 / 1 | 3.042334 / 1 | 3.046351 / 1 |
| detail:expert-gate | nested | 2.720912 / 16 | 2.628673 / 16 | 2.604824 / 16 |
| detail:expert-up | nested | 2.719249 / 16 | 2.642306 / 16 | 2.621144 / 16 |
| detail:expert-activation | nested | 0.088819 / 16 | 0.090819 / 16 | 0.093644 / 16 |
| detail:expert-down | nested | 2.918873 / 16 | 2.802975 / 16 | 2.827735 / 16 |
| EUP | nested | 0.643753 / 1 | 0.610611 / 1 | 0.600061 / 1 |
| experts-mix-normalize-up | boundary | 9.487445 / 1 | 9.133283 / 1 | 9.094109 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002695 / 1 | 0.002775 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000151 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006862 / 1 | 0.005190 / 1 | 0.005099 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.540329 / 1 | 112.084666 / 1 | 111.044507 / 1 |
| op:Q   int8 projection | nested | 10.464110 / 1 | 9.555799 / 1 | 9.546095 / 1 |
| op:X   mxfp4 expert proj | nested | 8.505990 / 1 | 8.222151 / 1 | 8.206159 / 1 |
| op:N   rmsnorm | nested | 0.024254 / 1 | 0.023363 / 1 | 0.023573 / 1 |
| op:SiTU + sigma | nested | 0.097165 / 1 | 0.098696 / 1 | 0.101549 / 1 |
| op:AR  snapshot aggregate | nested | 0.036448 / 1 | 0.035446 / 1 | 0.031739 / 1 |
| op:SA  softmax attention | nested | 0.006943 / 1 | 0.008235 / 1 | 0.011762 / 1 |
| op:router dot product | nested | 0.604209 / 1 | 0.565857 / 1 | 0.566919 / 1 |
| op:top-k selection | nested | 0.002175 / 1 | 0.002064 / 1 | 0.001913 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.185271 / 1 | 18.870023 / 1 | 18.824659 / 1 |

### Layer 56

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011732 / 1 | 0.003546 / 1 | 0.003666 / 1 |
| pre-attention-aggregation | boundary | 0.016181 / 1 | 0.015148 / 1 | 0.014757 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010690 / 1 | 0.010349 / 1 | 0.011011 / 1 |
| Q | nested | 2.562006 / 1 | 2.540216 / 1 | 2.444156 / 1 |
| K | nested | 2.329643 / 1 | 2.181015 / 1 | 2.149326 / 1 |
| V | nested | 2.174052 / 1 | 2.110312 / 1 | 2.089373 / 1 |
| B | nested | 0.066724 / 1 | 0.029004 / 1 | 0.028352 / 1 |
| FA | nested | 0.059952 / 1 | 0.025307 / 1 | 0.027041 / 1 |
| FB | nested | 0.076183 / 1 | 0.052859 / 1 | 0.047088 / 1 |
| G | nested | 2.182628 / 1 | 2.024172 / 1 | 2.013291 / 1 |
| O | nested | 2.195051 / 1 | 2.028069 / 1 | 2.017579 / 1 |
| attention | boundary | 12.740172 / 1 | 11.986032 / 1 | 11.475539 / 1 |
| attention-residual | boundary | 0.003456 / 1 | 0.003456 / 1 | 0.003646 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027291 / 1 | 0.024776 / 1 | 0.025959 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010270 / 1 | 0.002094 / 1 | 0.002174 / 1 |
| router-and-top16 | boundary | 0.603708 / 1 | 0.569534 / 1 | 0.562982 / 1 |
| EDOWN | nested | 0.644845 / 1 | 0.581987 / 1 | 0.584763 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.646087 / 1 | 0.583179 / 1 | 0.585975 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027402 / 1 | 0.017002 / 1 | 0.006102 / 1 |
| SH1 | nested | 1.138687 / 1 | 0.995650 / 1 | 1.012472 / 1 |
| SH3 | nested | 1.115615 / 1 | 1.015697 / 1 | 1.019365 / 1 |
| SH2 | nested | 1.078234 / 1 | 1.030034 / 1 | 1.017471 / 1 |
| shared-expert-during-read | boundary | 3.345280 / 1 | 3.054417 / 1 | 3.061690 / 1 |
| detail:expert-gate | nested | 2.718097 / 16 | 2.699126 / 16 | 2.500273 / 16 |
| detail:expert-up | nested | 2.722566 / 16 | 2.718858 / 16 | 2.530999 / 16 |
| detail:expert-activation | nested | 0.087714 / 16 | 0.094304 / 16 | 0.094937 / 16 |
| detail:expert-down | nested | 2.919410 / 16 | 2.879700 / 16 | 2.646884 / 16 |
| EUP | nested | 0.641168 / 1 | 0.611793 / 1 | 0.597196 / 1 |
| experts-mix-normalize-up | boundary | 9.469771 / 1 | 9.368202 / 1 | 8.748013 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002304 / 1 | 0.002795 / 1 | 0.002564 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000250 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.426266 / 1 | 0.383226 / 1 | 0.389728 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.169524 / 1 | 115.040608 / 1 | 105.417285 / 1 |
| op:Q   int8 projection | nested | 16.263264 / 1 | 15.220565 / 1 | 15.045920 / 1 |
| op:X   mxfp4 expert proj | nested | 8.507935 / 1 | 8.451541 / 1 | 7.832082 / 1 |
| op:N   rmsnorm | nested | 0.031639 / 1 | 0.037810 / 1 | 0.023043 / 1 |
| op:L   l2 per-head | nested | 0.006392 / 1 | 0.005200 / 1 | 0.005781 / 1 |
| op:SiTU + sigma | nested | 0.096172 / 1 | 0.102542 / 1 | 0.102391 / 1 |
| op:C   shortconv | nested | 0.063208 / 1 | 0.063429 / 1 | 0.034525 / 1 |
| op:AR  snapshot aggregate | nested | 0.033243 / 1 | 0.030557 / 1 | 0.030828 / 1 |
| op:D   kda delta-rule | nested | 0.146183 / 1 | 0.110647 / 1 | 0.104946 / 1 |
| op:router dot product | nested | 0.600843 / 1 | 0.567029 / 1 | 0.560447 / 1 |
| op:top-k selection | nested | 0.002485 / 1 | 0.002234 / 1 | 0.002124 / 1 |
| op:alpha / beta / gate | nested | 0.083386 / 1 | 0.073708 / 1 | 0.058690 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.353623 / 1 | 26.037103 / 1 | 24.907142 / 1 |

### Layer 57

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012103 / 1 | 0.005219 / 1 | 0.005670 / 1 |
| pre-attention-aggregation | boundary | 0.020969 / 1 | 0.016931 / 1 | 0.016300 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011231 / 1 | 0.011272 / 1 | 0.010951 / 1 |
| Q | nested | 2.503958 / 1 | 2.472369 / 1 | 2.472489 / 1 |
| K | nested | 2.343929 / 1 | 2.193197 / 1 | 2.158383 / 1 |
| V | nested | 2.199630 / 1 | 2.104582 / 1 | 2.087610 / 1 |
| B | nested | 0.083646 / 1 | 0.027872 / 1 | 0.028233 / 1 |
| FA | nested | 0.059661 / 1 | 0.029615 / 1 | 0.030747 / 1 |
| FB | nested | 0.080741 / 1 | 0.059180 / 1 | 0.052658 / 1 |
| G | nested | 2.189541 / 1 | 2.026155 / 1 | 2.031295 / 1 |
| O | nested | 2.193688 / 1 | 2.065549 / 1 | 2.053827 / 1 |
| attention | boundary | 12.720585 / 1 | 12.011691 / 1 | 11.603378 / 1 |
| attention-residual | boundary | 0.004328 / 1 | 0.004018 / 1 | 0.003576 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025979 / 1 | 0.024897 / 1 | 0.024656 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010440 / 1 | 0.002224 / 1 | 0.002214 / 1 |
| router-and-top16 | boundary | 0.604670 / 1 | 0.572890 / 1 | 0.566678 / 1 |
| EDOWN | nested | 0.654723 / 1 | 0.596555 / 1 | 0.594511 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.655905 / 1 | 0.597787 / 1 | 0.595723 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.049973 / 1 | 0.037110 / 1 | 0.006933 / 1 |
| SH1 | nested | 1.120864 / 1 | 0.982947 / 1 | 0.987295 / 1 |
| SH3 | nested | 1.108501 / 1 | 0.982145 / 1 | 0.984299 / 1 |
| SH2 | nested | 1.117588 / 1 | 1.024765 / 1 | 1.017741 / 1 |
| shared-expert-during-read | boundary | 3.359706 / 1 | 3.002680 / 1 | 3.001839 / 1 |
| detail:expert-gate | nested | 2.737074 / 16 | 2.731372 / 16 | 2.522866 / 16 |
| detail:expert-up | nested | 2.716163 / 16 | 2.698611 / 16 | 2.512333 / 16 |
| detail:expert-activation | nested | 0.088174 / 16 | 0.091693 / 16 | 0.094705 / 16 |
| detail:expert-down | nested | 2.900502 / 16 | 2.867615 / 16 | 2.663597 / 16 |
| EUP | nested | 0.635628 / 1 | 0.606493 / 1 | 0.600141 / 1 |
| experts-mix-normalize-up | boundary | 9.478538 / 1 | 9.388820 / 1 | 8.769995 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.002885 / 1 | 0.003006 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000170 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429964 / 1 | 0.398595 / 1 | 0.398365 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.264410 / 1 | 114.738195 / 1 | 105.600585 / 1 |
| op:Q   int8 projection | nested | 16.290456 / 1 | 15.169873 / 1 | 15.097568 / 1 |
| op:X   mxfp4 expert proj | nested | 8.503409 / 1 | 8.448995 / 1 | 7.853554 / 1 |
| op:N   rmsnorm | nested | 0.043982 / 1 | 0.037550 / 1 | 0.023844 / 1 |
| op:L   l2 per-head | nested | 0.005269 / 1 | 0.006041 / 1 | 0.005620 / 1 |
| op:SiTU + sigma | nested | 0.096559 / 1 | 0.100008 / 1 | 0.102440 / 1 |
| op:C   shortconv | nested | 0.063429 / 1 | 0.065082 / 1 | 0.033922 / 1 |
| op:AR  snapshot aggregate | nested | 0.036779 / 1 | 0.032300 / 1 | 0.030777 / 1 |
| op:D   kda delta-rule | nested | 0.140752 / 1 | 0.110216 / 1 | 0.116127 / 1 |
| op:router dot product | nested | 0.601553 / 1 | 0.570135 / 1 | 0.563713 / 1 |
| op:top-k selection | nested | 0.002454 / 1 | 0.002364 / 1 | 0.002495 / 1 |
| op:alpha / beta / gate | nested | 0.073487 / 1 | 0.075340 / 1 | 0.058519 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.401723 / 1 | 26.091224 / 1 | 25.024010 / 1 |

### Layer 58

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014146 / 1 | 0.005530 / 1 | 0.006091 / 1 |
| pre-attention-aggregation | boundary | 0.017673 / 1 | 0.017302 / 1 | 0.015739 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010369 / 1 | 0.011181 / 1 | 0.011622 / 1 |
| Q | nested | 2.666682 / 1 | 2.662855 / 1 | 2.645272 / 1 |
| K | nested | 2.468582 / 1 | 2.216451 / 1 | 2.197275 / 1 |
| V | nested | 2.238582 / 1 | 2.106005 / 1 | 2.072532 / 1 |
| B | nested | 0.086472 / 1 | 0.027091 / 1 | 0.026930 / 1 |
| FA | nested | 0.059101 / 1 | 0.027331 / 1 | 0.030617 / 1 |
| FB | nested | 0.066243 / 1 | 0.051787 / 1 | 0.050584 / 1 |
| G | nested | 2.192887 / 1 | 2.031675 / 1 | 2.031104 / 1 |
| O | nested | 2.233743 / 1 | 2.130480 / 1 | 2.118828 / 1 |
| attention | boundary | 13.061953 / 1 | 12.301783 / 1 | 11.896796 / 1 |
| attention-residual | boundary | 0.002685 / 1 | 0.003116 / 1 | 0.003066 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026119 / 1 | 0.026670 / 1 | 0.026159 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010069 / 1 | 0.002214 / 1 | 0.002074 / 1 |
| router-and-top16 | boundary | 0.615710 / 1 | 0.569514 / 1 | 0.577609 / 1 |
| EDOWN | nested | 0.679179 / 1 | 0.617614 / 1 | 0.599600 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.680451 / 1 | 0.618826 / 1 | 0.600762 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.032641 / 1 | 0.006442 / 1 | 0.024476 / 1 |
| SH1 | nested | 1.134299 / 1 | 1.016288 / 1 | 1.008033 / 1 |
| SH3 | nested | 1.089004 / 1 | 0.982556 / 1 | 0.969912 / 1 |
| SH2 | nested | 1.095958 / 1 | 1.034663 / 1 | 1.031317 / 1 |
| shared-expert-during-read | boundary | 3.332616 / 1 | 3.046090 / 1 | 3.022126 / 1 |
| detail:expert-gate | nested | 2.758336 / 16 | 2.663654 / 16 | 2.652628 / 16 |
| detail:expert-up | nested | 2.733114 / 16 | 2.686016 / 16 | 2.625322 / 16 |
| detail:expert-activation | nested | 0.091177 / 16 | 0.091168 / 16 | 0.096162 / 16 |
| detail:expert-down | nested | 2.941323 / 16 | 2.866875 / 16 | 2.799891 / 16 |
| EUP | nested | 0.632051 / 1 | 0.606583 / 1 | 0.597696 / 1 |
| experts-mix-normalize-up | boundary | 9.532559 / 1 | 9.286008 / 1 | 9.169971 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002575 / 1 | 0.002865 / 1 | 0.002825 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000251 / 1 | 0.000271 / 1 | 0.000231 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.404475 / 1 | 0.373808 / 1 | 0.391381 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.144674 / 1 | 113.358204 / 1 | 111.062339 / 1 |
| op:Q   int8 projection | nested | 16.640989 / 1 | 15.509455 / 1 | 15.377898 / 1 |
| op:X   mxfp4 expert proj | nested | 8.584771 / 1 | 8.368075 / 1 | 8.238634 / 1 |
| op:N   rmsnorm | nested | 0.031869 / 1 | 0.033994 / 1 | 0.024385 / 1 |
| op:L   l2 per-head | nested | 0.005891 / 1 | 0.005300 / 1 | 0.005801 / 1 |
| op:SiTU + sigma | nested | 0.100145 / 1 | 0.099293 / 1 | 0.104436 / 1 |
| op:C   shortconv | nested | 0.064821 / 1 | 0.060673 / 1 | 0.034194 / 1 |
| op:AR  snapshot aggregate | nested | 0.033993 / 1 | 0.033954 / 1 | 0.031830 / 1 |
| op:D   kda delta-rule | nested | 0.144700 / 1 | 0.130424 / 1 | 0.126817 / 1 |
| op:router dot product | nested | 0.613015 / 1 | 0.566598 / 1 | 0.574954 / 1 |
| op:top-k selection | nested | 0.002315 / 1 | 0.002545 / 1 | 0.002294 / 1 |
| op:alpha / beta / gate | nested | 0.074209 / 1 | 0.074379 / 1 | 0.058099 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.757066 / 1 | 26.284245 / 1 | 25.763572 / 1 |

### Layer 59

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012473 / 1 | 0.004288 / 1 | 0.004849 / 1 |
| pre-attention-aggregation | boundary | 0.029214 / 1 | 0.017843 / 1 | 0.017283 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000101 / 1 |
| pre-attention-normalization | boundary | 0.011362 / 1 | 0.011020 / 1 | 0.011051 / 1 |
| QA | nested | 0.322493 / 1 | 0.320358 / 1 | 0.317012 / 1 |
| QB | nested | 0.738419 / 1 | 0.696441 / 1 | 0.708884 / 1 |
| KA | nested | 0.123270 / 1 | 0.111528 / 1 | 0.110386 / 1 |
| KB | nested | 0.344263 / 1 | 0.299830 / 1 | 0.299419 / 1 |
| G | nested | 2.200902 / 1 | 2.013722 / 1 | 2.028711 / 1 |
| O | nested | 2.199158 / 1 | 1.993264 / 1 | 1.990729 / 1 |
| attention | boundary | 6.032360 / 1 | 5.537785 / 1 | 5.556801 / 1 |
| attention-residual | boundary | 0.002645 / 1 | 0.002795 / 1 | 0.003246 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025868 / 1 | 0.026179 / 1 | 0.026550 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.011862 / 1 | 0.001923 / 1 | 0.002214 / 1 |
| router-and-top16 | boundary | 0.599780 / 1 | 0.577659 / 1 | 0.570856 / 1 |
| EDOWN | nested | 0.644524 / 1 | 0.576467 / 1 | 0.575815 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.645706 / 1 | 0.577659 / 1 | 0.577048 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052107 / 1 | 0.011992 / 1 | 0.012333 / 1 |
| SH1 | nested | 1.105004 / 1 | 1.004587 / 1 | 1.008844 / 1 |
| SH3 | nested | 1.081832 / 1 | 0.964041 / 1 | 0.965925 / 1 |
| SH2 | nested | 1.077162 / 1 | 1.000960 / 1 | 0.994047 / 1 |
| shared-expert-during-read | boundary | 3.277012 / 1 | 2.982682 / 1 | 2.982051 / 1 |
| detail:expert-gate | nested | 2.808167 / 16 | 2.702729 / 16 | 2.507183 / 16 |
| detail:expert-up | nested | 2.817293 / 16 | 2.741842 / 16 | 2.487777 / 16 |
| detail:expert-activation | nested | 0.243144 / 16 | 0.090762 / 16 | 0.094358 / 16 |
| detail:expert-down | nested | 2.943377 / 16 | 2.863067 / 16 | 2.654930 / 16 |
| EUP | nested | 0.629957 / 1 | 0.613146 / 1 | 0.595873 / 1 |
| experts-mix-normalize-up | boundary | 9.847227 / 1 | 9.379573 / 1 | 8.700335 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002284 / 1 | 0.002975 / 1 | 0.003417 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000150 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006823 / 1 | 0.006943 / 1 | 0.005360 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.567153 / 1 | 114.265034 / 1 | 105.074907 / 1 |
| op:Q   int8 projection | nested | 10.465692 / 1 | 9.593022 / 1 | 9.594214 / 1 |
| op:X   mxfp4 expert proj | nested | 8.876865 / 1 | 8.460053 / 1 | 7.805429 / 1 |
| op:N   rmsnorm | nested | 0.024396 / 1 | 0.024085 / 1 | 0.023884 / 1 |
| op:SiTU + sigma | nested | 0.251931 / 1 | 0.099468 / 1 | 0.103105 / 1 |
| op:AR  snapshot aggregate | nested | 0.044523 / 1 | 0.032441 / 1 | 0.032691 / 1 |
| op:SA  softmax attention | nested | 0.006592 / 1 | 0.009377 / 1 | 0.012012 / 1 |
| op:router dot product | nested | 0.596164 / 1 | 0.574684 / 1 | 0.568051 / 1 |
| op:top-k selection | nested | 0.003266 / 1 | 0.002415 / 1 | 0.002354 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.569799 / 1 | 19.154395 / 1 | 18.486577 / 1 |

### Layer 60

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011151 / 1 | 0.003797 / 1 | 0.004067 / 1 |
| pre-attention-aggregation | boundary | 0.016301 / 1 | 0.014998 / 1 | 0.015098 / 1 |
| snapshot-push | boundary | 0.001964 / 1 | 0.001543 / 1 | 0.001663 / 1 |
| pre-attention-normalization | boundary | 0.011120 / 1 | 0.010730 / 1 | 0.010450 / 1 |
| Q | nested | 2.459454 / 1 | 2.430992 / 1 | 2.419190 / 1 |
| K | nested | 2.359878 / 1 | 2.181225 / 1 | 2.234024 / 1 |
| V | nested | 2.226490 / 1 | 2.096788 / 1 | 2.080157 / 1 |
| B | nested | 0.083927 / 1 | 0.021911 / 1 | 0.021150 / 1 |
| FA | nested | 0.057948 / 1 | 0.031950 / 1 | 0.033162 / 1 |
| FB | nested | 0.074669 / 1 | 0.048270 / 1 | 0.046857 / 1 |
| G | nested | 2.164895 / 1 | 1.958960 / 1 | 1.976873 / 1 |
| O | nested | 2.211451 / 1 | 2.035704 / 1 | 2.033599 / 1 |
| attention | boundary | 12.781238 / 1 | 11.831645 / 1 | 11.603138 / 1 |
| attention-residual | boundary | 0.002545 / 1 | 0.002365 / 1 | 0.002545 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028013 / 1 | 0.026529 / 1 | 0.025868 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010499 / 1 | 0.002414 / 1 | 0.002484 / 1 |
| router-and-top16 | boundary | 0.603487 / 1 | 0.568682 / 1 | 0.566498 / 1 |
| EDOWN | nested | 0.674189 / 1 | 0.603437 / 1 | 0.600963 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.675441 / 1 | 0.604619 / 1 | 0.602155 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.029124 / 1 | 0.012202 / 1 | 0.005841 / 1 |
| SH1 | nested | 1.135782 / 1 | 0.982937 / 1 | 0.978958 / 1 |
| SH3 | nested | 1.110725 / 1 | 0.995299 / 1 | 1.004597 / 1 |
| SH2 | nested | 1.112227 / 1 | 1.027319 / 1 | 1.023812 / 1 |
| shared-expert-during-read | boundary | 3.371960 / 1 | 3.018640 / 1 | 3.020644 / 1 |
| detail:expert-gate | nested | 2.741221 / 16 | 2.712587 / 16 | 2.640382 / 16 |
| detail:expert-up | nested | 2.731011 / 16 | 2.675306 / 16 | 2.641574 / 16 |
| detail:expert-activation | nested | 0.089374 / 16 | 0.089607 / 16 | 0.094788 / 16 |
| detail:expert-down | nested | 2.904537 / 16 | 2.867287 / 16 | 2.808456 / 16 |
| EUP | nested | 0.630969 / 1 | 0.594681 / 1 | 0.607975 / 1 |
| experts-mix-normalize-up | boundary | 9.477987 / 1 | 9.309192 / 1 | 9.140357 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002405 / 1 | 0.002875 / 1 | 0.003547 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000160 / 1 | 0.000301 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.413132 / 1 | 0.366935 / 1 | 0.381001 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.227171 / 1 | 113.570236 / 1 | 110.716774 / 1 |
| op:Q   int8 projection | nested | 16.300981 / 1 | 15.007527 / 1 | 15.059654 / 1 |
| op:X   mxfp4 expert proj | nested | 8.528255 / 1 | 8.407156 / 1 | 8.247456 / 1 |
| op:N   rmsnorm | nested | 0.033483 / 1 | 0.038963 / 1 | 0.023155 / 1 |
| op:L   l2 per-head | nested | 0.006352 / 1 | 0.005741 / 1 | 0.005720 / 1 |
| op:SiTU + sigma | nested | 0.097560 / 1 | 0.097923 / 1 | 0.103163 / 1 |
| op:C   shortconv | nested | 0.072185 / 1 | 0.060613 / 1 | 0.035305 / 1 |
| op:AR  snapshot aggregate | nested | 0.033852 / 1 | 0.031479 / 1 | 0.030718 / 1 |
| op:D   kda delta-rule | nested | 0.145773 / 1 | 0.112360 / 1 | 0.132046 / 1 |
| op:router dot product | nested | 0.600342 / 1 | 0.565457 / 1 | 0.563613 / 1 |
| op:top-k selection | nested | 0.002665 / 1 | 0.002855 / 1 | 0.002354 / 1 |
| op:alpha / beta / gate | nested | 0.072095 / 1 | 0.074519 / 1 | 0.058590 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.450203 / 1 | 25.790673 / 1 | 25.398861 / 1 |

### Layer 61

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011993 / 1 | 0.004388 / 1 | 0.004358 / 1 |
| pre-attention-aggregation | boundary | 0.030638 / 1 | 0.018064 / 1 | 0.017783 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012493 / 1 | 0.010370 / 1 | 0.011562 / 1 |
| Q | nested | 2.644851 / 1 | 2.644461 / 1 | 2.622891 / 1 |
| K | nested | 2.485233 / 1 | 2.224376 / 1 | 2.249713 / 1 |
| V | nested | 2.215970 / 1 | 2.129428 / 1 | 2.169243 / 1 |
| B | nested | 0.088435 / 1 | 0.023184 / 1 | 0.024566 / 1 |
| FA | nested | 0.056124 / 1 | 0.025467 / 1 | 0.027331 / 1 |
| FB | nested | 0.068218 / 1 | 0.048300 / 1 | 0.061064 / 1 |
| G | nested | 2.189300 / 1 | 1.996971 / 1 | 2.377591 / 1 |
| O | nested | 2.620065 / 1 | 2.156459 / 1 | 2.333199 / 1 |
| attention | boundary | 13.467170 / 1 | 12.282707 / 1 | 12.622282 / 1 |
| attention-residual | boundary | 0.003397 / 1 | 0.003336 / 1 | 0.003517 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027711 / 1 | 0.026720 / 1 | 0.027040 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010339 / 1 | 0.002665 / 1 | 0.002705 / 1 |
| router-and-top16 | boundary | 0.671374 / 1 | 0.565397 / 1 | 0.619868 / 1 |
| EDOWN | nested | 0.831484 / 1 | 0.592116 / 1 | 0.786810 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.832736 / 1 | 0.593388 / 1 | 0.788343 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027952 / 1 | 0.032510 / 1 | 0.006602 / 1 |
| SH1 | nested | 1.567428 / 1 | 1.034763 / 1 | 1.351044 / 1 |
| SH3 | nested | 1.772752 / 1 | 1.060651 / 1 | 1.402631 / 1 |
| SH2 | nested | 1.500373 / 1 | 0.993527 / 1 | 1.374447 / 1 |
| shared-expert-during-read | boundary | 4.856393 / 1 | 3.101715 / 1 | 4.143992 / 1 |
| detail:expert-gate | nested | 2.789943 / 16 | 2.726070 / 16 | 3.212431 / 16 |
| detail:expert-up | nested | 2.765024 / 16 | 2.706195 / 16 | 3.212923 / 16 |
| detail:expert-activation | nested | 0.094167 / 16 | 0.092562 / 16 | 0.111791 / 16 |
| detail:expert-down | nested | 2.933341 / 16 | 2.883266 / 16 | 3.694985 / 16 |
| EUP | nested | 0.634656 / 1 | 0.588359 / 1 | 0.776591 / 1 |
| experts-mix-normalize-up | boundary | 9.612168 / 1 | 9.390444 / 1 | 11.364291 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002995 / 1 | 0.002504 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000160 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.397212 / 1 | 0.379850 / 1 | 0.379679 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.149896 / 1 | 114.687200 / 1 | 105.588541 / 1 |
| op:Q   int8 projection | nested | 18.673096 / 1 | 15.516377 / 1 | 17.554996 / 1 |
| op:X   mxfp4 expert proj | nested | 8.645873 / 1 | 8.471396 / 1 | 10.294801 / 1 |
| op:N   rmsnorm | nested | 0.033282 / 1 | 0.032179 / 1 | 0.024375 / 1 |
| op:L   l2 per-head | nested | 0.005811 / 1 | 0.005490 / 1 | 0.005239 / 1 |
| op:SiTU + sigma | nested | 0.105067 / 1 | 0.100047 / 1 | 0.123001 / 1 |
| op:C   shortconv | nested | 0.070481 / 1 | 0.053931 / 1 | 0.033031 / 1 |
| op:AR  snapshot aggregate | nested | 0.048270 / 1 | 0.034564 / 1 | 0.034745 / 1 |
| op:D   kda delta-rule | nested | 0.137907 / 1 | 0.106629 / 1 | 0.116508 / 1 |
| op:router dot product | nested | 0.668740 / 1 | 0.562891 / 1 | 0.616772 / 1 |
| op:top-k selection | nested | 0.002194 / 1 | 0.002004 / 1 | 0.002484 / 1 |
| op:alpha / beta / gate | nested | 0.074099 / 1 | 0.072786 / 1 | 0.058379 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 29.979899 / 1 | 26.429306 / 1 | 30.009665 / 1 |

### Layer 62

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014687 / 1 | 0.005871 / 1 | 0.005891 / 1 |
| pre-attention-aggregation | boundary | 0.027011 / 1 | 0.016982 / 1 | 0.017923 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011462 / 1 | 0.010720 / 1 | 0.011631 / 1 |
| Q | nested | 2.470225 / 1 | 2.442543 / 1 | 2.718789 / 1 |
| K | nested | 2.354398 / 1 | 2.197996 / 1 | 2.155827 / 1 |
| V | nested | 2.223535 / 1 | 2.107667 / 1 | 2.093431 / 1 |
| B | nested | 0.084959 / 1 | 0.029254 / 1 | 0.028924 / 1 |
| FA | nested | 0.057127 / 1 | 0.028503 / 1 | 0.028674 / 1 |
| FB | nested | 0.075831 / 1 | 0.049352 / 1 | 0.049151 / 1 |
| G | nested | 2.171296 / 1 | 2.011157 / 1 | 1.999165 / 1 |
| O | nested | 2.238291 / 1 | 2.189641 / 1 | 2.193709 / 1 |
| attention | boundary | 12.758406 / 1 | 12.076402 / 1 | 11.981474 / 1 |
| attention-residual | boundary | 0.002995 / 1 | 0.003467 / 1 | 0.003887 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027421 / 1 | 0.025839 / 1 | 0.025497 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009979 / 1 | 0.002235 / 1 | 0.002284 / 1 |
| router-and-top16 | boundary | 0.608888 / 1 | 0.563473 / 1 | 0.564905 / 1 |
| EDOWN | nested | 0.675632 / 1 | 0.599420 / 1 | 0.594180 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.676915 / 1 | 0.600602 / 1 | 0.595473 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048000 / 1 | 0.006703 / 1 | 0.006973 / 1 |
| SH1 | nested | 1.120723 / 1 | 0.993967 / 1 | 0.991232 / 1 |
| SH3 | nested | 1.107048 / 1 | 1.000649 / 1 | 1.009326 / 1 |
| SH2 | nested | 1.090838 / 1 | 1.005087 / 1 | 1.001241 / 1 |
| shared-expert-during-read | boundary | 3.331214 / 1 | 3.012688 / 1 | 3.014822 / 1 |
| detail:expert-gate | nested | 2.741459 / 16 | 2.730061 / 16 | 2.555294 / 16 |
| detail:expert-up | nested | 2.707729 / 16 | 2.714461 / 16 | 2.560875 / 16 |
| detail:expert-activation | nested | 0.116856 / 16 | 0.091361 / 16 | 0.091919 / 16 |
| detail:expert-down | nested | 2.934321 / 16 | 2.875461 / 16 | 2.744736 / 16 |
| EUP | nested | 0.636609 / 1 | 0.596354 / 1 | 0.610881 / 1 |
| experts-mix-normalize-up | boundary | 9.550652 / 1 | 9.376146 / 1 | 8.895399 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002755 / 1 | 0.003176 / 1 | 0.003166 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000501 / 1 | 0.000251 / 1 | 0.000250 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.377916 / 1 | 0.359712 / 1 | 0.347910 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.467074 / 1 | 114.732877 / 1 | 108.354313 / 1 |
| op:Q   int8 projection | nested | 16.304789 / 1 | 15.249598 / 1 | 15.472798 / 1 |
| op:X   mxfp4 expert proj | nested | 8.564318 / 1 | 8.475835 / 1 | 8.015767 / 1 |
| op:N   rmsnorm | nested | 0.032512 / 1 | 0.034073 / 1 | 0.023854 / 1 |
| op:L   l2 per-head | nested | 0.005440 / 1 | 0.005661 / 1 | 0.005340 / 1 |
| op:SiTU + sigma | nested | 0.124722 / 1 | 0.099775 / 1 | 0.100114 / 1 |
| op:C   shortconv | nested | 0.051317 / 1 | 0.056776 / 1 | 0.034385 / 1 |
| op:AR  snapshot aggregate | nested | 0.044864 / 1 | 0.032571 / 1 | 0.033203 / 1 |
| op:D   kda delta-rule | nested | 0.132237 / 1 | 0.107230 / 1 | 0.115676 / 1 |
| op:router dot product | nested | 0.605841 / 1 | 0.560628 / 1 | 0.561669 / 1 |
| op:top-k selection | nested | 0.002665 / 1 | 0.002495 / 1 | 0.002615 / 1 |
| op:alpha / beta / gate | nested | 0.074550 / 1 | 0.074188 / 1 | 0.059161 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.462276 / 1 | 26.078270 / 1 | 25.491745 / 1 |

### Layer 63

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012743 / 1 | 0.005120 / 1 | 0.004879 / 1 |
| pre-attention-aggregation | boundary | 0.034965 / 1 | 0.018665 / 1 | 0.018133 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012363 / 1 | 0.011180 / 1 | 0.010600 / 1 |
| QA | nested | 0.323184 / 1 | 0.306182 / 1 | 0.316381 / 1 |
| QB | nested | 0.723181 / 1 | 0.718843 / 1 | 0.702803 / 1 |
| KA | nested | 0.125975 / 1 | 0.103243 / 1 | 0.101991 / 1 |
| KB | nested | 0.349543 / 1 | 0.326199 / 1 | 0.316932 / 1 |
| G | nested | 2.197645 / 1 | 1.992302 / 1 | 1.992973 / 1 |
| O | nested | 2.194820 / 1 | 2.017640 / 1 | 2.025534 / 1 |
| attention | boundary | 6.017883 / 1 | 5.563524 / 1 | 5.559296 / 1 |
| attention-residual | boundary | 0.003145 / 1 | 0.003095 / 1 | 0.003095 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026319 / 1 | 0.032170 / 1 | 0.024656 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010430 / 1 | 0.002444 / 1 | 0.002194 / 1 |
| router-and-top16 | boundary | 0.607285 / 1 | 0.573752 / 1 | 0.568873 / 1 |
| EDOWN | nested | 0.633734 / 1 | 0.571187 / 1 | 0.570886 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.635116 / 1 | 0.572389 / 1 | 0.572099 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.033352 / 1 | 0.017122 / 1 | 0.006172 / 1 |
| SH1 | nested | 1.110204 / 1 | 0.994568 / 1 | 0.990250 / 1 |
| SH3 | nested | 1.087672 / 1 | 0.988727 / 1 | 0.985150 / 1 |
| SH2 | nested | 1.099114 / 1 | 0.979209 / 1 | 0.983297 / 1 |
| shared-expert-during-read | boundary | 3.309513 / 1 | 2.975148 / 1 | 2.971742 / 1 |
| detail:expert-gate | nested | 2.750018 / 16 | 2.685716 / 16 | 2.460665 / 16 |
| detail:expert-up | nested | 2.731240 / 16 | 2.696730 / 16 | 2.458355 / 16 |
| detail:expert-activation | nested | 0.089426 / 16 | 0.090959 / 16 | 0.092119 / 16 |
| detail:expert-down | nested | 2.923813 / 16 | 2.865446 / 16 | 2.587816 / 16 |
| EUP | nested | 0.637541 / 1 | 0.604259 / 1 | 0.602646 / 1 |
| experts-mix-normalize-up | boundary | 9.529483 / 1 | 9.314402 / 1 | 8.542479 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002425 / 1 | 0.002735 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000161 / 1 | 0.000190 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006532 / 1 | 0.005851 / 1 | 0.005440 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.540308 / 1 | 114.233404 / 1 | 103.766095 / 1 |
| op:Q   int8 projection | nested | 10.481372 / 1 | 9.601037 / 1 | 9.587481 / 1 |
| op:X   mxfp4 expert proj | nested | 8.559742 / 1 | 8.406026 / 1 | 7.663887 / 1 |
| op:N   rmsnorm | nested | 0.025628 / 1 | 0.026881 / 1 | 0.023124 / 1 |
| op:SiTU + sigma | nested | 0.097321 / 1 | 0.099124 / 1 | 0.100576 / 1 |
| op:AR  snapshot aggregate | nested | 0.050404 / 1 | 0.036898 / 1 | 0.032821 / 1 |
| op:SA  softmax attention | nested | 0.006672 / 1 | 0.008856 / 1 | 0.011923 / 1 |
| op:router dot product | nested | 0.604460 / 1 | 0.571107 / 1 | 0.566358 / 1 |
| op:top-k selection | nested | 0.002354 / 1 | 0.002164 / 1 | 0.002094 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.254941 / 1 | 19.111956 / 1 | 18.306461 / 1 |

### Layer 64

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011551 / 1 | 0.003837 / 1 | 0.003657 / 1 |
| pre-attention-aggregation | boundary | 0.015699 / 1 | 0.014807 / 1 | 0.015349 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011171 / 1 | 0.010479 / 1 | 0.010410 / 1 |
| Q | nested | 2.676340 / 1 | 2.645111 / 1 | 2.642307 / 1 |
| K | nested | 2.496894 / 1 | 2.223504 / 1 | 2.169854 / 1 |
| V | nested | 2.244122 / 1 | 2.088843 / 1 | 2.082571 / 1 |
| B | nested | 0.083556 / 1 | 0.020458 / 1 | 0.020298 / 1 |
| FA | nested | 0.057237 / 1 | 0.025948 / 1 | 0.027672 / 1 |
| FB | nested | 0.076894 / 1 | 0.050505 / 1 | 0.046387 / 1 |
| G | nested | 2.160145 / 1 | 1.990249 / 1 | 1.974129 / 1 |
| O | nested | 2.222021 / 1 | 2.100925 / 1 | 2.098371 / 1 |
| attention | boundary | 13.122386 / 1 | 12.181418 / 1 | 11.748830 / 1 |
| attention-residual | boundary | 0.003346 / 1 | 0.003196 / 1 | 0.003115 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.028703 / 1 | 0.027311 / 1 | 0.026009 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010790 / 1 | 0.002084 / 1 | 0.002796 / 1 |
| router-and-top16 | boundary | 0.610601 / 1 | 0.572469 / 1 | 0.576267 / 1 |
| EDOWN | nested | 0.679509 / 1 | 0.613817 / 1 | 0.610571 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.680892 / 1 | 0.615079 / 1 | 0.611853 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027401 / 1 | 0.022242 / 1 | 0.006412 / 1 |
| SH1 | nested | 1.132987 / 1 | 1.022350 / 1 | 1.017992 / 1 |
| SH3 | nested | 1.121335 / 1 | 1.025767 / 1 | 1.024885 / 1 |
| SH2 | nested | 1.080228 / 1 | 0.981583 / 1 | 0.985451 / 1 |
| shared-expert-during-read | boundary | 3.348256 / 1 | 3.046602 / 1 | 3.041151 / 1 |
| detail:expert-gate | nested | 2.740048 / 16 | 2.724740 / 16 | 2.669457 / 16 |
| detail:expert-up | nested | 2.760756 / 16 | 2.753573 / 16 | 2.665869 / 16 |
| detail:expert-activation | nested | 0.088333 / 16 | 0.093532 / 16 | 0.094337 / 16 |
| detail:expert-down | nested | 2.923573 / 16 | 2.928152 / 16 | 2.810741 / 16 |
| EUP | nested | 0.637881 / 1 | 0.594471 / 1 | 0.600231 / 1 |
| experts-mix-normalize-up | boundary | 9.541797 / 1 | 9.480091 / 1 | 9.184909 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002876 / 1 | 0.002735 / 1 | 0.002605 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000231 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.428922 / 1 | 0.401891 / 1 | 0.397111 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.491719 / 1 | 114.734202 / 1 | 111.624037 / 1 |
| op:Q   int8 projection | nested | 16.667377 / 1 | 15.381927 / 1 | 15.299010 / 1 |
| op:X   mxfp4 expert proj | nested | 8.578797 / 1 | 8.564849 / 1 | 8.305949 / 1 |
| op:N   rmsnorm | nested | 0.034303 / 1 | 0.033983 / 1 | 0.023054 / 1 |
| op:L   l2 per-head | nested | 0.005771 / 1 | 0.005851 / 1 | 0.005590 / 1 |
| op:SiTU + sigma | nested | 0.097039 / 1 | 0.104552 / 1 | 0.102132 / 1 |
| op:C   shortconv | nested | 0.067616 / 1 | 0.068005 / 1 | 0.035235 / 1 |
| op:AR  snapshot aggregate | nested | 0.034194 / 1 | 0.032009 / 1 | 0.031369 / 1 |
| op:D   kda delta-rule | nested | 0.147215 / 1 | 0.111849 / 1 | 0.117699 / 1 |
| op:router dot product | nested | 0.607846 / 1 | 0.569484 / 1 | 0.573691 / 1 |
| op:top-k selection | nested | 0.002375 / 1 | 0.002625 / 1 | 0.002224 / 1 |
| op:alpha / beta / gate | nested | 0.071944 / 1 | 0.074449 / 1 | 0.058399 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.859858 / 1 | 26.399019 / 1 | 25.644530 / 1 |

### Layer 65

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012494 / 1 | 0.004909 / 1 | 0.005139 / 1 |
| pre-attention-aggregation | boundary | 0.017854 / 1 | 0.017213 / 1 | 0.016270 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010961 / 1 | 0.010289 / 1 | 0.011481 / 1 |
| Q | nested | 2.482829 / 1 | 2.463212 / 1 | 2.443374 / 1 |
| K | nested | 2.334311 / 1 | 2.187717 / 1 | 2.175424 / 1 |
| V | nested | 2.199729 / 1 | 2.107167 / 1 | 2.101677 / 1 |
| B | nested | 0.094767 / 1 | 0.026369 / 1 | 0.027622 / 1 |
| FA | nested | 0.056306 / 1 | 0.026339 / 1 | 0.031870 / 1 |
| FB | nested | 0.080861 / 1 | 0.051295 / 1 | 0.046527 / 1 |
| G | nested | 2.175415 / 1 | 1.961374 / 1 | 1.954902 / 1 |
| O | nested | 2.208837 / 1 | 2.114189 / 1 | 2.112948 / 1 |
| attention | boundary | 12.714634 / 1 | 11.970003 / 1 | 11.600151 / 1 |
| attention-residual | boundary | 0.003466 / 1 | 0.003397 / 1 | 0.004098 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027452 / 1 | 0.025888 / 1 | 0.025458 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010540 / 1 | 0.002444 / 1 | 0.002524 / 1 |
| router-and-top16 | boundary | 0.605251 / 1 | 0.572409 / 1 | 0.570837 / 1 |
| EDOWN | nested | 0.678728 / 1 | 0.633905 / 1 | 0.626300 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.679980 / 1 | 0.635187 / 1 | 0.627663 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048751 / 1 | 0.022812 / 1 | 0.013204 / 1 |
| SH1 | nested | 1.122668 / 1 | 0.988347 / 1 | 0.986423 / 1 |
| SH3 | nested | 1.101287 / 1 | 0.980382 / 1 | 0.974050 / 1 |
| SH2 | nested | 1.091359 / 1 | 0.990851 / 1 | 0.987996 / 1 |
| shared-expert-during-read | boundary | 3.328037 / 1 | 2.972263 / 1 | 2.961663 / 1 |
| detail:expert-gate | nested | 2.971941 / 16 | 2.750942 / 16 | 2.511490 / 16 |
| detail:expert-up | nested | 3.075858 / 16 | 2.729511 / 16 | 2.478891 / 16 |
| detail:expert-activation | nested | 0.097976 / 16 | 0.090693 / 16 | 0.093142 / 16 |
| detail:expert-down | nested | 2.960067 / 16 | 2.908484 / 16 | 2.607022 / 16 |
| EUP | nested | 0.633694 / 1 | 0.594281 / 1 | 0.597156 / 1 |
| experts-mix-normalize-up | boundary | 10.164760 / 1 | 9.465283 / 1 | 8.637707 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002294 / 1 | 0.002855 / 1 | 0.002785 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000151 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.410216 / 1 | 0.399146 / 1 | 0.395078 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 116.307520 / 1 | 114.951304 / 1 | 103.549226 / 1 |
| op:Q   int8 projection | nested | 16.259194 / 1 | 15.123925 / 1 | 15.064433 / 1 |
| op:X   mxfp4 expert proj | nested | 9.172106 / 1 | 8.545847 / 1 | 7.757531 / 1 |
| op:N   rmsnorm | nested | 0.038380 / 1 | 0.038652 / 1 | 0.023736 / 1 |
| op:L   l2 per-head | nested | 0.005991 / 1 | 0.005841 / 1 | 0.006382 / 1 |
| op:SiTU + sigma | nested | 0.105840 / 1 | 0.098655 / 1 | 0.101386 / 1 |
| op:C   shortconv | nested | 0.068097 / 1 | 0.059722 / 1 | 0.034495 / 1 |
| op:AR  snapshot aggregate | nested | 0.034894 / 1 | 0.032711 / 1 | 0.031940 / 1 |
| op:D   kda delta-rule | nested | 0.129673 / 1 | 0.110807 / 1 | 0.110587 / 1 |
| op:router dot product | nested | 0.602566 / 1 | 0.569504 / 1 | 0.567030 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002555 / 1 | 0.003326 / 1 |
| op:alpha / beta / gate | nested | 0.073818 / 1 | 0.074079 / 1 | 0.059090 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.051326 / 1 | 26.118235 / 1 | 24.888477 / 1 |

### Layer 66

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.015239 / 1 | 0.004959 / 1 | 0.005089 / 1 |
| pre-attention-aggregation | boundary | 0.023484 / 1 | 0.016761 / 1 | 0.016741 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011010 / 1 | 0.010199 / 1 | 0.010880 / 1 |
| Q | nested | 2.484432 / 1 | 2.532521 / 1 | 2.429309 / 1 |
| K | nested | 2.346053 / 1 | 2.191784 / 1 | 2.180554 / 1 |
| V | nested | 2.194420 / 1 | 2.109030 / 1 | 2.105354 / 1 |
| B | nested | 0.095999 / 1 | 0.023424 / 1 | 0.022222 / 1 |
| FA | nested | 0.058479 / 1 | 0.027341 / 1 | 0.028614 / 1 |
| FB | nested | 0.074199 / 1 | 0.048611 / 1 | 0.051085 / 1 |
| G | nested | 2.174602 / 1 | 2.020975 / 1 | 2.001429 / 1 |
| O | nested | 2.201623 / 1 | 2.070528 / 1 | 2.077952 / 1 |
| attention | boundary | 12.698063 / 1 | 12.049040 / 1 | 11.609449 / 1 |
| attention-residual | boundary | 0.004007 / 1 | 0.003466 / 1 | 0.003958 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027131 / 1 | 0.025789 / 1 | 0.025297 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009778 / 1 | 0.002555 / 1 | 0.002355 / 1 |
| router-and-top16 | boundary | 0.615890 / 1 | 0.566889 / 1 | 0.569133 / 1 |
| EDOWN | nested | 0.674099 / 1 | 0.602376 / 1 | 0.602857 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.675442 / 1 | 0.603658 / 1 | 0.604199 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.028724 / 1 | 0.006001 / 1 | 0.012793 / 1 |
| SH1 | nested | 1.118881 / 1 | 0.975843 / 1 | 0.973729 / 1 |
| SH3 | nested | 1.144628 / 1 | 1.059750 / 1 | 1.050152 / 1 |
| SH2 | nested | 1.125092 / 1 | 1.012091 / 1 | 1.018633 / 1 |
| shared-expert-during-read | boundary | 3.401675 / 1 | 3.060758 / 1 | 3.055098 / 1 |
| detail:expert-gate | nested | 2.729271 / 16 | 2.716085 / 16 | 2.521551 / 16 |
| detail:expert-up | nested | 2.733126 / 16 | 2.716433 / 16 | 2.518323 / 16 |
| detail:expert-activation | nested | 0.090068 / 16 | 0.091421 / 16 | 0.094518 / 16 |
| detail:expert-down | nested | 2.936013 / 16 | 2.887113 / 16 | 2.651583 / 16 |
| EUP | nested | 0.637671 / 1 | 0.599961 / 1 | 0.606994 / 1 |
| experts-mix-normalize-up | boundary | 9.519825 / 1 | 9.382218 / 1 | 8.773260 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002595 / 1 | 0.002956 / 1 | 0.002655 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000231 / 1 | 0.000271 / 1 | 0.000241 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.394737 / 1 | 0.377034 / 1 | 0.374119 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.843710 / 1 | 115.011201 / 1 | 106.344824 / 1 |
| op:Q   int8 projection | nested | 16.328224 / 1 | 15.272664 / 1 | 15.147249 / 1 |
| op:X   mxfp4 expert proj | nested | 8.556415 / 1 | 8.477920 / 1 | 7.854394 / 1 |
| op:N   rmsnorm | nested | 0.039685 / 1 | 0.042029 / 1 | 0.023805 / 1 |
| op:L   l2 per-head | nested | 0.005460 / 1 | 0.005440 / 1 | 0.005810 / 1 |
| op:SiTU + sigma | nested | 0.098163 / 1 | 0.099577 / 1 | 0.102292 / 1 |
| op:C   shortconv | nested | 0.055113 / 1 | 0.057056 / 1 | 0.036850 / 1 |
| op:AR  snapshot aggregate | nested | 0.040316 / 1 | 0.032430 / 1 | 0.031950 / 1 |
| op:D   kda delta-rule | nested | 0.136405 / 1 | 0.111919 / 1 | 0.111719 / 1 |
| op:router dot product | nested | 0.612885 / 1 | 0.564214 / 1 | 0.566378 / 1 |
| op:top-k selection | nested | 0.002565 / 1 | 0.002325 / 1 | 0.002325 / 1 |
| op:alpha / beta / gate | nested | 0.076373 / 1 | 0.073678 / 1 | 0.059020 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.442039 / 1 | 26.126861 / 1 | 25.079434 / 1 |

### Layer 67

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012563 / 1 | 0.004368 / 1 | 0.003958 / 1 |
| pre-attention-aggregation | boundary | 0.022602 / 1 | 0.022071 / 1 | 0.020398 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010910 / 1 | 0.010710 / 1 | 0.010901 / 1 |
| QA | nested | 0.319617 / 1 | 0.317043 / 1 | 0.325147 / 1 |
| QB | nested | 0.732920 / 1 | 0.691883 / 1 | 0.703695 / 1 |
| KA | nested | 0.127078 / 1 | 0.100899 / 1 | 0.099126 / 1 |
| KB | nested | 0.347168 / 1 | 0.305862 / 1 | 0.326641 / 1 |
| G | nested | 2.171717 / 1 | 1.987453 / 1 | 1.992523 / 1 |
| O | nested | 2.187256 / 1 | 1.982293 / 1 | 1.983114 / 1 |
| attention | boundary | 5.993968 / 1 | 5.484497 / 1 | 5.530653 / 1 |
| attention-residual | boundary | 0.002725 / 1 | 0.002675 / 1 | 0.003216 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027301 / 1 | 0.026299 / 1 | 0.025637 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010790 / 1 | 0.002014 / 1 | 0.002124 / 1 |
| router-and-top16 | boundary | 0.598578 / 1 | 0.570375 / 1 | 0.570706 / 1 |
| EDOWN | nested | 0.642350 / 1 | 0.576327 / 1 | 0.580264 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.643913 / 1 | 0.577669 / 1 | 0.581477 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027501 / 1 | 0.005871 / 1 | 0.006092 / 1 |
| SH1 | nested | 1.097290 / 1 | 0.985261 / 1 | 0.986604 / 1 |
| SH3 | nested | 1.112949 / 1 | 1.038901 / 1 | 1.030796 / 1 |
| SH2 | nested | 1.091018 / 1 | 0.989719 / 1 | 0.990681 / 1 |
| shared-expert-during-read | boundary | 3.314562 / 1 | 3.026905 / 1 | 3.020864 / 1 |
| detail:expert-gate | nested | 2.719077 / 16 | 2.666859 / 16 | 2.508766 / 16 |
| detail:expert-up | nested | 2.740932 / 16 | 2.677594 / 16 | 2.493617 / 16 |
| detail:expert-activation | nested | 0.090348 / 16 | 0.090444 / 16 | 0.092712 / 16 |
| detail:expert-down | nested | 2.907972 / 16 | 2.874483 / 16 | 2.669779 / 16 |
| EUP | nested | 0.631320 / 1 | 0.606173 / 1 | 0.596615 / 1 |
| experts-mix-normalize-up | boundary | 9.474190 / 1 | 9.280257 / 1 | 8.716514 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002685 / 1 | 0.002815 / 1 | 0.003296 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000150 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006863 / 1 | 0.005520 / 1 | 0.005611 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.670149 / 1 | 113.642929 / 1 | 106.322366 / 1 |
| op:Q   int8 projection | nested | 10.459490 / 1 | 9.580420 / 1 | 9.613940 / 1 |
| op:X   mxfp4 expert proj | nested | 8.525715 / 1 | 8.377911 / 1 | 7.832821 / 1 |
| op:N   rmsnorm | nested | 0.024475 / 1 | 0.023593 / 1 | 0.024013 / 1 |
| op:SiTU + sigma | nested | 0.099014 / 1 | 0.098551 / 1 | 0.100649 / 1 |
| op:AR  snapshot aggregate | nested | 0.038802 / 1 | 0.037391 / 1 | 0.035706 / 1 |
| op:SA  softmax attention | nested | 0.006783 / 1 | 0.008747 / 1 | 0.011542 / 1 |
| op:router dot product | nested | 0.595382 / 1 | 0.567560 / 1 | 0.567971 / 1 |
| op:top-k selection | nested | 0.002655 / 1 | 0.002314 / 1 | 0.002505 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.165564 / 1 | 19.036174 / 1 | 18.516052 / 1 |

### Layer 68

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011982 / 1 | 0.003387 / 1 | 0.003376 / 1 |
| pre-attention-aggregation | boundary | 0.016280 / 1 | 0.014818 / 1 | 0.014737 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011852 / 1 | 0.011171 / 1 | 0.010910 / 1 |
| Q | nested | 2.449937 / 1 | 2.423978 / 1 | 2.416975 / 1 |
| K | nested | 2.334692 / 1 | 2.179181 / 1 | 2.157480 / 1 |
| V | nested | 2.168882 / 1 | 2.084474 / 1 | 2.098931 / 1 |
| B | nested | 0.069279 / 1 | 0.024937 / 1 | 0.025167 / 1 |
| FA | nested | 0.057938 / 1 | 0.035256 / 1 | 0.038031 / 1 |
| FB | nested | 0.066134 / 1 | 0.051706 / 1 | 0.049673 / 1 |
| G | nested | 2.159385 / 1 | 1.944573 / 1 | 1.945976 / 1 |
| O | nested | 2.227942 / 1 | 2.126633 / 1 | 2.114170 / 1 |
| attention | boundary | 12.691089 / 1 | 11.921552 / 1 | 11.562591 / 1 |
| attention-residual | boundary | 0.003216 / 1 | 0.003567 / 1 | 0.003727 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026880 / 1 | 0.026560 / 1 | 0.025458 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010479 / 1 | 0.002535 / 1 | 0.001813 / 1 |
| router-and-top16 | boundary | 0.635588 / 1 | 0.583260 / 1 | 0.583250 / 1 |
| EDOWN | nested | 0.674289 / 1 | 0.603848 / 1 | 0.597356 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.675642 / 1 | 0.605270 / 1 | 0.598658 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027772 / 1 | 0.006492 / 1 | 0.012914 / 1 |
| SH1 | nested | 1.136282 / 1 | 0.995420 / 1 | 0.993546 / 1 |
| SH3 | nested | 1.106587 / 1 | 1.014235 / 1 | 1.026558 / 1 |
| SH2 | nested | 1.086440 / 1 | 0.982496 / 1 | 0.986803 / 1 |
| shared-expert-during-read | boundary | 3.345100 / 1 | 3.005124 / 1 | 3.019922 / 1 |
| detail:expert-gate | nested | 2.714351 / 16 | 2.689806 / 16 | 2.585388 / 16 |
| detail:expert-up | nested | 2.730322 / 16 | 2.672985 / 16 | 2.562458 / 16 |
| detail:expert-activation | nested | 0.090790 / 16 | 0.090448 / 16 | 0.093635 / 16 |
| detail:expert-down | nested | 2.941736 / 16 | 2.873797 / 16 | 2.710525 / 16 |
| EUP | nested | 0.628183 / 1 | 0.589572 / 1 | 0.595864 / 1 |
| experts-mix-normalize-up | boundary | 9.503074 / 1 | 9.293332 / 1 | 8.874590 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002415 / 1 | 0.002956 / 1 | 0.003166 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000170 / 1 | 0.000170 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.383867 / 1 | 0.367848 / 1 | 0.353741 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 114.926775 / 1 | 113.727513 / 1 | 108.167039 / 1 |
| op:Q   int8 projection | nested | 16.164027 / 1 | 15.054639 / 1 | 15.044896 / 1 |
| op:X   mxfp4 expert proj | nested | 8.546877 / 1 | 8.400003 / 1 | 8.021638 / 1 |
| op:N   rmsnorm | nested | 0.034625 / 1 | 0.035296 / 1 | 0.024205 / 1 |
| op:L   l2 per-head | nested | 0.005651 / 1 | 0.005651 / 1 | 0.006032 / 1 |
| op:SiTU + sigma | nested | 0.101049 / 1 | 0.098463 / 1 | 0.101560 / 1 |
| op:C   shortconv | nested | 0.083005 / 1 | 0.073277 / 1 | 0.034505 / 1 |
| op:AR  snapshot aggregate | nested | 0.033222 / 1 | 0.031029 / 1 | 0.030176 / 1 |
| op:D   kda delta-rule | nested | 0.149950 / 1 | 0.118381 / 1 | 0.118050 / 1 |
| op:router dot product | nested | 0.632642 / 1 | 0.580033 / 1 | 0.580425 / 1 |
| op:top-k selection | nested | 0.002344 / 1 | 0.002544 / 1 | 0.002384 / 1 |
| op:alpha / beta / gate | nested | 0.073988 / 1 | 0.072125 / 1 | 0.058840 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.360245 / 1 | 25.862758 / 1 | 25.083822 / 1 |

### Layer 69

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012263 / 1 | 0.004919 / 1 | 0.004689 / 1 |
| pre-attention-aggregation | boundary | 0.020428 / 1 | 0.018544 / 1 | 0.018544 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000021 / 1 |
| pre-attention-normalization | boundary | 0.012403 / 1 | 0.010950 / 1 | 0.011091 / 1 |
| Q | nested | 2.687250 / 1 | 2.668646 / 1 | 2.662223 / 1 |
| K | nested | 2.479122 / 1 | 2.213455 / 1 | 2.177758 / 1 |
| V | nested | 2.246146 / 1 | 2.092589 / 1 | 2.097017 / 1 |
| B | nested | 0.064340 / 1 | 0.020919 / 1 | 0.020578 / 1 |
| FA | nested | 0.059802 / 1 | 0.035055 / 1 | 0.033072 / 1 |
| FB | nested | 0.077534 / 1 | 0.050565 / 1 | 0.048941 / 1 |
| G | nested | 2.210039 / 1 | 2.069857 / 1 | 2.060960 / 1 |
| O | nested | 2.227672 / 1 | 2.087460 / 1 | 2.067032 / 1 |
| attention | boundary | 13.159164 / 1 | 12.288307 / 1 | 11.881387 / 1 |
| attention-residual | boundary | 0.003236 / 1 | 0.003326 / 1 | 0.003095 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027471 / 1 | 0.026640 / 1 | 0.026380 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010049 / 1 | 0.002414 / 1 | 0.002034 / 1 |
| router-and-top16 | boundary | 0.608256 / 1 | 0.578080 / 1 | 0.571959 / 1 |
| EDOWN | nested | 0.652389 / 1 | 0.588169 / 1 | 0.587147 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.653741 / 1 | 0.589512 / 1 | 0.588549 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.028333 / 1 | 0.012444 / 1 | 0.006582 / 1 |
| SH1 | nested | 1.144218 / 1 | 0.974170 / 1 | 0.969912 / 1 |
| SH3 | nested | 1.094415 / 1 | 0.991963 / 1 | 0.993957 / 1 |
| SH2 | nested | 1.085949 / 1 | 1.025276 / 1 | 1.015658 / 1 |
| shared-expert-during-read | boundary | 3.337615 / 1 | 3.005285 / 1 | 2.992601 / 1 |
| detail:expert-gate | nested | 2.725903 / 16 | 2.714161 / 16 | 2.516717 / 16 |
| detail:expert-up | nested | 2.758926 / 16 | 2.723808 / 16 | 2.480093 / 16 |
| detail:expert-activation | nested | 0.095359 / 16 | 0.090290 / 16 | 0.093574 / 16 |
| detail:expert-down | nested | 2.891684 / 16 | 2.859094 / 16 | 2.656892 / 16 |
| EUP | nested | 0.639685 / 1 | 0.605812 / 1 | 0.585364 / 1 |
| experts-mix-normalize-up | boundary | 9.518102 / 1 | 9.368933 / 1 | 8.702659 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002244 / 1 | 0.002675 / 1 | 0.002554 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000200 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.401991 / 1 | 0.378186 / 1 | 0.375832 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.136691 / 1 | 114.701626 / 1 | 105.630166 / 1 |
| op:Q   int8 projection | nested | 16.666775 / 1 | 15.422112 / 1 | 15.317927 / 1 |
| op:X   mxfp4 expert proj | nested | 8.542660 / 1 | 8.456910 / 1 | 7.817560 / 1 |
| op:N   rmsnorm | nested | 0.032000 / 1 | 0.033503 / 1 | 0.024647 / 1 |
| op:L   l2 per-head | nested | 0.005760 / 1 | 0.005911 / 1 | 0.006142 / 1 |
| op:SiTU + sigma | nested | 0.103813 / 1 | 0.098796 / 1 | 0.101660 / 1 |
| op:C   shortconv | nested | 0.066755 / 1 | 0.063429 / 1 | 0.034264 / 1 |
| op:AR  snapshot aggregate | nested | 0.038232 / 1 | 0.035156 / 1 | 0.034705 / 1 |
| op:D   kda delta-rule | nested | 0.142917 / 1 | 0.116538 / 1 | 0.112811 / 1 |
| op:router dot product | nested | 0.605291 / 1 | 0.575075 / 1 | 0.569414 / 1 |
| op:top-k selection | nested | 0.002595 / 1 | 0.002575 / 1 | 0.002084 / 1 |
| op:alpha / beta / gate | nested | 0.074509 / 1 | 0.073007 / 1 | 0.058659 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.812139 / 1 | 26.306596 / 1 | 25.204799 / 1 |

### Layer 70

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013596 / 1 | 0.005521 / 1 | 0.005711 / 1 |
| pre-attention-aggregation | boundary | 0.024305 / 1 | 0.016952 / 1 | 0.017472 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012022 / 1 | 0.012703 / 1 | 0.010800 / 1 |
| Q | nested | 2.469364 / 1 | 2.440890 / 1 | 2.426533 / 1 |
| K | nested | 2.335023 / 1 | 2.185754 / 1 | 2.174452 / 1 |
| V | nested | 2.199249 / 1 | 2.099823 / 1 | 2.110774 / 1 |
| B | nested | 0.067676 / 1 | 0.024255 / 1 | 0.023785 / 1 |
| FA | nested | 0.057688 / 1 | 0.030717 / 1 | 0.034365 / 1 |
| FB | nested | 0.070362 / 1 | 0.047549 / 1 | 0.047258 / 1 |
| G | nested | 2.170034 / 1 | 1.955353 / 1 | 1.932380 / 1 |
| O | nested | 2.268007 / 1 | 2.290760 / 1 | 2.289808 / 1 |
| attention | boundary | 12.723260 / 1 | 12.127928 / 1 | 11.782262 / 1 |
| attention-residual | boundary | 0.003066 / 1 | 0.003526 / 1 | 0.003897 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027001 / 1 | 0.027211 / 1 | 0.025538 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010109 / 1 | 0.002244 / 1 | 0.002434 / 1 |
| router-and-top16 | boundary | 0.605111 / 1 | 0.572740 / 1 | 0.570646 / 1 |
| EDOWN | nested | 0.653642 / 1 | 0.579433 / 1 | 0.567360 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.655084 / 1 | 0.580785 / 1 | 0.568723 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043221 / 1 | 0.022472 / 1 | 0.013656 / 1 |
| SH1 | nested | 1.138988 / 1 | 1.007122 / 1 | 1.002242 / 1 |
| SH3 | nested | 1.119221 / 1 | 1.029033 / 1 | 1.009195 / 1 |
| SH2 | nested | 1.083384 / 1 | 1.000960 / 1 | 0.999568 / 1 |
| shared-expert-during-read | boundary | 3.355438 / 1 | 3.050619 / 1 | 3.024150 / 1 |
| detail:expert-gate | nested | 2.712878 / 16 | 2.729059 / 16 | 2.626516 / 16 |
| detail:expert-up | nested | 2.697080 / 16 | 2.704643 / 16 | 2.628442 / 16 |
| detail:expert-activation | nested | 0.089575 / 16 | 0.091971 / 16 | 0.098015 / 16 |
| detail:expert-down | nested | 2.924335 / 16 | 2.900256 / 16 | 2.747825 / 16 |
| EUP | nested | 0.630418 / 1 | 0.590473 / 1 | 0.592346 / 1 |
| experts-mix-normalize-up | boundary | 9.468760 / 1 | 9.414849 / 1 | 9.018969 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002264 / 1 | 0.003136 / 1 | 0.002655 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000251 / 1 | 0.000231 / 1 | 0.000240 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.397833 / 1 | 0.365342 / 1 | 0.370532 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.129292 / 1 | 114.980016 / 1 | 110.323583 / 1 |
| op:Q   int8 projection | nested | 16.260918 / 1 | 15.280127 / 1 | 15.208532 / 1 |
| op:X   mxfp4 expert proj | nested | 8.497956 / 1 | 8.497367 / 1 | 8.172316 / 1 |
| op:N   rmsnorm | nested | 0.034304 / 1 | 0.040817 / 1 | 0.023364 / 1 |
| op:L   l2 per-head | nested | 0.005511 / 1 | 0.005971 / 1 | 0.007003 / 1 |
| op:SiTU + sigma | nested | 0.098412 / 1 | 0.100425 / 1 | 0.106009 / 1 |
| op:C   shortconv | nested | 0.052378 / 1 | 0.054872 / 1 | 0.032200 / 1 |
| op:AR  snapshot aggregate | nested | 0.041187 / 1 | 0.033643 / 1 | 0.032550 / 1 |
| op:D   kda delta-rule | nested | 0.136124 / 1 | 0.114564 / 1 | 0.110847 / 1 |
| op:router dot product | nested | 0.601864 / 1 | 0.569695 / 1 | 0.568072 / 1 |
| op:top-k selection | nested | 0.002465 / 1 | 0.002655 / 1 | 0.002064 / 1 |
| op:alpha / beta / gate | nested | 0.074810 / 1 | 0.073898 / 1 | 0.059181 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.359144 / 1 | 26.221498 / 1 | 25.432905 / 1 |

### Layer 71

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012924 / 1 | 0.004899 / 1 | 0.005150 / 1 |
| pre-attention-aggregation | boundary | 0.021220 / 1 | 0.017382 / 1 | 0.017303 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012253 / 1 | 0.011622 / 1 | 0.013345 / 1 |
| QA | nested | 0.331890 / 1 | 0.319096 / 1 | 0.312373 / 1 |
| QB | nested | 0.869264 / 1 | 0.690400 / 1 | 0.705248 / 1 |
| KA | nested | 0.145361 / 1 | 0.102361 / 1 | 0.103143 / 1 |
| KB | nested | 0.407341 / 1 | 0.310450 / 1 | 0.302455 / 1 |
| G | nested | 2.527232 / 1 | 2.022779 / 1 | 2.023631 / 1 |
| O | nested | 2.181475 / 1 | 1.954541 / 1 | 1.932290 / 1 |
| attention | boundary | 6.573370 / 1 | 5.500787 / 1 | 5.480309 / 1 |
| attention-residual | boundary | 0.002755 / 1 | 0.002665 / 1 | 0.002995 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026290 / 1 | 0.026469 / 1 | 0.025157 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010359 / 1 | 0.002014 / 1 | 0.002384 / 1 |
| router-and-top16 | boundary | 0.603527 / 1 | 0.572048 / 1 | 0.567150 / 1 |
| EDOWN | nested | 0.634816 / 1 | 0.575866 / 1 | 0.566178 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.636188 / 1 | 0.577208 / 1 | 0.567510 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.016942 / 1 | 0.011872 / 1 | 0.005901 / 1 |
| SH1 | nested | 1.105916 / 1 | 1.009937 / 1 | 1.016309 / 1 |
| SH3 | nested | 1.116757 / 1 | 1.012251 / 1 | 1.031287 / 1 |
| SH2 | nested | 1.104133 / 1 | 1.007963 / 1 | 0.992644 / 1 |
| shared-expert-during-read | boundary | 3.340320 / 1 | 3.042785 / 1 | 3.052974 / 1 |
| detail:expert-gate | nested | 2.704393 / 16 | 2.686209 / 16 | 2.538173 / 16 |
| detail:expert-up | nested | 2.710313 / 16 | 2.677953 / 16 | 2.514137 / 16 |
| detail:expert-activation | nested | 0.090981 / 16 | 0.091120 / 16 | 0.094197 / 16 |
| detail:expert-down | nested | 2.882754 / 16 | 2.887315 / 16 | 2.658575 / 16 |
| EUP | nested | 0.630107 / 1 | 0.611131 / 1 | 0.615180 / 1 |
| experts-mix-normalize-up | boundary | 9.397978 / 1 | 9.330512 / 1 | 8.778741 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003257 / 1 | 0.003456 / 1 | 0.003326 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000160 / 1 | 0.000161 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005601 / 1 | 0.006052 / 1 | 0.005901 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 114.580076 / 1 | 114.553125 / 1 | 106.367881 / 1 |
| op:Q   int8 projection | nested | 11.052806 / 1 | 9.615492 / 1 | 9.599455 / 1 |
| op:X   mxfp4 expert proj | nested | 8.459476 / 1 | 8.413767 / 1 | 7.876064 / 1 |
| op:N   rmsnorm | nested | 0.025338 / 1 | 0.024627 / 1 | 0.024976 / 1 |
| op:SiTU + sigma | nested | 0.099818 / 1 | 0.098725 / 1 | 0.102224 / 1 |
| op:AR  snapshot aggregate | nested | 0.036528 / 1 | 0.032801 / 1 | 0.031869 / 1 |
| op:SA  softmax attention | nested | 0.007314 / 1 | 0.009037 / 1 | 0.012173 / 1 |
| op:router dot product | nested | 0.600432 / 1 | 0.568993 / 1 | 0.564434 / 1 |
| op:top-k selection | nested | 0.002435 / 1 | 0.002574 / 1 | 0.002395 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.678272 / 1 | 19.124850 / 1 | 18.543173 / 1 |

### Layer 72

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010770 / 1 | 0.003927 / 1 | 0.003336 / 1 |
| pre-attention-aggregation | boundary | 0.015398 / 1 | 0.014968 / 1 | 0.015318 / 1 |
| snapshot-push | boundary | 0.001724 / 1 | 0.001673 / 1 | 0.001864 / 1 |
| pre-attention-normalization | boundary | 0.010860 / 1 | 0.010419 / 1 | 0.011311 / 1 |
| Q | nested | 2.467760 / 1 | 2.465205 / 1 | 2.452902 / 1 |
| K | nested | 2.270572 / 1 | 2.185574 / 1 | 2.163943 / 1 |
| V | nested | 2.235707 / 1 | 2.104953 / 1 | 2.078343 / 1 |
| B | nested | 0.068678 / 1 | 0.020769 / 1 | 0.020418 / 1 |
| FA | nested | 0.057647 / 1 | 0.032220 / 1 | 0.030126 / 1 |
| FB | nested | 0.067867 / 1 | 0.046717 / 1 | 0.047369 / 1 |
| G | nested | 2.160226 / 1 | 1.994897 / 1 | 1.984427 / 1 |
| O | nested | 2.300989 / 1 | 2.120001 / 1 | 2.122205 / 1 |
| attention | boundary | 12.693194 / 1 | 12.037699 / 1 | 11.599681 / 1 |
| attention-residual | boundary | 0.002454 / 1 | 0.002725 / 1 | 0.002174 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027341 / 1 | 0.026680 / 1 | 0.025708 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009478 / 1 | 0.002455 / 1 | 0.002304 / 1 |
| router-and-top16 | boundary | 0.596905 / 1 | 0.572139 / 1 | 0.563383 / 1 |
| EDOWN | nested | 0.645607 / 1 | 0.609649 / 1 | 0.604399 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.647009 / 1 | 0.611001 / 1 | 0.605742 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031639 / 1 | 0.006312 / 1 | 0.006332 / 1 |
| SH1 | nested | 1.102059 / 1 | 1.008103 / 1 | 1.006270 / 1 |
| SH3 | nested | 1.108822 / 1 | 1.021358 / 1 | 1.008595 / 1 |
| SH2 | nested | 1.082723 / 1 | 1.016639 / 1 | 1.007202 / 1 |
| shared-expert-during-read | boundary | 3.307048 / 1 | 3.059626 / 1 | 3.035110 / 1 |
| detail:expert-gate | nested | 2.695958 / 16 | 2.690766 / 16 | 2.699295 / 16 |
| detail:expert-up | nested | 2.685970 / 16 | 2.710285 / 16 | 2.704831 / 16 |
| detail:expert-activation | nested | 0.090158 / 16 | 0.090547 / 16 | 0.090822 / 16 |
| detail:expert-down | nested | 2.906830 / 16 | 2.876935 / 16 | 2.873828 / 16 |
| EUP | nested | 0.625499 / 1 | 0.594060 / 1 | 0.592808 / 1 |
| experts-mix-normalize-up | boundary | 9.402516 / 1 | 9.338747 / 1 | 9.313720 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003487 / 1 | 0.002925 / 1 | 0.002916 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000140 / 1 | 0.000141 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.383166 / 1 | 0.393856 / 1 | 0.386512 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.028912 / 1 | 114.895958 / 1 | 114.835232 / 1 |
| op:Q   int8 projection | nested | 16.192491 / 1 | 15.218563 / 1 | 15.117394 / 1 |
| op:X   mxfp4 expert proj | nested | 8.451182 / 1 | 8.442441 / 1 | 8.441011 / 1 |
| op:N   rmsnorm | nested | 0.030016 / 1 | 0.035576 / 1 | 0.023564 / 1 |
| op:L   l2 per-head | nested | 0.005501 / 1 | 0.005570 / 1 | 0.006041 / 1 |
| op:SiTU + sigma | nested | 0.098935 / 1 | 0.098994 / 1 | 0.098939 / 1 |
| op:C   shortconv | nested | 0.072225 / 1 | 0.067195 / 1 | 0.034163 / 1 |
| op:AR  snapshot aggregate | nested | 0.032610 / 1 | 0.031489 / 1 | 0.030918 / 1 |
| op:D   kda delta-rule | nested | 0.135783 / 1 | 0.130684 / 1 | 0.118091 / 1 |
| op:router dot product | nested | 0.593940 / 1 | 0.569614 / 1 | 0.560858 / 1 |
| op:top-k selection | nested | 0.002445 / 1 | 0.002164 / 1 | 0.002205 / 1 |
| op:alpha / beta / gate | nested | 0.073838 / 1 | 0.074789 / 1 | 0.058409 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.159340 / 1 | 26.101644 / 1 | 25.591460 / 1 |

### Layer 73

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010640 / 1 | 0.005120 / 1 | 0.005380 / 1 |
| pre-attention-aggregation | boundary | 0.018585 / 1 | 0.017893 / 1 | 0.019487 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011531 / 1 | 0.011842 / 1 | 0.011351 / 1 |
| Q | nested | 2.472068 / 1 | 2.478931 / 1 | 2.455327 / 1 |
| K | nested | 2.275551 / 1 | 2.203497 / 1 | 2.160546 / 1 |
| V | nested | 2.242991 / 1 | 2.103360 / 1 | 2.083733 / 1 |
| B | nested | 0.074729 / 1 | 0.029475 / 1 | 0.031289 / 1 |
| FA | nested | 0.058820 / 1 | 0.031779 / 1 | 0.029735 / 1 |
| FB | nested | 0.078016 / 1 | 0.051326 / 1 | 0.048099 / 1 |
| G | nested | 2.155437 / 1 | 2.009484 / 1 | 2.020756 / 1 |
| O | nested | 2.282304 / 1 | 2.135740 / 1 | 2.143634 / 1 |
| attention | boundary | 12.658409 / 1 | 12.104594 / 1 | 11.699807 / 1 |
| attention-residual | boundary | 0.003607 / 1 | 0.003687 / 1 | 0.004028 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026319 / 1 | 0.026429 / 1 | 0.032180 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009488 / 1 | 0.002294 / 1 | 0.002415 / 1 |
| router-and-top16 | boundary | 0.598528 / 1 | 0.569354 / 1 | 0.563061 / 1 |
| EDOWN | nested | 0.655184 / 1 | 0.589221 / 1 | 0.585494 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.656576 / 1 | 0.590734 / 1 | 0.586906 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026239 / 1 | 0.012483 / 1 | 0.006833 / 1 |
| SH1 | nested | 1.111797 / 1 | 1.000790 / 1 | 0.987064 / 1 |
| SH3 | nested | 1.095476 / 1 | 1.027028 / 1 | 1.035093 / 1 |
| SH2 | nested | 1.079948 / 1 | 1.026037 / 1 | 1.034593 / 1 |
| shared-expert-during-read | boundary | 3.300115 / 1 | 3.067501 / 1 | 3.070296 / 1 |
| detail:expert-gate | nested | 2.707416 / 16 | 2.726855 / 16 | 2.707092 / 16 |
| detail:expert-up | nested | 2.723727 / 16 | 2.698640 / 16 | 2.701115 / 16 |
| detail:expert-activation | nested | 0.093082 / 16 | 0.090501 / 16 | 0.091352 / 16 |
| detail:expert-down | nested | 2.932428 / 16 | 2.914764 / 16 | 2.881574 / 16 |
| EUP | nested | 0.626069 / 1 | 0.600803 / 1 | 0.609028 / 1 |
| experts-mix-normalize-up | boundary | 9.475052 / 1 | 9.415199 / 1 | 9.317637 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003055 / 1 | 0.002866 / 1 | 0.002866 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000180 / 1 | 0.000150 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.378618 / 1 | 0.404766 / 1 | 0.390279 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.006447 / 1 | 114.874849 / 1 | 114.946491 / 1 |
| op:Q   int8 projection | nested | 16.206747 / 1 | 15.285668 / 1 | 15.222348 / 1 |
| op:X   mxfp4 expert proj | nested | 8.531759 / 1 | 8.503888 / 1 | 8.455039 / 1 |
| op:N   rmsnorm | nested | 0.032912 / 1 | 0.032802 / 1 | 0.024455 / 1 |
| op:L   l2 per-head | nested | 0.005661 / 1 | 0.005851 / 1 | 0.005851 / 1 |
| op:SiTU + sigma | nested | 0.101188 / 1 | 0.098825 / 1 | 0.099807 / 1 |
| op:C   shortconv | nested | 0.054532 / 1 | 0.053761 / 1 | 0.032261 / 1 |
| op:AR  snapshot aggregate | nested | 0.035026 / 1 | 0.033752 / 1 | 0.041207 / 1 |
| op:D   kda delta-rule | nested | 0.117830 / 1 | 0.121617 / 1 | 0.108633 / 1 |
| op:router dot product | nested | 0.595803 / 1 | 0.566688 / 1 | 0.560317 / 1 |
| op:top-k selection | nested | 0.002043 / 1 | 0.002134 / 1 | 0.002274 / 1 |
| op:alpha / beta / gate | nested | 0.073427 / 1 | 0.074499 / 1 | 0.059280 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.192943 / 1 | 26.250442 / 1 | 25.728958 / 1 |

### Layer 74

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011742 / 1 | 0.004518 / 1 | 0.009558 / 1 |
| pre-attention-aggregation | boundary | 0.016972 / 1 | 0.018034 / 1 | 0.019767 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.011471 / 1 | 0.011010 / 1 | 0.010840 / 1 |
| Q | nested | 2.444557 / 1 | 2.460136 / 1 | 2.459826 / 1 |
| K | nested | 2.291611 / 1 | 2.188218 / 1 | 2.177909 / 1 |
| V | nested | 2.250574 / 1 | 2.101406 / 1 | 2.098531 / 1 |
| B | nested | 0.075031 / 1 | 0.020558 / 1 | 0.020308 / 1 |
| FA | nested | 0.060894 / 1 | 0.025668 / 1 | 0.030748 / 1 |
| FB | nested | 0.078116 / 1 | 0.049533 / 1 | 0.048090 / 1 |
| G | nested | 2.192507 / 1 | 2.082871 / 1 | 2.085296 / 1 |
| O | nested | 2.275701 / 1 | 2.121002 / 1 | 2.101296 / 1 |
| attention | boundary | 12.711738 / 1 | 12.124441 / 1 | 11.725335 / 1 |
| attention-residual | boundary | 0.003858 / 1 | 0.003477 / 1 | 0.003697 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026359 / 1 | 0.026570 / 1 | 0.025257 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010209 / 1 | 0.002495 / 1 | 0.002224 / 1 |
| router-and-top16 | boundary | 0.597887 / 1 | 0.565957 / 1 | 0.563763 / 1 |
| EDOWN | nested | 0.642320 / 1 | 0.584963 / 1 | 0.579593 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.643733 / 1 | 0.586435 / 1 | 0.581095 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041147 / 1 | 0.022643 / 1 | 0.013816 / 1 |
| SH1 | nested | 1.103001 / 1 | 1.000158 / 1 | 0.993907 / 1 |
| SH3 | nested | 1.121425 / 1 | 1.027089 / 1 | 1.023662 / 1 |
| SH2 | nested | 1.088193 / 1 | 1.002583 / 1 | 1.001110 / 1 |
| shared-expert-during-read | boundary | 3.325643 / 1 | 3.042685 / 1 | 3.032616 / 1 |
| detail:expert-gate | nested | 2.762841 / 16 | 2.626074 / 16 | 2.721013 / 16 |
| detail:expert-up | nested | 2.750828 / 16 | 2.649019 / 16 | 2.687521 / 16 |
| detail:expert-activation | nested | 0.090138 / 16 | 0.088727 / 16 | 0.096570 / 16 |
| detail:expert-down | nested | 2.921249 / 16 | 2.828043 / 16 | 2.892675 / 16 |
| EUP | nested | 0.630138 / 1 | 0.594941 / 1 | 0.612515 / 1 |
| experts-mix-normalize-up | boundary | 9.580989 / 1 | 9.172727 / 1 | 9.368492 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003116 / 1 | 0.002945 / 1 | 0.002976 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000240 / 1 | 0.000350 / 1 | 0.000241 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.354312 / 1 | 0.364732 / 1 | 0.356045 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.396485 / 1 | 111.635416 / 1 | 114.583125 / 1 |
| op:Q   int8 projection | nested | 16.252243 / 1 | 15.257244 / 1 | 15.231156 / 1 |
| op:X   mxfp4 expert proj | nested | 8.598834 / 1 | 8.267595 / 1 | 8.472908 / 1 |
| op:N   rmsnorm | nested | 0.033251 / 1 | 0.040495 / 1 | 0.024215 / 1 |
| op:L   l2 per-head | nested | 0.005310 / 1 | 0.005740 / 1 | 0.005650 / 1 |
| op:SiTU + sigma | nested | 0.098444 / 1 | 0.096491 / 1 | 0.105025 / 1 |
| op:C   shortconv | nested | 0.058919 / 1 | 0.058950 / 1 | 0.032360 / 1 |
| op:AR  snapshot aggregate | nested | 0.033212 / 1 | 0.034184 / 1 | 0.035305 / 1 |
| op:D   kda delta-rule | nested | 0.138719 / 1 | 0.145431 / 1 | 0.127018 / 1 |
| op:router dot product | nested | 0.595082 / 1 | 0.563373 / 1 | 0.561138 / 1 |
| op:top-k selection | nested | 0.002365 / 1 | 0.002084 / 1 | 0.001894 / 1 |
| op:alpha / beta / gate | nested | 0.075551 / 1 | 0.071824 / 1 | 0.058961 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.356528 / 1 | 25.965650 / 1 | 25.733536 / 1 |

### Layer 75

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011372 / 1 | 0.005431 / 1 | 0.005319 / 1 |
| pre-attention-aggregation | boundary | 0.017173 / 1 | 0.017994 / 1 | 0.017633 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011832 / 1 | 0.010690 / 1 | 0.010540 / 1 |
| QA | nested | 0.328383 / 1 | 0.308577 / 1 | 0.310611 / 1 |
| QB | nested | 0.740534 / 1 | 0.709555 / 1 | 0.705689 / 1 |
| KA | nested | 0.124904 / 1 | 0.111068 / 1 | 0.107351 / 1 |
| KB | nested | 0.348892 / 1 | 0.319056 / 1 | 0.313405 / 1 |
| G | nested | 2.161017 / 1 | 1.963598 / 1 | 1.961715 / 1 |
| O | nested | 2.192947 / 1 | 2.056772 / 1 | 2.056161 / 1 |
| attention | boundary | 6.008435 / 1 | 5.569084 / 1 | 5.555038 / 1 |
| attention-residual | boundary | 0.002655 / 1 | 0.002965 / 1 | 0.003035 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025788 / 1 | 0.026008 / 1 | 0.025908 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009708 / 1 | 0.001854 / 1 | 0.002034 / 1 |
| router-and-top16 | boundary | 0.602736 / 1 | 0.568412 / 1 | 0.570636 / 1 |
| EDOWN | nested | 0.644574 / 1 | 0.591224 / 1 | 0.598288 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.645957 / 1 | 0.593539 / 1 | 0.599761 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031719 / 1 | 0.007744 / 1 | 0.006211 / 1 |
| SH1 | nested | 1.103672 / 1 | 1.017551 / 1 | 1.023723 / 1 |
| SH3 | nested | 1.094665 / 1 | 0.993406 / 1 | 1.000169 / 1 |
| SH2 | nested | 1.090847 / 1 | 1.006731 / 1 | 1.009476 / 1 |
| shared-expert-during-read | boundary | 3.302089 / 1 | 3.030692 / 1 | 3.046462 / 1 |
| detail:expert-gate | nested | 2.716334 / 16 | 2.592604 / 16 | 2.694005 / 16 |
| detail:expert-up | nested | 2.689804 / 16 | 2.582876 / 16 | 2.695389 / 16 |
| detail:expert-activation | nested | 0.090656 / 16 | 0.089419 / 16 | 0.095759 / 16 |
| detail:expert-down | nested | 2.885781 / 16 | 2.727765 / 16 | 2.868722 / 16 |
| EUP | nested | 0.631129 / 1 | 0.596304 / 1 | 0.597407 / 1 |
| experts-mix-normalize-up | boundary | 9.409780 / 1 | 8.944871 / 1 | 9.305815 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002655 / 1 | 0.008446 / 1 | 0.003406 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000150 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005401 / 1 | 0.006632 / 1 | 0.006132 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 114.811873 / 1 | 109.080235 / 1 | 114.484666 / 1 |
| op:Q   int8 projection | nested | 10.460143 / 1 | 9.672498 / 1 | 9.682492 / 1 |
| op:X   mxfp4 expert proj | nested | 8.458362 / 1 | 8.067071 / 1 | 8.430170 / 1 |
| op:N   rmsnorm | nested | 0.024213 / 1 | 0.022862 / 1 | 0.023634 / 1 |
| op:SiTU + sigma | nested | 0.098642 / 1 | 0.097134 / 1 | 0.103744 / 1 |
| op:AR  snapshot aggregate | nested | 0.032840 / 1 | 0.033001 / 1 | 0.032620 / 1 |
| op:SA  softmax attention | nested | 0.007314 / 1 | 0.008616 / 1 | 0.012213 / 1 |
| op:router dot product | nested | 0.599821 / 1 | 0.565907 / 1 | 0.568231 / 1 |
| op:top-k selection | nested | 0.002484 / 1 | 0.002134 / 1 | 0.002114 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.105753 / 1 | 18.811363 / 1 | 19.173921 / 1 |

### Layer 76

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.010529 / 1 | 0.003697 / 1 | 0.003416 / 1 |
| pre-attention-aggregation | boundary | 0.015109 / 1 | 0.015239 / 1 | 0.016391 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011251 / 1 | 0.009748 / 1 | 0.010540 / 1 |
| Q | nested | 2.441280 / 1 | 2.454064 / 1 | 2.460838 / 1 |
| K | nested | 2.338759 / 1 | 2.173541 / 1 | 2.176105 / 1 |
| V | nested | 2.314153 / 1 | 2.097809 / 1 | 2.091147 / 1 |
| B | nested | 0.078586 / 1 | 0.024846 / 1 | 0.024886 / 1 |
| FA | nested | 0.058198 / 1 | 0.037791 / 1 | 0.037951 / 1 |
| FB | nested | 0.070862 / 1 | 0.047879 / 1 | 0.046748 / 1 |
| G | nested | 2.190743 / 1 | 2.006639 / 1 | 2.015635 / 1 |
| O | nested | 2.260252 / 1 | 2.156899 / 1 | 2.160346 / 1 |
| attention | boundary | 12.691190 / 1 | 11.989229 / 1 | 11.669562 / 1 |
| attention-residual | boundary | 0.004038 / 1 | 0.003116 / 1 | 0.003547 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024796 / 1 | 0.026600 / 1 | 0.026179 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009708 / 1 | 0.002244 / 1 | 0.002214 / 1 |
| router-and-top16 | boundary | 0.601794 / 1 | 0.564134 / 1 | 0.569183 / 1 |
| EDOWN | nested | 0.650094 / 1 | 0.591265 / 1 | 0.598087 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.651487 / 1 | 0.592778 / 1 | 0.599640 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041257 / 1 | 0.006583 / 1 | 0.006442 / 1 |
| SH1 | nested | 1.091188 / 1 | 0.979830 / 1 | 0.996311 / 1 |
| SH3 | nested | 1.093563 / 1 | 1.002142 / 1 | 1.012251 / 1 |
| SH2 | nested | 1.109222 / 1 | 1.063196 / 1 | 1.057115 / 1 |
| shared-expert-during-read | boundary | 3.307950 / 1 | 3.058664 / 1 | 3.079433 / 1 |
| detail:expert-gate | nested | 2.723741 / 16 | 2.686464 / 16 | 2.696017 / 16 |
| detail:expert-up | nested | 2.733676 / 16 | 2.678023 / 16 | 2.698842 / 16 |
| detail:expert-activation | nested | 0.087811 / 16 | 0.089941 / 16 | 0.092384 / 16 |
| detail:expert-down | nested | 2.905729 / 16 | 2.912942 / 16 | 2.889929 / 16 |
| EUP | nested | 0.628123 / 1 | 0.597607 / 1 | 0.600693 / 1 |
| experts-mix-normalize-up | boundary | 9.488827 / 1 | 9.345770 / 1 | 9.338346 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002906 / 1 | 0.003066 / 1 | 0.002986 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000150 / 1 | 0.000161 / 1 | 0.000151 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.372977 / 1 | 0.387464 / 1 | 0.369420 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.631876 / 1 | 114.247438 / 1 | 114.428216 / 1 |
| op:Q   int8 projection | nested | 16.323422 / 1 | 15.231835 / 1 | 15.276401 / 1 |
| op:X   mxfp4 expert proj | nested | 8.526748 / 1 | 8.443587 / 1 | 8.452961 / 1 |
| op:N   rmsnorm | nested | 0.032040 / 1 | 0.038983 / 1 | 0.022922 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.005771 / 1 | 0.005931 / 1 |
| op:SiTU + sigma | nested | 0.096198 / 1 | 0.098125 / 1 | 0.100679 / 1 |
| op:C   shortconv | nested | 0.067507 / 1 | 0.065873 / 1 | 0.033112 / 1 |
| op:AR  snapshot aggregate | nested | 0.030717 / 1 | 0.031770 / 1 | 0.031559 / 1 |
| op:D   kda delta-rule | nested | 0.103072 / 1 | 0.116978 / 1 | 0.108142 / 1 |
| op:router dot product | nested | 0.599260 / 1 | 0.561579 / 1 | 0.566659 / 1 |
| op:top-k selection | nested | 0.002144 / 1 | 0.002214 / 1 | 0.002274 / 1 |
| op:alpha / beta / gate | nested | 0.072245 / 1 | 0.073457 / 1 | 0.058139 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.252083 / 1 | 26.026123 / 1 | 25.715412 / 1 |

### Layer 77

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012323 / 1 | 0.005971 / 1 | 0.005180 / 1 |
| pre-attention-aggregation | boundary | 0.018475 / 1 | 0.019857 / 1 | 0.019306 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000030 / 1 | 0.000090 / 1 |
| pre-attention-normalization | boundary | 0.012714 / 1 | 0.011482 / 1 | 0.011692 / 1 |
| Q | nested | 2.444247 / 1 | 2.426674 / 1 | 2.428627 / 1 |
| K | nested | 2.354409 / 1 | 2.188448 / 1 | 2.156268 / 1 |
| V | nested | 2.282073 / 1 | 2.105243 / 1 | 2.092269 / 1 |
| B | nested | 0.085439 / 1 | 0.021330 / 1 | 0.021110 / 1 |
| FA | nested | 0.057437 / 1 | 0.033763 / 1 | 0.036197 / 1 |
| FB | nested | 0.073928 / 1 | 0.048620 / 1 | 0.050535 / 1 |
| G | nested | 2.167359 / 1 | 1.976313 / 1 | 1.965792 / 1 |
| O | nested | 2.272967 / 1 | 2.208887 / 1 | 2.179132 / 1 |
| attention | boundary | 12.724282 / 1 | 12.007343 / 1 | 11.631821 / 1 |
| attention-residual | boundary | 0.003737 / 1 | 0.003477 / 1 | 0.003477 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026239 / 1 | 0.026990 / 1 | 0.026099 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009648 / 1 | 0.002405 / 1 | 0.002595 / 1 |
| router-and-top16 | boundary | 0.596815 / 1 | 0.562831 / 1 | 0.569113 / 1 |
| EDOWN | nested | 0.647510 / 1 | 0.589441 / 1 | 0.584402 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.648923 / 1 | 0.590974 / 1 | 0.585875 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026719 / 1 | 0.017602 / 1 | 0.006262 / 1 |
| SH1 | nested | 1.101758 / 1 | 1.012161 / 1 | 1.005989 / 1 |
| SH3 | nested | 1.086570 / 1 | 1.023943 / 1 | 1.015797 / 1 |
| SH2 | nested | 1.090737 / 1 | 1.011460 / 1 | 0.998946 / 1 |
| shared-expert-during-read | boundary | 3.292642 / 1 | 3.061219 / 1 | 3.034208 / 1 |
| detail:expert-gate | nested | 2.706889 / 16 | 2.700265 / 16 | 2.619282 / 16 |
| detail:expert-up | nested | 2.710967 / 16 | 2.677330 / 16 | 2.610358 / 16 |
| detail:expert-activation | nested | 0.087876 / 16 | 0.088468 / 16 | 0.091451 / 16 |
| detail:expert-down | nested | 2.880851 / 16 | 2.871853 / 16 | 2.775146 / 16 |
| EUP | nested | 0.635678 / 1 | 0.603748 / 1 | 0.593088 / 1 |
| experts-mix-normalize-up | boundary | 9.418877 / 1 | 9.326043 / 1 | 9.050879 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003206 / 1 | 0.002945 / 1 | 0.002816 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000160 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.368849 / 1 | 0.383146 / 1 | 0.374038 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.069515 / 1 | 114.485536 / 1 | 110.454287 / 1 |
| op:Q   int8 projection | nested | 16.298350 / 1 | 15.248409 / 1 | 15.126230 / 1 |
| op:X   mxfp4 expert proj | nested | 8.465105 / 1 | 8.415082 / 1 | 8.173889 / 1 |
| op:N   rmsnorm | nested | 0.040124 / 1 | 0.033462 / 1 | 0.023745 / 1 |
| op:L   l2 per-head | nested | 0.005430 / 1 | 0.005520 / 1 | 0.010129 / 1 |
| op:SiTU + sigma | nested | 0.096299 / 1 | 0.096422 / 1 | 0.099356 / 1 |
| op:C   shortconv | nested | 0.055173 / 1 | 0.062677 / 1 | 0.032311 / 1 |
| op:AR  snapshot aggregate | nested | 0.034586 / 1 | 0.036068 / 1 | 0.035376 / 1 |
| op:D   kda delta-rule | nested | 0.125725 / 1 | 0.125254 / 1 | 0.115486 / 1 |
| op:router dot product | nested | 0.594381 / 1 | 0.560447 / 1 | 0.566118 / 1 |
| op:top-k selection | nested | 0.002034 / 1 | 0.002084 / 1 | 0.002405 / 1 |
| op:alpha / beta / gate | nested | 0.073046 / 1 | 0.071734 / 1 | 0.062998 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.180139 / 1 | 26.038897 / 1 | 25.341634 / 1 |

### Layer 78

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012844 / 1 | 0.004709 / 1 | 0.004979 / 1 |
| pre-attention-aggregation | boundary | 0.017884 / 1 | 0.018475 / 1 | 0.017332 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011201 / 1 | 0.011291 / 1 | 0.012474 / 1 |
| Q | nested | 2.667333 / 1 | 2.663987 / 1 | 2.658236 / 1 |
| K | nested | 2.515990 / 1 | 2.234695 / 1 | 2.170164 / 1 |
| V | nested | 2.371490 / 1 | 2.113359 / 1 | 2.092740 / 1 |
| B | nested | 0.075741 / 1 | 0.028814 / 1 | 0.028544 / 1 |
| FA | nested | 0.064571 / 1 | 0.035637 / 1 | 0.030677 / 1 |
| FB | nested | 0.076323 / 1 | 0.051316 / 1 | 0.049152 / 1 |
| G | nested | 2.174783 / 1 | 2.048177 / 1 | 2.037717 / 1 |
| O | nested | 2.243732 / 1 | 2.098500 / 1 | 2.072462 / 1 |
| attention | boundary | 13.151541 / 1 | 12.251949 / 1 | 11.809583 / 1 |
| attention-residual | boundary | 0.003657 / 1 | 0.003646 / 1 | 0.003476 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025678 / 1 | 0.026891 / 1 | 0.026870 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009368 / 1 | 0.002084 / 1 | 0.002394 / 1 |
| router-and-top16 | boundary | 0.604349 / 1 | 0.573361 / 1 | 0.568272 / 1 |
| EDOWN | nested | 0.643021 / 1 | 0.589441 / 1 | 0.599540 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.644505 / 1 | 0.591545 / 1 | 0.601083 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026018 / 1 | 0.014197 / 1 | 0.006282 / 1 |
| SH1 | nested | 1.096479 / 1 | 1.008564 / 1 | 0.999347 / 1 |
| SH3 | nested | 1.080869 / 1 | 0.972156 / 1 | 0.967107 / 1 |
| SH2 | nested | 1.100216 / 1 | 1.030144 / 1 | 1.033240 / 1 |
| shared-expert-during-read | boundary | 3.298412 / 1 | 3.024731 / 1 | 3.012879 / 1 |
| detail:expert-gate | nested | 2.692000 / 16 | 2.672340 / 16 | 2.626026 / 16 |
| detail:expert-up | nested | 2.721325 / 16 | 2.691998 / 16 | 2.648910 / 16 |
| detail:expert-activation | nested | 0.086755 / 16 | 0.088747 / 16 | 0.093444 / 16 |
| detail:expert-down | nested | 2.908912 / 16 | 2.855022 / 16 | 2.788359 / 16 |
| EUP | nested | 0.642621 / 1 | 0.608266 / 1 | 0.602776 / 1 |
| experts-mix-normalize-up | boundary | 9.448301 / 1 | 9.298211 / 1 | 9.100442 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002976 / 1 | 0.002625 / 1 | 0.002825 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000250 / 1 | 0.000461 / 1 | 0.000240 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.364922 / 1 | 0.386152 / 1 | 0.381733 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.393918 / 1 | 113.538114 / 1 | 112.398167 / 1 |
| op:Q   int8 projection | nested | 16.751638 / 1 | 15.481353 / 1 | 15.339848 / 1 |
| op:X   mxfp4 expert proj | nested | 8.486325 / 1 | 8.385515 / 1 | 8.234704 / 1 |
| op:N   rmsnorm | nested | 0.032741 / 1 | 0.037450 / 1 | 0.025517 / 1 |
| op:L   l2 per-head | nested | 0.005630 / 1 | 0.005480 / 1 | 0.005550 / 1 |
| op:SiTU + sigma | nested | 0.102534 / 1 | 0.097122 / 1 | 0.101311 / 1 |
| op:C   shortconv | nested | 0.060042 / 1 | 0.055524 / 1 | 0.032561 / 1 |
| op:AR  snapshot aggregate | nested | 0.033252 / 1 | 0.034735 / 1 | 0.034104 / 1 |
| op:D   kda delta-rule | nested | 0.086762 / 1 | 0.102221 / 1 | 0.095519 / 1 |
| op:router dot product | nested | 0.601744 / 1 | 0.570385 / 1 | 0.565426 / 1 |
| op:top-k selection | nested | 0.002224 / 1 | 0.002695 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.083136 / 1 | 0.072986 / 1 | 0.058530 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.638946 / 1 | 26.226467 / 1 | 25.567556 / 1 |

### Layer 79

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011782 / 1 | 0.005109 / 1 | 0.004969 / 1 |
| pre-attention-aggregation | boundary | 0.021641 / 1 | 0.021881 / 1 | 0.021981 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.019727 / 1 | 0.011822 / 1 | 0.011101 / 1 |
| QA | nested | 0.310700 / 1 | 0.310259 / 1 | 0.308266 / 1 |
| QB | nested | 0.723231 / 1 | 0.677847 / 1 | 0.691231 / 1 |
| KA | nested | 0.126657 / 1 | 0.102902 / 1 | 0.103403 / 1 |
| KB | nested | 0.350505 / 1 | 0.308417 / 1 | 0.311462 / 1 |
| G | nested | 2.184161 / 1 | 2.011929 / 1 | 2.019613 / 1 |
| O | nested | 2.170875 / 1 | 1.966494 / 1 | 1.964661 / 1 |
| attention | boundary | 5.974942 / 1 | 5.479437 / 1 | 5.499645 / 1 |
| attention-residual | boundary | 0.002625 / 1 | 0.002785 / 1 | 0.002755 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.030336 / 1 | 0.025738 / 1 | 0.026259 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009517 / 1 | 0.002114 / 1 | 0.002144 / 1 |
| router-and-top16 | boundary | 0.611994 / 1 | 0.569674 / 1 | 0.575155 / 1 |
| EDOWN | nested | 0.643232 / 1 | 0.581186 / 1 | 0.585113 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.644665 / 1 | 0.582618 / 1 | 0.586626 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.047659 / 1 | 0.006352 / 1 | 0.006362 / 1 |
| SH1 | nested | 1.087702 / 1 | 0.983077 / 1 | 0.983598 / 1 |
| SH3 | nested | 1.100897 / 1 | 1.005368 / 1 | 1.003936 / 1 |
| SH2 | nested | 1.098682 / 1 | 1.011270 / 1 | 1.010407 / 1 |
| shared-expert-during-read | boundary | 3.300636 / 1 | 3.013059 / 1 | 3.011426 / 1 |
| detail:expert-gate | nested | 2.749788 / 16 | 2.664119 / 16 | 2.633360 / 16 |
| detail:expert-up | nested | 2.768754 / 16 | 2.715584 / 16 | 2.623939 / 16 |
| detail:expert-activation | nested | 0.091121 / 16 | 0.090489 / 16 | 0.091713 / 16 |
| detail:expert-down | nested | 2.906027 / 16 | 2.901823 / 16 | 2.823276 / 16 |
| EUP | nested | 0.635928 / 1 | 0.608056 / 1 | 0.617684 / 1 |
| experts-mix-normalize-up | boundary | 9.580158 / 1 | 9.348215 / 1 | 9.132842 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003366 / 1 | 0.002905 / 1 | 0.002634 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000451 / 1 | 0.000160 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006552 / 1 | 0.005941 / 1 | 0.005500 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.390637 / 1 | 114.397530 / 1 | 112.039454 / 1 |
| op:Q   int8 projection | nested | 10.431237 / 1 | 9.565231 / 1 | 9.597740 / 1 |
| op:X   mxfp4 expert proj | nested | 8.593837 / 1 | 8.456309 / 1 | 8.252038 / 1 |
| op:N   rmsnorm | nested | 0.033091 / 1 | 0.024676 / 1 | 0.024245 / 1 |
| op:SiTU + sigma | nested | 0.099607 / 1 | 0.098415 / 1 | 0.099988 / 1 |
| op:AR  snapshot aggregate | nested | 0.040947 / 1 | 0.037321 / 1 | 0.037620 / 1 |
| op:SA  softmax attention | nested | 0.007304 / 1 | 0.009087 / 1 | 0.011461 / 1 |
| op:router dot product | nested | 0.609018 / 1 | 0.566749 / 1 | 0.572219 / 1 |
| op:top-k selection | nested | 0.002494 / 1 | 0.002344 / 1 | 0.002514 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.282913 / 1 | 19.093922 / 1 | 18.906121 / 1 |

### Layer 80

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011161 / 1 | 0.003507 / 1 | 0.003647 / 1 |
| pre-attention-aggregation | boundary | 0.015268 / 1 | 0.015349 / 1 | 0.015429 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010710 / 1 | 0.010830 / 1 | 0.010880 / 1 |
| Q | nested | 2.456560 / 1 | 2.453985 / 1 | 2.454956 / 1 |
| K | nested | 2.346784 / 1 | 2.191745 / 1 | 2.155707 / 1 |
| V | nested | 2.238813 / 1 | 2.109642 / 1 | 2.087310 / 1 |
| B | nested | 0.071334 / 1 | 0.021029 / 1 | 0.021650 / 1 |
| FA | nested | 0.059862 / 1 | 0.033493 / 1 | 0.034584 / 1 |
| FB | nested | 0.076302 / 1 | 0.051025 / 1 | 0.050434 / 1 |
| G | nested | 2.180043 / 1 | 1.954161 / 1 | 1.950965 / 1 |
| O | nested | 2.255434 / 1 | 2.190102 / 1 | 2.184481 / 1 |
| attention | boundary | 12.674209 / 1 | 12.016279 / 1 | 11.590544 / 1 |
| attention-residual | boundary | 0.003537 / 1 | 0.003357 / 1 | 0.003276 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027511 / 1 | 0.026119 / 1 | 0.025909 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.015970 / 1 | 0.002184 / 1 | 0.002495 / 1 |
| router-and-top16 | boundary | 0.605662 / 1 | 0.581186 / 1 | 0.575264 / 1 |
| EDOWN | nested | 0.659593 / 1 | 0.606894 / 1 | 0.612885 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.661045 / 1 | 0.608507 / 1 | 0.614408 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.037260 / 1 | 0.006512 / 1 | 0.006923 / 1 |
| SH1 | nested | 1.120153 / 1 | 1.028150 / 1 | 1.028472 / 1 |
| SH3 | nested | 1.090988 / 1 | 1.009737 / 1 | 0.977536 / 1 |
| SH2 | nested | 1.099915 / 1 | 1.007572 / 1 | 0.989078 / 1 |
| shared-expert-during-read | boundary | 3.323970 / 1 | 3.058714 / 1 | 3.008611 / 1 |
| detail:expert-gate | nested | 2.714489 / 16 | 2.656972 / 16 | 2.673615 / 16 |
| detail:expert-up | nested | 2.718848 / 16 | 2.651122 / 16 | 2.625142 / 16 |
| detail:expert-activation | nested | 0.088897 / 16 | 0.088294 / 16 | 0.093455 / 16 |
| detail:expert-down | nested | 2.898485 / 16 | 2.828233 / 16 | 2.768882 / 16 |
| EUP | nested | 0.639535 / 1 | 0.605030 / 1 | 0.615500 / 1 |
| experts-mix-normalize-up | boundary | 9.487064 / 1 | 9.208634 / 1 | 9.129156 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003377 / 1 | 0.002705 / 1 | 0.002604 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000220 / 1 | 0.000271 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.401660 / 1 | 0.385640 / 1 | 0.375050 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.465072 / 1 | 112.366336 / 1 | 111.266893 / 1 |
| op:Q   int8 projection | nested | 16.293701 / 1 | 15.260940 / 1 | 15.161727 / 1 |
| op:X   mxfp4 expert proj | nested | 8.503599 / 1 | 8.303473 / 1 | 8.240486 / 1 |
| op:N   rmsnorm | nested | 0.033172 / 1 | 0.038201 / 1 | 0.022903 / 1 |
| op:L   l2 per-head | nested | 0.005861 / 1 | 0.005691 / 1 | 0.005560 / 1 |
| op:SiTU + sigma | nested | 0.096682 / 1 | 0.095870 / 1 | 0.101419 / 1 |
| op:C   shortconv | nested | 0.061285 / 1 | 0.076734 / 1 | 0.034434 / 1 |
| op:AR  snapshot aggregate | nested | 0.030747 / 1 | 0.031299 / 1 | 0.031018 / 1 |
| op:D   kda delta-rule | nested | 0.108503 / 1 | 0.116237 / 1 | 0.100438 / 1 |
| op:router dot product | nested | 0.602746 / 1 | 0.578280 / 1 | 0.572279 / 1 |
| op:top-k selection | nested | 0.002424 / 1 | 0.002294 / 1 | 0.002635 / 1 |
| op:alpha / beta / gate | nested | 0.071313 / 1 | 0.071504 / 1 | 0.058690 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.297959 / 1 | 25.948388 / 1 | 25.383332 / 1 |

### Layer 81

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012764 / 1 | 0.005911 / 1 | 0.005400 / 1 |
| pre-attention-aggregation | boundary | 0.016440 / 1 | 0.016621 / 1 | 0.017663 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000021 / 1 |
| pre-attention-normalization | boundary | 0.011711 / 1 | 0.011241 / 1 | 0.011091 / 1 |
| Q | nested | 2.490603 / 1 | 2.462561 / 1 | 2.457511 / 1 |
| K | nested | 2.369166 / 1 | 2.217563 / 1 | 2.185252 / 1 |
| V | nested | 2.229134 / 1 | 2.107808 / 1 | 2.120611 / 1 |
| B | nested | 0.074950 / 1 | 0.028133 / 1 | 0.027792 / 1 |
| FA | nested | 0.063869 / 1 | 0.032791 / 1 | 0.033423 / 1 |
| FB | nested | 0.075130 / 1 | 0.052257 / 1 | 0.051166 / 1 |
| G | nested | 2.154064 / 1 | 1.969659 / 1 | 1.961324 / 1 |
| O | nested | 2.305227 / 1 | 2.213806 / 1 | 2.224956 / 1 |
| attention | boundary | 12.796187 / 1 | 12.096369 / 1 | 11.755953 / 1 |
| attention-residual | boundary | 0.003386 / 1 | 0.003376 / 1 | 0.003847 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025908 / 1 | 0.026510 / 1 | 0.025788 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009909 / 1 | 0.002365 / 1 | 0.002134 / 1 |
| router-and-top16 | boundary | 0.605722 / 1 | 0.570916 / 1 | 0.565887 / 1 |
| EDOWN | nested | 0.654693 / 1 | 0.584823 / 1 | 0.579122 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.656136 / 1 | 0.586426 / 1 | 0.580685 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.017693 / 1 | 0.012573 / 1 | 0.018494 / 1 |
| SH1 | nested | 1.107499 / 1 | 1.024765 / 1 | 0.999777 / 1 |
| SH3 | nested | 1.105586 / 1 | 1.035134 / 1 | 1.034082 / 1 |
| SH2 | nested | 1.099073 / 1 | 1.010928 / 1 | 1.016579 / 1 |
| shared-expert-during-read | boundary | 3.326154 / 1 | 3.084693 / 1 | 3.067642 / 1 |
| detail:expert-gate | nested | 2.720771 / 16 | 2.688453 / 16 | 2.514950 / 16 |
| detail:expert-up | nested | 2.753194 / 16 | 2.665630 / 16 | 2.478382 / 16 |
| detail:expert-activation | nested | 0.091720 / 16 | 0.090899 / 16 | 0.094117 / 16 |
| detail:expert-down | nested | 2.917789 / 16 | 2.866104 / 16 | 2.609897 / 16 |
| EUP | nested | 0.643102 / 1 | 0.593098 / 1 | 0.591936 / 1 |
| experts-mix-normalize-up | boundary | 9.529403 / 1 | 9.293022 / 1 | 8.665459 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003546 / 1 | 0.002735 / 1 | 0.002685 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000151 / 1 | 0.000170 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.386082 / 1 | 0.360924 / 1 | 0.361035 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.858384 / 1 | 113.495227 / 1 | 104.301694 / 1 |
| op:Q   int8 projection | nested | 16.370474 / 1 | 15.331582 / 1 | 15.281618 / 1 |
| op:X   mxfp4 expert proj | nested | 8.565711 / 1 | 8.391728 / 1 | 7.780225 / 1 |
| op:N   rmsnorm | nested | 0.036217 / 1 | 0.033212 / 1 | 0.023404 / 1 |
| op:L   l2 per-head | nested | 0.005280 / 1 | 0.005791 / 1 | 0.005510 / 1 |
| op:SiTU + sigma | nested | 0.100256 / 1 | 0.099426 / 1 | 0.102042 / 1 |
| op:C   shortconv | nested | 0.063178 / 1 | 0.059542 / 1 | 0.032000 / 1 |
| op:AR  snapshot aggregate | nested | 0.032261 / 1 | 0.032761 / 1 | 0.033873 / 1 |
| op:D   kda delta-rule | nested | 0.109525 / 1 | 0.111899 / 1 | 0.100849 / 1 |
| op:router dot product | nested | 0.602786 / 1 | 0.568091 / 1 | 0.563172 / 1 |
| op:top-k selection | nested | 0.002544 / 1 | 0.002384 / 1 | 0.002495 / 1 |
| op:alpha / beta / gate | nested | 0.072856 / 1 | 0.072886 / 1 | 0.058860 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.418795 / 1 | 26.091294 / 1 | 25.102017 / 1 |

### Layer 82

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013826 / 1 | 0.005050 / 1 | 0.004739 / 1 |
| pre-attention-aggregation | boundary | 0.017072 / 1 | 0.017462 / 1 | 0.017192 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010199 / 1 | 0.011832 / 1 | 0.011331 / 1 |
| Q | nested | 2.467329 / 1 | 2.452552 / 1 | 2.452302 / 1 |
| K | nested | 2.346724 / 1 | 2.199639 / 1 | 2.178780 / 1 |
| V | nested | 2.258570 / 1 | 2.093381 / 1 | 2.096487 / 1 |
| B | nested | 0.062426 / 1 | 0.029365 / 1 | 0.020268 / 1 |
| FA | nested | 0.061465 / 1 | 0.034615 / 1 | 0.033382 / 1 |
| FB | nested | 0.074981 / 1 | 0.051296 / 1 | 0.048230 / 1 |
| G | nested | 2.176145 / 1 | 2.035493 / 1 | 2.023951 / 1 |
| O | nested | 2.222732 / 1 | 2.084755 / 1 | 2.075117 / 1 |
| attention | boundary | 12.673477 / 1 | 11.997514 / 1 | 11.607826 / 1 |
| attention-residual | boundary | 0.003777 / 1 | 0.003537 / 1 | 0.003907 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025428 / 1 | 0.025949 / 1 | 0.026039 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009959 / 1 | 0.002485 / 1 | 0.002024 / 1 |
| router-and-top16 | boundary | 0.595773 / 1 | 0.567169 / 1 | 0.569715 / 1 |
| EDOWN | nested | 0.667567 / 1 | 0.612694 / 1 | 0.609329 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.669190 / 1 | 0.614558 / 1 | 0.610901 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.023714 / 1 | 0.012824 / 1 | 0.006843 / 1 |
| SH1 | nested | 1.095206 / 1 | 0.982064 / 1 | 0.979110 / 1 |
| SH3 | nested | 1.083104 / 1 | 0.958511 / 1 | 0.949023 / 1 |
| SH2 | nested | 1.109503 / 1 | 1.024174 / 1 | 1.027901 / 1 |
| shared-expert-during-read | boundary | 3.301769 / 1 | 2.978755 / 1 | 2.969789 / 1 |
| detail:expert-gate | nested | 2.756609 / 16 | 2.695605 / 16 | 2.570191 / 16 |
| detail:expert-up | nested | 2.726182 / 16 | 2.674833 / 16 | 2.583145 / 16 |
| detail:expert-activation | nested | 0.091261 / 16 | 0.092529 / 16 | 0.099053 / 16 |
| detail:expert-down | nested | 2.945653 / 16 | 2.843853 / 16 | 2.715969 / 16 |
| EUP | nested | 0.634706 / 1 | 0.590593 / 1 | 0.592006 / 1 |
| experts-mix-normalize-up | boundary | 9.556855 / 1 | 9.290697 / 1 | 8.927308 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003336 / 1 | 0.002845 / 1 | 0.002976 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000251 / 1 | 0.000511 / 1 | 0.000261 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.403734 / 1 | 0.363379 / 1 | 0.366835 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.963246 / 1 | 113.316327 / 1 | 107.855767 / 1 |
| op:Q   int8 projection | nested | 16.258366 / 1 | 15.147588 / 1 | 15.084151 / 1 |
| op:X   mxfp4 expert proj | nested | 8.601007 / 1 | 8.393028 / 1 | 8.052895 / 1 |
| op:N   rmsnorm | nested | 0.035837 / 1 | 0.049322 / 1 | 0.024325 / 1 |
| op:L   l2 per-head | nested | 0.005560 / 1 | 0.005360 / 1 | 0.005691 / 1 |
| op:SiTU + sigma | nested | 0.099827 / 1 | 0.101046 / 1 | 0.107208 / 1 |
| op:C   shortconv | nested | 0.054241 / 1 | 0.053581 / 1 | 0.032942 / 1 |
| op:AR  snapshot aggregate | nested | 0.032351 / 1 | 0.033052 / 1 | 0.032852 / 1 |
| op:D   kda delta-rule | nested | 0.103293 / 1 | 0.102712 / 1 | 0.097382 / 1 |
| op:router dot product | nested | 0.593108 / 1 | 0.564244 / 1 | 0.567009 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002485 / 1 | 0.002254 / 1 |
| op:alpha / beta / gate | nested | 0.073167 / 1 | 0.072526 / 1 | 0.059101 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.326102 / 1 | 25.912771 / 1 | 25.146219 / 1 |

### Layer 83

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013134 / 1 | 0.005070 / 1 | 0.005169 / 1 |
| pre-attention-aggregation | boundary | 0.018214 / 1 | 0.017813 / 1 | 0.016742 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000090 / 1 |
| pre-attention-normalization | boundary | 0.010390 / 1 | 0.011031 / 1 | 0.010730 / 1 |
| QA | nested | 0.310470 / 1 | 0.308476 / 1 | 0.306903 / 1 |
| QB | nested | 0.733741 / 1 | 0.701781 / 1 | 0.701731 / 1 |
| KA | nested | 0.118141 / 1 | 0.101259 / 1 | 0.105517 / 1 |
| KB | nested | 0.342640 / 1 | 0.302064 / 1 | 0.301553 / 1 |
| G | nested | 2.218314 / 1 | 2.064076 / 1 | 2.060049 / 1 |
| O | nested | 2.173561 / 1 | 1.985179 / 1 | 1.982053 / 1 |
| attention | boundary | 6.007874 / 1 | 5.569435 / 1 | 5.559787 / 1 |
| attention-residual | boundary | 0.002465 / 1 | 0.002885 / 1 | 0.002916 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026029 / 1 | 0.026469 / 1 | 0.026059 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009688 / 1 | 0.002474 / 1 | 0.002084 / 1 |
| router-and-top16 | boundary | 0.612675 / 1 | 0.570767 / 1 | 0.575595 / 1 |
| EDOWN | nested | 0.632952 / 1 | 0.578241 / 1 | 0.586055 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.634526 / 1 | 0.579834 / 1 | 0.587567 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026830 / 1 | 0.006492 / 1 | 0.006342 / 1 |
| SH1 | nested | 1.085128 / 1 | 0.974531 / 1 | 0.981494 / 1 |
| SH3 | nested | 1.103581 / 1 | 0.994778 / 1 | 0.990691 / 1 |
| SH2 | nested | 1.086990 / 1 | 0.994047 / 1 | 0.996692 / 1 |
| shared-expert-during-read | boundary | 3.289716 / 1 | 2.977573 / 1 | 2.985668 / 1 |
| detail:expert-gate | nested | 2.739899 / 16 | 2.510682 / 16 | 2.662005 / 16 |
| detail:expert-up | nested | 2.744536 / 16 | 2.533214 / 16 | 2.649341 / 16 |
| detail:expert-activation | nested | 0.096802 / 16 | 0.099236 / 16 | 0.100449 / 16 |
| detail:expert-down | nested | 2.886032 / 16 | 2.682250 / 16 | 2.818755 / 16 |
| EUP | nested | 0.639425 / 1 | 0.597937 / 1 | 0.600241 / 1 |
| experts-mix-normalize-up | boundary | 9.515127 / 1 | 8.806773 / 1 | 9.191371 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002545 / 1 | 0.003036 / 1 | 0.002715 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000141 / 1 | 0.000150 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006502 / 1 | 0.005711 / 1 | 0.005891 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.508108 / 1 | 105.607900 / 1 | 111.004909 / 1 |
| op:Q   int8 projection | nested | 10.443590 / 1 | 9.600999 / 1 | 9.611605 / 1 |
| op:X   mxfp4 expert proj | nested | 8.549213 / 1 | 7.909448 / 1 | 8.312772 / 1 |
| op:N   rmsnorm | nested | 0.023855 / 1 | 0.024767 / 1 | 0.023905 / 1 |
| op:SiTU + sigma | nested | 0.105297 / 1 | 0.107951 / 1 | 0.110798 / 1 |
| op:AR  snapshot aggregate | nested | 0.033853 / 1 | 0.033483 / 1 | 0.032481 / 1 |
| op:SA  softmax attention | nested | 0.006782 / 1 | 0.008596 / 1 | 0.012122 / 1 |
| op:router dot product | nested | 0.609468 / 1 | 0.567841 / 1 | 0.573191 / 1 |
| op:top-k selection | nested | 0.002755 / 1 | 0.002485 / 1 | 0.002114 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.193115 / 1 | 18.603035 / 1 | 18.996089 / 1 |

### Layer 84

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.011572 / 1 | 0.003517 / 1 | 0.003356 / 1 |
| pre-attention-aggregation | boundary | 0.015639 / 1 | 0.015339 / 1 | 0.015689 / 1 |
| snapshot-push | boundary | 0.001653 / 1 | 0.001553 / 1 | 0.001552 / 1 |
| pre-attention-normalization | boundary | 0.011131 / 1 | 0.011050 / 1 | 0.010250 / 1 |
| Q | nested | 2.491906 / 1 | 2.453022 / 1 | 2.452161 / 1 |
| K | nested | 2.388432 / 1 | 2.195832 / 1 | 2.149956 / 1 |
| V | nested | 2.239233 / 1 | 2.096096 / 1 | 2.094333 / 1 |
| B | nested | 0.087293 / 1 | 0.025017 / 1 | 0.028443 / 1 |
| FA | nested | 0.058990 / 1 | 0.027021 / 1 | 0.025838 / 1 |
| FB | nested | 0.074449 / 1 | 0.047479 / 1 | 0.047619 / 1 |
| G | nested | 2.174131 / 1 | 1.970451 / 1 | 1.973116 / 1 |
| O | nested | 2.216040 / 1 | 2.075217 / 1 | 2.073083 / 1 |
| attention | boundary | 12.827996 / 1 | 11.898429 / 1 | 11.497400 / 1 |
| attention-residual | boundary | 0.002144 / 1 | 0.002224 / 1 | 0.002124 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027050 / 1 | 0.027041 / 1 | 0.026609 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009949 / 1 | 0.001924 / 1 | 0.002284 / 1 |
| router-and-top16 | boundary | 0.603267 / 1 | 0.562521 / 1 | 0.565958 / 1 |
| EDOWN | nested | 0.644274 / 1 | 0.584071 / 1 | 0.579904 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.645816 / 1 | 0.585604 / 1 | 0.581547 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027120 / 1 | 0.017754 / 1 | 0.012573 / 1 |
| SH1 | nested | 1.136974 / 1 | 1.014435 / 1 | 1.019905 / 1 |
| SH3 | nested | 1.120142 / 1 | 1.004627 / 1 | 1.009045 / 1 |
| SH2 | nested | 1.092852 / 1 | 1.007713 / 1 | 1.006029 / 1 |
| shared-expert-during-read | boundary | 3.363875 / 1 | 3.040861 / 1 | 3.049216 / 1 |
| detail:expert-gate | nested | 2.753663 / 16 | 2.631122 / 16 | 2.589698 / 16 |
| detail:expert-up | nested | 2.730651 / 16 | 2.589497 / 16 | 2.583238 / 16 |
| detail:expert-activation | nested | 0.090228 / 16 | 0.090289 / 16 | 0.092411 / 16 |
| detail:expert-down | nested | 2.905729 / 16 | 2.779923 / 16 | 2.754495 / 16 |
| EUP | nested | 0.630207 / 1 | 0.607625 / 1 | 0.606914 / 1 |
| experts-mix-normalize-up | boundary | 9.529884 / 1 | 9.083991 / 1 | 8.997058 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002885 / 1 | 0.003206 / 1 | 0.003056 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000151 / 1 | 0.000371 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.419874 / 1 | 0.368328 / 1 | 0.364962 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.419729 / 1 | 109.208482 / 1 | 109.324680 / 1 |
| op:Q   int8 projection | nested | 16.353189 / 1 | 15.106913 / 1 | 15.064293 / 1 |
| op:X   mxfp4 expert proj | nested | 8.564469 / 1 | 8.174603 / 1 | 8.104941 / 1 |
| op:N   rmsnorm | nested | 0.040646 / 1 | 0.032269 / 1 | 0.023172 / 1 |
| op:L   l2 per-head | nested | 0.005390 / 1 | 0.005400 / 1 | 0.005661 / 1 |
| op:SiTU + sigma | nested | 0.098655 / 1 | 0.098845 / 1 | 0.100837 / 1 |
| op:C   shortconv | nested | 0.068789 / 1 | 0.073928 / 1 | 0.032941 / 1 |
| op:AR  snapshot aggregate | nested | 0.033032 / 1 | 0.032410 / 1 | 0.032050 / 1 |
| op:D   kda delta-rule | nested | 0.149069 / 1 | 0.102793 / 1 | 0.102862 / 1 |
| op:router dot product | nested | 0.600351 / 1 | 0.559595 / 1 | 0.563333 / 1 |
| op:top-k selection | nested | 0.002465 / 1 | 0.002624 / 1 | 0.002244 / 1 |
| op:alpha / beta / gate | nested | 0.073528 / 1 | 0.074178 / 1 | 0.058459 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.519242 / 1 | 25.643829 / 1 | 25.153733 / 1 |

### Layer 85

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014007 / 1 | 0.006753 / 1 | 0.006242 / 1 |
| pre-attention-aggregation | boundary | 0.021991 / 1 | 0.021310 / 1 | 0.021470 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011973 / 1 | 0.011251 / 1 | 0.011001 / 1 |
| Q | nested | 2.481976 / 1 | 2.461489 / 1 | 2.464164 / 1 |
| K | nested | 2.389784 / 1 | 2.221891 / 1 | 2.172700 / 1 |
| V | nested | 2.229185 / 1 | 2.137894 / 1 | 2.087129 / 1 |
| B | nested | 0.067456 / 1 | 0.022161 / 1 | 0.021220 / 1 |
| FA | nested | 0.066204 / 1 | 0.028504 / 1 | 0.028574 / 1 |
| FB | nested | 0.073908 / 1 | 0.048922 / 1 | 0.050965 / 1 |
| G | nested | 2.189050 / 1 | 2.014834 / 1 | 2.020906 / 1 |
| O | nested | 2.215419 / 1 | 2.072181 / 1 | 2.079986 / 1 |
| attention | boundary | 12.777351 / 1 | 11.968700 / 1 | 11.577930 / 1 |
| attention-residual | boundary | 0.004268 / 1 | 0.004348 / 1 | 0.004598 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026790 / 1 | 0.026480 / 1 | 0.025959 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010099 / 1 | 0.002254 / 1 | 0.002335 / 1 |
| router-and-top16 | boundary | 0.601073 / 1 | 0.572991 / 1 | 0.566348 / 1 |
| EDOWN | nested | 0.648341 / 1 | 0.590292 / 1 | 0.586967 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.649904 / 1 | 0.592788 / 1 | 0.588520 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027021 / 1 | 0.008245 / 1 | 0.005691 / 1 |
| SH1 | nested | 1.116296 / 1 | 0.992224 / 1 | 0.990881 / 1 |
| SH3 | nested | 1.104844 / 1 | 0.991803 / 1 | 0.978388 / 1 |
| SH2 | nested | 1.095607 / 1 | 1.007442 / 1 | 0.999858 / 1 |
| shared-expert-during-read | boundary | 3.330432 / 1 | 3.012769 / 1 | 2.982873 / 1 |
| detail:expert-gate | nested | 2.757523 / 16 | 2.674288 / 16 | 2.567697 / 16 |
| detail:expert-up | nested | 2.738887 / 16 | 2.637277 / 16 | 2.569067 / 16 |
| detail:expert-activation | nested | 0.092993 / 16 | 0.090619 / 16 | 0.091158 / 16 |
| detail:expert-down | nested | 2.945012 / 16 | 2.810692 / 16 | 2.668815 / 16 |
| EUP | nested | 0.643883 / 1 | 0.607085 / 1 | 0.598198 / 1 |
| experts-mix-normalize-up | boundary | 9.601338 / 1 | 9.207292 / 1 | 8.856486 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003296 / 1 | 0.002875 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000161 / 1 | 0.000150 / 1 | 0.000160 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.429773 / 1 | 0.380902 / 1 | 0.378737 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.686823 / 1 | 111.301867 / 1 | 107.029829 / 1 |
| op:Q   int8 projection | nested | 16.320340 / 1 | 15.195079 / 1 | 15.078301 / 1 |
| op:X   mxfp4 expert proj | nested | 8.619100 / 1 | 8.297705 / 1 | 7.982614 / 1 |
| op:N   rmsnorm | nested | 0.038563 / 1 | 0.033834 / 1 | 0.023894 / 1 |
| op:L   l2 per-head | nested | 0.005360 / 1 | 0.005310 / 1 | 0.005330 / 1 |
| op:SiTU + sigma | nested | 0.101069 / 1 | 0.106398 / 1 | 0.098874 / 1 |
| op:C   shortconv | nested | 0.054622 / 1 | 0.054111 / 1 | 0.032189 / 1 |
| op:AR  snapshot aggregate | nested | 0.038661 / 1 | 0.037529 / 1 | 0.037361 / 1 |
| op:D   kda delta-rule | nested | 0.159979 / 1 | 0.115917 / 1 | 0.120325 / 1 |
| op:router dot product | nested | 0.598128 / 1 | 0.569844 / 1 | 0.563273 / 1 |
| op:top-k selection | nested | 0.002665 / 1 | 0.002615 / 1 | 0.002294 / 1 |
| op:alpha / beta / gate | nested | 0.073647 / 1 | 0.075220 / 1 | 0.058700 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.527978 / 1 | 25.837942 / 1 | 25.049348 / 1 |

### Layer 86

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014648 / 1 | 0.006322 / 1 | 0.006001 / 1 |
| pre-attention-aggregation | boundary | 0.016872 / 1 | 0.017172 / 1 | 0.016641 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011311 / 1 | 0.010629 / 1 | 0.010229 / 1 |
| Q | nested | 2.427354 / 1 | 2.421093 / 1 | 2.425261 / 1 |
| K | nested | 2.370579 / 1 | 2.190542 / 1 | 2.184642 / 1 |
| V | nested | 2.246066 / 1 | 2.119630 / 1 | 2.089233 / 1 |
| B | nested | 0.069300 / 1 | 0.024426 / 1 | 0.023654 / 1 |
| FA | nested | 0.053289 / 1 | 0.025378 / 1 | 0.025848 / 1 |
| FB | nested | 0.077635 / 1 | 0.047379 / 1 | 0.047720 / 1 |
| G | nested | 2.181376 / 1 | 2.073554 / 1 | 2.061211 / 1 |
| O | nested | 2.198698 / 1 | 1.997422 / 1 | 1.999816 / 1 |
| attention | boundary | 12.729171 / 1 | 11.885254 / 1 | 11.532235 / 1 |
| attention-residual | boundary | 0.004388 / 1 | 0.004198 / 1 | 0.004108 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026209 / 1 | 0.025918 / 1 | 0.024977 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010039 / 1 | 0.001834 / 1 | 0.002354 / 1 |
| router-and-top16 | boundary | 0.607114 / 1 | 0.573962 / 1 | 0.571328 / 1 |
| EDOWN | nested | 0.660344 / 1 | 0.589531 / 1 | 0.599260 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.661997 / 1 | 0.591224 / 1 | 0.600853 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.017533 / 1 | 0.012633 / 1 | 0.012844 / 1 |
| SH1 | nested | 1.119983 / 1 | 0.973188 / 1 | 0.978799 / 1 |
| SH3 | nested | 1.086690 / 1 | 0.974380 / 1 | 0.972998 / 1 |
| SH2 | nested | 1.095887 / 1 | 1.028251 / 1 | 1.020556 / 1 |
| shared-expert-during-read | boundary | 3.320944 / 1 | 2.989706 / 1 | 2.985918 / 1 |
| detail:expert-gate | nested | 2.805449 / 16 | 2.609696 / 16 | 2.527131 / 16 |
| detail:expert-up | nested | 2.759647 / 16 | 2.553964 / 16 | 2.536192 / 16 |
| detail:expert-activation | nested | 0.093205 / 16 | 0.092372 / 16 | 0.092454 / 16 |
| detail:expert-down | nested | 2.933138 / 16 | 2.711096 / 16 | 2.613633 / 16 |
| EUP | nested | 0.636339 / 1 | 0.600391 / 1 | 0.594951 / 1 |
| experts-mix-normalize-up | boundary | 9.632766 / 1 | 8.946113 / 1 | 8.740579 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003417 / 1 | 0.003065 / 1 | 0.003957 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000250 / 1 | 0.000241 / 1 | 0.000251 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.393735 / 1 | 0.356405 / 1 | 0.346838 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.946590 / 1 | 106.892071 / 1 | 104.497642 / 1 |
| op:Q   int8 projection | nested | 16.221987 / 1 | 15.063540 / 1 | 15.022325 / 1 |
| op:X   mxfp4 expert proj | nested | 8.676166 / 1 | 8.051692 / 1 | 7.854771 / 1 |
| op:N   rmsnorm | nested | 0.042669 / 1 | 0.033783 / 1 | 0.023586 / 1 |
| op:L   l2 per-head | nested | 0.005640 / 1 | 0.005390 / 1 | 0.006232 / 1 |
| op:SiTU + sigma | nested | 0.105768 / 1 | 0.100409 / 1 | 0.100427 / 1 |
| op:C   shortconv | nested | 0.047149 / 1 | 0.053110 / 1 | 0.031038 / 1 |
| op:AR  snapshot aggregate | nested | 0.033463 / 1 | 0.033293 / 1 | 0.031769 / 1 |
| op:D   kda delta-rule | nested | 0.125475 / 1 | 0.101660 / 1 | 0.098865 / 1 |
| op:router dot product | nested | 0.604239 / 1 | 0.571177 / 1 | 0.568873 / 1 |
| op:top-k selection | nested | 0.002415 / 1 | 0.002314 / 1 | 0.001964 / 1 |
| op:alpha / beta / gate | nested | 0.073508 / 1 | 0.071844 / 1 | 0.058910 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.469029 / 1 | 25.443244 / 1 | 24.878328 / 1 |

### Layer 87

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013024 / 1 | 0.006212 / 1 | 0.006112 / 1 |
| pre-attention-aggregation | boundary | 0.018304 / 1 | 0.017102 / 1 | 0.017282 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010159 / 1 | 0.010640 / 1 | 0.010920 / 1 |
| QA | nested | 0.309899 / 1 | 0.303377 / 1 | 0.302124 / 1 |
| QB | nested | 0.746655 / 1 | 0.704296 / 1 | 0.708053 / 1 |
| KA | nested | 0.125795 / 1 | 0.104445 / 1 | 0.104445 / 1 |
| KB | nested | 0.352379 / 1 | 0.305100 / 1 | 0.306452 / 1 |
| G | nested | 2.186545 / 1 | 2.014684 / 1 | 2.010927 / 1 |
| O | nested | 2.183910 / 1 | 1.945485 / 1 | 1.953068 / 1 |
| attention | boundary | 6.022331 / 1 | 5.475720 / 1 | 5.487401 / 1 |
| attention-residual | boundary | 0.004048 / 1 | 0.004628 / 1 | 0.004148 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025909 / 1 | 0.026850 / 1 | 0.026079 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.009668 / 1 | 0.001914 / 1 | 0.002084 / 1 |
| router-and-top16 | boundary | 0.611363 / 1 | 0.586576 / 1 | 0.579973 / 1 |
| EDOWN | nested | 0.648312 / 1 | 0.596173 / 1 | 0.601404 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.649975 / 1 | 0.597837 / 1 | 0.603097 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.038121 / 1 | 0.011912 / 1 | 0.012213 / 1 |
| SH1 | nested | 1.094235 / 1 | 0.981604 / 1 | 0.986363 / 1 |
| SH3 | nested | 1.087271 / 1 | 0.996672 / 1 | 0.986763 / 1 |
| SH2 | nested | 1.096218 / 1 | 1.014585 / 1 | 1.025626 / 1 |
| shared-expert-during-read | boundary | 3.291510 / 1 | 3.006727 / 1 | 3.012789 / 1 |
| detail:expert-gate | nested | 2.756120 / 16 | 2.607542 / 16 | 2.634732 / 16 |
| detail:expert-up | nested | 2.750716 / 16 | 2.563890 / 16 | 2.640414 / 16 |
| detail:expert-activation | nested | 0.093934 / 16 | 0.093182 / 16 | 0.097702 / 16 |
| detail:expert-down | nested | 2.898847 / 16 | 2.740499 / 16 | 2.801100 / 16 |
| EUP | nested | 0.638593 / 1 | 0.593359 / 1 | 0.605080 / 1 |
| experts-mix-normalize-up | boundary | 9.559529 / 1 | 8.975157 / 1 | 9.137631 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002995 / 1 | 0.003075 / 1 | 0.002935 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000151 / 1 | 0.000150 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006503 / 1 | 0.005480 / 1 | 0.004859 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.701325 / 1 | 109.116625 / 1 | 110.801981 / 1 |
| op:Q   int8 projection | nested | 10.468187 / 1 | 9.558357 / 1 | 9.588984 / 1 |
| op:X   mxfp4 expert proj | nested | 8.585187 / 1 | 8.090233 / 1 | 8.261273 / 1 |
| op:N   rmsnorm | nested | 0.024006 / 1 | 0.023535 / 1 | 0.023635 / 1 |
| op:SiTU + sigma | nested | 0.101960 / 1 | 0.100747 / 1 | 0.105948 / 1 |
| op:AR  snapshot aggregate | nested | 0.033903 / 1 | 0.033102 / 1 | 0.032562 / 1 |
| op:SA  softmax attention | nested | 0.011031 / 1 | 0.009408 / 1 | 0.012263 / 1 |
| op:router dot product | nested | 0.608417 / 1 | 0.583931 / 1 | 0.577269 / 1 |
| op:top-k selection | nested | 0.002434 / 1 | 0.002345 / 1 | 0.002374 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.281871 / 1 | 18.748106 / 1 | 18.926179 / 1 |

### Layer 88

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013726 / 1 | 0.005801 / 1 | 0.005490 / 1 |
| pre-attention-aggregation | boundary | 0.015960 / 1 | 0.015279 / 1 | 0.016160 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010370 / 1 | 0.010560 / 1 | 0.010119 / 1 |
| Q | nested | 2.699092 / 1 | 2.653838 / 1 | 2.664067 / 1 |
| K | nested | 2.651524 / 1 | 2.222101 / 1 | 2.184591 / 1 |
| V | nested | 2.348738 / 1 | 2.128196 / 1 | 2.080557 / 1 |
| B | nested | 0.090519 / 1 | 0.025277 / 1 | 0.026419 / 1 |
| FA | nested | 0.060203 / 1 | 0.026219 / 1 | 0.025809 / 1 |
| FB | nested | 0.073648 / 1 | 0.046818 / 1 | 0.045756 / 1 |
| G | nested | 2.165205 / 1 | 1.993033 / 1 | 1.976813 / 1 |
| O | nested | 2.284258 / 1 | 2.200100 / 1 | 2.204569 / 1 |
| attention | boundary | 13.456169 / 1 | 12.274251 / 1 | 11.872360 / 1 |
| attention-residual | boundary | 0.003566 / 1 | 0.003596 / 1 | 0.003717 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027531 / 1 | 0.027211 / 1 | 0.026609 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010620 / 1 | 0.001984 / 1 | 0.002544 / 1 |
| router-and-top16 | boundary | 0.602405 / 1 | 0.573041 / 1 | 0.570075 / 1 |
| EDOWN | nested | 0.646898 / 1 | 0.590954 / 1 | 0.592407 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.648591 / 1 | 0.592608 / 1 | 0.594060 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027321 / 1 | 0.012002 / 1 | 0.006412 / 1 |
| SH1 | nested | 1.131695 / 1 | 0.987344 / 1 | 0.986232 / 1 |
| SH3 | nested | 1.122748 / 1 | 1.011210 / 1 | 1.018713 / 1 |
| SH2 | nested | 1.112819 / 1 | 1.018723 / 1 | 1.018182 / 1 |
| shared-expert-during-read | boundary | 3.383762 / 1 | 3.031855 / 1 | 3.040210 / 1 |
| detail:expert-gate | nested | 2.732785 / 16 | 2.600509 / 16 | 2.503401 / 16 |
| detail:expert-up | nested | 2.707748 / 16 | 2.608291 / 16 | 2.491628 / 16 |
| detail:expert-activation | nested | 0.094587 / 16 | 0.094663 / 16 | 0.097862 / 16 |
| detail:expert-down | nested | 2.871977 / 16 | 2.756530 / 16 | 2.657684 / 16 |
| EUP | nested | 0.639034 / 1 | 0.596724 / 1 | 0.591275 / 1 |
| experts-mix-normalize-up | boundary | 9.452540 / 1 | 9.054716 / 1 | 8.712176 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002314 / 1 | 0.002634 / 1 | 0.002325 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000390 / 1 | 0.000170 / 1 | 0.000161 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.450401 / 1 | 0.413352 / 1 | 0.398915 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.588642 / 1 | 110.127303 / 1 | 105.621097 / 1 |
| op:Q   int8 projection | nested | 17.024646 / 1 | 15.498853 / 1 | 15.413728 / 1 |
| op:X   mxfp4 expert proj | nested | 8.493598 / 1 | 8.154143 / 1 | 7.840149 / 1 |
| op:N   rmsnorm | nested | 0.036419 / 1 | 0.033342 / 1 | 0.024294 / 1 |
| op:L   l2 per-head | nested | 0.005530 / 1 | 0.005371 / 1 | 0.005941 / 1 |
| op:SiTU + sigma | nested | 0.103054 / 1 | 0.103130 / 1 | 0.107581 / 1 |
| op:C   shortconv | nested | 0.066995 / 1 | 0.079449 / 1 | 0.034104 / 1 |
| op:AR  snapshot aggregate | nested | 0.033342 / 1 | 0.032461 / 1 | 0.033293 / 1 |
| op:D   kda delta-rule | nested | 0.145122 / 1 | 0.103994 / 1 | 0.108192 / 1 |
| op:router dot product | nested | 0.599720 / 1 | 0.570476 / 1 | 0.567510 / 1 |
| op:top-k selection | nested | 0.002425 / 1 | 0.002235 / 1 | 0.002154 / 1 |
| op:alpha / beta / gate | nested | 0.074870 / 1 | 0.073447 / 1 | 0.058549 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 28.124523 / 1 | 26.037785 / 1 | 25.280159 / 1 |

### Layer 89

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014808 / 1 | 0.006061 / 1 | 0.005931 / 1 |
| pre-attention-aggregation | boundary | 0.020147 / 1 | 0.018264 / 1 | 0.017774 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010951 / 1 | 0.011001 / 1 | 0.010951 / 1 |
| Q | nested | 2.474503 / 1 | 2.444126 / 1 | 2.442032 / 1 |
| K | nested | 2.378965 / 1 | 2.177428 / 1 | 2.193939 / 1 |
| V | nested | 2.236468 / 1 | 2.107888 / 1 | 2.097258 / 1 |
| B | nested | 0.098654 / 1 | 0.029796 / 1 | 0.029976 / 1 |
| FA | nested | 0.058449 / 1 | 0.027150 / 1 | 0.026529 / 1 |
| FB | nested | 0.074129 / 1 | 0.046917 / 1 | 0.047219 / 1 |
| G | nested | 2.168321 / 1 | 1.964350 / 1 | 1.961364 / 1 |
| O | nested | 2.238222 / 1 | 2.162290 / 1 | 2.177158 / 1 |
| attention | boundary | 12.813298 / 1 | 11.963971 / 1 | 11.682395 / 1 |
| attention-residual | boundary | 0.004007 / 1 | 0.004658 / 1 | 0.004187 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027421 / 1 | 0.026960 / 1 | 0.026359 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010720 / 1 | 0.002224 / 1 | 0.001934 / 1 |
| router-and-top16 | boundary | 0.602105 / 1 | 0.579663 / 1 | 0.569254 / 1 |
| EDOWN | nested | 0.655264 / 1 | 0.610230 / 1 | 0.602386 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.656887 / 1 | 0.611813 / 1 | 0.604039 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042640 / 1 | 0.017573 / 1 | 0.006152 / 1 |
| SH1 | nested | 1.119040 / 1 | 0.996381 / 1 | 1.009586 / 1 |
| SH3 | nested | 1.103341 / 1 | 0.984930 / 1 | 0.993256 / 1 |
| SH2 | nested | 1.087832 / 1 | 1.002423 / 1 | 0.992314 / 1 |
| shared-expert-during-read | boundary | 3.325122 / 1 | 2.998171 / 1 | 3.009993 / 1 |
| detail:expert-gate | nested | 2.724040 / 16 | 2.609646 / 16 | 2.563629 / 16 |
| detail:expert-up | nested | 2.718269 / 16 | 2.636893 / 16 | 2.580311 / 16 |
| detail:expert-activation | nested | 0.094459 / 16 | 0.098053 / 16 | 0.104305 / 16 |
| detail:expert-down | nested | 2.899990 / 16 | 2.778901 / 16 | 2.687860 / 16 |
| EUP | nested | 0.625488 / 1 | 0.600992 / 1 | 0.600542 / 1 |
| experts-mix-normalize-up | boundary | 9.503815 / 1 | 9.118385 / 1 | 8.875792 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002225 / 1 | 0.002985 / 1 | 0.003026 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000230 / 1 | 0.000140 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.423090 / 1 | 0.364491 / 1 | 0.369290 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.389631 / 1 | 110.438098 / 1 | 107.933956 / 1 |
| op:Q   int8 projection | nested | 16.317055 / 1 | 15.153249 / 1 | 15.171788 / 1 |
| op:X   mxfp4 expert proj | nested | 8.523986 / 1 | 8.211050 / 1 | 8.024241 / 1 |
| op:N   rmsnorm | nested | 0.032449 / 1 | 0.037750 / 1 | 0.023613 / 1 |
| op:L   l2 per-head | nested | 0.006142 / 1 | 0.006051 / 1 | 0.005631 / 1 |
| op:SiTU + sigma | nested | 0.103314 / 1 | 0.106358 / 1 | 0.113270 / 1 |
| op:C   shortconv | nested | 0.060163 / 1 | 0.054582 / 1 | 0.031690 / 1 |
| op:AR  snapshot aggregate | nested | 0.037149 / 1 | 0.034485 / 1 | 0.033481 / 1 |
| op:D   kda delta-rule | nested | 0.141785 / 1 | 0.104385 / 1 | 0.107391 / 1 |
| op:router dot product | nested | 0.599210 / 1 | 0.577068 / 1 | 0.566779 / 1 |
| op:top-k selection | nested | 0.002565 / 1 | 0.002134 / 1 | 0.002205 / 1 |
| op:alpha / beta / gate | nested | 0.072626 / 1 | 0.072215 / 1 | 0.066374 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.478897 / 1 | 25.746430 / 1 | 25.207584 / 1 |

### Layer 90

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.014226 / 1 | 0.006763 / 1 | 0.006823 / 1 |
| pre-attention-aggregation | boundary | 0.020028 / 1 | 0.019907 / 1 | 0.020378 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010931 / 1 | 0.011712 / 1 | 0.011100 / 1 |
| Q | nested | 2.471778 / 1 | 2.444837 / 1 | 2.442122 / 1 |
| K | nested | 2.375227 / 1 | 2.199479 / 1 | 2.195622 / 1 |
| V | nested | 2.238021 / 1 | 2.111004 / 1 | 2.116574 / 1 |
| B | nested | 0.071434 / 1 | 0.029024 / 1 | 0.033904 / 1 |
| FA | nested | 0.065553 / 1 | 0.027191 / 1 | 0.031569 / 1 |
| FB | nested | 0.078217 / 1 | 0.049602 / 1 | 0.049472 / 1 |
| G | nested | 2.168071 / 1 | 2.008562 / 1 | 1.998163 / 1 |
| O | nested | 2.250645 / 1 | 2.173902 / 1 | 2.183700 / 1 |
| attention | boundary | 12.807437 / 1 | 12.067655 / 1 | 11.788244 / 1 |
| attention-residual | boundary | 0.004148 / 1 | 0.004108 / 1 | 0.004158 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026860 / 1 | 0.026389 / 1 | 0.026379 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.014638 / 1 | 0.002114 / 1 | 0.001973 / 1 |
| router-and-top16 | boundary | 0.602275 / 1 | 0.570786 / 1 | 0.577278 / 1 |
| EDOWN | nested | 0.689078 / 1 | 0.642530 / 1 | 0.627863 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.690881 / 1 | 0.644193 / 1 | 0.629446 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.033062 / 1 | 0.012394 / 1 | 0.013465 / 1 |
| SH1 | nested | 1.122748 / 1 | 0.989439 / 1 | 0.986013 / 1 |
| SH3 | nested | 1.136033 / 1 | 1.010078 / 1 | 1.014565 / 1 |
| SH2 | nested | 1.101507 / 1 | 1.021639 / 1 | 1.010768 / 1 |
| shared-expert-during-read | boundary | 3.375527 / 1 | 3.035090 / 1 | 3.025733 / 1 |
| detail:expert-gate | nested | 2.752623 / 16 | 2.623069 / 16 | 2.545327 / 16 |
| detail:expert-up | nested | 2.724952 / 16 | 2.627138 / 16 | 2.482800 / 16 |
| detail:expert-activation | nested | 0.093687 / 16 | 0.093546 / 16 | 0.095229 / 16 |
| detail:expert-down | nested | 2.943179 / 16 | 2.730069 / 16 | 2.655692 / 16 |
| EUP | nested | 0.631480 / 1 | 0.607966 / 1 | 0.593979 / 1 |
| experts-mix-normalize-up | boundary | 9.564208 / 1 | 9.066960 / 1 | 8.751550 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002635 / 1 | 0.002745 / 1 | 0.003176 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000230 / 1 | 0.000230 / 1 | 0.000221 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.419904 / 1 | 0.381152 / 1 | 0.387083 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.626796 / 1 | 108.943280 / 1 | 105.849201 / 1 |
| op:Q   int8 projection | nested | 16.398187 / 1 | 15.313308 / 1 | 15.282720 / 1 |
| op:X   mxfp4 expert proj | nested | 8.602982 / 1 | 8.163121 / 1 | 7.866987 / 1 |
| op:N   rmsnorm | nested | 0.040826 / 1 | 0.032701 / 1 | 0.022974 / 1 |
| op:L   l2 per-head | nested | 0.005441 / 1 | 0.005290 / 1 | 0.005480 / 1 |
| op:SiTU + sigma | nested | 0.102592 / 1 | 0.101769 / 1 | 0.103966 / 1 |
| op:C   shortconv | nested | 0.054592 / 1 | 0.059410 / 1 | 0.032510 / 1 |
| op:AR  snapshot aggregate | nested | 0.037339 / 1 | 0.036448 / 1 | 0.037220 / 1 |
| op:D   kda delta-rule | nested | 0.135202 / 1 | 0.108653 / 1 | 0.110196 / 1 |
| op:router dot product | nested | 0.599160 / 1 | 0.568231 / 1 | 0.574453 / 1 |
| op:top-k selection | nested | 0.002545 / 1 | 0.002154 / 1 | 0.002385 / 1 |
| op:alpha / beta / gate | nested | 0.074439 / 1 | 0.073778 / 1 | 0.058920 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 27.608328 / 1 | 25.873007 / 1 | 25.267265 / 1 |

### Layer 91

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.013245 / 1 | 0.006092 / 1 | 0.006432 / 1 |
| pre-attention-aggregation | boundary | 0.025267 / 1 | 0.021240 / 1 | 0.020578 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.010710 / 1 | 0.011722 / 1 | 0.011090 / 1 |
| QA | nested | 0.324155 / 1 | 0.311401 / 1 | 0.312303 / 1 |
| QB | nested | 0.721819 / 1 | 0.704536 / 1 | 0.681954 / 1 |
| KA | nested | 0.126927 / 1 | 0.098043 / 1 | 0.097662 / 1 |
| KB | nested | 0.351697 / 1 | 0.319708 / 1 | 0.318795 / 1 |
| G | nested | 2.166587 / 1 | 1.958839 / 1 | 1.975009 / 1 |
| O | nested | 2.195682 / 1 | 2.054297 / 1 | 2.040072 / 1 |
| attention | boundary | 5.996082 / 1 | 5.545992 / 1 | 5.526705 / 1 |
| attention-residual | boundary | 0.004458 / 1 | 0.004508 / 1 | 0.004468 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026659 / 1 | 0.025978 / 1 | 0.026249 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010299 / 1 | 0.002314 / 1 | 0.002384 / 1 |
| router-and-top16 | boundary | 0.597406 / 1 | 0.562371 / 1 | 0.567661 / 1 |
| EDOWN | nested | 0.635848 / 1 | 0.570326 / 1 | 0.569253 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.637521 / 1 | 0.571939 / 1 | 0.570876 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026549 / 1 | 0.006432 / 1 | 0.006513 / 1 |
| SH1 | nested | 1.086910 / 1 | 0.978578 / 1 | 0.974601 / 1 |
| SH3 | nested | 1.131864 / 1 | 1.091389 / 1 | 1.089105 / 1 |
| SH2 | nested | 1.083905 / 1 | 0.984770 / 1 | 0.973428 / 1 |
| shared-expert-during-read | boundary | 3.317217 / 1 | 3.069014 / 1 | 3.052042 / 1 |
| detail:expert-gate | nested | 2.736133 / 16 | 2.642226 / 16 | 2.571074 / 16 |
| detail:expert-up | nested | 2.717076 / 16 | 2.610629 / 16 | 2.549220 / 16 |
| detail:expert-activation | nested | 0.091440 / 16 | 0.096983 / 16 | 0.104602 / 16 |
| detail:expert-down | nested | 2.924544 / 16 | 2.799911 / 16 | 2.690035 / 16 |
| EUP | nested | 0.630939 / 1 | 0.602867 / 1 | 0.591245 / 1 |
| experts-mix-normalize-up | boundary | 9.507593 / 1 | 9.131801 / 1 | 8.845506 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002926 / 1 | 0.002635 / 1 | 0.002976 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000231 / 1 | 0.000351 / 1 | 0.000140 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006913 / 1 | 0.005751 / 1 | 0.005490 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.591618 / 1 | 110.650174 / 1 | 107.350916 / 1 |
| op:Q   int8 projection | nested | 10.455010 / 1 | 9.673313 / 1 | 9.622094 / 1 |
| op:X   mxfp4 expert proj | nested | 8.560162 / 1 | 8.238864 / 1 | 8.007313 / 1 |
| op:N   rmsnorm | nested | 0.024235 / 1 | 0.023963 / 1 | 0.023593 / 1 |
| op:SiTU + sigma | nested | 0.100045 / 1 | 0.105477 / 1 | 0.113180 / 1 |
| op:AR  snapshot aggregate | nested | 0.040937 / 1 | 0.036750 / 1 | 0.036318 / 1 |
| op:SA  softmax attention | nested | 0.006562 / 1 | 0.009297 / 1 | 0.011622 / 1 |
| op:router dot product | nested | 0.594931 / 1 | 0.559606 / 1 | 0.564985 / 1 |
| op:top-k selection | nested | 0.002184 / 1 | 0.002134 / 1 | 0.002024 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.202143 / 1 | 18.986431 / 1 | 18.668447 / 1 |

### Layer 92

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.012623 / 1 | 0.006302 / 1 | 0.006001 / 1 |
| pre-attention-aggregation | boundary | 0.016160 / 1 | 0.015809 / 1 | 0.015399 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.014918 / 1 | 0.010420 / 1 | 0.009548 / 1 |
| QA | nested | 0.277018 / 1 | 0.254024 / 1 | 0.259245 / 1 |
| QB | nested | 0.720196 / 1 | 0.691723 / 1 | 0.673609 / 1 |
| KA | nested | 0.130063 / 1 | 0.101770 / 1 | 0.099697 / 1 |
| KB | nested | 0.348952 / 1 | 0.303988 / 1 | 0.306302 / 1 |
| G | nested | 2.191304 / 1 | 2.034201 / 1 | 2.004284 / 1 |
| O | nested | 2.168021 / 1 | 2.001119 / 1 | 1.989878 / 1 |
| attention | boundary | 5.949385 / 1 | 5.499935 / 1 | 5.440104 / 1 |
| attention-residual | boundary | 0.004469 / 1 | 0.004399 / 1 | 0.004208 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026951 / 1 | 0.026890 / 1 | 0.026019 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.010630 / 1 | 0.002274 / 1 | 0.002315 / 1 |
| router-and-top16 | boundary | 0.599610 / 1 | 0.567450 / 1 | 0.573431 / 1 |
| EDOWN | nested | 0.640616 / 1 | 0.587567 / 1 | 0.593940 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.642219 / 1 | 0.589210 / 1 | 0.595553 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.021710 / 1 | 0.005811 / 1 | 0.005981 / 1 |
| SH1 | nested | 1.090196 / 1 | 1.000218 / 1 | 0.992033 / 1 |
| SH3 | nested | 1.080098 / 1 | 0.978999 / 1 | 0.983708 / 1 |
| SH2 | nested | 1.094194 / 1 | 1.011950 / 1 | 1.011720 / 1 |
| shared-expert-during-read | boundary | 3.278536 / 1 | 3.005224 / 1 | 3.001687 / 1 |
| detail:expert-gate | nested | 2.741874 / 16 | 2.629703 / 16 | 2.672171 / 16 |
| detail:expert-up | nested | 2.734106 / 16 | 2.630395 / 16 | 2.638811 / 16 |
| detail:expert-activation | nested | 0.098342 / 16 | 0.099107 / 16 | 0.097989 / 16 |
| detail:expert-down | nested | 2.945433 / 16 | 2.798349 / 16 | 2.841668 / 16 |
| EUP | nested | 0.642651 / 1 | 0.597306 / 1 | 0.600552 / 1 |
| experts-mix-normalize-up | boundary | 9.562996 / 1 | 9.136148 / 1 | 9.183146 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003006 / 1 | 0.003156 / 1 | 0.003016 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000060 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.006302 / 1 | 0.005179 / 1 | 0.005391 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 115.609486 / 1 | 111.370545 / 1 | 112.433128 / 1 |
| op:Q   int8 projection | nested | 10.381828 / 1 | 9.561540 / 1 | 9.513264 / 1 |
| op:X   mxfp4 expert proj | nested | 8.612241 / 1 | 8.246826 / 1 | 8.344439 / 1 |
| op:N   rmsnorm | nested | 0.028342 / 1 | 0.023785 / 1 | 0.023354 / 1 |
| op:SiTU + sigma | nested | 0.106608 / 1 | 0.107432 / 1 | 0.106465 / 1 |
| op:AR  snapshot aggregate | nested | 0.032341 / 1 | 0.031519 / 1 | 0.030977 / 1 |
| op:SA  softmax attention | nested | 0.007224 / 1 | 0.008055 / 1 | 0.011160 / 1 |
| op:router dot product | nested | 0.596614 / 1 | 0.565056 / 1 | 0.570966 / 1 |
| op:top-k selection | nested | 0.002585 / 1 | 0.001963 / 1 | 0.002024 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 20.169631 / 1 | 18.897465 / 1 | 18.891143 / 1 |

### Output

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| final-aggregation-and-normalization | boundary | not executed | 0.044242 / 1 | 0.043591 / 1 |
| head-fill-and-projection | boundary | not executed | 54.090383 / 1 | 50.937112 / 1 |
| argmax-and-cleanup | boundary | not executed | 0.160229 / 1 | 0.105818 / 1 |

