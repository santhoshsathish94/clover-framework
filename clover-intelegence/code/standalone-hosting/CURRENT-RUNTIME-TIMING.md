# Current Runtime Timing

Current source, AX102, 2026-10-02, default generation.json, 16 threads close/core affinity; timing only; 2 input positions and 1 continuation; no experimental decoder.

Source: [timing data](stage-profile-current-20261002.json). Output assertions passed: IDs 418 then 276 (Rain falls -> on the). All 279 layer totals are present, with 35-50 recorded measurements per layer, depending on layer type. These are boundaries and nested measurements, not a claim that every scalar operation has a separate timer.

Startup: **5.132054 s**. First output (two input positions): **44.868743 s**. Next output (one continuation position): **16.532759 s**. This measures the current source with profiling enabled; it is not a retroactive decomposition of the earlier 45.028-second run.

## Elapsed-Time Breakdown

| Work | First output, two input positions (s) | Next output (s) |
|---|---:|---:|
| Routed experts, mixing, latent norm and up projection | 26.340246 | 14.760274 |
| Vocabulary head preparation and projection | 15.295462 | 0.051458 |
| Attention across all 93 layers | 2.176674 | 1.178713 |
| Shared experts | 0.615059 | 0.315575 |
| Residual/state save, unbind and next-layer prediction | 0.161232 | 0.082961 |
| Latent down projection | 0.111183 | 0.059347 |
| Live router and top-16 | 0.103850 | 0.053125 |
| Layer-0 dense MLP | 0.033484 | 0.015998 |
| Other boundaries and unassigned timer overhead | 0.031553 | 0.015308 |
| Total | 44.868743 | 16.532759 |

First head use prepares the retained BF16 table and scores it. The head timer does not separate reading, decompression, layout conversion, checksums and scoring. The next output reuses that head table, not a cached answer. Dataset preparation is not run.

## Inside Routed Experts

Gate/up/down wall timers are nested inside routed-expert processing above; do not add them again. Each includes its block decoding, validation and multiplication.

| Nested measurement | First output (s) | Next output (s) |
|---|---:|---:|
| detail:expert-gate | 9.156122 | 4.945867 |
| detail:expert-up | 7.940388 | 4.794922 |
| detail:expert-activation | 0.017925 | 0.011806 |
| detail:expert-down | 9.038666 | 4.836570 |
| detail:read-ahead-wait | 0.000273 | 0.075618 |

Worker times below sum concurrent threads. They locate CPU work but are **not elapsed seconds**, and must not be added to the wall-time table. The read wrapper accesses already-prefetched bytes; it is not total disk I/O time.

| Worker phase | First output (summed worker s) | Next output (summed worker s) |
|---|---:|---:|
| read | 0.018349 | 0.009028 |
| decode | 197.348522 | 99.575387 |
| crc | 110.288644 | 55.716128 |
| math | 66.935094 | 34.382094 |

## Every Layer

Elapsed milliseconds. Layer 0 is dense; all remaining layers have routed and shared experts.

| Layer | Attention | First input | Second input | Continuation |
|---:|---|---:|---:|---:|
| 0 | KDA | 32.927382 | 30.136870 | 29.253833 |
| 1 | KDA | 164.339672 | 161.466989 | 185.991043 |
| 2 | KDA | 161.294034 | 161.801424 | 179.458102 |
| 3 | MLA | 155.016671 | 156.979138 | 175.025665 |
| 4 | KDA | 161.791844 | 162.090454 | 175.897353 |
| 5 | KDA | 161.281380 | 161.617961 | 174.836492 |
| 6 | KDA | 161.392388 | 161.548662 | 177.406621 |
| 7 | MLA | 154.761425 | 154.975297 | 171.854353 |
| 8 | KDA | 161.293693 | 161.597142 | 179.224968 |
| 9 | KDA | 161.337085 | 160.498761 | 173.753720 |
| 10 | KDA | 163.915590 | 163.127902 | 175.378083 |
| 11 | MLA | 153.821901 | 155.457056 | 159.639050 |
| 12 | KDA | 159.275915 | 159.490548 | 173.730226 |
| 13 | KDA | 161.022768 | 162.268316 | 180.955559 |
| 14 | KDA | 161.167027 | 162.172467 | 181.728222 |
| 15 | MLA | 152.797497 | 154.947054 | 171.720443 |
| 16 | KDA | 159.818699 | 159.974863 | 174.970582 |
| 17 | KDA | 161.516500 | 161.809640 | 171.304846 |
| 18 | KDA | 160.080228 | 160.933753 | 169.376204 |
| 19 | MLA | 162.292910 | 154.472016 | 165.579494 |
| 20 | KDA | 160.853121 | 166.456838 | 174.391602 |
| 21 | KDA | 159.971165 | 160.480688 | 179.962275 |
| 22 | KDA | 160.599277 | 161.268399 | 179.962986 |
| 23 | MLA | 154.045367 | 154.714379 | 170.279281 |
| 24 | KDA | 161.784641 | 162.834524 | 176.158661 |
| 25 | KDA | 161.815599 | 162.209497 | 184.858469 |
| 26 | KDA | 161.050540 | 165.212295 | 183.102569 |
| 27 | MLA | 162.512110 | 154.347915 | 174.872991 |
| 28 | KDA | 162.113145 | 161.505722 | 183.018032 |
| 29 | KDA | 164.068717 | 161.357706 | 181.934128 |
| 30 | KDA | 161.280600 | 161.651664 | 185.730798 |
| 31 | MLA | 155.059462 | 156.183252 | 171.189271 |
| 32 | KDA | 161.466998 | 162.651743 | 178.075061 |
| 33 | KDA | 161.955570 | 162.897953 | 182.812088 |
| 34 | KDA | 161.390615 | 162.099932 | 185.116802 |
| 35 | MLA | 155.649965 | 156.215163 | 175.416966 |
| 36 | KDA | 160.690648 | 161.370960 | 186.524600 |
| 37 | KDA | 161.866725 | 161.475235 | 185.796961 |
| 38 | KDA | 161.789190 | 161.556327 | 172.815999 |
| 39 | MLA | 155.190407 | 156.088977 | 167.261596 |
| 40 | KDA | 162.282541 | 161.505512 | 177.089571 |
| 41 | KDA | 161.968575 | 163.769801 | 178.499564 |
| 42 | KDA | 161.284076 | 161.722177 | 181.154722 |
| 43 | MLA | 154.576300 | 154.375697 | 178.814732 |
| 44 | KDA | 180.962997 | 180.569725 | 187.699405 |
| 45 | KDA | 160.378776 | 162.321507 | 184.259441 |
| 46 | KDA | 161.043929 | 161.430061 | 181.667259 |
| 47 | MLA | 154.639107 | 154.735840 | 176.688121 |
| 48 | KDA | 162.476824 | 162.589066 | 181.212761 |
| 49 | KDA | 162.411432 | 161.582176 | 177.020181 |
| 50 | KDA | 167.893850 | 161.652407 | 181.565119 |
| 51 | MLA | 154.406744 | 154.936535 | 176.148523 |
| 52 | KDA | 164.466801 | 168.462044 | 176.407878 |
| 53 | KDA | 163.837736 | 161.802478 | 183.653009 |
| 54 | KDA | 161.592973 | 161.043970 | 182.322884 |
| 55 | MLA | 156.651996 | 158.684434 | 177.595005 |
| 56 | KDA | 161.558089 | 163.036341 | 176.635212 |
| 57 | KDA | 161.590630 | 162.320114 | 174.826064 |
| 58 | KDA | 160.976933 | 162.388031 | 182.762916 |
| 59 | MLA | 154.455755 | 155.152007 | 169.833269 |
| 60 | KDA | 165.004394 | 168.531895 | 181.740166 |
| 61 | KDA | 163.024036 | 161.912863 | 183.217926 |
| 62 | KDA | 162.595266 | 161.318513 | 182.636480 |
| 63 | MLA | 153.482798 | 154.458382 | 166.169257 |
| 64 | KDA | 161.001499 | 160.903769 | 180.644831 |
| 65 | KDA | 160.124521 | 159.932876 | 170.179907 |
| 66 | KDA | 160.595261 | 161.152744 | 179.708131 |
| 67 | MLA | 153.477608 | 154.087289 | 166.307396 |
| 68 | KDA | 160.504621 | 160.503742 | 181.288282 |
| 69 | KDA | 160.702311 | 160.889593 | 175.762082 |
| 70 | KDA | 160.669479 | 160.960896 | 180.478911 |
| 71 | MLA | 152.550628 | 155.062871 | 167.152263 |
| 72 | KDA | 161.259141 | 159.992067 | 184.559451 |
| 73 | KDA | 159.994739 | 160.619739 | 192.729119 |
| 74 | KDA | 160.250177 | 161.386481 | 188.226009 |
| 75 | MLA | 153.524016 | 156.352731 | 178.534600 |
| 76 | KDA | 160.625207 | 161.227734 | 183.207006 |
| 77 | KDA | 160.595161 | 160.484537 | 180.569130 |
| 78 | KDA | 160.295320 | 164.962138 | 182.358551 |
| 79 | MLA | 153.983183 | 158.295729 | 178.402253 |
| 80 | KDA | 160.739451 | 160.881277 | 191.405828 |
| 81 | KDA | 163.593079 | 162.888406 | 180.521511 |
| 82 | KDA | 161.183900 | 162.428608 | 183.601934 |
| 83 | MLA | 154.246795 | 158.474162 | 175.934324 |
| 84 | KDA | 161.915327 | 163.953344 | 186.382396 |
| 85 | KDA | 160.439390 | 161.199502 | 182.454020 |
| 86 | KDA | 160.268050 | 161.264523 | 182.402845 |
| 87 | MLA | 154.046151 | 154.931757 | 175.122959 |
| 88 | KDA | 160.700899 | 161.448998 | 182.222377 |
| 89 | KDA | 161.306890 | 162.469664 | 183.462363 |
| 90 | KDA | 169.443145 | 161.993384 | 185.543660 |
| 91 | MLA | 154.927066 | 154.412477 | 178.782273 |
| 92 | MLA | 153.686439 | 153.867038 | 192.705585 |

## Every Recorded Measurement

Milliseconds. Boundary rows are disjoint within their layer; projections, operator aggregates, details and worker rows are nested or parallel and cannot be summed with them.

### Input

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| request-reset | boundary | 2.308231 / 1 | 0.145171 / 1 | 0.433179 / 1 |
| embedding | boundary | 1.595319 / 1 | 1.472059 / 1 | 2.264580 / 1 |
| scratch-setup | boundary | 0.015449 / 1 | 0.000281 / 1 | 0.000761 / 1 |

### Layer 0

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.015509 / 1 | 0.006051 / 1 | 0.005490 / 1 |
| pre-attention-aggregation | boundary | 0.016982 / 1 | 0.001964 / 1 | 0.002705 / 1 |
| snapshot-push | boundary | 0.000691 / 1 | 0.000491 / 1 | 0.000711 / 1 |
| pre-attention-normalization | boundary | 0.552962 / 1 | 0.024155 / 1 | 0.036057 / 1 |
| Q | nested | 3.557534 / 1 | 2.590387 / 1 | 2.448954 / 1 |
| K | nested | 2.816340 / 1 | 2.406834 / 1 | 2.313580 / 1 |
| V | nested | 2.341914 / 1 | 2.320223 / 1 | 2.190110 / 1 |
| B | nested | 0.023133 / 1 | 0.027011 / 1 | 0.023865 / 1 |
| FA | nested | 0.028494 / 1 | 0.029495 / 1 | 0.026960 / 1 |
| FB | nested | 0.061635 / 1 | 0.051697 / 1 | 0.047499 / 1 |
| G | nested | 2.156287 / 1 | 2.061780 / 1 | 1.996359 / 1 |
| O | nested | 2.095343 / 1 | 2.129347 / 1 | 2.000545 / 1 |
| attention | boundary | 14.221298 / 1 | 12.615159 / 1 | 12.216164 / 1 |
| attention-residual | boundary | 0.001322 / 1 | 0.001482 / 1 | 0.001322 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.039434 / 1 | 0.031719 / 1 | 0.025477 / 1 |
| MGATE | nested | 5.735661 / 1 | 5.458744 / 1 | 5.388192 / 1 |
| MUP | nested | 5.681109 / 1 | 5.481376 / 1 | 5.254924 / 1 |
| MDOWN | nested | 5.532482 / 1 | 5.392882 / 1 | 5.314676 / 1 |
| dense-mlp | boundary | 17.087260 / 1 | 16.397212 / 1 | 15.997686 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.566417 / 1 | 0.586816 / 1 | 0.567740 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.991080 / 1 | 1.057515 / 1 | 0.967176 / 1 |
| worker:read | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:decode | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:crc | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| worker:math | parallel worker | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| op:Q   int8 projection | nested | 30.028608 / 1 | 27.948614 / 1 | 27.004512 / 1 |
| op:N   rmsnorm | nested | 0.579203 / 1 | 0.058449 / 1 | 0.062116 / 1 |
| op:L   l2 per-head | nested | 0.006041 / 1 | 0.005760 / 1 | 0.005309 / 1 |
| op:SiTU + sigma | nested | 0.135493 / 1 | 0.038732 / 1 | 0.036388 / 1 |
| op:C   shortconv | nested | 0.096078 / 1 | 0.066563 / 1 | 0.070282 / 1 |
| op:AR  snapshot aggregate | nested | 0.022362 / 1 | 0.017263 / 1 | 0.014768 / 1 |
| op:D   kda delta-rule | nested | 0.179846 / 1 | 0.149109 / 1 | 0.210222 / 1 |
| op:alpha / beta / gate | nested | 0.087193 / 1 | 0.077345 / 1 | 0.075050 / 1 |
| detail:read-ahead-wait | nested | 0.000000 / 1 | 0.000000 / 1 | 0.000000 / 1 |
| total:layer | total | 32.927382 / 1 | 30.136870 / 1 | 29.253833 / 1 |

### Layer 1

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005681 / 1 | 0.005630 / 1 | 0.005230 / 1 |
| pre-attention-aggregation | boundary | 0.018955 / 1 | 0.017613 / 1 | 0.018995 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000080 / 1 | 0.000141 / 1 |
| pre-attention-normalization | boundary | 0.012122 / 1 | 0.015609 / 1 | 0.012393 / 1 |
| Q | nested | 3.461705 / 1 | 3.507901 / 1 | 3.708355 / 1 |
| K | nested | 3.313148 / 1 | 2.290448 / 1 | 3.387516 / 1 |
| V | nested | 3.002367 / 1 | 2.199187 / 1 | 3.390822 / 1 |
| B | nested | 0.025248 / 1 | 0.020789 / 1 | 0.031839 / 1 |
| FA | nested | 0.032310 / 1 | 0.032421 / 1 | 0.041607 / 1 |
| FB | nested | 0.049923 / 1 | 0.049623 / 1 | 0.071393 / 1 |
| G | nested | 2.149454 / 1 | 1.976992 / 1 | 2.779922 / 1 |
| O | nested | 2.078251 / 1 | 2.039018 / 1 | 2.107646 / 1 |
| attention | boundary | 15.086815 / 1 | 13.338119 / 1 | 16.499784 / 1 |
| attention-residual | boundary | 0.002745 / 1 | 0.003636 / 1 | 0.003597 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025577 / 1 | 0.023594 / 1 | 0.024897 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003116 / 1 | 0.002244 / 1 | 0.003216 / 1 |
| router-and-top16 | boundary | 0.583059 / 1 | 0.509662 / 1 | 0.584271 / 1 |
| EDOWN | nested | 0.601623 / 1 | 0.575585 / 1 | 0.599850 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.601964 / 1 | 0.575916 / 1 | 0.600421 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.104375 / 1 | 0.034485 / 1 | 0.016140 / 1 |
| SH1 | nested | 1.028120 / 1 | 0.970632 / 1 | 1.008493 / 1 |
| SH3 | nested | 1.022880 / 1 | 0.975822 / 1 | 1.006800 / 1 |
| SH2 | nested | 0.991491 / 1 | 0.987935 / 1 | 0.992153 / 1 |
| shared-expert-during-read | boundary | 3.052040 / 1 | 2.944989 / 1 | 3.017114 / 1 |
| detail:expert-gate | nested | 50.107324 / 16 | 49.955412 / 16 | 54.434055 / 16 |
| detail:expert-up | nested | 43.520501 / 16 | 42.863510 / 16 | 54.151929 / 16 |
| detail:expert-activation | nested | 0.095377 / 16 | 0.095481 / 16 | 0.129031 / 16 |
| detail:expert-down | nested | 48.987294 / 16 | 49.130331 / 16 | 54.355241 / 16 |
| EUP | nested | 0.604058 / 1 | 0.611702 / 1 | 0.619858 / 1 |
| experts-mix-normalize-up | boundary | 143.761973 / 1 | 143.028455 / 1 | 164.049818 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002364 / 1 | 0.002976 / 1 | 0.003136 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.604709 / 1 | 0.593158 / 1 | 0.641058 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.076901 / 1 | 0.962077 / 1 | 1.149757 / 1 |
| worker:read | parallel worker | 0.068015 / 1 | 0.069726 / 1 | 0.069762 / 1 |
| worker:decode | parallel worker | 1048.455337 / 1 | 1079.749273 / 1 | 1072.019308 / 1 |
| worker:crc | parallel worker | 580.683361 / 1 | 598.952391 / 1 | 594.877250 / 1 |
| worker:math | parallel worker | 371.780202 / 1 | 363.956814 / 1 | 378.846152 / 1 |
| op:Q   int8 projection | nested | 18.358754 / 1 | 16.236572 / 1 | 19.744653 / 1 |
| op:X   mxfp4 expert proj | nested | 142.723435 / 1 | 142.057682 / 1 | 163.085385 / 1 |
| op:N   rmsnorm | nested | 0.040305 / 1 | 0.042460 / 1 | 0.032639 / 1 |
| op:L   l2 per-head | nested | 0.005881 / 1 | 0.005791 / 1 | 0.005600 / 1 |
| op:SiTU + sigma | nested | 0.102260 / 1 | 0.103296 / 1 | 0.135403 / 1 |
| op:C   shortconv | nested | 0.076493 / 1 | 0.049352 / 1 | 0.038933 / 1 |
| op:AR  snapshot aggregate | nested | 0.034174 / 1 | 0.031178 / 1 | 0.033111 / 1 |
| op:D   kda delta-rule | nested | 0.134672 / 1 | 0.228306 / 1 | 0.156301 / 1 |
| op:router dot product | nested | 0.580003 / 1 | 0.506747 / 1 | 0.581145 / 1 |
| op:top-k selection | nested | 0.002515 / 1 | 0.002525 / 1 | 0.002745 / 1 |
| op:alpha / beta / gate | nested | 0.073988 / 1 | 0.076142 / 1 | 0.069670 / 1 |
| detail:read-ahead-wait | nested | 0.002074 / 1 | 0.001613 / 1 | 0.001594 / 1 |
| total:layer | total | 164.339672 / 1 | 161.466989 / 1 | 185.991043 / 1 |

### Layer 2

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006301 / 1 | 0.005520 / 1 | 0.005971 / 1 |
| pre-attention-aggregation | boundary | 0.017061 / 1 | 0.015840 / 1 | 0.017402 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012133 / 1 | 0.011311 / 1 | 0.012543 / 1 |
| Q | nested | 3.404618 / 1 | 3.325461 / 1 | 3.579685 / 1 |
| K | nested | 2.358655 / 1 | 2.418066 / 1 | 3.307527 / 1 |
| V | nested | 2.284847 / 1 | 2.312799 / 1 | 3.158068 / 1 |
| B | nested | 0.023053 / 1 | 0.023694 / 1 | 0.024916 / 1 |
| FA | nested | 0.029936 / 1 | 0.027852 / 1 | 0.037420 / 1 |
| FB | nested | 0.051135 / 1 | 0.051957 / 1 | 0.069139 / 1 |
| G | nested | 2.211820 / 1 | 2.062402 / 1 | 2.641012 / 1 |
| O | nested | 2.139706 / 1 | 2.077720 / 1 | 2.138103 / 1 |
| attention | boundary | 13.568619 / 1 | 13.470025 / 1 | 16.008185 / 1 |
| attention-residual | boundary | 0.003767 / 1 | 0.003837 / 1 | 0.003606 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025427 / 1 | 0.024706 / 1 | 0.025708 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002515 / 1 | 0.001934 / 1 | 0.002384 / 1 |
| router-and-top16 | boundary | 0.593048 / 1 | 0.580344 / 1 | 0.599690 / 1 |
| EDOWN | nested | 0.613105 / 1 | 0.587387 / 1 | 0.620428 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.613666 / 1 | 0.587858 / 1 | 0.620899 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.091701 / 1 | 0.022622 / 1 | 0.025247 / 1 |
| SH1 | nested | 1.045893 / 1 | 1.000979 / 1 | 1.040182 / 1 |
| SH3 | nested | 1.035955 / 1 | 1.011379 / 1 | 1.039782 / 1 |
| SH2 | nested | 1.010797 / 1 | 1.015226 / 1 | 1.018763 / 1 |
| shared-expert-during-read | boundary | 3.102324 / 1 | 3.036931 / 1 | 3.108104 / 1 |
| detail:expert-gate | nested | 49.208447 / 16 | 50.106944 / 16 | 54.344622 / 16 |
| detail:expert-up | nested | 42.783877 / 16 | 42.819495 / 16 | 52.010608 / 16 |
| detail:expert-activation | nested | 0.097577 / 16 | 0.093987 / 16 | 0.121086 / 16 |
| detail:expert-down | nested | 49.128785 / 16 | 49.105977 / 16 | 50.461379 / 16 |
| EUP | nested | 0.590854 / 1 | 0.607424 / 1 | 0.618635 / 1 |
| experts-mix-normalize-up | boundary | 142.244580 / 1 | 143.093617 / 1 | 157.916664 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002675 / 1 | 0.003126 / 1 | 0.003206 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586816 / 1 | 0.587778 / 1 | 0.621060 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.007752 / 1 | 0.941438 / 1 | 1.106266 / 1 |
| worker:read | parallel worker | 0.068653 / 1 | 0.067947 / 1 | 0.070056 / 1 |
| worker:decode | parallel worker | 1050.819944 / 1 | 1077.720266 / 1 | 1066.604316 / 1 |
| worker:crc | parallel worker | 584.185422 / 1 | 599.626989 / 1 | 594.529067 / 1 |
| worker:math | parallel worker | 371.229779 / 1 | 364.409853 / 1 | 370.277250 / 1 |
| op:Q   int8 projection | nested | 16.799063 / 1 | 16.520833 / 1 | 19.292099 / 1 |
| op:X   mxfp4 expert proj | nested | 141.232872 / 1 | 142.140136 / 1 | 156.954740 / 1 |
| op:N   rmsnorm | nested | 0.040244 / 1 | 0.038292 / 1 | 0.032699 / 1 |
| op:L   l2 per-head | nested | 0.005380 / 1 | 0.005561 / 1 | 0.005831 / 1 |
| op:SiTU + sigma | nested | 0.104822 / 1 | 0.101193 / 1 | 0.127177 / 1 |
| op:C   shortconv | nested | 0.062336 / 1 | 0.061575 / 1 | 0.040666 / 1 |
| op:AR  snapshot aggregate | nested | 0.032280 / 1 | 0.030278 / 1 | 0.032410 / 1 |
| op:D   kda delta-rule | nested | 0.151914 / 1 | 0.242973 / 1 | 0.148657 / 1 |
| op:router dot product | nested | 0.589821 / 1 | 0.577549 / 1 | 0.596814 / 1 |
| op:top-k selection | nested | 0.002775 / 1 | 0.002395 / 1 | 0.002415 / 1 |
| op:alpha / beta / gate | nested | 0.075712 / 1 | 0.078266 / 1 | 0.057147 / 1 |
| detail:read-ahead-wait | nested | 0.001452 / 1 | 0.001311 / 1 | 0.001252 / 1 |
| total:layer | total | 161.294034 / 1 | 161.801424 / 1 | 179.458102 / 1 |

### Layer 3

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005611 / 1 | 0.005179 / 1 | 0.006021 / 1 |
| pre-attention-aggregation | boundary | 0.016060 / 1 | 0.016321 / 1 | 0.017653 / 1 |
| snapshot-push | boundary | 0.000110 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011672 / 1 | 0.011932 / 1 | 0.013164 / 1 |
| QA | nested | 0.375601 / 1 | 0.359431 / 1 | 0.405307 / 1 |
| QB | nested | 0.960553 / 1 | 0.952158 / 1 | 1.120783 / 1 |
| KA | nested | 0.147365 / 1 | 0.135022 / 1 | 0.144710 / 1 |
| KB | nested | 0.435153 / 1 | 0.424573 / 1 | 0.401731 / 1 |
| G | nested | 2.972872 / 1 | 2.789489 / 1 | 3.160512 / 1 |
| O | nested | 2.193266 / 1 | 2.056540 / 1 | 3.168097 / 1 |
| attention | boundary | 7.258866 / 1 | 6.826068 / 1 | 8.542724 / 1 |
| attention-residual | boundary | 0.002665 / 1 | 0.002796 / 1 | 0.004398 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024296 / 1 | 0.023494 / 1 | 0.028153 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001743 / 1 | 0.001673 / 1 | 0.002335 / 1 |
| router-and-top16 | boundary | 0.578310 / 1 | 0.526754 / 1 | 0.687263 / 1 |
| EDOWN | nested | 0.635066 / 1 | 0.598658 / 1 | 0.867510 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.635446 / 1 | 0.599069 / 1 | 0.867851 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.084828 / 1 | 0.031519 / 1 | 0.022342 / 1 |
| SH1 | nested | 1.087922 / 1 | 1.024563 / 1 | 1.071050 / 1 |
| SH3 | nested | 1.081018 / 1 | 1.016768 / 1 | 1.093973 / 1 |
| SH2 | nested | 1.075989 / 1 | 1.041656 / 1 | 1.067294 / 1 |
| shared-expert-during-read | boundary | 3.254849 / 1 | 3.092836 / 1 | 3.242105 / 1 |
| detail:expert-gate | nested | 49.674535 / 16 | 50.695216 / 16 | 54.748656 / 16 |
| detail:expert-up | nested | 42.793234 / 16 | 43.759432 / 16 | 53.018584 / 16 |
| detail:expert-activation | nested | 0.095736 / 16 | 0.098273 / 16 | 0.123871 / 16 |
| detail:expert-down | nested | 48.950277 / 16 | 49.690063 / 16 | 52.104287 / 16 |
| EUP | nested | 0.597065 / 1 | 0.618004 / 1 | 0.603497 / 1 |
| experts-mix-normalize-up | boundary | 142.544049 / 1 | 145.235637 / 1 | 160.947664 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003246 / 1 | 0.003226 / 1 | 0.003607 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586496 / 1 | 0.595422 / 1 | 0.630778 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.592397 / 1 | 0.600140 / 1 | 0.637881 / 1 |
| worker:read | parallel worker | 0.070283 / 1 | 0.067198 / 1 | 0.068310 / 1 |
| worker:decode | parallel worker | 1055.362831 / 1 | 1080.901379 / 1 | 1077.935305 / 1 |
| worker:crc | parallel worker | 587.150513 / 1 | 600.807426 / 1 | 599.180258 / 1 |
| worker:math | parallel worker | 367.939125 / 1 | 364.426119 / 1 | 376.445608 / 1 |
| op:Q   int8 projection | nested | 11.560669 / 1 | 11.015641 / 1 | 13.103150 / 1 |
| op:X   mxfp4 expert proj | nested | 141.527649 / 1 | 144.257532 / 1 | 160.011435 / 1 |
| op:N   rmsnorm | nested | 0.033714 / 1 | 0.031519 / 1 | 0.033643 / 1 |
| op:SiTU + sigma | nested | 0.102811 / 1 | 0.105085 / 1 | 0.130144 / 1 |
| op:AR  snapshot aggregate | nested | 0.030017 / 1 | 0.030016 / 1 | 0.033362 / 1 |
| op:SA  softmax attention | nested | 0.019636 / 1 | 0.009247 / 1 | 0.013456 / 1 |
| op:router dot product | nested | 0.575474 / 1 | 0.524108 / 1 | 0.683667 / 1 |
| op:top-k selection | nested | 0.002384 / 1 | 0.002304 / 1 | 0.003206 / 1 |
| detail:read-ahead-wait | nested | 0.001584 / 1 | 0.001563 / 1 | 0.001332 / 1 |
| total:layer | total | 155.016671 / 1 | 156.979138 / 1 | 175.025665 / 1 |

### Layer 4

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005059 / 1 | 0.003847 / 1 | 0.005961 / 1 |
| pre-attention-aggregation | boundary | 0.015188 / 1 | 0.014658 / 1 | 0.016822 / 1 |
| snapshot-push | boundary | 0.000081 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011642 / 1 | 0.011101 / 1 | 0.011862 / 1 |
| Q | nested | 3.578884 / 1 | 3.386314 / 1 | 3.489627 / 1 |
| K | nested | 2.355559 / 1 | 2.379403 / 1 | 2.404901 / 1 |
| V | nested | 2.287542 / 1 | 2.338988 / 1 | 2.325703 / 1 |
| B | nested | 0.022973 / 1 | 0.022061 / 1 | 0.022331 / 1 |
| FA | nested | 0.030077 / 1 | 0.026629 / 1 | 0.030667 / 1 |
| FB | nested | 0.061775 / 1 | 0.067105 / 1 | 0.062838 / 1 |
| G | nested | 2.168539 / 1 | 2.068172 / 1 | 2.145998 / 1 |
| O | nested | 2.158251 / 1 | 2.090134 / 1 | 2.122674 / 1 |
| attention | boundary | 13.844705 / 1 | 13.443456 / 1 | 13.769144 / 1 |
| attention-residual | boundary | 0.003006 / 1 | 0.003907 / 1 | 0.002534 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025788 / 1 | 0.023533 / 1 | 0.025978 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002695 / 1 | 0.002194 / 1 | 0.002925 / 1 |
| router-and-top16 | boundary | 0.601313 / 1 | 0.573080 / 1 | 0.591986 / 1 |
| EDOWN | nested | 0.629506 / 1 | 0.599519 / 1 | 0.622303 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.629937 / 1 | 0.599910 / 1 | 0.622723 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.097633 / 1 | 0.031719 / 1 | 0.024525 / 1 |
| SH1 | nested | 1.062444 / 1 | 1.011709 / 1 | 1.043358 / 1 |
| SH3 | nested | 1.043328 / 1 | 1.013252 / 1 | 1.041966 / 1 |
| SH2 | nested | 1.015797 / 1 | 1.013032 / 1 | 1.004085 / 1 |
| shared-expert-during-read | boundary | 3.133832 / 1 | 3.048022 / 1 | 3.099658 / 1 |
| detail:expert-gate | nested | 49.605357 / 16 | 50.408568 / 16 | 52.833237 / 16 |
| detail:expert-up | nested | 42.674004 / 16 | 42.771123 / 16 | 50.120490 / 16 |
| detail:expert-activation | nested | 0.095496 / 16 | 0.097046 / 16 | 0.118911 / 16 |
| detail:expert-down | nested | 48.960732 / 16 | 49.117428 / 16 | 52.609863 / 16 |
| EUP | nested | 0.599390 / 1 | 0.599079 / 1 | 0.590933 / 1 |
| experts-mix-normalize-up | boundary | 142.379641 / 1 | 143.357159 / 1 | 156.640750 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003587 / 1 | 0.002795 / 1 | 0.002985 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.585043 / 1 | 0.590813 / 1 | 0.593909 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.034662 / 1 | 0.972266 / 1 | 1.076660 / 1 |
| worker:read | parallel worker | 0.067747 / 1 | 0.069727 / 1 | 0.069568 / 1 |
| worker:decode | parallel worker | 1057.779841 / 1 | 1078.478397 / 1 | 1070.397777 / 1 |
| worker:crc | parallel worker | 588.993644 / 1 | 600.876509 / 1 | 597.845798 / 1 |
| worker:math | parallel worker | 365.651178 / 1 | 363.597453 / 1 | 370.139292 / 1 |
| op:Q   int8 projection | nested | 17.012633 / 1 | 16.613636 / 1 | 16.905943 / 1 |
| op:X   mxfp4 expert proj | nested | 141.351171 / 1 | 142.409519 / 1 | 155.700775 / 1 |
| op:N   rmsnorm | nested | 0.037961 / 1 | 0.040634 / 1 | 0.032261 / 1 |
| op:L   l2 per-head | nested | 0.006092 / 1 | 0.005891 / 1 | 0.005761 / 1 |
| op:SiTU + sigma | nested | 0.102571 / 1 | 0.104210 / 1 | 0.125502 / 1 |
| op:C   shortconv | nested | 0.058378 / 1 | 0.055033 / 1 | 0.040406 / 1 |
| op:AR  snapshot aggregate | nested | 0.030407 / 1 | 0.028523 / 1 | 0.031539 / 1 |
| op:D   kda delta-rule | nested | 0.152354 / 1 | 0.112721 / 1 | 0.152996 / 1 |
| op:router dot product | nested | 0.598638 / 1 | 0.570205 / 1 | 0.588759 / 1 |
| op:top-k selection | nested | 0.002235 / 1 | 0.002505 / 1 | 0.002785 / 1 |
| op:alpha / beta / gate | nested | 0.074028 / 1 | 0.074008 / 1 | 0.059371 / 1 |
| detail:read-ahead-wait | nested | 0.001752 / 1 | 0.001775 / 1 | 0.001842 / 1 |
| total:layer | total | 161.791844 / 1 | 162.090454 / 1 | 175.897353 / 1 |

### Layer 5

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006492 / 1 | 0.005009 / 1 | 0.006072 / 1 |
| pre-attention-aggregation | boundary | 0.016721 / 1 | 0.016050 / 1 | 0.017433 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000100 / 1 | 0.000101 / 1 |
| pre-attention-normalization | boundary | 0.011551 / 1 | 0.011251 / 1 | 0.011301 / 1 |
| Q | nested | 3.519232 / 1 | 3.290495 / 1 | 3.668902 / 1 |
| K | nested | 2.400263 / 1 | 2.405432 / 1 | 3.383448 / 1 |
| V | nested | 2.303011 / 1 | 2.323560 / 1 | 2.787165 / 1 |
| B | nested | 0.029335 / 1 | 0.028123 / 1 | 0.031509 / 1 |
| FA | nested | 0.035617 / 1 | 0.033362 / 1 | 0.031890 / 1 |
| FB | nested | 0.051516 / 1 | 0.051155 / 1 | 0.051927 / 1 |
| G | nested | 2.187014 / 1 | 2.091766 / 1 | 2.224814 / 1 |
| O | nested | 2.151197 / 1 | 2.081036 / 1 | 2.124638 / 1 |
| attention | boundary | 13.760378 / 1 | 13.447884 / 1 | 15.363332 / 1 |
| attention-residual | boundary | 0.003997 / 1 | 0.004108 / 1 | 0.003677 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025899 / 1 | 0.024166 / 1 | 0.024136 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002855 / 1 | 0.001733 / 1 | 0.002795 / 1 |
| router-and-top16 | boundary | 0.595342 / 1 | 0.575545 / 1 | 0.594961 / 1 |
| EDOWN | nested | 0.618094 / 1 | 0.598978 / 1 | 0.608326 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.618525 / 1 | 0.599460 / 1 | 0.608857 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085970 / 1 | 0.017072 / 1 | 0.016080 / 1 |
| SH1 | nested | 1.036306 / 1 | 1.019554 / 1 | 1.038970 / 1 |
| SH3 | nested | 1.023942 / 1 | 1.005428 / 1 | 1.027128 / 1 |
| SH2 | nested | 0.997984 / 1 | 1.015326 / 1 | 1.001660 / 1 |
| shared-expert-during-read | boundary | 3.067990 / 1 | 3.050477 / 1 | 3.077647 / 1 |
| detail:expert-gate | nested | 49.394105 / 16 | 50.041529 / 16 | 53.830460 / 16 |
| detail:expert-up | nested | 42.639867 / 16 | 42.775061 / 16 | 50.022147 / 16 |
| detail:expert-activation | nested | 0.095276 / 16 | 0.095865 / 16 | 0.109002 / 16 |
| detail:expert-down | nested | 48.911222 / 16 | 49.051793 / 16 | 49.105538 / 16 |
| EUP | nested | 0.593969 / 1 | 0.595852 / 1 | 0.598728 / 1 |
| experts-mix-normalize-up | boundary | 142.058462 / 1 | 142.909092 / 1 | 154.032631 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003257 / 1 | 0.003085 / 1 | 0.002915 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.593458 / 1 | 0.591224 / 1 | 0.605771 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.020305 / 1 | 0.949643 / 1 | 1.071220 / 1 |
| worker:read | parallel worker | 0.066129 / 1 | 0.067384 / 1 | 0.069662 / 1 |
| worker:decode | parallel worker | 1060.565320 / 1 | 1079.964511 / 1 | 1076.357208 / 1 |
| worker:crc | parallel worker | 589.231569 / 1 | 599.630846 / 1 | 598.588173 / 1 |
| worker:math | parallel worker | 364.361855 / 1 | 362.300762 / 1 | 370.702071 / 1 |
| op:Q   int8 projection | nested | 16.945977 / 1 | 16.538413 / 1 | 18.577704 / 1 |
| op:X   mxfp4 expert proj | nested | 141.057755 / 1 | 141.980938 / 1 | 153.087103 / 1 |
| op:N   rmsnorm | nested | 0.039974 / 1 | 0.039002 / 1 | 0.031389 / 1 |
| op:L   l2 per-head | nested | 0.005611 / 1 | 0.005450 / 1 | 0.005621 / 1 |
| op:SiTU + sigma | nested | 0.102191 / 1 | 0.103021 / 1 | 0.115856 / 1 |
| op:C   shortconv | nested | 0.065482 / 1 | 0.046046 / 1 | 0.040736 / 1 |
| op:AR  snapshot aggregate | nested | 0.032031 / 1 | 0.030728 / 1 | 0.031629 / 1 |
| op:D   kda delta-rule | nested | 0.152394 / 1 | 0.241932 / 1 | 0.151172 / 1 |
| op:router dot product | nested | 0.592867 / 1 | 0.572680 / 1 | 0.591775 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002585 / 1 | 0.002675 / 1 |
| op:alpha / beta / gate | nested | 0.073517 / 1 | 0.073958 / 1 | 0.057447 / 1 |
| detail:read-ahead-wait | nested | 0.001492 / 1 | 0.001513 / 1 | 0.001574 / 1 |
| total:layer | total | 161.281380 / 1 | 161.617961 / 1 | 174.836492 / 1 |

### Layer 6

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005420 / 1 | 0.004558 / 1 | 0.006472 / 1 |
| pre-attention-aggregation | boundary | 0.016531 / 1 | 0.016351 / 1 | 0.017002 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000101 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011421 / 1 | 0.012864 / 1 | 0.012022 / 1 |
| Q | nested | 3.499475 / 1 | 3.367679 / 1 | 3.428353 / 1 |
| K | nested | 2.396145 / 1 | 2.400312 / 1 | 2.440107 / 1 |
| V | nested | 2.314132 / 1 | 2.316716 / 1 | 2.357964 / 1 |
| B | nested | 0.022422 / 1 | 0.021049 / 1 | 0.022192 / 1 |
| FA | nested | 0.029104 / 1 | 0.032801 / 1 | 0.029235 / 1 |
| FB | nested | 0.053620 / 1 | 0.050144 / 1 | 0.050134 / 1 |
| G | nested | 2.195951 / 1 | 2.090635 / 1 | 2.186844 / 1 |
| O | nested | 2.153361 / 1 | 2.086736 / 1 | 2.125890 / 1 |
| attention | boundary | 13.750439 / 1 | 13.535748 / 1 | 13.746622 / 1 |
| attention-residual | boundary | 0.003536 / 1 | 0.003927 / 1 | 0.003376 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025978 / 1 | 0.024115 / 1 | 0.024325 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002024 / 1 | 0.001944 / 1 | 0.002434 / 1 |
| router-and-top16 | boundary | 0.600411 / 1 | 0.576927 / 1 | 0.602235 / 1 |
| EDOWN | nested | 0.625539 / 1 | 0.592196 / 1 | 0.613415 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.625909 / 1 | 0.592576 / 1 | 0.613866 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.089857 / 1 | 0.021600 / 1 | 0.026169 / 1 |
| SH1 | nested | 1.048438 / 1 | 1.013412 / 1 | 1.040162 / 1 |
| SH3 | nested | 1.026557 / 1 | 1.013843 / 1 | 1.031496 / 1 |
| SH2 | nested | 1.012110 / 1 | 1.034883 / 1 | 1.007231 / 1 |
| shared-expert-during-read | boundary | 3.096793 / 1 | 3.072138 / 1 | 3.089039 / 1 |
| detail:expert-gate | nested | 49.472170 / 16 | 49.849625 / 16 | 55.498843 / 16 |
| detail:expert-up | nested | 42.550572 / 16 | 42.697448 / 16 | 52.576309 / 16 |
| detail:expert-activation | nested | 0.096480 / 16 | 0.097773 / 16 | 0.119141 / 16 |
| detail:expert-down | nested | 48.962466 / 16 | 49.115834 / 16 | 48.953079 / 16 |
| EUP | nested | 0.606984 / 1 | 0.602255 / 1 | 0.620810 / 1 |
| experts-mix-normalize-up | boundary | 142.133012 / 1 | 142.723296 / 1 | 158.133028 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.003226 / 1 | 0.003967 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590032 / 1 | 0.584301 / 1 | 0.629957 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.024182 / 1 | 0.955955 / 1 | 1.122677 / 1 |
| worker:read | parallel worker | 0.069157 / 1 | 0.068125 / 1 | 0.067936 / 1 |
| worker:decode | parallel worker | 1060.908071 / 1 | 1077.912291 / 1 | 1075.131582 / 1 |
| worker:crc | parallel worker | 590.118143 / 1 | 599.875352 / 1 | 598.114199 / 1 |
| worker:math | parallel worker | 364.067335 / 1 | 362.688589 / 1 | 368.479565 / 1 |
| op:Q   int8 projection | nested | 16.982416 / 1 | 16.621160 / 1 | 16.952367 / 1 |
| op:X   mxfp4 expert proj | nested | 141.099391 / 1 | 141.778221 / 1 | 157.168294 / 1 |
| op:N   rmsnorm | nested | 0.047988 / 1 | 0.043582 / 1 | 0.032751 / 1 |
| op:L   l2 per-head | nested | 0.005741 / 1 | 0.006352 / 1 | 0.005641 / 1 |
| op:SiTU + sigma | nested | 0.103132 / 1 | 0.105165 / 1 | 0.126385 / 1 |
| op:C   shortconv | nested | 0.060111 / 1 | 0.060974 / 1 | 0.040516 / 1 |
| op:AR  snapshot aggregate | nested | 0.032169 / 1 | 0.030557 / 1 | 0.030938 / 1 |
| op:D   kda delta-rule | nested | 0.155811 / 1 | 0.240850 / 1 | 0.152324 / 1 |
| op:router dot product | nested | 0.597827 / 1 | 0.574273 / 1 | 0.599169 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002364 / 1 | 0.002725 / 1 |
| op:alpha / beta / gate | nested | 0.073157 / 1 | 0.073818 / 1 | 0.058098 / 1 |
| detail:read-ahead-wait | nested | 0.001845 / 1 | 0.001756 / 1 | 0.001765 / 1 |
| total:layer | total | 161.392388 / 1 | 161.548662 / 1 | 177.406621 / 1 |

### Layer 7

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005420 / 1 | 0.008947 / 1 | 0.006072 / 1 |
| pre-attention-aggregation | boundary | 0.016671 / 1 | 0.015940 / 1 | 0.017633 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012253 / 1 | 0.011822 / 1 | 0.012744 / 1 |
| QA | nested | 0.360503 / 1 | 0.348111 / 1 | 0.407360 / 1 |
| QB | nested | 0.986953 / 1 | 0.914128 / 1 | 1.008173 / 1 |
| KA | nested | 0.138038 / 1 | 0.133049 / 1 | 0.149529 / 1 |
| KB | nested | 0.435203 / 1 | 0.403864 / 1 | 0.427889 / 1 |
| G | nested | 2.892422 / 1 | 2.771075 / 1 | 3.236054 / 1 |
| O | nested | 2.189278 / 1 | 2.107025 / 1 | 3.253606 / 1 |
| attention | boundary | 7.113243 / 1 | 6.790741 / 1 | 8.619036 / 1 |
| attention-residual | boundary | 0.002745 / 1 | 0.002745 / 1 | 0.003627 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024706 / 1 | 0.024466 / 1 | 0.028293 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001753 / 1 | 0.001964 / 1 | 0.002685 / 1 |
| router-and-top16 | boundary | 0.584292 / 1 | 0.528988 / 1 | 0.673518 / 1 |
| EDOWN | nested | 0.640536 / 1 | 0.612704 / 1 | 0.833487 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.640977 / 1 | 0.613216 / 1 | 0.833968 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.084017 / 1 | 0.033753 / 1 | 0.020528 / 1 |
| SH1 | nested | 1.081309 / 1 | 1.041766 / 1 | 1.074437 / 1 |
| SH3 | nested | 1.075538 / 1 | 1.047576 / 1 | 1.083173 / 1 |
| SH2 | nested | 1.072503 / 1 | 1.037968 / 1 | 1.065800 / 1 |
| shared-expert-during-read | boundary | 3.238768 / 1 | 3.137179 / 1 | 3.233228 / 1 |
| detail:expert-gate | nested | 49.456769 / 16 | 49.975910 / 16 | 53.654853 / 16 |
| detail:expert-up | nested | 42.770503 / 16 | 42.885809 / 16 | 51.470763 / 16 |
| detail:expert-activation | nested | 0.095119 / 16 | 0.102983 / 16 | 0.118903 / 16 |
| detail:expert-down | nested | 49.078182 / 16 | 49.259915 / 16 | 51.620962 / 16 |
| EUP | nested | 0.593368 / 1 | 0.602996 / 1 | 0.591945 / 1 |
| experts-mix-normalize-up | boundary | 142.434595 / 1 | 143.201929 / 1 | 157.798353 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002745 / 1 | 0.003025 / 1 | 0.003476 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589651 / 1 | 0.591425 / 1 | 0.589461 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.595572 / 1 | 0.597245 / 1 | 0.596874 / 1 |
| worker:read | parallel worker | 0.204957 / 1 | 0.196888 / 1 | 0.198454 / 1 |
| worker:decode | parallel worker | 1062.854302 / 1 | 1078.805957 / 1 | 1078.012427 / 1 |
| worker:crc | parallel worker | 590.331817 / 1 | 599.729335 / 1 | 599.546792 / 1 |
| worker:math | parallel worker | 367.151372 / 1 | 364.550930 / 1 | 375.933568 / 1 |
| op:Q   int8 projection | nested | 11.464149 / 1 | 11.018989 / 1 | 13.130102 / 1 |
| op:X   mxfp4 expert proj | nested | 141.419832 / 1 | 142.243950 / 1 | 156.888002 / 1 |
| op:N   rmsnorm | nested | 0.033112 / 1 | 0.031929 / 1 | 0.032362 / 1 |
| op:SiTU + sigma | nested | 0.101908 / 1 | 0.109905 / 1 | 0.125315 / 1 |
| op:AR  snapshot aggregate | nested | 0.030656 / 1 | 0.030026 / 1 | 0.034394 / 1 |
| op:SA  softmax attention | nested | 0.006362 / 1 | 0.009187 / 1 | 0.013926 / 1 |
| op:router dot product | nested | 0.581526 / 1 | 0.526423 / 1 | 0.670372 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002184 / 1 | 0.002795 / 1 |
| detail:read-ahead-wait | nested | 0.001304 / 1 | 0.001361 / 1 | 0.001223 / 1 |
| total:layer | total | 154.761425 / 1 | 154.975297 / 1 | 171.854353 / 1 |

### Layer 8

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005240 / 1 | 0.003687 / 1 | 0.005721 / 1 |
| pre-attention-aggregation | boundary | 0.015489 / 1 | 0.017543 / 1 | 0.016491 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011432 / 1 | 0.011371 / 1 | 0.012414 / 1 |
| Q | nested | 3.450423 / 1 | 3.359925 / 1 | 3.641070 / 1 |
| K | nested | 2.376478 / 1 | 2.370066 / 1 | 2.952214 / 1 |
| V | nested | 2.279697 / 1 | 2.334369 / 1 | 3.031260 / 1 |
| B | nested | 0.027992 / 1 | 0.026189 / 1 | 0.031419 / 1 |
| FA | nested | 0.028804 / 1 | 0.028524 / 1 | 0.036207 / 1 |
| FB | nested | 0.050965 / 1 | 0.050865 / 1 | 0.072565 / 1 |
| G | nested | 2.167328 / 1 | 2.068944 / 1 | 2.785913 / 1 |
| O | nested | 2.171115 / 1 | 2.086577 / 1 | 2.180732 / 1 |
| attention | boundary | 13.749357 / 1 | 13.385707 / 1 | 15.823711 / 1 |
| attention-residual | boundary | 0.003156 / 1 | 0.003757 / 1 | 0.003025 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025678 / 1 | 0.023884 / 1 | 0.024616 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002625 / 1 | 0.002054 / 1 | 0.002425 / 1 |
| router-and-top16 | boundary | 0.601473 / 1 | 0.573621 / 1 | 0.602255 / 1 |
| EDOWN | nested | 0.625618 / 1 | 0.604378 / 1 | 0.625177 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.626179 / 1 | 0.604879 / 1 | 0.625738 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.085800 / 1 | 0.021279 / 1 | 0.012664 / 1 |
| SH1 | nested | 1.069968 / 1 | 1.010417 / 1 | 1.056433 / 1 |
| SH3 | nested | 1.031006 / 1 | 1.013974 / 1 | 1.053317 / 1 |
| SH2 | nested | 1.019403 / 1 | 1.014664 / 1 | 1.023852 / 1 |
| shared-expert-during-read | boundary | 3.130476 / 1 | 3.049354 / 1 | 3.143380 / 1 |
| detail:expert-gate | nested | 49.332778 / 16 | 50.032618 / 16 | 53.168143 / 16 |
| detail:expert-up | nested | 42.674310 / 16 | 42.779333 / 16 | 50.680636 / 16 |
| detail:expert-activation | nested | 0.095248 / 16 | 0.100276 / 16 | 0.120563 / 16 |
| detail:expert-down | nested | 48.869571 / 16 | 49.062433 / 16 | 52.926382 / 16 |
| EUP | nested | 0.587427 / 1 | 0.602335 / 1 | 0.598778 / 1 |
| experts-mix-normalize-up | boundary | 141.999942 / 1 | 142.942114 / 1 | 157.860939 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003356 / 1 | 0.002895 / 1 | 0.002985 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584091 / 1 | 0.585133 / 1 | 0.605741 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.028872 / 1 | 0.951086 / 1 | 1.084195 / 1 |
| worker:read | parallel worker | 0.067218 / 1 | 0.065416 / 1 | 0.068412 / 1 |
| worker:decode | parallel worker | 1059.552709 / 1 | 1078.039326 / 1 | 1074.825028 / 1 |
| worker:crc | parallel worker | 591.019575 / 1 | 600.294532 / 1 | 599.832562 / 1 |
| worker:math | parallel worker | 365.810219 / 1 | 362.582693 / 1 | 369.103641 / 1 |
| op:Q   int8 projection | nested | 16.884703 / 1 | 16.569694 / 1 | 19.087076 / 1 |
| op:X   mxfp4 expert proj | nested | 140.992601 / 1 | 141.994423 / 1 | 156.920892 / 1 |
| op:N   rmsnorm | nested | 0.047419 / 1 | 0.038401 / 1 | 0.032902 / 1 |
| op:L   l2 per-head | nested | 0.005320 / 1 | 0.005590 / 1 | 0.005791 / 1 |
| op:SiTU + sigma | nested | 0.102201 / 1 | 0.107319 / 1 | 0.126978 / 1 |
| op:C   shortconv | nested | 0.059101 / 1 | 0.070512 / 1 | 0.040575 / 1 |
| op:AR  snapshot aggregate | nested | 0.030718 / 1 | 0.031940 / 1 | 0.030938 / 1 |
| op:D   kda delta-rule | nested | 0.151052 / 1 | 0.112641 / 1 | 0.153637 / 1 |
| op:router dot product | nested | 0.598989 / 1 | 0.570966 / 1 | 0.599219 / 1 |
| op:top-k selection | nested | 0.002044 / 1 | 0.002344 / 1 | 0.002645 / 1 |
| op:alpha / beta / gate | nested | 0.074449 / 1 | 0.074660 / 1 | 0.057677 / 1 |
| detail:read-ahead-wait | nested | 0.001712 / 1 | 0.001581 / 1 | 0.001808 / 1 |
| total:layer | total | 161.293693 / 1 | 161.597142 / 1 | 179.224968 / 1 |

### Layer 9

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005570 / 1 | 0.006012 / 1 | 0.005200 / 1 |
| pre-attention-aggregation | boundary | 0.016751 / 1 | 0.016361 / 1 | 0.017242 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000121 / 1 | 0.000121 / 1 |
| pre-attention-normalization | boundary | 0.010940 / 1 | 0.011351 / 1 | 0.011442 / 1 |
| Q | nested | 3.474178 / 1 | 3.284974 / 1 | 3.527718 / 1 |
| K | nested | 2.392969 / 1 | 2.404500 / 1 | 3.066988 / 1 |
| V | nested | 2.312028 / 1 | 2.333769 / 1 | 2.915315 / 1 |
| B | nested | 0.022191 / 1 | 0.022522 / 1 | 0.030396 / 1 |
| FA | nested | 0.032561 / 1 | 0.035536 / 1 | 0.042700 / 1 |
| FB | nested | 0.050384 / 1 | 0.049312 / 1 | 0.103162 / 1 |
| G | nested | 2.191393 / 1 | 2.067010 / 1 | 2.826378 / 1 |
| O | nested | 2.127082 / 1 | 2.072761 / 1 | 2.136991 / 1 |
| attention | boundary | 13.689456 / 1 | 13.430722 / 1 | 15.740205 / 1 |
| attention-residual | boundary | 0.003807 / 1 | 0.003517 / 1 | 0.003777 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025297 / 1 | 0.024055 / 1 | 0.024827 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002555 / 1 | 0.001763 / 1 | 0.002365 / 1 |
| router-and-top16 | boundary | 0.593989 / 1 | 0.568272 / 1 | 0.600632 / 1 |
| EDOWN | nested | 0.612724 / 1 | 0.588329 / 1 | 0.622763 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.613215 / 1 | 0.588810 / 1 | 0.623334 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.086933 / 1 | 0.021530 / 1 | 0.022903 / 1 |
| SH1 | nested | 1.045272 / 1 | 0.999176 / 1 | 1.026156 / 1 |
| SH3 | nested | 1.025385 / 1 | 1.000488 / 1 | 1.020455 / 1 |
| SH2 | nested | 1.004947 / 1 | 0.991050 / 1 | 1.012260 / 1 |
| shared-expert-during-read | boundary | 3.085613 / 1 | 3.000994 / 1 | 3.068781 / 1 |
| detail:expert-gate | nested | 49.386709 / 16 | 48.844407 / 16 | 52.401482 / 16 |
| detail:expert-up | nested | 42.732752 / 16 | 42.893303 / 16 | 48.905221 / 16 |
| detail:expert-activation | nested | 0.095717 / 16 | 0.100610 / 16 | 0.115665 / 16 |
| detail:expert-down | nested | 48.959831 / 16 | 49.087861 / 16 | 50.262546 / 16 |
| EUP | nested | 0.593608 / 1 | 0.601043 / 1 | 0.600782 / 1 |
| experts-mix-normalize-up | boundary | 142.210626 / 1 | 141.885471 / 1 | 152.630101 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003075 / 1 | 0.002775 / 1 | 0.002945 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.580303 / 1 | 0.584181 / 1 | 0.584852 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.984529 / 1 | 0.932672 / 1 | 0.995639 / 1 |
| worker:read | parallel worker | 0.201534 / 1 | 0.213261 / 1 | 0.194297 / 1 |
| worker:decode | parallel worker | 1062.954051 / 1 | 1076.632975 / 1 | 1077.308409 / 1 |
| worker:crc | parallel worker | 592.004042 / 1 | 599.057594 / 1 | 600.935870 / 1 |
| worker:math | parallel worker | 366.014336 / 1 | 364.006767 / 1 | 371.786540 / 1 |
| op:Q   int8 projection | nested | 16.883281 / 1 | 16.449008 / 1 | 18.930342 / 1 |
| op:X   mxfp4 expert proj | nested | 141.195290 / 1 | 140.946026 / 1 | 151.708990 / 1 |
| op:N   rmsnorm | nested | 0.038323 / 1 | 0.042148 / 1 | 0.032469 / 1 |
| op:L   l2 per-head | nested | 0.005480 / 1 | 0.005470 / 1 | 0.005891 / 1 |
| op:SiTU + sigma | nested | 0.102551 / 1 | 0.107992 / 1 | 0.122139 / 1 |
| op:C   shortconv | nested | 0.076543 / 1 | 0.047950 / 1 | 0.039584 / 1 |
| op:AR  snapshot aggregate | nested | 0.031870 / 1 | 0.030778 / 1 | 0.031179 / 1 |
| op:D   kda delta-rule | nested | 0.157264 / 1 | 0.241501 / 1 | 0.172201 / 1 |
| op:router dot product | nested | 0.591214 / 1 | 0.565426 / 1 | 0.597406 / 1 |
| op:top-k selection | nested | 0.002405 / 1 | 0.002515 / 1 | 0.002494 / 1 |
| op:alpha / beta / gate | nested | 0.072195 / 1 | 0.074179 / 1 | 0.058219 / 1 |
| detail:read-ahead-wait | nested | 0.001583 / 1 | 0.001673 / 1 | 0.001844 / 1 |
| total:layer | total | 161.337085 / 1 | 160.498761 / 1 | 173.753720 / 1 |

### Layer 10

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004759 / 1 | 0.005821 / 1 | 0.005460 / 1 |
| pre-attention-aggregation | boundary | 0.016341 / 1 | 0.016180 / 1 | 0.016220 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011201 / 1 | 0.011431 / 1 | 0.011201 / 1 |
| Q | nested | 3.427732 / 1 | 3.267261 / 1 | 3.377238 / 1 |
| K | nested | 2.483368 / 1 | 2.379193 / 1 | 2.443423 / 1 |
| V | nested | 2.333457 / 1 | 2.334389 / 1 | 2.329931 / 1 |
| B | nested | 0.028032 / 1 | 0.027051 / 1 | 0.028844 / 1 |
| FA | nested | 0.027752 / 1 | 0.028443 / 1 | 0.028774 / 1 |
| FB | nested | 0.051797 / 1 | 0.050464 / 1 | 0.051145 / 1 |
| G | nested | 2.188877 / 1 | 2.060678 / 1 | 2.141830 / 1 |
| O | nested | 2.122583 / 1 | 2.063353 / 1 | 2.133905 / 1 |
| attention | boundary | 13.755408 / 1 | 13.394735 / 1 | 13.696779 / 1 |
| attention-residual | boundary | 0.003968 / 1 | 0.003717 / 1 | 0.003496 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025267 / 1 | 0.024485 / 1 | 0.024275 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002184 / 1 | 0.001723 / 1 | 0.002184 / 1 |
| router-and-top16 | boundary | 0.596795 / 1 | 0.570736 / 1 | 0.592897 / 1 |
| EDOWN | nested | 0.612543 / 1 | 0.589451 / 1 | 0.606563 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.613054 / 1 | 0.590062 / 1 | 0.607094 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.077765 / 1 | 0.026089 / 1 | 0.015679 / 1 |
| SH1 | nested | 1.049179 / 1 | 0.997293 / 1 | 1.034913 / 1 |
| SH3 | nested | 1.027389 / 1 | 1.005618 / 1 | 1.025265 / 1 |
| SH2 | nested | 1.016068 / 1 | 1.002432 / 1 | 1.017860 / 1 |
| shared-expert-during-read | boundary | 3.103185 / 1 | 3.015612 / 1 | 3.088087 / 1 |
| detail:expert-gate | nested | 49.483230 / 16 | 49.138813 / 16 | 52.879040 / 16 |
| detail:expert-up | nested | 43.781241 / 16 | 43.865580 / 16 | 49.615849 / 16 |
| detail:expert-activation | nested | 0.095009 / 16 | 0.096880 / 16 | 0.120194 / 16 |
| detail:expert-down | nested | 50.291340 / 16 | 50.430210 / 16 | 52.728053 / 16 |
| EUP | nested | 0.596063 / 1 | 0.601413 / 1 | 0.604078 / 1 |
| experts-mix-normalize-up | boundary | 144.679197 / 1 | 144.499594 / 1 | 156.309131 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002825 / 1 | 0.003386 / 1 | 0.003116 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.588008 / 1 | 0.591886 / 1 | 0.597285 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.019073 / 1 | 0.960253 / 1 | 0.997673 / 1 |
| worker:read | parallel worker | 0.072089 / 1 | 0.067287 / 1 | 0.069262 / 1 |
| worker:decode | parallel worker | 1060.204646 / 1 | 1074.463173 / 1 | 1072.028733 / 1 |
| worker:crc | parallel worker | 596.969065 / 1 | 602.076606 / 1 | 603.713509 / 1 |
| worker:math | parallel worker | 363.644205 / 1 | 361.669041 / 1 | 367.086279 / 1 |
| op:Q   int8 projection | nested | 16.963267 / 1 | 16.405646 / 1 | 16.822215 / 1 |
| op:X   mxfp4 expert proj | nested | 143.672447 / 1 | 143.554344 / 1 | 155.369601 / 1 |
| op:N   rmsnorm | nested | 0.045715 / 1 | 0.044051 / 1 | 0.030748 / 1 |
| op:L   l2 per-head | nested | 0.005870 / 1 | 0.005270 / 1 | 0.005570 / 1 |
| op:SiTU + sigma | nested | 0.102100 / 1 | 0.104135 / 1 | 0.126736 / 1 |
| op:C   shortconv | nested | 0.063779 / 1 | 0.064440 / 1 | 0.038632 / 1 |
| op:AR  snapshot aggregate | nested | 0.032059 / 1 | 0.030777 / 1 | 0.030517 / 1 |
| op:D   kda delta-rule | nested | 0.154628 / 1 | 0.243806 / 1 | 0.196357 / 1 |
| op:router dot product | nested | 0.594119 / 1 | 0.568101 / 1 | 0.589871 / 1 |
| op:top-k selection | nested | 0.002094 / 1 | 0.002294 / 1 | 0.002554 / 1 |
| op:alpha / beta / gate | nested | 0.072025 / 1 | 0.073607 / 1 | 0.058329 / 1 |
| detail:read-ahead-wait | nested | 0.001826 / 1 | 0.001715 / 1 | 0.001665 / 1 |
| total:layer | total | 163.915590 / 1 | 163.127902 / 1 | 175.378083 / 1 |

### Layer 11

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006162 / 1 | 0.005370 / 1 | 0.005129 / 1 |
| pre-attention-aggregation | boundary | 0.017022 / 1 | 0.017743 / 1 | 0.016892 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012333 / 1 | 0.012743 / 1 | 0.011461 / 1 |
| QA | nested | 0.371043 / 1 | 0.371684 / 1 | 0.374149 / 1 |
| QB | nested | 1.002653 / 1 | 1.146842 / 1 | 0.963179 / 1 |
| KA | nested | 0.140352 / 1 | 0.135804 / 1 | 0.137006 / 1 |
| KB | nested | 0.410747 / 1 | 0.334665 / 1 | 0.429302 / 1 |
| G | nested | 2.778138 / 1 | 1.988614 / 1 | 2.868286 / 1 |
| O | nested | 2.183588 / 1 | 2.009934 / 1 | 2.101394 / 1 |
| attention | boundary | 6.995794 / 1 | 6.104370 / 1 | 6.980827 / 1 |
| attention-residual | boundary | 0.002745 / 1 | 0.002805 / 1 | 0.002906 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024927 / 1 | 0.023895 / 1 | 0.024035 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001964 / 1 | 0.001733 / 1 | 0.002033 / 1 |
| router-and-top16 | boundary | 0.566659 / 1 | 0.518779 / 1 | 0.537874 / 1 |
| EDOWN | nested | 0.624416 / 1 | 0.605531 / 1 | 0.606723 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.624897 / 1 | 0.606032 / 1 | 0.607214 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.064841 / 1 | 0.016311 / 1 | 0.014978 / 1 |
| SH1 | nested | 1.062304 / 1 | 1.027279 / 1 | 1.051303 / 1 |
| SH3 | nested | 1.067694 / 1 | 1.011228 / 1 | 1.034392 / 1 |
| SH2 | nested | 1.067604 / 1 | 1.016779 / 1 | 1.034963 / 1 |
| shared-expert-during-read | boundary | 3.208292 / 1 | 3.065786 / 1 | 3.130897 / 1 |
| detail:expert-gate | nested | 48.992887 / 16 | 50.332704 / 16 | 51.534662 / 16 |
| detail:expert-up | nested | 42.665207 / 16 | 43.810317 / 16 | 45.947760 / 16 |
| detail:expert-activation | nested | 0.095619 / 16 | 0.102463 / 16 | 0.107878 / 16 |
| detail:expert-down | nested | 48.923886 / 16 | 49.267688 / 16 | 49.146431 / 16 |
| EUP | nested | 0.586605 / 1 | 0.606392 / 1 | 0.607765 / 1 |
| experts-mix-normalize-up | boundary | 141.694462 / 1 | 144.473865 / 1 | 147.697770 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003346 / 1 | 0.003075 / 1 | 0.003386 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587156 / 1 | 0.595352 / 1 | 0.594040 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.593749 / 1 | 0.599940 / 1 | 0.599239 / 1 |
| worker:read | parallel worker | 0.067257 / 1 | 0.066642 / 1 | 0.067563 / 1 |
| worker:decode | parallel worker | 1061.253780 / 1 | 1078.480182 / 1 | 1071.621562 / 1 |
| worker:crc | parallel worker | 592.912098 / 1 | 602.380270 / 1 | 597.975915 / 1 |
| worker:math | parallel worker | 364.379377 / 1 | 363.756345 / 1 | 364.539908 / 1 |
| op:Q   int8 projection | nested | 11.293943 / 1 | 10.253561 / 1 | 11.207150 / 1 |
| op:X   mxfp4 expert proj | nested | 140.699967 / 1 | 143.536803 / 1 | 146.761119 / 1 |
| op:N   rmsnorm | nested | 0.032361 / 1 | 0.032490 / 1 | 0.031017 / 1 |
| op:SiTU + sigma | nested | 0.102622 / 1 | 0.109926 / 1 | 0.114702 / 1 |
| op:AR  snapshot aggregate | nested | 0.031719 / 1 | 0.031750 / 1 | 0.030678 / 1 |
| op:SA  softmax attention | nested | 0.006923 / 1 | 0.008666 / 1 | 0.012493 / 1 |
| op:router dot product | nested | 0.563903 / 1 | 0.516254 / 1 | 0.535130 / 1 |
| op:top-k selection | nested | 0.002324 / 1 | 0.002215 / 1 | 0.002414 / 1 |
| detail:read-ahead-wait | nested | 0.001712 / 1 | 0.001576 / 1 | 0.001512 / 1 |
| total:layer | total | 153.821901 / 1 | 155.457056 / 1 | 159.639050 / 1 |

### Layer 12

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004900 / 1 | 0.003546 / 1 | 0.005280 / 1 |
| pre-attention-aggregation | boundary | 0.015549 / 1 | 0.015539 / 1 | 0.015158 / 1 |
| snapshot-push | boundary | 0.001683 / 1 | 0.001233 / 1 | 0.001232 / 1 |
| pre-attention-normalization | boundary | 0.010479 / 1 | 0.011492 / 1 | 0.011231 / 1 |
| Q | nested | 3.351679 / 1 | 3.315221 / 1 | 3.374081 / 1 |
| K | nested | 2.464242 / 1 | 2.347774 / 1 | 2.437031 / 1 |
| V | nested | 2.376227 / 1 | 2.290708 / 1 | 2.333097 / 1 |
| B | nested | 0.027662 / 1 | 0.027080 / 1 | 0.028383 / 1 |
| FA | nested | 0.032932 / 1 | 0.034234 / 1 | 0.033743 / 1 |
| FB | nested | 0.054502 / 1 | 0.050525 / 1 | 0.050284 / 1 |
| G | nested | 2.156688 / 1 | 2.043727 / 1 | 2.120610 / 1 |
| O | nested | 2.165774 / 1 | 2.071909 / 1 | 2.150606 / 1 |
| attention | boundary | 13.828875 / 1 | 13.263009 / 1 | 13.595771 / 1 |
| attention-residual | boundary | 0.002124 / 1 | 0.002154 / 1 | 0.002224 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025378 / 1 | 0.032441 / 1 | 0.024706 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002685 / 1 | 0.001664 / 1 | 0.002114 / 1 |
| router-and-top16 | boundary | 0.603888 / 1 | 0.557030 / 1 | 0.601543 / 1 |
| EDOWN | nested | 0.623194 / 1 | 0.588890 / 1 | 0.620138 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.624015 / 1 | 0.589431 / 1 | 0.620709 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.078476 / 1 | 0.030267 / 1 | 0.026569 / 1 |
| SH1 | nested | 1.069257 / 1 | 0.999857 / 1 | 1.057184 / 1 |
| SH3 | nested | 1.043479 / 1 | 0.999267 / 1 | 1.039051 / 1 |
| SH2 | nested | 1.025916 / 1 | 0.999096 / 1 | 1.042848 / 1 |
| shared-expert-during-read | boundary | 3.149131 / 1 | 3.008579 / 1 | 3.149071 / 1 |
| detail:expert-gate | nested | 48.300330 / 16 | 49.036235 / 16 | 53.595212 / 16 |
| detail:expert-up | nested | 42.166213 / 16 | 42.323859 / 16 | 50.898583 / 16 |
| detail:expert-activation | nested | 0.095407 / 16 | 0.096221 / 16 | 0.120682 / 16 |
| detail:expert-down | nested | 48.310279 / 16 | 48.600788 / 16 | 49.073216 / 16 |
| EUP | nested | 0.597216 / 1 | 0.594951 / 1 | 0.615219 / 1 |
| experts-mix-normalize-up | boundary | 139.899941 / 1 | 141.017781 / 1 | 154.678046 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002926 / 1 | 0.003036 / 1 | 0.003026 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.578520 / 1 | 0.586686 / 1 | 0.585293 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.018242 / 1 | 0.948641 / 1 | 0.988726 / 1 |
| worker:read | parallel worker | 0.197746 / 1 | 0.202452 / 1 | 0.194964 / 1 |
| worker:decode | parallel worker | 1041.183950 / 1 | 1057.989907 / 1 | 1058.561401 / 1 |
| worker:crc | parallel worker | 591.697395 / 1 | 599.555437 / 1 | 602.212962 / 1 |
| worker:math | parallel worker | 363.749568 / 1 | 362.140630 / 1 | 371.809723 / 1 |
| op:Q   int8 projection | nested | 16.987254 / 1 | 16.361727 / 1 | 16.900714 / 1 |
| op:X   mxfp4 expert proj | nested | 138.896386 / 1 | 140.083327 / 1 | 153.717082 / 1 |
| op:N   rmsnorm | nested | 0.041728 / 1 | 0.050314 / 1 | 0.031207 / 1 |
| op:L   l2 per-head | nested | 0.005480 / 1 | 0.005270 / 1 | 0.006061 / 1 |
| op:SiTU + sigma | nested | 0.102703 / 1 | 0.103200 / 1 | 0.127205 / 1 |
| op:C   shortconv | nested | 0.065763 / 1 | 0.053479 / 1 | 0.038902 / 1 |
| op:AR  snapshot aggregate | nested | 0.031048 / 1 | 0.033733 / 1 | 0.030377 / 1 |
| op:D   kda delta-rule | nested | 0.147766 / 1 | 0.143378 / 1 | 0.122980 / 1 |
| op:router dot product | nested | 0.601142 / 1 | 0.554516 / 1 | 0.598708 / 1 |
| op:top-k selection | nested | 0.002315 / 1 | 0.002174 / 1 | 0.002304 / 1 |
| op:alpha / beta / gate | nested | 0.074319 / 1 | 0.074249 / 1 | 0.057498 / 1 |
| detail:read-ahead-wait | nested | 0.001634 / 1 | 0.001618 / 1 | 0.001523 / 1 |
| total:layer | total | 159.275915 / 1 | 159.490548 / 1 | 173.730226 / 1 |

### Layer 13

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006742 / 1 | 0.004548 / 1 | 0.005820 / 1 |
| pre-attention-aggregation | boundary | 0.016861 / 1 | 0.016791 / 1 | 0.017232 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011391 / 1 | 0.011732 / 1 | 0.011582 / 1 |
| Q | nested | 3.398126 / 1 | 3.218090 / 1 | 3.547145 / 1 |
| K | nested | 2.459753 / 1 | 2.373372 / 1 | 2.839944 / 1 |
| V | nested | 2.393370 / 1 | 2.325113 / 1 | 2.963805 / 1 |
| B | nested | 0.023143 / 1 | 0.020889 / 1 | 0.028684 / 1 |
| FA | nested | 0.029275 / 1 | 0.028333 / 1 | 0.036858 / 1 |
| FB | nested | 0.052528 / 1 | 0.050324 / 1 | 0.056396 / 1 |
| G | nested | 2.163821 / 1 | 2.032686 / 1 | 2.547617 / 1 |
| O | nested | 2.129026 / 1 | 2.066349 / 1 | 2.111002 / 1 |
| attention | boundary | 13.753294 / 1 | 13.268489 / 1 | 15.241423 / 1 |
| attention-residual | boundary | 0.004077 / 1 | 0.003667 / 1 | 0.003456 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024896 / 1 | 0.024085 / 1 | 0.025117 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.003416 / 1 | 0.001963 / 1 | 0.002535 / 1 |
| router-and-top16 | boundary | 0.602876 / 1 | 0.574733 / 1 | 0.584261 / 1 |
| EDOWN | nested | 0.618836 / 1 | 0.586295 / 1 | 0.588820 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.619387 / 1 | 0.586876 / 1 | 0.589371 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.086041 / 1 | 0.023374 / 1 | 0.015679 / 1 |
| SH1 | nested | 1.052746 / 1 | 1.412998 / 1 | 1.458664 / 1 |
| SH3 | nested | 1.039781 / 1 | 1.419972 / 1 | 1.540817 / 1 |
| SH2 | nested | 1.026768 / 1 | 1.384004 / 1 | 1.605618 / 1 |
| shared-expert-during-read | boundary | 3.129294 / 1 | 4.231271 / 1 | 4.621080 / 1 |
| detail:expert-gate | nested | 49.137897 / 16 | 49.669990 / 16 | 52.649234 / 16 |
| detail:expert-up | nested | 42.606477 / 16 | 42.772079 / 16 | 48.961899 / 16 |
| detail:expert-activation | nested | 0.095116 / 16 | 0.096941 / 16 | 0.114843 / 16 |
| detail:expert-down | nested | 48.899491 / 16 | 49.049680 / 16 | 51.661732 / 16 |
| EUP | nested | 0.594100 / 1 | 0.611913 / 1 | 0.611371 / 1 |
| experts-mix-normalize-up | boundary | 141.772128 / 1 | 142.560362 / 1 | 158.846621 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003186 / 1 | 0.002655 / 1 | 0.002946 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.585313 / 1 | 0.592316 / 1 | 0.589921 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.984068 / 1 | 0.952889 / 1 | 0.982926 / 1 |
| worker:read | parallel worker | 0.067891 / 1 | 0.066683 / 1 | 0.065250 / 1 |
| worker:decode | parallel worker | 1060.980981 / 1 | 1072.969849 / 1 | 1071.641156 / 1 |
| worker:crc | parallel worker | 592.829423 / 1 | 599.813602 / 1 | 598.999098 / 1 |
| worker:math | parallel worker | 365.199503 / 1 | 364.038875 / 1 | 368.579529 / 1 |
| op:Q   int8 projection | nested | 16.979740 / 1 | 17.528864 / 1 | 19.934628 / 1 |
| op:X   mxfp4 expert proj | nested | 140.761852 / 1 | 141.611340 / 1 | 153.420078 / 1 |
| op:N   rmsnorm | nested | 0.040797 / 1 | 0.040093 / 1 | 0.031259 / 1 |
| op:L   l2 per-head | nested | 0.004949 / 1 | 0.006151 / 1 | 0.005360 / 1 |
| op:SiTU + sigma | nested | 0.101790 / 1 | 0.107641 / 1 | 0.126063 / 1 |
| op:C   shortconv | nested | 0.065081 / 1 | 0.052919 / 1 | 0.037009 / 1 |
| op:AR  snapshot aggregate | nested | 0.031519 / 1 | 0.031308 / 1 | 0.032101 / 1 |
| op:D   kda delta-rule | nested | 0.183793 / 1 | 0.244798 / 1 | 0.219479 / 1 |
| op:router dot product | nested | 0.600131 / 1 | 0.571938 / 1 | 0.581376 / 1 |
| op:top-k selection | nested | 0.002294 / 1 | 0.002244 / 1 | 0.002475 / 1 |
| op:alpha / beta / gate | nested | 0.072115 / 1 | 0.081462 / 1 | 0.062126 / 1 |
| detail:read-ahead-wait | nested | 0.001722 / 1 | 0.001513 / 1 | 4.492029 / 1 |
| total:layer | total | 161.022768 / 1 | 162.268316 / 1 | 180.955559 / 1 |

### Layer 14

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005240 / 1 | 0.004849 / 1 | 0.005009 / 1 |
| pre-attention-aggregation | boundary | 0.017332 / 1 | 0.017663 / 1 | 0.016601 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000101 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.010891 / 1 | 0.011291 / 1 | 0.011511 / 1 |
| Q | nested | 3.391564 / 1 | 3.226576 / 1 | 3.562223 / 1 |
| K | nested | 2.441820 / 1 | 2.377119 / 1 | 2.971058 / 1 |
| V | nested | 2.339449 / 1 | 2.342484 / 1 | 2.521079 / 1 |
| B | nested | 0.024135 / 1 | 0.027832 / 1 | 0.023444 / 1 |
| FA | nested | 0.030357 / 1 | 0.029224 / 1 | 0.035176 / 1 |
| FB | nested | 0.054662 / 1 | 0.052248 / 1 | 0.051126 / 1 |
| G | nested | 2.168219 / 1 | 2.039629 / 1 | 2.130900 / 1 |
| O | nested | 2.128374 / 1 | 2.066259 / 1 | 2.091376 / 1 |
| attention | boundary | 13.695206 / 1 | 13.332739 / 1 | 14.504107 / 1 |
| attention-residual | boundary | 0.004227 / 1 | 0.003707 / 1 | 0.003888 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025848 / 1 | 0.023815 / 1 | 0.023884 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002464 / 1 | 0.001794 / 1 | 0.002194 / 1 |
| router-and-top16 | boundary | 0.600081 / 1 | 0.568712 / 1 | 0.584602 / 1 |
| EDOWN | nested | 0.609218 / 1 | 0.591114 / 1 | 0.594931 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609919 / 1 | 0.591675 / 1 | 0.595522 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.082464 / 1 | 0.027762 / 1 | 0.015459 / 1 |
| SH1 | nested | 1.043148 / 1 | 1.426233 / 1 | 1.396929 / 1 |
| SH3 | nested | 1.029372 / 1 | 1.451280 / 1 | 1.401297 / 1 |
| SH2 | nested | 1.032989 / 1 | 1.393212 / 1 | 1.415664 / 1 |
| shared-expert-during-read | boundary | 3.115498 / 1 | 4.284852 / 1 | 4.228537 / 1 |
| detail:expert-gate | nested | 49.296111 / 16 | 49.570084 / 16 | 55.226288 / 16 |
| detail:expert-up | nested | 42.644857 / 16 | 42.699672 / 16 | 53.247775 / 16 |
| detail:expert-activation | nested | 0.096038 / 16 | 0.098658 / 16 | 0.125837 / 16 |
| detail:expert-down | nested | 48.913525 / 16 | 48.992203 / 16 | 51.153291 / 16 |
| EUP | nested | 0.616391 / 1 | 0.607805 / 1 | 0.605531 / 1 |
| experts-mix-normalize-up | boundary | 142.002088 / 1 | 142.337245 / 1 | 160.729496 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003266 / 1 | 0.002695 / 1 | 0.002945 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.583139 / 1 | 0.590212 / 1 | 0.601754 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.986763 / 1 | 0.958050 / 1 | 0.998615 / 1 |
| worker:read | parallel worker | 0.215255 / 1 | 0.201964 / 1 | 0.191314 / 1 |
| worker:decode | parallel worker | 1064.936535 / 1 | 1073.370985 / 1 | 1079.495591 / 1 |
| worker:crc | parallel worker | 593.239667 / 1 | 600.196090 / 1 | 603.895921 / 1 |
| worker:math | parallel worker | 364.352961 / 1 | 363.386860 / 1 | 375.161153 / 1 |
| op:Q   int8 projection | nested | 16.908055 / 1 | 17.629463 / 1 | 18.799168 / 1 |
| op:X   mxfp4 expert proj | nested | 140.978839 / 1 | 141.388174 / 1 | 159.789411 / 1 |
| op:N   rmsnorm | nested | 0.046226 / 1 | 0.039635 / 1 | 0.031449 / 1 |
| op:L   l2 per-head | nested | 0.005911 / 1 | 0.005701 / 1 | 0.005199 / 1 |
| op:SiTU + sigma | nested | 0.102954 / 1 | 0.109085 / 1 | 0.136032 / 1 |
| op:C   shortconv | nested | 0.057708 / 1 | 0.059802 / 1 | 0.034585 / 1 |
| op:AR  snapshot aggregate | nested | 0.032781 / 1 | 0.031999 / 1 | 0.031048 / 1 |
| op:D   kda delta-rule | nested | 0.193752 / 1 | 0.245999 / 1 | 0.227064 / 1 |
| op:router dot product | nested | 0.597356 / 1 | 0.566308 / 1 | 0.582127 / 1 |
| op:top-k selection | nested | 0.002304 / 1 | 0.002084 / 1 | 0.002114 / 1 |
| op:alpha / beta / gate | nested | 0.072255 / 1 | 0.075120 / 1 | 0.058420 / 1 |
| detail:read-ahead-wait | nested | 0.001742 / 1 | 0.001716 / 1 | 0.001782 / 1 |
| total:layer | total | 161.167027 / 1 | 162.172467 / 1 | 181.728222 / 1 |

### Layer 15

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005270 / 1 | 0.004960 / 1 | 0.004950 / 1 |
| pre-attention-aggregation | boundary | 0.016180 / 1 | 0.016671 / 1 | 0.016591 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011211 / 1 | 0.011432 / 1 | 0.011943 / 1 |
| QA | nested | 0.359952 / 1 | 0.367025 / 1 | 0.385330 / 1 |
| QB | nested | 0.953801 / 1 | 0.937991 / 1 | 0.950234 / 1 |
| KA | nested | 0.139611 / 1 | 0.137787 / 1 | 0.145401 / 1 |
| KB | nested | 0.417590 / 1 | 0.426115 / 1 | 0.420075 / 1 |
| G | nested | 2.886520 / 1 | 2.723606 / 1 | 2.947384 / 1 |
| O | nested | 2.172828 / 1 | 2.026734 / 1 | 3.041029 / 1 |
| attention | boundary | 7.039676 / 1 | 6.733746 / 1 | 8.020759 / 1 |
| attention-residual | boundary | 0.002855 / 1 | 0.002795 / 1 | 0.003577 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025097 / 1 | 0.023844 / 1 | 0.026670 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002164 / 1 | 0.001613 / 1 | 0.002374 / 1 |
| router-and-top16 | boundary | 0.549736 / 1 | 0.521013 / 1 | 0.583330 / 1 |
| EDOWN | nested | 0.623254 / 1 | 0.595262 / 1 | 0.897586 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.623815 / 1 | 0.595823 / 1 | 0.898478 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.083617 / 1 | 0.018204 / 1 | 1.680669 / 1 |
| SH1 | nested | 1.064889 / 1 | 1.408410 / 1 | 1.420423 / 1 |
| SH3 | nested | 1.052666 / 1 | 1.426193 / 1 | 1.398401 / 1 |
| SH2 | nested | 1.044110 / 1 | 1.371852 / 1 | 1.337638 / 1 |
| shared-expert-during-read | boundary | 3.172124 / 1 | 4.220582 / 1 | 4.170779 / 1 |
| detail:expert-gate | nested | 48.579590 / 16 | 49.610760 / 16 | 52.520926 / 16 |
| detail:expert-up | nested | 42.372300 / 16 | 42.641483 / 16 | 49.603676 / 16 |
| detail:expert-activation | nested | 0.097022 / 16 | 0.095822 / 16 | 0.120788 / 16 |
| detail:expert-down | nested | 48.579300 / 16 | 48.864204 / 16 | 52.469177 / 16 |
| EUP | nested | 0.601072 / 1 | 0.615790 / 1 | 0.598688 / 1 |
| experts-mix-normalize-up | boundary | 140.660140 / 1 | 142.183528 / 1 | 155.685888 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003006 / 1 | 0.003557 / 1 | 0.003266 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591154 / 1 | 0.599339 / 1 | 0.598277 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.596985 / 1 | 0.603828 / 1 | 0.604068 / 1 |
| worker:read | parallel worker | 0.207364 / 1 | 0.187885 / 1 | 0.183162 / 1 |
| worker:decode | parallel worker | 1054.400378 / 1 | 1072.207090 / 1 | 1069.302888 / 1 |
| worker:crc | parallel worker | 593.203110 / 1 | 600.732511 / 1 | 600.691026 / 1 |
| worker:math | parallel worker | 362.125473 / 1 | 361.783394 / 1 | 367.876354 / 1 |
| op:Q   int8 projection | nested | 11.314901 / 1 | 12.035284 / 1 | 13.540395 / 1 |
| op:X   mxfp4 expert proj | nested | 139.656173 / 1 | 141.240045 / 1 | 154.749049 / 1 |
| op:N   rmsnorm | nested | 0.031929 / 1 | 0.031698 / 1 | 0.032021 / 1 |
| op:SiTU + sigma | nested | 0.104123 / 1 | 0.106289 / 1 | 0.131074 / 1 |
| op:AR  snapshot aggregate | nested | 0.030968 / 1 | 0.030528 / 1 | 0.032450 / 1 |
| op:SA  softmax attention | nested | 0.006282 / 1 | 0.009388 / 1 | 0.013586 / 1 |
| op:router dot product | nested | 0.547432 / 1 | 0.518488 / 1 | 0.579953 / 1 |
| op:top-k selection | nested | 0.001923 / 1 | 0.002184 / 1 | 0.002815 / 1 |
| detail:read-ahead-wait | nested | 0.001463 / 1 | 0.001484 / 1 | 1.653556 / 1 |
| total:layer | total | 152.797497 / 1 | 154.947054 / 1 | 171.720443 / 1 |

### Layer 16

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004979 / 1 | 0.003577 / 1 | 0.003676 / 1 |
| pre-attention-aggregation | boundary | 0.016140 / 1 | 0.015208 / 1 | 0.015359 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011151 / 1 | 0.011411 / 1 | 0.011091 / 1 |
| Q | nested | 3.436618 / 1 | 3.362860 / 1 | 3.370865 / 1 |
| K | nested | 2.435789 / 1 | 2.377410 / 1 | 2.353075 / 1 |
| V | nested | 2.331103 / 1 | 2.305705 / 1 | 2.298843 / 1 |
| B | nested | 0.022312 / 1 | 0.021651 / 1 | 0.021290 / 1 |
| FA | nested | 0.033994 / 1 | 0.031629 / 1 | 0.034534 / 1 |
| FB | nested | 0.053330 / 1 | 0.050705 / 1 | 0.054562 / 1 |
| G | nested | 2.111363 / 1 | 2.049848 / 1 | 2.080024 / 1 |
| O | nested | 2.136490 / 1 | 2.080976 / 1 | 2.092608 / 1 |
| attention | boundary | 13.665530 / 1 | 13.358156 / 1 | 13.365821 / 1 |
| attention-residual | boundary | 0.003767 / 1 | 0.003977 / 1 | 0.003696 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025007 / 1 | 0.031719 / 1 | 0.023574 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002294 / 1 | 0.001883 / 1 | 0.001633 / 1 |
| router-and-top16 | boundary | 0.600501 / 1 | 0.557962 / 1 | 0.583580 / 1 |
| EDOWN | nested | 0.616401 / 1 | 0.594029 / 1 | 0.598989 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.617112 / 1 | 0.594660 / 1 | 0.599610 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.059701 / 1 | 0.016060 / 1 | 0.019226 / 1 |
| SH1 | nested | 1.051504 / 1 | 1.011750 / 1 | 1.011028 / 1 |
| SH3 | nested | 1.053337 / 1 | 1.010888 / 1 | 1.017210 / 1 |
| SH2 | nested | 1.028751 / 1 | 1.005047 / 1 | 1.017681 / 1 |
| shared-expert-during-read | boundary | 3.144021 / 1 | 3.038314 / 1 | 3.055997 / 1 |
| detail:expert-gate | nested | 48.134812 / 16 | 48.714365 / 16 | 53.496989 / 16 |
| detail:expert-up | nested | 42.636740 / 16 | 42.757650 / 16 | 50.372380 / 16 |
| detail:expert-activation | nested | 0.095869 / 16 | 0.096369 / 16 | 0.119663 / 16 |
| detail:expert-down | nested | 48.775637 / 16 | 48.830394 / 16 | 51.181826 / 16 |
| EUP | nested | 0.597706 / 1 | 0.604559 / 1 | 0.728381 / 1 |
| experts-mix-normalize-up | boundary | 140.648379 / 1 | 141.355712 / 1 | 156.274076 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002736 / 1 | 0.003226 / 1 | 0.002866 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584752 / 1 | 0.590603 / 1 | 0.613896 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.011569 / 1 | 0.977285 / 1 | 1.004736 / 1 |
| worker:read | parallel worker | 0.066532 / 1 | 0.066746 / 1 | 0.068809 / 1 |
| worker:decode | parallel worker | 1057.461816 / 1 | 1066.360388 / 1 | 1071.904330 / 1 |
| worker:crc | parallel worker | 594.277364 / 1 | 600.564269 / 1 | 603.763172 / 1 |
| worker:math | parallel worker | 365.564719 / 1 | 364.053648 / 1 | 373.168806 / 1 |
| op:Q   int8 projection | nested | 16.907122 / 1 | 16.505632 / 1 | 16.677596 / 1 |
| op:X   mxfp4 expert proj | nested | 139.671385 / 1 | 140.427130 / 1 | 155.206815 / 1 |
| op:N   rmsnorm | nested | 0.040897 / 1 | 0.039846 / 1 | 0.030968 / 1 |
| op:L   l2 per-head | nested | 0.005730 / 1 | 0.005350 / 1 | 0.005561 / 1 |
| op:SiTU + sigma | nested | 0.103143 / 1 | 0.103991 / 1 | 0.126197 / 1 |
| op:C   shortconv | nested | 0.059491 / 1 | 0.057597 / 1 | 0.035687 / 1 |
| op:AR  snapshot aggregate | nested | 0.031038 / 1 | 0.037550 / 1 | 0.029396 / 1 |
| op:D   kda delta-rule | nested | 0.127388 / 1 | 0.156532 / 1 | 0.112249 / 1 |
| op:router dot product | nested | 0.597626 / 1 | 0.555518 / 1 | 0.580714 / 1 |
| op:top-k selection | nested | 0.002575 / 1 | 0.002054 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.072937 / 1 | 0.074259 / 1 | 0.057537 / 1 |
| detail:read-ahead-wait | nested | 0.001430 / 1 | 0.001453 / 1 | 0.001221 / 1 |
| total:layer | total | 159.818699 / 1 | 159.974863 / 1 | 174.970582 / 1 |

### Layer 17

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005110 / 1 | 0.005110 / 1 | 0.005680 / 1 |
| pre-attention-aggregation | boundary | 0.016861 / 1 | 0.017002 / 1 | 0.017473 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011491 / 1 | 0.012584 / 1 | 0.011241 / 1 |
| Q | nested | 3.377647 / 1 | 2.791182 / 1 | 3.273654 / 1 |
| K | nested | 2.423897 / 1 | 2.384153 / 1 | 2.742121 / 1 |
| V | nested | 2.322297 / 1 | 2.329440 / 1 | 2.353445 / 1 |
| B | nested | 0.028504 / 1 | 0.026840 / 1 | 0.026930 / 1 |
| FA | nested | 0.032822 / 1 | 0.032500 / 1 | 0.034364 / 1 |
| FB | nested | 0.055203 / 1 | 0.053159 / 1 | 0.054321 / 1 |
| G | nested | 2.157239 / 1 | 2.079293 / 1 | 2.083992 / 1 |
| O | nested | 2.123406 / 1 | 2.079163 / 1 | 2.083521 / 1 |
| attention | boundary | 13.799381 / 1 | 12.983216 / 1 | 13.836249 / 1 |
| attention-residual | boundary | 0.003868 / 1 | 0.003757 / 1 | 0.003686 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024336 / 1 | 0.023895 / 1 | 0.023634 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002394 / 1 | 0.001703 / 1 | 0.001813 / 1 |
| router-and-top16 | boundary | 0.596704 / 1 | 0.576897 / 1 | 0.572129 / 1 |
| EDOWN | nested | 0.623975 / 1 | 0.582027 / 1 | 0.595032 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.624576 / 1 | 0.582588 / 1 | 0.595653 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.071844 / 1 | 0.047027 / 1 | 0.009859 / 1 |
| SH1 | nested | 1.038158 / 1 | 0.990119 / 1 | 0.994067 / 1 |
| SH3 | nested | 1.050853 / 1 | 0.994407 / 1 | 0.998294 / 1 |
| SH2 | nested | 1.039491 / 1 | 1.003313 / 1 | 1.006379 / 1 |
| shared-expert-during-read | boundary | 3.139273 / 1 | 2.998369 / 1 | 3.009381 / 1 |
| detail:expert-gate | nested | 47.952179 / 16 | 49.031363 / 16 | 51.957553 / 16 |
| detail:expert-up | nested | 43.403116 / 16 | 43.583604 / 16 | 47.692106 / 16 |
| detail:expert-activation | nested | 0.094654 / 16 | 0.095663 / 16 | 0.112470 / 16 |
| detail:expert-down | nested | 49.730629 / 16 | 49.893650 / 16 | 51.549990 / 16 |
| EUP | nested | 0.598608 / 1 | 0.611552 / 1 | 0.602926 / 1 |
| experts-mix-normalize-up | boundary | 142.230814 / 1 | 143.609180 / 1 | 152.261503 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002716 / 1 | 0.003106 / 1 | 0.002465 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584972 / 1 | 0.591284 / 1 | 0.591134 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.980932 / 1 | 0.938282 / 1 | 0.947980 / 1 |
| worker:read | parallel worker | 0.211316 / 1 | 0.195756 / 1 | 0.188151 / 1 |
| worker:decode | parallel worker | 1056.411506 / 1 | 1070.573978 / 1 | 1068.051316 / 1 |
| worker:crc | parallel worker | 595.188968 / 1 | 601.115653 / 1 | 601.660477 / 1 |
| worker:math | parallel worker | 366.361446 / 1 | 364.953134 / 1 | 368.834899 / 1 |
| op:Q   int8 projection | nested | 16.870596 / 1 | 15.955748 / 1 | 16.847616 / 1 |
| op:X   mxfp4 expert proj | nested | 141.212601 / 1 | 142.632405 / 1 | 151.345923 / 1 |
| op:N   rmsnorm | nested | 0.039204 / 1 | 0.042238 / 1 | 0.031048 / 1 |
| op:L   l2 per-head | nested | 0.005440 / 1 | 0.005310 / 1 | 0.005330 / 1 |
| op:SiTU + sigma | nested | 0.101920 / 1 | 0.102843 / 1 | 0.119262 / 1 |
| op:C   shortconv | nested | 0.073697 / 1 | 0.057117 / 1 | 0.031098 / 1 |
| op:AR  snapshot aggregate | nested | 0.031619 / 1 | 0.031188 / 1 | 0.031780 / 1 |
| op:D   kda delta-rule | nested | 0.202719 / 1 | 0.244847 / 1 | 0.242933 / 1 |
| op:router dot product | nested | 0.594270 / 1 | 0.574353 / 1 | 0.569624 / 1 |
| op:top-k selection | nested | 0.001793 / 1 | 0.002194 / 1 | 0.002094 / 1 |
| op:alpha / beta / gate | nested | 0.072646 / 1 | 0.074659 / 1 | 0.058269 / 1 |
| detail:read-ahead-wait | nested | 0.001523 / 1 | 0.001530 / 1 | 0.001792 / 1 |
| total:layer | total | 161.516500 / 1 | 161.809640 / 1 | 171.304846 / 1 |

### Layer 18

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005330 / 1 | 0.005391 / 1 | 0.005531 / 1 |
| pre-attention-aggregation | boundary | 0.017292 / 1 | 0.016341 / 1 | 0.017203 / 1 |
| snapshot-push | boundary | 0.000110 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010911 / 1 | 0.011421 / 1 | 0.012103 / 1 |
| Q | nested | 3.385742 / 1 | 3.225493 / 1 | 3.239690 / 1 |
| K | nested | 2.453713 / 1 | 2.372651 / 1 | 2.365337 / 1 |
| V | nested | 2.296378 / 1 | 2.348075 / 1 | 2.305636 / 1 |
| B | nested | 0.022071 / 1 | 0.021640 / 1 | 0.021490 / 1 |
| FA | nested | 0.033002 / 1 | 0.034995 / 1 | 0.034805 / 1 |
| FB | nested | 0.051997 / 1 | 0.047709 / 1 | 0.051096 / 1 |
| G | nested | 2.128805 / 1 | 2.100503 / 1 | 2.052123 / 1 |
| O | nested | 2.117394 / 1 | 2.088911 / 1 | 2.045069 / 1 |
| attention | boundary | 13.693973 / 1 | 13.423008 / 1 | 13.300850 / 1 |
| attention-residual | boundary | 0.003867 / 1 | 0.003817 / 1 | 0.003697 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024355 / 1 | 0.024376 / 1 | 0.024416 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002575 / 1 | 0.001492 / 1 | 0.001983 / 1 |
| router-and-top16 | boundary | 0.596625 / 1 | 0.574803 / 1 | 0.571868 / 1 |
| EDOWN | nested | 0.613976 / 1 | 0.589852 / 1 | 0.587908 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.614698 / 1 | 0.590463 / 1 | 0.588529 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.061525 / 1 | 0.025709 / 1 | 0.017333 / 1 |
| SH1 | nested | 1.034602 / 1 | 1.011609 / 1 | 1.006600 / 1 |
| SH3 | nested | 1.040453 / 1 | 1.010037 / 1 | 1.001971 / 1 |
| SH2 | nested | 1.024493 / 1 | 0.997182 / 1 | 1.004566 / 1 |
| shared-expert-during-read | boundary | 3.109918 / 1 | 3.029347 / 1 | 3.023787 / 1 |
| detail:expert-gate | nested | 48.726846 / 16 | 49.719452 / 16 | 51.935313 / 16 |
| detail:expert-up | nested | 42.508246 / 16 | 42.674931 / 16 | 47.601779 / 16 |
| detail:expert-activation | nested | 0.093882 / 16 | 0.096411 / 16 | 0.111646 / 16 |
| detail:expert-down | nested | 48.572818 / 16 | 48.811226 / 16 | 50.233161 / 16 |
| EUP | nested | 0.604398 / 1 | 0.601593 / 1 | 0.599961 / 1 |
| experts-mix-normalize-up | boundary | 140.946476 / 1 | 142.274327 / 1 | 150.835840 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003346 / 1 | 0.002765 / 1 | 0.002855 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584461 / 1 | 0.587888 / 1 | 0.591685 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.982926 / 1 | 0.944083 / 1 | 0.963880 / 1 |
| worker:read | parallel worker | 0.068474 / 1 | 0.066964 / 1 | 0.066789 / 1 |
| worker:decode | parallel worker | 1059.390199 / 1 | 1073.685883 / 1 | 1073.882685 / 1 |
| worker:crc | parallel worker | 594.056725 / 1 | 599.577627 / 1 | 599.714395 / 1 |
| worker:math | parallel worker | 361.766988 / 1 | 362.352174 / 1 | 365.893109 / 1 |
| op:Q   int8 projection | nested | 16.805520 / 1 | 16.448839 / 1 | 16.314848 / 1 |
| op:X   mxfp4 expert proj | nested | 139.931309 / 1 | 141.331148 / 1 | 149.916345 / 1 |
| op:N   rmsnorm | nested | 0.044633 / 1 | 0.039073 / 1 | 0.031480 / 1 |
| op:L   l2 per-head | nested | 0.004980 / 1 | 0.005721 / 1 | 0.005891 / 1 |
| op:SiTU + sigma | nested | 0.100757 / 1 | 0.103184 / 1 | 0.118380 / 1 |
| op:C   shortconv | nested | 0.059101 / 1 | 0.064380 / 1 | 0.030657 / 1 |
| op:AR  snapshot aggregate | nested | 0.031739 / 1 | 0.030887 / 1 | 0.032019 / 1 |
| op:D   kda delta-rule | nested | 0.205524 / 1 | 0.244737 / 1 | 0.238175 / 1 |
| op:router dot product | nested | 0.593659 / 1 | 0.572079 / 1 | 0.568883 / 1 |
| op:top-k selection | nested | 0.002314 / 1 | 0.002364 / 1 | 0.002605 / 1 |
| op:alpha / beta / gate | nested | 0.073287 / 1 | 0.074269 / 1 | 0.058409 / 1 |
| detail:read-ahead-wait | nested | 0.001732 / 1 | 0.001593 / 1 | 0.001781 / 1 |
| total:layer | total | 160.080228 / 1 | 160.933753 / 1 | 169.376204 / 1 |

### Layer 19

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005260 / 1 | 0.004829 / 1 | 0.004588 / 1 |
| pre-attention-aggregation | boundary | 0.016541 / 1 | 0.015809 / 1 | 0.016691 / 1 |
| snapshot-push | boundary | 0.000031 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011170 / 1 | 0.011952 / 1 | 0.011191 / 1 |
| QA | nested | 0.350244 / 1 | 0.349032 / 1 | 0.349412 / 1 |
| QB | nested | 0.939815 / 1 | 0.928644 / 1 | 0.929816 / 1 |
| KA | nested | 0.137677 / 1 | 0.129292 / 1 | 0.135212 / 1 |
| KB | nested | 0.439571 / 1 | 0.384018 / 1 | 0.425965 / 1 |
| G | nested | 2.845935 / 1 | 2.711614 / 1 | 2.728757 / 1 |
| O | nested | 2.165705 / 1 | 2.025363 / 1 | 2.041152 / 1 |
| attention | boundary | 6.987349 / 1 | 6.658164 / 1 | 6.718237 / 1 |
| attention-residual | boundary | 0.002775 / 1 | 0.002756 / 1 | 0.002776 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024215 / 1 | 0.024115 / 1 | 0.024306 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002044 / 1 | 0.001473 / 1 | 0.001613 / 1 |
| router-and-top16 | boundary | 0.551149 / 1 | 0.520281 / 1 | 0.526423 / 1 |
| EDOWN | nested | 0.616011 / 1 | 0.598768 / 1 | 0.606132 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.616642 / 1 | 0.599389 / 1 | 0.606803 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052629 / 1 | 0.026049 / 1 | 0.017683 / 1 |
| SH1 | nested | 1.058567 / 1 | 1.021448 / 1 | 1.027769 / 1 |
| SH3 | nested | 1.062024 / 1 | 1.023832 / 1 | 1.020296 / 1 |
| SH2 | nested | 1.087902 / 1 | 1.033961 / 1 | 1.028711 / 1 |
| shared-expert-during-read | boundary | 3.219312 / 1 | 3.089791 / 1 | 3.087195 / 1 |
| detail:expert-gate | nested | 55.437178 / 16 | 49.963088 / 16 | 53.226414 / 16 |
| detail:expert-up | nested | 43.732772 / 16 | 42.853998 / 16 | 49.916421 / 16 |
| detail:expert-activation | nested | 0.098374 / 16 | 0.097444 / 16 | 0.126044 / 16 |
| detail:expert-down | nested | 49.894359 / 16 | 49.003306 / 16 | 49.550507 / 16 |
| EUP | nested | 0.594420 / 1 | 0.615920 / 1 | 0.751524 / 1 |
| experts-mix-normalize-up | boundary | 150.195999 / 1 | 142.904474 / 1 | 153.932062 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003466 / 1 | 0.003146 / 1 | 0.003096 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591835 / 1 | 0.598909 / 1 | 0.615189 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.597867 / 1 | 0.603807 / 1 | 0.620529 / 1 |
| worker:read | parallel worker | 0.066424 / 1 | 0.069062 / 1 | 0.066997 / 1 |
| worker:decode | parallel worker | 1061.774369 / 1 | 1074.093039 / 1 | 1075.601262 / 1 |
| worker:crc | parallel worker | 597.765527 / 1 | 600.744921 / 1 | 602.936421 / 1 |
| worker:math | parallel worker | 365.549356 / 1 | 364.107153 / 1 | 372.150685 / 1 |
| op:Q   int8 projection | nested | 11.296528 / 1 | 10.820710 / 1 | 11.043493 / 1 |
| op:X   mxfp4 expert proj | nested | 149.194301 / 1 | 141.948581 / 1 | 152.855923 / 1 |
| op:N   rmsnorm | nested | 0.032621 / 1 | 0.031429 / 1 | 0.031288 / 1 |
| op:SiTU + sigma | nested | 0.105496 / 1 | 0.104541 / 1 | 0.132627 / 1 |
| op:AR  snapshot aggregate | nested | 0.030778 / 1 | 0.029876 / 1 | 0.030727 / 1 |
| op:SA  softmax attention | nested | 0.006873 / 1 | 0.009828 / 1 | 0.012513 / 1 |
| op:router dot product | nested | 0.548425 / 1 | 0.517737 / 1 | 0.524029 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002144 / 1 | 0.002024 / 1 |
| detail:read-ahead-wait | nested | 0.001703 / 1 | 0.001656 / 1 | 0.001692 / 1 |
| total:layer | total | 162.292910 / 1 | 154.472016 / 1 | 165.579494 / 1 |

### Layer 20

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005511 / 1 | 0.004398 / 1 | 0.004719 / 1 |
| pre-attention-aggregation | boundary | 0.015709 / 1 | 0.015348 / 1 | 0.016761 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011391 / 1 | 0.010981 / 1 | 0.011141 / 1 |
| Q | nested | 3.384821 / 1 | 3.308158 / 1 | 3.306545 / 1 |
| K | nested | 2.407676 / 1 | 2.371749 / 1 | 2.678301 / 1 |
| V | nested | 2.298452 / 1 | 2.369325 / 1 | 2.709651 / 1 |
| B | nested | 0.029595 / 1 | 0.023334 / 1 | 0.028082 / 1 |
| FA | nested | 0.031889 / 1 | 0.032511 / 1 | 0.031879 / 1 |
| FB | nested | 0.048551 / 1 | 0.049462 / 1 | 0.055464 / 1 |
| G | nested | 2.126882 / 1 | 2.097407 / 1 | 2.451979 / 1 |
| O | nested | 2.151187 / 1 | 2.098919 / 1 | 2.462840 / 1 |
| attention | boundary | 13.671331 / 1 | 13.424961 / 1 | 14.779892 / 1 |
| attention-residual | boundary | 0.003697 / 1 | 0.003557 / 1 | 0.003767 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024506 / 1 | 0.024485 / 1 | 0.024075 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002154 / 1 | 0.001603 / 1 | 0.001944 / 1 |
| router-and-top16 | boundary | 0.595953 / 1 | 0.572419 / 1 | 0.601103 / 1 |
| EDOWN | nested | 0.611943 / 1 | 0.595462 / 1 | 0.697583 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.612574 / 1 | 0.596123 / 1 | 0.698274 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.090750 / 1 | 0.020618 / 1 | 0.011942 / 1 |
| SH1 | nested | 1.047356 / 1 | 1.008443 / 1 | 1.186986 / 1 |
| SH3 | nested | 1.058878 / 1 | 1.003064 / 1 | 1.171027 / 1 |
| SH2 | nested | 1.045913 / 1 | 1.006510 / 1 | 1.195553 / 1 |
| shared-expert-during-read | boundary | 3.162926 / 1 | 3.029007 / 1 | 3.567061 / 1 |
| detail:expert-gate | nested | 49.517444 / 16 | 51.282641 / 16 | 52.264966 / 16 |
| detail:expert-up | nested | 42.395964 / 16 | 44.076643 / 16 | 48.564815 / 16 |
| detail:expert-activation | nested | 0.099756 / 16 | 0.101490 / 16 | 0.116578 / 16 |
| detail:expert-down | nested | 48.561946 / 16 | 51.306492 / 16 | 51.802572 / 16 |
| EUP | nested | 0.598938 / 1 | 0.606173 / 1 | 0.591836 / 1 |
| experts-mix-normalize-up | boundary | 141.627969 / 1 | 147.753330 / 1 | 153.702193 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003006 / 1 | 0.003096 / 1 | 0.002595 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584933 / 1 | 0.589471 / 1 | 0.588479 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.018762 / 1 | 0.990019 / 1 | 0.959472 / 1 |
| worker:read | parallel worker | 0.067481 / 1 | 0.069121 / 1 | 0.066251 / 1 |
| worker:decode | parallel worker | 1054.865606 / 1 | 1076.659827 / 1 | 1073.237115 / 1 |
| worker:crc | parallel worker | 594.929102 / 1 | 603.188859 / 1 | 601.950128 / 1 |
| worker:math | parallel worker | 363.372568 / 1 | 366.053914 / 1 | 369.427104 / 1 |
| op:Q   int8 projection | nested | 16.840398 / 1 | 16.569004 / 1 | 18.566312 / 1 |
| op:X   mxfp4 expert proj | nested | 140.607060 / 1 | 146.798917 / 1 | 152.786021 / 1 |
| op:N   rmsnorm | nested | 0.039374 / 1 | 0.041618 / 1 | 0.031117 / 1 |
| op:L   l2 per-head | nested | 0.005480 / 1 | 0.005751 / 1 | 0.005410 / 1 |
| op:SiTU + sigma | nested | 0.106739 / 1 | 0.108974 / 1 | 0.126075 / 1 |
| op:C   shortconv | nested | 0.062056 / 1 | 0.051236 / 1 | 0.032210 / 1 |
| op:AR  snapshot aggregate | nested | 0.030377 / 1 | 0.030085 / 1 | 0.031188 / 1 |
| op:D   kda delta-rule | nested | 0.149980 / 1 | 0.141615 / 1 | 0.134010 / 1 |
| op:router dot product | nested | 0.593168 / 1 | 0.569934 / 1 | 0.598007 / 1 |
| op:top-k selection | nested | 0.002395 / 1 | 0.002154 / 1 | 0.002745 / 1 |
| op:alpha / beta / gate | nested | 0.072285 / 1 | 0.081301 / 1 | 0.057447 / 1 |
| detail:read-ahead-wait | nested | 0.001715 / 1 | 0.001674 / 1 | 0.001594 / 1 |
| total:layer | total | 160.853121 / 1 | 166.456838 / 1 | 174.391602 / 1 |

### Layer 21

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005821 / 1 | 0.005330 / 1 | 0.006141 / 1 |
| pre-attention-aggregation | boundary | 0.017312 / 1 | 0.017272 / 1 | 0.016551 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011000 / 1 | 0.013025 / 1 | 0.010841 / 1 |
| Q | nested | 3.300073 / 1 | 3.136297 / 1 | 3.218049 / 1 |
| K | nested | 2.406615 / 1 | 2.371519 / 1 | 2.360137 / 1 |
| V | nested | 2.304483 / 1 | 2.367671 / 1 | 2.332065 / 1 |
| B | nested | 0.021591 / 1 | 0.021090 / 1 | 0.022432 / 1 |
| FA | nested | 0.029305 / 1 | 0.029575 / 1 | 0.028643 / 1 |
| FB | nested | 0.048912 / 1 | 0.049272 / 1 | 0.058930 / 1 |
| G | nested | 2.091486 / 1 | 2.039218 / 1 | 2.081317 / 1 |
| O | nested | 2.090504 / 1 | 2.072310 / 1 | 2.077750 / 1 |
| attention | boundary | 13.442995 / 1 | 13.242710 / 1 | 13.365851 / 1 |
| attention-residual | boundary | 0.003747 / 1 | 0.003697 / 1 | 0.003527 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024666 / 1 | 0.023865 / 1 | 0.024014 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001803 / 1 | 0.002124 / 1 | 0.001844 / 1 |
| router-and-top16 | boundary | 0.583359 / 1 | 0.573642 / 1 | 0.585995 / 1 |
| EDOWN | nested | 0.607695 / 1 | 0.583900 / 1 | 0.595321 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.608356 / 1 | 0.584631 / 1 | 0.596023 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.058700 / 1 | 0.021480 / 1 | 0.011682 / 1 |
| SH1 | nested | 1.025114 / 1 | 1.007211 / 1 | 1.001981 / 1 |
| SH3 | nested | 1.027169 / 1 | 1.005307 / 1 | 1.002062 / 1 |
| SH2 | nested | 1.027198 / 1 | 1.009295 / 1 | 1.009545 / 1 |
| shared-expert-during-read | boundary | 3.090040 / 1 | 3.032443 / 1 | 3.024228 / 1 |
| detail:expert-gate | nested | 49.197327 / 16 | 49.455907 / 16 | 53.294476 / 16 |
| detail:expert-up | nested | 42.360998 / 16 | 42.632856 / 16 | 51.938515 / 16 |
| detail:expert-activation | nested | 0.095869 / 16 | 0.097420 / 16 | 0.132238 / 16 |
| detail:expert-down | nested | 48.494553 / 16 | 48.835158 / 16 | 52.089632 / 16 |
| EUP | nested | 0.603127 / 1 | 0.606723 / 1 | 0.618975 / 1 |
| experts-mix-normalize-up | boundary | 141.160626 / 1 | 142.002450 / 1 | 161.321403 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003186 / 1 | 0.003046 / 1 | 0.002554 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.585774 / 1 | 0.589962 / 1 | 0.592397 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.951688 / 1 | 0.947710 / 1 | 0.984689 / 1 |
| worker:read | parallel worker | 0.068774 / 1 | 0.068260 / 1 | 0.068692 / 1 |
| worker:decode | parallel worker | 1051.820593 / 1 | 1071.122680 / 1 | 1067.790836 / 1 |
| worker:crc | parallel worker | 594.612127 / 1 | 600.108654 / 1 | 604.078287 / 1 |
| worker:math | parallel worker | 363.110410 / 1 | 362.968327 / 1 | 369.726311 / 1 |
| op:Q   int8 projection | nested | 16.581787 / 1 | 16.297977 / 1 | 16.405715 / 1 |
| op:X   mxfp4 expert proj | nested | 140.183542 / 1 | 141.056202 / 1 | 157.500885 / 1 |
| op:N   rmsnorm | nested | 0.040186 / 1 | 0.039052 / 1 | 0.030887 / 1 |
| op:L   l2 per-head | nested | 0.006532 / 1 | 0.005621 / 1 | 0.006001 / 1 |
| op:SiTU + sigma | nested | 0.102853 / 1 | 0.104813 / 1 | 0.138599 / 1 |
| op:C   shortconv | nested | 0.054430 / 1 | 0.051787 / 1 | 0.031509 / 1 |
| op:AR  snapshot aggregate | nested | 0.032350 / 1 | 0.030878 / 1 | 0.030698 / 1 |
| op:D   kda delta-rule | nested | 0.240148 / 1 | 0.244346 / 1 | 0.239227 / 1 |
| op:router dot product | nested | 0.580825 / 1 | 0.570776 / 1 | 0.582838 / 1 |
| op:top-k selection | nested | 0.002244 / 1 | 0.002514 / 1 | 0.002755 / 1 |
| op:alpha / beta / gate | nested | 0.076373 / 1 | 0.073827 / 1 | 0.058058 / 1 |
| detail:read-ahead-wait | nested | 0.001692 / 1 | 0.001572 / 1 | 2.867154 / 1 |
| total:layer | total | 159.971165 / 1 | 160.480688 / 1 | 179.962275 / 1 |

### Layer 22

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004649 / 1 | 0.004548 / 1 | 0.005470 / 1 |
| pre-attention-aggregation | boundary | 0.017683 / 1 | 0.017272 / 1 | 0.016441 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000110 / 1 |
| pre-attention-normalization | boundary | 0.011341 / 1 | 0.011081 / 1 | 0.011731 / 1 |
| Q | nested | 3.326192 / 1 | 3.282340 / 1 | 3.568274 / 1 |
| K | nested | 2.399651 / 1 | 2.382179 / 1 | 2.789890 / 1 |
| V | nested | 2.314713 / 1 | 2.325142 / 1 | 2.846386 / 1 |
| B | nested | 0.025487 / 1 | 0.026890 / 1 | 0.031920 / 1 |
| FA | nested | 0.033413 / 1 | 0.028212 / 1 | 0.036499 / 1 |
| FB | nested | 0.049161 / 1 | 0.047188 / 1 | 0.057698 / 1 |
| G | nested | 2.111594 / 1 | 2.069285 / 1 | 2.388621 / 1 |
| O | nested | 2.100343 / 1 | 2.072200 / 1 | 2.442962 / 1 |
| attention | boundary | 13.510590 / 1 | 13.398211 / 1 | 15.318157 / 1 |
| attention-residual | boundary | 0.004108 / 1 | 0.003717 / 1 | 0.003637 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024987 / 1 | 0.024115 / 1 | 0.024085 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002124 / 1 | 0.002003 / 1 | 0.002345 / 1 |
| router-and-top16 | boundary | 0.582238 / 1 | 0.567630 / 1 | 0.590503 / 1 |
| EDOWN | nested | 0.601533 / 1 | 0.596695 / 1 | 0.688716 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.602234 / 1 | 0.597536 / 1 | 0.689518 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052378 / 1 | 0.036328 / 1 | 0.017132 / 1 |
| SH1 | nested | 1.022309 / 1 | 1.013082 / 1 | 1.370460 / 1 |
| SH3 | nested | 1.037478 / 1 | 1.004445 / 1 | 1.406487 / 1 |
| SH2 | nested | 1.047476 / 1 | 0.999076 / 1 | 1.401647 / 1 |
| shared-expert-during-read | boundary | 3.118233 / 1 | 3.027103 / 1 | 4.194263 / 1 |
| detail:expert-gate | nested | 49.459785 / 16 | 50.036602 / 16 | 53.140922 / 16 |
| detail:expert-up | nested | 42.489026 / 16 | 42.624400 / 16 | 51.568896 / 16 |
| detail:expert-activation | nested | 0.097742 / 16 | 0.096449 / 16 | 0.126515 / 16 |
| detail:expert-down | nested | 48.645264 / 16 | 48.871204 / 16 | 52.308549 / 16 |
| EUP | nested | 0.599139 / 1 | 0.600331 / 1 | 0.605341 / 1 |
| experts-mix-normalize-up | boundary | 141.693561 / 1 | 142.627798 / 1 | 158.116678 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003056 / 1 | 0.002795 / 1 | 0.003056 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590192 / 1 | 0.588629 / 1 | 0.593959 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.964161 / 1 | 0.940817 / 1 | 0.961977 / 1 |
| worker:read | parallel worker | 0.066953 / 1 | 0.067144 / 1 | 0.068940 / 1 |
| worker:decode | parallel worker | 1057.628884 / 1 | 1070.356232 / 1 | 1075.289422 / 1 |
| worker:crc | parallel worker | 594.636470 / 1 | 600.451923 / 1 | 603.957087 / 1 |
| worker:math | parallel worker | 363.317993 / 1 | 363.551311 / 1 | 371.708201 / 1 |
| op:Q   int8 projection | nested | 16.667044 / 1 | 16.445573 / 1 | 19.633266 / 1 |
| op:X   mxfp4 expert proj | nested | 140.725752 / 1 | 141.662575 / 1 | 157.189245 / 1 |
| op:N   rmsnorm | nested | 0.047790 / 1 | 0.045465 / 1 | 0.031850 / 1 |
| op:L   l2 per-head | nested | 0.006461 / 1 | 0.005400 / 1 | 0.005570 / 1 |
| op:SiTU + sigma | nested | 0.105276 / 1 | 0.103504 / 1 | 0.137286 / 1 |
| op:C   shortconv | nested | 0.061475 / 1 | 0.058098 / 1 | 0.031568 / 1 |
| op:AR  snapshot aggregate | nested | 0.032280 / 1 | 0.031689 / 1 | 0.030597 / 1 |
| op:D   kda delta-rule | nested | 0.231812 / 1 | 0.247602 / 1 | 0.251490 / 1 |
| op:router dot product | nested | 0.579713 / 1 | 0.564704 / 1 | 0.587727 / 1 |
| op:top-k selection | nested | 0.002204 / 1 | 0.002555 / 1 | 0.002215 / 1 |
| op:alpha / beta / gate | nested | 0.072546 / 1 | 0.074049 / 1 | 0.058419 / 1 |
| detail:read-ahead-wait | nested | 0.001102 / 1 | 0.001394 / 1 | 0.001121 / 1 |
| total:layer | total | 160.599277 / 1 | 161.268399 / 1 | 179.962986 / 1 |

### Layer 23

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005600 / 1 | 0.004268 / 1 | 0.005119 / 1 |
| pre-attention-aggregation | boundary | 0.016801 / 1 | 0.015830 / 1 | 0.016100 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011812 / 1 | 0.011372 / 1 | 0.010840 / 1 |
| QA | nested | 0.347088 / 1 | 0.347559 / 1 | 0.355914 / 1 |
| QB | nested | 0.926481 / 1 | 0.925158 / 1 | 0.912153 / 1 |
| KA | nested | 0.140472 / 1 | 0.138779 / 1 | 0.136214 / 1 |
| KB | nested | 0.366244 / 1 | 0.401841 / 1 | 0.413362 / 1 |
| G | nested | 2.620293 / 1 | 2.731131 / 1 | 2.723957 / 1 |
| O | nested | 2.140968 / 1 | 2.021375 / 1 | 2.018480 / 1 |
| attention | boundary | 6.645501 / 1 | 6.675236 / 1 | 6.680145 / 1 |
| attention-residual | boundary | 0.002935 / 1 | 0.002695 / 1 | 0.002886 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024236 / 1 | 0.024315 / 1 | 0.023964 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001974 / 1 | 0.001854 / 1 | 0.001553 / 1 |
| router-and-top16 | boundary | 0.526653 / 1 | 0.522366 / 1 | 0.528367 / 1 |
| EDOWN | nested | 0.627943 / 1 | 0.591184 / 1 | 0.587176 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.628594 / 1 | 0.591865 / 1 | 0.587837 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.070842 / 1 | 0.025798 / 1 | 0.010660 / 1 |
| SH1 | nested | 1.066031 / 1 | 1.020676 / 1 | 1.008032 / 1 |
| SH3 | nested | 1.053016 / 1 | 1.029773 / 1 | 1.018893 / 1 |
| SH2 | nested | 1.039832 / 1 | 1.029533 / 1 | 1.010196 / 1 |
| shared-expert-during-read | boundary | 3.169840 / 1 | 3.093407 / 1 | 3.049425 / 1 |
| detail:expert-gate | nested | 49.554051 / 16 | 49.758784 / 16 | 53.844029 / 16 |
| detail:expert-up | nested | 42.909333 / 16 | 43.372037 / 16 | 51.767286 / 16 |
| detail:expert-activation | nested | 0.098303 / 16 | 0.097673 / 16 | 0.125805 / 16 |
| detail:expert-down | nested | 48.742624 / 16 | 48.900183 / 16 | 51.713629 / 16 |
| EUP | nested | 0.599980 / 1 | 0.614417 / 1 | 0.720074 / 1 |
| experts-mix-normalize-up | boundary | 142.328998 / 1 | 143.129033 / 1 | 158.732779 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003106 / 1 | 0.003296 / 1 | 0.003677 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.595422 / 1 | 0.600100 / 1 | 0.612764 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.600672 / 1 | 0.605441 / 1 | 0.618525 / 1 |
| worker:read | parallel worker | 0.213704 / 1 | 0.211103 / 1 | 0.200678 / 1 |
| worker:decode | parallel worker | 1060.429730 / 1 | 1073.105164 / 1 | 1076.085606 / 1 |
| worker:crc | parallel worker | 599.288248 / 1 | 602.652193 / 1 | 607.678534 / 1 |
| worker:math | parallel worker | 362.944845 / 1 | 363.018134 / 1 | 370.579856 / 1 |
| op:Q   int8 projection | nested | 10.927126 / 1 | 10.850083 / 1 | 10.903221 / 1 |
| op:X   mxfp4 expert proj | nested | 141.340283 / 1 | 142.164702 / 1 | 157.494227 / 1 |
| op:N   rmsnorm | nested | 0.031769 / 1 | 0.031228 / 1 | 0.031347 / 1 |
| op:SiTU + sigma | nested | 0.105478 / 1 | 0.107522 / 1 | 0.133930 / 1 |
| op:AR  snapshot aggregate | nested | 0.030667 / 1 | 0.030347 / 1 | 0.030116 / 1 |
| op:SA  softmax attention | nested | 0.006072 / 1 | 0.008756 / 1 | 0.013125 / 1 |
| op:router dot product | nested | 0.523838 / 1 | 0.519690 / 1 | 0.525742 / 1 |
| op:top-k selection | nested | 0.002515 / 1 | 0.002355 / 1 | 0.002264 / 1 |
| detail:read-ahead-wait | nested | 0.001453 / 1 | 0.001583 / 1 | 0.001573 / 1 |
| total:layer | total | 154.045367 / 1 | 154.714379 / 1 | 170.279281 / 1 |

### Layer 24

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004418 / 1 | 0.004388 / 1 | 0.004619 / 1 |
| pre-attention-aggregation | boundary | 0.015018 / 1 | 0.016862 / 1 | 0.015609 / 1 |
| snapshot-push | boundary | 0.001543 / 1 | 0.001373 / 1 | 0.001493 / 1 |
| pre-attention-normalization | boundary | 0.011241 / 1 | 0.011642 / 1 | 0.011872 / 1 |
| Q | nested | 3.355987 / 1 | 3.292469 / 1 | 3.438331 / 1 |
| K | nested | 2.373061 / 1 | 2.361901 / 1 | 2.820837 / 1 |
| V | nested | 2.319121 / 1 | 2.347484 / 1 | 2.815719 / 1 |
| B | nested | 0.022643 / 1 | 0.022732 / 1 | 0.028904 / 1 |
| FA | nested | 0.031869 / 1 | 0.028483 / 1 | 0.033462 / 1 |
| FB | nested | 0.050995 / 1 | 0.052208 / 1 | 0.076933 / 1 |
| G | nested | 2.078572 / 1 | 2.098479 / 1 | 2.383601 / 1 |
| O | nested | 2.121562 / 1 | 2.089712 / 1 | 2.440899 / 1 |
| attention | boundary | 13.415333 / 1 | 13.364338 / 1 | 15.075033 / 1 |
| attention-residual | boundary | 0.002104 / 1 | 0.002214 / 1 | 0.002104 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025608 / 1 | 0.024556 / 1 | 0.024536 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002094 / 1 | 0.001884 / 1 | 0.001854 / 1 |
| router-and-top16 | boundary | 0.584722 / 1 | 0.574943 / 1 | 0.589290 / 1 |
| EDOWN | nested | 0.604389 / 1 | 0.597465 / 1 | 0.684679 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.605120 / 1 | 0.598308 / 1 | 0.685420 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.056426 / 1 | 0.028453 / 1 | 0.014577 / 1 |
| SH1 | nested | 1.029874 / 1 | 1.395286 / 1 | 1.387601 / 1 |
| SH3 | nested | 1.035734 / 1 | 1.425823 / 1 | 1.423839 / 1 |
| SH2 | nested | 1.027940 / 1 | 1.421345 / 1 | 1.398803 / 1 |
| shared-expert-during-read | boundary | 3.104538 / 1 | 4.257601 / 1 | 4.225652 / 1 |
| detail:expert-gate | nested | 49.822990 / 16 | 49.769114 / 16 | 52.708944 / 16 |
| detail:expert-up | nested | 42.914365 / 16 | 42.944779 / 16 | 49.188701 / 16 |
| detail:expert-activation | nested | 0.095525 / 16 | 0.097555 / 16 | 0.115623 / 16 |
| detail:expert-down | nested | 49.121023 / 16 | 49.196736 / 16 | 51.543018 / 16 |
| EUP | nested | 0.600171 / 1 | 0.597295 / 1 | 0.601373 / 1 |
| experts-mix-normalize-up | boundary | 142.963803 / 1 | 142.981227 / 1 | 154.505163 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002675 / 1 | 0.002545 / 1 | 0.002695 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.588089 / 1 | 0.590953 / 1 | 0.597856 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.982294 / 1 | 0.956336 / 1 | 0.990179 / 1 |
| worker:read | parallel worker | 0.066020 / 1 | 0.066384 / 1 | 0.068570 / 1 |
| worker:decode | parallel worker | 1072.674637 / 1 | 1081.269120 / 1 | 1084.750619 / 1 |
| worker:crc | parallel worker | 595.741479 / 1 | 601.125792 / 1 | 603.164760 / 1 |
| worker:math | parallel worker | 365.679050 / 1 | 364.989283 / 1 | 371.161135 / 1 |
| op:Q   int8 projection | nested | 16.650425 / 1 | 17.729019 / 1 | 19.533420 / 1 |
| op:X   mxfp4 expert proj | nested | 141.990825 / 1 | 142.043557 / 1 | 153.600694 / 1 |
| op:N   rmsnorm | nested | 0.039856 / 1 | 0.036478 / 1 | 0.032140 / 1 |
| op:L   l2 per-head | nested | 0.005329 / 1 | 0.005650 / 1 | 0.005300 / 1 |
| op:SiTU + sigma | nested | 0.103121 / 1 | 0.108144 / 1 | 0.126346 / 1 |
| op:C   shortconv | nested | 0.057558 / 1 | 0.058649 / 1 | 0.036889 / 1 |
| op:AR  snapshot aggregate | nested | 0.030467 / 1 | 0.032019 / 1 | 0.030528 / 1 |
| op:D   kda delta-rule | nested | 0.103353 / 1 | 0.125916 / 1 | 0.132117 / 1 |
| op:router dot product | nested | 0.581766 / 1 | 0.572209 / 1 | 0.586475 / 1 |
| op:top-k selection | nested | 0.002555 / 1 | 0.002374 / 1 | 0.002445 / 1 |
| op:alpha / beta / gate | nested | 0.072946 / 1 | 0.073678 / 1 | 0.057748 / 1 |
| detail:read-ahead-wait | nested | 0.001532 / 1 | 0.001685 / 1 | 0.001622 / 1 |
| total:layer | total | 161.784641 / 1 | 162.834524 / 1 | 176.158661 / 1 |

### Layer 25

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005019 / 1 | 0.005009 / 1 | 0.005109 / 1 |
| pre-attention-aggregation | boundary | 0.015919 / 1 | 0.016191 / 1 | 0.016320 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000030 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011922 / 1 | 0.011752 / 1 | 0.011632 / 1 |
| Q | nested | 3.353492 / 1 | 3.279535 / 1 | 3.370294 / 1 |
| K | nested | 2.400153 / 1 | 2.425359 / 1 | 2.853950 / 1 |
| V | nested | 2.312699 / 1 | 2.327768 / 1 | 2.749545 / 1 |
| B | nested | 0.030006 / 1 | 0.025137 / 1 | 0.032341 / 1 |
| FA | nested | 0.032801 / 1 | 0.039875 / 1 | 0.038452 / 1 |
| FB | nested | 0.051126 / 1 | 0.049152 / 1 | 0.057788 / 1 |
| G | nested | 2.091345 / 1 | 2.057633 / 1 | 2.618360 / 1 |
| O | nested | 2.109079 / 1 | 2.085885 / 1 | 2.076237 / 1 |
| attention | boundary | 13.522673 / 1 | 13.467741 / 1 | 14.951662 / 1 |
| attention-residual | boundary | 0.004027 / 1 | 0.004228 / 1 | 0.003847 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024686 / 1 | 0.024616 / 1 | 0.024386 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001653 / 1 | 0.001783 / 1 | 0.001744 / 1 |
| router-and-top16 | boundary | 0.584621 / 1 | 0.575685 / 1 | 0.574133 / 1 |
| EDOWN | nested | 0.606122 / 1 | 0.593499 / 1 | 0.583069 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.606844 / 1 | 0.594330 / 1 | 0.583800 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.056124 / 1 | 0.011892 / 1 | 0.014156 / 1 |
| SH1 | nested | 1.595570 / 1 | 1.016909 / 1 | 1.426164 / 1 |
| SH3 | nested | 1.478572 / 1 | 1.017340 / 1 | 1.366011 / 1 |
| SH2 | nested | 1.020325 / 1 | 1.012591 / 1 | 1.561536 / 1 |
| shared-expert-during-read | boundary | 4.106169 / 1 | 3.057730 / 1 | 4.369871 / 1 |
| detail:expert-gate | nested | 49.080028 / 16 | 50.423345 / 16 | 54.294216 / 16 |
| detail:expert-up | nested | 42.739385 / 16 | 42.865560 / 16 | 52.190236 / 16 |
| detail:expert-activation | nested | 0.095329 / 16 | 0.098216 / 16 | 0.127426 / 16 |
| detail:expert-down | nested | 48.971675 / 16 | 49.109342 / 16 | 51.105110 / 16 |
| EUP | nested | 0.603347 / 1 | 0.607755 / 1 | 0.722249 / 1 |
| experts-mix-normalize-up | boundary | 141.894327 / 1 | 143.476032 / 1 | 163.296381 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003076 / 1 | 0.003016 / 1 | 0.003096 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589040 / 1 | 0.589341 / 1 | 0.606383 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.966986 / 1 | 0.950866 / 1 | 0.993385 / 1 |
| worker:read | parallel worker | 0.067870 / 1 | 0.066925 / 1 | 0.066249 / 1 |
| worker:decode | parallel worker | 1072.205348 / 1 | 1082.788059 / 1 | 1085.849015 / 1 |
| worker:crc | parallel worker | 595.185547 / 1 | 600.915658 / 1 | 603.434044 / 1 |
| worker:math | parallel worker | 362.321071 / 1 | 363.737408 / 1 | 371.179574 / 1 |
| op:Q   int8 projection | nested | 17.682864 / 1 | 16.536822 / 1 | 19.454452 / 1 |
| op:X   mxfp4 expert proj | nested | 140.924548 / 1 | 142.534935 / 1 | 157.766375 / 1 |
| op:N   rmsnorm | nested | 0.039343 / 1 | 0.044223 / 1 | 0.031068 / 1 |
| op:L   l2 per-head | nested | 0.005741 / 1 | 0.005089 / 1 | 0.005470 / 1 |
| op:SiTU + sigma | nested | 0.102942 / 1 | 0.105447 / 1 | 0.138258 / 1 |
| op:C   shortconv | nested | 0.069250 / 1 | 0.056366 / 1 | 0.030657 / 1 |
| op:AR  snapshot aggregate | nested | 0.030747 / 1 | 0.030837 / 1 | 0.030758 / 1 |
| op:D   kda delta-rule | nested | 0.232444 / 1 | 0.243986 / 1 | 0.241020 / 1 |
| op:router dot product | nested | 0.582468 / 1 | 0.573221 / 1 | 0.571607 / 1 |
| op:top-k selection | nested | 0.001833 / 1 | 0.002104 / 1 | 0.002104 / 1 |
| op:alpha / beta / gate | nested | 0.073286 / 1 | 0.073458 / 1 | 0.058439 / 1 |
| detail:read-ahead-wait | nested | 0.001833 / 1 | 0.001713 / 1 | 4.502218 / 1 |
| total:layer | total | 161.815599 / 1 | 162.209497 / 1 | 184.858469 / 1 |

### Layer 26

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005601 / 1 | 0.005531 / 1 | 0.005480 / 1 |
| pre-attention-aggregation | boundary | 0.016721 / 1 | 0.016951 / 1 | 0.016862 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.012273 / 1 | 0.012363 / 1 | 0.011302 / 1 |
| Q | nested | 3.195507 / 1 | 3.262403 / 1 | 3.284904 / 1 |
| K | nested | 2.413768 / 1 | 2.419178 / 1 | 2.667432 / 1 |
| V | nested | 2.294866 / 1 | 2.329530 / 1 | 2.680746 / 1 |
| B | nested | 0.028543 / 1 | 0.029305 / 1 | 0.028854 / 1 |
| FA | nested | 0.027902 / 1 | 0.027171 / 1 | 0.029936 / 1 |
| FB | nested | 0.051406 / 1 | 0.051185 / 1 | 0.056285 / 1 |
| G | nested | 2.088229 / 1 | 2.080456 / 1 | 2.371338 / 1 |
| O | nested | 2.084844 / 1 | 2.079213 / 1 | 2.421422 / 1 |
| attention | boundary | 13.330465 / 1 | 13.450629 / 1 | 14.739927 / 1 |
| attention-residual | boundary | 0.003867 / 1 | 0.004098 / 1 | 0.003797 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024747 / 1 | 0.025107 / 1 | 0.023684 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001784 / 1 | 0.001833 / 1 | 0.001944 / 1 |
| router-and-top16 | boundary | 0.587777 / 1 | 0.577468 / 1 | 0.583680 / 1 |
| EDOWN | nested | 0.599941 / 1 | 0.599009 / 1 | 0.679920 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.600712 / 1 | 0.599790 / 1 | 0.680692 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.058219 / 1 | 0.018324 / 1 | 0.021229 / 1 |
| SH1 | nested | 1.029733 / 1 | 1.394624 / 1 | 1.222333 / 1 |
| SH3 | nested | 1.046034 / 1 | 1.411386 / 1 | 1.233444 / 1 |
| SH2 | nested | 1.049430 / 1 | 1.410154 / 1 | 1.254433 / 1 |
| shared-expert-during-read | boundary | 3.136207 / 1 | 4.231493 / 1 | 3.730206 / 1 |
| detail:expert-gate | nested | 49.462851 / 16 | 50.353570 / 16 | 54.879078 / 16 |
| detail:expert-up | nested | 42.791149 / 16 | 43.796502 / 16 | 54.553305 / 16 |
| detail:expert-activation | nested | 0.096119 / 16 | 0.100208 / 16 | 0.130883 / 16 |
| detail:expert-down | nested | 48.936478 / 16 | 50.068373 / 16 | 51.630942 / 16 |
| EUP | nested | 0.597806 / 1 | 0.602555 / 1 | 0.727979 / 1 |
| experts-mix-normalize-up | boundary | 142.301477 / 1 | 145.299948 / 1 | 162.286315 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003597 / 1 | 0.003115 / 1 | 0.002644 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.584562 / 1 | 0.588159 / 1 | 0.616311 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.958070 / 1 | 0.957108 / 1 | 0.986373 / 1 |
| worker:read | parallel worker | 0.070172 / 1 | 0.068470 / 1 | 0.070564 / 1 |
| worker:decode | parallel worker | 1071.641491 / 1 | 1082.379662 / 1 | 1095.398315 / 1 |
| worker:crc | parallel worker | 595.412928 / 1 | 601.215413 / 1 | 608.190184 / 1 |
| worker:math | parallel worker | 362.511332 / 1 | 363.180331 / 1 | 381.966315 / 1 |
| op:Q   int8 projection | nested | 16.506375 / 1 | 17.694705 / 1 | 18.657615 / 1 |
| op:X   mxfp4 expert proj | nested | 141.326013 / 1 | 144.359195 / 1 | 161.245631 / 1 |
| op:N   rmsnorm | nested | 0.044772 / 1 | 0.044544 / 1 | 0.031128 / 1 |
| op:L   l2 per-head | nested | 0.005700 / 1 | 0.005389 / 1 | 0.005721 / 1 |
| op:SiTU + sigma | nested | 0.103291 / 1 | 0.111057 / 1 | 0.146572 / 1 |
| op:C   shortconv | nested | 0.061315 / 1 | 0.061523 / 1 | 0.030937 / 1 |
| op:AR  snapshot aggregate | nested | 0.031459 / 1 | 0.032019 / 1 | 0.031399 / 1 |
| op:D   kda delta-rule | nested | 0.232615 / 1 | 0.245980 / 1 | 0.250758 / 1 |
| op:router dot product | nested | 0.585103 / 1 | 0.574693 / 1 | 0.580574 / 1 |
| op:top-k selection | nested | 0.002304 / 1 | 0.002354 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.073748 / 1 | 0.072485 / 1 | 0.058539 / 1 |
| detail:read-ahead-wait | nested | 0.001612 / 1 | 0.001724 / 1 | 0.001683 / 1 |
| total:layer | total | 161.050540 / 1 | 165.212295 / 1 | 183.102569 / 1 |

### Layer 27

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004679 / 1 | 0.005210 / 1 | 0.004428 / 1 |
| pre-attention-aggregation | boundary | 0.016271 / 1 | 0.016761 / 1 | 0.016321 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011451 / 1 | 0.011792 / 1 | 0.011040 / 1 |
| QA | nested | 0.346146 / 1 | 0.344784 / 1 | 0.365182 / 1 |
| QB | nested | 0.919627 / 1 | 0.903718 / 1 | 0.947540 / 1 |
| KA | nested | 0.133910 / 1 | 0.132337 / 1 | 0.140993 / 1 |
| KB | nested | 0.422829 / 1 | 0.406950 / 1 | 0.421667 / 1 |
| G | nested | 2.821529 / 1 | 2.800420 / 1 | 2.747611 / 1 |
| O | nested | 2.110661 / 1 | 2.086016 / 1 | 2.315825 / 1 |
| attention | boundary | 6.862746 / 1 | 6.787576 / 1 | 7.051519 / 1 |
| attention-residual | boundary | 0.002715 / 1 | 0.002735 / 1 | 0.002996 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025127 / 1 | 0.024145 / 1 | 0.024696 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001663 / 1 | 0.002064 / 1 | 0.001843 / 1 |
| router-and-top16 | boundary | 0.502077 / 1 | 0.527575 / 1 | 0.541642 / 1 |
| EDOWN | nested | 0.609818 / 1 | 0.611683 / 1 | 0.680340 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.610501 / 1 | 0.612444 / 1 | 0.681072 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.077966 / 1 | 0.021530 / 1 | 0.014487 / 1 |
| SH1 | nested | 1.429500 / 1 | 1.034913 / 1 | 1.160317 / 1 |
| SH3 | nested | 1.505060 / 1 | 1.042828 / 1 | 1.160898 / 1 |
| SH2 | nested | 1.015457 / 1 | 1.036035 / 1 | 1.172610 / 1 |
| shared-expert-during-read | boundary | 3.961388 / 1 | 3.124515 / 1 | 3.507420 / 1 |
| detail:expert-gate | nested | 50.635042 / 16 | 49.650943 / 16 | 54.830739 / 16 |
| detail:expert-up | nested | 45.892656 / 16 | 42.774691 / 16 | 54.648917 / 16 |
| detail:expert-activation | nested | 0.105177 / 16 | 0.094277 / 16 | 0.129381 / 16 |
| detail:expert-down | nested | 52.149569 / 16 | 49.086008 / 16 | 51.603636 / 16 |
| EUP | nested | 0.602615 / 1 | 0.615339 / 1 | 0.735724 / 1 |
| experts-mix-normalize-up | boundary | 149.810249 / 1 | 142.600056 / 1 | 162.379018 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003236 / 1 | 0.002795 / 1 | 0.003306 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.607625 / 1 | 0.594670 / 1 | 0.618956 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.613466 / 1 | 0.600241 / 1 | 0.624797 / 1 |
| worker:read | parallel worker | 0.199772 / 1 | 0.185710 / 1 | 0.198901 / 1 |
| worker:decode | parallel worker | 1078.288788 / 1 | 1081.589495 / 1 | 1094.826069 / 1 |
| worker:crc | parallel worker | 598.357872 / 1 | 600.408041 / 1 | 608.776851 / 1 |
| worker:math | parallel worker | 366.714990 / 1 | 361.720531 / 1 | 380.687978 / 1 |
| op:Q   int8 projection | nested | 11.915621 / 1 | 11.013867 / 1 | 11.847476 / 1 |
| op:X   mxfp4 expert proj | nested | 148.823667 / 1 | 141.646196 / 1 | 161.264228 / 1 |
| op:N   rmsnorm | nested | 0.032571 / 1 | 0.031259 / 1 | 0.032020 / 1 |
| op:SiTU + sigma | nested | 0.112462 / 1 | 0.101499 / 1 | 0.138857 / 1 |
| op:AR  snapshot aggregate | nested | 0.030878 / 1 | 0.030868 / 1 | 0.030537 / 1 |
| op:SA  softmax attention | nested | 0.006282 / 1 | 0.009087 / 1 | 0.012914 / 1 |
| op:router dot product | nested | 0.499263 / 1 | 0.524720 / 1 | 0.538766 / 1 |
| op:top-k selection | nested | 0.002434 / 1 | 0.002495 / 1 | 0.002524 / 1 |
| detail:read-ahead-wait | nested | 0.001552 / 1 | 0.001631 / 1 | 0.001564 / 1 |
| total:layer | total | 162.512110 / 1 | 154.347915 / 1 | 174.872991 / 1 |

### Layer 28

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004739 / 1 | 0.004478 / 1 | 0.004428 / 1 |
| pre-attention-aggregation | boundary | 0.015128 / 1 | 0.015319 / 1 | 0.015348 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011681 / 1 | 0.011401 / 1 | 0.011912 / 1 |
| Q | nested | 3.247024 / 1 | 3.351940 / 1 | 3.377928 / 1 |
| K | nested | 2.418096 / 1 | 2.399260 / 1 | 2.710462 / 1 |
| V | nested | 2.332276 / 1 | 2.319151 / 1 | 2.705162 / 1 |
| B | nested | 0.028223 / 1 | 0.025387 / 1 | 0.024105 / 1 |
| FA | nested | 0.026529 / 1 | 0.027722 / 1 | 0.031549 / 1 |
| FB | nested | 0.048771 / 1 | 0.051035 / 1 | 0.052769 / 1 |
| G | nested | 2.104661 / 1 | 2.082359 / 1 | 2.386517 / 1 |
| O | nested | 2.164643 / 1 | 2.087248 / 1 | 2.436851 / 1 |
| attention | boundary | 13.458673 / 1 | 13.393232 / 1 | 14.777797 / 1 |
| attention-residual | boundary | 0.003757 / 1 | 0.003637 / 1 | 0.003587 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024846 / 1 | 0.023835 / 1 | 0.024416 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002374 / 1 | 0.001713 / 1 | 0.001613 / 1 |
| router-and-top16 | boundary | 0.589201 / 1 | 0.573521 / 1 | 0.588098 / 1 |
| EDOWN | nested | 0.622392 / 1 | 0.601423 / 1 | 0.727018 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.623224 / 1 | 0.602255 / 1 | 0.727809 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.054852 / 1 | 0.011652 / 1 | 0.012013 / 1 |
| SH1 | nested | 1.455839 / 1 | 1.021458 / 1 | 1.235798 / 1 |
| SH3 | nested | 1.486727 / 1 | 1.013994 / 1 | 1.243422 / 1 |
| SH2 | nested | 1.421685 / 1 | 1.012250 / 1 | 1.253050 / 1 |
| shared-expert-during-read | boundary | 4.380060 / 1 | 3.058993 / 1 | 3.745695 / 1 |
| detail:expert-gate | nested | 48.979157 / 16 | 49.745198 / 16 | 53.766503 / 16 |
| detail:expert-up | nested | 42.799827 / 16 | 42.876074 / 16 | 53.473350 / 16 |
| detail:expert-activation | nested | 0.098386 / 16 | 0.095868 / 16 | 0.132237 / 16 |
| detail:expert-down | nested | 49.050593 / 16 | 49.155999 / 16 | 53.758438 / 16 |
| EUP | nested | 0.602606 / 1 | 0.592727 / 1 | 0.598287 / 1 |
| experts-mix-normalize-up | boundary | 141.943778 / 1 | 142.836567 / 1 | 162.103694 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002835 / 1 | 0.003076 / 1 | 0.002785 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.593688 / 1 | 0.589621 / 1 | 0.598838 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.985841 / 1 | 0.957168 / 1 | 0.989829 / 1 |
| worker:read | parallel worker | 0.067152 / 1 | 0.067643 / 1 | 0.067348 / 1 |
| worker:decode | parallel worker | 1072.972076 / 1 | 1082.110820 / 1 | 1090.831646 / 1 |
| worker:crc | parallel worker | 594.980134 / 1 | 600.442008 / 1 | 605.300761 / 1 |
| worker:math | parallel worker | 363.137113 / 1 | 362.760136 / 1 | 374.577761 / 1 |
| op:Q   int8 projection | nested | 17.958051 / 1 | 16.584424 / 1 | 18.781486 / 1 |
| op:X   mxfp4 expert proj | nested | 140.967615 / 1 | 141.913724 / 1 | 161.185221 / 1 |
| op:N   rmsnorm | nested | 0.038132 / 1 | 0.038992 / 1 | 0.031559 / 1 |
| op:L   l2 per-head | nested | 0.005309 / 1 | 0.005300 / 1 | 0.005170 / 1 |
| op:SiTU + sigma | nested | 0.109262 / 1 | 0.103364 / 1 | 0.141255 / 1 |
| op:C   shortconv | nested | 0.068417 / 1 | 0.054822 / 1 | 0.035315 / 1 |
| op:AR  snapshot aggregate | nested | 0.030236 / 1 | 0.029876 / 1 | 0.029455 / 1 |
| op:D   kda delta-rule | nested | 0.111940 / 1 | 0.111609 / 1 | 0.128530 / 1 |
| op:router dot product | nested | 0.586375 / 1 | 0.570566 / 1 | 0.585614 / 1 |
| op:top-k selection | nested | 0.002474 / 1 | 0.002564 / 1 | 0.002124 / 1 |
| op:alpha / beta / gate | nested | 0.072515 / 1 | 0.081592 / 1 | 0.057467 / 1 |
| detail:read-ahead-wait | nested | 0.001172 / 1 | 0.001384 / 1 | 0.001202 / 1 |
| total:layer | total | 162.113145 / 1 | 161.505722 / 1 | 183.018032 / 1 |

### Layer 29

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005070 / 1 | 0.005140 / 1 | 0.005921 / 1 |
| pre-attention-aggregation | boundary | 0.016290 / 1 | 0.016911 / 1 | 0.016601 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011612 / 1 | 0.011672 / 1 | 0.012102 / 1 |
| Q | nested | 3.311985 / 1 | 3.126248 / 1 | 3.450083 / 1 |
| K | nested | 2.393951 / 1 | 2.390534 / 1 | 2.893293 / 1 |
| V | nested | 2.290708 / 1 | 2.312028 / 1 | 2.800009 / 1 |
| B | nested | 0.028263 / 1 | 0.025728 / 1 | 0.031008 / 1 |
| FA | nested | 0.026850 / 1 | 0.031740 / 1 | 0.038382 / 1 |
| FB | nested | 0.049643 / 1 | 0.050814 / 1 | 0.062026 / 1 |
| G | nested | 2.099461 / 1 | 2.056852 / 1 | 2.544723 / 1 |
| O | nested | 2.088240 / 1 | 2.082699 / 1 | 2.060568 / 1 |
| attention | boundary | 13.420463 / 1 | 13.227853 / 1 | 15.035399 / 1 |
| attention-residual | boundary | 0.004098 / 1 | 0.003888 / 1 | 0.003787 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025077 / 1 | 0.024556 / 1 | 0.024155 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001904 / 1 | 0.002073 / 1 | 0.001683 / 1 |
| router-and-top16 | boundary | 0.582748 / 1 | 0.578100 / 1 | 0.576436 / 1 |
| EDOWN | nested | 0.606713 / 1 | 0.593028 / 1 | 0.598278 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.607524 / 1 | 0.593869 / 1 | 0.599089 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.074540 / 1 | 0.011602 / 1 | 0.016250 / 1 |
| SH1 | nested | 1.429690 / 1 | 1.014174 / 1 | 1.386048 / 1 |
| SH3 | nested | 1.446321 / 1 | 1.011359 / 1 | 1.399784 / 1 |
| SH2 | nested | 1.411807 / 1 | 1.009576 / 1 | 1.406216 / 1 |
| shared-expert-during-read | boundary | 4.302765 / 1 | 3.046158 / 1 | 4.207707 / 1 |
| detail:expert-gate | nested | 49.640013 / 16 | 49.754114 / 16 | 54.366350 / 16 |
| detail:expert-up | nested | 43.761332 / 16 | 42.878227 / 16 | 52.176371 / 16 |
| detail:expert-activation | nested | 0.097355 / 16 | 0.095759 / 16 | 0.126958 / 16 |
| detail:expert-down | nested | 49.482308 / 16 | 49.178732 / 16 | 52.656267 / 16 |
| EUP | nested | 0.603076 / 1 | 0.597656 / 1 | 0.735023 / 1 |
| experts-mix-normalize-up | boundary | 144.017651 / 1 | 142.873958 / 1 | 160.435950 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003096 / 1 | 0.002745 / 1 | 0.002715 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.598207 / 1 | 0.585975 / 1 | 0.615309 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.985921 / 1 | 0.949984 / 1 | 0.986141 / 1 |
| worker:read | parallel worker | 0.069009 / 1 | 0.067576 / 1 | 0.069390 / 1 |
| worker:decode | parallel worker | 1075.444020 / 1 | 1081.189899 / 1 | 1087.947656 / 1 |
| worker:crc | parallel worker | 596.972441 / 1 | 601.227412 / 1 | 605.426453 / 1 |
| worker:math | parallel worker | 362.052428 / 1 | 363.313191 / 1 | 374.033625 / 1 |
| op:Q   int8 projection | nested | 17.785136 / 1 | 16.300841 / 1 | 19.403517 / 1 |
| op:X   mxfp4 expert proj | nested | 143.022985 / 1 | 141.947879 / 1 | 159.379925 / 1 |
| op:N   rmsnorm | nested | 0.039794 / 1 | 0.039894 / 1 | 0.031558 / 1 |
| op:L   l2 per-head | nested | 0.005550 / 1 | 0.005381 / 1 | 0.005871 / 1 |
| op:SiTU + sigma | nested | 0.107743 / 1 | 0.103034 / 1 | 0.137054 / 1 |
| op:C   shortconv | nested | 0.054382 / 1 | 0.059671 / 1 | 0.032271 / 1 |
| op:AR  snapshot aggregate | nested | 0.030937 / 1 | 0.031539 / 1 | 0.031189 / 1 |
| op:D   kda delta-rule | nested | 0.235079 / 1 | 0.243184 / 1 | 0.241982 / 1 |
| op:router dot product | nested | 0.580153 / 1 | 0.575514 / 1 | 0.573551 / 1 |
| op:top-k selection | nested | 0.002194 / 1 | 0.002244 / 1 | 0.002545 / 1 |
| op:alpha / beta / gate | nested | 0.072496 / 1 | 0.072846 / 1 | 0.058480 / 1 |
| detail:read-ahead-wait | nested | 0.001622 / 1 | 0.001503 / 1 | 0.001392 / 1 |
| total:layer | total | 164.068717 / 1 | 161.357706 / 1 | 181.934128 / 1 |

### Layer 30

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005130 / 1 | 0.005450 / 1 | 0.004979 / 1 |
| pre-attention-aggregation | boundary | 0.017313 / 1 | 0.017052 / 1 | 0.016801 / 1 |
| snapshot-push | boundary | 0.000100 / 1 | 0.000110 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012173 / 1 | 0.012393 / 1 | 0.011671 / 1 |
| Q | nested | 3.239710 / 1 | 3.278532 / 1 | 3.280777 / 1 |
| K | nested | 2.395734 / 1 | 2.380606 / 1 | 2.753473 / 1 |
| V | nested | 2.296839 / 1 | 2.298062 / 1 | 2.685164 / 1 |
| B | nested | 0.024125 / 1 | 0.026219 / 1 | 0.027732 / 1 |
| FA | nested | 0.027432 / 1 | 0.036879 / 1 | 0.028844 / 1 |
| FB | nested | 0.049592 / 1 | 0.047559 / 1 | 0.053029 / 1 |
| G | nested | 2.101755 / 1 | 2.068583 / 1 | 2.398930 / 1 |
| O | nested | 2.096255 / 1 | 2.100032 / 1 | 2.433725 / 1 |
| attention | boundary | 13.377923 / 1 | 13.401487 / 1 | 14.861053 / 1 |
| attention-residual | boundary | 0.003968 / 1 | 0.003507 / 1 | 0.003887 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023825 / 1 | 0.024816 / 1 | 0.025227 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001763 / 1 | 0.001633 / 1 | 0.001784 / 1 |
| router-and-top16 | boundary | 0.582989 / 1 | 0.581486 / 1 | 0.591615 / 1 |
| EDOWN | nested | 0.608506 / 1 | 0.589471 / 1 | 0.693004 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609418 / 1 | 0.590283 / 1 | 0.693956 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.073227 / 1 | 0.006211 / 1 | 0.021129 / 1 |
| SH1 | nested | 1.032528 / 1 | 1.006149 / 1 | 1.180835 / 1 |
| SH3 | nested | 1.031407 / 1 | 1.012571 / 1 | 1.184672 / 1 |
| SH2 | nested | 1.029243 / 1 | 0.999657 / 1 | 1.182178 / 1 |
| shared-expert-during-read | boundary | 3.104427 / 1 | 3.032453 / 1 | 3.561141 / 1 |
| detail:expert-gate | nested | 49.646192 / 16 | 49.974971 / 16 | 54.340425 / 16 |
| detail:expert-up | nested | 42.734284 / 16 | 42.844196 / 16 | 54.555004 / 16 |
| detail:expert-activation | nested | 0.096328 / 16 | 0.098746 / 16 | 0.133650 / 16 |
| detail:expert-down | nested | 49.012022 / 16 | 49.125841 / 16 | 54.910706 / 16 |
| EUP | nested | 0.595913 / 1 | 0.601003 / 1 | 0.603407 / 1 |
| experts-mix-normalize-up | boundary | 142.523441 / 1 | 143.013728 / 1 | 164.945681 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003026 / 1 | 0.003447 / 1 | 0.002715 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.583239 / 1 | 0.587858 / 1 | 0.595823 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.928354 / 1 | 0.947780 / 1 | 0.979229 / 1 |
| worker:read | parallel worker | 0.207888 / 1 | 0.195055 / 1 | 0.190720 / 1 |
| worker:decode | parallel worker | 1073.261357 / 1 | 1081.677322 / 1 | 1087.484579 / 1 |
| worker:crc | parallel worker | 595.393292 / 1 | 600.218202 / 1 | 604.858193 / 1 |
| worker:math | parallel worker | 362.318667 / 1 | 362.524395 / 1 | 372.629655 / 1 |
| op:Q   int8 projection | nested | 16.527233 / 1 | 16.443910 / 1 | 18.504336 / 1 |
| op:X   mxfp4 expert proj | nested | 141.532648 / 1 | 142.087911 / 1 | 163.999688 / 1 |
| op:N   rmsnorm | nested | 0.043542 / 1 | 0.043933 / 1 | 0.031988 / 1 |
| op:L   l2 per-head | nested | 0.005350 / 1 | 0.006170 / 1 | 0.005389 / 1 |
| op:SiTU + sigma | nested | 0.103733 / 1 | 0.108625 / 1 | 0.142769 / 1 |
| op:C   shortconv | nested | 0.061365 / 1 | 0.061295 / 1 | 0.032240 / 1 |
| op:AR  snapshot aggregate | nested | 0.031378 / 1 | 0.031719 / 1 | 0.031830 / 1 |
| op:D   kda delta-rule | nested | 0.234728 / 1 | 0.239898 / 1 | 0.253714 / 1 |
| op:router dot product | nested | 0.580064 / 1 | 0.578310 / 1 | 0.588589 / 1 |
| op:top-k selection | nested | 0.002534 / 1 | 0.002795 / 1 | 0.002665 / 1 |
| op:alpha / beta / gate | nested | 0.070501 / 1 | 0.072696 / 1 | 0.058259 / 1 |
| detail:read-ahead-wait | nested | 0.001967 / 1 | 0.001934 / 1 | 0.001542 / 1 |
| total:layer | total | 161.280600 / 1 | 161.651664 / 1 | 185.730798 / 1 |

### Layer 31

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004519 / 1 | 0.005119 / 1 | 0.005309 / 1 |
| pre-attention-aggregation | boundary | 0.015790 / 1 | 0.021500 / 1 | 0.016501 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000101 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011832 / 1 | 0.011061 / 1 | 0.012082 / 1 |
| QA | nested | 0.310700 / 1 | 0.343231 / 1 | 0.357668 / 1 |
| QB | nested | 0.804392 / 1 | 0.927031 / 1 | 0.953530 / 1 |
| KA | nested | 0.122058 / 1 | 0.131997 / 1 | 0.134692 / 1 |
| KB | nested | 0.349863 / 1 | 0.424412 / 1 | 0.416809 / 1 |
| G | nested | 2.539313 / 1 | 2.605335 / 1 | 2.745577 / 1 |
| O | nested | 2.601047 / 1 | 2.082730 / 1 | 2.031714 / 1 |
| attention | boundary | 6.843991 / 1 | 6.632056 / 1 | 6.750227 / 1 |
| attention-residual | boundary | 0.003837 / 1 | 0.002615 / 1 | 0.002995 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.027191 / 1 | 0.029134 / 1 | 0.023905 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002575 / 1 | 0.001954 / 1 | 0.001813 / 1 |
| router-and-top16 | boundary | 0.547753 / 1 | 0.526954 / 1 | 0.528717 / 1 |
| EDOWN | nested | 0.694346 / 1 | 0.608005 / 1 | 0.597906 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.695098 / 1 | 0.608807 / 1 | 0.598698 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.066223 / 1 | 0.023153 / 1 | 0.010921 / 1 |
| SH1 | nested | 1.049651 / 1 | 1.433648 / 1 | 1.020746 / 1 |
| SH3 | nested | 1.056603 / 1 | 1.452283 / 1 | 1.018351 / 1 |
| SH2 | nested | 1.253581 / 1 | 1.356583 / 1 | 1.013853 / 1 |
| shared-expert-during-read | boundary | 3.370715 / 1 | 4.258473 / 1 | 3.063872 / 1 |
| detail:expert-gate | nested | 49.632590 / 16 | 50.151031 / 16 | 54.031916 / 16 |
| detail:expert-up | nested | 42.923801 / 16 | 42.938154 / 16 | 52.668803 / 16 |
| detail:expert-activation | nested | 0.096459 / 16 | 0.095388 / 16 | 0.130996 / 16 |
| detail:expert-down | nested | 49.165728 / 16 | 49.254437 / 16 | 51.739143 / 16 |
| EUP | nested | 0.616321 / 1 | 0.605721 / 1 | 0.603908 / 1 |
| experts-mix-normalize-up | boundary | 142.856854 / 1 | 143.444813 / 1 | 159.561486 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003136 / 1 | 0.002976 / 1 | 0.002875 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.594310 / 1 | 0.599630 / 1 | 0.595011 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.599780 / 1 | 0.604930 / 1 | 0.600762 / 1 |
| worker:read | parallel worker | 0.068828 / 1 | 0.066922 / 1 | 0.068414 / 1 |
| worker:decode | parallel worker | 1074.503188 / 1 | 1081.939733 / 1 | 1084.387743 / 1 |
| worker:crc | parallel worker | 595.508929 / 1 | 600.061610 / 1 | 603.301985 / 1 |
| worker:math | parallel worker | 365.081046 / 1 | 365.631342 / 1 | 372.749160 / 1 |
| op:Q   int8 projection | nested | 11.396661 / 1 | 11.969584 / 1 | 10.893503 / 1 |
| op:X   mxfp4 expert proj | nested | 141.860844 / 1 | 142.481788 / 1 | 158.628903 / 1 |
| op:N   rmsnorm | nested | 0.031778 / 1 | 0.031860 / 1 | 0.031289 / 1 |
| op:SiTU + sigma | nested | 0.103632 / 1 | 0.107063 / 1 | 0.137416 / 1 |
| op:AR  snapshot aggregate | nested | 0.031639 / 1 | 0.040305 / 1 | 0.030286 / 1 |
| op:SA  softmax attention | nested | 0.006973 / 1 | 0.008676 / 1 | 0.012483 / 1 |
| op:router dot product | nested | 0.544607 / 1 | 0.524209 / 1 | 0.526133 / 1 |
| op:top-k selection | nested | 0.002705 / 1 | 0.002405 / 1 | 0.002214 / 1 |
| detail:read-ahead-wait | nested | 0.001576 / 1 | 0.001532 / 1 | 0.001323 / 1 |
| total:layer | total | 155.059462 / 1 | 156.183252 / 1 | 171.189271 / 1 |

### Layer 32

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004869 / 1 | 0.004488 / 1 | 0.004989 / 1 |
| pre-attention-aggregation | boundary | 0.015519 / 1 | 0.015690 / 1 | 0.015168 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011351 / 1 | 0.012123 / 1 | 0.011231 / 1 |
| Q | nested | 3.375654 / 1 | 3.333305 / 1 | 3.316183 / 1 |
| K | nested | 2.371369 / 1 | 2.360448 / 1 | 2.326084 / 1 |
| V | nested | 2.294896 / 1 | 2.285889 / 1 | 2.288584 / 1 |
| B | nested | 0.026359 / 1 | 0.026129 / 1 | 0.023885 / 1 |
| FA | nested | 0.034074 / 1 | 0.030657 / 1 | 0.030176 / 1 |
| FB | nested | 0.049783 / 1 | 0.054963 / 1 | 0.049262 / 1 |
| G | nested | 2.074254 / 1 | 2.549501 / 1 | 2.049708 / 1 |
| O | nested | 2.092848 / 1 | 2.318901 / 1 | 2.072750 / 1 |
| attention | boundary | 13.352726 / 1 | 14.029060 / 1 | 13.202997 / 1 |
| attention-residual | boundary | 0.003898 / 1 | 0.003586 / 1 | 0.003606 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023814 / 1 | 0.023864 / 1 | 0.024085 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001884 / 1 | 0.001633 / 1 | 0.001833 / 1 |
| router-and-top16 | boundary | 0.585764 / 1 | 0.571127 / 1 | 0.568893 / 1 |
| EDOWN | nested | 0.601984 / 1 | 0.595162 / 1 | 0.591344 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.602846 / 1 | 0.596033 / 1 | 0.592246 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.064550 / 1 | 0.026710 / 1 | 0.008917 / 1 |
| SH1 | nested | 1.461399 / 1 | 1.009655 / 1 | 1.003594 / 1 |
| SH3 | nested | 1.608985 / 1 | 1.032899 / 1 | 1.008202 / 1 |
| SH2 | nested | 1.059409 / 1 | 1.008333 / 1 | 1.001951 / 1 |
| shared-expert-during-read | boundary | 4.141734 / 1 | 3.061888 / 1 | 3.024909 / 1 |
| detail:expert-gate | nested | 48.383025 / 16 | 50.100274 / 16 | 53.495065 / 16 |
| detail:expert-up | nested | 42.886991 / 16 | 42.923821 / 16 | 52.218200 / 16 |
| detail:expert-activation | nested | 0.094609 / 16 | 0.096990 / 16 | 0.126106 / 16 |
| detail:expert-down | nested | 49.240656 / 16 | 49.211532 / 16 | 52.789185 / 16 |
| EUP | nested | 0.603246 / 1 | 0.597626 / 1 | 0.605060 / 1 |
| experts-mix-normalize-up | boundary | 141.642185 / 1 | 143.324369 / 1 | 159.586442 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002865 / 1 | 0.002885 / 1 | 0.002445 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590212 / 1 | 0.589260 / 1 | 0.599770 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 1.002553 / 1 | 0.968148 / 1 | 1.016859 / 1 |
| worker:read | parallel worker | 0.068027 / 1 | 0.068133 / 1 | 0.066758 / 1 |
| worker:decode | parallel worker | 1074.288187 / 1 | 1080.662858 / 1 | 1088.507510 / 1 |
| worker:crc | parallel worker | 595.039851 / 1 | 600.980355 / 1 | 605.404977 / 1 |
| worker:math | parallel worker | 364.841669 / 1 | 364.729027 / 1 | 375.847307 / 1 |
| op:Q   int8 projection | nested | 17.652535 / 1 | 17.202015 / 1 | 16.365351 / 1 |
| op:X   mxfp4 expert proj | nested | 140.650213 / 1 | 142.377744 / 1 | 158.686270 / 1 |
| op:N   rmsnorm | nested | 0.038722 / 1 | 0.039173 / 1 | 0.031368 / 1 |
| op:L   l2 per-head | nested | 0.004919 / 1 | 0.006092 / 1 | 0.005811 / 1 |
| op:SiTU + sigma | nested | 0.101812 / 1 | 0.103694 / 1 | 0.132697 / 1 |
| op:C   shortconv | nested | 0.055825 / 1 | 0.073718 / 1 | 0.033842 / 1 |
| op:AR  snapshot aggregate | nested | 0.029605 / 1 | 0.029936 / 1 | 0.029676 / 1 |
| op:D   kda delta-rule | nested | 0.105077 / 1 | 0.120675 / 1 | 0.124392 / 1 |
| op:router dot product | nested | 0.582418 / 1 | 0.568101 / 1 | 0.566207 / 1 |
| op:top-k selection | nested | 0.002555 / 1 | 0.002635 / 1 | 0.002345 / 1 |
| op:alpha / beta / gate | nested | 0.072656 / 1 | 0.074379 / 1 | 0.057748 / 1 |
| detail:read-ahead-wait | nested | 0.001431 / 1 | 0.001502 / 1 | 0.001354 / 1 |
| total:layer | total | 161.466998 / 1 | 162.651743 / 1 | 178.075061 / 1 |

### Layer 33

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004799 / 1 | 0.004278 / 1 | 0.006422 / 1 |
| pre-attention-aggregation | boundary | 0.017162 / 1 | 0.015899 / 1 | 0.016881 / 1 |
| snapshot-push | boundary | 0.000130 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011562 / 1 | 0.011511 / 1 | 0.011902 / 1 |
| Q | nested | 2.928078 / 1 | 3.274415 / 1 | 3.400330 / 1 |
| K | nested | 2.404670 / 1 | 2.433765 / 1 | 2.923410 / 1 |
| V | nested | 2.291759 / 1 | 2.316757 / 1 | 2.766706 / 1 |
| B | nested | 0.023694 / 1 | 0.024726 / 1 | 0.029095 / 1 |
| FA | nested | 0.027451 / 1 | 0.028833 / 1 | 0.041287 / 1 |
| FB | nested | 0.052718 / 1 | 0.054221 / 1 | 0.071473 / 1 |
| G | nested | 2.102817 / 1 | 2.087619 / 1 | 2.942185 / 1 |
| O | nested | 2.098860 / 1 | 2.089863 / 1 | 2.283394 / 1 |
| attention | boundary | 13.068234 / 1 | 13.494300 / 1 | 15.646270 / 1 |
| attention-residual | boundary | 0.003777 / 1 | 0.003757 / 1 | 0.003637 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024246 / 1 | 0.024035 / 1 | 0.024616 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001953 / 1 | 0.001743 / 1 | 0.001993 / 1 |
| router-and-top16 | boundary | 0.587517 / 1 | 0.578390 / 1 | 0.600872 / 1 |
| EDOWN | nested | 0.601604 / 1 | 0.593187 / 1 | 0.657618 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.602595 / 1 | 0.594069 / 1 | 0.658489 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.063809 / 1 | 0.012113 / 1 | 0.019096 / 1 |
| SH1 | nested | 1.038089 / 1 | 1.018892 / 1 | 1.120483 / 1 |
| SH3 | nested | 1.040573 / 1 | 1.010878 / 1 | 1.097770 / 1 |
| SH2 | nested | 1.028971 / 1 | 1.013723 / 1 | 1.113559 / 1 |
| shared-expert-during-read | boundary | 3.119185 / 1 | 3.054935 / 1 | 3.345478 / 1 |
| detail:expert-gate | nested | 49.240458 / 16 | 49.766165 / 16 | 54.499191 / 16 |
| detail:expert-up | nested | 43.653605 / 16 | 43.738781 / 16 | 54.469525 / 16 |
| detail:expert-activation | nested | 0.099255 / 16 | 0.100506 / 16 | 0.132359 / 16 |
| detail:expert-down | nested | 49.417477 / 16 | 49.542932 / 16 | 51.342924 / 16 |
| EUP | nested | 0.601323 / 1 | 0.598628 / 1 | 0.637530 / 1 |
| experts-mix-normalize-up | boundary | 143.448408 / 1 | 144.111027 / 1 | 161.459310 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003226 / 1 | 0.003185 / 1 | 0.002385 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.597336 / 1 | 0.594249 / 1 | 0.618546 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.985751 / 1 | 0.978507 / 1 | 1.003644 / 1 |
| worker:read | parallel worker | 0.067553 / 1 | 0.067261 / 1 | 0.068590 / 1 |
| worker:decode | parallel worker | 1074.665132 / 1 | 1081.685641 / 1 | 1092.521890 / 1 |
| worker:crc | parallel worker | 595.910258 / 1 | 600.464353 / 1 | 607.300094 / 1 |
| worker:math | parallel worker | 363.112219 / 1 | 363.270696 / 1 | 375.474160 / 1 |
| op:Q   int8 projection | nested | 16.239184 / 1 | 16.543854 / 1 | 19.083337 / 1 |
| op:X   mxfp4 expert proj | nested | 142.456378 / 1 | 143.194453 / 1 | 160.500641 / 1 |
| op:N   rmsnorm | nested | 0.037671 / 1 | 0.040766 / 1 | 0.031801 / 1 |
| op:L   l2 per-head | nested | 0.005771 / 1 | 0.006091 / 1 | 0.005771 / 1 |
| op:SiTU + sigma | nested | 0.106460 / 1 | 0.107633 / 1 | 0.138760 / 1 |
| op:C   shortconv | nested | 0.057737 / 1 | 0.054872 / 1 | 0.033112 / 1 |
| op:AR  snapshot aggregate | nested | 0.031859 / 1 | 0.030326 / 1 | 0.031739 / 1 |
| op:D   kda delta-rule | nested | 0.231502 / 1 | 0.250368 / 1 | 0.253924 / 1 |
| op:router dot product | nested | 0.584582 / 1 | 0.575615 / 1 | 0.598387 / 1 |
| op:top-k selection | nested | 0.002435 / 1 | 0.002454 / 1 | 0.002104 / 1 |
| op:alpha / beta / gate | nested | 0.072896 / 1 | 0.074379 / 1 | 0.057948 / 1 |
| detail:read-ahead-wait | nested | 0.001795 / 1 | 0.001823 / 1 | 0.001475 / 1 |
| total:layer | total | 161.955570 / 1 | 162.897953 / 1 | 182.812088 / 1 |

### Layer 34

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005520 / 1 | 0.005330 / 1 | 0.005020 / 1 |
| pre-attention-aggregation | boundary | 0.016571 / 1 | 0.016601 / 1 | 0.018635 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011782 / 1 | 0.010910 / 1 | 0.011942 / 1 |
| Q | nested | 3.220935 / 1 | 3.307396 / 1 | 3.614861 / 1 |
| K | nested | 2.381638 / 1 | 2.376999 / 1 | 2.997288 / 1 |
| V | nested | 2.292261 / 1 | 2.304924 / 1 | 2.912599 / 1 |
| B | nested | 0.022422 / 1 | 0.022662 / 1 | 0.025848 / 1 |
| FA | nested | 0.030337 / 1 | 0.030748 / 1 | 0.029946 / 1 |
| FB | nested | 0.050444 / 1 | 0.051406 / 1 | 0.057758 / 1 |
| G | nested | 2.100913 / 1 | 2.057011 / 1 | 2.431230 / 1 |
| O | nested | 2.090935 / 1 | 2.081528 / 1 | 2.432924 / 1 |
| attention | boundary | 13.339060 / 1 | 13.410454 / 1 | 15.582912 / 1 |
| attention-residual | boundary | 0.003757 / 1 | 0.003547 / 1 | 0.003507 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024295 / 1 | 0.023835 / 1 | 0.024816 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002344 / 1 | 0.002264 / 1 | 0.002355 / 1 |
| router-and-top16 | boundary | 0.584601 / 1 | 0.576647 / 1 | 0.602495 / 1 |
| EDOWN | nested | 0.605130 / 1 | 0.597667 / 1 | 0.706108 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.606021 / 1 | 0.598668 / 1 | 0.706991 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031609 / 1 | 0.006242 / 1 | 0.019306 / 1 |
| SH1 | nested | 1.026717 / 1 | 0.995740 / 1 | 1.196364 / 1 |
| SH3 | nested | 1.025846 / 1 | 1.002432 / 1 | 1.179943 / 1 |
| SH2 | nested | 1.016408 / 1 | 1.012521 / 1 | 1.194090 / 1 |
| shared-expert-during-read | boundary | 3.080422 / 1 | 3.022344 / 1 | 3.584143 / 1 |
| detail:expert-gate | nested | 49.566746 / 16 | 50.219786 / 16 | 54.130994 / 16 |
| detail:expert-up | nested | 42.894415 / 16 | 42.966131 / 16 | 53.881033 / 16 |
| detail:expert-activation | nested | 0.095880 / 16 | 0.097133 / 16 | 0.134380 / 16 |
| detail:expert-down | nested | 49.156451 / 16 | 49.206430 / 16 | 54.437546 / 16 |
| EUP | nested | 0.590413 / 1 | 0.599109 / 1 | 0.606432 / 1 |
| experts-mix-normalize-up | boundary | 142.708717 / 1 | 143.467145 / 1 | 163.583597 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003296 / 1 | 0.003076 / 1 | 0.002495 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590462 / 1 | 0.591234 / 1 | 0.595823 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.961696 / 1 | 0.941999 / 1 | 0.957368 / 1 |
| worker:read | parallel worker | 0.066082 / 1 | 0.068762 / 1 | 0.068141 / 1 |
| worker:decode | parallel worker | 1072.158137 / 1 | 1079.535933 / 1 | 1088.568042 / 1 |
| worker:crc | parallel worker | 602.218263 / 1 | 607.465584 / 1 | 613.339619 / 1 |
| worker:math | parallel worker | 362.865120 / 1 | 362.845753 / 1 | 374.026469 / 1 |
| op:Q   int8 projection | nested | 16.452935 / 1 | 16.438659 / 1 | 19.383911 / 1 |
| op:X   mxfp4 expert proj | nested | 141.759446 / 1 | 142.535205 / 1 | 162.643983 / 1 |
| op:N   rmsnorm | nested | 0.047319 / 1 | 0.047969 / 1 | 0.031689 / 1 |
| op:L   l2 per-head | nested | 0.005711 / 1 | 0.005771 / 1 | 0.006061 / 1 |
| op:SiTU + sigma | nested | 0.103245 / 1 | 0.104697 / 1 | 0.143648 / 1 |
| op:C   shortconv | nested | 0.061225 / 1 | 0.056365 / 1 | 0.033863 / 1 |
| op:AR  snapshot aggregate | nested | 0.031187 / 1 | 0.030998 / 1 | 0.033774 / 1 |
| op:D   kda delta-rule | nested | 0.228637 / 1 | 0.244045 / 1 | 0.261458 / 1 |
| op:router dot product | nested | 0.581586 / 1 | 0.574022 / 1 | 0.599390 / 1 |
| op:top-k selection | nested | 0.002355 / 1 | 0.002325 / 1 | 0.002725 / 1 |
| op:alpha / beta / gate | nested | 0.072876 / 1 | 0.075291 / 1 | 0.056987 / 1 |
| detail:read-ahead-wait | nested | 0.001775 / 1 | 0.001773 / 1 | 0.001845 / 1 |
| total:layer | total | 161.390615 / 1 | 162.099932 / 1 | 185.116802 / 1 |

### Layer 35

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005240 / 1 | 0.005390 / 1 | 0.005310 / 1 |
| pre-attention-aggregation | boundary | 0.016821 / 1 | 0.016791 / 1 | 0.016351 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000101 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011872 / 1 | 0.011542 / 1 | 0.011291 / 1 |
| QA | nested | 0.344573 / 1 | 0.362276 / 1 | 0.359912 / 1 |
| QB | nested | 0.942501 / 1 | 0.919527 / 1 | 0.939054 / 1 |
| KA | nested | 0.138609 / 1 | 0.140011 / 1 | 0.137987 / 1 |
| KB | nested | 0.391030 / 1 | 0.413663 / 1 | 0.394617 / 1 |
| G | nested | 2.667311 / 1 | 2.806421 / 1 | 2.783278 / 1 |
| O | nested | 2.121391 / 1 | 2.063844 / 1 | 2.047724 / 1 |
| attention | boundary | 6.720320 / 1 | 6.816761 / 1 | 6.775984 / 1 |
| attention-residual | boundary | 0.002715 / 1 | 0.002775 / 1 | 0.002986 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024515 / 1 | 0.024757 / 1 | 0.024055 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001863 / 1 | 0.001963 / 1 | 0.001994 / 1 |
| router-and-top16 | boundary | 0.526764 / 1 | 0.532425 / 1 | 0.511335 / 1 |
| EDOWN | nested | 0.619878 / 1 | 0.614608 / 1 | 0.600672 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.620770 / 1 | 0.615469 / 1 | 0.601504 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.083716 / 1 | 0.012353 / 1 | 0.020207 / 1 |
| SH1 | nested | 1.455839 / 1 | 1.047045 / 1 | 1.028241 / 1 |
| SH3 | nested | 1.456641 / 1 | 1.057354 / 1 | 1.021978 / 1 |
| SH2 | nested | 1.384676 / 1 | 1.039711 / 1 | 1.028521 / 1 |
| shared-expert-during-read | boundary | 4.313124 / 1 | 3.155292 / 1 | 3.090292 / 1 |
| detail:expert-gate | nested | 49.896060 / 16 | 50.131532 / 16 | 54.331875 / 16 |
| detail:expert-up | nested | 42.697116 / 16 | 43.625149 / 16 | 54.165204 / 16 |
| detail:expert-activation | nested | 0.097832 / 16 | 0.100757 / 16 | 0.135764 / 16 |
| detail:expert-down | nested | 48.957870 / 16 | 49.542382 / 16 | 54.134838 / 16 |
| EUP | nested | 0.606793 / 1 | 0.614929 / 1 | 0.601864 / 1 |
| experts-mix-normalize-up | boundary | 142.709528 / 1 | 144.394978 / 1 | 163.739438 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003066 / 1 | 0.002846 / 1 | 0.003376 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.592567 / 1 | 0.604950 / 1 | 0.596294 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.597977 / 1 | 0.610560 / 1 | 0.602155 / 1 |
| worker:read | parallel worker | 0.065425 / 1 | 0.069530 / 1 | 0.070166 / 1 |
| worker:decode | parallel worker | 1072.646961 / 1 | 1080.247176 / 1 | 1087.927036 / 1 |
| worker:crc | parallel worker | 596.154834 / 1 | 600.886668 / 1 | 606.731619 / 1 |
| worker:math | parallel worker | 362.492937 / 1 | 363.531258 / 1 | 377.135998 / 1 |
| op:Q   int8 projection | nested | 12.127978 / 1 | 11.078158 / 1 | 10.942535 / 1 |
| op:X   mxfp4 expert proj | nested | 141.696748 / 1 | 143.448711 / 1 | 162.833737 / 1 |
| op:N   rmsnorm | nested | 0.031607 / 1 | 0.031637 / 1 | 0.030936 / 1 |
| op:SiTU + sigma | nested | 0.108874 / 1 | 0.107670 / 1 | 0.142254 / 1 |
| op:AR  snapshot aggregate | nested | 0.031138 / 1 | 0.030616 / 1 | 0.029906 / 1 |
| op:SA  softmax attention | nested | 0.006613 / 1 | 0.009117 / 1 | 0.012864 / 1 |
| op:router dot product | nested | 0.524169 / 1 | 0.529659 / 1 | 0.508730 / 1 |
| op:top-k selection | nested | 0.002184 / 1 | 0.002415 / 1 | 0.002264 / 1 |
| detail:read-ahead-wait | nested | 0.001733 / 1 | 0.001680 / 1 | 0.001695 / 1 |
| total:layer | total | 155.649965 / 1 | 156.215163 / 1 | 175.416966 / 1 |

### Layer 36

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004578 / 1 | 0.004738 / 1 | 0.004658 / 1 |
| pre-attention-aggregation | boundary | 0.015439 / 1 | 0.015158 / 1 | 0.016020 / 1 |
| snapshot-push | boundary | 0.001483 / 1 | 0.001222 / 1 | 0.001733 / 1 |
| pre-attention-normalization | boundary | 0.011201 / 1 | 0.011341 / 1 | 0.011581 / 1 |
| Q | nested | 3.312677 / 1 | 3.403426 / 1 | 3.408425 / 1 |
| K | nested | 2.369235 / 1 | 2.352773 / 1 | 2.895337 / 1 |
| V | nested | 2.315244 / 1 | 2.275169 / 1 | 2.766616 / 1 |
| B | nested | 0.028874 / 1 | 0.028393 / 1 | 0.027892 / 1 |
| FA | nested | 0.028934 / 1 | 0.032440 / 1 | 0.035196 / 1 |
| FB | nested | 0.049082 / 1 | 0.053039 / 1 | 0.071113 / 1 |
| G | nested | 2.080786 / 1 | 2.038928 / 1 | 2.809688 / 1 |
| O | nested | 2.098289 / 1 | 2.088881 / 1 | 2.072560 / 1 |
| attention | boundary | 13.345593 / 1 | 13.353788 / 1 | 15.097926 / 1 |
| attention-residual | boundary | 0.002184 / 1 | 0.002034 / 1 | 0.002094 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024215 / 1 | 0.024446 / 1 | 0.030046 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001944 / 1 | 0.001653 / 1 | 0.002023 / 1 |
| router-and-top16 | boundary | 0.578530 / 1 | 0.573311 / 1 | 0.564884 / 1 |
| EDOWN | nested | 0.608276 / 1 | 0.590623 / 1 | 0.597506 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609217 / 1 | 0.591585 / 1 | 0.598478 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.062316 / 1 | 0.011411 / 1 | 0.014587 / 1 |
| SH1 | nested | 1.030485 / 1 | 1.005398 / 1 | 1.446352 / 1 |
| SH3 | nested | 1.013313 / 1 | 1.010316 / 1 | 1.525399 / 1 |
| SH2 | nested | 1.015667 / 1 | 1.013122 / 1 | 1.502215 / 1 |
| shared-expert-during-read | boundary | 3.071145 / 1 | 3.040098 / 1 | 4.490517 / 1 |
| detail:expert-gate | nested | 48.883728 / 16 | 49.700244 / 16 | 53.689894 / 16 |
| detail:expert-up | nested | 42.838231 / 16 | 42.906697 / 16 | 52.387134 / 16 |
| detail:expert-activation | nested | 0.102351 / 16 | 0.096689 / 16 | 0.129833 / 16 |
| detail:expert-down | nested | 49.095542 / 16 | 49.067874 / 16 | 52.841663 / 16 |
| EUP | nested | 0.600882 / 1 | 0.596864 / 1 | 0.766391 / 1 |
| experts-mix-normalize-up | boundary | 141.960870 / 1 | 142.747010 / 1 | 164.611397 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003046 / 1 | 0.002665 / 1 | 0.002805 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586766 / 1 | 0.596093 / 1 | 0.636980 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.987945 / 1 | 0.979599 / 1 | 1.063316 / 1 |
| worker:read | parallel worker | 0.066290 / 1 | 0.068514 / 1 | 0.068059 / 1 |
| worker:decode | parallel worker | 1074.176904 / 1 | 1082.235976 / 1 | 1089.990480 / 1 |
| worker:crc | parallel worker | 595.587698 / 1 | 600.626683 / 1 | 605.423785 / 1 |
| worker:math | parallel worker | 363.097899 / 1 | 362.922592 / 1 | 374.953790 / 1 |
| op:Q   int8 projection | nested | 16.550330 / 1 | 16.487972 / 1 | 19.922898 / 1 |
| op:X   mxfp4 expert proj | nested | 140.970682 / 1 | 141.820911 / 1 | 159.112417 / 1 |
| op:N   rmsnorm | nested | 0.044724 / 1 | 0.037090 / 1 | 0.031038 / 1 |
| op:L   l2 per-head | nested | 0.005560 / 1 | 0.005060 / 1 | 0.005320 / 1 |
| op:SiTU + sigma | nested | 0.109251 / 1 | 0.103784 / 1 | 0.140082 / 1 |
| op:C   shortconv | nested | 0.059982 / 1 | 0.068879 / 1 | 0.035226 / 1 |
| op:AR  snapshot aggregate | nested | 0.030095 / 1 | 0.030306 / 1 | 0.036729 / 1 |
| op:D   kda delta-rule | nested | 0.104936 / 1 | 0.107941 / 1 | 0.115796 / 1 |
| op:router dot product | nested | 0.575455 / 1 | 0.570616 / 1 | 0.562240 / 1 |
| op:top-k selection | nested | 0.002685 / 1 | 0.002355 / 1 | 0.002284 / 1 |
| op:alpha / beta / gate | nested | 0.072636 / 1 | 0.074740 / 1 | 0.057147 / 1 |
| detail:read-ahead-wait | nested | 0.001927 / 1 | 0.001763 / 1 | 4.416238 / 1 |
| total:layer | total | 160.690648 / 1 | 161.370960 / 1 | 186.524600 / 1 |

### Layer 37

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005701 / 1 | 0.005751 / 1 | 0.005059 / 1 |
| pre-attention-aggregation | boundary | 0.016651 / 1 | 0.016370 / 1 | 0.017924 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000471 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012603 / 1 | 0.011852 / 1 | 0.011823 / 1 |
| Q | nested | 3.217489 / 1 | 3.299091 / 1 | 3.148260 / 1 |
| K | nested | 2.398669 / 1 | 2.380425 / 1 | 2.845034 / 1 |
| V | nested | 2.327607 / 1 | 2.293182 / 1 | 2.770865 / 1 |
| B | nested | 0.022662 / 1 | 0.021089 / 1 | 0.027872 / 1 |
| FA | nested | 0.032110 / 1 | 0.033122 / 1 | 0.028833 / 1 |
| FB | nested | 0.050043 / 1 | 0.052568 / 1 | 0.056265 / 1 |
| G | nested | 2.097788 / 1 | 2.058534 / 1 | 2.424658 / 1 |
| O | nested | 2.083801 / 1 | 2.088911 / 1 | 2.503436 / 1 |
| attention | boundary | 13.369718 / 1 | 13.393251 / 1 | 15.008388 / 1 |
| attention-residual | boundary | 0.003597 / 1 | 0.003937 / 1 | 0.003716 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024425 / 1 | 0.025097 / 1 | 0.024927 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002424 / 1 | 0.002034 / 1 | 0.001923 / 1 |
| router-and-top16 | boundary | 0.591876 / 1 | 0.581206 / 1 | 0.613435 / 1 |
| EDOWN | nested | 0.608536 / 1 | 0.597556 / 1 | 0.683817 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.609498 / 1 | 0.598468 / 1 | 0.684798 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.061415 / 1 | 0.026770 / 1 | 0.013324 / 1 |
| SH1 | nested | 1.601550 / 1 | 1.015517 / 1 | 1.256466 / 1 |
| SH3 | nested | 1.497306 / 1 | 1.022209 / 1 | 1.535899 / 1 |
| SH2 | nested | 1.019204 / 1 | 1.011419 / 1 | 1.621949 / 1 |
| shared-expert-during-read | boundary | 4.130012 / 1 | 3.060305 / 1 | 4.430945 / 1 |
| detail:expert-gate | nested | 49.181576 / 16 | 49.759575 / 16 | 53.848187 / 16 |
| detail:expert-up | nested | 42.732663 / 16 | 42.824486 / 16 | 52.563723 / 16 |
| detail:expert-activation | nested | 0.098173 / 16 | 0.095420 / 16 | 0.133846 / 16 |
| detail:expert-down | nested | 49.023552 / 16 | 49.093854 / 16 | 51.816890 / 16 |
| EUP | nested | 0.598658 / 1 | 0.602024 / 1 | 0.608807 / 1 |
| experts-mix-normalize-up | boundary | 142.064885 / 1 | 142.780203 / 1 | 163.970019 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002936 / 1 | 0.002775 / 1 | 0.003076 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591295 / 1 | 0.587697 / 1 | 0.595542 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.958109 / 1 | 0.955445 / 1 | 0.994898 / 1 |
| worker:read | parallel worker | 0.068478 / 1 | 0.067894 / 1 | 0.070082 / 1 |
| worker:decode | parallel worker | 1072.781096 / 1 | 1082.427282 / 1 | 1086.956219 / 1 |
| worker:crc | parallel worker | 596.072113 / 1 | 601.142340 / 1 | 604.557606 / 1 |
| worker:math | parallel worker | 362.587600 / 1 | 362.561574 / 1 | 369.617041 / 1 |
| op:Q   int8 projection | nested | 17.553871 / 1 | 16.474254 / 1 | 19.510445 / 1 |
| op:X   mxfp4 expert proj | nested | 141.086650 / 1 | 141.823707 / 1 | 158.430809 / 1 |
| op:N   rmsnorm | nested | 0.045545 / 1 | 0.039684 / 1 | 0.031347 / 1 |
| op:L   l2 per-head | nested | 0.005460 / 1 | 0.005831 / 1 | 0.005390 / 1 |
| op:SiTU + sigma | nested | 0.105286 / 1 | 0.102625 / 1 | 0.144048 / 1 |
| op:C   shortconv | nested | 0.060703 / 1 | 0.051767 / 1 | 0.032581 / 1 |
| op:AR  snapshot aggregate | nested | 0.031730 / 1 | 0.031208 / 1 | 0.032691 / 1 |
| op:D   kda delta-rule | nested | 0.229579 / 1 | 0.237644 / 1 | 0.272669 / 1 |
| op:router dot product | nested | 0.589131 / 1 | 0.578280 / 1 | 0.610540 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002615 / 1 | 0.002365 / 1 |
| op:alpha / beta / gate | nested | 0.072245 / 1 | 0.072876 / 1 | 0.057879 / 1 |
| detail:read-ahead-wait | nested | 0.001841 / 1 | 0.001743 / 1 | 4.609146 / 1 |
| total:layer | total | 161.866725 / 1 | 161.475235 / 1 | 185.796961 / 1 |

### Layer 38

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004899 / 1 | 0.005390 / 1 | 0.005080 / 1 |
| pre-attention-aggregation | boundary | 0.017452 / 1 | 0.019336 / 1 | 0.017052 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012333 / 1 | 0.012424 / 1 | 0.011511 / 1 |
| Q | nested | 3.167816 / 1 | 3.286458 / 1 | 3.255199 / 1 |
| K | nested | 2.398439 / 1 | 2.385685 / 1 | 2.349968 / 1 |
| V | nested | 2.291489 / 1 | 2.315765 / 1 | 2.277964 / 1 |
| B | nested | 0.026610 / 1 | 0.027802 / 1 | 0.026931 / 1 |
| FA | nested | 0.029074 / 1 | 0.030116 / 1 | 0.027972 / 1 |
| FB | nested | 0.049853 / 1 | 0.049182 / 1 | 0.049873 / 1 |
| G | nested | 2.090835 / 1 | 2.084923 / 1 | 2.037235 / 1 |
| O | nested | 2.089001 / 1 | 2.086357 / 1 | 2.057272 / 1 |
| attention | boundary | 13.278287 / 1 | 13.434700 / 1 | 13.283507 / 1 |
| attention-residual | boundary | 0.003356 / 1 | 0.003947 / 1 | 0.003848 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025227 / 1 | 0.025197 / 1 | 0.023935 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001773 / 1 | 0.001713 / 1 | 0.001823 / 1 |
| router-and-top16 | boundary | 0.586195 / 1 | 0.577449 / 1 | 0.578270 / 1 |
| EDOWN | nested | 0.604028 / 1 | 0.585634 / 1 | 0.594700 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.605019 / 1 | 0.586586 / 1 | 0.595672 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043902 / 1 | 0.011992 / 1 | 0.018695 / 1 |
| SH1 | nested | 1.637759 / 1 | 1.011940 / 1 | 1.006240 / 1 |
| SH3 | nested | 1.459605 / 1 | 1.015126 / 1 | 1.005929 / 1 |
| SH2 | nested | 1.003915 / 1 | 1.010878 / 1 | 1.009755 / 1 |
| shared-expert-during-read | boundary | 4.113131 / 1 | 3.049836 / 1 | 3.033345 / 1 |
| detail:expert-gate | nested | 49.141172 / 16 | 49.838495 / 16 | 53.641618 / 16 |
| detail:expert-up | nested | 42.831699 / 16 | 42.857351 / 16 | 50.278637 / 16 |
| detail:expert-activation | nested | 0.095418 / 16 | 0.096018 / 16 | 0.118460 / 16 |
| detail:expert-down | nested | 49.013280 / 16 | 49.078813 / 16 | 49.249454 / 16 |
| EUP | nested | 0.608617 / 1 | 0.593899 / 1 | 0.598418 / 1 |
| experts-mix-normalize-up | boundary | 142.108225 / 1 | 142.844423 / 1 | 154.264074 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003095 / 1 | 0.002825 / 1 | 0.003035 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586736 / 1 | 0.589541 / 1 | 0.589771 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.974369 / 1 | 0.969080 / 1 | 0.964701 / 1 |
| worker:read | parallel worker | 0.066169 / 1 | 0.069223 / 1 | 0.068921 / 1 |
| worker:decode | parallel worker | 1073.513292 / 1 | 1081.431517 / 1 | 1083.319230 / 1 |
| worker:crc | parallel worker | 595.762012 / 1 | 600.538848 / 1 | 603.105388 / 1 |
| worker:math | parallel worker | 363.570433 / 1 | 363.500201 / 1 | 369.010433 / 1 |
| op:Q   int8 projection | nested | 17.455566 / 1 | 16.482351 / 1 | 16.296031 / 1 |
| op:X   mxfp4 expert proj | nested | 141.134367 / 1 | 141.920593 / 1 | 153.348093 / 1 |
| op:N   rmsnorm | nested | 0.048290 / 1 | 0.046377 / 1 | 0.031029 / 1 |
| op:L   l2 per-head | nested | 0.005310 / 1 | 0.009628 / 1 | 0.005960 / 1 |
| op:SiTU + sigma | nested | 0.102662 / 1 | 0.103941 / 1 | 0.125013 / 1 |
| op:C   shortconv | nested | 0.053470 / 1 | 0.063308 / 1 | 0.031078 / 1 |
| op:AR  snapshot aggregate | nested | 0.032590 / 1 | 0.034324 / 1 | 0.031568 / 1 |
| op:D   kda delta-rule | nested | 0.230210 / 1 | 0.239808 / 1 | 0.240529 / 1 |
| op:router dot product | nested | 0.583509 / 1 | 0.574422 / 1 | 0.575815 / 1 |
| op:top-k selection | nested | 0.002324 / 1 | 0.002645 / 1 | 0.002104 / 1 |
| op:alpha / beta / gate | nested | 0.072636 / 1 | 0.077645 / 1 | 0.058299 / 1 |
| detail:read-ahead-wait | nested | 0.001774 / 1 | 0.001473 / 1 | 0.001753 / 1 |
| total:layer | total | 161.789190 / 1 | 161.556327 / 1 | 172.815999 / 1 |

### Layer 39

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.006462 / 1 | 0.005911 / 1 | 0.006051 / 1 |
| pre-attention-aggregation | boundary | 0.016932 / 1 | 0.016221 / 1 | 0.016931 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012664 / 1 | 0.011481 / 1 | 0.011512 / 1 |
| QA | nested | 0.342710 / 1 | 0.355724 / 1 | 0.400467 / 1 |
| QB | nested | 0.909438 / 1 | 0.927372 / 1 | 1.005077 / 1 |
| KA | nested | 0.133570 / 1 | 0.142086 / 1 | 0.137136 / 1 |
| KB | nested | 0.413201 / 1 | 0.416387 / 1 | 0.405567 / 1 |
| G | nested | 2.714129 / 1 | 2.702016 / 1 | 2.973714 / 1 |
| O | nested | 2.080876 / 1 | 2.078652 / 1 | 2.979855 / 1 |
| attention | boundary | 6.702988 / 1 | 6.737192 / 1 | 8.037000 / 1 |
| attention-residual | boundary | 0.002896 / 1 | 0.002855 / 1 | 0.003676 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024125 / 1 | 0.024125 / 1 | 0.027582 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001924 / 1 | 0.001563 / 1 | 0.001973 / 1 |
| router-and-top16 | boundary | 0.519971 / 1 | 0.529999 / 1 | 0.601644 / 1 |
| EDOWN | nested | 0.613295 / 1 | 0.606483 / 1 | 0.858193 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.614247 / 1 | 0.607424 / 1 | 0.859626 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.057317 / 1 | 0.023114 / 1 | 0.011772 / 1 |
| SH1 | nested | 1.544855 / 1 | 1.455238 / 1 | 1.241428 / 1 |
| SH3 | nested | 1.373094 / 1 | 1.447484 / 1 | 1.213636 / 1 |
| SH2 | nested | 1.010667 / 1 | 1.377402 / 1 | 1.216442 / 1 |
| shared-expert-during-read | boundary | 3.940869 / 1 | 4.296253 / 1 | 3.685152 / 1 |
| detail:expert-gate | nested | 49.604348 / 16 | 50.079213 / 16 | 52.227737 / 16 |
| detail:expert-up | nested | 42.779561 / 16 | 42.823894 / 16 | 49.436723 / 16 |
| detail:expert-activation | nested | 0.094327 / 16 | 0.097062 / 16 | 0.111690 / 16 |
| detail:expert-down | nested | 49.164523 / 16 | 49.221262 / 16 | 50.609185 / 16 |
| EUP | nested | 0.602825 / 1 | 0.606372 / 1 | 0.598357 / 1 |
| experts-mix-normalize-up | boundary | 142.675926 / 1 | 143.219893 / 1 | 153.375834 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003146 / 1 | 0.002925 / 1 | 0.003086 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591985 / 1 | 0.592977 / 1 | 0.598979 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.597667 / 1 | 0.598408 / 1 | 0.604940 / 1 |
| worker:read | parallel worker | 0.208122 / 1 | 0.187826 / 1 | 0.202970 / 1 |
| worker:decode | parallel worker | 1072.290327 / 1 | 1078.954920 / 1 | 1084.023962 / 1 |
| worker:crc | parallel worker | 596.595191 / 1 | 600.449418 / 1 | 603.755450 / 1 |
| worker:math | parallel worker | 363.058197 / 1 | 363.847835 / 1 | 369.984023 / 1 |
| op:Q   int8 projection | nested | 11.737377 / 1 | 12.113754 / 1 | 13.028661 / 1 |
| op:X   mxfp4 expert proj | nested | 141.693244 / 1 | 142.271201 / 1 | 152.443533 / 1 |
| op:N   rmsnorm | nested | 0.032500 / 1 | 0.032220 / 1 | 0.032282 / 1 |
| op:SiTU + sigma | nested | 0.102280 / 1 | 0.107795 / 1 | 0.120907 / 1 |
| op:AR  snapshot aggregate | nested | 0.031108 / 1 | 0.030337 / 1 | 0.033142 / 1 |
| op:SA  softmax attention | nested | 0.006272 / 1 | 0.008846 / 1 | 0.013285 / 1 |
| op:router dot product | nested | 0.517005 / 1 | 0.527144 / 1 | 0.598417 / 1 |
| op:top-k selection | nested | 0.002545 / 1 | 0.002494 / 1 | 0.002806 / 1 |
| detail:read-ahead-wait | nested | 0.001246 / 1 | 0.001433 / 1 | 0.001050 / 1 |
| total:layer | total | 155.190407 / 1 | 156.088977 / 1 | 167.261596 / 1 |

### Layer 40

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004608 / 1 | 0.005049 / 1 | 0.004038 / 1 |
| pre-attention-aggregation | boundary | 0.015549 / 1 | 0.015249 / 1 | 0.015760 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011451 / 1 | 0.011271 / 1 | 0.011472 / 1 |
| Q | nested | 3.269175 / 1 | 3.324539 / 1 | 3.279835 / 1 |
| K | nested | 2.382990 / 1 | 2.358064 / 1 | 2.335541 / 1 |
| V | nested | 2.334550 / 1 | 2.292842 / 1 | 2.318099 / 1 |
| B | nested | 0.021901 / 1 | 0.020518 / 1 | 0.020838 / 1 |
| FA | nested | 0.035166 / 1 | 0.036157 / 1 | 0.027521 / 1 |
| FB | nested | 0.048391 / 1 | 0.049432 / 1 | 0.049473 / 1 |
| G | nested | 2.162198 / 1 | 2.058976 / 1 | 2.066649 / 1 |
| O | nested | 2.067301 / 1 | 2.086447 / 1 | 2.074094 / 1 |
| attention | boundary | 13.378785 / 1 | 13.280882 / 1 | 13.248723 / 1 |
| attention-residual | boundary | 0.004098 / 1 | 0.003667 / 1 | 0.003827 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023945 / 1 | 0.026529 / 1 | 0.024516 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001753 / 1 | 0.001563 / 1 | 0.001824 / 1 |
| router-and-top16 | boundary | 0.568412 / 1 | 0.572910 / 1 | 0.565636 / 1 |
| EDOWN | nested | 0.587317 / 1 | 0.601713 / 1 | 0.594751 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.588319 / 1 | 0.602715 / 1 | 0.595722 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052679 / 1 | 0.022212 / 1 | 0.010209 / 1 |
| SH1 | nested | 1.407509 / 1 | 1.005869 / 1 | 1.008483 / 1 |
| SH3 | nested | 1.441251 / 1 | 1.009496 / 1 | 1.000639 / 1 |
| SH2 | nested | 1.419390 / 1 | 1.021337 / 1 | 1.010006 / 1 |
| shared-expert-during-read | boundary | 4.284682 / 1 | 3.048143 / 1 | 3.030850 / 1 |
| detail:expert-gate | nested | 49.337910 / 16 | 49.928502 / 16 | 53.259102 / 16 |
| detail:expert-up | nested | 42.821408 / 16 | 42.820015 / 16 | 51.350297 / 16 |
| detail:expert-activation | nested | 0.100096 / 16 | 0.096842 / 16 | 0.125465 / 16 |
| detail:expert-down | nested | 49.089815 / 16 | 49.088271 / 16 | 52.869096 / 16 |
| EUP | nested | 0.600962 / 1 | 0.601694 / 1 | 0.602565 / 1 |
| experts-mix-normalize-up | boundary | 142.366508 / 1 | 142.930794 / 1 | 158.581486 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003216 / 1 | 0.003186 / 1 | 0.003266 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587337 / 1 | 0.586776 / 1 | 0.600702 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.966034 / 1 | 0.969311 / 1 | 0.980611 / 1 |
| worker:read | parallel worker | 0.068045 / 1 | 0.066416 / 1 | 0.068362 / 1 |
| worker:decode | parallel worker | 1072.740057 / 1 | 1079.287145 / 1 | 1082.847398 / 1 |
| worker:crc | parallel worker | 596.566128 / 1 | 600.745960 / 1 | 604.617451 / 1 |
| worker:math | parallel worker | 363.641791 / 1 | 363.435490 / 1 | 370.365339 / 1 |
| op:Q   int8 projection | nested | 17.776250 / 1 | 16.465592 / 1 | 16.387071 / 1 |
| op:X   mxfp4 expert proj | nested | 141.400154 / 1 | 141.983903 / 1 | 157.671275 / 1 |
| op:N   rmsnorm | nested | 0.047530 / 1 | 0.039383 / 1 | 0.031558 / 1 |
| op:L   l2 per-head | nested | 0.005450 / 1 | 0.006002 / 1 | 0.005541 / 1 |
| op:SiTU + sigma | nested | 0.110826 / 1 | 0.103804 / 1 | 0.132199 / 1 |
| op:C   shortconv | nested | 0.049272 / 1 | 0.065643 / 1 | 0.031750 / 1 |
| op:AR  snapshot aggregate | nested | 0.029967 / 1 | 0.030177 / 1 | 0.030417 / 1 |
| op:D   kda delta-rule | nested | 0.112039 / 1 | 0.108172 / 1 | 0.138518 / 1 |
| op:router dot product | nested | 0.565626 / 1 | 0.570315 / 1 | 0.562731 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002354 / 1 | 0.002475 / 1 |
| op:alpha / beta / gate | nested | 0.074649 / 1 | 0.074248 / 1 | 0.057908 / 1 |
| detail:read-ahead-wait | nested | 0.001815 / 1 | 0.001691 / 1 | 0.001652 / 1 |
| total:layer | total | 162.282541 / 1 | 161.505512 / 1 | 177.089571 / 1 |

### Layer 41

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005841 / 1 | 0.005330 / 1 | 0.004439 / 1 |
| pre-attention-aggregation | boundary | 0.017002 / 1 | 0.016410 / 1 | 0.016941 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000090 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012644 / 1 | 0.012323 / 1 | 0.012022 / 1 |
| Q | nested | 3.132389 / 1 | 3.331602 / 1 | 3.408716 / 1 |
| K | nested | 2.377310 / 1 | 2.368303 / 1 | 2.826809 / 1 |
| V | nested | 2.343817 / 1 | 2.312419 / 1 | 2.833471 / 1 |
| B | nested | 0.024566 / 1 | 0.027291 / 1 | 0.027521 / 1 |
| FA | nested | 0.028664 / 1 | 0.034094 / 1 | 0.036889 / 1 |
| FB | nested | 0.049683 / 1 | 0.052157 / 1 | 0.056946 / 1 |
| G | nested | 2.094091 / 1 | 2.062572 / 1 | 2.223562 / 1 |
| O | nested | 2.076348 / 1 | 2.074204 / 1 | 2.061790 / 1 |
| attention | boundary | 13.251517 / 1 | 13.404132 / 1 | 14.625584 / 1 |
| attention-residual | boundary | 0.003757 / 1 | 0.003747 / 1 | 0.003667 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023984 / 1 | 0.024496 / 1 | 0.024035 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001673 / 1 | 0.001853 / 1 | 0.002014 / 1 |
| router-and-top16 | boundary | 0.583169 / 1 | 0.580894 / 1 | 0.573311 / 1 |
| EDOWN | nested | 0.600301 / 1 | 0.604048 / 1 | 0.595803 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.601302 / 1 | 0.605140 / 1 | 0.596805 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043721 / 1 | 0.016822 / 1 | 0.019096 / 1 |
| SH1 | nested | 1.415874 / 1 | 1.011639 / 1 | 1.006991 / 1 |
| SH3 | nested | 1.425913 / 1 | 1.015166 / 1 | 1.006169 / 1 |
| SH2 | nested | 1.379987 / 1 | 1.012792 / 1 | 1.008693 / 1 |
| shared-expert-during-read | boundary | 4.238686 / 1 | 3.051569 / 1 | 3.033565 / 1 |
| detail:expert-gate | nested | 49.482811 / 16 | 50.398910 / 16 | 53.312454 / 16 |
| detail:expert-up | nested | 42.658706 / 16 | 44.338420 / 16 | 52.020839 / 16 |
| detail:expert-activation | nested | 0.095396 / 16 | 0.100317 / 16 | 0.123880 / 16 |
| detail:expert-down | nested | 48.977966 / 16 | 49.228254 / 16 | 52.137880 / 16 |
| EUP | nested | 0.598859 / 1 | 0.606903 / 1 | 0.609618 / 1 |
| experts-mix-normalize-up | boundary | 142.223281 / 1 | 145.060932 / 1 | 158.601053 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003246 / 1 | 0.002765 / 1 | 0.003397 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.592076 / 1 | 0.591224 / 1 | 0.595252 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.945476 / 1 | 0.970742 / 1 | 0.970723 / 1 |
| worker:read | parallel worker | 0.066072 / 1 | 0.066259 / 1 | 0.068020 / 1 |
| worker:decode | parallel worker | 1070.196208 / 1 | 1081.181983 / 1 | 1084.905558 / 1 |
| worker:crc | parallel worker | 597.344393 / 1 | 603.360979 / 1 | 607.423394 / 1 |
| worker:math | parallel worker | 362.832959 / 1 | 364.282238 / 1 | 374.142103 / 1 |
| op:Q   int8 projection | nested | 17.546209 / 1 | 16.511668 / 1 | 17.701536 / 1 |
| op:X   mxfp4 expert proj | nested | 141.266655 / 1 | 144.117001 / 1 | 157.660626 / 1 |
| op:N   rmsnorm | nested | 0.038732 / 1 | 0.040555 / 1 | 0.031658 / 1 |
| op:L   l2 per-head | nested | 0.005601 / 1 | 0.005610 / 1 | 0.005560 / 1 |
| op:SiTU + sigma | nested | 0.106580 / 1 | 0.107619 / 1 | 0.130523 / 1 |
| op:C   shortconv | nested | 0.057578 / 1 | 0.047849 / 1 | 0.031409 / 1 |
| op:AR  snapshot aggregate | nested | 0.031500 / 1 | 0.031258 / 1 | 0.031328 / 1 |
| op:D   kda delta-rule | nested | 0.242904 / 1 | 0.240189 / 1 | 0.239858 / 1 |
| op:router dot product | nested | 0.580204 / 1 | 0.577699 / 1 | 0.570516 / 1 |
| op:top-k selection | nested | 0.002585 / 1 | 0.002785 / 1 | 0.002445 / 1 |
| op:alpha / beta / gate | nested | 0.073196 / 1 | 0.074489 / 1 | 0.058429 / 1 |
| detail:read-ahead-wait | nested | 0.001694 / 1 | 0.001682 / 1 | 0.001524 / 1 |
| total:layer | total | 161.968575 / 1 | 163.769801 / 1 | 178.499564 / 1 |

### Layer 42

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005099 / 1 | 0.005129 / 1 | 0.005600 / 1 |
| pre-attention-aggregation | boundary | 0.016230 / 1 | 0.016521 / 1 | 0.017452 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011131 / 1 | 0.014677 / 1 | 0.012083 / 1 |
| Q | nested | 3.260740 / 1 | 3.356869 / 1 | 3.396343 / 1 |
| K | nested | 2.381638 / 1 | 2.373563 / 1 | 2.827571 / 1 |
| V | nested | 2.329861 / 1 | 2.293473 / 1 | 2.793197 / 1 |
| B | nested | 0.022673 / 1 | 0.022543 / 1 | 0.023664 / 1 |
| FA | nested | 0.037310 / 1 | 0.031759 / 1 | 0.028363 / 1 |
| FB | nested | 0.049242 / 1 | 0.051456 / 1 | 0.052257 / 1 |
| G | nested | 2.119007 / 1 | 2.097267 / 1 | 2.669255 / 1 |
| O | nested | 2.065056 / 1 | 2.087939 / 1 | 2.056050 / 1 |
| attention | boundary | 13.431503 / 1 | 13.440711 / 1 | 14.997969 / 1 |
| attention-residual | boundary | 0.004158 / 1 | 0.003677 / 1 | 0.003817 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024325 / 1 | 0.024706 / 1 | 0.030607 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001833 / 1 | 0.001814 / 1 | 0.002675 / 1 |
| router-and-top16 | boundary | 0.571047 / 1 | 0.580213 / 1 | 0.566788 / 1 |
| EDOWN | nested | 0.597376 / 1 | 0.596003 / 1 | 0.596133 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.598408 / 1 | 0.597035 / 1 | 0.597125 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.067045 / 1 | 0.011892 / 1 | 0.008496 / 1 |
| SH1 | nested | 0.992634 / 1 | 1.025405 / 1 | 1.002773 / 1 |
| SH3 | nested | 0.997863 / 1 | 1.014485 / 1 | 1.004776 / 1 |
| SH2 | nested | 0.994097 / 1 | 1.010207 / 1 | 1.006069 / 1 |
| shared-expert-during-read | boundary | 2.996146 / 1 | 3.062138 / 1 | 3.025250 / 1 |
| detail:expert-gate | nested | 49.817055 / 16 | 50.084472 / 16 | 53.864053 / 16 |
| detail:expert-up | nested | 42.705124 / 16 | 42.728925 / 16 | 52.889774 / 16 |
| detail:expert-activation | nested | 0.096211 / 16 | 0.097623 / 16 | 0.129865 / 16 |
| detail:expert-down | nested | 48.933125 / 16 | 49.090466 / 16 | 53.012463 / 16 |
| EUP | nested | 0.601142 / 1 | 0.607004 / 1 | 0.606202 / 1 |
| experts-mix-normalize-up | boundary | 142.596277 / 1 | 142.993250 / 1 | 160.896750 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002715 / 1 | 0.002755 / 1 | 0.003056 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591194 / 1 | 0.588900 / 1 | 0.591164 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.945676 / 1 | 0.955144 / 1 | 0.972777 / 1 |
| worker:read | parallel worker | 0.197728 / 1 | 0.210015 / 1 | 0.201555 / 1 |
| worker:decode | parallel worker | 1071.511054 / 1 | 1077.896289 / 1 | 1083.949408 / 1 |
| worker:crc | parallel worker | 597.528880 / 1 | 601.805335 / 1 | 604.909791 / 1 |
| worker:math | parallel worker | 362.539210 / 1 | 363.179268 / 1 | 370.828481 / 1 |
| op:Q   int8 projection | nested | 16.447255 / 1 | 16.566328 / 1 | 18.061159 / 1 |
| op:X   mxfp4 expert proj | nested | 141.604616 / 1 | 142.053647 / 1 | 159.971192 / 1 |
| op:N   rmsnorm | nested | 0.043531 / 1 | 0.049331 / 1 | 0.036107 / 1 |
| op:L   l2 per-head | nested | 0.005451 / 1 | 0.007464 / 1 | 0.005741 / 1 |
| op:SiTU + sigma | nested | 0.103406 / 1 | 0.104635 / 1 | 0.136545 / 1 |
| op:C   shortconv | nested | 0.064400 / 1 | 0.056906 / 1 | 0.030498 / 1 |
| op:AR  snapshot aggregate | nested | 0.030967 / 1 | 0.031429 / 1 | 0.034313 / 1 |
| op:D   kda delta-rule | nested | 0.241681 / 1 | 0.245629 / 1 | 0.240509 / 1 |
| op:router dot product | nested | 0.568501 / 1 | 0.577378 / 1 | 0.564083 / 1 |
| op:top-k selection | nested | 0.002144 / 1 | 0.002434 / 1 | 0.002204 / 1 |
| op:alpha / beta / gate | nested | 0.072896 / 1 | 0.074539 / 1 | 0.058439 / 1 |
| detail:read-ahead-wait | nested | 0.001163 / 1 | 0.001392 / 1 | 0.001041 / 1 |
| total:layer | total | 161.284076 / 1 | 161.722177 / 1 | 181.154722 / 1 |

### Layer 43

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004900 / 1 | 0.004849 / 1 | 0.003917 / 1 |
| pre-attention-aggregation | boundary | 0.016331 / 1 | 0.016751 / 1 | 0.016651 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000160 / 1 |
| pre-attention-normalization | boundary | 0.011331 / 1 | 0.011502 / 1 | 0.011241 / 1 |
| QA | nested | 0.360173 / 1 | 0.355925 / 1 | 0.388055 / 1 |
| QB | nested | 0.936369 / 1 | 0.938573 / 1 | 0.862060 / 1 |
| KA | nested | 0.133369 / 1 | 0.133038 / 1 | 0.144069 / 1 |
| KB | nested | 0.422048 / 1 | 0.432537 / 1 | 0.451403 / 1 |
| G | nested | 2.745297 / 1 | 2.762479 / 1 | 2.897571 / 1 |
| O | nested | 2.023098 / 1 | 2.084954 / 1 | 2.960759 / 1 |
| attention | boundary | 6.732112 / 1 | 6.823172 / 1 | 7.847555 / 1 |
| attention-residual | boundary | 0.002805 / 1 | 0.002946 / 1 | 0.003597 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024636 / 1 | 0.024366 / 1 | 0.027812 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001533 / 1 | 0.001904 / 1 | 0.002485 / 1 |
| router-and-top16 | boundary | 0.520011 / 1 | 0.522506 / 1 | 0.601683 / 1 |
| EDOWN | nested | 0.596083 / 1 | 0.606082 / 1 | 0.857331 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.597115 / 1 | 0.607094 / 1 | 0.858703 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.033342 / 1 | 0.026850 / 1 | 0.015058 / 1 |
| SH1 | nested | 1.429790 / 1 | 1.048498 / 1 | 1.314856 / 1 |
| SH3 | nested | 1.404893 / 1 | 1.044139 / 1 | 1.158534 / 1 |
| SH2 | nested | 1.341505 / 1 | 1.037337 / 1 | 1.161949 / 1 |
| shared-expert-during-read | boundary | 4.192079 / 1 | 3.141416 / 1 | 3.653994 / 1 |
| detail:expert-gate | nested | 48.951715 / 16 | 49.706760 / 16 | 53.661884 / 16 |
| detail:expert-up | nested | 42.729807 / 16 | 42.717527 / 16 | 52.580004 / 16 |
| detail:expert-activation | nested | 0.094506 / 16 | 0.095509 / 16 | 0.131955 / 16 |
| detail:expert-down | nested | 49.029592 / 16 | 49.045113 / 16 | 54.166036 / 16 |
| EUP | nested | 0.605992 / 1 | 0.609709 / 1 | 0.600772 / 1 |
| experts-mix-normalize-up | boundary | 141.814848 / 1 | 142.576503 / 1 | 165.146586 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003056 / 1 | 0.003116 / 1 | 0.003156 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.603307 / 1 | 0.594780 / 1 | 0.599249 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.609157 / 1 | 0.600432 / 1 | 0.606272 / 1 |
| worker:read | parallel worker | 0.067694 / 1 | 0.067141 / 1 | 0.069351 / 1 |
| worker:decode | parallel worker | 1071.520974 / 1 | 1078.879117 / 1 | 1086.954584 / 1 |
| worker:crc | parallel worker | 595.554827 / 1 | 600.283293 / 1 | 606.167395 / 1 |
| worker:math | parallel worker | 363.409398 / 1 | 362.869401 / 1 | 374.622483 / 1 |
| op:Q   int8 projection | nested | 11.997304 / 1 | 11.052029 / 1 | 12.793000 / 1 |
| op:X   mxfp4 expert proj | nested | 140.860506 / 1 | 141.617528 / 1 | 160.610835 / 1 |
| op:N   rmsnorm | nested | 0.032079 / 1 | 0.031519 / 1 | 0.032029 / 1 |
| op:SiTU + sigma | nested | 0.105405 / 1 | 0.102384 / 1 | 0.145171 / 1 |
| op:AR  snapshot aggregate | nested | 0.030548 / 1 | 0.031158 / 1 | 0.032461 / 1 |
| op:SA  softmax attention | nested | 0.006553 / 1 | 0.009338 / 1 | 0.013365 / 1 |
| op:router dot product | nested | 0.517346 / 1 | 0.519831 / 1 | 0.598898 / 1 |
| op:top-k selection | nested | 0.002304 / 1 | 0.002344 / 1 | 0.002435 / 1 |
| detail:read-ahead-wait | nested | 0.001222 / 1 | 0.001322 / 1 | 3.620031 / 1 |
| total:layer | total | 154.576300 / 1 | 154.375697 / 1 | 178.814732 / 1 |

### Layer 44

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004308 / 1 | 0.003717 / 1 | 0.005059 / 1 |
| pre-attention-aggregation | boundary | 0.015359 / 1 | 0.015209 / 1 | 0.016221 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011051 / 1 | 0.012143 / 1 | 0.012062 / 1 |
| Q | nested | 3.290134 / 1 | 3.288571 / 1 | 3.398637 / 1 |
| K | nested | 2.351792 / 1 | 2.343637 / 1 | 2.897992 / 1 |
| V | nested | 2.290187 / 1 | 2.319842 / 1 | 2.822000 / 1 |
| B | nested | 0.021080 / 1 | 0.021450 / 1 | 0.029936 / 1 |
| FA | nested | 0.028794 / 1 | 0.027201 / 1 | 0.037370 / 1 |
| FB | nested | 0.049813 / 1 | 0.051466 / 1 | 0.070823 / 1 |
| G | nested | 2.042564 / 1 | 2.091566 / 1 | 2.875370 / 1 |
| O | nested | 2.075797 / 1 | 2.081497 / 1 | 2.042545 / 1 |
| attention | boundary | 13.243613 / 1 | 13.297463 / 1 | 15.241774 / 1 |
| attention-residual | boundary | 0.003637 / 1 | 0.004058 / 1 | 0.003757 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023764 / 1 | 0.024887 / 1 | 0.023724 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001492 / 1 | 0.001934 / 1 | 0.002104 / 1 |
| router-and-top16 | boundary | 0.560647 / 1 | 0.572409 / 1 | 0.553263 / 1 |
| EDOWN | nested | 0.588550 / 1 | 0.592677 / 1 | 0.584141 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.589902 / 1 | 0.593800 / 1 | 0.585173 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.039183 / 1 | 0.016421 / 1 | 0.012183 / 1 |
| SH1 | nested | 1.450780 / 1 | 1.002673 / 1 | 0.971675 / 1 |
| SH3 | nested | 1.461900 / 1 | 1.011219 / 1 | 0.966274 / 1 |
| SH2 | nested | 1.006380 / 1 | 1.014474 / 1 | 0.973648 / 1 |
| shared-expert-during-read | boundary | 3.931813 / 1 | 3.040027 / 1 | 2.923700 / 1 |
| detail:expert-gate | nested | 55.962197 / 16 | 55.224814 / 16 | 56.858934 / 16 |
| detail:expert-up | nested | 55.215934 / 16 | 56.249686 / 16 | 56.109465 / 16 |
| detail:expert-activation | nested | 0.097542 / 16 | 0.097702 / 16 | 0.136194 / 16 |
| detail:expert-down | nested | 49.263069 / 16 | 49.408550 / 16 | 53.195786 / 16 |
| EUP | nested | 0.601373 / 1 | 0.614107 / 1 | 0.615990 / 1 |
| experts-mix-normalize-up | boundary | 161.550084 / 1 | 161.983054 / 1 | 167.308975 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002935 / 1 | 0.002945 / 1 | 0.002364 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589852 / 1 | 0.594159 / 1 | 0.594601 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.971153 / 1 | 0.988115 / 1 | 0.995028 / 1 |
| worker:read | parallel worker | 0.067326 / 1 | 0.066936 / 1 | 0.069744 / 1 |
| worker:decode | parallel worker | 1075.472680 / 1 | 1083.558079 / 1 | 1089.331344 / 1 |
| worker:crc | parallel worker | 620.785559 / 1 | 624.144835 / 1 | 629.852346 / 1 |
| worker:math | parallel worker | 379.116368 / 1 | 381.090175 / 1 | 394.653103 / 1 |
| op:Q   int8 projection | nested | 17.257739 / 1 | 16.458946 / 1 | 18.284850 / 1 |
| op:X   mxfp4 expert proj | nested | 160.595270 / 1 | 161.037669 / 1 | 166.374257 / 1 |
| op:N   rmsnorm | nested | 0.045264 / 1 | 0.040045 / 1 | 0.030917 / 1 |
| op:L   l2 per-head | nested | 0.005320 / 1 | 0.005871 / 1 | 0.005650 / 1 |
| op:SiTU + sigma | nested | 0.104946 / 1 | 0.104825 / 1 | 0.142786 / 1 |
| op:C   shortconv | nested | 0.053771 / 1 | 0.074038 / 1 | 0.033453 / 1 |
| op:AR  snapshot aggregate | nested | 0.029666 / 1 | 0.029865 / 1 | 0.030757 / 1 |
| op:D   kda delta-rule | nested | 0.162303 / 1 | 0.115005 / 1 | 0.123972 / 1 |
| op:router dot product | nested | 0.558153 / 1 | 0.569794 / 1 | 0.550228 / 1 |
| op:top-k selection | nested | 0.002144 / 1 | 0.002335 / 1 | 0.002675 / 1 |
| op:alpha / beta / gate | nested | 0.072456 / 1 | 0.074880 / 1 | 0.065803 / 1 |
| detail:read-ahead-wait | nested | 0.001607 / 1 | 0.001585 / 1 | 0.001562 / 1 |
| total:layer | total | 180.962997 / 1 | 180.569725 / 1 | 187.699405 / 1 |

### Layer 45

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005690 / 1 | 0.004960 / 1 | 0.005160 / 1 |
| pre-attention-aggregation | boundary | 0.017653 / 1 | 0.017463 / 1 | 0.017232 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000030 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012523 / 1 | 0.012894 / 1 | 0.012423 / 1 |
| Q | nested | 2.859269 / 1 | 2.842288 / 1 | 3.454412 / 1 |
| K | nested | 2.360829 / 1 | 2.397588 / 1 | 2.834494 / 1 |
| V | nested | 2.297490 / 1 | 2.330843 / 1 | 2.756568 / 1 |
| B | nested | 0.024075 / 1 | 0.025457 / 1 | 0.027221 / 1 |
| FA | nested | 0.028283 / 1 | 0.028514 / 1 | 0.032451 / 1 |
| FB | nested | 0.049834 / 1 | 0.051176 / 1 | 0.072676 / 1 |
| G | nested | 2.046372 / 1 | 2.103168 / 1 | 2.745568 / 1 |
| O | nested | 2.076507 / 1 | 2.075466 / 1 | 2.063033 / 1 |
| attention | boundary | 12.933504 / 1 | 13.059619 / 1 | 15.115378 / 1 |
| attention-residual | boundary | 0.004108 / 1 | 0.003967 / 1 | 0.003696 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.023875 / 1 | 0.025327 / 1 | 0.024365 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002064 / 1 | 0.001683 / 1 | 0.002234 / 1 |
| router-and-top16 | boundary | 0.575826 / 1 | 0.584762 / 1 | 0.575685 / 1 |
| EDOWN | nested | 0.581797 / 1 | 0.589952 / 1 | 0.591685 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.582869 / 1 | 0.591044 / 1 | 0.592787 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041848 / 1 | 0.016230 / 1 | 0.016461 / 1 |
| SH1 | nested | 0.990680 / 1 | 1.003525 / 1 | 0.988787 / 1 |
| SH3 | nested | 1.003655 / 1 | 1.018181 / 1 | 0.990620 / 1 |
| SH2 | nested | 1.003494 / 1 | 1.016608 / 1 | 0.980081 / 1 |
| shared-expert-during-read | boundary | 3.009601 / 1 | 3.049846 / 1 | 2.971710 / 1 |
| detail:expert-gate | nested | 49.425212 / 16 | 50.335300 / 16 | 53.458866 / 16 |
| detail:expert-up | nested | 42.715461 / 16 | 43.506177 / 16 | 51.972320 / 16 |
| detail:expert-activation | nested | 0.094336 / 16 | 0.098293 / 16 | 0.129963 / 16 |
| detail:expert-down | nested | 48.946186 / 16 | 49.031068 / 16 | 54.196215 / 16 |
| EUP | nested | 0.614498 / 1 | 0.602696 / 1 | 0.604419 / 1 |
| experts-mix-normalize-up | boundary | 142.210287 / 1 | 143.966669 / 1 | 163.912322 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003266 / 1 | 0.003056 / 1 | 0.003005 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589361 / 1 | 0.585654 / 1 | 0.599780 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.941498 / 1 | 0.966385 / 1 | 0.992383 / 1 |
| worker:read | parallel worker | 0.066783 / 1 | 0.067645 / 1 | 0.069710 / 1 |
| worker:decode | parallel worker | 1071.692278 / 1 | 1079.986939 / 1 | 1083.536963 / 1 |
| worker:crc | parallel worker | 596.174277 / 1 | 600.173432 / 1 | 603.548942 / 1 |
| worker:math | parallel worker | 362.912279 / 1 | 363.141898 / 1 | 370.622675 / 1 |
| op:Q   int8 projection | nested | 15.935419 / 1 | 16.084006 / 1 | 18.140198 / 1 |
| op:X   mxfp4 expert proj | nested | 141.236918 / 1 | 143.029245 / 1 | 159.833946 / 1 |
| op:N   rmsnorm | nested | 0.040746 / 1 | 0.051707 / 1 | 0.031597 / 1 |
| op:L   l2 per-head | nested | 0.005441 / 1 | 0.005200 / 1 | 0.005370 / 1 |
| op:SiTU + sigma | nested | 0.101360 / 1 | 0.105255 / 1 | 0.136455 / 1 |
| op:C   shortconv | nested | 0.056767 / 1 | 0.048550 / 1 | 0.031479 / 1 |
| op:AR  snapshot aggregate | nested | 0.031970 / 1 | 0.032852 / 1 | 0.032040 / 1 |
| op:D   kda delta-rule | nested | 0.246240 / 1 | 0.239046 / 1 | 0.241842 / 1 |
| op:router dot product | nested | 0.573261 / 1 | 0.582197 / 1 | 0.572749 / 1 |
| op:top-k selection | nested | 0.002104 / 1 | 0.002234 / 1 | 0.002515 / 1 |
| op:alpha / beta / gate | nested | 0.082003 / 1 | 0.074309 / 1 | 0.058249 / 1 |
| detail:read-ahead-wait | nested | 0.001363 / 1 | 0.001221 / 1 | 3.094330 / 1 |
| total:layer | total | 160.378776 / 1 | 162.321507 / 1 | 184.259441 / 1 |

### Layer 46

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004789 / 1 | 0.005220 / 1 | 0.005049 / 1 |
| pre-attention-aggregation | boundary | 0.016501 / 1 | 0.016681 / 1 | 0.017543 / 1 |
| snapshot-push | boundary | 0.000131 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011411 / 1 | 0.011401 / 1 | 0.011251 / 1 |
| Q | nested | 3.267913 / 1 | 3.287640 / 1 | 3.256451 / 1 |
| K | nested | 2.385755 / 1 | 2.367771 / 1 | 2.339550 / 1 |
| V | nested | 2.278295 / 1 | 2.314232 / 1 | 2.271402 / 1 |
| B | nested | 0.020819 / 1 | 0.021941 / 1 | 0.022913 / 1 |
| FA | nested | 0.029856 / 1 | 0.028664 / 1 | 0.037270 / 1 |
| FB | nested | 0.050765 / 1 | 0.048320 / 1 | 0.051797 / 1 |
| G | nested | 2.058144 / 1 | 2.091255 / 1 | 2.037535 / 1 |
| O | nested | 2.063243 / 1 | 2.075987 / 1 | 2.068783 / 1 |
| attention | boundary | 13.324864 / 1 | 13.420733 / 1 | 13.307953 / 1 |
| attention-residual | boundary | 0.003847 / 1 | 0.003757 / 1 | 0.003496 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024416 / 1 | 0.024416 / 1 | 0.023924 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001964 / 1 | 0.002094 / 1 | 0.002013 / 1 |
| router-and-top16 | boundary | 0.569334 / 1 | 0.577859 / 1 | 0.576416 / 1 |
| EDOWN | nested | 0.599530 / 1 | 0.592687 / 1 | 0.587226 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.600602 / 1 | 0.593779 / 1 | 0.588369 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.056205 / 1 | 0.011792 / 1 | 0.009017 / 1 |
| SH1 | nested | 1.008323 / 1 | 1.008122 / 1 | 0.996070 / 1 |
| SH3 | nested | 1.001259 / 1 | 1.011028 / 1 | 0.997874 / 1 |
| SH2 | nested | 1.015686 / 1 | 1.009876 / 1 | 0.998775 / 1 |
| shared-expert-during-read | boundary | 3.037051 / 1 | 3.041009 / 1 | 3.004912 / 1 |
| detail:expert-gate | nested | 49.358278 / 16 | 49.720632 / 16 | 54.373253 / 16 |
| detail:expert-up | nested | 42.799880 / 16 | 42.855835 / 16 | 53.842271 / 16 |
| detail:expert-activation | nested | 0.096643 / 16 | 0.095457 / 16 | 0.132867 / 16 |
| detail:expert-down | nested | 49.126774 / 16 | 49.102910 / 16 | 53.748703 / 16 |
| EUP | nested | 0.605501 / 1 | 0.600651 / 1 | 0.605401 / 1 |
| experts-mix-normalize-up | boundary | 142.429876 / 1 | 142.753914 / 1 | 163.120924 / 1 |
| mlp-merge-and-cleanup | boundary | 0.004138 / 1 | 0.003106 / 1 | 0.003166 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589000 / 1 | 0.588850 / 1 | 0.598408 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.941889 / 1 | 0.950445 / 1 | 0.980110 / 1 |
| worker:read | parallel worker | 0.066915 / 1 | 0.068185 / 1 | 0.069095 / 1 |
| worker:decode | parallel worker | 1071.663283 / 1 | 1078.263171 / 1 | 1087.076363 / 1 |
| worker:crc | parallel worker | 596.587002 / 1 | 600.261551 / 1 | 606.469378 / 1 |
| worker:math | parallel worker | 365.294301 / 1 | 364.586795 / 1 | 378.374772 / 1 |
| op:Q   int8 projection | nested | 16.383698 / 1 | 16.456771 / 1 | 16.269603 / 1 |
| op:X   mxfp4 expert proj | nested | 141.439357 / 1 | 141.831612 / 1 | 162.175300 / 1 |
| op:N   rmsnorm | nested | 0.043762 / 1 | 0.043471 / 1 | 0.031859 / 1 |
| op:L   l2 per-head | nested | 0.005520 / 1 | 0.005660 / 1 | 0.006192 / 1 |
| op:SiTU + sigma | nested | 0.103735 / 1 | 0.102683 / 1 | 0.139832 / 1 |
| op:C   shortconv | nested | 0.062036 / 1 | 0.069099 / 1 | 0.031659 / 1 |
| op:AR  snapshot aggregate | nested | 0.031199 / 1 | 0.031118 / 1 | 0.031770 / 1 |
| op:D   kda delta-rule | nested | 0.248684 / 1 | 0.238837 / 1 | 0.247783 / 1 |
| op:router dot product | nested | 0.566488 / 1 | 0.575315 / 1 | 0.573631 / 1 |
| op:top-k selection | nested | 0.002434 / 1 | 0.002214 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.072957 / 1 | 0.079078 / 1 | 0.058118 / 1 |
| detail:read-ahead-wait | nested | 0.001442 / 1 | 0.001643 / 1 | 0.001481 / 1 |
| total:layer | total | 161.043929 / 1 | 161.430061 / 1 | 181.667259 / 1 |

### Layer 47

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004548 / 1 | 0.004719 / 1 | 0.005551 / 1 |
| pre-attention-aggregation | boundary | 0.020498 / 1 | 0.016550 / 1 | 0.016972 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000110 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011993 / 1 | 0.011622 / 1 | 0.011361 / 1 |
| QA | nested | 0.360894 / 1 | 0.348000 / 1 | 0.351196 / 1 |
| QB | nested | 0.924767 / 1 | 0.875615 / 1 | 0.924577 / 1 |
| KA | nested | 0.138518 / 1 | 0.140242 / 1 | 0.140603 / 1 |
| KB | nested | 0.398655 / 1 | 0.389768 / 1 | 0.429563 / 1 |
| G | nested | 2.757349 / 1 | 2.768440 / 1 | 2.732783 / 1 |
| O | nested | 2.039459 / 1 | 2.041292 / 1 | 2.014732 / 1 |
| attention | boundary | 6.724147 / 1 | 6.679754 / 1 | 6.704571 / 1 |
| attention-residual | boundary | 0.002836 / 1 | 0.002555 / 1 | 0.002645 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024707 / 1 | 0.024836 / 1 | 0.024366 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001583 / 1 | 0.001613 / 1 | 0.001874 / 1 |
| router-and-top16 | boundary | 0.524259 / 1 | 0.503330 / 1 | 0.522426 / 1 |
| EDOWN | nested | 0.598387 / 1 | 0.597005 / 1 | 0.604869 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.599479 / 1 | 0.598036 / 1 | 0.605981 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.051296 / 1 | 0.021370 / 1 | 0.010750 / 1 |
| SH1 | nested | 1.020486 / 1 | 1.018883 / 1 | 1.024093 / 1 |
| SH3 | nested | 1.027729 / 1 | 1.018422 / 1 | 1.024433 / 1 |
| SH2 | nested | 1.025114 / 1 | 1.026257 / 1 | 1.021668 / 1 |
| shared-expert-during-read | boundary | 3.086985 / 1 | 3.075584 / 1 | 3.082196 / 1 |
| detail:expert-gate | nested | 49.997109 / 16 | 50.177696 / 16 | 54.449675 / 16 |
| detail:expert-up | nested | 42.818111 / 16 | 42.798007 / 16 | 54.478460 / 16 |
| detail:expert-activation | nested | 0.095628 / 16 | 0.095908 / 16 | 0.140945 / 16 |
| detail:expert-down | nested | 49.018579 / 16 | 49.090748 / 16 | 55.001548 / 16 |
| EUP | nested | 0.614718 / 1 | 0.605380 / 1 | 0.600531 / 1 |
| experts-mix-normalize-up | boundary | 142.964726 / 1 | 143.171313 / 1 | 165.081144 / 1 |
| mlp-merge-and-cleanup | boundary | 0.004208 / 1 | 0.003236 / 1 | 0.003025 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.598267 / 1 | 0.602455 / 1 | 0.595833 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.603647 / 1 | 0.608166 / 1 | 0.601994 / 1 |
| worker:read | parallel worker | 0.066144 / 1 | 0.068435 / 1 | 0.070937 / 1 |
| worker:decode | parallel worker | 1072.389347 / 1 | 1078.234990 / 1 | 1092.182032 / 1 |
| worker:crc | parallel worker | 597.764008 / 1 | 601.232267 / 1 | 609.280505 / 1 |
| worker:math | parallel worker | 363.886838 / 1 | 363.401940 / 1 | 380.244951 / 1 |
| op:Q   int8 projection | nested | 10.904794 / 1 | 10.828060 / 1 | 10.867846 / 1 |
| op:X   mxfp4 expert proj | nested | 141.987359 / 1 | 142.220806 / 1 | 164.155313 / 1 |
| op:N   rmsnorm | nested | 0.031760 / 1 | 0.032140 / 1 | 0.031540 / 1 |
| op:SiTU + sigma | nested | 0.104202 / 1 | 0.102932 / 1 | 0.147403 / 1 |
| op:AR  snapshot aggregate | nested | 0.034966 / 1 | 0.030707 / 1 | 0.031138 / 1 |
| op:SA  softmax attention | nested | 0.006231 / 1 | 0.008887 / 1 | 0.012103 / 1 |
| op:router dot product | nested | 0.521543 / 1 | 0.500906 / 1 | 0.520111 / 1 |
| op:top-k selection | nested | 0.002384 / 1 | 0.002094 / 1 | 0.001914 / 1 |
| detail:read-ahead-wait | nested | 0.001080 / 1 | 0.001123 / 1 | 0.000961 / 1 |
| total:layer | total | 154.639107 / 1 | 154.735840 / 1 | 176.688121 / 1 |

### Layer 48

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003917 / 1 | 0.004729 / 1 | 0.004989 / 1 |
| pre-attention-aggregation | boundary | 0.015389 / 1 | 0.015309 / 1 | 0.016070 / 1 |
| snapshot-push | boundary | 0.002174 / 1 | 0.001643 / 1 | 0.002344 / 1 |
| pre-attention-normalization | boundary | 0.011412 / 1 | 0.011351 / 1 | 0.012213 / 1 |
| Q | nested | 3.320051 / 1 | 2.938217 / 1 | 3.316133 / 1 |
| K | nested | 2.311708 / 1 | 2.335432 / 1 | 2.839123 / 1 |
| V | nested | 2.291500 / 1 | 2.314563 / 1 | 2.738243 / 1 |
| B | nested | 0.026549 / 1 | 0.028984 / 1 | 0.029615 / 1 |
| FA | nested | 0.028583 / 1 | 0.030627 / 1 | 0.038341 / 1 |
| FB | nested | 0.055002 / 1 | 0.053500 / 1 | 0.079248 / 1 |
| G | nested | 2.072290 / 1 | 2.105482 / 1 | 2.921176 / 1 |
| O | nested | 2.065598 / 1 | 2.091235 / 1 | 2.082329 / 1 |
| attention | boundary | 13.238232 / 1 | 13.014034 / 1 | 15.087035 / 1 |
| attention-residual | boundary | 0.002044 / 1 | 0.002194 / 1 | 0.001994 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024757 / 1 | 0.024416 / 1 | 0.024976 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001934 / 1 | 0.001994 / 1 | 0.001953 / 1 |
| router-and-top16 | boundary | 0.563042 / 1 | 0.571387 / 1 | 0.564714 / 1 |
| EDOWN | nested | 0.589220 / 1 | 0.605631 / 1 | 0.588058 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590403 / 1 | 0.606733 / 1 | 0.589160 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042499 / 1 | 0.013715 / 1 | 0.009387 / 1 |
| SH1 | nested | 1.398361 / 1 | 1.411877 / 1 | 0.971033 / 1 |
| SH3 | nested | 1.432045 / 1 | 1.468603 / 1 | 0.975973 / 1 |
| SH2 | nested | 1.383163 / 1 | 1.394985 / 1 | 0.974890 / 1 |
| shared-expert-during-read | boundary | 4.231482 / 1 | 4.294851 / 1 | 2.934319 / 1 |
| detail:expert-gate | nested | 49.730031 / 16 | 49.877418 / 16 | 54.026727 / 16 |
| detail:expert-up | nested | 42.799998 / 16 | 42.900106 / 16 | 53.217937 / 16 |
| detail:expert-activation | nested | 0.096459 / 16 | 0.096589 / 16 | 0.138230 / 16 |
| detail:expert-down | nested | 49.147434 / 16 | 49.200924 / 16 | 52.596754 / 16 |
| EUP | nested | 0.592837 / 1 | 0.589751 / 1 | 0.599038 / 1 |
| experts-mix-normalize-up | boundary | 142.781543 / 1 | 143.053534 / 1 | 160.957083 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003416 / 1 | 0.002996 / 1 | 0.003096 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.588248 / 1 | 0.589180 / 1 | 0.599700 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.949874 / 1 | 0.955605 / 1 | 0.988827 / 1 |
| worker:read | parallel worker | 0.207659 / 1 | 0.186430 / 1 | 0.175972 / 1 |
| worker:decode | parallel worker | 1075.207363 / 1 | 1080.817766 / 1 | 1084.378805 / 1 |
| worker:crc | parallel worker | 598.161238 / 1 | 601.471380 / 1 | 605.900133 / 1 |
| worker:math | parallel worker | 363.088019 / 1 | 363.428019 / 1 | 371.277326 / 1 |
| op:Q   int8 projection | nested | 17.565415 / 1 | 17.367264 / 1 | 18.151688 / 1 |
| op:X   mxfp4 expert proj | nested | 141.834329 / 1 | 142.136012 / 1 | 160.063714 / 1 |
| op:N   rmsnorm | nested | 0.049651 / 1 | 0.041045 / 1 | 0.031550 / 1 |
| op:L   l2 per-head | nested | 0.005420 / 1 | 0.005380 / 1 | 0.005300 / 1 |
| op:SiTU + sigma | nested | 0.108294 / 1 | 0.107378 / 1 | 0.145091 / 1 |
| op:C   shortconv | nested | 0.054041 / 1 | 0.061605 / 1 | 0.034585 / 1 |
| op:AR  snapshot aggregate | nested | 0.030456 / 1 | 0.030476 / 1 | 0.031519 / 1 |
| op:D   kda delta-rule | nested | 0.156272 / 1 | 0.151763 / 1 | 0.126626 / 1 |
| op:router dot product | nested | 0.560606 / 1 | 0.568702 / 1 | 0.562150 / 1 |
| op:top-k selection | nested | 0.002114 / 1 | 0.002294 / 1 | 0.002224 / 1 |
| op:alpha / beta / gate | nested | 0.072405 / 1 | 0.074449 / 1 | 0.057378 / 1 |
| detail:read-ahead-wait | nested | 0.001603 / 1 | 0.001743 / 1 | 0.001602 / 1 |
| total:layer | total | 162.476824 / 1 | 162.589066 / 1 | 181.212761 / 1 |

### Layer 49

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004638 / 1 | 0.004499 / 1 | 0.005109 / 1 |
| pre-attention-aggregation | boundary | 0.016942 / 1 | 0.016711 / 1 | 0.016761 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000040 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012513 / 1 | 0.011241 / 1 | 0.011471 / 1 |
| Q | nested | 3.282029 / 1 | 3.241814 / 1 | 3.270789 / 1 |
| K | nested | 2.338156 / 1 | 2.330492 / 1 | 2.335541 / 1 |
| V | nested | 2.307509 / 1 | 2.309794 / 1 | 2.296218 / 1 |
| B | nested | 0.021971 / 1 | 0.021790 / 1 | 0.020859 / 1 |
| FA | nested | 0.032861 / 1 | 0.035406 / 1 | 0.032671 / 1 |
| FB | nested | 0.052818 / 1 | 0.067536 / 1 | 0.051546 / 1 |
| G | nested | 2.051581 / 1 | 2.090323 / 1 | 2.046282 / 1 |
| O | nested | 2.065698 / 1 | 2.067270 / 1 | 2.064476 / 1 |
| attention | boundary | 13.297553 / 1 | 13.314575 / 1 | 13.310077 / 1 |
| attention-residual | boundary | 0.003837 / 1 | 0.003757 / 1 | 0.003396 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024526 / 1 | 0.024957 / 1 | 0.024446 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001603 / 1 | 0.001744 / 1 | 0.002034 / 1 |
| router-and-top16 | boundary | 0.572510 / 1 | 0.569434 / 1 | 0.574142 / 1 |
| EDOWN | nested | 0.592487 / 1 | 0.592948 / 1 | 0.591815 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.593559 / 1 | 0.594050 / 1 | 0.592927 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.053630 / 1 | 0.012073 / 1 | 0.016100 / 1 |
| SH1 | nested | 1.409172 / 1 | 1.003975 / 1 | 1.015326 / 1 |
| SH3 | nested | 1.423478 / 1 | 1.006971 / 1 | 1.017120 / 1 |
| SH2 | nested | 1.396297 / 1 | 1.008593 / 1 | 1.009465 / 1 |
| shared-expert-during-read | boundary | 4.245849 / 1 | 3.031932 / 1 | 3.053893 / 1 |
| detail:expert-gate | nested | 49.762040 / 16 | 50.054729 / 16 | 53.404947 / 16 |
| detail:expert-up | nested | 42.685534 / 16 | 42.760629 / 16 | 51.245231 / 16 |
| detail:expert-activation | nested | 0.097301 / 16 | 0.096080 / 16 | 0.128652 / 16 |
| detail:expert-down | nested | 49.033729 / 16 | 49.132315 / 16 | 52.618669 / 16 |
| EUP | nested | 0.610239 / 1 | 0.603217 / 1 | 0.593138 / 1 |
| experts-mix-normalize-up | boundary | 142.622055 / 1 | 143.043234 / 1 | 158.426817 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003216 / 1 | 0.002996 / 1 | 0.002975 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591605 / 1 | 0.592897 / 1 | 0.595142 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.939495 / 1 | 0.936259 / 1 | 0.965283 / 1 |
| worker:read | parallel worker | 0.066491 / 1 | 0.067359 / 1 | 0.070039 / 1 |
| worker:decode | parallel worker | 1073.133911 / 1 | 1077.007094 / 1 | 1088.312621 / 1 |
| worker:crc | parallel worker | 597.549048 / 1 | 601.492162 / 1 | 607.410316 / 1 |
| worker:math | parallel worker | 361.719360 / 1 | 363.335529 / 1 | 375.256174 / 1 |
| op:Q   int8 projection | nested | 17.582736 / 1 | 16.378407 / 1 | 16.343584 / 1 |
| op:X   mxfp4 expert proj | nested | 141.641816 / 1 | 142.106135 / 1 | 157.475062 / 1 |
| op:N   rmsnorm | nested | 0.040844 / 1 | 0.039594 / 1 | 0.031238 / 1 |
| op:L   l2 per-head | nested | 0.005320 / 1 | 0.005520 / 1 | 0.005761 / 1 |
| op:SiTU + sigma | nested | 0.108492 / 1 | 0.103686 / 1 | 0.135263 / 1 |
| op:C   shortconv | nested | 0.053881 / 1 | 0.053610 / 1 | 0.031840 / 1 |
| op:AR  snapshot aggregate | nested | 0.031810 / 1 | 0.031539 / 1 | 0.031699 / 1 |
| op:D   kda delta-rule | nested | 0.244877 / 1 | 0.248525 / 1 | 0.242473 / 1 |
| op:router dot product | nested | 0.569634 / 1 | 0.566518 / 1 | 0.571427 / 1 |
| op:top-k selection | nested | 0.002494 / 1 | 0.002504 / 1 | 0.002455 / 1 |
| op:alpha / beta / gate | nested | 0.083336 / 1 | 0.072205 / 1 | 0.058249 / 1 |
| detail:read-ahead-wait | nested | 0.001662 / 1 | 0.001541 / 1 | 0.001373 / 1 |
| total:layer | total | 162.411432 / 1 | 161.582176 / 1 | 177.020181 / 1 |

### Layer 50

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004549 / 1 | 0.004589 / 1 | 0.004909 / 1 |
| pre-attention-aggregation | boundary | 0.017122 / 1 | 0.016972 / 1 | 0.017313 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.012343 / 1 | 0.011422 / 1 | 0.011461 / 1 |
| Q | nested | 3.262433 / 1 | 3.252834 / 1 | 3.253356 / 1 |
| K | nested | 2.349678 / 1 | 2.352784 / 1 | 2.339349 / 1 |
| V | nested | 2.306667 / 1 | 2.323539 / 1 | 2.296699 / 1 |
| B | nested | 0.027622 / 1 | 0.022542 / 1 | 0.024396 / 1 |
| FA | nested | 0.039644 / 1 | 0.036228 / 1 | 0.034304 / 1 |
| FB | nested | 0.050284 / 1 | 0.049673 / 1 | 0.050644 / 1 |
| G | nested | 2.048716 / 1 | 2.079955 / 1 | 2.072891 / 1 |
| O | nested | 2.079794 / 1 | 2.087639 / 1 | 2.063604 / 1 |
| attention | boundary | 13.332468 / 1 | 13.375108 / 1 | 13.315648 / 1 |
| attention-residual | boundary | 0.003617 / 1 | 0.003587 / 1 | 0.003346 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024415 / 1 | 0.024787 / 1 | 0.024746 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001914 / 1 | 0.001703 / 1 | 0.002865 / 1 |
| router-and-top16 | boundary | 0.571097 / 1 | 0.582979 / 1 | 0.570565 / 1 |
| EDOWN | nested | 0.594230 / 1 | 0.598909 / 1 | 0.593568 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595282 / 1 | 0.600041 / 1 | 0.594691 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.040426 / 1 | 0.006142 / 1 | 0.015439 / 1 |
| SH1 | nested | 1.008363 / 1 | 1.014224 / 1 | 1.011139 / 1 |
| SH3 | nested | 1.002463 / 1 | 1.003855 / 1 | 1.004135 / 1 |
| SH2 | nested | 1.003745 / 1 | 1.011569 / 1 | 1.006269 / 1 |
| shared-expert-during-read | boundary | 3.026322 / 1 | 3.042011 / 1 | 3.033436 / 1 |
| detail:expert-gate | nested | 52.719924 / 16 | 49.764072 / 16 | 54.676890 / 16 |
| detail:expert-up | nested | 44.301704 / 16 | 42.960359 / 16 | 54.312760 / 16 |
| detail:expert-activation | nested | 0.098023 / 16 | 0.096590 / 16 | 0.130333 / 16 |
| detail:expert-down | nested | 51.077178 / 16 | 49.172220 / 16 | 52.705076 / 16 |
| EUP | nested | 0.607765 / 1 | 0.619417 / 1 | 0.730394 / 1 |
| experts-mix-normalize-up | boundary | 149.256706 / 1 | 142.998039 / 1 | 162.971174 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003316 / 1 | 0.002745 / 1 | 0.002444 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.595432 / 1 | 0.590333 / 1 | 0.605030 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.989768 / 1 | 0.966976 / 1 | 0.981413 / 1 |
| worker:read | parallel worker | 0.066444 / 1 | 0.067406 / 1 | 0.068567 / 1 |
| worker:decode | parallel worker | 1080.494236 / 1 | 1079.922975 / 1 | 1089.600798 / 1 |
| worker:crc | parallel worker | 598.338907 / 1 | 601.245811 / 1 | 607.427792 / 1 |
| worker:math | parallel worker | 367.160953 / 1 | 365.266718 / 1 | 375.681474 / 1 |
| op:Q   int8 projection | nested | 16.379981 / 1 | 16.451764 / 1 | 16.479307 / 1 |
| op:X   mxfp4 expert proj | nested | 148.263099 / 1 | 142.054013 / 1 | 161.908677 / 1 |
| op:N   rmsnorm | nested | 0.045725 / 1 | 0.044302 / 1 | 0.031318 / 1 |
| op:L   l2 per-head | nested | 0.005360 / 1 | 0.005320 / 1 | 0.005400 / 1 |
| op:SiTU + sigma | nested | 0.104776 / 1 | 0.104046 / 1 | 0.136734 / 1 |
| op:C   shortconv | nested | 0.063157 / 1 | 0.058579 / 1 | 0.031760 / 1 |
| op:AR  snapshot aggregate | nested | 0.031690 / 1 | 0.031969 / 1 | 0.032500 / 1 |
| op:D   kda delta-rule | nested | 0.243315 / 1 | 0.238716 / 1 | 0.245749 / 1 |
| op:router dot product | nested | 0.568472 / 1 | 0.580334 / 1 | 0.567880 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002204 / 1 | 0.002274 / 1 |
| op:alpha / beta / gate | nested | 0.072886 / 1 | 0.074048 / 1 | 0.058259 / 1 |
| detail:read-ahead-wait | nested | 0.001532 / 1 | 0.001714 / 1 | 0.001942 / 1 |
| total:layer | total | 167.893850 / 1 | 161.652407 / 1 | 181.565119 / 1 |

### Layer 51

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005140 / 1 | 0.005851 / 1 | 0.004879 / 1 |
| pre-attention-aggregation | boundary | 0.017433 / 1 | 0.018103 / 1 | 0.016731 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000110 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011241 / 1 | 0.012955 / 1 | 0.011772 / 1 |
| QA | nested | 0.349723 / 1 | 0.366334 / 1 | 0.347408 / 1 |
| QB | nested | 0.906423 / 1 | 1.131063 / 1 | 0.888409 / 1 |
| KA | nested | 0.135423 / 1 | 0.135373 / 1 | 0.133489 / 1 |
| KB | nested | 0.389257 / 1 | 0.372114 / 1 | 0.409384 / 1 |
| G | nested | 2.721131 / 1 | 2.080205 / 1 | 2.736180 / 1 |
| O | nested | 2.017288 / 1 | 2.091847 / 1 | 2.423456 / 1 |
| attention | boundary | 6.625042 / 1 | 6.297481 / 1 | 7.060245 / 1 |
| attention-residual | boundary | 0.002985 / 1 | 0.002956 / 1 | 0.002846 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024676 / 1 | 0.024926 / 1 | 0.024957 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001904 / 1 | 0.001813 / 1 | 0.001443 / 1 |
| router-and-top16 | boundary | 0.511856 / 1 | 0.521023 / 1 | 0.534268 / 1 |
| EDOWN | nested | 0.589261 / 1 | 0.603347 / 1 | 0.696360 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590393 / 1 | 0.604509 / 1 | 0.697703 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031679 / 1 | 0.014167 / 1 | 0.010930 / 1 |
| SH1 | nested | 1.011679 / 1 | 1.412528 / 1 | 1.192307 / 1 |
| SH3 | nested | 1.023361 / 1 | 1.409452 / 1 | 1.186626 / 1 |
| SH2 | nested | 1.016959 / 1 | 1.379616 / 1 | 1.213306 / 1 |
| shared-expert-during-read | boundary | 3.063761 / 1 | 4.217686 / 1 | 3.606726 / 1 |
| detail:expert-gate | nested | 49.629695 / 16 | 49.447103 / 16 | 54.139230 / 16 |
| detail:expert-up | nested | 42.938709 / 16 | 42.844771 / 16 | 53.641954 / 16 |
| detail:expert-activation | nested | 0.094387 / 16 | 0.095015 / 16 | 0.133277 / 16 |
| detail:expert-down | nested | 49.223215 / 16 | 49.206223 / 16 | 54.632378 / 16 |
| EUP | nested | 0.606223 / 1 | 0.604429 / 1 | 0.602194 / 1 |
| experts-mix-normalize-up | boundary | 142.894975 / 1 | 142.590699 / 1 | 163.555245 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002805 / 1 | 0.002675 / 1 | 0.003276 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.599830 / 1 | 0.599440 / 1 | 0.596374 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.608286 / 1 | 0.605210 / 1 | 0.602625 / 1 |
| worker:read | parallel worker | 0.066526 / 1 | 0.067549 / 1 | 0.066920 / 1 |
| worker:decode | parallel worker | 1075.747202 / 1 | 1080.698158 / 1 | 1088.361585 / 1 |
| worker:crc | parallel worker | 597.696666 / 1 | 600.495335 / 1 | 606.269378 / 1 |
| worker:math | parallel worker | 364.652271 / 1 | 363.490759 / 1 | 374.958646 / 1 |
| op:Q   int8 projection | nested | 10.765494 / 1 | 11.585086 / 1 | 11.827988 / 1 |
| op:X   mxfp4 expert proj | nested | 141.948639 / 1 | 141.655111 / 1 | 162.632002 / 1 |
| op:N   rmsnorm | nested | 0.032451 / 1 | 0.032301 / 1 | 0.031639 / 1 |
| op:SiTU + sigma | nested | 0.101470 / 1 | 0.105384 / 1 | 0.142445 / 1 |
| op:AR  snapshot aggregate | nested | 0.031469 / 1 | 0.032500 / 1 | 0.031539 / 1 |
| op:SA  softmax attention | nested | 0.006312 / 1 | 0.008305 / 1 | 0.013655 / 1 |
| op:router dot product | nested | 0.509060 / 1 | 0.518218 / 1 | 0.531452 / 1 |
| op:top-k selection | nested | 0.002464 / 1 | 0.002464 / 1 | 0.002495 / 1 |
| detail:read-ahead-wait | nested | 0.001520 / 1 | 0.001617 / 1 | 0.001595 / 1 |
| total:layer | total | 154.406744 / 1 | 154.936535 / 1 | 176.148523 / 1 |

### Layer 52

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004479 / 1 | 0.004538 / 1 | 0.004970 / 1 |
| pre-attention-aggregation | boundary | 0.020208 / 1 | 0.015889 / 1 | 0.015709 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000021 / 1 |
| pre-attention-normalization | boundary | 0.012002 / 1 | 0.011311 / 1 | 0.011000 / 1 |
| Q | nested | 3.192372 / 1 | 3.383107 / 1 | 3.303941 / 1 |
| K | nested | 2.303792 / 1 | 2.718357 / 1 | 2.314512 / 1 |
| V | nested | 2.269949 / 1 | 3.006304 / 1 | 2.288163 / 1 |
| B | nested | 0.020789 / 1 | 0.028253 / 1 | 0.021039 / 1 |
| FA | nested | 0.029805 / 1 | 0.034805 / 1 | 0.028774 / 1 |
| FB | nested | 0.050795 / 1 | 0.071063 / 1 | 0.049102 / 1 |
| G | nested | 2.060218 / 1 | 2.916516 / 1 | 2.050229 / 1 |
| O | nested | 2.083761 / 1 | 2.923900 / 1 | 2.070237 / 1 |
| attention | boundary | 13.151460 / 1 | 16.131897 / 1 | 13.186997 / 1 |
| attention-residual | boundary | 0.003656 / 1 | 0.003878 / 1 | 0.003576 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024747 / 1 | 0.025538 / 1 | 0.023985 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001794 / 1 | 0.002063 / 1 | 0.002094 / 1 |
| router-and-top16 | boundary | 0.557702 / 1 | 0.645235 / 1 | 0.569423 / 1 |
| EDOWN | nested | 0.594901 / 1 | 0.852853 / 1 | 0.589161 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.596083 / 1 | 0.853995 / 1 | 0.590393 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.055965 / 1 | 0.016902 / 1 | 0.010209 / 1 |
| SH1 | nested | 1.004186 / 1 | 1.457692 / 1 | 1.009585 / 1 |
| SH3 | nested | 1.006840 / 1 | 1.459606 / 1 | 1.004616 / 1 |
| SH2 | nested | 1.004977 / 1 | 1.446402 / 1 | 1.019945 / 1 |
| shared-expert-during-read | boundary | 3.027995 / 1 | 4.378928 / 1 | 3.046740 / 1 |
| detail:expert-gate | nested | 49.586506 / 16 | 50.616568 / 16 | 53.337894 / 16 |
| detail:expert-up | nested | 44.628624 / 16 | 44.628355 / 16 | 50.926897 / 16 |
| detail:expert-activation | nested | 0.096068 / 16 | 0.096921 / 16 | 0.120576 / 16 |
| detail:expert-down | nested | 50.667030 / 16 | 49.030464 / 16 | 52.401561 / 16 |
| EUP | nested | 0.602355 / 1 | 0.610760 / 1 | 0.722600 / 1 |
| experts-mix-normalize-up | boundary | 146.026653 / 1 | 145.384426 / 1 | 157.922004 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002735 / 1 | 0.002705 / 1 | 0.002475 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587166 / 1 | 0.589781 / 1 | 0.612975 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.965553 / 1 | 0.970031 / 1 | 1.003073 / 1 |
| worker:read | parallel worker | 0.067597 / 1 | 0.068575 / 1 | 0.070093 / 1 |
| worker:decode | parallel worker | 1071.204520 / 1 | 1078.689227 / 1 | 1085.433401 / 1 |
| worker:crc | parallel worker | 606.073789 / 1 | 609.025159 / 1 | 612.129379 / 1 |
| worker:math | parallel worker | 363.134977 / 1 | 363.615746 / 1 | 371.067841 / 1 |
| op:Q   int8 projection | nested | 16.223338 / 1 | 20.908005 / 1 | 16.470480 / 1 |
| op:X   mxfp4 expert proj | nested | 145.042767 / 1 | 144.436117 / 1 | 156.868308 / 1 |
| op:N   rmsnorm | nested | 0.046136 / 1 | 0.039645 / 1 | 0.030596 / 1 |
| op:L   l2 per-head | nested | 0.005400 / 1 | 0.005360 / 1 | 0.005430 / 1 |
| op:SiTU + sigma | nested | 0.103002 / 1 | 0.107511 / 1 | 0.127889 / 1 |
| op:C   shortconv | nested | 0.056636 / 1 | 0.066013 / 1 | 0.033653 / 1 |
| op:AR  snapshot aggregate | nested | 0.035516 / 1 | 0.031118 / 1 | 0.030598 / 1 |
| op:D   kda delta-rule | nested | 0.164457 / 1 | 0.108182 / 1 | 0.121137 / 1 |
| op:router dot product | nested | 0.554856 / 1 | 0.642140 / 1 | 0.566478 / 1 |
| op:top-k selection | nested | 0.002475 / 1 | 0.002635 / 1 | 0.002525 / 1 |
| op:alpha / beta / gate | nested | 0.076282 / 1 | 0.074318 / 1 | 0.057848 / 1 |
| detail:read-ahead-wait | nested | 0.001183 / 1 | 0.001192 / 1 | 0.001142 / 1 |
| total:layer | total | 164.466801 / 1 | 168.462044 / 1 | 176.407878 / 1 |

### Layer 53

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004819 / 1 | 0.005661 / 1 | 0.005370 / 1 |
| pre-attention-aggregation | boundary | 0.017092 / 1 | 0.016531 / 1 | 0.017112 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000120 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011742 / 1 | 0.011832 / 1 | 0.011892 / 1 |
| Q | nested | 3.280336 / 1 | 3.301886 / 1 | 3.246803 / 1 |
| K | nested | 2.337996 / 1 | 2.437432 / 1 | 2.744185 / 1 |
| V | nested | 2.292702 / 1 | 2.323309 / 1 | 2.805389 / 1 |
| B | nested | 0.028203 / 1 | 0.025357 / 1 | 0.026680 / 1 |
| FA | nested | 0.028033 / 1 | 0.033112 / 1 | 0.033763 / 1 |
| FB | nested | 0.051156 / 1 | 0.050864 / 1 | 0.050986 / 1 |
| G | nested | 2.062231 / 1 | 2.078301 / 1 | 2.399732 / 1 |
| O | nested | 2.065878 / 1 | 2.097687 / 1 | 2.426222 / 1 |
| attention | boundary | 13.294788 / 1 | 13.492327 / 1 | 14.940412 / 1 |
| attention-residual | boundary | 0.003657 / 1 | 0.003556 / 1 | 0.003898 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024586 / 1 | 0.025137 / 1 | 0.024366 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001843 / 1 | 0.001853 / 1 | 0.001964 / 1 |
| router-and-top16 | boundary | 0.575896 / 1 | 0.588238 / 1 | 0.591424 / 1 |
| EDOWN | nested | 0.667536 / 1 | 0.600391 / 1 | 0.689387 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.668940 / 1 | 0.601563 / 1 | 0.690600 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048601 / 1 | 0.021160 / 1 | 0.008296 / 1 |
| SH1 | nested | 1.436483 / 1 | 1.019073 / 1 | 1.192758 / 1 |
| SH3 | nested | 1.441172 / 1 | 1.018863 / 1 | 1.188419 / 1 |
| SH2 | nested | 1.387802 / 1 | 1.018261 / 1 | 1.207495 / 1 |
| shared-expert-during-read | boundary | 4.282708 / 1 | 3.068410 / 1 | 3.603209 / 1 |
| detail:expert-gate | nested | 49.814065 / 16 | 49.957004 / 16 | 53.732679 / 16 |
| detail:expert-up | nested | 43.044737 / 16 | 42.741153 / 16 | 52.929179 / 16 |
| detail:expert-activation | nested | 0.094108 / 16 | 0.097679 / 16 | 0.131757 / 16 |
| detail:expert-down | nested | 49.932708 / 16 | 49.163303 / 16 | 54.718250 / 16 |
| EUP | nested | 0.603497 / 1 | 0.596004 / 1 | 0.606593 / 1 |
| experts-mix-normalize-up | boundary | 143.921582 / 1 | 142.968034 / 1 | 162.742708 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.002906 / 1 | 0.002455 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589892 / 1 | 0.581966 / 1 | 0.598437 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.962678 / 1 | 0.978798 / 1 | 0.993877 / 1 |
| worker:read | parallel worker | 0.067412 / 1 | 0.069571 / 1 | 0.067457 / 1 |
| worker:decode | parallel worker | 1070.946871 / 1 | 1078.001976 / 1 | 1084.066636 / 1 |
| worker:crc | parallel worker | 599.144340 / 1 | 601.595526 / 1 | 604.873016 / 1 |
| worker:math | parallel worker | 363.799186 / 1 | 362.895367 / 1 | 372.199921 / 1 |
| op:Q   int8 projection | nested | 17.681574 / 1 | 16.599038 / 1 | 18.616989 / 1 |
| op:X   mxfp4 expert proj | nested | 142.949135 / 1 | 142.024242 / 1 | 161.598922 / 1 |
| op:N   rmsnorm | nested | 0.041307 / 1 | 0.039674 / 1 | 0.031740 / 1 |
| op:L   l2 per-head | nested | 0.005460 / 1 | 0.005320 / 1 | 0.005821 / 1 |
| op:SiTU + sigma | nested | 0.104998 / 1 | 0.104605 / 1 | 0.140545 / 1 |
| op:C   shortconv | nested | 0.058820 / 1 | 0.060564 / 1 | 0.031429 / 1 |
| op:AR  snapshot aggregate | nested | 0.031979 / 1 | 0.031529 / 1 | 0.032070 / 1 |
| op:D   kda delta-rule | nested | 0.246811 / 1 | 0.232665 / 1 | 0.251720 / 1 |
| op:router dot product | nested | 0.572750 / 1 | 0.585453 / 1 | 0.588609 / 1 |
| op:top-k selection | nested | 0.002825 / 1 | 0.002405 / 1 | 0.002395 / 1 |
| op:alpha / beta / gate | nested | 0.074228 / 1 | 0.073838 / 1 | 0.064591 / 1 |
| detail:read-ahead-wait | nested | 0.001302 / 1 | 0.001402 / 1 | 0.000990 / 1 |
| total:layer | total | 163.837736 / 1 | 161.802478 / 1 | 183.653009 / 1 |

### Layer 54

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004228 / 1 | 0.005190 / 1 | 0.005140 / 1 |
| pre-attention-aggregation | boundary | 0.016751 / 1 | 0.018325 / 1 | 0.017352 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012393 / 1 | 0.012874 / 1 | 0.012163 / 1 |
| Q | nested | 3.289743 / 1 | 2.805961 / 1 | 3.486842 / 1 |
| K | nested | 2.341463 / 1 | 2.430589 / 1 | 2.702146 / 1 |
| V | nested | 2.318870 / 1 | 2.329481 / 1 | 2.898754 / 1 |
| B | nested | 0.023484 / 1 | 0.023925 / 1 | 0.031058 / 1 |
| FA | nested | 0.028864 / 1 | 0.030266 / 1 | 0.035937 / 1 |
| FB | nested | 0.057487 / 1 | 0.051296 / 1 | 0.072455 / 1 |
| G | nested | 2.059356 / 1 | 2.106644 / 1 | 2.751218 / 1 |
| O | nested | 2.060408 / 1 | 2.092338 / 1 | 2.059196 / 1 |
| attention | boundary | 13.350512 / 1 | 13.085768 / 1 | 15.201640 / 1 |
| attention-residual | boundary | 0.003687 / 1 | 0.003516 / 1 | 0.003507 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024787 / 1 | 0.024496 / 1 | 0.024846 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001573 / 1 | 0.001754 / 1 | 0.001804 / 1 |
| router-and-top16 | boundary | 0.570275 / 1 | 0.587708 / 1 | 0.572189 / 1 |
| EDOWN | nested | 0.585133 / 1 | 0.596494 / 1 | 0.595522 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.586325 / 1 | 0.597676 / 1 | 0.596734 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.042720 / 1 | 0.017212 / 1 | 0.010059 / 1 |
| SH1 | nested | 1.430442 / 1 | 1.015146 / 1 | 0.993515 / 1 |
| SH3 | nested | 1.422126 / 1 | 1.029492 / 1 | 1.015065 / 1 |
| SH2 | nested | 1.405254 / 1 | 1.029132 / 1 | 1.011329 / 1 |
| shared-expert-during-read | boundary | 4.274853 / 1 | 3.086334 / 1 | 3.032263 / 1 |
| detail:expert-gate | nested | 49.040248 / 16 | 49.689717 / 16 | 55.108647 / 16 |
| detail:expert-up | nested | 42.614864 / 16 | 42.706812 / 16 | 53.997461 / 16 |
| detail:expert-activation | nested | 0.093543 / 16 | 0.096149 / 16 | 0.130183 / 16 |
| detail:expert-down | nested | 48.951274 / 16 | 49.120121 / 16 | 51.371466 / 16 |
| EUP | nested | 0.606382 / 1 | 0.606092 / 1 | 0.772452 / 1 |
| experts-mix-normalize-up | boundary | 141.739918 / 1 | 142.626116 / 1 | 161.800158 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002945 / 1 | 0.002986 / 1 | 0.002495 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591485 / 1 | 0.588789 / 1 | 0.645105 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.943272 / 1 | 0.957418 / 1 | 1.025796 / 1 |
| worker:read | parallel worker | 0.067950 / 1 | 0.069153 / 1 | 0.072867 / 1 |
| worker:decode | parallel worker | 1071.198205 / 1 | 1080.089477 / 1 | 1092.303035 / 1 |
| worker:crc | parallel worker | 597.402471 / 1 | 600.635082 / 1 | 609.078835 / 1 |
| worker:math | parallel worker | 361.916313 / 1 | 361.863057 / 1 | 377.523005 / 1 |
| op:Q   int8 projection | nested | 17.627569 / 1 | 16.145413 / 1 | 18.423747 / 1 |
| op:X   mxfp4 expert proj | nested | 140.765827 / 1 | 141.676812 / 1 | 160.699370 / 1 |
| op:N   rmsnorm | nested | 0.044323 / 1 | 0.051086 / 1 | 0.031499 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.005120 / 1 | 0.005029 / 1 |
| op:SiTU + sigma | nested | 0.103964 / 1 | 0.103662 / 1 | 0.136846 / 1 |
| op:C   shortconv | nested | 0.053009 / 1 | 0.066094 / 1 | 0.031409 / 1 |
| op:AR  snapshot aggregate | nested | 0.032010 / 1 | 0.033392 / 1 | 0.032451 / 1 |
| op:D   kda delta-rule | nested | 0.243916 / 1 | 0.237563 / 1 | 0.241551 / 1 |
| op:router dot product | nested | 0.567670 / 1 | 0.585142 / 1 | 0.569574 / 1 |
| op:top-k selection | nested | 0.002275 / 1 | 0.002154 / 1 | 0.002204 / 1 |
| op:alpha / beta / gate | nested | 0.074869 / 1 | 0.088626 / 1 | 0.057879 / 1 |
| detail:read-ahead-wait | nested | 0.001295 / 1 | 0.001452 / 1 | 0.001101 / 1 |
| total:layer | total | 161.592973 / 1 | 161.043970 / 1 | 182.322884 / 1 |

### Layer 55

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004729 / 1 | 0.004969 / 1 | 0.005411 / 1 |
| pre-attention-aggregation | boundary | 0.016771 / 1 | 0.017553 / 1 | 0.017403 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011331 / 1 | 0.011551 / 1 | 0.012132 / 1 |
| QA | nested | 0.361355 / 1 | 0.368979 / 1 | 0.388776 / 1 |
| QB | nested | 0.969080 / 1 | 0.911843 / 1 | 0.963339 / 1 |
| KA | nested | 0.137006 / 1 | 0.133459 / 1 | 0.137016 / 1 |
| KB | nested | 0.392824 / 1 | 0.415797 / 1 | 0.429903 / 1 |
| G | nested | 2.689733 / 1 | 2.787186 / 1 | 2.852698 / 1 |
| O | nested | 2.040029 / 1 | 2.136750 / 1 | 2.956481 / 1 |
| attention | boundary | 6.703509 / 1 | 6.876272 / 1 | 7.861521 / 1 |
| attention-residual | boundary | 0.002735 / 1 | 0.002705 / 1 | 0.003666 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024195 / 1 | 0.024406 / 1 | 0.029455 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001763 / 1 | 0.002074 / 1 | 0.002655 / 1 |
| router-and-top16 | boundary | 0.522575 / 1 | 0.518979 / 1 | 0.625548 / 1 |
| EDOWN | nested | 0.598368 / 1 | 0.607845 / 1 | 0.846341 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.599580 / 1 | 0.609047 / 1 | 0.847934 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.046126 / 1 | 0.017773 / 1 | 0.010850 / 1 |
| SH1 | nested | 1.020135 / 1 | 1.461389 / 1 | 1.462010 / 1 |
| SH3 | nested | 1.019694 / 1 | 1.457772 / 1 | 1.386049 / 1 |
| SH2 | nested | 1.033290 / 1 | 1.350652 / 1 | 1.398302 / 1 |
| shared-expert-during-read | boundary | 3.085202 / 1 | 4.286415 / 1 | 4.264584 / 1 |
| detail:expert-gate | nested | 49.805914 / 16 | 49.907143 / 16 | 55.733996 / 16 |
| detail:expert-up | nested | 45.160689 / 16 | 45.660001 / 16 | 55.044280 / 16 |
| detail:expert-activation | nested | 0.100577 / 16 | 0.101131 / 16 | 0.136825 / 16 |
| detail:expert-down | nested | 48.882946 / 16 | 49.002743 / 16 | 51.344240 / 16 |
| EUP | nested | 0.616040 / 1 | 0.595232 / 1 | 0.621210 / 1 |
| experts-mix-normalize-up | boundary | 145.006749 / 1 | 145.690628 / 1 | 163.281845 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003146 / 1 | 0.003085 / 1 | 0.003026 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.602175 / 1 | 0.596213 / 1 | 0.602796 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.608206 / 1 | 0.602515 / 1 | 0.608898 / 1 |
| worker:read | parallel worker | 0.064565 / 1 | 0.067234 / 1 | 0.071279 / 1 |
| worker:decode | parallel worker | 1070.913457 / 1 | 1080.092528 / 1 | 1092.565102 / 1 |
| worker:crc | parallel worker | 607.672860 / 1 | 609.217838 / 1 | 617.864426 / 1 |
| worker:math | parallel worker | 363.721084 / 1 | 362.822840 / 1 | 376.538686 / 1 |
| op:Q   int8 projection | nested | 10.876313 / 1 | 12.225623 / 1 | 13.440653 / 1 |
| op:X   mxfp4 expert proj | nested | 144.016571 / 1 | 144.740885 / 1 | 162.349594 / 1 |
| op:N   rmsnorm | nested | 0.032151 / 1 | 0.031250 / 1 | 0.033203 / 1 |
| op:SiTU + sigma | nested | 0.107662 / 1 | 0.111599 / 1 | 0.147143 / 1 |
| op:AR  snapshot aggregate | nested | 0.031058 / 1 | 0.031869 / 1 | 0.034344 / 1 |
| op:SA  softmax attention | nested | 0.006391 / 1 | 0.009357 / 1 | 0.012924 / 1 |
| op:router dot product | nested | 0.520011 / 1 | 0.515983 / 1 | 0.622483 / 1 |
| op:top-k selection | nested | 0.002214 / 1 | 0.002354 / 1 | 0.002655 / 1 |
| detail:read-ahead-wait | nested | 0.001541 / 1 | 0.001365 / 1 | 0.001834 / 1 |
| total:layer | total | 156.651996 / 1 | 158.684434 / 1 | 177.595005 / 1 |

### Layer 56

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004028 / 1 | 0.004648 / 1 | 0.004308 / 1 |
| pre-attention-aggregation | boundary | 0.015549 / 1 | 0.016220 / 1 | 0.015690 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011011 / 1 | 0.011311 / 1 | 0.011762 / 1 |
| Q | nested | 3.299171 / 1 | 3.283282 / 1 | 3.073630 / 1 |
| K | nested | 2.296919 / 1 | 2.379203 / 1 | 2.379033 / 1 |
| V | nested | 2.291880 / 1 | 2.312087 / 1 | 2.330432 / 1 |
| B | nested | 0.026820 / 1 | 0.024737 / 1 | 0.026259 / 1 |
| FA | nested | 0.034605 / 1 | 0.038843 / 1 | 0.037350 / 1 |
| FB | nested | 0.049533 / 1 | 0.051666 / 1 | 0.051296 / 1 |
| G | nested | 2.054777 / 1 | 2.084603 / 1 | 2.043837 / 1 |
| O | nested | 2.063814 / 1 | 2.105212 / 1 | 2.104299 / 1 |
| attention | boundary | 13.207705 / 1 | 13.348619 / 1 | 13.122076 / 1 |
| attention-residual | boundary | 0.004027 / 1 | 0.003526 / 1 | 0.003537 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024085 / 1 | 0.024215 / 1 | 0.024355 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001763 / 1 | 0.001783 / 1 | 0.001903 / 1 |
| router-and-top16 | boundary | 0.564885 / 1 | 0.584431 / 1 | 0.576366 / 1 |
| EDOWN | nested | 0.583931 / 1 | 0.593589 / 1 | 0.591445 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.585193 / 1 | 0.594811 / 1 | 0.592687 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.029234 / 1 | 0.016821 / 1 | 0.010049 / 1 |
| SH1 | nested | 1.476537 / 1 | 1.018031 / 1 | 1.003714 / 1 |
| SH3 | nested | 1.448375 / 1 | 1.019674 / 1 | 1.004025 / 1 |
| SH2 | nested | 1.383524 / 1 | 1.023291 / 1 | 1.001220 / 1 |
| shared-expert-during-read | boundary | 4.326038 / 1 | 3.073610 / 1 | 3.021182 / 1 |
| detail:expert-gate | nested | 49.170283 / 16 | 50.017125 / 16 | 53.488101 / 16 |
| detail:expert-up | nested | 42.607451 / 16 | 43.770494 / 16 | 51.051610 / 16 |
| detail:expert-activation | nested | 0.094223 / 16 | 0.100626 / 16 | 0.125916 / 16 |
| detail:expert-down | nested | 48.916662 / 16 | 49.461419 / 16 | 52.603731 / 16 |
| EUP | nested | 0.609698 / 1 | 0.594200 / 1 | 0.599139 / 1 |
| experts-mix-normalize-up | boundary | 141.808597 / 1 | 144.349403 / 1 | 158.266317 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002815 / 1 | 0.002655 / 1 | 0.002685 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587818 / 1 | 0.590032 / 1 | 0.590252 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.956596 / 1 | 0.988105 / 1 | 0.966294 / 1 |
| worker:read | parallel worker | 0.066116 / 1 | 0.065484 / 1 | 0.067612 / 1 |
| worker:decode | parallel worker | 1068.421870 / 1 | 1080.458605 / 1 | 1084.095840 / 1 |
| worker:crc | parallel worker | 597.371417 / 1 | 602.466009 / 1 | 604.149347 / 1 |
| worker:math | parallel worker | 362.304409 / 1 | 363.851670 / 1 | 370.228756 / 1 |
| op:Q   int8 projection | nested | 17.618093 / 1 | 16.526925 / 1 | 16.244288 / 1 |
| op:X   mxfp4 expert proj | nested | 140.854185 / 1 | 143.416431 / 1 | 157.353565 / 1 |
| op:N   rmsnorm | nested | 0.047861 / 1 | 0.038483 / 1 | 0.031448 / 1 |
| op:L   l2 per-head | nested | 0.005840 / 1 | 0.005821 / 1 | 0.005130 / 1 |
| op:SiTU + sigma | nested | 0.105374 / 1 | 0.107691 / 1 | 0.132397 / 1 |
| op:C   shortconv | nested | 0.048812 / 1 | 0.058860 / 1 | 0.034375 / 1 |
| op:AR  snapshot aggregate | nested | 0.029896 / 1 | 0.031328 / 1 | 0.030376 / 1 |
| op:D   kda delta-rule | nested | 0.147886 / 1 | 0.102612 / 1 | 0.126817 / 1 |
| op:router dot product | nested | 0.562240 / 1 | 0.581476 / 1 | 0.573481 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002565 / 1 | 0.002525 / 1 |
| op:alpha / beta / gate | nested | 0.072425 / 1 | 0.075060 / 1 | 0.057217 / 1 |
| detail:read-ahead-wait | nested | 0.001285 / 1 | 0.001222 / 1 | 0.001040 / 1 |
| total:layer | total | 161.558089 / 1 | 163.036341 / 1 | 176.635212 / 1 |

### Layer 57

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005641 / 1 | 0.005460 / 1 | 0.004859 / 1 |
| pre-attention-aggregation | boundary | 0.016741 / 1 | 0.016781 / 1 | 0.016782 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000110 / 1 |
| pre-attention-normalization | boundary | 0.011963 / 1 | 0.011632 / 1 | 0.011652 / 1 |
| Q | nested | 3.242516 / 1 | 3.349907 / 1 | 3.311183 / 1 |
| K | nested | 2.338537 / 1 | 2.409289 / 1 | 2.336133 / 1 |
| V | nested | 2.302871 / 1 | 2.293774 / 1 | 2.271392 / 1 |
| B | nested | 0.024305 / 1 | 0.028142 / 1 | 0.022923 / 1 |
| FA | nested | 0.026339 / 1 | 0.033272 / 1 | 0.026219 / 1 |
| FB | nested | 0.049843 / 1 | 0.050995 / 1 | 0.053610 / 1 |
| G | nested | 2.052953 / 1 | 2.097928 / 1 | 2.085645 / 1 |
| O | nested | 2.067080 / 1 | 2.095424 / 1 | 2.062942 / 1 |
| attention | boundary | 13.266165 / 1 | 13.499882 / 1 | 13.356384 / 1 |
| attention-residual | boundary | 0.003557 / 1 | 0.003457 / 1 | 0.003607 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024125 / 1 | 0.024125 / 1 | 0.023954 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001874 / 1 | 0.002285 / 1 | 0.001954 / 1 |
| router-and-top16 | boundary | 0.569875 / 1 | 0.587778 / 1 | 0.572409 / 1 |
| EDOWN | nested | 0.590182 / 1 | 0.596935 / 1 | 0.588189 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.591435 / 1 | 0.598207 / 1 | 0.589471 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048771 / 1 | 0.041367 / 1 | 0.007023 / 1 |
| SH1 | nested | 1.464996 / 1 | 1.034181 / 1 | 1.016178 / 1 |
| SH3 | nested | 1.509099 / 1 | 1.027369 / 1 | 1.012541 / 1 |
| SH2 | nested | 1.056763 / 1 | 1.024183 / 1 | 1.005918 / 1 |
| shared-expert-during-read | boundary | 4.048961 / 1 | 3.098195 / 1 | 3.047482 / 1 |
| detail:expert-gate | nested | 49.217793 / 16 | 50.353302 / 16 | 54.121305 / 16 |
| detail:expert-up | nested | 42.667300 / 16 | 42.797835 / 16 | 50.845896 / 16 |
| detail:expert-activation | nested | 0.093235 / 16 | 0.097710 / 16 | 0.121840 / 16 |
| detail:expert-down | nested | 49.008524 / 16 | 49.154787 / 16 | 49.960215 / 16 |
| EUP | nested | 0.612363 / 1 | 0.598938 / 1 | 0.722800 / 1 |
| experts-mix-normalize-up | boundary | 142.034559 / 1 | 143.440086 / 1 | 156.181885 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002875 / 1 | 0.003427 / 1 | 0.003346 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589491 / 1 | 0.591274 / 1 | 0.612944 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.948040 / 1 | 0.971405 / 1 | 0.989278 / 1 |
| worker:read | parallel worker | 0.183245 / 1 | 0.199829 / 1 | 0.187400 / 1 |
| worker:decode | parallel worker | 1070.726939 / 1 | 1083.068187 / 1 | 1087.097108 / 1 |
| worker:crc | parallel worker | 598.569766 / 1 | 602.208273 / 1 | 605.947176 / 1 |
| worker:math | parallel worker | 362.622938 / 1 | 362.915534 / 1 | 370.727316 / 1 |
| op:Q   int8 projection | nested | 17.336395 / 1 | 16.638845 / 1 | 16.514242 / 1 |
| op:X   mxfp4 expert proj | nested | 141.053013 / 1 | 142.469371 / 1 | 155.133087 / 1 |
| op:N   rmsnorm | nested | 0.041218 / 1 | 0.038552 / 1 | 0.031677 / 1 |
| op:L   l2 per-head | nested | 0.006051 / 1 | 0.005771 / 1 | 0.005370 / 1 |
| op:SiTU + sigma | nested | 0.104755 / 1 | 0.104935 / 1 | 0.128683 / 1 |
| op:C   shortconv | nested | 0.054631 / 1 | 0.052347 / 1 | 0.031800 / 1 |
| op:AR  snapshot aggregate | nested | 0.031449 / 1 | 0.031249 / 1 | 0.031008 / 1 |
| op:D   kda delta-rule | nested | 0.244206 / 1 | 0.232475 / 1 | 0.242623 / 1 |
| op:router dot product | nested | 0.567179 / 1 | 0.585193 / 1 | 0.569573 / 1 |
| op:top-k selection | nested | 0.002334 / 1 | 0.002114 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.078907 / 1 | 0.072525 / 1 | 0.058128 / 1 |
| detail:read-ahead-wait | nested | 0.001362 / 1 | 0.001174 / 1 | 0.001176 / 1 |
| total:layer | total | 161.590630 / 1 | 162.320114 / 1 | 174.826064 / 1 |

### Layer 58

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005220 / 1 | 0.004318 / 1 | 0.005079 / 1 |
| pre-attention-aggregation | boundary | 0.017322 / 1 | 0.017192 / 1 | 0.016862 / 1 |
| snapshot-push | boundary | 0.000101 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.012193 / 1 | 0.011231 / 1 | 0.011742 / 1 |
| Q | nested | 3.282219 / 1 | 3.357450 / 1 | 3.308198 / 1 |
| K | nested | 2.359106 / 1 | 2.385295 / 1 | 2.761528 / 1 |
| V | nested | 2.291940 / 1 | 2.289465 / 1 | 2.721051 / 1 |
| B | nested | 0.023284 / 1 | 0.022552 / 1 | 0.029916 / 1 |
| FA | nested | 0.029526 / 1 | 0.029575 / 1 | 0.030917 / 1 |
| FB | nested | 0.049042 / 1 | 0.049282 / 1 | 0.056726 / 1 |
| G | nested | 2.055068 / 1 | 2.106464 / 1 | 2.402587 / 1 |
| O | nested | 2.063834 / 1 | 2.108568 / 1 | 2.439436 / 1 |
| attention | boundary | 13.351213 / 1 | 13.505381 / 1 | 14.950471 / 1 |
| attention-residual | boundary | 0.003697 / 1 | 0.004078 / 1 | 0.003727 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024085 / 1 | 0.024916 / 1 | 0.024616 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002114 / 1 | 0.001813 / 1 | 0.001964 / 1 |
| router-and-top16 | boundary | 0.567500 / 1 | 0.587086 / 1 | 0.591535 / 1 |
| EDOWN | nested | 0.599780 / 1 | 0.603317 / 1 | 0.684559 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.601072 / 1 | 0.604559 / 1 | 0.685841 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031108 / 1 | 0.006362 / 1 | 0.023243 / 1 |
| SH1 | nested | 1.012280 / 1 | 1.030264 / 1 | 1.188319 / 1 |
| SH3 | nested | 1.007131 / 1 | 1.023591 / 1 | 1.178020 / 1 |
| SH2 | nested | 1.014975 / 1 | 1.020696 / 1 | 1.195503 / 1 |
| shared-expert-during-read | boundary | 3.046840 / 1 | 3.087837 / 1 | 3.577150 / 1 |
| detail:expert-gate | nested | 49.604598 / 16 | 50.370088 / 16 | 53.955053 / 16 |
| detail:expert-up | nested | 42.635258 / 16 | 42.906639 / 16 | 52.745234 / 16 |
| detail:expert-activation | nested | 0.098505 / 16 | 0.098935 / 16 | 0.135262 / 16 |
| detail:expert-down | nested | 48.999549 / 16 | 49.168140 / 16 | 53.995812 / 16 |
| EUP | nested | 0.599028 / 1 | 0.601573 / 1 | 0.606383 / 1 |
| experts-mix-normalize-up | boundary | 142.354046 / 1 | 143.547105 / 1 | 161.880307 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002825 / 1 | 0.003226 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589270 / 1 | 0.589911 / 1 | 0.597365 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.940686 / 1 | 0.966364 / 1 | 0.971794 / 1 |
| worker:read | parallel worker | 0.067746 / 1 | 0.069462 / 1 | 0.069547 / 1 |
| worker:decode | parallel worker | 1071.825181 / 1 | 1082.923927 / 1 | 1088.823163 / 1 |
| worker:crc | parallel worker | 598.376134 / 1 | 602.586019 / 1 | 606.723960 / 1 |
| worker:math | parallel worker | 362.345476 / 1 | 363.620666 / 1 | 372.451970 / 1 |
| op:Q   int8 projection | nested | 16.385820 / 1 | 16.626669 / 1 | 18.601679 / 1 |
| op:X   mxfp4 expert proj | nested | 141.406737 / 1 | 142.612388 / 1 | 160.925063 / 1 |
| op:N   rmsnorm | nested | 0.051086 / 1 | 0.043411 / 1 | 0.031429 / 1 |
| op:L   l2 per-head | nested | 0.005350 / 1 | 0.006201 / 1 | 0.005751 / 1 |
| op:SiTU + sigma | nested | 0.105326 / 1 | 0.106608 / 1 | 0.144498 / 1 |
| op:C   shortconv | nested | 0.059822 / 1 | 0.058278 / 1 | 0.031880 / 1 |
| op:AR  snapshot aggregate | nested | 0.032290 / 1 | 0.032300 / 1 | 0.031749 / 1 |
| op:D   kda delta-rule | nested | 0.242152 / 1 | 0.229569 / 1 | 0.249346 / 1 |
| op:router dot product | nested | 0.564654 / 1 | 0.584491 / 1 | 0.589140 / 1 |
| op:top-k selection | nested | 0.002234 / 1 | 0.001964 / 1 | 0.002014 / 1 |
| op:alpha / beta / gate | nested | 0.074239 / 1 | 0.074319 / 1 | 0.057657 / 1 |
| detail:read-ahead-wait | nested | 0.001094 / 1 | 0.001184 / 1 | 0.001061 / 1 |
| total:layer | total | 160.976933 / 1 | 162.388031 / 1 | 182.762916 / 1 |

### Layer 59

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004949 / 1 | 0.005129 / 1 | 0.004909 / 1 |
| pre-attention-aggregation | boundary | 0.017022 / 1 | 0.016711 / 1 | 0.017092 / 1 |
| snapshot-push | boundary | 0.000091 / 1 | 0.000020 / 1 | 0.000080 / 1 |
| pre-attention-normalization | boundary | 0.011531 / 1 | 0.011581 / 1 | 0.011261 / 1 |
| QA | nested | 0.360774 / 1 | 0.350264 / 1 | 0.361745 / 1 |
| QB | nested | 0.925438 / 1 | 0.912133 / 1 | 0.920990 / 1 |
| KA | nested | 0.139651 / 1 | 0.136715 / 1 | 0.145222 / 1 |
| KB | nested | 0.411639 / 1 | 0.381061 / 1 | 0.406159 / 1 |
| G | nested | 2.756538 / 1 | 2.809327 / 1 | 2.749305 / 1 |
| O | nested | 2.031123 / 1 | 2.111153 / 1 | 2.021865 / 1 |
| attention | boundary | 6.741360 / 1 | 6.812432 / 1 | 6.722595 / 1 |
| attention-residual | boundary | 0.003066 / 1 | 0.002905 / 1 | 0.002816 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024526 / 1 | 0.025287 / 1 | 0.024707 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001794 / 1 | 0.001773 / 1 | 0.002044 / 1 |
| router-and-top16 | boundary | 0.522887 / 1 | 0.529529 / 1 | 0.527054 / 1 |
| EDOWN | nested | 0.589561 / 1 | 0.624787 / 1 | 0.593098 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590964 / 1 | 0.626320 / 1 | 0.594330 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.052288 / 1 | 0.014598 / 1 | 0.014167 / 1 |
| SH1 | nested | 1.019794 / 1 | 1.043138 / 1 | 1.008753 / 1 |
| SH3 | nested | 1.019204 / 1 | 1.032118 / 1 | 1.012631 / 1 |
| SH2 | nested | 1.024744 / 1 | 1.029834 / 1 | 1.019264 / 1 |
| shared-expert-during-read | boundary | 3.076455 / 1 | 3.117492 / 1 | 3.053562 / 1 |
| detail:expert-gate | nested | 49.935857 / 16 | 50.278595 / 16 | 53.711128 / 16 |
| detail:expert-up | nested | 42.675912 / 16 | 42.827080 / 16 | 50.809648 / 16 |
| detail:expert-activation | nested | 0.094808 / 16 | 0.097735 / 16 | 0.128991 / 16 |
| detail:expert-down | nested | 49.029148 / 16 | 49.146773 / 16 | 52.452346 / 16 |
| EUP | nested | 0.603577 / 1 | 0.605130 / 1 | 0.720676 / 1 |
| experts-mix-normalize-up | boundary | 142.782957 / 1 | 143.362280 / 1 | 158.221775 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002946 / 1 | 0.002995 / 1 | 0.003176 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.600742 / 1 | 0.596844 / 1 | 0.610981 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.606612 / 1 | 0.603156 / 1 | 0.617573 / 1 |
| worker:read | parallel worker | 0.065451 / 1 | 0.066548 / 1 | 0.067592 / 1 |
| worker:decode | parallel worker | 1071.784397 / 1 | 1082.334513 / 1 | 1087.675482 / 1 |
| worker:crc | parallel worker | 597.952554 / 1 | 599.849208 / 1 | 604.729987 / 1 |
| worker:math | parallel worker | 363.229740 / 1 | 362.711483 / 1 | 371.679528 / 1 |
| op:Q   int8 projection | nested | 10.880830 / 1 | 11.034527 / 1 | 10.958465 / 1 |
| op:X   mxfp4 expert proj | nested | 141.809660 / 1 | 142.419859 / 1 | 157.190928 / 1 |
| op:N   rmsnorm | nested | 0.031327 / 1 | 0.031650 / 1 | 0.031788 / 1 |
| op:SiTU + sigma | nested | 0.102302 / 1 | 0.105046 / 1 | 0.136264 / 1 |
| op:AR  snapshot aggregate | nested | 0.031418 / 1 | 0.031629 / 1 | 0.031589 / 1 |
| op:SA  softmax attention | nested | 0.006632 / 1 | 0.009608 / 1 | 0.012423 / 1 |
| op:router dot product | nested | 0.519821 / 1 | 0.526713 / 1 | 0.523758 / 1 |
| op:top-k selection | nested | 0.002725 / 1 | 0.002404 / 1 | 0.002705 / 1 |
| detail:read-ahead-wait | nested | 0.001294 / 1 | 0.001335 / 1 | 0.001071 / 1 |
| total:layer | total | 154.455755 / 1 | 155.152007 / 1 | 169.833269 / 1 |

### Layer 60

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004308 / 1 | 0.004258 / 1 | 0.004709 / 1 |
| pre-attention-aggregation | boundary | 0.015499 / 1 | 0.016291 / 1 | 0.015770 / 1 |
| snapshot-push | boundary | 0.001983 / 1 | 0.002134 / 1 | 0.002104 / 1 |
| pre-attention-normalization | boundary | 0.011792 / 1 | 0.011602 / 1 | 0.012133 / 1 |
| Q | nested | 3.182774 / 1 | 3.311905 / 1 | 3.074141 / 1 |
| K | nested | 3.101041 / 1 | 2.395875 / 1 | 2.732733 / 1 |
| V | nested | 3.019438 / 1 | 2.309984 / 1 | 2.670107 / 1 |
| B | nested | 0.027020 / 1 | 0.028683 / 1 | 0.031328 / 1 |
| FA | nested | 0.033803 / 1 | 0.026880 / 1 | 0.033803 / 1 |
| FB | nested | 0.051627 / 1 | 0.055013 / 1 | 0.058549 / 1 |
| G | nested | 2.080565 / 1 | 2.092488 / 1 | 2.409199 / 1 |
| O | nested | 2.044579 / 1 | 2.141559 / 1 | 2.368603 / 1 |
| attention | boundary | 14.530095 / 1 | 13.435731 / 1 | 14.463621 / 1 |
| attention-residual | boundary | 0.002785 / 1 | 0.002134 / 1 | 0.002113 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025868 / 1 | 0.025618 / 1 | 0.024536 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002064 / 1 | 0.001654 / 1 | 0.001623 / 1 |
| router-and-top16 | boundary | 0.551760 / 1 | 0.590233 / 1 | 0.579142 / 1 |
| EDOWN | nested | 0.582418 / 1 | 0.592686 / 1 | 0.701961 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.583760 / 1 | 0.594009 / 1 | 0.703283 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.023634 / 1 | 0.014026 / 1 | 0.006913 / 1 |
| SH1 | nested | 1.397980 / 1 | 1.572286 / 1 | 1.182499 / 1 |
| SH3 | nested | 1.429309 / 1 | 1.483530 / 1 | 1.164625 / 1 |
| SH2 | nested | 1.386910 / 1 | 1.047747 / 1 | 1.174032 / 1 |
| shared-expert-during-read | boundary | 4.232253 / 1 | 4.116838 / 1 | 3.536385 / 1 |
| detail:expert-gate | nested | 49.812743 / 16 | 51.031912 / 16 | 53.564494 / 16 |
| detail:expert-up | nested | 43.407503 / 16 | 46.129317 / 16 | 52.212658 / 16 |
| detail:expert-activation | nested | 0.094208 / 16 | 0.101458 / 16 | 0.134320 / 16 |
| detail:expert-down | nested | 49.737857 / 16 | 50.430457 / 16 | 54.467150 / 16 |
| EUP | nested | 0.600001 / 1 | 0.594270 / 1 | 0.604088 / 1 |
| experts-mix-normalize-up | boundary | 144.060092 / 1 | 148.701150 / 1 | 161.389310 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003166 / 1 | 0.002966 / 1 | 0.002755 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589621 / 1 | 0.597736 / 1 | 0.589571 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.938613 / 1 | 0.996101 / 1 | 0.977456 / 1 |
| worker:read | parallel worker | 0.200119 / 1 | 0.193159 / 1 | 0.174937 / 1 |
| worker:decode | parallel worker | 1063.825637 / 1 | 1079.621195 / 1 | 1083.014852 / 1 |
| worker:crc | parallel worker | 599.437456 / 1 | 606.856296 / 1 | 607.889695 / 1 |
| worker:math | parallel worker | 363.949227 / 1 | 368.024210 / 1 | 372.502410 / 1 |
| op:Q   int8 projection | nested | 18.935933 / 1 | 17.651193 / 1 | 18.204248 / 1 |
| op:X   mxfp4 expert proj | nested | 143.122298 / 1 | 147.766163 / 1 | 160.475165 / 1 |
| op:N   rmsnorm | nested | 0.052608 / 1 | 0.039936 / 1 | 0.031968 / 1 |
| op:L   l2 per-head | nested | 0.005411 / 1 | 0.005310 / 1 | 0.005421 / 1 |
| op:SiTU + sigma | nested | 0.105118 / 1 | 0.108893 / 1 | 0.143217 / 1 |
| op:C   shortconv | nested | 0.051126 / 1 | 0.071733 / 1 | 0.036268 / 1 |
| op:AR  snapshot aggregate | nested | 0.031068 / 1 | 0.032050 / 1 | 0.030767 / 1 |
| op:D   kda delta-rule | nested | 0.147846 / 1 | 0.107060 / 1 | 0.132768 / 1 |
| op:router dot product | nested | 0.548684 / 1 | 0.587146 / 1 | 0.576576 / 1 |
| op:top-k selection | nested | 0.002715 / 1 | 0.002775 / 1 | 0.002154 / 1 |
| op:alpha / beta / gate | nested | 0.072315 / 1 | 0.072296 / 1 | 0.057367 / 1 |
| detail:read-ahead-wait | nested | 0.001112 / 1 | 0.001492 / 1 | 0.001121 / 1 |
| total:layer | total | 165.004394 / 1 | 168.531895 / 1 | 181.740166 / 1 |

### Layer 61

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004849 / 1 | 0.005451 / 1 | 0.005901 / 1 |
| pre-attention-aggregation | boundary | 0.017122 / 1 | 0.017433 / 1 | 0.017643 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000030 / 1 | 0.000040 / 1 |
| pre-attention-normalization | boundary | 0.011522 / 1 | 0.011491 / 1 | 0.011822 / 1 |
| Q | nested | 3.470751 / 1 | 3.308259 / 1 | 3.337984 / 1 |
| K | nested | 3.042742 / 1 | 2.457399 / 1 | 2.825867 / 1 |
| V | nested | 2.816169 / 1 | 2.383952 / 1 | 2.735328 / 1 |
| B | nested | 0.025207 / 1 | 0.029025 / 1 | 0.030927 / 1 |
| FA | nested | 0.028844 / 1 | 0.031949 / 1 | 0.037049 / 1 |
| FB | nested | 0.049782 / 1 | 0.050615 / 1 | 0.068507 / 1 |
| G | nested | 2.056982 / 1 | 2.114499 / 1 | 2.820828 / 1 |
| O | nested | 2.066099 / 1 | 2.118947 / 1 | 2.073893 / 1 |
| attention | boundary | 14.738424 / 1 | 13.615747 / 1 | 15.085924 / 1 |
| attention-residual | boundary | 0.003837 / 1 | 0.004298 / 1 | 0.003627 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024706 / 1 | 0.026359 / 1 | 0.024747 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001994 / 1 | 0.001784 / 1 | 0.001984 / 1 |
| router-and-top16 | boundary | 0.572990 / 1 | 0.594110 / 1 | 0.576236 / 1 |
| EDOWN | nested | 0.603136 / 1 | 0.610140 / 1 | 0.586986 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.604469 / 1 | 0.611472 / 1 | 0.588309 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.028494 / 1 | 0.034164 / 1 | 0.011662 / 1 |
| SH1 | nested | 1.405024 / 1 | 1.590260 / 1 | 1.461689 / 1 |
| SH3 | nested | 1.434980 / 1 | 1.553060 / 1 | 1.539184 / 1 |
| SH2 | nested | 1.393563 / 1 | 1.125322 / 1 | 1.328972 / 1 |
| shared-expert-during-read | boundary | 4.251319 / 1 | 4.286224 / 1 | 4.349964 / 1 |
| detail:expert-gate | nested | 48.994317 / 16 | 48.773366 / 16 | 53.371893 / 16 |
| detail:expert-up | nested | 42.735876 / 16 | 42.727582 / 16 | 51.811989 / 16 |
| detail:expert-activation | nested | 0.094910 / 16 | 0.097212 / 16 | 0.125384 / 16 |
| detail:expert-down | nested | 48.931389 / 16 | 49.075990 / 16 | 51.933115 / 16 |
| EUP | nested | 0.610971 / 1 | 0.602245 / 1 | 0.609869 / 1 |
| experts-mix-normalize-up | boundary | 141.779763 / 1 | 141.708872 / 1 | 161.596578 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002946 / 1 | 0.002555 / 1 | 0.002916 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587387 / 1 | 0.590814 / 1 | 0.597636 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.961856 / 1 | 0.974971 / 1 | 0.921420 / 1 |
| worker:read | parallel worker | 0.184284 / 1 | 0.196350 / 1 | 0.191316 / 1 |
| worker:decode | parallel worker | 1065.918081 / 1 | 1075.245674 / 1 | 1082.314264 / 1 |
| worker:crc | parallel worker | 597.282356 / 1 | 600.520449 / 1 | 606.575580 / 1 |
| worker:math | parallel worker | 364.340183 / 1 | 363.098812 / 1 | 375.287435 / 1 |
| op:Q   int8 projection | nested | 19.002746 / 1 | 17.974018 / 1 | 19.455442 / 1 |
| op:X   mxfp4 expert proj | nested | 140.827514 / 1 | 140.743531 / 1 | 157.334688 / 1 |
| op:N   rmsnorm | nested | 0.038293 / 1 | 0.042198 / 1 | 0.032269 / 1 |
| op:L   l2 per-head | nested | 0.005270 / 1 | 0.005491 / 1 | 0.005631 / 1 |
| op:SiTU + sigma | nested | 0.105819 / 1 | 0.107882 / 1 | 0.136756 / 1 |
| op:C   shortconv | nested | 0.059030 / 1 | 0.052508 / 1 | 0.032972 / 1 |
| op:AR  snapshot aggregate | nested | 0.032261 / 1 | 0.033993 / 1 | 0.032662 / 1 |
| op:D   kda delta-rule | nested | 0.241110 / 1 | 0.245198 / 1 | 0.241992 / 1 |
| op:router dot product | nested | 0.570295 / 1 | 0.591434 / 1 | 0.573501 / 1 |
| op:top-k selection | nested | 0.002284 / 1 | 0.002244 / 1 | 0.002345 / 1 |
| op:alpha / beta / gate | nested | 0.073488 / 1 | 0.073136 / 1 | 0.058549 / 1 |
| detail:read-ahead-wait | nested | 0.001241 / 1 | 0.001444 / 1 | 3.324257 / 1 |
| total:layer | total | 163.024036 / 1 | 161.912863 / 1 | 183.217926 / 1 |

### Layer 62

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005070 / 1 | 0.004969 / 1 | 0.004719 / 1 |
| pre-attention-aggregation | boundary | 0.017853 / 1 | 0.017953 / 1 | 0.017523 / 1 |
| snapshot-push | boundary | 0.000111 / 1 | 0.000110 / 1 | 0.000021 / 1 |
| pre-attention-normalization | boundary | 0.011742 / 1 | 0.011211 / 1 | 0.011722 / 1 |
| Q | nested | 3.472645 / 1 | 3.298210 / 1 | 3.491551 / 1 |
| K | nested | 2.827780 / 1 | 2.436260 / 1 | 2.721232 / 1 |
| V | nested | 2.303481 / 1 | 2.334590 / 1 | 2.878426 / 1 |
| B | nested | 0.030867 / 1 | 0.027531 / 1 | 0.031088 / 1 |
| FA | nested | 0.027822 / 1 | 0.030608 / 1 | 0.037009 / 1 |
| FB | nested | 0.049352 / 1 | 0.052157 / 1 | 0.071624 / 1 |
| G | nested | 2.091486 / 1 | 2.094421 / 1 | 2.680256 / 1 |
| O | nested | 2.079313 / 1 | 2.092738 / 1 | 2.057112 / 1 |
| attention | boundary | 14.075245 / 1 | 13.507155 / 1 | 15.107724 / 1 |
| attention-residual | boundary | 0.003647 / 1 | 0.003456 / 1 | 0.004137 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025137 / 1 | 0.025959 / 1 | 0.025087 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002444 / 1 | 0.001973 / 1 | 0.002374 / 1 |
| router-and-top16 | boundary | 0.576547 / 1 | 0.587247 / 1 | 0.573692 / 1 |
| EDOWN | nested | 0.594300 / 1 | 0.605851 / 1 | 0.594660 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595893 / 1 | 0.607124 / 1 | 0.595953 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.048271 / 1 | 0.006282 / 1 | 0.011802 / 1 |
| SH1 | nested | 1.449627 / 1 | 1.030164 / 1 | 1.267958 / 1 |
| SH3 | nested | 1.459335 / 1 | 1.029112 / 1 | 1.541329 / 1 |
| SH2 | nested | 1.124740 / 1 | 1.017630 / 1 | 1.570663 / 1 |
| shared-expert-during-read | boundary | 4.053741 / 1 | 3.089570 / 1 | 4.399095 / 1 |
| detail:expert-gate | nested | 49.641327 / 16 | 49.751207 / 16 | 52.763409 / 16 |
| detail:expert-up | nested | 42.571450 / 16 | 42.685717 / 16 | 49.477490 / 16 |
| detail:expert-activation | nested | 0.095218 / 16 | 0.097231 / 16 | 0.121737 / 16 |
| detail:expert-down | nested | 48.856670 / 16 | 48.954885 / 16 | 52.844308 / 16 |
| EUP | nested | 0.603257 / 1 | 0.603707 / 1 | 0.598388 / 1 |
| experts-mix-normalize-up | boundary | 142.210477 / 1 | 142.487497 / 1 | 160.895889 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003417 / 1 | 0.002534 / 1 | 0.003166 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.592476 / 1 | 0.588359 / 1 | 0.593378 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.946527 / 1 | 0.947750 / 1 | 0.964030 / 1 |
| worker:read | parallel worker | 0.064216 / 1 | 0.069380 / 1 | 0.066392 / 1 |
| worker:decode | parallel worker | 1067.119500 / 1 | 1073.525028 / 1 | 1077.392638 / 1 |
| worker:crc | parallel worker | 598.755628 / 1 | 602.025075 / 1 | 604.054982 / 1 |
| worker:math | parallel worker | 361.763405 / 1 | 363.166848 / 1 | 368.712958 / 1 |
| op:Q   int8 projection | nested | 18.112535 / 1 | 16.651428 / 1 | 19.539541 / 1 |
| op:X   mxfp4 expert proj | nested | 141.235528 / 1 | 141.558982 / 1 | 155.298507 / 1 |
| op:N   rmsnorm | nested | 0.044312 / 1 | 0.044765 / 1 | 0.032151 / 1 |
| op:L   l2 per-head | nested | 0.005460 / 1 | 0.005730 / 1 | 0.005981 / 1 |
| op:SiTU + sigma | nested | 0.107853 / 1 | 0.104455 / 1 | 0.131945 / 1 |
| op:C   shortconv | nested | 0.057486 / 1 | 0.063459 / 1 | 0.031158 / 1 |
| op:AR  snapshot aggregate | nested | 0.032991 / 1 | 0.033593 / 1 | 0.032731 / 1 |
| op:D   kda delta-rule | nested | 0.239698 / 1 | 0.241321 / 1 | 0.242132 / 1 |
| op:router dot product | nested | 0.573832 / 1 | 0.584281 / 1 | 0.571097 / 1 |
| op:top-k selection | nested | 0.002374 / 1 | 0.002584 / 1 | 0.002304 / 1 |
| op:alpha / beta / gate | nested | 0.073898 / 1 | 0.073728 / 1 | 0.058219 / 1 |
| detail:read-ahead-wait | nested | 0.001420 / 1 | 0.001432 / 1 | 4.694120 / 1 |
| total:layer | total | 162.595266 / 1 | 161.318513 / 1 | 182.636480 / 1 |

### Layer 63

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004739 / 1 | 0.004819 / 1 | 0.004919 / 1 |
| pre-attention-aggregation | boundary | 0.017523 / 1 | 0.017492 / 1 | 0.018063 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011281 / 1 | 0.014187 / 1 | 0.011372 / 1 |
| QA | nested | 0.365603 / 1 | 0.349703 / 1 | 0.351827 / 1 |
| QB | nested | 0.925408 / 1 | 0.949543 / 1 | 0.905932 / 1 |
| KA | nested | 0.136305 / 1 | 0.136084 / 1 | 0.135493 / 1 |
| KB | nested | 0.405928 / 1 | 0.421677 / 1 | 0.415475 / 1 |
| G | nested | 2.753502 / 1 | 2.819355 / 1 | 2.682089 / 1 |
| O | nested | 2.045209 / 1 | 2.134846 / 1 | 2.037174 / 1 |
| attention | boundary | 6.744225 / 1 | 6.928098 / 1 | 6.644810 / 1 |
| attention-residual | boundary | 0.002775 / 1 | 0.002785 / 1 | 0.002796 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024917 / 1 | 0.025528 / 1 | 0.024475 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001784 / 1 | 0.001893 / 1 | 0.001833 / 1 |
| router-and-top16 | boundary | 0.523988 / 1 | 0.507838 / 1 | 0.528167 / 1 |
| EDOWN | nested | 0.592827 / 1 | 0.618806 / 1 | 0.603226 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.594400 / 1 | 0.620169 / 1 | 0.604579 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031619 / 1 | 0.018054 / 1 | 0.010219 / 1 |
| SH1 | nested | 1.020436 / 1 | 1.460548 / 1 | 1.024263 / 1 |
| SH3 | nested | 1.023932 / 1 | 1.448325 / 1 | 1.031416 / 1 |
| SH2 | nested | 1.029783 / 1 | 1.398582 / 1 | 1.032138 / 1 |
| shared-expert-during-read | boundary | 3.086144 / 1 | 4.324857 / 1 | 3.100240 / 1 |
| detail:expert-gate | nested | 49.315798 / 16 | 48.648242 / 16 | 52.634677 / 16 |
| detail:expert-up | nested | 42.566843 / 16 | 42.644409 / 16 | 49.320699 / 16 |
| detail:expert-activation | nested | 0.094878 / 16 | 0.094626 / 16 | 0.122010 / 16 |
| detail:expert-down | nested | 48.799212 / 16 | 48.961715 / 16 | 51.362339 / 16 |
| EUP | nested | 0.608286 / 1 | 0.613797 / 1 | 0.731687 / 1 |
| experts-mix-normalize-up | boundary | 141.807334 / 1 | 141.370811 / 1 | 154.568502 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003146 / 1 | 0.002875 / 1 | 0.003046 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.604279 / 1 | 0.594790 / 1 | 0.622723 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.610630 / 1 | 0.600903 / 1 | 0.628824 / 1 |
| worker:read | parallel worker | 0.067159 / 1 | 0.065912 / 1 | 0.068372 / 1 |
| worker:decode | parallel worker | 1067.908488 / 1 | 1073.217067 / 1 | 1080.410718 / 1 |
| worker:crc | parallel worker | 597.867952 / 1 | 600.173534 / 1 | 604.910195 / 1 |
| worker:math | parallel worker | 361.906277 / 1 | 362.152140 / 1 | 371.408496 / 1 |
| op:Q   int8 projection | nested | 10.906016 / 1 | 12.349975 / 1 | 10.949387 / 1 |
| op:X   mxfp4 expert proj | nested | 140.849175 / 1 | 140.422794 / 1 | 153.529534 / 1 |
| op:N   rmsnorm | nested | 0.032071 / 1 | 0.035126 / 1 | 0.031698 / 1 |
| op:SiTU + sigma | nested | 0.101412 / 1 | 0.105237 / 1 | 0.128311 / 1 |
| op:AR  snapshot aggregate | nested | 0.032290 / 1 | 0.033012 / 1 | 0.032091 / 1 |
| op:SA  softmax attention | nested | 0.006643 / 1 | 0.008977 / 1 | 0.012614 / 1 |
| op:router dot product | nested | 0.521364 / 1 | 0.505354 / 1 | 0.525481 / 1 |
| op:top-k selection | nested | 0.002294 / 1 | 0.002144 / 1 | 0.002284 / 1 |
| detail:read-ahead-wait | nested | 0.001331 / 1 | 0.001393 / 1 | 0.001162 / 1 |
| total:layer | total | 153.482798 / 1 | 154.458382 / 1 | 166.169257 / 1 |

### Layer 64

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004889 / 1 | 0.004628 / 1 | 0.003898 / 1 |
| pre-attention-aggregation | boundary | 0.016701 / 1 | 0.016170 / 1 | 0.016611 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010950 / 1 | 0.011551 / 1 | 0.012333 / 1 |
| Q | nested | 3.314099 / 1 | 3.331782 / 1 | 3.313267 / 1 |
| K | nested | 2.349908 / 1 | 2.390675 / 1 | 2.871472 / 1 |
| V | nested | 2.328599 / 1 | 2.329561 / 1 | 2.781505 / 1 |
| B | nested | 0.027582 / 1 | 0.024365 / 1 | 0.030878 / 1 |
| FA | nested | 0.028514 / 1 | 0.027832 / 1 | 0.034895 / 1 |
| FB | nested | 0.055353 / 1 | 0.047068 / 1 | 0.076392 / 1 |
| G | nested | 2.056330 / 1 | 2.086066 / 1 | 2.885498 / 1 |
| O | nested | 2.219504 / 1 | 2.164672 / 1 | 2.303372 / 1 |
| attention | boundary | 13.456400 / 1 | 13.476077 / 1 | 15.357881 / 1 |
| attention-residual | boundary | 0.003707 / 1 | 0.003406 / 1 | 0.003496 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025187 / 1 | 0.025067 / 1 | 0.025258 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001864 / 1 | 0.001953 / 1 | 0.001704 / 1 |
| router-and-top16 | boundary | 0.561859 / 1 | 0.583500 / 1 | 0.574353 / 1 |
| EDOWN | nested | 0.599029 / 1 | 0.602455 / 1 | 0.585413 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.600391 / 1 | 0.603888 / 1 | 0.586786 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026149 / 1 | 0.022041 / 1 | 0.008946 / 1 |
| SH1 | nested | 1.017019 / 1 | 1.021087 / 1 | 0.995640 / 1 |
| SH3 | nested | 1.015066 / 1 | 1.028881 / 1 | 0.998735 / 1 |
| SH2 | nested | 0.993866 / 1 | 1.021888 / 1 | 1.007472 / 1 |
| shared-expert-during-read | boundary | 3.038735 / 1 | 3.085081 / 1 | 3.014600 / 1 |
| detail:expert-gate | nested | 49.710304 / 16 | 49.394112 / 16 | 53.220009 / 16 |
| detail:expert-up | nested | 42.561863 / 16 | 42.666008 / 16 | 51.829264 / 16 |
| detail:expert-activation | nested | 0.095748 / 16 | 0.095539 / 16 | 0.129360 / 16 |
| detail:expert-down | nested | 48.878622 / 16 | 48.867669 / 16 | 53.810831 / 16 |
| EUP | nested | 0.594900 / 1 | 0.601563 / 1 | 0.614387 / 1 |
| experts-mix-normalize-up | boundary | 142.263125 / 1 | 142.048878 / 1 | 160.023319 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002494 / 1 | 0.003176 / 1 | 0.002705 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.592055 / 1 | 0.587227 / 1 | 0.589751 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.970673 / 1 | 0.999807 / 1 | 0.993766 / 1 |
| worker:read | parallel worker | 0.069466 / 1 | 0.069602 / 1 | 0.070113 / 1 |
| worker:decode | parallel worker | 1068.151131 / 1 | 1072.283884 / 1 | 1078.388749 / 1 |
| worker:crc | parallel worker | 598.165679 / 1 | 601.255676 / 1 | 605.251131 / 1 |
| worker:math | parallel worker | 362.840711 / 1 | 362.756246 / 1 | 371.561803 / 1 |
| op:Q   int8 projection | nested | 16.598368 / 1 | 16.676384 / 1 | 18.497514 / 1 |
| op:X   mxfp4 expert proj | nested | 141.319764 / 1 | 141.097701 / 1 | 159.088264 / 1 |
| op:N   rmsnorm | nested | 0.042981 / 1 | 0.038292 / 1 | 0.031248 / 1 |
| op:L   l2 per-head | nested | 0.005500 / 1 | 0.006492 / 1 | 0.005720 / 1 |
| op:SiTU + sigma | nested | 0.102871 / 1 | 0.102882 / 1 | 0.136094 / 1 |
| op:C   shortconv | nested | 0.051967 / 1 | 0.076032 / 1 | 0.037541 / 1 |
| op:AR  snapshot aggregate | nested | 0.032050 / 1 | 0.031439 / 1 | 0.032220 / 1 |
| op:D   kda delta-rule | nested | 0.150882 / 1 | 0.104636 / 1 | 0.130203 / 1 |
| op:router dot product | nested | 0.559204 / 1 | 0.580344 / 1 | 0.572069 / 1 |
| op:top-k selection | nested | 0.002315 / 1 | 0.002325 / 1 | 0.002044 / 1 |
| op:alpha / beta / gate | nested | 0.071473 / 1 | 0.073597 / 1 | 0.062026 / 1 |
| detail:read-ahead-wait | nested | 0.001414 / 1 | 0.001592 / 1 | 0.000852 / 1 |
| total:layer | total | 161.001499 / 1 | 160.903769 / 1 | 180.644831 / 1 |

### Layer 65

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005490 / 1 | 0.006011 / 1 | 0.004598 / 1 |
| pre-attention-aggregation | boundary | 0.017823 / 1 | 0.017513 / 1 | 0.017813 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012433 / 1 | 0.018034 / 1 | 0.011681 / 1 |
| Q | nested | 3.121409 / 1 | 2.904274 / 1 | 3.278653 / 1 |
| K | nested | 2.368403 / 1 | 2.396054 / 1 | 2.328368 / 1 |
| V | nested | 2.308051 / 1 | 2.300165 / 1 | 2.260822 / 1 |
| B | nested | 0.023283 / 1 | 0.022132 / 1 | 0.022813 / 1 |
| FA | nested | 0.031178 / 1 | 0.029655 / 1 | 0.028633 / 1 |
| FB | nested | 0.050535 / 1 | 0.051296 / 1 | 0.050825 / 1 |
| G | nested | 2.056540 / 1 | 2.068072 / 1 | 2.059646 / 1 |
| O | nested | 2.056491 / 1 | 2.068473 / 1 | 2.064455 / 1 |
| attention | boundary | 13.160327 / 1 | 13.012581 / 1 | 13.292183 / 1 |
| attention-residual | boundary | 0.003978 / 1 | 0.003706 / 1 | 0.003847 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024325 / 1 | 0.025017 / 1 | 0.024676 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002104 / 1 | 0.001904 / 1 | 0.002024 / 1 |
| router-and-top16 | boundary | 0.570826 / 1 | 0.577459 / 1 | 0.572098 / 1 |
| EDOWN | nested | 0.601003 / 1 | 0.588910 / 1 | 0.588209 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.602425 / 1 | 0.590282 / 1 | 0.589682 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.046066 / 1 | 0.022432 / 1 | 0.018385 / 1 |
| SH1 | nested | 1.001651 / 1 | 1.002833 / 1 | 0.997072 / 1 |
| SH3 | nested | 1.032348 / 1 | 1.015647 / 1 | 1.011039 / 1 |
| SH2 | nested | 0.998976 / 1 | 1.007041 / 1 | 1.000729 / 1 |
| shared-expert-during-read | boundary | 3.045367 / 1 | 3.037943 / 1 | 3.021823 / 1 |
| detail:expert-gate | nested | 48.811588 / 16 | 48.788402 / 16 | 51.179492 / 16 |
| detail:expert-up | nested | 42.690103 / 16 | 42.747591 / 16 | 48.188345 / 16 |
| detail:expert-activation | nested | 0.094969 / 16 | 0.096310 / 16 | 0.115546 / 16 |
| detail:expert-down | nested | 49.015164 / 16 | 48.973766 / 16 | 50.995575 / 16 |
| EUP | nested | 0.600010 / 1 | 0.605621 / 1 | 0.733820 / 1 |
| experts-mix-normalize-up | boundary | 141.662514 / 1 | 141.632650 / 1 | 151.614616 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.002976 / 1 | 0.002795 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589921 / 1 | 0.588309 / 1 | 0.607574 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.947810 / 1 | 0.965232 / 1 | 0.985240 / 1 |
| worker:read | parallel worker | 0.196036 / 1 | 0.223561 / 1 | 0.196568 / 1 |
| worker:decode | parallel worker | 1066.768520 / 1 | 1070.688246 / 1 | 1075.174868 / 1 |
| worker:crc | parallel worker | 597.803674 / 1 | 600.736174 / 1 | 603.914470 / 1 |
| worker:math | parallel worker | 366.694730 / 1 | 366.658057 / 1 | 371.804383 / 1 |
| op:Q   int8 projection | nested | 16.248345 / 1 | 16.058750 / 1 | 16.423631 / 1 |
| op:X   mxfp4 expert proj | nested | 140.688927 / 1 | 140.680211 / 1 | 150.566288 / 1 |
| op:N   rmsnorm | nested | 0.039053 / 1 | 0.041187 / 1 | 0.031880 / 1 |
| op:L   l2 per-head | nested | 0.005470 / 1 | 0.005390 / 1 | 0.004949 / 1 |
| op:SiTU + sigma | nested | 0.101562 / 1 | 0.103225 / 1 | 0.122317 / 1 |
| op:C   shortconv | nested | 0.054511 / 1 | 0.048409 / 1 | 0.036678 / 1 |
| op:AR  snapshot aggregate | nested | 0.032641 / 1 | 0.032522 / 1 | 0.032531 / 1 |
| op:D   kda delta-rule | nested | 0.241922 / 1 | 0.239226 / 1 | 0.242363 / 1 |
| op:router dot product | nested | 0.568201 / 1 | 0.574603 / 1 | 0.568792 / 1 |
| op:top-k selection | nested | 0.002255 / 1 | 0.002495 / 1 | 0.002936 / 1 |
| op:alpha / beta / gate | nested | 0.073658 / 1 | 0.077514 / 1 | 0.058860 / 1 |
| detail:read-ahead-wait | nested | 0.001315 / 1 | 0.001794 / 1 | 0.001272 / 1 |
| total:layer | total | 160.124521 / 1 | 159.932876 / 1 | 170.179907 / 1 |

### Layer 66

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005340 / 1 | 0.004187 / 1 | 0.005019 / 1 |
| pre-attention-aggregation | boundary | 0.016942 / 1 | 0.016972 / 1 | 0.017582 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000100 / 1 |
| pre-attention-normalization | boundary | 0.011722 / 1 | 0.011772 / 1 | 0.012493 / 1 |
| Q | nested | 3.259307 / 1 | 3.285806 / 1 | 3.276619 / 1 |
| K | nested | 2.363564 / 1 | 2.375797 / 1 | 2.741880 / 1 |
| V | nested | 2.320083 / 1 | 2.311046 / 1 | 2.691547 / 1 |
| B | nested | 0.024646 / 1 | 0.032540 / 1 | 0.026630 / 1 |
| FA | nested | 0.026199 / 1 | 0.026208 / 1 | 0.029595 / 1 |
| FB | nested | 0.052367 / 1 | 0.050966 / 1 | 0.055574 / 1 |
| G | nested | 2.065187 / 1 | 2.082328 / 1 | 2.408829 / 1 |
| O | nested | 2.075987 / 1 | 2.083551 / 1 | 2.439025 / 1 |
| attention | boundary | 13.355632 / 1 | 13.445028 / 1 | 14.864811 / 1 |
| attention-residual | boundary | 0.007334 / 1 | 0.003707 / 1 | 0.003546 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025037 / 1 | 0.024406 / 1 | 0.024205 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001693 / 1 | 0.002134 / 1 | 0.001693 / 1 |
| router-and-top16 | boundary | 0.565496 / 1 | 0.580905 / 1 | 0.594220 / 1 |
| EDOWN | nested | 0.591295 / 1 | 0.601323 / 1 | 0.697883 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.592737 / 1 | 0.602706 / 1 | 0.699256 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026019 / 1 | 0.006632 / 1 | 0.016621 / 1 |
| SH1 | nested | 1.010417 / 1 | 1.023091 / 1 | 1.188330 / 1 |
| SH3 | nested | 1.003204 / 1 | 1.013894 / 1 | 1.191074 / 1 |
| SH2 | nested | 1.008553 / 1 | 1.009194 / 1 | 1.201674 / 1 |
| shared-expert-during-read | boundary | 3.035118 / 1 | 3.059584 / 1 | 3.596577 / 1 |
| detail:expert-gate | nested | 49.563243 / 16 | 49.873591 / 16 | 53.277889 / 16 |
| detail:expert-up | nested | 42.553216 / 16 | 42.635965 / 16 | 51.674035 / 16 |
| detail:expert-activation | nested | 0.096170 / 16 | 0.096499 / 16 | 0.127909 / 16 |
| detail:expert-down | nested | 48.734923 / 16 | 48.790447 / 16 | 52.751012 / 16 |
| EUP | nested | 0.614788 / 1 | 0.609008 / 1 | 0.606663 / 1 |
| experts-mix-normalize-up | boundary | 141.980558 / 1 | 142.413077 / 1 | 158.877058 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002956 / 1 | 0.002765 / 1 | 0.003196 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590603 / 1 | 0.590162 / 1 | 0.591575 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.949924 / 1 | 0.959552 / 1 | 0.970332 / 1 |
| worker:read | parallel worker | 0.068445 / 1 | 0.065920 / 1 | 0.069333 / 1 |
| worker:decode | parallel worker | 1066.493604 / 1 | 1072.107906 / 1 | 1077.136584 / 1 |
| worker:crc | parallel worker | 598.496702 / 1 | 601.750015 / 1 | 605.467811 / 1 |
| worker:math | parallel worker | 361.908516 / 1 | 362.899855 / 1 | 371.818324 / 1 |
| op:Q   int8 projection | nested | 16.414001 / 1 | 16.503309 / 1 | 18.553879 / 1 |
| op:X   mxfp4 expert proj | nested | 141.022732 / 1 | 141.471487 / 1 | 157.932968 / 1 |
| op:N   rmsnorm | nested | 0.046728 / 1 | 0.044523 / 1 | 0.031160 / 1 |
| op:L   l2 per-head | nested | 0.005771 / 1 | 0.005300 / 1 | 0.005450 / 1 |
| op:SiTU + sigma | nested | 0.103392 / 1 | 0.104013 / 1 | 0.137267 / 1 |
| op:C   shortconv | nested | 0.056204 / 1 | 0.062076 / 1 | 0.030187 / 1 |
| op:AR  snapshot aggregate | nested | 0.031849 / 1 | 0.032101 / 1 | 0.032620 / 1 |
| op:D   kda delta-rule | nested | 0.241652 / 1 | 0.244006 / 1 | 0.251210 / 1 |
| op:router dot product | nested | 0.562821 / 1 | 0.578310 / 1 | 0.591555 / 1 |
| op:top-k selection | nested | 0.002305 / 1 | 0.002285 / 1 | 0.002374 / 1 |
| op:alpha / beta / gate | nested | 0.074178 / 1 | 0.074208 / 1 | 0.058730 / 1 |
| detail:read-ahead-wait | nested | 0.001275 / 1 | 0.001482 / 1 | 0.001183 / 1 |
| total:layer | total | 160.595261 / 1 | 161.152744 / 1 | 179.708131 / 1 |

### Layer 67

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004568 / 1 | 0.004618 / 1 | 0.004930 / 1 |
| pre-attention-aggregation | boundary | 0.016892 / 1 | 0.017633 / 1 | 0.017292 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011551 / 1 | 0.011351 / 1 | 0.011542 / 1 |
| QA | nested | 0.364641 / 1 | 0.359962 / 1 | 0.352489 / 1 |
| QB | nested | 0.936338 / 1 | 0.885403 / 1 | 0.942971 / 1 |
| KA | nested | 0.134601 / 1 | 0.133659 / 1 | 0.134481 / 1 |
| KB | nested | 0.380831 / 1 | 0.385851 / 1 | 0.401169 / 1 |
| G | nested | 2.632215 / 1 | 2.766527 / 1 | 2.683211 / 1 |
| O | nested | 2.014702 / 1 | 2.081467 / 1 | 2.021755 / 1 |
| attention | boundary | 6.585008 / 1 | 6.729738 / 1 | 6.652143 / 1 |
| attention-residual | boundary | 0.002985 / 1 | 0.002725 / 1 | 0.002715 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025348 / 1 | 0.024566 / 1 | 0.025137 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001813 / 1 | 0.001813 / 1 | 0.001563 / 1 |
| router-and-top16 | boundary | 0.522696 / 1 | 0.540700 / 1 | 0.526212 / 1 |
| EDOWN | nested | 0.602014 / 1 | 0.607595 / 1 | 0.592967 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.603407 / 1 | 0.608967 / 1 | 0.594339 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.025668 / 1 | 0.006392 / 1 | 0.009208 / 1 |
| SH1 | nested | 1.027679 / 1 | 1.045112 / 1 | 1.019374 / 1 |
| SH3 | nested | 1.015086 / 1 | 1.027980 / 1 | 1.019394 / 1 |
| SH2 | nested | 1.033621 / 1 | 1.025696 / 1 | 1.017630 / 1 |
| shared-expert-during-read | boundary | 3.088958 / 1 | 3.111300 / 1 | 3.069553 / 1 |
| detail:expert-gate | nested | 49.400215 / 16 | 49.798830 / 16 | 52.727399 / 16 |
| detail:expert-up | nested | 42.612668 / 16 | 42.647956 / 16 | 49.615964 / 16 |
| detail:expert-activation | nested | 0.097852 / 16 | 0.095169 / 16 | 0.122720 / 16 |
| detail:expert-down | nested | 48.810875 / 16 | 48.862382 / 16 | 51.286278 / 16 |
| EUP | nested | 0.617253 / 1 | 0.599890 / 1 | 0.596514 / 1 |
| experts-mix-normalize-up | boundary | 141.963005 / 1 | 142.400845 / 1 | 154.767414 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002946 / 1 | 0.002956 / 1 | 0.003095 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.595342 / 1 | 0.598458 / 1 | 0.597045 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.601263 / 1 | 0.604389 / 1 | 0.603588 / 1 |
| worker:read | parallel worker | 0.066507 / 1 | 0.066661 / 1 | 0.068211 / 1 |
| worker:decode | parallel worker | 1065.545679 / 1 | 1071.340060 / 1 | 1075.066934 / 1 |
| worker:crc | parallel worker | 598.314275 / 1 | 601.561678 / 1 | 604.095976 / 1 |
| worker:math | parallel worker | 363.042518 / 1 | 363.295645 / 1 | 369.099505 / 1 |
| op:Q   int8 projection | nested | 10.757831 / 1 | 10.917698 / 1 | 10.780644 / 1 |
| op:X   mxfp4 expert proj | nested | 140.998133 / 1 | 141.479422 / 1 | 153.849900 / 1 |
| op:N   rmsnorm | nested | 0.031980 / 1 | 0.034654 / 1 | 0.031839 / 1 |
| op:SiTU + sigma | nested | 0.104694 / 1 | 0.101821 / 1 | 0.129482 / 1 |
| op:AR  snapshot aggregate | nested | 0.031610 / 1 | 0.032151 / 1 | 0.032490 / 1 |
| op:SA  softmax attention | nested | 0.006802 / 1 | 0.009217 / 1 | 0.011912 / 1 |
| op:router dot product | nested | 0.520131 / 1 | 0.538035 / 1 | 0.523167 / 1 |
| op:top-k selection | nested | 0.002244 / 1 | 0.002354 / 1 | 0.002805 / 1 |
| detail:read-ahead-wait | nested | 0.001321 / 1 | 0.001293 / 1 | 0.001003 / 1 |
| total:layer | total | 153.477608 / 1 | 154.087289 / 1 | 166.307396 / 1 |

### Layer 68

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004358 / 1 | 0.004628 / 1 | 0.004468 / 1 |
| pre-attention-aggregation | boundary | 0.015299 / 1 | 0.016461 / 1 | 0.016291 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011462 / 1 | 0.011662 / 1 | 0.011863 / 1 |
| Q | nested | 3.176292 / 1 | 3.293020 / 1 | 3.300835 / 1 |
| K | nested | 2.351482 / 1 | 2.363424 / 1 | 2.301208 / 1 |
| V | nested | 2.311908 / 1 | 2.293803 / 1 | 2.283795 / 1 |
| B | nested | 0.021300 / 1 | 0.022031 / 1 | 0.020949 / 1 |
| FA | nested | 0.030948 / 1 | 0.033853 / 1 | 0.030166 / 1 |
| FB | nested | 0.049723 / 1 | 0.050755 / 1 | 0.050855 / 1 |
| G | nested | 2.034109 / 1 | 2.079674 / 1 | 2.042805 / 1 |
| O | nested | 2.062271 / 1 | 2.096686 / 1 | 2.072300 / 1 |
| attention | boundary | 13.135100 / 1 | 13.323752 / 1 | 13.183360 / 1 |
| attention-residual | boundary | 0.003757 / 1 | 0.003607 / 1 | 0.003667 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024386 / 1 | 0.030086 / 1 | 0.024356 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001733 / 1 | 0.002054 / 1 | 0.001873 / 1 |
| router-and-top16 | boundary | 0.568442 / 1 | 0.575565 / 1 | 0.573020 / 1 |
| EDOWN | nested | 0.604318 / 1 | 0.597266 / 1 | 0.591575 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.605731 / 1 | 0.598658 / 1 | 0.593008 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026139 / 1 | 0.005601 / 1 | 0.015619 / 1 |
| SH1 | nested | 1.007071 / 1 | 1.017730 / 1 | 1.003985 / 1 |
| SH3 | nested | 1.018702 / 1 | 1.009165 / 1 | 1.002181 / 1 |
| SH2 | nested | 1.022639 / 1 | 1.027158 / 1 | 1.001921 / 1 |
| shared-expert-during-read | boundary | 3.061397 / 1 | 3.066827 / 1 | 3.021172 / 1 |
| detail:expert-gate | nested | 49.491592 / 16 | 48.853604 / 16 | 53.948412 / 16 |
| detail:expert-up | nested | 42.705861 / 16 | 42.966427 / 16 | 52.496660 / 16 |
| detail:expert-activation | nested | 0.097142 / 16 | 0.096630 / 16 | 0.125155 / 16 |
| detail:expert-down | nested | 48.759640 / 16 | 48.940057 / 16 | 52.431728 / 16 |
| EUP | nested | 0.599560 / 1 | 0.612935 / 1 | 0.607535 / 1 |
| experts-mix-normalize-up | boundary | 142.070997 / 1 | 141.873961 / 1 | 162.796018 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002785 / 1 | 0.003346 / 1 | 0.003336 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587226 / 1 | 0.586065 / 1 | 0.603147 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.953431 / 1 | 0.968118 / 1 | 1.021027 / 1 |
| worker:read | parallel worker | 0.215665 / 1 | 0.214800 / 1 | 0.214275 / 1 |
| worker:decode | parallel worker | 1065.357911 / 1 | 1072.121604 / 1 | 1079.738763 / 1 |
| worker:crc | parallel worker | 601.082931 / 1 | 603.350927 / 1 | 614.424680 / 1 |
| worker:math | parallel worker | 362.307264 / 1 | 362.695705 / 1 | 374.213432 / 1 |
| op:Q   int8 projection | nested | 16.288949 / 1 | 16.495948 / 1 | 16.308649 / 1 |
| op:X   mxfp4 expert proj | nested | 141.133764 / 1 | 140.933844 / 1 | 159.103103 / 1 |
| op:N   rmsnorm | nested | 0.043231 / 1 | 0.038623 / 1 | 0.031919 / 1 |
| op:L   l2 per-head | nested | 0.005891 / 1 | 0.005911 / 1 | 0.005781 / 1 |
| op:SiTU + sigma | nested | 0.103883 / 1 | 0.103785 / 1 | 0.132207 / 1 |
| op:C   shortconv | nested | 0.051255 / 1 | 0.063970 / 1 | 0.032090 / 1 |
| op:AR  snapshot aggregate | nested | 0.030026 / 1 | 0.036940 / 1 | 0.030847 / 1 |
| op:D   kda delta-rule | nested | 0.155481 / 1 | 0.123410 / 1 | 0.140011 / 1 |
| op:router dot product | nested | 0.565647 / 1 | 0.572599 / 1 | 0.569975 / 1 |
| op:top-k selection | nested | 0.002334 / 1 | 0.002595 / 1 | 0.002645 / 1 |
| op:alpha / beta / gate | nested | 0.073567 / 1 | 0.072376 / 1 | 0.057608 / 1 |
| detail:read-ahead-wait | nested | 0.001193 / 1 | 0.001303 / 1 | 2.781676 / 1 |
| total:layer | total | 160.504621 / 1 | 160.503742 / 1 | 181.288282 / 1 |

### Layer 69

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004989 / 1 | 0.005430 / 1 | 0.006282 / 1 |
| pre-attention-aggregation | boundary | 0.016490 / 1 | 0.016751 / 1 | 0.018374 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011481 / 1 | 0.012102 / 1 | 0.012994 / 1 |
| Q | nested | 3.303139 / 1 | 3.283883 / 1 | 3.075534 / 1 |
| K | nested | 2.347344 / 1 | 2.383300 / 1 | 2.332516 / 1 |
| V | nested | 2.313902 / 1 | 2.298542 / 1 | 2.286089 / 1 |
| B | nested | 0.023965 / 1 | 0.024906 / 1 | 0.022833 / 1 |
| FA | nested | 0.028062 / 1 | 0.026730 / 1 | 0.027872 / 1 |
| FB | nested | 0.055253 / 1 | 0.048360 / 1 | 0.048911 / 1 |
| G | nested | 2.063815 / 1 | 2.088731 / 1 | 2.038257 / 1 |
| O | nested | 2.073572 / 1 | 2.080856 / 1 | 2.061661 / 1 |
| attention | boundary | 13.368004 / 1 | 13.389715 / 1 | 13.094374 / 1 |
| attention-residual | boundary | 0.003627 / 1 | 0.003557 / 1 | 0.003447 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024957 / 1 | 0.025258 / 1 | 0.024526 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001793 / 1 | 0.002023 / 1 | 0.001693 / 1 |
| router-and-top16 | boundary | 0.573271 / 1 | 0.584502 / 1 | 0.578370 / 1 |
| EDOWN | nested | 0.591234 / 1 | 0.591946 / 1 | 0.596273 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.592667 / 1 | 0.593388 / 1 | 0.597756 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026499 / 1 | 0.011351 / 1 | 0.006382 / 1 |
| SH1 | nested | 1.004306 / 1 | 1.007512 / 1 | 1.014605 / 1 |
| SH3 | nested | 1.002853 / 1 | 1.007361 / 1 | 0.999306 / 1 |
| SH2 | nested | 1.001620 / 1 | 1.012170 / 1 | 1.007832 / 1 |
| shared-expert-during-read | boundary | 3.021964 / 1 | 3.040128 / 1 | 3.034156 / 1 |
| detail:expert-gate | nested | 49.689978 / 16 | 49.665062 / 16 | 52.175752 / 16 |
| detail:expert-up | nested | 42.530416 / 16 | 42.588307 / 16 | 48.820514 / 16 |
| detail:expert-activation | nested | 0.096568 / 16 | 0.096791 / 16 | 0.121298 / 16 |
| detail:expert-down | nested | 48.732711 / 16 | 48.848707 / 16 | 52.422720 / 16 |
| EUP | nested | 0.599439 / 1 | 0.612524 / 1 | 0.599309 / 1 |
| experts-mix-normalize-up | boundary | 142.071668 / 1 | 142.222983 / 1 | 157.359575 / 1 |
| mlp-merge-and-cleanup | boundary | 0.004128 / 1 | 0.002455 / 1 | 0.003146 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591866 / 1 | 0.588109 / 1 | 0.598658 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.960774 / 1 | 0.960684 / 1 | 0.999597 / 1 |
| worker:read | parallel worker | 0.066159 / 1 | 0.067427 / 1 | 0.067246 / 1 |
| worker:decode | parallel worker | 1065.143465 / 1 | 1069.736336 / 1 | 1072.194363 / 1 |
| worker:crc | parallel worker | 598.138383 / 1 | 601.217684 / 1 | 603.335458 / 1 |
| worker:math | parallel worker | 361.701592 / 1 | 362.886559 / 1 | 369.058752 / 1 |
| op:Q   int8 projection | nested | 16.407090 / 1 | 16.465451 / 1 | 16.109615 / 1 |
| op:X   mxfp4 expert proj | nested | 141.129180 / 1 | 141.277176 / 1 | 153.636293 / 1 |
| op:N   rmsnorm | nested | 0.041919 / 1 | 0.039153 / 1 | 0.032182 / 1 |
| op:L   l2 per-head | nested | 0.005471 / 1 | 0.005800 / 1 | 0.005491 / 1 |
| op:SiTU + sigma | nested | 0.103994 / 1 | 0.104437 / 1 | 0.127148 / 1 |
| op:C   shortconv | nested | 0.058109 / 1 | 0.053359 / 1 | 0.031209 / 1 |
| op:AR  snapshot aggregate | nested | 0.031749 / 1 | 0.032220 / 1 | 0.033353 / 1 |
| op:D   kda delta-rule | nested | 0.243164 / 1 | 0.236862 / 1 | 0.243755 / 1 |
| op:router dot product | nested | 0.570725 / 1 | 0.581906 / 1 | 0.575655 / 1 |
| op:top-k selection | nested | 0.002154 / 1 | 0.002204 / 1 | 0.002304 / 1 |
| op:alpha / beta / gate | nested | 0.072275 / 1 | 0.071614 / 1 | 0.058910 / 1 |
| detail:read-ahead-wait | nested | 0.001351 / 1 | 0.001361 / 1 | 2.795209 / 1 |
| total:layer | total | 160.702311 / 1 | 160.889593 / 1 | 175.762082 / 1 |

### Layer 70

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004348 / 1 | 0.004478 / 1 | 0.004919 / 1 |
| pre-attention-aggregation | boundary | 0.020508 / 1 | 0.017182 / 1 | 0.017863 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.010921 / 1 | 0.011942 / 1 | 0.011471 / 1 |
| Q | nested | 3.189707 / 1 | 3.245712 / 1 | 3.438783 / 1 |
| K | nested | 2.340671 / 1 | 2.384863 / 1 | 2.903332 / 1 |
| V | nested | 2.309834 / 1 | 2.303672 / 1 | 2.806000 / 1 |
| B | nested | 0.020989 / 1 | 0.021140 / 1 | 0.023915 / 1 |
| FA | nested | 0.032951 / 1 | 0.032531 / 1 | 0.035987 / 1 |
| FB | nested | 0.049813 / 1 | 0.048741 / 1 | 0.051877 / 1 |
| G | nested | 2.044027 / 1 | 2.079353 / 1 | 2.048727 / 1 |
| O | nested | 2.055008 / 1 | 2.084573 / 1 | 2.064596 / 1 |
| attention | boundary | 13.227853 / 1 | 13.377874 / 1 | 14.527591 / 1 |
| attention-residual | boundary | 0.003978 / 1 | 0.003607 / 1 | 0.003546 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024035 / 1 | 0.025938 / 1 | 0.024606 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001683 / 1 | 0.001744 / 1 | 0.002425 / 1 |
| router-and-top16 | boundary | 0.567330 / 1 | 0.587117 / 1 | 0.570926 / 1 |
| EDOWN | nested | 0.603026 / 1 | 0.595332 / 1 | 0.589461 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.604489 / 1 | 0.596865 / 1 | 0.590933 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041217 / 1 | 0.021671 / 1 | 0.019116 / 1 |
| SH1 | nested | 1.016148 / 1 | 1.021658 / 1 | 1.458263 / 1 |
| SH3 | nested | 1.016579 / 1 | 1.017731 / 1 | 1.530438 / 1 |
| SH2 | nested | 0.999587 / 1 | 1.009826 / 1 | 1.030835 / 1 |
| shared-expert-during-read | boundary | 3.045477 / 1 | 3.062378 / 1 | 4.038081 / 1 |
| detail:expert-gate | nested | 49.657027 / 16 | 49.575925 / 16 | 53.895469 / 16 |
| detail:expert-up | nested | 42.666721 / 16 | 42.737936 / 16 | 52.575789 / 16 |
| detail:expert-activation | nested | 0.099805 / 16 | 0.094689 / 16 | 0.129011 / 16 |
| detail:expert-down | nested | 48.676633 / 16 | 48.845277 / 16 | 52.069144 / 16 |
| EUP | nested | 0.603207 / 1 | 0.596975 / 1 | 0.611151 / 1 |
| experts-mix-normalize-up | boundary | 142.140697 / 1 | 142.273067 / 1 | 159.673767 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002956 / 1 | 0.003216 / 1 | 0.003256 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589771 / 1 | 0.582287 / 1 | 0.593448 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.949764 / 1 | 0.953922 / 1 | 0.968809 / 1 |
| worker:read | parallel worker | 0.067853 / 1 | 0.065161 / 1 | 0.070342 / 1 |
| worker:decode | parallel worker | 1065.241377 / 1 | 1071.760897 / 1 | 1078.901506 / 1 |
| worker:crc | parallel worker | 598.704633 / 1 | 602.710247 / 1 | 607.734320 / 1 |
| worker:math | parallel worker | 361.772318 / 1 | 362.927413 / 1 | 375.470236 / 1 |
| op:Q   int8 projection | nested | 16.280034 / 1 | 16.440624 / 1 | 18.591881 / 1 |
| op:X   mxfp4 expert proj | nested | 141.180548 / 1 | 141.331058 / 1 | 158.773795 / 1 |
| op:N   rmsnorm | nested | 0.048961 / 1 | 0.044133 / 1 | 0.031569 / 1 |
| op:L   l2 per-head | nested | 0.005370 / 1 | 0.005600 / 1 | 0.005541 / 1 |
| op:SiTU + sigma | nested | 0.106851 / 1 | 0.101884 / 1 | 0.139221 / 1 |
| op:C   shortconv | nested | 0.061245 / 1 | 0.059781 / 1 | 0.031659 / 1 |
| op:AR  snapshot aggregate | nested | 0.035075 / 1 | 0.033162 / 1 | 0.032811 / 1 |
| op:D   kda delta-rule | nested | 0.246300 / 1 | 0.240069 / 1 | 0.240188 / 1 |
| op:router dot product | nested | 0.564253 / 1 | 0.584561 / 1 | 0.568071 / 1 |
| op:top-k selection | nested | 0.002364 / 1 | 0.002275 / 1 | 0.002165 / 1 |
| op:alpha / beta / gate | nested | 0.074609 / 1 | 0.075341 / 1 | 0.058399 / 1 |
| detail:read-ahead-wait | nested | 0.001303 / 1 | 0.001336 / 1 | 0.001000 / 1 |
| total:layer | total | 160.669479 / 1 | 160.960896 / 1 | 180.478911 / 1 |

### Layer 71

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004258 / 1 | 0.005190 / 1 | 0.005610 / 1 |
| pre-attention-aggregation | boundary | 0.017112 / 1 | 0.017402 / 1 | 0.021601 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011011 / 1 | 0.012403 / 1 | 0.011271 / 1 |
| QA | nested | 0.358760 / 1 | 0.373237 / 1 | 0.350375 / 1 |
| QB | nested | 0.913205 / 1 | 0.898127 / 1 | 0.909659 / 1 |
| KA | nested | 0.135573 / 1 | 0.136545 / 1 | 0.132417 / 1 |
| KB | nested | 0.412500 / 1 | 0.421797 / 1 | 0.412740 / 1 |
| G | nested | 2.767157 / 1 | 2.721022 / 1 | 2.707256 / 1 |
| O | nested | 2.041172 / 1 | 2.085875 / 1 | 2.023469 / 1 |
| attention | boundary | 6.743693 / 1 | 6.762960 / 1 | 6.655419 / 1 |
| attention-residual | boundary | 0.002776 / 1 | 0.002875 / 1 | 0.002685 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025217 / 1 | 0.025197 / 1 | 0.025157 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001553 / 1 | 0.001733 / 1 | 0.001553 / 1 |
| router-and-top16 | boundary | 0.521694 / 1 | 0.521935 / 1 | 0.527365 / 1 |
| EDOWN | nested | 0.597206 / 1 | 0.610240 / 1 | 0.593999 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.598668 / 1 | 0.611672 / 1 | 0.595482 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.016191 / 1 | 0.012924 / 1 | 0.007303 / 1 |
| SH1 | nested | 1.031175 / 1 | 1.428598 / 1 | 1.023170 / 1 |
| SH3 | nested | 1.021808 / 1 | 1.422256 / 1 | 1.016819 / 1 |
| SH2 | nested | 1.028531 / 1 | 1.370108 / 1 | 1.023772 / 1 |
| shared-expert-during-read | boundary | 3.094348 / 1 | 4.239177 / 1 | 3.076807 / 1 |
| detail:expert-gate | nested | 48.461888 / 16 | 49.657737 / 16 | 52.756474 / 16 |
| detail:expert-up | nested | 42.577749 / 16 | 42.581841 / 16 | 49.542364 / 16 |
| detail:expert-activation | nested | 0.097972 / 16 | 0.096930 / 16 | 0.123237 / 16 |
| detail:expert-down | nested | 48.715918 / 16 | 48.859409 / 16 | 52.151183 / 16 |
| EUP | nested | 0.611763 / 1 | 0.607485 / 1 | 0.602114 / 1 |
| experts-mix-normalize-up | boundary | 140.881887 / 1 | 142.216060 / 1 | 155.597674 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003045 / 1 | 0.003496 / 1 | 0.003186 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.603357 / 1 | 0.601564 / 1 | 0.594830 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.609639 / 1 | 0.607805 / 1 | 0.601544 / 1 |
| worker:read | parallel worker | 0.187548 / 1 | 0.184424 / 1 | 0.178091 / 1 |
| worker:decode | parallel worker | 1063.836145 / 1 | 1069.737874 / 1 | 1073.757449 / 1 |
| worker:crc | parallel worker | 597.565548 / 1 | 600.841271 / 1 | 603.659177 / 1 |
| worker:math | parallel worker | 362.910146 / 1 | 364.302631 / 1 | 369.757133 / 1 |
| op:Q   int8 projection | nested | 10.917376 / 1 | 12.074028 / 1 | 10.794587 / 1 |
| op:X   mxfp4 expert proj | nested | 139.935378 / 1 | 141.275692 / 1 | 154.676322 / 1 |
| op:N   rmsnorm | nested | 0.032141 / 1 | 0.032610 / 1 | 0.032199 / 1 |
| op:SiTU + sigma | nested | 0.105036 / 1 | 0.107640 / 1 | 0.129759 / 1 |
| op:AR  snapshot aggregate | nested | 0.031969 / 1 | 0.032300 / 1 | 0.035756 / 1 |
| op:SA  softmax attention | nested | 0.006182 / 1 | 0.009177 / 1 | 0.012514 / 1 |
| op:router dot product | nested | 0.519059 / 1 | 0.518849 / 1 | 0.524530 / 1 |
| op:top-k selection | nested | 0.002264 / 1 | 0.002655 / 1 | 0.002494 / 1 |
| detail:read-ahead-wait | nested | 0.001342 / 1 | 0.001380 / 1 | 0.000910 / 1 |
| total:layer | total | 152.550628 / 1 | 155.062871 / 1 | 167.152263 / 1 |

### Layer 72

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004358 / 1 | 0.004468 / 1 | 0.004729 / 1 |
| pre-attention-aggregation | boundary | 0.015700 / 1 | 0.016211 / 1 | 0.016371 / 1 |
| snapshot-push | boundary | 0.001964 / 1 | 0.002104 / 1 | 0.002284 / 1 |
| pre-attention-normalization | boundary | 0.012113 / 1 | 0.011902 / 1 | 0.012113 / 1 |
| Q | nested | 3.349265 / 1 | 3.262443 / 1 | 3.383799 / 1 |
| K | nested | 2.308722 / 1 | 2.319171 / 1 | 2.735298 / 1 |
| V | nested | 2.281361 / 1 | 2.269218 / 1 | 2.827971 / 1 |
| B | nested | 0.025928 / 1 | 0.027201 / 1 | 0.030286 / 1 |
| FA | nested | 0.031589 / 1 | 0.032812 / 1 | 0.036308 / 1 |
| FB | nested | 0.056966 / 1 | 0.048210 / 1 | 0.068578 / 1 |
| G | nested | 2.030862 / 1 | 2.075135 / 1 | 2.802785 / 1 |
| O | nested | 2.064185 / 1 | 2.075225 / 1 | 2.041362 / 1 |
| attention | boundary | 13.217263 / 1 | 13.162341 / 1 | 14.919332 / 1 |
| attention-residual | boundary | 0.002133 / 1 | 0.002094 / 1 | 0.002244 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024505 / 1 | 0.025618 / 1 | 0.025057 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001563 / 1 | 0.001853 / 1 | 0.001944 / 1 |
| router-and-top16 | boundary | 0.564715 / 1 | 0.572439 / 1 | 0.546571 / 1 |
| EDOWN | nested | 0.588008 / 1 | 0.601935 / 1 | 0.579783 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.589491 / 1 | 0.603417 / 1 | 0.581336 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.033453 / 1 | 0.008296 / 1 | 0.008566 / 1 |
| SH1 | nested | 1.462191 / 1 | 1.601972 / 1 | 0.983176 / 1 |
| SH3 | nested | 1.497026 / 1 | 1.389124 / 1 | 0.978076 / 1 |
| SH2 | nested | 1.389786 / 1 | 1.014674 / 1 | 0.982324 / 1 |
| shared-expert-during-read | boundary | 4.366715 / 1 | 4.019717 / 1 | 2.957022 / 1 |
| detail:expert-gate | nested | 48.935269 / 16 | 47.840521 / 16 | 54.071909 / 16 |
| detail:expert-up | nested | 42.539240 / 16 | 42.647082 / 16 | 54.158592 / 16 |
| detail:expert-activation | nested | 0.094767 / 16 | 0.096788 / 16 | 0.136898 / 16 |
| detail:expert-down | nested | 48.830832 / 16 | 48.946548 / 16 | 54.865033 / 16 |
| EUP | nested | 0.614197 / 1 | 0.604469 / 1 | 0.595913 / 1 |
| experts-mix-normalize-up | boundary | 141.446882 / 1 | 140.550019 / 1 | 164.471487 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003256 / 1 | 0.002635 / 1 | 0.002885 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.594110 / 1 | 0.586936 / 1 | 0.598868 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.954993 / 1 | 0.988245 / 1 | 0.985570 / 1 |
| worker:read | parallel worker | 0.066651 / 1 | 0.069828 / 1 | 0.069234 / 1 |
| worker:decode | parallel worker | 1068.111767 / 1 | 1072.056930 / 1 | 1082.248472 / 1 |
| worker:crc | parallel worker | 597.807467 / 1 | 600.276223 / 1 | 608.352062 / 1 |
| worker:math | parallel worker | 361.810784 / 1 | 362.082118 / 1 | 378.596926 / 1 |
| op:Q   int8 projection | nested | 17.698543 / 1 | 17.320115 / 1 | 18.044057 / 1 |
| op:X   mxfp4 expert proj | nested | 140.483214 / 1 | 139.615893 / 1 | 163.350330 / 1 |
| op:N   rmsnorm | nested | 0.038242 / 1 | 0.040014 / 1 | 0.032742 / 1 |
| op:L   l2 per-head | nested | 0.005821 / 1 | 0.006432 / 1 | 0.006162 / 1 |
| op:SiTU + sigma | nested | 0.105389 / 1 | 0.103599 / 1 | 0.143409 / 1 |
| op:C   shortconv | nested | 0.057466 / 1 | 0.053179 / 1 | 0.033532 / 1 |
| op:AR  snapshot aggregate | nested | 0.030667 / 1 | 0.031649 / 1 | 0.031109 / 1 |
| op:D   kda delta-rule | nested | 0.149930 / 1 | 0.112350 / 1 | 0.138488 / 1 |
| op:router dot product | nested | 0.561870 / 1 | 0.569804 / 1 | 0.544036 / 1 |
| op:top-k selection | nested | 0.002545 / 1 | 0.002234 / 1 | 0.002084 / 1 |
| op:alpha / beta / gate | nested | 0.073147 / 1 | 0.077765 / 1 | 0.057407 / 1 |
| detail:read-ahead-wait | nested | 0.001373 / 1 | 0.001325 / 1 | 0.001121 / 1 |
| total:layer | total | 161.259141 / 1 | 159.992067 / 1 | 184.559451 / 1 |

### Layer 73

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005310 / 1 | 0.005430 / 1 | 0.006432 / 1 |
| pre-attention-aggregation | boundary | 0.017914 / 1 | 0.018635 / 1 | 0.017452 / 1 |
| snapshot-push | boundary | 0.000021 / 1 | 0.000140 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.010880 / 1 | 0.012273 / 1 | 0.011511 / 1 |
| Q | nested | 3.200016 / 1 | 2.895076 / 1 | 3.412162 / 1 |
| K | nested | 2.326665 / 1 | 2.377380 / 1 | 2.895748 / 1 |
| V | nested | 2.292431 / 1 | 2.279878 / 1 | 2.758803 / 1 |
| B | nested | 0.022282 / 1 | 0.021250 / 1 | 0.033041 / 1 |
| FA | nested | 0.035366 / 1 | 0.035897 / 1 | 0.035186 / 1 |
| FB | nested | 0.051907 / 1 | 0.053119 / 1 | 0.071864 / 1 |
| G | nested | 2.053976 / 1 | 2.075226 / 1 | 2.517622 / 1 |
| O | nested | 2.070867 / 1 | 2.083692 / 1 | 2.060829 / 1 |
| attention | boundary | 13.217854 / 1 | 13.010397 / 1 | 14.951181 / 1 |
| attention-residual | boundary | 0.003907 / 1 | 0.003687 / 1 | 0.003677 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024706 / 1 | 0.025167 / 1 | 0.024646 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001933 / 1 | 0.001934 / 1 | 0.002265 / 1 |
| router-and-top16 | boundary | 0.571037 / 1 | 0.575705 / 1 | 0.575675 / 1 |
| EDOWN | nested | 0.595031 / 1 | 0.599009 / 1 | 0.581376 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.597276 / 1 | 0.600492 / 1 | 0.582849 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026579 / 1 | 0.011782 / 1 | 0.009548 / 1 |
| SH1 | nested | 1.007070 / 1 | 1.021828 / 1 | 1.441862 / 1 |
| SH3 | nested | 1.010617 / 1 | 1.023070 / 1 | 1.512845 / 1 |
| SH2 | nested | 1.009375 / 1 | 1.022179 / 1 | 1.507045 / 1 |
| shared-expert-during-read | boundary | 3.040127 / 1 | 3.079942 / 1 | 4.481349 / 1 |
| detail:expert-gate | nested | 49.073034 / 16 | 49.629831 / 16 | 53.753134 / 16 |
| detail:expert-up | nested | 42.578353 / 16 | 42.610207 / 16 | 53.780218 / 16 |
| detail:expert-activation | nested | 0.095599 / 16 | 0.094886 / 16 | 0.137017 / 16 |
| detail:expert-down | nested | 48.717351 / 16 | 48.946267 / 16 | 56.465792 / 16 |
| EUP | nested | 0.604289 / 1 | 0.595873 / 1 | 0.601283 / 1 |
| experts-mix-normalize-up | boundary | 141.493408 / 1 | 142.286451 / 1 | 171.045654 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003075 / 1 | 0.002986 / 1 | 0.002936 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586926 / 1 | 0.590443 / 1 | 0.596004 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.958480 / 1 | 0.963299 / 1 | 0.991662 / 1 |
| worker:read | parallel worker | 0.071059 / 1 | 0.068886 / 1 | 0.068355 / 1 |
| worker:decode | parallel worker | 1068.690443 / 1 | 1073.964886 / 1 | 1081.127705 / 1 |
| worker:crc | parallel worker | 598.544483 / 1 | 601.861562 / 1 | 607.196825 / 1 |
| worker:math | parallel worker | 361.425875 / 1 | 362.933683 / 1 | 377.882506 / 1 |
| op:Q   int8 projection | nested | 16.278441 / 1 | 16.082005 / 1 | 19.427841 / 1 |
| op:X   mxfp4 expert proj | nested | 140.547193 / 1 | 141.361932 / 1 | 164.257125 / 1 |
| op:N   rmsnorm | nested | 0.039083 / 1 | 0.044082 / 1 | 0.031077 / 1 |
| op:L   l2 per-head | nested | 0.005691 / 1 | 0.005330 / 1 | 0.006132 / 1 |
| op:SiTU + sigma | nested | 0.102361 / 1 | 0.101992 / 1 | 0.146966 / 1 |
| op:C   shortconv | nested | 0.055213 / 1 | 0.055764 / 1 | 0.032400 / 1 |
| op:AR  snapshot aggregate | nested | 0.032641 / 1 | 0.034154 / 1 | 0.032521 / 1 |
| op:D   kda delta-rule | nested | 0.244457 / 1 | 0.243365 / 1 | 0.240519 / 1 |
| op:router dot product | nested | 0.568682 / 1 | 0.572860 / 1 | 0.573000 / 1 |
| op:top-k selection | nested | 0.002084 / 1 | 0.002194 / 1 | 0.002255 / 1 |
| op:alpha / beta / gate | nested | 0.073757 / 1 | 0.072034 / 1 | 0.058409 / 1 |
| detail:read-ahead-wait | nested | 0.001604 / 1 | 0.001672 / 1 | 5.890739 / 1 |
| total:layer | total | 159.994739 / 1 | 160.619739 / 1 | 192.729119 / 1 |

### Layer 74

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005851 / 1 | 0.005070 / 1 | 0.005620 / 1 |
| pre-attention-aggregation | boundary | 0.017602 / 1 | 0.017483 / 1 | 0.017774 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011702 / 1 | 0.011431 / 1 | 0.011561 / 1 |
| Q | nested | 3.159571 / 1 | 3.268123 / 1 | 3.503483 / 1 |
| K | nested | 2.324952 / 1 | 2.360488 / 1 | 2.822321 / 1 |
| V | nested | 2.285458 / 1 | 2.284957 / 1 | 2.838251 / 1 |
| B | nested | 0.025658 / 1 | 0.027121 / 1 | 0.024426 / 1 |
| FA | nested | 0.029385 / 1 | 0.034474 / 1 | 0.030817 / 1 |
| FB | nested | 0.051086 / 1 | 0.049883 / 1 | 0.054292 / 1 |
| G | nested | 2.040591 / 1 | 2.059987 / 1 | 2.041473 / 1 |
| O | nested | 2.064335 / 1 | 2.091085 / 1 | 2.072791 / 1 |
| attention | boundary | 13.131934 / 1 | 13.345783 / 1 | 14.503836 / 1 |
| attention-residual | boundary | 0.004068 / 1 | 0.003447 / 1 | 0.004258 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025117 / 1 | 0.025327 / 1 | 0.025006 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001974 / 1 | 0.001824 / 1 | 0.002254 / 1 |
| router-and-top16 | boundary | 0.569824 / 1 | 0.571387 / 1 | 0.572860 / 1 |
| EDOWN | nested | 0.597166 / 1 | 0.596625 / 1 | 0.596464 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.598658 / 1 | 0.598258 / 1 | 0.597987 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043341 / 1 | 0.022261 / 1 | 0.019446 / 1 |
| SH1 | nested | 1.015737 / 1 | 1.003204 / 1 | 1.443827 / 1 |
| SH3 | nested | 1.001460 / 1 | 1.009765 / 1 | 1.579209 / 1 |
| SH2 | nested | 0.999467 / 1 | 1.006229 / 1 | 1.708441 / 1 |
| shared-expert-during-read | boundary | 3.029698 / 1 | 3.032123 / 1 | 4.752125 / 1 |
| detail:expert-gate | nested | 49.167380 / 16 | 49.916763 / 16 | 54.232119 / 16 |
| detail:expert-up | nested | 42.674465 / 16 | 42.846699 / 16 | 54.410924 / 16 |
| detail:expert-activation | nested | 0.094858 / 16 | 0.096983 / 16 | 0.136065 / 16 |
| detail:expert-down | nested | 48.841439 / 16 | 48.892898 / 16 | 53.102021 / 16 |
| EUP | nested | 0.598578 / 1 | 0.603527 / 1 | 0.608537 / 1 |
| experts-mix-normalize-up | boundary | 141.835237 / 1 | 142.775364 / 1 | 166.704537 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002816 / 1 | 0.002896 / 1 | 0.002725 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.590974 / 1 | 0.584882 / 1 | 0.597486 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.949733 / 1 | 0.952919 / 1 | 0.983306 / 1 |
| worker:read | parallel worker | 0.067458 / 1 | 0.066785 / 1 | 0.069021 / 1 |
| worker:decode | parallel worker | 1068.030439 / 1 | 1072.403032 / 1 | 1081.456383 / 1 |
| worker:crc | parallel worker | 599.453743 / 1 | 603.202661 / 1 | 607.872538 / 1 |
| worker:math | parallel worker | 362.532115 / 1 | 363.684802 / 1 | 377.937106 / 1 |
| op:Q   int8 projection | nested | 16.192043 / 1 | 16.394055 / 1 | 19.322687 / 1 |
| op:X   mxfp4 expert proj | nested | 140.861600 / 1 | 141.835241 / 1 | 162.002598 / 1 |
| op:N   rmsnorm | nested | 0.046929 / 1 | 0.042159 / 1 | 0.030957 / 1 |
| op:L   l2 per-head | nested | 0.005320 / 1 | 0.005069 / 1 | 0.005861 / 1 |
| op:SiTU + sigma | nested | 0.101378 / 1 | 0.103395 / 1 | 0.146533 / 1 |
| op:C   shortconv | nested | 0.052027 / 1 | 0.063609 / 1 | 0.031199 / 1 |
| op:AR  snapshot aggregate | nested | 0.032871 / 1 | 0.032872 / 1 | 0.033251 / 1 |
| op:D   kda delta-rule | nested | 0.242913 / 1 | 0.242042 / 1 | 0.243475 / 1 |
| op:router dot product | nested | 0.567199 / 1 | 0.569053 / 1 | 0.570095 / 1 |
| op:top-k selection | nested | 0.002214 / 1 | 0.001994 / 1 | 0.002415 / 1 |
| op:alpha / beta / gate | nested | 0.073918 / 1 | 0.072546 / 1 | 0.058750 / 1 |
| detail:read-ahead-wait | nested | 0.001425 / 1 | 0.001343 / 1 | 3.779747 / 1 |
| total:layer | total | 160.250177 / 1 | 161.386481 / 1 | 188.226009 / 1 |

### Layer 75

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005861 / 1 | 0.006182 / 1 | 0.007293 / 1 |
| pre-attention-aggregation | boundary | 0.018505 / 1 | 0.017773 / 1 | 0.017964 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000021 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012413 / 1 | 0.012072 / 1 | 0.011952 / 1 |
| QA | nested | 0.369009 / 1 | 0.352869 / 1 | 0.381112 / 1 |
| QB | nested | 0.950255 / 1 | 0.920810 / 1 | 1.046755 / 1 |
| KA | nested | 0.137427 / 1 | 0.135032 / 1 | 0.141895 / 1 |
| KB | nested | 0.365814 / 1 | 0.423631 / 1 | 0.390079 / 1 |
| G | nested | 2.387719 / 1 | 2.655470 / 1 | 2.855122 / 1 |
| O | nested | 2.007700 / 1 | 2.007058 / 1 | 2.948476 / 1 |
| attention | boundary | 6.336323 / 1 | 6.614222 / 1 | 7.908229 / 1 |
| attention-residual | boundary | 0.003115 / 1 | 0.002745 / 1 | 0.003777 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024616 / 1 | 0.024686 / 1 | 0.027632 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001653 / 1 | 0.002004 / 1 | 0.002044 / 1 |
| router-and-top16 | boundary | 0.520663 / 1 | 0.499943 / 1 | 0.605060 / 1 |
| EDOWN | nested | 0.594310 / 1 | 0.595151 / 1 | 0.905521 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595822 / 1 | 0.596594 / 1 | 0.907896 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031329 / 1 | 0.006181 / 1 | 0.008947 / 1 |
| SH1 | nested | 1.016488 / 1 | 1.029794 / 1 | 1.468803 / 1 |
| SH3 | nested | 1.020756 / 1 | 1.009345 / 1 | 1.517023 / 1 |
| SH2 | nested | 1.017861 / 1 | 1.024593 / 1 | 1.035785 / 1 |
| shared-expert-during-read | boundary | 3.067969 / 1 | 3.076706 / 1 | 4.040546 / 1 |
| detail:expert-gate | nested | 49.702258 / 16 | 50.709250 / 16 | 54.123338 / 16 |
| detail:expert-up | nested | 42.560401 / 16 | 43.966736 / 16 | 54.058940 / 16 |
| detail:expert-activation | nested | 0.095927 / 16 | 0.099777 / 16 | 0.137624 / 16 |
| detail:expert-down | nested | 48.855598 / 16 | 49.071452 / 16 | 55.015562 / 16 |
| EUP | nested | 0.614207 / 1 | 0.611011 / 1 | 0.590894 / 1 |
| experts-mix-normalize-up | boundary | 142.269387 / 1 | 144.860338 / 1 | 164.360850 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002876 / 1 | 0.003056 / 1 | 0.003076 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.602715 / 1 | 0.602235 / 1 | 0.594510 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.609158 / 1 | 0.609368 / 1 | 0.601614 / 1 |
| worker:read | parallel worker | 0.065293 / 1 | 0.066167 / 1 | 0.069069 / 1 |
| worker:decode | parallel worker | 1066.714431 / 1 | 1073.096049 / 1 | 1084.343885 / 1 |
| worker:crc | parallel worker | 599.214691 / 1 | 602.785495 / 1 | 609.498091 / 1 |
| worker:math | parallel worker | 361.885403 / 1 | 364.054596 / 1 | 379.708116 / 1 |
| op:Q   int8 projection | nested | 10.480341 / 1 | 10.763572 / 1 | 13.280113 / 1 |
| op:X   mxfp4 expert proj | nested | 141.298903 / 1 | 143.934357 / 1 | 163.456862 / 1 |
| op:N   rmsnorm | nested | 0.031949 / 1 | 0.032772 / 1 | 0.032151 / 1 |
| op:SiTU + sigma | nested | 0.102862 / 1 | 0.106391 / 1 | 0.148324 / 1 |
| op:AR  snapshot aggregate | nested | 0.033252 / 1 | 0.032610 / 1 | 0.034774 / 1 |
| op:SA  softmax attention | nested | 0.006132 / 1 | 0.009397 / 1 | 0.015189 / 1 |
| op:router dot product | nested | 0.518067 / 1 | 0.497339 / 1 | 0.601894 / 1 |
| op:top-k selection | nested | 0.002274 / 1 | 0.002234 / 1 | 0.002796 / 1 |
| detail:read-ahead-wait | nested | 0.001171 / 1 | 0.001231 / 1 | 0.000942 / 1 |
| total:layer | total | 153.524016 / 1 | 156.352731 / 1 | 178.534600 / 1 |

### Layer 76

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004669 / 1 | 0.005350 / 1 | 0.004338 / 1 |
| pre-attention-aggregation | boundary | 0.016270 / 1 | 0.016521 / 1 | 0.017142 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011672 / 1 | 0.011802 / 1 | 0.012043 / 1 |
| Q | nested | 3.347281 / 1 | 3.283732 / 1 | 3.282671 / 1 |
| K | nested | 2.301979 / 1 | 2.367000 / 1 | 2.822652 / 1 |
| V | nested | 2.293403 / 1 | 2.298783 / 1 | 2.735969 / 1 |
| B | nested | 0.023183 / 1 | 0.027502 / 1 | 0.038191 / 1 |
| FA | nested | 0.027472 / 1 | 0.033623 / 1 | 0.037169 / 1 |
| FB | nested | 0.054702 / 1 | 0.058499 / 1 | 0.066765 / 1 |
| G | nested | 2.046301 / 1 | 2.089082 / 1 | 2.976839 / 1 |
| O | nested | 2.134827 / 1 | 2.090283 / 1 | 2.078211 / 1 |
| attention | boundary | 13.315196 / 1 | 13.325866 / 1 | 15.069563 / 1 |
| attention-residual | boundary | 0.003617 / 1 | 0.004318 / 1 | 0.003567 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024505 / 1 | 0.032981 / 1 | 0.024286 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001854 / 1 | 0.001763 / 1 | 0.002414 / 1 |
| router-and-top16 | boundary | 0.588539 / 1 | 0.572068 / 1 | 0.569795 / 1 |
| EDOWN | nested | 0.709084 / 1 | 0.590092 / 1 | 0.585964 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.710657 / 1 | 0.591685 / 1 | 0.587497 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.041678 / 1 | 0.005951 / 1 | 0.008426 / 1 |
| SH1 | nested | 1.054740 / 1 | 1.010437 / 1 | 0.979549 / 1 |
| SH3 | nested | 1.015076 / 1 | 1.013473 / 1 | 0.983397 / 1 |
| SH2 | nested | 0.998354 / 1 | 1.020095 / 1 | 0.981884 / 1 |
| shared-expert-during-read | boundary | 3.081164 / 1 | 3.057099 / 1 | 2.958515 / 1 |
| detail:expert-gate | nested | 49.386159 / 16 | 49.917100 / 16 | 53.785598 / 16 |
| detail:expert-up | nested | 42.581059 / 16 | 42.643589 / 16 | 53.521873 / 16 |
| detail:expert-activation | nested | 0.094024 / 16 | 0.096887 / 16 | 0.135272 / 16 |
| detail:expert-down | nested | 48.735674 / 16 | 48.952599 / 16 | 54.458375 / 16 |
| EUP | nested | 0.598848 / 1 | 0.597085 / 1 | 0.597155 / 1 |
| experts-mix-normalize-up | boundary | 141.843001 / 1 | 142.616759 / 1 | 162.926190 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002766 / 1 | 0.003166 / 1 | 0.002735 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.588009 / 1 | 0.584531 / 1 | 0.598638 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.958870 / 1 | 0.961836 / 1 | 0.997583 / 1 |
| worker:read | parallel worker | 0.067830 / 1 | 0.068661 / 1 | 0.066812 / 1 |
| worker:decode | parallel worker | 1068.399498 / 1 | 1072.956565 / 1 | 1078.500551 / 1 |
| worker:crc | parallel worker | 598.848319 / 1 | 602.245295 / 1 | 605.243895 / 1 |
| worker:math | parallel worker | 361.626881 / 1 | 362.829829 / 1 | 372.052577 / 1 |
| op:Q   int8 projection | nested | 16.603807 / 1 | 16.478121 / 1 | 18.164662 / 1 |
| op:X   mxfp4 expert proj | nested | 140.884092 / 1 | 141.693545 / 1 | 162.018097 / 1 |
| op:N   rmsnorm | nested | 0.045234 / 1 | 0.040616 / 1 | 0.031519 / 1 |
| op:L   l2 per-head | nested | 0.005601 / 1 | 0.005680 / 1 | 0.005330 / 1 |
| op:SiTU + sigma | nested | 0.100849 / 1 | 0.103689 / 1 | 0.141856 / 1 |
| op:C   shortconv | nested | 0.055904 / 1 | 0.061756 / 1 | 0.034434 / 1 |
| op:AR  snapshot aggregate | nested | 0.031359 / 1 | 0.034524 / 1 | 0.031990 / 1 |
| op:D   kda delta-rule | nested | 0.150932 / 1 | 0.115176 / 1 | 0.118781 / 1 |
| op:router dot product | nested | 0.585774 / 1 | 0.569223 / 1 | 0.567299 / 1 |
| op:top-k selection | nested | 0.002104 / 1 | 0.002334 / 1 | 0.002134 / 1 |
| op:alpha / beta / gate | nested | 0.072014 / 1 | 0.074039 / 1 | 0.057437 / 1 |
| detail:read-ahead-wait | nested | 0.001202 / 1 | 0.001291 / 1 | 0.000932 / 1 |
| total:layer | total | 160.625207 / 1 | 161.227734 / 1 | 183.207006 / 1 |

### Layer 77

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003697 / 1 | 0.003817 / 1 | 0.005781 / 1 |
| pre-attention-aggregation | boundary | 0.017252 / 1 | 0.017493 / 1 | 0.017613 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011432 / 1 | 0.011292 / 1 | 0.011712 / 1 |
| Q | nested | 3.297358 / 1 | 3.284334 / 1 | 3.475561 / 1 |
| K | nested | 2.342645 / 1 | 2.382639 / 1 | 2.749716 / 1 |
| V | nested | 2.327987 / 1 | 2.298632 / 1 | 2.853419 / 1 |
| B | nested | 0.020979 / 1 | 0.020969 / 1 | 0.027010 / 1 |
| FA | nested | 0.028363 / 1 | 0.028643 / 1 | 0.043601 / 1 |
| FB | nested | 0.050384 / 1 | 0.051256 / 1 | 0.060613 / 1 |
| G | nested | 2.064846 / 1 | 2.061851 / 1 | 2.645922 / 1 |
| O | nested | 2.071558 / 1 | 2.079714 / 1 | 2.072931 / 1 |
| attention | boundary | 13.354770 / 1 | 13.364849 / 1 | 15.086124 / 1 |
| attention-residual | boundary | 0.003868 / 1 | 0.003967 / 1 | 0.003637 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024867 / 1 | 0.024616 / 1 | 0.024014 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002044 / 1 | 0.001764 / 1 | 0.001793 / 1 |
| router-and-top16 | boundary | 0.570986 / 1 | 0.573461 / 1 | 0.570616 / 1 |
| EDOWN | nested | 0.590873 / 1 | 0.589711 / 1 | 0.588169 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.592426 / 1 | 0.591284 / 1 | 0.589692 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026690 / 1 | 0.017603 / 1 | 0.008325 / 1 |
| SH1 | nested | 1.003013 / 1 | 1.005969 / 1 | 1.008814 / 1 |
| SH3 | nested | 0.998535 / 1 | 1.009044 / 1 | 1.004606 / 1 |
| SH2 | nested | 1.004456 / 1 | 1.012481 / 1 | 1.008494 / 1 |
| shared-expert-during-read | boundary | 3.019449 / 1 | 3.040518 / 1 | 3.035149 / 1 |
| detail:expert-gate | nested | 49.350189 / 16 | 48.982559 / 16 | 53.704938 / 16 |
| detail:expert-up | nested | 42.672028 / 16 | 42.719701 / 16 | 52.331943 / 16 |
| detail:expert-activation | nested | 0.093604 / 16 | 0.098994 / 16 | 0.129124 / 16 |
| detail:expert-down | nested | 48.851027 / 16 | 49.021818 / 16 | 53.005382 / 16 |
| EUP | nested | 0.607505 / 1 | 0.609147 / 1 | 0.602415 / 1 |
| experts-mix-normalize-up | boundary | 142.001538 / 1 | 141.859303 / 1 | 160.214737 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002665 / 1 | 0.003006 / 1 | 0.003186 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589110 / 1 | 0.582468 / 1 | 0.590233 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.941759 / 1 | 0.949503 / 1 | 0.972906 / 1 |
| worker:read | parallel worker | 0.068152 / 1 | 0.068038 / 1 | 0.073439 / 1 |
| worker:decode | parallel worker | 1067.822998 / 1 | 1070.923169 / 1 | 1082.756111 / 1 |
| worker:crc | parallel worker | 598.055430 / 1 | 600.559752 / 1 | 606.351142 / 1 |
| worker:math | parallel worker | 363.953531 / 1 | 364.176095 / 1 | 376.927101 / 1 |
| op:Q   int8 projection | nested | 16.407131 / 1 | 16.432948 / 1 | 18.139689 / 1 |
| op:X   mxfp4 expert proj | nested | 141.054799 / 1 | 140.908460 / 1 | 159.292736 / 1 |
| op:N   rmsnorm | nested | 0.041396 / 1 | 0.043813 / 1 | 0.030668 / 1 |
| op:L   l2 per-head | nested | 0.005440 / 1 | 0.005631 / 1 | 0.006292 / 1 |
| op:SiTU + sigma | nested | 0.100478 / 1 | 0.105907 / 1 | 0.135733 / 1 |
| op:C   shortconv | nested | 0.054943 / 1 | 0.051395 / 1 | 0.030848 / 1 |
| op:AR  snapshot aggregate | nested | 0.032000 / 1 | 0.031990 / 1 | 0.032390 / 1 |
| op:D   kda delta-rule | nested | 0.244647 / 1 | 0.242653 / 1 | 0.240560 / 1 |
| op:router dot product | nested | 0.568592 / 1 | 0.570986 / 1 | 0.567560 / 1 |
| op:top-k selection | nested | 0.001984 / 1 | 0.002074 / 1 | 0.002364 / 1 |
| op:alpha / beta / gate | nested | 0.081031 / 1 | 0.072205 / 1 | 0.058920 / 1 |
| detail:read-ahead-wait | nested | 0.001344 / 1 | 0.001283 / 1 | 0.000880 / 1 |
| total:layer | total | 160.595161 / 1 | 160.484537 / 1 | 180.569130 / 1 |

### Layer 78

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004349 / 1 | 0.005320 / 1 | 0.005130 / 1 |
| pre-attention-aggregation | boundary | 0.016401 / 1 | 0.017223 / 1 | 0.018164 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.010931 / 1 | 0.010860 / 1 | 0.011712 / 1 |
| Q | nested | 3.253276 / 1 | 3.269506 / 1 | 3.319289 / 1 |
| K | nested | 2.327727 / 1 | 2.361330 / 1 | 2.872174 / 1 |
| V | nested | 2.300216 / 1 | 2.302591 / 1 | 2.742462 / 1 |
| B | nested | 0.023844 / 1 | 0.025217 / 1 | 0.029374 / 1 |
| FA | nested | 0.034014 / 1 | 0.032321 / 1 | 0.042560 / 1 |
| FB | nested | 0.050344 / 1 | 0.052127 / 1 | 0.084107 / 1 |
| G | nested | 2.035711 / 1 | 2.089782 / 1 | 2.954718 / 1 |
| O | nested | 2.067311 / 1 | 2.087699 / 1 | 2.072240 / 1 |
| attention | boundary | 13.278197 / 1 | 13.402900 / 1 | 15.274927 / 1 |
| attention-residual | boundary | 0.003898 / 1 | 0.003788 / 1 | 0.003767 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025177 / 1 | 0.025297 / 1 | 0.024356 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002144 / 1 | 0.001754 / 1 | 0.001793 / 1 |
| router-and-top16 | boundary | 0.567570 / 1 | 0.578230 / 1 | 0.570926 / 1 |
| EDOWN | nested | 0.590082 / 1 | 0.598528 / 1 | 0.587397 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.591635 / 1 | 0.600221 / 1 | 0.588970 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.025898 / 1 | 0.011621 / 1 | 0.008556 / 1 |
| SH1 | nested | 1.007903 / 1 | 1.018583 / 1 | 0.995659 / 1 |
| SH3 | nested | 1.007721 / 1 | 1.015025 / 1 | 1.000067 / 1 |
| SH2 | nested | 0.999176 / 1 | 1.010847 / 1 | 1.001270 / 1 |
| shared-expert-during-read | boundary | 3.028346 / 1 | 3.061217 / 1 | 3.010442 / 1 |
| detail:expert-gate | nested | 49.197569 / 16 | 51.037952 / 16 | 54.896151 / 16 |
| detail:expert-up | nested | 42.590435 / 16 | 44.992965 / 16 | 53.781127 / 16 |
| detail:expert-activation | nested | 0.094114 / 16 | 0.100257 / 16 | 0.134903 / 16 |
| detail:expert-down | nested | 48.830409 / 16 | 49.058681 / 16 | 51.852146 / 16 |
| EUP | nested | 0.612564 / 1 | 0.603828 / 1 | 0.734221 / 1 |
| experts-mix-normalize-up | boundary | 141.760437 / 1 | 146.227061 / 1 | 161.816799 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.002925 / 1 | 0.002785 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587266 / 1 | 0.586234 / 1 | 0.608537 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.955524 / 1 | 0.991962 / 1 | 0.997473 / 1 |
| worker:read | parallel worker | 0.186036 / 1 | 0.208199 / 1 | 0.186658 / 1 |
| worker:decode | parallel worker | 1067.233378 / 1 | 1076.397537 / 1 | 1080.912884 / 1 |
| worker:crc | parallel worker | 598.805276 / 1 | 602.767569 / 1 | 606.569043 / 1 |
| worker:math | parallel worker | 362.916382 / 1 | 364.994569 / 1 | 374.204094 / 1 |
| op:Q   int8 projection | nested | 16.308456 / 1 | 16.465871 / 1 | 18.433977 / 1 |
| op:X   mxfp4 expert proj | nested | 140.801728 / 1 | 145.283337 / 1 | 160.785775 / 1 |
| op:N   rmsnorm | nested | 0.043102 / 1 | 0.046077 / 1 | 0.031720 / 1 |
| op:L   l2 per-head | nested | 0.005580 / 1 | 0.006141 / 1 | 0.005670 / 1 |
| op:SiTU + sigma | nested | 0.100919 / 1 | 0.110327 / 1 | 0.141507 / 1 |
| op:C   shortconv | nested | 0.063649 / 1 | 0.060694 / 1 | 0.030737 / 1 |
| op:AR  snapshot aggregate | nested | 0.031840 / 1 | 0.032490 / 1 | 0.033252 / 1 |
| op:D   kda delta-rule | nested | 0.244136 / 1 | 0.237012 / 1 | 0.239267 / 1 |
| op:router dot product | nested | 0.565116 / 1 | 0.575575 / 1 | 0.568111 / 1 |
| op:top-k selection | nested | 0.002084 / 1 | 0.002395 / 1 | 0.002275 / 1 |
| op:alpha / beta / gate | nested | 0.073978 / 1 | 0.074079 / 1 | 0.062407 / 1 |
| detail:read-ahead-wait | nested | 0.001323 / 1 | 0.001120 / 1 | 0.001012 / 1 |
| total:layer | total | 160.295320 / 1 | 164.962138 / 1 | 182.358551 / 1 |

### Layer 79

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005390 / 1 | 0.005621 / 1 | 0.004619 / 1 |
| pre-attention-aggregation | boundary | 0.017122 / 1 | 0.018624 / 1 | 0.017392 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000100 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011511 / 1 | 0.012653 / 1 | 0.011662 / 1 |
| QA | nested | 0.356165 / 1 | 0.394356 / 1 | 0.372896 / 1 |
| QB | nested | 0.941238 / 1 | 0.947449 / 1 | 0.994307 / 1 |
| KA | nested | 0.133359 / 1 | 0.147576 / 1 | 0.142827 / 1 |
| KB | nested | 0.410236 / 1 | 0.422870 / 1 | 0.390319 / 1 |
| G | nested | 2.780463 / 1 | 2.113526 / 1 | 2.870992 / 1 |
| O | nested | 2.049568 / 1 | 2.108658 / 1 | 2.839072 / 1 |
| attention | boundary | 6.786063 / 1 | 6.262606 / 1 | 7.746577 / 1 |
| attention-residual | boundary | 0.002705 / 1 | 0.003006 / 1 | 0.003277 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024646 / 1 | 0.024877 / 1 | 0.028754 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001793 / 1 | 0.001893 / 1 | 0.002434 / 1 |
| router-and-top16 | boundary | 0.525652 / 1 | 0.522566 / 1 | 0.585023 / 1 |
| EDOWN | nested | 0.595563 / 1 | 0.608396 / 1 | 0.855598 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.597095 / 1 | 0.609929 / 1 | 0.858414 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.053611 / 1 | 0.006272 / 1 | 0.011001 / 1 |
| SH1 | nested | 1.016478 / 1 | 1.050201 / 1 | 1.466629 / 1 |
| SH3 | nested | 1.040793 / 1 | 1.030123 / 1 | 1.387842 / 1 |
| SH2 | nested | 1.024904 / 1 | 1.034953 / 1 | 1.453424 / 1 |
| shared-expert-during-read | boundary | 3.095360 / 1 | 3.128211 / 1 | 4.326380 / 1 |
| detail:expert-gate | nested | 49.852368 / 16 | 50.991070 / 16 | 54.575161 / 16 |
| detail:expert-up | nested | 42.497624 / 16 | 45.961607 / 16 | 53.971491 / 16 |
| detail:expert-activation | nested | 0.099376 / 16 | 0.105129 / 16 | 0.134804 / 16 |
| detail:expert-down | nested | 48.691988 / 16 | 48.980142 / 16 | 54.333109 / 16 |
| EUP | nested | 0.615860 / 1 | 0.611151 / 1 | 0.717640 / 1 |
| experts-mix-normalize-up | boundary | 142.225596 / 1 | 147.070465 / 1 | 164.157570 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003056 / 1 | 0.002484 / 1 | 0.002955 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.603336 / 1 | 0.596144 / 1 | 0.609879 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.612304 / 1 | 0.603587 / 1 | 0.617183 / 1 |
| worker:read | parallel worker | 0.066297 / 1 | 0.067518 / 1 | 0.068338 / 1 |
| worker:decode | parallel worker | 1067.110582 / 1 | 1075.132192 / 1 | 1082.808225 / 1 |
| worker:crc | parallel worker | 598.661839 / 1 | 603.398106 / 1 | 607.812281 / 1 |
| worker:math | parallel worker | 361.949916 / 1 | 365.016465 / 1 | 377.106294 / 1 |
| op:Q   int8 projection | nested | 10.963394 / 1 | 10.468028 / 1 | 13.490055 / 1 |
| op:X   mxfp4 expert proj | nested | 141.230666 / 1 | 146.134377 / 1 | 163.138315 / 1 |
| op:N   rmsnorm | nested | 0.031089 / 1 | 0.032941 / 1 | 0.032863 / 1 |
| op:SiTU + sigma | nested | 0.106166 / 1 | 0.111730 / 1 | 0.145000 / 1 |
| op:AR  snapshot aggregate | nested | 0.031729 / 1 | 0.033502 / 1 | 0.033904 / 1 |
| op:SA  softmax attention | nested | 0.005881 / 1 | 0.008586 / 1 | 0.013515 / 1 |
| op:router dot product | nested | 0.522697 / 1 | 0.519690 / 1 | 0.581566 / 1 |
| op:top-k selection | nested | 0.002645 / 1 | 0.002495 / 1 | 0.002955 / 1 |
| detail:read-ahead-wait | nested | 0.001455 / 1 | 0.001231 / 1 | 0.000852 / 1 |
| total:layer | total | 153.983183 / 1 | 158.295729 / 1 | 178.402253 / 1 |

### Layer 80

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005010 / 1 | 0.004208 / 1 | 0.008266 / 1 |
| pre-attention-aggregation | boundary | 0.015970 / 1 | 0.016511 / 1 | 0.017192 / 1 |
| snapshot-push | boundary | 0.000030 / 1 | 0.000021 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011422 / 1 | 0.011141 / 1 | 0.011591 / 1 |
| Q | nested | 3.300043 / 1 | 3.331211 / 1 | 3.285015 / 1 |
| K | nested | 2.322828 / 1 | 2.347414 / 1 | 2.877383 / 1 |
| V | nested | 2.284917 / 1 | 2.284606 / 1 | 2.800269 / 1 |
| B | nested | 0.022211 / 1 | 0.021420 / 1 | 0.030406 / 1 |
| FA | nested | 0.034455 / 1 | 0.033963 / 1 | 0.039173 / 1 |
| FB | nested | 0.055604 / 1 | 0.053400 / 1 | 0.056976 / 1 |
| G | nested | 2.047213 / 1 | 2.069575 / 1 | 2.838150 / 1 |
| O | nested | 2.100593 / 1 | 2.092066 / 1 | 2.438885 / 1 |
| attention | boundary | 13.261987 / 1 | 13.298596 / 1 | 15.428984 / 1 |
| attention-residual | boundary | 0.003898 / 1 | 0.004308 / 1 | 0.003777 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024546 / 1 | 0.025137 / 1 | 0.024796 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002034 / 1 | 0.001894 / 1 | 0.001633 / 1 |
| router-and-top16 | boundary | 0.559104 / 1 | 0.576126 / 1 | 0.576046 / 1 |
| EDOWN | nested | 0.593558 / 1 | 0.596113 / 1 | 0.683637 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595181 / 1 | 0.597736 / 1 | 0.685921 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.036708 / 1 | 0.005811 / 1 | 0.009167 / 1 |
| SH1 | nested | 1.000218 / 1 | 1.016188 / 1 | 1.491185 / 1 |
| SH3 | nested | 1.008263 / 1 | 1.022460 / 1 | 1.349720 / 1 |
| SH2 | nested | 1.002502 / 1 | 1.021808 / 1 | 1.500371 / 1 |
| shared-expert-during-read | boundary | 3.024308 / 1 | 3.073560 / 1 | 4.361005 / 1 |
| detail:expert-gate | nested | 49.629714 / 16 | 49.387528 / 16 | 54.091559 / 16 |
| detail:expert-up | nested | 42.604277 / 16 | 42.727142 / 16 | 54.130038 / 16 |
| detail:expert-activation | nested | 0.097131 / 16 | 0.095309 / 16 | 0.139409 / 16 |
| detail:expert-down | nested | 48.804275 / 16 | 49.033592 / 16 | 55.322347 / 16 |
| EUP | nested | 0.615630 / 1 | 0.605751 / 1 | 0.597245 / 1 |
| experts-mix-normalize-up | boundary | 142.206180 / 1 | 142.271754 / 1 | 169.252766 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002735 / 1 | 0.002875 / 1 | 0.003416 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.602495 / 1 | 0.587818 / 1 | 0.598198 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.968488 / 1 | 0.968599 / 1 | 0.996691 / 1 |
| worker:read | parallel worker | 0.067964 / 1 | 0.068193 / 1 | 0.070740 / 1 |
| worker:decode | parallel worker | 1067.835945 / 1 | 1073.925465 / 1 | 1080.174821 / 1 |
| worker:crc | parallel worker | 600.432682 / 1 | 602.021719 / 1 | 606.577668 / 1 |
| worker:math | parallel worker | 362.355625 / 1 | 363.712935 / 1 | 373.547443 / 1 |
| op:Q   int8 projection | nested | 16.386685 / 1 | 16.494543 / 1 | 19.986593 / 1 |
| op:X   mxfp4 expert proj | nested | 141.233405 / 1 | 141.331093 / 1 | 163.808479 / 1 |
| op:N   rmsnorm | nested | 0.040024 / 1 | 0.044653 / 1 | 0.031078 / 1 |
| op:L   l2 per-head | nested | 0.005871 / 1 | 0.005420 / 1 | 0.005240 / 1 |
| op:SiTU + sigma | nested | 0.103582 / 1 | 0.102052 / 1 | 0.149971 / 1 |
| op:C   shortconv | nested | 0.065602 / 1 | 0.054953 / 1 | 0.035055 / 1 |
| op:AR  snapshot aggregate | nested | 0.030827 / 1 | 0.031619 / 1 | 0.032521 / 1 |
| op:D   kda delta-rule | nested | 0.148457 / 1 | 0.106088 / 1 | 0.133530 / 1 |
| op:router dot product | nested | 0.555788 / 1 | 0.573521 / 1 | 0.573010 / 1 |
| op:top-k selection | nested | 0.002485 / 1 | 0.002224 / 1 | 0.002655 / 1 |
| op:alpha / beta / gate | nested | 0.072485 / 1 | 0.072405 / 1 | 0.057798 / 1 |
| detail:read-ahead-wait | nested | 0.001426 / 1 | 0.001271 / 1 | 4.543304 / 1 |
| total:layer | total | 160.739451 / 1 | 160.881277 / 1 | 191.405828 / 1 |

### Layer 81

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.003957 / 1 | 0.005110 / 1 | 0.005601 / 1 |
| pre-attention-aggregation | boundary | 0.016531 / 1 | 0.017653 / 1 | 0.017953 / 1 |
| snapshot-push | boundary | 0.000110 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011752 / 1 | 0.011211 / 1 | 0.011381 / 1 |
| Q | nested | 3.329147 / 1 | 3.311103 / 1 | 3.300053 / 1 |
| K | nested | 2.386748 / 1 | 2.400583 / 1 | 2.358114 / 1 |
| V | nested | 2.314563 / 1 | 2.306688 / 1 | 2.319502 / 1 |
| B | nested | 0.024456 / 1 | 0.027571 / 1 | 0.027021 / 1 |
| FA | nested | 0.030858 / 1 | 0.041447 / 1 | 0.042740 / 1 |
| FB | nested | 0.047409 / 1 | 0.048291 / 1 | 0.050955 / 1 |
| G | nested | 2.052794 / 1 | 2.070146 / 1 | 2.070266 / 1 |
| O | nested | 2.073643 / 1 | 2.093459 / 1 | 2.071017 / 1 |
| attention | boundary | 13.365610 / 1 | 13.456811 / 1 | 13.447183 / 1 |
| attention-residual | boundary | 0.003447 / 1 | 0.003627 / 1 | 0.003697 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025137 / 1 | 0.025157 / 1 | 0.024566 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001913 / 1 | 0.001964 / 1 | 0.002073 / 1 |
| router-and-top16 | boundary | 0.567250 / 1 | 0.575154 / 1 | 0.571707 / 1 |
| EDOWN | nested | 0.593809 / 1 | 0.595803 / 1 | 0.600011 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.595372 / 1 | 0.597466 / 1 | 0.601694 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.018355 / 1 | 0.012013 / 1 | 0.021570 / 1 |
| SH1 | nested | 1.401607 / 1 | 1.015557 / 1 | 1.010628 / 1 |
| SH3 | nested | 1.425382 / 1 | 1.016618 / 1 | 1.008283 / 1 |
| SH2 | nested | 1.388463 / 1 | 1.018963 / 1 | 1.002162 / 1 |
| shared-expert-during-read | boundary | 4.234708 / 1 | 3.064783 / 1 | 3.034698 / 1 |
| detail:expert-gate | nested | 49.521854 / 16 | 50.203774 / 16 | 54.118717 / 16 |
| detail:expert-up | nested | 44.176951 / 16 | 43.587740 / 16 | 53.476153 / 16 |
| detail:expert-activation | nested | 0.101039 / 16 | 0.099032 / 16 | 0.135182 / 16 |
| detail:expert-down | nested | 48.953381 / 16 | 49.195803 / 16 | 52.974292 / 16 |
| EUP | nested | 0.604118 / 1 | 0.600421 / 1 | 0.611272 / 1 |
| experts-mix-normalize-up | boundary | 143.776983 / 1 | 144.114305 / 1 | 161.767978 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002745 / 1 | 0.003236 / 1 | 0.002655 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591194 / 1 | 0.588238 / 1 | 0.594741 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.946788 / 1 | 0.976494 / 1 | 0.986643 / 1 |
| worker:read | parallel worker | 0.187223 / 1 | 0.205250 / 1 | 0.189623 / 1 |
| worker:decode | parallel worker | 1074.795375 / 1 | 1080.649261 / 1 | 1088.569740 / 1 |
| worker:crc | parallel worker | 599.694653 / 1 | 603.045705 / 1 | 608.325285 / 1 |
| worker:math | parallel worker | 363.956254 / 1 | 363.941741 / 1 | 377.479709 / 1 |
| op:Q   int8 projection | nested | 17.671294 / 1 | 16.545207 / 1 | 16.470531 / 1 |
| op:X   mxfp4 expert proj | nested | 142.846017 / 1 | 143.178927 / 1 | 160.833282 / 1 |
| op:N   rmsnorm | nested | 0.039192 / 1 | 0.042269 / 1 | 0.030757 / 1 |
| op:L   l2 per-head | nested | 0.005230 / 1 | 0.005149 / 1 | 0.005429 / 1 |
| op:SiTU + sigma | nested | 0.112371 / 1 | 0.106388 / 1 | 0.141625 / 1 |
| op:C   shortconv | nested | 0.060372 / 1 | 0.057126 / 1 | 0.031489 / 1 |
| op:AR  snapshot aggregate | nested | 0.032240 / 1 | 0.032991 / 1 | 0.033102 / 1 |
| op:D   kda delta-rule | nested | 0.249796 / 1 | 0.239337 / 1 | 0.240199 / 1 |
| op:router dot product | nested | 0.564464 / 1 | 0.572249 / 1 | 0.568812 / 1 |
| op:top-k selection | nested | 0.002415 / 1 | 0.002284 / 1 | 0.002134 / 1 |
| op:alpha / beta / gate | nested | 0.075351 / 1 | 0.071905 / 1 | 0.058539 / 1 |
| detail:read-ahead-wait | nested | 0.001362 / 1 | 0.001215 / 1 | 0.000882 / 1 |
| total:layer | total | 163.593079 / 1 | 162.888406 / 1 | 180.521511 / 1 |

### Layer 82

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004217 / 1 | 0.004609 / 1 | 0.005260 / 1 |
| pre-attention-aggregation | boundary | 0.017553 / 1 | 0.018635 / 1 | 0.018985 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000090 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.011421 / 1 | 0.012413 / 1 | 0.011742 / 1 |
| Q | nested | 3.293240 / 1 | 2.750005 / 1 | 3.298800 / 1 |
| K | nested | 2.338187 / 1 | 2.376869 / 1 | 2.867254 / 1 |
| V | nested | 2.285187 / 1 | 2.303762 / 1 | 2.754304 / 1 |
| B | nested | 0.021380 / 1 | 0.021179 / 1 | 0.028904 / 1 |
| FA | nested | 0.029976 / 1 | 0.030116 / 1 | 0.035175 / 1 |
| FB | nested | 0.051487 / 1 | 0.051436 / 1 | 0.055533 / 1 |
| G | nested | 2.048836 / 1 | 2.056701 / 1 | 2.910335 / 1 |
| O | nested | 2.058694 / 1 | 2.095573 / 1 | 2.060939 / 1 |
| attention | boundary | 13.275301 / 1 | 12.916071 / 1 | 15.166834 / 1 |
| attention-residual | boundary | 0.003607 / 1 | 0.003978 / 1 | 0.003626 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024616 / 1 | 0.024686 / 1 | 0.025187 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002024 / 1 | 0.001763 / 1 | 0.002374 / 1 |
| router-and-top16 | boundary | 0.573892 / 1 | 0.578982 / 1 | 0.579712 / 1 |
| EDOWN | nested | 0.588849 / 1 | 0.587237 / 1 | 0.587217 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590482 / 1 | 0.588870 / 1 | 0.588769 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.021841 / 1 | 0.012243 / 1 | 0.008936 / 1 |
| SH1 | nested | 1.000748 / 1 | 1.002843 / 1 | 0.992885 / 1 |
| SH3 | nested | 1.006931 / 1 | 1.002983 / 1 | 1.000929 / 1 |
| SH2 | nested | 1.002803 / 1 | 1.006720 / 1 | 0.996060 / 1 |
| shared-expert-during-read | boundary | 3.024217 / 1 | 3.026192 / 1 | 3.003359 / 1 |
| detail:expert-gate | nested | 49.892802 / 16 | 50.446778 / 16 | 54.035932 / 16 |
| detail:expert-up | nested | 42.692528 / 16 | 43.549499 / 16 | 53.551000 / 16 |
| detail:expert-activation | nested | 0.100529 / 16 | 0.100961 / 16 | 0.144628 / 16 |
| detail:expert-down | nested | 48.936290 / 16 | 49.130305 / 16 | 54.429708 / 16 |
| EUP | nested | 0.616471 / 1 | 0.601062 / 1 | 0.590733 / 1 |
| experts-mix-normalize-up | boundary | 142.668663 / 1 | 144.263985 / 1 | 163.194191 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003056 / 1 | 0.003076 / 1 | 0.003236 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589711 / 1 | 0.589270 / 1 | 0.595753 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.939485 / 1 | 0.947861 / 1 | 0.966164 / 1 |
| worker:read | parallel worker | 0.067149 / 1 | 0.068252 / 1 | 0.068655 / 1 |
| worker:decode | parallel worker | 1074.261156 / 1 | 1080.258477 / 1 | 1091.233388 / 1 |
| worker:crc | parallel worker | 598.473610 / 1 | 602.109837 / 1 | 608.386317 / 1 |
| worker:math | parallel worker | 362.001807 / 1 | 363.376528 / 1 | 377.373049 / 1 |
| op:Q   int8 projection | nested | 16.341338 / 1 | 15.885105 / 1 | 18.177454 / 1 |
| op:X   mxfp4 expert proj | nested | 141.713928 / 1 | 143.321724 / 1 | 162.286474 / 1 |
| op:N   rmsnorm | nested | 0.045423 / 1 | 0.048940 / 1 | 0.031249 / 1 |
| op:L   l2 per-head | nested | 0.006132 / 1 | 0.005029 / 1 | 0.005891 / 1 |
| op:SiTU + sigma | nested | 0.107622 / 1 | 0.108163 / 1 | 0.151343 / 1 |
| op:C   shortconv | nested | 0.057047 / 1 | 0.066795 / 1 | 0.030867 / 1 |
| op:AR  snapshot aggregate | nested | 0.032682 / 1 | 0.033683 / 1 | 0.034314 / 1 |
| op:D   kda delta-rule | nested | 0.248924 / 1 | 0.239498 / 1 | 0.244156 / 1 |
| op:router dot product | nested | 0.571237 / 1 | 0.576156 / 1 | 0.577118 / 1 |
| op:top-k selection | nested | 0.002255 / 1 | 0.002314 / 1 | 0.002184 / 1 |
| op:alpha / beta / gate | nested | 0.074119 / 1 | 0.079799 / 1 | 0.058499 / 1 |
| detail:read-ahead-wait | nested | 0.001182 / 1 | 0.001292 / 1 | 0.001032 / 1 |
| total:layer | total | 161.183900 / 1 | 162.428608 / 1 | 183.601934 / 1 |

### Layer 83

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.005179 / 1 | 0.004830 / 1 | 0.004628 / 1 |
| pre-attention-aggregation | boundary | 0.017082 / 1 | 0.016471 / 1 | 0.017773 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000300 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011161 / 1 | 0.011422 / 1 | 0.011311 / 1 |
| QA | nested | 0.359141 / 1 | 0.342941 / 1 | 0.357338 / 1 |
| QB | nested | 0.936919 / 1 | 0.946367 / 1 | 0.929817 / 1 |
| KA | nested | 0.137567 / 1 | 0.133159 / 1 | 0.138679 / 1 |
| KB | nested | 0.412690 / 1 | 0.414854 / 1 | 0.407591 / 1 |
| G | nested | 2.754434 / 1 | 2.791483 / 1 | 2.763862 / 1 |
| O | nested | 2.044809 / 1 | 2.086577 / 1 | 2.034499 / 1 |
| attention | boundary | 6.765314 / 1 | 6.839863 / 1 | 6.749976 / 1 |
| attention-residual | boundary | 0.002695 / 1 | 0.002755 / 1 | 0.002796 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025037 / 1 | 0.025277 / 1 | 0.025007 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001894 / 1 | 0.001784 / 1 | 0.001593 / 1 |
| router-and-top16 | boundary | 0.523948 / 1 | 0.525752 / 1 | 0.523428 / 1 |
| EDOWN | nested | 0.602796 / 1 | 0.608696 / 1 | 0.606212 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.604419 / 1 | 0.610309 / 1 | 0.607845 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026269 / 1 | 0.005921 / 1 | 0.007864 / 1 |
| SH1 | nested | 1.029702 / 1 | 1.043459 / 1 | 1.030244 / 1 |
| SH3 | nested | 1.019134 / 1 | 1.033460 / 1 | 1.025425 / 1 |
| SH2 | nested | 1.023782 / 1 | 1.053508 / 1 | 1.027839 / 1 |
| shared-expert-during-read | boundary | 3.086474 / 1 | 3.144613 / 1 | 3.097545 / 1 |
| detail:expert-gate | nested | 49.523848 / 16 | 50.809103 / 16 | 54.268049 / 16 |
| detail:expert-up | nested | 42.776554 / 16 | 45.268169 / 16 | 53.763647 / 16 |
| detail:expert-activation | nested | 0.104145 / 16 | 0.108934 / 16 | 0.145762 / 16 |
| detail:expert-down | nested | 49.103191 / 16 | 49.421825 / 16 | 55.049797 / 16 |
| EUP | nested | 0.606744 / 1 | 0.606923 / 1 | 0.589500 / 1 |
| experts-mix-normalize-up | boundary | 142.549651 / 1 | 146.660280 / 1 | 164.252418 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002775 / 1 | 0.002695 / 1 | 0.003476 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.596354 / 1 | 0.593067 / 1 | 0.596975 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.602886 / 1 | 0.599640 / 1 | 0.604288 / 1 |
| worker:read | parallel worker | 0.066609 / 1 | 0.066027 / 1 | 0.066729 / 1 |
| worker:decode | parallel worker | 1079.725462 / 1 | 1090.339104 / 1 | 1090.534423 / 1 |
| worker:crc | parallel worker | 598.794489 / 1 | 603.151765 / 1 | 606.231581 / 1 |
| worker:math | parallel worker | 362.403094 / 1 | 365.364735 / 1 | 373.273312 / 1 |
| op:Q   int8 projection | nested | 10.926566 / 1 | 11.060274 / 1 | 10.909854 / 1 |
| op:X   mxfp4 expert proj | nested | 141.600289 / 1 | 145.700323 / 1 | 163.355926 / 1 |
| op:N   rmsnorm | nested | 0.032019 / 1 | 0.033441 / 1 | 0.030875 / 1 |
| op:SiTU + sigma | nested | 0.111058 / 1 | 0.116656 / 1 | 0.152614 / 1 |
| op:AR  snapshot aggregate | nested | 0.032090 / 1 | 0.031217 / 1 | 0.032701 / 1 |
| op:SA  softmax attention | nested | 0.006091 / 1 | 0.009548 / 1 | 0.012343 / 1 |
| op:router dot product | nested | 0.518569 / 1 | 0.522907 / 1 | 0.520402 / 1 |
| op:top-k selection | nested | 0.005029 / 1 | 0.002515 / 1 | 0.002604 / 1 |
| detail:read-ahead-wait | nested | 0.001192 / 1 | 0.001313 / 1 | 0.001043 / 1 |
| total:layer | total | 154.246795 / 1 | 158.474162 / 1 | 175.934324 / 1 |

### Layer 84

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.004058 / 1 | 0.004368 / 1 | 0.004900 / 1 |
| pre-attention-aggregation | boundary | 0.016140 / 1 | 0.017874 / 1 | 0.017432 / 1 |
| snapshot-push | boundary | 0.001994 / 1 | 0.002003 / 1 | 0.002515 / 1 |
| pre-attention-normalization | boundary | 0.011471 / 1 | 0.012663 / 1 | 0.012183 / 1 |
| Q | nested | 3.269967 / 1 | 2.916406 / 1 | 3.366307 / 1 |
| K | nested | 2.328449 / 1 | 2.369796 / 1 | 2.845795 / 1 |
| V | nested | 2.298904 / 1 | 2.329029 / 1 | 2.783588 / 1 |
| B | nested | 0.026881 / 1 | 0.025367 / 1 | 0.033943 / 1 |
| FA | nested | 0.030417 / 1 | 0.034524 / 1 | 0.037160 / 1 |
| FB | nested | 0.051056 / 1 | 0.061475 / 1 | 0.065292 / 1 |
| G | nested | 2.060949 / 1 | 2.067752 / 1 | 2.096345 / 1 |
| O | nested | 2.086166 / 1 | 2.105893 / 1 | 2.078852 / 1 |
| attention | boundary | 13.237371 / 1 | 13.000308 / 1 | 14.367532 / 1 |
| attention-residual | boundary | 0.001954 / 1 | 0.002003 / 1 | 0.002335 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024696 / 1 | 0.025517 / 1 | 0.026078 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002104 / 1 | 0.001603 / 1 | 0.002044 / 1 |
| router-and-top16 | boundary | 0.557962 / 1 | 0.573431 / 1 | 0.563843 / 1 |
| EDOWN | nested | 0.596444 / 1 | 0.592947 / 1 | 0.582508 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.598167 / 1 | 0.594630 / 1 | 0.584241 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.027722 / 1 | 0.019296 / 1 | 0.015399 / 1 |
| SH1 | nested | 1.377733 / 1 | 1.414461 / 1 | 1.442354 / 1 |
| SH3 | nested | 1.398842 / 1 | 1.419130 / 1 | 1.560404 / 1 |
| SH2 | nested | 1.382472 / 1 | 1.401488 / 1 | 1.483290 / 1 |
| shared-expert-during-read | boundary | 4.177932 / 1 | 4.253745 / 1 | 4.505334 / 1 |
| detail:expert-gate | nested | 49.716485 / 16 | 50.233848 / 16 | 54.440500 / 16 |
| detail:expert-up | nested | 42.646814 / 16 | 43.537556 / 16 | 53.699796 / 16 |
| detail:expert-activation | nested | 0.094997 / 16 | 0.099065 / 16 | 0.129612 / 16 |
| detail:expert-down | nested | 48.760390 / 16 | 49.511435 / 16 | 52.234709 / 16 |
| EUP | nested | 0.605150 / 1 | 0.603797 / 1 | 0.594751 / 1 |
| experts-mix-normalize-up | boundary | 142.259769 / 1 | 144.415467 / 1 | 165.234161 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002715 / 1 | 0.002675 / 1 | 0.003046 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.587707 / 1 | 0.602625 / 1 | 0.603477 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.967838 / 1 | 1.002422 / 1 | 1.013783 / 1 |
| worker:read | parallel worker | 0.211706 / 1 | 0.216620 / 1 | 0.179272 / 1 |
| worker:decode | parallel worker | 1070.072752 / 1 | 1073.332657 / 1 | 1082.581948 / 1 |
| worker:crc | parallel worker | 599.268692 / 1 | 602.457416 / 1 | 608.043281 / 1 |
| worker:math | parallel worker | 363.190749 / 1 | 364.453074 / 1 | 377.266308 / 1 |
| op:Q   int8 projection | nested | 17.511924 / 1 | 17.340482 / 1 | 18.968575 / 1 |
| op:X   mxfp4 expert proj | nested | 141.312750 / 1 | 143.476823 / 1 | 160.634780 / 1 |
| op:N   rmsnorm | nested | 0.044262 / 1 | 0.041476 / 1 | 0.031738 / 1 |
| op:L   l2 per-head | nested | 0.005760 / 1 | 0.005370 / 1 | 0.005220 / 1 |
| op:SiTU + sigma | nested | 0.105094 / 1 | 0.109505 / 1 | 0.139250 / 1 |
| op:C   shortconv | nested | 0.055143 / 1 | 0.051626 / 1 | 0.033413 / 1 |
| op:AR  snapshot aggregate | nested | 0.031178 / 1 | 0.033913 / 1 | 0.032981 / 1 |
| op:D   kda delta-rule | nested | 0.146744 / 1 | 0.110687 / 1 | 0.120716 / 1 |
| op:router dot product | nested | 0.555247 / 1 | 0.569895 / 1 | 0.561238 / 1 |
| op:top-k selection | nested | 0.002385 / 1 | 0.002915 / 1 | 0.002254 / 1 |
| op:alpha / beta / gate | nested | 0.073958 / 1 | 0.073287 / 1 | 0.057808 / 1 |
| detail:read-ahead-wait | nested | 0.001352 / 1 | 0.001482 / 1 | 3.692436 / 1 |
| total:layer | total | 161.915327 / 1 | 163.953344 / 1 | 186.382396 / 1 |

### Layer 85

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.008195 / 1 | 0.007344 / 1 | 0.005720 / 1 |
| pre-attention-aggregation | boundary | 0.018705 / 1 | 0.017924 / 1 | 0.018604 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000290 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012283 / 1 | 0.011542 / 1 | 0.012153 / 1 |
| Q | nested | 3.129013 / 1 | 3.305513 / 1 | 3.500738 / 1 |
| K | nested | 2.327597 / 1 | 2.410201 / 1 | 2.865692 / 1 |
| V | nested | 2.283625 / 1 | 2.323880 / 1 | 2.945791 / 1 |
| B | nested | 0.021570 / 1 | 0.022192 / 1 | 0.027221 / 1 |
| FA | nested | 0.029535 / 1 | 0.027371 / 1 | 0.035937 / 1 |
| FB | nested | 0.051797 / 1 | 0.050194 / 1 | 0.064580 / 1 |
| G | nested | 2.050279 / 1 | 2.081497 / 1 | 2.813474 / 1 |
| O | nested | 2.072851 / 1 | 2.086076 / 1 | 2.064686 / 1 |
| attention | boundary | 13.134027 / 1 | 13.484853 / 1 | 15.478837 / 1 |
| attention-residual | boundary | 0.004208 / 1 | 0.004408 / 1 | 0.004177 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025277 / 1 | 0.025477 / 1 | 0.025588 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001733 / 1 | 0.001923 / 1 | 0.001773 / 1 |
| router-and-top16 | boundary | 0.566308 / 1 | 0.573190 / 1 | 0.571528 / 1 |
| EDOWN | nested | 0.598427 / 1 | 0.595422 / 1 | 0.592105 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.600241 / 1 | 0.597175 / 1 | 0.593789 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026289 / 1 | 0.006201 / 1 | 0.008446 / 1 |
| SH1 | nested | 1.013583 / 1 | 1.015616 / 1 | 1.014515 / 1 |
| SH3 | nested | 1.003624 / 1 | 1.022038 / 1 | 1.001671 / 1 |
| SH2 | nested | 1.004516 / 1 | 1.009045 / 1 | 1.007482 / 1 |
| shared-expert-during-read | boundary | 3.035559 / 1 | 3.060706 / 1 | 3.037533 / 1 |
| detail:expert-gate | nested | 49.639135 / 16 | 49.792880 / 16 | 54.230145 / 16 |
| detail:expert-up | nested | 42.543088 / 16 | 42.638575 / 16 | 53.234707 / 16 |
| detail:expert-activation | nested | 0.096683 / 16 | 0.096942 / 16 | 0.131363 / 16 |
| detail:expert-down | nested | 48.705969 / 16 | 48.883623 / 16 | 52.893871 / 16 |
| EUP | nested | 0.600601 / 1 | 0.596534 / 1 | 0.730494 / 1 |
| experts-mix-normalize-up | boundary | 142.028327 / 1 | 142.427545 / 1 | 161.656480 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002795 / 1 | 0.002806 / 1 | 0.002955 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.586786 / 1 | 0.588118 / 1 | 0.615880 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.950555 / 1 | 0.954753 / 1 | 1.011468 / 1 |
| worker:read | parallel worker | 0.065735 / 1 | 0.067653 / 1 | 0.068428 / 1 |
| worker:decode | parallel worker | 1068.050436 / 1 | 1072.712610 / 1 | 1080.067868 / 1 |
| worker:crc | parallel worker | 599.533481 / 1 | 602.184071 / 1 | 607.121448 / 1 |
| worker:math | parallel worker | 362.503257 / 1 | 363.248754 / 1 | 373.735438 / 1 |
| op:Q   int8 projection | nested | 16.185625 / 1 | 16.543986 / 1 | 18.662721 / 1 |
| op:X   mxfp4 expert proj | nested | 141.079265 / 1 | 141.506025 / 1 | 160.619972 / 1 |
| op:N   rmsnorm | nested | 0.040234 / 1 | 0.046096 / 1 | 0.031480 / 1 |
| op:L   l2 per-head | nested | 0.005601 / 1 | 0.005680 / 1 | 0.005570 / 1 |
| op:SiTU + sigma | nested | 0.103956 / 1 | 0.104606 / 1 | 0.138067 / 1 |
| op:C   shortconv | nested | 0.054331 / 1 | 0.053410 / 1 | 0.031018 / 1 |
| op:AR  snapshot aggregate | nested | 0.034133 / 1 | 0.033713 / 1 | 0.033993 / 1 |
| op:D   kda delta-rule | nested | 0.243845 / 1 | 0.241652 / 1 | 0.242554 / 1 |
| op:router dot product | nested | 0.563722 / 1 | 0.570575 / 1 | 0.569072 / 1 |
| op:top-k selection | nested | 0.002214 / 1 | 0.002385 / 1 | 0.002154 / 1 |
| op:alpha / beta / gate | nested | 0.073477 / 1 | 0.073457 / 1 | 0.058149 / 1 |
| detail:read-ahead-wait | nested | 0.001222 / 1 | 0.001465 / 1 | 0.000992 / 1 |
| total:layer | total | 160.439390 / 1 | 161.199502 / 1 | 182.454020 / 1 |

### Layer 86

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.007664 / 1 | 0.008345 / 1 | 0.008165 / 1 |
| pre-attention-aggregation | boundary | 0.017393 / 1 | 0.017773 / 1 | 0.018234 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000090 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011993 / 1 | 0.011752 / 1 | 0.011983 / 1 |
| Q | nested | 3.252525 / 1 | 3.269476 / 1 | 3.316203 / 1 |
| K | nested | 2.355589 / 1 | 2.408278 / 1 | 2.730961 / 1 |
| V | nested | 2.296890 / 1 | 2.335051 / 1 | 2.683973 / 1 |
| B | nested | 0.025217 / 1 | 0.022963 / 1 | 0.023744 / 1 |
| FA | nested | 0.032030 / 1 | 0.032050 / 1 | 0.034074 / 1 |
| FB | nested | 0.049172 / 1 | 0.049563 / 1 | 0.050384 / 1 |
| G | nested | 2.049538 / 1 | 2.074013 / 1 | 2.399141 / 1 |
| O | nested | 2.057613 / 1 | 2.082139 / 1 | 2.446799 / 1 |
| attention | boundary | 13.303685 / 1 | 13.459496 / 1 | 14.926496 / 1 |
| attention-residual | boundary | 0.004568 / 1 | 0.004909 / 1 | 0.004318 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024215 / 1 | 0.025167 / 1 | 0.024666 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001754 / 1 | 0.002214 / 1 | 0.002264 / 1 |
| router-and-top16 | boundary | 0.572550 / 1 | 0.576297 / 1 | 0.592868 / 1 |
| EDOWN | nested | 0.588449 / 1 | 0.595292 / 1 | 0.693966 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590102 / 1 | 0.596985 / 1 | 0.695679 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.016621 / 1 | 0.012733 / 1 | 0.013315 / 1 |
| SH1 | nested | 1.001901 / 1 | 1.019163 / 1 | 1.202546 / 1 |
| SH3 | nested | 1.002022 / 1 | 1.011459 / 1 | 1.182789 / 1 |
| SH2 | nested | 0.997173 / 1 | 1.010127 / 1 | 1.180184 / 1 |
| shared-expert-during-read | boundary | 3.015031 / 1 | 3.054384 / 1 | 3.581899 / 1 |
| detail:expert-gate | nested | 49.193396 / 16 | 49.776338 / 16 | 53.860597 / 16 |
| detail:expert-up | nested | 42.611479 / 16 | 42.722236 / 16 | 52.833085 / 16 |
| detail:expert-activation | nested | 0.098153 / 16 | 0.098413 / 16 | 0.132328 / 16 |
| detail:expert-down | nested | 48.771809 / 16 | 48.900331 / 16 | 53.644892 / 16 |
| EUP | nested | 0.602475 / 1 | 0.605370 / 1 | 0.596535 / 1 |
| experts-mix-normalize-up | boundary | 141.719591 / 1 | 142.515910 / 1 | 161.513884 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003356 / 1 | 0.002925 / 1 | 0.002895 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.589792 / 1 | 0.585373 / 1 | 0.599700 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.952739 / 1 | 0.952118 / 1 | 0.981934 / 1 |
| worker:read | parallel worker | 0.067484 / 1 | 0.070333 / 1 | 0.068465 / 1 |
| worker:decode | parallel worker | 1069.751081 / 1 | 1075.659892 / 1 | 1083.485992 / 1 |
| worker:crc | parallel worker | 598.989185 / 1 | 602.616273 / 1 | 607.349270 / 1 |
| worker:math | parallel worker | 361.820146 / 1 | 362.580976 / 1 | 372.652600 / 1 |
| op:Q   int8 projection | nested | 16.309178 / 1 | 16.513471 / 1 | 18.539874 / 1 |
| op:X   mxfp4 expert proj | nested | 140.771129 / 1 | 141.591264 / 1 | 160.601547 / 1 |
| op:N   rmsnorm | nested | 0.045846 / 1 | 0.044505 / 1 | 0.031618 / 1 |
| op:L   l2 per-head | nested | 0.005380 / 1 | 0.005610 / 1 | 0.005631 / 1 |
| op:SiTU + sigma | nested | 0.105535 / 1 | 0.105725 / 1 | 0.141471 / 1 |
| op:C   shortconv | nested | 0.063920 / 1 | 0.063158 / 1 | 0.033763 / 1 |
| op:AR  snapshot aggregate | nested | 0.032351 / 1 | 0.033401 / 1 | 0.033292 / 1 |
| op:D   kda delta-rule | nested | 0.243565 / 1 | 0.243956 / 1 | 0.253152 / 1 |
| op:router dot product | nested | 0.569734 / 1 | 0.573380 / 1 | 0.590092 / 1 |
| op:top-k selection | nested | 0.002375 / 1 | 0.002384 / 1 | 0.002194 / 1 |
| op:alpha / beta / gate | nested | 0.074038 / 1 | 0.071965 / 1 | 0.059100 / 1 |
| detail:read-ahead-wait | nested | 0.001403 / 1 | 0.001635 / 1 | 0.000962 / 1 |
| total:layer | total | 160.268050 / 1 | 161.264523 / 1 | 182.402845 / 1 |

### Layer 87

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.008626 / 1 | 0.008125 / 1 | 0.008255 / 1 |
| pre-attention-aggregation | boundary | 0.017503 / 1 | 0.017192 / 1 | 0.017773 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011792 / 1 | 0.012984 / 1 | 0.011722 / 1 |
| QA | nested | 0.363168 / 1 | 0.349222 / 1 | 0.361725 / 1 |
| QB | nested | 0.909930 / 1 | 0.910480 / 1 | 0.951658 / 1 |
| KA | nested | 0.142306 / 1 | 0.138890 / 1 | 0.138158 / 1 |
| KB | nested | 0.419804 / 1 | 0.404526 / 1 | 0.403554 / 1 |
| G | nested | 2.736300 / 1 | 2.746198 / 1 | 2.734317 / 1 |
| O | nested | 2.027286 / 1 | 2.062272 / 1 | 2.023669 / 1 |
| attention | boundary | 6.717124 / 1 | 6.727604 / 1 | 6.733816 / 1 |
| attention-residual | boundary | 0.004538 / 1 | 0.004439 / 1 | 0.004258 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025828 / 1 | 0.025869 / 1 | 0.025758 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001673 / 1 | 0.001683 / 1 | 0.001603 / 1 |
| router-and-top16 | boundary | 0.523127 / 1 | 0.526312 / 1 | 0.523578 / 1 |
| EDOWN | nested | 0.601894 / 1 | 0.607705 / 1 | 0.596524 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.603587 / 1 | 0.609408 / 1 | 0.598228 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.035867 / 1 | 0.011962 / 1 | 0.012654 / 1 |
| SH1 | nested | 1.028491 / 1 | 1.049731 / 1 | 1.014144 / 1 |
| SH3 | nested | 1.024162 / 1 | 1.046825 / 1 | 1.014855 / 1 |
| SH2 | nested | 1.025205 / 1 | 1.046774 / 1 | 1.015055 / 1 |
| shared-expert-during-read | boundary | 3.091393 / 1 | 3.157227 / 1 | 3.057760 / 1 |
| detail:expert-gate | nested | 49.255937 / 16 | 49.790771 / 16 | 54.322510 / 16 |
| detail:expert-up | nested | 42.978473 / 16 | 43.222367 / 16 | 53.772473 / 16 |
| detail:expert-activation | nested | 0.097121 / 16 | 0.099116 / 16 | 0.137377 / 16 |
| detail:expert-down | nested | 48.963160 / 16 | 49.073046 / 16 | 54.184583 / 16 |
| EUP | nested | 0.620218 / 1 | 0.596874 / 1 | 0.606031 / 1 |
| experts-mix-normalize-up | boundary | 142.373382 / 1 | 143.204286 / 1 | 163.465719 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003036 / 1 | 0.003537 / 1 | 0.002745 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.598197 / 1 | 0.591344 / 1 | 0.628654 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.604690 / 1 | 0.597957 / 1 | 0.635908 / 1 |
| worker:read | parallel worker | 0.224142 / 1 | 0.218869 / 1 | 0.191361 / 1 |
| worker:decode | parallel worker | 1072.664426 / 1 | 1078.118992 / 1 | 1089.997083 / 1 |
| worker:crc | parallel worker | 602.324600 / 1 | 605.299939 / 1 | 612.709888 / 1 |
| worker:math | parallel worker | 362.174239 / 1 | 363.299875 / 1 | 378.169582 / 1 |
| op:Q   int8 projection | nested | 10.897542 / 1 | 10.958205 / 1 | 10.858265 / 1 |
| op:X   mxfp4 expert proj | nested | 141.392420 / 1 | 142.279788 / 1 | 162.554817 / 1 |
| op:N   rmsnorm | nested | 0.031469 / 1 | 0.032379 / 1 | 0.031768 / 1 |
| op:SiTU + sigma | nested | 0.104212 / 1 | 0.106550 / 1 | 0.143959 / 1 |
| op:AR  snapshot aggregate | nested | 0.032861 / 1 | 0.032832 / 1 | 0.033313 / 1 |
| op:SA  softmax attention | nested | 0.006142 / 1 | 0.009758 / 1 | 0.012373 / 1 |
| op:router dot product | nested | 0.520503 / 1 | 0.523337 / 1 | 0.520893 / 1 |
| op:top-k selection | nested | 0.002334 / 1 | 0.002635 / 1 | 0.002325 / 1 |
| detail:read-ahead-wait | nested | 0.001321 / 1 | 0.001522 / 1 | 0.000861 / 1 |
| total:layer | total | 154.046151 / 1 | 154.931757 / 1 | 175.122959 / 1 |

### Layer 88

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.008596 / 1 | 0.010700 / 1 | 0.009047 / 1 |
| pre-attention-aggregation | boundary | 0.016691 / 1 | 0.020298 / 1 | 0.018124 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| pre-attention-normalization | boundary | 0.012133 / 1 | 0.012082 / 1 | 0.012564 / 1 |
| Q | nested | 3.203963 / 1 | 3.305333 / 1 | 3.130937 / 1 |
| K | nested | 2.305816 / 1 | 2.360598 / 1 | 2.537149 / 1 |
| V | nested | 2.286761 / 1 | 2.312860 / 1 | 2.407697 / 1 |
| B | nested | 0.029886 / 1 | 0.029845 / 1 | 0.027271 / 1 |
| FA | nested | 0.027151 / 1 | 0.027782 / 1 | 0.027913 / 1 |
| FB | nested | 0.050094 / 1 | 0.051917 / 1 | 0.053270 / 1 |
| G | nested | 2.047474 / 1 | 2.139926 / 1 | 2.163811 / 1 |
| O | nested | 2.066659 / 1 | 2.084923 / 1 | 2.123646 / 1 |
| attention | boundary | 13.118268 / 1 | 13.377282 / 1 | 13.603195 / 1 |
| attention-residual | boundary | 0.004168 / 1 | 0.004748 / 1 | 0.002404 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024927 / 1 | 0.026009 / 1 | 0.029135 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002064 / 1 | 0.001623 / 1 | 0.002534 / 1 |
| router-and-top16 | boundary | 0.562069 / 1 | 0.574242 / 1 | 0.617793 / 1 |
| EDOWN | nested | 0.589010 / 1 | 0.592185 / 1 | 0.612303 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.590783 / 1 | 0.594040 / 1 | 0.615189 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026168 / 1 | 0.011562 / 1 | 0.006783 / 1 |
| SH1 | nested | 1.010467 / 1 | 1.013152 / 1 | 1.020826 / 1 |
| SH3 | nested | 1.034542 / 1 | 1.013663 / 1 | 1.007702 / 1 |
| SH2 | nested | 1.001972 / 1 | 1.008864 / 1 | 0.992343 / 1 |
| shared-expert-during-read | boundary | 3.063812 / 1 | 3.049706 / 1 | 3.035168 / 1 |
| detail:expert-gate | nested | 49.636316 / 16 | 49.837221 / 16 | 53.857839 / 16 |
| detail:expert-up | nested | 42.645832 / 16 | 42.777887 / 16 | 53.555377 / 16 |
| detail:expert-activation | nested | 0.100695 / 16 | 0.100536 / 16 | 0.141985 / 16 |
| detail:expert-down | nested | 48.874446 / 16 | 49.017412 / 16 | 54.582495 / 16 |
| EUP | nested | 0.595752 / 1 | 0.604940 / 1 | 0.586917 / 1 |
| experts-mix-normalize-up | boundary | 142.286179 / 1 | 142.768502 / 1 | 163.181107 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003066 / 1 | 0.002424 / 1 | 0.002334 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.591084 / 1 | 0.587106 / 1 | 0.597095 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.957638 / 1 | 0.970372 / 1 | 1.060150 / 1 |
| worker:read | parallel worker | 0.068364 / 1 | 0.068971 / 1 | 0.072881 / 1 |
| worker:decode | parallel worker | 1074.070923 / 1 | 1079.927031 / 1 | 1090.698811 / 1 |
| worker:crc | parallel worker | 598.571908 / 1 | 602.938578 / 1 | 608.688604 / 1 |
| worker:math | parallel worker | 361.997582 / 1 | 363.122564 / 1 | 377.574392 / 1 |
| op:Q   int8 projection | nested | 16.248095 / 1 | 16.544596 / 1 | 16.690341 / 1 |
| op:X   mxfp4 expert proj | nested | 141.357097 / 1 | 141.829970 / 1 | 162.271567 / 1 |
| op:N   rmsnorm | nested | 0.043642 / 1 | 0.040848 / 1 | 0.033784 / 1 |
| op:L   l2 per-head | nested | 0.005350 / 1 | 0.006192 / 1 | 0.005671 / 1 |
| op:SiTU + sigma | nested | 0.108571 / 1 | 0.107579 / 1 | 0.148857 / 1 |
| op:C   shortconv | nested | 0.055975 / 1 | 0.063028 / 1 | 0.040877 / 1 |
| op:AR  snapshot aggregate | nested | 0.032381 / 1 | 0.036128 / 1 | 0.036778 / 1 |
| op:D   kda delta-rule | nested | 0.146744 / 1 | 0.108703 / 1 | 0.137236 / 1 |
| op:router dot product | nested | 0.559375 / 1 | 0.571528 / 1 | 0.614377 / 1 |
| op:top-k selection | nested | 0.002274 / 1 | 0.002314 / 1 | 0.002595 / 1 |
| op:alpha / beta / gate | nested | 0.073587 / 1 | 0.073717 / 1 | 0.057457 / 1 |
| detail:read-ahead-wait | nested | 0.001404 / 1 | 0.001173 / 1 | 0.000973 / 1 |
| total:layer | total | 160.700899 / 1 | 161.448998 / 1 | 182.222377 / 1 |

### Layer 89

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.009648 / 1 | 0.009067 / 1 | 0.006101 / 1 |
| pre-attention-aggregation | boundary | 0.018405 / 1 | 0.018645 / 1 | 0.019858 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.012283 / 1 | 0.012073 / 1 | 0.011652 / 1 |
| Q | nested | 3.140505 / 1 | 3.117903 / 1 | 3.517779 / 1 |
| K | nested | 2.342134 / 1 | 2.371509 / 1 | 3.221095 / 1 |
| V | nested | 2.304734 / 1 | 2.326464 / 1 | 2.965308 / 1 |
| B | nested | 0.026509 / 1 | 0.026529 / 1 | 0.030066 / 1 |
| FA | nested | 0.034895 / 1 | 0.030697 / 1 | 0.040656 / 1 |
| FB | nested | 0.049582 / 1 | 0.050193 / 1 | 0.078196 / 1 |
| G | nested | 2.052613 / 1 | 2.055649 / 1 | 2.745988 / 1 |
| O | nested | 2.052102 / 1 | 2.091256 / 1 | 2.142651 / 1 |
| attention | boundary | 13.157100 / 1 | 13.240347 / 1 | 15.775682 / 1 |
| attention-residual | boundary | 0.004358 / 1 | 0.004058 / 1 | 0.003156 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.025778 / 1 | 0.025558 / 1 | 0.028453 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002074 / 1 | 0.001783 / 1 | 0.003406 / 1 |
| router-and-top16 | boundary | 0.572789 / 1 | 0.577389 / 1 | 0.598698 / 1 |
| EDOWN | nested | 0.587097 / 1 | 0.589441 / 1 | 0.616150 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.588860 / 1 | 0.591114 / 1 | 0.618025 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.043441 / 1 | 0.017973 / 1 | 0.007634 / 1 |
| SH1 | nested | 1.397460 / 1 | 1.003934 / 1 | 1.044992 / 1 |
| SH3 | nested | 1.416576 / 1 | 1.007713 / 1 | 1.028150 / 1 |
| SH2 | nested | 1.390747 / 1 | 1.006860 / 1 | 1.007432 / 1 |
| shared-expert-during-read | boundary | 4.224840 / 1 | 3.033125 / 1 | 3.095110 / 1 |
| detail:expert-gate | nested | 49.144405 / 16 | 50.047430 / 16 | 53.675461 / 16 |
| detail:expert-up | nested | 42.544996 / 16 | 43.649014 / 16 | 53.082743 / 16 |
| detail:expert-activation | nested | 0.101392 / 16 | 0.108904 / 16 | 0.144600 / 16 |
| detail:expert-down | nested | 48.835081 / 16 | 49.115958 / 16 | 54.294998 / 16 |
| EUP | nested | 0.599250 / 1 | 0.591916 / 1 | 0.595712 / 1 |
| experts-mix-normalize-up | boundary | 141.685047 / 1 | 143.953304 / 1 | 162.228889 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003196 / 1 | 0.002945 / 1 | 0.002815 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.592807 / 1 | 0.590783 / 1 | 0.601423 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.932041 / 1 | 0.957187 / 1 | 1.036125 / 1 |
| worker:read | parallel worker | 0.069034 / 1 | 0.069533 / 1 | 0.069262 / 1 |
| worker:decode | parallel worker | 1072.815947 / 1 | 1082.354790 / 1 | 1089.587948 / 1 |
| worker:crc | parallel worker | 599.176703 / 1 | 601.615664 / 1 | 606.231670 / 1 |
| worker:math | parallel worker | 360.964523 / 1 | 363.170746 / 1 | 371.559474 / 1 |
| op:Q   int8 projection | nested | 17.392669 / 1 | 16.268600 / 1 | 19.032613 / 1 |
| op:X   mxfp4 expert proj | nested | 140.724402 / 1 | 143.022906 / 1 | 161.331543 / 1 |
| op:N   rmsnorm | nested | 0.039522 / 1 | 0.051235 / 1 | 0.034577 / 1 |
| op:L   l2 per-head | nested | 0.005610 / 1 | 0.005460 / 1 | 0.005380 / 1 |
| op:SiTU + sigma | nested | 0.113192 / 1 | 0.116288 / 1 | 0.151634 / 1 |
| op:C   shortconv | nested | 0.052778 / 1 | 0.061885 / 1 | 0.040937 / 1 |
| op:AR  snapshot aggregate | nested | 0.034334 / 1 | 0.034284 / 1 | 0.038140 / 1 |
| op:D   kda delta-rule | nested | 0.242503 / 1 | 0.240028 / 1 | 0.156493 / 1 |
| op:router dot product | nested | 0.569844 / 1 | 0.574262 / 1 | 0.595792 / 1 |
| op:top-k selection | nested | 0.002615 / 1 | 0.002695 / 1 | 0.002244 / 1 |
| op:alpha / beta / gate | nested | 0.074008 / 1 | 0.074860 / 1 | 0.057457 / 1 |
| detail:read-ahead-wait | nested | 0.001503 / 1 | 0.001262 / 1 | 0.001353 / 1 |
| total:layer | total | 161.306890 / 1 | 162.469664 / 1 | 183.462363 / 1 |

### Layer 90

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.007194 / 1 | 0.007173 / 1 | 0.007374 / 1 |
| pre-attention-aggregation | boundary | 0.017393 / 1 | 0.024496 / 1 | 0.020799 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000090 / 1 | 0.000101 / 1 |
| pre-attention-normalization | boundary | 0.011672 / 1 | 0.011422 / 1 | 0.011392 / 1 |
| Q | nested | 2.944789 / 1 | 3.307056 / 1 | 3.487873 / 1 |
| K | nested | 2.480743 / 1 | 2.373102 / 1 | 3.142469 / 1 |
| V | nested | 2.297681 / 1 | 2.289115 / 1 | 3.047551 / 1 |
| B | nested | 0.022592 / 1 | 0.021490 / 1 | 0.028714 / 1 |
| FA | nested | 0.032551 / 1 | 0.031409 / 1 | 0.038332 / 1 |
| FB | nested | 0.048150 / 1 | 0.048430 / 1 | 0.060643 / 1 |
| G | nested | 2.069635 / 1 | 2.084482 / 1 | 2.951832 / 1 |
| O | nested | 2.063855 / 1 | 2.082028 / 1 | 2.151828 / 1 |
| attention | boundary | 13.084856 / 1 | 13.418159 / 1 | 16.000001 / 1 |
| attention-residual | boundary | 0.004278 / 1 | 0.004398 / 1 | 0.004048 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024777 / 1 | 0.025839 / 1 | 0.028794 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001553 / 1 | 0.001693 / 1 | 0.002875 / 1 |
| router-and-top16 | boundary | 0.569143 / 1 | 0.581296 / 1 | 0.597255 / 1 |
| EDOWN | nested | 0.587938 / 1 | 0.588879 / 1 | 0.617914 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.589732 / 1 | 0.590653 / 1 | 0.619607 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.031148 / 1 | 0.012413 / 1 | 0.019666 / 1 |
| SH1 | nested | 1.005037 / 1 | 1.009725 / 1 | 1.018542 / 1 |
| SH3 | nested | 1.011749 / 1 | 1.021969 / 1 | 1.012521 / 1 |
| SH2 | nested | 1.004556 / 1 | 1.013512 / 1 | 0.989047 / 1 |
| shared-expert-during-read | boundary | 3.035499 / 1 | 3.059924 / 1 | 3.034798 / 1 |
| detail:expert-gate | nested | 51.755552 / 16 | 50.108037 / 16 | 54.874671 / 16 |
| detail:expert-up | nested | 45.461841 / 16 | 42.888957 / 16 | 54.465855 / 16 |
| detail:expert-activation | nested | 0.104013 / 16 | 0.100710 / 16 | 0.140477 / 16 |
| detail:expert-down | nested | 52.680962 / 16 | 49.134309 / 16 | 53.443007 / 16 |
| EUP | nested | 0.590202 / 1 | 0.612755 / 1 | 0.723912 / 1 |
| experts-mix-normalize-up | boundary | 151.056317 / 1 | 143.266451 / 1 | 164.123407 / 1 |
| mlp-merge-and-cleanup | boundary | 0.002495 / 1 | 0.003366 / 1 | 0.003376 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.595091 / 1 | 0.589261 / 1 | 0.609668 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.981383 / 1 | 0.961255 / 1 | 1.044160 / 1 |
| worker:read | parallel worker | 0.070317 / 1 | 0.068330 / 1 | 0.069419 / 1 |
| worker:decode | parallel worker | 1082.212100 / 1 | 1081.716927 / 1 | 1095.792130 / 1 |
| worker:crc | parallel worker | 602.198218 / 1 | 602.172038 / 1 | 609.954607 / 1 |
| worker:math | parallel worker | 370.780275 / 1 | 366.476514 / 1 | 383.830893 / 1 |
| op:Q   int8 projection | nested | 16.158123 / 1 | 16.482548 / 1 | 19.269586 / 1 |
| op:X   mxfp4 expert proj | nested | 150.113195 / 1 | 142.330794 / 1 | 163.064420 / 1 |
| op:N   rmsnorm | nested | 0.045353 / 1 | 0.045735 / 1 | 0.032230 / 1 |
| op:L   l2 per-head | nested | 0.005391 / 1 | 0.006152 / 1 | 0.006062 / 1 |
| op:SiTU + sigma | nested | 0.111447 / 1 | 0.108065 / 1 | 0.147704 / 1 |
| op:C   shortconv | nested | 0.057869 / 1 | 0.053701 / 1 | 0.039683 / 1 |
| op:AR  snapshot aggregate | nested | 0.032482 / 1 | 0.040245 / 1 | 0.039364 / 1 |
| op:D   kda delta-rule | nested | 0.241711 / 1 | 0.237814 / 1 | 0.158756 / 1 |
| op:router dot product | nested | 0.566528 / 1 | 0.578591 / 1 | 0.594791 / 1 |
| op:top-k selection | nested | 0.002204 / 1 | 0.002354 / 1 | 0.002004 / 1 |
| op:alpha / beta / gate | nested | 0.074419 / 1 | 0.082754 / 1 | 0.058559 / 1 |
| detail:read-ahead-wait | nested | 0.001311 / 1 | 0.001461 / 1 | 0.001183 / 1 |
| total:layer | total | 169.443145 / 1 | 161.993384 / 1 | 185.543660 / 1 |

### Layer 91

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.007685 / 1 | 0.008456 / 1 | 0.007273 / 1 |
| pre-attention-aggregation | boundary | 0.017964 / 1 | 0.018244 / 1 | 0.019055 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000100 / 1 | 0.000021 / 1 |
| pre-attention-normalization | boundary | 0.011612 / 1 | 0.012142 / 1 | 0.011852 / 1 |
| QA | nested | 0.363409 / 1 | 0.344994 / 1 | 0.324065 / 1 |
| QB | nested | 0.954733 / 1 | 0.954944 / 1 | 0.781650 / 1 |
| KA | nested | 0.134261 / 1 | 0.133249 / 1 | 0.147085 / 1 |
| KB | nested | 0.427168 / 1 | 0.392914 / 1 | 0.414775 / 1 |
| G | nested | 2.821770 / 1 | 2.619892 / 1 | 3.037813 / 1 |
| O | nested | 2.088140 / 1 | 2.082479 / 1 | 2.944079 / 1 |
| attention | boundary | 6.909293 / 1 | 6.649999 / 1 | 7.791070 / 1 |
| attention-residual | boundary | 0.004218 / 1 | 0.004278 / 1 | 0.005770 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.026099 / 1 | 0.026089 / 1 | 0.030136 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.001703 / 1 | 0.002304 / 1 | 0.002375 / 1 |
| router-and-top16 | boundary | 0.538596 / 1 | 0.524319 / 1 | 0.627892 / 1 |
| EDOWN | nested | 0.614778 / 1 | 0.602926 / 1 | 0.884532 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.616541 / 1 | 0.604599 / 1 | 0.887327 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.026860 / 1 | 0.006813 / 1 | 0.007785 / 1 |
| SH1 | nested | 1.039190 / 1 | 1.038910 / 1 | 1.542561 / 1 |
| SH3 | nested | 1.035955 / 1 | 1.038379 / 1 | 1.512885 / 1 |
| SH2 | nested | 1.042206 / 1 | 1.033841 / 1 | 1.383634 / 1 |
| shared-expert-during-read | boundary | 3.131398 / 1 | 3.125567 / 1 | 4.459788 / 1 |
| detail:expert-gate | nested | 50.109317 / 16 | 49.709616 / 16 | 54.720212 / 16 |
| detail:expert-up | nested | 42.826398 / 16 | 42.816489 / 16 | 54.557038 / 16 |
| detail:expert-activation | nested | 0.099077 / 16 | 0.104414 / 16 | 0.139401 / 16 |
| detail:expert-down | nested | 48.917962 / 16 | 49.142806 / 16 | 53.692503 / 16 |
| EUP | nested | 0.603317 / 1 | 0.606272 / 1 | 0.740012 / 1 |
| experts-mix-normalize-up | boundary | 143.005622 / 1 | 142.799670 / 1 | 164.276944 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003436 / 1 | 0.002855 / 1 | 0.003106 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.595341 / 1 | 0.594861 / 1 | 0.611793 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.602205 / 1 | 0.601513 / 1 | 0.619837 / 1 |
| worker:read | parallel worker | 0.067531 / 1 | 0.066329 / 1 | 0.067224 / 1 |
| worker:decode | parallel worker | 1078.186559 / 1 | 1083.003975 / 1 | 1101.986232 / 1 |
| worker:crc | parallel worker | 599.051905 / 1 | 601.181948 / 1 | 612.124435 / 1 |
| worker:math | parallel worker | 362.532299 / 1 | 362.942889 / 1 | 384.094945 / 1 |
| op:Q   int8 projection | nested | 11.123715 / 1 | 10.847596 / 1 | 13.711677 / 1 |
| op:X   mxfp4 expert proj | nested | 142.054630 / 1 | 141.874733 / 1 | 163.243152 / 1 |
| op:N   rmsnorm | nested | 0.032100 / 1 | 0.031699 / 1 | 0.032421 / 1 |
| op:SiTU + sigma | nested | 0.106388 / 1 | 0.112449 / 1 | 0.150501 / 1 |
| op:AR  snapshot aggregate | nested | 0.033462 / 1 | 0.033694 / 1 | 0.036788 / 1 |
| op:SA  softmax attention | nested | 0.006472 / 1 | 0.008996 / 1 | 0.013916 / 1 |
| op:router dot product | nested | 0.536011 / 1 | 0.521674 / 1 | 0.624857 / 1 |
| op:top-k selection | nested | 0.002274 / 1 | 0.002315 / 1 | 0.002705 / 1 |
| detail:read-ahead-wait | nested | 0.001723 / 1 | 0.001801 / 1 | 0.002015 / 1 |
| total:layer | total | 154.927066 / 1 | 154.412477 / 1 | 178.782273 / 1 |

### Layer 92

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| binding-and-stored-folds | boundary | 0.008205 / 1 | 0.008086 / 1 | 0.007083 / 1 |
| pre-attention-aggregation | boundary | 0.016871 / 1 | 0.016832 / 1 | 0.017633 / 1 |
| snapshot-push | boundary | 0.000020 / 1 | 0.000020 / 1 | 0.000020 / 1 |
| pre-attention-normalization | boundary | 0.011331 / 1 | 0.011301 / 1 | 0.011521 / 1 |
| QA | nested | 0.355724 / 1 | 0.350073 / 1 | 0.334174 / 1 |
| QB | nested | 0.992283 / 1 | 0.945466 / 1 | 1.050391 / 1 |
| KA | nested | 0.132327 / 1 | 0.137397 / 1 | 0.141504 / 1 |
| KB | nested | 0.405457 / 1 | 0.348050 / 1 | 0.409385 / 1 |
| G | nested | 2.772297 / 1 | 2.824215 / 1 | 2.946072 / 1 |
| O | nested | 2.032275 / 1 | 2.018510 / 1 | 2.978944 / 1 |
| attention | boundary | 6.830225 / 1 | 6.777507 / 1 | 8.037911 / 1 |
| attention-residual | boundary | 0.004148 / 1 | 0.004469 / 1 | 0.005180 / 1 |
| pre-mlp-aggregation-and-normalization | boundary | 0.024967 / 1 | 0.025347 / 1 | 0.028763 / 1 |
| moe-scratch-and-parameter-bind | boundary | 0.002194 / 1 | 0.001673 / 1 | 0.001874 / 1 |
| router-and-top16 | boundary | 0.500665 / 1 | 0.501827 / 1 | 0.608657 / 1 |
| EDOWN | nested | 0.604750 / 1 | 0.606172 / 1 | 0.883880 / 1 |
| prefetch-submit-and-latent-projection | boundary | 0.606443 / 1 | 0.607906 / 1 | 0.886695 / 1 |
| result-lookup-and-first-expert-submit | boundary | 0.021631 / 1 | 0.005891 / 1 | 0.009087 / 1 |
| SH1 | nested | 1.031306 / 1 | 1.034542 / 1 | 1.516382 / 1 |
| SH3 | nested | 1.030605 / 1 | 1.037758 / 1 | 1.172430 / 1 |
| SH2 | nested | 1.037838 / 1 | 1.028852 / 1 | 1.168442 / 1 |
| shared-expert-during-read | boundary | 3.113825 / 1 | 3.115528 / 1 | 3.875467 / 1 |
| detail:expert-gate | nested | 49.822555 / 16 | 49.974717 / 16 | 54.475517 / 16 |
| detail:expert-up | nested | 42.712206 / 16 | 42.813844 / 16 | 53.698204 / 16 |
| detail:expert-activation | nested | 0.101072 / 16 | 0.106049 / 16 | 0.144097 / 16 |
| detail:expert-down | nested | 48.852303 / 16 | 48.840865 / 16 | 53.241999 / 16 |
| EUP | nested | 0.586876 / 1 | 0.604669 / 1 | 0.641508 / 1 |
| experts-mix-normalize-up | boundary | 142.512211 / 1 | 142.758122 / 1 | 179.174635 / 1 |
| mlp-merge-and-cleanup | boundary | 0.003106 / 1 | 0.002905 / 1 | 0.002815 / 1 |
| detail:cross-layer-predict-and-submit | nested | 0.000020 / 1 | 0.000020 / 1 | 0.000030 / 1 |
| mlp-residual-cache-save-and-unbind | boundary | 0.005620 / 1 | 0.005630 / 1 | 0.006752 / 1 |
| worker:read | parallel worker | 0.067250 / 1 | 0.067476 / 1 | 0.070389 / 1 |
| worker:decode | parallel worker | 1076.428825 / 1 | 1078.455673 / 1 | 1093.158455 / 1 |
| worker:crc | parallel worker | 600.207157 / 1 | 602.725939 / 1 | 610.653379 / 1 |
| worker:math | parallel worker | 362.446999 / 1 | 362.963799 / 1 | 379.692823 / 1 |
| op:Q   int8 projection | nested | 10.980616 / 1 | 10.934609 / 1 | 13.241308 / 1 |
| op:X   mxfp4 expert proj | nested | 141.589268 / 1 | 141.835732 / 1 | 161.701915 / 1 |
| op:N   rmsnorm | nested | 0.024024 / 1 | 0.023404 / 1 | 0.026651 / 1 |
| op:SiTU + sigma | nested | 0.108562 / 1 | 0.113664 / 1 | 0.153785 / 1 |
| op:AR  snapshot aggregate | nested | 0.031919 / 1 | 0.032019 / 1 | 0.034414 / 1 |
| op:SA  softmax attention | nested | 0.008055 / 1 | 0.008937 / 1 | 0.013135 / 1 |
| op:router dot product | nested | 0.497719 / 1 | 0.499412 / 1 | 0.605451 / 1 |
| op:top-k selection | nested | 0.002525 / 1 | 0.002084 / 1 | 0.002634 / 1 |
| detail:read-ahead-wait | nested | 0.001313 / 1 | 0.001371 / 1 | 14.760786 / 1 |
| total:layer | total | 153.686439 / 1 | 153.867038 / 1 | 192.705585 / 1 |

### Output

| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |
|---|---|---:|---:|---:|
| final-aggregation-and-normalization | boundary | not executed | 0.045024 / 1 | 0.054171 / 1 |
| head-fill-and-projection | boundary | not executed | 15295.461546 / 1 | 51.457970 / 1 |
| argmax-and-cleanup | boundary | not executed | 0.234799 / 1 | 0.100577 / 1 |

