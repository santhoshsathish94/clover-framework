# v3 — repetitive

**Probes:** what routing does when the input token is the same at every position. Twelve
copies of one token, then four generated.

```
prompt   "the the the the the the the the the the the the"  (47 bytes, sha1 f91356683290)
tokens   12 ids, all 276 = ' the'
gen      4
trace    /root/k3flow/v3.jsonl        69.4 s wall, I/O 30.4%
```

## Outcome

Every generated token is **276 `' the'`**. The model continues the repetition exactly.

| step | token | seconds | expert GB read |
|---:|---:|---:|---:|
| 0 (prefill, T=12) | 276 | 44.25 | 114.78 |
| 1 | 276 | 8.22 | 24.53 |
| 2 | 276 | 7.79 | 11.55 |
| 3 | 276 | 9.16 | 11.28 |

## Repetition concentrates routing — the clearest result so far

Against v1 and v2, held at identical settings:

| | v1 / v2 (factual) | v3 (repetitive) |
|---|---|---|
| in-layer diversity, prefill | 77.2% (5 positions) | **37.0%** (12 positions) |
| prefill slots → distinct pairs | 7360 → 5683 | 17664 → **6541** |
| mean adjacent-step reuse | 42.4% | **65.5%** |
| pairs used by *every* step | 0.5% | **9.3%** |
| residual per position, prefill | 36.92 | 29.93 |

Twelve identical tokens fill 17664 expert slots but need only 6541 distinct
(layer, expert) pairs. The twelve positions are not routed independently — they agree
with each other about two thirds of the time. With five *different* tokens in v1 the
same measure was 77.2% distinct.

This is the mechanism behind the expert cache's behavior made visible: **repetitive
input is cheap because it routes narrowly**, not because anything special is done for
it. Reads settle at 11.3 GB per step against 12.3 GB for v2, and the all-step
intersection rises eighteenfold.

## What did not change

Snapshot layers 0, 12, 24, 36, 48, 60, 72, 84; residual replaced not added at each;
layer chain bit-identical 368/368; routed experts distinct 1380/1380; no non-finite
value in 3348 tensors. The architecture behaves identically — only the routing
distribution moved.

Decode-step magnitudes are also stable and slightly below v2's: residual per position
23.5–24.0 against v2's 23.9–28.5.

## Caveat

The in-layer diversity comparison is across different position counts — 12 here against
5 in v1 — and more positions give more chances to differ, which would push v3's figure
*up*, not down. The effect survives that bias, but a same-length comparison has not been
run.
