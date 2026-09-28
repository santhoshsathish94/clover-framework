# v1 — factual-short

**Probes:** the baseline. One token from a short factual prompt, everything else held
constant. Every later variation is read against this.

```
prompt   "The capital of France is"   (24 bytes, ascii, sha1 2f0ef9e54ace)
gen      1
flags    --incremental --trunk /root/k3trunk_i8 --trunk-gb 60 --cache-gb 30
branch   clover/observe @ f97eaaa      trace /root/k3flow/v1.jsonl (1498 records)
```

## Outcome

**Token 17374 = `" Paris"`.** 32.1 s wall, I/O 45.4% of it (trunk 5.5 s, experts 9.1 s).
5 prompt tokens in, 1 generated.

## What the trace shows

| | |
|---|---|
| embed L2 → logits L2 | 4.138 → 1330.2 (max logit 18.113) |
| residual first / last / max / mean | 1.92 / 170.97 / 482.16 / 82.56 |
| residual at snapshot layers | 7.14 |
| snapshot layers | 0, 12, 24, 36, 48, 60, 72, 84 |
| KDA attention out, mean L2 | 15.546 (69 layers) |
| MLA attention out, mean L2 | 19.485 (24 layers) |
| norm before attention | 15.21 … 126.64, mean 49.19 |
| norm before MLP | 1.72 … 46.48, mean 24.24 |
| routing decisions | 460 = 92 MoE layers × 5 positions |
| expert slots → distinct (layer, expert) | 7360 → 5683 |
| in-layer diversity | 77.2% |
| routing weight sum | exactly 1.00000 everywhere |
| largest single expert share | 0.6450 |
| top weight not the first pick | 20.0% |

Invariants: layer chain bit-identical 92/92; snapshot-replaces / ordinary-adds 93/93;
routed experts distinct 460/460; no non-finite value in 837 tensors.

## Worth noting

**The trace agrees with the engine's own counters.** The trace yields 5683 distinct
(layer, expert) pairs; the engine's cache statistics independently report
`requests 5683`. Two different mechanisms counting the same thing and landing on the
same number is the best evidence available that the instrument measures what it claims.

**7360 slots against 5683 unique requests** means 1677 of the picks — 22.8% — were an
expert the pass had already needed. That reuse is what the expert cache exists to
capture.

**Identical to the pre-removal run.** v1 reproduces the earlier traced run exactly
(same token, same residual figures, same routing statistics) after the `--direction`
boundary was stripped out. Removing it changed no behavior.

## Open from this run

- 896 experts exist per layer and only 16 fire per token; nothing here says whether the
  other 880 are ever used, or how that changes with input.
- One prompt, five positions. In-layer diversity of 77.2% over five positions is not
  comparable to the depth-dependent cache hit rates measured earlier over long runs.
