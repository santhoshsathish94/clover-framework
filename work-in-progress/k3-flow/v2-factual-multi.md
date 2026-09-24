# v2 — factual-multi

**Probes:** what changes between prefill and decode, and how much routing carries over
from one generated token to the next. Same prompt as v1, eight tokens instead of one.

```
prompt   "The capital of France is"   (identical to v1, sha1 2f0ef9e54ace)
gen      8
flags    --incremental --trunk /root/k3trunk_i8 --trunk-gb 60 --cache-gb 30
trace    /root/k3flow/v2.jsonl        89.1 s wall, I/O 27.8%
```

## Outcome

```
 Paris.",
+            "The Eiffel
```

17374 `' Paris'`, 20829 `'.",\n'`, 10 `'+'`, 427 (spaces), 414 `' "'`, 1008 `'The'`,
606 `' E'`, 142957 `'iffel'`.

It answers correctly and then drifts into what looks like a source diff. There is no
chat template here — this is raw completion, so the continuation is whatever the prompt
most resembles in training data.

## Prefill is one step; the other seven are a different regime

| step | token | seconds | expert GB read |
|---:|---:|---:|---:|
| 0 (prefill, T=5) | 17374 | 30.39 | 99.72 |
| 1 | 20829 | 11.21 | 24.95 |
| 2 | 10 | 10.38 | 19.78 |
| 3 | 427 | 7.39 | 14.07 |
| 4 | 414 | 7.10 | 12.28 |
| 5 | 1008 | 7.67 | 12.88 |
| 6 | 606 | 7.37 | 15.09 |
| 7 | 142957 | 7.63 | 18.11 |

**The cache warms and then holds.** Expert reads fall 99.72 → 24.95 → … → 12.28 GB and
step time settles around 7.1–7.7 s. Prefill is 4× the cost of a decode step here, and
I/O share falls from 45.4% (v1, prefill only) to 27.8% across the whole run.

## Adjacent-step reuse is what the cache lives on

Between consecutive steps, the fraction of (layer, expert) pairs already needed by the
previous step:

```
54.1%  23.4%  45.5%  52.4%  50.1%  41.6%  29.9%      mean 42.4%
```

The all-step figure is far harsher and nearly useless for this purpose: of 9991 distinct
(layer, expert) pairs across the run, only **54 — 0.5% — are used by every step**. Each
token needs roughly 1472 experts and the cache holds 1708 slots, so it can keep about
one token's worth. That is why mean adjacent reuse of 42.4%, not the 0.5% intersection,
is the number that predicts the read volume.

## Decode positions carry a smaller residual than prefill ones

Raw L2 is not comparable between a 5-position prefill buffer and a 1-position decode
buffer, so it is normalized per position:

| | prefill (step 0) | decode (steps 1–7) |
|---|---|---|
| residual mean, per position | 36.92 | 23.9 – 28.5 |
| residual at snapshot layers | 7.14 | 2.75 – 3.00 |
| KDA attention out, mean | 15.55 | 6.7 – 8.0 |
| MLA attention out, mean | 19.49 | 9.0 – 13.2 |
| norm before attention, mean | 49.19 | ~21.8 |
| largest single expert share | 0.6450 | 0.31 – 0.36 |
| top weight not the first pick | 20.0% | 20.7% – 28.3% |

The sawtooth and the snapshot layers are identical in both regimes: 0, 12, 24, 36, 48,
60, 72, 84, residual replaced not added, every step.

Invariants across all eight steps: layer chain bit-identical 736/736; snapshot-replaces /
ordinary-adds 744/744; routed experts distinct 1104/1104; no non-finite value in 6696
tensors.

## A metric that had to be discarded

"In-layer diversity" reads 100.0% for every decode step. That is degenerate, not
interesting: with one position there is a single pick set per layer, so sixteen of
sixteen slots are trivially distinct. The measure only means something when T > 1, which
makes v1's 77.2% and v6 the only places it can be read.

## Open from this run

- The 0.645 largest-share in prefill against ~0.33 in decode is probably just
  max-of-460 against max-of-92, not a real regime difference. Untested.
- Whether the low reuse at step 1→2 (23.4%) tracks the content shift from `'.",\n'` to
  `'+'` is a guess; the other low value, 29.9%, sits on a within-word continuation, so
  the obvious story does not hold.
