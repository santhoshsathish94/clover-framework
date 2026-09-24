# v7 — nonsense

**Probes:** out-of-distribution input. Twelve token ids drawn deterministically from
across the vocabulary, forming no real text.

```
ids      5002, 15976, 39064, 73907, 88317, 97385, 112854, 117120,
         142248, 144245, 147463, 150069        (seed 20260924, 12 ids)
gen      4
trace    /root/k3flow/v7.jsonl        85.1 s wall, I/O 38.0%
```

## Outcome

The model falls into a **2-cycle**: 220, 269, 220, 269 — `' '` and `'的'` alternating.
Given nonsense it produces a degenerate loop rather than anything resembling text.

| step | token | seconds | expert GB read |
|---:|---:|---:|---:|
| 0 (prefill, T=12) | 220 `' '` | 61.16 | 186.46 |
| 1 | 269 `'的'` | 8.10 | 25.32 |
| 2 | 220 `' '` | 7.05 | 14.60 |
| 3 | 269 `'的'` | 8.84 | 15.13 |

## Confidence collapses, and it is visible in the logits

Peak logit at the prefill step, across every run:

| run | logits L2 | max logit |
|---|---:|---:|
| v1 factual | 1330.2 | 18.113 |
| v3 repetitive | 1180.8 | 19.025 |
| v4 code | 1398.0 | 23.504 |
| v5 french | 1159.0 | 18.724 |
| v6 long prose | 1336.9 | 18.095 |
| **v7 nonsense** | **812.2** | **8.385** |

Every run on real text lands between 18.1 and 23.5. Nonsense comes in at **8.385** —
less than half the lowest of them — and its logit vector is a third shorter. The model's
uncertainty is legible directly in the output layer, without any probability calibration
or sampling.

## The cleanest diversity comparison in the set

v3 and v7 both have **exactly 12 positions**, so position count is held constant and only
content differs:

| | v3 repetition | v7 nonsense |
|---|---:|---:|
| expert slots | 17,664 | 17,664 |
| distinct (layer, expert) pairs | 6,541 | **10,626** |
| in-layer diversity | 37.0% | **60.2%** |
| expert GB read, prefill | 114.78 | **186.46** |

Twelve identical tokens and twelve unrelated tokens cost 115 GB and 186 GB respectively,
for the same amount of work. Routing breadth — not sequence length — is what the expert
reads track. Every earlier diversity comparison was confounded by differing position
counts; this pair is not.

## Other readings

- Lift over chance for sharing v1's experts: **3.00×** — above code (2.61×), below
  repetition (4.66×), well below the French statement of the same fact (8.85×).
- **No final-layer amplification**: 0.11×, max element 0.811, consistent with every
  T ≤ 12 run.
- Residual per position 29.48, in line with v3's 29.93 at the same length.
- Largest single expert share 0.5598; top weight not first 22.7% — both unremarkable.
  Routing *mechanics* are unchanged by nonsense; only the distribution moves.
