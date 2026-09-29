# v5 — french

**Probes:** the same fact in another language. If routing were lexical, two prompts
sharing almost no tokens should share almost no experts.

```
prompt   "La capitale de la France est"   (28 bytes, sha1 9cd7df561afc, 7 ids)
gen      4
trace    /root/k3flow/v5.jsonl        65.2 s wall, I/O 31.7%
```

## Outcome

Generated `' Paris.",\n+           '` — tokens **17374, 20829, 10, 427**.

Those are the **identical first four tokens v2 produced from the English prompt**. Two
prompts in different languages, sharing essentially no tokens, converge on the same
continuation — including the drift into diff-like punctuation, which appears to be a
property of raw completion without a chat template rather than of either language.

| step | token | seconds | expert GB read |
|---:|---:|---:|---:|
| 0 (prefill, T=7) | 17374 | 36.01 | 106.37 |
| 1 | 20829 | 11.59 | 25.08 |
| 2 | 10 | 9.87 | 18.11 |
| 3 | 427 | 7.72 | 14.13 |

## Routing looks semantic rather than lexical

Jaccard overlap of the prefill (layer, expert) sets — intersection over union:

| | v1 factual EN | v5 factual FR | v3 repetitive | v4 code |
|---|---:|---:|---:|---:|
| **v1 factual EN** | 100% | **46.0%** | 20.8% | 13.9% |
| **v5 factual FR** | 46.0% | 100% | 19.0% | 13.8% |
| **v3 repetitive** | 20.8% | 19.0% | 100% | 14.0% |
| **v4 code** | 13.9% | 13.8% | 14.0% | 100% |

The English and French statements of the same fact share **46.0%** of their routed
experts — 3.3× their overlap with code and 2.2× with repeated text. On the decode step
where both runs emitted token 20829, overlap is 45.5%: 920 of 1472 (layer, expert) pairs
shared.

## The control this does not have

v1 and v5 are both **natural-language prose**, while v4 is code and v3 is degenerate
repetition. So the 46% may be measuring "prose against non-prose" rather than "the same
fact in two languages". Separating those needs a prose prompt on an unrelated topic, and
**v6 is exactly that** — long English technical prose. If v1–v6 overlap lands near 46%,
the result here is about register, not meaning. If it lands near 20%, it is about
meaning.

Recorded before v6 runs, so the prediction is on the record either way.

## Other readings

- **No final-layer amplification**: L91 attention / L90 residual = 0.14×, max element
  1.174. So v4's 19.1× spike is not simply a long-prompt effect — v5 has 7 positions,
  v1 has 5, v4 has 18, and only v4 spikes.
- In-layer diversity 58.8% over 7 positions, between v1's 77.2% (5) and v3's 37.0% (12).
  Position count and diversity move together here, which is a confound the current set
  does not resolve.
- Mean adjacent-step reuse 45.6%, close to v2's 42.4%.
- Residual per position 33.70, against v1's 36.92.
