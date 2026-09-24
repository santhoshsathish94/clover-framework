# v6 — long-context

**Probes:** prefill at scale, and the control for v5's prediction — English prose on an
unrelated topic.

```
prompt   225 ids of technical prose about the engine's own streaming design
         (1175 bytes, sha1 8d503e639313)
gen      4
trace    /root/k3flow/v6.jsonl        7 m 58 s wall, I/O 22.0%
```

## Outcome

Generated `' The engine is a'` — 646, 6052, 387, 261.

| step | seconds | expert GB read | cache hit |
|---:|---:|---:|---:|
| 0 (prefill, T=225) | 443.20 | **831.07** | 72.8% |
| 1 | 11.19 | 25.57 | 100% |
| 2 | 10.01 | 19.37 | 100% |
| 3 | 9.55 | 15.69 | 100% |

Prefill read **831 GB of experts** — 8.3× v1's 99.7 GB for 45× the positions. KV cache
544 MB for 230 positions. Decode steps are indistinguishable from every other run.

**In-layer diversity falls to 14.3%**: 331,200 expert slots across 225 positions need
only 47,362 distinct (layer, expert) pairs. The more positions there are, the more they
agree — 77.2% at 5 positions, 58.8% at 7, 45.2% at 18, 14.3% at 225. This is the
saturation the expert cache depends on at long context.

## The prediction I recorded was not answerable as written

Before this run the context file said: v1-vs-v6 overlap **near 46% means register, near
20% means meaning**. Observed Jaccard was **8.5%** — below both.

The thresholds were invalid. v6's expert set holds 47,362 pairs against v1's 5,683, and
Jaccard is not robust to an 8× size difference: even if v1 were *entirely contained* in
v6, Jaccard could not exceed 12.0%. The measure could not have produced 46% whatever the
answer. That is a badly designed test, not a surprising result.

### What a size-robust measure says

Overlap with v1, against what the set sizes alone would give by chance. The space is
92 MoE layers × 896 experts = 82,432 pairs.

| compared with v1 | set size | coverage | observed | chance | **lift** |
|---|---:|---:|---:|---:|---:|
| v5 French, same fact | 6,062 | 7.4% | 65.1% | 7.4% | **8.85×** |
| v3 repetition | 6,541 | 7.9% | 37.0% | 7.9% | 4.66× |
| v4 code | 11,971 | 14.5% | 38.0% | 14.5% | 2.61× |
| **v6 English prose, other topic** | 47,362 | 57.5% | 73.3% | 57.5% | **1.28×** |

**The control comes down on the side of meaning.** v6 is the same language and the same
register as v1 and shows the *lowest* enrichment of the four. If the French result were
about register, v6 would be high. It is the same-meaning pair that is enriched 8.85×.

Caveat kept in view: v6 covers 57.5% of the whole pair space, so its lift is compressed
toward 1 by saturation. The ordering is informative; the exact 1.28× is not. A short
English prose prompt on an unrelated topic would be a cleaner control and has not been
run.

### Overturned by v8

The paragraph above is wrong, and the caveat under it is the reason. v6 covers 57.5% of
the pair space, so its lift is compressed toward 1 by saturation and cannot carry the
argument. **v8** — "The chemical symbol for iron is", 6,009 pairs against v5's 6,062, a
properly matched control — reverses it: different fact, same register, **6.94×** against
the same-fact French run's 8.85×. Register accounts for most of the overlap and meaning
adds a smaller further increment. See `v8-control.md`.

## The final-layer amplification is not code-specific

v6 spikes too: L91 attention / L90 residual = **7.30×**, maximum element **3319.94**.
This contradicts what v4's file first concluded, and that file has been corrected.
Across all five prefills the spike appears at T = 18 and T = 225 but not at T ≤ 12, with
the maximum element landing near 3300–3500 in both spiking runs. Content and length vary
together here, so neither is isolated.

## Other readings

- Residual per position 44.07, the highest of any run; raw mean 661.06, last-layer 17324.
- Largest single expert share **0.8515** — one expert taking 85% of a token's routed
  contribution, the most concentrated seen.
- Mean adjacent-step reuse 53.6%, with the first transition at 96.5%: the token
  immediately after a 225-token prefill reuses almost everything the prefill touched.
- Top weight not the first pick: 23.8%, in line with every other run (20–28%).
