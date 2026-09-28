# v8 — control

**Probes:** the control v5 needed and v6 could not provide. Short English prose, the same
factual register as v1, a different fact. Added after v6 showed the original control was
saturated.

```
prompt   "The chemical symbol for iron is"   (31 bytes, 6 ids)
gen      4
trace    /root/k3flow/v8.jsonl
```

## Outcome

Generated `' Fe, and the'` — correct.

| step | seconds | expert GB read |
|---:|---:|---:|
| 0 (prefill, T=6) | 36.58 | 105.44 |
| 1 | 10.07 | 24.78 |
| 2 | 8.34 | 14.56 |
| 3 | 8.32 | 13.72 |

## The matched comparison

v5 and v8 have almost the same expert-set size — 6,062 and 6,009 — so lift over chance
can be compared between them directly, which was impossible against v6's 47,362.

Overlap with v1 (English, "The capital of France is"), against chance:

| | set size | coverage | observed | **lift** |
|---|---:|---:|---:|---:|
| v5 French, **same fact** | 6,062 | 7.4% | 65.1% | **8.85×** |
| v8 English, **other fact, same register** | 6,009 | 7.3% | 50.6% | **6.94×** |
| v3 repetition | 6,541 | 7.9% | 37.0% | 4.66× |
| v7 nonsense ids | 10,626 | 12.9% | 38.7% | 3.00× |
| v4 code | 11,971 | 14.5% | 38.0% | 2.61× |
| v6 long prose | 47,362 | 57.5% | 73.3% | 1.28× |

## What this settles, and what it overturns

**Register dominates; meaning adds a real but smaller increment.**

Short English factual prose shares v1's experts at 6.94× chance *regardless of what it is
about*. Making it the same fact in another language raises that to 8.85× — a further
1.28×, or 824 more shared pairs out of 5,683. Against sample sizes in the thousands that
increment is not noise, but it is a minority of the total enrichment.

Everything that is **not** short natural prose — code, repetition, nonsense — sits at
2.6–4.7×, clearly separated from both prose runs.

**This overturns the conclusion recorded in v6.** That file read the control as favoring
meaning, on the strength of v6's low 1.28× lift. v6 covers 57.5% of the whole
(layer, expert) space, so its lift is compressed toward 1 by saturation and it could not
carry that argument. v8 is the matched control and says the opposite: most of the
French–English overlap is register, not meaning. The v6 file has been corrected.

## The sequence of corrections this took

Worth keeping, because the first three answers were all wrong in the same way:

1. Jaccard at 46% — invalid, not robust to set-size differences.
2. Containment at 65.1% — still rewards larger comparison sets.
3. Lift over chance against v6 — valid measure, but v6 is saturated, so the comparison
   was unsound even though the statistic was.
4. Lift over chance against a **size-matched** control — the first comparison that could
   answer the question, and it reversed the answer.
