# r — reverse replication

**Probes:** whether any of it depends on the order the runs were done in. Every earlier
conclusion rests on comparing runs that were executed in a particular sequence on one
machine. If order matters — through machine state, caches, thermal behavior, anything —
then the comparisons between variations are not comparisons of the model.

So the whole set is being re-run **from the last run to the first**, and each replay is
compared against its original **field by field, exactly**.

## Method

`trace_diff.py` compares two traces record by record. It compares only the keys present in
**both** records, because the tap gained fields over time — per-position rows, then the
channel index — and an older trace simply lacks them. That lets an old run and a new one
be checked against each other on everything they both recorded.

Floating point is compared **exactly**, not approximately. These runs are deterministic by
design; the engine documents bit-identical results at any thread count. "Close" is not the
claim being tested. Any difference at all is a finding.

## Method check, before any replay

Two independent runs of the same code prompt were already on disk from earlier cycles —
`v4rows` and `v4ch`, separate processes, run with different versions of the tap.

```
3067 records each
24,812 field values compared
fields: layer, pos, site, tok, value, bad, fnv, l2, max, mean, min, n,
        cols, rows, row_l2, row_absmax, ids, wt
VERDICT: byte-for-byte identical on every shared field
```

Two separate processes, same prompt, identical down to the last bit. That establishes the
baseline the replays are measured against, and it establishes it from data that already
existed rather than from a run made to prove the point.

## Reverse sequence

Executed last-to-first. `v4rows` and `v6rows` are skipped as exact duplicates of `v4ch`
and `v6ch` — same prompt, same generation count, same settings, differing only in the tap
version, and the newer tap is a superset. That is stated rather than quietly dropped.

| # | run | original | result |
|---:|---|---|---|
| 1 | v6ch — prose 225, rows+channel | `sweep/v6ch.jsonl` | running |
| 2 | c3-no-newline | `sweep/c3-no-newline.jsonl` | pending |
| 3 | c2-javascript | `sweep/c2-javascript.jsonl` | pending |
| 4 | c1-factorial | `sweep/c1-factorial.jsonl` | pending |
| 5 | s160 | `sweep/s160.jsonl` | pending |
| 6 | s80 | `sweep/s80.jsonl` | pending |
| 7 | v4ch — code 18, rows+channel | `sweep/v4ch.jsonl` | pending |
| 8 | s40 | `sweep/s40.jsonl` | pending |
| 9 | s18 | `sweep/s18.jsonl` | pending |
| 10 | s12 | `sweep/s12.jsonl` | pending |
| 11 | s5 | `sweep/s5.jsonl` | pending |
| 12 | v8 control | `v8.jsonl` | pending |
| 13 | v7 nonsense | `v7.jsonl` | pending |
| 14 | v6 long-context, gen 4 | `v6.jsonl` | pending |
| 15 | v5 french | `v5.jsonl` | pending |
| 16 | v4 code, gen 4 | `v4.jsonl` | pending |
| 17 | v3 repetitive | `v3.jsonl` | pending |
| 18 | v2 factual-multi, gen 8 | `v2.jsonl` | pending |
| 19 | v1 factual-short | `v1.jsonl` | pending |

## What would falsify what

Stated before the results exist, so it cannot be fitted afterwards:

- **Any** difference in a hash or a value means the runs are not deterministic, and every
  cross-variation comparison in this investigation would need re-examining, because those
  comparisons assume the only thing that changed between runs was the prompt.
- Differences confined to record *counts* would mean the flow changed, not the arithmetic.
- Identical traces throughout would mean order has no effect and the earlier conclusions
  rest on the model rather than on the sequence they were gathered in.
