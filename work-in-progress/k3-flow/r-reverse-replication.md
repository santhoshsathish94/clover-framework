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

| # | run | records | fields compared | result |
|---:|---|---:|---:|---|
| 1 | v6ch — prose 225, rows+channel | 22,111 | 158,493 | **identical** |
| 2 | c3-no-newline | 2,791 | 23,253 | **identical** |
| 3 | c2-javascript | 3,435 | 27,761 | **identical** |
| 4 | c1-factorial | 2,975 | 24,541 | **identical** |
| 5 | s160 | 16,131 | 116,633 | **identical** |
| 6 | s80 | 8,771 | 65,113 | **identical** |
| 7 | v4ch — code 18, rows+channel | 3,067 | 25,185 | **identical** |
| 8 | s40 | 5,091 | 38,980 | **identical** |
| 9 | s18 | 3,067 | 24,812 | **identical** |
| 10 | s12 | 2,515 | 20,948 | **identical** |
| 11 | s5 | 1,871 | 16,440 | **identical** |
| 12 | v8 control | 4,980 | 46,740 | **identical** |
| 13 | v7 nonsense | 5,532 | 50,604 | **identical** |
| 14 | v6 long-context, gen 4 | | | running |
| 15 | v5 french | | | pending |
| 16 | v4 code, gen 4 | | | pending |
| 17 | v3 repetitive | | | pending |
| 18 | v2 factual-multi, gen 8 | | | pending |
| 19 | v1 factual-short | | | pending |

The v-series originals predate the per-position tap, so those replays are run **without**
`K3_TRACE_ROWS` to match the original record set exactly rather than relying on the
comparator's key intersection to paper over a different flow.

## A trap worth recording

Partway through, a comparison was attempted against `s80` while its process was still
running. The trace's final line was half-written, and the comparator failed with a JSON
parse error at column 295.

That looks exactly like a corrupt file. It was a file being appended to. The guard in use
was "the trace is non-empty", which is not sufficient; it has to be "the process has
exited". Changed to check `pgrep` before comparing.

## What would falsify what

Stated before the results exist, so it cannot be fitted afterwards:

- **Any** difference in a hash or a value means the runs are not deterministic, and every
  cross-variation comparison in this investigation would need re-examining, because those
  comparisons assume the only thing that changed between runs was the prompt.
- Differences confined to record *counts* would mean the flow changed, not the arithmetic.
- Identical traces throughout would mean order has no effect and the earlier conclusions
  rest on the model rather than on the sequence they were gathered in.
