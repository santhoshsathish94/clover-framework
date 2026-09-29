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
| 14 | v6 long-context, gen 4 | 25,128 | 187,776 | **identical** |
| 15 | v5 french | 5,072 | 47,384 | **identical** |
| 16 | v4 code, gen 4 | 6,084 | 54,468 | **identical** |
| 17 | v3 repetitive | 5,532 | 50,604 | **identical** |
| 18 | v2 factual-multi, gen 8 | 9,408 | 89,616 | **identical** |
| 19 | v1 factual-short | 1,498 | 13,456 | **identical** |

**Totals: 135,059 records, 1,082,807 field values, zero differences.** With the method
check, 1,107,619 field values compared in all.

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

## Flow is identical. Timing is not.

The trace records no time, so the comparisons above say nothing about it. Checked
separately from the run logs, over 19 completed steps:

**Identical in every replay:**

| quantity | agreement |
|---|---|
| token emitted | exact |
| expert bytes read | exact, to two decimals — e.g. 831.07 GB, 226.97 GB, 98.30 GB |
| cache hit rate | exact |
| every trace field | exact |

**Not identical:**

```
wall clock delta over 19 steps:  mean +5.21%   min -10.93%   max +46.75%   stdev 12.87
```

The two largest deviations are both the final decode step of a four-token run — v8 step 3
at +46.8% and v7 step 3 at +29.9% — on steps taking eight seconds, where a small absolute
wobble is a large relative one. Not investigated further; recorded as observed.

### What this costs the earlier conclusions

Nothing that was measured from the traces, which is almost all of it. But several claims
in the variation files are **wall-clock** claims, and wall clock moves by up to a factor
of 1.5 on a single step:

- "prefill is 4× the cost of a decode step"
- "step time settles around 7.1–7.7 s"
- "I/O share of wall clock 27.8%"

Those should be read as approximate on this machine, not as measurements. The **bytes
read** figures alongside them are exact and reproducible, and are the better basis for any
statement about cost — reads falling 99.72 → 24.95 → … → 12.28 GB is a reproducible fact;
the seconds attached to it are not.

This distinction was not in the plan. It came from being asked whether flow and timing
were the same, having only checked one of them.

### Final timing figures, all 44 generation steps

```
token emitted identical    44 / 44
expert bytes identical     44 / 44
cache hit rate identical   44 / 44
wall clock                 mean +1.72%   min -32.42%   max +70.73%   stdev 19.66
steps within +/-10% on time  25 / 44
```

Only 57% of steps land within 10% of their original time, and the extremes span a factor
of 2.5 between the slowest and fastest relative outcome. Wall clock on this machine is not
a stable measurement at single-step resolution.

## Verdict

**Order has no effect.** Running the entire set backwards reproduced every trace exactly:
135,059 records and 1,082,807 field values with zero differences. Every hash, norm,
minimum, maximum, mean, per-position magnitude, channel index, routing choice and routing
weight came back the same.

That matters because every cross-variation conclusion in this investigation assumes the
only thing that differed between two runs was the prompt. That assumption is now checked
rather than relied upon.

**What this does not establish.** It validates the measurements, not the reasoning built
on them. All the derived figures — the 8.85× lift, the ~3400 ceiling, channel 4590, the
layer-90 onset — were computed from these traces, so identical traces give identical
numbers by construction. Replication confirms the foundation is solid; it cannot confirm
that the interpretations drawn from it are correct. Those still rest on the arguments made
for them, and two of them have already been overturned once.

**One claim weakened.** Timing-based statements are not measurements at this resolution
and have been qualified in place.

## Where the timing variance actually comes from

Identical work taking different time has to come from outside the computation. The
obvious suspect on this machine is the disk: there is a documented hardware fault, nvme1n1
negotiating a PCIe **x2** link instead of x4, delivering 473 MB/s against its RAID1 twin's
702 MB/s at double the latency.

The engine reports wall clock split into read time and everything else, so the guess can
be checked rather than assumed. Over 19 runs:

| | mean | min | max | stdev |
|---|---:|---:|---:|---:|
| I/O seconds | +0.20% | −0.97% | +1.00% | **0.41** |
| disk throughput MB/s | +0.01% | −0.96% | +2.07% | **0.64** |
| compute seconds | +0.03% | **−18.29%** | **+14.42%** | **7.93** |

**It is not the disk.** Expert bytes read were identical in all 19 runs, and the drives
delivered them at the same rate to within 1%. I/O time is the most stable quantity in the
entire measurement — a twentyfold smaller spread than compute.

All of the variance is CPU-side. This machine is a Ryzen 9 7950X3D, which has **two
asymmetric core complexes** — one with the stacked cache, one without — and the runs do no
thread pinning. Which complex the OpenMP threads land on can differ between runs. That is
a hypothesis consistent with the data, **not** a finding; confirming it would need runs
with threads pinned to one complex and then the other.

### A note for the hardware ticket

The x2 link is a real and constant handicap, but this data says it is **consistent**, not
erratic: 19 runs moving 98 GB to 831 GB each delivered throughput within a 3-point band.
Whatever the degraded link costs, it costs it the same way every time. That is worth
knowing separately from the ticket's bandwidth argument.

### And a correction to the obvious inference

Having a PCIe fault report open on the same machine made "it must be the disk" the natural
reading, and it is wrong. The measurement that was available all along says the opposite.


## What would falsify what

Stated before the results exist, so it cannot be fitted afterwards:

- **Any** difference in a hash or a value means the runs are not deterministic, and every
  cross-variation comparison in this investigation would need re-examining, because those
  comparisons assume the only thing that changed between runs was the prompt.
- Differences confined to record *counts* would mean the flow changed, not the arithmetic.
- Identical traces throughout would mean order has no effect and the earlier conclusions
  rest on the model rather than on the sequence they were gathered in.
