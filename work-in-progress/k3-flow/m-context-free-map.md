# m — the context-free map

**One token in, 93 layers, one token out, for every id in the vocabulary.**

Question: if a single token is fed with no context at all, what does the model produce?
Repeated over all 163,840 ids, that is a complete and finite table — the only slice of
this model that can be exhaustively enumerated.

Started 2026-09-24. Running. Resumable.

## Why this slice and not a bigger one

The model's output is not a function of the last token. v3 measured that directly: twelve
copies of the same token ` the` at twelve positions routed **differently at 91 of 92
layers and identically at none**. So there is no token → token table for the general case.

But a *single* token has no preceding context by definition, so its output is well defined
and fixed. The table is 163,840 rows. The next one up, two-token prefixes, is 2.68 × 10¹⁰
rows — about 6 years of machine time — and would still say nothing about length three.

One property worth naming rather than discovering later: a lone token always sits at
position 0, and position 0 is the attention sink — we measured it receiving up to 99.8% of
attention from other positions. This table is therefore specifically *"what each token
predicts when it is the first and only token"*. That is a real, well-defined slice, not a
distorted one, but it is not "what each token means".

## How it is run

Separate processes were not viable: one invocation measured **20.51 s**, of which **12.0 s
is re-reading the 108.81 GB trunk**. Times 163,840 that is 39 years of trunk loading for
163,840 useful forward passes.

So `--sweep-out / --sweep-from / --sweep-to` were added to `src/cli/k3_run.c` on
`clover/observe`: bind the model **once**, then loop `forward()` with `T = 1` per id.

Safe because `forward()` is self-contained whenever `w->kvc` is NULL:

```c
memset(br, 0, (size_t)T * maxb * E * sizeof(float));
if (!w->kvc) {                       /* no --incremental */
    memset(kstate, 0, kper * slots * sizeof(float));
}
```

`br` and the KDA recurrent state are cleared on every call and `h` is fully overwritten by
the embedding, so nothing survives between ids. The sweep **refuses `--incremental`**,
which is the one flag that would make `w->kvc` non-NULL and let id *N−1* leak into id *N*.

Command:

```
./bin/k3 /root/k3model --tok /root/k3model --trunk /root/k3trunk \
  --trunk-gb 114 --cache-gb 2 --ids 0 --gen 1 \
  --sweep-out /root/k3map/map.tsv --sweep-from 0 --sweep-to 163839
```

bf16 trunk deliberately. int8 and MXFP4 both change the output — measured, first divergence
at generated token 21 and 3 respectively — so a map built on them would be a map of a
different model.

Per row: `in`, `out`, `top_logit`, `margin`, `mean`, `secs`, then the 2nd–5th ranked ids and
their logits. The runners-up are recorded because a 0.07 margin and a 2.4 margin are very
different claims about the same `out` value.

## Two gates passed before starting

**1. The sweep agrees with a separate process.** Token 17374 (` Paris`) run as its own
process produced `generated_ids: [11]`. The sweep row for 17374 is `out = 11`. Same answer,
so binding once does not change the result.

**2. The sweep is deterministic.** The range 17370–17380 was run twice into separate files
and compared field by field with `secs` removed:

```
rows A=11  B=11
fields per row (secs removed): 13
total field values compared: 143
differences: 0
```

The first attempt at this check reported "identical" while comparing **1 field per row** —
a malformed comparison that would have passed no matter what. It was redone to print the
field and value counts, so a pass now proves something was actually read.

## Cost

| | |
|---|---:|
| separate process per id | 20.51 s → **39 days** |
| bound once, per id | **5.61 s** → **10.6 days** |
| first id of a run | 18.1 s (cold expert cache) |

5.61 s per id is roughly 2.27 s of trunk out of DRAM plus 25.83 GB of expert reads from
disk. The expert cache is only 2 GB because the bf16 trunk pins 114 GB of the 124 GB box,
and a larger cache would mean a smaller trunk, which costs more than it saves.

## Results

Running. First 12 rows, `in → out`:

| in | out | top logit | margin |
|---:|---:|---:|---:|
| 7 | 16 | 10.702 | 0.355 |
| 8 | 440 | 9.441 | 0.757 |
| 9 | 220 | 8.763 | 1.129 |
| 17370 | 11 | 9.124 | 0.128 |
| 17372 | 566 | 9.262 | 0.599 |
| 17374 | 11 | 10.099 | 1.171 |
| 17375 | 1686 | 11.616 | 2.413 |

Too early for any claim. Questions to put to the finished table, written down now so they
are not invented to fit the data afterwards:

- How concentrated is the image? Token 11 already appears three times in eleven rows.
- How many distinct output tokens does the whole vocabulary produce?
- Are there fixed points, `f(x) = x`? Id 0 → 0 in the first row.
- Are there cycles under repeated application?
- How does `margin` distribute — is the model confident with no context, or nearly tied?
- Do any of the 240 unreachable ids (163,600–163,839) ever appear as an output? They are
  **not** masked and their `lm_head` rows are fully populated, so they can win an argmax.

## The table we already had — 267 pairs from runs already on disk

Built while the sweep runs, from every result JSON with token ids
(`/root/k3map/build-observed-map.py` → `/root/k3map/observed-pairs.tsv`).

**These pairs are not context-free.** Each generated token saw everything before it. So
this table answers the opposite question: given that a token was preceded by something,
how often is it followed by the same thing?

43 files reduce to **10 distinct runs** and, more importantly, only **4 distinct prompts**
— most of the benchmark runs are the same prompt under different memory settings. Counting
raw occurrences would therefore have counted the same context several times over.

Counting only **distinct preceding contexts**:

| | |
|---|---:|
| in-tokens appearing in 2+ distinct contexts | 35 |
| produced the same output every time | 8 |
| **produced different outputs** | **27 (77%)** |

And ambiguity rises monotonically with how often a token is seen, which is what says the
"stable" cases are simply under-sampled rather than genuinely deterministic:

| times seen | tokens | ambiguous |
|---|---:|---:|
| exactly once | 72 | 0% |
| 2–3 times | 34 | 47% |
| 4–9 times | 14 | 64% |
| **10+ times** | **2** | **100%** |

**Every token observed ten or more times produced more than one output.**

The clearest single case is token 220, a bare space, seen 26 times:

```
220 " "  ->  16 "1"   x7
        ->  17 "2"    x6
        ->  18 "3"    x5
        ->  19 "4"    x4
        ->  20 "5"    x3
        -> 2205 "60"  x1
```

The model is counting. The same input token produces the next number in a list, and which
number depends entirely on what came before. Nothing about the token itself decides it.

Others: ` the` gave 7 different outputs in 9 sightings, ` a` gave 6 in 8, `,` gave 4 in 10,
`.` gave 5 in 9.

The 8 that held constant are all thin evidence — 2 or 3 contexts each — and four of them
are the digits 2, 3, 4, 5 each followed by a space, which is list formatting rather than
meaning:

```
17 "2" -> 220 " "   in 2 contexts
18 "3" -> 220 " "   in 2 contexts
19 "4" -> 220 " "   in 2 contexts
20 "5" -> 220 " "   in 3 contexts
691  " can"     -> 859  " do"     in 2 contexts
2009 " where"   -> 1129 " they"   in 2 contexts
5849 " discuss" -> 1632 " how"    in 2 contexts
24650 " Finally"-> 11   ","       in 2 contexts
```

**What this settles.** There is no token → token table for the model in use. This is the
third independent measurement to say so, after v3's twelve copies of ` the` routing
differently at 91 of 92 layers, and the KV cache and recurrent state existing at all.

**What it does not settle.** It says nothing about the context-free case, which is exactly
why the sweep is still running. A lone token at position 0 has no preceding context to vary,
so it genuinely does have one fixed answer. These two tables are complementary, not
competing.

**A defect worth recording.** The first version of the writer emitted token text into a TSV
unescaped. Token bytes routinely contain tabs and newlines, so the column layout was
destroyed and the reader crashed on a `None`. Text is now JSON-escaped. A crash was lucky —
the same bug could as easily have shifted columns silently and produced a plausible, wrong
table.

## Files

- `/root/k3map/map.tsv` — the context-free table, appended as it runs
- `/root/k3map/sweep.log` — progress and ETA
- `/root/k3map/run-sweep.sh` — resumable launcher; re-running continues after the last id
  already on disk
- `/root/k3map/observed-pairs.tsv` — the 267 context-dependent pairs, copied to
  `c:\personal\oss\k3-results\observed-pairs.tsv`
- `/root/k3map/build-observed-map.py` — builds the above from the run JSONs
- engine change on `clover/observe`, `src/cli/k3_run.c`, builds with zero warnings
