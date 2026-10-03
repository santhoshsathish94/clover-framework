# Prompts, Outputs And Time Taken

Measured runs of the standalone engine on AX102, 2026-10-03. Every row below is a
real request through `run.sh`, not a reconstruction. Timings are whatever the
machine did on the day; the caveats at the end say what they do and do not cover.

## Machine and build

| | |
|---|---|
| Host | AX102, 124 GB RAM, expert data on `/dev/md2` |
| Threads | `OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores` |
| Binary | `bin/clover-one`, built by `build.sh` with `CLOVER_NPOS_SLOTS=8` |
| Config | `bin/configs/generation.json` |
| Warm-up | off (`K3_RESIDENT_WARMUP=0`, the default) |
| Expert storage | `experts.direct`, mapped; no decompression, no validation at run time |

## Prompts and outputs

Text is shown exactly as the vocabulary decodes it, including leading spaces and
newlines.

### 1. `The capital of France is`

```
input  ids : [1008, 10484, 318, 15383, 387]
output ids : [17374, 20829, 10, 427, 414, 1008, 606, 142957, 37092, 387, 7081, 306]
```

```
The capital of France is Paris.",
            "The Eiffel Tower is located in
```

12 new tokens, stop reason `length`, **109.336068450 s**, 9.11 s per token.

The answer is correct and then drifts into JSON punctuation. That is the shape the
stored five-field france records were written in, so it is the data showing
through rather than a decoding fault.

### 2. `Rain falls`, 12 new tokens

```
input  ids : [91019, 25528]
output ids : [418, 276, 13159, 318, 261, 4645, 316, 62598, 306, 261, 12981, 72919]
```

```
Rain falls on the roof of a house and collects in a rain gutter
```

12 new tokens, stop reason `length`, **81.484618991 s**, 6.79 s per token.

### 3. `Rain falls`, 32 new tokens

```
Rain falls on the roof of a house and collects in a rain gutter that drains into
a rain barrel. The base of the barrel is a circle with an area of
```

32 new tokens, stop reason `length`, **211.510645285 s**, 6.61 s per token.

The first 12 tokens are identical to run 2, ids included. Two separate processes,
same prompt, same output: the engine is deterministic across runs.

### 4. `In 1969, humans first walked on`

```
In 1969, humans first walked on the moon. The Apollo 11 mission was a historic
achievement, but it was
```

16 new tokens, stop reason `length`, **159.952222649 s**, 10.00 s per token.

### 5. `Rain falls`, 2 new tokens

```
input  ids : [91019, 25528]   output ids : [418, 276]   ->   " on the"
```

Used throughout as the short regression prompt. Warm, it completes in about 3.7 s.

## Where the time goes

Per-token cost is not constant. It tracks how much expert data the continuation
has not read yet.

| run | in | out | total | per token |
|---|---|---|---|---|
| `Rain falls` | 2 | 32 | 211.51 s | 6.61 s |
| `Rain falls` | 2 | 12 | 81.48 s | 6.79 s |
| `The capital of France is` | 5 | 12 | 109.34 s | 9.11 s |
| `In 1969, humans first walked on` | 9 | 16 | 159.95 s | 10.00 s |
| `Rain falls` (warm, short) | 2 | 1 | 3.66 s | 3.66 s |

Every generated token picks its own sixteen experts per layer, independently of
the tokens before it. A long continuation therefore keeps pulling expert data it
has never read, and 124 GB of page cache cannot hold a working set drawn from
1.45 TB. The short warm request is fast because it revisits experts already in
cache; the long ones are not, and no amount of batching changes that.

## Effect of prefill batching

Prefill positions do not project, so they can share one sweep over the weights.
`NPOS_SLOTS` sets how many may be batched. Warm page cache, repeated runs:

| request | one position at a time | batched |
|---|---|---|
| 2 in, 1 out | 4.81 / 4.86 s | 3.65 / 3.68 / 3.66 s |
| 8 in, 1 out | 52.6 / 55.2 s (2 slots) | 49.6 / 52.3 s (8 slots) |

The two-token gain is about a quarter and the spreads do not overlap. The
eight-token figures overlap, so that row shows the absence of harm, not a gain.
Batching amortises the dense weights only; the routed experts are different per
position and do not shrink.

Output is unchanged by batching. `The capital of France is` style prompts and an
eight-token prompt with three outputs produced identical ids at one, two and eight
slots, and the france reference check keeps all 7,360 reference expert matches.

## How the text was produced

The engine emits token ids only. `test-decode` in this directory is the zlib
decoder, not a detokenizer, and the host has no `tiktoken`, `regex` or
`transformers` installed. Ids were mapped through `dataset/tiktoken.model`
(163,584 base64 rank pairs; the remaining 256 of the 163,840 vocabulary are
special tokens) using a short script kept outside the repository, applying the
Kimi pretokenizer pattern restricted to ASCII plus tiktoken byte-pair merging.

That script is a reimplementation, so it was only trusted after it reproduced ids
already observed from other evidence: `Rain falls` encodes to 91019, 25528 and
`The capital` to 1008, 10484, both exact. Non-ASCII prompts were not checked and
should not be trusted to it.

## What these numbers do not cover

- One machine, one day. Timings are AX102 with this disk layout and 16 threads.
  Nothing here transfers to other hardware, and a GPU would reorder the costs
  entirely.
- Cold and warm page cache differ by roughly a factor of two on the same request.
  Each measurement says which it was; a figure without that is meaningless here.
- Sample counts are small. Where only one or two runs exist the figure is a
  single observation, not a mean.
- Output quality was read, not scored. Four prompts is an illustration of what the
  engine emits, not an evaluation of the model.
