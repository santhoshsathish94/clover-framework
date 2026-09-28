# Does it actually give the same answer?

Two completely different programs were given the same 34 prompts and the same
1.5 TB of Kimi K3 weights, on the same machine.

**All 34 answers were identical. `clover-k3` took 3.40x less time.**

This page is the evidence. No arithmetic below is estimated; every number comes
from a recorded run, and section 7 lists the things this does *not* show.

---

## 1. What "the answer" means here

Both programs read a prompt and emit **the next token** — one piece of text,
usually a word or part of one.

```
   "The capital of France is"   ->   token 17374   ->   " Paris"
```

They produce a list of 163,840 scores, one per possible token, and the answer
is the highest. That number, 17374, is what has to match.

Section 4 goes further and generates whole sentences, which is a different and
much slower activity. The 34-prompt comparison below is one token per prompt.

The two programs being compared:

| | what it is |
|---|---|
| **clover-k3** | one C file, ~2,500 lines, written from the model's equation. `clover-k3.c` in this folder. |
| **the engine** | [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), a separate, independent C implementation of the same model. |

They were written from different starting points and share no code. If they
agree on all 34 prompts, that is two independent programs reaching the same
answer — not one program agreeing with itself.

---

## 2. Ten prompts anyone can check

| prompt | answer | clover-k3 | the engine |
|---|---|---|---|
| The capital of France is | ` Paris` | **7.23 s** | 23.60 s |
| The chemical symbol for gold is | ` Au` | **7.72 s** | 25.55 s |
| The largest planet in our solar system is | ` Jupiter` | **9.35 s** | 30.13 s |
| The author of Pride and Prejudice was | ` Jane` | **10.29 s** | 33.83 s |
| In 1969 the first humans landed on the | ` moon` | **11.80 s** | 38.01 s |
| The mitochondria is often called the powerhouse of the | ` cell` | **10.64 s** | 35.99 s |
| Paris is the capital of France and Berlin is the capital of | ` Germany` | **11.37 s** | 39.36 s |
| Paris is the capital of France and Rome is the capital of | ` Italy` | **11.38 s** | 39.38 s |
| La capitale de la France est | ` Paris` | **7.94 s** | 26.69 s |
| Die Hauptstadt von Frankreich ist | ` Paris` | **8.25 s** | 27.78 s |

Same answer every time. Between three and three and a half times faster every
time.

### Three that are worth pausing on

**The same question in three languages.** English, French and German all emit
token **17374**, ` Paris` — the identical token id, not merely the same idea.

**An analogy it has to complete, not recall.** `...and Berlin is the capital of`
gives ` Germany`; swap Berlin for Rome and the same sentence gives ` Italy`. The
prompt structure is identical; only one word changed.

**Two prompts that share eleven words and diverge:**

```
Paris is the capital of France, a city on the river Seine that has long   ->  " been"
Paris is the capital of France, a city that sits far inland away from any ->  " ocean"
```

Both programs agree on both, including the disagreement between them.

---

## 3. All 34

`clover-k3` is the current build with the cross-layer lookahead enabled. Both
columns are wall-clock seconds for the whole run, cold start to emitted token.

| # | len | prompt | answer | clover-k3 | engine | ratio |
|---|---|---|---|---|---|---|
| 1 | 5 | The capital of France is | ` Paris` | 7.23 | 23.60 | 3.26x |
| 2 | 6 | The chemical symbol for gold is | ` Au` | 7.72 | 25.55 | 3.31x |
| 3 | 8 | The largest planet in our solar system is | ` Jupiter` | 9.35 | 30.13 | 3.22x |
| 4 | 6 | Water boils at a temperature of | ` ` | 7.45 | 24.84 | 3.33x |
| 5 | 9 | The author of Pride and Prejudice was | ` Jane` | 10.29 | 33.83 | 3.29x |
| 6 | 10 | In 1969 the first humans landed on the | ` moon` | 11.80 | 38.01 | 3.22x |
| 7 | 9 | The speed of light in a vacuum is approximately | ` ` | 9.98 | 33.51 | 3.36x |
| 8 | 10 | The mitochondria is often called the powerhouse of the | ` cell` | 10.64 | 35.99 | 3.38x |
| 9 | 23 | `def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(` | ` n` | 20.03 | 63.53 | 3.17x |
| 10 | 14 | `SELECT name, COUNT(*) FROM users WHERE active = 1 GROUP BY` | ` name` | 12.86 | 44.54 | 3.46x |
| 11 | 13 | `public static void main(String[] args) { System.out.println(` | `“` | 11.46 | 40.45 | 3.53x |
| 12 | 22 | `for (int i = 0; i < n; i++) { total += arr[i]; } return` | ` total` | 16.71 | 59.12 | 3.54x |
| 13 | 12 | `git commit -m "fix: handle null pointer in the` | ` user` | 12.02 | 41.40 | 3.44x |
| 14 | 10 | The derivative of x squared with respect to x is | ` ` | 10.51 | 35.55 | 3.38x |
| 15 | 13 | If 3x plus 7 equals 22 then x equals | ` ` | 11.77 | 41.29 | 3.51x |
| 16 | 10 | The sum of the interior angles of a triangle is | ` ` | 10.26 | 35.13 | 3.42x |
| 17 | 6 | Seven multiplied by eight equals fifty | `-six` | 7.38 | 24.46 | 3.31x |
| 18 | 12 | yes yes yes yes yes yes yes yes yes yes yes yes | ` yes` | 9.45 | 33.70 | 3.57x |
| 19 | 12 | the the the the the the the the the the the the | ` the` | 9.45 | 33.54 | 3.55x |
| 20 | 11 | one two three one two three one two three one two | ` three` | 9.80 | 32.59 | 3.33x |
| 21 | 24 | a a a a b b b b a a a a b b b b a a a a b b b b | ` a` | 15.30 | 54.28 | 3.55x |
| 22 | 12 | Paris is the capital of France and Berlin is the capital of | ` Germany` | 11.37 | 39.36 | 3.46x |
| 23 | 12 | Paris is the capital of France and Rome is the capital of | ` Italy` | 11.38 | 39.38 | 3.46x |
| 24 | 16 | Paris is the capital of France, a city on the river Seine that has long | ` been` | 14.29 | 49.22 | 3.44x |
| 25 | 16 | Paris is the capital of France, a city that sits far inland away from any | ` ocean` | 14.88 | 50.91 | 3.42x |
| 26 | 17 | She opened the door slowly, listening for any sound in the hallway beyond, but heard | ` nothing` | 15.35 | 52.41 | 3.41x |
| 27 | 14 | The committee reviewed the proposal for three hours before deciding that further work was | ` needed` | 14.23 | 47.28 | 3.32x |
| 28 | 16 | Despite the heavy rain that had fallen throughout the night, the river remained below its | ` banks` | 14.91 | 51.13 | 3.43x |
| 29 | 15 | Machine learning models trained on large corpora of text tend to exhibit behaviour that | ` is` | 14.52 | 48.95 | 3.37x |
| 30 | 15 | The quick brown fox jumps over the lazy dog while the cat watches from the | ` window` | 13.56 | 47.02 | 3.47x |
| 31 | 17 | Once upon a time in a village at the foot of a great mountain there lived a | ` young` | 14.50 | 51.05 | 3.52x |
| 32 | 7 | La capitale de la France est | ` Paris` | 7.94 | 26.69 | 3.36x |
| 33 | 7 | Die Hauptstadt von Frankreich ist | ` Paris` | 8.25 | 27.78 | 3.37x |
| 34 | 7 | El idioma oficial de Mexico es | ` el` | 8.30 | 28.01 | 3.37x |

```
token agreement            34 / 34
per-prompt ratio           3.17x to 3.57x
total wall  clover-k3  394.94 s      engine  1344.23 s      3.40x
```

**The lookahead is measured here too, and gated per prompt.** Reading the next
layer's experts a layer early is worth **51.03 s over the 34 prompts, 11.4%**
(445.97 s without it, 394.94 s with). Every prompt was run both ways and the
logits md5 compared: **0 of 34 differed**, so the speedup changes no answer.

**About the blank answers.** Prompts 4, 7, 14, 15 and 16 emit token **220**,
which is a single space. That is a real answer, not a failure: the model is
about to write a number, and the space comes first. Both programs emit it, and
both emit exactly token 220.

**Why longer prompts take longer.** Each extra word can pull in more distinct
experts, and every expert has to be read from disk. It is an I/O cost, not a
thinking cost.

---

## 4. Whole sentences, not just one word

One token is the unit the comparison above uses, but the program can generate.
`clover-k3.c` has no loop of its own; [`gen.py`](gen.py) drives it, and
`K3_PFXOUT` lets each step hand the next one its carried state — the attention
cache, the recurrent state and the convolution history — so only the new
position is computed.

```
In 1969 the first humans landed on the
  ->  moon. The Apollo 11 mission was a monumental achievement in human history
```

```
   1   14.94 s   10 positions   ' moon'          <- reading the prompt
   2    6.60 s    1 position    '.'
   3    6.64 s    1 position    ' The'
   4    6.64 s    1 position    ' Apollo'
   ...
  14    6.65 s    1 position    ' history'

  14 tokens in 101.3 s   7.23 s per token including the prompt
```

**The state reuse is checked against full recompute, not assumed.** Generating
the same six tokens both ways gives the identical token sequence:

| | 6 tokens | per token |
|---|---|---|
| recompute every position each step | 83.2 s | 13.87 s |
| reuse the prefix state | **35.6 s** | **5.93 s** |

```
tokens identical : YES        speedup : 2.34x
```

Steady state once the prompt has been read is **5.1 s per token** at a 5-token
prompt and **6.6 s** at a 10-token one. For reference the engine, generating on
this same machine, measured 23.71 s per token over 8 tokens and 12.42 s over
128.

An independent check fell out of this: the second token of
`The capital of France is` is **20829**, exactly the token
`k3-analysis/k3-model-equation.md` §6 records the engine emitting on its decode
step.

**None of this is interactive.** Six seconds a word is not a slow version of
usable; a 128-word answer is around thirteen minutes. What it suits is work
where the answer matters more than the wait.

---

## 5. Where the time goes

Both programs read **exactly the same 99.72 GB of expert weights** for prompt 1.
That is the anchor that makes the comparison meaningful — neither is doing less
arithmetic than the other.

From the engine's own log for prompt 1:

```
PEAK RSS for the whole run: 51.09 GB
read from disk: 99.72 GB in 8.41 s (11860 MB/s)      <- experts
read 54.47 GB in 5.73 s (9500 MB/s)                  <- trunk
I/O share of wall clock: 72.7%
```

clover-k3 on the same prompt: the same 99.72 GB of experts, and peak RSS
58.05 GB.

The gap on prompt 1 is 23.60 - 7.23 = **16.37 s**, from three places and only
two of them are cleverness:

1. **The engine re-reads the 54.47 GB trunk from disk on every run** — 5.73 s,
   **35% of the gap**. clover-k3 keeps it mapped and warm in page cache, paying
   that once per machine rather than once per prompt. A configuration choice,
   not an algorithmic win.
2. **Overlap.** The engine spends 72.7% of its wall clock on I/O, so reading and
   arithmetic are substantially serialised. clover-k3 issues expert reads a
   layer ahead and its device measures 98-99% busy.
3. **The lookahead itself**, worth 1.50 s on this prompt and 11.4% across all 34.

Both use comparable memory: 51.09 GB against 58.05 GB. Neither is a low-memory
configuration. The engine *can* run in far less RAM — that is what it is designed
for — and it was not asked to here.

---

## 6. How the runs were done

| | |
|---|---|
| machine | Ryzen 9 7950X3D, 16 cores, 124 GB RAM, NVMe RAID1 |
| weights | the released Kimi K3 checkpoint, ~1.5 TB, identical for both |
| engine campaign | 2026-09-27 |
| clover-k3 campaign | 2026-09-29, same box, same checkpoint |
| prompts | `prompts.tsv` in this folder, token ids included so they can be replayed exactly |
| what is timed | whole process, cold start to emitted token |

The two campaigns are **two days apart rather than in one session**, which is
the weakest part of the setup. Nothing on the box is known to have changed
between them, and the engine numbers were not re-run.

### Reproducing it

```sh
cd clover-k3
./build.sh          # NPOS=5 by default; a build is for one prompt length
./gate.sh           # must print md5 23d162dcefb18211a7540ef12948f1eb
```

Every prompt in `prompts.tsv` carries its token ids, so any row can be replayed
with `K3_IDS=...` and a build at the matching `NPOS`.
[`proof-campaign.sh`](proof-campaign.sh) runs the whole table;
[`gen.py`](gen.py) drives generation.

---

## 7. What this does not show

Stated plainly, because a proof that overclaims is not a proof.

- **The 34-prompt table is one token per prompt.** Section 4 generates
  sentences, but the head-to-head comparison does not.
- **Generation is not compared head to head.** clover-k3's 6.6 s per token and
  the engine's 12.42 s were measured on different prompts at different lengths.
  Treat them as two separate observations, not as a ratio.
- **This is not a comparison against the official Kimi runtime**, or against a
  GPU, or vLLM. It is one CPU C implementation against another CPU C
  implementation of the same checkpoint.
- **34 prompts, English-dominant, mostly short.** The longest is 24 tokens.
  Nothing here speaks to long context.
- **The trunk caching difference is not controlled for.** Section 5 quantifies
  it at 5.73 s of the engine's 23.60 s; a version of this test with both
  programs cold, or both warm, has not been run.
- **Agreement is on the emitted token, not on every intermediate value** —
  except on prompt 1, where clover-k3 is checked float-for-float against a
  preserved baseline, md5 `23d162dcefb18211a7540ef12948f1eb`, and that gate has
  held through every optimisation. Across all 34, the lookahead arm is checked
  against the no-lookahead arm at full float precision.
- **Single runs, not repeated trials.** Each cell is one measurement. Run-to-run
  spread on this machine is about 0.1 s within a session and up to 2 s across
  sessions, which is small against a 3.4x gap but not zero.
- **One machine.** Every number here comes from the same box.

---

## 8. The one-line version

Two independently written programs, the same 1.5 TB of weights, the same 34
prompts, the same machine: **the same 34 answers, and clover-k3 got there in
under a third of the time** — roughly 35% of the gap from holding the trunk in
memory instead of re-reading it, and the rest from overlapping the reading with
the arithmetic.
