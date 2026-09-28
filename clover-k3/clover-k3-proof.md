# Does it actually give the same answer?

Two completely different programs were given the same 34 prompts and the same
1.5 TB of Kimi K3 weights, on the same machine, on the same day.

**All 34 answers were identical. `clover-k3` took 2.95x less time.**

This page is the evidence. No arithmetic below is estimated; every number comes
from a recorded run, and section 6 lists the things this does *not* show.

---

## 1. What "the answer" means here

Both programs do one thing: read the prompt, and emit **the single next token**
the model considers most likely. Not a sentence — one token.

```
   "The capital of France is"   ->   token 17374   ->   " Paris"
```

A token is a piece of text, usually a word or part of one. The programs produce
a list of 163,840 scores, one per possible token, and the answer is the highest.
That number, 17374, is what has to match.

The two programs being compared:

| | what it is |
|---|---|
| **clover-k3** | one C file, ~2,400 lines, written from the model's equation. `clover-k3.c` in this folder. |
| **the engine** | [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c), a separate, independent C implementation of the same model. |

They were written from different starting points and share no code. If they
agree on all 34 prompts, that is two independent programs reaching the same
answer — not one program agreeing with itself.

---

## 2. Ten prompts anyone can check

| prompt | answer | clover-k3 | the engine |
|---|---|---|---|
| The capital of France is | ` Paris` | **8.81 s** | 23.60 s |
| The chemical symbol for gold is | ` Au` | **9.53 s** | 25.55 s |
| The largest planet in our solar system is | ` Jupiter` | **10.91 s** | 30.13 s |
| The author of Pride and Prejudice was | ` Jane` | **11.98 s** | 33.83 s |
| In 1969 the first humans landed on the | ` moon` | **13.62 s** | 38.01 s |
| The mitochondria is often called the powerhouse of the | ` cell` | **12.43 s** | 35.99 s |
| Paris is the capital of France and Berlin is the capital of | ` Germany` | **13.23 s** | 39.36 s |
| Paris is the capital of France and Rome is the capital of | ` Italy` | **13.14 s** | 39.38 s |
| La capitale de la France est | ` Paris` | **9.53 s** | 26.69 s |
| Die Hauptstadt von Frankreich ist | ` Paris` | **9.93 s** | 27.78 s |

Same answer every time. Between two and three times faster every time.

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

`s` is wall-clock seconds for the whole run, cold start to emitted token.

| # | len | prompt | answer | clover-k3 | engine | ratio |
|---|---|---|---|---|---|---|
| 1 | 5 | The capital of France is | ` Paris` | 8.81 | 23.60 | 2.68x |
| 2 | 6 | The chemical symbol for gold is | ` Au` | 9.53 | 25.55 | 2.68x |
| 3 | 8 | The largest planet in our solar system is | ` Jupiter` | 10.91 | 30.13 | 2.76x |
| 4 | 6 | Water boils at a temperature of | ` ` | 8.96 | 24.84 | 2.77x |
| 5 | 9 | The author of Pride and Prejudice was | ` Jane` | 11.98 | 33.83 | 2.82x |
| 6 | 10 | In 1969 the first humans landed on the | ` moon` | 13.62 | 38.01 | 2.79x |
| 7 | 9 | The speed of light in a vacuum is approximately | ` ` | 11.55 | 33.51 | 2.90x |
| 8 | 10 | The mitochondria is often called the powerhouse of the | ` cell` | 12.43 | 35.99 | 2.90x |
| 9 | 23 | `def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(` | ` n` | 22.10 | 63.53 | 2.87x |
| 10 | 14 | `SELECT name, COUNT(*) FROM users WHERE active = 1 GROUP BY` | ` name` | 14.82 | 44.54 | 3.01x |
| 11 | 13 | `public static void main(String[] args) { System.out.println(` | `“` | 13.34 | 40.45 | 3.03x |
| 12 | 22 | `for (int i = 0; i < n; i++) { total += arr[i]; } return` | ` total` | 18.82 | 59.12 | 3.14x |
| 13 | 12 | `git commit -m "fix: handle null pointer in the` | ` user` | 13.91 | 41.40 | 2.98x |
| 14 | 10 | The derivative of x squared with respect to x is | ` ` | 12.14 | 35.55 | 2.93x |
| 15 | 13 | If 3x plus 7 equals 22 then x equals | ` ` | 13.58 | 41.29 | 3.04x |
| 16 | 10 | The sum of the interior angles of a triangle is | ` ` | 11.88 | 35.13 | 2.96x |
| 17 | 6 | Seven multiplied by eight equals fifty | `-six` | 8.84 | 24.46 | 2.77x |
| 18 | 12 | yes yes yes yes yes yes yes yes yes yes yes yes | ` yes` | 10.87 | 33.70 | 3.10x |
| 19 | 12 | the the the the the the the the the the the the | ` the` | 10.83 | 33.54 | 3.10x |
| 20 | 11 | one two three one two three one two three one two | ` three` | 10.74 | 32.59 | 3.03x |
| 21 | 24 | a a a a b b b b a a a a b b b b a a a a b b b b | ` a` | 16.61 | 54.28 | 3.27x |
| 22 | 12 | Paris is the capital of France and Berlin is the capital of | ` Germany` | 13.23 | 39.36 | 2.98x |
| 23 | 12 | Paris is the capital of France and Rome is the capital of | ` Italy` | 13.14 | 39.38 | 3.00x |
| 24 | 16 | Paris is the capital of France, a city on the river Seine that has long | ` been` | 16.39 | 49.22 | 3.00x |
| 25 | 16 | Paris is the capital of France, a city that sits far inland away from any | ` ocean` | 17.10 | 50.91 | 2.98x |
| 26 | 17 | She opened the door slowly, listening for any sound in the hallway beyond, but heard | ` nothing` | 17.62 | 52.41 | 2.97x |
| 27 | 14 | The committee reviewed the proposal for three hours before deciding that further work was | ` needed` | 16.40 | 47.28 | 2.88x |
| 28 | 16 | Despite the heavy rain that had fallen throughout the night, the river remained below its | ` banks` | 17.23 | 51.13 | 2.97x |
| 29 | 15 | Machine learning models trained on large corpora of text tend to exhibit behaviour that | ` is` | 16.72 | 48.95 | 2.93x |
| 30 | 15 | The quick brown fox jumps over the lazy dog while the cat watches from the | ` window` | 15.60 | 47.02 | 3.01x |
| 31 | 17 | Once upon a time in a village at the foot of a great mountain there lived a | ` young` | 16.78 | 51.05 | 3.04x |
| 32 | 7 | La capitale de la France est | ` Paris` | 9.53 | 26.69 | 2.80x |
| 33 | 7 | Die Hauptstadt von Frankreich ist | ` Paris` | 9.93 | 27.78 | 2.80x |
| 34 | 7 | El idioma oficial de Mexico es | ` el` | 10.00 | 28.01 | 2.80x |

```
token agreement   34 / 34
total wall        clover-k3  455.94 s      engine  1344.23 s      2.95x
```

**About the blank answers.** Prompts 4, 7, 14, 15 and 16 emit token **220**,
which is a single space. That is a real answer, not a failure: the model is
about to write a number, and the space comes first. Both programs emit it, and
both emit exactly token 220.

**Why longer prompts take longer.** Each extra word can pull in more distinct
experts, and every expert has to be read from disk. It is an I/O cost, not a
thinking cost.

---

## 4. Where the time actually goes

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

clover-k3 on the same prompt: the same 99.72 GB of experts, and peak RSS 56.8 GB
(measured on the current build).

The gap on prompt 1 is 23.60 - 8.81 = **14.79 s**, and it comes from two places,
only one of which is cleverness:

1. **The engine re-reads the 54.47 GB trunk from disk on every run** — 5.73 s,
   which is **39% of the gap**. clover-k3 keeps the trunk mapped and warm in
   page cache, paying that once per machine rather than once per prompt. This
   is a configuration choice, not an algorithmic win.
2. **The rest, about 9 s, is overlap.** The engine spends 72.7% of its wall
   clock on I/O, so its reading and its arithmetic are substantially serialised.
   clover-k3 issues expert reads a layer ahead; on the current build its device
   measures 98-99% busy for the body of the run. That utilisation figure is from
   the current build, not from the 8.81 s build in the table — no equivalent
   measurement was taken during the campaign.

Both use comparable memory: 51.09 GB against 56.8 GB. Neither is a low-memory
configuration. The engine *can* run in far less RAM — that is what it is designed
for — and it was not asked to here.

---

## 5. How the runs were done

| | |
|---|---|
| machine | Ryzen 9 7950X3D, 16 cores, 124 GB RAM, NVMe RAID1 |
| weights | the released Kimi K3 checkpoint, ~1.5 TB, identical for both |
| clover-k3 campaign | 2026-09-27 20:25 |
| engine campaign | 2026-09-27 23:47, same box, ~3 hours later |
| prompts | `prompts.tsv` in this folder, token ids included so they can be replayed exactly |
| what is timed | whole process, cold start to emitted token |

Three things that make these numbers **conservative rather than flattering**:

- The clover-k3 runs had provenance instrumentation switched on (`K3_PROV`),
  which the engine runs did not have an equivalent of.
- clover-k3 has since got faster. Prompt 1 now runs in **7.25 s**, not 8.81 s,
  which would make the ratio 3.26x. The table keeps the contemporaneous number.
- The engine was given its own preferred configuration and was not tuned down.

### Reproducing it

```sh
cd clover-k3
./build.sh          # NPOS=5 by default; a build is for one prompt length
./gate.sh           # must print md5 23d162dcefb18211a7540ef12948f1eb
```

Every prompt in `prompts.tsv` carries its token ids, so any row can be replayed
with `K3_IDS=...` and a build at the matching `NPOS`.

---

## 6. What this does not show

Stated plainly, because a proof that overclaims is not a proof.

- **One token, not a sentence.** Both programs were asked for the next token
  only. No generation loop, no conversation, no long output was compared.
- **This is not a comparison against the official Kimi runtime**, or against a
  GPU, or against vLLM. It is one CPU C implementation against another CPU C
  implementation of the same checkpoint.
- **34 prompts, English-dominant, mostly short.** The longest is 24 tokens.
  Nothing here speaks to long context.
- **The trunk caching difference is not controlled for.** Section 4 quantifies
  it at 5.73 s of the engine's 23.60 s; a version of this test with both
  programs cold, or both warm, has not been run.
- **Agreement is on the emitted token, not on every intermediate value.** The
  two programs agree on the answer; a float-level comparison across all 34
  prompts has not been done. (clover-k3 *is* checked float-for-float against a
  preserved baseline on prompt 1 — md5 `23d162dcefb18211a7540ef12948f1eb` — and
  that gate has held through every optimisation.)
- **Single runs, not repeated trials.** Each cell is one measurement. Run-to-run
  spread on this machine is about 0.1 s within a session and up to 2 s across
  sessions, which is small against a 2.7-3.3x gap but not zero.

---

## 7. The one-line version

Two independently written programs, the same 1.5 TB of weights, the same 34
prompts, the same machine: **the same 34 answers, and clover-k3 got there in a
third of the time** — roughly 39% of the gap from holding the trunk in memory
instead of re-reading it, and the rest from overlapping the reading with the
arithmetic.
