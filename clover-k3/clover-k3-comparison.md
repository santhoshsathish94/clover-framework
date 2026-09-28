# The same 34 prompts through two independent implementations

`clover-k3` against [`kimi-k3-in-c`](https://github.com/FareedKhan-dev/kimi-k3-in-c),
a separate C implementation of the same model sharing no code with this one.
Same machine, same 1.5 TB checkpoint, same prompts, one token each.

**All 34 answers identical. 3.40x less wall time.**

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

Three worth pausing on. **The same question in three languages** — English,
French and German all emit token **17374**, the identical id and not merely
the same idea. **An analogy rather than a recall** — `...and Berlin is the
capital of` gives ` Germany`, and swapping Berlin for Rome gives ` Italy` from
an otherwise identical sentence. **Two prompts sharing eleven words** that
diverge to ` been` and ` ocean`, with both programs agreeing on both.

Prompts 4, 7, 14, 15 and 16 answer with token **220**, a single space. That is
a real answer — the model is about to write a number and the space comes
first — and both programs emit exactly that token.

## The lookahead, gated per prompt

Reading the next layer's experts a layer early is worth **51.03 s over the 34
prompts, 11.4%** (445.97 s without, 394.94 s with). Every prompt ran both ways
and the logits md5 of the two arms was compared: **0 of 34 differed**, so the
speedup changes no answer.

## Where the time goes

Both programs read **exactly the same 99.72 GB of expert weights** on prompt 1,
so neither is doing less arithmetic. From the engine's own log:

```
PEAK RSS for the whole run: 51.09 GB
read from disk: 99.72 GB in 8.41 s (11860 MB/s)      <- experts
read 54.47 GB in 5.73 s (9500 MB/s)                  <- trunk
I/O share of wall clock: 72.7%
```

clover-k3 on the same prompt: the same 99.72 GB, peak RSS 58.05 GB. The gap is
23.60 - 7.23 = **16.37 s**, and only part of it is cleverness:

1. **The engine re-reads the 54.47 GB trunk every run** — 5.73 s, **35% of the
   gap**. clover-k3 keeps it mapped and warm. A configuration choice.
2. **Overlap.** The engine spends 72.7% of wall on I/O; clover-k3 issues expert
   reads a layer ahead and its device measures 98-99% busy.
3. **The lookahead**, 1.50 s on this prompt.

## What this does not show

- **One token per prompt.** Whole-sentence output is in
  [`clover-k3-proof.md`](clover-k3-proof.md), and generation was not compared
  head to head.
- **Not the official Kimi runtime, not a GPU, not vLLM.** One CPU C
  implementation against another.
- **34 prompts, English-dominant, longest 24 tokens.** Nothing about long
  context.
- **The trunk caching difference is quantified but not controlled for.** A run
  with both programs cold, or both warm, has not been done.
- **Two campaigns two days apart** — the engine on 2026-09-27, clover-k3 on
  2026-09-29, same box. Nothing is known to have changed between them, and the
  engine was not re-run.
- **Agreement is on the emitted token, not every intermediate value**, except
  prompt 1 where clover-k3 is checked float-for-float against a preserved
  baseline, md5 `23d162dcefb18211a7540ef12948f1eb`.
- **Single runs.** Spread on this machine is about 0.1 s within a session and
  up to 2 s across sessions.
