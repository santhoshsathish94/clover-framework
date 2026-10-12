# Simplification experiments

Can the computation be reduced without losing the answer? Each method below was tried
against the same harness and the same prompts, and each has its own context file saying
what it did, what held, and what did not.

The analysis in [../OBSERVED-ARCHITECTURE.md](../OBSERVED-ARCHITECTURE.md) suggested
these candidates, and nothing here was chosen because it sounded plausible: every method
points at something that was measured first.

| method | motivated by | result | file |
|---|---|---|---|
| A, drop block snapshots | the tail gives seven of its nine sources about 5% of the weight between them | **inconclusive** — the ablation hook collapses positions rather than swapping prompts, and the no-op control caught it | [METHOD-A-SNAPSHOTS.md](METHOD-A-SNAPSHOTS.md) |
| B, cache pod 12's attention | it is the only layer of 92 whose attention output is near a fixed direction | **held** — 12 of 12 tokens identical, worst pod-92 cosine 0.996102 | [METHOD-B-CONST-ATTENTION.md](METHOD-B-CONST-ATTENTION.md) |
| C, the same at every layer | the single-token result in B did not fit the reason B was tried | **held per layer, 68 of 68**, and up to **32 of 68 together** when chosen by measured tolerance; all 68 at once collapses. Also shows B's success was not caused by what motivated it | [METHOD-C-ALL-LAYERS.md](METHOD-C-ALL-LAYERS.md) |

## The harness

`fidelity.sh` runs a prompt set through the production fleet twice, once as a baseline
and once with a method enabled, and compares two things:

- **the generated token**, which is what actually matters, and
- **the cosine of pod 92's output against the baseline**, which is continuous and far
  more sensitive than token identity. A method can leave the token alone while moving
  the representation a long way, and that distinction should be visible rather than
  hidden.

Only owner 92 is traced, so a run costs about 17 MB instead of 1.5 GB.

## What this harness does not establish

**Matching the greedy top-1 token on a handful of short prompts is weak evidence of
fidelity.** It is the signal available here, not a substitute for an evaluation. A
method that keeps every token in this set may still be worse on long generations,
on reasoning, or anywhere the margin between the top two candidates is thin — and
earlier measurement showed that margin is thin exactly where context is short, 0.34 at
position 0 against 3.68 at position 4.

Any method recorded here as "held" means: held on this prompt set, by these two
measures. Deciding whether to keep a simplification is a model-quality decision with a
real evaluation behind it, and that evaluation has not been run.
