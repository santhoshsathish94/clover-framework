# Clover Qwen C1/C2/C3 pilot on k3: does the plan route work in the one family where every part is trained?

Started 2026-11-18 by the AI session, on the human's direction ("use k3 and do the pilot tests ... use a qwen model to see if our plan
really works with both hybridagent, ncm and base model ... do A and B one after the other ... download the Qwen for B while you run A ...
record all findings to files, create a branch, push it"). Branch `research/qwen-c1-c2-c3-pilot-20261118`, worktree
`C:\personal\oss\clover-qwen-pilot-wt`. Code and raw results: `research/qwen-c1-c2-c3-pilot/`. k3 working folder (not in git, holds states
and checkpoints): `/opt/clover-k3/qwen-pilot-20261118`. Nothing was pushed to Hugging Face, nothing changed on k3 outside that folder
and the Hugging Face cache (Qwen3-8B-Base was added, 16 GB). States: **known** (measured, file named), **inferred**, **unknown**.

STATUS: all planned runs finished 2026-11-19 03:34 CET (A1, A2, A3, B0-B3, B4). Single runs, small schedules; see section 10.

## 0. The short answer

Does the plan route work in the one family where every part is trained? **Partly, and the part that works is the writer.**
- Works (known): link reads held-out states (0.31-0.37 top-1 vs controls <= 0.02); a linear bridge preserves that readout when the two
  spaces are close (cos 0.978); the released NCM, fine-tuned 700 steps on CPU, renders a plan into fluent text keeping 76% of the plan
  words and 79% of numbers (wrong or blank plan: about 1%).
- Does not work (known): the link as a latent step on a FROZEN backbone (collapses to repetition); C3 alone cannot supply correctness
  (word substitutions, duplicates, lost relations); HybridAgent-8B's own text plan is chain-of-thought, which C3 renders as such.
- Therefore for a frozen Clover model: C3 (plan in the prefix) is worth building per model; the plan writer cannot be the link on the
  model's states unless the backbone is trained; the nearest working choices are the model's own text or HybridAgent-8B as writer.

Next experiments this suggests (not started, need the human's direction): (1) Clover-35B C3: same A2 recipe with Clover's vocabulary
and a VQ-VAE adaptation; (2) the copy-pointer decoder on top of A2 to fix substituted words; (3) a plan writer that answers directly
(HybridAgent-8B with its latent mode and anchors through its real engine) rather than thinking in text.

## 1. Question and design

Last session's findings (see `clover-ncm-hybridagent-training-readiness-context.md`) left one question open: do C1 state -> link (C2) ->
plan -> NCM (C3) work at all, or is only our Clover-specific training too small? The Qwen family is the one place where each part
exists at the paper's training scale (HybridAgent-8B and its link: togethercomputer; NCM tokenizer built on the Qwen3-0.6B vocabulary).

- **A (matched, as in the paper):** C1 = C2 = HybridAgent-8B (Qwen3-8B post-trained), C3 = released WMT14 NCM.
- **B (split, like Clover):** C1 = frozen Qwen3-8B-Base; a link trained on its states; a bridge into HybridAgent space; same C3.

Data: databricks-dolly-15k (CC-BY-SA-3.0), no-context rows, unique instructions, split by sha1 of the instruction (held-out = hash%5==0).
Hardware: k3 (Ryzen 9 7950X3D, 16 cores / 32 threads, 124 GB, CPU only, shared; jobs ran as `systemd-run` units, Nice=15, memory capped,
no other service touched). Measured: bf16 forward of the 8B models 101 tokens/s.

## 2. A1 - positive control: the paper-trained HybridLink on held-out text (known: `results/a1.json`)

HybridAgent-8B's own `adapter.pt` (ln_res_adapter: pre_ln, proj1, GELU, proj2, residual, post_ln), applied to its own last-layer state
(HF `hidden_states[-1]`, what the engine feeds the link). Readout = nearest input-embedding row by cosine (the repo's `readout.py` rule),
target = the next response token. 120 held-out prompts, 5,856 positions; controls on 1,500 sampled positions.

| Readout of the next token | top-1 | top-5 | median rank |
|---|---|---|---|
| paper link `R(h)` (all 5,856 positions) | 0.347 | 0.507 | 4 |
| model's own `lm_head` (reference) | 0.592 | 0.815 | |
| raw `h`, no link | 0.0007 | 0.003 | 50,078 |
| zeros | 0.000 | 0.001 | |
| random state of matched norm, through the link | 0.011 | 0.047 | 3,651 |
| shuffled link outputs | 0.019 | 0.046 | 2,581 |

Reading (inferred): the instrument separates a paper-trained link from every control by more than 17x on top-1. The paper's latent
steps need not decode to the next TEXT token, so 0.347 is not a ceiling for what the link carries. It is the number to compare with
last session's Clover-35B link (0.381 top-1 held-out): same instrument, different model, text and training, so close but not a
like-for-like comparison (the Clover run read the model's own answers; this one reads human Dolly answers).

## 3. A2 - C3 on the Qwen vocabulary (known: `results/a2_recon.json`, `results/a2_eval_zeroshot.json`)

Target windows: Dolly responses of 12-63 Qwen tokens (64-token windows, EOS then PAD). Plan = grounded plan: the response's content
words (>=4 letters, small stop list) and numbers in order of first appearance, at most 30 items (as in Cycle 48 of the HybridAgent
research context: a plan built from the target, carrying facts but not sentences).

- **Tokenizer gate (released WMT14 VQ-VAE, no training), 200 held-out windows:** token accuracy 0.9286, exact windows 0.415, number
  recall 0.9798. Below the 0.95 token gate I used for the Kimi adaptation, but numbers survive. Errors are word substitutions, e.g.
  "Interstellar" -> "Intermedi".
- **Zero-shot (the released translation generator, plan in the 96-token prefix), 64 held-out windows:**

| sampling | plan retention, matched | shuffled plan | blank plan | true text (ceiling) | number recall, matched / shuffled / blank |
|---|---|---|---|---|---|
| temp 1.0 | 0.228 | 0.006 | 0.000 | 1.0 | 0.348 / 0.033 / 0.000 |
| temp 0.6, top-k 50, top-p 0.9 | 0.387 | 0.005 | 0.000 | 1.0 | 0.660 / 0.044 / 0.000 |

Read samples (known): the released translation model copies the plan's words ("Sasando string Ranat percussion" -> "Sasando string
Ranat percussion."); it does not write sentences, and it garbles some tokens ("Oceans" -> "Oporto"). That it follows a plan prefix
with matched >> shuffled/blank without any training on plans is the finding: the prefix path of the released NCM carries content.

## 4. A2 fine-tune: plan in the prefix -> answer window (known: `results/a2_eval_trained.json`, `runs/a2/log.jsonl` on k3)

Training: 700 steps, batch 16, lr 1e-4 (cosine), bf16 autocast, 16 threads, tokenizer frozen, 3,740 train pairs, 64 held-out windows for
validation, ~6-8 s per step on k3 CPU (fp32 8 s; two earlier longer schedules were aborted for time, logs kept in `logs/`).
Validation cross-entropy (mean over the 11 levels), with the correct plan / another window's plan / blank plan:

| step | matched | shuffled | blank |
|---|---|---|---|
| 0 (released translation generator) | 2.842 | 3.313 | 4.889 |
| 175 | 2.309 | 3.636 | 4.221 |
| 350 | 2.252 | 3.692 | 4.266 |
| 525 | 2.238 | 3.737 | 4.289 |
| 700 | 2.224 | 3.729 | 4.309 |

The gap between a right and a wrong plan grew from 0.47 to 1.51 nats per code: the generator learned to depend on the plan.

Held-out generation, 64 windows (grounded plans; retention = share of the plan's words/numbers present in the output):

| sampling | retention matched | shuffled plan | blank plan | number recall matched / shuffled / blank |
|---|---|---|---|---|
| temp 1.0 | 0.649 | 0.008 | 0.003 | 0.770 / 0.056 / 0.067 |
| temp 0.6, top-k 50, top-p 0.9 | 0.760 | 0.011 | 0.001 | 0.787 / 0.061 / 0.000 |

Zero-shot (before training) was 0.228 / 0.387 retention. Read samples (known): the fine-tuned C3 now writes sentences around the
plan: "countries Northern Hemisphere include Brazil China India Canada Southern Australia South Africa Argentina Fiji" ->
"The Northern Alliance include Brazil, China, India, Canada, Australia, Australia Australia, South Africa, Argentina, and Fiji."
It also makes the artifacts the NCM paper lists (word substitutions "Oceans" -> "Oimbabwe", repeated words, a lost meaning: "Southern
Hemisphere" came out as "Australia" only). Shuffled and blank plans give unrelated or garbage text, so the content really comes from
the plan. The test shows C3 renders a plan into fluent text; it does not show the text is correct or that the plan was right.

## 5. A3 - end to end in the matched family (known: `results/a3_trained.json`, `a3_zeroshot.json`, `a3_answers.json`)

C1 = C2 = HybridAgent-8B in text mode on 40 held-out questions (greedy, 60 tokens); plan = grounded plan of that text; C3 renders it.
Important (known): HybridAgent-8B opens with chain-of-thought ("Okay, so the user is asking for a list of fantastic movies. Hmm, I
need to figure out ..."), also with the "answer in two sentences" instruction; a prefilled "Final answer:" cue made one output empty and
was dropped. So its first 60 tokens are a thinking plan, not an answer, the same form as the Studio route.

| C3 | retention matched | shuffled plan | blank | number recall matched / shuffled / blank |
|---|---|---|---|---|
| fine-tuned (A2), temp 0.6 | 0.744 | 0.105 | 0.004 | 0.75 / 0 / 0 |
| fine-tuned (A2), temp 1.0 | 0.625 | 0.067 | 0.001 | 1.00 / 0 / 0 (few numbers in these 40) |
| released translation generator, temp 0.6 | 0.113 | 0.030 | 0.000 | 0.50 / 0 / 0 |

Reading the output (known): the render keeps the plan's words in a fluent frame but the content is the model's thinking, e.g.
"Okay user asking list fantastic movies need figure mean fantasy maybe ..." became "I am a user of user asking me for asking a list
of some great TV movies. You need to figure mean I mean fantasy". C3 faithfully renders whatever the plan says; a thinking plan
produces thinking text. The one question where HybridAgent answered directly ("The Pacific Northwest is unique due to its diverse
geography ...") rendered as a readable paraphrase with small errors ("unique and unique diverse geography ...").

## 6. B - the split design with a frozen Qwen3-8B-Base (known: `results/b_link.json`, `b3_answers.json`, `b3_render_trained.json`)

Qwen3-8B-Base (16 GB, 5 shards) was downloaded to the k3 Hugging Face cache for this. States: last-layer `hidden_states[-1]` at the
response positions of Dolly text in a plain format ("Question: ...\nAnswer: ..."), identical token sequences for both models so that
states pair position by position. Train 1,500 prompts (about 75K positions), held-out 300 prompts, exported at 133 tokens/s.
Readout rule and controls are the same as A1; readouts over the base model's own embedding table unless noted.

**B0 - the paper's link applied unchanged to the frozen base model's states (no training):**

| held-out next-token readout | top-1 | top-5 | median rank |
|---|---|---|---|
| paper link on Base states | 0.307 | 0.403 | 26 |
| zeros / random state through link / shuffled outputs / raw state | 0.000 / 0.008 / 0.015 / 0.000 | | |

**B1 - a link trained on frozen Base states** (same architecture; objective cosine + 0.2 MSE on unit-normalised vectors to the next token's
embedding, AdamW 2e-4, batch 256, 1,500 steps; the objective's normalisation is my choice, not taken from the paper):

| init | top-1 | top-5 | median rank | controls (zeros / random / shuffled) |
|---|---|---|---|---|
| random | 0.251 | 0.369 | 48 | 0.000 / 0.017 / 0.017 |
| the paper link | 0.372 | 0.539 | 3 | 0.000 / 0.017 / 0.014 |

**B2 - a bridge from Base state space to HybridAgent-8B space** (ridge regression, closed form, fitted on the train positions):
cosine between a bridged Base state and the true HybridAgent state on held-out positions 0.978 (0.881 with no bridge), relative squared
error 0.033. The paper link on the true HybridAgent states reads out at 0.373 top-1 (HA table), on bridged Base states at 0.357,
on Base states with no bridge at 0.307. Controls in all rows stay at or below 0.017.

**B3 - the frozen Base model's own answers rendered by the A2 C3** (40 held-out questions, plain completion with "Answer in at most two
short sentences", plan = grounded plan of its answer): retention 0.755 (temp 0.6) / 0.654 (temp 1.0), shuffled plan 0.010 / 0.007,
blank 0.000 / 0.002; number recall 0.917 / 0.667. The Base model answers directly (no chain of thought), e.g. "Ranat ek is a percussion
instrument, while Sasando is a string instrument." rendered as "Ranat is percussion instrument, Sasando is string string."; the
countries question rendered as "India blossom, Fiji, Brazil, Australia, Argentina, Argentina, South Africa, Northern China, India, and
Canada," (Hemisphere words dropped, duplicates); the water/land question rendered as "The river is usually used for water, water, water
..." (content lost). So the render keeps most plan words and often breaks the relations between them.

## 7. B4 - the paper's link in a latent loop (known: `results/b4_ha.json`, `results/b4_base.json`, script `b4_latent_loop.py`)

After the prompt (and the `<latent>` anchor for HybridAgent-8B), the paper link's output is fed back as the next input embedding K times,
then the model continues in text (greedy, 48 tokens, 16 held-out questions, one run). Controls: plain text, and K random vectors of the
link output's norm. This is my own minimal loop with a KV cache, not the HybridAgent engine (`hybrid_agent/engine.py`), so differences
from the engine are possible. "Degenerate" = empty, fewer than 3 words, or under 40% distinct words. Overlap = share of the reference's
content words (first 5 letters) found in the output.

| model | variant | overlap with own plain answer | overlap with human answer | degenerate | empty outputs |
|---|---|---|---|---|---|
| HybridAgent-8B (backbone trained with the link) | plain text | 1.00 | 0.27 | 0.00 | 0 |
| | latent K=4 | 0.50 | 0.23 | 0.31 | 5 |
| | random K=4 | 0.25 | 0.15 | 0.63 | 10 |
| | latent K=16 | 0.32 | 0.25 | 0.44 | 7 |
| frozen Qwen3-8B-Base (link only) | plain text | 1.00 | 0.51 | 0.00 | 0 |
| | latent K=4 | 0.08 | 0.09 | 0.81 | 0 |
| | random K=4 | 0.31 | 0.27 | 0.69 | 8 |
| | latent K=16 | 0.00 | 0.00 | 1.00 | 0 |

Read samples (known): on the frozen Base model every latent-loop output is "the the the the the ..." or "to the to the to the ..."
(all 16 at K=16), whereas the same prompts in text mode answer well ("Jet ski and boat are used on water, while car, bike, kayak, and
skateboard are used on land."). HybridAgent-8B keeps writing coherent text after 4 link steps in 11 of 16 questions (it sometimes
returns a different but fluent opening, "To solve this task, I need to categorize ..."), and does worse with random vectors; five latent
runs and ten random runs gave an empty output, a cause I did not investigate (it may be an end-of-turn token right after the exit anchor
in my loop).

Reading (inferred): with the backbone frozen, the paper's link used as a latent step damages the model more than random vectors do (0.08
vs 0.31 overlap). That matches the HybridAgent paper's own statement that link activation alone is insufficient and that the
backbone must be trained with the link (HybridSFT). For Clover this fits the constraint noted earlier: a frozen GGUF/compiled model
can carry a link that READS its states (sections 2, 6) but cannot be made to think in latent steps without training its backbone.

## 8. What this shows (inferred unless marked known)

1. **The matched Qwen family can do every stage with real data on a CPU machine (known numbers above).** The paper link reads
   next tokens from held-out HybridAgent states at 0.347-0.373 against controls <= 0.02; the released NCM, fine-tuned for 700 steps on
   3,740 plan-to-text pairs, renders plans with 0.76 retention against 0.011 for a wrong plan.
2. **The split design is viable in this family, and for a simple reason (inferred).** HybridAgent-8B's states and Qwen3-8B-Base's states
   are close (identity cosine 0.88): the paper link transfers to Base states with 82% of its own score (0.307 vs 0.373) with no
   training; training a link from the paper's init on Base states recovers 0.372; training from random initialisation on 75K positions
   gives 0.251 (the 2048-wide link that scored 0.381 on Clover-35B last session was in the same range); a linear bridge preserves the
   readout (0.357). This is the proof of concept for a bridge: it works when the two spaces are as close as a base model and its own
   post-trained derivative. It says nothing yet about a bridge between models that do not share a base (Kimi, Nemotron, LFM3).
3. **What a link reads out is the next token of text the model would write anyway (known readouts, inferred meaning).** In every row the
   target is the next response token, teacher-forced. None of these results shows a link producing a plan of the answer before the answer
   exists. In A3 the plan came from HybridAgent-8B's own text output.
4. **A frozen backbone cannot think through the link (known, section 7).** Fed back as latent steps, the paper link collapses the frozen
   Base model into repetition while HybridAgent-8B, whose backbone was trained with it, stays coherent in 11 of 16 questions.
   So the plan writer for a frozen Clover model cannot be "the link on the model's own states"; it has to be a model whose backbone was
   trained (HybridAgent-8B today) or a model's own text.
5. **C3 renders; correctness lives in the plan (known samples).** Fine-tuned C3 writes fluent sentences that contain the plan words, and
   keeps numbers (0.79 recall), but it substitutes some words, duplicates others and can drop the relations ("Southern Hemisphere" lost
   in the countries example). With a wrong or thinking-style plan it renders the wrong or thinking-style text.
6. **The earlier Clover failures read differently now (inferred).** Last session's "R/G" tests gave C3 states without plan text and its
   link had no plan-writing role; this pilot gave C3 a grounded plan in the prefix and it worked, as Cycles 48/50 predicted. What
   transfers: the C3 recipe (plan in prefix, fine-tune the released NCM) and the instrument (positive control passes). What does not
   transfer yet: a trained plan writer for a non-Qwen model.

## 9. What was not tested

- A link that writes a plan before the answer exists on a non-trained backbone (B4 tests the paper link in a latent loop on Base, not a
  trained plan writer).
- A bridge between models without a shared base; the Qwen -> Clover-35B case (Clover-35B is Qwen3.6-derived but its width is 2048 and
  it is a mixture-of-experts model: the ridge bridge here needed the same width, so it would have to be a learned map between widths).
- The copy-pointer decoder (Cycle 50) and VQ-VAE adaptation (token accuracy 0.929 on Dolly; word substitutions in the renders are the
  likely cost): not run in this pilot.
- Any quality score by a human, any answer-correctness score, speed against plain decoding (CPU-only, bf16 forward 101-133 tokens/s;
  NCM generation of 3 x 64 windows (matched, shuffled, blank) took 52-56 s on CPU at batch 16, not compared with token-by-token decoding).
- More than one run per cell; 40 questions in A3/B3, 64 windows in A2, 300 held-out prompts in B1/B2.

## 10. Limits and honest caveats

- CPU-only pilot: 700 steps at batch 16 on 3,740 pairs against the paper's 150,000 steps at batch 2,048.
- Dolly human answers are short and list-like; the grounded plan is an oracle plan built from the target text, not a model-written plan
  (in A3/B3 it is built from the model's own answer, still not a separately generated plan).
- HybridAgent-8B in text mode opens with chain-of-thought even when told to answer briefly; a prefilled answer cue made one output empty
  and was dropped.
- The link training objective (cosine + 0.2 MSE on unit-normalised vectors) is my choice; the repo's exact loss normalisation was not read.
- Held-out Dolly rows were never used for training; the split is by instruction hash, so the same questions appear in A1, A2, A3, B*.
- Clock note: k3 prints CET; Python logs of one venv printed other hours; all timings above come from the logs and are step-relative.
