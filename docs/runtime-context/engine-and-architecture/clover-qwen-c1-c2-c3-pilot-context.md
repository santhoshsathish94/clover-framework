# Clover Qwen C1/C2/C3 pilot on k3: does the plan route work in the one family where every part is trained?

Started 2026-11-18 by the AI session, on the human's direction ("use k3 and do the pilot tests ... use a qwen model to see if our plan
really works with both hybridagent, ncm and base model ... do A and B one after the other ... download the Qwen for B while you run A ...
record all findings to files, create a branch, push it"). Branch `research/qwen-c1-c2-c3-pilot-20261118`, worktree
`C:\personal\oss\clover-qwen-pilot-wt`. Code and raw results: `research/qwen-c1-c2-c3-pilot/`. k3 working folder (not in git, holds states
and checkpoints): `/opt/clover-k3/qwen-pilot-20261118`. Nothing was pushed to Hugging Face, nothing changed on k3 outside that folder
and the Hugging Face cache (Qwen3-8B-Base was added, 16 GB). States: **known** (measured, file named), **inferred**, **unknown**.

STATUS: in progress. Sections marked PENDING are filled in as the chain on k3 finishes.

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

## 4. PENDING - A2 fine-tune (plan -> answer window), A3 (HybridAgent-8B text plan -> C3), B0-B3

Training: 700 steps, batch 16, lr 1e-4 (cosine), bf16 autocast, tokenizer frozen, 3,740 train pairs, ~6-8 s per step on k3 CPU
(fp32 8 s; two earlier longer schedules were aborted for time and their logs are kept in `logs/`).

## 5. Limits stated up front

- CPU-only pilot: small schedules (hundreds of steps, a few thousand pairs) against the paper's 150,000 steps at batch 2,048.
- Dolly human answers are short and often list-like; plan = grounded plan is an oracle plan, not a model-written plan.
- HybridAgent-8B in text mode opens with chain-of-thought ("Okay, so the user is asking ..."), so its first 60 tokens are a thinking
  plan, not an answer; prefilling an answer cue made one output empty and was dropped.
- No human rating; scores are keyword retention and exact number recall plus reading samples.
