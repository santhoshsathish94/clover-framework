# Hybrid Reasoning Agents (HybridAgent): research notes

Started 2026-10-21. Direction (the creator): research the paper and the official code, and create a context file for it. No use for Clover K3 was stated; the section on K3 near the end is my reading, not Direction.
This file records what was read, what it says, what the released code does, and what was not looked at. **Everything here was written from reading only. Nothing was run, installed or downloaded onto any server.**

## What was read, and how

| Source | How it was read | Not covered |
|---|---|---|
| Paper, arXiv 2610.18861 v1 (13 Oct 2026), "Hybrid Reasoning Agents", Zou, Qi, Chen, Bianchi, Zhu, Lu, He, Zou (Stanford, Together AI, UIUC, Meta). cs.CL, 45 pages, 12 figures, CC BY 4.0 | The HTML version (`arxiv.org/html/2610.18861v1`) by text extraction: abstract, sections 1 to 9, appendices A to H | Figures and plots (only captions seen); the PDF; appendix I (prompt templates) and appendix J (case studies), which the extraction did not return; some table cells and column labels were lost in extraction (noted where it matters) |
| Code, github.com/togethercomputer/HybridAgent, commit `1b41ceb` (18 Oct 2026, "Add X and LinkedIn badges, drop license badge (#4)"), MIT, Python only | Shallow clone at `c:\personal\oss\HybridAgent` (outside this repo). Read in full: `README.md`, `hybrid_agent/{hybridlink,model,engine,sampling,readout,constants,prompts}.py`, `hybrid_agent/README.md`, `hybrid_sft/{model,train,data}.py`, `hybrid_sft/README.md`, `hybrid_rl/{config,replay,grpo,reward,policy,rollout,train}.py`, `hybrid_rl/README.md`, `hybrid_inference/run_batch.py`, `interpretable_latent_mode/README.md`, `requirements.txt` | `hybrid_rl/{data,dist,minibatch,optimizer,trajectory}.py`, `hybrid_agent/{answer.py,tools/*}`, `interpretable_latent_mode/{__main__,render,trace}.py`, `scripts/*.sh`, `hybrid_sft/configs/*`, `examples/`, the three PNGs in `assets/` |
| Hugging Face | The model card of `togethercomputer/HybridAgent-8B-Agentic`; the `config.json` of Qwen3-4B/8B/14B-Base (for hidden sizes) | The 14B card, the datasets, file sizes, the weights themselves |
| GitHub | The issue and pull request lists | Issue and PR bodies |

I used the clean URLs; the links in the request carried a `utm_source=chatgpt.com` tracking parameter.

Every number in "Published results" is the authors', from their hardware and their infrastructure, **published, not reproduced by us**.

## What HybridAgent is

An LLM agent whose single generation process moves between two modes:

- **Text mode:** ordinary token sampling. Tool calls (`<python>`, `<search>`), the final answer and a few decisive reasoning steps are written here.
- **Latent mode:** no token is sampled. The last-layer hidden state `h_t` of the step is mapped by a small adapter, **HybridLink**, into the input-embedding space, and that vector `R(h_t)` is fed in as the next input. The span runs for tens to hundreds of forward passes.

Eq. 1 to 5 of the paper, in words:

- Text step: next input is `Emb(y)` for a sampled token `y`. Latent step: `R(h_t)` (older latent methods such as Coconut feed `h_t` itself, which does not match the pretrained input space).
- **HybridLink** `R(h) = h + W2 GELU(W1 h)`, about `2 d_h^2` parameters, paper says about 10M for a 4B model.
- Entry: the model samples the special token `<latent>` in text mode. Exit: the paper says the span closes when the nearest vocabulary embedding (cosine) of `R(h_t)` is `</latent>` (Eq. 5). See the differences list below for what the released engine does.
- **Working memory:** the KV cache of one trajectory, kept across turns. Text tokens, latent inputs and tool observations all go through the same input space and the same cache. A tool observation is appended as tokens to the existing cache; nothing earlier is re-encoded. Theorem A.1 proves that cached continuation equals a full-prefix forward pass (same deterministic ops, masks and positions).
- **Interpretable latent mode:** each latent input can be mapped to its nearest token embedding to give readable "pseudo-text". The paper reports this as readable and context-specific (Fig. 2, appendix F), and the intervention results below say the model uses the content.

A trajectory (Eq. 3): `[text] <latent> ... </latent> [text] [tool_call] [tool_response] ...`.

## How it is trained

Backbones: Qwen3-4B/8B/14B-Base. No pretraining from scratch; the architecture of the backbone is unchanged (anchor tokens added, HybridLink added).

**HybridSFT** (54K hybrid trajectories):

1. Data: existing text agent trajectories (DAPO-style math, ARPO/ToolStar-style search) converted by an automatic pipeline. Each reasoning step goes to latent or text by (a) external LLM judges (GPT-6.1-Sol / GPT-5.6-Luna) and (b) the frozen base model's mean token entropy (low entropy goes latent); then a "visibility audit" by an LLM critic restores steps to text when the visible trace no longer explains the decisions; adjacent latent steps are merged and the anchors inserted; a check that removing the anchors recovers the original text byte for byte.
2. Format activation: text SFT of the backbone so it emits `<latent>` and `</latent>` at the right places.
3. HybridLink activation: backbone frozen; HybridLink trained so `R(h)` at latent positions matches the embedding of the next target token. Loss: `lambda_cos (1 - cos) + lambda_mse ||R(h) - Emb||^2`, weights 1.0 and 0.2.
4. **Self-conditioned HybridSFT:** two passes. Pass 1 teacher-forced; HybridLink turns the hidden states into predictions of the next latent inputs. Pass 2 replaces the ground-truth latent embeddings with those predictions and trains backbone and HybridLink jointly (CE at text positions, regression at latent positions). The reason given (appendix B.1): latent inputs are continuous and fed back unquantised, so small errors compound over a span (exposure bias) in a way text tokens cannot, because a sampled token always lies in the discrete vocabulary.

**HybridRL** (GRPO on hybrid trajectories, HybridLink frozen):

- Plain PPO/GRPO needs a per-token likelihood at every position; latent positions have none. Two answers: **hybrid policy factorisation** (Prop. 4.1: with HybridLink fixed, latent steps and tool observations have probability 1, so the trajectory likelihood is the product over the sampled text positions only) and **exact rollout replay** (record the latent vectors during rollout, then in the update replay the whole trajectory in one differentiable forward pass in embedding space with those vectors as constants, and recompute the text log-probs). Theorem C.3 says the resulting text-token gradient equals the advantage-weighted gradient of the full recorded trajectory at `theta = theta_old`.
- Reward `r = r_correct + r_format + r_tool` (Eq. 11). "No explicit reward for mode usage".
- Rollout batch 512 trajectories (64 prompts x 8), one pass over 8 minibatches. 12K deep-reasoning prompts (math) and 1K agentic search prompts. All training and inference reported on 32 H100s.

## Published results (authors' hardware; published, not reproduced)

Abstract claims: 11 benchmarks, 3 scales, average +10.4% over the compared systems, 35.5% to 54.7% fewer tokens, 1.8x to 2.6x faster inference, 2.7x to 3.7x lower cost. Section 5.2 gives 2.1x as the average speedup and "up to 54%" fewer generation steps.

**Agentic benchmarks (Table 1; avg columns):**

| System | GAIA | WebWalker | HLE | SealQA S-0 / Hard | DSQA Acc / F1 |
|---|---|---|---|---|---|
| HybridAgent 14B | 52.4 | 49.5 | 16.0 | 24.3 / 37.8 | 21.7 / 46.8 |
| HybridAgent 8B | 43.7 | 42.5 | 14.4 | 22.5 / 35.8 | 20.0 / 40.7 |
| ARPO 14B (text RL) | 43.7 | 40.5 | 10.0 | 17.1 / 22.8 | 16.3 / 38.2 |
| Kimi-K3 | 41.7 | 10.0 | 31.4 | 11.7 / 28.0 | 21.7 / 46.0 |
| GPT-5.6-Luna | 37.9 | 11.0 | 29.0 | 11.7 / 29.1 | 23.3 / 49.4 |
| DeepSeekV4.1-Flash | 39.8 | 16.5 | 28.4 | 14.4 / 33.5 | 19.3 / 38.6 |

HLE is the column where the frontier models are ahead of HybridAgent in this table. Kimi-K3 appears as a baseline.

**Math and knowledge (Table 2), 14B:** AIME25 / AIME26 / MATH500 / GPQA-D / MedQA / MuSiQue. HybridAgent (RL) 43.3 / 45.8 / 88.2 / 55.1 / 81.7 / 34.5; ARPO 42.1 / 43.3 / 87.0 / 54.0 / 80.3 / 31.5; GRPO 40.8 / 42.5 / 87.4 / 52.5 / 81.0 / 32.0; Standard Text SFT 30.8 / 31.7 / 83.8 / 43.9 / 78.0 / 28.0; HybridAgent (SFT) 32.5 / 31.7 / 84.2 / 45.5 / 78.7 / 29.5. The 8B rows follow the same ordering (HybridAgent RL 39.2 / 43.3 / 84.6 / 48.0 / 75.3 / 32.0). Std over 5 seeds is about 0.031 (math) and 0.018 (search).

**Efficiency (Table 9 and Fig. 6, 7):** decoding compute per question 51.2 to 13.9 TFLOP (4B), 62.5 to 17.2 (8B), 66.9 to 25.1 (14B); estimated cost per million questions $359 to $98, $439 to $121, $470 to $176 at $2.50 per H100-hour and 10% of peak BF16. **The cost model charges every latent step a full backbone forward pass**, so the saving comes from fewer total steps, not cheaper steps. It excludes prefill, context-dependent attention and tool execution.

**What the ablations say:**

- Table 3, Qwen3-14B-Base, AIME25 / AIME26 / MATH500: base 4.9 / 3.0 / 47.0; + format activation 20.6 / 19.8 / 62.4; + HybridLink activation 23.8 / 22.5 / 73.8; + self-conditioning 32.5 / 31.7 / 84.2; Coconut 16.3 / 15.2 / 61.2.
- Table 6, HybridLink design (14B; MATH500 / MedQA / MuSiQue): identity 71.4 / 73.7 / 24.5; 1-layer 83.6 / 79.7 / 28.5; 1-layer + residual 86.4 / 80.3 / 31.0; 2-layer 85.8 / 81.0 / 31.5; 2-layer + residual (adopted) 88.2 / 81.7 / 34.5.
- Table 5, interventions on latent thoughts (MATH500, 8B / 14B, intact 84.6 / 88.2): cross-problem replacement 44.8 / 48.6; random-token replacement 41.2 / 44.6; mean-latent replacement 53.6 / 58.4; within-block permutation 73.2 / 76.4; one step per span 72.2 / 75.0; no latent mode at all 80.4 / 78.8; Gaussian noise at 0.5 and 2.0 relative strength 72.6 / 75.0 and 66.4 / 70.2; nearest-token replacement 82.6 / 87.2. Reading from the paper: the content, order and length of the latent span matter.
- Table 4, reward ablation (8B, MATH500 / GAIA / WebWalker; intact 84.6 / 43.7 / 42.5). The row labels were lost in extraction; I assigned them by matching the percentages in the text: without `r_correct` 79.2 / 35.9 / 37.5, without `r_format` 80.6 / 40.7 / 40.5, without `r_tool` 84.2 / 38.8 / 41.0.
- Table 7, auditability (MATH500 and AIME, 560 problems): text tokens per instance 302 (8B) and 284 (14B) against 596 and 497 for a matched text agent, with 170 and 183 latent steps; weak-monitor AUROC 0.840 / 0.860 against 0.772 / 0.818; strong-monitor AUROC 0.961 / 0.989 against 0.935 / 0.990.
- Fig. 9 (caption only): during RL the share of generation in latent mode rises while tool use and reward rise.

Limits the authors state: the Discussion section names harness and post-training directions (latent working memory as a first-class part of an agent harness, step-level credit on latent spans). I did not find a section titled "Limitations" in the text I extracted. Scale evidence is three sizes of one model family (Qwen3), 4B to 14B.

## What the code shows

Read from the source at `1b41ceb`.

**Stack.** `requirements.txt` pins torch 2.11.0, transformers 5.13.0, accelerate 1.14.0, deepspeed 0.19.2, math-verify, tavily-python. No vLLM and no verl. SFT: accelerate + DeepSpeed ZeRO-2 (8 GPUs, batch 1 x accumulation 2 x 8 = 16). RL: `torchrun`, own data-parallel code and a `ZeroOneAdamW` (ZeRO-1 sharded AdamW; not read).

**Engine (`hybrid_agent/engine.py`).**

- `HybridGenerationEngine` generates B rollouts of one prompt **in lockstep over one global KV cache**. The prompt is prefilled once at batch 1, then the cache is expanded to B rows. A row that is not producing a real slot in a step (waiting on a tool, finished) gets a padding slot that is masked out of attention for ever; per-row position ids keep the true RoPE positions. The cache length therefore grows with the number of lockstep steps, not with each row's own length (read from code, not measured).
- Per-row states: `DECODE`, `LATENT`, `FEED` (feeding a tool observation, token by token), `IDLE` (waiting on a tool), `DONE`. Tool calls run concurrently in a thread pool at a barrier; observations are tokenised and fed as ordinary tokens.
- Text mode blocks `</latent>` always and `<latent>` outside `<think>`; with `think_guards` it also keeps the `<think>` tags nested.
- One latent step: `injected = R(last_hidden)` is the next input. The exit test uses `readout_id = next_logits.argmax()`, the **LM-head argmax** at that step (see difference 1), and closes the span when it equals `</latent>` after `min_latent_steps`, or forces the exit at `max_latent_steps`. On exit the real `</latent>` token embedding is fed. `row_result` reports `anchor_exits` and `forced_exits`, so how often spans end naturally can be read off any run.
- `record_latent_readout` additionally computes the cosine nearest token of `R(h)` for each step, used only for the pseudo-text and the readability figure.
- KV memory of latent slots is tracked as `None` in a per-row ledger; the RL rollout records each latent vector (`R(h)`, one hidden-size vector per step) per trajectory for replay.

**HybridLink (`hybridlink.py`).** `post_ln(h + proj2(GELU(proj1(pre_ln(h)))))`: a pre-LayerNorm and a post-LayerNorm around the residual block. Parameters `2 d^2 + 6 d`. With the hidden sizes from the Hugging Face configs (2560, 4096, 5120) that is **13.12M (4B), 33.58M (8B), 52.46M (14B)**, computed by hand, not by loading the model.

**SFT details (`hybrid_sft/{data,model}.py`).**

- Token masks: text tokens CE weight 1; `<latent>` CE 1 and latent-mask 1; latent content tokens CE 0 and latent-mask 1 (the content text is only a regression target, never a CE target); `</latent>` CE weight **2.0** and latent-mask 0; tool results context only; EOS appended.
- HybridLink loss as in the paper, plus a second term over the exit positions (the last latent position of each span, target `Emb(</latent>)`) with weight 1.0.
- Self-conditioned pass: pass 1 is **not** detached (only the target embeddings are computed under `no_grad`), so gradients flow through pass 1 into the inputs of pass 2. In pass 2 the input at position `t+1` is replaced when `latent_mask[t] = 1`, which includes the `</latent>` position; at inference that position holds the true `</latent>` embedding.
- Default epochs 1 / 2 / 1 per stage (README: 1/1/1 and 1/1/2 also fine); backbone lr 7e-6, HybridLink lr 2e-4 (activation) and 5e-5 (self-conditioned); max length 16,888; AdamW betas (0.9, 0.95).

**RL details (`hybrid_rl/*`).**

- Rollout: temperature 1.0, top-p 1.0, top-k 0, `think_guards=False`, `reject_fabricated_result=False`, tool tokens counted in the budget, hard cap 16,888 slots for prompt + response (replay length). Each recorded text action stores its blocked-token set, and replay re-applies exactly that support before the log-softmax, so rollout and replay score the same distribution.
- Replay: one forward over the whole trajectory, recorded latents `detach()`ed and written into the embedding sequence, log-probs only at the sampled text positions in chunks of 256. The trainer logs `replay_agreement_abs_logp` (mean absolute difference between replayed and recorded log-probs before the first update of each step); the value is in `metrics.jsonl` of a run, not in the paper text I read.
- Objective (`grpo.py`): group-normalised advantage `(r - mean)/(std + 1e-6)`; PPO clip 0.2; **dual clip with c = 3.0 for negative advantages and a log-ratio clamp of 20**; token-level normalisation by the global token count of the minibatch; no KL term, no entropy term; zero-advantage trajectories skipped; incomplete or invalid groups dropped. These two clip details are in the code and not in Eq. 14 and 15 as extracted.
- Reward (`reward.py`): math = Math-Verify equivalence of the last `\boxed{}`; search = token-level F1; `r_format = -1` on an invalid trajectory; `r_tool = +0.1` only when the answer is correct and both a search and a python call were made. **`validate_format` requires at least one `<think>` pair and at least one `<latent>` pair**, so a trajectory with no latent span is penalised as malformed. This is a structural floor of one span, not a reward for how much latent mode is used.
- Learning rate default 1e-6; HybridLink frozen and a guard raises if it receives a gradient.

**System prompt (`prompts.py`).** It tells the model to put "routine internal reasoning (planning, recalling, restating, routine double-checking)" in latent mode, and only key facts, decisions and conclusions in text, and to open and close every think block in text.

**Released artefacts.** `togethercomputer/HybridAgent-8B-Agentic` and `-14B-Agentic` (Qwen3 base, HybridSFT then HybridRL on the agentic task, MIT), `HybridAgent-HybridSFT-v1` and `-v2` datasets (v2 from the same pipeline with a stronger labelling model), `HybridAgent-HybridRL` (math and search parquet). **No 4B checkpoint and no math-RL checkpoint** in the README table. A checkpoint is a normal Hugging Face backbone plus `adapter.pt` and `adapter_config.json`; loading the backbone alone gives a text-only model. Inference defaults in `run_batch.py`: temperature 0.6, top-p 0.95, top-k 20, **repetition penalty 1.1**, 4,096 response tokens, 5 tool calls, latent span 4 to 200 steps. The search tool needs a Tavily API key; the python tool runs generated code as the current user in a subprocess with timeout and memory limit.

GitHub state: one open issue (#3, "Verify evals on Papers with Code"), three merged pull requests (#1, #2, #4), none open.

### Differences between the paper text and the released code

Found by reading both side by side. I ran nothing to settle them, and I do not know which version produced the paper's numbers; they may simply describe the same thing at different levels of detail.

1. **Exit rule.** Paper Eq. 5: the span closes when `argmax_v cos(R(h_t), Emb(v)) = </latent>`. Released engine: the exit test is the argmax of the LM-head logits at that step; the cosine readout of `R(h)` is computed only for the pseudo-text. Appendix B.2 also says the backbone is trained with a CE target `</latent>` from the same hidden state, which is consistent with the LM-head route.
2. **HybridLink.** Paper Eq. 4 has no LayerNorm and says about 10M parameters for 4B. Released: two LayerNorms, 13.12M for 4B.
3. **Latent span bounds.** Paper H.3: minimum 16 and maximum 256 steps per span (training tables and inference). Released defaults: engine 4 / 200, `run_batch` 4 / 200, RL presets 4 / 180 (math) and 4 / 200 (search).
4. **Tool-call caps.** Paper: 10 (agentic search) and 5 (deep reasoning). Released RL presets: 6 (search) and 3 (math); `run_batch` default 5.
5. **Sampling at evaluation.** Paper: temperature 0.6, top-p 0.95, top-k 20. `run_batch` adds a repetition penalty of 1.1 by default; I did not find it in the extracted paper text.
6. **SFT epochs.** Paper Table 10: "3 (1 / 1 / 1)"; text H.3: one epoch each of the two activation stages "followed by one additional epoch of self-conditioned training". Code defaults: 1 / 2 / 1.
7. **Infrastructure.** Paper H.3 says the verl and vLLM infrastructure was extended for multi-turn hybrid generation. The released repository has its own Hugging Face based engine and trainer, with neither as a dependency.
8. **Mode usage in the reward.** Paper: no explicit reward for mode usage. Code: the format check requires at least one latent span per trajectory (a floor, not a gradient).
9. **Exit anchor weight.** Paper (B.2): the exit target is "given slightly more weight". Code: an extra mean over the exit positions at weight 1.0 in the regression, and CE weight 2.0 on `</latent>`.
10. **Policy objective.** Paper Eq. 15 shows the clipped objective; the code also applies a dual clip at c = 3.0 and a log-ratio clamp.

## How this might connect to Clover K3 (my reading; not Direction)

- Our own measurements, as recorded in `clover-ncm-research-context.md` and not re-read this session: K3 per-token cost is dominated by reading experts, and the number of tokens in a pass is nearly free until compute binds.
- The paper's own cost model charges a latent step a full backbone pass. On our system a latent step would therefore cost what a decoded token costs; a gain would have to come from fewer total steps (paper: up to 54% fewer generation steps), not from cheaper steps.
- The mechanism needs a trained HybridLink (about `2 d^2` parameters) and an SFT plus RL post-training of the backbone (54K trajectories, then 150 to 200 RL steps of 512 rollouts, on 32 H100s in the paper). The released checkpoints are Qwen3 8B and 14B. Nothing in what I read says it can be applied to a model we did not train.
- Kimi-K3 is one of the baselines in Table 1 (published numbers above). That says something about K3 as an agent, not about our engine.
- Which of these, if any, matters for K3 is the creator's call.

## What is not known

- Whether the released checkpoints, run in the released engine, give the paper's numbers or its speedups (not run). The released engine is Hugging Face based and steps through Python per token; the paper's timings were taken with infrastructure the repository does not contain.
- Which of differences 1 to 10 reflects the version behind the paper's results.
- The paper's learning rates (the table cells were empty in the extraction) and appendix I prompts.
- Whether `hidden_states[-1]` returned by the engine's forward equals `last_hidden_state` used in SFT and replay in transformers 5.13 (it matters for HybridLink input scale; the checkpoints presumably work, but I did not check).
- How often spans end on the anchor and how often at the step cap in real runs (the engine reports both).

## Smallest next cycles (suggestions; the creator chooses)

1. On the GPU server, load `HybridAgent-8B-Agentic` and run the five questions in `examples/questions.jsonl` with `batch_size 1` and the readout on. Needs the 8B weights (about 16 GB in BF16, an estimate) and a `TAVILY_API_KEY` for the search tool, which only the creator can put on the host; the python tool alone works without it. Observe the real latent spans, `anchor_exits` against `forced_exits`, and steps per question. A short cycle, no timing claims from the first run.
2. Same engine, same questions, a matched text-only run of the base Qwen3-8B with latent mode blocked, to see steps and wall time in one reality before comparing with the paper's ratios.
3. At one exit step, record both the LM-head argmax and the cosine nearest token of `R(h)` to see whether difference 1 matters in practice.
4. One-line check of the `hidden_states[-1]` against `last_hidden_state` question.

## What I did this session

Fetched the arXiv abstract and HTML text, shallow-cloned the repository to `c:\personal\oss\HybridAgent` (commit `1b41ceb`), read the files listed above, fetched three Qwen3 configs, one Hugging Face model card and the GitHub issue and PR lists, and computed the HybridLink parameter counts by hand. Wrote this file. Nothing was run on `gpu` or `k3`; no packages were installed; nothing in the repository `clover-kimi-k3` other than this file was changed. This record is offered: keep it, correct it, or delete it.

## Cycle 2 (2026-10-21, later): work moved to a worktree, k3 cleaned, Clover K3 run

Direction (the creator): use a separate worktree and a new branch because another session works in the main working copy; use the CPU box `k3` for inference; the GPU box stays with the NCM research; clear old tmp files, logs, data captures and old Kimi K3 files if space is needed but keep the final `clover-kimi-k3` setup; run it so it can be compared with the HybridAgent research. Answers given to my questions: run both (Clover K3 first, then HybridAgent), stop the other session's job on k3, and delete these four groups: `/root/k3trunk_i8`, old logs, `archive-before-restructure`, the old dated folders in `/opt/clover-k3`. Not approved and left alone: `/tmp/ds4bit`, `/tmp/kda-genesis`, `/tmp/rt-*` (about 31 GB).

**Worktree.** `c:\personal\oss\clover-framework-hybrid`, branch `hybridagent-compare`, created from `main` (`e48f49b`), not pushed. This copy of this file is the one to keep; the untracked copy in the main working copy is older.

**Another session's job on k3 (stopped on the creator's request).** The driver `k3run.sh` (k3ncm prompts p1 to p5, chat mode, DSpark draft, 260 new tokens each) had finished p1 to p4 (429, 431, 311, 360 s, all exit 0) and was 1 minute into p5. I stopped the driver and its `whole-model`; p5 has no result; a note is in `src/work/results/k3ncm/p5.stopped.txt`. p1 to p4 logs and ids are intact.

**Cleanup on k3 (script `/root/clean1.sh`, log `/root/cleanup-20261021.log`, list of removed folders `/root/cleanup-20261021-dirs.txt`).** Checked first, read-only: the engine opens only the 809 files of `datasets/computed` (all present at the recorded size, all real files, none a symlink into the archive) plus `datasets/third-party/kimi-k3-open-weights` (shard 94, tiktoken.model, tokenizer_config.json; 4.4 GiB, read on every run, link count 1, so kept). Free space 20 GB, then 70 after `k3trunk_i8` (51 GB trunk.bin), 71 after logs (the journal had nothing to vacuum; rotated logs went from 1.1 GB to 195 MB), 113 after 83 dated folders, **158 GB free at the end** (disk 91% used). Kept: `datasets/computed` (1.4 TB), the open weights, `kimi-k3-dspark`, `work/inputs/refs`, `hf-push` (the upload scripts use it), `clover-data`, loose dated `.py`, `.json` and the `checkpoint-20260930-a.tar` file. Consequences I noticed and did not repair: `/opt/clover-k3/clover-intelegence/dataset` (a symlink to the archive) and `/opt/clover-k3/build/eqidx.bin` now dangle (they belong to the old distributed-hosting tree), `/opt/clover-k3/config.env` still names `/root/k3trunk_i8` for the old trunk gate, and two symlinks in `work/inputs/tokfiles` were already broken before today (they point to `clover-kimi-k3/dataset` and `clover-kimi-k3/assets`, which do not exist). The deletions cannot be undone.

**Clover K3, recorded 1000-token prompt, run on k3 (`BUILD=0 ./run-model.sh recorded`, log `/root/runs/recorded-20261021.log`, nothing else running on the box).** Arithmetic: trunk exact, experts exact. One-time setup 9.63 s. Request, words to next token: 745 words = 1000 tokens, **301.535 s** in total (layer 0 1,114 ms, layer 1 3,306 ms, layers 2 to 92 297,050 ms: 67 KDA layers 3,240 to 3,328 ms each and 24 MLA layers 3,160 to 3,229 ms each, tail 55 ms). Next token id 3027, " team". Against the pod's own chain: embeddings 1000 of 1000 rows, layer 92 output 1000 of 1000 rows, 8 snapshots 8000 of 8000 rows bit-identical. One run, one box, as measured today; it agrees with the 297 to 302 s recorded in the README.

**HybridAgent-8B-Agentic, run on k3 CPU (same box, after the Clover K3 run, nothing else running).**
Set-up (script `/root/setup-hybrid.sh`, log `/root/hybrid-setup.log`, 2 min 41 s): repository at commit `1b41ceb` in `/root/hybridagent`, Python 3.12.3 venv `/root/hybridagent-venv`, torch 2.11.0+cpu, transformers 5.13.0 (the pins of the repository, with the CPU wheel for torch and without deepspeed), checkpoint `togethercomputer/HybridAgent-8B-Agentic` (snapshot `13a5c61b`, 16 GB) under `/root/hf`. Disk after: not re-read; 158 GB free before.
Run: `OMP_NUM_THREADS=16`, `run_python_only.py` (my launcher, `/root/hybridagent/`; it is `hybrid_inference.run_batch` with `build_tool_executor(require_search=False)`, because the released `run_batch` raises without a `TAVILY_API_KEY`), question q4 of `examples/questions.jsonl` ("How many trailing zeros does 100! have?", answer 24), one sample, `--max_response_tokens 1024 --max_search_calls 0`, default sampling of `run_batch` (temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.1), seed 0, bf16. **This is a different protocol from the paper and the README example: the search tool was off.**
Observed (record `/root/runs/hybrid/q4.jsonl`, copy of my read in `oss/scratch-agents/hybrid-clean/q4-result.txt`): wall 452.6 s for the sample (about 0.44 s per generation step, prefill included); 1,024 generation steps, of which **323 latent steps in 12 latent spans** (102, 13, 6, 4, 24, 35, 71, 23, 9, 18, 11 steps, 11 ended on the exit anchor, the last one was cut by the budget) and 701 text tokens; no tool call; **end reason `budget`, no `<answer>` block, so the extracted prediction is empty**. The visible text reaches "24" twice (floor(100/5) + floor(100/25) = 20 + 4) and then re-checks the formula on 25!, where the model first writes a wrong digit count for 25! and corrects itself. The pseudo-text readouts are partly garbled (repeated words, a few non-Latin tokens); readability scores were not computed.
What this does not show: one question, one sample, one seed, one box, a cap of 1,024; no text-only control with the same weights (the released engine has no switch that blocks `<latent>`, and the other released artefacts are different models); no comparison of steps with a text agent; no search question. The 0.44 s per step is this box and this engine (Hugging Face, one row, bf16 on CPU), not the paper's setup.

**Same box, two systems, as measured today (one run each, different things):** Clover K3 prompt pass of a 1,000-token prompt 301.5 s, bit-identical to the pod chain; its decode cost from the earlier records (README and `clover-kimi-k3-cpu-context.md`, not re-measured today) is about 3 s per token with the expert cache. HybridAgent-8B generation step 0.44 s. These are different models of different sizes, so the ratio says nothing about the method; the paper's own comparison is steps, and here the step count of the latent run for the same answer was not compared with anything.

Open questions that came out of this run: does the same prompt finish with `--max_response_tokens 2048 or 4096` (the paper uses 4,096 for math), what share of steps is latent across the five example questions, and is the pseudo-text readability on this box in the paper's range. Each is one more run of about 8 to 15 minutes.


## Cycle 3 (2026-10-22): step-by-step trace of HybridAgent-8B on k3 (Direction: see each step, stage, layer; do not assume)

**Instrument.** `trace_hybrid.py` (local copy `oss/scratch-agents/hybrid-clean/`, remote `/root/hybridagent/`), observation only, no engine file edited. Hooks: forward pre-hook on the embedding-side input, post-hook on each of the 36 decoder layers, final norm and LM head (all `torch` hooks, last position only); wrappers around `_lockstep_forward`, `run_hybridlink` and the sampler for timing. Per latent step it also reads the LM-head distribution and the cosine nearest token of `R(h)` (my own extra work, timed separately as `obs` and subtracted from the "net" column). Same question (q4), same settings and seed 0 as the 1,024-step run, search tool off, batch 1, `OMP_NUM_THREADS=16`, budget 450 steps, nothing else running on `k3`. Raw record `/root/runs/hybrid/trace1.json`, log `trace1.log`; the printed analysis is saved at `oss/scratch-agents/hybrid-clean/trace1-analysis.txt`.

**Gate (checked first).** The 1,019 characters of output are identical to the first 1,019 of the earlier 1,024-step run (`first_diff` none). The instrument did not change the text in this run. The 4 spans (102, 13, 6, 4 latent steps) and 4 anchor exits equal the first four spans of the earlier run.

**Totals.** 450 generation steps (+1 pad step), 125 latent steps, 4 spans, 4 anchor exits, 0 forced exits, end reason `budget`, wall 196.7 s with the instrument (0.437 s per step), prefill of 529 prompt tokens 2.68 s. One run; no repeat of the trace itself.

**Time per step by kind (ms, mean; net = without my own readout work).**

| kind | n | net total | forward | of which 36 layers | final norm | LM head | HybridLink | sampling | other |
|---|---|---|---|---|---|---|---|---|---|
| text token | 317 | 408.1 | 402.5 | 373.5 | 0.09 | 28.3 | - | 5.2 | 0.5 |
| step that sampled `<latent>` | 4 | 409.6 | 404.0 | 375.1 | 0.10 | 28.2 | - | 5.2 | 0.4 |
| latent step | 125 | 404.3 | 402.2 | 373.2 | 0.09 | 28.2 | 1.9 | - | 0.2 |
| exit step (`</latent>` fed) | 4 | 409.7 | 407.7 | 378.7 | 0.09 | 28.2 | 1.9 | - | 0.2 |

Read: a latent step costs the same as a text step (404 against 408 ms), as the paper's cost model says. The 36 layers are 92% of a step (about 10.4 ms per layer on average), the LM head 7%, the HybridLink 0.5%, sampling 1.3%, everything else under 0.2%. Each stage is measured with a timer around it, so hook overhead is inside the numbers and was not measured on its own; the comparison with the untraced run (0.442 s per step, 1,024 steps) agrees but is not a head-to-head.

**Fed vector.** Norm of the fed embedding at text steps 1.64; norm of `R(h)` at latent steps 8.83 (5.4 times larger). Layer 0 (last position): output norm 10.9 (text) against 24.3 (latent); relative update 6.32 against 2.32. After the final norm the vector read by the head and the HybridLink has norm 152 at text steps and 71 at latent steps.

**Layers (selected; last position; mean over 317 text steps / 125 latent steps).** Output norm grows with depth in both (layer 8: 56 / 59; layer 24: 311 / 258; layer 34: 1356 / 1401). Relative update is 0.35 to 0.6 in most layers in both kinds. Two places differ: layer 16 (text 0.998, latent 0.309) and layer 35 (output norm text 1,378, latent 2,336; relative update 0.44 / 0.81). Means only; I have not looked at their distribution over steps, so a few tokens could carry them. Unexplained.

**`hidden_states[-1]` question (open item above): answered for this run.** The vector the engine keeps as `last_hidden` equals the output of the final norm exactly: maximum absolute difference 0.0 over all 451 steps (transformers 5.13.0, this engine). Last position only.

**Latent-step readouts (129 latent-mode decision steps: 125 continued, 4 exited).**
- LM-head P(`</latent>`): 0.0005 mean and 0.035 at most on continued steps; 0.998, 0.957, 0.987, 0.972 on the four exits.
- Cosine of `R(h)` to `Emb(</latent>)`: 0.063 mean on continued steps; 0.999, 0.998, 0.999, 0.988 on exits.
- The two readouts agreed on all 129 steps (both say `</latent>` on the 4 exits, neither says it on the 125 others). Difference 1 (LM-head argmax in the engine versus cosine for pseudo-text) made no difference to exits in this run.
- Cosine of `R(h)` to its nearest token embedding: mean 0.855, min 0.273, max 0.998.
- LM-head top-1 probability in latent steps 0.88, plain-softmax entropy 0.42; at text steps the sampling distribution (temperature, top-k, top-p applied) has entropy 0.07 and P(sampled token) 0.96. These two entropies are of different distributions and are not comparable.
- P(`<latent>`) at the steps that sampled it: 1.0, 1.0, 1.0, 0.931; mean over all text steps 0.0022.
- Spans: 102, 13, 6, 4 latent steps; none open at the end of the budget.

**Not observed.** Per-layer attention versus MLP split; KV cache size or cost growth with length (only 529 + 451 positions); any step after 450; one question, one seed, batch 1, search off, one box; how steps would differ with the search tool; the layer 16 and 35 distributions.

**Reading, not Direction: HybridAgent step stages next to Clover K3 stages.** HybridAgent step: input (embedding of a token, or `R(h)` from the previous step) -> 36 layers (92% of time) -> final norm -> LM head (7%) -> sample token, or HybridLink -> `R(h)` fed to the next step (0.5%). Clover K3 token: tokenise -> embedding -> layer 0 -> layers 1 to 92 -> tail. The same step shape; the HybridLink sits where K3's tail output meets the next embedding. What this trace says about K3 is limited to this: in a step, the cost of the latent feedback itself (a small two-layer map and a norm) was 1.9 ms of 404; all the cost stays in the layers, which for K3 are expert reads. No K3 per-stage numbers were taken in this cycle.

**Smallest next cycles (the creator chooses).** (a) From the existing `trace1.json`, the distribution over steps of layer 16 and layer 35 values and of the span position (no new run); (b) a longer budget run (2,048 or 4,096) to see a complete answer and more spans; (c) K3's own per-layer stage times from an existing log for the side-by-side; (d) the other four example questions.


## Cycle 4 (2026-10-22): distributions behind the layer 16 and layer 35 means (existing trace1.json, no new run)

Script `oss/scratch-agents/hybrid-clean/dist_trace.py`, output `dist1.txt`. Same 317 text steps and 125 latent steps as Cycle 3.

**Correction of Cycle 3 (layer 16).** The mean relative update of 0.998 at layer 16 on text steps came from two steps, not from the layer's behaviour. Median relative update at layer 16: text 0.337, latent 0.298; quartiles overlap (text 0.300 to 0.373, latent 0.261 to 0.346). Two text steps have a relative update above 0.9 (maximum 117.8; layer-16 output norm up to 8,547 against a median of 86); their input ids are 220 and 15, one step each. Excluding those two, the text mean is 0.337. So "layer 16 differs between text and latent steps" was an artefact of using means; it is withdrawn. The two outlier steps are real observations (a very large layer-16 update at two steps) and are not explained; which positions they are, and whether they recur on other prompts, was not checked.

**Layer 35 (last layer, before the final norm): the difference holds.** Output norm, text median 1,351 (p10 993, p90 1,795) against latent median 2,321 (p10 1,729, p90 3,019). Relative update, text median 0.419 (p90 0.628) against latent median 0.766 (p90 1.286). The distributions overlap only in their tails (latent p10 1,729 is near text p90 1,795). Cosine of input and output at layer 35: text median 0.939, latent 0.952, so the vector grows in size without turning more.

**Along the run.** At steps 125 to 200 (inside the 102-step span) the layer-35 output norm is 2,345 to 2,931 and relative update 0.94 to 1.62; after the span (step 225 onward) it returns to 900 to 1,950. The span means of layer-35 output norm are 2,443 (102 steps), 1,929 (13), 1,836 (6) and 1,695 (4): longer spans sit higher. The final-normed norm that the head and HybridLink read is 53 to 93 inside the long span and 96 to 219 on text steps, so the final norm shrinks these vectors more.

**Inside the long span.** Relative update at layer 16 and the norm of `R(h)` at the input stay in a narrow band from the first to the last latent step (0.245 to 0.371; 6.6 to 10.0). The cosine of `R(h)` to its nearest token embedding jumps between 0.33 and 0.997 from one step to the next (steps 164 to 168: 0.995, 0.699, 0.994, 0.997, 0.994), and P(`</latent>`) is below 0.0001 at every step shown. Five-step samples at three places of one span, not all 102 steps.

**Not observed / limits.** One prompt, one seed. No decoding of which tokens the two outlier steps are; no per-step view of what the nearest-token readouts spell; layer 35 not split into attention and MLP.

**Next (running):** the same trace with budget 2,048 (`trace2.json` on k3, started 01:58 k3 time), to see more spans and, if it finishes within the budget, the answer.


## Cycle 5 (2026-10-22): the same trace with budget 2,048 (job on k3, `trace2.json`, log `trace2.log`; analysis `trace2-analysis.txt`, `trace2-tail.txt`, `exits2.txt` in `oss/scratch-agents/hybrid-clean/`)

Same question (q4), settings, seed 0, search off, batch 1, `OMP_NUM_THREADS=16`, nothing else on `k3`; disk was not a constraint (141 GB free), nothing deleted.

**Gate.** `first_diff` none: the new output (2,704 characters) starts with the full 2,202 characters of the earlier 1,024-step run. Raising the budget from 1,024 to 2,048 did not change the text up to where the earlier run stopped (seed 0, this box).

**Outcome of the run.** It ended by itself (end reason `eos`) after **1,243 generation steps**, wall 553.4 s with the instrument (0.445 s per step), prefill 2.6 s. 329 latent steps in **13 latent spans** (102, 13, 6, 4, 24, 35, 71, 23, 9, 18, 11, 8, 5), 13 anchor exits, 0 forced exits, no tool call. The output ends with `</think><answer> ... 20 + 4 = 24 ... \boxed{24}</answer>`; the answer is 24, which is the answer in `examples/questions.jsonl` for q4 (the engine's own `prediction` field was not printed by my script, so I compared by reading). One question, one sample.

**Per-step times (ms, net of my own readout work), 888 text steps / 329 latent steps / 13 exit steps.** Text 422.3 (layers 387.2, final norm 0.09, head 28.3, sampling 5.5); latent 416.7 (layers 385.5, head 28.4, HybridLink 1.9); exit 421.8. In the first 450 steps the same kinds were 408.1 / 404.3 / 409.7, so the mean step is about 3.5% slower over the longer run (the KV cache grows from 529 to about 1,770 positions); I did not break the time down by position. A latent step is still as expensive as a text step.

**Gate-like repeat of Cycle 3/4 numbers on 2.7 times more steps.** The final-normed vector norm read by the head and the HybridLink: 140 (text) against 75 (latent); `hidden_states[-1]` against the final-norm output: maximum difference 0.0 over all 1,243 steps. Fed vector norm: 1.64 (embedding) against 8.42 (`R(h)`). Layer 35 (before the final norm): output norm median text 1,467 (p10 1,034, p90 1,940) against latent 2,096 (p10 1,481, p90 2,756); relative update median 0.447 against 0.616. The separation seen in the first 450 steps is smaller here (text p90 1,940 is above latent p25 1,775), so the layer-35 difference is real on average but the two kinds overlap more over a whole answer than in the first 450 steps. Layer 16: median relative update text 0.324 against latent 0.290; the same two text steps (input ids 220 and 15; maximum 117.84, output norm 8,547, values identical to Cycle 4) are the only ones above 0.9 in all 888 text steps, so they sit in the first 450 steps and nothing like them happened later.

**Where the two exit readouts differ (Direction item: do not assume).** 342 latent-mode decision steps. Engine exits (13): the LM-head says `</latent>` at top-1 on all 13, with P between 0.49 and 0.998 (P = 0.49 at step 535, 0.63 at step 723, 0.67 at step 819, 0.92 to 0.998 at the other ten), and the nearest token embedding to `R(h)` is also `</latent>` on those 13 (cosine 0.897 to 0.999). On 3 further steps (523, 673, 721; positions 24, 67, 22 in their spans) the nearest token embedding of `R(h)` is `</latent>` but with a cosine of only 0.46, 0.54 and 0.51 to it, while the LM-head's top-1 is another token (P of the top token 0.84, 0.57, 0.99; P(`</latent>`) 0.0, 0.018, 0.0) and the engine, which uses the LM-head, did not exit. So the exit rule choice (LM-head argmax in the code, cosine in the pseudo-text readout, difference 1) makes no difference on the 13 exits and would have caused 3 early exits out of 329 continued steps if the cosine nearest-neighbour had been the rule. Max P(`</latent>`) on continued steps 0.19. In the first 450 steps (Cycle 3) the two readouts agreed everywhere; this is the first disagreement, and it is on low-cosine steps.

**Not observed.** Other questions; a text-only run of the same weights to count steps saved (not possible with the released artefacts, see Cycle 2); search-tool runs; time by position in the sequence; what text the latent states spell for the 13 spans; any other seed or sampling setting.

**Next, single cycles (the creator chooses).** (a) Same trace, other questions of `examples/questions.jsonl` (python tool only, search off); (b) time per step against position from the existing `trace2.json` (no new run); (c) K3-side per-stage timings from an existing log for the side-by-side with the table above.


## Cycle 6 (2026-10-22): step time against position, from `trace2.json` (no new run; `time_by_pos.py`, `timepos2.txt`)

Net step time (without my own readout work), mean per bucket of 150 steps, q4, 1,243 steps, positions 529 to 1,772: 404.4 ms (positions 529 to 679), 412.7 (829 to 979), 421.6 (1,129 to 1,279), 432.3 (1,429 to 1,579), 440.3 (1,729 to 1,772). Layer time rises 370.5 to 405.2 ms over the same range; the LM head stays 28.3 ms in every bucket. Text steps and latent steps rise together (for example 417.4 and 417.0 at positions 979 to 1,129). Least-squares slope about 31 ms per 1,000 positions (0.031 ms per step); first 50 steps 403.1 ms, last 50 steps 440.0 ms (+9%). Observed: the growth is in the 36 layers only (the KV cache grows with the sequence on this CPU engine); not split into attention against the rest. One question, one box.


## Cycle 7 (2026-10-22): the other four example questions with the same trace (budget 2,048, seed 0, search off, python tool on, batch 1, `k3`)

Records `/root/runs/hybrid/trace-q1.json`, `-q2`, `-q3`, `-q5` (q4 is `trace2.json` from Cycle 5); summary `oss/scratch-agents/hybrid-clean/qs-summary.txt`, tails `qs-tails.txt`, scripts `summ_qs.py`, `trace_hybrid.py` (now takes the question id). The runs went one after another, nothing else on the box.

| question | steps | latent steps (share) | spans (latent steps each) | python calls | ended | wall | answer |
|---|---|---|---|---|---|---|---|
| q1 sound, 10 km (29) | 147 | 59 (40%) | 1 [59] | 0 | at the first `<search>` request | 68 s | none (stopped) |
| q2 light Sun-Earth (8) | 286 | 63 (22%) | 2 [48, 15] | 0 | at the first `<search>` request | 127 s | none (stopped) |
| q3 milk vs water (30) | 232 | 55 (24%) | 1 [55] | 0 | at the first `<search>` request | 103 s | none (stopped) |
| q4 100! zeros (24) | 1,243 | 329 (26%) | 13 | 0 | `eos` | 553 s | 24, correct |
| q5 reflected circle ((-12, -8, 36)) | 1,035 | 240 (23%) | 8 [56, 40, 15, 30, 35, 18, 22, 24] | 3 | `eos` | 462 s | `\boxed{(-12, -8, 36)}`, correct |

**Why q1 to q3 stop.** Each output ends with `<search>...</search>` (q1: "speed of sound in air at 20 degrees Celsius"; q2: "speed of light in meters per second to km per minute"; q3: "density of whole milk at room temperature"). With search off (`--max_search_calls 0`, `terminate_on_tool_cap`) the engine ends the run at that request, so these three runs show the part before the first search only; they say nothing about whether the model would answer them with a search tool. The paper's protocol uses search; a Tavily key on the host (only the creator can place it) is needed for that.

**Same per-step picture on every question.** Net step time text / latent (ms): q1 409.4 / 405.7, q2 412.8 / 416.9, q3 410.1 / 413.3, q4 422.3 / 416.7, q5 422.7 / 419.7: a latent step costs what a text step costs on all five. The vector the engine keeps as `last_hidden` equals the final-norm output exactly (max difference 0.0) on all five. Fed norm: embedding 1.64 to 1.69, `R(h)` 8.4 to 9.2. Layer-35 output norm median, text against latent: q1 1,558 / 2,586, q2 1,324 / 2,435, q3 1,309 / 2,499, q4 1,466 / 2,096, q5 1,549 / 2,222: latent is higher on every question; the gap is larger on the three short runs.

**Layer 16 outliers.** Text steps with a relative update above 0.9: q4 two (input ids 220 and 15), q1, q2, q3 and q5 none (0 of 86, 219, 175 and 779). The two q4 steps are not a general property of this model on these questions.

**Exit readouts.** Exits (LM-head top-1 is `</latent>` at every exit): q1 1 (P 0.999), q2 2 (min P 0.53), q3 1 (P 0.999), q4 13 (min 0.49), q5 8 (min 0.69). Steps where only the cosine nearest token is `</latent>` and the engine did not exit: q1, q2, q3 none; q4 3; q5 1. Steps where only the LM-head says it: none on any question. Maximum P(`</latent>`) on continued steps: q1 0.000, q2 0.004, q3 0.034, q4 0.192, q5 0.345.

**Latent share.** 22% to 40% of generation steps are latent across the five (40% in q1 which has one 59-step span in 147 steps; 22% to 26% in the four others). This is a count of steps in one sample per question, not a saving: no text-only run of the same weights exists to say how many steps the same answer would take.

**Not observed.** Search-assisted runs (needs the key); other seeds or sampling settings; a text-only control; K3 numbers.

**Next, single cycles (the creator chooses).** (a) A Tavily key on the host, then q1 to q3 with search on; (b) a second seed for q4 and q5 to see how much the spans vary; (c) K3 per-stage timings from an existing log for the side-by-side with the HybridAgent step table.


## Cycle 8 (2026-10-22): a second seed for q4 and q5 (seed 1; budget 2,048, search off, python tool on, batch 1, `k3`)

Records `/root/runs/hybrid/trace-q4-s1.json`, `trace-q5-s1.json`; `trace_hybrid.py` now takes the seed as its fourth argument; summaries `seeds-summary2.txt`, `outliers.txt` in `oss/scratch-agents/hybrid-clean/`. Seed 0 are the Cycle 5 and Cycle 7 runs. One run per seed.

| run | steps | latent steps (share) | spans | python calls | ended | wall | answer |
|---|---|---|---|---|---|---|---|
| q4 seed 0 | 1,243 | 329 (26%) | 13: 102, 13, 6, 4, 24, 35, 71, 23, 9, 18, 11, 8, 5 | 0 | `eos` | 553 s | `\boxed{24}`, correct |
| q4 seed 1 | 1,208 | 172 (14%) | 10: 39, 6, 5, 27, 8, 27, 16, 10, 4, 30 | 3 | `eos` | 531 s | `\boxed{24}`, correct |
| q5 seed 0 | 1,035 | 240 (23%) | 8: 56, 40, 15, 30, 35, 18, 22, 24 | 3 | `eos` | 462 s | `(-12, -8, 36)`, correct |
| q5 seed 1 | 1,148 | 324 (28%) | 10: 39, 74, 38, 38, 40, 31, 17, 6, 23, 18 | 1 | `eos` | 517 s | `(-12, -8, 36)`, correct |

**What the second seed shows.** The two seeds write different text from early on (first differing character 87 for q4, 59 for q5), take 1,035 to 1,243 steps, and differ in the number and length of latent spans, in the share of latent steps (q4 14% against 26%; q5 28% against 23%) and in how many python calls they make (q4: 0 against 3; q5: 3 against 1). All four runs end with a boxed answer that matches the example file. So the latent share of a run is not a fixed property of the question: q4 moved from 26% to 14% between two seeds. Two seeds per question, one run each.

**Same picture per step.** Net step time text / latent (ms): q4 seed 1 425.1 / 426.0, q5 seed 1 424.4 / 419.0 (seed 0: 422.3 / 416.7 and 422.7 / 419.7). `hidden_states[-1]` equals the final-norm output exactly (0.0) in both. Layer-35 output norm median text / latent: 1,494 / 2,140 and 1,580 / 2,156.

**Exit readouts.** q4 seed 1: 10 exits (minimum P(`</latent>`) 0.35 at an exit, so the LM-head argmax is `</latent>` there with a probability under one half), 1 step where only the cosine nearest token is `</latent>`; q5 seed 1: 10 exits (minimum 0.63), 3 cosine-only steps. Steps where only the LM-head says it: none. One new thing: in q5 seed 1 at step 436, position 4 of its span, the LM-head top-1 is `</latent>` with P = 0.994 and the engine did not exit. The code (`engine.py` line 334) exits only when `row.latent_steps >= min_latent_steps` (4) and the readout is the closing token; by my count that step is the fourth decision, with three latent steps done, so the minimum-length rule is the likely reason (inferred from code and position count, not printed by the engine). First observed case of that rule acting in these 11 runs.

**Layer-16 outliers, now with the tokens (tokenizer lookup, `outliers.txt`).** Text steps with a relative update above 0.9 at layer 16, all traces: q4 seed 0 step 54 (fed token `' '`, after "... 10 is 2 times"; relative update 92.6) and step 371 (`'0'`, after "(100/25) + floor(10"; 117.8); q4 seed 1 step 214 (`'0'`, same words "(100/25) + floor(10"; 123.4); q5 seed 1 step 552 (`'1'`, after "ordered triple (b, c, d) should be (-1"; 169.5); none in q1, q2, q3, q5 seed 0. Layer-16 output norm at those steps 8,547 to 11,312 against a median near 85. So in these traces the outlier appears on a digit or space token inside an arithmetic expression; 3 of the 7 traces contain one (q1 to q3 and q5 seed 0 contain none), and the same token in the same words gave it in both q4 seeds. Not explained; not tested on other tokens.

**Not observed.** Third seed; search runs (no key on the host: checked only that `TAVILY_API_KEY` is not in the login shell or the shell profile files of root on `k3`); a text-only control.


## Cycle 9 (2026-10-22): Clover K3 decode measured today, side by side with the HybridAgent step table

Run on `k3` after the HybridAgent runs had finished (nothing else active): `BUILD=0 PF_DECODE=30 PF_CACHE=32 ./run-model.sh words 20`, log `/root/runs/k3-decode-20261022.log` (copy `oss/scratch-agents/hybrid-clean/k3-decode-log.txt`). Prompt: the first 20 words of `inputs/text.txt` (24 tokens, raw text, not chat), then 30 decode steps; 32 experts cached per layer (51.7 GB), no early read (`LB_GUESS` 0), exact arithmetic ("trunk exact; experts exact"), 16 threads. One run.

**Observed (K3).** One-time setup 53.87 s. Prompt pass 20.369 s for 24 tokens (layer 0 42.9 ms, layer 1 337.8 ms, layers 2 to 92 19,933 ms, tail 53.9 ms). Decode: 30 more tokens in 90.551 s, **mean 3.018 s per token**, first step 3.854 s; layers 89.045 s in total (2.968 s per token, 32.3 ms per layer on average), tail 1.505 s (0.050 s per token). Per decode token, summed over the 92 layers: before the router 0.978 s, latent 0.052 s, experts 1.582 s, after the router 0.330 s, readers start + guess 0.000 s. Expert cache: 20,653 of 44,160 needed experts already in RAM (46.8%), 23,507 read from disk (412.5 GB, about 13.8 GB per token). 7 major page faults in the decode phase. The 31 generated tokens read as sensible English (" on this repository.** ... This file tells you how to behave ..."), that is the continuation of the first words of `AGENTS.md`; no reference exists for this prompt, so nothing was compared. Same setting as the 24-token line of the earlier record (3.07 s per token at C=32, one run then): 3.018 now.

**Side by side, one decode step (different models; this is a table of shapes, not a ratio of methods).**

| | HybridAgent-8B step (Hugging Face, bf16, CPU, 16 threads; q4, q5, seeds 0 and 1) | Clover K3 decode step (C, exact, CPU, 16 threads, 32 cached experts per layer) |
|---|---|---|
| step time | about 405 to 440 ms (rises with position, Cycle 6) | 3.018 s (24-token prompt) |
| input stage | embedding lookup or `R(h)`; not separately timed (inside "other" 0.2 to 0.5 ms) | tokenise and embedding 1.4 ms at prompt time; layer 0 and layer 1 are separate stages |
| layers | 36 layers, 92% of the step, 10.4 ms per layer on average | 92 layers, 98.3% of the step (2.968 s), 32.3 ms per layer on average |
| inside the layers | not split (attention against MLP not measured) | before the router 33% (0.978 s), latent 1.8%, experts 53% (1.582 s), after the router 11% (0.330 s) |
| final norm and head | 28.4 ms (7%) | tail 50 ms (1.7%) |
| sampling | 5.5 ms (1.3%) | not split out in this log |
| what bounds it | not measured with counters here | the log's own reading (Cycle 2 record): disk and memory; 46.8% of experts needed were not in RAM |

**Reading (labelled as such, not Direction).** (1) In HybridAgent one latent step costs the same as one text step (Cycles 3, 5, 7, 8); the HybridLink itself is 1.9 ms of about 420. (2) The feedback map `R(h)` of the released 8B model has `2 d^2 + 6 d` parameters with `d = 4096`; the same shape at K3's hidden size of 7,168 would be about 102.8 million parameters. As arithmetic only: scaling the measured 1.9 ms by the parameter ratio (about 3.06) would give about 5.8 ms on this CPU, around 0.2% of a 3.018 s K3 step; this was not run and no such map exists for K3. (3) On K3 a latent step would be a full K3 step (about 3 s here), so any gain would have to come from fewer total steps, as the paper's own cost model says, and from the share of steps that are latent in a given run: 14% to 28% over the four HybridAgent q4 and q5 runs (Cycle 8). The latent share varied between seeds of the same question. (4) K3 step cost was not measured against position here (24-token prompt only); the earlier 1,000-token decode record (3.167 s per token with cache, one run) is consistent with little growth but used 28 cached experts.

**Not observed.** K3 on a longer context in this cycle; K3 attention against MLP split; the arithmetic of any trained feedback map on K3 (none exists); a latent-style readout on K3 states.

**Open for the creator.** (a) Tavily key on `k3` to run q1 to q3 with search (only the creator can place it; checked that none is set in root's login shell or profile files); (b) whether any of this feeds a decision for Clover K3, which is the creator's Direction, not mine.


## Cycle 10 (2026-10-22): q1 to q3 with the search tool on (budget 2,048, seed 0, batch 1, `k3`)

**Key handling.** The creator placed the Tavily key in the local clone's `HybridAgent/.env.example` (a tracked template file, now locally modified; it also holds two other key lines, lengths 24 and 26, which I did not use). I copied only the `TAVILY_API_KEY` line over ssh to `/root/.tavily.env` on `k3` (mode 600, root; first copy had a CR in it, stripped), sourced it inside `run_search.sh` only, and never printed the value (only its length, 58). `trace_hybrid.py` now has a `SEARCH=1` switch (search tool on, up to 5 search calls). Records `/root/runs/hybrid/trace-q1-search.json`, `-q2-search`, `-q3-search`; summary `oss/scratch-agents/hybrid-clean/search-summary.txt`. The key is not in any file of this repository.

| question | steps | latent steps (share) | spans | search calls (queries) | python calls | ended | wall | answer |
|---|---|---|---|---|---|---|---|---|
| q1 sound, 10 km (29) | 1,983 | 151 (8%) | 7 | 1 ("speed of sound in air at 20 degrees Celsius") | 1 | `eos` | 871 s | `\boxed{29}`, correct |
| q2 light Sun-Earth (8) | 3,673 | 191 (5%) | 10 | 2 ("speed of light in meters per second to km per minute"; "Average distance from Sun to Earth in kilometers") | 3 | `eos` | 1,724 s | `\boxed{8}`, correct |
| q3 milk vs water (30) | 1,816 | 140 (8%) | 5 | 1 ("density of whole milk at room temperature") | 0 | `eos` | 807 s | `\boxed{30000}`, **wrong** (expected 30) |

**What was observed.** The search tool worked: each search returned a block of 3,000 to 4,000 characters, and with it all three runs finished by themselves instead of stopping at the first request (Cycle 7). Two of three answers match the example file. In q3 the final text says "30000 grams" and rounds that to 30000; the question asks for the difference per litre, so the unit step (a litre against a cubic metre) was lost in the model's own reasoning; I read the tail of the output, not the whole run. The runs are much longer than the search-off ones (1,816 to 3,673 steps) and the latent share is lower: 5% to 8%, against 22% to 40% for the same questions up to the first search request in Cycle 7 and 14% to 28% for q4 and q5. Search-result tokens are not counted as steps (`count_tool_tokens` is false), so steps count only the model's own generation. One run per question; the lower latent share in these three runs is an observation of these runs, not an explanation.

**Per step.** Net step time text / latent (ms): q1 427.0 / 438.1, q2 462.3 / 452.4, q3 433.1 / 439.0: a latent step still costs the same as a text step; the step time is higher than in the search-off runs because the sequence is longer (search results are in the context). `hidden_states[-1]` equals the final-norm output exactly (0.0) in all three. Layer-35 output norm median text / latent: 1,674 / 2,359, 1,689 / 2,217, 1,747 / 2,480. Layer-16 outlier steps (relative update above 0.9): 1, 2 and 1 steps (ids 315; 15 and 220; 220). Exit readouts: q3 had 5 exits, 2 steps where only the LM-head says `</latent>` (the first time in these runs; minimum P at an exit 0.59) and 1 where only the cosine readout does; q1 and q2 none of either.

**Not observed.** A text-only control; whether the low latent share is a property of search runs in general (three runs); the cause of the q3 unit error; third seed.

**Housekeeping for the creator.** The key sits in a tracked file in your `HybridAgent` clone (`.env.example`, shown as modified). Move it to a git-ignored `.env` and restore the template (`git checkout -- .env.example`) before any commit or push from that clone; I have not changed that file.


## Cycle 11 (2026-10-22): q3 with search, second seed (seed 1; budget 2,048, `k3`; record `trace-q3-search-s1.json`, summary `q3-s1-summary.txt`)

Part of a queue run one job at a time (`run_all.sh` on `k3`: this run, then q4 and q5 with search, then text-only controls); nothing else was on the box.

| run | steps | latent steps (share) | spans | search calls | ended | wall | answer |
|---|---|---|---|---|---|---|---|
| q3 search seed 0 (Cycle 10) | 1,816 | 140 (8%) | 5 | 1 | `eos` | 807 s | 30000, wrong |
| q3 search seed 1 | 1,722 | 262 (15%) | 6: 41, 47, 25, 44, 35, 70 | 1 ("density of whole milk at room temperature") | `eos` | 764 s | `\boxed{30}`, **correct** |

**Observed.** The wrong answer of Cycle 10 is not repeated by the second seed: both seeds make the same one search, and the texts differ from character 139. In seed 1 the closing text reasons in grams ("30 grams would round to 30"), in seed 0 it ended on "30000 grams"; so for this question the outcome depends on the seed. The latent share is 15% here against 8% for seed 0 of the same question. Net step time text / latent 422.8 / 438.2 ms. `hidden_states[-1]` equals the final-norm output exactly (0.0). Exits: 6, minimum P(`</latent>`) 0.95; 2 steps where only the cosine readout says `</latent>`; none where only the LM head does. One outlier text step at layer 16 (id 16, the digit `1`).

**What this does and does not say.** Two seeds, one search result each; the answer difference is an observation of two samples, not a rate of error. It does say that a single run of a question cannot be read as the model's ability on it.


## Cycle 12 (2026-10-22): q4 and q5 with the search tool on (seed 0, budget 2,048, `k3`; records `trace-q4-search.json`, `trace-q5-search.json`, summary `q45-search-summary.txt`)

| run | steps | latent steps (share) | spans | search calls | python calls | ended | wall | answer |
|---|---|---|---|---|---|---|---|---|
| q4 search on | 1,243 | 329 (26%) | 13 | **0** | 0 | `eos` | 570 s | `\boxed{24}`, correct |
| q5 search on | 1,035 | 240 (23%) | 8 | **0** | 3 | `eos` | 468 s | `(-12, -8, 36)`, correct |

**Observed.** With the search tool available, the model made no search call on either question (it computed, and on q5 used the python tool three times). The two runs have the same step counts, latent steps, span lengths (13 spans 102, 13, 6, 4, 24, 35, 71, 23, 9, 18, 11, 8, 5 for q4; 8 spans for q5), python calls and layer-35 medians as the search-off runs of seed 0 (Cycles 5 and 7). So a second, separate process run with the same seed on `k3` reproduced the first one in every count I compared (I did not diff the full texts; the answers and the closing text read the same). Practical meaning: on this box, one seed is one trajectory; differences between runs in earlier cycles come from the seed, not from run-to-run noise. And the model uses the search tool when it needs a fact (q1 to q3) and not when it can compute (q4, q5): that is a choice made inside its own text, visible as `<search>` after a `</think>`.

**Per step.** Net step time text / latent (ms): q4 430.5 / 424.8, q5 425.5 / 415.9 (earlier runs 422.3 / 416.7 and 422.7 / 419.7: the same within about 2%). Wall 570 s and 468 s against 553 s and 462 s.


## Cycle 13 (2026-10-22): text-only control with the same weights (the `<latent>` token blocked), q4 and q5, seeds 0 and 1 (budget 4,096, search off, python on, `k3`; records `trace-q4-nolat*.json`, `trace-q5-nolat*.json`, summary `control-summary.txt`)

**What the control is.** In `trace_hybrid.py` the switch `LATENT=0` sets the engine's own text-mode mask for `<latent>` (`_blocked_masks["in"]`), so the same HybridAgent-8B-Agentic weights can only write text. Nothing in the engine file is edited. This answers the open item "a text-only control with the same weights" from Cycle 2 in the only way the released code allows, and it has a built-in limit: the model was trained to use latent spans, so with them blocked it is run off the distribution it was trained on. It is not the paper's text-only baseline (a different trained model).

| question and seed | latent run: steps (latent) | text-only run: steps | ratio text-only / latent | text-only ended | answer text-only |
|---|---|---|---|---|---|
| q4 seed 0 | 1,243 (329) | 241 | not valid | `tool_cap`: it wrote `<search>how many trailing zeros in 100!</search>` and search was off in this run | none |
| q4 seed 1 | 1,208 (172) | 494 | 0.41 | `eos` | `\boxed{24}`, correct |
| q5 seed 0 | 1,035 (240) | 1,316 | 1.27 | `eos` | `\boxed{(-12,-8,36)}`, correct |
| q5 seed 1 | 1,148 (324) | 2,870 | 2.50 | `eos` | `\boxed{(-12, -8, 36)}`, correct |

Wall time follows steps (ratios 0.39, 1.22, 2.51; one step is about 0.42 to 0.45 s in both modes). The q4 seed-0 control was stopped by a search request, so it is not a comparison; it is re-run with search on (next cycle).

**Reading (labelled as such).** On the three valid pairs the text-only runs took 0.41, 1.27 and 2.50 times the steps of the latent runs of the same question and seed, so the same weights wrote fewer steps with latent spans on two of three pairs and more on one. Summed over those three pairs: 3,391 steps with latent spans against 4,680 text-only (ratio 1.38). Against that, one sample per pair, the ratio ranges over a factor of six, the text-only runs are off-policy, and the answers are right in all valid runs, so steps are the only difference measured. The paper's claim is a step saving; this control shows it can go either way for one question and one seed, and it is not enough to put a number on it.

**For Clover K3.** A latent step costs what a text step costs here (Cycles 3 to 12) and, as arithmetic only, a K3 decode step is about 3 s (Cycle 9). The saving available from a latent-span mechanism is therefore set by the steps it removes in a given trajectory, which in these 4 pairs varied from fewer (0.41) to more (2.50) than the same weights without it.


## Cycle 14 (2026-10-22): q4 seed-0 text-only control re-run with the search tool on (`trace-q4-nolat-search.json`, `control-q4s0.txt`)

The Cycle 13 control for q4 seed 0 had been stopped by a search request while search was off. Re-run with `SEARCH=1 LATENT=0`: **1,519 steps**, one search call ("how many trailing zeros in 100!"), no python call, ended by itself, answer `\boxed{24}`, correct, wall 655 s. The latent run of the same question and seed had 1,243 steps (329 latent) and no search call, so this pair is 1.22 text-only to latent.

**All four pairs, same weights, same seed, `<latent>` allowed against blocked.**

| question and seed | steps with latent allowed (latent steps) | steps with latent blocked | ratio blocked / allowed | both answers correct |
|---|---|---|---|---|
| q4 seed 0 | 1,243 (329) | 1,519 | 1.22 | yes |
| q4 seed 1 | 1,208 (172) | 494 | 0.41 | yes |
| q5 seed 0 | 1,035 (240) | 1,316 | 1.27 | yes |
| q5 seed 1 | 1,148 (324) | 2,870 | 2.50 | yes |
| total | 4,634 | 6,199 | 1.34 | |

**Reading (labelled as such).** Over the four pairs the same weights took 34% more steps with `<latent>` blocked than allowed (6,199 against 4,634), and 3 of 4 pairs went the same way, but the pair ratios run from 0.41 to 2.50, so one question and one seed cannot be read as the saving. Four pairs, one run each; the blocked runs are off the training distribution; all eight runs reached a correct answer, so steps (and wall time, which follows steps at 0.42 to 0.45 s each) are the only difference measured here. The paper's step saving is a different comparison (a separately trained text agent); this is the closest the released code allows.

**For Clover K3 (arithmetic from measured numbers, not a measurement).** One K3 decode step is about 3.0 s on this box (Cycle 9) and a latent step costs a full step. If the 34% step difference observed here carried over, 4,634 against 6,199 steps would be a difference of 1,565 steps, about 78 minutes at 3.0 s per step; but the spread over the four pairs (0.41 to 2.50) means the sign is not secure for a single run, and no latent-style mechanism exists for K3 (it needs a trained feedback map and a post-trained backbone; Cycle 9).

**Not observed.** A text-only run of the base model without the HybridAgent training; the same pairs on other questions (q1 to q3 with latent blocked); more seeds per pair.


## Cycle 15 (2026-10-22): q1 to q3 with `<latent>` blocked and search on (seed 0, budget 4,096, `k3`; records `trace-q1-nolat-search.json`, `trace-q2-nolat-search.json`, `trace-q3-nolat-search.json`, summaries `cycle15-summ.txt`, `cycle15-lat.txt`)

Same weights, same seed 0, same question as the Cycle 10 runs; the only change is `LATENT=0` (and budget 4,096 against 2,048 in Cycle 10; the q3 control ended at the budget, so a larger budget could matter only for the cases that did not finish). Run one job at a time, nothing else on the box.

| question | steps with latent allowed (latent steps) | steps with latent blocked | ratio blocked / allowed | blocked: search / python calls | blocked ended | wall allowed / blocked | answers |
|---|---|---|---|---|---|---|---|
| q1 (29) | 1,983 (151) | 872 | 0.44 | 0 / 1 | `eos` | 871 s / 369 s | both `\boxed{29}`, correct |
| q2 (8) | 3,673 (191) | 2,589 | 0.70 | 2 / 0 | `eos` | 1,724 s / 1,131 s | both `\boxed{8}`, correct |
| q3 (30) | 1,816 (140) | 5,175 | at least 2.85 | 1 / 0 | **`budget`** (no answer block) | 807 s / 2,516 s | allowed: 30000, wrong; blocked: none |

Net step time with latent blocked (text only): q1 419.5 ms, q2 434.1 ms, q3 484.9 ms (q3 is higher because the sequence reached 5,175 steps plus search text). Layer 16 had 1 outlier step in each of the three runs (relative update above 0.9; ids 220, 13, 16). Latent blocked: 0 latent steps and 0 exits in all three, as expected. Search calls in the latent-allowed runs: 1, 2, 1; blocked: 0, 2, 1 (q1 answered without search; it used python once instead).

**Observed.** With latent blocked the same weights wrote fewer steps on q1 and q2 and many more on q3, where the run reached the 4,096 budget without an answer (steps exceed the budget because search text and python output are not counted, as in Cycle 10; I did not read the q3 text to see why it kept going).

**All seed-0 and seed-1 pairs so far (seven pairs; q3 seed 1 allowed has no blocked partner):** q4 s0 1,243 / 1,519; q4 s1 1,208 / 494; q5 s0 1,035 / 1,316; q5 s1 1,148 / 2,870; q1 s0 1,983 / 872; q2 s0 3,673 / 2,589; q3 s0 1,816 / 5,175 (not finished). Summed: allowed 12,106, blocked 14,835 (ratio 1.23, where the q3 blocked run is a lower bound). By pair the ratio runs from 0.41 to at least 2.85; 4 pairs above 1 and 3 below. So the sign differs between questions, and the pooled number moves from 1.34 (Cycle 14) to 1.23 when three more pairs are added.

**Reading (labelled as such).** The step saving seen in Cycle 14 does not hold up as a rule once q1 and q2 are added: it depends on the question and the seed, with q1 and q2 faster without latent spans and q3 much slower. One run per pair, off-policy control, q3 truncated by the budget; correctness is equal on q1 and q2 and unknown on the q3 control.

**Not observed.** A seed beyond 0 for q1 to q3 with latent blocked; why the q3 blocked run did not finish; a base-model control.


## Cycle 16 (2026-10-22): seed 2 for q4 and q5, latent allowed against blocked (search on, budget 4,096, `k3`; records `trace-q4-s2.json`, `trace-q4-nolat-s2.json`, `trace-q5-s2.json`, `trace-q5-nolat-s2.json`, summary `cycle16-summ.txt`)

| question and seed | steps with latent allowed (latent steps) | steps with latent blocked | ratio blocked / allowed | blocked ended | answers |
|---|---|---|---|---|---|
| q4 seed 2 | 471 (64, 14%) | 502 | 1.07 | `eos` | both `\boxed{24}`, correct |
| q5 seed 2 | 1,894 (523, 28%) | 4,097 | at least 2.16 | **`budget`**, no answer block | allowed: `(-12, -8, 36)`, correct; blocked: none |

Wall time follows steps: 206 s / 213 s for q4, 890 s / 1,892 s for q5.

**All nine pairs so far** (q4 s0 to s2, q5 s0 to s2, q1 s0, q2 s0, q3 s0): latent allowed 14,471 steps, latent blocked 19,434 steps, ratio 1.34 (a lower bound: q3 s0 and q5 s2 blocked runs stopped at the budget). By pair, 6 are above 1 and 3 below (0.41, 0.44, 0.70); the range is 0.41 to at least 2.85. The pooled number was 1.34 (four pairs), 1.23 (seven pairs), 1.34 (nine pairs): it does not settle, and it is driven by the two pairs that did not finish without latent spans.

**Reading (labelled as such).** The two blocked runs that ran out of budget are both cases where the same weights, with `<latent>` allowed, finished correctly in fewer steps (1,816 and 1,894). The blocked runs are off the training distribution, one run per pair, so the pooled ratio is a description of these nine pairs and not a rate. Correctness: every blocked run that finished is correct; the two that did not finish have no answer.

**Not observed.** A base-model control; a third seed for q1 to q3; why the blocked q3 and q5 s2 runs keep going; any K3-side run of a latent mechanism (none exists).


## Cycle 17 (2026-10-22): the paper's own metrics, read off our nine pairs; and a correction to the step counts of Cycles 10 to 16 (script `cycle17_metrics.py`, output `cycle17-metrics.txt`, no new run)

**Direction.** The creator asked to observe the other efficiency claims of the paper (35.5% to 54.7% fewer tokens, up to 54% fewer generation steps, 1.8x to 2.6x faster) in our own reality, one step at a time, without treating a different number on our hardware as a statement about theirs.

**What their protocol counts (read from the paper text, Sections 5.2, 6.3, Appendix G.2 and H.3; and from `engine.py`).**
- "Text tokens" are tokens written in text mode (Table 7: 302 against 596 per instance at 8B, 284 against 497 at 14B, MATH500 and AIME, "matched same-size text agents"). "Generation steps" are text tokens plus latent steps; the generation budget counts both (`generation_steps` in the engine, 4,096 for math and search, 24,000 for agentic tasks).
- Their cost model (Table 9) charges every text or latent step one full backbone forward pass and **excludes prompt prefill, context-dependent attention and tool execution**. Observations are tokenised and appended to the working memory; they are not generation steps.
- Their comparator is a separately trained text agent (GRPO or ARPO after text SFT, same backbone, same data). Ours is the same weights with `<latent>` blocked, so the two comparisons are different experiments.
- Their inference setting has latent spans of 16 to 256 steps, tool calls up to 10 per trajectory, evaluation temperature 0.6, top-p 0.95, top-k 20, five seeds, H100 GPUs, 11 benchmarks. Ours (released defaults, `trace_hybrid.py`): spans 4 to 200, 5 tool calls, repetition penalty 1.1 (not in the paper text), batch 1, CPU, five hand-written questions.

**Correction (mine).** The `steps` column of Cycles 3 to 16 counted every forward pass recorded by the tracer, including the forward passes that feed a tool observation into the cache one token at a time (`FEED` mode in the released engine). Cycle 10 wrote that search-result tokens are "not counted as steps"; that is true of the budget (`count_tool_tokens` false) and false of my `steps` column. Example: q2 with search is 3,673 forward passes = 1,349 generation steps (1,158 text, 191 latent) + 2,324 observation-feed passes. So the pooled ratios 1.34 / 1.23 / 1.34 of Cycles 14 to 16 are ratios of all forward passes, a mixture of the two things. Cycle 16 also said "budget 4,096" for all runs; the latent-allowed runs had budget 2,048 (none reached it: at most 1,798 generation steps) and the blocked runs 4,096 (two reached it). Earlier rows are left as they were; this section replaces their reading.

**Per pair, same question and seed, latent allowed against blocked (generation steps = text + latent; feed = observation passes).**

| pair | text tokens allowed / blocked | generation steps allowed / blocked | feed passes allowed / blocked | blocked ended |
|---|---|---|---|---|
| q4 s0 | 914 / 525 | 1,243 / 525 | 0 / 994 | `eos` |
| q4 s1 | 803 / 486 | 975 / 486 | 233 / 8 | `eos` |
| q4 s2 | 399 / 494 | 463 / 494 | 8 / 8 | `eos` |
| q5 s0 | 748 / 1,289 | 988 / 1,289 | 47 / 27 | `eos` |
| q5 s1 | 803 / 2,785 | 1,127 / 2,785 | 21 / 85 | `eos` |
| q5 s2 | 1,275 / 4,096 | 1,798 / 4,096 | 96 / 0 | **`budget`** |
| q1 s0 | 503 / 816 | 654 / 816 | 1,329 / 56 | `eos` |
| q2 s0 | 1,158 / 607 | 1,349 / 607 | 2,324 / 1,982 | `eos` |
| q3 s0 | 428 / 4,096 | 568 / 4,096 | 1,248 / 1,078 | **`budget`** |

Sums over nine pairs: text tokens 7,031 / 15,194; generation steps 9,165 / 15,194; observation-feed passes 5,306 / 4,238; all forward passes 14,471 / 19,434. Generation-step time (sum of recorded step times, tool time excluded) 4,205 s / 6,970 s.

| reading of the same nine pairs | latent allowed against blocked |
|---|---|
| text tokens, all nine pairs | 53.7% fewer (lower bound: two blocked runs stopped at their budget with no answer) |
| generation steps, all nine pairs | 39.7% fewer (same lower bound) |
| all forward passes including observation feeding, all nine | 25.5% fewer |
| text tokens, the seven pairs where both runs finished | 23.9% fewer (5,328 / 7,002) |
| generation steps, the seven finished pairs | 2.9% fewer (6,799 / 7,002) |
| all forward passes, the seven finished pairs | 5.9% **more** (10,761 / 10,162) |

Per pair the direction is the same for text tokens and generation steps: latent allowed is lower in 6 of 9 pairs (q4 s2, q5 s0 to s2, q1, q3) and higher in 3 (q4 s0, q4 s1, q2). In q4 s0 and q2 the blocked run took a different path (blocked q4 s0 made a search call of 994 observation tokens; the allowed run made none).

**What this shows, and what it does not.**
- On the paper's own two metrics, our nine pairs land at 53.7% fewer text tokens and 39.7% fewer generation steps when the two truncated blocked runs are included as they stand, and at 23.9% and 2.9% when only the finished pairs are used. The paper's 43% to 49% text-token figure (Table 7) sits between those two readings; its "up to 54%" generation-step figure is above both of ours in the finished-only reading. These are numbers from different protocols (their trained text agent against our blocked copy of the same model; benchmarks against five questions), so this is a description of how the mechanism behaves in our reality, not a comparison of claims.
- The released engine feeds tool observations one token per forward pass: 5,306 of the 14,471 forward passes of the latent-allowed runs (36.7%) and 2,338 s of their step time are observation feeding. The paper's cost model leaves this out (prefill and tool execution are excluded). With feeding included, the finished pairs show no saving in forward passes. For a serving system that prefills observations in parallel (K3 would), the relevant comparison is generation steps.
- Not measured: the speedup claim (1.8x to 2.6x) in wall-clock time on H100 with a trained text agent as comparator; five seeds per benchmark; the 24,000-step agentic budget.

**For Clover K3 (arithmetic from measured numbers, not a K3 measurement).** One K3 decode step is about 3.0 s (Cycle 9). Generation-step differences: 6,029 steps over nine pairs when the truncated blocked runs are taken at their cap (about 5.0 hours, a lower bound), 203 steps over the seven finished pairs (about 10 minutes). The sign flips by question and seed, and almost the whole difference sits in the two pairs where the blocked run never reached an answer. K3 has no latent mechanism; it would need a trained feedback map and a post-trained backbone (Cycle 9).

**Next (Cycle 18, started).** Run the mechanism with the paper's own latent bounds (16 to 256 steps per span) in our engine on q4 s0, q5 s0 and q1 s0 against the existing blocked runs, to see how the span length they chose changes steps and the text-token share in our reality. `trace_hybrid.py` gets `MINLAT` and `MAXLAT` environment switches (defaults 4 and 200, unchanged).


## Cycle 18 (2026-10-22): the paper's latent-span bounds (16 to 256 steps) in our engine, q4, q5, q1 at seed 0 (search on, budget 2,048, `k3`; records `trace-q4-b16-s0.json`, `trace-q5-b16-s0.json`, `trace-q1-b16-s0.json`, output `cycle18-metrics.txt`)

Only `MINLAT=16 MAXLAT=256` changed against the default-bound runs of the same question and seed (`trace_hybrid.py` switches; defaults 4 and 200 unchanged). The blocked runs of Cycles 13 to 15 are the reference. All three runs ended by `eos`. **Correction (Cycle 22): this sentence originally said all three answers were correct. I had not read the q1 answer: it is `\boxed{28}` (the model computed 10,000 / 343 as 28.03 after an arithmetic slip), the correct value is 29. q4 (24) and q5 ((-12, -8, 36)) were correct.** The q4 run took a different path from the default-bound run (one python call against none), as any change of the bounds can do to a sampled trajectory.

| pair (seed 0) | setting | text tokens | latent steps (spans) | generation steps | observation-feed passes | wall |
|---|---|---|---|---|---|---|
| q4 | bounds 16 to 256 | 645 | 435 (4: 102, 130, 116, 87) | 1,080 | 7 | 510 s |
| q4 | default 4 to 200 | 914 | 329 (13) | 1,243 | 0 | 570 s |
| q4 | latent blocked | 525 | 0 | 525 | 994 | 655 s |
| q5 | bounds 16 to 256 | 717 | 718 (20, median about 27, one of 156) | 1,435 | 0 | 695 s |
| q5 | default | 748 | 240 (8) | 988 | 47 | 468 s |
| q5 | latent blocked | 1,289 | 0 | 1,289 | 27 | 562 s |
| q1 | bounds 16 to 256 | 531 | 194 (7) | 725 | 1,309 | 897 s |
| q1 | default | 503 | 151 (7) | 654 | 1,329 | 871 s |
| q1 | latent blocked | 816 | 0 | 816 | 56 | 369 s |

Sums over the three pairs: text tokens 1,893 (bounds 16 to 256), 2,165 (default), 2,630 (blocked); generation steps 3,240, 2,885, 2,630.

**Observed.** With the paper's bounds the latent share of generation rises (q4 40%, q5 50%, q1 27%, against 26%, 24% and 23% with the default bounds) and the text-token count falls: 28.0% fewer text tokens than the blocked runs (12.6% fewer than the default-bound runs). Generation steps go the other way for these three pairs: 23.2% more than the blocked runs (3,240 against 2,630) and 12.3% more than the default-bound runs, because each latent step is a full forward pass. Per pair, text tokens are below the blocked run in q5 and q1 and above it in q4; generation steps are above the blocked run in q4 and q5 and below it in q1.

**What this does and does not say.** Three pairs, one seed each, one comparator (our blocked copy of the same weights, not their trained text agent), five-question set instead of their benchmarks, batch 1 on CPU. In this reality the 16-step floor moves work from text into latent steps without lowering the number of forward passes; the paper's cost model charges latent steps as forward passes too, so the saving it reports must come from the text agent's longer trajectories (596 against 302 text tokens in its Table 7), which our blocked copy does not reproduce. I did not test that reading; it is an inference from their table and ours. The text-token share is what the latent span changes in our runs; the generation-step total depends on how many latent steps it costs.

**For Clover K3.** A latent mechanism trades text tokens for latent forward passes at one K3 decode step (about 3.0 s, Cycle 9) each. Whether it saves time then depends on the number of latent steps per text token removed, which in our three pairs is above 1 (q4: 435 latent steps for 269 text tokens fewer than the default run's 914; q5: 718 for 31 fewer; q1: none saved). It is a quantity a K3 training plan would have to set (the span floor and ceiling) and measure.

**Not observed.** Other seeds and questions with the 16 to 256 bounds; their trained text agent as comparator; why q4 took a different path under different bounds (not examined).


## Cycle 19 (2026-10-22): what the latent steps read as (paper Fig 2, Fig 8, Appendix F), from the 17 existing latent-allowed traces (script `cycle19_readout.py`, output `cycle19-readout.txt`; no new run)

**Direction.** The creator asked that HybridAgent be fully observed and tested before any K3 training plan is written. This cycle looks at the interpretability claim: every latent input `R(h)` can be read as its nearest vocabulary embedding (Eq. 5). The traces already hold, for every latent step, the nearest token (`link_nn_id`) and its cosine (`link_nn_cos`).

**Method (their definitions, our data).** Latent span = consecutive latent steps between `<latent>` and `</latent>`. Readability (Eq. 19) = mean nearest-token cosine over the span. Noise reference = mean nearest-token cosine of 2,000 Gaussian vectors against the same embedding matrix: **0.069** (max 0.092). Context specificity (Eq. 20) = Jaccard of the span's pseudo-text content words with its own question, divided by the mean of that Jaccard with the other questions' words; defined only when that mean is positive. Our set is five questions, theirs 200 questions over six benchmarks.

**Observed.**
- 131 latent spans, 4,160 latent steps, five questions, seed 0 to 2, search on or off, default and 16-to-256 bounds.
- Readability per span: mean 0.844, median 0.863, range 0.600 to 0.990. All 131 spans are above the noise reference (0.069). Per latent step: median cosine 0.963, 62% of steps above 0.9, 93% above 0.5.
- The pseudo-text reads as sentences about the problem. Examples (nearest-token readout of single spans): "Let me start by searching for the speed of light in minutes per minute." (q2); "Wait, maybe I should verify with a example. Let me check 100!" (q4); "Therefore, the equation of the new circle should be (x - 6)² + (y - 4)² = 16." (q5); "For milk, if I use the average of of 1.03 g cm³, then 1 liters would be 1.03 * 1000 = 1030 grams." (q3). The garbled characters in the console output were encoding of `²` and `³`, not data.
- Specificity cases: 17 spans have a defined, positive score and all 17 are above 1 (median 4.52). 46 spans share at least one content word with their own question and none with any other question, so the score is undefined (denominator zero) although the overlap is entirely with the right question. 68 spans share no content word with any of the five questions (mostly arithmetic and notation). So 63 of 131 spans (48%) overlap their own question and none overlap a different one in the 46 cases where the score is undefined.

**What this shows and does not.** In our runs the latent steps are, step by step, close to a vocabulary token (their stated property) and the readout is coherent text about the right problem; the five-question set is too small and too narrow in vocabulary for the specificity score to be defined for most spans, so I report the three counts instead of a single number. I did not observe whether the pseudo-text is what the model *uses* (their causal interventions, Table 5), which is the next kind of observation. The readout is a property of a single trajectory per file; I did not compare spans of the same question across seeds.

**For Clover K3 (reading, not a measurement on K3).** The readout requires the latent input to sit near a vocabulary embedding (mean cosine 0.84 here). A K3 latent mechanism that is meant to be auditable would need the same property from its own feedback map; whether K3's embedding space gives it is not known.

**Not observed.** Causal interventions (noise, replacement, permutation, one-step spans; only "block entry" was run, Cycles 13 to 18); the working-memory equivalence of Theorem A.1; the exit rule across many spans; more seeds for the 16 to 256 bounds; their trained text agent as a comparator.


## Cycle 21 (2026-10-22): the exit rule over every latent span (paper Eq. 5, Appendix B.2; script `cycle21_exit.py`, output `cycle21-exit.txt`; the 17 existing latent-allowed traces, no new run)

**What is read.** For each span, the step on which it closed: the LM-head probability of `</latent>`, the cosine of `R(h)` to the `</latent>` embedding, and whether the two readouts agree. The released engine closes a span when the LM-head argmax is `</latent>` after `min_latent_steps` (paper text: the cosine nearest token), or forces it at `max_latent_steps`.

**Observed (131 spans, 4,160 continuing latent steps).**
- All 131 spans ended by the model's own exit; 0 were forced at the maximum (the longest span was 156 steps against the 200 and 256 maxima).
- Span length (continuing steps before the exit step): median 24, mean 31.8, min 4, max 156. Histogram (<=8, 9 to 16, 17 to 32, 33 to 64, 65 to 128, 129+): 17, 20, 48, 33, 11, 2.
- At the exit step the LM-head P(`</latent>`) has median 0.986 (min 0.350; above 0.5 in 95% of exits, above 0.9 in 74%). The cosine to the `</latent>` embedding has median 0.998 (min 0.219).
- The two readouts agree on 129 of 131 exits (both name `</latent>`); on 2 the LM head names it and the cosine nearest token does not; on none does only the cosine name it. So the engine's LM-head rule and the paper's cosine rule gave the same exit in 98% of the spans.
- Continuing steps: P(`</latent>`) is above 0.1 on 1.03% of them and above 0.5 on 0.31% (13 steps), maximum 0.999. 19 steps had the LM-head argmax on `</latent>` while the span was still shorter than that run's minimum (4, or 16 in the paper-bounds runs) and were held open by `min_latent_steps`. The cosine to `</latent>` is median 0.052 on continuing steps (max 0.999).

**What this shows and does not.** In our runs the exit is a sharp event, not a threshold on a slowly rising score: the probability sits near zero for most steps and near one at the exit. The minimum span length is active (19 steps held open); the maximum never was. Nothing here says whether a span ends at a *good* point, only how decisively it ends. One checkpoint, five questions, 131 spans.

**For Clover K3 (reading).** A K3 latent mechanism would need an exit signal with the same property (a decisive anchor decision). The span floor changed behaviour in our runs (19 held-open steps), so it is a real setting and not only a safety bound.

**Not observed.** Why the 2 LM-only exits differ in the cosine readout; exits under the one-step rule (Cycle 20) and under interventions on the latent vector (next cycles).


## Cycle 20 (2026-10-22): one-step latent spans (script `cmp_runs.py`, output `cycle20-one.txt`; released engine with `min_latent_steps=1 max_latent_steps=1`, search on, budget 2,048, seed 0, questions q4, q5, q1)

**What was run.** Each `<latent>` span is closed by force after exactly one latent step, so the continuous channel carries one vector per span. Everything else is as in the default-bound runs. Three questions, one seed each; the default-bound (4 to 200) and blocked (`<latent>` suppressed) runs for the same question and seed are the comparison.

| question | default-bound: generation steps (text + latent, spans) | one-step: generation steps (text + latent, spans) | blocked: generation steps | answer, all three |
|---|---|---|---|---|
| q4 | 1,243 (914 + 329, 13 spans) | 1,115 (1,101 + 14, 14 spans) | 525 | 24 |
| q5 | 988 (748 + 240, 8 spans) | 1,419 (1,393 + 26, 26 spans) | 1,289 | (-12, -8, 36) |
| q1 | 654 (503 + 151, 7 spans) | 502 (496 + 6, 6 spans) | 816 | 29 |
| sum | 2,885 (2,165 + 720) | 3,036 (2,990 + 46) | 2,630 | |

**Observed.**
- All spans under the one-step rule ended by force (14, 26, 6), none by the model's anchor; latent steps were 1.2%, 1.8% and 1.2% of generation steps.
- All nine runs finished (`eos`) and produced the same final answer on each question.
- Summed over the three questions the one-step runs used 2,990 text tokens against 2,165 with the default bounds and 2,630 with `<latent>` blocked; generation steps 3,036, 2,885 and 2,630. The ordering by question is not uniform: on q1 one-step is lower than both others (502 against 654 and 816), on q4 it is between them (1,115 against 1,243 and 525), on q5 it is above both (1,419 against 988 and 1,289).
- The model opened as many spans under the one-step rule as under the default one on q4 (14 against 13) and q1 (6 against 7), and more on q5 (26 against 8).
- Tool use differs between the runs (for example q4 default: no python or search call recorded; q4 one-step: one python call), so trajectories diverge early and the step counts are not a controlled comparison of the span rule alone.

**What this shows and does not.** In our reality, with one seed per question, trimming each span to one step removed the text-token reduction seen with the default bounds (2,165 against 2,990) while the answers stayed the same. It does not show that the span length is the cause, because the trajectories diverge on tool calls and one seed per question cannot separate that from run-to-run variation (Cycle 16 and 17 differences between seeds were of the same order on single questions). Three questions, one seed, one checkpoint, answers compared by the final boxed value only.

**For Clover K3 (reading).** Whatever the continuous channel does, in these runs it was the longer spans (median 13 to 27 continuing steps) that coincided with fewer text tokens; the same number of span openings with one step each did not. Cycle 22 (paper bounds 16 to 256, seeds 1 and 2) will give the other end of the span-length axis.

**Not observed.** Spans between 1 and 4 steps; a second seed of the one-step rule.

## Queue started after Cycle 21 (2026-10-22)
Tracer now has `INTERV=noise:<d>|random|nearest` (applied to the latent vector after its readout; the readout fields in the trace describe the vector before the change, `iv_rel` records `||new - z|| / ||z||`) and `CACHECHECK=1` (records every real forward step's input embedding and final-layer hidden, then runs one full-prefix forward over prompt + all inputs and compares final-layer hidden states and LM-head argmax; writes `<trace>.cache.json`). Defaults unchanged. Detached queue on `k3` (`run_c22.sh`, one model job at a time): Cycle 24 first (q4 seed 2 and q1 seed 0 with `CACHECHECK=1`), then Cycle 23 (q4 and q5 at seed 0 with `noise:0.5`, `noise:2.0`, `random`, `nearest`), then Cycle 22 (paper bounds 16 to 256, seeds 1 and 2, q4, q5, q1). 16 model runs, total about 3 hours.


## Cycle 24 (2026-10-22): cached continuation against a full-prefix forward (paper Theorem A.1; tracer `CACHECHECK=1`, outputs `trace-q4-cc2-s2.json.cache.json`, `trace-q1-cc-s0.json.cache.json`, `cycle24-cmp.txt`)

**What was run.** The released engine runs normally (KV cache, one position per forward pass, latent vectors fed back as input embeddings). The tracer records every real forward step's input embedding and the final-layer hidden state it produced. After the run, one forward pass over the whole sequence (prompt embeddings, then every recorded input embedding, no cache) gives a hidden state at every position. The two hidden states per position are compared. To tell rounding from a difference in the mechanism, the same inputs also went through a float32 copy of the model and both bf16 results were compared with it. The model is bf16 on the CPU.

**Observed.**

| run | positions compared | median rel. difference, cached vs full-prefix (bf16) | max rel. difference | LM-head argmax same | cached vs fp32: median, argmax same | full-prefix vs fp32: median, argmax same |
|---|---|---|---|---|---|---|
| q4 seed 2 (3 latent spans, 64 latent steps) | 472 | 0.0123 | 0.180 (token step) | 468 of 472 | 0.01157, 469 | 0.01145, 468 |
| q1 seed 0 (8 spans, 202 latent steps, search + python) | 2,406 | 0.0124 | 0.175 (token step) | 2,390 of 2,406 | 0.01218, 2,393 | 0.01238, 2,395 |

- The cached result is as far from the float32 result as the full-prefix result is (median 0.0116 against 0.0115 on q4, 0.0122 against 0.0124 on q1). The cached-versus-full difference is the same size as each one's distance to float32.
- The maximum relative difference per position class: latent positions 0.029 (q4) and 0.034 (q1); token positions 0.180 and 0.175 (token positions include the tool-observation tokens fed one per pass). Against float32 the largest per-position values were 0.129 (cached) and 0.097 (full-prefix) on q4, 0.319 and 0.239 on q1, so a few token positions are far from float32 in both.
- Minimum cosine between the cached and full-prefix hidden states: 0.990 (q4), 0.985 (q1).
- The q4 run reproduced the earlier seed-2 run exactly (463 generation steps, 3 spans, answer 24). The q1 run at seed 0 did not reproduce the earlier one (934 generation steps and 8 spans against 654 and 7; same answer 29; this run made 3 python calls and 1 search, the earlier one 1 and 1), so live search results or tool timing probably differ between runs; the tool results were not saved, so this is not attributed further.

**What this shows and does not.** The recorded latent steps (fed back as vectors) and the text and observation tokens sit in the cache so that a full-prefix recomputation reproduces the hidden states to the size of bf16 rounding; nothing in these two runs separates the two beyond the noise floor of the precision. It does not show bit equality, because the bf16 test cannot: that would need float32 cached against float32 full-prefix, which was not run. It does not cover positions the engine masks (pad slots), because neither run had one mid-sequence. Two runs, one checkpoint, one precision.

**For Clover K3 (reading).** The property the paper states as a theorem (the cache holds the same state as recomputing everything) held within rounding for a mixed latent and text sequence in a real run. For K3 this is a test to repeat on its own engine with its own precision (the recorded run already has bit-identical replays, so a float64 or same-kernel comparison is possible there); the comparison method above (an independent higher-precision reference to size the noise floor) is reusable.

**Not observed.** Float32 against float32; pad positions; sequences with many more spans than 8.


## Cycle 23 (2026-10-22): interventions on the latent vector (tracer `INTERV`, `cmp_runs.py`, output `cycle23-iv.txt`; q4 and q5, seed 0, default bounds 4 to 200, search on, budget 2,048)

**What was run.** Every vector the HybridLink produces during a span is changed before it is fed back as the next input embedding. `noise:d`: add Gaussian noise of norm `d` times the vector's norm (relative change exactly `d`). `random`: replace with a uniformly sampled token embedding rescaled to the vector's norm. `nearest`: replace with the nearest token embedding (by cosine) rescaled to the vector's norm. The readouts in the trace (nearest token, cosine to `</latent>`) describe the vector before the change. The exit decision is the model's own, taken afterwards from the LM head. A separate seeded generator drives the interventions. The comparison is the unmodified seed-0 run of the same question.

**Observed.**

| run | generation steps | text + latent (latent share) | spans (median, max steps) | end | final answer |
|---|---|---|---|---|---|
| q4 unmodified | 1,243 | 914 + 329 (26%) | 13 (13, 102) | eos | 24 |
| q4 noise 0.5 (rel. change 0.50) | 481 | 455 + 26 (5%) | 2 (13, 16) | eos | 24 |
| q4 noise 2.0 (rel. change 2.00) | 422 | 328 + 94 (22%) | 3 (25, 60) | eos | 24 |
| q4 random token (rel. change 1.39) | 517 | 482 + 35 (7%) | 7 (4, 9) | eos | 24 |
| q4 nearest token (rel. change 0.42) | 888 | 574 + 314 (35%) | 7 (23, 127) | eos | 24 |
| q5 unmodified | 988 | 748 + 240 (24%) | 8 (27, 56) | eos | (-12, -8, 36) |
| q5 noise 0.5 | 848 | 620 + 228 (27%) | 5 (48, 65) | eos | (-12, -8, 36) |
| q5 noise 2.0 | 2,048 | 1,620 + 428 (21%) | 31 (7, 51) | budget | none |
| q5 random token (rel. change 1.39) | 1,406 | 1,330 + 76 (5%) | 18 (4, 6) | eos | (-12,-8,36) |
| q5 nearest token (rel. change 0.38) | 1,150 | 869 + 281 (24%) | 4 (62, 154) | eos | (-12, -8, 36) |

- Seven of the eight modified runs ended with the same final answer as the unmodified run; the eighth (q5 noise 2.0) used the whole budget with 31 spans and gave no answer.
- All spans in the eight runs ended by the model's own exit; none was forced.
- Replacing the vector with a random token embedding (almost orthogonal to the original, relative change 1.39) gave the shortest spans in both questions (median 4, which is the minimum span length; max 9 and 6) and the lowest latent share (7% and 5%). The nearest-token replacement gave spans as long as or longer than the unmodified run (median 23 against 13 on q4, 62 against 27 on q5).
- Every modified run diverged from the unmodified one in text early (q4 noise 0.5: first differing character 436 of 2,202 in the unmodified output), so every step count above is for a different trajectory, with different tool use (the unmodified q4 run recorded no python call, each modified one 1; q5: 3 unmodified, 0 to 4 modified).

**What this shows and does not.** In our reality, on these two questions at one seed, the final answer survived every replacement of the latent content except the largest noise on q5, so the answer on these questions did not require the latent vectors to be what the HybridLink produced. What changed was behaviour: how long the model stayed in a span (random replacement pushed it to the floor of 4 steps; nearest replacement left it long), and, for noise 2.0 on q5, whether it finished within the budget. The step counts are not comparable between rows beyond that, since trajectories differ. Two questions, one seed, eight modified runs; correctness is the final boxed value only. Whether the answer would survive on a harder question, where the latent steps carry more of the work, was not tested; both questions are ones the model solves with python calls and short text.

**For Clover K3 (reading).** The signal that the latent channel was being used shows in how the model *behaves around* the vector (span length, budget use), not in the final answer on easy questions. An evaluation of a K3 latent channel built on the same interventions would need questions where the text-only version fails or is long, otherwise "answer unchanged" is uninformative.

**Not observed.** Interventions on harder questions; a second seed; permutation of vectors within a span; intervention only at selected positions in a span (for example only the last one before the exit).


## Cycle 22 (2026-10-22): the paper's span bounds (16 to 256) at three seeds (script `cycle22_sum.py`, output `cycle22-b16.txt`; q4, q5, q1 at seeds 0 (Cycle 18), 1 and 2 (new), search on, budget 2,048, `MINLAT=16 MAXLAT=256`)

The comparators are the default-bound (4 to 200) and `<latent>`-blocked runs of the same question and seed from Cycles 11 to 17. q1 has those only at seed 0.

| run | generation steps | text + latent (share) | spans | end | final answer |
|---|---|---|---|---|---|
| q4 s0 | 1,080 | 645 + 435 (40%) | 4 | eos | 24 |
| q4 s1 | 482 | 409 + 73 (15%) | 2 | eos | 24 |
| q4 s2 | 636 | 434 + 202 (32%) | 5 | eos | 24 |
| q5 s0 | 1,435 | 717 + 718 (50%) | 20 | eos | (-12, -8, 36) |
| q5 s1 | 1,079 | 771 + 308 (29%) | 8 | eos | (-12, -8, 36) |
| q5 s2 | 1,280 | 737 + 543 (42%) | 8 | eos | **(-12,-8,52), wrong** |
| q1 s0 | 725 | 531 + 194 (27%) | 7 | eos | **28, wrong (correct is 29)** |
| q1 s1 | 677 | 427 + 250 (37%) | 6 | eos | 29 |
| q1 s2 | 707 | 477 + 230 (33%) | 6 | eos | 29 |

**Observed.**
- Nine runs, all ended by `eos`; every span in every run closed by the model's own exit (0 forced). Seven of nine answers are correct. The two wrong ones: q1 s0 is an arithmetic slip in the reasoning (10,000 / 343 worked out as 28.03); q5 s2 followed a failed python call (a `NameError` in its own code) and then expanded the equation by hand and concluded a constant of 52 where the correct value is 36; I did not work out where in its text the slip is. The corresponding default-bound runs of the same seeds were correct on q5 s2; q1 s0 default was correct too; so with the paper's bounds the q1 s0 and q5 s2 answers changed from correct to wrong.
- Cycle 18 had recorded all three seed-0 answers as correct; I had not read the q1 answer. Corrected in the Cycle 18 section.
- Over the seven pairs that have both comparators (q1 s1 and s2 have none): text tokens 4,244 (bounds 16 to 256), 5,445 (default), 10,491 (blocked): 22.1% fewer than the default runs and 59.5% fewer than the blocked runs. Generation steps 6,717, 7,248, 10,491: 7.3% fewer than default and 36.0% fewer than blocked. The blocked total includes q5 s2 at its 4,096 cap (no answer), so it is a lower bound.
- Over the six pairs where both the 16-to-256 run and the blocked run finished (all but q5 s2): text tokens 3,507 against 6,395 (45.2% fewer), generation steps 5,437 against 6,395 (15.0% fewer).
- Per pair against the blocked run: text tokens lower in six of seven (q4 s0, 645 against 525, is the exception); generation steps lower in four of seven (q4 s1 482 against 486, q5 s1, q5 s2, q1 s0) and higher in three (q4 s0, q4 s2, q5 s0).
- Latent share of generation: 15% to 50%, median 33%. Seed to seed on one question it ranged 15% to 40% (q4), 29% to 50% (q5), 27% to 37% (q1).

**What this shows and does not.** Over these nine runs, three questions, three seeds, the paper's span bounds reduced text tokens against both comparators, and generation steps against the blocked runs once the budget-capped one is included or excluded (36.0% and 15.0%); the first holds in six of seven pairs, the second in four of seven, so it depends on the question and seed. The comparison is against a blocked copy of the same weights, not their trained text agent. In these runs the paper's bounds also produced two wrong answers where the default bounds had produced correct ones: whether the longer spans cause that, or it is run-to-run variation (one sampled trajectory each), is not separated by this test. Seven correct answers from nine do not say anything about a rate. Correctness is the final boxed value against a value I know for these five questions.

**For Clover K3 (reading).** Two things this cycle puts on the table for any K3 latent design: the text-token reduction is large and fairly robust; the effect on forward passes depends on the question; and a longer span floor traded some correctness in our reality on a small set. An evaluation of a K3 version needs answer-correctness measured on many questions per setting, since a single wrong answer here is within what one sampled trajectory can do.

**Not observed.** Many more questions and seeds per setting; the same bounds with a trained text agent as comparator; why the two wrong answers happened (not examined past reading the end of each output).


## Observation summary after Cycle 24 (2026-10-22): what was observed, in our reality, and what was not

No K3 training plan is written here; the creator asked for it only after HybridAgent is fully observed, and will decide whether the observation is complete.

**Observed by running the released 8B checkpoint on `k3` (CPU, bf16, batch 1, one checkpoint, five questions, up to three seeds).**
- *Mechanism.* A span opens with `<latent>`, each latent step feeds the HybridLink output back as the next input embedding, the model closes it by its own LM-head decision (131 of 131 spans in Cycle 21; 0 forced; LM head and cosine readouts agreed on 129), and text resumes. Latent and text steps cost the same on `k3` (about 0.42 to 0.44 s each, Cycle 3 onward); on K3 one decode step is about 3.0 s (Cycle 9, a different system).
- *Cache.* Cached continuation equals a full-prefix recomputation to bf16 rounding, as far from float32 as the full-prefix result is (Cycle 24, two runs, 2,878 positions).
- *Content of the vector.* Read out through the embedding matrix it is readable pseudo-text (readability 0.844 against a noise reference of 0.069, Cycle 19).
- *Dependence on content.* Final answers on q4 and q5 survived noise, random-token and nearest-token replacement except q5 at relative noise 2.0 (budget, no answer); what changed was span length and budget use (Cycle 23). Not a test of harder questions.
- *Cost.* The paper's own metrics on nine pairs (Cycle 17): fewer text tokens with latent allowed, fewer generation steps once truncated blocked runs are counted; on finished pairs only the generation-step saving is small. With the paper's span bounds over three seeds (Cycle 22): text tokens 22% below the default bounds and 59% below the blocked runs over seven pairs; generation steps 36% below the blocked runs (15% over the six that finished); higher in three of seven pairs. One-step spans removed the text-token reduction (Cycle 20).
- *Answers.* Both paper-bound runs that were wrong (q1 s0, q5 s2) were correct at the default bounds; one trajectory each.

**Not tested.** Their trained text agent as comparator; their benchmarks (MATH500, GAIA, WebWalker, etc.) or any statistic over many questions; training in any form; permutation interventions and position-selective interventions; harder questions; float32 against float32 for the cache; the 14B checkpoint; GPU execution (the `gpu` box was not touched); wall-clock on H100.

**Housekeeping.** `/root/.tavily.env` on `k3` and the tracked `oss/HybridAgent/.env.example` both contain a key line; the clone must never be committed or pushed, and the key should be rotated and moved to an ignored file. Traces are in `/root/runs/hybrid/` on `k3`; scripts in `oss/scratch-agents/hybrid-clean/`.


## Cycle 25 (2026-10-22): HybridAgent output handed to the released XSum NCM (creator's Direction: "merge both approaches in the available setup and see what they create together, before K3"; scripts `ncm-merge/` in `scratch-agents/hybrid-clean`, results `merge-score.txt`, `render-out.json`)

**What was run.** No new HybridAgent run: the text comes from the existing seed-0 default-bound traces of q4, q5 and q1 (final answers 24, (-12, -8, 36) and 29, all correct). Per question three inputs: `think_head` (the visible reasoning with latent tags, tool calls and tool results removed; first 512 GPT-2 tokens), `think_tail` (last 500 tokens of the same text; identical to `think_head` for q5 and q1, which are shorter than 500 tokens, so only q4 differs), and `answer` (text inside `<answer>`, 12 to 34 tokens). Control: the news-style facts text used earlier on the GPU server. NCM side: the released XSum checkpoint (88.02M generator, 60.63M tokenizer, prefix 512, 9 levels, 48 output tokens), shipped decoding (frequency and adjacency penalties), fp32, on the `k3` CPU (16 threads) in a separate venv (`/root/ncm-cpu`, torch 2.11.0+cpu), repo pinned to `c702742`. The `gpu` box was not touched. The key strings and the scoring script were written before any NCM output on HybridAgent text was read; my prediction was "fluent news-like text that keeps almost none of the facts".

**Instrument check.** On the control the CPU run produced "Plans for a major shopping centre in Portsmouth have been approved.", the same text the GPU server gave for this input in the earlier smoke run (one sample, same checkpoint and decoding).

**Observed.**
- One rendering (9 forwards plus the decoder pass) took 0.89 to 0.98 s on the `k3` CPU at batch 1, whatever the input (34 to 512 source tokens).
- Control (news facts, 79 tokens): fluent one-sentence summary, 0 of 10 key strings kept (the output names a different town; it is a news-summary prior, as in the K3 test).
- The nine HybridAgent-derived inputs gave text that is not readable as a summary: for example q4 `think_head` -> "I'm struggling to find out more ways to explain how it has to reduce number of people from trailing slightly below 18 countries."; q5 `answer` -> "The world's richest and earth scientists on the planet - ..."; q1 `answer` -> "What does the biggest answer question ask on the questions??"".
- Facts: 48 key strings were present in the nine sources; 5 of them appear in the shipped NCM text, and every one of those five is a topic word ("zeros", "circle", "sound"), never a number: none of 24, 36, -12, -8, 29 or 343 survives. Mean of 8 sampled generations per input: 0.0 to 1.0 keys.

**What this shows and does not.** In our reality the released XSum NCM, given HybridAgent's reasoning or answer text, does not carry the content through to its output; this matches what it did with K3's answers (Cycle on the NCM branch: 0 of 8, 0 of 8, 0 of 11). The checkpoint was trained on news articles of up to 512 tokens and produces a one-sentence news summary; reasoning text and one-line answers are a different input shape, and the output goes from fluent (news input) to incoherent (reasoning input). This describes THIS checkpoint on THIS input shape, not the NCM mechanism and not a trained HybridAgent plus NCM system. The pipeline itself is cheap: HybridAgent steps cost about 0.43 s each on `k3`, one NCM rendering about 0.9 s. What was not tested: any NCM-style generator trained on HybridAgent text (no training code exists in the NCM repo and none was written), the other released checkpoints, longer or news-shaped inputs built from HybridAgent text, a second seed of the sampled texts.

**For Clover K3 (reading).** The two pieces fit together mechanically (HybridAgent writes less; NCM writes the text in 9 passes), but the released NCM weights are not a usable second stage, so the combination only becomes testable with a generator trained on plan-to-text pairs. Pairs would come from HybridAgent at about 0.43 s per step on `k3` (against about 3.0 s per step on K3), which makes generating a training set several times cheaper than from K3 (arithmetic from the two measured step times, not a measurement of data generation). Whether to train anything is the creator's Direction; no training plan is written.


## Cycle 26 (2026-10-22): NCM data fetched from the `gpu` box, and a calibration of the Cycle 25 yardstick (creator: "did we merge them ... connect to the gpu box and get the data you need for ncm, I believe all might not be checked in"; the creator's permission to read the `gpu` box was given in this request)

**Did we merge?** Only partly, and I should say it plainly. Cycle 25 ran one pipeline direction: HybridAgent's text as the INPUT of the released XSum NCM, on the `k3` CPU. That is a test of whether the released NCM can be a second stage; it is not a combined system, and no NCM-style generator conditioned on a HybridAgent plan exists. The result was negative (incoherent text, no numbers kept).

**What is on the `gpu` box (read-only listing, nothing changed there; box idle at the time: load 0.15, 8 cards at 0 MiB).** `~/ncm`: repo `c702742` (clean), `artifacts/` (xsum 624 MB, tinystories 552 MB, wmt14 1.1 GB, sudoku 465 MB, lm1b 4.9 GB; `owt` holds only `TODO.md`), `data/`, the run scripts `run1.sh` to `run15.sh`, `render.py`, `main_fp32.py`, and the outputs `out/` (traces, speed, eval-xsum, eval-xsum-full, eval-lm1b-small), `out11/`, `out12/`, `out15-smoke/`, `out-k3a/`; `~/k3ncm` (K3 prompts p1..p5, `k3_summary.json`, `results/p1.log` only); run records `~/runs/ncm-*`, `aurora-*`, `elf-*`, `flm-*`. None of this is in git: the NCM context file `clover-ncm-research-context.md` is untracked in `c:\personal\oss\clover-framework` (branch `gpu-merged-optimised-layers`, used by another session) and `scratch-agents` is not a repository.

**Copied to this machine (6.3 MB, `scratch-agents/ncm/gpu-pull/`, plus `data-xsum/` with the XSum test passages and references, 20 MB):** the five output folders, the scripts, `k3ncm/`. The release-sized files (artifacts, datasets) were not copied; the XSum checkpoint is already on `k3` from Cycle 25.

**Calibration (script `calibrate_xsum.py`, output `calibrate-xsum.txt`; offline, from the stored full-test-set NCM hypotheses; definition fixed before the numbers).** Why: Cycle 25 scored NCM by key strings of my own choosing; I needed to know what the same kind of score gives on in-domain news. Facts of a document = numbers and capitalised words (not the first word, not a small stop list) of its reference summary that also occur in its source passage. On all 11,334 XSum test documents (10,139 with at least one fact, 37,068 facts): the NCM shipped hypothesis keeps **17,187 (46.4%)**; the first 48 GPT-2 tokens of the source (a trivial lead summary) keep 7,498 (20.2%).

**What this shows and does not.** On real news the released NCM carries about half of the reference's names and numbers, more than twice a lead baseline, so the yardstick is not simply harsh. On HybridAgent text it kept 5 of 48 keys (all topic words, no number) and on K3 text 0 of 27 (NCM branch), so the failure belongs to the input shape (reasoning text, one-line answers, synthetic fact lists), not to a too strict score. The two scores are not the same definition (reference-derived facts here, my key list there), so the comparison is a reading, not a ratio. I considered a reverse pipeline (NCM compresses a news article, HybridAgent answers from the summary) and did not run it: whatever HybridAgent says about the article would be bounded by what the NCM sentence kept (46% of the facts on average), so it would measure NCM again.

**What would make a real merge testable (reading).** A generator that takes a short plan and writes the longer text, trained on plan-to-text pairs (the released checkpoints are compressors or translators); the NCM repo has no training code. Not started; the creator decides whether to go there.

**Practical note.** The shared PowerShell terminal sometimes delays or hides output of long chained commands; writing results to files and reading them back worked.

---

## Cycle 27 — Context for the plan-to-text generator (what NCM ships, what the paper says, what is available)

**Direction (creator, 2026-10-22):** "go forward with training of plan to text generator we should properly merge hybridagent with ncm before even thinking about clover-k3 training". No K3 training plan is written here.

**Read (source, not summaries), `c:\personal\oss\ncm` at `c702742`.** `models/vqvae.py`: `VQVAE.forward(x)` returns `(logits, vq_loss, per-level utilisation, global utilisation, ppl)`; `get_ncm_inputs(ids)` returns `(x, idx[:, levels[0]:])`. `models/quant.py` `MultiscaleResidualVectorQuantizer.forward`: EMA codebook update inside the forward (decay per level), `vq_loss = mean over levels of (commitment * mse(z_hat.detach(), z) + mse(z_hat, z.detach()))`, straight-through, dead-code restart, k-means init, and nested dropout as `aux.fine_drop` (`prob`, `min_keep`). `models/next_concept.py` `NextConceptTransformer.forward(x, prefix_ids, prefix_pad_mask)` returns the logits of every target level in one pass (block-diagonal mask, per-level RoPE reset), `predict_level0` adds a head `lm_head_l0` that reads the hidden state of the last prefix position, `query_level0` uses learned queries instead, `generate(idx, prefix_tokens, ...)` takes `idx=None` only when one of those two is on. **Not in the repo (checked by listing all `def`/`class` in `models/`, `tokenization/`, `utils/`, `scripts/`):** an optimiser or training loop, the generator loss, the conditioning corruption, the data pipeline, the level-0 prior used for the unconditional TinyStories model (`level0_unigram` is stored in the generator checkpoint).

**Read (paper arXiv 2610.14162, appendix text via a web fetch; some table cells had lost labels, so values below are only those stated in prose or matched by position).** Tokenizer loss: token cross-entropy at a sampled cutoff level (nested dropout) + beta*commit MSE + latent MSE, straight-through. Tokenizer settings (columns of Table 5 matched to datasets by position): AdamW betas 0.9/0.95, weight decay 0.01, clip 1, warmup 1,000, EMA decay 0.95 (TinyStories, XSum) or 0.99, nested dropout 0.5, decoder 1 block, no encoder blocks, width 384, 4,096 codes. Generator: 12 blocks, width 768, 12 heads, RoPE base 1e5; loss `w_k` proportional to `l_k^-alpha` (alpha 0 to 1 gave the same Gen. PPL on TinyStories); **conditioning corruption** = replace a fraction of the *input* codes with random codes of the same level, rate decreasing linearly across levels, mean rate 0.5 was the chosen recipe, targets stay clean; conditional models are first trained unpaired and then on pairs. Budgets in the paper are large (tokenizer 30k updates at batch 4,096 for TinyStories; generator 50k updates; XSum generator 350k); the learning-rate values did not survive the fetch (not known to me).

**What exists on the `gpu` box (read-only):** `~/ncm/artifacts/tinystories` holds the released TinyStories tokenizer and generator and BPE files: `config.yaml` says 12 levels `1,8,16,24,32,40,52,64,76,92,108,128` (641 codes), 128-token windows, codebooks 4,096, generator prefix length 1, `predict_level0: false` (level 0 is drawn from a stored unigram prior), `tie_heads_to_codebook: true`. The HuggingFace cache there has the XSum data, GPT-2, Qwen3-0.6B, no TinyStories text and no larger writer model. The box can reach huggingface.co.

**Observed blocker (2026-10-22 23:13 box time).** All 8 cards hold about 22 GB each (compute-apps list: one process, pid 681638, `whole-model-gpu` of another session, 7 min old, one card at 100%), load 16.8, 46 GB RAM free. Nothing of mine is running there. The creator's earlier permission for the `gpu` box was to read data; launching a job or writing files there has not been agreed, and the cards are not free today.

**What this changes.** Because the TinyStories tokenizer is released and the stage-1 (unpaired) generator is released, the paper's two-stage recipe leaves only stage 2 (plan-conditioned) to train, with the tokenizer frozen. That removes the largest cost (tokenizer training) and the largest risk (my own tokenizer being incompatible). It costs domain: TinyStories openings (children's stories, 128 tokens), not the reasoning or QA text of HybridAgent. That is a limitation of the first merge, to be named in every result, not a reason to skip it.

**Design to check before building (reading, nothing run):** plan = a short telegraphic description of the story opening (characters, setting, events in order), up to 48 tokens, written by a model; text = the story opening (128 BPE tokens); generator = the released one with `prefix_len` P, `predict_level0: true` (one new head, all other weights load), prefix embedded with the frozen tokenizer encoder's token table (as the shipped XSum config does), loss = cross-entropy over level 0 plus levels 1 to 11 with corruption. Success: facts of the plan (names, objects, events) that appear in the generated story, against the same count for the true story and for the stage-1 model given no plan. Gate before any training: the released tokenizer reconstructs held-out stories and the released generator reproduces its published behaviour on this machine.

---

## Cycle 28 — Gate on the released TinyStories NCM, stage-2 code written, CPU smoke (result of the long run pending)

**Why this cycle.** Cycle 27 found that the repo has the model and the VQ forward but no loop, loss, corruption or data code, and that the released TinyStories tokenizer and generator (stage 1, unpaired) are available. Direction: build stage 2 (plan-conditioned) on top, after a gate that shows the released parts behave here as published.

**Gate (k3 CPU, torch 2.11.0+cpu, `gate_ts.py`, output `gate-ts.json`; artifacts byte-identical in size to those on the gpu box: generator 356,160,651, tokenizer 221,861,835).** Data: the first 128 BPE tokens of the first 128 stories with at least 128 tokens in `TinyStories-valid.txt` (21,990 stories read). (1) Tokenizer, exact token accuracy of the cumulative reconstruction after each of the 12 levels: 8.2%, 11.5%, 22.7%, 37.0%, 56.2%, 69.6%, 81.9%, 94.0%, 96.5%, 97.9%, 98.8%, **99.963%** (16,384 tokens; the paper reports 99.79% to 100% for its tokenizers; this is 128 windows, one sample). (2) Generator, teacher-forced cross-entropy per level (nats, clean inputs, prefix = EOS) `7.53 5.41 4.06 3.16 1.92 1.90 0.86 1.33 1.72 1.61 0.33` for levels 2 to 12, against `7.57 8.78 10.5 12.3 13.4 15.2 18.9 15.4 14.5 14.2 21.2` when each story's inputs are replaced by another story's (uniform reference ln 4096 = 8.32): the generator uses its inputs, strongly at the finer levels. (3) Six unconditional samples (level 0 from the stored unigram, shipped sampling) are fluent TinyStories openings. Not tested: Gen. PPL against GPT-2-large (not computed here), so "reproduces the published number" is NOT claimed, only "behaves as a working released model".

**Stage-2 code (`ncm_stage2.py`, written for this study, repo untouched).** Loads the frozen tokenizer and the released generator, sets `prefix_len = 48` and `predict_level0 = true`, drops only the shape-mismatched buffer `attn_mask_for_masking` (rebuilt by the constructor); load report: missing = that buffer + the two parameters of the new level-0 head, unexpected = none; parameters 88.90 M (released 88.61 M). Loss = cross-entropy over level 0 and levels 1 to 11 (641 codes), per-level weights `l_k^-alpha` (alpha = 0 default), conditioning corruption in code space (random same-level codes, rate falling linearly from 0.9 at level 0 to 0.1 at level 10, mean 0.5; targets clean). Optimiser: AdamW (0.9, 0.95), weight decay 0.01, clip 1, warmup then cosine to 10%; **the learning rate (2e-4) and the exact corruption schedule are my choices, not the paper's**. Evaluation: validation CE with the true plan vs another story's plan, and generation from the plan with shipped sampling, scored by plan-word retention (content words of the plan, length at least 4, first-5-letter match in the text) for the generated story, for the generation from another story's plan (control), and for the true story (ceiling).

**Smoke data (not the real plan source).** `make_pairs_kw.py`: 3,000 pairs from the same validation file; plan = first 10 distinct content words of the window ("forest, lived, rhinoceros, named, roxy, ..."). A keyword plan is only a test of the code path.

**What ran.** 20 training steps (batch 8, CPU): 8.7 s per step, loss 4.92 to 3.66, validation CE (mean over levels, 32 stories) 4.114 matched / 4.145 shuffled before, 3.271 / 3.343 after; the plan-conditioned generation path works (`generate(None, plan, ...)` with the new head) and gives fluent stories that are only loosely about the plan. On 16 stories after those 20 steps: retention 0.41 for the generated stories, 0.20 for the other-plan control, 1.00 for the true stories (ceiling is trivial here because the plan is extracted from the story). **One seed, 16 stories, 20 steps: a sign that the code path carries information, not a result.** A 1,200-step run (batch 8, about 3 h) was started on k3 at 01:33 local (`smoke/t1200`, log `smoke/t1200.log`, `t1200.done` at the end).

**Resources found, still unresolved (not mine to decide).** The 8 cards on the gpu box were full (about 22 GB each, another session's job) at 23:13 box time, and nothing there was launched or written by me; my permission for that box is read-only so far. Plan generation for tens of thousands of stories (the real plan source) and a real-size run need those cards.

---

## Cycle 29 — HybridAgent as plan writer (text mode), 9-story test on k3 CPU

**Creator answers (2026-10-22):** plan source = HybridAgent-8B itself; TinyStories accepted as the first merge domain; on the gpu box: "it was said it will take 45 minutes do we want to wait or use the cpu machine" (a question back to me, so permission to write/launch on the gpu box is still NOT given; the other session's job is reported to end in about 45 minutes).

**What ran (`plan_writer.py`, output `plans_test2.jsonl`).** The checkpoint `togethercomputer/HybridAgent-8B-Agentic` loads as a plain Qwen3 causal model (`model.safetensors`; the HybridLink is in a separate `adapter.pt`, not loaded), so this is HybridAgent **in text mode, no latent steps**. Prompt: one example story and plan, "write a plan in one line... at most 40 words", the answer prefilled with `Plan:`, greedy, 64 new tokens, bf16 on the CPU (about 14 s per story at batch 3 while the smoke training also ran). First try without the example and prefill (3 stories): 1 of 3 usable, one answer started with "Okay, let me try to figure this out", one ran past 40 words. After the change: **9 of 9 read as plans** ("little girl Lily; at home with her mom; Lily loves playing dress-up; ...; at the play, the room is dark but stage is lit"), one-line, 30 to 40 words; some plans extend past what the 128-token window shows (e.g. "moral: ..." for the beetle story), which I have **not** checked story by story. 9 stories is a format check, not a quality measure. The script flags outputs that mention "the user"/"let me" or are very short (`bad`) and caps at 40 words.

**Open.** The 1,200-step keyword smoke is at step 25+ (about 9.6 s per step) on the same CPUs. Real plan data for tens of thousands of stories needs the gpu box (not launched, not permitted yet); on this CPU box the rate is about 14 s per plan.

---

## Cycle 30 — gpu box: data, plan writing at scale (creator: "gpu is done you can use it and go ahead yes")

**Authority and state.** Creator gave permission to use the gpu box. Before launching, read-only check at 23:45 box time: 8 cards at 0 MiB, load 0, nobody else running (the other session's job had ended). Everything of mine is under `~/plan2text` (data, out, scripts); jobs go through `~/probe/run-observed.sh` (logs in `~/runs/p2t-*`). Other sessions also use this box, so each launch is preceded by a card check. Superseded: the CPU keyword smoke on k3 was stopped at step 100 of 1,200 (no checkpoint; log kept), because real plans and the cards replace it.

**Data.** TinyStories train and valid text (HF `roneneldan/TinyStories`, `TinyStories-train.txt` 1.92 GB) and the HybridAgent-8B-Agentic weights (16.4 GB) downloaded into `~/plan2text/data` and the HF cache. `make_pairs_ts.py`: first 128 BPE tokens of each story with at least 128 tokens, in file order: 80,000 train windows (from the first stories of the train file) and 600 validation windows (from the validation file).

**Plan writer (HybridAgent-8B in text mode, no latent steps; 8 shards, one per card, batch 64, greedy, 64 new tokens).** Version 1 prompt on the 600 validation windows: 600 plans in 25 s wall (43 plans/s); 8 flagged, 357 cut at 40 words (many mid-phrase, some with a leaked "Now, write the..."), and **66.5% of a plan's content words occur in the story window** (the same measure on the true story is the ceiling for any generated text; it is below 100% partly because plans paraphrase and partly because plans name events beyond the 128-token window). Version 2 (prompt says only use what the text says and do not guess what happens after; cut at the last semicolon when over 35-40 words; leak check): 0 flagged, 320 of 600 over the cap (cut cleanly), mean 34.6 words, **67.4%** content words present. So the stricter prompt barely changed that number; I did not find a way to remove the gap with the prompt and keep it as a measured property, filtering instead at pair-build time (`build_pairs.py`: drop flagged, leaked-instruction plans such as "(The story ends here...)", and, for training only, plans with fewer than 40% of their content words in the window).

**Launch.** Plans for the 80,000 train windows started 23:52 box time (`p2t-plans-train`): about 26 plans/s measured over the first 12 minutes (18,624 done, 8 shards alive; the cards sit at 22 to 23 GB and the allocator logged three OOM-retry warnings, no crash so far), so about 50 minutes in total. Training is not started: the cards are in use by the writer.

**Code changes for the box.** `ncm_stage2.py` now runs under torchrun (DDP, rank-0 evaluation and saving), takes a separate validation pairs file, and can keep periodic checkpoints. Not yet run on the cards: the first thing to do after the plans is a short smoke on the 8 cards.

---

## Cycle 31 — Stage-2 training on 78,823 HybridAgent plans, evaluation, end-to-end test (gpu box)

**What ran.** `p2t-run1`: released TinyStories generator (88.61 M) + new level-0 head (88.90 M), tokenizer frozen, prefix 64 plan tokens (right-padded), loss = mean cross-entropy over the 641 codes (alpha 0), conditioning corruption linearly 0.9 to 0.1 (mean 0.5), AdamW (0.9, 0.95), wd 0.01, clip 1, lr 2e-4 with 300 warm-up steps and cosine to 10% (my choices), bf16, 8 cards, global batch 256, **5,000 steps = 16.2 passes over the 78,823 pairs**, 0.63 to 0.66 s per step, about 58 min including 100 s of story encoding per rank. 512 validation pairs (from `TinyStories-valid.txt`, never trained on) were scored every 500 steps.

**Validation cross-entropy (mean over the 12 levels, nats; true plan / another story's plan).** step 0 (new head untrained) 4.727 / 4.809; 500 2.778 / 3.700; 1,000 2.733 / 3.815; 2,000 2.697 / 3.889; 3,000 2.664 / 3.918; 4,000 2.640 / 3.958; 5,000 **2.631 / 3.975**. The true-plan loss was still falling at 5,000 (no overfitting visible; more steps were not tried), and the other-plan loss kept rising, so the model relies on the plan more over training. **Per level at step 5,000 (true plan):** level 0 (one code) 7.87, the same as with another story's plan (7.87): the first code carries no plan information yet (uniform would be 8.32); levels 1 to 11: 4.73, 4.47, 3.33, 2.41, 1.81, 1.41, 0.53, 1.30, 1.66, 1.79, 0.26, against 9.58, 8.45, 6.55, 4.80, 2.92, ... with another story's plan.

**Generation, validation plans (598 plans written by HybridAgent from the true story window, last checkpoint, shipped sampling temperature 1.0 / top-k 0 / top-p 1.0, 3 sampling seeds).** Share of a plan's content words (length at least 4, first-5-letter match) found in the text: **generated from the true plan 0.612** (seeds 0.609 / 0.613 / 0.615); generated from another story's plan, scored against this plan (chance level) **0.085**; the true story **0.675** (the ceiling of this measure, below 1 because plans paraphrase and name events past the 128-token window). So the generated stories carry about 91% of what the true stories carry on this yardstick, against 13% for the control. Generation speed: 3,588 stories (598 x 3 seeds x 2 conditions) in 74 s on one card at batch 64 (about 48 stories/s); the plan writer produced about 5 plans/s per card. Read, not measured further: the first lines of the sampled stories follow the plan names, objects and event order closely (examples in `eval_run1.json`); the word-overlap measure is dominated by names and nouns and says nothing about grammar or the later events.

**End-to-end front half (the real merge test).** `plan_from_topic.py`: HybridAgent-8B in text mode turns 48 short requests I wrote ("a girl who finds a lost puppy", ...) into plans, with no story text; then the trained generator writes the story (`eval_e2e.json`, `topic_plans.jsonl`). Share of plan words in the generated stories **0.596** (seeds 0.595 / 0.611 / 0.583), control 0.060; 48 stories in 6.2 s per condition on one card. **What reading the outputs showed (5 of 48 stories read, the first 420 characters of each, plus the first lines of 4 more; chosen by list position, not by quality; the other 39 were not read):** requests with common names and objects (puppy, garden, teddy bear start) give a story that follows the plan; **names and words outside the 8,192-word-piece TinyStories vocabulary are garbled** (Mousie became "namedousie", "Mousa"; R2D2 became "C2 Dare" and the robot story is mostly broken); the snowman story degenerates ("winter summer, the snowyy winter"); several stories repeat words ("started started", "owner owner"). Only the first two or three plan events fit into 128 tokens. 2 of 48 request plans end with leaked `</think>` tokens (my flag did not test for them); they cost plan tokens but were not removed.

**What this does and does not show.** Shown, in our reality (TinyStories openings, HybridAgent-8B plans, one trained model, one run, one hardware setup): a plan written by HybridAgent can drive an NCM-style generator that writes the whole 128-token text in 12 forward passes, and the plan is used (validation loss 2.63 vs 3.98 with another plan; word retention 0.61 vs 0.09). Not shown: quality of the text beyond word overlap (no language-model perplexity or human rating was computed); any use of HybridAgent's latent steps (it was used as a plain text model); anything about reasoning or QA text (no tokenizer for it exists); whether more steps or a different learning rate would help (one setting, no sweep); the level-0 code is not conditioned on the plan yet.

---

## Cycle 32 — Quality round: measure first, then decoding levers on run1 (training of run2 started)

**Direction (creator, 2026-10-22):** "go ahead with another round to improve the quality". Cycle 31 had no fluency number, so I measured before changing anything (Context): `quality_eval.py` scores, for the 598 validation plans, plan-word retention, Gen. PPL under GPT-2-large (exp of total NLL / total tokens, the paper's reference model), adjacent repeated-word rate, repeated-trigram rate (share of word trigrams that already occurred earlier in the same text) and mean per-sample unigram entropy of GPT-2 tokens. Reference rows: **true story windows: retention 0.675, Gen. PPL 7.66, adjacent repeats 0.0037, repeated trigrams 0.0178, entropy 4.033.** `run1` (step 5,000) at the shipped sampling (temperature 1.0, 3 seeds): retention 0.612, **Gen. PPL 16.35**, adjacent repeats 0.0128, repeated trigrams 0.0182, entropy 3.996.

**Is fluency lost in stage 2?** `uncond_ppl.py`: the released unconditional TinyStories generator through the same code (598 stories x 3 seeds, level 0 from the stored unigram): Gen. PPL **18.67** at temperature 1.0 and 13.71 at temperature 0.6 with the adjacency penalty (adjacent repeats 0.0101 / 0.0089). So the plan-conditioned model (16.35 / 12.18 under the same two settings) is *more* fluent than the unconditional one by this measure; the gap to the true text (7.66) belongs to the released tokenizer-plus-generator, not to stage 2. Consequence: more stage-2 training can only move this a little; the cheap levers are at decoding time.

**Decoding levers on run1 (no retraining; 598 plans; 3 seeds, or one MBR selection of K candidates).** Single-lever rows, retention / Gen. PPL: temperature 0.8: 0.625 / 13.48; 0.6: 0.630 / 12.15; top-p 0.9: 0.616 / 14.41; adjacency penalty (a,window,decay) 5,4,0.5: 0.614 / 16.14 (adjacent repeats 0.0089), 10,4,0.5: 0.614 / 16.28 (adjacent repeats **0.0063**); MBR over 8 candidates (keep the one with the highest mean word-set Jaccard to the others, never looks at the plan): 0.633 / 14.38. Combinations: temperature 0.6 + adjacency 10,4,0.5 + MBR8: **0.649 / 11.56**, repeated trigrams **0.0348 (about twice the true text)**, entropy 3.925; MBR16: 0.649 / 11.34. **The fluency gain from low temperature is partly bought with phrase repetition**, which the adjacent-word rate does not show; I only saw it because I added the trigram metric after reading "They gathered sticks and leaves ... sticks together" in the stories. Wider adjacency window (10,8,0.8) and a soft token n-gram penalty (n=3, 4 logits, 8 logits; token n-grams already produced in the same text) reduced it only partly, because the decoder maps the whole code stack to all 128 tokens in parallel, so the penalty can only change a choice where the decoder already had a close alternative. Balanced rows (all MBR8, adjacency 10,8,0.8), retention / Gen. PPL / repeated trigrams / entropy: temperature 1.0: 0.632 / 14.79 / 0.0197 / 3.994; **0.8 with n-gram 3,8: 0.646 / 13.04 / 0.0225 / 3.970**; 0.7 with n-gram 3,8: 0.646 / 12.51 / 0.0264 / 3.955; 0.6 with n-gram 3,8: 0.651 / 12.27 / 0.0281 / 3.940. **Current choice (a judgment, not a measured optimum): temperature 0.8, adjacency 10,8,0.8, n-gram 3,8, MBR over 8 candidates**: Gen. PPL 13.04 (-20% from 16.35; true text 7.66), retention 0.646 (+0.034; 96% of the 0.675 ceiling), repeated trigrams 0.0225 (true 0.0178), adjacent repeats 0.0077 (base 0.0128, true 0.0037). Cost: 8 times the generation; 598 plans (4,784 candidates) were generated, selected and scored in 112 to 120 s on one card (baseline: 3 x 598 stories and scoring in 52 s), about 5 selected stories per second including GPT-2-large scoring.

**End-to-end requests (48 requests, plans written by HybridAgent from the request alone).** Same decoding comparison, retention / Gen. PPL: old plans, run1 baseline 0.596 / 24.39, temperature 0.6 + adjacency + MBR8 0.634 / 15.16. New plan prompt (`plan_from_topic3.py`: "short common first names (Tim, Lily, ...), plain everyday words, no abbreviations or digits"; leaked `</think>` removed): baseline 0.585 / 21.47, same decoding 0.640 / **13.66**. Reading the first six requests (fixed by position) for both prompts under that decoding: stories now follow the plan in order (puppy, treehouse, fish, cat), the "R2D2" request no longer produces garbage words because the plan says "Robot named Tim" (with the old plan the text still contains "R2 1 22", "DoryD Doug"), and repeats remain ("close his eyes, close his eyes"; "They gathered the leaves, sticks, and sticks together"; "again and again again"). 6 + 6 stories read, first 430 characters each; 36 of 48 per prompt not read.

**Not tried / open.** Training-side levers (plan dropout for classifier-free guidance, level weighting, more data, more steps): `run2` (continue from step 5,000, 5,000 more steps, fresh warm-up and cosine at peak lr 1e-4, global batch 256) started on the 8 cards to test the cheapest one, duration. No human rating, no second reference LM (selection and scoring never share a model: MBR uses word overlap, scoring uses GPT-2-large), one validation set of 598 plans, and the repeated-trigram and PPL numbers depend on the temperature in opposite directions, so "best" depends on the weight given to each.

---

## Cycle 33 - Independent judge (Qwen3-8B, no GPT-2) built on k3; controls passed; main run launched

**Direction (creator, 2026-10-22):** build a quality judge that does not use GPT-2 or word overlap; build it on k3, start the real run detached, report back. Rules: everything under `/root/judge` on k3, nothing else touched, gpu box read-only (only `pairs_valid.jsonl` was read from it).

**Built (k3 `/root/judge`; local copy `scratch-agents/hybrid-clean/ncm-merge/judge.py`):** `Qwen/Qwen3-8B` (16 GB, 5 safetensors shards, 8.19 B parameters, 36 layers) in bfloat16 on CPU, one prefill pass per call (no generation), chat template with `enable_thinking=False` (empty think block, assistant turn open), verdict = next-token logit of "A" (id 32) vs "B" (id 33). Pairwise prompt: plan + two stories truncated to 110 words, "(a) fluent and coherent, no garbled words or repeated phrases AND (b) follows the plan". Every pair judged in both orders; win only if it wins in both, else tie. Data: `data.json` = 598 plans + four systems' `all_generated_first_run` lists.

**Controls (first 20 plans, base.json):** negative (same story as A and B): judge picks "A" **20/20**, mean logit margin A-B +9.2 (position bias is strong; under the both-orders rule an identical pair is a tie by construction, so the informative number is the A-rate). Positive (real story vs its words shuffled): real wins in both orders 17/20, tie 3, shuffled wins 0; single calls prefer the real story 37/40; picks A 23/40 over the pos control. Probability mass on A+B after the verdict position 0.996 (mean). Prompt token count 368-401.

**Speed:** median 1.7 s per call (forward pass only, 16 threads, Ryzen 9 7950X3D with avx512_bf16; a 8.19 B model at ~386 tokens is ~6.3 TFLOP per call, so ~3.7 TFLOP/s, plausible for that CPU, not independently counted). 360 calls therefore ~11 min; all 60 plans kept.

**Main run:** `judge.py main 60` detached, results appended to `/root/judge/results.jsonl`, `summary.txt` written at the end. Results NOT yet read when this entry was written. Not established: that this judge agrees with humans (only a shuffle control), that Qwen3-8B is unbiased towards text sampled at lower temperature, that 60 plans is enough; the judge is also a different model family from the generator but trained on web text that includes stories like TinyStories.

---

## Cycle 34 — Level-0 head defect found; two fixes training in parallel; code audit by a subagent

**Direction (creator, 2026-10-22):** use both machines (8-card gpu box, big CPU box k3) and subagents as much as possible. Split of work: gpu box = training (two experiments on four cards each), k3 = independent judge (Cycle 33, built by a subagent), a read-only audit subagent = second pair of eyes on my code.

**Defect (found from a number that looked too clean).** The per-level validation loss of `run1` at level 0 was **7.867858409881592 with the true plan and exactly the same with another story's plan** (all levels 1 to 11 differ). `diag_l0.py` (CPU, run1 step 5,000, 8 validation rows): level-0 logits identical across rows (max abs difference 0.0) while the hidden state the head reads differs by about 5 between rows and the head's `proj` output by about 3; level-1 logits change by up to 32.4 when the plans are rolled. Cause: `lm_head_l0` is the repo's tied-codebook head, logits `scale * (2 z.E - |E|^2) / mean|E|^2`, and **every row of the level-0 codebook has norm about 4e10** (min 3.7e10, max 1.2e11, typical 4.2e10; levels 1, 2, 5 and 11 have norms of about 0.1 to 55), so `mean|E|^2 = 4.6e21` and the `z.E` term, with z of order 1, is about 1e-11 of the logit: the head can only output a fixed prior. (Why the codebook is that large is not established; the released tokenizer's level-0 path goes through a learned down/up-sampler, so the scale may be compensated inside it; I did not check.) **Consequence for every earlier result:** the single level-0 code (the coarsest, one of 4,096) was sampled from a plan-independent distribution in all generations of Cycles 31 and 32, so the plan could steer only levels 1 to 11. The released unconditional model uses a stored unigram for the same code, so nothing was lost relative to it, but a plan-conditioned level 0 was never trained.

**Fix, two variants (both continue from run1 step 5,000, 4,000 steps, global batch 128 = 4 cards x 32, peak lr 1e-4 with 200 warm-up steps, level-0 head re-initialised with 10x the lr):** `e1_untied` = plain linear level-0 head (cards 0 to 3); `e2_tiednorm` = the same tied head with the codebook divided by its rms norm (cards 4 to 7). `ncm_stage2.py` records the head type in each checkpoint and `load_models` restores it. Also added (not yet used): plan dropout (`--plan-dropout`, null prefix = EOS then PAD, the form `generate(cfg_scale>0)` builds) for classifier-free guidance. First validation values at step 500 (mean over the 12 levels; true plan / other plan): e1 2.632 / 3.956, e2 2.627 / 3.964 (run1 final 2.631 / 3.975, whose level-0 term was constant; step 0 with the fresh head: e1 2.689, e2 2.684). Per-level level-0 numbers are in `log.jsonl` and will be read at the end of the runs. `run2` (continue from run1 with the same head, lr 1e-4) was **stopped** at step 3,000 of 5,000 (validation true-plan loss 2.648 at step 1,500 and 3,000, 2.655 / 2.650 / 2.651 at 500 / 1,000 / 2,500, no better than run1's 2.631, so the learning rate was still high when it stopped): so more steps with the broken head did not help in the part I ran; its final value is unknown.

**Audit (read-only subagent; its points weighed by me by reasoning, not re-measured).** No further defect that changes results was reported. Accepted: the retention measure is crude (a plan word counted anywhere, 5-letter stems, words of 4+ letters; its ceiling on true stories is 0.675, measured), so presence does not show binding; the null-prefix path for classifier-free guidance should be paired-tested against the level-0 slice before guidance is used. Not accepted as a defect: the validation plans are unfiltered on purpose (documented earlier), so the evaluation is not selected for easy plans.

---

## Cycle 35 — Level-0 fix results (E1 untied, E2 tied-normalised) and the Cycle 33 judge result

**Judge result (Cycle 33 main run, read from `/root/judge/summary.txt`; 60 plans, both orders, Qwen3-8B on CPU).** base vs `t08_ng3p8` (temp 0.8, adjacency, n-gram, MBR8): base wins 0, tie 52, decoded wins 8 (two-sided sign test p=0.0078). base vs `t06_adj10_mbr8`: 0 / 52 / 8 (same p). `t08_ng3p8` vs `t06_adj10_mbr8`: 4 / 52 / 4. The judge picked position A in 336 of 360 calls (0.933), so only 8 of 60 pairs were decisive per comparison. What this shows: the decoded stories are never judged worse than the base by an independent model and are better in the pairs where it could decide. What it does not show: any difference between temperature 0.8 and 0.6, or agreement with a human reader (shuffled-text control only), or anything about the 538 other plans.

**Training outcome (both from run1 step 5,000, 4,000 steps, global batch 128, peak lr 1e-4, level-0 head re-initialised with 10x lr; validation = 512 of the 598 pairs, as in the trainer).** Final (step 4,000) per-level validation loss, true plan / other story's plan:

| | level 0 | mean over 12 levels | other-plan mean |
|---|---|---|---|
| run1 (head broken) | 7.868 / 7.868 | 2.631 | 3.975 |
| `e1_untied` | 7.603 / 7.703 | 2.6205 | 4.062 |
| `e2_tiednorm` | 7.142 / 7.525 | 2.5826 | 4.049 |

Levels 1 to 11 are within 0.01 of each other between E1 and E2 (I did not compare them with run1 level by level here). The mean-loss gap between E2 and E1 (0.038) equals the level-0 gap divided by 12 (0.46 / 12), so it is all level 0; the same arithmetic against run1 gives 0.73 / 12 = 0.06 from level 0 alone, which is at least the whole 0.048 mean improvement (E2 vs run1), so no other level has to have improved (run2, which continued run1 with the old head and did not improve, also argues against extra steps, but I did not run a head-only comparison). The level-0 head now responds to the plan: gap between other and true plan 0.10 (E1) and 0.38 (E2) nats, against exactly 0 before. Absolute level-0 loss is still 7.1 to 7.6 of a maximum ln 4096 = 8.32, so the first code is only weakly predictable from the plan; I read this as the plan carrying little information about the single coarsest code, and did not test that.

**Generation quality (same 598 validation plans, `quality_eval.py`, base = temp 1.0 single candidate, 3 seeds; balanced = temp 0.8 + adjacency 10,8,0.8 + n-gram 3,8 + MBR8, one run):**

| | retention | Gen. PPL (GPT-2-large) | repeated trigrams |
|---|---|---|---|
| true text | 0.675 | 7.66 | 0.0178 |
| run1 base | 0.612 | 16.35 | 0.0182 |
| `e1_untied` base | 0.6165 | 15.20 (seeds 15.02 / 15.27 / 15.31) | 0.0182 |
| `e2_tiednorm` base | 0.6171 | 16.79 (seeds 16.63 / 16.79 / 16.93) | 0.0184 |
| run1 balanced | 0.646 | 13.04 | 0.0225 |
| `e1_untied` balanced | 0.6481 | 12.75 | 0.0221 |
| `e2_tiednorm` balanced | 0.6495 | 13.04 | 0.0213 |

Reading: fixing level 0 lowered validation cross-entropy (E2 more than E1) and moved the text metrics very little: retention +0.004 to +0.005, PPL -1.15 (E1 base, outside its seed spread of about 0.15) or +0.4 (E2 base, also outside spread; run1 base had no per-seed record here), balanced PPL 12.75 (E1) and 13.04 (E2) from one run each. The two heads rank differently on the two measures (E2 on validation loss, E1 on PPL), so I do not call either a winner. The retention metric counts plan words anywhere in the text and cannot see a change in a global code. No stories from these models have been read yet. Artefacts: `gpu-out/c35_*.json`, `gpu-out/c34_e*_log.jsonl`, launcher `q_c35.sh`.

**Running:** `e3_drop` (cards 4 to 7, from E2's final checkpoint, 3,000 steps, lr 5e-5, plan dropout 0.1) to make classifier-free guidance usable. Not evaluated yet; the null-prefix check against the level-0 slice is still to do before guidance.

**Stories read (balanced decoding, validation plans 3, 50, 200, 400, first run each; four plans, chosen by index not by quality):** run1, E1 and E2 begin the same way and follow the same plot. Visible glitches (dropped or doubled words such as "lots of other the bees", "the the clouds", "faster and and faster", "his noise made him loud noises") occur in all three; in these four, E1's text looks to me the least garbled and run1's the most, which matches E1's lower PPL, but four stories and one reader are not evidence of a difference. The level-0 fix is not visible by eye.

---

## Cycle 36 — E3 (plan dropout 10%) and classifier-free guidance

**Trained (`e3_drop`, cards 4 to 7, from E2's final checkpoint, 3,000 steps, global batch 128, peak lr 5e-5, tied-normalised level-0 head kept, 10% of training plans replaced by the null prefix [EOS, PAD, ...]).** Mean validation loss (true plan / other plan): 2.5757 / 4.0117 at step 3,000 (E2: 2.5826 / 4.049). Plan-conditioned loss did not get worse from dropping 10% of plans; the 0.007 gain over E2 comes with 3,000 extra steps and I did not separate the two. The ssh session dropped once while I waited; the detached job finished (`done` file present).

**Evaluation (598 validation plans, `quality_eval.py`, E3 last.pt; balanced = temp 0.8 + adjacency 10,8,0.8 + n-gram 3,8 + MBR8; guidance scale s applies `(1+t) * cond - t * null` with t = s * level / 11 as in the repo's `_sample_level`; base rows are 3 seeds, balanced rows one run):**

| config | retention | Gen. PPL | adj. repeat | repeated trigrams | entropy |
|---|---|---|---|---|---|
| true text | 0.675 | 7.66 | 0.0037 | 0.0178 | 4.033 |
| E3 base (s=0) | 0.6164 | 16.59 | 0.0139 | 0.0189 | 3.992 |
| E3 base s=1 | 0.6596 | 16.80 | 0.0161 | 0.0229 | 3.972 |
| E3 base s=2 | 0.6733 | 18.25 | 0.0191 | 0.0248 | 3.955 |
| E3 balanced s=0 | 0.6454 | 13.04 | 0.0077 | 0.0221 | 3.972 |
| E3 balanced s=0.5 | 0.6636 | 13.42 | 0.0079 | 0.0234 | 3.965 |
| E3 balanced s=1 | 0.6767 | 13.85 | 0.0083 | 0.0230 | 3.961 |
| E3 balanced s=2 | 0.6879 | 15.07 | 0.0088 | 0.0239 | 3.953 |
| E3 balanced s=4 | 0.6949 | 20.40 | 0.0084 | 0.0206 | 3.960 |

Reading (judgment, labelled): guidance raises plan-word retention steadily, and at s=1 with balanced decoding it reaches 0.677, which is at the retention of the true stories (0.675); beyond that the number says the text uses plan words more often than the real stories do, so higher retention is no longer a better score. PPL pays for it: 13.04 to 13.85 at s=1, 15.07 at s=2, 20.40 at s=4. Without guidance E3 matches E2 (balanced PPL 13.04, retention 0.645 vs 0.6495), so plan dropout cost nothing measurable. **Stories read** (plans 3, 50, 400, first run, balanced, s=0 / 1 / 4): at s=1 the text follows the plan more closely in all three (for plan 400, "Mommy, Mommy, make the noise to stop!" is in the plan; for plan 3, "I don't know how to play with a bee" is in the plan) and is no more garbled than s=0; at s=4 it is visibly worse ("named.. Billy", "The so was very pretty", "his runny's nose"), which agrees with the PPL of 20.4. Three stories and one reader; the choice of s=1 as the balance point is a judgment from these rows, not a measured optimum. Not done: judge run on the guided outputs (Qwen3-8B had 8 of 60 decisive pairs in Cycle 33, a weak instrument), an end-to-end 48-request re-run with E3, a separate test that the CFG null prefix matches the training null at the level-0 slice (the code path reads the same form, [EOS, PAD, ...], but I did not run a paired test; the gain at s=1 is indirect evidence that the null branch is meaningful).

Artefacts: `ncm-merge/q_c35.sh`, `q_c36.sh`; results on the gpu box in `~/plan2text/out/q-c36/` and `out/e3_drop/`.

---

## Cycle 37 — Guidance null-prefix checks, 48-request end-to-end with E3, judge on guided stories

**Null prefix (the Cycle 32 audit's open point), three checks on `e3_drop` and, as a contrast, on `e2_tiednorm` (trained without plan dropout):**
1. Ids: `net.prefix_pad_id` = 1 = the pad id I train with; null = [EOS=0, PAD=1, ...] in both `generate()` (`cfg_null_eos`) and the trainer's plan dropout. Same form, read from the code and printed.
2. Level-0 logits, exact (`cfg_l0.py`; level 0 depends only on the prefix): in a guided batch [real; null], the null half equals a plain null-plan run (max abs difference 3.8e-6 for E3, 4.8e-6 for E2, mean |logit| about 1.6 to 1.9) and the conditional half equals a plain real-plan run (1.4e-5, 1.3e-5); argmax equal in 48/48 in all four comparisons. Control (real vs null plain): max difference 8.6 (E3) and 16.5 (E2), argmax equal in 3/48 and 0/48. So the doubled batch builds the branches it is supposed to, at level 0, in both models.
3. Teacher-forced validation loss with the null prefix (598 validation pairs; `cfg_check.py`, mean over 12 levels; level 0 in brackets), true plan / other plan / null: E3 2.590 (7.158) / 4.019 (7.557) / **3.052 (7.456)**; E2 2.598 (7.182) / 4.056 (7.573) / **3.424 (9.616)**. E3's null prefix is a sensible unconditional distribution (between the true and a wrong plan); E2's was never trained and its level-0 loss under null is above ln 4096 = 8.32 (worse than uniform), so guidance on a model without plan dropout would be unsound. Plan dropout is what makes the null branch meaningful, as designed.
Guidance strength is `t = s * level / (levels - 1)` in the repo, so it is exactly 0 at level 0: the guided level-0 code is never changed (verified: at temperature 0, real plans, scale 2 vs 0, the level-0 code differs in 0 of 96, other codes differ in all 96 rows, so the control for "guidance does something" passes).
A stricter test failed (explained in Cycle 38): at temperature 0 with null plans, scale 2 vs scale 0 should give identical codes (both branches equal), but 96/96 rows differed in E3 (288 of 61,536 codes) and 38,688 codes in E2. I did not diagnose it; E3's 0.5% is consistent with floating-point differences between batch sizes being amplified by sequential decoding, but E2's 63% is not explained by that and I did not test it (E2's null prefix is untrained, so its null logits are probably large and fragile; a guess).

**End-to-end, 48 requests (plans v3, `quality_eval.py --n 48`; balanced = temp 0.8 + adjacency 10,8,0.8 + n-gram 3,8 + MBR8; base = 3 seeds):**

| config | retention | Gen. PPL | adj. repeat | repeated trigrams | entropy |
|---|---|---|---|---|---|
| run1, balanced | 0.6318 | 15.92 | 0.0087 | 0.0167 | 3.984 |
| E3, base | 0.5971 | 20.85 | 0.0156 | 0.0152 | 4.013 |
| E3, balanced | 0.6287 | 15.63 | 0.0076 | 0.0188 | 3.987 |
| E3, balanced + guidance s=1 | 0.6764 | 17.69 | 0.0104 | 0.0205 | 3.962 |

(Cycle 32 recorded 0.640 / 13.66 for run1 with temp 0.6 + adjacency 10,4,0.5 + MBR8, a different decoding; not comparable.) E3 equals run1 under the same decoding (retention 0.629 vs 0.632, PPL 15.6 vs 15.9), as on the validation plans. Guidance at s=1 raises retention by 0.048 and PPL by 2.05, the same direction and size as on validation (+0.031 and +0.81). **Stories read** (requests 0, 11, 23, 35; first 380 characters; run1 / E3 / E3 with s=1): all three have garbled spots ("fur fur", "the pup...", "her her", "around around", "Hello girl young", "gononfires", "replied Lily tree"). Guidance adds the plan's content where the others miss it (request 35: "a frog named Tim" appears only with guidance; request 0: "she found a lost little puppy ... I will comfort you" follows the plan; request 11: "festival ... bonfires" in all three) and adds some doubled words ("around around", "her her", "enjoyed enjoying"), which agrees with the adjacent-repeat rate rising from 0.0076 to 0.0104. Four stories, one reader.

**Judge on guided stories (k3, `/root/judge37`, 150 validation plans, pairs e3 s=0 vs s=1, s=0 vs s=4, run1 balanced vs e3 s=0, both orders):** finished (900 calls, 1.7 s each). Raw numbers below. New in `judge.py`: a position-bias-free margin, the mean of (A-B logit when X is first) and (B-A logit when X is second), which uses every call instead of only plans where both orders agree. Applied to the Cycle 33 raw calls (`/root/judge37/c33`): base vs `t08_ng3p8` margin -3.19 (bootstrap 95% CI -4.67 to -1.73; positive favours base), base ahead in 19 of 60 plans; base vs `t06_adj10_mbr8` -3.38 (-4.71 to -2.09), 18 of 60; `t08_ng3p8` vs `t06_adj10_mbr8` +0.09 (-1.30 to +1.55), 28 of 60. So the continuous margin supports the decoded stories over the base more clearly than the 8-0 tie-heavy count did, and still shows no difference between 0.8 and 0.6; judgment: Qwen3-8B can see the decoding gain and cannot separate the two decodings.

**Guided-stories judge result (150 plans, both orders, margin positive favours the first-named system; position A picked in 854 of 900 calls = 0.949):** `e3_s0` vs `e3_s1` (no guidance vs scale 1): margin +0.19 (bootstrap 95% CI -0.69 to +1.04), `e3_s0` ahead in 75 of 150; both-orders count 6 / tie 136 / 8, sign test p=0.79. `e3_s0` vs `e3_s4`: margin **+2.71 (CI +1.88 to +3.58)**, ahead in 104 of 150; count 16 / 132 / 2, p=0.0013. `run1_bal` vs `e3_s0`: margin -0.25 (CI -1.11 to +0.62), 71 of 150; count 6 / 137 / 7, p=1.0. Reading: (1) the judge sees the degradation at scale 4 (Gen. PPL 20.4 there), so it is sensitive to garbling of that size; (2) it finds no difference between scale 0 and scale 1 although retention rises 0.645 to 0.677, i.e. its combined fluency-plus-plan verdict is a wash at scale 1 (the interval, about +-0.9 logits, does not exclude a small effect either way); (3) it finds no difference between run1 and E3, matching the metrics. Limits: one judge model, one prompt that asks for fluency and plan-following at once, truncation to 110 words, 150 of 598 plans, position bias 0.95 (the margin cancels it by construction but a biased judge has less signal left).

Not done: nothing run on K3 itself; no human rating beyond the 4 + 4 + 3 + 6 stories I read across cycles; the 8 judged-by-both-orders count and the margin are from one prompt wording.

Artefacts: `ncm-merge/cfg_check.py`, `cfg_l0.py`, `e2e_c37.sh`, `judge.py` (env `JUDGE_ROOT`, `JUDGE_PAIRS`), `gpu-out/data37.json`; gpu `~/plan2text/out/e2e37/` (cfgcheck_*.json, cfgl0_*.json, run1_bal / e3_* e2e results).

---

## Cycle 38 — The temperature-0 null-identity failure is cuDNN TF32 convolution, not the guidance path

**Question (open item of Cycle 37):** with a null plan in both halves, guided generation at temperature 0 (scale 2) should reproduce the unguided codes, but did not (E3: 3 codes per row; E2: about 400 per row).
**What I measured (`cfg_diag.py`, one card, 48 identical null rows, so each model gives one trajectory, not 48 observations):** in the default configuration, E3 differed in 144 codes and E2 in 19,344. At the first differing level the unguided top-1 to top-2 margin was small against the difference between the guided run's conditional-half logits and the unguided logits: E3 first differs at level index 10 (margin 0.573, logit difference 4.49), E2 at level index 3 (margin 0.0046, difference 0.206). Both runs had identical codes at all earlier levels, so these logit differences are not caused by different inputs; they are arithmetic differences that depend on the batch size (96 rows guided, 48 unguided). With `torch.backends.cudnn.allow_tf32 = False` (and matmul precision forced to "highest") **both models give identical codes in all rows (0 differing codes)**. The matmul flags already read "highest" / TF32 off in the default run (printed), so the flag that actually changed is cuDNN's TF32 switch (default on). The NCM generator runs a Conv1d residual after each upsample (`models/quant.py`, `get_next_autoregressive_input`), and the repo sets no TF32 flag (grep). I did not toggle the three flags one at a time, so "cuDNN TF32 convolution" is the explanation consistent with what changed, not an isolated cause.
**Consequences.** (1) The Cycle 37 guidance checks stand: with full-precision convolutions the doubled batch reproduces the unguided run exactly for a null prefix, in both models. (2) Generation in this setup is not bit-reproducible across batch sizes (relative error of TF32 convolutions, amplified by greedy decoding); every table in Cycles 31 to 37 uses sampling at temperature 0.6 to 1.0 and MBR selection, where this is far below sampling noise, and all comparisons use the same batch sizes, so I see no effect on the reported differences; I did not re-run any of them with TF32 off. (3) For any exact-equality test between batch sizes, set `torch.backends.cudnn.allow_tf32 = False`.
Artefacts: `ncm-merge/cfg_diag.py`; gpu `out/e2e37/cfgdiag_*.json`.

---

## Cycle 39 — Human rating: set-up (ratings not collected yet when this entry was written)

**Direction (creator, 2026-10-22):** "let's do the human rating". The only readers so far have been me (about 17 stories across cycles) and a Qwen3-8B judge with 95% position bias, so the rating is done by the creator, blind.
**Design (fixed before any rating):** 20 validation plans (random seed 37, stories of at least 40 words in all four kinds) x four kinds of story: the true TinyStories window, E3 without guidance (s0), E3 with guidance scale 1 (s1), E3 with scale 4 (s4), all from the same balanced decoding used in Cycle 36 (s0, s1, s4 are the first run of each, the 598-plan files). 80 items plus 4 repeats of earlier items (test-retest), 84 in all, shown one at a time with the plan, in 5 chunks of 16 to 17 items; each chunk holds all four kinds for 4 plans, so stopping after a whole chunk keeps the comparison balanced. Item order inside a chunk is shuffled; item ids carry no information; the system of each item is in `human/key.json`, which the rater does not open. Two 1 to 5 scales: fluency (clean English: no garbled or doubled words, no repeated phrases) and plan-following (covers the plan's events in order). Stories are shown whole (about 100 words each, the same length for all kinds, so no truncation).
**Analysis fixed in advance (`analyze_rating.py`):** only plans with all four kinds rated; per-plan paired differences with a bootstrap 95% interval over plans and an exact sign test. Primary: s1 - s0 on both scales. Positive control: s0 - s4 on fluency (s4 had Gen. PPL 20.4 against 13.0, and the Qwen judge saw it); anchor: true - s0. Test-retest agreement from the repeats. The script was run only on random synthetic ratings written outside `human/`, to check that it runs; those numbers mean nothing.
**Limits stated up front:** one rater (the creator, who knows the project), 20 plans, so only differences of roughly half a point or more can show; a rater who sees four stories per plan in one chunk may compare across kinds, which the shuffling reduces but does not remove.
Files (`scratch-agents/hybrid-clean/ncm-merge/`): `build_rating.py`, `human/rate.html` (the page; ratings are saved in the browser and exported as CSV), `human/key.json`, `analyze_rating.py`, `decode_true.py`, `gpu-out/true_valid.json`.

---

## Cycle 40 — Ratings: the creator (20 items) and three model raters (all 84 items)

**Direction (creator, 2026-10-22):** he rated the first 20 items of the page himself (`human/ratings.csv`, r001 to r020; the page orders items in chunks, so these are all 16 items of chunk 1 and 4 of chunk 2) and asked for independent ratings of all items from three model subagents. Model names as available in the tool (the creator typed "gemini 4 agron", "gpt 6 astra", "claude opus 5.5"): Gemini 4 Argon, GPT-6 Astra, Claude Opus 5.5; the first call with "Gemini 4 Agron" failed on the name and I used the listed "Gemini 4 Argon". Each rater was a fresh subagent that got the same rubric as the page and one file with the 84 items (plan and story only, `rater_pack/items.txt`), was told to read nothing else and to answer with `id,fluency,plan` lines; I saved the replies unchanged as `human/ratings_gemini4argon.csv`, `ratings_gpt6astra.csv`, `ratings_claude_opus55.csv`. Whether a subagent respected "read nothing else" (for example did not open `human/key.json`) I could not check.
**Analysis:** `analyze_multi.py`, the same comparisons as set before (per-plan paired differences, bootstrap interval over plans, sign test, only plans with all four kinds rated), now per rater and for the mean of the three model raters.
**What the model raters say (20 plans each; mean of the three raters in brackets):**
- Primary, guidance 1 minus no guidance: fluency +0.10 / +0.10 / 0.00 for Gemini / GPT / Opus (mean +0.07, interval -0.17 to +0.33, sign p=0.45); plan-following +0.15 / +0.15 / +0.05 (mean +0.12, interval -0.13 to +0.35, p=0.34). No rater finds a difference.
- Control, no guidance minus guidance 4, fluency: +0.80 / +0.65 / +0.70 (mean +0.72, interval +0.43 to +0.98, p=0.001): all three see the degraded text at scale 4. On plan-following the same difference is -0.05 / -0.10 / +0.05, i.e. none.
- Anchor, true minus no guidance: fluency +2.45 / +2.85 / +2.35; plan +0.95 / +1.10 / +0.80. All three gave every true story fluency 5 (GPT 5.00, Gemini 5.00, Opus 5.00) and plan 5 except two GPT scores of 4; generated stories average 2.2 to 2.7 on fluency and 3.8 to 4.2 on plan-following.
- Agreement among the model raters is high on fluency (Spearman 0.93 to 0.94 over 84 items) and lower on plan (0.79 to 0.84). Test-retest on the 4 repeated items: identical scores for all three raters (4 of 4 on both scales). These raters share training data and prompt, so their agreement is not independent evidence.
**What the creator's own ratings say (20 items, only 4 plans have all four kinds, so no comparison is meaningful):** means on those 4 plans, fluency true 4.00 / s0 3.25 / s1 3.00 / s4 2.75, plan 3.75 / 3.75 / 3.25 / 2.75; intervals include zero everywhere. One comment on r002 reads "it's the same story", consistent with the cross-comparison risk named in the set-up.
**Model raters against the creator on the same 20 items:** Spearman 0.59 to 0.63 on fluency and 0.60 to 0.63 on plan per rater (mean of the three: 0.63 and 0.67). The models are more lenient on plan-following: model minus creator +1.05 (Gemini), +0.80 (GPT), +1.05 (Opus) points on average, and on fluency -0.25, -0.55, -0.20. Mean absolute difference about 1 point on both scales, exact agreement 4 to 7 of 20.
**Reading (judgment):** (1) Fluency: the models see no gain or loss from guidance 1 and a clear loss at guidance 4, matching the Qwen judge and Gen. PPL; the creator's 20 items cannot confirm or contradict it. (2) Plan-following: neither the models nor the Qwen judge find any change from guidance although the plan-word retention rises 0.645 to 0.677; the models score plan-following near 4 for every generated story, which leaves little room to show a difference (ceiling), and they are about one point more lenient than the creator, so I read the plan scores of the models as weak evidence. (3) With only 20 of 84 items rated by the creator, the creator-versus-model agreement (about 0.6) rests on 20 items per scale and its interval is wide; it says the models are related to, not a substitute for, his ratings. The remaining 64 items are not human-rated.
Artefacts: `ncm-merge/analyze_multi.py`, `make_rater_pack.py`, `human/ratings*.csv`, `rater_pack/items.txt`.
