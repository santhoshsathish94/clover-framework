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
