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
