# Clover NCM + HybridAgent: what the papers and our history say about training, and where the 2026-11-17 runs stand

Written 2026-11-18 by the AI session that ran the 2026-11-17 night runs, after the human asked it to read AGENTS.md, both
official papers and the git history first, "step by step", and to keep its own context file with a meaningful name.
This file is offered, not imposed (AGENTS.md section 5): the human decides whether to keep, correct or delete it.
It does not edit the existing `clover-hybridagent-research-context.md` (the earlier sessions' record).

States used below follow AGENTS.md section 11: **known** (read from a source or measured), **inferred** (my reading,
not verified), **unknown**.

## 1. Intended outcome (human's words, this session)

Train the per-model C2 (HybridAgent link/bridges) and C3 (NCM) on real data so that Clover's stages work, starting
with Clover-35B (box 1) and Nemotron (box 2), Kimi after its download. Before training: a complete picture of the
setup (2026-11-17), then "read AGENTS.md fully and the official research paper fully for both NCM and hybridagent
first and look at git history ... create your own context file ... first do this then let's see where we are".

## 2. What was read, and how

| Source | How read | Complete? |
|---|---|---|
| `AGENTS.md` (clover-framework) | `read_file`, lines 1-1300 | yes, all 22 sections |
| NCM paper, arXiv 2610.14162 (Kumbong et al., 9 Oct 2026) | PDF downloaded, text extracted with pypdf (40 pages, 111,653 chars), read in chunks | yes: main text, appendices A-J, tables, samples |
| HybridAgent paper, arXiv 2610.18861 (Zou et al., 13 Oct 2026) | same (45 pages, 138,632 chars) | yes: main text, appendices A-I; Appendix J (case studies) not read |
| git history | `git log --all` by keyword (NCM 44, HybridAgent 76, HybridLink 4, VQ-VAE 4, C3 29 commits) and the commit texts of the cycle commits | commit messages and the diffs of `bfa024fa`; not every diff |
| `clover-hybridagent-research-context.md` | read in earlier turns: What HybridAgent is / How it is trained / What the code shows, Cycles 25-33, 47-50, 66-68 | partly: the rest of its 68 cycles not re-read |

Not reachable: `clover-ncm-research-context.md`. Several commits and docs cite it, but the research-context file itself
says it is "untracked ... awaits his decision"; it exists in no worktree and in no commit I could find (**unknown**
contents; the human may have it).

Local copies of the papers: `%TEMP%\papers\{ncm,hybridagent}.{pdf,txt}` on this PC only, not in the repo (the NCM
paper is under arXiv's non-exclusive distribution licence).

## 3. What the NCM paper establishes (known, from the paper)

- Two stages: a multiscale residual VQ-VAE tokenizer (levels of 1..L codes, nested dropout) and a transformer that
  generates one level at a time, all codes of a level in parallel.
- Conditioning is a **prefix LM**: prompt tokens are embedded with the frozen VQ-VAE encoder's token embedding table
  and projected to the hidden width. A frozen T5-small encoder was only +0.10/+0.15/+0.24 ROUGE better and was dropped.
- Conditional tasks tested: WMT14 De->En translation and XSum summarization. In both, the target is a transformation
  of text that is in the prompt. No QA, instruction, dialogue, or LLM-hidden-state prefix appears anywhere in the paper.
- "We train a separate tokenizer and generator for each dataset" (section 4). XSum and WMT14 generators are trained on
  unpaired text first, then on source-target pairs.
- Budgets (Table 5/6): WMT14 tokenizer 60,000 updates at global batch 4,096 (144.00M params, 11 levels, 4,096 codes,
  Qwen3-0.6B vocabulary 151,680); WMT14 generator 150,000 updates at batch 2,048 (119.96M params, prefix 96, target 64).
- Ablations: moderate code corruption "greatly improves" quality, too much hurts; loss weighting alpha 0..1 barely
  matters (Gen PPL 18.9-19.0, 19.4 at 1.5); block-diagonal attention is best.
- Stated limits (Appendix I): fixed hierarchy depth and lengths, separately trained tokenizer, ~100M-725M parameters,
  occasional artifacts at coarse-level transitions. Tokenizer errors are mostly substitutions of related words,
  entities and numerals (Table 11).
- Sample quality shown is modest: the paper's own conditional samples contain repeated words and wrong entities
  ("Samsung, BMW and HTC", "operating operating system").

## 4. What the HybridAgent paper establishes (known, from the paper)

- HybridLink `R(h) = h + W2 GELU(W1 h)` maps the last-layer hidden state to the next input embedding; the paper's
  version has no LayerNorm and ~2*d^2 parameters (the released code adds two LayerNorms; see the research context).
- Training is **post-training of the backbone**, not a frozen link on an arbitrary model: format activation (SFT), then
  HybridLink activation (backbone frozen; loss `1.0*(1-cos) + 0.2*MSE` against the next token embedding), then
  **self-conditioned HybridSFT** (two passes, backbone and link trained jointly), then HybridRL (GRPO, link frozen).
- Table 3 (Qwen3-14B-Base, AIME25/AIME26/MATH500): base 4.9/3.0/47.0; + format 20.6/19.8/62.4; + link activation
  23.8/22.5/73.8; + self-conditioning 32.5/31.7/84.2. Activation alone is called "insufficient"; self-conditioning added
  an average of 30.5% over activation.
- Evidence of use of the latent content (Table 5, MATH500 8B): cross-problem replacement 44.8, random-token 41.2,
  intact 84.6.
- Data: 54K hybrid trajectories converted from DAPO/ARPO/ToolStar-style agent data; RL on 12K math + 1K search
  prompts, 150-200 steps, rollout batch 512; all on 32 H100s.
- Models: Qwen3-Base 4B/8B/14B only. The only generality statement is a figure caption: "can be converted from any
  standard pre-trained LLM through a dedicated post-training pipeline". No other model family was run.

## 5. What our git history and context files add (known = read; I did not re-run these)

- 2026-10-21..28: one commit per research cycle on the HybridAgent context file (Cycles 1-68). The scripts behind them:
  `clover-kimi-k3/research/hybridagent-ncm/ncm-merge/` (commit `a6be3e25`).
- Cycles 27-41 (stories): released TinyStories NCM + stage-2 plan-to-text trained on 78,823 HybridAgent-8B plans
  (5,000 steps, global batch 256). Retention 0.612 vs 0.085 control; Gen PPL 16.35 vs true text 7.66.
- Cycle 47-48: WMT14 tokenizer reconstructs Kimi outputs at 0.95-1.00 token accuracy; a stage-2 trained with a
  **grounded** plan (plan built from the target window) and a combined prefix `[R(h) 48] + [BPE plan 48]` reached
  content-word retention 0.886 on 500 held-out windows (ungrounded XSum-summary plans: 0.33).
- Cycle 48 modality ablation: both 0.8857, plan tokens only 0.8711, `R(h)` only 0.3961.
- Cycle 50: adding a plan copy-pointer decoder moved exact number recall 0.3427 -> 0.7527 and Kimi factual keys
  37/47 -> 39/47. DSpark verifier removed (no effect).
- Cycle 66: the Kimi "E12" training used 1,490 curated Q&A/conversation/CS windows through all 93 layers and trained
  the NCM end to end with the decoder unfrozen; reported ncm_ce 3.41 -> 0.38 (train).
- **Cycle 42 on the `ncm-merge-qa` branch (local, not pushed; its worktree is gone)** is the gate for QA/reasoning text:
  the released WMT14 tokenizer reconstructs one-line GSM8K answers at 0.983 token accuracy, 0.993 number recall, with
  no training; **no generator was run**; 67% of GSM8K answers exceed one 64-token window; decision (3) was left to the
  human: "the released WMT14 generator ... fine-tuned on plan-to-answer pairs, which also needs its own gate".
- Production (commit `3ae993ea`, 2026-11-10): `c1_inference_only` takes C1 states from a llama-server with
  `--embeddings --pooling none`; C2 = plan-trunk-0 + JEPA; C3 decodes "all 12 NCM blocks over 48 plan positions and
  levels to 64 (as trained)". Same extraction method as the 2026-11-17 runs.
- Earlier this session (Studio, 2026-11-17, recorded in `clover-ai-system/studio/CONTEXT.md`): C3 returned its input
  unchanged on 18/18 texts and for zeroed/random C2 state; the Kimi link read punctuation (0/65 next-token) because it
  was trained on a 2-layer stand-in's states with 8 templated Q&A.

## 6. What was done on 2026-11-17 (box 1, Clover-35B; one run each; all counts from result files read on 2026-11-18)

Data: 6,000 prompts (OASST1 Apache-2.0 + Dolly CC-BY-SA-3.0, tree-disjoint, prompts unique): 5,000 train / 1,000
held out; Clover-35B (Q4_K_M, bartowski GGUF, sha256 verified) answered each greedily with no thinking block; per-token
final hidden states exported for prompt+answer through llama-server embeddings (2048-d).

| Result | Value | Controls |
|---|---|---|
| Model's own lm_head on exported held-out states, top-1 next token | 0.9429 | shuffled states 1.4% (5-example check) |
| C2 link, bottleneck 2048, held-out next-token readout top-1 / top-5 | 0.3808 / 0.4605 | zeros 0.045, random 0.015, shuffled 0.0125; raw state no link 0.005 |
| C2 link, bottleneck 512 | 0.038 | same as controls: failed |
| VQ-VAE vocabulary graft Qwen3 -> Clover (131,612 shared strings) | zero-shot recon 0.9109 | ridge held-out cosine enc 0.104, dec 0.916 |
| VQ-VAE adaptation, 1,500 steps batch 64 | recon 0.911 (no change) | |
| C3 "R" (answer decoded from the answer's own states, prefix = bridge only) | token acc 0.544, word Jaccard 0.373 | shuffled 0.007, zeros 0.011, random 0.004 |
| C3 "R" via the C2 link | 0.480 / 0.283 | collapses likewise |
| C3 "G" (prompt states only) | 0.031 / 0.094 | 0.006-0.008 |

Not done: reading the decoded text samples (they are stored in `/root/c3/c3_*.json` on box 1); the Nemotron pipeline
(model downloaded and verified, answers a test question, nothing exported); Kimi (1.4 TB downloaded, files not checked
against `dataset-files.tsv`, engine not run); nothing was pushed to Hugging Face or k3.

## 7. What the papers and history say about those results (inferred, labelled)

1. **"G" (answer from prompt states alone) was the wrong test (inferred, strong).** Neither paper nor any cycle
   claims NCM turns a question's hidden states into an answer. NCM renders a prefix that already holds the content
   (translation source, article, or a grounded plan). The research design puts the content in a plan (HybridAgent text
   plan, Cycle 29-31/48) and, optionally, `R(h)`. My G result is consistent with that and does not test C3.
2. **"R" is closer to what the paper trains but is not the research design (inferred).** R gives C3 the model's own
   answer-position states with no plan tokens. The research's best configuration had plan tokens in the prefix and a
   copy-pointer decoder (Cycles 48, 50). I used neither, and 96 prefix positions where production trains 48.
3. **Training scale is far below the paper's and below our own earlier cycles (known numbers).** Paper WMT14
   tokenizer: 60,000 x 4,096; generator 150,000 x 2,048. Cycle 48: 16,000 grounded windows, 2,000 steps at global
   batch 256. Mine: adapt 1,500 x 64; C3 2,500 x 32 on 4,970 examples.
4. **The VQ-VAE adaptation not improving (0.911 -> 0.911) is unexplained (unknown).** Candidate causes I can name but
   have not tested: windows of mostly short chat text vs the XSum/GSM8K windows Kimi's adaptation used; the new
   116,708 ridge-initialised tokens (encoder ridge cosine only 0.10) never seen in training windows; the frozen
   codebooks. The research's Kimi adaptation reached 0.960 on held-out XSum windows after 800 steps at batch 512.
5. **The link result is the strongest piece (known numbers, inferred meaning).** It matches the paper's "HybridLink
   activation" objective on a frozen model, with 101,743 held-out positions and controls that collapse. The paper
   itself says activation alone is "insufficient" without self-conditioned training that updates the backbone, which
   we did not do and cannot do with a GGUF.
6. **Qwen lineage (known facts + inferred consequence).** HybridAgent is Qwen3-Base; the WMT14 tokenizer uses the
   Qwen3-0.6B vocabulary; Clover-35B is derived from Qwen3.6 (model card) and 131,612 of Qwen3's 151,669 token
   strings exist in its vocabulary. So Clover-35B is the closest of the four models to the tested recipe. Kimi,
   Nemotron and LFM3 need real vocabulary grafts. Whether the Qwen3-8B HybridAgent's latent space transfers to
   Clover-35B through the existing `QwenToClover35B` bridge is **unknown** (the Kimi analogue read as junk on real states).

## 8. What is ruled out (this session's evidence)

- A 512-bottleneck link on real Clover-35B states (no better than controls).
- Reading "G" as evidence about C3's ability (see 7.1).
- Extra VQ-VAE steps at batch 64 on this data as a fix for reconstruction (no change in 1,500 steps).

## 9. What remains unknown

- The decoded text of the C3 outputs (stored, unread).
- Whether C3 with grounded plan tokens + `R(h)` + a copy decoder, at the research's schedule, works for Clover-35B.
- What `clover-ncm-research-context.md` says.
- Whether `QwenToClover35B`/HybridAgent-8B latents carry usable content for Clover-35B.
- Nemotron and Kimi: everything beyond download/model-load.
- Whether the human wants Qwen-derived HybridAgent-8B used as plan writer (the CSI research boundary says "No
  Qwen-based components"; Clover-35B itself derives from Qwen3.6).

## 10. What should be different in the next cycle

- Direction first: the human decides what C3 is for (render a plan) and the plan source (HybridAgent-8B text mode,
  the model's own plan, or both). Do not train another "answer from states" variant.
- If training C3: grounded plans in the prefix, 48 plan positions as production does, `R(h)` optional, copy-pointer
  decoder, the research's corruption and dropout recipe, and budgets stated against the paper's.
- Read the stored decoded samples before reporting any number as quality.
- Check what the VQ-VAE adaptation windows contain before blaming the step count.
- Keep context records in `docs/runtime-context/<topic>/<subject>-context.md`, not appended to another work's file.

## 11. Housekeeping done in this cycle

- Moved my four sections out of `clover-ai-system/studio/CONTEXT.md` (untracked, about the Studio integrity work) into
  this file; backup of the original: `%TEMP%\studio-CONTEXT.before-trim.md`.
- Nothing committed, nothing pushed, no tests modified.

## 12. Records carried over from the sections moved out of studio/CONTEXT.md (2026-11-17)

**Kimi, Qwen-bridge readout (one run, k3 CPU, ~1.1 s per prompt; known).** Real Kimi final states (93-layer engine,
correct `normalization_process`, 2 prompts, 36 and 31 positions) were passed through `qwen_to_k31_bridge` (`to_ha`
+ MLP, 4096-d) and read out as nearest tokens over HybridAgent-8B's embedding table, and via `from_ha` over Kimi's.
Real states: mean cosine 0.069-0.078; random 0.069-0.084; zeros 0.068-0.085: no difference from controls; readouts
are junk tokens; zeros collapse to one token. Limits: the readout (nearest embedding token) may be a weak instrument
for a bridge latent; HybridAgent continued from the vector was not tested (no embedding-input path in either engine).
Cause found in `train_kimi_k31_hybridagent_gpu.py` / `kimi_k31_gpu_backbone.py`: the Kimi link and bridge were
trained on states of a 2-layer stage model (trunk-0 + Layer 3 + tail), 8 hand-written Q&A templates (640 train / 64
eval), 40/60/60 steps. Real 93-layer states are out of that distribution (hypothesis; retraining not run).

**System layers (2026-11-17, from `CLOVER_FRAMEWORK_ARCHITECTURE.md`, `semantics.py`, `personal.py`, `collective.py`;
tests run on this PC with Python).** C1 Inference, C2 Plan, C3 Decode; S1 Memory, S2 Relationships, S3 Tools;
engines around them: semantics (rules), CSI (research, off by default), JEPA (NumPy `CLVRTRK1` trunks), personal
learning, collective learning. Test results: `test_semantics_frame` 1 fail now (103 passed at HEAD; the failing check
expects the appended correction note that the integrity cycle removed; the human approved updating such tests, not
yet done); `test_personal` 2 fails (the same two on a clean HEAD copy; git-filter-repo absent); `test_collective` 35
pass; `test_mobile_s1_s2_s3` 13/13 pass. Not measured: whether any of those engines change answer quality for the
four hosted models.

**Direction recorded 2026-11-17 (human, pasted plan, checked against the repo).** Four independent systems (Kimi,
Nemotron, Clover-35B, LFM3 Omni), each with its own NCM; C3 is not considered complete until it decodes; build the C3
corpus from real C1 -> C2 -> C3 traces; train order VQ-VAE, NextConceptTransformer, LexicalPointerDecoder, end to end;
first experiment about 5,000 conversations with 1,000 held out. Corrections found by checking: the repo's names are
C1 Inference / C2 Plan / C3 Decode; Clover-35B is its own C1 (not LFM3 Omni). Licences read from the HF API on
2026-11-17: oasst1 Apache-2.0, dolly CC-BY-SA-3.0, WildChat-1M ODC-By, FineWeb ODC-By, No Robots CC-BY-NC-4.0 (excluded
by the non-commercial rule).

**Machines (2026-11-17/18).** Box 1 (GB10, 16 GB disk, 60 GB /dev/shm): Clover-35B work, files under `/dev/shm/data`
and `/root/c3` (lost if the container recycles). Box 2 (GB10, 2 TB disk): Nemotron Q4_K_M (lmstudio-community,
sha256 verified, answers a test question at ~26 tok/s) and the Kimi download (1.4 TB on disk, not yet verified
against `src/scripts/dataset-files.tsv`). Notes on both are in repo memory `gx10-training-box.md`.

Raw text of the four moved sections: `%TEMP%\moved-sections.md` (this PC only).
