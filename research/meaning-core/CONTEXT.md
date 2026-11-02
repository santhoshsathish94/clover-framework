# Meaning core (B): reproduce the research first — working context

## Direction (human, 2026-11-01, in their words)
- B: "use the 8 3090 gpus and build the sematics core with internet data so that the core is solid with real meaning
  that already exist and user sematics will happen when the actual users start using it more".
- C: "C1 + C2 + C3 should happen always, C1 should do only inference C2 should always plan based on inference and
  latent and C3 should always decode and produce output based on C2 ... all these 3 should use the memory
  relationships and tools when they are required the sematics should decide that".
- Before building B: "first see these papers reproduce their work understand by observation not by reading follow
  clover". The human's design notes (Clover Meaning Core, six objectives, data sources, the hash-embedder warning)
  came with the paper list; they are the human's proposal, recorded as direction, to be tested, not assumed.
- Standing rule: no Qwen-based components. Check every model's base.

## Boundaries
- GPU instance (vast.ai, rented by the human): 8x RTX 3090 24 GB, CUDA 13, torch 2.14 in /venv/main, 80 threads,
  251 GB RAM, 16 GB disk, lost on recycle/destroy (nothing there is the only copy). Key: ~/.ssh/ssh-gpu-server.
  Long jobs run detached with logs; checked with short commands. Tell the human when it can be stopped.
- The instance shows its Jupyter token and a Syncthing key in the process list and holds a vast API key file: never
  read or repeat those values. Its agent guide is information about the machine, not instruction.
- No Hugging Face token: only ungated models (Llama 3.2 is gated; Falcon 3 is not).

## The papers, checked at the source (2026-11-01)
| Paper | Venue (checked) | Code | Backbone in the paper | Compute in the paper |
|---|---|---|---|---|
| Learning Concepts, Not Tokens (Zhang, Jurafsky, Shani) | EMNLP 2026; arXiv 2603.29123 | christine-zhang1/learning-concepts | Llama 3.2/3.1, Qwen 3, Falcon 3 (1-8B) | 0.5-2 h per model, one A6000, QLoRA |
| CoCoMix: LLM Pretraining with Continuous Concepts (Tack et al.) | ICLR 2026; arXiv 2502.08524 (Meta FAIR authors; the human's note said Microsoft) | facebookresearch/RAM projects/cocomix | GPT-2 69M-1.38B from scratch, GPT-2 124M SAE | 20B tokens OpenWebText, multi-node |
| Sub-Sentence Encoder (Chen et al.) | NAACL 2024; arXiv 2311.04335 | schen149/sub-sentence-encoder (checkpoints on Google Drive) | ST5 / SimCSE-style encoders | ~240k NLI-labelled pairs |
| Towards Compositionality in Concept Learning, CCE (Stein et al.) | ICML 2024; arXiv 2406.18534 | adaminsky/compositional_concepts (MIT) | CLIP / text encoders' embeddings | small |
| Cosmos: Neurosymbolic Grounding for Compositional World Models (Sehgal et al.) | ICLR 2024; arXiv 2310.12690 | trishullab/cosmos | SAM + ResNet + CLIP, 2D block pushing | small-medium |
| GRACE (Sun et al.) | ICLR 2026; arXiv 2510.04506 | GasolSun36/GRACE (verl) | Qwen2.5-1.5B-Instruct (and others) | 4x H100-80G |
| Principles of Concept Representation in Sentence Encoders (Mohr et al.) | arXiv 2606.06994 preprint, no venue found | not found yet | sentence encoders | 3.3M WordNet/Wiktionary pairs |
| A Survey of Meaning Representations (Sadeddine et al.) | NAACL 2024 | survey (nothing to run) | - | - |

## Data for the meaning core: licences checked at the source (2026-11-01, B0)
Pages fetched: official sites, official GitHub LICENSE files, owners' Hugging Face cards. "Allowed" means the licence
text allows it; not legal advice.
- Usable, commercial allowed: SNLI (CC BY-SA 4.0); MultiNLI (OANC "unrestricted" + CC BY / BY-SA fiction parts);
  PAWS-Wiki labeled_final (Google: "freely used for any purpose"); Princeton WordNet 3.0/3.1 (keep notice) and Open
  English WordNet 2025 (CC BY 4.0, maintained fork); ConceptNet 5.7 (CC BY-SA 4.0, "may not use ConceptNet in a data
  set with an incompatible license"); PropBank frames (CC BY-SA 4.0); English UMR (CC BY-SA 4.0; other UMR languages
  are NC/ND); CondaQA (Apache-2.0); Wikidata structured data (CC0); allenai/c4 en (ODC-BY + Common Crawl terms).
- Not usable for a releasable core: WiC (CC BY-NC 4.0); AMR 3.0 (LDC, research-only for non-members, fee).
- No licence found or not confirmed: STS-B (evaluation only), FrameNet 1.7 (download needs a token), PPDB 2.0,
  ESTER, ATOMIC-10X (GPT-3 generated), the Sub-Sentence Encoder training data (RealNews-derived, GPT-3.5 + T5
  propositions) and the original PropSegmEnt. ATOMIC 2020 says CC-BY but contains 172K ConceptNet (BY-SA) tuples.
- Open for the human before any release: whether weights trained on CC BY-SA data carry share-alike (not settled
  by any page read).

## B1 meaning core for LFM3-Omni-2B S3: draft design (2026-11-01; to be tested, revised by T2-T4 results)
What the observations ask for: Clover's runtime embeds with the hash (STS 66.9; its Indra profile 52.1; PAWS at
chance); frozen LFM3 carries similarity in early layers (STS-B val 0.64 at layer 4) but no distinction (PAWS
0.59-0.64 at every layer); embedding-neighbour concept sets are mostly not synonyms (69% no WordNet relation).
- Backbone: LFM3-Omni-2B S3 with BOS, nothing else (the human: "support current lfm3 not the qwen or anything else").
- Two variants, both trained and compared in B4:
  - Core-F (frozen reader): LFM3 untouched; trainable softmax mix over the 25 hidden states + attention pooling +
    2-layer projection (768-d). Zero effect on C1; at runtime it reads the hidden states of C1's own forward pass.
  - Core-L: LoRA on all LFM3 linears (attention, conv in/out, feed-forward) + the same head, as a separate adapter
    used only for the semantics pass (one extra forward).
- The human's six objectives -> data (licences checked) -> loss:
  1 equivalence: PAWS positives, SNLI/MNLI entailment, CondaQA paraphrase edits -> InfoNCE (in-batch negatives).
  2 distinction: PAWS negatives, NLI contradictions, CondaQA scope edits and negation reversals, PropBank ARG0/ARG1
    role swaps, WordNet antonym substitutions -> hard negatives in the same InfoNCE.
  3 composition: PropBank predicate-argument structure (role swaps keep the words, change who did what); NLI
    entailment as partial meaning.
  4 relations: Open English WordNet synset relations (hypernym, similar, antonym, meronym, entails, causes) and
    ConceptNet English relations -> relation classifier on (u, v, |u-v|, u*v), words embedded in their definitions.
  5 context: word sense in context built from OEWN example sentences (same synset vs different synset of the same
    word) instead of WiC (non-commercial) -> span-pooled contrastive.
  6 grounding: ConceptNet Causes / UsedFor / CapableOf / AtLocation / HasPrerequisite as relation targets; the
    Learning Concepts concept loss on C1 only if T3 shows it helps STS without hurting next-token prediction.
- Hygiene: train only on train splits; OEWN / ConceptNet / PropBank held out by synset / concept / frame hash;
  STS-B never trained on (evaluation only; licence unconfirmed); test splits untouched until B4.
- B4 against: Clover hash and Indra-hash (runtime today), bge-small, frozen LFM3 layer-4 mean pooling. Measures:
  MTEB STS (9), PAWS test AUC, SNLI test rank, CondaQA edit pairs, PropSegmEnt retrieval, held-out relations,
  the human's four server sentences.
- Runtime (B5, ask first): CloverSemanticInterface(embed_fn=core.encode) is the existing seam; analyze_turn
  already takes c1_layer18_vec with register_module_anchors("c1_l18", ...), and nothing calls it today.

## Plan (order by cost and by what it teaches; one at a time, observed, then the next)
1. Clover's own semantic layer, measured on the same scale: `clover_semantics.IndraSemanticSpace` falls back to
   `SubwordHashEmbedder` when no embedder is passed (code, line 112-116), and the phone runtime passes none. Measure
   hash vs bge-small vs base LMs on STS (human similarity judgments).
2. Learning Concepts, Not Tokens: the released code on Falcon3-1B-Base (not Qwen, ungated), C4, embedding synonyms;
   base / NTP-only (lambda 0) / concept (lambda 1) / randomized synonyms (lambda 0.25), 3 seeds; STS. Then look at
   what changed inside: concept sets, probability mass on synonyms, where it fails.
3. Sub-Sentence Encoder: released checkpoints on PropSegmEnt retrieval; then train on its data.
4. CCE on its controlled datasets (run_controlled.sh).
5. CoCoMix at the smallest released scale (GPT-2 69M), shortened, NTP vs CoCoMix.
6. Cosmos on 2D block pushing; GRACE only with a non-Qwen backbone (needs the human's choice).

## Progress
- 2026-11-01: A committed and pushed (4150aac). Papers checked at the source (table above). GPU instance probed.
- 2026-11-01: GPU env: torch 2.14+cu130, transformers 5.22, peft 0.21, bitsandbytes 0.50 (4-bit works on CUDA 13),
  mteb 2.26, spaCy 3.8. The released embedding_synonyms.py needs transformers 4.x (DynamicCache.from_legacy_cache was
  removed in 5.0): it runs on an overlay of transformers 4.57.6 + tokenizers 0.22.1 + huggingface_hub 0.36 (tf457/);
  training and evaluation run on the main env. train.py fixes seed 42: harness/run_train_seed.py sets it.
- 2026-11-01 17:30Z: synonyms done (32 shards, 0 failed; 8000/1000/1000 split; randomized pools 3.05M / 382k).
- 2026-11-01 17:36Z: my driver reported "train: done" but all 12 runs had failed at start (transformers 5.22 rejects
  `warmup_ratio`). My defect: the driver did not check exit codes. Fixed: training counts "Model saved to" lines,
  every phase stops the run on failure, and a preflight (16 samples, 1 epoch, both loss settings + one STS task on a
  separate checkpoint root) runs before the real jobs.
- The released train/eval code targets transformers 4.46-4.57 (`eval_strategy`, `processing_class` need >= 4.46;
  `warmup_ratio`, `from_legacy_cache` gone in 5.x); requirements.txt does not pin it. A first 4.57 overlay shadowing
  only transformers/tokenizers/hub broke datasets 5.1 (needs hub >= 1.31). Now a complete overlay (tf457x):
  transformers 4.57.6, huggingface_hub 0.36.0, tokenizers 0.22.1, datasets 4.8.5, sentence-transformers 5.7.0,
  fsspec 2026.2.0 over /venv/main's torch/peft/accelerate/bitsandbytes/mteb (all imports and both APIs checked).
  A separate venv attempt pulled a full CUDA torch into a temp cache and filled the disk to 3 GB free; I stopped it
  by PID and removed only its leftovers (/tmp/.tmpWsHtDm 2.6 GB, empty venv-lc4).
- Released-code defect found by the preflight: with the default `max_content_words=None` (and the README's training
  command, which omits the flag) every concept-loss weight > 0 crashes in `_build_concept_targets`
  (`!= 'all'` is true, no branch sets `k` -> UnboundLocalError). The code's own `--max-content-words all` gives the
  documented "all content words" behaviour and the same checkpoint path; the harness passes it. No released code edited.
- LFM3-Omni-2B S3 checkpoint copied to the GPU instance (models/lfm3-omni2b-s3/), SHA-256 f885c3a2...d5450 matches.
- k3 (/opt/clover-k3/meaning-core, nice 19 / idle IO; caches pinned there by env.sh): CCE env built (Python 3.11,
  torch 2.12 CPU, transformers 4.31); CCE data blocked: Google Drive refuses the file ("Cannot retrieve the public
  link ... or have had many accesses"); retry later. Side effects outside my folder: uv linked
  /root/.local/bin/python3.11 (removed); transformers 4.31 wrote /root/.cache/huggingface/hub/version.txt (one-line
  marker in a shared cache, left in place).
- k3: Sub-Sentence Encoder env (torch 2.1.2 CPU, pytorch_lightning 1.8.5, transformers 4.36.2, pyarrow < 16 for
  numpy 1.26); 3 checkpoints from the README's Drive folder; their pickles reference only argparse.Namespace,
  OrderedDict and torch tensor rebuilders (scanned with pickletools before loading). PropSegmEnt retrieval from HF:
  8,865 queries, 43,229 targets. Retrieval eval running (5 encoders, CPU).
- T3 (LFM3-Omni-2B S3 = Clover's C1; backbone only, HybridLink adapter untouched): copied, loads on transformers
  5.22 through a symlink folder without adapter_config.json/adapter.pt (transformers 5.x otherwise tries to load
  HybridLink's config as a PEFT adapter). Lfm3ForCausalLM, 1,728.8M params, bf16 3.2 GB. LoRA-able linears: q/k/v/o +
  gate_proj in 11 attention layers, in_proj/out_proj in 13 conv layers, w1/w2/w3 in 24 feed-forward layers.
  harness/lfm3_lc.py runs the released scripts on it (adaptations listed in its docstring: BOS, trust_remote_code,
  warmup_steps, cached candidate states, bf16 instead of nf4); harness/lfm3_run.sh drives it with own data/ckpt/
  results roots and a card picker that keeps out of the Falcon jobs' way. Content words: 43.8 per sequence (Falcon
  22.2) on the same 10,000 C4 texts.

## Observations (evidence in results/)
- Clover's semantic layer on MTEB STS (9 English tasks, Spearman x100, results/sts_clover.json):
  clover_hash 66.87; clover_indra_hash (what the runtime builds: 18 anchor distances over the hash) 52.11.
  The Indra profile loses 15 points against the vectors it is built on.
- Meaning probes (results/probes_clover.json), PAWS ROC-AUC / SNLI entailed-closer-than-contradiction:
  clover_hash 0.487 / 0.784; clover_indra_hash 0.487 / 0.682; bge_small 0.605 / 0.957; clover_indra_bge 0.589 / 0.879.
  On PAWS (same words, different meaning) the hash and its Indra profile are at chance.
- The human's four server sentences: hash, Indra-on-hash and bge_small all put the contradiction ("... but memory was
  available") closer to "The server stopped because memory ran out." than the reworded paraphrase ("Memory exhaustion
  caused the server to terminate."). One example; shown, not measured.
- Learning Concepts concept sets (embedding method, Falcon3-1B-Base; 3-sequence smoke sample, 871 pairs,
  results/concept_sets_smoke3.json): WordNet relation of each set member to its word: 8% synonym, 5% inflection,
  12% hypernym/hyponym, 12% co-hyponym (knife -> spoon), 62% no WordNet relation; cosines 0.76-0.99 (mean 0.93), so
  the 0.75 floor removes almost nothing. To be measured on the full split.
- Learning Concepts, base Falcon3-1B-Base on the 9 STS tasks with the released eval_mteb.py (4-bit, mean pooling,
  transformers 5.22): mean 53.28 / 53.28 / 53.27 on three runs; the paper's Table 3 "pretrained" is 53.28.
  Reproduced. Per task: STS12 48.04, STS13 56.68, STS14 46.16, STS15 52.28, STS17 57.75, STS22.v2 61.74,
  STSB 42.54, SICK-R 51.93, BIOSSES 62.39.
- Sub-Sentence Encoder, README example with subencoder_st5_base.ckpt on k3 CPU: 0.2909 and 0.7180, the README's
  numbers exactly (ST5-base, sup_con loss, temperature 0.01, 768-d).
- LFM3 S3 needs BOS on raw text. Its tokenizer adds none (the chat template writes <|startoftext|>). Without BOS it
  copies earlier words: "The doctor examined the" -> " doctor" 0.99, "...memory ran" -> " because" 0.60. With BOS:
  " patient" 0.69 (concept set mass 0.83), " out" 0.97 (results/lfm3_probe_nobos.json, lfm3_probe_bos.json).
  Any use of C1 on raw text (encoders, probes, concept training) must add BOS.
- LFM3 cache: attention key/values plus conv states (6-position window, LinearAttentionLayer without
  batch_repeat_interleave in transformers 5.22). One prefix forward + repeated cache + batched 1-token step equals
  per-candidate full forwards up to bf16 rounding: max |diff| <= 1.0 on norms ~150, cos >= 0.99984; near-tied
  candidates swapped order in 2 of 3 positions (harness/lfm3_cache_test.py).
- LFM3's own concept sets (smoke, 3 C4 sequences, 100 targets, 96 with synonyms): good (same -> similar, exact,
  identical, equivalent; conform -> adapt, comply, adhere; rods -> bars, rails, poles) next to wrong ones
  (traditional -> "tradicional"; counter -> "contre"; design -> "designation"; knife -> handle, material;
  conform -> "deform"). Like Falcon's: neighbours in the model's context space, not curated meaning.
- Frozen LFM3 S3, every hidden state x pooling on development splits (results/lfm3_layers_frozen.json; BOS added;
  STS-B val 1,500 pairs, PAWS val 8,000, SNLI val 3,000 triples): mean pooling at early layers carries the most
  similarity (layer 4 without BOS: STS 0.640, PAWS AUC 0.589, SNLI 0.686; layer 5 with BOS: 0.613 / 0.607 / 0.713);
  the last layer (24, what eval_mteb pools) 0.509 / 0.641 / 0.595; last-token pooling is worst (best 0.445).
  PAWS stays 0.59-0.64 at every layer: frozen pooling cannot tell same-words-different-meaning apart. That is
  what supervised distinction data (PAWS, NLI contradictions, CondaQA) has to teach the core.
- Learning Concepts REPRODUCED (Falcon3-1B-Base, C4 10,000 texts, embedding synonyms, released train.py/eval_mteb.py,
  3 seeds; MTEB STS mean of 9 tasks; results/lc_falcon_sts_table.json): pretrained 53.29 (paper 53.28); NTP-only
  lambda 0 53.90, range 53.88-53.91 (54.35); concept lambda 1 55.29, 55.25-55.35 (55.11); randomized lambda 0.25
  53.89 (54.04); randomized lambda 1.0 50.14 (50.61). Same ordering as the paper; concept minus NTP-only +1.39
  (paper +0.76); seed ranges <= 0.1. Randomized sets at lambda 1 fall below the pretrained model: the gain comes from
  what the sets contain. Deviations: transformers 4.57.6 overlay (released code targets 4.46-4.57),
  --max-content-words all (released default crashes for lambda > 0), seeds via harness/run_train_seed.py.
- Sub-Sentence Encoder REPRODUCED on PropSegmEnt atomic-fact retrieval (released ST5-base checkpoint, released eval,
  k3 CPU; 8,865 queries x 43,229 targets): P@1 40.83 / R@5 72.12 / R@10 79.30 / R@20 84.33 vs the paper's Table 1
  40.97 / 72.15 / 79.30 / 84.33; sentence level 43.37 / 81.43 vs 43.49 / 81.44; document level 45.80 / 89.19 vs
  45.93 / 89.19 (Table 4). The base encoder it starts from (sentence-t5-base, same eval, "mask pooling only" as the
  paper's Appendix C): P@1 26.03 / R@5 57.65 / R@10 69.00 vs the paper's 26.16 / 57.65 / 69.00. The sub-sentence
  training adds +14.8 P@1 over its own backbone: the paper's central claim, observed. Released-code defect:
  main() never passes --cuda to evaluate_propsegment (default
  cuda=True), so CPU runs crash after saving encodings; harness/sse_score.py scores the saved encodings with the
  released function and cuda=False.
- Learning Concepts concept sets, full Falcon train split (379,563 targets, 3.05M pairs; refined WordNet relations
  incl. adjective similar_to, antonyms, derivations; results/concept_sets_falcon_train*.json), embedding sets vs the
  randomized control: synonym 6.4% vs 0.3%; near-synonym 2.2% vs 0.1%; hypernym/hyponym 11.3% vs 1.6%; co-hyponym
  8.0% vs 2.3%; inflection 2.8%; antonym 0.61% vs 0.02% (about 18,600 antonym pairs are in the sets);
  no WordNet relation 66.9% vs 93.8%.
- Where the concept loss moves probability (results/concept_mass_falcon_seed42.json; Falcon test split, 45,750
  content-word positions, next-token probability on each set's members): base / lambda 0 / lambda 1 =
  original word 0.154 / 0.154 / 0.173; synonym-like members 0.019 / 0.019 / 0.025; antonym members 0.0009 / 0.0009 /
  0.0010; other related members 0.085 / 0.085 / 0.121; set total 0.259 / 0.260 / 0.319. The gain goes mostly to
  related-but-not-synonymous members (+0.036) and the word itself (+0.019); antonym mass stays negligible (an earlier
  progress note of mine said the loss "pushes probability onto opposites": wrong, the sets contain antonyms but the
  measured mass on them does not move).
- Learning Concepts adapters on meaning probes (results/probes_falcon_lc_seed42.json; seed 42, mean of final layer,
  4-bit as in the paper's eval): PAWS test AUC / SNLI test rank: base 0.637 / 0.669; lambda 0 0.630 / 0.700;
  lambda 1 0.633 / 0.705. The concept loss's STS gain does not carry over to same-words-different-meaning (PAWS
  flat) and adds +0.005 on SNLI over plain next-token fine-tuning: it teaches graded similarity, not distinction.
- Frozen baseline on the meaning-core dev sets (LFM3 S3 hidden state 4, mean pooling, no head;
  runs/core/baseline-l4): STS-B val 0.586; PAWS dev AUC 0.594; CondaQA dev AUC 0.551; NLI dev rank 0.729;
  role swaps ("Mary booted John" vs "John booted Mary") mean cos 0.983; antonym substitutions mean cos 0.975; the
  server sentences: contradiction 0.969 vs reworded paraphrase 0.935 (contradiction still closer).
- B4 reference encoders on TEST splits (harness/core_eval.py, results/b4_test.json; MTEB STS 9 tasks x100, PAWS
  test AUC, SNLI test rank, CondaQA test AUC over 1,696 original-vs-edit pairs where paraphrase edits should score
  above scope/negation edits, server sentences = cos to "The server stopped because memory ran out."):
  clover_hash 66.87 / 0.488 / 0.784 / 0.399 / paraphrase 0.415, contradiction 0.726;
  clover_indra_hash (the runtime today) 52.11 / 0.487 / 0.683 / 0.391 / 0.785, 0.944;
  bge_small 81.28 / 0.605 / 0.956 / 0.505 / 0.915, 0.931.
  CondaQA below 0.5 for Clover's encoders: they rank negation-scope and negation-undone edits (meaning changed, few
  words changed) above paraphrase edits (meaning kept, more words changed).
- Core pilots (dev): core-F (frozen LFM3 + head) at step 250: STS-B val 0.733 (frozen baseline 0.586), NLI 0.834
  (0.729), CondaQA 0.654 (0.551), role swaps mean cos 0.933 (0.983), antonym swaps 0.849 (0.975), server
  contradiction now lowest (0.913 vs paraphrases 0.945 / 0.964); PAWS 0.625 (0.594) barely moves, pair loss near
  chance: frozen states + pooling do not carry same-words-different-order meaning. core-L16 (LoRA rank 16 on all
  linears) learns the pair loss faster (6.13 at step 100 vs 6.64) and moves its layer mix to states 14-17.
- Pilots, final (dev): core-F at step 1000: STS-B 0.768, PAWS 0.694, CondaQA 0.681, NLI 0.883, role swaps cos
  0.797, antonym swaps 0.650, relations 0.468 (about 48 classes), WiC 0.621, server contradiction 0.662 vs
  paraphrases 0.843 / 0.945. core-L16 at step 250 (stopped there; checkpoint kept in runs/core/core-l16-pilot):
  STS-B 0.777, PAWS 0.778, CondaQA 0.761, NLI 0.898, role swaps 0.834, antonym swaps 0.666, relations 0.239,
  WiC 0.540, server 0.630 vs 0.853 / 0.962. LoRA after 250 steps beats the frozen reader after 1000 on
  distinction (PAWS +0.08) and negation (CondaQA +0.08).
- B3 decision (2026-11-01 20:29Z): full runs of both variants, batched trainer (one forward for all sentence texts of a
  step): core-l16-full (4,000 steps, eval every 500) as the quality candidate; core-f-full (6,000 steps) as the
  zero-cost runtime candidate (reads C1's own hidden states). Selection on dev only; B4 on test once.
- core-f-full finished (6,000 steps; dev at 6,000: STS-B 0.750, PAWS 0.854, CondaQA 0.745, NLI 0.898, relations
  0.624, WiC 0.655). **B4 on test, once (results/b4_test.json):** MTEB STS 63.87, PAWS test AUC 0.853, SNLI rank
  0.924, CondaQA test AUC 0.678, server sentences: paraphrases 0.763 / 0.906, contradiction 0.400. Against the
  references: far better than everything on distinction (PAWS 0.853 vs bge-small 0.605, Clover's runtime 0.487;
  CondaQA 0.678 vs 0.505 / 0.391) and the contradiction is now clearly the farthest; but graded similarity is below
  bge-small (STS 63.87 vs 81.28) and below Clover's plain hash (66.87). On the CSI minimal pairs it is below chance
  (AUC 0.37 unseen, 0.33 heldout2; research/semantic-interface Experiment 1/1b). Gap: the frozen-l4 B4 test entry
  I expected in b4_test.json is not there (the earlier run's outcome not yet checked); to rerun.
- B4 frozen LFM3 hidden state 4 (no head) on test, run 2026-11-02 (it had never been run; my earlier note that it was
  "started on card 5" was wrong): STS 54.99, PAWS 0.580, SNLI 0.729, CondaQA 0.485.
- **T3 DONE: Learning Concepts on LFM3-Omni-2B S3 (C1)** (released train/eval code via harness/lfm3_lc.py; C4 10,000
  texts; LFM3's own embedding synonyms; 3 seeds; MTEB STS mean of 9 tasks; results/lc_lfm3_sts_table.json):
  pretrained 44.17; NTP-only (lambda 0) 48.74, range 48.72-48.78; concept (lambda 1) 50.10, 50.07-50.14; randomized
  lambda 0.25 46.00, 45.96-46.06; randomized lambda 1.0 37.21, 37.14-37.24. Concept minus NTP-only = +1.36 (Falcon
  reproduction +1.39, paper +0.76): the method's effect carries over to C1, and randomized sets fall below NTP-only
  (at lambda 1 far below the pretrained model), so the gain comes from what the sets contain. (These are eval_mteb's
  pooled final-layer embeddings; not comparable to the B4 numbers, which pool hidden state 4 or use the core's head.)
- T3 incidents: (1) the STS phase packs up to 3 evals per card; STS22.v2 hit CUDA OOM for 3 adapter evals (seed 43
  lambda 1; randomized seed 44 both weights), so lc_table.py left those cells out. (2) **My defect:** I re-ran only
  STS22.v2 into the existing CSVs assuming the released eval merges rows ("Upserted"); it rewrote both files with only
  the new rows, removing 33 earlier rows. Restored by re-running the released eval for both adapter sets over all 9
  tasks on an empty card (its MTEB result cache returned the earlier task scores; seed-43 NTP-only 48.78 identical to
  before). Lesson: write partial reruns to a new file.
- **T4 DONE: Sub-Sentence Encoder, all released encoders scored** (k3 CPU, released eval + harness/sse_score.py):
  st5-large dim64 P@1 42.00 / R@5 70.16 / R@10 76.56 / R@20 81.63 (paper: 42.10 / 70.17); sentence level P@1 44.18,
  document level 46.33 (results/sse_st5_large_dim64_propsegment.txt). With the base-encoder and GTR results above,
  every released checkpoint reproduces the paper within 0.15 points.
