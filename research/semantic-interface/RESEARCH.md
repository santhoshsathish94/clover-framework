# CSI deliverables 2 and 4: research review, linguistic resources, data strategy

Labels: **[fact]** checked at a primary source or measured by us (with the location); **[hypothesis]** to be tested;
**[proposal]** a design choice. Nothing here is claimed as validated unless marked [fact].

## 1. What the research says about the central question

| Finding | Source | Consequence for CSI |
|---|---|---|
| [fact] Models trained on MNLI, including BERT, perform very poorly on HANS: they rely on lexical overlap, subsequence and constituent heuristics. | McCoy, Pavlick, Linzen, ACL 2019, arXiv 1902.01007 | High accuracy on a standard test set does not show understanding; CSI needs controlled challenge sets where shallow cues fail. |
| [fact] Transformers and LSTMs reach 96-99% in distribution on COGS but 16-35% on systematic gaps (new combinations of familiar words and structures), seed-sensitive (+/-6-8%). | Kim, Linzen, EMNLP 2020, arXiv 2010.05465 | Generalisation must be tested on held-out combinations, never on random splits only. |
| [fact] Clover's current representations are at chance on PAWS (0.488 / 0.487) and below chance on CondaQA (0.399 / 0.391); bge-small reaches 0.605 / 0.505. | research/meaning-core, results/b4_test.json | Similarity embeddings, even learned ones, do not separate "same words, different meaning". |
| [fact] Learning Concepts (reproduced: STS +1.39 over NTP-only on Falcon3-1B) improves graded similarity, not distinction: PAWS 0.630 -> 0.633, SNLI 0.700 -> 0.705. | research/meaning-core, lc_falcon_sts_table.json, probes_falcon_lc_seed42.json | Concept-level next-token objectives are not enough for CSI's distinctions. |
| [fact] The Sub-Sentence Encoder (reproduced: P@1 40.83 vs paper 40.97) adds +14.8 P@1 over its own backbone by contrasting propositions within a sentence. | research/meaning-core, sse_*_propsegment.txt | Proposition-level contrastive training is a working mechanism for composition-sensitive representations. |
| [fact] A meaning head trained on LFM3 hidden states with explicit distinction data moves PAWS dev AUC from 0.594 to 0.922 and pushes the contradiction of the server example below both paraphrases (0.496 vs 0.79 / 0.88) at step 1000 (dev splits; PAWS is in-domain training data). | research/meaning-core, runs/core/core-l16-full (in progress) | Distinction is learnable when it is in the training signal; whether it generalises to unseen constructions is open. |
| [hypothesis] A small system that combines a learned syntactic analysis with an explicit proposition structure (predicate, roles, polarity, tense, aspect, modality) distinguishes negation, tense, modality and role reversal more reliably than similarity embeddings, and treats active/passive as equivalent. | to test in E6/E7 | First experiment. |
| [hypothesis] Structural objectives (dependencies, roles, proposition contrast) added to small-model training improve compositional generalisation over next-token prediction at equal compute. | to test in E8 (nanochat-based) | Later experiment. |

## 2. Approaches (computational families) and what each can and cannot give CSI

| Family | Gives | Does not give | Role in CSI |
|---|---|---|---|
| Distributional semantics (embeddings) | graded similarity, robustness to wording | operators (negation, tense, modality), roles, ambiguity sets | one learned signal inside the Meaning Engine, never the whole interface |
| Dependency grammar (UD) | head-dependent structure, subject/object, morph features (Tense, Aspect, Mood, Voice, Polarity) | word senses, coreference across sentences | syntax layer; UD English-EWT for training and evaluation |
| Constituency grammar | phrase boundaries | (EWT has no gold constituency in UD format) | evaluation only where licensed data exists (open) |
| Frame / role semantics (PropBank, FrameNet) | predicate senses and numbered roles; voice-independent arguments | full sentence meaning | role layer; PropBank frames (CC BY-SA) |
| Formal semantics / meaning representations (AMR, UMR) | sentence graphs with negation and modality | AMR 3.0 is LDC research-only; English UMR is small (2,186 sentences in 1.0) | evaluation and design reference, not training |
| Lexical semantics (WordNet / OEWN) | senses, glosses, examples, relations | context selection (that must be learned) | sense inventory with provenance |
| Pragmatics | - | no licensed large resource found yet | open; out of the first experiments |

## 3. Linguistic resources

| Resource | Contains | Does not contain | Format | Quality | Licence (checked) | Use |
|---|---|---|---|---|---|---|
| Open English WordNet 2025 | 107,519 synsets (core), glosses, examples, sense relations (antonym, derivation, pertainym) and synset relations (hypernym, similar, meronym, entails, causes) | frequency of senses in context, syntax | LMF XML | curated, maintained fork of Princeton WordNet | CC BY 4.0 + Princeton WordNet licence | train (relations, word-sense pairs from examples) and lexical layer |
| UD English-EWT 2.18 | 16,622 sentences, 254,818 words; UPOS, XPOS, lemmas, features, basic + enhanced dependencies | semantic roles, senses (except STREUSLE supersenses on reviews) | CoNLL-U | dependencies manual (approx. 96% inter-annotator agreement on the double-annotated part); features mostly automatic with cleanup | CC BY-SA 4.0 (annotations); texts from LDC2012T13 sources | train/dev/test for morphology and syntax (its own splits) |
| PropBank frames | 7,568 frame files with rolesets and examples with ARG spans | the annotated corpus text (needs LDC OntoNotes) | XML | curated; some examples have shifted indices (we found 833 in research/meaning-core B2) | CC BY-SA 4.0 | role inventory; role-swap pairs |
| ConceptNet 5.7 | commonsense relations with per-edge licence | linguistic structure; quality varies by source | TSV + JSON | crowd + resources; noisy | CC BY-SA 4.0 | relation evaluation, cautious training |
| SNLI / MultiNLI | 570k / 433k sentence pairs with entailment labels | structure; HANS shows label shortcuts | JSONL | crowd-written; known artefacts | CC BY-SA 4.0 / OANC + CC BY(-SA) parts | train (equivalence/distinction), with challenge-set evaluation |
| PAWS-Wiki labeled_final | 65k pairs, same words, paraphrase or not | operators beyond reordering | TSV | high-overlap adversarial | "freely used for any purpose" | train / test (its own splits) |
| CondaQA | 14,182 QA pairs over negation edits (paraphrase / scope / undone) | general paraphrase | JSON | crowd + verification | Apache-2.0 | negation distinction (test split held out) |
| English UMR 1.0 | document-level graphs incl. modality, aspect | size | text graphs | expert | CC BY-SA 4.0 (English only) | evaluation reference |
| FrameNet 1.7 | 1,200+ frames, 200k annotations | - | XML | expert | **not confirmed** (no licence on site; download needs a token) | not used until confirmed |
| AMR 3.0 | 59,255 sentence graphs | - | text graphs | expert | LDC, research-only for non-members | not used |
| WiC | word-in-context pairs | - | TSV | expert | CC BY-NC 4.0 | not used (non-commercial); replaced by OEWN-derived pairs |
| Wikidata | entities and relations (structured part CC0) | language structure | JSON/RDF | community | CC0 (structured) | entity layer later, out of the first experiments |

Reconciliation [proposal]: the structure layer speaks UD (dependency relations, features); roles map to PropBank-style
ARG0/ARG1 through voice (UD `nsubj:pass`, `obl:agent`); senses are OEWN synset ids. Every field in the interface
records which resource or model produced it (provenance), so resources are linked, not merged.

## 4. nanochat (karpathy/nanochat), evaluated for CSI experiments

[fact] (README, checked 2026-11-02): MIT licence; a minimal PyTorch harness covering tokenizer training, pretraining,
SFT, RL, evaluation and inference; one complexity dial (`--depth`) sets width, heads, learning rate and horizon
compute-optimally; a 12-layer (GPT-1 size) run takes about 5 minutes on 8xH100, GPT-2 capability (depth 24-26) about
1.6-3 hours; runs on one GPU via gradient accumulation; GPUs under 80 GB need a smaller `--device-batch-size`.
No earlier nanochat exploration was found in this workspace or in the session history; this evaluation starts fresh.

Assessment:
- [fact] nanochat's objective is next-token prediction: by itself it is exactly the "predicting likely sentence
  continuations" that the central question contrasts with. It does not provide CSI.
- [proposal] Its value is as a controlled training harness: small models trained from scratch on controlled data
  where we choose the objective (next-token only vs next-token plus dependency, role and proposition-contrast heads)
  at equal compute, then tested on COGS-style held-out combinations and on our minimal pairs (E8).
- [estimate, not measured] On 8x RTX 3090 a depth-12 run should take tens of minutes rather than 5 (lower bf16
  throughput, 24 GB batches); feasible. To be measured before planning around it.
- Alternatives for the learned parts: a learned parser trained on UD (spaCy's `en_core_web_sm`, MIT, trained on
  OntoNotes: POS 97.3, LAS 89.9 self-reported; or our own parser trained on EWT); a small trained sentence encoder
  (research/meaning-core data); adapters on a connected LLM (research/meaning-core core-L/core-F, model-dependent).

## 5. Training and data strategy [proposal]
1. Syntax and morphology: learned from UD English-EWT (its own train/dev/test). First experiments use an existing
   learned parser as a measured component; replacing it with our own trained one is a later experiment.
2. Senses and lexical relations: OEWN inventory; the choice among senses in context is learned (word-sense pairs from
   OEWN examples, as built in research/meaning-core B2), never hand-written.
3. Proposition structure: predicate, roles, polarity, tense, aspect, mood, voice, modality come from the analysis; the
   mapping from UD to roles is a small, documented set of structural correspondences (voice), listed in
   ARCHITECTURE.md, measured, and replaced by learning if it fails.
4. Equivalence and distinction: trained on SNLI/MNLI, PAWS, CondaQA, PropBank role swaps and OEWN antonym
   substitutions (research/meaning-core B2 data), evaluated on held-out generated minimal pairs whose words and
   constructions are controlled.
5. Leakage: test splits never used for training or selection; generated challenge sets split by lexical item, so
   "unseen word" results are measured on words absent from any tuning.
