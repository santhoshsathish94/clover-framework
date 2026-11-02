# CSI deliverable 7: experiment definitions

Every experiment records: hypothesis, methodology, dataset, configuration, evaluation, results, failure cases,
conclusion. Results files live in results/. Status: E6+E7 defined and run first; the others are defined, not run.

## Experiment 1 = E6 + E7: semantic equivalence vs semantic distinction on controlled minimal pairs

**Hypothesis.** [hypothesis] An explicit, model-independent proposition analysis (learned parser + the written
correspondences in ARCHITECTURE.md section 3) judges equivalence (active/passive, dative alternation) and distinction
(negation, tense, aspect, modality, attempt/intention embedding, role reversal, predicate change) far more accurately
than cosine similarity of sentence embeddings, including Clover's current representations.

**Dataset.** Generated minimal pairs (`prototype/csi/minimal_pairs.py`, seed 1234). A base event = agent noun, verb,
patient noun (and recipient for dative verbs), drawn from fixed vocabularies. Each base sentence "The child opened the
door." is paired with:
- equivalent: passive ("The door was opened by the child."); dative alternation for give/send/hand/show
  ("The child gave the dog a ball." / "The child gave a ball to the dog.");
- not equivalent: negation ("did not open"), future ("will open"), progressive ("was opening"), modality
  ("might open"), attempt ("tried to open"), role reversal ("The door opened the child." made only from animate pairs:
  "The teacher thanked the student." / "The student thanked the teacher."), predicate change ("closed").
Vocabulary is split: a **seen** lexicon (used during development of the written correspondences) and an **unseen**
lexicon (other nouns and verbs, used only for the final run) to measure generalisation to new words. Correspondences
were written before generating the unseen set and are not changed after looking at its results (a change starts a new
run with a new held-out lexicon).

**Systems (identical inputs).**
- CSI v0.1 (`compare()`), model-independent.
- Embedding cosine baselines, thresholded at the value that maximises accuracy on the **seen** pairs (generous to
  them): Clover hash, Clover Indra-hash (runtime today), bge-small, and the research/meaning-core encoders (frozen
  LFM3 layer 4; trained cores when available). Threshold-free AUC is also reported.
- Later (Experiment A/B/C of the brief): an LLM without CSI vs the same LLM given CSI's analysis (E6/E7 as prompts).

**Metrics.** Accuracy and AUC per phenomenon type; balanced accuracy overall; separately for seen and unseen
lexicons; coverage (share of sentences CSI analyses without an `unsupported` entry); every error listed.

**Success / failure criteria (pre-registered).** CSI v0.1 is useful if balanced accuracy on the unseen lexicon is
at least 0.90 and at least 0.15 above the best embedding baseline, with no phenomenon type below 0.80. If it fails,
the failure cases decide the next step (parser error -> E3 with a better parser; correspondence gap -> extend the
list once, new held-out lexicon; or replace the written layer with a learned one).

**Known limits of this design.** Generated sentences are simple and grammatical; success here does not show
robustness to natural text (E8 and natural-text sets do). Written correspondences are a measured baseline, not the
goal.

**Results (run 2026-11-01 ~23:12 UTC, GPU server; `results/e1.json`, `results/e1.log`).** 531 pairs per lexicon,
23% equivalent. Baseline thresholds chosen on seen, applied unchanged to unseen.

| System | Seen bal. acc | Seen AUC | Unseen bal. acc | Unseen AUC |
|---|---|---|---|---|
| CSI v0.1 (coverage 1.0 on both) | 0.992 | - | **0.988** | - |
| bge-small (best baseline) | 0.724 | 0.779 | 0.766 | 0.815 |
| Clover Indra-hash (runtime today) | 0.713 | 0.748 | 0.697 | 0.746 |
| Clover hash | 0.717 | 0.792 | 0.689 | 0.774 |
| LFM3-Omni-2B S3 frozen, hidden state 4 | 0.560 | 0.291 | 0.520 | 0.272 |
| Trained meaning core core-f-full (frozen S3 + head, 6,000 steps; research/meaning-core B3) | 0.573 | 0.397 | 0.470 | 0.372 |

CSI per phenomenon: 1.0 on every distinction type and on passive / negated-passive equivalence, both lexicons;
dative equivalence 10/12 seen, 9/12 unseen (Wilson 95% interval for 9/12: 0.47-0.91; the smallest bucket).

**Against the pre-registered criteria.** Unseen 0.988 >= 0.90: met. Margin over bge-small 0.222 >= 0.15: met. No
phenomenon below 0.80: **not met** (dative equivalence 0.75 unseen). Experiment 1 does not pass as pre-registered.

**Failure analysis (all 5 errors).** All are dative alternations judged "not equivalent", and all are parser (L1)
errors, not missing correspondences: in "offered the soldier the parcel" the parser gave the theme no object relation;
in "sold the sailor the basket/parcel" it made the recipient the direct object; in two "showed" pairs the theme or the
"to" phrase was not attached to the verb. Baseline observation: the frozen LFM3 cosine has AUC 0.27 (below chance): it
ranks meaning-changing edits ("did not open") closer than meaning-preserving restructurings (passive, dative), i.e. it
tracks surface form. The trained meaning core (core-f-full, trained with PAWS, PropBank role swaps and CondaQA) is also
below chance (AUC 0.37 unseen): training on paraphrase data did not make its cosine separate these structural cases.

**Decision (pre-registered rule: parser error -> E3 with a better parser).** The written correspondences are not
changed. The E1 seen/unseen sets are now development data. Next: Experiment 2 (E3) measures parsers on gold data and
chooses one on EWT dev; Experiment 1b re-tests with it on a new lexicon.

## Experiment 1b (pre-registered 2026-11-01, not run)
Same design, systems and criteria as Experiment 1; new generator seed; a third lexicon of nouns and verbs that appear
in neither earlier lexicon; dative bucket enlarged to at least 60 pairs. Parser = the one chosen by Experiment 2 on EWT
dev (not chosen on E1 results). If that parser uses UD labels, the same correspondences are written for UD labels
before the run; nothing is changed after seeing the 1b results.
**Set built (`minimal_pairs.py`, lexicon "heldout2", seed 20261102):** 627 pairs - 11 transitive verbs, 7 animate-object
verbs, 10 ditransitive verbs (toss, mail, loan, rent, ship, kick, post, leave, owe, slide), new nouns; 60 dative
equivalence pairs. Seen/unseen sets verified unchanged (531 each, same distribution; k3 rerun reproduces the GPU-server
CSI numbers exactly). **Baselines run (thresholds from seen; `results/e1b_baselines.json`):** balanced accuracy / AUC on
heldout2 - Clover hash 0.707 / 0.769, Clover Indra-hash 0.707 / 0.737, bge-small 0.706 / 0.783, frozen LFM3 hidden
state 4 0.533 / 0.442, trained core core-f-full 0.448 / 0.327. Best baseline 0.707, so the pre-registered bar for CSI
is 0.90 (the 0.15 margin needs only 0.857). The CSI part waits for the Experiment 2 parser decision.

**Results (k3, 2026-11-02 01:22 UTC; `results/e1b_csi_*.json`).**

| Parser | Status | Balanced acc | Coverage | Dative equiv | Passive | Negated passive | Lowest other |
|---|---|---|---|---|---|---|---|
| ours seed 3 (UD rules) | **pre-registered** | **0.743** | 0.976 | 0.05 | 0.80 | 0.78 | 0.93 |
| en_core_web_trf | secondary | 0.991 | 1.000 | 1.00 | 0.94 | 1.00 | 1.00 |
| en_core_web_sm | secondary | 0.817 | 0.994 | 0.10 | 0.93 | 0.94 | 0.98 |

**Against the criteria: Experiment 1b fails as pre-registered** (0.743 < 0.90; dative, passive and negated passive
below 0.80). Errors of the primary run: 56 of 60 dative pairs mis-parsed (recipient taken as the object), lemmatiser
errors on unseen forms ("scrubbed" -> "scrubb"/"scrubbe") that break passive equivalence, past tense missing. The rule
that chose this parser (EWT dev, web text) did not transfer to this domain - the flaw I noted before the run.
Secondary findings (not pre-registered): with trf the unchanged v0.1 correspondences reach 0.991 on the new lexicon
(3 errors, all the lemma "forgave" not reduced to "forgive"), so the correspondences generalise to new words when the
parser is reliable; sm drops to 0.817 because it mis-parses 54 of 60 double objects with the new ditransitive verbs,
so its 0.988 in Experiment 1 depended on the verbs in that set. The trf result is tested cleanly in Experiment 1c.

## Experiment 1c (pre-registered 2026-11-02 01:25 UTC, before generating its data)
Same design, systems, metrics and criteria as Experiment 1. Parser: en_core_web_trf, chosen because of Experiment
1b's secondary result; correspondences unchanged (section 3). Data: a fourth lexicon "heldout3" (new seed 20261103;
nouns and verbs in none of seen/unseen/heldout2; 60 dative pairs); baselines thresholded on seen. Pass = balanced
accuracy >= 0.90, >= 0.15 above the best baseline, no phenomenon below 0.80.

**Results (k3 + GPU server, 2026-11-02 ~01:30 UTC; `results/e1c_csi_trf.json`, `results/e1c_baselines.json`).**
627 pairs, new words throughout. CSI v0.1 + trf: **balanced accuracy 0.994**, coverage 1.000, 2 errors (both dative
parses: "leased the tailor the hammer" theme not found; "bounced the pirate the key" recipient not found); lowest
phenomenon dative equivalence 0.967. Baselines (balanced accuracy / AUC): bge-small 0.746 / 0.771, Clover Indra-hash
0.746 / 0.762, Clover hash 0.703 / 0.792, frozen LFM3 0.514 / 0.418, trained core core-f-full 0.434 / 0.321.
**Experiment 1c passes all three pre-registered criteria** (0.994 >= 0.90; margin 0.248 over bge-small; minimum
phenomenon 0.967). **What this establishes (and what not):** the v0.1 written correspondences, given a reliable parser,
separate meaning-preserving restructurings from meaning-changing edits on new words, which no embedding baseline does.
It does not establish robustness to natural text (Experiment 1N: 0.60), lexical paraphrase, or any learned
correspondence - the correspondences here are still written, not learned.

## Experiment 1N: external natural-text check on PAWS (pre-registered 2026-11-01)
**Hypothesis.** CSI v0.1 compares only the main proposition (predicate, ARG0-2 heads, polarity, tense, aspect,
modality, embedding) and ignores modifiers and other clauses. On PAWS labeled_final test (8,000 Wikipedia sentence
pairs, word-scrambled non-paraphrases; Google, free use) we therefore expect (i) coverage well below 1.0 and (ii) many
non-paraphrases judged equivalent, with balanced accuracy near chance (expected <= 0.65). This measures the scope gap of
v0.1 on natural text; it is not a success test. **Metrics.** Coverage; balanced accuracy on covered pairs and with
unsupported counted as errors; confusion matrix; categories of 50 randomly sampled errors (seed 7).

**Results (`results/e1n_paws.json`, `e1n_paws_errors.txt`).** Coverage 0.915 (my "well below 1.0" was too
pessimistic); balanced accuracy 0.602 on covered pairs, 0.551 with unsupported counted as errors (expectation <= 0.65:
met). Non-paraphrases judged equivalent: 2,062 of 4,464 (46%); paraphrases judged different: 940 of 3,536 (27%).
Unsupported pairs: 499 have a side with no subject, 188 a side with no verbal root (8 both). **Error sample (50, seed 7), my categorisation:** 30 false
"equivalent" - every one differs only outside what v0.1 compares (swapped modifiers, prepositional phrases, coordinated
nouns, first names under the same surname head, a second clause); 20 false "not equivalent" - 7 lexical paraphrases
(show/depict, live/reside, publish/release, river/stream, were/included), 8 restructurings that change which clause is
"main" (participle vs finite clause, sentence split or merge, copular inversion, apposition), 1 modal near-synonym
(may/can), 1 "to" phrase taken as recipient, 3 whose gold label looks wrong to me (imitators/imitations;
retained/secured swapped; a fragment without a subject). **Conclusion.** On natural text v0.1 is near chance because
of scope, not parsing: it needs lexical knowledge (E1), comparison of all propositions and modifiers (E4), and
alignment that does not depend on which clause is main.

## Experiment 2 = E3 run plan (pre-registered 2026-11-01)
**Data.** UD English-EWT r2.18 (tag commit b7711cce, CC BY-SA 4.0): train for our parser, dev for model choice, test
reported once. Gold word segmentation is given to every system (tokenisation differences do not count).
**Systems.** spaCy en_core_web_sm / md / trf 3.8 (MIT; OntoNotes-trained, ClearNLP labels mapped to UD for the
measured relations) and a pipeline trained by us on EWT train (spaCy tagger + morphologizer + trainable lemmatizer +
parser, CPU efficiency config, seeds 1-3; learns POS, morphological features, lemmas and dependencies from data).
**Metrics.** UPOS and XPOS accuracy; Tense / VerbForm / Mood / Voice / Polarity accuracy where gold has the feature,
plus false positives; lemma accuracy; precision/recall/F1 of verb-headed nsubj, nsubj:pass, obj, iobj, obl:agent, aux,
aux:pass, xcomp and negator attachment; UAS/LAS for the UD-native pipeline only.
**Hypotheses.** (i) sm's iobj recall is low (consistent with the Experiment 1 errors); (ii) md/trf improve obj/iobj;
(iii) our EWT-trained pipeline reaches LAS >= 0.80 with gold words. **Decision rule.** The Experiment 1b parser is the
system with the highest macro-F1 over the CSI relations on EWT **dev**.

**Interim results: pretrained parsers (k3, `results/e3_pretrained.json`; EWT r2.18 has 12,544 / 2,001 / 2,077
sentences).**

| System | UPOS dev/test | CSI macro-F1 dev/test | nsubj test | obj test | nsubj:pass test | iobj test | xcomp test |
|---|---|---|---|---|---|---|---|
| en_core_web_sm | 0.925 / 0.926 | 0.827 / 0.808 | 0.911 | 0.881 | 0.859 | 0.580 | 0.620 |
| en_core_web_md | 0.928 / 0.927 | 0.828 / 0.809 | 0.908 | 0.875 | 0.853 | 0.577 | 0.638 |
| en_core_web_trf | 0.943 / 0.943 | 0.849 / 0.829 | 0.933 | 0.917 | 0.919 | 0.621 | 0.637 |

**Diagnostic of the low iobj F1 (`prototype/e3_iobj.py`, dev+test, 146 gold iobj under a verb).** The low score is
mostly a convention difference, not a parsing failure: 76 recipients sit in double-object clauses (the construction of
the Experiment 1 errors); sm labels 64 of them "dative" (84%), md 68 (89%), trf 73 (96%). The other 70 are recipients
without a nominal theme ("told me that ..."), which UD calls iobj and the ClearNLP scheme calls dobj (60-61 of 70 for
every model). So hypothesis (i) holds only in the narrow sense (sm misses 12 of 76 double-object recipients, trf 3). Likewise
ClearNLP "aux" includes infinitival "to" (UD: mark), lowering mapped aux precision for all spaCy models.
**Decision-rule correction (made before our own pipeline is scored; it does not change the order of the pretrained
models).** Macro-F1 over iobj and aux would favour UD-native systems by convention. The Experiment 1b parser is
therefore chosen by dev macro-F1 over nsubj, nsubj:pass, obj, obl:agent, aux:pass, xcomp and negator, with
double-object recipient and theme recall reported beside it. Pretrained values under this rule (dev): sm 0.863, md
0.861, trf 0.881. xcomp may also differ by convention (ClearNLP oprd/acomp for secondary predicates); not diagnosed.
Pretrained-only reading so far: trf is the best of the three and labels double-object recipients far better than sm.
**Development measurement (not a test; E1 seen/unseen are development data):** CSI v0.1 with en_core_web_trf instead
of sm scores 1.000 balanced accuracy on both E1 sets (0 errors; sm: 5 dative errors), k3 run
`results/e1dev_en_core_web_trf.json`. Consistent with the parser-error diagnosis; Experiment 1b is the test.

**Final results incl. our EWT-trained pipeline (k3; `results/e3_pretrained.json`, `results/e3_ours.json`,
`prototype/e3_decide.py`).** Corrected-rule macro-F1 (nsubj, nsubj:pass, obj, obl:agent, aux:pass, xcomp, negator):

| System | Dev | Test | UPOS test | Lemma test | UAS / LAS test | Double-object theme dev / test |
|---|---|---|---|---|---|---|
| en_core_web_sm | 0.863 | 0.842 | 0.926 | 0.971 | - | 36/42 / 30/34 |
| en_core_web_md | 0.861 | 0.843 | 0.927 | 0.971 | - | 36/42 / 30/34 |
| en_core_web_trf | 0.881 | 0.863 | 0.943 | 0.973 | - | 40/42 / 29/34 |
| ours seed 1 | 0.879 | 0.883 | 0.943 | 0.983 | 0.850 / 0.815 | 36/42 / 32/34 |
| ours seed 2 | 0.871 | 0.877 | 0.944 | 0.982 | 0.850 / 0.817 | 37/42 / 30/34 |
| ours seed 3 | **0.886** | 0.884 | 0.944 | 0.982 | 0.848 / 0.814 | 31/42 / 32/34 |

Our pipeline (a small CNN trained only on EWT train, CPU, about 50 minutes) learns UD morphology the spaCy models do not
output: Voice 0.85-0.87 and Mood 0.93-0.94 on test (spaCy: 0.0 and 0.49), Polarity 0.97-0.99. **Hypotheses:** (i)
holds only narrowly (diagnostic above); (ii) trf improves obj (test 0.917 vs 0.881), md does not; (iii) LAS >= 0.80
with gold words: met (test 0.814-0.817). **Decision (corrected rule, applied as written):** ours seed 3 (dev 0.886)
is the Experiment 1b parser. Its margin over trf (0.005) is inside our own seed range (0.871-0.886), so the choice is
not evidence that our pipeline is better than trf; it is the rule's outcome. Because it is UD-native, the same
correspondences were written for UD labels (ARCHITECTURE.md 3b) before any 1b run. **Secondary, not pre-registered:**
1b is also run with trf and labelled as such.
**Development check before the 1b run (E1 dev sets; `results/e1dev_ours_seed3.json`, `prototype/e1_errors.py`):**
CSI with ours seed 3 scores 0.953 seen / 0.851 unseen (coverage 0.917), far below sm (0.988) and trf (1.000). I read
all 74 errors: none is a defect of the UD correspondences; they are errors of our small parser on these simple
templated sentences - "covered" not tagged as a verb (no verbal root), "called" tagged without past tense, lemmas
"blamed" -> "blam" and "stole" -> "stole", by-phrases not labelled obl:agent, double objects mis-parsed. So the
decision rule (EWT dev, web text) picked a parser that does worse on this domain; I record that as a flaw of the
rule. As pre-registered, the choice is not changed after seeing E1 results: 1b runs with ours seed 3 as primary.

## Experiment 2 = E3: grammar and syntax on natural text (defined)
Hypothesis: the learned parser's tense/aspect/voice/negation/subject/object on UD English-EWT test (gold features and
dependencies) is accurate enough for L3 (UPOS, LAS, feature accuracy for Tense/Aspect/Mood/Voice/Polarity). Compare
spaCy sm vs a parser trained by us on EWT train. Failure analysis by construction type.

## Experiment 3 = E1: lexical understanding (defined)
Senses, definitions, synonyms/antonyms, word forms, POS from OEWN with held-out synsets; learned sense choice in
context vs most-frequent-sense baseline; relation prediction on held-out synset pairs.

**Experiment 3a = Dictionary layer v1: what a word means in context (pre-registered 2026-11-02 05:40 UTC, before any
3a code or data; `prototype/dict_meaning.py`).** The human: "The next challenge is teaching the dictionary layer to
understand what words mean, not merely whether they exist." Question: can a learned dictionary layer choose the
meaning (OEWN synset) a word has in a sentence, **including meanings it never saw used, by reading their
definitions**? That is the dictionary's analogue of the morphology's unseen-lemma test: memorised sense-context pairs
cannot answer it. **Inventory:** OEWN 2025 (CC BY 4.0) nouns, verbs, adjectives (head + satellite). **Data:** OEWN
usage examples, each belonging to a synset; the target is the first token matching a single-word member lemma of that
synset (or its listed/inflected form); only polysemous targets (>= 2 synsets for that lemma and part of speech).
Splits by seed 3101: test-Z = all examples of 10% of synsets (**zero-shot meanings**: never seen used; their glosses
never shown in training); dev-Z = another 5%; test-L = all examples whose target lemma is among 5% held-out lemmas
(**unseen words**); for remaining synsets with >= 2 examples, 10% of examples to test-S (**seen meanings, new
sentence**) and 5% to dev-S; the rest trains. Candidates at test = all synsets of the lemma and part of speech; in
training only training synsets. **Natural text (N):** SemCor 3.0 (NLTK package; Princeton WordNet licence, permissive),
sense keys mapped to OEWN ids, 20 Brown documents chosen by seed 3101, the same target filter; never used to train.
**Learner D:** a bi-encoder after Blevins & Zettlemoyer (2020): a context encoder gives the target word's vector, a
gloss encoder reads each candidate's definition, score = dot product, cross-entropy over candidates (up to 16 sampled
in training, all at test); both encoders start from bert-base-uncased (Apache-2.0); trained on OEWN examples only;
model selection on dev-S + dev-Z only; 3 seeds. **Baselines:** random; most-frequent sense (OEWN sense order); a
simplified Lesk (content-word overlap of the sentence with each candidate's gloss and its other examples) as the
hand-written rule; the same encoder untrained (target-word vector vs mean-pooled gloss, cosine); and a
**shuffled-gloss control** (D trained with definitions permuted among training synsets, tested with the true ones).
**Criteria, each for every seed, "beats" = the 95% bootstrap interval of the accuracy difference excludes 0:**
D1 test-S: D beats MFS, Lesk and the untrained encoder. **D2 test-Z (the understanding test): D beats MFS, Lesk and the
untrained encoder.** D3 test-L: the same. D4: the shuffled-gloss control on test-Z is within 0.05 of random (the
learner uses what definitions say). D5 natural text: D beats Lesk and the untrained encoder; MFS is reported but is
not a fair bar there, because WordNet's sense order was derived from SemCor counts. Seed range on D1/D2 reported.
**Decision:** D1-D4 passing validates Dictionary v1 (choice of meaning) on dictionary contexts and D5 says whether it
holds on running text; putting senses into CSI's analysis and comparison (e.g. "buy" and "purchase" sharing a
meaning) is a separate pre-registered experiment. If D2 fails, the learner has memorised uses, not definitions; that
is analysed before anything else. Relations between meanings (hypernymy, antonymy on held-out synsets) follow as 3b.

**3a results (GPU server; `results/e3a_*.json`, `e3a_report.json`; data sha256 items.jsonl 06d31a02..., synsets.json
9c221dfa...).** Data: 20,545 training examples; test-S 2,492; test-Z 3,056 (2,092 meanings never seen used); test-L
1,243 (512 words never seen); SemCor 12,766 tokens from 20 documents (852 SemCor senses had no OEWN id and were
dropped). Mean candidates per item 6.2-7.4.

| Accuracy | D (seeds 1 / 2 / 3) | shuffled glosses | random | MFS | Lesk | untrained encoder |
|---|---|---|---|---|---|---|
| test-S seen meanings, new sentence | 0.658 / 0.663 / 0.658 | 0.266 | 0.262 | 0.365 | 0.398 | 0.382 |
| **test-Z meanings never seen used** | **0.668 / 0.659 / 0.658** | 0.264 | 0.276 | 0.332 | 0.361 | 0.402 |
| test-L words never seen | 0.681 / 0.685 / 0.661 | 0.267 | 0.279 | 0.310 | 0.343 | 0.436 |
| SemCor running text | 0.502 / 0.497 / 0.488 | 0.238 | 0.229 | 0.662 | 0.593 | 0.365 |

**D1, D2, D3 and D4 pass; D5 fails.** For every seed D beats MFS, Lesk and the untrained encoder on test-S, test-Z
and test-L (smallest difference +0.22, every 95% interval above 0); the shuffled-gloss control is at chance on
test-Z (-0.011 from random); seed range 0.006 (S) and 0.010 (Z). On SemCor D beats the untrained encoder (+0.12 to
+0.14) but is below Lesk by 0.09-0.10 (intervals entirely below 0) and far below MFS. **Reading.** The dictionary
layer has learned to read definitions: it picks a meaning it has never seen used as accurately as one it has
(0.66 vs 0.66), and a meaning of a word it has never seen (0.67); with the definitions shuffled it falls to chance,
so it is the content of the definitions it uses. It has not learned how meanings are *used*: in running text a few
senses of each word dominate, OEWN gives roughly one example per meaning, and the learner has no notion of
frequency. Lesk wins there through its tie-break to WordNet's first sense (most SemCor tokens share no word with any
gloss), and MFS (0.662) is partly fitted to this test, since WordNet's sense order was derived from SemCor counts. By
the pre-registered decision, Dictionary v1 is validated for choosing meanings in dictionary-style contexts and is
not validated for running text.

## Experiment 4 = E2: morphology (defined)
Lemmatisation, inflection and derivation on UD features and OEWN derivation links, including unseen lemma+affix
combinations; learned vs lookup baselines.

**Experiment 4a: lemmatisation of unseen verbs (pre-registered 2026-11-02 01:40 UTC, before running;
`prototype/run_e2.py`).** Motivation: lemma errors broke CSI equivalence in Experiments 6a, 1b and 1c ("hoped" ->
"hop", "scrubbed" -> "scrubbe", "forgave" -> "forgave"). Data: UniMorph English (Wiktionary-derived, CC BY-SA 3.0,
github unimorph/eng 66e0e9e): verbs with past, past participle, gerund and 3rd-singular forms, placed in fixed frames
("The man {past} the box.", "The box was {participle} by the man.", "The man was {gerund} the box.", "The man {3sg}
the box."). Split by lemma: **seen** = occurs as a VERB lemma in EWT train; **unseen** = lemma and forms absent from
EWT train; 500 lemmas each (seed 7). Systems: spaCy sm and trf (rule + lookup lemmatiser, POS-driven) and our
EWT-trained edit-tree lemmatiser (seeds 1-3). Metric: exact lemma accuracy on the verb token, per split and form type;
rate of verb tagging. **Hypotheses:** (a) all systems >= 0.95 on seen lemmas; (b) on unseen lemmas our learned
lemmatiser (trained on about 200k words) drops more than sm/trf; (c) unseen errors concentrate on e-final stems and
doubled consonants.

**Results (k3; `results/e2a_lemmas.json`; 1,459 seen and 20,487 unseen UniMorph verbs available, 500 + 500 sampled,
4 forms each).**

| System | Seen | Unseen | Unseen past / participle / gerund / 3sg | Verb tagged (unseen) |
|---|---|---|---|---|
| en_core_web_sm | 0.975 | 0.742 | 0.686 / 0.680 / 0.688 / 0.912 | 0.990 |
| en_core_web_trf | 0.978 | 0.738 | 0.686 / 0.668 / 0.688 / 0.908 | 0.987 |
| ours seeds 1-3 | 0.769-0.774 | 0.598-0.637 | 0.55-0.61 / 0.56-0.60 / 0.42-0.43 / 0.85-0.92 | 0.94-0.96 |

(a) holds for sm/trf, **not for ours** (0.77 on seen lemmas in these frames, gerunds 0.62-0.65, although its EWT test
lemma accuracy is 0.98). (b) **Not supported as stated**: ours is lower everywhere, but sm/trf drop more (0.23 vs 0.15).
(c) Holds for ours, reversed for sm/trf: the rule lemmatisers, meeting a word missing from their lists, keep an "e"
("hanked" -> "hanke", "blebbed" -> "blebbe", "reamplified" -> "reamplifie"); ours drops it ("novated" -> "novat",
"jawboned" -> "jawbon"). **Conclusion:** none of the three has learned English inflection; the rule lemmatisers
know a word list, ours learned from 200k words of web text. Every past, participle and gerund of a verb outside the
known lists is about a one-in-three coin flip for both - a direct source of CSI predicate errors.

**Experiment 4b: learned lemmatisation from spelling (pre-registered 2026-11-02 01:45 UTC, before running;
`prototype/morph_learn.py`).** Can English verb inflection be *learned* well enough to generalise to unseen verbs?
Train on UniMorph verb forms of every lemma except the 1,000 Experiment 4a test lemmas (and the E1 lexicon verbs);
the model predicts an edit (strip k final letters, append s) from the form's last 1-6 letters (logistic regression;
edits seen only once dropped as unlearnable irregulars); no word list, no sentence context. Test: the same 4,000
items as 4a. **Hypothesis:** >= 0.95 on unseen lemmas, well above sm/trf (0.74) and ours (0.62); residual errors
are irregular verbs and true spelling ambiguity.

**Results (k3; `results/e2b_learned_lemmas.json`).** Seen 0.969, **unseen 0.966** (past 0.948, participle 0.950,
gerund 0.970, 3sg 0.994). Hypothesis met: +0.23 over the rule lemmatisers and +0.33 over our EWT lemmatiser on unseen
verbs. Residual unseen errors are genuine spelling ambiguities ("winged" -> wing vs winge, "crandalled" -> crandal vs
crandall, "schapped" -> schap vs schappe); the seen-lemma errors include irregular verbs, which a spelling model
cannot learn (one-off edits were dropped). **Conclusion (validated finding):** English verb inflection is learnable
from spelling well enough to generalise to unseen verbs, and doing so removes the largest lemma-error source seen in
CSI. A CSI v0.2 morphology layer = this learned model for unknown words + a list for irregulars; it changes CSI, so it
needs a fresh held-out test before any claim. Licence note: trained on CC BY-SA 3.0 data (share-alike applies if the
model is distributed).

**Experiment 4c = Morphology v2: is 4b a genuinely generalisable, learned capability? (pre-registered 2026-11-02
02:10 UTC, before any code for it existed; `prototype/morph_v2.py`, component `prototype/csi/morphology.py`).**
Direction (human): establish whether 96.6% is genuine; if it holds, freeze it as a validated CSI component. Data:
UniMorph English 66e0e9e (23,916 verb lemmas with past / participle / gerund / 3sg; 125,314 noun plurals; 2,008
adjectives with comparative and superlative). The component = an exception lexicon (forms whose spelling the learned
model gets wrong in training, i.e. memorised irregulars) + the learned edit model (form -> strip k letters, append s;
logistic regression over the last 1-6 letters). All splits are by lemma. Criteria for "genuine" (all must hold):
- **C1 stability:** 5-fold cross-validation over all verb lemmas: mean unseen-lemma accuracy >= 0.95, fold range <= 0.02.
- **C2 novel endings:** test lemmas whose last three letters occur in no training lemma: >= 0.90.
- **C3 nonce verbs (wug test):** invented monosyllabic verbs absent from UniMorph, 7 spelling classes (consonant
  doubling "blip -> blipped", clusters "plemp -> plemped", silent e "brone -> broned", consonant+y "glemmy ->
  glemmied", sibilants "glush -> glushes", -ck "blick -> blicked", vowel+w/y "blaw -> blawed"), gold from standard
  orthography for monosyllables: >= 0.90 overall and >= 0.80 in every class.
- **C4 real text:** verb tokens of UD EWT dev+test (VERB with VBD/VBN/VBG/VBZ, alphabetic; gold lemmas): the component
  trained on all UniMorph >= the trf lemmatiser overall, and >= 0.90 on lemmas absent from UniMorph (n reported).
- **C5 learned, not trivial:** accuracy rises with training data (1%, 3%, 10%, 30%, 100% of lemmas); a shuffled-label
  control stays within 0.05 of the majority-edit baseline; the learned model >= a hand-written orthographic rule set
  (strip -ed/-ing/-s with doubling, e-restoration and y rules, no lexicon) on the held-out folds.
- **C6 generation (the other direction):** lemma -> each form on held-out lemmas >= 0.90 (stress-dependent doubling
  in long verbs is a known ambiguity; errors categorised).
- **C7 other inflection classes:** nouns plural -> singular >= 0.95 and adjectives comparative/superlative -> base
  >= 0.90 on held-out lemmas (5-fold).
Also reported: accuracy on forms whose spelling is ambiguous in UniMorph (>1 lemma) vs unambiguous; irregular verbs
with and without the exception lexicon; suffix-length ablation (2, 3, 4, 6). **Decision:** if C1-C5 hold, the
component is frozen (trained on all UniMorph; classes that pass C7 included) and must then pass an integration test
inside CSI on fresh minimal pairs with nonce verbs (pre-registered separately before its data is generated). If any of
C1-C5 fails, it is not frozen; the failure is analysed first.

**4c results (k3; `results/e4c_morph_v2.json`; C3/C4 below when finished).** Incident: the first nonce generator could
loop forever when a spelling class had fewer than 100 unused words (it hung that task); fixed with a wider onset list
and an attempt cap before any nonce result existed; all other tasks had finished and were kept.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| 4b replication (same split) | unseen 0.9665, seen 0.966 (4b: 0.9655 / 0.969) | - | replicated |
| C1 5-fold, model / component | 0.9585 (range 0.0026) / 0.9548 (0.0038) | >= 0.95, range <= 0.02 | **pass** |
| C2 novel three-letter endings | **0.622** (past 0.52, participle 0.53, gerund 0.54, 3sg 0.90) | >= 0.90 | **fail** |
| C5 learning curve 1/3/10/30/100% | 0.845 / 0.895 / 0.930 / 0.947 / 0.958 | rises | pass |
| C5 shuffled labels vs majority edit | 0.457 vs 0.505 | within 0.05 | pass |
| C5 learned vs hand-written rules | 0.958 vs 0.809 | learned >= rules | pass |
| C6 generation 5-fold | 0.976 | >= 0.90 | pass |
| C7 nouns plural -> singular | 0.972 (unambiguous 0.992) | >= 0.95 | pass |
| C7 adjectives | 0.896 (component 0.882) | >= 0.90 | fail (by 0.004) |
| C3 nonce verbs, 7 classes x 100 x 4 forms | analysis 0.973 (lowest class 0.93), generation 0.989 | >= 0.90, every class >= 0.80 | pass |
| C4 real text, EWT dev+test verb tokens (n = 2,580) | component **0.908** vs trf 0.967, sm 0.958, ours 0.907 | >= trf | **fail** |
| C4, lemmas absent from UniMorph (n = 416) | **0.740** | >= 0.90 | **fail** |

**Why C4 fails (`results/e4c_morph_v2.json` component errors; UniMorph checked directly):** UniMorph English, which
the component learns from, is incomplete and partly archaic for common verbs. "appear", "confirm", "fear" and "find"
are listed with their base form only; "be" is absent; "appeared" is listed as the past of an archaic lemma "appeare",
"added" also under "adde", and "fix" has only the past "fixt". The component learned those entries faithfully
("appeared" -> "appeare", "found" -> "foind", "fixed" -> "fixe"). The spaCy lemmatisers win on real text because
running text is dominated by frequent words, which their curated tables cover; the learned model wins on rare and
unseen words (Experiment 4a/4b).

Within C1: ambiguous spellings (1.3% of forms, e.g. a form shared by two lemmas) 0.35 model / 0.07 component - the
exception lexicon hurts exactly there; irregular verbs (116 forms) 0.44 (an unseen irregular cannot be learned).
Suffix ablation: last 2 / 3 / 4 / 6 letters -> 0.66 / 0.76 / 0.89 / 0.96. Noun "ambiguous" items are mostly a
UniMorph artefact (the strings "countable"/"uncountable" listed as plural forms). **Post-hoc radius analysis
(`results/e4c_radius.json`)**, holding out whole ending families of k letters: k = 2: 0.310, k = 3: 0.622, k = 4: 0.843,
k = 5: 0.939 (random lemmas: 0.958); errors are dominated by silent e ("abraded" -> "abrad", "abused" -> "abus"), then
y/ie and consonant doubling. **4e (character transformer, same C2 split): 0.758** (past 0.68, participle 0.69, gerund
0.74, 3sg 0.92) - better than the suffix model (+0.136) but below the hypothesised 0.80.
**Reading so far.** The 96.6% is a genuine, learned capability *within the ending families English verbs use*: stable
across 24k held-out lemmas, it generates as well as it analyses, extends to nouns, rises with data and beats
hand-written rules by 0.15. It does **not** transfer to an entire unseen ending family (0.62; a character model 0.76):
it has learned English endings, not abstract spelling rules that would recover a silent e from letters alone. By the
pre-registered rule the component is **not frozen** (C2, C4 and the adjective part of C7 fail).

**Experiment 4d: integration of the morphology component into CSI (pre-registered 2026-11-02 02:35 UTC, before its
data was generated; runs only if 4c passes).** Minimal pairs with the Experiment 1 design built on **nonce verbs**
(lexicon "nonce1": 14 transitive + 7 animate-object invented verbs, 2 resp. 1 per spelling class of 4c, generated with a
new seed and excluded from UniMorph and from the 4c nonce set; new nouns; no dative pairs, so parser behaviour on
unknown ditransitives does not enter). Systems: CSI v0.1 + trf with spaCy's lemmas (as in 1c) vs the same with the
frozen component supplying lemmas for inflected verbs (VBD/VBN/VBG/VBZ), plural nouns and comparative/superlative
adjectives. **Pass:** with the component, zero predicate mismatches between inflected forms of the same nonce verb,
and balanced accuracy >= 0.95 on covered pairs (coverage reported; trf may not parse every nonce verb as a verb, which
is a parser limit, not a morphology result). The spaCy-lemma run is reported beside it.

**4d amended (2026-11-02 05:20 UTC, after the human approved the scoped freeze and before the nonce1 data existed).**
The component is the **frozen layered verb lemmatiser** (Morphology artefact `morph-verb-v1` + OEWN 2025 + the
three-step composition), applied only inside its scope: VERB tokens tagged VBD/VBN/VBG/VBZ; nouns and adjectives keep
spaCy's lemmas (not frozen). nonce1 verbs come from the 4c spelling classes with seed 20261104 and are excluded from
UniMorph, from OEWN lemmas and forms, and from the 4c/4f/4g nonce sets (seeds 19, 107, 113); the 20 nouns are real
words absent from the earlier lexicons, chosen by hand. **Predicate mismatch** = a sentence whose CSI predicate is an
inflected form of a nonce1 verb but whose predicate lemma is not that verb's base form. Criteria unchanged.
Deviation found while generating (before any 4d run): after those exclusions the -ck and vowel+w/y classes had only 1
and 2 unused invented verbs left (their spelling space is small), so nonce1 has 13 transitive + 5 animate-object
verbs instead of 14 + 7 (`morph-verb-v1.card.json` lists them).

**Freeze (2026-11-02, human-approved scope).** `morph-verb-v1`: the 4c verb analyser trained on all 95,053 UniMorph
verb pairs (116 edit classes, 24,882 suffix features; `morphology.json` sha256 21b22e4e..., `morphology.npz` 88004090...,
4.8 MB), composed with OEWN 2025 (sha256 9ca6d1dc...). Reloaded from disk it reproduces 4g exactly (GUM verbs 0.9878,
0 of 3,439 predictions differ; nonce seed 113 0.9557). Card: `results/morph-verb-v1.card.json`.

**4d results (k3, trf; `results/e4d_*.json`). 4d fails as pre-registered; the frozen lemmatiser does not enter CSI.**

| Lemmas | Balanced accuracy (coverage 1.0) | equiv:negated_passive | Predicate mismatches (of 275 sentences) |
|---|---|---|---|
| parser (trf) | 0.741 | 0.093 | 241, all 16 verbs affected |
| frozen lemmatiser | **0.926** (< 0.95) | 0.833 | **26** (> 0), 2 verbs |

The 26 mismatches are two verbs. "cribed" -> "crib" (14): the composition preferred a real word, although "crib"
would inflect as "cribbed" - the known cost seen in 4g, now inside CSI. "sproched" -> "sproche" (12): the Morphology's
own silent-e error after "ch" (the weakest 4c nonce class). All 16 remaining CSI errors are equivalence pairs: 6
involve those two verbs' mismatches, and 11 involve a problem outside the frozen scope that occurs in both runs (one
error has both): the parser's lemmatiser gives the noun "glazier" the lemma "glazi" in active sentences, so the roles
differ. Without the noun errors the run would score 0.977; without the verb mismatches 0.954; the mismatch criterion
needs the verbs fixed in any case. **Analysis.** (1) A dictionary word should be accepted as the lemma only if the Morphology,
generating the observed form type from it, reproduces the observed form (analysis by synthesis: "crib" -> "cribbed",
not "cribed"); this removes the error class that cost 2.2 points on nonce verbs in 4g. (2) An unseen-word lemma is
uncertain by nature ("sproch"/"sproche" both inflect as "sproched"); CSI should carry the ranked alternatives rather
than one lemma. (3) Nouns need the layered lemmatiser as well. Each is a change to a frozen component, so each needs
its own pre-registered re-validation on data not yet examined.

**Experiment 4h = verb lemmatiser v1.1: analysis by synthesis (pre-registered 2026-11-02 06:10 UTC, after the 4d
analysis, before any 4h code or data).** One change to the composition, nothing else: in step 2 a candidate the
Dictionary knows is accepted only if the Morphology, generating the observed form type (from the tag) from that
candidate, reproduces the observed form; the generators are the 4c verb generators trained on all UniMorph verbs and
used without their exception lexicon (UniMorph's archaic forms such as "fixt" must not veto "fixed"); listed forms
(step 1) are unchanged. Artefact `morph-verb-v1.1` = the v1 analyser unchanged + those generators. Data never
examined: **H1** real text = UD EWT *train* verb tokens (never used to evaluate lemmatisation; same selection as C4):
v1.1 >= trf and v1.1 >= v1 - 0.002. **H2** nonce verbs, seed 127 (excluded from UniMorph, OEWN and every earlier nonce
set): v1.1 >= 0.95 overall and >= Morphology alone - 0.005 (the 4g cost removed). **H3** integration: the 4d design on
a new lexicon nonce2 (seed 20261105, same exclusions plus nonce1; only the five spelling classes that still have
unused invented verbs - the -ck and vowel+w/y spaces are exhausted - 2 transitive + 1 animate-object verb each), with
the heldout3 nouns so that the test isolates the verb lemmatiser (trf lemmatised those nouns correctly in 1c): zero
predicate mismatches and balanced accuracy >= 0.95 with v1.1; v1 and parser lemmas reported beside it. **Decision:**
H1-H3 passing lets v1.1 enter CSI for inflected verbs; otherwise the failure is analysed.

**4h results (k3; `results/e4h_v11.json`, `e4h_nonce2_*.json`, `e4h_check_*.json`). 4h passes all three criteria;
v1.1 is the CSI verb lemmatiser from now on.** Artefact `morph-verb-v1.1` (14 MB; `morphology.json` sha256
3ac4000b..., `morphology.npz` 9e50db6f...; kept on k3 with v1, not in the repository: size, and the weights derive
from UniMorph, CC BY-SA 3.0).

| Test | v1.1 | v1 | Morphology alone | trf | Criterion |
|---|---|---|---|---|---|
| H1 EWT train verbs (n = 11,714) | **0.9925** | 0.9939 | 0.913 | 0.975 | >= trf and >= v1 - 0.002: **pass** (margin 0.0006) |
| H2 nonce seed 127 (n = 2,000; 5 classes) | **0.972** | 0.942 | 0.972 | - | >= 0.95 and >= Morphology alone - 0.005: **pass** |
| H3 nonce2 minimal pairs (420; trf) | **1.000, 0 mismatches** | 1.000, 3 mismatches | - | parser lemmas 0.800, 183 mismatches | 0 mismatches, >= 0.95: **pass** |

What the synthesis check does: it removes the known-word errors on invented verbs completely (H2: the 4g/4d cost of
3 points is gone; H3: v1's "bexing" -> "be" is rejected because "be" would give "being"). What it costs: on real
text it rejects 19 correct dictionary lemmas where the generator mis-predicts stress-dependent doubling ("vomited",
"orbiting", "totaling", "libeled" -> "vomitted" etc.) or where the treebank tags a base form as inflected ("follow",
"save" tagged VBN), and fixes 2 - a net loss of 0.14 points that the pre-registered margin allows by 0.0006. Not
covered: nouns and adjectives (not in scope), the -ck and vowel+w/y classes in H2/H3 (their invented-word space is
used up), and H3's nouns were chosen to isolate verbs, so the parser's noun-lemma error of 4d ("glazier") remains.

**Experiment 4e: is the novel-ending failure a property of the learner or of the task? (pre-registered 2026-11-02
02:50 UTC, after the first 4c results showed C2 = 0.62 and before running 4e; `prototype/char_seq2seq.py`).** A
character-level transformer encoder-decoder trained from scratch (no pretraining, so no memorised words; about 3M
parameters) maps the inflected form to its lemma letter by letter. It is trained and tested on exactly the 4c splits:
C1 fold 0 (lemmas held out), C2 (whole three-letter ending families held out), and the C3 nonce verbs (trained on all
verbs). **Hypothesis:** it matches the suffix model on C1 (about 0.96) and does clearly better on novel endings
(>= 0.80), because it can represent letter patterns (consonant-vowel-consonant doubling, silent e) without the
specific ending string. If it also stays near 0.6, the failure belongs to the task: inflection of an unseen ending
family is not predictable from spelling alone.

**Experiment 4f = Morphology v2b (pre-registered 2026-11-02 03:05 UTC, after the 4c C2 failure and the first 4e
result, before any 4f run).** Question: can a learner with an abstract representation generalise across ending
families while keeping in-family accuracy? Learners: (S) the 4c suffix model; (SS) suffix + consonant/vowel shape
features (letter classes of the word end and of the stem left after a candidate suffix, plus that stem's last letter
and whether it ends in a doubled letter; nothing names an ending family); (T) the 4e character transformer. **Fresh
splits never used before:** fold seed 101, ending-family seed 103 (families of 3 and of 4 letters), nonce seed 107;
the 4c seeds (11/13/19) are now development data. **Measures:** the full 4c battery for S and SS, and C1/C2/nonce for
T. **Improvement criterion:** SS or T reaches >= 0.80 on the fresh three-letter families while keeping fresh C1 >=
0.95. The 4c freeze rule is not changed by this experiment.

**Experiment 4g = Morphology v3: does real-text evidence close the C4 gap? (pre-registered 2026-11-02 04:05 UTC,
after the 4c C4 failure and its diagnosis, before any 4g code or data existed.)** The 4c error list on EWT dev+test
was examined to diagnose C4, so **EWT dev+test is development data from now on** and the test uses real text never
examined: **UD_English-GUM r2.18 (dev+test) and UD_English-PUD r2.18 (test)**, the same token selection as C4 (VERB
with VBD/VBN/VBG/VBZ, alphabetic, gold lemmas). **v3 = the 4c component (suffix features, unchanged) plus two
learned-from-data changes, nothing hand-written:** (a) the verb pairs of UD EWT *train* (form, gold lemma, form type
from the tag) are added to the UniMorph training pairs; (b) a form with more than one known lemma is resolved by
lemma frequency in EWT train (the model decides when no candidate is attested). **G1 real text:** on the fresh
tokens (pooled; each treebank reported) v3 >= the trf lemmatiser, and v3 > the 4c component. **G2 the learned
generalisation is kept:** fresh 5-fold lemma holdout (fold seed 109; held-out lemmas removed from both UniMorph and EWT
pairs, and from the frequency prior) >= 0.95, and fresh nonce verbs (seed 113, also excluded from EWT train) >= 0.90
overall and >= 0.80 in every class. Also reported: EWT dev+test (development), the sm lemmatiser, the error list.
**Decision:** G1 and G2 passing makes v3 the candidate for a *scoped* freeze (validated inside English ending
families, with unseen ending families as the documented boundary); because C2 of the 4c rule is still unmet, that
freeze is the human's decision, and 4d must pass before the component enters CSI. Robustness check beside 4e/4f (not
a criterion): the character transformer on C2 with training seeds 2 and 3, on the 4c split and on the fresh split.

**4g amended (2026-11-02 04:25 UTC, after the human's layer model and before any 4g code or fresh data existed).** The
human's model: "Dictionary teaches what a word means. Morphology teaches how a word is constructed and changes. Grammar
teaches how words function together. Semantics teaches what the resulting expression means." The C4 errors are
dictionary facts (which words exist, which lemma is current English, which forms are irregular), so (a) and (b) above
would have put dictionary knowledge inside the morphology model. **Replaced by a layered lemmatiser, each layer doing
its own job:** *Grammar* gives the word's class and form (the tag; gold tags as in C4, so this tests lemmatisation
given the class); *Morphology* is the 4c learned edit model, unchanged, without its exception lexicon (memorised
words are a dictionary function), returning its 5 best-ranked candidate lemmas; the *Dictionary* is Open English
WordNet 2025 (CC BY 4.0; lemma inventory per part of speech and its 4,473 listed irregular or spelling-changed forms).
Composition (written, three steps): a form the Dictionary lists -> its lemma (ties broken by the Morphology ranking);
otherwise the best-ranked candidate the Dictionary knows for that class; otherwise the top candidate, marked unknown.
**G1** as above on the fresh GUM/PUD verb tokens: layered >= trf, and layered > the 4c component. **G2** replaced:
the Morphology layer is unchanged, so its unseen-word results (C1/C3/C6/C7) stand; what must be shown is that the
composition does not hurt unknown words: fresh nonce verbs (seed 113, excluded from UniMorph and from OEWN lemmas and
forms) >= 0.90 overall and >= 0.80 in every class with the layered lemmatiser. Errors on the fresh verbs are
attributed to a layer: gold lemma unknown to the Dictionary; gold known but not among the Morphology candidates; gold
among them but another known candidate ranked higher. Secondary (reported, no criterion): plural nouns (NNS) and
comparative/superlative adjectives (JJR/JJS) on the same treebanks; EWT dev+test as development data.
Data note (04:35 UTC, after cloning, before any 4g run): GUM r2.18 (b58e74b; CC BY-NC-SA 4.0, evaluation only) has
manual lemmas; PUD r2.18 (e173a1b; CC BY-SA 3.0) has *automatic* lemmas, so **G1 is decided on GUM dev+test** (3,476
verb tokens); PUD (1,596) and the pooled numbers are reported beside it.

**4f results (fresh splits: folds 101, endings 103, nonce 107; `results/e4f_suffix.json`, `e4f_suffix_shape.json`,
`e4f_char_*.json`; seed replicates `e4g_charseed_*.json`).**

| Measure | Suffix (S) | Suffix + shape (SS) | Character transformer (T) |
|---|---|---|---|
| C1 lemma holdout, 5 folds | 0.9585 (range 0.007) | 0.9601 (range 0.0065) | 0.9449 (fold 0) |
| C2 unseen three-letter families | 0.556 | 0.637 | 0.755 (training seeds 2/3: 0.749 / 0.759) |
| C2 unseen four-letter families | 0.854 | 0.869 | - |
| C3 nonce verbs (lowest class) | 0.963 (0.905) | 0.969 (0.935) | 0.970 (0.868) |
| C5 curve 1% -> 100%; shuffled vs majority; rules | 0.866 -> 0.957; 0.462 vs 0.510; 0.813 | 0.894 -> 0.958; 0.463 vs 0.510; 0.813 | - |
| C6 generation | 0.977 | 0.978 | - |
| C7 nouns / adjectives | 0.972 / 0.896 | 0.972 / 0.904 | - |
| C4 EWT dev+test (development) | 0.911 | 0.925 | - |

**Improvement criterion not met:** on fresh three-letter families SS reaches 0.637 and T 0.755 (and T's C1 is 0.945).
T on the 4c split with training seeds 2 and 3: 0.735 and 0.728 (first run 0.758), so over six runs it stays between
0.73 and 0.76. Everything else replicates on fresh splits: in-family accuracy 0.96, nonce 0.96-0.97, generation 0.98,
nouns 0.97, learning curve, controls. The unseen-family boundary is a property of the task as these learners see it,
not of the suffix features: a model that sees every letter, and one given letter-class shapes, still cannot recover
how an ending family it has never seen inflects. One unexplained difference: the 4c and 4f suffix runs train the C4
component on identical data yet give 0.908 and 0.911 (9 of 2,580 tokens); not investigated (the two runs used
different solver thread counts).

**4g results (k3; `results/e4g_layered.json`).** Verbs, lemma accuracy given the class:

| Set | n | Layered | Morphology alone | 4c component | trf | sm |
|---|---|---|---|---|---|---|
| **GUM dev+test (fresh, manual lemmas; decides G1)** | 3,439 | **0.988** | 0.931 | 0.928 | 0.960 | 0.949 |
| PUD test (fresh, automatic lemmas) | 1,579 | 0.999 | 0.932 | 0.929 | 0.979 | 0.971 |
| EWT dev+test (development) | 2,580 | 0.994 | 0.914 | 0.908 | 0.967 | 0.958 |

**G1 passes** (0.988 >= trf 0.960; > the 4c component 0.928). The Dictionary knows 99.5% of the GUM gold verb lemmas.
Of the 42 GUM errors, 34 are "another known candidate chosen", about 30 of them one token type: "gon" (GUM splits
"gonna" into "gon" + "na"), which the Morphology ranks as "gin" (the won/win pattern) and the Dictionary knows as a
verb; 4 have a gold lemma OEWN lacks (e.g. "pend"; also gold-side typos such as "embd", "flead"); 4 are not among the
Morphology candidates (misspellings, "recognises" with the American lemma "recognize"). PUD's 0.999 is likely inflated
by its automatic lemmas. **G2 passes:** fresh nonce verbs (seed 113) layered 0.956 (every class >= 0.88; silent e
lowest), but the composition **costs 2.2 points** against Morphology alone (0.978): preferring a known word turns
"spinded" into "spin" and "skibs" into "ski". Secondary classes: plural nouns layered 0.944 / 0.918 / 0.900 vs trf
0.941 / 0.948 / 0.943 (GUM / PUD / EWT) - at trf level on GUM, below it on the other two; comparative/superlative
adjectives 0.66 / 0.54 / 0.75 vs trf 0.90 / 0.96 / 0.90 - poor, because real JJR/JJS tokens are dominated by words
like "more", "most", "less" whose treebank lemma is the word itself, a case absent from UniMorph's adjective data.
**Reading.** The C4 failure belonged to the dictionary, not to morphology: with each layer doing its own job, verbs
on fresh real text go from 0.928 to 0.988, above the trf lemmatiser. By the amended 4g rule the layered verb
lemmatiser is the candidate for a scoped freeze; the decision is the human's (C2 still unmet), and 4d must pass
first.

## Experiment 5 = E5: context and ambiguity (defined)
Polysemy ("The bank is closed." with and without disambiguating context), structural ambiguity ("I saw her duck."),
pronoun reference (Winograd-style pairs). Success = alternative readings returned when evidence is insufficient and
the right one ranked first when it is sufficient; calibration measured.

## Experiment 6 = E4: compositional semantics (defined)
Meaning built from learned parts on COGS-style held-out combinations (new verb in a known structure, known verb in a
new structure).

**Experiment 6a: external role check on COGS (pre-registered 2026-11-01 UTC, before running; `prototype/run_cogs.py`).**
COGS (Kim & Linzen, EMNLP 2020; MIT; github najoungkim/COGS main 165a7b6) gives gold logical forms with
agent/theme/recipient for every event. CSI is not trained on COGS, so this is not a compositional-learning test; it
checks whether CSI's main proposition matches the gold roles of the same event across COGS structures (test split =
in-distribution structures, 3,000; gen split = 21 generalisation cases, 21,000). Mapping: agent = ARG0, theme = ARG1,
recipient = ARG2; CSI's embedded proposition (rule 7) is compared with the gold xcomp event. Metrics: coverage,
predicate match, exact match of the role set, per-role precision/recall, per gen case; error patterns; 40 sampled
errors (seed 7). **Expectations.** Transitive active/passive and dative sentences mostly right (parser errors on
double objects possible, Experiment 2); **unaccusative subjects wrong by design** ("The cake burned": COGS theme, CSI
v0.1 maps every active subject to ARG0 because it has no verb-class knowledge), so cases built on unaccusatives near
0; overall test exact match between 0.6 and 0.85.

**Results (`results/e6a_cogs.json`; spaCy sm parser).** Test: coverage 0.979, predicate 0.965, exact 0.786 (expected
range met); role P/R ARG0 0.946/0.957, ARG1 0.825/0.791, ARG2 1.000/0.585. Gen: coverage 0.978, predicate 0.939,
exact 0.835. By case: active->passive 0.988, passive->active 0.975, unaccusative->transitive 0.976, object-omitted
->transitive 0.971, pp_recursion 0.893, cp_recursion 0.862; lowest: **pp_dative->double object 0.353**,
obj_pp_to_subj_pp 0.682, prim_to_inf_arg 0.763, prim_to_obj_proper 0.764, subj_to_obj_proper 0.773.
**Failure analysis (`prototype/cogs_examples.py`).** (1) Missing recipient, the largest error (1,750 of 21,000 gen
sentences), mostly double objects: "Grace shipped Emma the cake" -> recipient taken as the object, theme dropped (parser, L1; the same error
as Experiment 1, now at scale, worst with proper names). (2) Recipient passives "The girl was given a box": rule 2
makes every passive subject ARG1 - a **missing correspondence** (L3), not a parser error. (3) Lemmatiser errors:
"hoped" -> "hop", "hated" -> "hat" (L1), the main cause of cp_recursion predicate errors. (4) Rare verbs:
"A prisoner attempted to crawl" -> no verbal root (parser). (5) Unaccusative subjects ("A donut on a table grew":
gold theme, CSI ARG0) as predicted, but only 86 gen sentences: my expectation that the "unacc" cases would be near 0
was **wrong** - those cases mostly put the unaccusative in an embedded clause, which v0.1 does not compare. Because
these failures have now been read, COGS is development data for any later change; such changes will be reported as
"informed by COGS gen errors".

**Experiment 6b: learned linking vs sequence learning on COGS (pre-registered 2026-11-02 00:10 UTC, before any
code was run).** Question: does a system that *learns* the syntax-to-meaning linking over explicit structure
generalise to new combinations better than a pretrained LM that learns the mapping as a sequence? All systems learn
from COGS train only (24,155 lines), choose checkpoints/settings on COGS dev (in-distribution), and are scored once on
test and gen. **S1, sequence learner:** LFM3-Omni-2B S3 (Clover's C1) + LoRA rank 16 on all linear layers, trained
to write COGS's logical form after the sentence (BOS + sentence + " =>" + LF + EOS, loss on the LF only), at most 3
epochs, best epoch by dev exact match, greedy decoding. **S2, learned linker:** spaCy en_core_web_trf parses each
sentence (not trained on COGS, frozen); a multinomial logistic regression learns, for every (verb, noun) pair, the
role agent / theme / recipient / none from the dependency path (labels, directions, prepositions), voice, relative
position, verb lemma and whether the verb has two objects; regularisation chosen on dev. **Reference:** CSI v0.1
written correspondences with the same parser (k3 run, `results/e6a_cogs_en_core_web_trf.json`). **Metrics:** main-event
role-set exact match (Experiment 6a metric) for all three, per gen case; full logical-form exact match for S1 (the
standard COGS metric). **Hypotheses.** (a) S1 >= 0.95 full-LF exact match on test, lower on gen, lowest on the
structural cases (cp_recursion, pp_recursion, obj_pp_to_subj_pp), as COGS reported for sequence models; (b) S2 >= S1
on the structural cases at the role level, and S2 >= CSI v0.1 overall (it can learn recipient passives and
unaccusative verbs from data); (c) S2's remaining errors are mostly parser errors (double objects, rare verbs).
Licence: COGS MIT; LFM3 S3 is Clover's own checkpoint; the LoRA adapter is a research artefact, not for the runtime.
**Addition before any S1 result (2026-11-02 00:25 UTC):** S1 is also trained with seeds 2 and 3 (same settings; the
first run uses seed 1234) to measure seed variance; S1 is reported as mean and range over the three seeds. Two GPUs
were idle, which is why this was added now; it does not change any hypothesis.
**S2 result (k3; `results/e6b_s2_trf.json`; C = 10 chosen on dev):** role-set exact match test 0.838, gen 0.913
(index-based 0.849 / 0.950) - identical overall to CSI v0.1's written correspondences with the same parser (0.838 /
0.913; `prototype/e6b_table.py`); per case S2 is higher on obj_to_subj / prim_to_obj_proper (+0.003 to +0.025) and
lower on obj_pp_to_subj_pp (0.714 vs 0.760). So hypothesis (b) is **not supported** in its second half: learned
linking with my pre-registered features equals the written rules, it does not exceed them. The reason is visible in
the features: nothing tells the model that a passive verb still has an object ("The girl was given a box"), so it
cannot learn recipient passives. **Post-hoc S2b (pre-registered 2026-11-02 01:55 UTC as a development run on COGS,
which is development data since its errors were read):** add that one cue (voice x number of remaining objects).
Hypothesis: obj_pp_to_subj_pp and test exact match rise above the written rules, showing that a missing
correspondence can be learned from data once the structure offers the cue.
**S2b result (development run, `results/e6b_s2b_trf.json`):** test 0.890 (S2 0.838, written rules 0.838), gen 0.928
(0.913 / 0.913), obj_pp_to_subj_pp 0.848 (0.714 / 0.760), subj_to_obj_proper 0.852 (0.793 / 0.791). With one
structural cue the learned linker absorbs recipient passives from data and exceeds the written correspondences. This
is post-hoc on development data: it shows feasibility, not a validated result; a validated claim needs a fresh test.
**S1 results and the 6b comparison (`results/e6b_s1_summary.json`, `prototype/e6b_table.py`; main-event role-set
exact match, S1 also full logical form).**

| | CSI-sm | CSI-trf | S2 | S2b (post-hoc) | S1 mean of 3 seeds (range) | S1 full LF |
|---|---|---|---|---|---|---|
| test (in-distribution) | 0.786 | 0.838 | 0.838 | 0.890 | 1.000 | 0.994 |
| gen, all 21 cases | 0.835 | 0.913 | 0.913 | 0.928 | 0.910 (0.903-0.921) | 0.817 |
| structural cases (cp_recursion, pp_recursion, obj_pp_to_subj_pp) | 0.812 | 0.871 | 0.856 | 0.901 | **0.591** (0.558-0.620) | **0.051** |
| cp_recursion | 0.862 | 0.882 | 0.882 | 0.882 | 0.514 | 0.018 |
| obj_pp_to_subj_pp | 0.682 | 0.760 | 0.714 | 0.848 | 0.463 | 0.001 |
| pp_recursion | 0.893 | 0.971 | 0.971 | 0.972 | 0.796 | 0.135 |
| lexical cases (18) | lower: parser errors | 0.77-0.99 | 0.79-0.98 | 0.83-0.99 | mostly 1.000; proper-name objects 0.73, unacc->transitive 0.88 | |

**Against the hypotheses.** (a) **Confirmed**: S1 reaches 0.994 full-LF exact match in distribution, 0.817 on gen, and
collapses on the structural cases (full LF 0.001-0.135) - the COGS pattern for sequence models, now shown for Clover's
C1 with LoRA. (b) First half **confirmed** (structural, role level: S2 0.856 vs S1 0.591); second half not supported
(S2 = written rules; only the post-hoc cue variant exceeds them). (c) Not analysed. **Conclusion:** the two kinds of
learner fail in opposite places. The sequence learner generalises to new words in known structures almost perfectly
and fails on new structures (deeper recursion, modified subjects); learning over explicit structure generalises across
structures and loses on lexical cases through parser errors. This is direct evidence for CSI's split: structure from an
explicit layer, word knowledge learned.

## Experiment 7 = E8: generalisation and training objectives (defined; nanochat-based)
Small models trained from scratch at equal compute: next-token only vs next-token + dependency + role +
proposition-contrast heads; evaluated on E1-E7 held-out sets and COGS generalisation splits.

## Experiment 8 = A/B/C: LLM without CSI, LLM with CSI, CSI alone (defined)
Same inputs from E5-E7; the connected LLM is LFM3-Omni-2B S3 (Clover's C1), with CSI's JSON in its prompt vs not.

**First run plan (pre-registered 2026-11-01, on the Experiment 1 pairs; `prototype/run_abc.py`).** The model gets
its own chat template with one user message: "Sentence 1: ... / Sentence 2: ... / Do these two sentences mean the same
thing? Answer Yes or No." Conditions: **A** = that message only; **B1** = plus CSI's proposition analysis of each
sentence as JSON (predicate, role texts, polarity, tense, aspect, modality, voice, embedded_under; no verdict);
**B2** = B1 plus CSI's comparison (equivalent, differences, notes); **C** = CSI alone. Score = log P(Yes) - log P(No)
over the first answer token (single-token case/space variants summed). Decisions: the model's own choice (score > 0)
and a threshold chosen on seen and applied to unseen (as for the embedding baselines); AUC threshold-free; per
phenomenon; the probability mass on Yes/No is reported to show whether the model answers directly.
**Hypotheses.** (a) A is below C on distinctions with high word overlap; (b) B1 improves A by >= 0.10 unseen balanced
accuracy (the representation helps the LLM without being told the answer); (c) B2 is close to C; where B2 and C
disagree we check whether the LLM corrected CSI (the dative errors) or overrode a correct verdict.
**Protocol correction before the full run.** The S3 template ends the generation prompt with "<think>". The first
smoke run scored the token right after it: Yes/No probability mass was ~0 (the model writes "</think>" first), so
those scores were meaningless; that run was stopped and kept as `results/e8_smoke_INVALID_think_position.*`. A greedy
probe (`prototype/abc_probe.py`) showed the model's own output in 12/12 cases is "</think>", "Yes"/"No",
"<|im_end|>"; the answer token is therefore scored after "</think>" (appended as its token id).

**Results (GPU server card 2, `results/e8.json`, `e8_disagree.json`; Yes/No probability mass 0.995-0.9995).**

| Condition | Own choice, unseen (seen) | Seen threshold, unseen (seen) | AUC unseen (seen) | Said Yes, unseen |
|---|---|---|---|---|
| A: LFM3 S3 alone | 0.851 (0.884) | 0.920 (0.944) | 0.984 (0.988) | 0.45 |
| B1: + CSI analysis, no verdict | 0.910 (0.940) | 0.919 (0.943) | 0.973 (0.976) | 0.35 |
| B2: + CSI analysis and verdict | 0.988 (0.992) | 0.988 (0.995) | 0.994 (0.999) | 0.22 |
| C: CSI alone | 0.988 (0.992) | - | - | - |

A per phenomenon (unseen, own choice): progressive 0.02, attempt 0.57, passive vs negated active 0.74, future 0.80,
negation 0.94, role reversal 1.0, equivalences 0.92-1.0. B1 vs A item flips: unseen 58 fixed (attempt 21,
passive vs negated active 14, future 11) and 7 broken (modality 5); progressive stays wrong (0.07).
**Against the hypotheses.** (a) **Confirmed**: alone, the LLM treats aspect and attempt as "same meaning". (b) **Not
met**: B1 adds 0.059 at the model's own choice, nothing at a fitted threshold (0.919 vs 0.920) and slightly lowers AUC;
the analysis without a verdict shifts the decision bias and fixes some phenomena but does not improve the model's
ranking. (c) **Confirmed**: on unseen, B2 equals C on all 531 items - the LLM followed CSI everywhere, including CSI's 3
wrong dative verdicts. On seen, 4 disagreements: it corrected 1 CSI dative error and overrode 3 correct verdicts
(2 predicate changes, 1 progressive). **Conclusion.** With this 2B model, CSI helps mainly as a verdict the model
follows; it does not supply error correction, so CSI's own errors pass through. Label note: "was opening" vs "opened"
is labelled a meaning difference (aspect; the brief lists tense/aspect); the LLM disagrees in 98% of cases, so this
labelling choice matters for any LLM comparison.

**Experiment 8b: A/B/C on natural text (pre-registered 2026-11-01 UTC, before running; `prototype/run_abc_paws.py`).**
Same model, prompts, answer position and conditions as Experiment 8, on all 8,000 PAWS labeled_final test pairs, where
CSI v0.1 alone is near chance (Experiment 1N). Metrics: own-choice balanced accuracy, AUC, Yes share, and how often the
LLM's answer agrees with CSI's verdict when CSI is right vs wrong (covered pairs). **Hypotheses.** (a) A's AUC is well
above CSI alone; (b) B2 lowers A's own-choice balanced accuracy because the LLM follows CSI's unreliable verdicts
(the integration risk Experiment 8 suggests); (c) B1 changes A's balanced accuracy by less than 0.03. If (b) holds,
CSI must not hand an LLM a verdict without calibrated confidence and coverage limits.

**Results (`results/e8b.json`; Yes/No mass 0.990-0.998).**

| Condition | Own choice bal. acc | AUC | Agrees with CSI when CSI is right / wrong |
|---|---|---|---|
| A: LFM3 S3 alone | 0.812 | 0.910 | 0.84 / 0.26 |
| B1: + CSI analysis | 0.819 | 0.895 | 0.90 / 0.30 |
| B2: + CSI analysis and verdict | **0.719** | 0.826 | 0.98 / **0.68** |
| C: CSI alone (covered 91.5%) | 0.602 | - | - |

All three hypotheses hold: (a) the LLM alone is far better than CSI v0.1 on natural text; (b) given CSI's verdict, it
follows it even when CSI is wrong (68% vs 26% without the verdict) and loses 0.093 balanced accuracy; (c) the analysis
without a verdict changes almost nothing (+0.007). **Conclusion (integration rule, validated on this data):** an LLM
connected to CSI defers to CSI's verdict. CSI may only pass a verdict where it is reliable, with calibrated confidence
and explicit coverage; v0.1 has neither, so its verdicts must not be given to an LLM on open text.
