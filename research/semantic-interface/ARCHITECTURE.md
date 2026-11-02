# CSI deliverables 3 and 5: proposed architecture and interface contract (v0.1)

Status: **proposal**. Only the parts marked "prototype" exist as code (prototype/); everything else is a design to be
tested. Stage 2 (experiential process understanding) is out of scope.

## 1. Layers (Meaning Engine inside, CSI contract outside)

| Layer | Produces | Learned or written | v0.1 prototype source |
|---|---|---|---|
| L1 tokens and morphology | tokens, lemmas, UPOS, morphological features (Tense, Aspect, Mood, Voice, VerbForm, Number, Polarity) | learned (parser/tagger trained on a treebank) | spaCy `en_core_web_sm` 3.8.0 (MIT; trained on OntoNotes; ClearNLP dependency labels) |
| L2 syntax | dependency tree (head, relation), phrase spans | learned | same parser |
| L3 proposition structure | predicate, roles (ARG0 agent / ARG1 patient-theme / ARG2 recipient), polarity, tense, aspect, modality, voice, embedding predicate (e.g. "tried to") | **written**: a small, listed set of structural correspondences (section 3), measured in E6/E7 | prototype `csi/analyze.py` |
| L4 lexical semantics | senses (OEWN synset ids) with probabilities, relations | inventory curated (OEWN); choice in context learned | not in v0.1 (E1, E5) |
| L5 meaning representation | per-proposition vectors and a sentence vector | learned | not in v0.1 (uses research/meaning-core encoders as comparison only) |
| L6 comparison | equivalent / not, with typed differences and confidence | written on top of L3 in v0.1; learned later | prototype `csi/compare.py` |

Principle: written parts stay small, listed and measured; whenever a written part fails measurably, the next experiment
replaces it with a learned one. Every field records which layer and which component produced it.

Layer model (human, 2026-11-02; **a core CSI design principle**, decided by the human after Experiment 4g):
**Dictionary** = what a word means (L4, OEWN; also which words exist and their
listed irregular forms); **Morphology** = how a word is constructed and changes (the learned component
`csi/morphology.py`, split out of L1); **Grammar** = how words function together (L2 and the correspondences of L3);
**Semantics** = what the resulting expression means (L3 propositions, L5, L6). Each layer is validated on its own job,
and a layer does not absorb another's knowledge: lemmatisation composes Grammar (class), Morphology (ranked
candidates) and Dictionary (`csi/dictionary.py`, Experiment 4g).
Boundary (human, 2026-11-02, after 3c): the Dictionary reports candidate meanings and evidence; CSI decides how those
meanings relate in the context of the whole proposition; the dictionary never declares two sentences equivalent.

Frozen components (artefact, validated scope, status; details and checksums in EXPERIMENTS.md):

| Component | Validated scope | Documented boundary | Status |
|---|---|---|---|
| `morph-verb-v1` (verb lemmatiser: Morphology + OEWN) | inflected verbs (VBD/VBN/VBG/VBZ) inside English ending families: held-out lemmas 0.96, fresh real text 0.988 (trf 0.960) | ending families never seen (0.56-0.76); prefers a known word for unknown verbs | frozen; not in CSI (4d failed) |
| `morph-verb-v1.1` (v1 + analysis by synthesis) | as v1; real text 0.9925 (trf 0.975), invented verbs 0.972, CSI minimal pairs with invented verbs 1.000 and no predicate mismatch | ending families never seen; stress-dependent doubling in its generator can veto a correct lemma | frozen; **the CSI verb lemmatiser** (4h passed; enabled with `CSI_MORPH` + `CSI_DICT`, recorded in each proposition's provenance) |
| Dictionary v1.1 (`csi/senses.py` + checkpoint `dictionary-v1.1`) | meaning of a single-word noun/verb/adjective in context among its OEWN meanings: running text 0.71 (frequency table 0.61-0.63), meanings never trained on 0.67; contract section 2b | lemmas without SemCor training examples (OEWN's first sense better: 0.81 vs 0.66-0.69); other text domains untested; multiword expressions; uncalibrated scores; two synonyms in the same sentence often get neighbouring, different synsets (Experiment 3c); the cosine between its meaning vectors measures relatedness, not sameness (3c2) | frozen after checks F1-F5; **not in CSI's comparison** (3c failed: synset identity too brittle; 3c2 failed: closeness of its meaning vectors does not separate synonyms from related meanings) |

## 2. Interface contract v0.1 (model-independent; JSON)

`analyze(text) -> Analysis`

```json
{
  "csi_version": "0.1",
  "text": "The door was opened by the child.",
  "readings": [
    {
      "confidence": 0.94,
      "tokens": [{"i": 1, "text": "door", "lemma": "door", "upos": "NOUN", "feats": {"Number": "Sing"}, "head": 3, "dep": "nsubjpass"}],
      "propositions": [
        {
          "predicate": {"lemma": "open", "token": 3},
          "roles": {"ARG0": {"text": "the child", "head_lemma": "child", "tokens": [5, 6]},
                    "ARG1": {"text": "The door", "head_lemma": "door", "tokens": [0, 1]}},
          "polarity": "positive",
          "tense": "past", "aspect": "simple", "modality": null, "voice": "passive",
          "embedded_under": null,
          "provenance": {"syntax": "spacy/en_core_web_sm@3.8.0", "structure": "csi.analyze@0.1"}
        }
      ]
    }
  ],
  "ambiguity": [],
  "unsupported": []
}
```

`compare(text_a, text_b) -> Comparison`

```json
{"csi_version": "0.1", "equivalent": false, "confidence": 0.9,
 "differences": [{"type": "polarity", "a": "positive", "b": "negative", "proposition": 0}],
 "notes": ["voice differs (active vs passive): grammatical, not a meaning difference"]}
```

Contract rules:
- Versioned (`csi_version`); fields are only added in minor versions, never repurposed.
- `readings` is a list: more than one when the analysis is ambiguous (v0.1 returns one; E5 tests alternatives).
- Fields CSI cannot determine are `null` and listed in `unsupported` (e.g. a construction outside L3's coverage),
  never guessed.
- Provenance on every proposition; confidence is calibrated in E8/E10 (v0.1 reports the parser-agnostic value 1.0 or
  a coverage-based lower value, documented as uncalibrated).
- Model-independent: no field depends on a particular LLM; a connected LLM receives the JSON (or a rendered text
  form) as input. As a service: same JSON over HTTP (later). Inside Clover: through the existing
  `CloverSemanticInterface` seam (runtime integration needs the human's approval).

## 2b. Dictionary layer contract (Dictionary v1.1, `csi/senses.py`; written 2026-11-02 for freeze check F4)

`Senses(checkpoint, oewn_path).choose(text, start, end, lemma, upos) -> SenseChoice`

```json
{"lemma": "bank", "upos": "NOUN", "span": [4, 8], "status": "chosen", "windowed": false, "usage_trained": true,
 "senses": [{"synset": "oewn-08437235-n", "definition": "a financial institution that accepts deposits ...",
             "score": 12.3, "p": 0.91}, {"synset": "oewn-09236472-n", "definition": "sloping land ...",
             "score": 9.9, "p": 0.08}],
 "scores_calibrated": false,
 "provenance": {"component": "csi.senses", "version": "dictionary-v1.1", "code_sha256": "...",
                "checkpoint_sha256": {"context/model.safetensors": "e78597e79e7caa50",
                                      "gloss/model.safetensors": "0d913fb27fe37703", "...": "configs"},
                "weights_stored": "float16", "compute": "float32", "tokenizer": "bert-base-uncased-86b5e09",
                "tokenizer_sha256": "...", "inventory": "english-wordnet-2025.xml.gz",
                "inventory_sha256": "9ca6d1dcb75f822f"}}
```

- **Inputs.** The sentence, the character span of the word, its lemma (parser, or the frozen verb lemmatiser for
  inflected verbs) and its UD part of speech. Grammar and Morphology supply these; the dictionary layer never
  re-tokenises or re-lemmatises.
- **Scope.** Single-word lemmas tagged NOUN, VERB or ADJ. Candidates are all OEWN 2025 meanings of (lemma, part of
  speech), built exactly as in training: OEWN sense order within an entry; where entries merge, head adjectives come
  before satellites and lower-case lemmas before capitalised ones (our construction, not OEWN's, for 576 keys).
- **Outside the scope, nothing is guessed:** other parts of speech, multiword expressions and non-alphabetic lemmas
  -> `out_of_scope`; an empty or out-of-range span -> `invalid_span`; a lemma OEWN does not list ->
  `not_in_dictionary`; a word that cannot be located even in a window -> `target_not_located`; all with an empty
  `senses` list. A lemma with one meaning -> `single_meaning` with that meaning, p = 1.0 and no model call.
- **Output.** All candidates, best first, with the raw dot-product `score` and a softmax `p` over the candidates.
  `scores_calibrated` is false: p is a ranking aid, not a probability anyone may act on (Experiment 8b: no verdict to
  an LLM without calibrated confidence). `windowed` is true when the word lay beyond 128 word pieces and was encoded
  in a 250-character window around it. `usage_trained` says whether the lemma had SemCor training examples: where it
  did not, OEWN's first sense beat the model on running text (0.81 vs 0.66-0.69), so a consumer should weigh the
  choice accordingly.
- **Determinism and provenance.** Weights stored in float16, computed in float32 in evaluation mode; the same input
  gives the same output, independent of earlier queries and across processes. Every result names the component
  version and the checksums of the code, weights, configs, tokenizer and inventory.
- **Not a complete dictionary.** No definitions are generated, no relations between meanings, no adverbs, multiword
  expressions or words outside OEWN. Accuracy 0.71 on running text assumes the correct lemma, part of speech and span
  and covers only words with at least two meanings whose SemCor sense maps to OEWN, in SemCor (Brown) text; OEWN's
  first sense scores 0.684 on the same items; other text domains are untested.

## 3. v0.1 structural correspondences (the complete written list)

Voice to roles (ClearNLP labels from the parser):
1. Active clause: `nsubj` -> ARG0; `dobj` -> ARG1; `dative` / indirect object -> ARG2 (a `dative` preposition
   contributes its `pobj`); when there is a `dobj` and no `dative`, a `prep` "to" + `pobj` -> ARG2 ("gave a ball to
   the dog").
2. Passive clause (`nsubjpass` or `auxpass` present): `nsubjpass` -> ARG1; `agent` ("by" + `pobj`) -> ARG0.
Tense, aspect, modality from auxiliaries and the predicate's tag:
3. Modal auxiliaries can/could/may/might/must/shall/should/would -> modality (lemma); will/shall -> tense future.
4. `have` auxiliary + past participle -> aspect perfect; `be` auxiliary + gerund -> aspect progressive.
5. Otherwise tense from the finite verb's morph feature (Past / Pres).
Polarity:
6. A `neg` dependent of the predicate or of its auxiliary chain -> negative.
Embedding:
7. A root predicate with an `xcomp` verb ("tried to open", "wants to open") -> the proposition is the `xcomp` verb,
   `embedded_under` = the root lemma (attempt, desire, ...), which is a meaning difference.
Comparison:
8. Two analyses are equivalent iff their main propositions match on predicate lemma, role head lemmas (ARG0, ARG1,
   ARG2), polarity, tense, aspect, modality and embedding; voice is ignored (grammatical difference).

## 3b. The same correspondences for UD labels (written 2026-11-02 for Experiment 1b, before its run)
Used automatically when the parser's label set is UD (e.g. the pipeline we trained on UD English-EWT). Rule by rule:
1. Active: `nsubj` -> ARG0; `obj` -> ARG1; `iobj` -> ARG2; when there is an `obj` and no `iobj`, an `obl` with a
   `case` dependent "to" -> ARG2.
2. Passive (`nsubj:pass`, `aux:pass` or `csubj:pass` on the predicate): `nsubj:pass` -> ARG1; `obl:agent` -> ARG0.
3-5. Unchanged (auxiliaries are `aux` / `aux:pass` children of the root; modal = tag MD; tense from tags).
6. An `advmod` dependent with Polarity=Neg (not, n't, never) on the predicate or root -> negative.
7-8. Unchanged (`xcomp` has the same name in both schemes).
No correspondence was added or removed; the recipient passive found in Experiment 6a is still not covered.

## 3c. Request correspondences (v0.1r; written 2026-11-02 for Experiment 1d, before its data)
9. Every sentence is analysed. In a sentence, the root verb's clause and each verb attached to it by `conj` or
   `parataxis` form their own propositions, in text order. A clause without its own subject takes the subject of the
   clause it is attached to; polarity, tense, aspect and modality are read from each clause's own dependents (a
   negation is not carried into another clause).
10. A clause whose verb is in base form (tag VB), with no subject of its own and no auxiliary other than do-support
    ("do", "don't", "do not"), is an imperative: mood "imperative", ARG0 = the addressee (implicit, head lemma
    "you"), tense "unmarked"; a clause attached to an imperative without its own subject is an imperative too. Every
    other proposition has mood "indicative". Negation as rule 6.
11. A token belongs to the proposition of its nearest ancestor (itself included) that is a proposition's predicate. A
    negation inside a subordinate clause that forms no proposition does not change that proposition's polarity.
9b. (v0.1r2, Experiment 1e, written after 1d) A verb attached to a clause's head by `advcl` or `ccomp` forms its own
    proposition when it comes before that head, a comma, semicolon or colon stands between them, and it has no
    subordinating word (`mark`) or infinitival "to": the parser's reading of two main clauses joined in one sentence
    ("Don't search the web ..., just tell me ..."). It takes no subject from that head.
12. (v0.1r3, Experiment 1f, written after 1e; amended before any 1f test data) A sentence the parser reads without a
    verbal root is re-read as a request only when the Grammar layer supports that reading twice: (a) its tagger gives
    the sentence's first word, as written, a probability of at least 0.02 of being a base-form verb (tag VB); (b) the
    parser, reading the same words as a polite request ("Please" + the sentence with its first letter lowercased),
    makes that word a VB heading the sentence with no subject, finds one sentence, attaches nothing to the added
    "Please", and the words align one to one. Then the sentence is read from that parse with "Please" removed, by
    rules 1-11 unchanged (rule 10 makes it an imperative with the addressee as ARG0); otherwise it stays unread. A
    sentence with a verbal root is never re-read. The request form adds no content ("Please forward the timetable." is
    the request "Forward the timetable."); the threshold and the form were chosen on development data (EXPERIMENTS.md,
    Experiment 1f and its amendment).
Comparison (rule 8) adds mood to the compared fields and still compares each text's first proposition.
