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

Layer model (human, 2026-11-02; **a core CSI design principle**, decided by the human after Experiment 4g): **Dictionary** = what a word means (L4, OEWN; also which words exist and their
listed irregular forms); **Morphology** = how a word is constructed and changes (the learned component
`csi/morphology.py`, split out of L1); **Grammar** = how words function together (L2 and the correspondences of L3);
**Semantics** = what the resulting expression means (L3 propositions, L5, L6). Each layer is validated on its own job,
and a layer does not absorb another's knowledge: lemmatisation composes Grammar (class), Morphology (ranked
candidates) and Dictionary (`csi/dictionary.py`, Experiment 4g).

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
