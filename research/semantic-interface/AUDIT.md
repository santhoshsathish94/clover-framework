# CSI deliverable 1: audit of Clover's existing semantic implementation

Scope: clover-framework main at 4150aac, plus branch experiment/meaning-development-pilot-cycle2 and the measurements
in research/meaning-core. Every line reference below was checked against the source on 2026-11-02.
Categories: **learned** (parameters fitted to data), **deterministic** (fixed computation, e.g. hashing),
**hardcoded** (hand-written lists, rules, regexes), **model-dependent** (needs a specific LLM's internals).

## 1. Components

| Component | Where | Category | What it actually does |
|---|---|---|---|
| Intent and act routing | `clover-ai-system/mobile/semantics.py` L91-169 (regexes), `interpret()` L398, `Frame` L208 | hardcoded | Regexes classify a message into an act (social, concept, action, question, ...) and tools to offer. `NEGATION` (L165) is a keyword list used to check reply claims, not negation scope. |
| `SubwordHashEmbedder` | `clover-ai-system/mobile/memory_index.py` L168-195 | deterministic | 384-d vector from MD5 hashes of words (L182) and character trigrams (L189), signed, L2-normalised. No learned meaning. |
| `Embedder` (bge-small) | `memory_index.py` L139 | learned (external model) | BAAI/bge-small-en-v1.5 at a pinned revision (L37), used when present on disk. |
| `CLOVER_SEMANTIC_ANCHORS` | `clover-ai-system/mobile/clover_semantics.py` L67 | hardcoded | 18 hand-written anchor sentences: intents (tools), query views, domains, a risk anchor. |
| `IndraSemanticSpace` | `clover_semantics.py` L97 (falls back to the hash embedder when no `embed_fn`, L112-116) | deterministic over its embedder | Profile = angular distances of a text to the 18 anchors. Meaning comes only from the anchor labels and the embedder. |
| `CloverSemanticInterface.analyze_turn` | `clover_semantics.py` L581, L629-641 | composite | Builds a `SemanticFrame` (L563) from the Indra profile, regex query view, feasibility rules. Accepts `c1_layer18_vec` for a C1-state profile; nothing in the runtime calls it with one. |
| Runtime construction | `clover-ai-system/mobile/s23_on_device_clover.py` L2226 | - | `CloverSemanticInterface()` with no `embed_fn`: the phone runtime uses the hash embedder path. |
| Meaning-development pilot | `src/tools/hybrid-ncm-k3/meaning_development_pilot.py` L814-871 | hardcoded | "Paraphrase identification" = per-concept hand-written keyword lists (e.g. `["flat","unchanged","stuck",...]` AND `["size","capacity","doubling",...]`) matched against 4 hand-written test sentences that contain those words. |
| Semantic architecture study | `clover-ai-system/mobile/semantic_architecture_study_report.json` | measurement | Real LFM3-Omni-2B hidden-state measurements on 30 concepts (e.g. peak RSA 0.554 at layer 18; mean-centring gain +0.18 to +0.20). About representational geometry, not linguistic analysis. |
| Ontology graph | `src/tools/hybrid-ncm-k3/k3_ontology_graph.py`, `symbolic-ontology-graph.py` | hardcoded (curated edges + Horn rules) | CVE/CWE/CAPEC property graph and rule inference; domain knowledge, no language analysis. |
| C embedding tables | `src/code/common/words-to-embeddings.c`, `embedding-and-head-tables.h` | model-dependent | Token-to-embedding lookup for the model engines. |

No runtime code uses WordNet, a POS tagger, a parser, Universal Dependencies, FrameNet, PropBank or AMR (searched
2026-11-02; the only uses are in research harnesses under research/meaning-core and scratch folders).

## 2. Measured behaviour (research/meaning-core, test splits; harness/core_eval.py)

| Encoder | STS (9 MTEB tasks) | PAWS AUC | SNLI rank | CondaQA AUC | "The server stopped because memory ran out." vs paraphrase / contradiction |
|---|---|---|---|---|---|
| Clover hash (phone path) | 66.87 | 0.488 | 0.784 | 0.399 | 0.415 / **0.726** |
| Clover Indra-hash (runtime profile) | 52.11 | 0.487 | 0.683 | 0.391 | 0.785 / **0.944** |
| bge-small | 81.28 | 0.605 | 0.956 | 0.505 | 0.915 / 0.931 |

PAWS (same words, different meaning) is at chance for both Clover representations. On CondaQA they rank
meaning-changing negation edits above meaning-preserving paraphrases (AUC below 0.5): edits that keep more words win
(paraphrase edits share Jaccard 0.652 of their words with the original, scope edits 0.744).

## 3. Verdict per Stage 1 capability

| Capability | Exists today? |
|---|---|
| Lexical senses, definitions, POS | no |
| Morphology (lemmas, inflection, derivation) | no |
| Grammar (tense, aspect, mood, voice, negation, questions, conditionals) | no (keyword regexes only, for routing) |
| Syntax (dependencies, subject/object, phrases) | no |
| Semantic relations (synonym, antonym, hypernym, meronym, roles) | no (curated domain graphs only) |
| Composition | no |
| Context and ambiguity (sense, structure, reference) | no |
| Equivalence vs distinction | no (measured above: fails) |
| Structured, versioned interface | no: `SemanticFrame`/`Frame` are unversioned routing records |

What is genuinely learned today: only the optional bge-small embedder and the LFM3 model itself. Everything called
"semantic" in the runtime is hardcoded routing plus deterministic geometry over hashes or that embedder.
