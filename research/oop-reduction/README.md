# Object-Oriented Reduction of the Clover Repository (398 → 88 → 24 → 12 → 5)

This investigation inspects all source code across the Clover repository (`C`, `CUDA/C++`, `Rust`, `Python`, and `JavaScript`), catalogs every existing `class`, `struct`, `trait`, and `enum` alongside its relationships, and reduces them step by step into a clean, complete-English Object-Oriented architecture implemented in [`clover_object_model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover_object_model.py) and tested in [`test_clover_object_model.py`](../../scripts/tests/test_clover_object_model.py).

## Complete-English Naming Grammar

Every symbol in our reduced object model follows an unambiguous grammatical role so a human reader can immediately distinguish classes, methods, variables, boolean predicates, and constants at a glance:

| Code Element | Casing & English Grammar Rule | Examples in `clover_object_model.py` |
|---|---|---|
| **Class / Interface** | `UpperCamelCase` **Complete Noun Phrase** (*what the object is*, zero abbreviations) | `VerifiedKnowledgeGraph`, `ConversationSession`, `ForwardPassWorkspace`, `MemoryStorageTier`, `ExpertSlotCache`, `TransformerStage`, `NeuralStagePipeline`, `ArtifactVerifier`, `VerificationVerdict` |
| **Function / Method** | `snake_case` **Complete Verb-Phrase Sentence** (*what action the object performs*) | `encode_input_into_activation_vector()`, `ensure_expert_tensor_resident()`, `compact_oldest_turns_into_summary_when_full()`, `verify_candidate_artifact_against_context()`, `execute_complete_clover_cycle()` |
| **Variable / Attribute** | `snake_case` **Complete Descriptive Noun Phrase** (*what state the variable holds*, plural for collections, explicit units) | `resident_entries_by_identifier`, `carbon_copy_snapshots`, `protected_file_paths`, `capacity_in_slots`, `maximum_token_budget`, `backing_storage_tier` |
| **Boolean Property / Predicate** | `@property` `snake_case` **Question Clause** (`is_*`, `has_*`, `can_*`, `allows_*`) | `has_passed_all_checks`, `has_exceeded_maximum_token_budget`, `is_cache_currently_full`, `is_currently_active`, `allows_modification_of_path()` |
| **Constant** | `UPPER_SNAKE_CASE` **Complete Quantified Noun Phrase** | `MAXIMUM_SESSION_CONTEXT_TOKENS`, `DEFAULT_SNAPSHOT_INTERVAL_IN_STAGES`, `TOTAL_ROUTED_EXPERTS_PER_STAGE` |
| **Decorator** | `@snake_case` **Declarative Verb Phrase** | `@register_component_in_catalog`, `@enforce_direction_boundary`, `@record_execution_telemetry`, `@emit_clover_cycle_stage` |

## Step-by-Step Reduction Files

| Step | Document | Reduction | Core Question Answered |
|---:|---|---:|---|
| **0** | [All Current Classes & Relationships](00-current-classes-and-relationships.md) | **398 raw types** (`330` runtime + `68` driver helpers) | What classes, structs, traits, and enums existed across the repository, and how were they connected? |
| **1** | [Reduction 1 — Eliminate Clones & Mirrors](01-reduction-1-eliminate-clones-and-mirrors.md) | **398 → 88 classes** | Which types are byte-for-byte clones, C/C++/Rust/Python FFI mirrors, or file-scoped aliases of the exact same structure? |
| **2** | [Reduction 2 — Polymorphic Domain Archetypes](02-reduction-2-domain-archetypes-and-polymorphism.md) | **88 → 24 classes** | How do composition (`has-a`) and polymorphism (`is-a`) unify layer-index variants, memory pools, codecs, bridges, graphs, and verifiers? |
| **3** | [Reduction 3 — Complete-English Domain Objects & Design Patterns](03-reduction-3-real-world-objects.md) | **24 → 12 objects** | How do we organize the 24 archetypes into 12 complete-English domain objects (keeping `ConversationSession` vs. `ForwardPassWorkspace` and `ArtifactVerifier` vs. `VerificationVerdict` cleanly separated) using Decorators, Strategy, Chain of Responsibility, Memento, and Builder patterns? |
| **4** | [Reduction 4 — Irreducible Clover Core & Executable Model](04-reduction-4-irreducible-clover-core.md) | **12 → 5 stage roots** | How do the 12 domain objects map onto the 5 irreducible responsibilities of the Clover cycle (`Context`, `Direction`, `Execution`, `Outcome`, `Growth`) in [`clover_object_model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover_object_model.py)? |
