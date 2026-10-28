# Object-Oriented Reduction of the Clover Repository (398 → 88 → 24 → 12 → 5)

This investigation inspects all source code across the Clover repository (`C`, `CUDA/C++`, `Rust`, `Python`, and `JavaScript`), catalogs every existing `class`, `struct`, `trait`, and `enum` alongside its relationships, and reduces them step by step into a clean, hyphenated complete-English Object-Oriented architecture implemented in [`clover-object-model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-object-model.py) (powered by [`clover_hyphen_runtime.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover_hyphen_runtime.py)) and tested in [`test-clover-object-model.py`](../../scripts/tests/test-clover-object-model.py).

## Hyphenated Complete-English Role-Prefix Naming (`class-*`, `function-*`, `variable-*`, `flag-*`, `constant-*`, `decorator-*`)

Every symbol in [`clover-object-model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-object-model.py) uses literal ASCII hyphens `-` and an explicit role prefix so a human reader can immediately distinguish classes, functions, variables, boolean flags, constants, and decorators at a glance:

| Code Element | Role Prefix & Hyphenated English Rule | Examples in [`clover-object-model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-object-model.py) |
|---|---|---|
| **Class / Interface** | `class-*` (**Complete Noun Phrase** with `-`, zero abbreviations) | `class-task-direction`, `class-conversation-session`, `class-forward-pass-workspace`, `class-verified-knowledge-graph`, `class-memory-storage-tier`, `class-expert-slot-cache`, `class-transformer-stage`, `class-neural-stage-pipeline`, `class-artifact-verifier`, `class-verification-verdict` |
| **Function / Method** | `function-*` (**Complete Verb-Phrase Sentence** with `-`) | `function-encode-input-into-activation-vector()`, `function-ensure-expert-tensor-resident()`, `function-compact-oldest-turns-into-summary-when-full()`, `function-verify-candidate-artifact-against-context()`, `function-execute-complete-clover-cycle()` |
| **Variable / Attribute** | `variable-*` (**Complete Descriptive Noun Phrase** with `-`) | `variable-resident-entries-by-identifier`, `variable-carbon-copy-snapshots`, `variable-protected-file-paths`, `variable-capacity-in-slots`, `variable-maximum-token-budget`, `variable-backing-storage-tier` |
| **Boolean Flag / Predicate** | `flag-*` (**Question Clause** with `-`) | `flag-has-passed-all-checks`, `flag-has-exceeded-maximum-token-budget`, `flag-is-cache-currently-full`, `flag-is-currently-active`, `flag-has-non-empty-goal` |
| **Constant** | `constant-*` (**Complete Quantified Noun Phrase** with `-`) | `constant-maximum-session-context-tokens`, `constant-default-snapshot-interval-in-stages`, `constant-total-routed-experts-per-stage` |
| **Decorator** | `@decorator-*` (**Declarative Verb Phrase** with `-`) | `@decorator-register-component-in-catalog`, `@decorator-enforce-direction-boundary`, `@decorator-record-execution-telemetry`, `@decorator-emit-clover-cycle-stage` |

## Hyphenated Complete-English Module File Names

All cryptic `k3_*.py` module names in `clover-kimi-k3/src/tools/hybrid-ncm-k3/` have been given self-explanatory hyphenated English file names (with lightweight compatibility shims at the old names so existing external scripts continue to work):

| Old Abbreviated Name | New Hyphenated Complete-English File Name | Purpose |
|---|---|---|
| *(new unified model)* | [`clover-object-model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-object-model.py) | Canonical 12-object / 5-stage OOP model using `class-*`, `function-*`, `variable-*`, `flag-*`, `constant-*`, `decorator-*` |
| `k3_generative_compiler.py` | [`generative-code-compiler.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/generative-code-compiler.py) | On-the-fly Python & Rust partial-program sealing and statement-boundary compiler verifier |
| `k3_memory_graph.py` | [`memory-knowledge-graph.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/memory-knowledge-graph.py) | File-backed Markdown memory tree, staging/promotion gate, and relational graph |
| `k3_ontology_graph.py` | [`symbolic-ontology-graph.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/symbolic-ontology-graph.py) | Deterministic Horn-rule ontology graph and McCulloch-Pitts symbolic claim verifier |
| `k3_agent_engine.py` | [`clover-agent-engine.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-agent-engine.py) | Unified multi-step agent planner, Kimi K3 MoE expert runner, and NCM decoder |
| `k3_chat_server.py` | [`clover-chat-server.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-chat-server.py) | Multi-session, dynamic-batching HTTP/SSE server and web UI |
| `k3_fast_chat.py` | [`conversational-chat-engine.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/conversational-chat-engine.py) | Direct conversational planning and window rendering engine |
| `k3_multimodal_chat.py` | [`multimodal-vision-chat-engine.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/multimodal-vision-chat-engine.py) | Image/video MoonViT3d + Qwen3-VL multimodal chat engine |
| `k3_data_movement_ledger.py` | [`tiered-data-movement-ledger.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/tiered-data-movement-ledger.py) | 4-tier VRAM/RAM/NVMe data-movement and Eyeriss energy ledger |
| `k3_tokenizer.py` | [`text-bpe-tokenizer.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/text-bpe-tokenizer.py) | Native 163,840-vocabulary BPE/Tiktoken Ctypes tokenizer |
| `live_web_search.py` | [`live-web-search-tool.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/live-web-search-tool.py) | Live Wikipedia, arXiv, DuckDuckGo, and GitHub URL reader tool |
| `clover_paths.py` | [`workspace-path-resolver.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/workspace-path-resolver.py) | Portable CPU/GPU dataset, checkpoint, and `sys.path` discovery |
| `test_k3_memory_graph.py` | [`test-memory-knowledge-graph.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/test-memory-knowledge-graph.py) | 11-test verification suite for the file-backed memory & ontology graph |

## Step-by-Step Reduction Files

| Step | Document | Reduction | Core Question Answered |
|---:|---|---:|---|
| **0** | [All Current Classes & Relationships](00-current-classes-and-relationships.md) | **398 raw types** (`330` runtime + `68` driver helpers) | What classes, structs, traits, and enums existed across the repository, and how were they connected? |
| **1** | [Reduction 1 — Eliminate Clones & Mirrors](01-reduction-1-eliminate-clones-and-mirrors.md) | **398 → 88 classes** | Which types are byte-for-byte clones, C/C++/Rust/Python FFI mirrors, or file-scoped aliases of the exact same structure? |
| **2** | [Reduction 2 — Polymorphic Domain Archetypes](02-reduction-2-domain-archetypes-and-polymorphism.md) | **88 → 24 classes** | How do composition (`has-a`) and polymorphism (`is-a`) unify layer-index variants, memory pools, codecs, bridges, graphs, and verifiers? |
| **3** | [Reduction 3 — Complete-English Domain Objects & Design Patterns](03-reduction-3-real-world-objects.md) | **24 → 12 objects** | How do we organize the 24 archetypes into 12 `class-*` domain objects using `function-*`, `variable-*`, `flag-*`, `constant-*`, and `decorator-*` with Strategy, Chain of Responsibility, Memento, and Builder patterns? |
| **4** | [Reduction 4 — Irreducible Clover Core & Executable Model](04-reduction-4-irreducible-clover-core.md) | **12 → 5 stage roots** | How do the 12 `class-*` objects map onto the 5 irreducible responsibilities of the Clover cycle (`Context`, `Direction`, `Execution`, `Outcome`, `Growth`) in [`clover-object-model.py`](../../clover-kimi-k3/src/tools/hybrid-ncm-k3/clover-object-model.py)? |
