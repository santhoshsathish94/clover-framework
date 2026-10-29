# Object-Oriented Reduction of the Clover Repository (398 → 88 → 24 → 12 → 5)

This investigation inspects all source code across the Clover repository (`C`, `CUDA/C++`, `Rust`, `Python`, and `JavaScript`), catalogs every existing `class`, `struct`, `trait`, and `enum` alongside its relationships, and reduces them step by step into a clean, hyphenated complete-English Object-Oriented architecture implemented in [`clover-object-model.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) (powered by [`clover_hyphen_runtime.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover_hyphen_runtime.py)) and tested in [`test-clover-object-model.py`](../../scripts/tests/test-clover-object-model.py).

## Hyphenated Complete-English Role-Prefix Naming (`class-*`, `function-*`, `variable-*`, `flag-*`, `constant-*`, `decorator-*`)

Every symbol in [`clover-object-model.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) uses literal ASCII hyphens `-` and an explicit role prefix so a human reader can immediately distinguish classes, functions, variables, boolean flags, constants, and decorators at a glance:

| Code Element | Role Prefix & Hyphenated English Rule | Examples in [`clover-object-model.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) |
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
| *(new unified model)* | [`clover-object-model.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py) | Canonical 12-object / 5-stage OOP model using `class-*`, `function-*`, `variable-*`, `flag-*`, `constant-*`, `decorator-*` |
| `k3_generative_compiler.py` | [`generative-code-compiler.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/generative-code-compiler.py) | On-the-fly Python & Rust partial-program sealing and statement-boundary compiler verifier |
| `k3_memory_graph.py` | [`memory-knowledge-graph.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/memory-knowledge-graph.py) | File-backed Markdown memory tree, staging/promotion gate, and relational graph |
| `k3_ontology_graph.py` | [`symbolic-ontology-graph.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/symbolic-ontology-graph.py) | Deterministic Horn-rule ontology graph and McCulloch-Pitts symbolic claim verifier |
| `k3_agent_engine.py` | [`clover-agent-engine.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-agent-engine.py) | Unified multi-step agent planner, Kimi K3 MoE expert runner, and NCM decoder |
| `k3_chat_server.py` | [`clover-chat-server.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py) | Multi-session, dynamic-batching HTTP/SSE server and web UI |
| `k3_fast_chat.py` | [`conversational-chat-engine.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/conversational-chat-engine.py) | Direct conversational planning and window rendering engine |
| `k3_multimodal_chat.py` | [`multimodal-vision-chat-engine.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/multimodal-vision-chat-engine.py) | Image/video MoonViT3d + Qwen3-VL multimodal chat engine |
| `k3_data_movement_ledger.py` | [`tiered-data-movement-ledger.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/tiered-data-movement-ledger.py) | 4-tier VRAM/RAM/NVMe data-movement and Eyeriss energy ledger |
| `k3_tokenizer.py` | [`text-bpe-tokenizer.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/text-bpe-tokenizer.py) | Native 163,840-vocabulary BPE/Tiktoken Ctypes tokenizer |
| `live_web_search.py` | [`live-web-search-tool.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/live-web-search-tool.py) | Live Wikipedia, arXiv, DuckDuckGo, and GitHub URL reader tool |
| `clover_paths.py` | [`workspace-path-resolver.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/workspace-path-resolver.py) | Portable CPU/GPU dataset, checkpoint, and `sys.path` discovery |
| `test_k3_memory_graph.py` | [`test-memory-knowledge-graph.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/test-memory-knowledge-graph.py) | 11-test verification suite for the file-backed memory & ontology graph |

## Hyphenated Complete-English Dataset & Checkpoint Names (`dataset-*` & `checkpoint-*`)

All benchmark datasets, memory-graph JSON stores, and trained model checkpoints follow the same `dataset-*` and `checkpoint-*` naming convention:

| Old Name | New Hyphenated Complete-English Name | Category |
|---|---|---|
| `benchmarks/gsm8k_test.jsonl` | [`benchmarks/dataset-grade-school-math-test.jsonl`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/dataset-grade-school-math-test.jsonl) | Benchmark 1 (`GSM8K` test split) |
| `benchmarks/HumanEval.jsonl` | [`benchmarks/dataset-human-eval-python-coding.jsonl`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/dataset-human-eval-python-coding.jsonl) | Benchmark 3 (`OpenAI HumanEval`) |
| `benchmarks/data/test/astronomy_test.csv` | [`benchmarks/data/test/dataset-mmlu-astronomy-test.csv`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/data/test/dataset-mmlu-astronomy-test.csv) | Benchmark 2 (`MMLU` Astronomy) |
| `benchmarks/data/test/college_biology_test.csv` | [`benchmarks/data/test/dataset-mmlu-college-biology-test.csv`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/data/test/dataset-mmlu-college-biology-test.csv) | Benchmark 2 (`MMLU` College Biology) |
| `benchmarks/data/test/college_computer_science_test.csv` | [`benchmarks/data/test/dataset-mmlu-college-computer-science-test.csv`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/data/test/dataset-mmlu-college-computer-science-test.csv) | Benchmark 2 (`MMLU` College CS) |
| `benchmarks/data/test/college_physics_test.csv` | [`benchmarks/data/test/dataset-mmlu-college-physics-test.csv`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/data/test/dataset-mmlu-college-physics-test.csv) | Benchmark 2 (`MMLU` College Physics) |
| `benchmarks/data/test/high_school_us_history_test.csv` | [`benchmarks/data/test/dataset-mmlu-high-school-us-history-test.csv`](../../clover-ai-system/src/tools/hybrid-ncm-k3/benchmarks/data/test/dataset-mmlu-high-school-us-history-test.csv) | Benchmark 2 (`MMLU` US History) |
| `memory_tree/_index/memory_graph_index.json` | [`memory_tree/_index/dataset-memory-graph-index.json`](../../clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree/_index/dataset-memory-graph-index.json) | Rebuildable Memory Graph Index |
| `memory_tree/_ontology/ontology_graph.json` | [`memory_tree/_ontology/dataset-symbolic-ontology-graph.json`](../../clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree/_ontology/dataset-symbolic-ontology-graph.json) | Deterministic Symbolic Ontology Graph |
| `memory_tree/_rules/memory_retrieval_rules.json` | [`memory_tree/_rules/dataset-memory-retrieval-rules.json`](../../clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree/_rules/dataset-memory-retrieval-rules.json) | Retrieval & Symbolic Rules |
| `memory_tree/_rules/visual_templates.json` | [`memory_tree/_rules/dataset-visual-scene-templates.json`](../../clover-ai-system/src/tools/hybrid-ncm-k3/memory_tree/_rules/dataset-visual-scene-templates.json) | SVG Scene & Chart Templates |
| `e12_clover_k3_true_ha_ncm.pt` | `checkpoint-clover-true-hybrid-agent-ncm.pt` | Trained Clover K3 + HybridAgent + NCM Checkpoint |
| `e11_clover_k3_native_ha_ncm.pt` | `checkpoint-clover-native-hybrid-agent-ncm.pt` | Trained Native HybridAgent + NCM Checkpoint |
| `e8_k3_direct.pt` | `checkpoint-next-concept-direct-decoder.pt` | Direct Next-Concept Decoder Checkpoint |
| `e6_last.pt` | `checkpoint-next-concept-generator-base.pt` | Base 11-Pass Next-Concept Generator Checkpoint |
| `ha_8b_int8_cpu.pt` | `checkpoint-hybrid-agent-8b-int8-cpu.pt` | `fbgemm` Int8 CPU-Quantized HybridAgent-8B Checkpoint |
| `k3_vqvae_adapted.pt` | `checkpoint-concept-vector-quantizer-adapted.pt` | Adapted Multi-Scale Residual VQ-VAE Checkpoint |
| `k3_vqvae_init.pt` | `checkpoint-concept-vector-quantizer-initial.pt` | Initial Multi-Scale Residual VQ-VAE Checkpoint |
| `k3_hybrid_bridges.pt` | `checkpoint-latent-representation-bridges.pt` | Cross-Model Latent Bridge Projections |
| `staged_32win_inputs.pt` | `dataset-staged-32-window-inputs.pt` | Staged 32-Window Evaluation Tensor Dataset |
| `data/wmt14/reference_translations.pickle` | `data/wmt14/dataset-wmt14-reference-translations.pickle` | Benchmark 5A (`WMT14` Reference Sentences) |
| `data/xsum/passages.pickle` | `data/xsum/dataset-xsum-news-passages.pickle` | Benchmark 5B (`XSum` News Passages) |
| `data/xsum/reference_summaries.pickle` | `data/xsum/dataset-xsum-reference-summaries.pickle` | Benchmark 5B (`XSum` Reference Summaries) |
| `data/sudoku/easy/puzzles.pickle` | `data/sudoku/easy/dataset-sudoku-easy-puzzles.pickle` | Benchmark 5C (`Sudoku-Easy` 9x9 Puzzles) |
| `data/sudoku/easy/solutions.pickle` | `data/sudoku/easy/dataset-sudoku-easy-solutions.pickle` | Benchmark 5C (`Sudoku-Easy` 9x9 Solutions) |

## Step-by-Step Reduction Files

| Step | Document | Reduction | Core Question Answered |
|---:|---|---:|---|
| **0** | [All Current Classes & Relationships](00-current-classes-and-relationships.md) | **398 raw types** (`330` runtime + `68` driver helpers) | What classes, structs, traits, and enums existed across the repository, and how were they connected? |
| **1** | [Reduction 1 — Eliminate Clones & Mirrors](01-reduction-1-eliminate-clones-and-mirrors.md) | **398 → 88 classes** | Which types are byte-for-byte clones, C/C++/Rust/Python FFI mirrors, or file-scoped aliases of the exact same structure? |
| **2** | [Reduction 2 — Polymorphic Domain Archetypes](02-reduction-2-domain-archetypes-and-polymorphism.md) | **88 → 24 classes** | How do composition (`has-a`) and polymorphism (`is-a`) unify layer-index variants, memory pools, codecs, bridges, graphs, and verifiers? |
| **3** | [Reduction 3 — Complete-English Domain Objects & Design Patterns](03-reduction-3-real-world-objects.md) | **24 → 12 objects** | How do we organize the 24 archetypes into 12 `class-*` domain objects using `function-*`, `variable-*`, `flag-*`, `constant-*`, and `decorator-*` with Strategy, Chain of Responsibility, Memento, and Builder patterns? |
| **4** | [Reduction 4 — Irreducible Clover Core & Executable Model](04-reduction-4-irreducible-clover-core.md) | **12 → 5 stage roots** | How do the 12 `class-*` objects map onto the 5 irreducible responsibilities of the Clover cycle (`Context`, `Direction`, `Execution`, `Outcome`, `Growth`) in [`clover-object-model.py`](../../clover-ai-system/src/tools/hybrid-ncm-k3/clover-object-model.py)? |

## Public Website (`site/`) OOP Reduction, Hyphenated File/Folder Names & Role-Prefix Naming

The public website in [`site/`](../../site/README.md) ([`site/website-interaction-controllers.js`](../../site/website-interaction-controllers.js), [`site/clover-ai/index.html`](../../site/clover-ai/index.html), and [`site/botanical-design-system.css`](../../site/botanical-design-system.css)) applies the exact same OOP reduction, complete-English hyphenated file/folder names (`styles.css` → `botanical-design-system.css`, `app.js` → `website-interaction-controllers.js`, `check.ps1` → `validate-website-pages.ps1`, `favicon.svg` → `clover-leaf-favicon.svg`, `ai/` → `clover-ai/`, `start/` → `getting-started/`), and role-prefix naming (`class_*`, `function_*`, `variable_*`, `flag_*`, `constant_*` in JavaScript and `--constant-*` in CSS), reducing 32 procedural functions and 25 mutable closure variables into **8 cohesive domain classes** mapped directly onto the 12 core objects and 5 Clover stages:

| Website File | Website Domain Class (`class_*`) | Responsibility | Canonical 12-Object & 5-Stage Mapping |
|---|---|---|---|
| [`site/website-interaction-controllers.js`](../../site/website-interaction-controllers.js) | `class_viewport_scroll_navigator` | Eased cubic-bezier anchor scrolling, hash alignment, and same-page link delegation | `ModalityCodec` / `TaskDirection` (`Direction → Execution`) |
| [`site/website-interaction-controllers.js`](../../site/website-interaction-controllers.js) | `class_five_stage_clover_mark_controller` | Pinned 5-leaf stage highlighting, continuous 360° scroll rotation, and real-world stone transition | `CloverCycleCoordinator` (`Context → Direction → Execution → Outcome → Growth`) |
| [`site/website-interaction-controllers.js`](../../site/website-interaction-controllers.js) | `class_site_navigation_drawer_controller` | Accessible responsive header menu drawer and keyboard `Escape` handling | `TaskDirection` (`Direction`) |
| [`site/website-interaction-controllers.js`](../../site/website-interaction-controllers.js) | `class_accessible_tab_and_glossary_controller` | Accessible ARIA tabs (`getting-started/`) and live glossary term filtering (`glossary/`) | `VerifiedKnowledgeGraph` (`Context`) |
| [`site/clover-ai/index.html`](../../site/clover-ai/index.html) | `class_browser_conversation_session_store` | Multi-session browser storage, session lifecycle, and prior-turn history extraction | `ConversationSession` (`Context` & `Growth`) |
| [`site/clover-ai/index.html`](../../site/clover-ai/index.html) | `class_multimodal_attachment_and_speech_codec` | Image/video contact-sheet/text/binary attachment ingestion, Web Speech STT, and neural/browser TTS | `ModalityCodec` (`Execution`) |
| [`site/clover-ai/index.html`](../../site/clover-ai/index.html) | `class_markdown_and_media_reply_renderer` | Code-block, Markdown table, SVG diagram, PNG chart, and URL linkification renderer | `ModalityCodec` / `VerificationVerdict` (`Execution → Outcome`) |
| [`site/clover-ai/index.html`](../../site/clover-ai/index.html) | `class_clover_interactive_studio_coordinator` | Coordinates session selection, backend health probing, `/api/chat` execution, and reply regeneration | `CloverCycleCoordinator` (`Context → Direction → Execution → Outcome → Growth`) |
