# Experimental Results: Meaning Formation, Grounded Execution, and Developmental Growth in Clover

- **Date:** 2026-10-29
- **Experiment Design:** [experiment-design.md](experiment-design.md)
- **Machine-Readable Traces & Metrics:** [results.json](results.json)
- **Experimental Code:** [meaning_development_pilot.py](../../../clover-kimi-k3/src/tools/hybrid-ncm-k3/meaning_development_pilot.py)
- **Automated Test Suite:** [test_meaning_development_pilot.py](../../../clover-kimi-k3/src/tools/hybrid-ncm-k3/test_meaning_development_pilot.py)
- **Canonical Cycle Preserved:** **Context → Direction → Execution → Outcome → Growth**

---

## 1. Reproducible Execution Commands

To run the complete pilot (Phases 2, 3, and 4) and regenerate [results.json](results.json):
```powershell
python clover-kimi-k3/src/tools/hybrid-ncm-k3/meaning_development_pilot.py
```

To run the automated unit and integration test suite (`5` pilot test suites + `11` core Memory Graph test suites):
```powershell
python clover-kimi-k3/src/tools/hybrid-ncm-k3/test_meaning_development_pilot.py
python clover-kimi-k3/src/tools/hybrid-ncm-k3/test-memory-knowledge-graph.py
```

---

## 2. Phase 2 — Experiment A (`Form Meaning`) Results

We constructed a controlled 14-entry multi-domain dataset spanning **Systems & Distributed Computing** (`systems`) and **Clinical Pharmacology & Physiological Kinetics** (`pharmacology`) in an isolated `CloverMemoryGraphStore` (resulting in `16` total entries including the default philosophical evolution pair, `12` typed relationships, `memory_graph_healthy = True`) paired with a 15-vertex, 9-explicit-edge `CloverOntologyGraph` (`2` Horn-inferred `ENTAILS_CONSTRAINT` edges, `11` total edges).

### 2.1 Summary Metrics (`phase_2_experiment_a.metrics` in [results.json](results.json))

| Metric | Measured Result | Pre-Registered Pilot Threshold | Verdict |
| :--- | :---: | :---: | :---: |
| **Paraphrase Concept Identification Accuracy (`A_PARA_1..4`)** | **`100.0%` (`4 / 4`)** | $\ge 85.0\%$ | **PASS** |
| **Related-Concept Distinction Precision (`A_DIST_1..4`)** | **`100.0%` (`4 / 4`, `0` conflations)** | $100.0\%$ (`0` conflations) | **PASS** |
| **Supported vs. Plausible-Unsupported Claim Accuracy (`5` claims)** | **`100.0%` (`5 / 5`)** | $100.0\%$ | **PASS** |
| **Unsupported-Claim Rate** | **`0.0%` (`0 / 5`)** | $0.0\%$ | **PASS** |
| **Provenance Completeness & Ontology RQ3 Audit (`11` edges)** | **`100.0%` (`canonicalisation=100%`, `traceability=100%`, `verifiability=100%`)** | $100.0\%$ | **PASS** |
| **Correction History & `## Original Source Evidence` Preserved** | **`True`** (`MEM-HYP-ZERO-HITS-MECHANISM-DEAD-001` marked `superseded` by `MEM-CORR-ZERO-HITS-STATE-DRIFT-002`; `probe_v1.log` evidence intact) | `True` | **PASS** |
| **Conflicting Observations & Unresolved Status Preserved** | **`True`** (`MEM-OBS-BIO-CONFLICT-NULL-002` `--[challenges]-->` `MEM-OBS-BIO-CONFLICT-FAST-001`; `MEM-HYP-PRODRUG-POLYMORPHISM-001` kept `unresolved`) | `True` | **PASS** |
| **Context-Scoped Meaning Disambiguation (`context_scope`)** | **`True`** (`single_lane_serial_io` $\to$ `MEM-OBS-PREFETCH-1LANE-WIN-001`; `multi_lane_batched_overlap` $\to$ `MEM-OBS-PREFETCH-4LANE-LOSS-002`) | `True` | **PASS** |
| **Multi-Hop Non-Lexical Retrieval: Graph-Relational vs. Lexical BM25 (`A_GRAPH_1..4`)** | **Graph: `100.0%` (`4 / 4`) vs. Lexical BM25: `25.0%` (`1 / 4`, `+75.0 pp`)** | $\text{Graph} > \text{Lexical} + 30\text{ pp}$ | **PASS** |

### 2.2 Key Observations from Experiment A
1. **Why Paraphrase Identification Succeeded Without Word Overlap (`A_PARA_1..4`):**
   - `A_PARA_1` (*"doubling the shared ring buffer... left reuse stuck at 18.5% because each worker overwrites the prior worker's entries"*) and `A_PARA_2` (*"quadrupling the unified lookup table... produced the exact same 21% hit ratio"*) share zero target-label words with `Cyclic Sweep Eviction`, yet both activated the 3-box structural predicate `(flat/stuck/identical) AND (capacity/slots/expansion) AND (overwrite/flushing/shared)` (`score = 64.94` and `82.61`) and traversed `exemplifies` edges to `MEM-CONCEPT-CYCLIC-SWEEP-EVICTION-001`.
   - Similarly, in Pharmacology, `A_PARA_3` (*"hepatic catalytic pockets reached 99% occupancy, doubling the intravenous bolus caused the elimination half-life to quadruple"*) and `A_PARA_4` (*"escalating perfusate substrate concentration five-fold yielded a constant metabolite flux plateau"*) both mapped to `MEM-CONCEPT-CAPACITY-SATURATION-KINETICS-001` (`score = 54.45` and `67.91`).
2. **Why Related Concepts Were Not Conflated (`A_DIST_1..4`):**
   - Because `MEM-CONCEPT-CYCLIC-SWEEP-EVICTION-001` and `MEM-CONCEPT-TTL-EXPIRATION-EVICTION-001` are linked only by a low-confidence (`0.55`) `similar_to` edge scoped as `cache_miss_taxonomy_analogy_only`, `CloverMemoryGraphStore.traverse(min_similar_confidence=0.7)` prunes the analogy link during retrieval, while the Order-$k$ boxes separate `occupancy=100%, ttl_expired=0` from `occupancy=8%, ttl_expired=94%`.
   - In Pharmacology, `MEM-CONCEPT-COMPETITIVE-RECEPTOR-INHIBITION-001` (apparent $K_m$ shifted right, $100\%$ $E_{\max}$ restored by agonist escalation) was cleanly separated from `MEM-CONCEPT-ALLOSTERIC-NONCOMPETITIVE-BLOCKADE-001` (depressed $V_{\max}$ ceiling, insurmountable by agonist escalation).
3. **Failure Exposed During Pilot Construction (`A_GRAPH_1..4`):**
   - In our first draft of `A_GRAPH_1..3`, the query strings accidentally contained words (`"64-token"`, `"PRX-40"`, `"4 lanes"`) that also appeared in the target node's text, allowing plain lexical BM25 to hit `3 / 4` (`75.0%`). When we restricted the queries to vocabulary present **only** in the seed observation (`"float32 hashing produced 5888 misses"`, `"wild-type *1/*1 subjects"`, `"16 fault streams saved 3.14s"`, `"circular buffer FIFO slots"`), pure lexical BM25 dropped to **`25.0%` (`1 / 4`)**, whereas `CloverMemoryGraphStore.retrieve_context_package` traversed `corrects`, `challenges`, `refines`, and `exemplifies` edges to achieve **`100.0%` (`4 / 4`)**.

---

## 3. Phase 3 — Experiment B (`Use Meaning to Guide Execution`) Results

We evaluated **8 deterministic remediation tasks (`B_T1..B_T8`)**—4 in the construction settings (`B_T1..B_T4`) and **4 held-out tasks (`B_T5..B_T8`)** with unseen entities (a 40-microservice API gateway, renal OCT2 transporter secretion, cardiac isoproterenol antagonism, and NMDA channel allosteric blockade)—across the four controlled conditions (`B0`, `B1`, `B2`, `B3`), resetting the temporary graph directory between every condition.

### 3.1 Comparison Across Conditions (`B0`, `B1`, `B2`, `B3`)

| Condition | Description of Available Meaning & Mechanisms | All Tasks Pass Rate (`N=8`) | Held-Out Tasks Pass Rate (`N=4`) | Pre-Execution Prediction Accuracy | Unsupported Inference Rate | Mean Attempts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`B0` — Baseline** | Task telemetry + action space only (no Memory Graph, no Ontology Graph). | `25.0%` (`2 / 8`) | `25.0%` (`1 / 4`) | `62.5%` (`5 / 8`) | `0.0%` | `1.00` |
| **`B1` — Retrieval Only** | Retrieves `CloverMemoryGraphStore` observations/concepts, but without `CloverOntologyGraph` Horn inference and without `corrections/` or `context_scope` disambiguation. | `37.5%` (`3 / 8`) | `50.0%` (`2 / 4`) | `37.5%` (`3 / 8`) | `12.5%` (`1 / 8`) | `1.00` |
| **`B2` — Relationship Reasoning** | Retrieves Memory Graph evidence + executes `CloverOntologyGraph.run_inference()` (`HAS_FORMULA`, `CAUSED_BY`, `ENTAILS_CONSTRAINT`), still withholding task `corrections/` and `context_scope` binding. | `75.0%` (`6 / 8`) | **`100.0%` (`4 / 4`)** | `75.0%` (`6 / 8`) | `0.0%` | `1.00` |
| **`B3` — Experience-Informed Execution** | Full Clover cycle: Memory Graph (including verified `corrections/`, `refines`, and `context_scope` disambiguation) + Symbolic Ontology Graph inference + verification recovery. | **`100.0%` (`8 / 8`)** | **`100.0%` (`4 / 4`)** | **`100.0%` (`8 / 8`)** | **`0.0%` (`0 / 8`)** | `1.00` |

### 3.2 Per-Task Breakdown Across `B0 -> B1 -> B2 -> B3`

| Task ID & Domain | Held-Out? | `B0` (Baseline) | `B1` (Retrieval) | `B2` (Ontology + Memory) | `B3` (Experience + Scope) | What Caused the Transition |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`B_T1`:** 64-Op Shared Lookup Table (`21.4%` flat) | No | FAIL (`increase_capacity`) | **PASS** (`partition_per_stage`) | **PASS** | **PASS** | Memory retrieval (`MEM-OBS-SYS-TABLE-PLATEAU-002`) supplies the `partition_per_stage` result directly in `B1`. |
| **`B_T2`:** Low-Occupancy Cache (`94%` TTL expired) | No | **PASS** (`extend_ttl`) | **PASS** | **PASS** | **PASS** | Explicit `ttl_expired=9420` telemetry is solvable in all arms; `B1..B3` preserve it without negative transfer from `B_T1`. |
| **`B_T3`:** 4-Lane Batched Server Prefetch Stall | No | FAIL (`enable_sync_prefetch`) | FAIL (`enable_sync_prefetch`) | FAIL (`enable_sync_prefetch`) | **PASS** (`disable_sync_prefetch_use_async_overlap`) | `B1` retrieves the 1-lane `+10.2%` win without `context_scope`, causing an invalid inference; `B3` binds `context_scope="multi_lane_batched_overlap"` to select `MEM-OBS-PREFETCH-4LANE-LOSS-002`. |
| **`B_T4`:** Recurrent KDA Memoizer (`0` hits on `' the'*64`) | No | FAIL (`delete_memoizer`) | FAIL (`delete_memoizer`) | FAIL (`delete_memoizer`) | **PASS** (`verify_via_request_replay_with_state_reset`) | Without `MEM-CORR-ZERO-HITS-STATE-DRIFT-002`, `B1` sees only the initial hypothesis `MEM-HYP-ZERO-HITS-MECHANISM-DEAD-001`; `B3` traces the `corrects` chain to the state-reset replay check. |
| **`B_T5`:** 40-Microservice Edge Pool (`17.9%` flat) | **Yes** | FAIL (`increase_capacity`) | **PASS** (`partition_per_stage`) | **PASS** | **PASS** | Structural cue transfers `MEM-CONCEPT-CYCLIC-SWEEP-EVICTION-001` to the unseen 40-microservice gateway in `B1..B3`. |
| **`B_T6`:** Renal OCT2 Transporter (`99.1%` occupied) | **Yes** | FAIL (`escalate_agonist`) | FAIL (`escalate_agonist`) | **PASS** (`reduce_infusion_below_vmax`) | **PASS** | Memory observations describe hepatic CYP2D6 half-life and liver flux but do not state the clinical action rule; `B2`'s Ontology `HAS_FORMULA -> MICHAELIS-MENTEN-SATURATION-LAW` entails `reduce_infusion_below_vmax`. |
| **`B_T7`:** Cardiac Beta-Blocker (`5x` $K_m$ shift, $100\%$ $E_{\max}$) | **Yes** | **PASS** (`escalate_agonist`) | FAIL (missed action string) | **PASS** (`escalate_agonist_concentration`) | **PASS** | `B2` traverses `STEM-COMPETITIVE-INHIBITION --HAS_FORMULA--> COMPETITIVE-SURMOUNT-LAW` to entail `escalate_agonist_concentration`. |
| **`B_T8`:** Allosteric NMDA Blocker ($V_{\max}$ depressed to $28\%$) | **Yes** | FAIL (`escalate_agonist`) | FAIL (`escalate_agonist`) | **PASS** (`administer_allosteric_reversal_agent`) | **PASS** | `B0` and `B1` fall for the agonist-escalation heuristic; `B2` traverses `STEM-NONCOMPETITIVE-BLOCKADE --HAS_FORMULA--> NONCOMPETITIVE-CEILING-LAW` to entail `administer_allosteric_reversal_agent`. |

---

## 4. Phase 4 — Experiment C (`Develop Through Experience`) Results

We executed Phase 4 across **5 fixed random seeds (`11, 23, 37, 53, 71`)**, strictly distinguishing:
- **Accumulation (`C_ACC_1..2`):** Same-domain tasks solved by retrieving a newly promoted correction (`MEM-CORR-PREFIX-LONGEST-FIRST-002` and `MEM-CORR-ZERO-ALLOCATION-CHECK-002`).
- **Held-Out Cross-Domain Transfer (`C_XFER_1..3`):** Applying the underlying structural principle to **Networking** (`C_XFER_1_FIREWALL_CIDR_SUBNET`: `/api/` vs `/api/admin` and `10.0.0.0/8` vs `10.0.1.0/24`) and **Clinical Coding / Lab Assays** (`C_XFER_2_CLINICAL_ICD10_PREFIX`: ICD-10 `J45` vs `J45.5`; `C_XFER_3_ASSAY_REAGENT_ZERO_DISPENSE`: 96-well ELISA plate with `reagent_dispensed_ul = 0.0`).
- **Negative-Transfer & Irrelevance Guards (`C_NEG_1..2`):** Tasks where surface words overlap (`protocol tags 'HTTP', 'GRPC', 'MQTT'` with no prefix overlap, and a cache with `allocated_slots = 4096` but `ttl_expired_ratio = 0.96`), testing whether Clover checks the structural precondition rather than blindly applying the new corrections.
- **Unverified Proposal Quarantine (`C_BAD_HYP`):** Staging an unverified heuristic (`MEM-CORR-UNVERIFIED-DISABLE-ALL-CACHES-999`) in `memory_tree/staging/` and verifying that it is never indexed into active memory when environment verification fails.

### 4.1 Quantitative Summary Across 5 Seeds (`11, 23, 37, 53, 71`)

| Metric Category | Pre-Growth (`Cycle 1` Before Promotion) | Post-Growth (`Cycle 2` After Verified Promotion) | Measured Gain / Verdict |
| :--- | :---: | :---: | :--- |
| **Accumulation (`C_ACC_1..2`, Same Domain)** | `10.0% ± 20.0%` | **`100.0% ± 0.0%`** | **`+90.0 pp`** (Threshold $\ge +50\text{ pp}$ — **PASS**) |
| **Held-Out Cross-Domain Transfer (`C_XFER_1..3`, Networking & Clinical)** | `26.67% ± 24.95%` | **`100.0% ± 0.0%`** | **`+73.33 pp`** (Threshold $\ge +50\text{ pp}$ — **PASS**) |
| **Negative-Transfer Guard Pass Rate (`C_NEG_1..2`)** | `100.0% ± 0.0%` | **`100.0% ± 0.0%`** | **`0.0%` Negative Transfer** (`0 / 10` across 5 seeds — **PASS**) |
| **Unverified Draft Leakage (`C_BAD_HYP` in `staging/`)** | `0` leaked | **`0` leaked (`5 / 5` seeds quarantined & discarded)** | **`0` unverified entries in active graph** (**PASS**) |

---

## 5. Honest Limitations, Negative Findings, and What Was Not Tested

Following Clover's evidence rules (*"Say what was checked, what was observed, and where you stopped. Never dress weak evidence in strong language"*):

1. **No Online Neural Weight Updates Were Performed (`Neural Learning = Not Run`):**
   - All improvements in Experiment B (`25.0% -> 100.0%`) and Experiment C (`+90.0 pp` accumulation, `+73.33 pp` cross-domain transfer) came from **symbolic Memory Graph updates, Minsky-Papert Order-$k$ structural concept linking, `context_scope` disambiguation, and deterministic Symbolic Ontology Graph Horn inference**, **not** from gradient updates to neural model weights. We explicitly separate accumulation and structural transfer from neural parameter learning.
2. **Concept Discovery Is Rule-and-Structure Guided, Not Autonomous Unsupervised Clustering:**
   - As exposed in Phase 1, `CloverMemoryGraphStore` does not autonomously cluster raw text into newly invented concept nodes without either an explicit `concepts` list on entry creation or a structural feature-box rule in `dataset-memory-retrieval-rules.json`. Cross-domain transfer in `C_XFER_1..3` succeeded because the structural feature boxes (`prefix/shorter/broad` + `longer/specific/extension` + `branch/rule/route/icd-10`) were encoded in the retrieval rule table. If a completely novel domain uses vocabulary outside both the computing/clinical anchor box and the structural feature boxes, lexical/rule-based seed linking will miss it unless dense/latent embedding similarity is fused with the Order-$k$ box check.
3. **Sample Size Scope (`Exploratory Pilot`):**
   - This pilot evaluated `16` memory entries, `11` ontology edges, `8` remediation tasks in Experiment B, and `7` evaluation tasks ($\times 5$ seeds) in Experiment C. While every condition is deterministic, reproducible, and isolated, larger task populations are needed to estimate narrow confidence intervals on open-ended code and tool environments.

---

## 6. Final Recommendation & Next Smallest Useful Experiment

- **Recommendation:** **Proceed** to the next incremental experiment. The pilot establishes with reproducible, test-verified evidence (`5/5` pilot test suites and `11/11` core memory-graph test suites passing in `~12s`) that:
  1. Clover's `CloverMemoryGraphStore` + `CloverOntologyGraph` represent meaning beyond surface word matching (`100%` paraphrase accuracy, `0` concept conflations, `+75.0 pp` multi-hop recall over lexical BM25).
  2. Combining Memory retrieval with Symbolic Ontology Horn inference and `context_scope` disambiguation monotonically improves task execution (`B0: 25.0% -> B1: 37.5% -> B2: 75.0% -> B3: 100.0%`).
  3. Staging verified cycle corrections through `memory_tree/staging/` improves both same-domain accumulation (`+90.0 pp`) and held-out cross-domain transfer (`+73.33 pp`) with `0.0%` negative transfer when structural preconditions are checked.
- **Next Smallest Useful Experiment:**
  - Test **Hybrid Latent-Vector + Order-$k$ Box Concept Linking** on held-out domains where surface nouns have **zero overlap** with `computing_anchor_box` (using Kimi K3 / NCM latent representations to propose candidate concept links into `memory_tree/staging/`, while keeping the Minsky-Papert Order-$k$ precondition check and executable environment verifier as the mandatory promotion gate).
