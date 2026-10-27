# Observe — Clover's Next Experiment

**Experiment name:** Observe  
**Base:** Clover AI v4, the current seven-layer Clover AI system (frozen release baseline)  
**Status:** Milestone 1–3 (`Experiment A — Reciprocal change`) implemented and verified on GPU (`run_observe_experiment_a.py`, [`results/experiment_a_reciprocal_change_results.json`](results/experiment_a_reciprocal_change_results.json)); Milestones 4–8 (multi-host distribution, replication, and coordination) remain proposed future phases  
**Primary objective:** Prove that reciprocal observation can change participating systems and that the resulting experience can be preserved and reused.  
**Initial performance constraint:** None. Correctness and demonstrable operation come first; latency is measured, not used as an initial pass/fail threshold.

## 1. Why Observe

Clover AI v4 is the current seven-layer system. Observe is the next experiment, not an eighth layer of v4 and not a claim that a distributed network already exists.

The guiding hypothesis is:

> Every interaction is an opportunity for reciprocal change; every meaningful change can become experience; and experience can improve future interaction.

The experiment takes the observer/observed relationship as dynamic. Any participating system may observe another, be observed by it, contribute to the interaction, and be changed by the result. A participant is not permanently assigned to only one role.

This is an engineering hypothesis about observable system state and learning, not a claim that software has consciousness.

## 2. Core hypothesis

When two or more participating systems interact, record and expose enough of the interaction for each side to:
- observe relevant information, actions and outcomes from the other side, within explicit permissions;
- form and record an interpretation with evidence and uncertainty;
- receive feedback or a response that may change its own memory, beliefs, policy, or future behavior;
- observe and verify the changes that occurred on both sides;
- preserve useful experience so it can improve a later interaction.

The shared network may make verified learnings available to other authorized participants. It must not silently replace each participant's perspective with a single global interpretation.

## 3. Distributed access goal

Any compatible internet-connected system should be able to reach Clover's public capabilities without needing to host the entire Clover implementation. Intended clients include:
- mobile devices and browsers;
- laptops and workstations;
- human-operated applications;
- independent AI agents through documented, typed interfaces;
- participating peers contributing memory, storage, model components or compute.

Access and resource placement must remain separate. A stable public interface should resolve the memory, model and compute resources needed for a request, regardless of which peer currently hosts them.

The initial access interface should distinguish:
1. **Knowledge:** retrieve learning objects, relationships, evidence and provenance.
2. **Execution:** submit a task and receive a result plus verification information.
3. **Discovery:** find available peers, capabilities, resource identifiers and trust metadata.

Adapters such as HTTPS APIs, OpenAPI/SDKs and MCP may be used as client surfaces. They are access mechanisms, not the experiment's core hypothesis.

## 4. Memory and experience model

Treat memory records as versioned, addressable objects with stable identifiers and provenance. Relationships between records form a graph index that can be rebuilt from canonical records.

An interaction record should capture, where permitted:
- participant identities or pseudonymous identifiers;
- the question, task, or observation that initiated the interaction;
- relevant evidence and source references;
- each participant's interpretation and epistemic status;
- the before/after state changes each participant reports or exposes;
- feedback, actions and outcomes;
- verification method and result;
- what can be shared, with whom, and under which permissions.

Keep observations, externally verified facts, hypotheses, participant-stated beliefs and unresolved claims distinct. A stored claim does not become true merely because it is stored, replicated or popular.

Experience is more than a transcript. The experiment should test whether explicit before/after changes and verified outcomes improve performance on subsequent held-out interactions.

## 5. Distributed resources

Observe should ultimately investigate distribution of:
- memory objects and relational graph data;
- datasets and supporting evidence;
- model trunks, individual experts and other model artifacts;
- compute workers using CPU, GPU or other suitable resources;
- verification work and audit records.

For the first proof, do not require distribution of the entire Kimi K3 model. Start with small memory objects and a known computation across multiple independent hosts, then increase fragmentation in later experiments. The existing seven-layer Clover AI remains the baseline.

Use content hashes and signed manifests to establish resource identity. Replicate resources when availability requires it. Keep large data and compute off-chain; blockchain may coordinate identities, commitments, job receipts, reputation and payments. Blockchain records can provide tamper-evident coordination, but do not alone prove that a worker performed a neural-network computation correctly. Test independent verification, redundant execution or challenge-based checks separately.

## 6. Principles and boundaries

- **Correctness before performance:** no initial latency deadline. Measure latency, bandwidth, cost and energy without using them as first-phase pass/fail criteria.
- **Prove operation before scale claims:** a small working network is evidence for that configuration, not proof that millions of peers will work.
- **Reciprocity:** every participant can be an observer, observed participant, and actor during an interaction.
- **Change must be observable:** distinguish intended changes from changes verified in actual state.
- **Evidence before promotion:** new claims remain hypotheses until supported; retain failures and contradictory results.
- **No forced consensus:** participants can preserve different interpretations and disagreement.
- **Least privilege:** public reachability does not grant access to private memory, credentials, administrative tools or unrestricted compute.
- **Privacy by design:** do not publish private prompts, memories, or activations by default. Define consent, encryption, access controls and retention rules.
- **Fault tolerance:** peers may disconnect, return malformed data or act maliciously. Test these cases explicitly.
- **Human accountability:** system changes and external actions remain bounded by explicit authorization and policy.
- **No consciousness claim:** reciprocal observation and state change are testable system behaviors, not proof of subjective experience.

## 7. Initial proof of concept

Build the smallest reproducible experiment that demonstrates the core mechanism.

### Experiment A — Reciprocal change

1. Create two independently stateful participants with inspectable memory and a simple shared task.
2. Capture each participant's relevant initial state.
3. Let participant A observe B and produce an evidence-linked interpretation.
4. Let B observe A's interpretation, respond or challenge it, and record its own change.
5. Let A incorporate the response and record its own change.
6. Run a later held-out task where the recorded experience could help.
7. Compare against a control where the interaction is merely logged and not used to update future behavior.

**Success criterion:** both participants exhibit a verifiable, relevant state change caused by the interaction, and the later task can establish whether that change helped, harmed or made no difference.

### Experiment B — Distributed knowledge access

1. Store a known learning object and its relationships across at least two independent hosts.
2. Retrieve it from a browser/API client and an independent agent.
3. Verify the content hash, provenance and version.
4. make one host unavailable and retrieve from a replica.
5. Submit a new contribution from a peer and keep its trust status explicit.

**Success criterion:** the same authorized learning can be retrieved and verified without depending on one original host.

### Experiment C — Distributed execution

1. Define a known task with deterministic expected output.
2. Divide the task into independently executable work units.
3. Execute across multiple hosts and combine results.
4. Verify against the local reference result.
5. Disconnect a worker and corrupt one returned result to test recovery and detection.

**Success criterion:** the task completes across multiple machines, the output passes the defined correctness check, and at least one worker failure and one invalid result are detected or recovered from.

## 8. Evaluation

Do not impose a time budget in the initial phase. Record:
- correctness and verified task completion;
- state changes on each participant;
- whether changes persist and influence a held-out follow-up task;
- evidence completeness and provenance;
- false, corrupted or contradictory contributions;
- recovery after peer failure;
- retrieval consistency across replicas;
- total elapsed time, bandwidth and resource consumption for later optimization.

Report null or negative effects honestly. A change to state is not automatically learning; it must be evaluated for its effect on future outcomes.

## 9. Milestones

1. **Specification:** define participant state, observation records, change records, permissions and verification contract.
2. **Local reciprocal prototype:** two participants, one task, before/after state and replayable evidence.
3. **Held-out evaluation:** compare reciprocal learning against logging-only and no-memory controls.
4. **Multi-host memory:** content-addressed records, replicas, retrieval and host failure.
5. **Multi-host compute:** deterministic split task, result verification, failure and corruption tests.
6. **Public access surface:** documented endpoint and independent-agent integration.
7. **Trust and coordination:** evaluate signed manifests, blockchain-backed commitments and audit records only after core behavior works.
8. **Scale experiments:** increase the number of peers and decrease per-peer resources; report the limits found.

## 10. Required deliverables

- a runnable, reproducible proof of concept;
- protocol and record schemas;
- explicit trust and permission model;
- experiment configuration and baseline controls;
- raw outputs and verification evidence;
- failures, limitations and unresolved questions;
- a clear comparison between what was implemented, what was observed, and what remains hypothetical.

## 11. Definition of the first success

The first success is not speed, scale, a token, or a claim of general intelligence.

It is a reproducible demonstration that:
1. two participants can observe and change one another through an interaction;
2. the changes and evidence can be inspected;
3. experience can affect a later outcome and be evaluated;
4. knowledge can be retrieved across independent hosts;
5. distributed execution can produce a verified result despite a tested failure.

Only after this proof should the experiment expand toward internet-wide participation and fragmented model execution.

---

## 12. Measured Results — Experiment A (Reciprocal Observation, Mutual State Change & Held-Out Evaluation)

We implemented and executed **Experiment A** (Milestones 1–3) in [`run_observe_experiment_a.py`](run_observe_experiment_a.py) (with full raw state snapshots, SHA-256 digests, observation records, and live 93-layer Clover AI v4 server responses saved in [`results/experiment_a_reciprocal_change_results.json`](results/experiment_a_reciprocal_change_results.json)) while keeping **Clover AI v4 completely frozen**.

### 12.1 Two Independently Stateful Participants & Initial Asymmetry (`t = 0`)
- **`participant_A` (Pipeline Capacity Analyst):**
  - Holds `MEM-OBS-A-PIPELINE-FLAT-001` (`observed`: sweeping a shared global LRU pool from `512` to `2,048` slots across a 32-stage media transcode pipeline stays flat at `21.0%` hit rate) and `MEM-OBS-A-BURST-SKEW-002` (`observed`: telemetry shows 4 reference-frame stages `3, 11, 19, 27` require `256` slots each while 28 pass-through stages require `36` slots each; `4×256 + 28×36 = 2,032 <= 2,048` slots).
  - Holds a flawed active hypothesis `MEM-HYP-A-NEED-8192-SLOTS-001` (`hypothesis`, `status=active`): concludes from the flat `21.0%` curve that `2,048` slots is too small and recommends quadrupling global memory to `8,192` slots.
  - Also holds a restricted internal entry `MEM-PRIV-A-INTERNAL-TOKEN-001` (`scope:private_internal`) to test least-privilege observation boundaries.
- **`participant_B` (Stage-Isolation & Eviction Analyst):**
  - Holds `MEM-OBS-B-CYCLIC-EVICTION-001` (`observed`: partitioning the same `2,048`-slot budget into `64` slots/stage eliminates cross-stage cyclic-sweep eviction and raises hit rate from `21.0%` to `49.5%`).
  - Holds a flawed active hypothesis `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001` (`hypothesis`, `status=active`): assumes equal-sized `64`-slot quotas (`2,048 / 32`) are optimal for all 32 stages.

### 12.2 Three-Round Reciprocal Observation & Verified State Changes
1. **Round 1 (`OBS-REC-001-A-OBSERVES-B`):**
   - `participant_A` observes `participant_B`'s permitted `scope:public_systems` entries (`MEM-OBS-B-CYCLIC-EVICTION-001`, `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001`), promotes `MEM-CORR-A-STAGE-PARTITION-001` (`externally_verified`), and marks `MEM-HYP-A-NEED-8192-SLOTS-001` as `superseded`.
   - Simultaneously, `participant_A` challenges `participant_B`'s uniform `64`-slot hypothesis using `MEM-OBS-A-BURST-SKEW-002` (under `64` slots/stage, the 4 reference-frame stages thrash at `18.2%` while 28 pass-through stages leave `28` slots/stage idle).
2. **Round 2 (`OBS-REC-002-B-OBSERVES-A`):**
   - `participant_B` observes `participant_A`'s permitted `scope:public_systems` entries (`MEM-PRIV-A-INTERNAL-TOKEN-001` is strictly withheld by the permission filter), verifies that allocating `256` slots to the 4 reference-frame stages and `36` slots to the 28 pass-through stages (`2,032 <= 2,048` slots) lifts hit rate from `49.5%` to `68.4%`, promotes `MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001`, and marks `MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001` as `superseded`.
3. **Round 3 (`OBS-REC-003-A-SYNTHESIZES-B`):**
   - `participant_A` observes `participant_B`'s promoted correction and records `MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001` (`externally_verified`, capturing all three regimes: `21.0%` global $\to$ `49.5%` uniform `64`-slot $\to$ `68.4%` weighted `256/36`-slot).

| Participant | State SHA-256 Before (`t = 0`) | State SHA-256 After Reciprocal Change | Active Entries (`Before -> After`) | Superseded Hypotheses (`Before -> After`) |
| :--- | :--- | :--- | :--- | :--- |
| **`participant_A`** | `6562c63a2bb69631...` | `a26b46c70f532a97...` | `4 -> 5` (`+MEM-CORR-A-STAGE-PARTITION-001`, `+MEM-OUT-A-WEIGHTED-PARTITION-VERIFIED-001`) | `[] -> ['MEM-HYP-A-NEED-8192-SLOTS-001']` |
| **`participant_B`** | `c32fe88254041428...` | `da048eca3b04a3a5...` | `2 -> 2` (`+MEM-CORR-B-WEIGHTED-STAGE-QUOTA-001`) | `[] -> ['MEM-HYP-B-UNIFORM-QUOTA-ALWAYS-001']` |

### 12.3 Held-Out Task Evaluation (`N = 6`) & Live 93-Layer Clover AI v4 Server Verification

We evaluated `participant_A` and `participant_B` across 6 held-out tasks (4 held-out diagnosis & cross-domain transfer tasks, 1 out-of-domain distractor control, and 1 privacy-boundary audit) across **three controlled arms**:
- **Control 1 (`No Interaction`):** Initial isolated state (`t = 0`).
- **Control 2 (`Interaction Logged Only`):** The 3-round interaction is logged and candidates are written to `staging/` (`status='under_review'`), but **not** promoted into active canonical memory.
- **Treatment (`Reciprocal State Change`):** Both participants promote verified corrections and mark refuted hypotheses `superseded`.

| Evaluation Arm | Held-Out Reasoning & Transfer (`Tasks 1–4`) | Distractor & Privacy Controls (`Tasks 5–6`) | Total Held-Out Score (`N=6`) | Live 93-Layer Clover AI v4 Server E2E (`Tasks 1 & 2`) |
| :--- | :---: | :---: | :---: | :---: |
| **Control 1: No Interaction (`t = 0`)** | `0 / 4` (`0.0%`) | `2 / 2` (`100.0%`) | **`2 / 6` (`33.33%`)** | `0 / 2` (`False` on A, `False` on B) |
| **Control 2: Interaction Logged Only (`staging/` only)** | `0 / 4` (`0.0%`) | `2 / 2` (`100.0%`) | **`2 / 6` (`33.33%`)** | `0 / 2` (unrefuted hypotheses remain `active`) |
| **Treatment: Reciprocal State Change (Promoted)** | **`4 / 4` (`100.0%`)** | **`2 / 2` (`100.0%`)** | **`6 / 6` (`100.00%`)** | **`2 / 2` (`True` on A in `27.9 s`, `True` on B in `27.5 s`)** |

**What remains unbuilt / hypothetical (Milestones 4–8):**
- Experiment A proves local reciprocal observation, mutual state change, least-privilege filtering (`0` private leaks), and held-out transfer on a single host (`gpu`). Multi-host content-addressed memory replication (`Experiment B`), multi-host distributed execution with worker-failure/corruption recovery (`Experiment C`), and cryptographic/blockchain coordination have **not** been implemented yet and remain future milestones.
