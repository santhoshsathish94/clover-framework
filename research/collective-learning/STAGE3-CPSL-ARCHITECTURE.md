# Stage 3 — Collective Problem-Solving Learning (CPSL) Architecture & Protocol

**Date:** 2026-11-05  
**Code Package:** [`cpsl/`](cpsl/) (`collective.py`, `collective_gateway.py`, `collective_evaluator.py`, `core_registry.py`, `core_sync.py`, `strategies.py`, `run_cpsl_experiment.py`)  
**Unit Test Suite:** [`tests/test_cpsl.py`](tests/test_cpsl.py) (`4/4` pass locally and on `k3`)

---

## 1. Core Architectural Principle

> **Clover must learn from people without learning to become those people.**  
> It should discover what is generally useful from their experiences, while preserving what is personally theirs.

Clover therefore separates learning into two distinct systems connected only through a controlled, privacy-filtered, and independently validated boundary:
1. **Personal Learning (`personal.py` / PPL):** What Clover learns about one individual—their personal shorthand habits, response-style preferences (`format`, `length`, `confirm`, `plan`, `options`, `sources`), recurring routines (`"the usual"`), and open items. This stays strictly on the user's device or private repository.
2. **Collective Learning (`cpsl/`):** General problem-solving decompositions, cross-domain tool-use patterns, and universal semantic relationships that hold across independent users and unseen problem phrasings.

---

## 2. Strict Boundary Between Personal and Collective Learning

Enforced in [`cpsl/collective.py`](cpsl/collective.py) (`is_allowed_collective_feature`, `scrub_experience_for_collective`, `LocalCollectiveExtractor`):

| Learning / Record Type | Exported to Collective Gateway? | How Boundary Is Enforced in Code |
|---|:---:|---|
| **Raw conversation text, prompts, or reactions** | **Never** | `scrub_experience_for_collective()` drops `text` and `reactions` before candidate synthesis |
| **Object names, file paths, URLs, emails, SSNs, or numbers** | **Never** | `PII_OR_SECRET_RE` + `load_closed_general_vocabulary()` (OEWN 2025 + CSI closed lemma inventory) rejects any non-general token *before* `Learner.observe` |
| **User's preferred response style (`format`, `length`, `confirm`, `plan`, `options`, `sources`)** | **Never** | `pres_feedback` and `accepted_presentation` are stripped; `LocalCollectiveExtractor` iterates only over task/strategy hypotheses |
| **User's personal routines (`"the usual"`, `"the Friday routine"`) & open items** | **Never** | `PERSONAL_ROUTINE_WORDS` filter + `objects=[]` scrubbing |
| **Explicit user statements (`"When I say X, I mean Y"`)** | **Never** | `h.origin == "statement"` is excluded (statements have single-turn support and are personal instructions) |
| **Unvalidated (`provisional` or `retired`) local hypotheses** | **Never** | Requires `h.status == "validated"` (`support >= 2`, `precision >= 0.70`) |
| **General semantic relationship (`tok=<oewn_lemma>, ctx=<domain> -> intent`)** | **Candidate** | Exported as `category="semantics"` if `consent_contribute == True` |
| **Reusable multi-step problem-solving strategy (`trigger -> (step_1, step_2, ...)` )** | **Candidate** | Exported as `category="problem-solving"` if `consent_contribute == True` |
| **Cross-domain reasoning or tool-use pattern (`tok=<oewn_lemma> -> intent`)** | **Candidate** | Exported as `category="reasoning-patterns"` or `"tool-use-patterns"` if `consent_contribute == True` |

### Why Pre-Extraction Closed-Vocabulary Scrubbing Matters (Stage 2 `M5` Lesson)
In Stage 2 (`M5`), we discovered that if a learner relies on **multi-example set intersection** (`h.trigger & tf`) to remove a private token (such as `tok=secret_project_acmecorp`), deleting the second experience causes `Learner.rebuild` to revert the trigger to the first experience alone—**unmasking the private token (`20/20 = 100%` leak rate)**.

In `cpsl/collective.py`, `scrub_experience_for_collective()` filters every experience's features against the **frozen Open English WordNet 2025 (`OEWN`) + CSI general vocabulary** *before* constructing collective candidates. Even if a user has only one remaining experience after deleting others, no private token can ever enter a candidate trigger (`0/20 = 0.0%` leak rate).

---

## 3. Six-Component Implementation & Promotion Lifecycle

```
Individual Clover Installations (Group A: 20 isolated instances)
  └── personal.py (Local Store + Learner: habits, preferences, routines stay local)
        │
        ▼  [Opt-in: consent_contribute == True + preview_candidates() inspection]
  └── collective.py (LocalCollectiveExtractor: OEWN/CSI closed-vocab scrub -> candidate JSON)
        │
        ▼
  └── collective_gateway.py (CollectiveGateway)
        • Schema validation (learning-event.schema.json)
        • Defense-in-depth PII & closed-vocabulary check
        • Installation deduplication (Sybil defense: N reports from 1 user = 1 installation vote)
        • Cross-user conflict tracking (detects when users assign different meanings to one word)
        │
        ▼
  └── collective_evaluator.py (CollectiveEvaluator — Quarantine & MAML Outer Loop)
        • Gate G1 (Independent Quorum): independent_installations >= 3
        • Gate G2 (Cross-User Consistency): cross_user_agreement >= 0.75
        • Gate G3 (Held-Out Generalisation): >= 0.80 accuracy & positive lift on unseen phrasings/objects
        • Gate G4 (Cross-Domain Transfer): verifies domain-general rules across domains
        • Gate G5 (Zero-Regression & Safety): 0 regressions on general requests & 0 personal hijacks
        │
        ▼
  └── core_registry.py (CoreRegistry — Git-backed clover-collective/ repository)
        • Writes promoted rules to core/{semantics,reasoning-patterns,problem-solving,tool-use-patterns}/
        • Writes evaluation evidence to evaluations/{regression,generalisation,cross-domain}/
        • Signs releases/manifest.json (HMAC-SHA256 + per-file SHA-256) and commits to Git
        • Supports withdraw_installation(): removes contributor, recomputes gates, demotes rules < quorum
        │
        ▼  [Opt-in: consent_receive == True + signature & SHA-256 verification]
  └── core_sync.py (CoreSync + CollectiveAssistant on Recipient Installations)
```

---

## 4. Two Independent Consent Controls (`whats-needed-for-clover.txt` §5)

1. **`consent_contribute` (Opt-in, default `False`):**
   - Controls whether `LocalCollectiveExtractor.export_candidates()` may emit candidate packages.
   - `preview_candidates()` returns the exact JSON payload beforehand so the user can inspect every field.
   - If the user later withdraws consent (`gateway.remove_installation(inst_id)`), their support is removed from all candidates and the next core release is recomputed from the remaining consenting installations.
2. **`consent_receive` (Opt-in, default `False`):**
   - Controls whether `CoreSync.verify_and_load()` downloads and applies promoted rules from `releases/manifest.json`.
   - Verifies both the release HMAC-SHA256 signature and every individual rule file's SHA-256 digest before loading; any tampering raises `CoreSyncError`.

---

## 5. Priority Hierarchy on Device (`CollectiveAssistant.respond`)

When a Clover installation has both **Personal Learning (`PPL`)** and **Collective Learning (`CPSL`)** enabled, how do they interact?

In [`cpsl/core_sync.py`](cpsl/core_sync.py), `CollectiveAssistant.respond()` resolves every request in strict order:
1. **Safety & Declined Capabilities (`semantics.py` / `AGENTS.md` §9):** Vault paths, risky actions requiring confirmation, and declined tools are never overridden.
2. **Validated Personal Understanding (`self.learner`, `personal.py`):** If this specific user has a validated personal habit (e.g., for User A, `"clean up {code}"` means `code.format`), their personal reading wins *for User A*.
3. **Signed Collective Core (`self.collective_rules`, `core_sync.py`):** If no personal habit overrides the message, any matching validated collective strategy (e.g., `"preflight {code}" -> (code.test, code.format)`, or `"bookmark {code}" -> notes.todo`) is applied.
4. **Frozen General Interpreter (`self.general`, `B0` + `CSI v0.1r3`):** The immutable baseline core handles all standard language requests.

This ordering guarantees that **collective learning never erases a person's own validated habits**, while giving every user immediate access to problem-solving strategies discovered by the wider community.
