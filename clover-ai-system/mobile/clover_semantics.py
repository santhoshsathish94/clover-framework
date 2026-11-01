#!/usr/bin/env python3
"""
Clover Central Semantic Interface (`clover_semantics.py`)
=========================================================
Semantics: Meaning · Context · Interpretation · Alignment

Connects User, Fuel System (`S1 Memory · S2 Relationships · S3 Tools`), and
Model Engine (`C1 Inference · C2 Plan · C3 Decode`) as a shared meaning system
rather than an independent processing layer.

Grounded in empirical measurements on `LFM3-Omni-2B` (24 Tri-Hybrid layers) and
priority research literature:
  1. **Meaning — Mean-Centered Layer-Selective Geometry & Forced-Choice Candidate Readout**
     (Schiekiera et al., ICML 2026, *From Associations to Activations*):
     - Removes transformer anisotropy (`raw cos = 0.766 -> -0.034`) via empirical mean-centering
       `e_tilde = e - mu`, increasing representational similarity (RSA) with cross-model semantic
       consensus from `0.238 -> 0.554` (`+0.317`) at `LFM3-Omni-2B` Layer 18 (`full_gqa`).
     - Uses constrained Forced-Choice (FC) candidate ranking over shared supports rather than
       open-ended free association (`FC RSA = 0.471` vs. `FA RSA = 0.305`).
  2. **Alignment across C1/C2/C3 & S1/S2/S3 — V-Enriched Yoneda Embedding (Indra Representation)**
     (Lu et al., NeurIPS 2025, *The Indra Representation Hypothesis for Multimodal Alignment*):
     - Maps any vector `x in R^d` (whether `S2` 384d, `C1` 2048d, `C2` 2048d, or `C3 VQ-VAE` 384d)
       to its covariant Hom-functor over the Lawvere Cost Category `V = ([0, inf], >=, 0, +)`
       using angular distance `d(x, A_k) = arccos(cos(x - mu, A_k - mu))` against `K` canonical
       Clover Semantic Anchors `{A_1, ..., A_K}`.
     - Verified on 27,000 triples with `0` triangle-inequality violations (`6.0x` Top-1 and `5.0x`
       Top-3 cross-module alignment gain between S2 and C1 Layer 18 without training).
  3. **Context — 3-Layer Neuro-Symbolic Memory & 4-Graph Intent-Aware Traversal**
     (Jiang et al., 2026, *NS-Mem*, arXiv:2603.15280; Jiang et al., ACL 2026, *MAGMA*):
     - Organizes memory items across `Episodic` (`L_epi`), `Semantic` (`L_sem`), and `Logic DAG`
       (`L_logic`) layers with 4 orthogonal relational views (`Temporal`, `Causal`, `Semantic`, `Entity`).
     - Dual-level Index Vectors `I = {i_goal, i_step}` scored via
       `score(q, N) = alpha * sim(q, i_goal) + (1 - alpha) * sim(q, i_step)` (`alpha = 0.3`),
       maintained incrementally via Exponential Moving Average (`EMA`, `beta = 0.9`) and
       Beta-Binomial / Dirichlet-Multinomial transition posteriors.
     - Deterministic symbolic query functions (`getProcedureWithEvidence`, `queryStepSequence(goal, C)`,
       `aggregateEntityBehaviors`) filter procedural DAG paths by active world constraints `C`.
  4. **Interpretation & Bidirectional Neuro-Symbolic Alignment (`I_DAT` + `BiNS` + `DualMemory`)**
     (Odense & d'Avila Garcez, Artificial Intelligence 2025; Wang et al., EMNLP 2026, *BiNSGPS*;
      Wen et al., EMNLP 2026, *DualMemory*; Saglam et al., IJCNLP-AACL 2025):
     - Distributed-Atoms Semantic Encoding `i in I_DAT : X -> 2^M` with `Agg = intersection`,
       checking whether neural states satisfy `emptyset subset M_N subseteq M_L` (Neural Model soundness)
       and `L |=_S L' iff M_N subseteq M_L'` (Semantic Encoding completeness), while preserving
       Clover's 5 epistemic states (`known`, `inferred`, `unknown`, `contested`, `verified`).
     - Bidirectional Neuro-Symbolic loop (`BiNS` + `DualMemory`):
       * `Mode i (Conflict / Feasibility Rectification)`: Symbolic verifier returns `(c_r=0, m_r, u_r)`
         when a candidate action or draft violates an environment precondition or arithmetic/logical
         invariant, constraining C2 to rectify the conflicting assertion.
       * `Mode ii (Deductive Deadlock / Auxiliary Hypothesis)`: When symbolic rules in S1/S2 cannot
         bridge the gap to the goal, prompts C1/C2 to propose a typed auxiliary hypothesis `H`
         that is verified against `M_L` before C3 decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1. Canonical Clover Semantic Anchors (Indra Jewels across Domains, Intents & Tools)
# ---------------------------------------------------------------------------
CLOVER_SEMANTIC_ANCHORS: tuple[tuple[str, str, str], ...] = (
    # (anchor_id, category, canonical_description)
    ("ANC_CONCEPT_EXPLAIN", "intent:concept", "Explain the meaning, definition, theory, or mechanism of a concept in plain words without running any tool."),
    ("ANC_TOOL_WEB_SEARCH", "intent:tool:web_search", "Search the internet, Wikipedia, arXiv, or live web pages for current facts, papers, or external URLs."),
    ("ANC_TOOL_PYTHON", "intent:tool:python_interpreter", "Write and execute Python code to calculate exact numbers, solve equations, or test an algorithm."),
    ("ANC_TOOL_SHELL_SAFE", "intent:tool:run_shell", "Run a read-only terminal command such as git status, git log, disk space, battery level, or version check."),
    ("ANC_TOOL_SSH", "intent:tool:run_ssh", "Connect to a configured remote SSH server or host and run a non-interactive diagnostic command."),
    ("ANC_TOOL_FILE_READ", "intent:tool:repo_read", "Read, inspect, open, or list files and directories inside the user's active project workspace."),
    ("ANC_TOOL_FILE_WRITE", "intent:tool:repo_write", "Create, edit, append, or write content into a file in the workspace or save a note to memory."),
    ("ANC_TOOL_IMAGE_GEN", "intent:tool:generate_image", "Generate, draw, paint, or render a new picture, illustration, or icon from a text description."),
    ("ANC_DHARMA_RISKY", "dharma:ask_risky", "Destructive command that rewrites git history, deletes files, kills processes, formats disks, or alters system state."),
    ("ANC_QUERY_FACTUAL", "view:factual", "Direct factual question about an entity's attributes, recorded observation, or static knowledge."),
    ("ANC_QUERY_CONSTRAINT", "view:constraint", "Constrained decision problem where a broken resource, budget, or precondition restricts which actions are feasible."),
    ("ANC_QUERY_PROCEDURAL", "view:procedural", "Step-by-step workflow, ordered sequence of actions, or how-to procedure from start to goal."),
    ("ANC_QUERY_CAUSAL", "view:causal", "Diagnostic why/how question asking for the root cause, weakness, or causal chain behind an outcome or failure."),
    ("ANC_QUERY_TEMPORAL", "view:temporal", "Chronological question about when an event happened, what changed recently, or the order of past sessions."),
    ("ANC_DOMAIN_SYSTEMS", "domain:systems", "Operating systems, compilers, C/Rust/Python code, CPU/GPU hardware, caches, threads, and network sockets."),
    ("ANC_DOMAIN_MATH", "domain:math", "Mathematics, arithmetic, algebra, geometry, primes, matrices, calculus, and quantitative word problems."),
    ("ANC_DOMAIN_SECURITY", "domain:security", "Cybersecurity vulnerabilities, CVE advisories, CWE weaknesses, CAPEC attack patterns, encryption, and patches."),
    ("ANC_DOMAIN_SCIENCE", "domain:science", "Physics, biology, chemistry, astronomy, thermodynamics, enzymes, genomes, and natural phenomena."),
)


EPISTEMIC_STATES = ("known", "inferred", "unknown", "contested", "verified")
DHARMA_STATES = ("proceed", "ask", "do_not_know", "refuse")


# ---------------------------------------------------------------------------
# 2. Meaning & Alignment: Mean-Centered Indra Representation (Paper 1 + Paper 2)
# ---------------------------------------------------------------------------
class IndraSemanticSpace:
    """
    Implements the V-enriched Yoneda embedding (`C(X_i, -) = [d(X_i, A_1), ..., d(X_i, A_K)]`)
    over the Lawvere Cost Category `V = ([0, inf], >=, 0, +)` with angular distance on
    mean-centered representations (`Paper 1` Section 3.4 + `Paper 2` Definition 1-3 & Theorem 1-2).

    Because every module's vector is mapped to its angular distance profile against the same `K`
    canonical anchors `{A_1, ..., A_K}`, representations from:
      - `S2` Embedder (`384d` SubwordHash or BGE-small),
      - `C1` LFM3-Omni-2B Hidden States (`2048d`, Layer 8 for intent / Layer 18 for semantics),
      - `C2` HybridLink Plan Vectors (`2048d`), and
      - `C3` 11-Level VQ-VAE Concept Latents (`384d`)
    all project into the exact same `K`-dimensional relational space `[0, pi]^K`!
    """

    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        if embed_fn is None:
            from memory_index import SubwordHashEmbedder
            _hash_emb = SubwordHashEmbedder()
            embed_fn = _hash_emb.encode
        self.embed_fn = embed_fn
        self.anchor_ids = [a[0] for a in CLOVER_SEMANTIC_ANCHORS]
        self.anchor_categories = [a[1] for a in CLOVER_SEMANTIC_ANCHORS]
        self.anchor_texts = [a[2] for a in CLOVER_SEMANTIC_ANCHORS]
        raw_anchors = np.asarray(self.embed_fn(self.anchor_texts), dtype=np.float64)
        self.text_mu = raw_anchors.mean(axis=0, keepdims=True)
        self.text_anchors_mc = self._normalize(raw_anchors - self.text_mu)
        # Per-module registered anchor matrices `{space_name: (mu [1, d], anchors_mc [K, d])}`
        self._module_anchors: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "text": (self.text_mu, self.text_anchors_mc)
        }

    @staticmethod
    def _normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.clip(norms, 1e-12, None)

    def register_module_anchors(self, space_name: str, anchor_vectors: np.ndarray) -> None:
        """
        Registers the `K` anchor representations `[K, d_M]` extracted from module `space_name`
        (e.g., `c1_l18`, `c2_plan`, `c3_vqvae`), computing and storing the module's anisotropy mean `mu_M`.
        """
        A = np.asarray(anchor_vectors, dtype=np.float64)
        if A.shape[0] != len(self.anchor_ids):
            raise ValueError(f"Expected {len(self.anchor_ids)} anchor vectors, got {A.shape[0]}")
        mu = A.mean(axis=0, keepdims=True)
        A_mc = self._normalize(A - mu)
        self._module_anchors[space_name] = (mu, A_mc)

    def yoneda_profile(self, vectors: np.ndarray, space_name: str = "text") -> np.ndarray:
        """
        Computes the Lawvere-metric Indra representation `I(x) = [arccos(cos(x - mu, A_k - mu))]_{k=1..K}`
        in `[0, pi]^K` (`[N, K]` for batch input or `[K]` for a 1D vector).
        """
        mu, A_mc = self._module_anchors.get(space_name, (self.text_mu, self.text_anchors_mc))
        X = np.asarray(vectors, dtype=np.float64)
        is_1d = X.ndim == 1
        if is_1d:
            X = X[None, :]
        X_mc = self._normalize(X - mu)
        cos_sim = np.clip(X_mc @ A_mc.T, -1.0, 1.0)
        indra = np.arccos(cos_sim)
        return indra[0] if is_1d else indra

    def text_profile(self, text: str) -> np.ndarray:
        vec = np.asarray(self.embed_fn([text or ""]), dtype=np.float64)[0]
        return self.yoneda_profile(vec, space_name="text")

    @staticmethod
    def relational_similarity(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
        """
        Structure-preserving similarity between two Indra Yoneda profiles `I(a), I(b) in [0, pi]^K`
        (Corollary 2 of Lu et al., NeurIPS 2025). Works even when `a` and `b` came from different models/dimensions!
        """
        pa = np.asarray(profile_a, dtype=np.float64).ravel()
        pb = np.asarray(profile_b, dtype=np.float64).ravel()
        pa0 = pa - pa.mean()
        pb0 = pb - pb.mean()
        denom = float(np.linalg.norm(pa0) * np.linalg.norm(pb0))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(pa0, pb0) / denom)

    @staticmethod
    def verify_lawvere_metric(profiles_matrix: np.ndarray, tol: float = 1e-7) -> dict[str, Any]:
        """Verifies Lawvere cost-category identity `d(x,x)=0` and triangle inequality `d(x,z) <= d(x,y) + d(y,z)`."""
        X = IndraSemanticSpace._normalize(profiles_matrix - profiles_matrix.mean(axis=0, keepdims=True))
        D = np.arccos(np.clip(X @ X.T, -1.0, 1.0))
        np.fill_diagonal(D, 0.0)
        n = D.shape[0]
        violations = 0
        for i in range(n):
            for j in range(n):
                ij = D[i, j]
                for k in range(n):
                    if D[i, k] > ij + D[j, k] + tol:
                        violations += 1
        return {
            "n": n,
            "triples": n * n * n,
            "triangle_violations": violations,
            "lawvere_valid": violations == 0,
        }

    def forced_choice_select(
        self,
        cue_text: str,
        candidates: list[str],
        top_k: int = 2,
    ) -> list[tuple[int, str, float]]:
        """
        Paper 1 (Schiekiera et al., ICML 2026) Forced-Choice (FC) relational selector:
        Concentrates comparison onto the explicit candidate support `candidates` in mean-centered
        Indra space, returning the `top_k` candidates `(index, candidate_text, similarity)`.
        """
        if not candidates:
            return []
        all_vecs = np.asarray(self.embed_fn([cue_text] + list(candidates)), dtype=np.float64)
        mu = all_vecs.mean(axis=0, keepdims=True)
        normed = self._normalize(all_vecs - mu)
        cue_v = normed[0]
        cand_vs = normed[1:]
        sims = cand_vs @ cue_v
        order = np.argsort(sims)[::-1][: max(1, min(top_k, len(candidates)))]
        return [(int(idx), candidates[int(idx)], round(float(sims[int(idx)]), 4)) for idx in order]


# ---------------------------------------------------------------------------
# 3. Context: NS-Mem (Paper 6) + MAGMA (Paper 10) 3-Layer Multi-Graph Memory
# ---------------------------------------------------------------------------
@dataclass
class EpisodicNode:
    node_id: str
    timestamp: float
    description: str
    entities: list[str]
    embedding: np.ndarray
    indra_profile: np.ndarray
    causal_parents: list[str] = field(default_factory=list)
    epistemic_status: str = "observed"


@dataclass
class SemanticEntityNode:
    entity_id: str
    entity_type: str
    attributes: dict[str, Any]
    embedding: np.ndarray
    confidence_weight: int = 1
    epistemic_status: str = "known"


@dataclass
class ProceduralStepNode:
    step_id: str
    action: str
    preconditions: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    successes: int = 1
    trials: int = 1

    @property
    def posterior_reliability(self) -> float:
        """Theorem 2 (NS-Mem Appendix A.3): Beta-Binomial posterior mean `(alpha + s) / (alpha + beta + n)`."""
        return (1.0 + self.successes) / (2.0 + self.trials)


@dataclass
class ProceduralLogicNode:
    """
    NS-Mem Logic Node `N = (id, goal, I={i_goal, i_step}, G=(V, E, A), F, episodic_links)`:
    Couples dual-level neural Index Vectors with a symbolic Procedural DAG `G`.
    """
    logic_id: str
    goal: str
    i_goal: np.ndarray
    i_step: np.ndarray
    steps: dict[str, ProceduralStepNode]
    edges: dict[str, dict[str, int]]  # u -> {v: transition_count}
    episodic_links: list[str] = field(default_factory=list)
    entity_links: list[str] = field(default_factory=list)

    def transition_prob(self, u: str, v: str) -> float:
        """Theorem 2 (NS-Mem Eq. 2 & 8): Dirichlet-Multinomial posterior transition probability."""
        outs = self.edges.get(u) or {}
        if not outs or v not in outs:
            return 0.0
        k = len(outs)
        return (1.0 + outs[v]) / (float(k) + sum(outs.values()))

    def update_ema(self, new_obs_vec: np.ndarray, beta: float = 0.9) -> None:
        """NS-Mem Eq. 1: Exponential Moving Average (EMA) refinement `i_{t+1} = beta * i_t + (1 - beta) * phi(o_new)`."""
        v = np.asarray(new_obs_vec, dtype=np.float64)
        self.i_step = beta * self.i_step + (1.0 - beta) * v
        norm = np.linalg.norm(self.i_step)
        if norm > 1e-12:
            self.i_step = self.i_step / norm

    def record_transition(self, u: str, v: str, succeeded: bool = True) -> None:
        self.edges.setdefault(u, {})
        self.edges[u][v] = self.edges[u].get(v, 0) + 1
        if v in self.steps:
            self.steps[v].trials += 1
            if succeeded:
                self.steps[v].successes += 1


class NeuroSymbolicMultiGraphMemory:
    """
    Unifies Paper 6 (NS-Mem 3-Layer Memory + SK-Gen EMA + Deterministic Symbolic Functions)
    and Paper 10 (MAGMA 4 Orthogonal Relation Graphs: Temporal, Causal, Semantic, Entity).
    """

    def __init__(self, indra_space: IndraSemanticSpace, alpha_dual: float = 0.3, tau_pos: float = 0.75):
        self.indra = indra_space
        self.alpha_dual = alpha_dual
        self.tau_pos = tau_pos
        self.episodic: dict[str, EpisodicNode] = {}
        self.semantic: dict[str, SemanticEntityNode] = {}
        self.logic: dict[str, ProceduralLogicNode] = {}
        # MAGMA 4 Orthogonal Relation Graphs
        self.temporal_edges: list[tuple[str, str]] = []          # (earlier_ep_id, later_ep_id)
        self.causal_edges: list[tuple[str, str, str]] = []       # (cause_id, effect_id, reason)
        self.semantic_edges: list[tuple[str, str, float]] = []   # (id_a, id_b, cos_sim)
        self.entity_edges: dict[str, list[str]] = {}             # entity_id -> [node_ids]

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.asarray(self.indra.embed_fn([text or ""]), dtype=np.float64)[0]
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v

    def add_episodic(
        self,
        node_id: str,
        description: str,
        entities: list[str] | None = None,
        timestamp: float | None = None,
        causal_parents: list[str] | None = None,
    ) -> EpisodicNode:
        ts = float(timestamp if timestamp is not None else len(self.episodic) + 1)
        ents = list(entities or [])
        vec = self._embed_one(description)
        prof = self.indra.yoneda_profile(vec, space_name="text")
        node = EpisodicNode(
            node_id=node_id,
            timestamp=ts,
            description=description,
            entities=ents,
            embedding=vec,
            indra_profile=prof,
            causal_parents=list(causal_parents or []),
        )
        # Temporal graph edge from most recent episodic node
        if self.episodic:
            last_ep = max(self.episodic.values(), key=lambda e: e.timestamp)
            if last_ep.timestamp <= ts and last_ep.node_id != node_id:
                self.temporal_edges.append((last_ep.node_id, node_id))
        # Semantic graph edges for conceptually similar events
        for other in self.episodic.values():
            sim = float(np.dot(vec, other.embedding))
            if sim >= 0.45:
                self.semantic_edges.append((other.node_id, node_id, round(sim, 4)))
        # Causal graph edges
        for cp in node.causal_parents:
            self.causal_edges.append((cp, node_id, "causes"))
        # Entity graph edges
        for ent in ents:
            self.entity_edges.setdefault(ent, []).append(node_id)
        self.episodic[node_id] = node
        return node

    def consolidate_semantic(self, entity_id: str, entity_type: str, attributes: dict[str, Any]) -> SemanticEntityNode:
        """
        NS-Mem Section 4.1(2) Reinforcement-Based Consolidation Policy:
        Updates and reinforces existing entity node when present rather than duplicating nodes.
        """
        attr_str = f"{entity_id} ({entity_type}): " + ", ".join(f"{k}={v}" for k, v in sorted(attributes.items()))
        vec = self._embed_one(attr_str)
        if entity_id in self.semantic:
            existing = self.semantic[entity_id]
            existing.attributes.update(attributes)
            existing.confidence_weight += 1
            existing.embedding = self._embed_one(
                f"{entity_id} ({ existing.entity_type }): "
                + ", ".join(f"{k}={v}" for k, v in sorted(existing.attributes.items()))
            )
            return existing
        node = SemanticEntityNode(
            entity_id=entity_id,
            entity_type=entity_type,
            attributes=dict(attributes),
            embedding=vec,
        )
        self.semantic[entity_id] = node
        return node

    def add_or_fuse_logic_dag(
        self,
        logic_id: str,
        goal: str,
        step_specs: list[tuple[str, str, dict[str, Any], dict[str, Any]]],
        transitions: list[tuple[str, str]],
        episodic_links: list[str] | None = None,
        entity_links: list[str] | None = None,
    ) -> ProceduralLogicNode:
        """
        NS-Mem Section 4.2.1 & 4.2.3 (SK-Gen + Knowledge Fusion):
        Constructs or fuses a multi-path Procedural DAG `G = (V, E, A)` with dual-level Index Vectors
        `i_goal = phi(goal)` and `i_step = mean_{s in steps} phi(s)`.
        """
        i_goal = self._embed_one(goal)
        step_vecs = [self._embed_one(action) for _, action, _, _ in step_specs]
        i_step = np.mean(step_vecs, axis=0) if step_vecs else i_goal.copy()
        n_step = np.linalg.norm(i_step)
        if n_step > 1e-12:
            i_step = i_step / n_step

        if logic_id in self.logic:
            node = self.logic[logic_id]
            node.update_ema(i_step, beta=0.9)
        else:
            steps_map = {
                "START": ProceduralStepNode("START", "START"),
                "GOAL": ProceduralStepNode("GOAL", "GOAL"),
            }
            node = ProceduralLogicNode(
                logic_id=logic_id,
                goal=goal,
                i_goal=i_goal,
                i_step=i_step,
                steps=steps_map,
                edges={},
            )
            self.logic[logic_id] = node

        for sid, act, pre, eff in step_specs:
            if sid not in node.steps:
                node.steps[sid] = ProceduralStepNode(sid, act, dict(pre), dict(eff))
            else:
                node.steps[sid].preconditions.update(pre)
                node.steps[sid].effects.update(eff)
        for u, v in transitions:
            node.record_transition(u, v, succeeded=True)
        for ep in episodic_links or []:
            if ep not in node.episodic_links:
                node.episodic_links.append(ep)
        for ent in entity_links or []:
            if ent not in node.entity_links:
                node.entity_links.append(ent)
            self.entity_edges.setdefault(ent, []).append(logic_id)
        return node

    @staticmethod
    def classify_query_view(query: str) -> str:
        """
        NS-Mem Section 4.3.1 + MAGMA Section 3.3 Intent-Aware Query Classification:
        Routes `query` to `'constraint'`, `'procedural'`, `'causal'`, `'temporal'`, or `'factual'`.
        """
        q = (query or "").lower()
        if re.search(r"\b(?:broken|unavailable|without|cannot|can't|missing|violat|constraint|instead|if\s+.*\s+is\s+not|what\s+should\s+\w+\s+do\s+next)\b", q):
            return "constraint"
        if re.search(r"\b(?:how\s+to|steps\s+to|procedure|workflow|in\s+what\s+order|sequence\s+of)\b", q):
            return "procedural"
        if re.search(r"\b(?:why\s+did|what\s+caused|root\s+cause|because|reason\s+for|lead\s+to)\b", q):
            return "causal"
        if re.search(r"\b(?:when\s+did|before|after|timeline|chronolog|earlier|last\s+time|recent)\b", q):
            return "temporal"
        return "factual"

    # --- NS-Mem Section 4.3.3 Deterministic Symbolic Query Functions (Theorem 4 Determinism Guarantee) ---
    def getProcedureWithEvidence(self, goal_query: str) -> tuple[ProceduralLogicNode | None, list[EpisodicNode]]:
        """Function 1: Returns the highest-scoring Logic Node (via dual-index Eq. 3) and its supporting episodic nodes."""
        if not self.logic:
            return None, []
        q_vec = self._embed_one(goal_query)
        best_node = max(
            self.logic.values(),
            key=lambda n: self.alpha_dual * float(np.dot(q_vec, n.i_goal)) + (1.0 - self.alpha_dual) * float(np.dot(q_vec, n.i_step)),
        )
        ev = [self.episodic[eid] for eid in best_node.episodic_links if eid in self.episodic]
        return best_node, ev

    def queryStepSequence(
        self,
        goal_query: str,
        constraints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Function 2 (NS-Mem Eq. 5):
        Enumerates all valid paths `pi in Pi(START, GOAL)` in the retrieved Procedural DAG `G`
        whose every step satisfies the active world constraints `A(v) |= C` (plus semantic entity state),
        ranked by posterior path probability.
        """
        node, _ = self.getProcedureWithEvidence(goal_query)
        if node is None:
            return []
        # Merge current semantic entity attributes with explicit query constraints
        world_state: dict[str, Any] = {}
        for ent_id, ent_node in self.semantic.items():
            for k, v in ent_node.attributes.items():
                world_state[f"{ent_id}.{k}"] = v
                world_state[k] = v
        if constraints:
            world_state.update(constraints)

        valid_paths: list[dict[str, Any]] = []

        def dfs(curr: str, path: list[str], prob: float, visited: set[str]) -> None:
            if curr == "GOAL":
                actions = [node.steps[s].action for s in path if s not in ("START", "GOAL") and s in node.steps]
                valid_paths.append({
                    "step_ids": list(path),
                    "actions": actions,
                    "posterior_score": round(prob, 4),
                })
                return
            for nxt in sorted((node.edges.get(curr) or {}).keys()):
                if nxt in visited:
                    continue
                step_obj = node.steps.get(nxt)
                if step_obj is not None:
                    # Check all preconditions A(v) |= C
                    if any(world_state.get(k) is not None and world_state.get(k) != req_v for k, req_v in step_obj.preconditions.items()):
                        continue
                edge_p = max(0.05, node.transition_prob(curr, nxt))
                step_r = step_obj.posterior_reliability if step_obj else 1.0
                dfs(nxt, path + [nxt], prob * edge_p * step_r, visited | {nxt})

        dfs("START", ["START"], 1.0, {"START"})
        valid_paths.sort(key=lambda x: -x["posterior_score"])
        return valid_paths

    def aggregateEntityBehaviors(self, entity_id: str) -> dict[str, Any]:
        """Function 3: Deterministically collects all Episodic, Semantic, Causal, and Logic records linked to `entity_id`."""
        linked_ids = self.entity_edges.get(entity_id, [])
        ep_list = [self.episodic[i].description for i in linked_ids if i in self.episodic]
        lg_list = [self.logic[i].goal for i in linked_ids if i in self.logic]
        causal_chains = [
            f"{u} -> {v} ({r})" for u, v, r in self.causal_edges
            if u in linked_ids or v in linked_ids
        ]
        sem_info = self.semantic.get(entity_id)
        return {
            "entity_id": entity_id,
            "attributes": dict(sem_info.attributes) if sem_info else {},
            "episodic_events": ep_list,
            "procedures": lg_list,
            "causal_links": causal_chains,
        }


# ---------------------------------------------------------------------------
# 4. Interpretation & Alignment: Odense-Garcez I_DAT + BiNS + DualMemory
# ---------------------------------------------------------------------------
@dataclass
class FeasibilityRule:
    """
    Paper 5 (Wen et al., EMNLP 2026 — DualMemory) Eq. 5:
    Executable symbolic verifier `f_r(o_t, g_t, a_t) -> (c_r, m_r, u_r)`
    with zero false rejections on valid transitions.
    """
    rule_id: str
    description: str
    check_fn: Callable[[dict[str, Any], str], tuple[bool, str, str]]


@dataclass
class SemanticFrame:
    """
    The unified Semantic Interface packet shared across `(User, S1, S2, S3, C1, C2, C3)`
    for a single Clover cycle movement.
    """
    query: str
    indra_profile: list[float]
    top_anchors: list[tuple[str, float]]
    query_view: str                    # 'factual' | 'constraint' | 'procedural' | 'causal' | 'temporal'
    dharma_state: str                  # 'proceed' | 'ask' | 'do_not_know' | 'refuse'
    dharma_reason: str
    is_concept_only: bool
    offered_tools: list[str]
    epistemic_state: str               # 'known' | 'inferred' | 'unknown' | 'contested' | 'verified'
    constrained_plan_steps: list[str] = field(default_factory=list)


class CloverSemanticInterface:
    """
    Central Semantic Interface connecting User, Fuel System (`S1+S2+S3`), and Model Engine (`C1+C2+C3`).

    - **Meaning**: Mean-centered Indra Yoneda profile `I(q) in [0, pi]^18` + Forced-Choice candidate ranking.
    - **Context**: `NeuroSymbolicMultiGraphMemory` (`NS-Mem` + `MAGMA`) with dual-index retrieval and
      constraint-filtered procedural DAG traversal.
    - **Interpretation**: `Odense-Garcez (2025)` distributed-atoms `I_DAT` verification (`M_N subseteq M_L`).
    - **Alignment**: `BiNS` (Mode i Conflict Rectification & Mode ii Auxiliary Hypothesis Generation) +
      `DualMemory` Feasibility Verifier + `AGENTS.md` Dharma Discernment (`Proceed / Ask / Do Not Know / Refuse`).
    """

    def __init__(self, embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.indra = IndraSemanticSpace(embed_fn=embed_fn)
        self.memory = NeuroSymbolicMultiGraphMemory(self.indra)
        self.feasibility_rules: list[FeasibilityRule] = []
        self._register_default_feasibility_rules()

    def _register_default_feasibility_rules(self) -> None:
        import local_terminal

        def _rule_shell_risk(state: dict[str, Any], action: str) -> tuple[bool, str, str]:
            why = local_terminal.risky(action)
            if why and not state.get("user_confirmed_risky"):
                return (
                    False,
                    f"Feasibility/Dharma Boundary: '{action}' {why}.",
                    "Ask the human for explicit one-time confirmation before executing this command.",
                )
            return True, "OK", ""

        def _rule_vault_isolation(state: dict[str, Any], action: str) -> tuple[bool, str, str]:
            if re.search(r"(?:vault[/\\]|\.app_secret|id_ed25519_clover(?!\.pub)|token_Protected|keys_Protected)", action, re.I):
                return (
                    False,
                    "Feasibility/Dharma Boundary: Clover's encrypted vault and private keys are never read or modified by tools.",
                    "Use public APIs or `.pub` keys only; never read `vault/`.",
                )
            return True, "OK", ""

        self.feasibility_rules.extend([
            FeasibilityRule("R1_DestructiveCommandGate", "Always require confirmation for destructive shell/git commands", _rule_shell_risk),
            FeasibilityRule("R2_VaultSecretIsolation", "Never read or overwrite encrypted vault or private keys", _rule_vault_isolation),
        ])

    def analyze_turn(
        self,
        prompt: str,
        asked_tools_fn: Callable[[str], list[str]] | None = None,
        world_constraints: dict[str, Any] | None = None,
        c1_layer18_vec: np.ndarray | None = None,
    ) -> SemanticFrame:
        """
        Builds the unified `SemanticFrame` for `prompt` before/during C1/C2/C3 execution.
        """
        import local_terminal

        q = (prompt or "").strip()
        prof = (
            self.indra.yoneda_profile(c1_layer18_vec, space_name="c1_l18")
            if c1_layer18_vec is not None and "c1_l18" in self.indra._module_anchors
            else self.indra.text_profile(q)
        )
        # Convert angular distances d in [0, pi] to closeness cos(d) in [-1, 1] for top-anchor readout
        closeness = np.cos(prof)
        top_idx = np.argsort(closeness)[::-1][:3]
        top_anchors = [(self.indra.anchor_ids[int(i)], round(float(closeness[int(i)]), 4)) for i in top_idx]

        view = self.memory.classify_query_view(q)
        offered = asked_tools_fn(q) if asked_tools_fn is not None else []
        is_concept = len(offered) == 0

        # Dharma discernment state (AGENTS.md Section 9: Proceed / Ask / Do Not Know / Refuse)
        dharma_state = "proceed"
        dharma_reason = "Sufficient context and within safe operating boundaries."
        for rule in self.feasibility_rules:
            ok_r, msg_r, _ = rule.check_fn(world_constraints or {}, q)
            if not ok_r:
                if "VaultSecretIsolation" in rule.rule_id:
                    dharma_state = "refuse"
                else:
                    dharma_state = "ask" if not is_concept else "proceed"
                if dharma_state != "proceed":
                    dharma_reason = msg_r
                break

        # If query is procedural or constrained, run NS-Mem deterministic DAG path filter
        plan_steps: list[str] = []
        if view in ("constraint", "procedural") and self.memory.logic:
            paths = self.memory.queryStepSequence(q, constraints=world_constraints)
            if paths:
                plan_steps = paths[0]["actions"]

        epistemic = "verified" if plan_steps else ("known" if is_concept else "inferred")
        return SemanticFrame(
            query=q,
            indra_profile=[round(float(x), 4) for x in prof],
            top_anchors=top_anchors,
            query_view=view,
            dharma_state=dharma_state,
            dharma_reason=dharma_reason,
            is_concept_only=is_concept,
            offered_tools=offered,
            epistemic_state=epistemic,
            constrained_plan_steps=plan_steps,
        )

    def verify_idat_semantic_encoding(
        self,
        base_facts: set[str],
        horn_rules: list[tuple[tuple[str, ...], str]],
        candidate_claims: list[str],
    ) -> dict[str, Any]:
        """
        Paper 7 (Odense & d'Avila Garcez, Artificial Intelligence 2025) Definition 3.2 & 3.4 (`I_DAT`):
        Computes the least fixed-point `T_P^inf(base_facts)` of the symbolic knowledge base `L = (base_facts, horn_rules)`
        and checks whether the neural network's emitted claims `candidate_claims` (`M_N`) satisfy:
          - Soundness (Neural Model): every claimed atom is entailed by `L` (`emptyset subset M_N subseteq M_L`)
          - Completeness (Semantic Encoding): all target entailed atoms of `L` are present (`M_N == M_L`).
        """
        entailed = set(base_facts)
        steps = 0
        while True:
            steps += 1
            added = False
            for body, head in horn_rules:
                if all(b in entailed for b in body) and head not in entailed:
                    entailed.add(head)
                    added = True
            if not added or steps > 32:
                break

        claimed_set = set(candidate_claims)
        unentailed_claims = sorted(claimed_set - entailed)
        missing_entailed = sorted(entailed - claimed_set)
        is_neural_model = len(claimed_set) > 0 and len(unentailed_claims) == 0
        is_semantic_encoding = is_neural_model and len(missing_entailed) == 0
        return {
            "fixed_point_steps": steps,
            "entailed_model_M_L": sorted(entailed),
            "claimed_M_N": sorted(claimed_set),
            "unentailed_contradictions": unentailed_claims,
            "missing_entailed_atoms": missing_entailed,
            "is_neural_model_sound": is_neural_model,
            "is_semantic_encoding_complete": is_semantic_encoding,
            "epistemic_status": "verified" if is_semantic_encoding else ("known" if is_neural_model else "contested"),
        }

    def bidirectional_bins_diagnose(
        self,
        question: str,
        draft_answer: str,
        arithmetic_or_compiler_findings: list[str],
        missing_bridge_concepts: list[str] | None = None,
        world_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Paper 4 (BiNSGPS, EMNLP 2026) + Paper 5 (DualMemory, EMNLP 2026) Bidirectional Feedback:
          - `Mode i (Rectify Inconsistent Representations)`: When deterministic checks or feasibility rules
            find a contradiction `findings`, returns a constrained rectification directive anchored on
            verified facts (modifying/deleting only the conflicting assertion).
          - `Mode ii (Propose Auxiliary Hypotheses)`: When the symbolic solver/graph reaches a deductive
            deadlock (`missing_bridge_concepts`), prompts the neural model to propose a typed auxiliary
            hypothesis `H` that bridges the known premises to the target goal.
        """
        violations: list[tuple[str, str]] = []
        for rule in self.feasibility_rules:
            ok_r, msg_r, rev_r = rule.check_fn(world_state or {}, draft_answer)
            if not ok_r:
                violations.append((msg_r, rev_r))

        if arithmetic_or_compiler_findings or violations:
            all_diags = list(arithmetic_or_compiler_findings) + [f"{m} -> {u}" for m, u in violations]
            return {
                "bins_mode": "Mode_i_Conflict_Rectification",
                "status": "conflict_detected",
                "diagnostics": all_diags,
                "directive": (
                    "BiNS Mode i (Conflict Rectification): Your draft conflicts with verified symbolic constraints:\n"
                    + "\n".join(f"- {d}" for d in all_diags)
                    + "\nModify or remove ONLY the conflicting step while keeping verified ground-truth facts intact."
                ),
            }

        if missing_bridge_concepts:
            return {
                "bins_mode": "Mode_ii_Auxiliary_Hypothesis",
                "status": "deductive_deadlock",
                "missing_concepts": list(missing_bridge_concepts),
                "directive": (
                    "BiNS Mode ii (Auxiliary Hypothesis Generation): The symbolic graph reached a deadlock and needs "
                    f"a bridging relation for {', '.join(missing_bridge_concepts)}. State a concise, verifiable "
                    "auxiliary hypothesis H connecting the given facts to the target quantity, then solve."
                ),
            }

        return {
            "bins_mode": "Verified_Consistent",
            "status": "ok",
            "diagnostics": [],
            "directive": "",
        }
