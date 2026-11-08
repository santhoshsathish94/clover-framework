"""Clover 5.x Curiosity Foundation (`curiosity_foundation.py`).

Implements the persistent, developmental curiosity and metacognitive feedback substrate for Clover 5.x across all 13
foundational research papers (`01`–`13`), governed by three architectural principles:
  1. **Reuse Clover's existing components wherever possible** (`CloverMemoryGraphStore`, `CloverOntologyGraph`,
     `CloverJEPARuntime`, `LayerKit`, `clover_cycle`, `Repo`, `MemoryIndex`, `IsolatedPythonSandbox`, `semantics.py`).
  2. **Build the missing underlying motivation and feedback mechanisms rather than adding another question-generating agent**:
     - `CuriosityRegionTree` (`R-IAC` Criterion C2 output-variance splitting + parent-to-child state inheritance, Paper 01)
     - Non-saturating Plasticity-Weighted Progress + Habituation Cooldown Trace (Paper 02)
     - Endogenous Interest Emergence inside Clover's `semantics.py` Discernment Hierarchy (Paper 03)
     - Coupled Knowledge Progress (`LPM`, Eq. 11) + Competence Progress (`CPM`, Eq. 24) + Predictive Surprise Ratio
       (`SM`, Eq. 15) + Controllability Gate defeating the Passive Spectator Trap (Paper 04)
     - Dual-Channel Phasic Surprise vs. Tonic Regional Progress in Category-Level Microcolumns (`MR + metaMR`, Paper 05)
     - POMDP Observing Action Value of Information (`VoI`) with Indifference Threshold `p* = 1 + r_o` + Local `LP`
       on Branch-Expanding Frontiers (Paper 06)
     - Endogenous Belief-Based Utility Loop + Anti-Confirmation Active Sampler separating Neutral Diagnosticity from
       Confirmation Counts (Paper 07)
     - Closed `LP <-> State Curiosity <-> Memory Consolidation` Feedback Loop (Fig. 1B) + `ZPDES` Dynamic Zone of
       Proximal Development with Lateral Prerequisite/Modality Switching (Fig. 5, Paper 08)
     - Deterministic Structural-First Feedback (`_eval`, Compiler, Horn Inference, Sandbox) Driving Emergent
       Workspace-Specific Developmental Attractors (Paper 09)
     - Confidence/Reliability Module `C(R_n)` predicting Expected Error `y_C`, Signed Reliability Improvement
       `dot_y_C(t) = y_C_old - y_C_new`, and Smoothed Signed Shift `|E[Delta z]|` (Jensen's Inequality Guard, Paper 10)
     - `SGIM-ACTS` Epistemic Strategy Bandit (`(region, strategy, actor)`) with Automatic Compensation Against
       Static/Biased Sources & 7-View Coverage Tracking (Paper 11)
     - `SGIM-D` Social Goal Emulation + Policy Imitation with Self-Execution Calibration, `|LP|` Competence-Drop
       Vigilance (Eq. 3), and Reliability-Weighted Retrieval `dist + alpha * var_h` (Algorithm 4.6, Paper 12)
     - `IMOL` Action-Controllable Subspace Filter (`R^2(Action -> Outcome)`) + Dual Episodic-Semantic Memory
       Consolidation with Non-Regression Replay (Paper 13)
  3. **Keep curiosity persistent across interactions** (`<repo_dir>/growth/curiosity/curiosity_state.json`), with its
     own evolving interests, unresolved questions, exploration history, and learning progress — and **zero hardcoded
     lists of topics, books, or questions Clover should find interesting**.
"""
from __future__ import annotations

import collections
import copy
import datetime
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

SCHEMA_VERSION = "5.1.0"
D_LATENT = 64

# Clover's 5 existing epistemic strategy primitives (Paper 11 SGIM-ACTS)
EPISTEMIC_STRATEGIES: tuple[str, ...] = (
    "memory_graph_recall",
    "workspace_code_inspect",
    "sandbox_counterexample_probe",
    "web_primary_research",
    "human_clarification_ask",
)

# Observation cost r_o per epistemic strategy (Paper 06 Fig. 1A POMDP VoI)
STRATEGY_OBSERVATION_COST: dict[str, float] = {
    "memory_graph_recall": 0.04,
    "workspace_code_inspect": 0.07,
    "sandbox_counterexample_probe": 0.10,
    "web_primary_research": 0.14,
    "human_clarification_ask": 0.18,
}

# Strategy-to-action_prior_8 dimension mapping (bridges into jepa_engine.semantic_action_prior_8)
STRATEGY_TO_ACTION_DIM: dict[str, int] = {
    "web_primary_research": 0,         # Dim 0: web_factual
    "sandbox_counterexample_probe": 2, # Dim 2: code_python (also boosts Dim 1 math_calc)
    "memory_graph_recall": 3,          # Dim 3: memory_knowledge
    "workspace_code_inspect": 4,       # Dim 4: files_workspace
    "human_clarification_ask": 7,      # Dim 7: general_dialogue
}

STOPWORDS: frozenset[str] = frozenset(
    "a about above after again against all am an and any are as at be because been before being below between both "
    "but by can could did do does doing down during each few for from further had has have having he her here hers "
    "herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on "
    "once only or other our ours ourselves out over own please same she should so some such than that the their "
    "theirs them themselves then there these they this those through to too under until up very was we were what "
    "when where which while who whom why will with would you your yours yourself yourselves clover question answer "
    "use using used".split()
)


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_emergent_terms(text: str, max_terms: int = 12) -> list[str]:
    """Extract content terms from `text` without any hardcoded topic list."""
    words = re.findall(r"[a-z][a-z0-9_\-]{2,}", (text or "").lower())
    filtered = [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words if w not in STOPWORDS]
    counts = collections.Counter(filtered)
    return [w for w, _ in counts.most_common(max_terms)]


def encode_text_to_64d(text: str, latent_cue_64: np.ndarray | Sequence[float] | None = None) -> np.ndarray:
    """Project `text` (and optional `latent_cue_64` from `CloverJEPARuntime`) into a unit-normalized 64-d vector."""
    v = np.zeros(D_LATENT, dtype=np.float64)
    terms = extract_emergent_terms(text, max_terms=32)
    if not terms:
        terms = ["empty"]
    for t in terms:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        for i in range(0, 32, 2):
            dim = digest[i] % D_LATENT
            sign = 1.0 if (digest[i + 1] & 1) == 0 else -1.0
            v[dim] += sign
    norm = float(np.linalg.norm(v))
    if norm > 1e-9:
        v /= norm
    if latent_cue_64 is not None:
        cue = np.asarray(latent_cue_64, dtype=np.float64).reshape(-1)
        if cue.size == D_LATENT and np.all(np.isfinite(cue)):
            c_norm = float(np.linalg.norm(cue))
            if c_norm > 1e-9:
                v = 0.55 * v + 0.45 * (cue / c_norm)
                v /= max(1e-9, float(np.linalg.norm(v)))
    return v


# ---------------------------------------------------------------------------
# 1. Data Structures for Persistent Curiosity State
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Paper 11 (`SGIM-ACTS`) & Paper 06 (`VoI`): per-(region, strategy) empirical progress and reliability."""
    strategy: str
    attempts: int = 0
    progress_window: list[float] = field(default_factory=list)
    error_window: list[float] = field(default_factory=list)
    views_discovered: int = 0
    last_used_step: int = 0

    @property
    def empirical_lp(self) -> float:
        if not self.progress_window:
            return 0.08  # initial optimistic exploration prior for untried strategy
        return float(np.mean(self.progress_window[-6:]))

    @property
    def outcome_variance(self) -> float:
        if len(self.error_window) < 2:
            return 0.02
        return float(np.var(self.error_window[-8:]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyStats":
        return cls(
            strategy=str(d.get("strategy") or "memory_graph_recall"),
            attempts=int(d.get("attempts") or 0),
            progress_window=[float(x) for x in (d.get("progress_window") or [])][-16:],
            error_window=[float(x) for x in (d.get("error_window") or [])][-16:],
            views_discovered=int(d.get("views_discovered") or 0),
            last_used_step=int(d.get("last_used_step") or 0),
        )


@dataclass
class CuriosityRegion:
    """Cortical Microcolumn (`MR + metaMR`, Paper 05) & `R-IAC` Tree Node (Paper 01, 04, 10, 12, 13)."""
    region_id: str
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    split_dim: int | None = None
    split_threshold: float | None = None
    centroid_64: list[float] = field(default_factory=lambda: [0.0] * D_LATENT)
    emergent_keywords: list[str] = field(default_factory=list)
    exemplar_vectors: list[list[float]] = field(default_factory=list)
    exemplar_outcomes: list[float] = field(default_factory=list)
    pred_error_window: list[float] = field(default_factory=list)
    comp_error_window: list[float] = field(default_factory=list)
    # Paper 10 (Schmidhuber 1991): Confidence module C(R_n) expected error y_C and signed shift vector E[Delta z]
    expected_error_yc: float = 0.35
    prev_expected_error_yc: float = 0.35
    signed_shift_ema_64: list[float] = field(default_factory=lambda: [0.0] * D_LATENT)
    # Paper 04 & 13: Controllability R^2(Action -> Outcome)
    controllability_score: float = 0.50
    controllable_Try_count: int = 0
    uncontrollable_Try_count: int = 0
    # Tonic progress signals
    tonic_lpm: float = 0.05
    tonic_cpm: float = 0.05
    dot_yc_reliability: float = 0.0
    abs_lp_vigilance: float = 0.0
    phasic_surprise_sm: float = 0.0
    habituation_trace: float = 0.0
    state_curiosity: float = 0.05
    visits: int = 0
    last_visited_step: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def is_leaf(self) -> bool:
        return self.left_child_id is None and self.right_child_id is None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep exemplar buffers bounded for compact persistence
        d["exemplar_vectors"] = self.exemplar_vectors[-40:]
        d["exemplar_outcomes"] = self.exemplar_outcomes[-40:]
        d["pred_error_window"] = self.pred_error_window[-32:]
        d["comp_error_window"] = self.comp_error_window[-32:]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CuriosityRegion":
        return cls(
            region_id=str(d["region_id"]),
            parent_id=d.get("parent_id"),
            left_child_id=d.get("left_child_id"),
            right_child_id=d.get("right_child_id"),
            split_dim=int(d["split_dim"]) if d.get("split_dim") is not None else None,
            split_threshold=float(d["split_threshold"]) if d.get("split_threshold") is not None else None,
            centroid_64=[float(x) for x in (d.get("centroid_64") or [0.0] * D_LATENT)],
            emergent_keywords=list(d.get("emergent_keywords") or []),
            exemplar_vectors=[[float(v) for v in row] for row in (d.get("exemplar_vectors") or [])],
            exemplar_outcomes=[float(x) for x in (d.get("exemplar_outcomes") or [])],
            pred_error_window=[float(x) for x in (d.get("pred_error_window") or [])],
            comp_error_window=[float(x) for x in (d.get("comp_error_window") or [])],
            expected_error_yc=float(d.get("expected_error_yc", 0.35)),
            prev_expected_error_yc=float(d.get("prev_expected_error_yc", 0.35)),
            signed_shift_ema_64=[float(x) for x in (d.get("signed_shift_ema_64") or [0.0] * D_LATENT)],
            controllability_score=float(d.get("controllability_score", 0.50)),
            controllable_Try_count=int(d.get("controllable_Try_count", 0)),
            uncontrollable_Try_count=int(d.get("uncontrollable_Try_count", 0)),
            tonic_lpm=float(d.get("tonic_lpm", 0.05)),
            tonic_cpm=float(d.get("tonic_cpm", 0.05)),
            dot_yc_reliability=float(d.get("dot_yc_reliability", 0.0)),
            abs_lp_vigilance=float(d.get("abs_lp_vigilance", 0.0)),
            phasic_surprise_sm=float(d.get("phasic_surprise_sm", 0.0)),
            habituation_trace=float(d.get("habituation_trace", 0.0)),
            state_curiosity=float(d.get("state_curiosity", 0.05)),
            visits=int(d.get("visits", 0)),
            last_visited_step=int(d.get("last_visited_step", 0)),
            created_at=str(d.get("created_at") or utc_now_iso()),
            updated_at=str(d.get("updated_at") or utc_now_iso()),
        )


@dataclass
class UnresolvedQuestionNode:
    """Papers 06, 07, 08, 13: Dynamic `ZPDES` epistemic question/hypothesis node (never pre-scripted)."""
    question_id: str
    region_id: str
    summary: str
    origin_kind: str  # "turn_discrepancy" | "memory_contradiction" | "unverified_hypothesis" | "competence_regression" | "child_frontier"
    memory_entry_id: str | None = None
    parent_question_id: str | None = None
    prerequisite_ids: list[str] = field(default_factory=list)
    lateral_modality_ids: list[str] = field(default_factory=list)
    zpd_status: str = "active_zpd"  # "active_zpd" | "stalled_deactivated" | "prerequisite_pending" | "resolved_verified"
    attempts: int = 0
    error_history: list[float] = field(default_factory=list)
    diagnosticity_score: float = 0.50  # Paper 07 neutral validity / expected uncertainty reduction
    views_covered: list[str] = field(default_factory=list)  # Paper 11 multi-view accumulation (up to 7 dimensions)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @property
    def recent_lp(self) -> float:
        if len(self.error_history) < 2:
            return 0.06
        half = max(1, len(self.error_history) // 2)
        pst = float(np.mean(self.error_history[:half]))
        rec = float(np.mean(self.error_history[half:]))
        return pst - rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnresolvedQuestionNode":
        return cls(
            question_id=str(d["question_id"]),
            region_id=str(d.get("region_id") or "R0"),
            summary=str(d.get("summary") or ""),
            origin_kind=str(d.get("origin_kind") or "turn_discrepancy"),
            memory_entry_id=d.get("memory_entry_id"),
            parent_question_id=d.get("parent_question_id"),
            prerequisite_ids=list(d.get("prerequisite_ids") or []),
            lateral_modality_ids=list(d.get("lateral_modality_ids") or []),
            zpd_status=str(d.get("zpd_status") or "active_zpd"),
            attempts=int(d.get("attempts") or 0),
            error_history=[float(x) for x in (d.get("error_history") or [])][-16:],
            diagnosticity_score=float(d.get("diagnosticity_score", 0.50)),
            views_covered=list(d.get("views_covered") or []),
            created_at=str(d.get("created_at") or utc_now_iso()),
            updated_at=str(d.get("updated_at") or utc_now_iso()),
        )


# ---------------------------------------------------------------------------
# 2. Core Mathematical Primitives (Criterion C2, LPM+CPM, VoI, Controllability)
# ---------------------------------------------------------------------------

def find_criterion_c2_split(
    vectors: Sequence[Sequence[float]],
    outcomes: Sequence[float],
    candidate_dims: Sequence[int] | None = None,
) -> tuple[int, float, float] | None:
    """Paper 01 (Oudeyer, Kaplan & Hafner 2007, p. 270 Criterion C2):
    Find split dimension `j*` and threshold `v*` that minimize the sample-weighted sum of child outcome variances:
      Gamma_n(j, v) = (|L| / N) * Var(outcomes_L) + (|R| / N) * Var(outcomes_R).
    Returns `(best_dim, best_threshold, variance_reduction_ratio)` or `None` if no valid split reduces variance.
    """
    n = len(vectors)
    if n < 8 or len(outcomes) != n:
        return None
    X = np.asarray(vectors, dtype=np.float64)
    Y = np.asarray(outcomes, dtype=np.float64)
    parent_var = float(np.var(Y))
    if parent_var < 1e-6:
        return None

    dims = list(candidate_dims) if candidate_dims is not None else list(range(min(D_LATENT, X.shape[1])))
    best_dim = -1
    best_thresh = 0.0
    best_weighted_var = parent_var

    for j in dims:
        col = X[:, j]
        uniq = np.unique(np.round(col, 6))
        if len(uniq) < 2:
            continue
        midpoints = 0.5 * (uniq[:-1] + uniq[1:])
        if len(midpoints) > 7:
            idx = np.linspace(0, len(midpoints) - 1, 7).astype(int)
            midpoints = midpoints[idx]
        for v in midpoints:
            v_flt = float(v)
            mask_l = col <= v_flt
            n_l = int(np.sum(mask_l))
            n_r = n - n_l
            if n_l < 3 or n_r < 3:
                continue
            w_var = (n_l / n) * float(np.var(Y[mask_l])) + (n_r / n) * float(np.var(Y[~mask_l]))
            if w_var < best_weighted_var:
                best_weighted_var = w_var
                best_dim = int(j)
                best_thresh = v_flt

    if best_dim < 0:
        return None
    reduction_ratio = (parent_var - best_weighted_var) / max(1e-9, parent_var)
    if reduction_ratio < 0.05:
        return None
    return best_dim, best_thresh, round(float(reduction_ratio), 4)


def compute_windowed_lp_and_vigilance(
    errors: Sequence[float],
    window_half: int = 6,
    noise_discount: float = 0.85,
) -> tuple[float, float, float]:
    """Compute:
      - `lp_positive`: Variance-discounted positive Learning Progress `<e(t-theta)> - <e(t)> - beta * SE(e_rec)`
        (Papers 01, 04, 05),
      - `abs_lp_vigilance`: Competence-drop vigilance `|<e(t-theta)> - <e(t)>|` when a previously mastered region
        (`<e(t-theta)> <= 0.20`) regresses (`<e(t)> - <e(t-theta)> >= 0.18`, Paper 12 Eq. 3 & Paper 13),
      - `plasticity_factor`: Non-saturating plasticity weight `min_floor + 4*p*(1-p)` (Paper 02 E3A fix: never 0 at extremes).
    """
    if len(errors) < 2:
        return 0.05, 0.0, 0.65
    arr = np.asarray(errors[-2 * window_half :], dtype=np.float64)
    mid = max(1, len(arr) // 2)
    pst = arr[:mid]
    rec = arr[mid:]
    mean_pst = float(np.mean(pst))
    mean_rec = float(np.mean(rec))
    std_rec = float(np.std(rec))

    raw_drop = mean_pst - mean_rec
    # Discount erratic high-variance noise and non-monotonic upward oscillations (Papers 01, 04, 05, 10)
    diffs = np.diff(arr)
    up_jumps = float(np.mean(np.maximum(0.0, diffs))) if len(diffs) > 0 else 0.0
    se_penalty = 0.0
    if mean_rec > 0.25:
        se_penalty += noise_discount * std_rec + 1.5 * up_jumps
    if mean_rec > 0.50 and float(np.min(rec)) > 0.45:
        se_penalty += 0.50 * (mean_rec - 0.45)
    lp_pos = max(0.0, raw_drop - se_penalty)

    # Paper 12 Eq. (3): |LP| vigilance when competence in a previously mastered region suddenly drops
    if mean_pst <= 0.20 and (mean_rec - mean_pst) >= 0.18:
        vigilance = abs(raw_drop)
    else:
        vigilance = 0.0

    # Paper 02 E3A: Plasticity factor bounded away from zero (`>= 0.15`) so severe errors are never ignored
    p_clamped = min(0.98, max(0.02, mean_rec))
    plasticity = max(0.15, 4.0 * p_clamped * (1.0 - p_clamped))
    return round(lp_pos, 5), round(vigilance, 5), round(plasticity, 4)


def compute_pomdp_voi(
    expected_error_yc: float,
    strategy_gain_rate: float,
    observation_cost_ro: float,
) -> dict[str, Any]:
    """Paper 06 (Gottlieb et al. 2013, Fig. 1A):
    - Prior certainty `p = 1.0 - min(0.5, expected_error_yc)` in `[0.50, 1.00]`.
    - Blind action value `V_blind(p) = p`.
    - Observing action value `V_obs(p) = p + (1 - p) * strategy_gain_rate - |r_o|`.
    - `VoI = V_obs(p) - V_blind(p) = (1 - p) * strategy_gain_rate - |r_o|`.
    - Indifference threshold `p* = 1.0 - |r_o| / max(1e-6, strategy_gain_rate)`.
    """
    err = float(np.clip(expected_error_yc, 0.0, 0.50))
    p_certainty = 1.0 - err
    cost = abs(float(observation_cost_ro))
    gain_rate = max(0.05, float(strategy_gain_rate))
    expected_reduction = err * gain_rate
    voi = expected_reduction - cost
    p_star = max(0.50, 1.0 - cost / gain_rate)
    return {
        "prior_certainty_p": round(p_certainty, 4),
        "indifference_threshold_p_star": round(p_star, 4),
        "observation_cost": round(cost, 4),
        "value_of_information_voi": round(voi, 4),
        "should_observe": bool(voi > 1e-6),
    }


def compute_neutral_diagnosticity(
    support_count: int,
    challenge_count: int,
    is_unverified_hypothesis: bool,
) -> float:
    """Paper 07 (Gottlieb & Oudeyer 2018, Fig. 3c vs Fig. 4a):
    Neutral diagnosticity / cue validity score that peaks when a hypothesis or claim has high epistemic uncertainty
    or conflicting evidence (`support` vs `challenge`), rather than inflating already-confirmed `supports` items.
    """
    tot = support_count + challenge_count
    if tot == 0:
        return 0.85 if is_unverified_hypothesis else 0.45
    p_pos = (support_count + 0.5) / (tot + 1.0)
    # Binary Shannon entropy of evidence balance in [0, 1]
    h_ent = -(p_pos * math.log2(p_pos) + (1.0 - p_pos) * math.log2(1.0 - p_pos))
    sample_factor = 1.0 / (1.0 + 0.15 * tot)
    base = 0.65 * h_ent + 0.35 * sample_factor
    if is_unverified_hypothesis:
        base = max(base, 0.60)
    return round(float(np.clip(base, 0.05, 1.0)), 4)


# ---------------------------------------------------------------------------
# 3. Persistent Curiosity Engine (`PersistentCuriosityEngine`)
# ---------------------------------------------------------------------------

class PersistentCuriosityEngine:
    """Persistent developmental curiosity engine integrated with `LayerKit`, `CloverMemoryGraphStore`,
    `CloverOntologyGraph`, `CloverJEPARuntime`, and `Repo` (`growth/curiosity/curiosity_state.json`).
    """

    def __init__(
        self,
        storage_path: str | Path,
        *,
        split_threshold: int = 12,
        habituation_decay: float = 0.72,
        habituation_weight: float = 0.55,
        zpd_stall_attempts: int = 3,
        zpd_stall_eps: float = 0.015,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.split_threshold = int(split_threshold)
        self.habituation_decay = float(habituation_decay)
        self.habituation_weight = float(habituation_weight)
        self.zpd_stall_attempts = int(zpd_stall_attempts)
        self.zpd_stall_eps = float(zpd_stall_eps)
        self.lock = threading.RLock()

        self.step_counter: int = 0
        self.regions: dict[str, CuriosityRegion] = {}
        self.unresolved_questions: dict[str, UnresolvedQuestionNode] = {}
        self.strategy_bandit: dict[str, dict[str, StrategyStats]] = {}
        self.exploration_history: list[dict[str, Any]] = []
        # Paper 13: 64-d action-controllability subspace weights R^2_j in [0, 1]
        self.controllable_subspace_r2: np.ndarray = np.ones(D_LATENT, dtype=np.float64)
        self._action_history_u: list[np.ndarray] = []
        self._state_delta_history: list[np.ndarray] = []

        self.load()

    # ---------------- Persistence (`growth/curiosity/curiosity_state.json`) ----------------

    def _ensure_root_region(self) -> None:
        if "R0" not in self.regions:
            self.regions["R0"] = CuriosityRegion(region_id="R0")
        if "R0" not in self.strategy_bandit:
            self.strategy_bandit["R0"] = {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}

    def load(self) -> bool:
        with self.lock:
            if not self.storage_path.is_file():
                self._ensure_root_region()
                return False
            try:
                doc = json.loads(self.storage_path.read_text(encoding="utf-8"))
                self.step_counter = int(doc.get("step_counter") or 0)
                self.regions = {
                    k: CuriosityRegion.from_dict(v) for k, v in (doc.get("regions") or {}).items()
                }
                self.unresolved_questions = {
                    k: UnresolvedQuestionNode.from_dict(v)
                    for k, v in (doc.get("unresolved_questions") or {}).items()
                }
                self.strategy_bandit = {}
                for r_id, s_map in (doc.get("strategy_bandit") or {}).items():
                    self.strategy_bandit[r_id] = {
                        s: StrategyStats.from_dict(s_map[s]) if s in s_map else StrategyStats(strategy=s)
                        for s in EPISTEMIC_STRATEGIES
                    }
                self.exploration_history = list(doc.get("exploration_history") or [])[-120:]
                r2_raw = doc.get("controllable_subspace_r2")
                if isinstance(r2_raw, list) and len(r2_raw) == D_LATENT:
                    self.controllable_subspace_r2 = np.asarray(r2_raw, dtype=np.float64)
                self._ensure_root_region()
                return True
            except Exception:
                self._ensure_root_region()
                return False

    def save(self) -> Path:
        with self.lock:
            self._ensure_root_region()
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now_iso(),
                "step_counter": self.step_counter,
                "controllable_subspace_r2": [round(float(x), 4) for x in self.controllable_subspace_r2],
                "regions": {k: v.to_dict() for k, v in self.regions.items()},
                "unresolved_questions": {k: v.to_dict() for k, v in self.unresolved_questions.items()},
                "strategy_bandit": {
                    r_id: {s: st.to_dict() for s, st in s_map.items()}
                    for r_id, s_map in self.strategy_bandit.items()
                },
                "exploration_history": self.exploration_history[-120:],
            }
            tmp = self.storage_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.storage_path)
            return self.storage_path

    # ---------------- Region Tree Routing & Criterion C2 Splitting (Paper 01) ----------------

    def leaf_regions(self) -> list[CuriosityRegion]:
        with self.lock:
            self._ensure_root_region()
            return [r for r in self.regions.values() if r.is_leaf]

    def locate_leaf_region(self, vec_64: np.ndarray | Sequence[float]) -> CuriosityRegion:
        """Route a 64-d vector `vec_64` down the `CuriosityRegionTree` to its unique leaf region."""
        with self.lock:
            self._ensure_root_region()
            v = np.asarray(vec_64, dtype=np.float64).reshape(-1)
            cur = self.regions["R0"]
            visited_guard: set[str] = set()
            while not cur.is_leaf and cur.region_id not in visited_guard:
                visited_guard.add(cur.region_id)
                j = cur.split_dim if cur.split_dim is not None else 0
                thresh = cur.split_threshold if cur.split_threshold is not None else 0.0
                next_id = cur.left_child_id if float(v[j]) <= thresh else cur.right_child_id
                if not next_id or next_id not in self.regions:
                    break
                cur = self.regions[next_id]
            return cur

    def _recompute_region_metrics(self, reg: CuriosityRegion) -> None:
        """Update `tonic_lpm`, `tonic_cpm`, `abs_lp_vigilance`, `habituation_trace`, and `state_curiosity`."""
        lpm, _, plast_p = compute_windowed_lp_and_vigilance(reg.pred_error_window)
        cpm, vig, plast_c = compute_windowed_lp_and_vigilance(reg.comp_error_window)
        reg.tonic_lpm = lpm
        reg.tonic_cpm = cpm
        reg.abs_lp_vigilance = vig

        # Paper 10 Section 4: norm of the smoothed signed shift vector ||E[Delta z | R_n]||
        shift_norm = float(np.linalg.norm(np.asarray(reg.signed_shift_ema_64, dtype=np.float64)))
        reliability_gain = max(0.0, reg.dot_yc_reliability)

        # Paper 04 & 13: Controllability gate prevents Passive Spectator Trap (`controllability_score == 0` -> suppressed)
        ctrl_gate = float(np.clip(reg.controllability_score, 0.02, 1.0))
        plasticity = 0.5 * (plast_p + plast_c)

        # Combine Knowledge Progress (LPM), Competence Progress (CPM), Reliability Improvement (dot_y_C), and |LP| Vigilance
        raw_interest = (
            0.30 * lpm
            + 0.45 * cpm
            + 0.15 * reliability_gain
            + 0.10 * min(0.25, shift_norm)
            + 0.65 * vig
        ) * (0.45 + 0.55 * plasticity) * ctrl_gate

        # Paper 02 E3B: Divide by habitation trace `1 + lambda_h * h_R(t)` so repeated visits cool down smoothly
        denom = 1.0 + self.habituation_weight * reg.habituation_trace
        reg.state_curiosity = round(float(raw_interest / denom), 5)

    def _maybe_split_region_c2(self, reg: CuriosityRegion) -> tuple[CuriosityRegion, CuriosityRegion] | None:
        """Paper 01 (Criterion C2): Split leaf region `reg` if it has accumulated `> self.split_threshold` exemplars
        and a split reduces weighted outcome variance by `>= 5%`.
        Crucial boundary fix (from Paper 01 E2): Children inherit the parent's partitioned exemplar histories,
        error windows, reliability state `expected_error_yc`, and strategy bandit stats so splitting NEVER
        manufactures artificial learning progress in an already-mastered region!
        """
        if not reg.is_leaf or len(reg.exemplar_vectors) <= self.split_threshold:
            return None
        # Prefer splitting along controllable dimensions (Paper 13)
        ctrl_dims = [int(i) for i in np.argsort(-self.controllable_subspace_r2)[:24]]
        split_res = find_criterion_c2_split(reg.exemplar_vectors, reg.exemplar_outcomes, candidate_dims=ctrl_dims)
        if split_res is None:
            return None
        j_star, v_star, _ = split_res
        left_id = f"{reg.region_id}.0"
        right_id = f"{reg.region_id}.1"
        if left_id in self.regions or right_id in self.regions:
            return None

        X = np.asarray(reg.exemplar_vectors, dtype=np.float64)
        Y = np.asarray(reg.exemplar_outcomes, dtype=np.float64)
        P_err = np.asarray(reg.pred_error_window[-len(Y) :], dtype=np.float64)
        C_err = np.asarray(reg.comp_error_window[-len(Y) :], dtype=np.float64)
        mask_l = X[:, j_star] <= v_star

        def _make_child(cid: str, mask: np.ndarray) -> CuriosityRegion:
            sub_x = X[mask]
            sub_y = Y[mask]
            sub_p = P_err[mask] if len(P_err) == len(mask) else sub_y
            sub_c = C_err[mask] if len(C_err) == len(mask) else sub_y
            cent = np.mean(sub_x, axis=0) if len(sub_x) > 0 else np.asarray(reg.centroid_64, dtype=np.float64)
            c_norm = float(np.linalg.norm(cent))
            if c_norm > 1e-9:
                cent = cent / c_norm
            # Inherit partitioned outcome errors and parent's reliability baseline (no artificial reset!)
            inherited_p = [float(v) for v in sub_p[-16:]] if len(sub_p) >= 2 else list(reg.pred_error_window[-8:])
            inherited_c = [float(v) for v in sub_c[-16:]] if len(sub_c) >= 2 else list(reg.comp_error_window[-8:])
            child = CuriosityRegion(
                region_id=cid,
                parent_id=reg.region_id,
                centroid_64=[round(float(v), 5) for v in cent],
                emergent_keywords=list(reg.emergent_keywords[:8]),
                exemplar_vectors=[[round(float(v), 5) for v in row] for row in sub_x[-24:]],
                exemplar_outcomes=[round(float(v), 5) for v in sub_y[-24:]],
                pred_error_window=inherited_p,
                comp_error_window=inherited_c,
                expected_error_yc=float(np.mean(inherited_c)) if inherited_c else reg.expected_error_yc,
                prev_expected_error_yc=reg.expected_error_yc,
                signed_shift_ema_64=list(reg.signed_shift_ema_64),
                controllability_score=reg.controllability_score,
                controllable_Try_count=reg.controllable_Try_count,
                uncontrollable_Try_count=reg.uncontrollable_Try_count,
                habituation_trace=reg.habituation_trace,
                visits=int(np.sum(mask)),
                last_visited_step=reg.last_visited_step,
            )
            self._recompute_region_metrics(child)
            return child

        left_child = _make_child(left_id, mask_l)
        right_child = _make_child(right_id, ~mask_l)

        reg.split_dim = j_star
        reg.split_threshold = v_star
        reg.left_child_id = left_id
        reg.right_child_id = right_id

        self.regions[left_id] = left_child
        self.regions[right_id] = right_child
        parent_bandit = self.strategy_bandit.get(reg.region_id) or {
            s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES
        }
        self.strategy_bandit[left_id] = copy.deepcopy(parent_bandit)
        self.strategy_bandit[right_id] = copy.deepcopy(parent_bandit)

        # Re-home any unresolved questions that belonged to `reg` into the appropriate child leaf
        for q_node in self.unresolved_questions.values():
            if q_node.region_id == reg.region_id:
                q_vec = encode_text_to_64d(q_node.summary)
                q_node.region_id = left_id if float(q_vec[j_star]) <= v_star else right_id

        return left_child, right_child

    # ---------------- Paper 13: Action-Controllable Subspace Estimation ----------------

    def update_controllable_subspace(
        self,
        action_vec_8: np.ndarray | Sequence[float],
        state_delta_64: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Paper 13 (Santucci et al. 2020 IMOL): Estimate per-dimension controllability `R^2_j(Action -> Delta z_j)`
        across the 64-d latent space so goal similarity and splitting prioritize controllable dimensions over
        uncontrollable background distractors.
        """
        with self.lock:
            u = np.asarray(action_vec_8, dtype=np.float64).reshape(-1)
            dz = np.asarray(state_delta_64, dtype=np.float64).reshape(-1)
            if u.size == 0 or dz.size != D_LATENT:
                return self.controllable_subspace_r2
            self._action_history_u.append(u)
            self._state_delta_history.append(dz)
            if len(self._action_history_u) > 64:
                self._action_history_u = self._action_history_u[-64:]
                self._state_delta_history = self._state_delta_history[-64:]

            if len(self._action_history_u) >= 10:
                U = np.asarray(self._action_history_u, dtype=np.float64)
                S = np.asarray(self._state_delta_history, dtype=np.float64)
                n_obs, p_dim = U.shape
                gram = U.T @ U + 0.5 * np.eye(p_dim, dtype=np.float64)
                r2 = np.zeros(D_LATENT, dtype=np.float64)
                for j in range(D_LATENT):
                    var_j = float(np.var(S[:, j]))
                    if var_j < 1e-8:
                        r2[j] = 0.05
                        continue
                    w = np.linalg.solve(gram, U.T @ S[:, j])
                    pred_j = U @ w
                    res_var = float(np.var(S[:, j] - pred_j))
                    raw_r2 = 1.0 - res_var / var_j
                    # Adjusted R^2 penalizing degrees of freedom p_dim so small-sample noise does not overfit
                    dof_denom = max(1, n_obs - p_dim - 1)
                    adj_r2 = 1.0 - (1.0 - raw_r2) * ((n_obs - 1.0) / dof_denom)
                    r2[j] = float(np.clip(adj_r2, 0.05, 1.0))
                self.controllable_subspace_r2 = 0.2 * self.controllable_subspace_r2 + 0.8 * r2
            return self.controllable_subspace_r2

    def controllable_distance(self, vec_a: np.ndarray | Sequence[float], vec_b: np.ndarray | Sequence[float]) -> float:
        """Weighted Euclidean distance in 64-d space weighted by `self.controllable_subspace_r2` (Paper 13)."""
        a = np.asarray(vec_a, dtype=np.float64).reshape(D_LATENT)
        b = np.asarray(vec_b, dtype=np.float64).reshape(D_LATENT)
        w = self.controllable_subspace_r2 / max(1e-9, float(np.mean(self.controllable_subspace_r2)))
        return float(np.sqrt(np.sum(w * ((a - b) ** 2))))

    # ---------------- Hook A: Pre-Turn Meaning Bridge, VoI & Reliability-Weighted Retrieval ----------------

    def recommend_strategy_and_voi(
        self,
        query: str,
        *,
        latent_cue_64: np.ndarray | Sequence[float] | None = None,
        discernment_state: str = "proceed",
        declined_capabilities: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Hook A (`before_turn` / `LayerKit.compute_meaning_bridge`):
        1. Respects Paper 03 / `AGENTS.md` §9 Discernment Hierarchy: if `discernment_state` is `"refuse"` or `"ask"`,
           curiosity never overrides safety/authority boundaries.
        2. Locates the active leaf region `R_n` for `query`.
        3. Uses the `SGIM-ACTS` bandit (Paper 11) + POMDP `VoI` (Paper 06) to select the highest-value epistemic
           strategy and produces an 8-d belief-based utility boost for `action_prior_8` (Paper 07 Fig. 1b).
        """
        with self.lock:
            q_vec = encode_text_to_64d(query, latent_cue_64=latent_cue_64)
            reg = self.locate_leaf_region(q_vec)
            if discernment_state in ("refuse", "ask", "do_not_know"):
                return {
                    "region_id": reg.region_id,
                    "preempted_by_discernment": discernment_state,
                    "selected_strategy": "human_clarification_ask" if discernment_state != "refuse" else "none",
                    "voi": 0.0,
                    "should_observe": False,
                    "belief_action_boost_8": np.zeros(8, dtype=np.float32),
                    "active_zpd_questions": [],
                }

            bandit = self.strategy_bandit.setdefault(
                reg.region_id, {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}
            )
            declined = set(declined_capabilities or ())
            best_strat = "memory_graph_recall"
            best_score = -1e9
            best_voi_info: dict[str, Any] = {}

            for strat in EPISTEMIC_STRATEGIES:
                if strat == "web_primary_research" and ("web.search" in declined or "web.page" in declined):
                    continue
                if strat == "sandbox_counterexample_probe" and "python" in declined:
                    continue
                st = bandit[strat]
                # Paper 11 SGIM-ACTS: empirical LP minus variance penalty
                gain_rate = max(0.05, st.empirical_lp * 3.5)
                voi_info = compute_pomdp_voi(
                    reg.expected_error_yc,
                    strategy_gain_rate=gain_rate,
                    observation_cost_ro=STRATEGY_OBSERVATION_COST[strat],
                )
                # Combine empirical LP with VoI and small exploration bonus for untried strategies
                score = st.empirical_lp + 0.5 * voi_info["value_of_information_voi"] - 0.15 * st.outcome_variance
                if score > best_score:
                    best_score = score
                    best_strat = strat
                    best_voi_info = voi_info

            boost_8 = np.zeros(8, dtype=np.float32)
            if best_voi_info.get("should_observe"):
                dim_idx = STRATEGY_TO_ACTION_DIM.get(best_strat, 3)
                boost_8[dim_idx] = np.float32(min(0.65, 0.25 + 2.0 * reg.state_curiosity))
                if best_strat == "sandbox_counterexample_probe":
                    boost_8[1] = np.float32(0.20)  # math_calc synergy

            open_qs = [
                q.to_dict()
                for q in self.unresolved_questions.values()
                if q.region_id == reg.region_id and q.zpd_status == "active_zpd"
            ]
            open_qs.sort(key=lambda d: -float(d.get("diagnosticity_score", 0.5)))

            return {
                "region_id": reg.region_id,
                "emergent_keywords": list(reg.emergent_keywords[:6]),
                "state_curiosity": reg.state_curiosity,
                "expected_error_yc": round(reg.expected_error_yc, 4),
                "tonic_lpm": reg.tonic_lpm,
                "tonic_cpm": reg.tonic_cpm,
                "abs_lp_vigilance": reg.abs_lp_vigilance,
                "selected_strategy": best_strat,
                "voi_analysis": best_voi_info,
                "should_observe": bool(best_voi_info.get("should_observe")),
                "belief_action_boost_8": boost_8,
                "active_zpd_questions": open_qs[:3],
            }

    @staticmethod
    def reliability_weighted_rank(
        candidates: Sequence[dict[str, Any]],
        alpha_variance: float = 0.50,
    ) -> list[dict[str, Any]]:
        """Paper 12 (`SGIM-D` Algorithm 4.6, Section 4.4.2):
        Rank retrieved memory/concept exemplars by minimizing `dist(tau_h, tau_g) + alpha * var_h`
        where `var_h` is the outcome variance of exemplar `h` (`worked` vs `did_not_work`).
        """
        scored = []
        for item in candidates:
            dist = float(item.get("distance", 1.0 - float(item.get("score", 0.5))))
            w = int(item.get("worked", 0))
            d = int(item.get("did_not_work", 0))
            tot = w + d
            if tot == 0:
                var_h = float(item.get("variance", 0.04))
            else:
                p_fail = d / tot
                # Bernoulli outcome variance p*(1-p) + failure penalty when d > w
                var_h = p_fail * (1.0 - p_fail) + (0.35 if d > w else 0.0)
            rel_cost = dist + alpha_variance * var_h
            scored.append((rel_cost, dict(item, reliability_cost=round(rel_cost, 4), outcome_variance=round(var_h, 4))))
        scored.sort(key=lambda pair: pair[0])
        return [ pair[1] for pair in scored ]

    # ---------------- Hook B: Post-Turn Feedback (`on_turn_outcome`) ----------------

    def record_interaction_outcome(
        self,
        query: str,
        *,
        pred_error: float,
        comp_error: float,
        strategy_used: str = "memory_graph_recall",
        actor: str = "clover",
        controllable: bool = True,
        latent_before_64: np.ndarray | Sequence[float] | None = None,
        latent_after_64: np.ndarray | Sequence[float] | None = None,
        unresolved_summary: str | None = None,
        memory_entry_id: str | None = None,
        parent_question_id: str | None = None,
        prerequisite_ids: Sequence[str] = (),
        lateral_modality_ids: Sequence[str] = (),
        view_dimension_added: str | None = None,
        session_id: str = "",
        turn_n: int = 0,
    ) -> dict[str, Any]:
        """Hook B (`on_turn_outcome`): Update `CuriosityRegionTree`, Confidence Module `C(R_n)`, `SGIM-ACTS` strategy
        bandit, and `ZPDES` unresolved question frontier from an interaction outcome.
        """
        with self.lock:
            self.step_counter += 1
            q_vec = encode_text_to_64d(query, latent_cue_64=latent_before_64)

            # Decay habitation traces across all regions before boosting the visited region (Paper 02 E3B)
            for r in self.regions.values():
                r.habituation_trace = round(r.habituation_trace * self.habituation_decay, 5)
                if r.is_leaf and r.visits > 0:
                    self._recompute_region_metrics(r)

            reg = self.locate_leaf_region(q_vec)
            reg.visits += 1
            reg.last_visited_step = self.step_counter
            reg.habituation_trace = round(reg.habituation_trace + 1.0, 5)
            reg.updated_at = utc_now_iso()

            # Competitive Hebbian prototype adaptation (`MR` microcolumn receptive field, Paper 05)
            cent = np.asarray(reg.centroid_64, dtype=np.float64)
            if reg.visits == 1 or float(np.linalg.norm(cent)) < 1e-9:
                cent = q_vec.copy()
            else:
                cent = 0.88 * cent + 0.12 * q_vec
                cent /= max(1e-9, float(np.linalg.norm(cent)))
            reg.centroid_64 = [round(float(x), 5) for x in cent]

            # Update emergent keywords from observed query terms (zero hardcoded topics)
            new_terms = extract_emergent_terms(query, max_terms=6)
            merged_kw = list(dict.fromkeys(new_terms + reg.emergent_keywords))[:10]
            reg.emergent_keywords = merged_kw

            # Record exemplar for Criterion C2 splitting (Paper 01)
            combined_err = float(np.clip(0.4 * pred_error + 0.6 * comp_error, 0.0, 1.5))
            reg.exemplar_vectors.append([round(float(x), 5) for x in q_vec])
            reg.exemplar_outcomes.append(round(combined_err, 5))
            reg.pred_error_window.append(round(float(pred_error), 5))
            reg.comp_error_window.append(round(float(comp_error), 5))

            # Paper 04 & 13: Update regional controllability score
            if controllable:
                reg.controllable_Try_count += 1
            else:
                reg.uncontrollable_Try_count += 1
            tot_ctrl = reg.controllable_Try_count + reg.uncontrollable_Try_count
            reg.controllability_score = round(reg.controllable_Try_count / max(1, tot_ctrl), 4)

            # Paper 04 Eq. (15): Phasic predictive surprise ratio SM(t) = Er(t) / <Er_R>
            reg.phasic_surprise_sm = round(combined_err / max(0.02, reg.expected_error_yc), 4)

            # Paper 10 (Schmidhuber 1991 Section 2 & 3): Confidence Module C(R_n) & signed reliability derivative dot_y_C
            yc_old = reg.expected_error_yc
            eta_c = 0.22
            yc_new = (1.0 - eta_c) * yc_old + eta_c * combined_err
            reg.prev_expected_error_yc = round(yc_old, 5)
            reg.expected_error_yc = round(yc_new, 5)
            reg.dot_yc_reliability = round(yc_old - yc_new, 5)

            # Paper 10 Section 4: Smoothed signed shift vector E[Delta z | R_n] (Jensen's inequality guard)
            if latent_before_64 is not None and latent_after_64 is not None:
                zb = np.asarray(latent_before_64, dtype=np.float64).reshape(D_LATENT)
                za = np.asarray(latent_after_64, dtype=np.float64).reshape(D_LATENT)
                dz = za - zb
                cur_ema = np.asarray(reg.signed_shift_ema_64, dtype=np.float64)
                new_ema = 0.82 * cur_ema + 0.18 * dz
                reg.signed_shift_ema_64 = [round(float(x), 5) for x in new_ema]

            prev_curiosity = reg.state_curiosity
            self._recompute_region_metrics(reg)

            # Update SGIM-ACTS strategy bandit (Paper 11)
            bandit = self.strategy_bandit.setdefault(
                reg.region_id, {s: StrategyStats(strategy=s) for s in EPISTEMIC_STRATEGIES}
            )
            s_key = strategy_used if strategy_used in bandit else "memory_graph_recall"
            st = bandit[s_key]
            st.attempts += 1
            st.last_used_step = self.step_counter
            prev_s_err = st.error_window[-1] if st.error_window else 0.55
            st.error_window.append(round(float(comp_error), 5))
            s_prog = max(0.0, prev_s_err - float(comp_error)) if controllable else 0.0
            st.progress_window.append(round(s_prog, 5))
            if view_dimension_added:
                st.views_discovered += 1

            # Update or spawn ZPDES unresolved question node (Papers 06, 07, 08, 12)
            spawned_qid = None
            resolved_qids: list[str] = []
            if unresolved_summary or comp_error > 0.25 or reg.abs_lp_vigilance > 0.0:
                q_text = (unresolved_summary or query).strip()[:240]
                qid = "Q-" + hashlib.sha1(q_text.lower().encode("utf-8")).hexdigest()[:10].upper()
                q_node = self.unresolved_questions.get(qid)
                if q_node is None:
                    origin = (
                        "competence_regression"
                        if reg.abs_lp_vigilance > 0.0
                        else ("child_frontier" if parent_question_id else "turn_discrepancy")
                    )
                    q_node = UnresolvedQuestionNode(
                        question_id=qid,
                        region_id=reg.region_id,
                        summary=q_text,
                        origin_kind=origin,
                        memory_entry_id=memory_entry_id,
                        parent_question_id=parent_question_id,
                        prerequisite_ids=list(prerequisite_ids),
                        lateral_modality_ids=list(lateral_modality_ids),
                        zpd_status="prerequisite_pending" if prerequisite_ids else "active_zpd",
                    )
                    self.unresolved_questions[qid] = q_node
                    spawned_qid = qid
                q_node.attempts += 1
                q_node.error_history.append(round(float(comp_error), 4))
                q_node.updated_at = utc_now_iso()
                if view_dimension_added and view_dimension_added not in q_node.views_covered:
                    q_node.views_covered.append(view_dimension_added)

                if comp_error <= 0.10:
                    q_node.zpd_status = "resolved_verified"
                    resolved_qids.append(qid)
                    self._on_question_progressed_or_resolved(q_node)
                elif (
                    q_node.attempts >= self.zpd_stall_attempts
                    and q_node.recent_lp <= self.zpd_stall_eps
                    and comp_error > 0.30
                ):
                    # Paper 08 Figure 5 (ZPDES): stall detected -> deactivate stalled node and open lateral/prerequisite nodes
                    q_node.zpd_status = "stalled_deactivated"
                    for lat_id in list(q_node.lateral_modality_ids) + list(q_node.prerequisite_ids):
                        if lat_id in self.unresolved_questions:
                            lat_q = self.unresolved_questions[lat_id]
                            if lat_q.zpd_status in ("prerequisite_pending", "stalled_deactivated"):
                                lat_q.zpd_status = "active_zpd"
                elif q_node.recent_lp > self.zpd_stall_eps:
                    self._on_question_progressed_or_resolved(q_node)

            # Check Criterion C2 region split (Paper 01)
            split_children = self._maybe_split_region_c2(reg)

            episode = {
                "step": self.step_counter,
                "at": utc_now_iso(),
                "session": session_id,
                "turn": turn_n,
                "region_id": reg.region_id,
                "strategy": s_key,
                "actor": actor,
                "pred_error": round(float(pred_error), 4),
                "comp_error": round(float(comp_error), 4),
                "phasic_surprise_sm": reg.phasic_surprise_sm,
                "tonic_lpm": reg.tonic_lpm,
                "tonic_cpm": reg.tonic_cpm,
                "dot_yc_reliability": reg.dot_yc_reliability,
                "abs_lp_vigilance": reg.abs_lp_vigilance,
                "state_curiosity_before": prev_curiosity,
                "state_curiosity_after": reg.state_curiosity,
                "split_into": [split_children[0].region_id, split_children[1].region_id] if split_children else None,
                "spawned_question_id": spawned_qid,
                "resolved_question_ids": resolved_qids,
            }
            self.exploration_history.append(episode)
            self.save()
            return episode

    def _on_question_progressed_or_resolved(self, progressed_node: UnresolvedQuestionNode) -> None:
        """Paper 08 Fig. 5 (`ZPDES`): When a prerequisite or lateral modality question progresses or resolves,
        re-activate any stalled or pending questions that depended on it.
        """
        qid = progressed_node.question_id
        for other in self.unresolved_questions.values():
            if other.question_id == qid or other.zpd_status == "resolved_verified":
                continue
            if qid in other.prerequisite_ids or qid in other.lateral_modality_ids or other.question_id in progressed_node.lateral_modality_ids:
                if other.zpd_status in ("stalled_deactivated", "prerequisite_pending"):
                    other.zpd_status = "active_zpd"
                    # Reset recent stall window so it gets a fresh evaluation with the newly acquired skill
                    if len(other.error_history) >= 2:
                        other.error_history = [other.error_history[-1]]

    # ---------------- Candidate Selection Without Hardcoded Lists or `solved_ids` Crutch ----------------

    def rank_candidates_by_curiosity(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank candidate problems, reading units, or topics by developmental curiosity (Papers 01, 02, 04, 08, 10, 12)
        WITHOUT using a hardcoded order or a binary `solved_ids` exclusion set:
          - Evaluates each candidate's leaf region `R_n` in `CuriosityRegionTree`
          - Combines `state_curiosity` (`LPM + CPM + dot_y_C + |LP| vigilance`, cooled by `habituation_trace`)
            with `ZPD` readiness (penalizing candidates whose prerequisites are still unmastered or whose region
            has already plateaued at near-zero error).
        """
        with self.lock:
            scored: list[tuple[float, dict[str, Any]]] = []
            for cand in candidates:
                text = str(cand.get("text") or cand.get("question") or cand.get("title") or "")
                cue = cand.get("cue_64")
                vec = encode_text_to_64d(text, latent_cue_64=cue)
                reg = self.locate_leaf_region(vec)

                # Check if candidate has an associated question node in ZPD
                qid = cand.get("question_id")
                zpd_mult = 1.0
                if qid and qid in self.unresolved_questions:
                    q_node = self.unresolved_questions[qid]
                    if q_node.zpd_status == "stalled_deactivated":
                        zpd_mult = 0.15
                    elif q_node.zpd_status == "prerequisite_pending":
                        # Check if all prerequisites are now resolved
                        if all(
                            p in self.unresolved_questions
                            and self.unresolved_questions[p].zpd_status == "resolved_verified"
                            for p in q_node.prerequisite_ids
                        ):
                            q_node.zpd_status = "active_zpd"
                            zpd_mult = 1.35
                        else:
                            zpd_mult = 0.20
                    elif q_node.zpd_status == "resolved_verified":
                        zpd_mult = 0.05 if reg.abs_lp_vigilance == 0.0 else 1.50
                    else:
                        zpd_mult = 1.25 * (0.5 + 0.5 * q_node.diagnosticity_score)

                # If region is already mastered (expected_error_yc < 0.06) and has zero vigilance spike, interest is low
                mastery_gate = 0.08 if (reg.expected_error_yc < 0.06 and reg.abs_lp_vigilance == 0.0 and reg.visits >= 2) else 1.0
                # Within-region exemplar habitation cooldown (Paper 01 & Paper 02 E3B):
                # Cools down specific stimuli recently repeated inside R_n so diverse points in R_n are sampled
                local_hab = 0.0
                if reg.exemplar_vectors:
                    recent_vecs = np.asarray(reg.exemplar_vectors[-12:], dtype=np.float64)
                    sims = np.maximum(0.0, recent_vecs @ vec)
                    weights = np.array([self.habituation_decay ** (len(sims) - 1 - i) for i in range(len(sims))], dtype=np.float64)
                    local_hab = float(np.sum(weights * (sims ** 4)))
                local_cooldown = 1.0 / (1.0 + self.habituation_weight * local_hab)
                score = reg.state_curiosity * zpd_mult * mastery_gate * local_cooldown
                scored.append((
                    score,
                    dict(
                        cand,
                        region_id=reg.region_id,
                        curiosity_score=round(float(score), 5),
                        tonic_lpm=reg.tonic_lpm,
                        tonic_cpm=reg.tonic_cpm,
                        expected_error_yc=reg.expected_error_yc,
                        habituation_trace=reg.habituation_trace,
                        controllability=reg.controllability_score,
                    ),
                ))
            scored.sort(key=lambda pair: -pair[0])
            return [pair[1] for pair in scored]

    # ---------------- Hook C: Idle Background Consolidation & Non-Regression Replay (Papers 08 & 13) ----------------

    def run_idle_consolidation_step(
        self,
        memory_store: Any | None = None,
        ontology_graph: Any | None = None,
        sandbox_runner: Callable[[str], tuple[bool, Any, str]] | None = None,
        regression_tests: Sequence[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        """Hook C (`on_idle_step` for `studio_idle()`):
        1. Syncs any `unresolved`, `hypothesis`, or `contested` entries from `CloverMemoryGraphStore` into `self.unresolved_questions`
           with neutral diagnosticity scores (Paper 07).
        2. Selects the highest-`state_curiosity` active ZPD region/question (closing Fig. 1B of Paper 08:
           `LP -> State Curiosity -> Memory Consolidation Priority`).
        3. If a candidate hypothesis has a verification script (`sandbox_counterexample_probe`), executes it in
           `IsolatedPythonSandbox` AND runs **non-regression replay** (`regression_tests`, Paper 13) before promoting
           the hypothesis to an active `externally_verified` concept. If any regression test fails, promotion is
           rolled back and the hypothesis is marked `contested`!
        """
        with self.lock:
            synced_from_memory = 0
            if memory_store is not None and hasattr(memory_store, "entries"):
                for e in list(memory_store.entries.values()):
                    is_open = (
                        getattr(e, "epistemic_status", "") in ("unresolved", "hypothesis")
                        or getattr(e, "status", "") in ("contested", "under_review")
                    )
                    if not is_open:
                        continue
                    qid = f"Q-MEM-{e.id}"
                    if qid not in self.unresolved_questions:
                        vec = encode_text_to_64d(f"{e.title or ''} {e.summary or ''}")
                        reg = self.locate_leaf_region(vec)
                        rels = getattr(e, "relationships", ()) or ()
                        sup_c = sum(1 for r in rels if getattr(r, "type", "") == "supports")
                        chal_c = sum(1 for r in rels if getattr(r, "type", "") in ("challenges", "unresolved_with"))
                        prereqs = [f"Q-MEM-{r.target}" for r in rels if getattr(r, "type", "") == "depends_on"]
                        diag = compute_neutral_diagnosticity(
                            sup_c, chal_c, is_unverified_hypothesis=(getattr(e, "entry_type", "") == "hypothesis")
                        )
                        self.unresolved_questions[qid] = UnresolvedQuestionNode(
                            question_id=qid,
                            region_id=reg.region_id,
                            summary=str(e.summary or e.title or e.id)[:240],
                            origin_kind="unverified_hypothesis" if getattr(e, "entry_type", "") == "hypothesis" else "memory_contradiction",
                            memory_entry_id=e.id,
                            prerequisite_ids=prereqs,
                            diagnosticity_score=diag,
                        )
                        synced_from_memory += 1

            # Rank active ZPD questions by regional state_curiosity * diagnosticity (Fig. 1B of Paper 08)
            active_qs = [
                q for q in self.unresolved_questions.values() if q.zpd_status == "active_zpd"
            ]
            active_qs.sort(
                key=lambda q: -(
                    self.regions.get(q.region_id, self.regions["R0"]).state_curiosity * (0.5 + 0.5 * q.diagnosticity_score)
                )
            )
            top_q = active_qs[0] if active_qs else None

            consolidation_result: dict[str, Any] = {
                "synced_open_memory_entries": synced_from_memory,
                "active_zpd_count": len(active_qs),
                "selected_question_id": top_q.question_id if top_q else None,
                "selected_region_id": top_q.region_id if top_q else "R0",
                "promoted": False,
                "rolled_back_on_regression": False,
            }

            if top_q and sandbox_runner is not None and memory_store is not None and top_q.memory_entry_id:
                e_hyp = memory_store.entries.get(top_q.memory_entry_id)
                v_code = getattr(e_hyp, "verification_code", None) if e_hyp else None
                if not v_code and e_hyp:
                    m_code = re.search(r"```python\s*\n([\s\S]*?)```", str(getattr(e_hyp, "content", "") or ""))
                    if m_code:
                        v_code = m_code.group(1).strip()
                if v_code:
                    ok_new, val_new, _ = sandbox_runner(v_code)
                    passed_new = bool(ok_new and (val_new is True or (isinstance(val_new, dict) and val_new.get("passed") is True)))
                    # Paper 13: Run non-regression replay on existing validated concepts before allowing promotion
                    reg_ok = True
                    for rt in regression_tests or ():
                        ok_r, val_r, _ = sandbox_runner(str(rt.get("code") or ""))
                        if not (ok_r and (val_r is True or (isinstance(val_r, dict) and val_r.get("passed") is True))):
                            reg_ok = False
                            break
                    if passed_new and reg_ok and e_hyp is not None:
                        e_hyp.epistemic_status = "externally_verified"
                        e_hyp.status = "active"
                        if hasattr(memory_store, "_save_entry_in_place"):
                            memory_store._save_entry_in_place(e_hyp)
                        top_q.zpd_status = "resolved_verified"
                        self._on_question_progressed_or_resolved(top_q)
                        consolidation_result["promoted"] = True
                    elif passed_new and not reg_ok and e_hyp is not None:
                        e_hyp.status = "contested"
                        if hasattr(memory_store, "_save_entry_in_place"):
                            memory_store._save_entry_in_place(e_hyp)
                        consolidation_result["rolled_back_on_regression"] = True

            self.save()
            return consolidation_result

    # ---------------- Human-Readable Summary for `clover_repo.py` (`CONTEXT.md`) ----------------

    def summary_markdown(self) -> tuple[str, list[str]]:
        """Return `(headline, markdown_lines)` for `clover_repo.py` `_growth_summary()`."""
        with self.lock:
            leaves = self.leaf_regions()
            leaves.sort(key=lambda r: -r.state_curiosity)
            open_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "active_zpd"]
            stalled_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "stalled_deactivated"]
            resolved_zpd = [q for q in self.unresolved_questions.values() if q.zpd_status == "resolved_verified"]

            head = (
                f"{len(leaves)} emergent interest regions, {len(open_zpd)} active ZPD questions "
                f"({len(stalled_zpd)} stalled/rerouted, {len(resolved_zpd)} resolved)"
            )
            lines = [
                "## Curiosity & Developmental Frontier",
                "",
                f"- {head}; {self.step_counter} exploration episodes logged.",
            ]
            for r in leaves[:6]:
                kws = ", ".join(r.emergent_keywords[:5]) or "initial root region"
                lines.append(
                    f"- Region `{r.region_id}` ({kws}): curiosity `{r.state_curiosity:.4f}`, "
                    f"LPM `{r.tonic_lpm:.4f}`, CPM `{r.tonic_cpm:.4f}`, expected error `{r.expected_error_yc:.3f}`, "
                    f"controllability `{r.controllability_score:.2f}`, visits `{r.visits}`"
                )
            if open_zpd:
                lines.append("")
                lines.append("### Active Unresolved Questions (ZPD)")
                for q in open_zpd[:6]:
                    lines.append(
                        f"- `{q.question_id}` (`{q.region_id}`, {q.origin_kind}, diagnosticity `{q.diagnosticity_score:.2f}`): "
                        f"{q.summary[:140]}"
                    )
            lines.append("")
            return head, lines


_ENGINES_BY_PATH: dict[str, PersistentCuriosityEngine] = {}
_ENGINES_LOCK = threading.Lock()


def for_path(storage_path: str | Path) -> PersistentCuriosityEngine:
    p = str(Path(storage_path).resolve())
    with _ENGINES_LOCK:
        if p not in _ENGINES_BY_PATH:
            _ENGINES_BY_PATH[p] = PersistentCuriosityEngine(Path(p))
        return _ENGINES_BY_PATH[p]
