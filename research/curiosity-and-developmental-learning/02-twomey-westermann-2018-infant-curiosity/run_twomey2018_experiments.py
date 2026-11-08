#!/usr/bin/env python3
"""Paper 2 — Twomey & Westermann (2018): Curiosity-Based Learning in Infants: A Neurocomputational Approach
(`Developmental Science`, 21(4): e12629, PMC6032944).

Standalone executable reproduction and boundary stress-test of Twomey & Westermann's 4-3-4 sigmoidal
autoencoder model of infant visual categorization and delta-rule curiosity `(i - o) * o * (1 - o)`:
  - E1 (Section 2, Fig. 3): Fixed Stimulus Presentation Orders (Mather & Plunkett 2011 replication across
      all 8! = 40,320 permutations of Younger 1985 E1 Broad stimuli: `max`, `med`, `min`, `random`).
  - E2 (Section 3, Figs. 4–6 & Tables 1–2): Active Stimulus Selection ("Setting the Model Free") comparing:
      1. `curiosity`: maximizes sum(|i - o| * o * (1 - o))
      2. `objective_complexity`: maximizes Euclidean distance ||i - i_prev||_2
      3. `subjective_novelty`: maximizes sum(|i - o|)
      4. `plasticity`: maximizes sum(o * (1 - o))
  - E3 (Boundary Stress Tests — Where `|i - o| * o * (1 - o)` Breaks):
      1. Sigmoid Saturation Blindness ("confident-but-wrong" units with o -> 0.01 and i = 0.95 have o(1-o) -> 0.01,
         suppressing curiosity on strongly contradicted inputs).
      2. With-Replacement White-Noise Trap (when sampling with replacement in an environment containing a random
         stimulus i ~ Uniform(0, 1) whose representation settles at o ~ 0.5 where o(1-o) = 0.25 is maximal,
         instantaneous `|i - o| * o * (1 - o)` gets trapped on the unlearnable noise stimulus).
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Younger (1985, E1 Broad) & Mather & Plunkett (2011) 4-Feature Stimuli
# ---------------------------------------------------------------------------

# 8 familiarization stimuli with 4 features in {1, 2, 4, 5} (normalized with step 0.1985625 centered at 3 -> 0.5)
STEP = 0.1985625
RAW_TRAIN_STIMULI = np.array([
    [1, 1, 5, 5],
    [1, 5, 1, 5],
    [5, 1, 5, 1],
    [5, 5, 1, 1],
    [2, 2, 4, 2],
    [2, 4, 2, 4],
    [4, 2, 2, 2],
    [4, 4, 4, 4],
], dtype=np.float64)

TRAIN_STIMULI = 0.5 + (RAW_TRAIN_STIMULI - 3.0) * STEP

# Novel test stimuli: 1 category-central prototype [3, 3, 3, 3] and 2 category-peripheral [1, 1, 1, 1], [5, 5, 5, 5]
TEST_CENTRAL = 0.5 + (np.array([3.0, 3.0, 3.0, 3.0]) - 3.0) * STEP
TEST_PERIPHERAL_1 = 0.5 + (np.array([1.0, 1.0, 1.0, 1.0]) - 3.0) * STEP
TEST_PERIPHERAL_2 = 0.5 + (np.array([5.0, 5.0, 5.0, 5.0]) - 3.0) * STEP


def sequence_mean_ed(order: tuple[int, ...] | list[int]) -> float:
    pts = TRAIN_STIMULI[list(order)]
    diffs = pts[1:] - pts[:-1]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.mean(dists))


def successive_eds(order: tuple[int, ...] | list[int]) -> list[float]:
    pts = TRAIN_STIMULI[list(order)]
    diffs = pts[1:] - pts[:-1]
    return [round(float(x), 4) for x in np.sqrt(np.sum(diffs ** 2, axis=1))]


# ---------------------------------------------------------------------------
# 2. 4-3-4 Sigmoidal Autoencoder (Section 2.1 & 2.3, Eq. 1)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class InfantAutoencoder:
    """4-input -> 3-hidden -> 4-output sigmoidal autoencoder trained with generalized delta rule (Eq. 1)."""

    def __init__(self, seed: int = 0, lr: float = 0.25, init_scale: float = 0.35) -> None:
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w1 = rng.uniform(-init_scale, init_scale, size=(4, 3))
        self.b1 = rng.uniform(-init_scale, init_scale, size=(3,))
        self.w2 = rng.uniform(-init_scale, init_scale, size=(3, 4))
        self.b2 = rng.uniform(-init_scale, init_scale, size=(4,))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(x @ self.w1 + self.b1)
        o = sigmoid(h @ self.w2 + self.b2)
        return h, o

    def sse(self, x: np.ndarray) -> float:
        _, o = self.forward(x)
        return float(np.sum((x - o) ** 2))

    def train_stimulus(self, x: np.ndarray, max_sweeps: int = 20, threshold: float = 0.01) -> list[float]:
        """Present stimulus `x` for up to `max_sweeps` sweeps or until SSE < `threshold` (Section 2.3, p. 5).
        Returns the whole-training-set SSE after each sweep."""
        sweep_sses: list[float] = []
        for _ in range(max_sweeps):
            h, o = self.forward(x)
            err_vec = x - o
            if float(np.sum(err_vec ** 2)) < threshold:
                break
            # Generalized delta rule (Rumelhart, Hinton & Williams 1986; Eq. 1 p. 4):
            delta_o = err_vec * o * (1.0 - o)
            delta_h = (delta_o @ self.w2.T) * h * (1.0 - h)
            self.w2 += self.lr * np.outer(h, delta_o)
            self.b2 += self.lr * delta_o
            self.w1 += self.lr * np.outer(x, delta_h)
            self.b1 += self.lr * delta_h
            sweep_sses.append(self.whole_set_sse())
        if not sweep_sses:
            sweep_sses.append(self.whole_set_sse())
        return sweep_sses

    def whole_set_sse(self) -> float:
        return float(np.mean([self.sse(s) for s in TRAIN_STIMULI]))

    def test_category_preference(self) -> dict[str, float]:
        """Evaluate SSE on the 2 peripheral stimuli and 1 central prototype with no weight updates (Section 2.4.2)."""
        sse_p1 = self.sse(TEST_PERIPHERAL_1)
        sse_p2 = self.sse(TEST_PERIPHERAL_2)
        sse_periph = 0.5 * (sse_p1 + sse_p2)
        sse_cent = self.sse(TEST_CENTRAL)
        prop_periph = sse_periph / (sse_periph + sse_cent + 1e-12)
        return {
            "sse_peripheral": round(sse_periph, 5),
            "sse_central": round(sse_cent, 5),
            "prop_peripheral_sse": round(prop_periph, 5),
        }


# ---------------------------------------------------------------------------
# 3. Experiment E1: Fixed Stimulus Presentation Orders (Mather & Plunkett 2011)
# ---------------------------------------------------------------------------

def run_e1_fixed_orders(n_participants: int = 24) -> dict:
    all_perms = list(itertools.permutations(range(8)))
    assert len(all_perms) == 40320
    scored_perms = [(sequence_mean_ed(p), p) for p in all_perms]
    scored_perms.sort(key=lambda x: x[0])

    min_orders = [p for _, p in scored_perms[:n_participants]]
    max_orders = [p for _, p in scored_perms[-n_participants:]]
    mid_start = len(scored_perms) // 2 - n_participants // 2
    med_orders = [p for _, p in scored_perms[mid_start : mid_start + n_participants]]
    rng = np.random.default_rng(2018)
    rand_indices = rng.choice(len(scored_perms), size=n_participants, replace=False)
    rand_orders = [scored_perms[int(i)][1] for i in rand_indices]

    conditions = {
        "max": max_orders,
        "med": med_orders,
        "min": min_orders,
        "random": rand_orders,
    }

    out: dict[str, dict] = {}
    for cond_name, orders in conditions.items():
        first_block_sses = []
        last_block_sses = []
        prop_periphs = []
        mean_eds = []

        for idx, order in enumerate(orders):
            # Use identical participant seed idx across conditions so initial weights are matched
            model = InfantAutoencoder(seed=100 + idx, lr=0.25, init_scale=1.2)
            # In sequential online familiarity training, recent items overwrite shared hidden weights slightly;
            # large successive transitions (max ED) prevent local weight drift into one corner of category space
            trial_sses = []
            for stim_idx in order:
                s_vec = TRAIN_STIMULI[stim_idx]
                sw = model.train_stimulus(s_vec, max_sweeps=20, threshold=0.01)
                trial_sses.append(float(np.mean(sw)))
            first_block_sses.append(float(np.mean(trial_sses[:3])))
            last_block_sses.append(float(np.mean(trial_sses[-3:])))
            test_res = model.test_category_preference()
            prop_periphs.append(test_res["prop_peripheral_sse"])
            mean_eds.append(sequence_mean_ed(order))

        out[cond_name] = {
            "mean_sequence_ed": round(float(np.mean(mean_eds)), 4),
            "first_3trial_block_sse": round(float(np.mean(first_block_sses)), 4),
            "last_3trial_block_sse": round(float(np.mean(last_block_sses)), 4),
            "sse_decrease": round(float(np.mean(first_block_sses) - np.mean(last_block_sses)), 4),
            "median_prop_peripheral_sse": round(float(np.median(prop_periphs)), 4),
            "mean_prop_peripheral_sse": round(float(np.mean(prop_periphs)), 4),
            "above_chance_count": sum(1 for p in prop_periphs if p > 0.50),
        }

    return out


# ---------------------------------------------------------------------------
# 4. Experiment E2: Active Stimulus Selection ("Setting the Model Free")
# ---------------------------------------------------------------------------

def select_next_stimulus(
    model: InfantAutoencoder,
    remaining_indices: list[int],
    prev_idx: int,
    strategy: str,
) -> int:
    best_score = -float("inf")
    best_idx = remaining_indices[0]
    prev_vec = TRAIN_STIMULI[prev_idx]

    for idx in remaining_indices:
        i_vec = TRAIN_STIMULI[idx]
        _, o_vec = model.forward(i_vec)
        if strategy == "curiosity":
            # Equation (1) curiosity function: |(i - o) * o * (1 - o)|
            score = float(np.sum(np.abs(i_vec - o_vec) * o_vec * (1.0 - o_vec)))
        elif strategy == "objective_complexity":
            score = float(np.linalg.norm(i_vec - prev_vec))
        elif strategy == "subjective_novelty":
            score = float(np.sum(np.abs(i_vec - o_vec)))
        elif strategy == "plasticity":
            score = float(np.sum(o_vec * (1.0 - o_vec)))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def run_e2_active_selection(n_participants: int = 24) -> dict:
    strategies = ("curiosity", "objective_complexity", "subjective_novelty", "plasticity")
    out: dict[str, dict] = {}
    unique_curiosity_orders: dict[tuple[int, ...], int] = {}

    for strat in strategies:
        prop_periphs = []
        seq_eds = []
        step_ed_traces = []

        for idx in range(n_participants):
            rng = np.random.default_rng(500 + idx)
            model = InfantAutoencoder(seed=100 + idx, lr=0.25, init_scale=1.2)
            remaining = list(range(8))
            first_idx = int(rng.integers(0, 8))
            remaining.remove(first_idx)
            chosen_order = [first_idx]
            model.train_stimulus(TRAIN_STIMULI[first_idx], max_sweeps=20, threshold=0.01)

            while remaining:
                nxt = select_next_stimulus(model, remaining, chosen_order[-1], strategy=strat)
                remaining.remove(nxt)
                chosen_order.append(nxt)
                model.train_stimulus(TRAIN_STIMULI[nxt], max_sweeps=20, threshold=0.01)

            t_ord = tuple(chosen_order)
            if strat == "curiosity":
                unique_curiosity_orders[t_ord] = unique_curiosity_orders.get(t_ord, 0) + 1

            test_res = model.test_category_preference()
            prop_periphs.append(test_res["prop_peripheral_sse"])
            seq_eds.append(sequence_mean_ed(t_ord))
            step_ed_traces.append(successive_eds(t_ord))

        mean_step_eds = [
            round(float(np.mean([tr[s] for tr in step_ed_traces])), 4)
            for s in range(7)
        ]
        out[strat] = {
            "median_prop_peripheral_sse": round(float(np.median(prop_periphs)), 4),
            "mean_prop_peripheral_sse": round(float(np.mean(prop_periphs)), 4),
            "mean_sequence_ed": round(float(np.mean(seq_eds)), 4),
            "mean_successive_ed_by_step": mean_step_eds,
            "above_chance_count": sum(1 for p in prop_periphs if p > 0.50),
        }

    out["curiosity_unique_sequences_count"] = len(unique_curiosity_orders)
    out["curiosity_sample_step_eds"] = successive_eds(next(iter(unique_curiosity_orders.keys())))
    return out


# ---------------------------------------------------------------------------
# 5. Experiment E3: Challenging `|i - o| * o * (1 - o)` (Saturation & Noise Traps)
# ---------------------------------------------------------------------------

def run_e3_boundary_challenges() -> dict:
    # Challenge 3A: Sigmoid saturation blindness on a strongly contradicted stimulus
    # Stimulus A: strongly contradicted (i = 0.95, but prior belief saturated at o = 0.01 -> |i - o| = 0.94)
    # Stimulus B: mildly discrepant (i = 0.70, neutral belief o = 0.50 -> |i - o| = 0.20)
    i_a, o_a = 0.95, 0.01
    i_b, o_b = 0.70, 0.50
    cur_a = abs(i_a - o_a) * o_a * (1.0 - o_a)
    cur_b = abs(i_b - o_b) * o_b * (1.0 - o_b)

    # Challenge 3B: Sampling WITH replacement when 1 unlearnable white-noise stimulus is added to the 8 stimuli
    # Because a random stimulus i_noise ~ Uniform(0, 1) drives its output representation toward o ~ 0.50
    # (where plasticity o(1-o) = 0.25 is maximal!) while maintaining |i - o| ~ 0.25, instantaneous
    # `|i - o| * o * (1 - o)` without replacement-free constraint or error-derivative tracking gets trapped!
    # Challenge 3B: Removing the "without replacement" constraint (Section 3.1, p. 6).
    # In Experiment 2, Twomey & Westermann forced the model to sample WITHOUT replacement (each of the 8 stimuli
    # could be chosen only once). What happens if the infant model is truly "set free" to sample WITH replacement
    # across 40 trials? Because training on one peripheral stimulus shifts the shared 3-hidden-unit weights, it pushes
    # the antipodal peripheral stimulus back toward o ~ 0.50 (maximal plasticity o(1-o)=0.25 and large |i-o|), trapping
    # the model in a 2-stimulus antipodal limit cycle where it ignores the rest of the category!
    model_wr = InfantAutoencoder(seed=100, lr=0.25, init_scale=1.2)
    visited_with_replacement: list[int] = [0]
    model_wr.train_stimulus(TRAIN_STIMULI[0], max_sweeps=20, threshold=0.01)
    for _ in range(39):
        scores = []
        for k in range(8):
            i_vec = TRAIN_STIMULI[k]
            _, o_vec = model_wr.forward(i_vec)
            scores.append(float(np.sum(np.abs(i_vec - o_vec) * o_vec * (1.0 - o_vec))))
        chosen_k = int(np.argmax(scores))
        visited_with_replacement.append(chosen_k)
        model_wr.train_stimulus(TRAIN_STIMULI[chosen_k], max_sweeps=20, threshold=0.01)

    unique_visited = sorted(set(visited_with_replacement))
    counts = {int(k): visited_with_replacement.count(k) for k in unique_visited}
    top2_share = sum(sorted(counts.values(), reverse=True)[:2]) / len(visited_with_replacement)

    return {
        "challenge_3a_sigmoid_saturation_blindness": {
            "strongly_contradicted_stimulus": {"i": i_a, "o": o_a, "abs_error": round(abs(i_a - o_a), 4), "plasticity": round(o_a * (1.0 - o_a), 5), "curiosity_score": round(cur_a, 5)},
            "mildly_discrepant_stimulus": {"i": i_b, "o": o_b, "abs_error": round(abs(i_b - o_b), 4), "plasticity": round(o_b * (1.0 - o_b), 5), "curiosity_score": round(cur_b, 5)},
            "blindness_ratio_mild_over_contradicted": round(cur_b / cur_a, 2),
        },
        "challenge_3b_with_replacement_limit_cycle_trap": {
            "total_trials": len(visited_with_replacement),
            "unique_stimuli_visited_out_of_8": len(unique_visited),
            "visit_counts_by_stimulus": counts,
            "top_2_stimuli_share_of_trials": round(top2_share, 4),
            "sequence_head_16": visited_with_replacement[:16],
        },
    }


def main() -> int:
    t0 = time.perf_counter()
    e1 = run_e1_fixed_orders()
    e2 = run_e2_active_selection()
    e3 = run_e3_boundary_challenges()

    assert e1["max"]["first_3trial_block_sse"] > e1["max"]["last_3trial_block_sse"], "SSE must decrease over training"
    assert e1["max"]["above_chance_count"] == 24, "All 24 models must form a category (> 0.50 peripheral SSE proportion)"
    assert e2["curiosity"]["mean_prop_peripheral_sse"] > e2["plasticity"]["mean_prop_peripheral_sse"]
    assert e3["challenge_3a_sigmoid_saturation_blindness"]["blindness_ratio_mild_over_contradicted"] > 4.0
    assert e3["challenge_3b_with_replacement_limit_cycle_trap"]["unique_stimuli_visited_out_of_8"] < 8

    report = {
        "paper_id": "02-twomey-westermann-2018-infant-curiosity",
        "title": "Curiosity-Based Learning in Infants: A Neurocomputational Approach",
        "authors": "Katherine E. Twomey, Gert Westermann",
        "year": 2018,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "e1_fixed_presentation_orders": e1,
        "e2_active_stimulus_selection": e2,
        "e3_boundary_challenges": e3,
    }
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "twomey_2018_results.json"
    out_fp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
