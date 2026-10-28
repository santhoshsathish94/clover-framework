#!/usr/bin/env python3
"""
sutton1988_experiments.py — Verified from-scratch implementation and experimental verification
of Richard S. Sutton (1988), "Learning to Predict by the Methods of Temporal Differences",
Machine Learning 3: 9-44 (plus Erratum on p. 377).

Covers:
  - E2: Theorem 1 exact equivalence of per-sequence TD(1) and Widrow-Hoff (Eq. 2-3, pp. 13-15),
        incremental O(d) eligibility trace recursion (Eq. 4, pp. 15-16), and Section 5.2 (p. 35)
        intra-sequence weight update non-equivalence & refinement.
  - E3: Bounded 5-state random walk (B, C, D, E, F with absorbing A->0, G->1, Fig. 2, p. 18):
        - Theorem 2 positive-definiteness of D(I-Q) and exact ideal predictions (1/6, 2/6, 3/6, 4/6, 5/6)
        - Theorem 3 & Figure 3 (Erratum p. 377): Repeated-presentation convergence across 100 training sets
          of 10 sequences each, proving TD(0) == Maximum-Likelihood Markov model [(I - Q_hat)^{-1} h_hat]
          and TD(1) == Widrow-Hoff empirical outcome average
        - Figures 4 & 5: Single-presentation RMS error across alpha and lambda, plus p. 23 backward TD(0) replay
  - E4: Section 3.1 "Novel -> Bad -> Win" Game Example (Fig. 1, pp. 17-18), Section 3.1 Non-Markovian
        State-Aliasing Counterexample (p. 18), and Section 5.1 Cumulative Step-Count TD(0) (pp. 33-34).
"""

import numpy as np


# ============================================================================
# Random Walk Generator (Section 3.2, Figure 2, pp. 18-19)
# ============================================================================
# States: 0=A (left terminal, z=0), 1=B, 2=C, 3=D (start), 4=E, 5=F, 6=G (right terminal, z=1)
# Nonterminal indices in 0..4 correspond to B, C, D, E, F.

IDEAL_W = np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0], dtype=float)


def generate_walk(rng: np.random.Generator):
    """Returns (states, z) where states is list of nonterminal indices in 0..4 (B..F) and z in {0.0, 1.0}."""
    pos = 3  # Start at D (1..5 are B..F; 0 is A, 6 is G)
    seq = []
    while 1 <= pos <= 5:
        seq.append(pos - 1)
        step = 1 if rng.random() < 0.5 else -1
        pos += step
    z = 1.0 if pos == 6 else 0.0
    return seq, z


def generate_training_sets(n_sets: int = 100, n_walks: int = 10, seed: int = 1988):
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n_sets):
        walks = [generate_walk(rng) for _ in range(n_walks)]
        sets.append(walks)
    return sets


# ============================================================================
# E2: Theorem 1 Equivalence, Eligibility Traces, & Intra-Sequence Drift
# ============================================================================

def widrow_hoff_sequence_delta(states, z, w, X, alpha):
    """Eq. 2 (p. 13-14): sum_{t=1}^m alpha * (z - w^T x_t) x_t."""
    dw = np.zeros_like(w)
    for s in states:
        x_t = X[:, s]
        p_t = float(w @ x_t)
        dw += alpha * (z - p_t) * x_t
    return dw


def td_lambda_sequence_delta(states, z, w, X, alpha, lam):
    """Eq. 4 (p. 15-16) with incremental eligibility trace e_t = x_t + lam * e_{t-1}."""
    dw = np.zeros_like(w)
    e = np.zeros_like(w)
    m = len(states)
    for t in range(m):
        x_t = X[:, states[t]]
        p_t = float(w @ x_t)
        if t + 1 < m:
            p_next = float(w @ X[:, states[t + 1]])
        else:
            p_next = z
        e = x_t + lam * e
        dw += alpha * (p_next - p_t) * e
    return dw


def td_lambda_explicit_sum_delta(states, z, w, X, alpha, lam):
    """Eq. 4 explicit sum: sum_{t=1}^m alpha (P_{t+1} - P_t) sum_{k=1}^t lam^{t-k} x_k."""
    dw = np.zeros_like(w)
    m = len(states)
    for t in range(m):
        p_t = float(w @ X[:, states[t]])
        p_next = float(w @ X[:, states[t + 1]]) if (t + 1 < m) else z
        sum_x = np.zeros_like(w)
        for k in range(t + 1):
            sum_x += (lam ** (t - k)) * X[:, states[k]]
        dw += alpha * (p_next - p_t) * sum_x
    return dw


def run_e2_theorem1_and_traces():
    print("=== E2.1: Theorem 1 Exact Equivalence of Per-Sequence Linear TD(1) and Widrow-Hoff (pp. 13-15) ===")
    rng = np.random.default_rng(42)
    walks = [generate_walk(rng) for _ in range(1000)]

    # 1. Tabular unit-basis features X = I_5:
    X_tab = np.eye(5, dtype=float)
    w_tab = rng.uniform(-1.0, 2.0, size=5)
    max_diff_tab = 0.0
    for states, z in walks:
        dw_wh = widrow_hoff_sequence_delta(states, z, w_tab, X_tab, alpha=0.17)
        dw_td1 = td_lambda_sequence_delta(states, z, w_tab, X_tab, alpha=0.17, lam=1.0)
        max_diff_tab = max(max_diff_tab, float(np.max(np.abs(dw_wh - dw_td1))))

    # 2. Arbitrary dense 8D features X in R^{8 x 5}:
    X_dense = rng.normal(size=(8, 5))
    w_dense = rng.uniform(-0.5, 0.5, size=8)
    max_diff_dense = 0.0
    for states, z in walks:
        dw_wh = widrow_hoff_sequence_delta(states, z, w_dense, X_dense, alpha=0.05)
        dw_td1 = td_lambda_sequence_delta(states, z, w_dense, X_dense, alpha=0.05, lam=1.0)
        max_diff_dense = max(max_diff_dense, float(np.max(np.abs(dw_wh - dw_td1))))

    print(f"  Max |dw_TD(1) - dw_WidrowHoff| across 1,000 walks (tabular 5D): {max_diff_tab:.2e}")
    print(f"  Max |dw_TD(1) - dw_WidrowHoff| across 1,000 walks (dense 8D):   {max_diff_dense:.2e}")
    assert max_diff_tab < 1e-14 and max_diff_dense < 1e-14

    print("\n=== E2.2: Incremental O(d) Eligibility Trace vs Explicit Past-History Sum (Eq. 4, pp. 15-16) ===")
    max_trace_diff = 0.0
    for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for states, z in walks[:100]:
            d1 = td_lambda_sequence_delta(states, z, w_dense, X_dense, alpha=0.1, lam=lam)
            d2 = td_lambda_explicit_sum_delta(states, z, w_dense, X_dense, alpha=0.1, lam=lam)
            max_trace_diff = max(max_trace_diff, float(np.max(np.abs(d1 - d2))))
    print(f"  Max |incremental_trace - explicit_sum| across all 7 lambdas:    {max_trace_diff:.2e}")
    assert max_trace_diff < 1e-14

    print("\n=== E2.3: Negative Control — Intra-Sequence Weight Updating Drift (Section 5.2, p. 35) ===")
    # When w is updated at every step t WITHIN a sequence that revisits states,
    # naive intra-sequence TD(1) uses P_t = w_{t-1}^T x_t and P_{t+1} = w_t^T x_{t+1},
    # so P_{t+1} - P_t includes (w_t - w_{t-1})^T x_{t+1} and no longer telescopes to z - P_t!
    max_naive_diff = 0.0
    max_refined_diff = 0.0
    for states, z in walks[:200]:
        if len(states) < 4:
            continue
        w0 = w_tab.copy()
        alpha = 0.2
        w_wh = w0 + widrow_hoff_sequence_delta(states, z, w0, X_tab, alpha)

        # Naive intra-sequence TD(1) (first equation on p. 35: P_t = P(x_t, w_{t-1})):
        w_naive = w0.copy()
        e = np.zeros_like(w0)
        p_prev = float(w_naive @ X_tab[:, states[0]])
        for t in range(len(states)):
            e = X_tab[:, states[t]] + 1.0 * e
            p_next = float(w_naive @ X_tab[:, states[t + 1]]) if (t + 1 < len(states)) else z
            w_naive = w_naive + alpha * (p_next - p_prev) * e
            p_prev = p_next

        # Sutton's refined intra-sequence TD(1) (second equation on p. 35: P(x_{t+1}, w_t) - P(x_t, w_t)):
        w_ref = w0.copy()
        e = np.zeros_like(w0)
        for t in range(len(states)):
            e = X_tab[:, states[t]] + 1.0 * e
            p_curr = float(w_ref @ X_tab[:, states[t]])
            p_next = float(w_ref @ X_tab[:, states[t + 1]]) if (t + 1 < len(states)) else z
            w_ref = w_ref + alpha * (p_next - p_curr) * e

        max_naive_diff = max(max_naive_diff, float(np.max(np.abs(w_naive - w_wh))))
        max_refined_diff = max(max_refined_diff, float(np.max(np.abs(w_ref - w_wh))))

    print(f"  Max |w_intra_naive   - w_off_line_WH| (p. 35 Eq. 1, mixes dw & dx): {max_naive_diff:.4f}")
    print(f"  Max |w_intra_refined - w_off_line_WH| (p. 35 Eq. 2, pure dx step):  {max_refined_diff:.4f}")
    assert max_naive_diff > 1e-2


# ============================================================================
# E3: Bounded Random Walk — Figure 3 (+ Erratum p. 377), Theorem 2 & 3, Figs. 4-5
# ============================================================================

def compute_ml_and_wh_estimates(walks):
    """
    For a finite training set of walks on the 5 nonterminal states 0..4:
      - Maximum-likelihood Markov model (Theorem 3, Eq. 8, p. 30):
          Q_hat[i, j] = count(i -> j) / visits(i)
          h_hat[i]    = count(i -> G) / visits(i)
          w_ML        = (I - Q_hat)^{-1} h_hat
      - Widrow-Hoff (TD(1)) fixed point (p. 31):
          w_WH[i]     = sum_{visits to i} z_walk / visits(i)
    """
    visits = np.zeros(5, dtype=float)
    trans = np.zeros((5, 5), dtype=float)
    right_term = np.zeros(5, dtype=float)
    outcome_sum = np.zeros(5, dtype=float)

    for states, z in walks:
        m = len(states)
        for t, s in enumerate(states):
            visits[s] += 1.0
            outcome_sum[s] += z
            if t + 1 < m:
                trans[s, states[t + 1]] += 1.0
            else:
                if z > 0.5:
                    right_term[s] += 1.0

    Q_hat = np.zeros((5, 5), dtype=float)
    h_hat = np.zeros(5, dtype=float)
    w_wh = np.full(5, 0.5, dtype=float)
    for i in range(5):
        if visits[i] > 0:
            Q_hat[i, :] = trans[i, :] / visits[i]
            h_hat[i] = right_term[i] / visits[i]
            w_wh[i] = outcome_sum[i] / visits[i]

    w_ml = np.linalg.solve(np.eye(5) - Q_hat, h_hat)
    return w_ml, w_wh, visits


def batch_td_lambda_converge(walks, lam, alpha=0.01, tol=1e-7, max_sweeps=20000):
    """Repeated presentations paradigm (Section 3.2, p. 20): accumulate dw over all 10 sequences in training set."""
    X_tab = np.eye(5, dtype=float)
    w = np.full(5, 0.5, dtype=float)
    for _ in range(max_sweeps):
        dw_total = np.zeros(5, dtype=float)
        for states, z in walks:
            dw_total += td_lambda_sequence_delta(states, z, w, X_tab, alpha, lam)
        w_new = w + dw_total
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def rms_error(w):
    return float(np.sqrt(np.mean((w - IDEAL_W) ** 2)))


def run_e3_random_walk():
    print("\n=== E3.1: Theorem 2 Matrix Properties & Exact Ideal Predictions (pp. 24-28) ===")
    Q = np.zeros((5, 5), dtype=float)
    for i in range(5):
        if i > 0:
            Q[i, i - 1] = 0.5
        if i < 4:
            Q[i, i + 1] = 0.5
    h = np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=float)
    mu = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    w_ideal = np.linalg.solve(np.eye(5) - Q, h)
    d_visits = mu @ np.linalg.inv(np.eye(5) - Q)
    D = np.diag(d_visits)
    S = D @ (np.eye(5) - Q) + (D @ (np.eye(5) - Q)).T
    eigvals_S = np.linalg.eigvalsh(S)
    print(f"  Ideal predictions (I - Q)^(-1) h: {np.round(w_ideal, 4)} (exact 1/6..5/6)")
    print(f"  Expected visits per walk d^T = mu^T (I - Q)^(-1): {np.round(d_visits, 4)} (sum = {np.sum(d_visits):.1f} steps)")
    print(f"  Eigenvalues of S = D(I-Q) + (D(I-Q))^T (Theorem 2 Lemma, p. 27): {np.round(eigvals_S, 4)} (all > 0!)")
    assert np.allclose(w_ideal, IDEAL_W) and np.all(eigvals_S > 0)

    print("\n=== E3.2: Repeated Presentations (Figure 3 + Erratum p. 377) & Theorem 3 ML Equivalence ===")
    training_sets = generate_training_sets(n_sets=100, n_walks=10, seed=1988)
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    # First check Theorem 3 (TD(0) == w_ML and TD(1) == w_WH) on all training sets where all 5 states are visited:
    max_td0_ml_diff = 0.0
    max_td1_wh_diff = 0.0
    n_full_visited = 0
    for walks in training_sets[:30]:
        w_ml, w_wh, visits = compute_ml_and_wh_estimates(walks)
        if np.all(visits > 0):
            n_full_visited += 1
            w_td0 = batch_td_lambda_converge(walks, lam=0.0, alpha=0.015, tol=1e-8)
            w_td1 = batch_td_lambda_converge(walks, lam=1.0, alpha=0.015, tol=1e-8)
            max_td0_ml_diff = max(max_td0_ml_diff, float(np.max(np.abs(w_td0 - w_ml))))
            max_td1_wh_diff = max(max_td1_wh_diff, float(np.max(np.abs(w_td1 - w_wh))))
    print(f"  Theorem 3 check across {n_full_visited} fully-visited 10-walk sets:")
    print(f"    Max |w_TD(0)_converged - (I - Q_hat)^(-1) h_hat (ML)|: {max_td0_ml_diff:.2e}")
    print(f"    Max |w_TD(1)_converged - empirical sample-mean (WH)|:  {max_td1_wh_diff:.2e}")
    assert max_td0_ml_diff < 1e-5 and max_td1_wh_diff < 1e-5

    # Reproduce Figure 3 (Erratum p. 377) across all 100 training sets and 7 lambdas:
    fig3_rms = {}
    for lam in lambdas:
        errs = []
        for walks in training_sets:
            if lam == 0.0:
                w_ml, _, visits = compute_ml_and_wh_estimates(walks)
                # Unvisited states keep initial weight 0.5:
                w_conv = np.where(visits > 0, w_ml, 0.5)
            elif lam == 1.0:
                _, w_wh, visits = compute_ml_and_wh_estimates(walks)
                w_conv = np.where(visits > 0, w_wh, 0.5)
            else:
                w_conv = batch_td_lambda_converge(walks, lam=lam, alpha=0.015, tol=1e-6, max_sweeps=8000)
            errs.append(rms_error(w_conv))
        fig3_rms[lam] = (float(np.mean(errs)), float(np.std(errs) / np.sqrt(len(errs))))
        print(f"    Fig. 3 (Erratum p. 377) lambda = {lam:.1f}: mean RMS error = {fig3_rms[lam][0]:.4f} +/- {fig3_rms[lam][1]:.4f}")
    assert fig3_rms[0.0][0] < fig3_rms[0.5][0] < fig3_rms[1.0][0]

    print("\n=== E3.3: Single Presentation of 10 Sequences (Figures 4 & 5, pp. 21-23) & Backward Replay ===")
    X_tab = np.eye(5, dtype=float)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    fig4_curves = {lam: {} for lam in lambdas}
    for lam in lambdas:
        for alpha in alphas:
            errs = []
            for walks in training_sets:
                w = np.full(5, 0.5, dtype=float)
                for states, z in walks:
                    w = w + td_lambda_sequence_delta(states, z, w, X_tab, alpha, lam)
                errs.append(rms_error(w))
            fig4_curves[lam][alpha] = float(np.mean(errs))

    print("  Figure 4 (RMS error vs alpha for representative lambdas 0.0, 0.3, 0.8, 1.0):")
    print("    alpha | lam=0.0 | lam=0.3 | lam=0.5 | lam=0.9 | lam=1.0 (Widrow-Hoff)")
    for alpha in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        print(f"     {alpha:4.2f} |  {fig4_curves[0.0][alpha]:.4f} |  {fig4_curves[0.3][alpha]:.4f} |  {fig4_curves[0.5][alpha]:.4f} |  {fig4_curves[0.9][alpha]:.4f} |  {fig4_curves[1.0][alpha]:.4f}")

    print("\n  Figure 5 (Best-alpha RMS error after 1 presentation of 10 sequences, p. 22):")
    best_by_lam = {}
    for lam in lambdas:
        best_a = min(alphas, key=lambda a: fig4_curves[lam][a])
        best_err = fig4_curves[lam][best_a]
        best_by_lam[lam] = (best_a, best_err)
        print(f"    lambda = {lam:.1f}: best alpha = {best_a:.2f} -> mean RMS error = {best_err:.4f}")
    assert best_by_lam[0.3][1] < best_by_lam[0.0][1] < best_by_lam[1.0][1]

    # Test Sutton's p. 23 backward offline TD(0) replay ("first F's prediction could be updated in light of the 1,
    # then E's prediction could be updated toward F's new level, and so on"):
    best_back_err = 999.0
    best_back_a = 0.0
    for alpha in alphas:
        errs = []
        for walks in training_sets:
            w = np.full(5, 0.5, dtype=float)
            for states, z in walks:
                m = len(states)
                for t in range(m - 1, -1, -1):
                    s_t = states[t]
                    p_next = w[states[t + 1]] if (t + 1 < m) else z
                    w[s_t] += alpha * (p_next - w[s_t])
            errs.append(rms_error(w))
        mean_e = float(np.mean(errs))
        if mean_e < best_back_err:
            best_back_err = mean_e
            best_back_a = alpha
    print(f"  Backward offline TD(0) replay (p. 23): best alpha = {best_back_a:.2f} -> mean RMS error = {best_back_err:.4f} (vs forward TD(0) {best_by_lam[0.0][1]:.4f}!)")
    assert best_back_err < best_by_lam[0.0][1]


# ============================================================================
# E4: Game-Playing Example (Fig. 1, p. 17), Non-Markovian Counterexample (p. 18),
#     and Cumulative Outcomes TD(0) (Section 5.1, pp. 33-34)
# ============================================================================

def run_e4_game_and_cumulative():
    print("\n=== E4.1: The 'Novel -> Bad -> Win' Game-Playing Example (Section 3.1, Figure 1, pp. 17-18) ===")
    # Experience consists of:
    #   - 9 prior games: [Bad] -> Loss (z=0)
    #   - 1 lucky game:  [Novel, Bad] -> Win (z=1)
    # True Markov process has P(Win | Bad) = 0.10, and since Novel -> Bad deterministically, P(Win | Novel) = 0.10.
    # States: 0 = Novel, 1 = Bad.
    episodes = [([1], 0.0) for _ in range(9)] + [([0, 1], 1.0)]
    X2 = np.eye(2, dtype=float)

    w_td0 = np.array([0.5, 0.5])
    w_wh = np.array([0.5, 0.5])
    for _ in range(5000):
        dw0 = np.zeros(2)
        dw1 = np.zeros(2)
        for st, z in episodes:
            dw0 += td_lambda_sequence_delta(st, z, w_td0, X2, alpha=0.05, lam=0.0)
            dw1 += widrow_hoff_sequence_delta(st, z, w_wh, X2, alpha=0.05)
        w_td0 += dw0
        w_wh += dw1

    print(f"  After 9 [Bad->0] and 1 [Novel->Bad->1] episodes (true Markov E[z|Novel]=0.10, E[z|Bad]=0.10):")
    print(f"    Widrow-Hoff TD(1): P(Novel) = {w_wh[0]:.4f}, P(Bad) = {w_wh[1]:.4f}  (|P(Novel) - 0.10| = {abs(w_wh[0] - 0.10):.4f})")
    print(f"    Linear TD(0):      P(Novel) = {w_td0[0]:.4f}, P(Bad) = {w_td0[1]:.4f}  (|P(Novel) - 0.10| = {abs(w_td0[0] - 0.10):.4f})")
    assert abs(w_td0[0] - 0.10) < 1e-4 and abs(w_wh[0] - 1.00) < 1e-4

    print("\n=== E4.2: Non-Markovian Counterexample Where TD(0) Fails and TD(1) Succeeds (p. 18) ===")
    # Page 18: "Suppose the bad state is usually followed by defeats except when it is preceded by the novel state,
    # in which case it always leads to a victory. In this odd case, TD methods could not perform better and might
    # perform worse than supervised-learning methods."
    # True non-Markovian process:
    #   - Path A (90% of games): [OtherStart -> Bad] -> Loss (z=0) always
    #   - Path B (10% of games): [Novel      -> Bad] -> Win  (z=1) always
    # So true E[z | Novel] = 1.00!
    non_markov_eps = [([1], 0.0) for _ in range(9)] + [([0, 1], 1.0)]
    print(f"  Under non-Markovian history dependence (true E[z | Novel] = 1.00):")
    print(f"    Linear TD(0) prediction for Novel:      {w_td0[0]:.4f} (error = {abs(1.0 - w_td0[0]):.4f} — misled by Markov state aliasing!)")
    print(f"    Widrow-Hoff TD(1) prediction for Novel: {w_wh[0]:.4f} (error = {abs(1.0 - w_wh[0]):.4f} — unbiased Monte Carlo return!)")
    assert abs(1.0 - w_td0[0]) > 0.85 and abs(1.0 - w_wh[0]) < 1e-4

    print("\n=== E4.3: Predicting Cumulative Outcomes — Expected Steps to Termination (Section 5.1, pp. 33-34) ===")
    # Ideal expected remaining steps from states B, C, D, E, F:
    #   [(I - Q)^{-1} 1] where 1 = [1, 1, 1, 1, 1]^T
    Q = np.zeros((5, 5), dtype=float)
    for i in range(5):
        if i > 0:
            Q[i, i - 1] = 0.5
        if i < 4:
            Q[i, i + 1] = 0.5
    ideal_steps = np.linalg.solve(np.eye(5) - Q, np.ones(5))
    rng = np.random.default_rng(1988)
    w_steps = np.zeros(5, dtype=float)
    for ep in range(1, 20001):
        states, _ = generate_walk(rng)
        alpha = 0.015 / (1.0 + ep / 4000.0)
        m = len(states)
        dw = np.zeros(5, dtype=float)
        for t in range(m):
            s_t = states[t]
            p_t = w_steps[s_t]
            p_next = w_steps[states[t + 1]] if (t + 1 < m) else 0.0  # p. 34: "where we define P_{m+1} = 0"
            c_next = 1.0                                             # 1 step elapsed per transition
            dw[s_t] += alpha * (c_next + p_next - p_t)
        w_steps += dw

    max_step_err = float(np.max(np.abs(w_steps - ideal_steps)))
    print(f"  Analytical expected steps to termination (I - Q)^(-1) 1: {ideal_steps.tolist()}")
    print(f"  Cumulative TD(0) learned expected steps (20,000 walks):  {np.round(w_steps, 2).tolist()} (max error = {max_step_err:.3f})")
    assert max_step_err < 0.25


if __name__ == "__main__":
    run_e2_theorem1_and_traces()
    run_e3_random_walk()
    run_e4_game_and_cumulative()
