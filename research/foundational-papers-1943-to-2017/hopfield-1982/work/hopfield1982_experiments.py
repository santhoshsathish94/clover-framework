"""
Experiments E1b, E2, E3, and E4 for J. J. Hopfield (1982),
"Neural networks and physical systems with emergent collective computational abilities",
Proc. Natl. Acad. Sci. USA 79: 2554-2558 (April 1982).
"""
import math
import random
from typing import List, Tuple
import numpy as np


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================================
# E2: Energy Monotonicity (Eq. 7-8) and {0,1} vs {-1,+1} Factor-of-2 Capacity
# ============================================================================

def run_e2_energy_and_capacity():
    print("=== E2.1: Energy Monotonicity (Eq. 7-8, p. 2556) & Negative Controls ===")
    rng = np.random.default_rng(1982)
    N = 30
    n_pat = 5

    # Generate symmetric T_ij with T_ii = 0:
    pats = rng.choice([-1, 1], size=(n_pat, N))
    T_sym = (pats.T @ pats).astype(float)
    np.fill_diagonal(T_sym, 0.0)

    def energy_01(V: np.ndarray, T: np.ndarray) -> float:
        # Eq. (7) p. 2556: E = -0.5 * sum_{i != j} T_ij V_i V_j
        return float(-0.5 * (V @ (T @ V)))

    # 1. Asynchronous updates on symmetric T_sym (T_ii = 0) across 500 random initial states:
    async_energy_increases = 0
    async_converged = 0
    total_flips = 0
    for _ in range(500):
        V = rng.choice([0, 1], size=N).astype(float)
        E_curr = energy_01(V, T_sym)
        for sweep in range(50):
            changed = False
            for i in rng.permutation(N):
                h_i = float(T_sym[i] @ V)
                new_vi = 1.0 if h_i > 0.0 else (0.0 if h_i < 0.0 else V[i])
                if new_vi != V[i]:
                    V[i] = new_vi
                    E_next = energy_01(V, T_sym)
                    if E_next > E_curr + 1e-9:
                        async_energy_increases += 1
                    E_curr = E_next
                    total_flips += 1
                    changed = True
            if not changed:
                async_converged += 1
                break

    # 2. Negative Control A: Synchronous parallel updates on anti-ferromagnetic / frustrated symmetric T_ij
    sync_cycles = 0
    for _ in range(200):
        W_rand = rng.normal(0, 1, size=(N, N))
        W_sym = 0.5 * (W_rand + W_rand.T)
        np.fill_diagonal(W_sym, 0.0)
        s = rng.choice([-1.0, 1.0], size=N)
        history = [tuple(s)]
        for _ in range(60):
            h = W_sym @ s
            s = np.where(h >= 0, 1.0, -1.0)
            tup = tuple(s)
            if tup in history:
                idx = history.index(tup)
                cycle_len = len(history) - idx
                if cycle_len >= 2:
                    sync_cycles += 1
                break
            history.append(tup)

    # 3. Negative Control B: Asynchronous updates with negative self-coupling diagonal T_ii = -20.0
    T_neg_diag = T_sym.copy()
    np.fill_diagonal(T_neg_diag, -20.0)
    neg_diag_increases = 0
    for _ in range(200):
        V = rng.choice([0, 1], size=N).astype(float)
        E_curr = float(-0.5 * (V @ (T_neg_diag @ V)))
        for i in range(N):
            h_i = float(T_neg_diag[i] @ V)
            new_vi = 1.0 if h_i > 0.0 else 0.0
            if new_vi != V[i]:
                V[i] = new_vi
                E_next = float(-0.5 * (V @ (T_neg_diag @ V)))
                if E_next > E_curr + 1e-9:
                    neg_diag_increases += 1
                E_curr = E_next

    print(f"  Asynchronous (T_ij=T_ji, T_ii=0): converged={async_converged}/500, E_increases={async_energy_increases}/{total_flips} flips")
    print(f"  Negative Control A (Synchronous parallel updates on symmetric W): limit cycles of length >=2 in {sync_cycles}/200 runs")
    print(f"  Negative Control B (Negative diagonal T_ii=-20.0): E increased in {neg_diag_increases} flips across 200 runs")
    assert async_converged == 500 and async_energy_increases == 0 and sync_cycles > 50 and neg_diag_increases > 100

    print("\n=== E2.2: {0, 1} (Eq. 2-4, 10) vs Bipolar {-1, +1} Factor-of-2 Storage Capacity (pp. 2556-2557) ===")
    # Hopfield Eq. (10) p. 2556 for V_j in {0, 1}:
    #   Signal = (N-1)/2 (or N/2), noise variance = (n-1)*(N-1)/4 (or (n-1)*N/2 in Hopfield's unnormalized T_ij where 2V-1 = +-1 and V in {0,1}):
    #   Wait: in Eq. (3)-(4), each term (2V_i^s - 1)(2V_j^s - 1) V_j^{s'} is +-1 when V_j^{s'}=1 (prob 1/2) and 0 when V_j^{s'}=0 (prob 1/2).
    #   So for s != s', each j != i has variance 1/2, giving sigma^2 = (n-1)*N/2, and signal mean = N/2!
    #   Thus SNR_{0,1} = (N/2) / sqrt((n-1)*N/2) = sqrt(N / (2*(n-1))).
    #   For bipolar mu_j in {-1, +1}: signal mean = N, noise variance = (n-1)*N, so SNR_{+-1} = N / sqrt((n-1)*N) = sqrt(N / (n-1))!
    #   Hence SNR_{+-1}(2n - 1) == SNR_{0,1}(n) IDENTICALLY!
    N_cap = 100
    print("  n_01 | n_bipolar=2n-1 | Eq.10 P_err | Emp P_err {0,1}(n) | Emp P_err {-1,+1}(2n-1) | Zero-err states {0,1} vs {-1,+1}")
    for n_01 in (5, 8, 10, 13, 15):
        n_bip = 2 * n_01 - 1
        snr_theo = math.sqrt((N_cap - 1) / (2.0 * (n_01 - 1)))
        p_theo_paper = 1.0 - normal_cdf(math.sqrt(N_cap / (2.0 * (n_01 - 1))))

        errs_01, zero_01, tot_states_01 = 0, 0, 0
        errs_bip, zero_bip, tot_states_bip = 0, 0, 0
        n_trials = 60
        for _ in range(n_trials):
            # {0, 1} trial with n_01 memories:
            V_mem = rng.choice([0, 1], size=(n_01, N_cap))
            S_mem = 2 * V_mem - 1
            T01 = S_mem.T @ S_mem
            np.fill_diagonal(T01, 0)
            H01 = V_mem @ T01.T  # shape (n_01, N_cap)
            V_out = (H01 > 0).astype(int)
            bit_diffs_01 = np.sum(V_out != V_mem, axis=1)
            errs_01 += int(np.sum(bit_diffs_01))
            zero_01 += int(np.sum(bit_diffs_01 == 0))
            tot_states_01 += n_01

            # Bipolar {-1, +1} trial with n_bip = 2*n_01 - 1 memories:
            M_mem = rng.choice([-1, 1], size=(n_bip, N_cap))
            Tbip = M_mem.T @ M_mem
            np.fill_diagonal(Tbip, 0)
            Hbip = M_mem @ Tbip.T
            M_out = np.where(Hbip >= 0, 1, -1)
            bit_diffs_bip = np.sum(M_out != M_mem, axis=1)
            errs_bip += int(np.sum(bit_diffs_bip))
            zero_bip += int(np.sum(bit_diffs_bip == 0))
            tot_states_bip += n_bip

        p_emp_01 = errs_01 / (tot_states_01 * N_cap)
        p_emp_bip = errs_bip / (tot_states_bip * N_cap)
        z_01 = zero_01 / tot_states_01
        z_bip = zero_bip / tot_states_bip
        print(f"   {n_01:2d}  |       {n_bip:2d}       |   {p_theo_paper:.5f}   |      {p_emp_01:.5f}       |         {p_emp_bip:.5f}         |     {z_01*100:5.1f}%   vs   {z_bip*100:5.1f}%")

    print("\n=== E2.3: Reproduction of Figure 2 (p. 2557) Multi-Step Error Histograms at N=100 (n=5, 10, 15) ===")
    for mode_name, is_bipolar in [("{0, 1} states (Eq. 1-4, Fig. 2)", False), ("Bipolar {-1, +1} states (p. 2557)", True)]:
        print(f"  >>> {mode_name}:")
        for n_mem in (5, 10, 15):
            final_errs = []
            for _ in range(30):
                M = rng.choice([-1, 1], size=(n_mem, N_cap))
                T = M.T @ M
                np.fill_diagonal(T, 0)
                for s_idx in range(n_mem):
                    state = M[s_idx].copy() if is_bipolar else ((M[s_idx] + 1) // 2).copy()
                    target = state.copy()
                    for _ in range(15):
                        changed = False
                        for i in rng.permutation(N_cap):
                            h_i = int(T[i] @ state)
                            if is_bipolar:
                                nv = 1 if h_i > 0 else (-1 if h_i < 0 else state[i])
                            else:
                                nv = 1 if h_i > 0 else (0 if h_i < 0 else state[i])
                            if nv != state[i]:
                                state[i] = nv
                                changed = True
                        if not changed:
                            break
                    final_errs.append(int(np.sum(state != target)))
            arr = np.array(final_errs)
            b_0 = np.mean(arr == 0) * 100
            b_lt5 = np.mean(arr < 5) * 100
            b_ge10 = np.mean(arr >= 10) * 100
            print(f"    n={n_mem:2d}: exact(0 err)={b_0:5.1f}% | <5 errs (well retained)={b_lt5:5.1f}% | >=10 errs (avalanche)={b_ge10:5.1f}%")


# ============================================================================
# E3: Basin of Attraction vs Distance (N=30, n=5) & Memory Merging (N=100, n=8)
# ============================================================================

def run_e3_basins_and_merging():
    print("\n=== E3.1: Basin of Attraction vs Initial Hamming Distance (N=30, n=5, p. 2557) ===")
    rng = np.random.default_rng(1982)
    N = 30
    n_mem = 5

    for d_init in (1, 2, 3, 5, 8, 10, 12):
        hits_target = 0
        hits_nearest = 0
        total = 500
        for _ in range(total):
            M = rng.choice([-1, 1], size=(n_mem, N))
            T = M.T @ M
            np.fill_diagonal(T, 0)
            state = M[0].copy()
            flip_idx = rng.choice(N, size=d_init, replace=False)
            state[flip_idx] *= -1

            # Determine which stored memory is initially closest to `state`:
            dists_init = [int(np.sum(state != M[k])) for k in range(n_mem)]
            nearest_k = int(np.argmin(dists_init))

            # Evolve asynchronously to fixed point:
            for _ in range(20):
                changed = False
                for i in rng.permutation(N):
                    h_i = int(T[i] @ state)
                    nv = 1 if h_i > 0 else (-1 if h_i < 0 else state[i])
                    if nv != state[i]:
                        state[i] = nv
                        changed = True
                if not changed:
                    break
            if np.array_equal(state, M[0]):
                hits_target += 1
            if np.array_equal(state, M[nearest_k]):
                hits_nearest += 1
        print(f"  Initial Hamming distance d={d_init:2d}/30: recovered original M[0] = {hits_target/total*100:5.1f}% | recovered nearest M[k] = {hits_nearest/total*100:5.1f}%")

    # Also test completely random starting states (N=30, n=5, 10 signed attractors +-M[k], p. 2557):
    exact_assigned, near_1bit, spurious = 0, 0, 0
    for _ in range(500):
        M = rng.choice([-1, 1], size=(n_mem, N))
        T = M.T @ M
        np.fill_diagonal(T, 0)
        state = rng.choice([-1, 1], size=N)
        for _ in range(25):
            changed = False
            for i in rng.permutation(N):
                h_i = int(T[i] @ state)
                nv = 1 if h_i > 0 else (-1 if h_i < 0 else state[i])
                if nv != state[i]:
                    state[i] = nv
                    changed = True
            if not changed:
                break
        # Distance to the 10 signed nominal states +-M[0..4]:
        min_d = min(min(int(np.sum(state != M[k])), int(np.sum(state != -M[k]))) for k in range(n_mem))
        if min_d == 0:
            exact_assigned += 1
        elif min_d <= 2:
            near_1bit += 1
        else:
            spurious += 1
    print(f"  Random initial states (N=30, n=5, 10 signed memories): exact={exact_assigned/5:.1f}%, near(1-2 bits)={near_1bit/5:.1f}%, spurious={spurious/5:.1f}%")

    print("\n=== E3.2: Memory Merging vs Similarity (N=100, n=8, 8th memory at distance d=30, 20, 10 from #1, p. 2557) ===")
    N_m = 100
    for d_sep in (30, 20, 10):
        both_exact = 0
        distinct_displaced = 0
        fused_same = 0
        trials = 200
        for _ in range(trials):
            M = rng.choice([-1, 1], size=(8, N_m))
            # Make M[7] at exact Hamming distance d_sep from M[0]:
            M[7] = M[0].copy()
            flip_pos = rng.choice(N_m, size=d_sep, replace=False)
            M[7, flip_pos] *= -1
            T = M.T @ M
            np.fill_diagonal(T, 0)

            end_states = []
            for idx in (0, 7):
                s = M[idx].copy()
                for _ in range(20):
                    changed = False
                    for i in rng.permutation(N_m):
                        h_i = int(T[i] @ s)
                        nv = 1 if h_i > 0 else (-1 if h_i < 0 else s[i])
                        if nv != s[i]:
                            s[i] = nv
                            changed = True
                    if not changed:
                        break
                end_states.append(s)
            d_between_ends = int(np.sum(end_states[0] != end_states[1]))
            d0 = int(np.sum(end_states[0] != M[0]))
            d7 = int(np.sum(end_states[1] != M[7]))
            if d_between_ends == 0:
                fused_same += 1
            elif d0 == 0 and d7 == 0:
                both_exact += 1
            else:
                distinct_displaced += 1
        print(f"  Distance d={d_sep:2d}/100 between M[0] & M[7]: both exact={both_exact/2:5.1f}% | distinct displaced={distinct_displaced/2:5.1f}% | fused(merged)={fused_same/2:5.1f}%")


# ============================================================================
# E4: Clipped Synapses (2/pi), Saturating +-3 Forgetting, Unidirectional, & Overload Familiarity
# ============================================================================

def run_e4_robustness_and_familiarity():
    print("\n=== E4.1: Clipped Synapses (2/pi capacity factor) & Unidirectional Dilution (1/sqrt(2) SNR, p. 2557) ===")
    rng = np.random.default_rng(1982)
    N = 100
    n_mem = 11  # odd n so T_ij != 0 off-diagonal

    snrs_unclipped, snrs_clipped, snrs_unidir = [], [], []
    for _ in range(100):
        M = rng.choice([-1, 1], size=(n_mem, N))
        T_full = (M.T @ M).astype(float)
        np.fill_diagonal(T_full, 0.0)

        T_clip = np.sign(T_full)
        np.fill_diagonal(T_clip, 0.0)

        # Unidirectional: keep only one of (i,j) or (j,i) at random:
        T_uni = np.zeros_like(T_full)
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.5:
                    T_uni[i, j] = T_full[i, j]
                else:
                    T_uni[j, i] = T_full[j, i]

        # Compute aligned local fields y_i = M[0, i] * (T @ M[0])_i for each matrix:
        for T_mat, bucket in [(T_full, snrs_unclipped), (T_clip, snrs_clipped), (T_uni, snrs_unidir)]:
            aligned = M[0] * (T_mat @ M[0])
            bucket.extend(aligned.tolist())

    snr_u = float(np.mean(snrs_unclipped) / np.std(snrs_unclipped))
    snr_c = float(np.mean(snrs_clipped) / np.std(snrs_clipped))
    snr_d = float(np.mean(snrs_unidir) / np.std(snrs_unidir))
    print(f"  Unclipped SNR:      {snr_u:.4f} (theoretical sqrt((N-1)/(n-1)) = {math.sqrt(99/10):.4f})")
    print(f"  Clipped (+-1) SNR:  {snr_c:.4f} -> ratio Clipped/Unclipped = {snr_c/snr_u:.4f} (theoretical sqrt(2/pi) = {math.sqrt(2/math.pi):.4f})")
    print(f"  Unidirectional SNR: {snr_d:.4f} -> ratio Unidir/Unclipped  = {snr_d/snr_u:.4f} (theoretical 1/sqrt(2)  = {1/math.sqrt(2):.4f}, with degree var sqrt(10/10.5)/sqrt(2) = {math.sqrt(10/10.5)/math.sqrt(2):.4f})")
    assert abs(snr_c / snr_u - math.sqrt(2 / math.pi)) < 0.03 and abs(snr_d / snr_u - 1 / math.sqrt(2)) < 0.03

    print("\n=== E4.2: Bounded Saturating Synapses T_ij in {-3..+3} Palimpsest Forgetting (N=100, 40 sequential memories, p. 2557) ===")
    # Store 40 memories sequentially into N=100:
    #   - Unclipped: T += M_s M_s^T
    #   - Saturating [-3, +3]: T = clip(T + M_s M_s^T, -3, +3)
    n_seq = 40
    old_unclip, rec_unclip = [], []
    old_sat, rec_sat = [], []
    for _ in range(40):
        M = rng.choice([-1, 1], size=(n_seq, N))
        T_u = np.zeros((N, N), dtype=int)
        T_s = np.zeros((N, N), dtype=int)
        for s in range(n_seq):
            outer = np.outer(M[s], M[s])
            np.fill_diagonal(outer, 0)
            T_u += outer
            T_s = np.clip(T_s + outer, -3, 3)

        def recall_bit_acc(T_mat: np.ndarray, idx_range) -> float:
            accs = []
            for idx in idx_range:
                st = M[idx].copy()
                for _ in range(10):
                    changed = False
                    for i in rng.permutation(N):
                        h_i = int(T_mat[i] @ st)
                        nv = 1 if h_i > 0 else (-1 if h_i < 0 else st[i])
                        if nv != st[i]:
                            st[i] = nv
                            changed = True
                    if not changed:
                        break
                accs.append(float(np.mean(st == M[idx])))
            return sum(accs) / len(accs)

        old_unclip.append(recall_bit_acc(T_u, range(0, 5)))
        rec_unclip.append(recall_bit_acc(T_u, range(35, 40)))
        old_sat.append(recall_bit_acc(T_s, range(0, 5)))
        rec_sat.append(recall_bit_acc(T_s, range(37, 40)))

    print(f"  Unclipped T_ij (40 memories in N=100, overloaded): Old(#1..#5) bit acc = {np.mean(old_unclip)*100:5.1f}% | Recent(#36..#40) = {np.mean(rec_unclip)*100:5.1f}%")
    print(f"  Saturating T_ij in [-3, +3] (palimpsest, p. 2557): Old(#1..#5) bit acc = {np.mean(old_sat)*100:5.1f}% | Recent(#38..#40) = {np.mean(rec_sat)*100:5.1f}%")
    assert np.mean(rec_sat) > 0.94 and np.mean(old_sat) < 0.65

    print("\n=== E4.3: Familiarity Recognition via Initial Processing Rate under 25x Overload (N=100, n=500, p. 2558) ===")
    # "We examined the case N = 100, n = 500, in which there is a memory overload of a factor of 25.
    #  None of the memory states assigned were stable. The initial rate of processing of a starting state is defined
    #  as the number of neuron state readjustments that occur in a time 1/2W. Familiar and unfamiliar states were
    #  distinguishable most of the time at this level of overload on the basis of the initial processing rate, which
    #  was faster for unfamiliar states."
    n_over = 500
    M_over = rng.choice([-1, 1], size=(n_over, N))
    T_over = M_over.T @ M_over
    np.fill_diagonal(T_over, 0)

    # Fraction of neurons that want to flip at t=0 (aligned field < 0):
    fam_unstable = []
    unfam_unstable = []
    for s in range(n_over):
        st_fam = M_over[s]
        h_fam = T_over @ st_fam
        fam_unstable.append(int(np.sum((st_fam * h_fam) < 0)))

        st_unfam = rng.choice([-1, 1], size=N)
        h_unfam = T_over @ st_unfam
        unfam_unstable.append(int(np.sum((st_unfam * h_unfam) < 0)))

    # Compute accuracy at threshold = midpoint & ROC AUC:
    fam_arr = np.array(fam_unstable)
    unfam_arr = np.array(unfam_unstable)
    thresh = 0.5 * (np.mean(fam_arr) + np.mean(unfam_arr))
    cls_acc = 0.5 * (np.mean(fam_arr < thresh) + np.mean(unfam_arr > thresh))
    auc = float(np.mean(fam_arr[:, None] < unfam_arr[None, :]) + 0.5 * np.mean(fam_arr[:, None] == unfam_arr[None, :]))
    print(f"  N=100, n=500 (25x overload): stable memories = {int(np.sum(fam_arr == 0))}/500")
    print(f"  Mean unstable neurons at t=0: Familiar = {np.mean(fam_arr):.2f} +/- {np.std(fam_arr):.2f} (theo 100*(1-Phi(1/sqrt(5))) = {100*(1-normal_cdf(1/math.sqrt(5))):.2f})")
    print(f"                                Unfamiliar = {np.mean(unfam_arr):.2f} +/- {np.std(unfam_arr):.2f} (theo 50.00)")
    print(f"  Single-step familiarity classification accuracy = {cls_acc*100:5.1f}%, ROC AUC = {auc:.4f}")
    assert int(np.sum(fam_arr == 0)) == 0 and cls_acc > 0.90 and auc > 0.97


if __name__ == "__main__":
    run_e2_energy_and_capacity()
    run_e3_basins_and_merging()
    run_e4_robustness_and_familiarity()
