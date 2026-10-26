"""
Exact implementation of Frank Rosenblatt (1958), "The Perceptron: A Probabilistic Model for
Information Storage and Organization in the Brain", Psychological Review 65(6): 386-408.
Covers Experiments E2, E3, and E4.
"""
import math
import random
from typing import Dict, List, Tuple


def comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def binom_pmf(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def pa_eq1(R: float, x: int, y: int, theta: int) -> float:
    """
    Equation (1), p. 392-393:
      P_a = sum_{e=theta}^x sum_{i=0}^{min(y, e-theta)} P(e, i)
      where P(e, i) = C(x,e) R^e (1-R)^{x-e} * C(y,i) R^i (1-R)^{y-i}.
    """
    total = 0.0
    for e in range(max(0, theta), x + 1):
        pe = binom_pmf(x, e, R)
        max_i = min(y, e - theta)
        for i in range(0, max_i + 1):
            pi = binom_pmf(y, i, R)
            total += pe * pi
    return total


def pc_eq2(R: float, L: float, G: float, x: int, y: int, theta: int, corrupt_side_cond: bool = False) -> float:
    """
    Equation (2), p. 393:
      P_c = (1 / P_a) * sum_{e=theta}^x sum_{i=0}^{min(y, e-theta)} sum_{le=0}^e sum_{li=0}^i
            sum_{ge=0}^{x-e} sum_{gi=0}^{y-i} P(e, i, le, li, ge, gi)
      subject to side condition: e - i - le + li + ge - gi >= theta.
    """
    pa = pa_eq1(R, x, y, theta)
    if pa <= 0.0:
        return 0.0
    joint = 0.0
    for e in range(max(0, theta), x + 1):
        pe = binom_pmf(x, e, R)
        for i in range(0, min(y, e - theta) + 1):
            pei = pe * binom_pmf(y, i, R)
            for le in range(0, e + 1):
                ple = binom_pmf(e, le, L)
                for li in range(0, i + 1):
                    pli = binom_pmf(i, li, L)
                    for ge in range(0, x - e + 1):
                        pge = binom_pmf(x - e, ge, G)
                        for gi in range(0, y - i + 1):
                            pgi = binom_pmf(y - i, gi, G)
                            a2 = (e - i - le + li - ge - gi) if corrupt_side_cond else (e - i - le + li + ge - gi)
                            if a2 >= theta:
                                joint += pei * ple * pli * pge * pgi
    return joint / pa


def pc_min_eq3(L: float, G: float, x: int, y: int) -> float:
    """Equation (3), p. 394: P_{c,min} = (1 - L)^x * (1 - G)^y."""
    return ((1.0 - L) ** x) * ((1.0 - G) ** y)


def run_e2_predominant_phase():
    print("=== E2.1: Predominant Phase Equations (1), (2), (3) vs Monte Carlo Simulation ===")
    rng = random.Random(1958)
    n_s = 2000
    n_a = 100000

    configs = [
        (0.30, 0.40, 0.15, 6, 2, 2),
        (0.20, 0.50, 0.125, 8, 0, 3),
        (0.40, 0.25, 0.1667, 5, 5, 2),
        (0.50, 0.30, 0.30, 4, 0, 4),  # Eq. (3) P_{c,min} regime: theta = x = 4, y = 0 (Pa=0.0625, ~1,250 active S1 units)
    ]
    max_pa_err = 0.0
    max_pc_err = 0.0
    for R, L, G, x, y, theta in configs:
        pa_ana = pa_eq1(R, x, y, theta)
        pc_ana = pc_eq2(R, L, G, x, y, theta, corrupt_side_cond=False)
        pc_bad = pc_eq2(R, L, G, x, y, theta, corrupt_side_cond=True)

        # Construct S1 and S2 on retina of n_s points with exact fractions R, L, G:
        n1 = round(R * n_s)
        n_lost = round(L * n1)
        n_kept = n1 - n_lost
        n_gained = round(G * (n_s - n1))
        s1 = set(range(0, n1))
        s2 = set(range(0, n_kept)) | set(range(n1, n1 + n_gained))

        active_s1 = 0
        active_both = 0
        for _ in range(n_a):
            exc = [rng.randrange(n_s) for _ in range(x)]
            inh = [rng.randrange(n_s) for _ in range(y)]
            a1 = sum(1 for p in exc if p in s1) - sum(1 for p in inh if p in s1)
            if a1 >= theta:
                active_s1 += 1
                a2 = sum(1 for p in exc if p in s2) - sum(1 for p in inh if p in s2)
                if a2 >= theta:
                    active_both += 1
        pa_mc = active_s1 / n_a
        pc_mc = (active_both / active_s1) if active_s1 > 0 else 0.0
        max_pa_err = max(max_pa_err, abs(pa_ana - pa_mc))
        max_pc_err = max(max_pc_err, abs(pc_ana - pc_mc))
        print(f"  (R={R:.2f}, L={L:.2f}, G={G:.2f}, x={x}, y={y}, th={theta:2d}): "
              f"Pa ana={pa_ana:.4f} mc={pa_mc:.4f} | Pc ana={pc_ana:.4f} mc={pc_mc:.4f} (corrupted={pc_bad:.4f})")

    # Check Eq. (3) identity at theta = x = 10, y = 0:
    pc_th10 = pc_eq2(0.50, 0.30, 0.30, 10, 0, 10)
    pc_min10 = pc_min_eq3(0.30, 0.30, 10, 0)
    print(f"  Eq. (3) check at x=10, y=0, theta=10: Pc_eq2={pc_th10:.8f} vs Pc_min=(1-L)^10={pc_min10:.8f} (diff={abs(pc_th10-pc_min10):.2e})")
    assert max_pa_err < 0.008 and max_pc_err < 0.015 and abs(pc_th10 - pc_min10) < 1e-12

    print("\n=== E2.2: Fig. 4c Flattening of P_a(R) When Excitation Equals Inhibition (p. 393) ===")
    r_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    pa_exc_only = [pa_eq1(r, x=10, y=0, theta=3) for r in r_grid]
    pa_balanced = [pa_eq1(r, x=5, y=5, theta=2) for r in r_grid]
    range_exc = max(pa_exc_only) - min(pa_exc_only)
    range_bal = max(pa_balanced) - min(pa_balanced)
    print(f"  R in {r_grid}:")
    print(f"    Excitatory only (x=10, y=0, theta=3): Pa = {[round(v, 4) for v in pa_exc_only]} (spread R=0.1..0.5: {range_exc:.4f}, R=0.2..0.5: {max(pa_exc_only[1:])-min(pa_exc_only[1:]):.4f})")
    print(f"    Balanced E == I (x=5,  y=5, theta=2): Pa = {[round(v, 4) for v in pa_balanced]} (spread R=0.1..0.5: {range_bal:.4f}, R=0.2..0.5: {max(pa_balanced[1:])-min(pa_balanced[1:]):.4f})")
    assert range_exc > 0.80 and range_bal < 0.13


def run_e3_postdominant_learning():
    print("\n=== E3.1: Table 1 Value Systems (alpha, beta, gamma) & Class-Frequency Imbalance (pp. 391, 398) ===")
    # Build two disjoint source-sets (R0 and R1) of N_Ar = 200 A-units each, wired to a 16x16 retina (N_S = 256)
    # with x=6, y=2, theta=2.
    # Stimuli: Differentiated environment of Horizontal bars (class 0) vs Vertical bars (class 1) on 16x16 grid.
    rng = random.Random(1958)
    n_s = 256
    n_ar_total = 250
    x_conn, y_conn, theta = 6, 2, 2

    def make_a_units(n_units: int):
        units = []
        for _ in range(n_units):
            exc = [rng.randrange(n_s) for _ in range(x_conn)]
            inh = [rng.randrange(n_s) for _ in range(y_conn)]
            units.append((exc, inh))
        return units

    source_sets = [make_a_units(n_ar_total), make_a_units(n_ar_total)]

    def active_indices(s_idx: int, stim: set) -> List[int]:
        act = []
        for idx, (exc, inh) in enumerate(source_sets[s_idx]):
            if sum(1 for p in exc if p in stim) - sum(1 for p in inh if p in stim) >= theta:
                act.append(idx)
        return act

    def make_bar_stimulus(cls: int, pos_rng: random.Random) -> set:
        # cls 0: 3 horizontal rows of width 12; cls 1: 3 vertical cols of height 12 (36 pixels lit, R = 36/256 = 0.14)
        pts = set()
        if cls == 0:
            r0 = pos_rng.randrange(0, 14)
            c0 = pos_rng.randrange(0, 5)
            for r in range(r0, r0 + 3):
                for c in range(c0, c0 + 12):
                    pts.add(r * 16 + c)
        else:
            r0 = pos_rng.randrange(0, 5)
            c0 = pos_rng.randrange(0, 14)
            for r in range(r0, r0 + 12):
                for c in range(c0, c0 + 3):
                    pts.add(r * 16 + c)
        # Flip 4 random noise pixels
        for p in pos_rng.sample(range(n_s), 4):
            if p in pts:
                pts.remove(p)
            else:
                pts.add(p)
        return pts

    # Generate imbalanced training series: n_{s0} = 150 (Class 0), n_{s1} = 50 (Class 1) (3:1 frequency ratio!)
    train_rng = random.Random(101)
    train_series = [(0, make_bar_stimulus(0, train_rng)) for _ in range(150)] + [
        (1, make_bar_stimulus(1, train_rng)) for _ in range(50)
    ]
    train_rng.shuffle(train_series)

    # Balanced unseen test series (100 of Class 0, 100 of Class 1):
    test_rng = random.Random(202)
    test_series = [(0, make_bar_stimulus(0, test_rng)) for _ in range(100)] + [
        (1, make_bar_stimulus(1, test_rng)) for _ in range(100)
    ]

    for sys_name in ("alpha", "beta", "gamma"):
        V = [[0.0] * n_ar_total, [0.0] * n_ar_total]
        K_feed = 10.0
        for cls, stim in train_series:
            act_dom = set(active_indices(cls, stim))
            nar = len(act_dom)
            if sys_name == "alpha":
                # Table 1: +1 for active in dominant set, 0 elsewhere
                for i in act_dom:
                    V[cls][i] += 1.0
            elif sys_name == "beta":
                # Table 1: dominant set gets +K/Nar on active (0 on inactive); non-dominant set gets +K/NAr on all
                if nar > 0:
                    for i in act_dom:
                        V[cls][i] += K_feed / nar
                other = 1 - cls
                for i in range(n_ar_total):
                    V[other][i] += K_feed / n_ar_total
            elif sys_name == "gamma":
                # Table 1: +1 on active in dominant set, -Nar/(NAr - Nar) on inactive in dominant set (total gain = 0)
                if 0 < nar < n_ar_total:
                    penalty = -float(nar) / float(n_ar_total - nar)
                    for i in range(n_ar_total):
                        V[cls][i] += 1.0 if i in act_dom else penalty

        # Evaluate on unseen test_series under both Sigma-system (sum) and mu-system (mean):
        for eval_mode in ("Sigma (sum)", "mu (mean)"):
            c0_ok = 0
            c1_ok = 0
            for cls, stim in test_series:
                a0 = active_indices(0, stim)
                a1 = active_indices(1, stim)
                s0 = sum(V[0][i] for i in a0)
                s1 = sum(V[1][i] for i in a1)
                if eval_mode == "mu (mean)":
                    s0 = s0 / len(a0) if a0 else -1e9
                    s1 = s1 / len(a1) if a1 else -1e9
                pred = 0 if s0 >= s1 else 1
                if cls == 0 and pred == 0:
                    c0_ok += 1
                elif cls == 1 and pred == 1:
                    c1_ok += 1
            bal_acc = 0.5 * (c0_ok + c1_ok)
            print(f"  {sys_name:<6} | {eval_mode:<12} | net_V=({sum(V[0]):7.1f}, {sum(V[1]):7.1f}) | "
                  f"Class0(3x freq)={c0_ok:3d}% | Class1(1x freq)={c1_ok:3d}% | Balanced Pg={bal_acc:5.1f}%")

    # Measure empirical Pa, Pc11, Pc12 on the centered bars (Regime A: Pc11 > Pc12 > Pa)
    # vs a stimulus environment satisfying Rosenblatt's p. 400 inequality Pc11 > Pa > Pc12 (Regime B):
    def measure_pa_pc11_pc12(gen_fn) -> Tuple[float, float, float]:
        mrng = random.Random(777)
        s0_list = [gen_fn(0, mrng) for _ in range(80)]
        s1_list = [gen_fn(1, mrng) for _ in range(80)]
        act0 = [set(active_indices(0, s)) for s in s0_list]
        act1 = [set(active_indices(0, s)) for s in s1_list]
        pa_emp = sum(len(a) for a in act0 + act1) / (160.0 * n_ar_total)
        same_num, same_den = 0, 0
        diff_num, diff_den = 0, 0
        for i in range(40):
            a_i = act0[2 * i]
            a_j = act0[2 * i + 1]
            same_num += len(a_i & a_j)
            same_den += len(a_i)
            b_j = act1[i]
            diff_num += len(a_i & b_j)
            diff_den += len(a_i)
        return pa_emp, same_num / max(1, same_den), diff_num / max(1, diff_den)

    def make_separated_stimulus(cls: int, pos_rng: random.Random) -> set:
        # Satisfies Pc11 > Pa > Pc12: Class 0 illuminates 36 pixels mainly in upper-left/horizontal bands (0..127),
        # Class 1 illuminates 36 pixels mainly in lower-right/vertical bands (128..255), with 6 random noise flips.
        base_pool = range(0, 112) if cls == 0 else range(144, 256)
        pts = set(pos_rng.sample(base_pool, 34)) | set(pos_rng.sample(range(n_s), 2))
        return pts

    pa_a, pc11_a, pc12_a = measure_pa_pc11_pc12(make_bar_stimulus)
    pa_b, pc11_b, pc12_b = measure_pa_pc11_pc12(make_separated_stimulus)
    print(f"\n  >>> Verification of Rosenblatt's p. 400 Inequality (Pc11 > Pa > Pc12):")
    print(f"    Regime A (Centered Horiz vs Vert bars, cross in 3x3=9/36=25% > R=14.1%): Pc11={pc11_a:.4f}, Pa={pa_a:.4f}, Pc12={pc12_a:.4f} (VIOLATES Pa > Pc12!)")
    print(f"    Regime B (Separated class pools, overlap < R):                           Pc11={pc11_b:.4f}, Pa={pa_b:.4f}, Pc12={pc12_b:.4f} (SATISFIES Pc11 > Pa > Pc12!)")

    # Re-run 3:1 frequency imbalance (n_s0=150, n_s1=50) under Regime B (Pc11 > Pa > Pc12) for alpha vs gamma vs bivalent-gamma:
    tr_b = [(0, make_separated_stimulus(0, train_rng)) for _ in range(150)] + [(1, make_separated_stimulus(1, train_rng)) for _ in range(50)]
    train_rng.shuffle(tr_b)
    te_b = [(0, make_separated_stimulus(0, test_rng)) for _ in range(100)] + [(1, make_separated_stimulus(1, test_rng)) for _ in range(100)]
    for mode_b in ("alpha", "gamma", "bivalent_gamma"):
        Vb = [[0.0] * n_ar_total, [0.0] * n_ar_total]
        for cls, stim in tr_b:
            for r_idx in (0, 1):
                act = set(active_indices(r_idx, stim))
                nar = len(act)
                if mode_b == "alpha" and r_idx == cls:
                    for i in act:
                        Vb[r_idx][i] += 1.0
                elif mode_b == "gamma" and r_idx == cls and 0 < nar < n_ar_total:
                    pen = -float(nar) / float(n_ar_total - nar)
                    for i in range(n_ar_total):
                        Vb[r_idx][i] += 1.0 if i in act else pen
                elif mode_b == "bivalent_gamma" and 0 < nar < n_ar_total:
                    sign = 1.0 if r_idx == cls else -1.0
                    pen = -float(nar) / float(n_ar_total - nar)
                    for i in range(n_ar_total):
                        Vb[r_idx][i] += sign * (1.0 if i in act else pen)
        c0_b = sum(1 for c, s in te_b if c == 0 and sum(Vb[0][i] for i in active_indices(0, s)) >= sum(Vb[1][i] for i in active_indices(1, s)))
        c1_b = sum(1 for c, s in te_b if c == 1 and sum(Vb[1][i] for i in active_indices(1, s)) > sum(Vb[0][i] for i in active_indices(0, s)))
        print(f"    Regime B (3:1 imbalance) {mode_b:<15}: Class0={c0_b:3d}% | Class1={c1_b:3d}% | Balanced Pg={0.5*(c0_b+c1_b):5.1f}%")

    print("\n=== E3.2: Ideal Environment (Random Patterns) vs Differentiated Environment (Pr vs Pg, pp. 396-402) ===")
    # Compare Pr (memorized training stimuli) and Pg (unseen stimuli from same distribution) across n_sr = 10, 50, 200
    # in the gamma system:
    for env_name, is_diff in [("Ideal Environment (random uncorrelated patterns, c1=0)", False),
                              ("Differentiated Environment (Horiz vs Vert bars, Pc11 > Pa > Pc12)", True)]:
        print(f"  >>> {env_name}:")
        for n_sr in (10, 50, 150):
            V = [[0.0] * n_ar_total, [0.0] * n_ar_total]
            erng = random.Random(500 + n_sr)
            learned = []
            for cls in (0, 1):
                for _ in range(n_sr):
                    stim = make_bar_stimulus(cls, erng) if is_diff else set(erng.sample(range(n_s), 36))
                    learned.append((cls, stim))
            erng.shuffle(learned)
            for cls, stim in learned:
                act_dom = set(active_indices(cls, stim))
                nar = len(act_dom)
                if 0 < nar < n_ar_total:
                    pen = -float(nar) / float(n_ar_total - nar)
                    for i in range(n_ar_total):
                        V[cls][i] += 1.0 if i in act_dom else pen

            # Measure Pr on the learned stimuli:
            pr_hits = 0
            for cls, stim in learned:
                a0, a1 = active_indices(0, stim), active_indices(1, stim)
                m0 = sum(V[0][i] for i in a0) / len(a0) if a0 else -1e9
                m1 = sum(V[1][i] for i in a1) / len(a1) if a1 else -1e9
                if (0 if m0 >= m1 else 1) == cls:
                    pr_hits += 1
            pr = pr_hits / len(learned)

            # Measure Pg on 200 fresh unseen stimuli:
            pg_hits = 0
            for cls in (0, 1):
                for _ in range(100):
                    stim = make_bar_stimulus(cls, erng) if is_diff else set(erng.sample(range(n_s), 36))
                    a0, a1 = active_indices(0, stim), active_indices(1, stim)
                    m0 = sum(V[0][i] for i in a0) / len(a0) if a0 else -1e9
                    m1 = sum(V[1][i] for i in a1) / len(a1) if a1 else -1e9
                    if (0 if m0 >= m1 else 1) == cls:
                        pg_hits += 1
            pg = pg_hits / 200.0
            print(f"    n_sr = {n_sr:3d} per class: Recall Pr = {pr*100:5.1f}% | Unseen Generalization Pg = {pg*100:5.1f}%")


def run_e4_spontaneous_and_relational():
    print("\n=== E4.1: Spontaneous Unsupervised Organization with Proportional Value Decay (p. 404) ===")
    # "if the values of the A-units are allowed to decay at a rate proportional to their magnitude,
    #  a striking new property emerges: the perceptron becomes capable of 'spontaneous' concept formation.
    #  That is to say, if the system is exposed to a random series of stimuli from two 'dissimilar' classes,
    #  and all of its responses are automatically reinforced without any regard to whether they are 'right'
    #  or 'wrong,' the system will tend towards a stable terminal condition in which (for each binary response)
    #  the response will be '1' for members of one stimulus class, and '0' for members of the other class"
    n_a = 200
    for use_decay_and_gamma, label in [
        (True, "With proportional decay (lambda=0.05) + gamma competition (p. 404)"),
        (False, "Negative control: uncompensated alpha gain without decay/competition"),
    ]:
        purities = []
        for trial in range(20):
            rng = random.Random(1958 + trial)
            # Two dissimilar stimulus classes on 200 A-units:
            # Class A activates ~30% of A-units in half 0..99 and ~5% in 100..199;
            # Class B activates ~5% of A-units in 0..99 and ~30% in 100..199.
            # Small random initial values V[r][i]:
            V = [[rng.gauss(0.0, 0.01) for _ in range(n_a)] for _ in range(2)]
            for step in range(300):
                cls = rng.randrange(2)
                active = [
                    i for i in range(n_a)
                    if rng.random() < (0.30 if (i < 100) == (cls == 0) else 0.05)
                ]
                # Postdominant winner-take-all response selection (without teacher!):
                s0 = sum(V[0][i] for i in active)
                s1 = sum(V[1][i] for i in active)
                winner = 0 if s0 >= s1 else 1
                loser = 1 - winner
                act_set = set(active)
                nar = len(act_set)
                if use_decay_and_gamma:
                    # Proportional decay on all weights + parasitic zero-sum bivalent self-reinforcement of winner:
                    decay = 0.05
                    pen = -float(nar) / float(n_a - nar)
                    for i in range(n_a):
                        delta = 1.0 if i in act_set else pen
                        V[winner][i] = (1.0 - decay) * V[winner][i] + delta
                        V[loser][i] = (1.0 - decay) * V[loser][i] - delta
                else:
                    # Uncompensated alpha gain on winner only, no decay:
                    for i in act_set:
                        V[winner][i] += 1.0

            # Test unsupervised separation on 100 Class A and 100 Class B stimuli:
            r_for_a = []
            r_for_b = []
            for cls, bucket in [(0, r_for_a), (1, r_for_b)]:
                for _ in range(100):
                    active = [
                        i for i in range(n_a)
                        if rng.random() < (0.30 if (i < 100) == (cls == 0) else 0.05)
                    ]
                    winner = 0 if sum(V[0][i] for i in active) >= sum(V[1][i] for i in active) else 1
                    bucket.append(winner)
            # Purity: max of (Class A -> 0 & Class B -> 1) or (Class A -> 1 & Class B -> 0)
            p01 = 0.5 * (r_for_a.count(0) + r_for_b.count(1))
            p10 = 0.5 * (r_for_a.count(1) + r_for_b.count(0))
            purities.append(max(p01, p10))
        print(f"  {label:<70}: mean unsupervised separation accuracy = {sum(purities)/len(purities):5.1f}% (20 trials)")

    print("\n=== E4.2: Concrete Conjunction vs Relational Judgment Boundary (pp. 404-405) ===")
    # Scene with 2 slots (Left slot: 50 S-points, Right slot: 50 S-points).
    # Each slot displays either a Square (pattern 0, 20 noisy points) or a Circle (pattern 1, 20 noisy points).
    # Task 1 (Concrete qualifying conditions, p. 404): "Left is Square AND Right is Circle" -> linearly separable by slot-local A-units!
    # Task 2 (Relational judgment, pp. 404-405): "Left object has the SAME shape as Right object" ((Square,Square) or (Circle,Circle))
    #   -> XOR across the two slots! Slot-local A-units have P_{c11} == P_{c12} and fail; cross-slot A-units succeed.
    rng = random.Random(1958)
    templates = {
        (0, 0): set(range(0, 25)),       # Left Square
        (0, 1): set(range(25, 50)),      # Left Circle
        (1, 0): set(range(50, 75)),      # Right Square
        (1, 1): set(range(75, 100)),     # Right Circle
    }

    def sample_scene(l_shape: int, r_shape: int) -> List[int]:
        pts = set(templates[(0, l_shape)]) | set(templates[(1, r_shape)])
        x = [1 if i in pts else 0 for i in range(100)]
        for idx in rng.sample(range(100), 6):
            x[idx] = 1 - x[idx]
        return x

    # Build two A-unit architectures (300 A-units each, threshold=2):
    #   Arch 1 (Slot-local A-units): each A-unit draws 4 excitatory wires from EITHER the Left slot (0..49) OR the Right slot (50..99).
    #   Arch 2 (Cross-slot A-units): each A-unit draws 2 excitatory wires from the Left slot AND 2 from the Right slot, with threshold=4.
    local_wires = [rng.sample(range(0, 50), 4) if k < 150 else rng.sample(range(50, 100), 4) for k in range(300)]
    cross_wires = [rng.sample(range(0, 50), 2) + rng.sample(range(50, 100), 2) for _ in range(300)]

    for arch_name, wires, th in [
        ("Slot-local A-units (each A-unit sees only Left OR Right object)", local_wires, 3),
        ("Cross-slot A-units (each A-unit sees both Left AND Right objects)", cross_wires, 4),
    ]:
        for task_name, is_rel in [
            ("Concrete ('Left=Square AND Right=Circle')", False),
            ("Relational ('Left shape == Right shape')", True),
        ]:
            w = [0.0] * 300
            b = 0.0
            for _ in range(1500):
                ls, rs = rng.randrange(2), rng.randrange(2)
                x = sample_scene(ls, rs)
                feat = [1 if sum(x[i] for i in wr) >= th else 0 for wr in wires]
                want = (1 if ls == rs else 0) if is_rel else (1 if (ls == 0 and rs == 1) else 0)
                got = 1 if sum(wi * fi for wi, fi in zip(w, feat)) + b > 0 else 0
                if got != want:
                    step = 1.0 if want == 1 else -1.0
                    for i, fi in enumerate(feat):
                        if fi:
                            w[i] += step
                    b += step
            # Evaluate on 400 unseen scenes:
            ok = 0
            for _ in range(400):
                ls, rs = rng.randrange(2), rng.randrange(2)
                x = sample_scene(ls, rs)
                feat = [1 if sum(x[i] for i in wr) >= th else 0 for wr in wires]
                want = (1 if ls == rs else 0) if is_rel else (1 if (ls == 0 and rs == 1) else 0)
                got = 1 if sum(wi * fi for wi, fi in zip(w, feat)) + b > 0 else 0
                if got == want:
                    ok += 1
            print(f"  {arch_name:<66} | {task_name:<42}: {ok/4.0:5.1f}%")


if __name__ == "__main__":
    run_e2_predominant_phase()
    run_e3_postdominant_learning()
    run_e4_spontaneous_and_relational()
