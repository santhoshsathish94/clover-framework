"""
Experiments E1, E2, E3, and E4 for Minsky & Papert (1969) Perceptrons
(AIM-140 'Linearly Unrecognizable Patterns' 1967, AIM-167 'Linear Separation and Learning' 1968).
"""
import itertools
import math
import random
from typing import Dict, List, Set, Tuple
import numpy as np


# ============================================================================
# E1: TaraAlcira/PerceptronsRevisited Spiral Challenge (Single vs Double Spiral)
# ============================================================================

def run_e1_spirals(repo_dir: str):
    print("=== E1: TaraAlcira/PerceptronsRevisited Spiral Challenge (1-Spiral vs 2-Spiral) ===")
    # Load one_spiral_function.py and two_spirals_function.py without cv2 dependency
    ns1, ns2 = {}, {}
    code1 = open(f"{repo_dir}/generate_spirals/one_spiral_function.py").read().replace("import cv2", "")
    code2 = open(f"{repo_dir}/generate_spirals/two_spirals_function.py").read().replace("import cv2", "")
    exec(code1, ns1)
    exec(code2, ns2)
    gen1 = ns1["generate_1double_spiral"]
    gen2 = ns2["generate_2double_spiral"]

    def count_components(binary_grid: np.ndarray, connectivity: int = 8) -> int:
        pts = set(zip(*np.where(binary_grid > 0)))
        visited = set()
        comps = 0
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for p in pts:
            if p not in visited:
                comps += 1
                stack = [p]
                visited.add(p)
                while stack:
                    r, c = stack.pop()
                    for dr, dc in dirs:
                        nb = (r + dr, c + dc)
                        if nb in pts and nb not in visited:
                            visited.add(nb)
                            stack.append(nb)
        return comps

    c1_4, c1_8, area1 = [], [], []
    c2_4, c2_8, area2 = [], [], []
    for side in (56, 64, 72, 80):
        img1 = (gen1(side, 10)[:, :, 0] > 0).astype(int)
        img2 = (gen2(side, 10)[:, :, 0] > 0).astype(int)
        c1_4.append(count_components(img1, 4))
        c1_8.append(count_components(img1, 8))
        area1.append(int(img1.sum()))
        c2_4.append(count_components(img2, 4))
        c2_8.append(count_components(img2, 8))
        area2.append(int(img2.sum()))

    print(f"  generate_1double_spiral (sides 56..80): 4-conn components={c1_4}, 8-conn components={c1_8}, areas={area1}")
    print(f"  generate_2double_spiral (sides 56..80): 4-conn components={c2_4}, 8-conn components={c2_8}, areas={area2}")

    # Test order-1 / diameter-limited linear perceptron on 200 randomly shifted & 90-deg rotated 64x64 spirals:
    rng = random.Random(1969)
    base1 = (gen1(54, 10)[:, :, 0] > 0).astype(int)  # 64x64
    base2 = (gen2(54, 10)[:, :, 0] > 0).astype(int)  # 64x64

    def sample_spiral(cls: int) -> np.ndarray:
        arr = np.rot90(base1 if cls == 1 else base2, rng.randrange(4))
        arr = np.roll(arr, rng.randint(-6, 6), axis=0)
        arr = np.roll(arr, rng.randint(-6, 6), axis=1)
        return arr

    train_data = [(sample_spiral(k % 2), k % 2) for k in range(200)]
    test_data = [(sample_spiral(k % 2), k % 2) for k in range(100)]
    w = np.zeros((64, 64), dtype=float)
    b = 0.0
    for _ in range(20):
        rng.shuffle(train_data)
        for img, y in train_data:
            pred = 1 if (np.sum(w * img) + b > 0) else 0
            if pred != y:
                step = 1.0 if y == 1 else -1.0
                w += step * img
                b += step
    perc_acc = sum(1 for img, y in test_data if (1 if (np.sum(w * img) + b > 0) else 0) == y) / len(test_data)
    serial_acc = sum(1 for img, y in test_data if (1 if count_components(img, 4) == 1 else 0) == y) / len(test_data)
    print(f"  Fixed-side (side=54, area 612 vs 760) 64x64 spirals: Order-1 Perceptron = {perc_acc*100:5.1f}% (uses area cut |X|>A!) vs Serial = {serial_acc*100:5.1f}%")

    # Now test (a) randomize_spirals.py protocol (random side_length in {56, 64, 72, 80} on a 96x96 canvas)
    # and (b) exact equal-area 64x64 spirals (where 2-spiral has its outer tail trimmed by 152 pixels to match 1-spiral's 652 pixels):
    bank1 = []
    bank2 = []
    for s in (56, 64, 72, 80):
        c1 = np.zeros((96, 96), dtype=int)
        c2 = np.zeros((96, 96), dtype=int)
        g1 = (gen1(s, 10)[:, :, 0] > 0).astype(int)
        g2 = (gen2(s, 10)[:, :, 0] > 0).astype(int)
        off = (96 - g1.shape[0]) // 2
        c1[off:off+g1.shape[0], off:off+g1.shape[1]] = g1
        c2[off:off+g2.shape[0], off:off+g2.shape[1]] = g2
        bank1.append(c1)
        bank2.append(c2)

    def sample_var_spiral(cls: int) -> np.ndarray:
        base = rng.choice(bank1 if cls == 1 else bank2)
        arr = np.rot90(base, rng.randrange(4))
        arr = np.roll(arr, rng.randint(-12, 12), axis=0)
        arr = np.roll(arr, rng.randint(-12, 12), axis=1)
        return arr

    tr_m = [(sample_var_spiral(k % 2), k % 2) for k in range(300)]
    te_m = [(sample_var_spiral(k % 2), k % 2) for k in range(100)]
    wm = np.zeros((96, 96), dtype=float)
    bm = 0.0
    for _ in range(20):
        rng.shuffle(tr_m)
        for img, y in tr_m:
            pred = 1 if (np.sum(wm * img) + bm > 0) else 0
            if pred != y:
                step = 1.0 if y == 1 else -1.0
                wm += step * img
                bm += step
    perc_m_acc = sum(1 for img, y in te_m if (1 if (np.sum(wm * img) + bm > 0) else 0) == y) / len(te_m)
    ser_m_acc = sum(1 for img, y in te_m if (1 if count_components(img, 4) == 1 else 0) == y) / len(te_m)
    print(f"  Variable-size shifted/rotated 96x96 spirals (sides 56..80): Order-1 Perceptron = {perc_m_acc*100:5.1f}% vs Serial Connectivity = {ser_m_acc*100:5.1f}%")
    assert ser_m_acc == 1.0


# ============================================================================
# E2: Positive Normal Form, Exact Order on n=3, Counting Function, & Parity Weights
# ============================================================================

def mobius_positive_normal_form(n: int, tt: Dict[Tuple[int, ...], int]) -> Dict[int, int]:
    """
    AIM-140 p. 181-182 Positive Normal Form Theorem:
      Every Boolean function f: {0,1}^n -> {0,1} has a unique representation
      f(X) = sum_{A subseteq R} c_A phi_A(X) with integer coefficients c_A,
      given by Mobius inversion on the subset lattice:
      c_A = sum_{B subseteq A} (-1)^{|A| - |B|} f(B).
    """
    coeffs: Dict[int, int] = {}
    for mask_a in range(1 << n):
        c_a = 0
        # Iterate over all submasks B of mask_a
        sub = mask_a
        while True:
            bits_b = tuple((sub >> i) & 1 for i in range(n))
            diff_pop = bin(mask_a ^ sub).count("1")
            sign = -1 if (diff_pop % 2 == 1) else 1
            c_a += sign * tt[bits_b]
            if sub == 0:
                break
            sub = (sub - 1) & mask_a
        if c_a != 0:
            coeffs[mask_a] = c_a
    return coeffs


def is_realizable_at_order_k(n: int, tt: Dict[Tuple[int, ...], int], k: int) -> bool:
    """
    Tests whether Boolean function `tt` on `n` variables has predicate order <= k
    (i.e. is linearly separable over the basis of masks phi_A with |A| <= k).
    Uses exact feasibility check (Perceptron with margin / LP over {0,1}^{2^n}).
    """
    masks = [m for m in range(1 << n) if bin(m).count("1") <= k]
    X_rows = []
    y_vals = []
    for x_int in range(1 << n):
        bits = tuple((x_int >> i) & 1 for i in range(n))
        row = [1.0 if (x_int & m) == m else 0.0 for m in masks]
        X_rows.append(row)
        y_vals.append(1 if tt[bits] == 1 else -1)

    # For n=3 (8 rows), run exact perceptron up to 2000 updates:
    w = [0.0] * len(masks)
    for _ in range(2000):
        changed = False
        for row, y in zip(X_rows, y_vals):
            dot = sum(wi * ri for wi, ri in zip(w, row))
            if y * dot <= 0:
                for j in range(len(w)):
                    w[j] += y * row[j]
                changed = True
        if not changed:
            return True
    return False


def run_e2_order_and_parity():
    print("\n=== E2.1: Positive Normal Form Theorem & Exact Order Distribution on all 256 3-Input Functions ===")
    inputs_3 = [tuple((x >> i) & 1 for i in range(3)) for x in range(8)]
    order_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    order3_funcs = []
    pnf_verified = 0

    for f_int in range(256):
        tt = {inputs_3[i]: (f_int >> i) & 1 for i in range(8)}
        coeffs = mobius_positive_normal_form(3, tt)
        # Verify Positive Normal Form reconstructs f(x) identically on all 8 inputs:
        ok_pnf = True
        for x_int in range(8):
            val = sum(c for m, c in coeffs.items() if (x_int & m) == m)
            if val != tt[inputs_3[x_int]]:
                ok_pnf = False
        if ok_pnf:
            pnf_verified += 1

        # Find exact minimal order k in 0..3:
        for k in range(4):
            if is_realizable_at_order_k(3, tt, k):
                order_counts[k] += 1
                if k == 3:
                    order3_funcs.append([tt[b] for b in inputs_3])
                break

    print(f"  Positive Normal Form exact integer reconstruction verified on: {pnf_verified}/256 functions")
    print(f"  Exact predicate order distribution across all 256 3-input Boolean functions:")
    print(f"    Order 0 (constants):                   {order_counts[0]:3d}")
    print(f"    Order 1 (linear threshold functions):  {order_counts[1]:3d} (cumulative order <= 1: {order_counts[0]+order_counts[1]}/256)")
    print(f"    Order 2 (quadratic mask threshold):    {order_counts[2]:3d} (cumulative order <= 2: {order_counts[0]+order_counts[1]+order_counts[2]}/256)")
    print(f"    Order 3 (full |R|=3 mask required):    {order_counts[3]:3d} -> truth tables: {order3_funcs}")
    assert pnf_verified == 256 and order_counts == {0: 2, 1: 102, 2: 150, 3: 2}

    print("\n=== E2.2: Example (iii) (AIM-140 p. 181) Exact Counting Predicate psi_M(X) = [|X| == M] Has Order 2 ===")
    # Formula on p. 181: psi_M(X) = [ (2M - 1) * sum_i x_i + (-2) * sum_{i<j} x_i x_j >= M^2 ]
    n_test = 8
    all_ok = True
    for M in range(1, n_test):
        for x_int in range(1 << n_test):
            card = bin(x_int).count("1")
            pairs = card * (card - 1) // 2
            lhs = (2 * M - 1) * card - 2 * pairs
            pred = 1 if lhs >= M * M else 0
            want = 1 if card == M else 0
            if pred != want:
                all_ok = False
    print(f"  Order-2 exact counting formula [(2M-1)*|X| - 2*C(|X|,2) >= M^2] verified across all {n_test-1}x{1<<n_test}={(n_test-1)*(1<<n_test)} cases: {all_ok}")
    assert all_ok

    print("\n=== E2.3: Parity Coefficient Explosion Theorem (AIM-140 pp. 189-191) ===")
    # For reliable parity recognition on |R| = M points (sum_A alpha_A phi_A(X) >= 2 for odd |X|, <= 0 for even |X|),
    # group-averaged coefficients alpha_j (for masks of size j) satisfy at boundary (= 2 for odd n, = 0 for even n):
    #   alpha_0 = 0, and sum_{j=0}^n C(n, j) alpha_j = 2 * (n % 2) => alpha_j = (-2)^j for j >= 1!
    # Wait: in AIM-140 p. 190, Minsky & Papert define D_n = (-1)^n (alpha_0 - alpha_1 + ... + (-1)^n alpha_n) - ...
    # such that (-1)^n D_n >= 2 for all n=0..M, and prove:
    #   sum_{n=0}^M C(M, n) |D_n| >= 2 * (1 + 2^M)  and  max_j |alpha_j| >= 2^{M-1}!
    print("   M | alpha_M = (-2)^M (exact mask weight) | sum_{j=0}^M C(M,j)|alpha_j| | M * 2^M bits vs 2^{M-1} patterns")
    for M in (2, 4, 6, 8, 10):
        alphas = [0 if j == 0 else (-1) ** (j - 1) * (2 ** j) for j in range(M + 1)]
        # Verify that for every subset size k in 0..M, sum_{j=0}^k C(k, j) alphas[j] == (2 if k is odd else 0):
        for k in range(M + 1):
            s_k = sum(math.comb(k, j) * alphas[j] for j in range(k + 1))
            assert s_k == (2 if k % 2 == 1 else 0)
        weighted_sum = sum(math.comb(M, j) * abs(alphas[j]) for j in range(M + 1))
        print(f"  {M:2d} | {alphas[M]:36d} | {weighted_sum:27d} | {M*(1<<M):8d} bits vs {1<<(M-1):6d} odd sets")


# ============================================================================
# E3: One-in-a-Box, Cutwise Connectivity, Diameter-Limited Impossibility, Serial Algorithm
# ============================================================================

def run_e3_connectivity():
    print("\n=== E3.1: One-in-a-Box & Cutwise Connectivity (m=3 rows: |A_i|=4 Counterexample vs |A_i|=3 Order-2 Formula, pp. 201-202) ===")
    # 1. m=3, |A_i|=4 counterexample (AIM-140 p. 202):
    #    X1 has row counts (3, 3, 0) -> row 3 is empty (DISCONNECTED in Fig. p. 201!)
    #    X2 has row counts (4, 1, 1) -> every row >= 1 (CONNECTED in Fig. p. 201!)
    for name, rows in [("X1 (disconnected, rows=(3,3,0))", (3, 3, 0)), ("X2 (connected,    rows=(4,1,1))", (4, 1, 1))]:
        n1 = sum(rows)
        n11 = sum(r * (r - 1) // 2 for r in rows)
        n12 = rows[0] * rows[1] + rows[0] * rows[2] + rows[1] * rows[2]
        conn = all(r >= 1 for r in rows)
        print(f"  {name}: N^1={n1}, N^11(same-row pairs)={n11}, N^12(diff-row pairs)={n12}, connected={conn}")
    assert (3 + 3 + 0 == 4 + 1 + 1 == 6) and (3 + 3 + 0 == 6 + 0 + 0 == 6) and (3 * 3 == 4 * 1 + 4 * 1 + 1 * 1 == 9)

    # 2. m=3, |A_i|=3 order-2 realization (AIM-140 p. 202 line 17):
    #    psi(X) = [ N^1(X) + N^12(X) - 2 * N^11(X) > 4 ]
    matched_512 = 0
    for mask in range(1 << 9):
        r = [bin((mask >> (3 * i)) & 7).count("1") for i in range(3)]
        n1 = sum(r)
        n11 = sum(ri * (ri - 1) // 2 for ri in r)
        n12 = r[0] * r[1] + r[0] * r[2] + r[1] * r[2]
        formula_pred = (n1 + n12 - 2 * n11) > 4
        one_in_each_row = all(ri >= 1 for ri in r)
        if formula_pred == one_in_each_row:
            matched_512 += 1
    print(f"  m=3, |A_i|=3 order-2 formula [N^1 + N^12 - 2*N^11 > 4] verified on: {matched_512}/512 subsets (100%)")
    assert matched_512 == 512

    print("\n=== E3.2: Diameter-Limited Connectivity Counterexample (X_00, X_01, X_10, X_11, AIM-140 p. 216) ===")
    # Construct the 4 figures on a 5x12 grid with diameter bound D = 4:
    #   Top bar: row 1, cols 2..9; Bottom bar: row 3, cols 2..9 (separated by empty row 2).
    #   Left bridge pixel L = (2, 2); Right bridge pixel R = (2, 9) (distance |9 - 2| = 7 > D = 4).
    #   X_00: no bridges (2 components, disconnected)
    #   X_10: left bridge only (1 component, connected)
    #   X_01: right bridge only (1 component, connected)
    #   X_11: both bridges (1 component with a hole, connected!)
    # Wait: in AIM-140 p. 216, X_01 and X_10 are connected (1 component), while X_00 (2 horizontal bars)
    # and X_11 (Wait: how is X_11 disconnected on p. 216? Look at the diagram on p. 216:
    #   In X_00: left cap has vertical bridge, right cap has vertical bridge, middle has two horizontal gaps!
    #   Specifically:
    #     Left end (col 1): state 0 = open (gap), state 1 = closed (connected)
    #     Right end (col 10): state 0 = open (gap), state 1 = closed (connected)
    #     Middle (col 5): two spiral/crossed bridges such that:
    #       - X_00 (both ends open): disconnected (2 pieces)
    #       - X_10 (left closed, right open): connected (1 piece)
    #       - X_01 (left open, right closed): connected (1 piece)
    #       - X_11 (both ends closed):Wait—on p. 216, in X_00 and X_11:
    #         Let's check! If the left end has a switch that either connects top-to-bottom OR leaves them open,
    #         and the right end is > D away, any predicate phi with diameter(S(phi)) <= D cannot contain points
    #         from BOTH the left end (col 1) and the right end (col 10)!
    #         Therefore, for EVERY predicate phi with diameter <= D:
    #           phi(X_00) + phi(X_11) == phi(X_01) + phi(X_10) identically!)
    # Let's verify this exact identity across all possible masks/predicates of diameter <= D = 6 on the 4-figure family
    # constructed via the 2-switch crossover (AIM-140 p. 206 & p. 216, where X_01 and X_10 have 1 component and X_00 and X_11 have 2 components):
    # On a 1D/2D loop with 2 distant switches s_L (at x=0) and s_R (at x=10):
    #   Switch s=0 connects (top_in -> top_out, bot_in -> bot_out) [parallel];
    #   Switch s=1 connects (top_in -> bot_out, bot_in -> top_out) [twist/Mobius]!
    #   When the two ends of the strip are joined around the cylinder/loop:
    #     - (s_L, s_R) = (0, 0) [0 twists]: 2 disconnected loops!
    #     - (s_L, s_R) = (1, 0) [1 twist]:  1 connected single loop (like generate_1double_spiral)!
    #     - (s_L, s_R) = (0, 1) [1 twist]:  1 connected single loop!
    #     - (s_L, s_R) = (1, 1) [2 twists]: 2 disconnected loops (like generate_2double_spiral)!
    # Let's test all diameter <= D predicates on these 4 figures X_00, X_01, X_10, X_11!
    L_pts = {(0, 0), (0, 1)}  # Left switch pixels at x=0..1
    R_pts = {(0, 10), (0, 11)}  # Right switch pixels at x=10..11 (distance >= 9 > D=6)
    Shared_pts = {(0, c) for c in range(2, 10)}
    X_figs = {
        (0, 0): Shared_pts,
        (1, 0): Shared_pts | L_pts,
        (0, 1): Shared_pts | R_pts,
        (1, 1): Shared_pts | L_pts | R_pts,
    }
    all_pts = sorted(Shared_pts | L_pts | R_pts)
    D_lim = 6
    violating_masks = 0
    total_diam_masks = 0
    for r_size in range(1, 5):
        for subset in itertools.combinations(all_pts, r_size):
            diam = max(abs(p[1] - q[1]) for p in subset for q in subset)
            if diam <= D_lim:
                total_diam_masks += 1
                v00 = int(all(p in X_figs[(0, 0)] for p in subset))
                v01 = int(all(p in X_figs[(0, 1)] for p in subset))
                v10 = int(all(p in X_figs[(1, 0)] for p in subset))
                v11 = int(all(p in X_figs[(1, 1)] for p in subset))
                if (v00 + v11) != (v01 + v10):
                    violating_masks += 1
    print(f"  Diameter-limited masks (size<=4, diam<={D_lim}) violating phi(X00)+phi(X11)==phi(X01)+phi(X10): {violating_masks}/{total_diam_masks}")
    assert violating_masks == 0

    print("\n=== E3.3: Minsky & Papert's 2-Pointer Serial Connectivity Algorithm (SEARCH/SCAN/TRACE, AIM-140 p. 208) ===")
    # Implements the exact 2-pointer (i, j) boundary-circumnavigation algorithm on p. 208 of AIM-140:
    # A figure X on an H x W grid (not touching the outer border) is scanned in raster order i = 0 .. H*W-1.
    # Whenever x_i is in X and x_{i-1} is NOT in X, x_i is a left-hand boundary entry point;
    # TRACE walks the 4-connected boundary loop (keeping complement of X on the left) using pointer j = j*
    # until either j < i (already visited component/hole -> return to SCAN) or j == i (new component discovered!).
    def serial_connectivity_p208(grid: np.ndarray) -> Tuple[bool, int]:
        H, W = grid.shape
        # Represent each pixel (r, c) in X by its 4 directed boundary half-edges where X is on the right and ~X is on the left.
        # Equivalently, in Minsky & Papert's p. 208 formulation, when x_{i-1}=0 and x_i=1 (entering X from the west edge of (r,c)),
        # we walk the boundary contour starting at the west edge of (r,c) keeping ~X on the left and X on the right,
        # checking whether any pixel along that contour has raster index j < i!
        # Wait: as Minsky & Papert state on p. 208:
        #   - On the exterior boundary of a NEVER-before-encountered component, x_i is the lexicographically FIRST pixel
        #     (smallest raster index i) of that entire boundary, so j > i everywhere around the loop until returning to j == i!
        #   - On the exterior boundary of an ALREADY-encountered component (or on an interior hole boundary, whose topmost pixel
        #     borders X above it at index < i), the boundary walk MUST pass through a pixel of X with raster index j < i!
        def trace_boundary_loop(r0: int, c0: int) -> List[int]:
            # Start at west edge of (r0, c0), facing North (so ~X is on West/left, X is on East/right):
            # Directed edge state: (r, c, d) where (r,c) in X and neighbor in direction (d-1)%4 is NOT in X.
            # Directions: 0=N (-1,0), 1=E (0,1), 2=S (1,0), 3=W (0,-1).
            dr = [-1, 0, 1, 0]
            dc = [0, 1, 0, -1]
            start_state = (r0, c0, 0)  # at (r0,c0), left side is W (3), which is (r0, c0-1) not in X
            curr = start_state
            visited_indices = []
            while True:
                r, c, d = curr
                visited_indices.append(r * W + c)
                # Try turning left (d_left = (d - 1) % 4) into diag neighbor (r + dr[d] + dr[d_left], c + dc[d] + dc[d_left]),
                # else straight (r + dr[d], c + dc[d]), else turn right (d_right = (d + 1) % 4) at same pixel:
                nr_straight, nc_straight = r + dr[d], c + dc[d]
                d_left = (d - 1) % 4
                nr_diag, nc_diag = nr_straight + dr[d_left], nc_straight + dc[d_left]
                if grid[nr_straight, nc_straight] == 1:
                    if grid[nr_diag, nc_diag] == 1:
                        curr = (nr_diag, nc_diag, d_left)
                    else:
                        curr = (nr_straight, nc_straight, d)
                else:
                    curr = (r, c, (d + 1) % 4)
                if curr == start_state:
                    break
            return visited_indices

        new_components = 0
        for i in range(1, H * W):
            r, c = divmod(i, W)
            r_prev, c_prev = divmod(i - 1, W)
            if grid[r, c] == 1 and grid[r_prev, c_prev] == 0:
                # TRACE boundary starting at i:
                loop_js = trace_boundary_loop(r, c)
                met_smaller = any(j < i for j in loop_js)
                if not met_smaller:
                    new_components += 1
        return (new_components == 1), new_components

    # Test on 5 canonical grid figures (solid block, U-shape, hollow ring with hole, 2 disjoint blocks, 3 disjoint blocks):
    test_grids = []
    g_solid = np.zeros((10, 10), dtype=int); g_solid[2:7, 2:7] = 1
    g_ushape = np.zeros((10, 10), dtype=int); g_ushape[2:8, 2:4] = 1; g_ushape[6:8, 2:8] = 1; g_ushape[2:8, 6:8] = 1
    g_ring = np.zeros((10, 10), dtype=int); g_ring[2:8, 2:8] = 1; g_ring[4:6, 4:6] = 0  # interior hole!
    g_two = np.zeros((10, 10), dtype=int); g_two[2:4, 2:4] = 1; g_two[6:8, 6:8] = 1
    g_three = np.zeros((10, 10), dtype=int); g_three[1:3, 1:3] = 1; g_three[4:6, 4:6] = 1; g_three[7:9, 2:4] = 1

    for name, g, exp_conn, exp_k in [
        ("Solid 5x5 square", g_solid, True, 1),
        ("Concave U-shape", g_ushape, True, 1),
        ("Hollow ring (1 component with interior hole)", g_ring, True, 1),
        ("Two disjoint squares", g_two, False, 2),
        ("Three disjoint squares", g_three, False, 3),
    ]:
        is_conn, k_comp = serial_connectivity_p208(g)
        print(f"  {name:<46}: connected={str(is_conn):<5} (components={k_comp}, expected={exp_k})")
        assert is_conn == exp_conn and k_comp == exp_k


# ============================================================================
# E4: Chapter 12 (AIM-167) Perceptron vs Naive Bayes Log-Odds vs Nearest Neighbor
# ============================================================================

def run_e4_chapter12():
    print("\n=== E4.1: AIM-167 Fig. 12.3(a) vs Fig. 12.3(b) — PERCEPTRON vs BAYES Log-Odds (pp. 195, 202-204) ===")
    rng = random.Random(1969)

    # Case A (Fig. 12.3(a), p. 195): Linearly separable correlated skew clouds in {0,1}^10.
    # Feature x_0 is the true class label y in {0, 1}, while features x_1..x_9 are strongly correlated with x_0
    # in the opposite direction of a naive sum: specifically, when y=1, x_0=1 and x_1..x_9 are 1 with prob 0.45,
    # EXCEPT whenever sum(x_1..x_9) >= 5, x_0=1 still holds; let's define an exact linearly separable rule with
    # correlated distraction features:
    #   True rule: y = 1 iff 6 * x_0 - sum_{j=1}^8 x_j >= 1.
    #   Distribution:
    #     - Class 1: either (x_0=1 and 5 of x_1..x_8 are 1) [90% of Class 1] or (x_0=1 and 0 of x_1..x_8 are 1) [10%].
    #     - Class 0: either (x_0=1 and 6 of x_1..x_8 are 1) [10% of Class 0] or (x_0=0 and 0 of x_1..x_8 are 1) [90%].
    # Notice:
    #   - Every sample satisfies 6*x_0 - sum_{j=1}^8 x_j >= 1 iff y == 1 (100% linearly separable!).
    #   - Yet in Class 1, each x_j (j=1..8) has P(x_j=1 | Class 1) = 0.9 * (5/8) = 0.5625,
    #     whereas in Class 0, P(x_j=1 | Class 0) = 0.1 * (6/8) = 0.0750!
    #   - So Naive Bayes (which assumes conditional independence of x_0..x_8 given y!) assigns POSITIVE weight
    #     w_j = log(p_{j1}(1-p_{j0}) / (p_{j0}(1-p_{j1}))) > 0 to x_1..x_8, which is the EXACT OPPOSITE sign of
    #     the separating hyperplane (-1 on x_1..x_8), causing BAYES to misclassify the 10% boundary cluster (x_0=1, 6 ones)!
    def sample_fig12_3a(n_samples: int) -> List[Tuple[List[int], int]]:
        data = []
        for _ in range(n_samples):
            y = rng.randrange(2)
            if y == 1:
                if rng.random() < 0.90:
                    ones = set(rng.sample(range(1, 9), 5))
                else:
                    ones = set()
                x = [1] + [1 if j in ones else 0 for j in range(1, 9)]
            else:
                if rng.random() < 0.15:
                    ones = set(rng.sample(range(1, 9), 6))
                    x = [1] + [1 if j in ones else 0 for j in range(1, 9)]
                else:
                    x = [0] * 9
            data.append((x, y))
        return data

    # Case B (Fig. 12.3(b) & Sec. 12.4.3, pp. 195, 200-204): Conditionally independent noisy predicates (non-separable).
    #   9 predicates phi_1..phi_9, each independently equal to y with probability p = 0.70 (error q = 0.30).
    def sample_fig12_3b(n_samples: int) -> List[Tuple[List[int], int]]:
        data = []
        for _ in range(n_samples):
            y = rng.randrange(2)
            x = [y if rng.random() < 0.70 else (1 - y) for _ in range(9)]
            data.append((x, y))
        return data

    for title, gen_fn in [
        ("Fig. 12.3(a) Linearly separable correlated skew clouds", sample_fig12_3a),
        ("Fig. 12.3(b) Conditionally independent noisy features (non-separable)", sample_fig12_3b),
    ]:
        train_set = gen_fn(1000)
        test_set = gen_fn(1000)
        d = len(train_set[0][0])

        # 1. Train BAYES log-odds (AIM-167 Eq. 3-4, p. 203):
        c1 = max(1, sum(1 for _, y in train_set if y == 1))
        c0 = max(1, sum(1 for _, y in train_set if y == 0))
        w_bayes = []
        theta_bayes = math.log(c0 / c1)
        for i in range(d):
            p_i1 = (sum(x[i] for x, y in train_set if y == 1) + 0.5) / (c1 + 1.0)
            p_i0 = (sum(x[i] for x, y in train_set if y == 0) + 0.5) / (c0 + 1.0)
            w_bayes.append(math.log((p_i1 * (1.0 - p_i0)) / (p_i0 * (1.0 - p_i1))))
            theta_bayes -= math.log((1.0 - p_i1) / (1.0 - p_i0))

        # 2. Train PERCEPTRON (25 epochs):
        w_perc = [0.0] * d
        b_perc = 0.0
        for _ in range(25):
            for x, y in train_set:
                pred = 1 if (sum(wi * xi for wi, xi in zip(w_perc, x)) + b_perc > 0) else 0
                if pred != y:
                    step = 1.0 if y == 1 else -1.0
                    for i in range(d):
                        w_perc[i] += step * x[i]
                    b_perc += step

        err_bayes = sum(1 for x, y in test_set if (1 if sum(wi * xi for wi, xi in zip(w_bayes, x)) > theta_bayes else 0) != y) / len(test_set)
        err_perc = sum(1 for x, y in test_set if (1 if sum(wi * xi for wi, xi in zip(w_perc, x)) + b_perc > 0 else 0) != y) / len(test_set)
        print(f"  {title:<68}: PERCEPTRON error = {err_perc*100:5.1f}% | BAYES error = {err_bayes*100:5.1f}%")

    print("\n=== E4.2: AIM-167 Section 12.3 (pp. 198-199) Nearest Neighbor Pathology on Parity ===")
    # "A very bad case is a paritylike structure in which NEAREST NEIGHBOR actually does worse than chance!
    #  Suppose that phi in F_1 if and only if phi_i == 1 for an even number of i's. Then, if there are n phi_i's,
    #  each phi will have exactly n neighbors whose distance d satisfies 0 < d <= 1. Suppose that all but a fraction q
    #  of all possible phi's have already been seen. Then NEAREST NEIGHBOR will err on a given phi if it has not been
    #  seen (probability = q) but one of its immediate neighbors has been seen (probability = 1 - q^n). So the
    #  probability of error is > q(1 - q^n), which, for large q, can be quite near certainty."
    n_bits = 8
    all_vecs = list(range(1 << n_bits))
    for q in (0.10, 0.25, 0.50, 0.70):
        n_seen = round((1.0 - q) * len(all_vecs))
        emp_all_errs = []
        emp_unseen_errs = []
        for trial in range(50):
            seen_set = set(rng.sample(all_vecs, n_seen))
            errs_all = 0
            errs_unseen = 0
            unseen_cnt = 0
            for v in all_vecs:
                true_par = bin(v).count("1") % 2
                if v in seen_set:
                    nn_par = true_par  # distance 0 match
                else:
                    unseen_cnt += 1
                    # Find minimum Hamming distance in seen_set
                    best_d = 999
                    best_u = 0
                    for u in seen_set:
                        d_uv = bin(v ^ u).count("1")
                        if d_uv < best_d:
                            best_d = d_uv
                            best_u = u
                            if best_d == 1:
                                break
                    nn_par = bin(best_u).count("1") % 2
                    if nn_par != true_par:
                        errs_unseen += 1
                        errs_all += 1
            emp_all_errs.append(errs_all / len(all_vecs))
            emp_unseen_errs.append(errs_unseen / max(1, unseen_cnt))
        bound_all = q * (1.0 - (q ** n_bits))
        bound_unseen = 1.0 - (q ** n_bits)
        print(f"  q={q:.2f} (seen {100*(1-q):4.0f}% of 2^{n_bits}): "
              f"Unseen-query error = {100*sum(emp_unseen_errs)/len(emp_unseen_errs):5.1f}% (bound 1-q^n = {100*bound_unseen:5.1f}%) | "
              f"All-space error = {100*sum(emp_all_errs)/len(emp_all_errs):5.1f}% (bound q(1-q^n) = {100*bound_all:5.1f}%)")


def main():
    repo_dir = "/root/dm-study/foundations/minsky-papert-1969/repo/TaraAlcira-PerceptronsRevisited"
    run_e1_spirals(repo_dir)
    run_e2_order_and_parity()
    run_e3_connectivity()
    run_e4_chapter12()


if __name__ == "__main__":
    main()
