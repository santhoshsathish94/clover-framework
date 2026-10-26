"""
Experiments E2, E3, and E4 for Rumelhart, Hinton & Williams (1986):
  - Nature 323: 533-536 (9 October 1986), "Learning representations by back-propagating errors"
  - ICS Report 8506 / PDP Vol. 1 Ch. 8 (1985/1986), "Learning Internal Representations by Error Propagation"
"""
import itertools
import math
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class MLP1986:
    """
    Feedforward network implementing Rumelhart, Hinton & Williams (1986) Equations (7)-(16):
      - Logistic activation: o_pj = 1 / (1 + exp(-(sum_i w_ji o_pi + theta_j)))
      - Error: E = sum_p (1/2) sum_j (t_pj - o_pj)^2
      - Output delta (Eq. 13): delta_pj = (t_pj - o_pj) * o_pj * (1 - o_pj)
      - Hidden delta (Eq. 14): delta_pj = o_pj * (1 - o_pj) * sum_k delta_pk w_kj
      - Momentum update (Eq. 16): dw_ji(n+1) = eta * (sum_p delta_pj o_pi) + alpha * dw_ji(n)
    """
    def __init__(self, layer_sizes, seed=0, init_range=0.5, equal_init=None):
        self.sizes = list(layer_sizes)
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        self.vW = []
        self.vb = []
        for s_in, s_out in zip(self.sizes[:-1], self.sizes[1:]):
            if equal_init is not None:
                w = np.full((s_out, s_in), float(equal_init))
                bi = np.full(s_out, float(equal_init))
            else:
                w = rng.uniform(-init_range, init_range, size=(s_out, s_in))
                bi = rng.uniform(-init_range, init_range, size=s_out)
            self.W.append(w)
            self.b.append(bi)
            self.vW.append(np.zeros_like(w))
            self.vb.append(np.zeros_like(bi))

    def forward(self, X: np.ndarray):
        acts = [X]
        for w, bi in zip(self.W, self.b):
            net = acts[-1] @ w.T + bi
            acts.append(sigmoid(net))
        return acts

    def loss(self, X: np.ndarray, T: np.ndarray) -> float:
        acts = self.forward(X)
        return float(0.5 * np.sum((T - acts[-1]) ** 2))

    def backward_deltas(self, X: np.ndarray, T: np.ndarray, omit_hidden_deriv: bool = False):
        """Returns (dE/dW_list, dE/db_list) where dE/dw_ji = -sum_p delta_pj o_pi."""
        acts = self.forward(X)
        n_layers = len(self.W)
        deltas = [None] * n_layers
        # Output layer (Eq. 13, note delta_pj = -dE_p/dnet_pj = (t_pj - o_pj) o_pj (1 - o_pj)):
        out = acts[-1]
        deltas[-1] = (T - out) * out * (1.0 - out)
        # Hidden layers (Eq. 14):
        for l in range(n_layers - 2, -1, -1):
            h = acts[l + 1]
            back = deltas[l + 1] @ self.W[l + 1]
            if omit_hidden_deriv:
                deltas[l] = back
            else:
                deltas[l] = back * h * (1.0 - h)

        dW = [-(deltas[l].T @ acts[l]) for l in range(n_layers)]
        db = [-np.sum(deltas[l], axis=0) for l in range(n_layers)]
        return dW, db

    def step(self, X: np.ndarray, T: np.ndarray, eta: float = 0.5, alpha: float = 0.9, omit_hidden_deriv: bool = False) -> float:
        dW, db = self.backward_deltas(X, T, omit_hidden_deriv=omit_hidden_deriv)
        for l in range(len(self.W)):
            self.vW[l] = -eta * dW[l] + alpha * self.vW[l]
            self.vb[l] = -eta * db[l] + alpha * self.vb[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]
        return self.loss(X, T)


# ============================================================================
# E2: Gradient Check, Symmetry-Breaking Saddle Point, Momentum, & Width Scaling
# ============================================================================

def run_e2_mechanics():
    print("=== E2.1: Generalized Delta Rule (Eq. 9-14) vs Central Finite Differences ===")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    T = np.array([[0], [1], [1], [0]], dtype=float)
    net = MLP1986([2, 3, 2, 1], seed=1986)
    dW_ana, db_ana = net.backward_deltas(X, T, omit_hidden_deriv=False)
    dW_bad, _ = net.backward_deltas(X, T, omit_hidden_deriv=True)

    eps = 1e-6
    max_err_ana = 0.0
    max_err_bad = 0.0
    for l in range(len(net.W)):
        for idx in np.ndindex(net.W[l].shape):
            orig = net.W[l][idx]
            net.W[l][idx] = orig + eps
            e_up = net.loss(X, T)
            net.W[l][idx] = orig - eps
            e_dn = net.loss(X, T)
            net.W[l][idx] = orig
            fd = (e_up - e_dn) / (2.0 * eps)
            max_err_ana = max(max_err_ana, abs(dW_ana[l][idx] - fd))
            max_err_bad = max(max_err_bad, abs(dW_bad[l][idx] - fd))

    print(f"  Max |analytical dE/dW - finite_diff| across 4-layer net [2,3,2,1]: {max_err_ana:.2e}")
    print(f"  Negative control (omitting hidden f'(net)=o(1-o) in Eq. 14):      {max_err_bad:.2e}")
    assert max_err_ana < 1e-9 and max_err_bad > 0.01

    print("\n=== E2.2: Symmetry-Breaking Saddle Point at Equal Initial Weights (p. 10) ===")
    for init_val in (0.0, 0.25):
        net_eq = MLP1986([2, 2, 1], equal_init=init_val)
        for _ in range(500):
            net_eq.step(X, T, eta=0.5, alpha=0.9)
        w_diff = float(np.max(np.abs(net_eq.W[0][0] - net_eq.W[0][1])))
        final_e = net_eq.loss(X, T)
        acc_eq = int(np.sum((net_eq.forward(X)[-1] >= 0.5).astype(float) == T))
        print(f"  Equal init w={init_val:4.2f} after 500 sweeps: max |w_h0 - w_h1| = {w_diff:.2e}, final E = {final_e:.4f}, accuracy = {acc_eq}/4 (cannot solve XOR!)")
        assert w_diff == 0.0 and final_e >= 0.25 and acc_eq < 4

    print("\n=== E2.3: Momentum (alpha=0.9 vs alpha=0.0, Eq. 16) & Chauvin's Hidden-Unit Scaling (p. 12) ===")
    for alpha_val in (0.0, 0.9):
        sweeps_list = []
        for s in range(15):
            net_m = MLP1986([2, 4, 1], seed=100 + s, init_range=0.5)
            for ep in range(1, 4001):
                e = net_m.step(X, T, eta=0.5, alpha=alpha_val)
                if e <= 0.04:  # 0.01 per pattern * 4 patterns
                    sweeps_list.append(ep)
                    break
        print(f"  XOR [2,4,1] eta=0.5, momentum alpha={alpha_val:.1f}: solved={len(sweeps_list)}/15, mean sweeps={np.mean(sweeps_list):.1f}, median={np.median(sweeps_list):.1f}")

    # Chauvin's scaling law (p. 12): vary H in {2, 4, 8, 16, 32} on XOR with eta=0.25, alpha=0.9:
    print("  Chauvin hidden-unit scaling on XOR (eta=0.25, alpha=0.9, target E <= 0.04):")
    means_by_h = []
    for H in (2, 4, 8, 16, 32):
        sw = []
        for s in range(15):
            net_h = MLP1986([2, H, 1], seed=200 + s, init_range=0.5)
            for ep in range(1, 3001):
                if net_h.step(X, T, eta=0.25, alpha=0.9) <= 0.04:
                    sw.append(ep)
                    break
        means_by_h.append(float(np.mean(sw)))
        chauvin_pred = 280.0 - 33.0 * math.log2(H)
        print(f"    H={H:2d} hidden units: solved={len(sw):2d}/15 | mean sweeps={np.mean(sw):6.1f} | Chauvin p.12 formula (280 - 33 log2 H) = {chauvin_pred:5.1f}")
    assert means_by_h[-1] < means_by_h[0]


# ============================================================================
# E3: 4-Bit Parity Counter (Table 3) & 8-3-8 Bottleneck Encoder (Table 5)
# ============================================================================

def run_e3_representations():
    print("\n=== E3.1: 4-Bit Parity [4 -> 4 -> 1] Thermometer Counter Representation (Fig. 6 & Table 3, pp. 12-13) ===")
    X_par = np.array(list(itertools.product([0.0, 1.0], repeat=4)))
    T_par = np.array([[float(int(np.sum(row)) % 2)] for row in X_par])

    # Search a few seeds for a clean convergence on 4-bit parity (4-4-1 is famously tricky without enough epochs):
    solved_net = None
    solved_ep = 0
    for seed in range(1, 20):
        net = MLP1986([4, 4, 1], seed=seed, init_range=0.8)
        for ep in range(1, 8001):
            e = net.step(X_par, T_par, eta=0.5, alpha=0.9)
            if e < 0.02:
                solved_net = net
                solved_ep = ep
                break
        if solved_net is not None:
            break

    assert solved_net is not None, "4-bit parity did not converge"
    acts = solved_net.forward(X_par)
    H_acts = acts[1]
    O_acts = acts[2]
    acc = int(np.sum((O_acts >= 0.5).astype(float) == T_par))
    print(f"  4-bit Parity [4 -> 4 -> 1] solved in {solved_ep} sweeps: accuracy = {acc}/16 (E = {solved_net.loss(X_par, T_par):.5f})")
    print("  Input-to-hidden weights W[0] (4 hidden units x 4 inputs):")
    for j in range(4):
        w_row = solved_net.W[0][j]
        print(f"    Hidden unit {j}: weights = {np.round(w_row, 2)} (bias = {solved_net.b[0][j]:+.2f})")
    print("  Hidden unit binary patterns by number of ON input bits |x| in 0..4 (cf. Table 3, p. 13):")
    for k_on in range(5):
        idxs = [i for i, row in enumerate(X_par) if int(np.sum(row)) == k_on]
        pats = {tuple(np.round(H_acts[i]).astype(int).tolist()) for i in idxs}
        out_mean = float(np.mean([O_acts[i, 0] for i in idxs]))
        print(f"    |x| = {k_on} ({len(idxs)} inputs): distinct rounded hidden patterns = {pats} -> mean output = {out_mean:.3f}")

    print("\n=== E3.2: 8-3-8 Identity Bottleneck Encoder (Fig. 7 & Table 5, pp. 14-15, and p. 9 {0.1, 0.9} targets) ===")
    X_enc = np.eye(8, dtype=float)
    # Control: {0.0, 1.0} targets saturate logistic f'(net)=o(1-o) on 7/8 zero targets
    net_enc01 = MLP1986([8, 3, 8], seed=1986, init_range=0.5)
    for ep01 in range(1, 5001):
        e01 = net_enc01.step(X_enc, X_enc, eta=0.4, alpha=0.9)
    enc01_ok = int(np.sum(np.argmax(net_enc01.forward(X_enc)[2], axis=1) == np.arange(8)))
    print(f"  8-3-8 Encoder with raw {{0.0, 1.0}} targets (5000 sweeps): argmax reconstruction = {enc01_ok}/8 (E = {e01:.5f}, logistic tail saturation!)")

    T_enc = np.where(X_enc > 0.5, 0.9, 0.1)  # p. 9: "we typically use the values of 0.1 and 0.9 as the targets"
    net_enc = MLP1986([8, 3, 8], seed=1, init_range=0.5)
    for ep in range(1, 5001):
        e = net_enc.step(X_enc, T_enc, eta=0.25, alpha=0.9)
        if e < 0.02:
            break
    acts_enc = net_enc.forward(X_enc)
    H_enc = acts_enc[1]
    O_enc = acts_enc[2]
    enc_ok = int(np.sum(np.argmax(O_enc, axis=1) == np.arange(8)))
    print(f"  8-3-8 Encoder (with p.9 targets {{0.1, 0.9}}, eta=0.25) converged in {ep} sweeps: argmax reconstruction = {enc_ok}/8 (E = {e:.5f})")
    for i in range(8):
        h_vals = [round(float(v), 2) for v in H_enc[i]]
        print(f"    Input {i} ({X_enc[i].astype(int).tolist()}) -> Hidden 3-unit code: {h_vals}")
    assert enc_ok == 8

    print("\n=== E3.3: 6-Bit Mirror Symmetry [6 -> 2 -> 1] Geometric Ratios 1:-2:4 (Fig. 9 & Table 6, pp. 17-18) ===")
    X_sym = np.array([[int(b) for b in format(i, "06b")] for i in range(64)], dtype=float)
    T_sym_bin = np.array([
        [1.0 if (row[0] == row[5] and row[1] == row[4] and row[2] == row[3]) else 0.0]
        for row in X_sym
    ])
    T_sym = np.where(T_sym_bin > 0.5, 0.9, 0.1)
    for seed in range(1, 30):
        net_sym = MLP1986([6, 2, 1], seed=seed, init_range=0.3)
        for ep in range(1, 2501):
            # p. 17: "arrived at after 1,206 presentations of each six-bit pattern with eta = 0.1"
            e = net_sym.step(X_sym, T_sym, eta=0.1, alpha=0.9)
            if e < 0.05:
                break
        preds = (net_sym.forward(X_sym)[-1] >= 0.5).astype(float)
        acc = int(np.sum(preds == T_sym_bin))
        if acc == 64:
            print(f"  6-bit Mirror Symmetry [6 -> 2 -> 1] (seed={seed}, eta=0.1) solved in {ep} sweeps: accuracy = {acc}/64 (E = {e:.5f})")
            for h_idx in range(2):
                w_row = np.round(net_sym.W[0][h_idx], 2)
                w_sum_pairs = np.round([w_row[0] + w_row[5], w_row[1] + w_row[4], w_row[2] + w_row[3]], 2)
                ratios = np.round(np.abs(w_row[:3]) / max(np.min(np.abs(w_row[:3])), 1e-6), 2)
                print(f"    Hidden unit {h_idx}: W = {w_row}, bias = {net_sym.b[0][h_idx]:+.2f} | mirror sums (w0+w5,w1+w4,w2+w3) = {w_sum_pairs} | |w0:w1:w2| ratio = {ratios}")
            print(f"    Output weights W[1] = {np.round(net_sym.W[1][0], 2)}, output bias = {net_sym.b[1][0]:+.2f}")
            break
    assert acc == 64


# ============================================================================
# E4: Order-3 T-C Problem with Weight Sharing (ICS-8506 pp. 22-25)
# ============================================================================

def run_e4_tc_problem():
    print("\n=== E4: The Order-3 T-C Problem with 3x3 Spatial Weight Sharing (ICS-8506 Figs. 13-15, pp. 22-25) ===")
    # Construct the 5-square block 'T' and 5-square block 'C' on a 3x3 patch (Fig. 13, p. 23):
    T_base = np.array([
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], dtype=float)
    C_base = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=float)

    def distance_spectrum(patch: np.ndarray):
        pts = list(zip(*np.where(patch > 0.5)))
        spec = {}
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d2 = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
                spec[d2] = spec.get(d2, 0) + 1
        return dict(sorted(spec.items()))

    spec_T = [distance_spectrum(np.rot90(T_base, k)) for k in range(4)]
    spec_C = [distance_spectrum(np.rot90(C_base, k)) for k in range(4)]
    print(f"  5-square T pairwise squared-distance spectrum (all 4 rotations): {dict((int(k), int(v)) for k, v in spec_T[0].items())}")
    print(f"  5-square C pairwise squared-distance spectrum (all 4 rotations): {dict((int(k), int(v)) for k, v in spec_C[0].items())}")
    assert spec_T == spec_C and all(s == {1: 4, 2: 2, 4: 2, 5: 2} for s in spec_T)
    print("  Verified Minsky & Papert order-3 property (p. 22): T and C have IDENTICAL pairwise distance spectra {1:4, 2:2, 4:2, 5:2}!")

    # All 25 3x3 receptive fields that can overlap a 3x3 T or C placed on a 2D toroidal/padded grid (5x5=25 RFs):
    # Note: on p. 24, Rumelhart, Hinton & Williams state: "The C turns off 20 hidden units, and the T turns off 21."
    patterns = []
    labels_bin = []
    for cls, base in [(1.0, T_base), (0.0, C_base)]:
        for rot in range(4):
            pat3 = np.rot90(base, rot)
            grid = np.zeros((7, 7), dtype=float)
            grid[2:5, 2:5] = pat3
            patterns.append(grid)
            labels_bin.append(cls)

    # Extract the 5x5 = 25 3x3 receptive fields centered around the pattern (rows 0..4, cols 0..4):
    P_patches = np.zeros((8, 25, 9), dtype=float)
    for p_idx, grid in enumerate(patterns):
        k = 0
        for r in range(5):
            for c in range(5):
                patch = grid[r:r + 3, c:c + 3].reshape(9)
                P_patches[p_idx, k] = patch
                k += 1
    Y_tc_bin = np.array(labels_bin, dtype=float)
    Y_tc = np.where(Y_tc_bin > 0.5, 0.9, 0.1)

    # First verify the analytical compactness count cited by Rumelhart, Hinton & Williams (Fig. 15D, p. 24-25):
    # "The strength of the inhibitory weights are such that if a character overlaps the receptive field of a hidden unit,
    #  that unit turns off... The T turns off 21 hidden units, and the C turns off only 20."
    overlap_counts = np.sum(np.sum(P_patches, axis=2) > 0, axis=1)
    print(f"  Number of 3x3 receptive fields overlapped (turned off in Fig. 15D): T rotations = {overlap_counts[:4].tolist()} (21), C rotations = {overlap_counts[4:].tolist()} (20)!")
    assert all(overlap_counts[:4] == 21) and all(overlap_counts[4:] == 20)

    for seed in range(1, 20):
        rng = np.random.default_rng(seed)
        w_rf = rng.uniform(-0.5, 0.5, size=9)
        b_rf = float(rng.uniform(-0.2, 0.2))
        w_out = float(rng.uniform(0.5, 1.0))
        b_out = float(rng.uniform(-0.2, 0.2))
        vw_rf = np.zeros(9)
        vb_rf, vw_out, vb_out = 0.0, 0.0, 0.0

        # Note: each weight in the 3x3 kernel w_rf is active on 5/25 of the patches, whereas b_rf and w_out
        # sum over all 25 spatial units; dividing shared-summed updates for b_rf, w_out, b_out by 25 keeps per-parameter step sizes matched:
        eta, alpha = 0.25, 0.9
        for ep in range(1, 6001):
            net_h = P_patches @ w_rf + b_rf          # (8, 25)
            act_h = sigmoid(net_h)                   # (8, 25)
            sum_h = np.sum(act_h, axis=1)            # (8,)
            net_o = w_out * sum_h + b_out            # (8,)
            act_o = sigmoid(net_o)                   # (8,)

            err = Y_tc - act_o                       # (8,)
            loss = float(0.5 * np.sum(err ** 2))
            if loss < 0.01:
                break

            delta_o = err * act_o * (1.0 - act_o)    # (8,)
            delta_h = (delta_o[:, None] * w_out) * act_h * (1.0 - act_h)  # (8, 25)

            dw_out = float(np.sum(delta_o * sum_h)) / 25.0
            db_out = float(np.sum(delta_o))
            dw_rf = np.sum(delta_h[:, :, None] * P_patches, axis=(0, 1)) / 5.0
            db_rf = float(np.sum(delta_h)) / 25.0

            vw_rf = eta * dw_rf + alpha * vw_rf
            vb_rf = eta * db_rf + alpha * vb_rf
            vw_out = eta * dw_out + alpha * vw_out
            vb_out = eta * db_out + alpha * vb_out

            w_rf += vw_rf
            b_rf += vb_rf
            w_out += vw_out
            b_out += vb_out

        preds = (act_o >= 0.5).astype(float)
        acc_tc = int(np.sum(preds == Y_tc_bin))
        if acc_tc == 8 and loss < 0.02:
            print(f"  Weight-shared 3x3 T-C network (seed={seed}) converged in {ep} sweeps: accuracy = {acc_tc}/8 (loss = {loss:.5f})")
            print(f"  Learned 3x3 receptive field kernel (b_rf={b_rf:+.2f}, w_out={w_out:+.2f}, b_out={b_out:+.2f}):")
            for row in w_rf.reshape(3, 3):
                print("    " + " ".join(f"{v:+6.2f}" for v in row))
            print(f"  Mean summed hidden activation across the 25 RFs: T = {np.mean(sum_h[:4]):.3f} vs C = {np.mean(sum_h[4:]):.3f}")
            break
    assert acc_tc == 8


if __name__ == "__main__":
    run_e2_mechanics()
    run_e3_representations()
    run_e4_tc_problem()
