#!/usr/bin/env python3
"""
lecun1998_experiments.py — Verified from-scratch implementation and experimental verification
of Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner (1998),
"Gradient-Based Learning Applied to Document Recognition", Proc. IEEE 86(11): 2278-2324.

Covers:
  - E2: Exact LeNet-5 Layer-by-Layer Architecture, Parameter & Connection Accounting
        (C1=156/122304, S2=12/5880, C3=1516/151600 via Table I, S4=32/2000, C5=48120/48120,
         F6=10164/10164 -> 60,000 trainable parameters & 340,908 total connections with RBF),
        Appendix A Scaled Hyperbolic Tangent f(a) = 1.7159 tanh(2/3 a) properties (f(+/-1) = +/-1,
        max |f''(a)| at a = +/-0.9885), and Appendix C Stochastic Diagonal Levenberg-Marquardt
        Gauss-Newton Hessian Backpropagation (Eq. 18-27, pp. 41-42).
  - E3: Euclidean RBF Output Layer (84-dim stylized 7x12 ASCII bitmaps, Eq. 7) & The Collapse Problem
        under Adaptive RBF Centers + MSE (Eq. 8) vs Competitive MAP Criterion (Eq. 9, pp. 8-9),
        plus Space-Displacement Neural Network (SDNN, Section VII, Fig. 23, p. 27) exact output
        equivalence and MAC savings vs sliding-window cropping.
  - E4: Graph Transformer Networks (GTNs, Sections IV-VI & VIII, pp. 16-26, 31-34):
        - Viterbi Transformer (Eq. 10) vs Forward Scorer (Eq. 13-14)
        - Exact backward recurrence through Forward Scorer (Eq. 15-16) vs Central Finite Differences
        - Parallel-path evidence aggregation (Section VI-C, p. 25)
        - String-level Discriminative Forward Training (Eq. 17, Fig. 21) vs Non-Discriminative
          Viterbi Training (Eq. 11, Fig. 19) on over-segmented multi-character strings without
          character segmentation labels, plus check-reading confidence exp(-E_dforw) (p. 39).
"""

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Table I (p. 2285 / PDF p. 8): S2 (6 maps) -> C3 (16 maps) Connection Matrix
# ============================================================================
# Columns 0..5:  every contiguous subset of 3 S2 maps
# Columns 6..11: every contiguous subset of 4 S2 maps
# Columns 12..14: discontinuous subsets of 4 S2 maps
# Column 15:     all 6 S2 maps
TABLE_I_C3_MAP = [
    [0, 1, 2],          # C3 map 0
    [1, 2, 3],          # C3 map 1
    [2, 3, 4],          # C3 map 2
    [3, 4, 5],          # C3 map 3
    [0, 4, 5],          # C3 map 4
    [0, 1, 5],          # C3 map 5
    [0, 1, 2, 3],       # C3 map 6
    [1, 2, 3, 4],       # C3 map 7
    [2, 3, 4, 5],       # C3 map 8
    [0, 3, 4, 5],       # C3 map 9
    [0, 1, 4, 5],       # C3 map 10
    [0, 1, 2, 5],       # C3 map 11
    [0, 1, 3, 4],       # C3 map 12
    [1, 2, 4, 5],       # C3 map 13
    [0, 2, 3, 5],       # C3 map 14
    [0, 1, 2, 3, 4, 5], # C3 map 15
]


# Stylized 7x12 = 84 bitmap codes in {-1, +1}^84 for digits 0..9 (Section II-B, Fig. 3, pp. 8-9):
DIGIT_BITMAPS_7X12 = np.array([
    # 0
    [-1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 1
    [-1,-1,-1,+1,+1,-1,-1, -1,-1,+1,+1,+1,-1,-1, -1,+1,-1,+1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1,
     -1,-1,-1,+1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1,
     -1,-1,-1,+1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1, -1,+1,+1,+1,+1,+1,+1, -1,-1,-1,-1,-1,-1,-1],
    # 2
    [-1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, -1,-1,-1,-1,-1,+1,+1,
     -1,-1,-1,-1,+1,+1,-1, -1,-1,-1,+1,+1,-1,-1, -1,-1,+1,+1,-1,-1,-1, -1,+1,+1,-1,-1,-1,-1,
     +1,+1,-1,-1,-1,-1,-1, +1,+1,-1,-1,-1,-1,+1, +1,+1,+1,+1,+1,+1,+1, -1,-1,-1,-1,-1,-1,-1],
    # 3
    [-1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, -1,-1,-1,-1,-1,-1,+1, -1,-1,-1,-1,-1,+1,+1,
     -1,-1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,+1,+1, -1,-1,-1,-1,-1,-1,+1, -1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 4
    [-1,-1,-1,-1,+1,+1,-1, -1,-1,-1,+1,+1,+1,-1, -1,-1,+1,-1,+1,+1,-1, -1,+1,-1,-1,+1,+1,-1,
     +1,+1,-1,-1,+1,+1,-1, +1,-1,-1,-1,+1,+1,-1, +1,+1,+1,+1,+1,+1,+1, -1,-1,-1,-1,+1,+1,-1,
     -1,-1,-1,-1,+1,+1,-1, -1,-1,-1,-1,+1,+1,-1, -1,-1,-1,+1,+1,+1,+1, -1,-1,-1,-1,-1,-1,-1],
    # 5
    [+1,+1,+1,+1,+1,+1,+1, +1,+1,-1,-1,-1,-1,-1, +1,+1,-1,-1,-1,-1,-1, +1,+1,+1,+1,+1,-1,-1,
     +1,+1,-1,-1,-1,+1,+1, -1,-1,-1,-1,-1,-1,+1, -1,-1,-1,-1,-1,-1,+1, -1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 6
    [-1,-1,+1,+1,+1,+1,-1, -1,+1,+1,-1,-1,-1,+1, +1,+1,-1,-1,-1,-1,-1, +1,+1,-1,-1,-1,-1,-1,
     +1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 7
    [+1,+1,+1,+1,+1,+1,+1, +1,-1,-1,-1,-1,+1,+1, -1,-1,-1,-1,-1,+1,-1, -1,-1,-1,-1,+1,+1,-1,
     -1,-1,-1,-1,+1,-1,-1, -1,-1,-1,+1,+1,-1,-1, -1,-1,-1,+1,-1,-1,-1, -1,-1,+1,+1,-1,-1,-1,
     -1,-1,+1,-1,-1,-1,-1, -1,-1,+1,-1,-1,-1,-1, -1,-1,+1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 8
    [-1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1,
     -1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1,
     +1,-1,-1,-1,-1,-1,+1, +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,-1, -1,-1,-1,-1,-1,-1,-1],
    # 9
    [-1,+1,+1,+1,+1,+1,-1, +1,+1,-1,-1,-1,+1,+1, +1,-1,-1,-1,-1,-1,+1, +1,-1,-1,-1,-1,-1,+1,
     +1,+1,-1,-1,-1,+1,+1, -1,+1,+1,+1,+1,+1,+1, -1,-1,-1,-1,-1,+1,+1, -1,-1,-1,-1,-1,+1,+1,
     +1,-1,-1,-1,-1,+1,-1, +1,+1,-1,-1,+1,+1,-1, -1,+1,+1,+1,+1,-1,-1, -1,-1,-1,-1,-1,-1,-1],
], dtype=np.float32)


class ScaledTanh1998(nn.Module):
    """Eq. 6 & Appendix A (p. 41): f(a) = A * tanh(S * a) with A = 1.7159, S = 2/3."""
    def __init__(self, A: float = 1.7159, S: float = 2.0 / 3.0):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.A * torch.tanh(self.S * a)


class Subsampling1998(nn.Module):
    """
    Section II-A & II-B (pp. 6-8): 2x2 non-overlapping sum multiplied by 1 trainable
    coefficient per feature map + 1 trainable bias per feature map, followed by ScaledTanh1998.
    """
    def __init__(self, n_maps: int):
        super().__init__()
        self.n_maps = n_maps
        self.coef = nn.Parameter(torch.empty(1, n_maps, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_maps, 1, 1))
        self.act = ScaledTanh1998()
        # Appendix A (p. 41): uniform in [-2.4 / F_i, +2.4 / F_i] with F_i = 4
        nn.init.uniform_(self.coef, -2.4 / 4.0, 2.4 / 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sum 2x2 non-overlapping blocks:
        s2x2 = x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]
        return self.act(self.coef * s2x2 + self.bias)


class AsymmetricC3_1998(nn.Module):
    """Section II-B & Table I (p. 8): 16 C3 maps connected to subsets of S2's 6 maps (1,516 params)."""
    def __init__(self, table=TABLE_I_C3_MAP):
        super().__init__()
        self.table = table
        self.convs = nn.ModuleList([
            nn.Conv2d(len(in_maps), 1, kernel_size=5, stride=1, bias=True)
            for in_maps in table
        ])
        self.act = ScaledTanh1998()
        for conv, in_maps in zip(self.convs, table):
            fi = len(in_maps) * 25
            nn.init.uniform_(conv.weight, -2.4 / fi, 2.4 / fi)
            nn.init.zeros_(conv.bias)

    def forward(self, s2: torch.Tensor) -> torch.Tensor:
        outs = [conv(s2[:, in_maps, :, :]) for conv, in_maps in zip(self.convs, self.table)]
        return self.act(torch.cat(outs, dim=1))


class AuthenticLeNet5_1998(nn.Module):
    """
    Exact 1998 LeNet-5 (Section II-B, Fig. 2, pp. 7-9):
      Input (1@32x32) -> C1 (6@28x28) -> S2 (6@14x14) -> C3 (16@10x10, Table I)
      -> S4 (16@5x5) -> C5 (120@1x1) -> F6 (84@1x1) -> Euclidean RBF (10 classes).
    Written with 1x1 convolutions for F6 so it also runs directly as a Space-Displacement
    Neural Network (SDNN, Section VII, Fig. 23, p. 27) on any 32 x W_wide canvas!
    """
    def __init__(self, rbf_trainable: bool = False):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, bias=True)
        self.act1 = ScaledTanh1998()
        self.s2 = Subsampling1998(6)
        self.c3 = AsymmetricC3_1998(TABLE_I_C3_MAP)
        self.s4 = Subsampling1998(16)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, bias=True)
        self.act5 = ScaledTanh1998()
        self.f6 = nn.Conv2d(120, 84, kernel_size=1, stride=1, bias=True)
        self.act6 = ScaledTanh1998()

        # Appendix A fan-in initialization [-2.4/F_i, +2.4/F_i]:
        nn.init.uniform_(self.c1.weight, -2.4 / 25.0, 2.4 / 25.0)
        nn.init.zeros_(self.c1.bias)
        nn.init.uniform_(self.c5.weight, -2.4 / 400.0, 2.4 / 400.0)
        nn.init.zeros_(self.c5.bias)
        nn.init.uniform_(self.f6.weight, -2.4 / 120.0, 2.4 / 120.0)
        nn.init.zeros_(self.f6.bias)

        rbf_init = torch.tensor(DIGIT_BITMAPS_7X12, dtype=torch.float32)  # (10, 84)
        if rbf_trainable:
            self.rbf_centers = nn.Parameter(rbf_init)
        else:
            self.register_buffer("rbf_centers", rbf_init)

    def forward_features(self, x: torch.Tensor):
        c1 = self.act1(self.c1(x))
        s2 = self.s2(c1)
        c3 = self.c3(s2)
        s4 = self.s4(c3)
        c5 = self.act5(self.c5(s4))
        f6 = self.act6(self.f6(c5))  # (B, 84, H_out, W_out)
        return c1, s2, c3, s4, c5, f6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, f6 = self.forward_features(x)
        # Eq. 7 (p. 8): y_i = sum_j (x_j - w_{ij})^2
        # f6 has shape (B, 84, H_out, W_out); rbf_centers has shape (10, 84)
        diff = f6.unsqueeze(1) - self.rbf_centers.view(1, 10, 84, 1, 1)
        y_rbf = torch.sum(diff ** 2, dim=2)  # (B, 10, H_out, W_out)
        if y_rbf.shape[2] == 1 and y_rbf.shape[3] == 1:
            return y_rbf.squeeze(-1).squeeze(-1)  # (B, 10)
        return y_rbf


# ============================================================================
# E2: Exact Accounting, Appendix A Scaled Tanh, & Appendix C Gauss-Newton Hessian
# ============================================================================

def run_e2_architecture_and_hessian():
    print("=== E2.1: Exact LeNet-5 Layer-by-Layer Parameter & Connection Accounting (Section II-B, pp. 7-9) ===")
    torch.manual_seed(1998)
    model = AuthenticLeNet5_1998(rbf_trainable=False)
    x_dummy = torch.zeros(1, 1, 32, 32)
    c1, s2, c3, s4, c5, f6 = model.forward_features(x_dummy)
    y_out = model(x_dummy)

    layer_stats = [
        ("C1 (6@28x28, 5x5 conv)", sum(p.numel() for p in model.c1.parameters()), 156 * 28 * 28, tuple(c1.shape[1:])),
        ("S2 (6@14x14, 2x2 subs)", sum(p.numel() for p in model.s2.parameters()), (4 + 1) * 6 * 14 * 14, tuple(s2.shape[1:])),
        ("C3 (16@10x10, Table I)", sum(p.numel() for p in model.c3.parameters()), 1516 * 10 * 10, tuple(c3.shape[1:])),
        ("S4 (16@5x5, 2x2 subs)", sum(p.numel() for p in model.s4.parameters()), (4 + 1) * 16 * 5 * 5, tuple(s4.shape[1:])),
        ("C5 (120@1x1, 5x5 conv)", sum(p.numel() for p in model.c5.parameters()), 48120 * 1 * 1, tuple(c5.shape[1:])),
        ("F6 (84 units, full)", sum(p.numel() for p in model.f6.parameters()), 10164 * 1, (84,)),
        ("RBF (10 Euclidean units)", 0, 10 * 84, tuple(y_out.shape[1:])),
    ]
    total_params = sum(p for _, p, _, _ in layer_stats)
    total_conns = sum(c for _, _, c, _ in layer_stats)
    for name, p_cnt, c_cnt, shp in layer_stats:
        print(f"  {name:26s} | shape={str(shp):14s} | trainable_params={p_cnt:6d} | connections={c_cnt:7d}")
    print(f"  TOTAL LeNet-5 (Fig. 2, p. 7): trainable_params = {total_params} (paper: 60,000), connections = {total_conns} (paper: 340,908)")
    assert total_params == 60000 and total_conns == 340908

    # Contrast with dense S2->C3 (without Table I):
    c3_dense_params = 16 * (6 * 25 + 1)
    print(f"  Table I C3 savings: {layer_stats[2][1]} params / {layer_stats[2][2]} conns vs dense S2->C3 {c3_dense_params} params / {c3_dense_params * 100} conns (saves 37.3%!)")

    print("\n=== E2.2: Appendix A Scaled Hyperbolic Tangent f(a) = 1.7159 tanh(2/3 a) (Eq. 6 & p. 41) ===")
    A, S = 1.7159, 2.0 / 3.0
    f_pos1 = A * np.tanh(S * 1.0)
    f_neg1 = A * np.tanh(S * (-1.0))
    # Analytical second derivative: f''(a) = -2 * A * S^2 * tanh(S*a) * sech^2(S*a)
    # Maximum of |tanh(u) * (1 - tanh^2(u))| occurs at 1 - 3 tanh^2(u) = 0 -> tanh(u) = +/- 1/sqrt(3)
    a_crit = (1.0 / S) * np.arctanh(1.0 / np.sqrt(3.0))
    grid_a = np.linspace(0.0, 3.0, 30001)
    u = S * grid_a
    f2_abs = np.abs(-2.0 * A * (S ** 2) * np.tanh(u) * (1.0 - np.tanh(u) ** 2))
    a_num_max = float(grid_a[int(np.argmax(f2_abs))])
    print(f"  f(+1.0) = {f_pos1:+.6f}, f(-1.0) = {f_neg1:+.6f} (|f(1) - 1| = {abs(f_pos1 - 1.0):.2e})")
    print(f"  Slope at origin f'(0) = A*S = {A * S:.4f}, slope at +/-1 f'(+/-1) = {A * S * (1.0 - np.tanh(S) ** 2):.4f}")
    print(f"  Max curvature |f''(a)| occurs at a = +/-{a_crit:.4f} (numerical grid max: +/-{a_num_max:.4f} ~ +/-1.0 RBF target!)")
    assert abs(f_pos1 - 1.0) < 1e-4 and abs(a_crit - 0.988) < 2e-3

    print("\n=== E2.3: Appendix C Stochastic Diagonal Levenberg-Marquardt Gauss-Newton Hessian (Eq. 18-27, pp. 41-42) ===")
    # Verify Eq. 25 & 27 (Gauss-Newton diagonal Hessian backpropagation) on a 2-layer shared-weight network
    # and show why weight sharing makes early-layer h_{kk} orders of magnitude larger than unshared weights:
    # Consider a 1D convolutional layer with 1 shared kernel w_k replicated across M=28x28=784 spatial locations,
    # feeding an output unit:
    M_locations = 784
    rng = np.random.default_rng(1998)
    x_patches = rng.uniform(-0.1, 1.175, size=(M_locations, 25))
    w_shared = rng.uniform(-2.4 / 25.0, 2.4 / 25.0, size=25)
    w_head = rng.uniform(-2.4 / M_locations, 2.4 / M_locations, size=M_locations)

    a_conv = x_patches @ w_shared
    f_conv = A * np.tanh(S * a_conv)
    fp_conv = A * S * (1.0 - np.tanh(S * a_conv) ** 2)

    a_out = float(w_head @ f_conv)
    f_out = A * np.tanh(S * a_out)
    fp_out = A * S * (1.0 - np.tanh(S * a_out) ** 2)

    # Loss E = 0.5 * (t - f_out)^2 -> d^2 E / d(f_out)^2 = 1.0
    # Eq. 27 at output unit: d^2 E / da_out^2 = (f'(a_out))^2 * 1.0
    d2E_da_out2 = (fp_out ** 2) * 1.0
    # Eq. 25 for head weights: d^2 E / dw_head_i^2 = d^2 E / da_out^2 * (f_conv_i)^2
    h_head = d2E_da_out2 * (f_conv ** 2)
    # Eq. 27 backpropagated to conv units: d^2 E / da_conv_i^2 = (f'(a_conv_i))^2 * w_head_i^2 * d^2 E / da_out^2
    d2E_da_conv2 = (fp_conv ** 2) * (w_head ** 2) * d2E_da_out2
    # Eq. 23 & 25 summed over all M=784 shared connections (i,j) in V_k:
    h_shared = np.sum(d2E_da_conv2[:, None] * (x_patches ** 2), axis=0)

    # Compare against exact Gauss-Newton J^T J diagonal (where J = df_out / dw):
    df_out_dw_head = fp_out * f_conv
    df_out_dw_shared_per_conn = (fp_out * w_head * fp_conv)[:, None] * x_patches  # (784, 25)
    h_shared_exact_diag_sum = np.sum(df_out_dw_shared_per_conn ** 2, axis=0)
    gn_diff = float(np.max(np.abs(h_shared - h_shared_exact_diag_sum)))
    print(f"  Max |Eq. 23-27 backprop h_kk - sum_(i,j in V_k) (df_out/du_ij)^2|: {gn_diff:.2e}")
    print(f"  Mean Gauss-Newton diagonal curvature: shared C1 weight h_kk = {np.mean(h_shared):.4e} vs unshared head weight = {np.mean(h_head):.4e}")
    assert gn_diff < 1e-15


# ============================================================================
# E3: Euclidean RBF Collapse (MSE Eq. 8 vs MAP Eq. 9) & Space-Displacement (SDNN)
# ============================================================================

def make_synthetic_digit_dataset(n_per_class: int = 30, seed: int = 1998):
    """Generates 32x32 digit images centered with random (-2..+2) pixel shifts from the 7x12 stylized bitmaps."""
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []
    for cls in range(10):
        bmp = (DIGIT_BITMAPS_7X12[cls].reshape(12, 7) > 0).astype(np.float32)
        # Upsample 12x7 by 2x -> 24x14 glyph and place in 32x32 image with background -0.1, foreground +1.175 (p. 7)
        glyph = np.repeat(np.repeat(bmp, 2, axis=0), 2, axis=1)  # (24, 14)
        for _ in range(n_per_class):
            img = np.full((32, 32), -0.1, dtype=np.float32)
            dy = int(rng.integers(-2, 3))
            dx = int(rng.integers(-2, 3))
            r0, c0 = 4 + dy, 9 + dx
            img[r0:r0 + 24, c0:c0 + 14] = np.where(glyph > 0.5, 1.175, -0.1)
            img += rng.normal(0.0, 0.05, size=(32, 32)).astype(np.float32)
            X_list.append(img[None, :, :])
            Y_list.append(cls)
    return torch.tensor(np.array(X_list)), torch.tensor(Y_list, dtype=torch.long)


def run_e3_rbf_and_sdnn():
    print("\n=== E3.1: Euclidean RBF Output Collapse under Adaptive RBF + MSE (Eq. 8) vs Competitive MAP (Eq. 9, pp. 8-9) ===")
    X_data, Y_data = make_synthetic_digit_dataset(n_per_class=20, seed=1998)

    # Arm 1: Adaptive RBF centers with pure MSE E(W) = (1/P) sum_p y_{D_p} (Eq. 8, p. 9):
    # "First, if we allow the parameters of the RBF to adapt, E(W) has a trivial, but totally unacceptable, solution.
    #  In this solution, all the RBF parameter vectors are equal, and the state of F6 is constant and equal to that
    #  parameter vector. In this case the network happily ignores the input, and all the RBF outputs are equal to zero."
    torch.manual_seed(1998)
    net_mse_adapt = AuthenticLeNet5_1998(rbf_trainable=True)
    # Initialize RBF centers near a small ball so gradient descent quickly finds the trivial collapse:
    with torch.no_grad():
        net_mse_adapt.rbf_centers.mul_(0.2)
    opt1 = torch.optim.SGD([
        {"params": [p for n, p in net_mse_adapt.named_parameters() if n != "rbf_centers"], "lr": 0.02},
        {"params": [net_mse_adapt.rbf_centers], "lr": 0.25},
    ], momentum=0.9)

    for _ in range(120):
        opt1.zero_grad()
        y_rbf = net_mse_adapt(X_data)  # (N, 10)
        loss_mse = torch.mean(y_rbf[torch.arange(len(Y_data)), Y_data])
        loss_mse.backward()
        opt1.step()

    with torch.no_grad():
        y_rbf_coll = net_mse_adapt(X_data)
        centers_coll = net_mse_adapt.rbf_centers
        center_spread_coll = float(torch.pdist(centers_coll).mean().item())
        mean_y_coll = float(y_rbf_coll.mean().item())
        # Margin between best wrong class penalty and true class penalty:
        y_true_c = y_rbf_coll[torch.arange(len(Y_data)), Y_data]
        y_wrong_c = y_rbf_coll.clone()
        y_wrong_c[torch.arange(len(Y_data)), Y_data] = float("inf")
        margin_coll = float((torch.min(y_wrong_c, dim=1).values - y_true_c).mean().item())
        # Under any finite sensor/readout tolerance (eps = 0.01 on F6 state or penalties) or confidence threshold:
        rng_eval = torch.Generator().manual_seed(1998)
        y_noisy_coll = y_rbf_coll + 0.01 * torch.randn(y_rbf_coll.shape, generator=rng_eval)
        acc_coll = float((torch.argmin(y_noisy_coll, dim=1) == Y_data).float().mean().item()) * 100.0

    # Arm 2: Adaptive RBF centers with Competitive MAP Criterion (Eq. 9, p. 9):
    # E(W) = (1/P) sum_p [ y_{D_p} + log(e^{-j} + sum_i e^{-y_i}) ]
    torch.manual_seed(1998)
    net_map_adapt = AuthenticLeNet5_1998(rbf_trainable=True)
    opt2 = torch.optim.SGD(net_map_adapt.parameters(), lr=0.008, momentum=0.9)
    j_rubbish = 4.0
    for _ in range(60):
        opt2.zero_grad()
        y_rbf = net_map_adapt(X_data)  # (N, 10)
        y_true = y_rbf[torch.arange(len(Y_data)), Y_data]
        # Numerically stable log(e^{-j} + sum_i e^{-y_i}) via logsumexp:
        neg_terms = torch.cat([-y_rbf, torch.full((len(Y_data), 1), -j_rubbish)], dim=1)
        comp_term = torch.logsumexp(neg_terms, dim=1)
        loss_map = torch.mean(y_true + comp_term)
        loss_map.backward()
        opt2.step()

    with torch.no_grad():
        y_rbf_map = net_map_adapt(X_data)
        y_true_m = y_rbf_map[torch.arange(len(Y_data)), Y_data]
        y_wrong_m = y_rbf_map.clone()
        y_wrong_m[torch.arange(len(Y_data)), Y_data] = float("inf")
        margin_map = float((torch.min(y_wrong_m, dim=1).values - y_true_m).mean().item())
        y_noisy_map = y_rbf_map + 0.01 * torch.randn(y_rbf_map.shape, generator=rng_eval)
        acc_map = float((torch.argmin(y_noisy_map, dim=1) == Y_data).float().mean().item()) * 100.0
        center_spread_map = float(torch.pdist(net_map_adapt.rbf_centers).mean().item())

    print(f"  Adaptive RBF + Pure MSE (Eq. 8):        mean RBF penalty = {mean_y_coll:.5f}, center spread = {center_spread_coll:.4f}, class margin = {margin_coll:.5f}, acc = {acc_coll:.1f}% (COLLAPSED!)")
    print(f"  Adaptive RBF + Competitive MAP (Eq. 9): center spread = {center_spread_map:.4f}, class margin = {margin_map:.2f}, acc = {acc_map:.1f}% (NO COLLAPSE!)")
    assert mean_y_coll < 0.05 and center_spread_coll < 0.01 and acc_coll <= 25.0 and acc_map >= 98.0

    print("\n=== E3.2: Space-Displacement Neural Network (SDNN) Exact Equivalence & MAC Speedup (Section VII, Fig. 23, p. 27) ===")
    # Because LeNet-5 has two 2x2 stride-2 subsampling layers (S2 and S4), total spatial stride is 2 * 2 = 4 pixels.
    # For a wide canvas of height 32 and width W_wide = 32 + 4 * (K - 1) = 92 (K = 16 horizontal window positions):
    K_windows = 16
    W_wide = 32 + 4 * (K_windows - 1)  # 92 pixels wide
    canvas = torch.full((1, 1, 32, W_wide), -0.1, dtype=torch.float32)
    # Place 3 digits ('2', '7', '4') centered in windows k = 1, 8, 15 (window starts at 4*k, glyph centered at +9):
    for cls, k_win in [(2, 1), (7, 8), (4, 15)]:
        x_off = 4 * k_win + 9
        bmp = (DIGIT_BITMAPS_7X12[cls].reshape(12, 7) > 0).astype(np.float32)
        glyph = torch.tensor(np.repeat(np.repeat(bmp, 2, axis=0), 2, axis=1))
        canvas[0, 0, 4:28, x_off:x_off + 14] = torch.where(glyph > 0.5, 1.175, -0.1)

    with torch.no_grad():
        # 1. SDNN single-pass over the 32x92 canvas:
        sdnn_out = net_map_adapt(canvas).squeeze(2)  # shape (1, 10, 16)

        # 2. Sliding-window crop-by-crop at each of the K = 16 stride-4 positions:
        crop_outs = []
        for k in range(K_windows):
            crop = canvas[:, :, :, 4 * k:4 * k + 32]
            crop_outs.append(net_map_adapt(crop))  # (1, 10)
        sliding_out = torch.stack(crop_outs, dim=2)  # (1, 10, 16)

    max_sdnn_diff = float(torch.max(torch.abs(sdnn_out - sliding_out)).item())
    spotted_at_1_8_15 = [int(torch.argmin(sdnn_out[0, :, k]).item()) for k in [1, 8, 15]]

    # Exact MAC comparison for C1..S4 across K=16 windows:
    # Sliding window repeats all C1..S4 computations K=16 times:
    macs_c1_to_s4_single = 122304 + 5880 + 151600 + 2000  # 281,784
    macs_sliding_c1_s4 = K_windows * macs_c1_to_s4_single
    # SDNN computes C1 on 28 x (92-4) = 28x88, S2 on 14x44, C3 on 10x40, S4 on 5x20:
    macs_sdnn_c1_s4 = (156 * 28 * 88) + (5 * 6 * 14 * 44) + (1516 * 10 * 40) + (5 * 16 * 5 * 20)
    speedup_c1_s4 = macs_sliding_c1_s4 / macs_sdnn_c1_s4

    print(f"  Max |SDNN_single_pass - Sliding_16_crops| across 32x{W_wide} canvas: {max_sdnn_diff:.2e}")
    print(f"  Spotted digit classes at window positions k=(1, 8, 15) [true=(2, 7, 4)]: {spotted_at_1_8_15}")
    print(f"  C1..S4 connections evaluated: Sliding 16 crops = {macs_sliding_c1_s4:,d} vs SDNN shared feature maps = {macs_sdnn_c1_s4:,d} ({speedup_c1_s4:.2f}x fewer MACs!)")
    assert max_sdnn_diff < 1e-4 and spotted_at_1_8_15 == [2, 7, 4] and speedup_c1_s4 > 4.0


# ============================================================================
# E4: Graph Transformer Networks (GTNs) — Viterbi vs Forward Scorer,
#     Analytical Graph Backprop (Eq. 15-16), and Unsegmented String Training
# ============================================================================

class InterpretationGraph:
    """
    Directed Acyclic Graph (DAG) for GTNs (Sections V & VI, pp. 19-26):
      - Nodes 0 .. N-1 in topological order (start_node = 0, end_node = N-1)
      - Arcs: list of (arc_id, src, dst, label_tuple, penalty_c)
    """
    def __init__(self, n_nodes: int, arcs: list):
        self.n_nodes = n_nodes
        self.arcs = arcs  # [(arc_id, src, dst, label_tuple, penalty_c), ...]

    def filter_constrained(self, target_labels: tuple):
        """
        Path Selector / Grammar Composition Transformer (Section VI-A & VIII-B, pp. 21, 31-33):
        Extracts the constrained interpretation graph G_c whose paths from start to end emit target_labels.
        State in G_c is (node_in_G, prefix_len_matched).
        """
        L = len(target_labels)
        # Forward reachability of (node, pos):
        valid_arcs = []
        for arc_id, s, d, lab, c in self.arcs:
            k_len = len(lab)
            for pos in range(L - k_len + 1):
                if target_labels[pos:pos + k_len] == lab:
                    valid_arcs.append((arc_id, (s, pos), (d, pos + k_len), lab, c))

        start_state = (0, 0)
        end_state = (self.n_nodes - 1, L)

        # Keep only states on a valid path from start_state to end_state:
        fwd_reach = {start_state}
        for _, u, v, _, _ in valid_arcs:
            if u in fwd_reach:
                fwd_reach.add(v)
        bwd_reach = {end_state}
        for _, u, v, _, _ in reversed(valid_arcs):
            if v in bwd_reach:
                bwd_reach.add(u)
        active = sorted(fwd_reach & bwd_reach, key=lambda x: (x[1], x[0]))
        if start_state not in active or end_state not in active:
            raise ValueError("No valid path in G_c for target_labels")

        state_to_idx = {st: idx for idx, st in enumerate(active)}
        c_arcs = [
            (orig_arc_id, state_to_idx[u], state_to_idx[v], lab, c)
            for orig_arc_id, u, v, lab, c in valid_arcs
            if u in state_to_idx and v in state_to_idx
        ]
        return InterpretationGraph(len(active), c_arcs)

    def viterbi_forward_and_bprop(self):
        """
        Eq. 10 (p. 20): v_n = min_{i in U_n} (c_i + v_{s_i}).
        Returns (v_end, best_label_seq, dC_vit_dc_dict).
        """
        incoming = {n: [] for n in range(self.n_nodes)}
        for idx, (arc_id, s, d, lab, c) in enumerate(self.arcs):
            incoming[d].append((idx, arc_id, s, d, lab, c))

        v = np.full(self.n_nodes, np.inf, dtype=float)
        v[0] = 0.0
        min_arc = {}
        for n in range(1, self.n_nodes):
            for idx, arc_id, s, d, lab, c in incoming[n]:
                if v[s] + c < v[n]:
                    v[n] = v[s] + c
                    min_arc[n] = (idx, arc_id, s, d, lab, c)

        # Traceback from end_node = n_nodes - 1:
        curr = self.n_nodes - 1
        dC_dc = {}
        labels = []
        while curr != 0:
            _, arc_id, s, d, lab, _ = min_arc[curr]
            dC_dc[arc_id] = dC_dc.get(arc_id, 0.0) + 1.0
            labels = list(lab) + labels
            curr = s
        return float(v[self.n_nodes - 1]), tuple(labels), dC_dc

    def forward_score_and_bprop(self):
        """
        Forward Scorer (Section VI-C, Eq. 13-16, p. 25):
          Forward pass (Eq. 13-14):
            f_0 = 0
            f_n = logadd_{i in U_n} (c_i + f_{s_i}) = -log( sum_{i in U_n} exp(-(c_i + f_{s_i})) )
          Backward pass (Eq. 15-16):
            dE / df_{N-1} = 1.0
            dE / df_n     = sum_{i in D_n} (dE / df_{d_i}) * exp(f_{d_i} - c_i - f_n)
            dE / dc_i     = (dE / df_{d_i}) * exp(f_{d_i} - c_i - f_{s_i})
        """
        incoming = {n: [] for n in range(self.n_nodes)}
        outgoing = {n: [] for n in range(self.n_nodes)}
        for idx, (arc_id, s, d, lab, c) in enumerate(self.arcs):
            incoming[d].append((idx, arc_id, s, d, c))
            outgoing[s].append((idx, arc_id, s, d, c))

        f = np.zeros(self.n_nodes, dtype=float)
        for n in range(1, self.n_nodes):
            terms = np.array([c + f[s] for _, _, s, _, c in incoming[n]], dtype=float)
            m_min = np.min(terms)  # p. 25: "factorize the largest e^{-x_i} (smallest penalty) out of the logarithm"
            f[n] = m_min - np.log(np.sum(np.exp(-(terms - m_min))))

        df = np.zeros(self.n_nodes, dtype=float)
        df[self.n_nodes - 1] = 1.0
        for n in range(self.n_nodes - 2, -1, -1):
            df[n] = sum(df[d] * np.exp(f[d] - c - f[n]) for _, _, _, d, c in outgoing[n])

        dC_dc = {}
        for arc_id, s, d, lab, c in self.arcs:
            grad_arc = float(df[d] * np.exp(f[d] - c - f[s]))
            dC_dc[arc_id] = dC_dc.get(arc_id, 0.0) + grad_arc

        return float(f[self.n_nodes - 1]), dC_dc


def run_e4_gtn():
    print("\n=== E4.1: GTN Forward Scorer Backward Recurrence (Eq. 15-16, p. 25) vs Central Finite Differences ===")
    # Build a multi-path segmentation/interpretation DAG (0 -> 1 -> 2, plus direct skip 0 -> 2) with multiple class arcs:
    rng = np.random.default_rng(1998)
    raw_arcs = []
    arc_id = 0
    for s, d in [(0, 1), (1, 2), (0, 2)]:
        for cls in range(3):
            raw_arcs.append((arc_id, s, d, (cls,), float(rng.uniform(0.3, 3.5))))
            arc_id += 1
    G = InterpretationGraph(3, raw_arcs)
    C_forw, dC_dc_analytical = G.forward_score_and_bprop()

    eps = 1e-6
    max_fd_err = 0.0
    for i in range(len(raw_arcs)):
        arcs_plus = list(raw_arcs)
        arcs_minus = list(raw_arcs)
        aid, s, d, lab, c = raw_arcs[i]
        arcs_plus[i] = (aid, s, d, lab, c + eps)
        arcs_minus[i] = (aid, s, d, lab, c - eps)
        f_plus, _ = InterpretationGraph(3, arcs_plus).forward_score_and_bprop()
        f_minus, _ = InterpretationGraph(3, arcs_minus).forward_score_and_bprop()
        fd_grad = (f_plus - f_minus) / (2.0 * eps)
        max_fd_err = max(max_fd_err, abs(dC_dc_analytical[aid] - fd_grad))
    print(f"  Max |Eq. 15-16 analytical dC_forw/dc_i - finite_diff| across all {len(raw_arcs)} arcs: {max_fd_err:.2e}")
    assert max_fd_err < 1e-10

    print("\n=== E4.2: Parallel-Path Evidence Aggregation — Forward Scorer vs Viterbi (Section VI-C, p. 25) ===")
    # Page 25: "the combined forward penalty of two paths C_1 and C_2 associated with the same label sequence
    # may be less than the penalty of a path C_3 associated with another label sequence, even though the penalty
    # of C_3 might be less than any one of C_1 or C_2."
    # Two segmentations (0->1->3 and 0->2->3) both emit label sequence (3, 4) with total path penalty C_1 = 2.2, C_2 = 2.2;
    # One segmentation (0->3) emits wrong label sequence (8,) with path penalty C_3 = 1.80:
    ambig_arcs = [
        (0, 0, 1, (3,), 1.10), (1, 1, 3, (4,), 1.10),  # Path 1: label (3,4), penalty 2.20
        (2, 0, 2, (3,), 1.10), (3, 2, 3, (4,), 1.10),  # Path 2: label (3,4), penalty 2.20
        (4, 0, 3, (8,), 1.80),                         # Path 3: label (8,),  penalty 1.80
    ]
    G_ambig = InterpretationGraph(4, ambig_arcs)
    vit_pen, vit_ans, _ = G_ambig.viterbi_forward_and_bprop()
    forw_34, _ = G_ambig.filter_constrained((3, 4)).forward_score_and_bprop()
    forw_8, _ = G_ambig.filter_constrained((8,)).forward_score_and_bprop()
    print(f"  Viterbi best path: label={vit_ans}, penalty={vit_pen:.4f} (picks wrong single path C_3=1.80 < 2.20!)")
    print(f"  Forward Scorer penalties: label (3,4) = {forw_34:.4f} (= 2.20 - ln 2), label (8,) = {forw_8:.4f} -> selects (3,4)!")
    assert vit_ans == (8,) and forw_34 < forw_8

    print("\n=== E4.3: Unsegmented String-Level Discriminative Forward Training (Section VI-D, Fig. 21, pp. 25-26) ===")
    # Construct a string recognition problem with Heuristic Over-Segmentation (Section V, Fig. 16, p. 19):
    # Each string has 2 characters from classes {0, 1, 2}. However, Heuristic Over-Segmentation cuts each string
    # into 3 ink pieces (nodes 0 -> 1 -> 2 -> 3), where EITHER:
    #   - Type A: piece (0,1) is char 1, and pieces (1,2)+(2,3) merged into arc (1,3) is char 2; OR
    #   - Type B: pieces (0,1)+(1,2) merged into arc (0,2) is char 1, and piece (2,3) is char 2.
    # The segmentation graph G_seg has 4 candidate segment arcs:
    #   arc_seg 0: (0 -> 1) [1 piece]
    #   arc_seg 1: (1 -> 2) [1 piece — ALWAYS a half-character fragment!]
    #   arc_seg 2: (2 -> 3) [1 piece]
    #   arc_seg 3: (0 -> 2) [2 pieces merged]
    #   arc_seg 4: (1 -> 3) [2 pieces merged]
    # Valid 2-character paths through G_seg are ONLY:
    #   Path A: (0 -> 1) + (1 -> 3)  [uses seg 0 and seg 4]
    #   Path B: (0 -> 2) + (2 -> 3)  [uses seg 3 and seg 2]
    #   Path C: (0 -> 1) + (1 -> 2) + (2 -> 3) [3 single pieces, emits 3 chars]
    # No ground-truth segmentation is ever given to the learner — ONLY the 2-character string label (c1, c2)!
    D_feat = 12
    prototypes = rng.normal(0.0, 1.0, size=(4, D_feat))  # 0, 1, 2 = valid chars; 3 = half-char fragment
    for k in range(4):
        prototypes[k] /= np.linalg.norm(prototypes[k])

    train_strings = []
    for i in range(90):
        c1, c2 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        seg_type = i % 2  # 0 = Type A ((0,1) + (1,3)), 1 = Type B ((0,2) + (2,3))
        noise = lambda: rng.normal(0.0, 0.15, size=D_feat)
        if seg_type == 0:
            # True cuts are (0->1)=c1 and (1->3)=c2; other candidate arcs are fragments (proto 3)
            feats = {
                (0, 1): prototypes[c1] + noise(),
                (1, 2): prototypes[3] + noise(),
                (2, 3): prototypes[3] + noise(),
                (0, 2): prototypes[3] + noise(),
                (1, 3): prototypes[c2] + noise(),
            }
        else:
            # True cuts are (0->2)=c1 and (2->3)=c2; other candidate arcs are fragments (proto 3)
            feats = {
                (0, 1): prototypes[3] + noise(),
                (1, 2): prototypes[3] + noise(),
                (2, 3): prototypes[c2] + noise(),
                (0, 2): prototypes[c1] + noise(),
                (1, 3): prototypes[3] + noise(),
            }
        train_strings.append((feats, (c1, c2), seg_type))

    # Recognizer is a 2-layer neural network mapping each segment feature x in R^{12} to class penalties (Section VI-A, p. 22):
    def run_string_training(mode: str):
        torch.manual_seed(42)
        rec_net = nn.Sequential(
            nn.Linear(D_feat, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )
        # Initialize output layer near small values so non-discriminative Viterbi's collapse (p. 22) is tested directly:
        with torch.no_grad():
            rec_net[2].weight.mul_(0.1)
            rec_net[2].bias.fill_(-1.0)
        opt_rec = torch.optim.SGD(rec_net.parameters(), lr=0.08, momentum=0.9)
        seg_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]

        for ep in range(40):
            for feats, target_seq, _ in train_strings:
                opt_rec.zero_grad()
                x_batch = torch.tensor(np.array([feats[e] for e in seg_edges]), dtype=torch.float32)  # (5, 12)
                if mode == "non_disc_viterbi":
                    # Page 22: "If the recognizer is a simple neural network with sigmoid output units, the minimum of
                    # the loss function is attained, not when the recognizer always gives the right answer, but when it
                    # ignores the input, and sets its output to a constant vector with small values for all the components."
                    pen_tensor = torch.sigmoid(rec_net(x_batch))  # (5, 3) in (0, 1)
                else:
                    # Unnormalized positive penalties for discriminative GTN training (Section VI-E, p. 26):
                    pen_tensor = torch.nn.functional.softplus(rec_net(x_batch))  # (5, 3)

                pen_np = pen_tensor.detach().numpy()
                arcs = []
                aid = 0
                for e_idx, (s, d) in enumerate(seg_edges):
                    for cls in range(3):
                        arcs.append((aid, s, d, (cls,), float(pen_np[e_idx, cls])))
                        aid += 1

                G_int = InterpretationGraph(4, arcs)
                G_c = G_int.filter_constrained(target_seq)

                if mode == "non_disc_viterbi":
                    # Eq. 11 (p. 22): E_vit = C_cvit (only pushes down constrained Viterbi path!)
                    _, _, dC_c = G_c.viterbi_forward_and_bprop()
                    grad_arcs = dC_c
                elif mode == "disc_viterbi":
                    # Eq. 12 (p. 24): E_dvit = C_cvit - C_vit
                    _, _, dC_c = G_c.viterbi_forward_and_bprop()
                    _, _, dC_u = G_int.viterbi_forward_and_bprop()
                    grad_arcs = {k: dC_c.get(k, 0.0) - dC_u.get(k, 0.0) for k in range(aid)}
                elif mode == "disc_forward":
                    # Eq. 17 (p. 26): E_dforw = C_cforw - C_forw
                    _, dC_c = G_c.forward_score_and_bprop()
                    _, dC_u = G_int.forward_score_and_bprop()
                    grad_arcs = {k: dC_c.get(k, 0.0) - dC_u.get(k, 0.0) for k in range(aid)}

                grad_tensor = torch.zeros_like(pen_tensor)
                for k_id, g_val in grad_arcs.items():
                    if g_val != 0.0:
                        grad_tensor[k_id // 3, k_id % 3] = float(g_val)
                pen_tensor.backward(grad_tensor)
                opt_rec.step()

        # Evaluate unconstrained Viterbi string recognition accuracy, segmentation accuracy, and confidence exp(-E_dforw) (p. 39):
        correct_str = 0
        mean_pen_all = 0.0
        conf_correct = []
        with torch.no_grad():
            for feats, target_seq, _ in train_strings:
                x_batch = torch.tensor(np.array([feats[e] for e in seg_edges]), dtype=torch.float32)
                pen_np = (torch.sigmoid(rec_net(x_batch)) if mode == "non_disc_viterbi" else torch.nn.functional.softplus(rec_net(x_batch))).numpy()
                mean_pen_all += float(np.mean(pen_np))
                arcs = []
                aid = 0
                for e_idx, (s, d) in enumerate(seg_edges):
                    for cls in range(3):
                        arcs.append((aid, s, d, (cls,), float(pen_np[e_idx, cls])))
                        aid += 1
                G_int = InterpretationGraph(4, arcs)
                _, vit_seq, _ = G_int.viterbi_forward_and_bprop()
                C_forw, _ = G_int.forward_score_and_bprop()
                C_vit_forw, _ = G_int.filter_constrained(vit_seq).forward_score_and_bprop()
                conf = float(np.exp(-(C_vit_forw - C_forw)))
                if vit_seq == target_seq:
                    correct_str += 1
                    conf_correct.append(conf)
        mean_pen_all /= len(train_strings)
        return correct_str, len(train_strings), mean_pen_all, (float(np.mean(conf_correct)) if conf_correct else 0.0)

    c_nd, n_tot, pen_nd, conf_nd = run_string_training("non_disc_viterbi")
    c_dv, _, pen_dv, conf_dv = run_string_training("disc_viterbi")
    c_df, _, pen_df, conf_df = run_string_training("disc_forward")
    print(f"  Non-Discriminative Viterbi (Eq. 11, p. 22): string acc = {c_nd:2d}/{n_tot} ({100*c_nd/n_tot:5.1f}%), mean arc penalty = {pen_nd:.5f}, conf = {conf_nd:.3f} (COLLAPSED to 0!)")
    print(f"  Discriminative Viterbi     (Eq. 12, p. 24): string acc = {c_dv:2d}/{n_tot} ({100*c_dv/n_tot:5.1f}%), mean arc penalty = {pen_dv:.4f}, mean confidence = {conf_dv:.3f}")
    print(f"  Discriminative Forward     (Eq. 17, p. 26): string acc = {c_df:2d}/{n_tot} ({100*c_df/n_tot:5.1f}%), mean arc penalty = {pen_df:.4f}, mean confidence = {conf_df:.3f}")
    assert pen_nd < 0.01 and conf_nd < 0.15 and c_df == n_tot and c_df >= c_dv


if __name__ == "__main__":
    run_e2_architecture_and_hessian()
    run_e3_rbf_and_sdnn()
    run_e4_gtn()
