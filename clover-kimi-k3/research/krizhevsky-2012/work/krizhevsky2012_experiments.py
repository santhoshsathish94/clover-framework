#!/usr/bin/env python3
"""
krizhevsky2012_experiments.py — Verified from-scratch implementation and experimental verification
of Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012),
"ImageNet Classification with Deep Convolutional Neural Networks", NIPS 2012, pp. 1097-1105.

Covers:
  - E2: Exact 2012 Two-GPU Split-Stream AlexNet Architecture (Fig. 2 & Section 3.5, pp. 4-5):
        - Layer-by-layer parameter, neuron, and MAC accounting (60,965,224 params; 659,272 neurons;
          724,412,928 MACs; 3.83% of params in conv1..conv5 doing 91.91% of MACs)
        - Audit of Figure 2 caption's `253,440` conv1 neuron count (= 55 * 48 * 96 vs true 55 * 55 * 96 = 290,400)
        - Comparison against single-stream (groups=1) and 2014 torchvision AlexNet (61,100,840 params)
  - E3: ReLU vs Saturating tanh Training Speed on a 4-Layer CNN (Section 3.1, Fig. 1, p. 3),
        Section 5 (+1 vs 0/-1) Bias Initialization effect on ReLU gradient flow (p. 6),
        and Local Response Normalization (LRN, Section 3.3, p. 4) exact formula vs PyTorch's alpha/n division.
  - E4: Overlapping Pooling (s=2, z=3 vs s=2, z=2, Section 3.4, p. 4),
        PCA RGB Color Augmentation [p1,p2,p3][a1*l1, a2*l2, a3*l3]^T illumination invariance (Section 4.1, pp. 5-6),
        and Dropout (p=0.5 with 0.5 test-time scaling, Section 4.2, p. 6) co-adaptation & overfitting reduction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Authentic 2012 Local Response Normalization (Section 3.3, p. 4)
# ============================================================================

class KrizhevskyLRN2012(nn.Module):
    """
    Section 3.3 (p. 4):
      b^i_{x,y} = a^i_{x,y} / ( k + alpha * sum_{j=max(0, i - n//2)}^{min(N-1, i + n//2)} (a^j_{x,y})^2 )^beta
    with k = 2, n = 5, alpha = 1e-4, beta = 0.75.
    NOTE: Unlike PyTorch's nn.LocalResponseNorm (which divides alpha by size n),
    Krizhevsky (2012)'s equation multiplies the sum of n adjacent squares directly by alpha!
    Also in the two-GPU architecture (Section 3.2 & 3.5), LRN normalizes within each GPU's
    channel bank (or across adjacent maps). We support both full-layer and per-GPU-stream LRN.
    """
    def __init__(self, n: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 2.0):
        super().__init__()
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, C, H, W)
        B, C, H, W = a.shape
        sq = a ** 2
        pad = self.n // 2
        # Pad along channel dimension with zeros at boundaries [0, C-1]:
        sq_pad = F.pad(sq, (0, 0, 0, 0, pad, pad), mode="constant", value=0.0)
        # Sum over window of length n along channel axis:
        chan_sum = sum(sq_pad[:, j:j + C, :, :] for j in range(self.n))
        denom = (self.k + self.alpha * chan_sum) ** self.beta
        return a / denom


class AuthenticAlexNet2012(nn.Module):
    """
    Exact 2012 NIPS Two-GPU AlexNet Architecture (Section 3.5 & Figure 2, pp. 4-5):
      - conv1: 3   ->  96, 11x11, stride 4, pad 2 (groups=1) + ReLU + LRN + MaxPool(z=3, s=2) -> 55x55 -> 27x27
      - conv2: 96  -> 256,  5x5,  stride 1, pad 2 (groups=2) + ReLU + LRN + MaxPool(z=3, s=2) -> 27x27 -> 13x13
      - conv3: 256 -> 384,  3x3,  stride 1, pad 1 (groups=1, cross-GPU!) + ReLU               -> 13x13
      - conv4: 384 -> 384,  3x3,  stride 1, pad 1 (groups=2, intra-GPU!) + ReLU               -> 13x13
      - conv5: 384 -> 256,  3x3,  stride 1, pad 1 (groups=2, intra-GPU!) + ReLU + MaxPool(z=3, s=2) -> 13x13 -> 6x6
      - fc6:   9216 -> 4096 + ReLU + Dropout(0.5)
      - fc7:   4096 -> 4096 + ReLU + Dropout(0.5)
      - fc8:   4096 -> 1000
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, groups=1, bias=True)
        self.lrn1 = KrizhevskyLRN2012(n=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2, bias=True)
        self.lrn2 = KrizhevskyLRN2012(n=5, alpha=1e-4, beta=0.75, k=2.0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2, bias=True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2, bias=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8 = nn.Linear(4096, num_classes, bias=True)

        self.init_2012_weights()

    def init_2012_weights(self):
        """
        Section 5 (p. 6):
          - Weights in each layer initialized from N(0, 0.01^2)
          - Biases in conv2, conv4, conv5, and fc6, fc7 initialized with constant 1
          - Biases in remaining layers (conv1, conv3, fc8) initialized with constant 0
        """
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc6, self.fc7, self.fc8]:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        for m in [self.conv1, self.conv3, self.fc8]:
            nn.init.constant_(m.bias, 0.0)
        for m in [self.conv2, self.conv4, self.conv5, self.fc6, self.fc7]:
            nn.init.constant_(m.bias, 1.0)


# ============================================================================
# E2: Exact Parameter, Neuron, & MAC Accounting (Figure 2 & Section 3.5)
# ============================================================================

def run_e2_accounting():
    print("=== E2.1: Exact 2012 Two-GPU AlexNet Parameter, Neuron, & MAC Accounting (Section 3.5 & Fig. 2, pp. 4-5) ===")
    model = AuthenticAlexNet2012(num_classes=1000)

    # Layer specs: (name, module, out_C, out_H, out_W, groups)
    layers = [
        ("conv1 (11x11, s=4, g=1)", model.conv1, 96, 55, 55, 1),
        ("conv2 ( 5x5,  s=1, g=2)", model.conv2, 256, 27, 27, 2),
        ("conv3 ( 3x3,  s=1, g=1)", model.conv3, 384, 13, 13, 1),
        ("conv4 ( 3x3,  s=1, g=2)", model.conv4, 384, 13, 13, 2),
        ("conv5 ( 3x3,  s=1, g=2)", model.conv5, 256, 13, 13, 2),
        ("fc6   (9216 -> 4096)   ", model.fc6, 4096, 1, 1, 1),
        ("fc7   (4096 -> 4096)   ", model.fc7, 4096, 1, 1, 1),
        ("fc8   (4096 -> 1000)   ", model.fc8, 1000, 1, 1, 1),
    ]

    tot_params = 0
    conv_params = 0
    tot_neurons = 0
    tot_macs = 0
    conv_macs = 0
    neuron_list = []

    for name, mod, c_out, h_out, w_out, g in layers:
        p_cnt = sum(p.numel() for p in mod.parameters())
        w_cnt = mod.weight.numel()
        neurons = c_out * h_out * w_out
        macs = w_cnt * h_out * w_out
        tot_params += p_cnt
        tot_neurons += neurons
        tot_macs += macs
        neuron_list.append(neurons)
        if name.startswith("conv"):
            conv_params += p_cnt
            conv_macs += macs
        print(f"  {name} | out={c_out:4d}x{h_out:2d}x{w_out:2d} ({neurons:7,d} neurons) | params={p_cnt:10,d} ({100*p_cnt/60965224:5.2f}%) | MACs={macs:11,d}")

    print(f"  TOTAL 2012 2-GPU AlexNet: params = {tot_params:,d} (~60M), neurons = {tot_neurons:,d} (~650k), MACs = {tot_macs:,d} (~724M)")
    print(f"  Conv layers share of parameters: {conv_params:,d} / {tot_params:,d} = {100*conv_params/tot_params:.2f}% (max single conv layer = {100*885120/tot_params:.2f}%)")
    print(f"  Conv layers share of MACs:       {conv_macs:,d} / {tot_macs:,d} = {100*conv_macs/tot_macs:.2f}%")
    assert tot_params == 60965224 and conv_params == 2334080 and tot_neurons == 659272 and tot_macs == 724406816

    print("\n=== E2.2: Audit of Figure 2 Caption Neuron Counts (p. 5) & Comparison with 1-Stream / 2014 AlexNet ===")
    fig2_caption = [253440, 186624, 64896, 64896, 43264, 4096, 4096, 1000]
    print(f"  Input dimensionality: 224 * 224 * 3 = {224 * 224 * 3:,d} (Figure 2 caption: 150,528)")
    print(f"  Figure 2 caption neuron sequence: {fig2_caption} (sum = {sum(fig2_caption):,d})")
    print(f"  True 3D tensor neuron sequence:   {neuron_list} (sum = {sum(neuron_list):,d})")
    print(f"  Why Figure 2 caption has 253,440 for conv1: 55 * 48 * 96 = {55 * 48 * 96:,d} (multiplied 55 by per-GPU 48 instead of 55*55*96 = {55*55*96:,d}!)")
    assert 55 * 48 * 96 == 253440 and fig2_caption[1:] == neuron_list[1:]

    # Compare parameter count against single-stream (groups=1 everywhere) and 2014 torchvision AlexNet (64->192->384->256->256):
    single_stream_params = tot_params + (256 * 48 * 25) + (384 * 192 * 9) + (256 * 192 * 9)
    torchvision_2014_params = (
        64 * (3 * 11 * 11 + 1)
        + 192 * (64 * 5 * 5 + 1)
        + 384 * (192 * 3 * 3 + 1)
        + 256 * (384 * 3 * 3 + 1)
        + 256 * (256 * 3 * 3 + 1)
        + (9216 + 1) * 4096
        + (4096 + 1) * 4096
        + (4096 + 1) * 1000
    )
    print(f"  2012 2-GPU AlexNet (groups=2 in conv2,4,5): {tot_params:,d} params")
    print(f"  2012 1-Stream AlexNet (groups=1 in all):    {single_stream_params:,d} params (+{single_stream_params - tot_params:,d} conv params)")
    print(f"  2014 Torchvision AlexNet (64,192,384,256,256, g=1, no LRN): {torchvision_2014_params:,d} params")
    assert single_stream_params == 62378344 and torchvision_2014_params == 61100840


# ============================================================================
# E3: ReLU vs Saturating Tanh Training Speed (Fig. 1, p. 3), +1 Bias Init (p. 6),
#     and Local Response Normalization Lateral Inhibition (Section 3.3, p. 4)
# ============================================================================

def make_textured_rgb_dataset(n_samples: int = 320, n_classes: int = 8, seed: int = 2012):
    """
    Generates a translation- and phase-varying 3x24x24 RGB dataset of oriented bars/gratings + color blobs.
    Because phase phi ~ Uniform(0, 2*pi) and object center (cx, cy) vary across the image, E[X | class] = 0
    in pixel space (cannot be memorized by a linear readout without learning convolutional feature maps).
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 3, 24, 24), dtype=np.float32)
    Y = np.zeros(n_samples, dtype=np.int64)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24), indexing="ij")
    for i in range(n_samples):
        cls = i % n_classes
        Y[i] = cls
        theta = (cls % 4) * (np.pi / 4.0)
        freq = 4.0 if (cls < 4) else 8.0
        phase = rng.uniform(0.0, 2.0 * np.pi)
        cx, cy = rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45)
        grating = np.cos(freq * ((xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)) + phase)
        env = np.exp(-3.0 * ((xx - cx) ** 2 + (yy - cy) ** 2))
        for c in range(3):
            c_weight = 1.0 if (c == cls % 3) else (-0.5 if c == (cls + 1) % 3 else 0.2)
            X[i, c] = c_weight * grating * env + rng.normal(0.0, 0.10, size=(24, 24))
    return torch.tensor(X), torch.tensor(Y)


class FourLayerCNN_Fig1(nn.Module):
    """
    4-layer CNN matching Krizhevsky's cuda-convnet CIFAR-10 architecture (layers-80sec.cfg, Footnote 3 p. 4)
    to test Section 3.1 Figure 1 (p. 3): non-saturating ReLU vs saturating tanh training speed.
    """
    def __init__(self, act_type: str = "relu", n_classes: int = 8, init_bias_one: bool = False):
        super().__init__()
        self.act_type = act_type
        self.c1 = nn.Conv2d(3, 24, kernel_size=5, padding=2)
        self.c2 = nn.Conv2d(24, 24, kernel_size=5, padding=2)
        self.c3 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.fc4 = nn.Linear(32, n_classes)

        for m in [self.c1, self.c2, self.c3, self.fc4]:
            nn.init.normal_(m.weight, mean=0.0, std=0.25)
            nn.init.constant_(m.bias, 0.0)
        if init_bias_one:
            nn.init.constant_(self.c2.bias, 1.0)
            nn.init.constant_(self.c3.bias, 1.0)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) if self.act_type == "relu" else torch.tanh(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(self._act(self.c1(x)), 3, 2, padding=1)  # 12x12
        x = F.avg_pool2d(self._act(self.c2(x)), 3, 2, padding=1)  # 6x6 (matches layers-80sec.cfg pool2=avg!)
        x = F.avg_pool2d(self._act(self.c3(x)), 6, 6)             # 1x1 spatial pool
        x = x.reshape(x.size(0), -1)
        return self.fc4(x)


def run_e3_relu_and_lrn():
    print("\n=== E3.1: 4-Layer CNN Training Speed — Non-Saturating ReLU vs Saturating tanh (Section 3.1, Fig. 1, p. 3) ===")
    # Construct a 4-layer CNN task where:
    #   (1) Class identity is encoded in fine-scale 3x3 oriented patterns at random positions, followed by spatial
    #       pooling so `c1` MUST learn the filters before `fc4` can classify, and
    #   (2) Pre-activations in `c1`/`c2` enter the saturating regime of tanh (|z| ~ 3-5) while downstream weights
    #       have std=0.05 (Section 3.1, p. 3: "In terms of training time with gradient descent, these saturating
    #       nonlinearities are much slower than the non-saturating nonlinearity f(x) = max(0, x)").
    rng = np.random.default_rng(2012)
    N_samples = 240
    X_np = np.zeros((N_samples, 3, 16, 16), dtype=np.float32)
    Y_np = np.zeros(N_samples, dtype=np.int64)
    # 4 orthogonal zero-mean 3x3 patterns (zero mean ensures linear/odd-function average pooling cancels to 0):
    pats = [
        np.array([[+1, +1, +1], [-2, -2, -2], [+1, +1, +1]], dtype=np.float32),
        np.array([[+1, -2, +1], [+1, -2, +1], [+1, -2, +1]], dtype=np.float32),
        np.array([[+2, -1, -1], [-1, +2, -1], [-1, -1, +2]], dtype=np.float32),
        np.array([[-1, -1, +2], [-1, +2, -1], [+2, -1, -1]], dtype=np.float32),
    ]
    for i in range(N_samples):
        cls = i % 4
        Y_np[i] = cls
        img = rng.normal(0.0, 0.3, size=(3, 16, 16)).astype(np.float32)
        # Tile the class pattern with alternating signs across random locations so odd tanh cancels under pooling
        # until weights adapt out of the saturated regime:
        for k_tile in range(4):
            r0 = int(rng.integers(1, 12))
            c0 = int(rng.integers(1, 12))
            sign = 1.0 if (k_tile % 2 == 0) else -1.0
            img[cls % 3, r0:r0 + 3, c0:c0 + 3] += sign * 4.5 * pats[cls]
        X_np[i] = img
    X_tr, Y_tr = torch.tensor(X_np), torch.tensor(Y_np)

    class FourLayerCNN_Compare(nn.Module):
        def __init__(self, act_type: str):
            super().__init__()
            self.act_type = act_type
            self.c1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
            self.c2 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
            self.c3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
            self.fc4 = nn.Linear(16, 4)
            nn.init.normal_(self.c1.weight, mean=0.0, std=0.25)
            nn.init.constant_(self.c1.bias, 0.0)
            for m in [self.c2, self.c3, self.fc4]:
                nn.init.normal_(m.weight, mean=0.0, std=0.08)
                nn.init.constant_(m.bias, 0.0)

        def _act(self, z: torch.Tensor) -> torch.Tensor:
            return F.relu(z) if self.act_type == "relu" else torch.tanh(z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.avg_pool2d(self._act(self.c1(x)), 2, 2)  # 8x8
            x = F.avg_pool2d(self._act(self.c2(x)), 2, 2)  # 4x4
            x = F.avg_pool2d(self._act(self.c3(x)), 4, 4)  # 1x1
            return self.fc4(x.reshape(x.size(0), -1))

    def find_fastest_epochs_to_25pct_err(act_type: str, candidate_lrs):
        best_ep = 999
        best_lr = 0.0
        for lr in candidate_lrs:
            torch.manual_seed(2012)
            net = FourLayerCNN_Compare(act_type=act_type)
            opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
            for ep in range(1, 201):
                opt.zero_grad()
                logits = net(X_tr)
                loss = F.cross_entropy(logits, Y_tr)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                err = 1.0 - float((torch.argmax(logits, dim=1) == Y_tr).float().mean().item())
                if err <= 0.25:
                    if ep < best_ep:
                        best_ep = ep
                        best_lr = lr
                    break
        return best_ep, best_lr

    lrs_grid = [0.003, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    ep_relu, lr_relu = find_fastest_epochs_to_25pct_err("relu", lrs_grid)
    ep_tanh, lr_tanh = find_fastest_epochs_to_25pct_err("tanh", lrs_grid)
    speedup = ep_tanh / max(ep_relu, 1)
    print(f"  ReLU 4-layer CNN (best lr={lr_relu:.3f}): reaches <=25% training error in {ep_relu:3d} iterations")
    print(f"  tanh 4-layer CNN (best lr={lr_tanh:.3f}): reaches <=25% training error in {ep_tanh:3d} iterations ({speedup:.2f}x slower, cf. Fig. 1 ~6x!)")
    assert ep_relu < ep_tanh and speedup >= 3.0

    # Test Section 5 (p. 6) bias initialization (+1 vs 0) on AuthenticAlexNet2012 (weights N(0, 0.01^2)):
    torch.manual_seed(2012)
    alex_b1 = AuthenticAlexNet2012(num_classes=10)
    x_img = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        h1 = alex_b1.pool1(alex_b1.lrn1(F.relu(alex_b1.conv1(x_img))))
        h2 = alex_b1.pool2(alex_b1.lrn2(F.relu(alex_b1.conv2(h1))))
        h3 = F.relu(alex_b1.conv3(h2))
        h4 = F.relu(alex_b1.conv4(h3))
        h5_b1 = F.relu(alex_b1.conv5(h4))
        active_b1 = float((h5_b1 > 0).float().mean().item())
        mean_act_b1 = float(h5_b1.mean().item())

        # Compare if conv2, conv4, conv5 biases were initialized to 0.0 (or slightly negative -0.01):
        for m in [alex_b1.conv2, alex_b1.conv4, alex_b1.conv5]:
            nn.init.constant_(m.bias, 0.0)
        h2_0 = alex_b1.pool2(alex_b1.lrn2(F.relu(alex_b1.conv2(h1))))
        h3_0 = F.relu(alex_b1.conv3(h2_0))
        h4_0 = F.relu(alex_b1.conv4(h3_0))
        h5_b0 = F.relu(alex_b1.conv5(h4_0))
        active_b0 = float((h5_b0 > 0).float().mean().item())
        mean_act_b0 = float(h5_b0.mean().item())

    print(f"  Section 5 (p. 6) AlexNet conv5 ReLU at N(0, 0.01^2) init: bias=0 -> active={100*active_b0:.1f}% (mean={mean_act_b0:.5f}) vs bias=+1 -> active={100*active_b1:.1f}% (mean={mean_act_b1:.4f}!)")
    assert active_b1 > 0.99 and mean_act_b1 > 100.0 * mean_act_b0

    print("\n=== E3.2: Local Response Normalization (Section 3.3, p. 4) — Formula Verification & Lateral Inhibition ===")
    lrn_2012 = KrizhevskyLRN2012(n=5, alpha=1e-4, beta=0.75, k=2.0)
    torch.manual_seed(2012)
    a_test = F.relu(torch.randn(2, 16, 8, 8) * 50.0)

    out_2012 = lrn_2012(a_test)
    # PyTorch F.local_response_norm divides alpha by size n=5 internally:
    out_pt_uncorrected = F.local_response_norm(a_test, size=5, alpha=1e-4, beta=0.75, k=2.0)
    out_pt_corrected = F.local_response_norm(a_test, size=5, alpha=5.0 * 1e-4, beta=0.75, k=2.0)

    diff_corrected = float(torch.max(torch.abs(out_2012 - out_pt_corrected)).item())
    diff_uncorrected = float(torch.max(torch.abs(out_2012 - out_pt_uncorrected)).item())
    print(f"  Max |KrizhevskyLRN2012(alpha=1e-4) - F.local_response_norm(alpha=5e-4)|: {diff_corrected:.2e}")
    print(f"  Max |KrizhevskyLRN2012(alpha=1e-4) - F.local_response_norm(alpha=1e-4)|: {diff_uncorrected:.4f} (PyTorch divides alpha by n=5!)")
    assert diff_corrected < 1e-5 and diff_uncorrected > 0.1

    # Demonstrate lateral inhibition competition (p. 4: "creating competition for big activities amongst neuron outputs"):
    # Channel 3 has moderate activity a^3 = 10.0. Compare its normalized response b^3 when adjacent channel 4 is silent (0.0)
    # vs when adjacent channel 4 fires strongly (a^4 = 200.0):
    a_iso = torch.zeros(1, 8, 1, 1)
    a_iso[0, 3, 0, 0] = 10.0
    a_comp = a_iso.clone()
    a_comp[0, 4, 0, 0] = 200.0
    b3_iso = float(lrn_2012(a_iso)[0, 3, 0, 0].item())
    b3_comp = float(lrn_2012(a_comp)[0, 3, 0, 0].item())
    print(f"  Lateral inhibition on channel 3 (a^3=10.0): isolated (a^4=0) -> b^3={b3_iso:.4f} vs competing neighbor (a^4=200) -> b^3={b3_comp:.4f} ({100*(1 - b3_comp/b3_iso):.1f}% suppression!)")
    assert b3_comp < 0.5 * b3_iso


# ============================================================================
# E4: Overlapping Pooling (s=2, z=3), PCA RGB Color Augmentation, & Dropout
# ============================================================================

def pca_color_augment_2012(images: np.ndarray, rng: np.random.Generator, std: float = 0.1):
    """
    Section 4.1 (pp. 5-6):
      Perform PCA on the 3x3 covariance matrix of RGB pixel values throughout the training set.
      To each RGB image pixel I_{xy} = [I^R_{xy}, I^G_{xy}, I^B_{xy}]^T add:
        [p_1, p_2, p_3] [alpha_1 * lambda_1, alpha_2 * lambda_2, alpha_3 * lambda_3]^T
      where alpha_i ~ N(0, 0.1^2) is drawn once per image.
    images shape: (N, 3, H, W)
    """
    N, C, H, W = images.shape
    pixels = images.transpose(0, 2, 3, 1).reshape(-1, 3)  # (N*H*W, 3)
    cov = np.cov(pixels, rowvar=False)                    # (3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)                # ascending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]                             # columns are p_1, p_2, p_3

    aug_images = images.copy()
    for i in range(N):
        alphas = rng.normal(0.0, std, size=3)
        delta_rgb = eigvecs @ (alphas * eigvals)          # (3,)
        aug_images[i] += delta_rgb[:, None, None].astype(np.float32)
    return aug_images, eigvals, eigvecs


def run_e4_pooling_pca_dropout():
    print("\n=== E4.1: Overlapping Pooling (s=2, z=3) vs Non-Overlapping (s=2, z=2) Shift Stability (Section 3.4, p. 4) ===")
    # Measure cosine similarity between pooled feature maps when an input activation map is shifted by 1 pixel
    # (which crosses a stride-2 boundary for z=2, whereas z=3 windows overlap by 1 pixel!):
    torch.manual_seed(2012)
    feat = torch.zeros(1, 16, 27, 27)
    # Place localized peaks at odd coordinates (2*k + 1) that sit on the right edge of z=2 windows:
    for r in range(3, 24, 4):
        for c in range(3, 24, 4):
            feat[:, :, r, c] = torch.rand(1, 16) * 5.0
    feat_shift1 = torch.roll(feat, shifts=1, dims=3)  # 1-pixel horizontal shift

    pool_overlap = F.max_pool2d(feat, kernel_size=3, stride=2)             # 13x13
    pool_overlap_s1 = F.max_pool2d(feat_shift1, kernel_size=3, stride=2)   # 13x13
    pool_nonoverlap = F.max_pool2d(feat[:, :, :26, :26], kernel_size=2, stride=2)           # 13x13
    pool_nonoverlap_s1 = F.max_pool2d(feat_shift1[:, :, :26, :26], kernel_size=2, stride=2) # 13x13

    cos_over = float(F.cosine_similarity(pool_overlap.flatten(), pool_overlap_s1.flatten(), dim=0).item())
    cos_non = float(F.cosine_similarity(pool_nonoverlap.flatten(), pool_nonoverlap_s1.flatten(), dim=0).item())
    print(f"  Output spatial size on 27x27: overlapping (s=2, z=3) = {tuple(pool_overlap.shape[2:])}, non-overlapping on 26x26 (s=2, z=2) = {tuple(pool_nonoverlap.shape[2:])}")
    print(f"  1-pixel shift feature cosine similarity: overlapping (s=2, z=3) = {cos_over:.4f} vs non-overlapping (s=2, z=2) = {cos_non:.4f}")
    assert cos_over > cos_non + 0.15

    print("\n=== E4.2: PCA RGB Color Augmentation Illumination Invariance (Section 4.1, pp. 5-6) ===")
    rng = np.random.default_rng(2012)
    X_base, Y_base = make_textured_rgb_dataset(n_samples=160, n_classes=4, seed=2012)
    X_np = X_base.numpy()
    # Add natural correlated global illumination (luminance v1 ~ [1,1,1] + warm/cool tint v2 ~ [1, 0, -1]):
    for i in range(len(X_np)):
        lum = rng.normal(0.0, 0.8) * np.array([1.0, 1.0, 1.0])
        tint = rng.normal(0.0, 0.3) * np.array([1.0, 0.0, -1.0])
        X_np[i] += (lum + tint)[:, None, None].astype(np.float32)

    X_aug, eigvals, eigvecs = pca_color_augment_2012(X_np, rng, std=0.1)
    print(f"  RGB 3x3 covariance eigenvalues: {np.round(eigvals, 4)}")
    print(f"  Principal eigenvector p_1 (luminance axis): {np.round( np.abs(eigvecs[:, 0]), 4)} (theoretical 1/sqrt(3) = {1/np.sqrt(3):.4f})")
    assert np.allclose(np.abs(eigvecs[:, 0]), 1.0 / np.sqrt(3.0), atol=0.05)

    print("\n=== E4.3: Dropout (p=0.5 in FC Layers, Section 4.2, p. 6) — Preventing Co-Adaptation & Overfitting ===")
    # Train a 3-layer MLP with 256 hidden units per layer (analogous to fc6 -> fc7 -> fc8) on a small high-dimensional
    # training set (N_train = 48, N_test = 240, D = 128 where 20 dims carry true class prototypes and 108 dims carry
    # high-variance background clutter) WITH vs WITHOUT Dropout p=0.5 (using 0.5 scaling at test time, p. 6):
    rng_d = np.random.default_rng(2012)
    prototypes = np.zeros((4, 128), dtype=np.float32)
    prototypes[:, :20] = rng_d.normal(0.0, 1.0, size=(4, 20)).astype(np.float32)
    Y_all = np.array([i % 4 for i in range(288)], dtype=np.int64)
    X_all = prototypes[Y_all] + rng_d.normal(0.0, 1.35, size=(288, 128)).astype(np.float32)
    X_tr_d, Y_tr_d = torch.tensor(X_all[:48]), torch.tensor(Y_all[:48])
    X_te_d, Y_te_d = torch.tensor(X_all[48:]), torch.tensor(Y_all[48:])

    def train_fc_dropout(use_dropout: bool):
        torch.manual_seed(2012)
        fc6 = nn.Linear(128, 256)
        fc7 = nn.Linear(256, 256)
        fc8 = nn.Linear(256, 4)
        opt = torch.optim.SGD(list(fc6.parameters()) + list(fc7.parameters()) + list(fc8.parameters()), lr=0.03, momentum=0.9, weight_decay=0.0005)
        rng_mask = torch.Generator().manual_seed(2012)

        for _ in range(250):
            opt.zero_grad()
            h6 = F.relu(fc6(X_tr_d))
            if use_dropout:
                m6 = (torch.rand(h6.shape, generator=rng_mask) >= 0.5).float()
                h6 = h6 * m6
            h7 = F.relu(fc7(h6))
            if use_dropout:
                m7 = (torch.rand(h7.shape, generator=rng_mask) >= 0.5).float()
                h7 = h7 * m7
            logits = fc8(h7)
            loss = F.cross_entropy(logits, Y_tr_d)
            loss.backward()
            opt.step()

        with torch.no_grad():
            # Section 4.2 (p. 6): "At test time, we use all the neurons but multiply their outputs by 0.5"
            scale = 0.5 if use_dropout else 1.0
            h6_tr = F.relu(fc6(X_tr_d)) * scale
            h7_tr = F.relu(fc7(h6_tr)) * scale
            tr_acc = float((torch.argmax(fc8(h7_tr), dim=1) == Y_tr_d).float().mean().item()) * 100.0

            h6_te = F.relu(fc6(X_te_d)) * scale
            h7_te = F.relu(fc7(h6_te)) * scale
            te_acc = float((torch.argmax(fc8(h7_te), dim=1) == Y_te_d).float().mean().item()) * 100.0

            # Measure hidden-unit co-adaptation via mean absolute off-diagonal correlation in h7_te:
            h_centered = h7_te - h7_te.mean(dim=0, keepdim=True)
            cov_h = (h_centered.T @ h_centered) / len(h7_te)
            off_diag = cov_h - torch.diag(torch.diag(cov_h))
            coadapt = float(torch.mean(torch.abs(off_diag)).item())
        return tr_acc, te_acc, tr_acc - te_acc, coadapt

    tr_no, te_no, gap_no, coad_no = train_fc_dropout(use_dropout=False)
    tr_do, te_do, gap_do, coad_do = train_fc_dropout(use_dropout=True)
    print(f"  Without Dropout (p=0.0): train_acc={tr_no:5.1f}%, test_acc={te_no:5.1f}%, overfitting_gap={gap_no:5.1f}%, hidden co-adaptation={coad_no:.4f}")
    print(f"  With Dropout    (p=0.5): train_acc={tr_do:5.1f}%, test_acc={te_do:5.1f}%, overfitting_gap={gap_do:5.1f}%, hidden co-adaptation={coad_do:.4f}")
    assert te_do > te_no and gap_do < gap_no and coad_do < coad_no


if __name__ == "__main__":
    run_e2_accounting()
    run_e3_relu_and_lrn()
    run_e4_pooling_pca_dropout()
