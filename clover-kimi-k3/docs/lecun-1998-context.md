# Yann LeCun, Léon Bottou, Yoshua Bengio & Patrick Haffner (1998) — *Gradient-Based Learning Applied to Document Recognition* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & inspect reference repositories on k3 CPU: `jmlee8939/LeNet-5_1998_pytorch`, `pop123-ux/LeNet_5-from-scratch`, `VTongTV/LeNet-5-scratch`, `SunnyHaze/LeNet5-MNIST-Pytorch`)**: DONE
- **E2 (Exact LeNet-5 Layer-by-Layer Parameter & Connection Accounting `60,000` params / `340,908` connections, Table I asymmetric `S2->C3` map, Appendix A scaled $\tanh$, and Appendix C Gauss–Newton diagonal Hessian backprop)**: DONE
- **E3 (Euclidean RBF 84-unit Output Layer Collapse under MSE Eq. 8 vs Competitive MAP Eq. 9, and Space-Displacement Neural Network SDNN equivalence & MAC speedup Section VII)**: DONE
- **E4 (Graph Transformer Networks Sections IV–VIII: Viterbi Eq. 10 vs Forward Scorer Eq. 13–16 backprop check, parallel-path evidence aggregation, and unsegmented string-level Discriminative Forward training Eq. 17)**: DONE

## 2. What this is

Authors say (Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, "Gradient-Based Learning Applied to Document Recognition", *Proceedings of the IEEE*, Vol. 86, No. 11, pp. 2278–2324, November 1998):
- **Convolutional Neural Networks & LeNet-5 (Section II, Figs. 2–3, Table I, pp. 2283–2287 / PDF pp. 6–9)**:
  - Replaces hand-crafted feature extractors with three architectural invariances: **local receptive fields**, **shared weights** across feature maps, and **spatial sub-sampling** ($2 \times 2$ pooling with a trainable scale coefficient and trainable bias passed through a squashing sigmoid).
  - **Exact LeNet-5 Architecture (Fig. 2, pp. 7–9)**: Takes a $32 \times 32$ input image (normalized to background $-0.1$, foreground $+1.175$) so that stroke endpoints/corners of a $20 \times 20$ digit can appear at the center of the highest-level receptive fields:
    1. `C1`: 6 feature maps of $28 \times 28$ ($5 \times 5$ kernels): `156` trainable parameters, `122,304` connections.
    2. `S2`: 6 feature maps of $14 \times 14$ ($2 \times 2$ non-overlapping sum $\times$ trainable weight $+$ trainable bias): `12` trainable parameters, `5,880` connections.
    3. `C3`: 16 feature maps of $10 \times 10$ ($5 \times 5$ kernels connected to subsets of `S2` via **Table I**: 6 maps of 3 contiguous `S2` maps, 6 maps of 4 contiguous `S2` maps, 3 maps of 4 discontinuous `S2` maps, and 1 map of all 6 `S2` maps = 60 kernels): `1,516` trainable parameters, `151,600` connections.
    4. `S4`: 16 feature maps of $5 \times 5$ ($2 \times 2$ sub-sampling): `32` trainable parameters, `2,000` connections.
    5. `C5`: 120 feature maps of $1 \times 1$ ($5 \times 5$ kernels across all 16 `S4` maps): `48,120` trainable parameters and `48,120` connections.
    6. `F6`: 84 units fully connected to `C5`: `10,164` trainable parameters and `10,164` connections.
    7. `Output`: 10 (or 95 ASCII) **Euclidean Radial Basis Function (RBF)** units $y_i = \sum_{j=1}^{84} (x_j - w_{ij})^2$ (Eq. 7) whose parameter vectors $w_i \in \{-1, +1\}^{84}$ are stylized $7 \times 12$ character bitmaps (Fig. 3).
    - Total: **`60,000` trainable parameters** (`156 + 12 + 1,516 + 32 + 48,120 + 10,164`) and **`340,908` connections** (`340,068` through `F6` $+ 840$ RBF connections).
  - **Scaled Hyperbolic Tangent (Eq. 6 & Appendix A, p. 41)**: $f(a) = A \tanh(S a)$ with $A = 1.7159, S = \frac{2}{3}$, chosen so $f(\pm 1) = \pm 1$ and $|f''(a)|$ is maximized at $a \approx \pm 1$, keeping `F6` units unsaturated against $\pm 1$ RBF targets.
  - **Loss Function & The Collapse Problem (Section II-C, Eq. 8–9, p. 9)**: Pure MSE $E(W) = \frac{1}{P} \sum_p y_{D_p}(Z^p, W)$ (Eq. 8) collapses to zero when RBF centers are allowed to adapt; the discriminative **MAP criterion** (Eq. 9) adds a competitive log-sum-exp penalty with rubbish class $e^{-j}$:
    $$E(W) = \frac{1}{P} \sum_{p=1}^P \left( y_{D_p}(Z^p, W) + \log\left(e^{-j} + \sum_i e^{-y_i(Z^p, W)}\right) \right)$$
- **Space-Displacement Neural Networks (SDNN, Section VII, Figs. 22–26, pp. 27–30)**: Rather than running a sliding window crop-by-crop, extending the horizontal width of a convolutional network shares intermediate feature map activations across overlapping positions.
- **Graph Transformer Networks (GTNs, Sections IV–VI, VIII–X, pp. 16–40)**: Multi-module networks that communicate **directed acyclic graphs (DAGs)** carrying arc penalties $c_i$ and labels, trained globally at the word/check level without character-segmentation labels via:
  - **Viterbi Transformer (Eq. 10, p. 20)** & **Discriminative Viterbi Loss (Eq. 12, p. 24)**: $v_n = \min_{i \in U_n} (c_i + v_{s_i})$, $E_{\text{dvit}} = C_{\text{cvit}} - C_{\text{vit}}$.
  - **Forward Scorer (Eq. 13–16, p. 25)** & **Discriminative Forward Loss (Eq. 17, p. 26)**: $f_n = \text{logadd}_{i \in U_n}(c_i + f_{s_i}) = -\log \sum_{i \in U_n} e^{-(c_i + f_{s_i})}$, $E_{\text{dforw}} = C_{\text{cforw}} - C_{\text{forw}}$, with check-level confidence $\exp(-E_{\text{dforw}})$ (p. 39).

## 3. Experiments

### E1: Clone and inspect reference repositories (`jmlee8939/LeNet-5_1998_pytorch`, `pop123-ux/LeNet_5-from-scratch`, `VTongTV/LeNet-5-scratch`, `SunnyHaze/LeNet5-MNIST-Pytorch`)
- **Question**: Which reference repositories implement the authentic 1998 LeNet-5 architecture (Table I asymmetric `S2->C3` connection matrix, trainable subsampling `S2`/`S4`, $1.7159 \tanh(\frac{2}{3}a)$, 84-unit Euclidean RBF output with $7 \times 12$ bitmaps and MAP loss Eq. 9) versus simplified modern `Conv2d + Pool + Linear` approximations?
- **Prediction** (logged in `work/predictions.txt` at 2026-10-26 18:10 UTC):
  - `SunnyHaze/LeNet5-MNIST-Pytorch` will use dense `S2->C3` and standard pooling/cross-entropy, whereas `jmlee8939/LeNet-5_1998_pytorch` and `pop123-ux/LeNet_5-from-scratch` will implement the authentic Table I connection scheme, scaled $\tanh$, and Euclidean RBF output layer.
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - Cloned and inspected four reference repositories in `/root/dm-study/foundations/lecun-1998/repo/`, finding that **none of the four** simultaneously got all three 1998 architectural specifications right (`60,000` trainable parameters via Table I `S2->C3` + trainable `S2`/`S4` + $7 \times 12 = 84$ Euclidean RBF bitmaps):
    1. `pop123-ux/LeNet_5-from-scratch` (`b53f2f8`): Passed all 7 unit tests in `tests/test_model.py` (`rc=0`) and faithfully implemented the $7 \times 12 = 84$ ASCII art `DIGIT_GLYPHS` in `LeNetRBFSublayer` (`build_digit_bitmaps()`), but omitted `S2`/`S4` trainable subsampling parameters (`12 + 32 = 44`) and used a dense `6->16` `conv2` (`2,416` params instead of Table I's `1,516` params), yielding **`60,856` trainable parameters** (`60000 - 44 + 900 = 60856`) instead of the paper's `60,000`.
    2. `jmlee8939/LeNet-5_1998_pytorch` (`69931f4`): Implemented `1.7159 * torch.tanh(x * 2/3)`, trainable `Layer_S2`/`Layer_S4`, and Table I `map_combine_list()` in `Layer_C3`, **but** had a bug in `Layer_C3.forward()` (`model.py` lines 86–88): iterating `for i in range(6)` over the 6 `S2` channels added `+ self.bias[:, list_[i], :, :]` on every input channel `i`, multiplying each `C3` feature map's bias by its fan-in (`3x`, `4x`, or `6x`)!
    3. `VTongTV/LeNet-5-scratch` (`8afbe2e`): Implemented `SubsampleLayer` with trainable weights/biases in NumPy, **but** its `C3_CONNECTION_TABLE` in `config.py` connected 6 maps to 1 `S2` map, 6 maps to 2 `S2` maps, and 4 maps to 3 `S2` maps (`30` kernels instead of Table I's `60` kernels), and its `RBF_TARGETS` set digit `0` to all `-1`s and digits `1..9` to a single `+1` out of 84 (a 1-of-84 place code rather than the $7 \times 12$ stylized character bitmaps of Fig. 3).
    4. `SunnyHaze/LeNet5-MNIST-Pytorch` (`3ffa181`): Replaced scaled $\tanh$ with `ReLU`, `S2`/`S4` with `MaxPool2d`, Table I `C3` with dense `Conv2d(6, 16, 5)`, and RBF output with `Linear(84, 10)`.

### E2: Exact LeNet-5 Layer-by-Layer Parameter & Connection Accounting (`60,000` params / `340,908` connections), Appendix A Scaled $\tanh$, and Appendix C Gauss–Newton Diagonal Hessian Backprop
- **Question**:
  1. Does our `AuthenticLeNet5_1998` implementation match every single layer's trainable parameter count, connection count, and output shape in Section II-B (pp. 7–9), summing to **exactly `60,000` trainable parameters** and **`340,908` connections**?
  2. Does $f(a) = 1.7159 \tanh(\frac{2}{3} a)$ satisfy $f(\pm 1) = \pm 1$ and place the global maximum of $|f''(a)|$ at $a \approx \pm 1$ (Appendix A, p. 41)?
  3. Does the Gauss–Newton diagonal Hessian backpropagation rule $\frac{\partial^2 E^p}{\partial a_i^2} = f'(a_i)^2 \sum_k u_{ki}^2 \frac{\partial^2 E^p}{\partial a_k^2}$ (Eq. 27, p. 42) summed over shared connections $(i, j) \in V_k$ (Eq. 23, 25) match $\sum_{(i,j) \in V_k} (\frac{\partial f_{\text{out}}}{\partial u_{ij}})^2$?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Exact layer-by-layer match (`156 + 12 + 1516 + 32 + 48120 + 10164 = 60,000` trainable parameters; `122304 + 5880 + 151600 + 2000 + 48120 + 10164 + 840 = 340,908` connections).
  - $|f(\pm 1) \mp 1| < 10^{-4}$ and $\arg\max |f''(a)| = \pm \frac{3}{2} \operatorname{arctanh}(1/\sqrt{3}) = \pm 0.9877 \approx \pm 1$.
  - Max difference between Eq. 23–27 Gauss–Newton Hessian backprop and explicit squared Jacobian sum $< 10^{-15}$.
- **Run**: `work/lecun1998_experiments.py` (`run_e2_architecture_and_hessian()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E2.1 (Exact LeNet-5 Accounting, Section II-B, pp. 7–9)**:
    | Layer | Output Shape | Trainable Parameters | Connections | Paper Citation (pp. 7–8) |
    | :--- | :---: | ---: | ---: | :--- |
    | `C1` ($5 \times 5$ conv) | `(6, 28, 28)` | `156` | `122,304` | *"156 trainable parameters, and 122,304 connections"* |
    | `S2` ($2 \times 2$ subs) | `(6, 14, 14)` | `12` | `5,880` | *"12 trainable parameters and 5,880 connections"* |
    | `C3` ($5 \times 5$ Table I) | `(16, 10, 10)` | `1,516` | `151,600` | *"1,516 trainable parameters and 151,600 connections"* |
    | `S4` ($2 \times 2$ subs) | `(16, 5, 5)` | `32` | `2,000` | *"32 trainable parameters and 2,000 connections"* |
    | `C5` ($5 \times 5$ conv) | `(120, 1, 1)` | `48,120` | `48,120` | *"48,120 trainable connections"* |
    | `F6` (full) | `(84,)` | `10,164` | `10,164` | *"10,164 trainable parameters"* |
    | `RBF` (10 Euclidean units) | `(10,)` | `0` (fixed) | `840` | *"10 units, 84 inputs each"* |
    | **TOTAL** | — | **`60,000`** | **`340,908`** | *"340,908 connections, but only 60,000 trainable free parameters"* |
    Table I saves `37.3%` of `C3` parameters and connections (`1,516 / 151,600` vs `2,416 / 241,600` for dense `S2->C3`).
  - **E2.2 (Appendix A Scaled Hyperbolic Tangent, Eq. 6 & p. 41)**:
    - $f(+1.0) = +0.999997$, $f(-1.0) = -0.999997$ ($|f(1) - 1| = 2.74 \times 10^{-6}$).
    - Slope at origin $f'(0) = A S = 1.1439$; slope at $\pm 1$ is $f'(\pm 1) = 0.7554$.
    - Maximum curvature $|f''(a)|$ occurs at $a = \pm \frac{3}{2} \operatorname{arctanh}(1/\sqrt{3}) = \mathbf{\pm 0.9877}$ (matching the $\pm 1$ RBF target coordinates!).
  - **E2.3 (Appendix C Gauss–Newton Diagonal Hessian Backprop, Eq. 18–27, pp. 41–42)**:
    - Max difference between backpropagated Gauss–Newton diagonal Hessian $h_{kk}$ (Eq. 23–27) and $\sum_{(i,j) \in V_k} (\frac{\partial f_{\text{out}}}{\partial u_{ij}})^2$: **`6.51e-19`**.

### E3: Euclidean RBF Collapse under Adaptive Centers + MSE (Eq. 8) vs Competitive MAP (Eq. 9, p. 9), and Space-Displacement Neural Network (SDNN, Section VII, p. 27)
- **Question**:
  1. When the 84-dimensional Euclidean RBF centers $w_i$ are allowed to adapt alongside the network weights, what happens under pure MSE (Eq. 8) versus the competitive MAP criterion (Eq. 9, p. 9)?
  2. When `AuthenticLeNet5_1998` is applied directly as a Space-Displacement Neural Network (SDNN) to a wide $32 \times 92$ multi-digit canvas (16 stride-4 window positions), does a single convolutional pass produce the exact same $10 \times 16$ RBF penalty matrix as cropping 16 separate $32 \times 32$ windows, and how many `C1..S4` MACs does feature sharing save?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Under Adaptive RBF + pure MSE (Eq. 8), all RBF centers collapse together (center spread $\to 0$, class margin $\to 0$, accuracy $\le 25\%$), whereas under Competitive MAP (Eq. 9) the centers stay separated and reach $> 98\%$ accuracy.
  - SDNN single-pass output matches 16 sliding-window crops to float32 precision ($< 10^{-4}$), spots all 3 digits `[2, 7, 4]` at their true window positions `k = (1, 8, 15)`, and reduces `C1..S4` MACs by $> 4\times$.
- **Run**: `work/lecun1998_experiments.py` (`run_e3_rbf_and_sdnn()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E3.1 (RBF Collapse under MSE Eq. 8 vs Competitive MAP Eq. 9, p. 9)**:
    - **Adaptive RBF + Pure MSE (Eq. 8)**: Mean RBF penalty across *all* classes collapsed to **`0.00001`**, pairwise center distance collapsed to **`0.0017`**, class separation margin $\min_{c \ne y} y_c - y_y$ collapsed to **`0.00000`**, and classification accuracy under $\sigma = 0.01$ readout tolerance dropped to **`12.5%`** (random chance).
    - **Adaptive RBF + Competitive MAP (Eq. 9)**: Pairwise center distance remained **`7.4574`**, class separation margin reached **`26.86`**, and classification accuracy reached **`100.0%`**.
  - **E3.2 (Space-Displacement Neural Network SDNN Equivalence & Speedup, Section VII, Fig. 23, p. 27)**:
    - Across the $32 \times 92$ multi-digit canvas ($K = 16$ stride-4 windows), $\max |\text{SDNN\_single\_pass} - \text{Sliding\_16\_crops}| = \mathbf{1.53 \times 10^{-5}}$ (float32 1-ULP identity), spotting the exact digit classes **`[2, 7, 4]`** at window indices `k = (1, 8, 15)`.
    - Evaluating 16 independent $32 \times 32$ crops requires **`4,508,544`** `C1..S4` connections, whereas SDNN evaluates **`1,017,264`** `C1..S4` connections — a **`4.43x` MAC reduction** from spatial feature sharing!

### E4: Graph Transformer Networks (GTNs) — Forward Scorer Backprop (Eq. 15–16), Parallel-Path Evidence Aggregation (p. 25), and Unsegmented String-Level Discriminative Training (Sections V–VI, pp. 19–26)
- **Question**:
  1. Does the analytical backward recurrence through the Forward Scorer (Eq. 15–16, p. 25) match central finite differences across a multi-path DAG?
  2. When two distinct segmentations emit the same label sequence `(3, 4)` with penalties $C_1 = 2.20, C_2 = 2.20$ while a wrong single path emits `(8,)` with penalty $C_3 = 1.80$ (Section VI-C, p. 25), what do Viterbi and the Forward Scorer select?
  3. On multi-character strings over-segmented into candidate ink pieces without character-level segmentation labels, how do Non-Discriminative Viterbi (Eq. 11, Fig. 19), Discriminative Viterbi (Eq. 12, Fig. 20), and Discriminative Forward (Eq. 17, Fig. 21) compare in string accuracy, penalty collapse, and check-level confidence $\exp(-E_{\text{dforw}})$ (p. 39)?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Analytical Forward Scorer backprop (Eq. 15–16) matches central finite differences to $< 10^{-10}$.
  - Viterbi picks the wrong single path `(8,)` ($1.80 < 2.20$), whereas Forward Scorer aggregates the two parallel `(3, 4)` segmentations to $2.20 - \ln 2 = 1.5069 < 1.80$ and picks `(3, 4)`.
  - Non-Discriminative Viterbi collapses all arc penalties to $0$ (`conf < 0.15`), Discriminative Viterbi stops pushing once the margin is $0$ (`low confidence`), and Discriminative Forward builds a large margin (`conf ~ 1.0`).
- **Run**: `work/lecun1998_experiments.py` (`run_e4_gtn()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E4.1 (Forward Scorer Backprop Check, Eq. 15–16, p. 25)**: Across all 9 arcs of the multi-path DAG, $\max |\frac{\partial C_{\text{forw}}}{\partial c_i} - \text{FD}| = \mathbf{8.66 \times 10^{-11}}$.
  - **E4.2 (Parallel-Path Evidence Aggregation, p. 25)**: Viterbi picked the wrong interpretation `(8,)` (`penalty = 1.8000 < 2.2000`), whereas the Forward Scorer combined the two `(3, 4)` segmentation paths to **`1.5069`** ($= 2.20 - \ln 2 < 1.8000$) and selected `(3, 4)`.
  - **E4.3 (Unsegmented String-Level GTN Training Comparison, Sections VI-A..D & Section X-C)**:
    | GTN Training Criterion | String Accuracy | Mean Arc Penalty | Mean Check Confidence $\exp(-E_{\text{dforw}})$ (p. 39) | Behavior Observed |
    | :--- | :---: | :---: | :---: | :--- |
    | **Non-Discriminative Viterbi** (Eq. 11, p. 22) | `90/90` | **`0.00032`** | **`0.044`** ($\approx 1/27$ uniform!) | Collapses all penalties to $0$; no rejection/confidence possible |
    | **Discriminative Viterbi** (Eq. 12, p. 24) | `90/90` | `0.3201` | **`0.063`** | Zero gradient once $C_{\text{cvit}} = C_{\text{vit}}$; fails to build margin (p. 24) |
    | **Discriminative Forward** (Eq. 17, p. 26) | `90/90` | **`9.0237`** | **`1.000`** | Pushes up all competing paths & half-character fragments! |

## 4. Findings

1. **Exact verification of LeNet-5's `60,000` trainable parameters and `340,908` connections (Section II-B, pp. 7–9)**: Implementing the exact 1998 architecture (`C1=156`, `S2=12`, `C3=1,516` via Table I, `S4=32`, `C5=48,120`, `F6=10,164`, plus `840` fixed RBF connections) reproduces the paper's headline counts (`60,000` parameters and `340,908` connections) to the exact integer, whereas all four inspected GitHub repositories either omitted `S2`/`S4` parameters, used dense `C3`, had a channel-loop bias multiplication bug (`jmlee8939`), or used a 30-kernel `C3` table (`VTongTV`).
2. **Why $f(a) = 1.7159 \tanh(\frac{2}{3}a)$, $\pm 1$ RBF bitmaps, and the MAP criterion (Eq. 9) form a coupled system (pp. 8–9, 41)**: $A = 1.7159, S = \frac{2}{3}$ sets $f(\pm 1) = \pm 0.999997$ and places the maximum second derivative $|f''(a)|$ at $a = \pm 0.9877 \approx \pm 1$, keeping `F6` units unsaturated against $\pm 1$ RBF bitmap targets. When RBF centers adapt, pure MSE (Eq. 8) collapses all centers together (`spread = 0.0017`, `margin = 0.00000`, `12.5%` accuracy), whereas the competitive MAP loss (Eq. 9) maintains `spread = 7.4574`, `margin = 26.86`, and `100.0%` accuracy.
3. **Space-Displacement Neural Networks (SDNN) and Graph Transformer Networks (GTNs) enable segmentation-free recognition and end-to-end string-level training (Sections V–VIII)**: Evaluating LeNet-5 as an SDNN over a $32 \times 92$ canvas matches 16 sliding-window crops to $1.53 \times 10^{-5}$ while cutting `C1..S4` MACs by $4.43\times$. In GTNs, Discriminative Forward training ($E_{\text{dforw}} = C_{\text{cforw}} - C_{\text{forw}}$, Eq. 17) prevents the penalty collapse of Non-Discriminative Viterbi (`0.00032`, `conf = 0.044`) and the zero-margin stall of Discriminative Viterbi (`conf = 0.063`), achieving `100.0%` unsegmented string recognition with `confidence = 1.000`.

## 5. Surprises / bugs hit during the work

- **Custom Type-3 font encoding in `yann.lecun.com/exdb/publis/pdf/lecun-98.pdf`**: Direct PDF text extraction from the original 1998 PostScript-generated PDF produced control-byte glyph indices (`\x00\x02...`) because the 1998 Type-3 fonts lacked a Unicode `ToUnicode` CMap. Rendering all 46 pages at 150 DPI with PyMuPDF and running parallel Tesseract OCR (`OMP_THREAD_LIMIT=1`) across 6 CPU cores extracted all 6,202 lines (251 KB) cleanly in under 45 seconds.
- **Topological ordering in Viterbi graph dynamic programming (Eq. 10, p. 20)**: In our first run of `E4.3`, `viterbi_forward_and_bprop` iterated over `self.arcs` in edge-list order `[(0,1), (1,2), (2,3), (0,2), (1,3)]`, which processed `(2,3)` *before* `(0,2)` had updated `v[2]`, breaking 50% of 2-character segmentations (`64.4%` accuracy). Grouping incoming arcs $U_n$ by destination node $n = 1 \dots N-1$ in topological order (exact Eq. 10) immediately took accuracy to `90/90` (`100.0%`).

## 6. READ-LOG

- `research/lecun-1998/work/lecun1998.txt` (lines 1–6202: full 46-page 1998 *Proceedings of the IEEE* paper, Sections I–XI and Appendices A–C)
- `jmlee8939/LeNet-5_1998_pytorch`: `model.py`, `README.md`
- `pop123-ux/LeNet_5-from-scratch`: `src/model.py`, `tests/test_model.py`
- `VTongTV/LeNet-5-scratch`: `config.py`, `lenet/layers.py`, `lenet/network.py`
- `SunnyHaze/LeNet5-MNIST-Pytorch`: `LeNet-5_GPU.py`

## 7. Files produced

- `research/lecun-1998/work/predictions.txt`
- `research/lecun-1998/work/lecun1998.txt`
- `research/lecun-1998/work/lecun1998_experiments.py`
- `research/lecun-1998/scripts/setup_lecun1998.sh`
- `research/lecun-1998/scripts/fetch_clean_lecun98.sh`
- `research/lecun-1998/scripts/e1_repos.sh`
- `research/lecun-1998/scripts/e2_e3_e4_run.sh`
- `research/lecun-1998/results/setup_lecun1998.out`
- `research/lecun-1998/results/fetch_clean_lecun98.out`
- `research/lecun-1998/results/e1_repos.out`
- `research/lecun-1998/results/e2_e3_e4_run.out`
