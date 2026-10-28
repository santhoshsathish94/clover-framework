# Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton (2012) — *ImageNet Classification with Deep Convolutional Neural Networks* — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh.

## 1. Status

- **E1 (Clone & inspect reference repositories on k3 CPU + compare with `torchvision.models.alexnet` and PyTorch `nn.LocalResponseNorm`)**: DONE
- **E2 (Exact 2012 Two-GPU AlexNet Architecture Parameter, Neuron, & MAC Accounting `60,965,224` params / `659,272` neurons / `724M` MACs, plus Figure 2 caption `253,440` audit)**: DONE
- **E3 (ReLU vs saturating $\tanh$ $6\times$ training speedup Fig. 1 p. 3, $+1$ bias initialization p. 6, and Local Response Normalization lateral inhibition p. 4)**: DONE
- **E4 (Overlapping pooling $s=2, z=3$ vs $s=2, z=2$ p. 4, PCA RGB illumination augmentation pp. 5–6, and Dropout $p=0.5$ co-adaptation & generalization p. 6)**: DONE

## 2. What this is

Authors say (Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", *Advances in Neural Information Processing Systems 25 (NIPS 2012)*, pp. 1097–1105):
- **The Problem & Result (Sections 1–2, 6, pp. 1–2, 7)**: Trained an 8-layer deep convolutional neural network (`60 million parameters` and `650,000 neurons`) on 1.2 million $256 \times 256$ RGB images across 1,000 categories in ILSVRC-2010 and ILSVRC-2012, reducing ILSVRC-2010 test error to `37.5%` top-1 / `17.0%` top-5 (vs `45.7%` / `25.7%` for SIFT + Fisher Vectors) and winning ILSVRC-2012 with `15.3%` top-5 test error (vs `26.2%` second-best).
- **Architectural Mechanisms (Section 3, Figs. 1–2, pp. 2–5)**:
  1. **ReLU Nonlinearity (Section 3.1, Fig. 1, p. 3)**: Non-saturating $f(x) = \max(0, x)$ trains **six times faster** than saturating $\tanh(x)$ on a 4-layer CNN on CIFAR-10.
  2. **Two-GPU Split-Stream Topology (Section 3.2 & 3.5, Fig. 2, pp. 3–5)**: Spreads half the kernels onto each GTX 580 3GB GPU with cross-GPU communication only in `conv3` and `fc6–fc8`:
     - `conv1`: 96 kernels of $11 \times 11 \times 3$, stride 4 (`groups=1`) $\to$ LRN $\to$ MaxPool ($s=2, z=3$)
     - `conv2`: 256 kernels of $5 \times 5 \times 48$ (`groups=2`, 128 per GPU) $\to$ LRN $\to$ MaxPool ($s=2, z=3$)
     - `conv3`: 384 kernels of $3 \times 3 \times 256$ (`groups=1`, cross-GPU!)
     - `conv4`: 384 kernels of $3 \times 3 \times 192$ (`groups=2`, intra-GPU)
     - `conv5`: 256 kernels of $3 \times 3 \times 192$ (`groups=2`, intra-GPU) $\to$ MaxPool ($s=2, z=3$)
     - `fc6`: 4,096 neurons (Dropout $p=0.5$) $\to$ `fc7`: 4,096 neurons (Dropout $p=0.5$) $\to$ `fc8`: 1,000-way softmax.
  3. **Local Response Normalization (Section 3.3, p. 4)**:
     $$b^i_{x,y} = a^i_{x,y} \left( k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a^j_{x,y})^2 \right)^{-\beta}, \qquad k=2, \; n=5, \; \alpha=10^{-4}, \; \beta=0.75$$
  4. **Overlapping Pooling (Section 3.4, p. 4)**: Pooling neighborhoods of size $z \times z = 3 \times 3$ with stride $s = 2$ ($s < z$).
- **Overfitting Prevention & Learning Details (Sections 4–5, pp. 5–6)**:
  - **10-crop translation/reflection augmentation** ($224 \times 224$ patches from $256 \times 256$) and **PCA RGB color augmentation** adding $[p_1, p_2, p_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$ where $\alpha_i \sim \mathcal{N}(0, 0.1^2)$.
  - **Dropout ($p=0.5$, Section 4.2, p. 6)** in `fc6` and `fc7`, multiplying outputs by $0.5$ at test time.
  - **SGD with momentum $0.9$, weight decay $0.0005$**, $\mathcal{N}(0, 0.01^2)$ weight init, and **$+1$ bias initialization** in `conv2, conv4, conv5, fc6, fc7` to provide positive initial inputs to ReLUs.

## 3. Experiments

### E1: Clone and inspect reference repositories and compare with `torchvision.models.alexnet`
- **Question**: How do open-source AlexNet repositories and `torchvision.models.alexnet` differ from Krizhevsky, Sutskever & Hinton (2012)'s exact two-GPU architecture and Local Response Normalization formula?
- **Prediction** (logged in `work/predictions.txt` at 2026-10-26 18:40 UTC):
  - `torchvision.models.alexnet` implements Krizhevsky (2014)'s single-stream `64->192->384->256->256` network without LRN (`61,100,840` parameters) rather than the 2012 `96->256->384->384->256` 2-stream LRN network (`60,965,224` parameters).
  - Repositories using PyTorch's `nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)` unknowingly divide `alpha` by `size=5`.
- **Run**: `scripts/e1_repos.sh` and `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - Cloned five repositories in `/root/dm-study/foundations/krizhevsky-2012/repo/`:
    1. `ulrichstern/cuda-convnet` (`3113628`): Alex Krizhevsky's original 2012 C++/CUDA source code exported from `http://code.google.com/p/cuda-convnet/` (cited in Footnotes 1 & 3 on pp. 2, 4). Inspecting `example-layers/layers-19pct.cfg`, `layers-18pct.cfg`, `layers-80sec.cfg`, `include/neuron.cuh`, and `layer.py` revealed the exact 4-layer CIFAR-10 CNN architecture (`conv1` $5\times 5$, 32 filters, `max_pool` $z=3, s=2$; `conv2` $5\times 5$, 32 filters, `avg_pool` $z=3, s=2$; `conv3` $5\times 5$, 64 filters, `avg_pool` $z=3, s=2$; `fc10`), its per-layer initial weight stds (`initW=0.0001, 0.01, 0.01, 0.01`), and the `TanhNeuron` (`f(x) = a * tanh(b * x)`), `ReluNeuron`, and `CrossMapResponseNormLayer` CUDA kernels.
    2. `Brilland-baba/alexnet-from-scratch` (`0078a57`): Implemented Krizhevsky (2012)'s exact `LRN` formula (`F.avg_pool1d(sq, n, stride=1, padding=n//2) * self.n`, multiplying by `n` to cancel `avg_pool1d`'s $1/n$ division!), but omitted the 2-GPU `groups=2` split in `conv2, conv4, conv5` and used `he_init` with zero biases instead of Section 5's $\mathcal{N}(0, 0.01^2)$ + $+1$ bias initialization.
    3. `matteo-dunnhofer/imagenet` (`3b88897`): Implemented `bcnn1=0, bcnn2=1, bcnn3=0, bcnn4=1, bcnn5=1` (matching Section 5 for `conv1..5`, though setting `bfc1=bfc2=0` instead of $1$), omitted the `groups=2` split in `conv2, conv4, conv5` (`[5, 5, 96, 256]`), and called `tf.nn.lrn(..., bias=1.0, alpha=2e-05, beta=0.75)` (setting `bias=1.0` instead of $k=2.0$ and dividing $\alpha=10^{-4}$ by $5$).
    4. `jcwchen/tensorflow_alexnet_classification` (`857d0c2`) & `animikhaich/AlexNet-Tensorflow` (`8a616ce`): Resized inputs to $227 \times 227$ with `VALID` padding instead of $224 \times 224$ with `padding=2` in `conv1`.

### E2: Exact 2012 Two-GPU AlexNet Parameter, Neuron, & MAC Accounting (`60,965,224` params / `659,272` neurons / `724M` MACs) and Figure 2 Caption Audit
- **Question**:
  1. What are the exact parameter, neuron, and multiply-accumulate (MAC) counts of each of the 8 learned layers in Krizhevsky, Sutskever & Hinton (2012)'s two-GPU architecture (`AuthenticAlexNet2012`), and how do they compare against a single-stream (`groups=1`) version and Krizhevsky (2014)'s `torchvision.models.alexnet`?
  2. Why does the caption of Figure 2 (p. 5) state that the number of neurons in the 8 layers is `253,440–186,624–64,896–64,896–43,264–4096–4096–1000`?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Total parameters $= 60,965,224$, of which `conv1..conv5` contain only `2,334,080` (`3.83%`), while performing `665,784,864` / `724,406,816` (`91.91%`) of all MACs.
  - True neuron counts sum to `659,272` (`290,400` in `conv1`), and Figure 2's `253,440` for `conv1` comes from `55 * 48 * 96 = 253,440` (multiplying one spatial dimension `55` by the per-GPU channel count `48` instead of `55 * 55 * 96 = 290,400`).
- **Run**: `work/krizhevsky2012_experiments.py` (`run_e2_accounting()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E2.1 (Exact Layer-by-Layer Accounting, Section 3.5 & Fig. 2, pp. 4–5)**:
    | Layer | Kernel / Stride / Groups | Output Tensor | Neurons | Trainable Parameters | % of Params | Multiply-Accumulates (MACs) |
    | :--- | :---: | :---: | ---: | ---: | ---: | ---: |
    | `conv1` | $11\times 11, s=4, g=1$ | $96 \times 55 \times 55$ | `290,400` | `34,944` | `0.06%` | `105,415,200` |
    | `conv2` | $5\times 5, s=1, g=2$ | $256 \times 27 \times 27$ | `186,624` | `307,456` | `0.50%` | `223,948,800` |
    | `conv3` | $3\times 3, s=1, g=1$ | $384 \times 13 \times 13$ | `64,896` | `885,120` | `1.45%` | `149,520,384` |
    | `conv4` | $3\times 3, s=1, g=2$ | $384 \times 13 \times 13$ | `64,896` | `663,936` | `1.09%` | `112,140,288` |
    | `conv5` | $3\times 3, s=1, g=2$ | $256 \times 13 \times 13$ | `43,264` | `442,624` | `0.73%` | `74,760,192` |
    | `fc6` | $9216 \to 4096$ | $4096 \times 1 \times 1$ | `4,096` | `37,752,832` | **`61.93%`** | `37,748,736` |
    | `fc7` | $4096 \to 4096$ | $4096 \times 1 \times 1$ | `4,096` | `16,781,312` | **`27.53%`** | `16,777,216` |
    | `fc8` | $4096 \to 1000$ | $1000 \times 1 \times 1$ | `1,000` | `4,097,000` | `6.72%` | `4,096,000` |
    | **TOTAL** | — | — | **`659,272`** | **`60,965,224`** | `100.00%` | **`724,406,816`** |
    All 5 convolutional layers combined contain only **`3.83%`** of the parameters (`2,334,080 / 60,965,224`), yet perform **`91.91%`** of all multiply-accumulates (`665,784,864 / 724,406,816`).
  - **E2.2 (Figure 2 Caption Neuron Audit & Variant Comparison)**:
    - The input dimensionality is $224 \times 224 \times 3 = \mathbf{150,528}$ (exact match to Figure 2 caption).
    - Every layer's neuron count in Figure 2's caption (`[253440, 186624, 64896, 64896, 43264, 4096, 4096, 1000]`) matches the true 3D tensor neuron count except the first (`253,440` vs `290,400`), because $55 \times 48 \times 96 = \mathbf{253,440}$ (multiplying one spatial dimension `55` by the per-GPU slice `48` instead of $55 \times 55 \times 96 = 290,400$).
    - Removing the 2-GPU split (`groups=1` in all conv layers) adds `+1,413,120` parameters (`62,378,344` total); Krizhevsky (2014)'s `torchvision` variant (`64, 192, 384, 256, 256` with `groups=1` and no LRN) has `61,100,840` parameters.

### E3: ReLU vs Saturating $\tanh$ Training Speed (Section 3.1, Fig. 1, p. 3), Section 5 $+1$ Bias Init (p. 6), and Local Response Normalization (Section 3.3, p. 4)
- **Question**:
  1. On a 4-layer CNN with independently tuned optimal learning rates and no regularization (Figure 1, p. 3), how much faster does non-saturating ReLU reach $\le 25\%$ training error than saturating $\tanh$, and how does Section 5's $+1$ bias initialization affect deep ReLU signal propagation at $\mathcal{N}(0, 0.01^2)$ weight initialization?
  2. How does Krizhevsky (2012)'s LRN formula compare against PyTorch's `F.local_response_norm`, and how strongly does LRN suppress a channel when an adjacent channel fires strongly (lateral inhibition)?
- **Prediction** (logged in `work/predictions.txt` before running):
  - ReLU reaches $\le 25\%$ training error $> 3\times$ faster than $\tanh$ at their respective best learning rates.
  - At $\mathcal{N}(0, 0.01^2)$ weight initialization, $+1$ bias initialization keeps $> 99\%$ of `conv5` ReLUs active with $> 100\times$ larger mean activation than $0$ bias initialization.
  - `KrizhevskyLRN2012(alpha=1e-4)` matches `F.local_response_norm(alpha=5e-4)` to $< 10^{-5}$, whereas `F.local_response_norm(alpha=1e-4)` differs because PyTorch divides `alpha` by `size=5`.
- **Run**: `work/krizhevsky2012_experiments.py` (`run_e3_relu_and_lrn()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E3.1 (ReLU vs $\tanh$ 4-Layer CNN Speedup & Section 5 $+1$ Bias Init)**:
    - With identical zero-bias initialization and independently selected optimal learning rates across `lrs_grid`, the ReLU 4-layer CNN (`best lr = 0.300`) reached $\le 25\%$ training error in **`10` iterations**, whereas the $\tanh$ 4-layer CNN (`best lr = 0.200`) required **`78` iterations** — a **`7.80x` speedup** (cf. Figure 1's $\sim 6\times$ on CIFAR-10, p. 3).
    - In `AuthenticAlexNet2012` with $\mathcal{N}(0, 0.01^2)$ weights (Section 5, p. 6), setting `conv2, conv4, conv5` biases to $0.0$ causes signal attenuation across the 5 layers so `conv5` mean activation is only **`0.00059`** (`54.7%` active), whereas Krizhevsky's $+1.0$ bias initialization keeps **`100.0%`** of `conv5` ReLUs active with mean activation **`1.0314`** ($1,748\times$ stronger initial signal!).
  - **E3.2 (Local Response Normalization Verification & Lateral Inhibition, Section 3.3, p. 4)**:
    - $\max |\text{KrizhevskyLRN2012}(\alpha=10^{-4}) - \text{F.local\_response\_norm}(\alpha=5\times 10^{-4})| = \mathbf{3.81 \times 10^{-6}}$, whereas calling `F.local_response_norm(alpha=1e-4)` produces a max error of **`31.9499`** because PyTorch divides `alpha` by `size=5`.
    - **Lateral inhibition**: For a channel with activity $a^3 = 10.0$, its normalized output is **`5.9238`** when adjacent channels are silent, but drops to **`2.6052`** (**`56.0%` suppression**) when adjacent channel $a^4 = 200.0$ fires strongly.

### E4: Overlapping Pooling ($s=2, z=3$, Section 3.4, p. 4), PCA RGB Illumination Augmentation (Section 4.1, pp. 5–6), and Dropout Co-Adaptation Prevention (Section 4.2, p. 6)
- **Question**:
  1. How does overlapping max-pooling ($s=2, z=3$) compare to non-overlapping max-pooling ($s=2, z=2$) in output spatial dimensions and 1-pixel shift stability?
  2. What does the $3 \times 3$ RGB pixel covariance eigensystem $[p_1, p_2, p_3], [\lambda_1, \lambda_2, \lambda_3]$ extract on images with global luminance and color-temperature variations (Section 4.1, pp. 5–6)?
  3. How does Dropout ($p=0.5$ in hidden FC layers, scaled by $0.5$ at test time, Section 4.2, p. 6) affect test accuracy, train-test overfitting gap, and hidden-unit co-adaptation?
- **Prediction** (logged in `work/predictions.txt` before running):
  - Overlapping pooling ($s=2, z=3$) maps $27 \times 27 \to 13 \times 13$ with higher 1-pixel shift cosine similarity than non-overlapping pooling ($s=2, z=2$).
  - The top eigenvector $p_1$ of the $3 \times 3$ RGB covariance matrix aligns with the luminance axis $\frac{1}{\sqrt{3}}[1, 1, 1]^T \approx [0.5774, 0.5774, 0.5774]^T$.
  - Dropout ($p=0.5$) increases test accuracy, reduces the train-test overfitting gap, and reduces hidden-unit off-diagonal covariance (co-adaptation).
- **Run**: `work/krizhevsky2012_experiments.py` (`run_e4_pooling_pca_dropout()`) via `scripts/e2_e3_e4_run.sh`
- **Observation**:
  - **E4.1 (Overlapping Pooling $s=2, z=3$ vs $s=2, z=2$, p. 4)**: Both produce `(13, 13)` output maps, but under a 1-pixel horizontal shift across stride-2 boundaries, non-overlapping $s=2, z=2$ feature cosine similarity drops to **`0.0000`**, whereas overlapping $s=2, z=3$ maintains cosine similarity **`0.7071`** because adjacent $3 \times 3$ windows share a 1-pixel border.
  - **E4.2 (PCA RGB Color Augmentation, pp. 5–6)**: The $3 \times 3$ RGB covariance matrix has eigenvalues `[2.2815, 0.2067, 0.0467]` ($90.0\%$ of variance in $\lambda_1$), and its principal eigenvector $p_1 = [0.5670, 0.5757, 0.5891]^T$ matches the theoretical luminance axis $\frac{1}{\sqrt{3}}[1, 1, 1]^T = [0.5774, 0.5774, 0.5774]^T$ to within `0.012`.
  - **E4.3 (Dropout $p=0.5$ Co-Adaptation & Overfitting Reduction, p. 6)**:
    - Without Dropout ($p=0.0$): `train_acc = 100.0%`, `test_acc = 73.3%`, `overfitting_gap = 26.7%`, hidden co-adaptation $= \mathbf{0.0146}$.
    - With Dropout ($p=0.5$, test-time $\times 0.5$ scaling): `train_acc = 100.0%`, `test_acc = 77.1%` (**`+3.8%`**), `overfitting_gap = 22.9%` (**`-3.8%`**), hidden co-adaptation $= \mathbf{0.0058}$ (**$2.52\times$ lower co-adaptation**).

## 4. Findings

1. **Exact verification of 2012 Two-GPU AlexNet (`60,965,224` parameters, `659,272` neurons, `724,406,816` MACs) and the Figure 2 caption `253,440` typo (Section 3.5 & Fig. 2, pp. 4–5)**: All five convolutional layers combined hold only `3.83%` of the network's parameters (`2,334,080`) while performing `91.91%` of its MACs (`665,784,864`), whereas `fc6..fc8` hold `96.17%` of the parameters (`58,631,144`). Figure 2's caption prints `253,440` for `conv1` because $55 \times 48 \times 96 = 253,440$ (multiplying one spatial dimension `55` by the per-GPU channel count `48` instead of $55 \times 55 \times 96 = 290,400$).
2. **Why non-saturating ReLUs train $7.8\times$ faster than $\tanh$ and why Section 5 initializes hidden biases to $+1$ (pp. 3, 6)**: Under spatial average pooling (`layers-19pct.cfg` / `layers-80sec.cfg` in `cuda-convnet`), odd-symmetric $\tanh(-z) = -\tanh(z)$ cancels positive and negative half-cycles of oriented textures and saturates in its tails ($1 - \tanh^2(z) \to 0$), requiring `78` iterations to reach $\le 25\%$ training error vs `10` iterations for ReLU (`7.80x` speedup). Furthermore, with Section 5's small $\mathcal{N}(0, 0.01^2)$ weight initialization, zero biases attenuate the forward signal by $1,748\times$ by `conv5` (`0.00059` vs `1.0314` with $+1$ biases).
3. **Local Response Normalization (`alpha` scaling difference vs PyTorch), Overlapping Pooling ($s=2, z=3$), PCA RGB Augmentation, and Dropout ($p=0.5$)**: Krizhevsky (2012)'s LRN formula multiplies the sum of $n=5$ adjacent squared activities directly by $\alpha = 10^{-4}$ (whereas PyTorch's `F.local_response_norm` divides `alpha` by `size=5`), creating $56.0\%$ lateral inhibition when a neighboring kernel fires strongly. Overlapping pooling ($s=2, z=3$) preserves 1-pixel shift similarity (`0.7071` vs `0.0000`), PCA color augmentation perturbs along the luminance and chromaticity eigenvectors of the $3 \times 3$ RGB covariance matrix, and Dropout ($p=0.5$) reduces hidden-unit co-adaptation by $2.52\times$ and improves test accuracy from `73.3%` to `77.1%`.

## 5. Surprises / bugs hit during the work

- **Why a random-feature shortcut masked the ReLU vs $\tanh$ difference on our first synthetic dataset**: In our first pass of `E3.1`, we placed fixed-phase gratings into a `24 -> 32 -> 32` max-pooled CNN followed by a `1,152`-unit linear readout `fc4` on `240` examples. Because $240 < 1,152$ and max-pooling already rectifies $\tanh$, `fc4` linearly separated the random untrained features in 3–4 steps without `c1..c3` ever learning! Reading Krizhevsky's actual `layers-19pct.cfg` and `layers-80sec.cfg` in `ulrichstern/cuda-convnet` showed that `pool2` and `pool3` are **average pooling** (`pool=avg`) feeding a compact output head: when class patterns occur at random locations/polarities, odd $\tanh$ cancels under average pooling and saturates in its tails, immediately reproducing the $7.80\times$ ReLU speedup (`10` vs `78` iterations).

## 6. READ-LOG

- `research/krizhevsky-2012/work/krizhevsky2012.txt` (lines 1–581: full 9-page 2012 NIPS paper)
- `ulrichstern/cuda-convnet`: `example-layers/layers-19pct.cfg`, `layer-params-19pct.cfg`, `layers-18pct.cfg`, `layers-80sec.cfg`, `layer-params-80sec.cfg`, `include/neuron.cuh`, `layer.py`
- `Brilland-baba/alexnet-from-scratch`: `alexnet.py`, `train.py`, `Plots.py`
- `matteo-dunnhofer/imagenet`: `tf/models/alexnet.py`
- `jcwchen/tensorflow_alexnet_classification`: `model.py`, `data.py`, `train.py`

## 7. Files produced

- `research/krizhevsky-2012/work/predictions.txt`
- `research/krizhevsky-2012/work/krizhevsky2012.txt`
- `research/krizhevsky-2012/work/krizhevsky2012_experiments.py`
- `research/krizhevsky-2012/scripts/setup_krizhevsky2012.sh`
- `research/krizhevsky-2012/scripts/e1_repos.sh`
- `research/krizhevsky-2012/scripts/e2_e3_e4_run.sh`
- `research/krizhevsky-2012/results/setup_krizhevsky2012.out`
- `research/krizhevsky-2012/results/e1_repos.out`
- `research/krizhevsky-2012/results/e2_e3_e4_run.out`
