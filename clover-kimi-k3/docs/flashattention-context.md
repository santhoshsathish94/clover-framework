# FlashAttention Study (Round 2) — Context File

Offered, not imposed. Direction owner: Santhosh (the creator). Date: 2026-10-26.

## 1. Status

- **E1** (Triton interpreter on CPU): **done**
- **E2** (Repository Triton attention forward on CPU): **done**
- **E3** (Repository Triton attention backward, block sizes, load/store counts): **done**
- **E4** (Algorithm 1 tiled attention, exactness, HBM traffic counter & crossover $M$): **done**
- **E5** (Online softmax step-by-step trace & negative controls): **done**
- **E6** (Backward pass with recomputation, exactness, bytes & FLOPs): **done**
- **E7** (Split-KV combine step, equal/uneven/masked chunks): **done**
- **E8** (FA1 vs FA2 loop order write counts; CUDA header tile sizes & shared memory): **done** (backward smem numbers corrected, see E8)

## 2. What this is

Authors say (FlashAttention-1, `2205.14135/src/abstract.tex`, lines 6–8): "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM."
In plain words, standard attention computes $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$, writes the $N \times N$ matrix $\mathbf{S}$ and softmax matrix $\mathbf{P}$ to slow main memory (HBM), and reads them back to multiply by $\mathbf{V}$.
FlashAttention splits $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into small blocks that fit in fast on-chip SRAM and uses an online softmax rescaling identity (tracking running row maximum $m$ and running sum of exponentials $\ell$) so $\mathbf{O} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d})\mathbf{V}$ is computed exactly in one pass without writing $\mathbf{S}$ or $\mathbf{P}$ to HBM (`2205.14135/src/algo.tex`, Section 3.1).
In the backward pass, instead of storing the $N \times N$ matrix $\mathbf{P}$ from the forward pass, it stores only the output $\mathbf{O}$ and the per-row logsumexp $L$, recomputing $\mathbf{S}_{ij}$ and $\mathbf{P}_{ij}$ tile-by-tile on-chip (`2205.14135/src/algo.tex`, lines 110–125).
Authors say (FlashAttention-2, `2307.08691/src/abstract.tex`, lines 18–24): "In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory."

## 3. Experiments

### E1: Triton interpreter installation and CPU kernel execution (`vector_add` and `tl.dot` matmul)

- **Question**: Can we install Triton into `/root/dm-study/flashattention/w2/pylib` and execute `@triton.jit` kernels (`vector_add` with `tl.load`/`tl.store` and tiled matrix multiplication with `tl.dot`) on PyTorch CPU tensors using `TRITON_INTERPRET=1`? What fails when `TRITON_INTERPRET=0` (negative control)?
- **Prediction**: Both `vector_add` (1024 elements, `float32`) and tiled `matmul` ($64 \times 64$, `BLOCK=32`, `float32`) will run on CPU tensors under `TRITON_INTERPRET=1`, matching `torch` reference (`max_abs_diff = 0.0` for `vector_add`, `< 1e-5` for `matmul`). Negative control (`TRITON_INTERPRET=0` on CPU tensor) will fail with a `RuntimeError` because no GPU driver/device is active.
- **Run**: Script [flashattn-w2-e1-triton.sh](../research/flashattention/scripts/flashattn-w2-e1-triton.sh) invoking [e1_triton_probe.py](../research/flashattention/work/e1_triton_probe.py) via `k3.ps1 -Action run -Name flashattn-w2-e1-triton -Cpus 6-13`. Output: [flashattn-w2-e1-triton.out](../research/flashattention/results/flashattn-w2-e1-triton.out).
- **Observation**:
  - Versions recorded on `k3`: Python `3.12.3`, NumPy `2.5.4`, PyTorch `2.11.0+cpu`, Triton `3.8.0` installed in `/root/dm-study/flashattention/w2/pylib` (`897 MB`).
  - First attempt at negative control via `python -c` failed before launch with `ValueError: @jit functions should be defined in a Python file` because `JITFunction.__init__` calls `inspect.getsourcelines(fn)`. Moving the negative control into `e1_triton_probe.py --neg-control` with `TRITON_INTERPRET=0` produced the expected failure: `RuntimeError: 0 active drivers ([]). There should only be one.`
  - With `TRITON_INTERPRET=1`:
    - `add_kernel` (`n=1024` and `n=1000`, `BLOCK_SIZE=256`, `float32`): `max_abs_diff = 0.000e+00`, `exact=True` on both.
    - `matmul_kernel` (`64x64x64`, `BLOCK=32`) with `tl.dot`:
      - `float32`: `max_abs_diff = 1.144e-05`, `max_rel_diff = 1.798e-04` (slightly above the `< 1e-5` prediction).
      - `float16`: `max_abs_diff = 7.629e-06`, `max_rel_diff = 1.213e-04` against `torch.matmul(a.float(), b.float())`.
      - `bfloat16`: `max_abs_diff = 8.854e+10`, `max_rel_diff = 1.448e+14` (corrupted output in Triton 3.8.0 interpreter for `bfloat16`).
- **What it showed**: Triton 3.8.0's CPU interpreter (`TRITON_INTERPRET=1`) executes `@triton.jit` kernels with `tl.load`, `tl.store`, masks, and `tl.dot` on CPU tensors in `float32` and `float16`. Two things differed from prediction: `float32` `tl.dot` had `max_abs_diff = 1.144e-05` (just over `1e-5`), and `bfloat16` produced garbage (`8.854e+10`) in the interpreter.

### E2: Repository Triton attention forward pass in the CPU interpreter

- **Question**: Which files in `Dao-AILab/flash-attention` implement Triton attention? Does unmodified `_flash_attn_forward` in `flash_attn/flash_attn_triton.py` run on CPU tensors under `TRITON_INTERPRET=1`, or what exact error does it raise? With the smallest patch in a copy under `work/`, does the real `_fwd_kernel` execute across $N \in \{64, 113, 128, 192, 256\}$, $d \in \{32, 64\}$, causal and non-causal, and match `float32`/`float64` reference attention?
- **Prediction**: Unmodified `_flash_attn_forward` fails at line 720 (`AssertionError: Only support fp16 and bf16`) on `float32` and line 721 (`assert q.is_cuda`) on `float16` CPU tensors. Relaxing only those host assertions in a copy under `work/` (zero edits to `@triton.jit def _fwd_kernel`) will allow `_fwd_kernel` to run for all shapes ($N \in \{64, 113, 128, 256\}$, $d \in \{32, 64\}$, causal and non-causal) with `max_abs_diff < 1e-4` (`float32`) and `< 2e-3` (`float16`) against `float64` reference. Negative control (causal kernel vs non-causal reference) will fail with `max_abs_diff > 0.1`.
- **Run**: Script [flashattn-w2-e2-triton-fwd.sh](../research/flashattention/scripts/flashattn-w2-e2-triton-fwd.sh) invoking [e2_triton_fwd.py](../research/flashattention/work/e2_triton_fwd.py) via `k3.ps1 -Action run -Name flashattn-w2-e2-fwd -Cpus 6-13`. Output: [flashattn-w2-e2-triton-fwd.out](../research/flashattention/results/flashattn-w2-e2-triton-fwd.out).
- **Observation**:
  - **Triton attention files in repo** (`98c1cee0ed59307833841cdbf567fe336ceec13a`): `flash_attn/flash_attn_triton.py` (1,160 lines; full forward/backward with causal, non-causal, arbitrary `seqlen`, `headdim <= 128`, bias) and `flash_attn/flash_attn_triton_og.py` (365 lines; tutorial 06 version, causal only, multiples of 128). Other Triton files in `flash_attn/ops/triton/` are non-attention ops (`cross_entropy.py`, `k_activations.py`, `layer_norm.py`, `linear.py`, `mlp.py`, `rotary.py`).
  - **Unmodified errors**:
    1. `orig_flash_attn_triton._flash_attn_forward` failed at **line 820** (`AssertionError: Only support fp16 and bf16`) on `float32` and **line 821** (`AssertionError` on `assert q.is_cuda and k.is_cuda and v.is_cuda`) on `float16`.
    2. `orig_flash_attn_triton_og.py` failed on import at **line 12** (`ModuleNotFoundError: No module named 'pytest'`).
    3. **Prediction miss on zero kernel edits**: When only the host assertions were patched (`host_patched_flash_attn_triton.py`), `_fwd_kernel` failed inside the Triton 3.8.0 interpreter at **line 183** (`qk += tl.dot(q, k, trans_b=True)`) with `InterpreterError: TypeError("dot() got an unexpected keyword argument 'trans_b'")` (and similarly at line 85 of `flash_attn_triton_og.py`). The repo kernels were written for `triton==2.0.0.dev20221202`, whereas Triton 3.8.0 dropped `trans_a`/`trans_b` keyword arguments from `tl.dot`.
  - **Smallest patch that worked**: In `work_e2/patched_flash_attn_triton.py`, in addition to the 2 host-assertion edits, replacing `tl.dot(q, k, trans_b=True)` with `tl.dot(q, tl.trans(k))` (1 line in `_fwd_kernel`, plus the 4 `tl.dot` transpose calls in `_bwd_kernel_one_col_block`) let both `flash_attn_triton_og.py` and `flash_attn_triton.py` run cleanly in `TRITON_INTERPRET=1`.
  - **Numerical results across 18 configurations** ($B=1, H=2, N \in \{64, 113, 128, 192, 256\}, D \in \{32, 64\}$, `causal` $\in \{\text{False}, \text{True}\}$):
    - `float32` kernel vs `float64` reference: `max_abs_diff = 7.143e-07`, `max_rel_diff = 7.313e-07`, `lse_diff <= 5.467e-07` (`max_abs_diff <= 8.345e-07` vs `float32` ref).
    - `float16` kernel vs `float64` reference: `max_abs_diff = 8.930e-04`, `max_rel_diff = 4.630e-04`, `lse_diff <= 5.738e-07`.
    - `patched_flash_attn_triton_og` (`fp16`, $Z=1, H=2, N=128, D=64$, causal): `max_abs_diff = 1.595e-03`, `max_rel_diff = 4.341e-04`.
  - **Negative controls**: Causal kernel vs non-causal reference failed with `max_abs_diff = 2.368e+00`; wrong `softmax_scale=1.0` vs `1/sqrt(64)` failed with `max_abs_diff = 3.879e+00`.
- **What it showed**: My prediction that `_fwd_kernel` needed zero edits was wrong: Triton 3.8 removed `trans_b=True` from `tl.dot`, requiring `tl.dot(q, tl.trans(k))` at line 183. With that 1-line kernel change and host guard removal, the repository's real `_fwd_kernel` runs on CPU under `TRITON_INTERPRET=1` for even and odd sequence lengths ($N=113$), $d \in \{32, 64\}$, causal and non-causal, matching exact `float64` attention to $7.14 \times 10^{-7}$ in `float32` and $8.93 \times 10^{-4}$ in `float16`.

### E3: Repository Triton attention backward pass, block-size sweep, and `tl.load`/`tl.store` instrumentation

- **Question**: (1) Does `_flash_attn_backward` in `patched_flash_attn_triton.py` run on CPU under `TRITON_INTERPRET=1`, or does `@triton.autotune` on `_bwd_kernel` fail because it has 2 configs (`SEQUENCE_PARALLEL=False/True`)? (2) How close are $\mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$ from the real Triton backward kernels (`_bwd_preprocess_do_o_dot` and `_bwd_kernel_one_col_block`) to PyTorch `autograd` (`float64`/`float32`), and what happens when we vary `(BLOCK_M, BLOCK_N)` across $\{(128,128), (64,64), (32,32)\}$? (3) Counting actual `tl.load` and `tl.store` calls in the interpreter, how many calls and elements do `_fwd_kernel` and `_bwd_kernel` execute as a function of $N$ and block size?
- **Prediction**: (1) `_bwd_kernel` with `@triton.autotune` (2 configs) will fail in `TRITON_INTERPRET=1` because `Autotuner.run` calls `do_bench` (which requires a GPU driver when `len(configs) > 1`); commenting out `@triton.autotune` on `_bwd_kernel` (matching `_fwd_kernel`) and zeroing `dq_accum` will succeed. (2) $\mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$ will match `float64` autograd with `max_abs_diff < 1e-4` (`float32`) and `< 5e-3` (`float16`) across all tested shapes and block sizes (`128`, `64`, `32`). (3) Because `_fwd_kernel` stores and reloads `acc_o_scale` and `o_scale` via the `TMP` buffer on every step, for $B=1, H=1$ with $T_r = \lceil N/\text{BLOCK\_M}\rceil$ and $I_{\text{total}}$ inner loop steps, `_fwd_kernel` executes $2 T_r + 3 I_{\text{total}}$ `tl.load` calls and $3 T_r + I_{\text{total}}$ `tl.store` calls (for $N=128, D=64$: 5 loads/4 stores at `BLOCK=128`; 16 loads/10 stores non-causal and 13 loads/9 stores causal at `BLOCK=64`; 52 loads/28 stores non-causal and 38 loads/22 stores causal at `BLOCK=32`). In backward (`SEQUENCE_PARALLEL=False`), `_bwd_kernel` executes $2 T_c + 5 J_{\text{total}}$ `tl.load` calls and $2 T_c + J_{\text{total}}$ `tl.store` calls.
- **Run**: Script [flashattn-w2-e3-triton-bwd.sh](../research/flashattention/scripts/flashattn-w2-e3-triton-bwd.sh) invoking [e3_triton_bwd_and_counts.py](../research/flashattention/work/e3_triton_bwd_and_counts.py) via `k3.ps1 -Action run -Name flashattn-w2-e3-bwd -Cpus 6-13`. Output: [flashattn-w2-e3-triton-bwd.out](../research/flashattention/results/flashattn-w2-e3-triton-bwd.out).
- **Observation**:
  - **Autotune failure confirmed**: Calling `_flash_attn_backward` with `@triton.autotune` enabled on `_bwd_kernel` (2 configs) failed at `triton/runtime/driver.py:20 (_create_driver)` with `RuntimeError: 0 active drivers ([]). There should only be one.` Commenting out `@triton.autotune` on `_bwd_kernel` (matching `_fwd_kernel`) made both `SEQUENCE_PARALLEL=False` (sequential column-block loop) and `SEQUENCE_PARALLEL=True` (parallel column blocks with `tl.atomic_add` on `DQ`) run on CPU.
  - **Backward gradient accuracy across 14 configurations** ($N \in \{64, 113, 128, 256\}, D \in \{32, 64\}$, causal and non-causal, vs `float64` autograd):
    - `float32`, `SEQUENCE_PARALLEL=False`: `max_abs_diff` $\mathbf{dQ} = 1.286 \times 10^{-6}$, $\mathbf{dK} = 1.733 \times 10^{-6}$, $\mathbf{dV} = 2.279 \times 10^{-6}$.
    - `float32`, `SEQUENCE_PARALLEL=True` (using `tl.atomic_add`): `max_abs_diff` $\mathbf{dQ} = 2.238 \times 10^{-6}$, $\mathbf{dK} = 2.185 \times 10^{-6}$, $\mathbf{dV} = 2.474 \times 10^{-6}$.
    - `float16`, `SEQUENCE_PARALLEL=False`: `max_abs_diff` $\mathbf{dQ} = 1.243 \times 10^{-3}$, $\mathbf{dK} = 1.391 \times 10^{-3}$, $\mathbf{dV} = 1.510 \times 10^{-3}$.
    - `float16`, `SEQUENCE_PARALLEL=True`: `max_abs_diff` $\mathbf{dQ} = 1.338 \times 10^{-3}$, $\mathbf{dK} = 1.295 \times 10^{-3}$, $\mathbf{dV} = 1.568 \times 10^{-3}$.
  - **Block-size sweep and measured `tl.load`/`tl.store` counts** (hooked on `interpreter_builder.create_masked_load` and `create_masked_store`, calibrated on a 2-block vector copy):
    - Changing `(BLOCK_M, BLOCK_N)` among `(128,128)`, `(64,64)`, and `(32,32)` kept forward and backward `float32` errors within $2.2 \times 10^{-7} \dots 1.7 \times 10^{-6}$ while scaling the number of tile loads/stores:
    - For $N=128, D=64$:
      - `BM=BN=128`: FWD `loads=5` ($24,832$ el), `stores=4` ($8,576$ el); BWD (`_bwd_preprocess_do_o_dot` + `_bwd_kernel`) `loads=9` ($57,600$ el), `stores=4` ($24,704$ el).
      - `BM=BN=64`: non-causal FWD `loads=16` ($41,344$ el), `stores=10` ($8,704$ el); causal FWD `loads=13` ($33,088$ el), `stores=9` ($8,640$ el). BWD non-causal `loads=28` ($82,432$ el), `stores=10` ($32,896$ el); causal `loads=23` ($70,016$ el), `stores=9` ($28,800$ el).
      - `BM=BN=32`: non-causal FWD `loads=56` ($74,368$ el), `stores=28` ($8,960$ el); causal FWD `loads=38` ($49,600$ el), `stores=22` ($8,768$ el). BWD non-causal `loads=96` ($132,096$ el), `stores=28` ($49,280$ el); causal `loads=66` ($94,848$ el), `stores=22` ($36,992$ el).
      - Note an arithmetic slip in my prediction text: my formula $2 T_r + 3 I_{\text{total}}$ at $T_r=4, I_{\text{total}}=16$ evaluates to $8 + 48 = 56$ (which is what the interpreter measured), whereas I had written `52` in the text.
    - For $N=113$ (non-multiple of 32/64/128), tile `tl.load`/`tl.store` counts and padded element counts are identical to $N=128$ because $\lceil 113/\text{BLOCK} \rceil = \lceil 128/\text{BLOCK} \rceil$ and masked loads still pass full `[BLOCK, D]` tensor handles in the interpreter.
  - **Negative control**: Zeroing `delta` ($D_i = \text{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i)$) after `_bwd_preprocess_do_o_dot` caused $\mathbf{dQ}$ and $\mathbf{dK}$ to fail (`dQ_err = 2.438e-01`, `dK_err = 1.295e-01`) while $\mathbf{dV}$ remained exact (`dV_err = 2.779e-07`).
- **What it showed**: Both backward modes (`SEQUENCE_PARALLEL=False` with `evict_last` load/store on $\mathbf{dQ}$, and `SEQUENCE_PARALLEL=True` with `tl.atomic_add` on $\mathbf{dQ}$) execute in the CPU interpreter once `@triton.autotune` is disabled. Instrumenting the interpreter confirmed the algebraic load/store formulas and exposed two concrete implementation details of `flash_attn_triton.py`: (1) `_fwd_kernel` performs an extra global `tl.store` and `tl.load` of `[BLOCK_M]` to `TMP` on every inner iteration plus the epilogue (lines 228–229, 255–256, marked `# BUG: have to store and immediately load`), and (2) when `SEQUENCE_PARALLEL=False`, `_bwd_kernel_one_col_block` loads and stores the `[BLOCK_M, D]` tile of `DQ` from global memory on every inner loop step ($J_{\text{total}}$ times), whereas `K` and `V` are loaded once per column block and `DK, DV` are stored once per column block.

### E4: Paper Algorithm 1 (tiled attention with online softmax), exactness, HBM traffic counter, and crossover $M$

- **Question**: (1) Does our implementation of FlashAttention-1 Algorithm 1 (with $B_c = \lceil M/(4d) \rceil$, $B_r = \min(B_c, d)$) in `float64` match exact standard attention across $N \in \{64, 113, 128, 256, 512, 517, 1024, 2048\}$, $d \in \{32, 64, 128\}$, causal and non-causal, and $M \in \{1024, 3000, 4096, 16384, 65536\}$? (2) What are the exact element read/write counts for the hand-countable case $N=4, d=2$ at $M=16$ and $M=8$? (3) At $N=4096, d=64$, what is the traffic ratio $E_{\mathrm{std}} / E_{\mathrm{fa1}}$ at $M=65,536$ and $M=98,304$, and below what SRAM size $M$ does tiled Algorithm 1 move MORE data than standard attention?
- **Prediction**: (1) Exactness against `float64` reference will hold across all tested configurations with `max_abs_diff < 1e-14`. (2) For $N=4, d=2$: standard attention moves $4(4)(2) + 4(16) = 96$ elements ($56$ reads, $40$ writes); at $M=16$ ($B_c=2, B_r=2, T_c=2, T_r=2$), FA1 matrix traffic is $(3\cdot 2+2)\cdot 8 = 64$ elements ($48$ reads, $16$ writes) and total traffic including $\mathbf{O}$-init and $(m, \ell)$ vectors is $112$ elements; at $M=8$ ($B_c=1, B_r=1, T_c=4, T_r=4$), FA1 matrix traffic is $(3\cdot 4+2)\cdot 8 = 112 > 96$. (3) At $N=4096, d=64$: $E_{\mathrm{std}} = 68,157,440$ elements; at $M=65,536$ ($B_c=256, B_r=64, T_c=16$), $E_{\mathrm{fa1,qkvo}} = 13,107,200$ elements, giving ratio $E_{\mathrm{std}} / E_{\mathrm{fa1,qkvo}} = 5.200000\times$ (and $E_{\mathrm{std}} / E_{\mathrm{fa1,all}} = 4.996997\times$); at $M=98,304$ ($B_c=384, B_r=64, T_c=11$), $E_{\mathrm{fa1,qkvo}} = 9,175,040$, giving ratio $7.428571\times$. Crossover: $E_{\mathrm{fa1,qkvo}} > E_{\mathrm{std}}$ when $T_c \ge 87$ ($B_c \le 47$), i.e., $M \le 12,032$ elements (and $E_{\mathrm{fa1,all}} > E_{\mathrm{std}}$ for $M \le 12,288$ elements).
- **Run**: Script [flashattn-w2-e4-alg1-traffic.sh](../research/flashattention/scripts/flashattn-w2-e4-alg1-traffic.sh) invoking [e4_alg1_and_traffic.py](../research/flashattention/work/e4_alg1_and_traffic.py) via `k3.ps1 -Action run -Name flashattn-w2-e4-alg1 -Cpus 6-13`. Output: [flashattn-w2-e4-alg1-traffic.out](../research/flashattention/results/flashattn-w2-e4-alg1-traffic.out).
- **Observation**:
  - **Hand-countable case ($N=4, d=2$)**:
    - Standard attention: `mat_reads = 56`, `mat_writes = 40`, `total_mat = 96` (exact match to hand count).
    - FA1 at $M=16$ ($B_c=2, B_r=2, T_c=2, T_r=2$): `mat_reads = 48`, `mat_writes = 16`, `total_qkvo = 64`, `total_all = 112`, `max_diff = 1.67e-16`.
    - FA1 at $M=8$ ($B_c=1, B_r=1, T_c=4, T_r=4$): `mat_reads = 80`, `mat_writes = 32`, `total_qkvo = 112` ($> 96$), `total_all = 192`, `max_diff = 1.11e-16`.
    - Negative control (`drop_last_outer=True` at $M=16$): `total_qkvo = 32` ($\neq 64$) and `max_diff = 4.457e-01`.
  - **Exactness sweep across 194 configurations** ($N \in \{64, 113, 128, 256, 512, 517, 1024, 2048\}$, $d \in \{32, 64, 128\}$, causal and non-causal, $M \in \{1024, 3000, 4096, 16384, 65536\}$ with $T_c T_r \le 4096$): `max_abs_diff = 2.442e-15`, `max_rel_diff = 3.157e-15`.
  - **Traffic ratio and crossover at $N=4096, d=64$** ($E_{\mathrm{std}} = 68,157,440$ elements: $34,340,864$ reads + $33,816,576$ writes):
    - At $M=65,536$ ($B_c=256, B_r=64, T_c=16, T_r=64$): $E_{\mathrm{qkvo}} = 13,107,200$, ratio $= 5.200000\times$; $E_{\mathrm{all}} = 13,639,680$, ratio $= 4.996997\times$ (exact match to prediction).
    - At $M=98,304$ ($B_c=384, B_r=64, T_c=11, T_r=64$): $E_{\mathrm{qkvo}} = 9,175,040$, ratio $= 7.428571\times$; $E_{\mathrm{all}} = 9,625,600$, ratio $= 7.080851\times$ (exact match to prediction).
    - **Crossover $M$**:
      - For matrix traffic $E_{\mathrm{qkvo}} = (3 T_c + 2) N d$: at $M = 12,032$ ($B_c = 47, T_c = 88$), $E_{\mathrm{qkvo}} = 69,730,304 > 68,157,440$ (ratio $0.977444\times$); at $M \in [12033, 12288]$ ($B_c = 48, T_c = 86$), $E_{\mathrm{qkvo}} = 68,157,440 = E_{\mathrm{std}}$ (ratio $1.000000\times$); at $M = 12,289$ ($B_c = 49, T_c = 84$), $E_{\mathrm{qkvo}} = 66,584,576 < E_{\mathrm{std}}$ (ratio $1.023622\times$).
      - Slight refinement on $E_{\mathrm{all}}$: at $M = 12,289$ ($B_c = 49, T_c = 84$), including the $\mathbf{O}$-init write ($262,144$ el) and $(m, \ell)$ vector reads/writes ($1,384,448$ el) gives $E_{\mathrm{all}} = 68,231,168 > 68,157,440$ (ratio $0.998919\times$), so $E_{\mathrm{all}}$ crosses below $E_{\mathrm{std}}$ at $B_c = 50$ ($M = 12,545$, $T_c = 82$), one tile step higher than $12,288$.
- **What it showed**: Tiled Algorithm 1 is exact to machine precision ($2.44 \times 10^{-15}$) across arbitrary sequence lengths (including primes $N=113, 517$), head dimensions, and SRAM sizes, and reduces HBM matrix traffic by $5.20\times$ at $M=65,536$ and $7.43\times$ at $M=98,304$ for $N=4096, d=64$. However, when $M \le 12,032$ ($B_c \le 47 < \frac{3}{4}d$), $T_c \ge 88$ outer passes each stream $\mathbf{Q}$ (read) and $\mathbf{O}$ (read+write) of width $d=64$ across HBM ($3d / B_c = 192 / 47 = 4.085 > 4$ HBM accesses per $(q, k)$ pair), moving more data than standard attention's $4$ accesses per $(q, k)$ pair ($\mathbf{S}$ write/read + $\mathbf{P}$ write/read).

### E5: Online softmax step-by-step block trace and negative controls

- **Question**: When processing key blocks $j = 1 \dots T_c$ one at a time for a single query row ($N=16, d=4, B_c=4, T_c=4$), how do the running maximum $m^{(j)}$, running sum $\ell^{(j)}$, and running output $\mathbf{O}^{(j)}$ evolve after each block, and does the final result after block $T_c$ equal plain $\mathrm{softmax}(\mathbf{q}\mathbf{K}^\top / \sqrt{d})\mathbf{V}$? What happens in the two negative controls: (a) leaving out the $\exp(m_{\text{old}} - m_{\text{new}})$ rescale factor, and (b) shifting the causal mask by $+1$?
- **Prediction**: After block $T_c = 4$, both the FA1 (normalized every step) and FA2 (unnormalized until the end) online updates will match plain `float64` softmax with `max_abs_diff < 1e-15`, and $m^{(4)} + \log \ell^{(4)}$ will match `torch.logsumexp` to `< 1e-15`. Negative control (a) (omitting $\exp(m_{\text{old}} - m_{\text{new}})$) will fail with `max_abs_diff > 0.1` whenever $m^{(j)}$ increases across blocks. Negative control (b) (shifting causal mask by $+1$) will fail with `max_abs_diff > 0.1`.
- **Run**: Script [flashattn-w2-e5-online-softmax.sh](../research/flashattention/scripts/flashattn-w2-e5-online-softmax.sh) invoking [e5_online_softmax.py](../research/flashattention/work/e5_online_softmax.py) via `k3.ps1 -Action run -Name flashattn-w2-e5-softmax -Cpus 6-13`. Output: [flashattn-w2-e5-online-softmax.out](../research/flashattention/results/flashattn-w2-e5-online-softmax.out).
- **Observation**:
  - **Step-by-step trace ($N=16, d=4, B_c=4, T_c=4$, plain reference $\mathbf{O} = [+0.231204, -0.139544, +0.375889, -0.040121], L = +4.537275$)**:
    - Block 1 (`keys 0..3`): $\tilde{m}=-0.0009, \tilde{\ell}=3.5089 \implies m^{(1)}=-0.0009, \ell^{(1)}=3.5089$, $\mathbf{O}^{(1)} = [-0.775328, -0.283623, -0.116234, +0.453355]$ (`err_vs_prefix_ref = 5.55e-17`).
    - Block 2 (`keys 4..7`): $\tilde{m}=+1.8374 > m^{(1)} \implies m^{(2)}=+1.8374$, $\alpha = \exp(-0.0009 - 1.8374) = 0.1591$, $\beta = 1.0000$, $\ell^{(2)} = 0.1591(3.5089) + 2.9246 = 3.4829$, $\mathbf{O}^{(2)} = [+0.555435, -0.726488, +0.538052, +0.598000]$ (`err_vs_prefix_ref = 1.11e-16`).
    - Block 3 (`keys 8..11`): $\tilde{m}=+0.3586 < m^{(2)} \implies m^{(3)}=+1.8374$, $\alpha = 1.0000$, $\beta = \exp(0.3586 - 1.8374) = 0.2279$, $\ell^{(3)} = 3.4829 + 0.2279(2.6937) = 4.0968$, $\mathbf{O}^{(3)} = [+0.452661, -0.641572, +0.447221, +0.563335]$ (`err_vs_prefix_ref = 2.22e-16`).
    - Block 4 (`keys 12..15`): $\tilde{m}=+2.9580 > m^{(3)} \implies m^{(4)}=+2.9580$, $\alpha = \exp(1.8374 - 2.9580) = 0.3261$, $\beta = 1.0000$, $\ell^{(4)} = 0.3261(4.0968) + 3.5157 = 4.8516$, $\mathbf{O}^{(4)} = [+0.231204, -0.139544, +0.375889, -0.040121]$.
    - Final error vs plain full-row softmax: `5.551e-17` for both FA1 and FA2 updates; LSE diff vs `torch.logsumexp`: `0.000e+00`.
  - **Negative Control A (omitting $\exp(m_{\text{old}} - m_{\text{new}})$ rescale)**:
    - Block 1 is unaffected (`5.55e-17`), but at Block 2 when $m$ increases from $-0.0009$ to $+1.8374$, setting $\alpha=1$ instead of $0.1591$ inflates $\ell^{(2)}$ to $6.4335$ (instead of $3.4829$) and jumps the output error to `6.10e-01`. After Block 4, $\ell^{(4)} = 12.6429$ (instead of $4.8516$), final output `max_abs_diff = 3.139e-01`, and LSE error is `9.578e-01`.
  - **Negative Control B (shifting causal mask by $+1$ and $-1$ at $N=64, d=32, B_c=16, B_r=16$)**:
    - Exact causal mask (`shift=0`): `max_abs_diff = 9.437e-16`.
    - Shifted by $+1$ (`col <= row + 1`): `max_abs_diff = 1.482e+00` (`row 0 err = 1.482e+00`, whereas `last row N-1 err = 1.110e-16` because row $N-1$ already attends to all $N$ keys).
    - Shifted by $-1$ (`col <= row - 1`): `max_abs_diff = 2.408e+00`.
- **What it showed**: At the end of every block $j$, online softmax maintains the exact softmax attention over the prefix of keys $1 \dots j B_c$ seen so far (`err <= 2.22e-16`). Whenever a new block has a larger maximum ($\tilde{m} > m_{\text{old}}$), the factor $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ rescales all previously accumulated exponentials to the new maximum; omitting $\alpha$ over-weights earlier blocks by $\exp(m_{\text{new}} - m_{\text{old}})$ and breaks the output (`3.14e-01`).

### E6: Backward pass with recomputation (exactness, stored bytes, and matmul FLOPs)

- **Question**: (1) Does our tiled backward pass with recomputation (keeping only $\mathbf{O}$ and logsumexp $L = m + \log \ell$ from the forward pass, precomputing $D_i = \text{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i)$, and recomputing $\mathbf{S}_{ij} = \tau \mathbf{Q}_i \mathbf{K}_j^\top$ and $\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - L_i)$ tile by tile) match PyTorch `float64` `autograd` across $N \in \{64, 113, 128, 256, 512, 1024\}$, $d \in \{32, 64, 128\}$, causal and non-causal? (2) How many bytes are stored from the forward pass and how many matmul FLOPs are executed compared to standard attention backward that stores $\mathbf{P}$?
- **Prediction**: (1) Tiled backward with recomputation will match `float64` autograd with `max_abs_diff < 1e-13` across all tested configurations, while omitting $D_i$ in $d\mathbf{S}_{ij} = \mathbf{P}_{ij} \circ (d\mathbf{P}_{ij} - D_i)$ will fail with `max_abs_diff > 0.1` on $\mathbf{dQ}$ and $\mathbf{dK}$. (2) Standard backward performs 4 matmuls totaling $8 N^2 d$ FLOPs ($536,870,912$ at $N=1024, d=64$), whereas recomputation adds 1 matmul ($\mathbf{Q}_i \mathbf{K}_j^\top$) of $2 N^2 d$ FLOPs for $10 N^2 d$ FLOPs ($671,088,640$ at $N=1024, d=64$, exactly $1.25\times$). Storing $(\mathbf{O}, L)$ ($2Nd + 4N$ bytes in FP16/FP32) instead of $\mathbf{P}$ ($2N^2$ bytes in FP16) shrinks saved state from $2,097,152\text{ B}$ to $135,168\text{ B}$ ($15.52\times$) at $N=1024, d=64$ and from $33,554,432\text{ B}$ to $540,672\text{ B}$ ($62.06\times$, or $2048\times$ for $L$ vs $\mathbf{P}$ alone) at $N=4096, d=64$.
- **Run**: Script [flashattn-w2-e6-bwd-recompute.sh](../research/flashattention/scripts/flashattn-w2-e6-bwd-recompute.sh) invoking [e6_bwd_recompute.py](../research/flashattention/work/e6_bwd_recompute.py) via `k3.ps1 -Action run -Name flashattn-w2-e6-bwd -Cpus 6-13`. Output: [flashattn-w2-e6-bwd-recompute.out](../research/flashattention/results/flashattn-w2-e6-bwd-recompute.out).
- **Observation**:
  - **Negative controls ($N=128, d=64$)**:
    - Omitting $D_i = \text{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i)$ failed on $\mathbf{dQ}$ (`3.148e-01`) and $\mathbf{dK}$ (`2.194e-01`) while leaving $\mathbf{dV}$ (`6.661e-16`) intact.
    - Shifting $L_i \leftarrow L_i + 1.0$ (which scales recomputed $\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - L_i)$ by $e^{-1}$) failed on all three gradients: `dQ_err = 4.017e-01`, `dK_err = 4.210e-01`, `dV_err = 5.058e-01`.
  - **Exactness across 72 configurations** ($N \in \{64, 113, 128, 256, 512, 1024\}$, $d \in \{32, 64, 128\}$, causal and non-causal, $(B_r, B_c) \in \{(32, 32), (64, 64)\}$): `max_abs_diff` $\mathbf{dQ} = 4.607 \times 10^{-15}$, $\mathbf{dK} = 1.066 \times 10^{-14}$, $\mathbf{dV} = 1.066 \times 10^{-14}$.
  - **Stored bytes and counted matmul FLOPs ($d=64$)**:
    - Counted matmul FLOPs in recomputing backward are exactly $1.2500\times$ standard backward at every $N$ ($5$ matmuls vs $4$ matmuls: e.g., at $N=1024, d=64$, $671,088,640$ vs $536,870,912$ FLOPs; at $N=4096, d=64$, $10,737,418,240$ vs $8,589,934,592$ FLOPs).
    - Stored state bytes ($\mathbf{P}_{\text{fp16}} = 2N^2$ bytes vs $\mathbf{O}_{\text{fp16}} + L_{\text{fp32}} = 2Nd + 4N$ bytes):
      - At $N=64, d=64$: $8,192\text{ B}$ vs $8,448\text{ B}$ ($0.97\times$; $L$ alone is $256\text{ B}$, $32\times$ smaller than $\mathbf{P}$).
      - At $N=1024, d=64$: $2,097,152\text{ B}$ vs $135,168\text{ B}$ ($15.52\times$; $L$ alone is $4,096\text{ B}$, $512\times$ smaller).
      - At $N=4096, d=64$: $33,554,432\text{ B}$ vs $540,672\text{ B}$ ($62.06\times$; $L$ alone is $16,384\text{ B}$, $2048\times$ smaller).
- **What it showed**: Knowing only the per-row scalar logsumexp $L_i = m_i + \log \ell_i$ from the forward pass allows $\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - L_i)$ to be reconstructed tile-by-tile on-chip without a second pass over the row, and precomputing $D_i = \text{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i) = \text{rowsum}(\mathbf{P}_i \circ d\mathbf{P}_i)$ supplies the full-row softmax Jacobian contraction before any tile loop starts. This trades $+25\%$ matmul FLOPs ($10 N^2 d$ vs $8 N^2 d$) to avoid storing and re-reading the $N \times N$ matrix $\mathbf{P}$.

### E7: Split-KV parallel chunk execution and combine step

- **Question**: When the KV sequence of length $N_k$ is split into $S$ chunks processed separately (each returning partial output $\mathbf{O}^{(s)}$ and logsumexp $L^{(s)}$ or $(m^{(s)}, \ell^{(s)})$) and combined at the end via $L_i = \mathrm{logsumexp}_s(L^{(s)}_i)$ and $\mathbf{O}_i = \sum_s \exp(L^{(s)}_i - L_i)\mathbf{O}^{(s)}_i$, does the combined output match single-pass `float64` attention across (a) equal chunks, (b) uneven chunks ($N_k=517$ and irregular sizes $[1, 3, 19, 100, 394]$), and (c) causally masked chunks where some splits have all keys masked out ($L^{(s)}_i = -\infty$)?
- **Prediction**: Both LSE combine and $(m, \ell)$ combine will match single-pass `float64` attention across all equal, uneven, and causally masked chunk configurations with `max_abs_diff < 1e-15` on $\mathbf{O}$ and $L$. Negative control (1) combining chunks with uniform weight $1/S$ instead of $\exp(L^{(s)}_i - L_i)$ will fail with `max_abs_diff > 0.1`; negative control (2) omitting the $L^{(s)}_i = -\infty$ guard on fully masked causal chunks will produce `NaN`s (`exp(-inf - (-inf))`).
- **Run**: Script [flashattn-w2-e7-splitkv.sh](../research/flashattention/scripts/flashattn-w2-e7-splitkv.sh) invoking [e7_splitkv_combine.py](../research/flashattention/work/e7_splitkv_combine.py) via `k3.ps1 -Action run -Name flashattn-w2-e7-splitkv -Cpus 6-13`. Output: [flashattn-w2-e7-splitkv.out](../research/flashattention/results/flashattn-w2-e7-splitkv.out).
- **Observation**:
  - **Equal chunks** ($N_k=1024, d=64, N_q \in \{1, 8, 64\}, S \in \{1, 2, 4, 8, 16\}$): `max_O_lse_err = 6.384e-16`, `max_O_ml_err = 6.384e-16`, `max_L_err = 1.776e-15`.
  - **Uneven & prime chunks** ($N_k=517, d=64, N_q=16$):
    - `S=3 ceil` ($[173, 173, 171]$): `O_lse_err = 3.608e-16`, `O_ml_err = 4.441e-16`, `L_err = 8.882e-16`.
    - `S=5 ceil` ($[104, 104, 104, 104, 101]$): `O_lse_err = 4.718e-16`, `O_ml_err = 5.829e-16`, `L_err = 8.882e-16`.
    - `S=7 ceil` ($[74 \times 6, 73]$): `O_lse_err = 4.718e-16`, `O_ml_err = 4.718e-16`, `L_err = 8.882e-16`.
    - `S=11 with empty tail` ($[52 \times 9, 49, 0]$): `O_lse_err = 4.996e-16`, `O_ml_err = 5.551e-16`, `L_err = 0.000e+00`.
    - Irregular $[1, 3, 19, 100, 394]$: `O_lse_err = 5.829e-16`, `O_ml_err = 4.718e-16`, `L_err = 8.882e-16`.
    - Irregular $[250, 1, 1, 265]$: `O_lse_err = 4.996e-16`, `O_ml_err = 4.718e-16`, `L_err = 8.882e-16`.
  - **Causally masked chunks** (where later chunks are completely masked out for early query rows, producing $32 \dots 384$ entries with $L^{(s)}_i = -\infty$): `O_lse_err <= 8.882e-16`, `O_ml_err <= 8.882e-16`, `L_err <= 8.882e-16`.
  - **Negative controls**:
    - Uniform $1/S$ averaging on $[1, 3, 19, 100, 394]$ failed with `max_abs_diff = 8.572e-01`.
    - Omitting the $-\infty$ guard when combining masked splits produced `NaN`s (`unguarded has_nan=True` vs `guarded has_nan=False`).
- **What it showed**: Because logsumexp satisfies $\log \sum_{s=1}^S \exp(L^{(s)}_i) = \log \sum_{s=1}^S \sum_{k \in C_s} \exp(S_{ik}) = L_i$ and $\exp(L^{(s)}_i - L_i) = \ell^{(s)}_i \exp(m^{(s)}_i - m_i) / \ell_i$, disjoint KV chunks of any positive (or even zero/masked) length can be reduced independently in parallel and combined at the end with a single rescaled linear combination.

### E8: FlashAttention-1 vs FlashAttention-2 loop order write counts; CUDA kernel header tile sizes and shared-memory formula (`[read, not run]`)

- **Question**: (1) In two small programs implementing the FA1 loop order (outer loop $j=1 \dots T_c$ over $\mathbf{K}, \mathbf{V}$ blocks, inner loop $i=1 \dots T_r$ over $\mathbf{Q}, \mathbf{O}$ blocks) and the FA2 loop order (outer loop $i=1 \dots T_r$ over $\mathbf{Q}, \mathbf{O}$ blocks, inner loop $j=1 \dots T_c$ over $\mathbf{K}, \mathbf{V}$ blocks), how many HBM writes and reads of $\mathbf{O}, m, \ell$ (or $L$) occur at $N=1024, d=64, B_r=64, B_c=64$ ($T_r=16, T_c=16$), and do both produce identical `float64` outputs? (2) Reading `csrc/flash_attn/src/kernel_traits.h` and `flash_fwd_launch_template.h` (`[read, not run]`), what is the exact shared-memory formula `kSmemSize` and what are the tile sizes (`kBlockM`, `kBlockN`, `kNWarps`) and shared-memory bytes for head dim 64 and 128 (`fp16`, `sm80`)?
- **Prediction**: (1) Both programs will match `float64` reference attention (and each other) to `< 1e-15`. At $N=1024, d=64, B_r=B_c=64$ ($T_r=T_c=16$): FA1 writes $\mathbf{O}$ to HBM $T_c=16$ times per row block ($256$ tile writes $= 1,048,576$ elements, plus $1,048,576$ elements read back) and writes $m, \ell$ $16$ times each ($16,384$ elements written and $16,384$ read each); FA2 keeps $\tilde{\mathbf{O}}, m, \ell$ on-chip across the inner loop and writes $\mathbf{O}$ to HBM once per row block ($16$ tile writes $= 65,536$ elements, $0$ reads of $\mathbf{O}$, $16\times$ fewer $\mathbf{O}$ writes) and writes $L = m + \log \ell$ once ($1,024$ elements, $0$ reads). (2) (`[read, not run]`) From `kernel_traits.h:107-109`, `kSmemSize = Share_Q_K_smem ? max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize`, where `kSmemQSize = 2*kBlockM*d` and `kSmemKVSize = 4*kBlockN*d` bytes in `fp16`. From `flash_fwd_launch_template.h:204-244` (`fp16`, `sm80`, no dropout): for $d=64$, `kBlockM=128, kBlockN=128, kNWarps=4, Share_Q_K_smem=false` $\implies \text{kSmemSize} = 16,384 + 32,768 = 49,152\text{ B}$ ($48\text{ KB}$); for $d=128$ on `sm80`, `kBlockM=128, kBlockN=64, kNWarps=8, Share_Q_K_smem=false` $\implies \text{kSmemSize} = 32,768 + 32,768 = 65,536\text{ B}$ ($64\text{ KB}$).
- **Run**: Script [flashattn-w2-e8-fa2-smem.sh](../research/flashattention/scripts/flashattn-w2-e8-fa2-smem.sh) invoking [e8_fa1_vs_fa2_and_smem.py](../research/flashattention/work/e8_fa1_vs_fa2_and_smem.py). Output: [flashattn-w2-e8-fa2-smem.out](../research/flashattention/results/flashattn-w2-e8-fa2-smem.out). Part 3 of the script reads the C++ header text and evaluates the formulas by hand in Python; it does not compile or run CUDA.
- **Observation**:
  - **Loop-order counts** (`float64`, $d=64$, $B_r=B_c=64$): outputs of the two loop orders agree to `1.67e-16` (O) and `1.78e-15` (L) at the largest. At $N=1024$ non-causal, FA1 order writes $\mathbf{O}$ in `256` tiles (`1,048,576` elements) and reads `1,048,576` back; FA2 order writes `16` tiles (`65,536`) and reads `0`. $(m,\ell)$ writes: FA1 `32,768`, FA2 (single $L$) `1,024`. Causal at $N=1024$: FA1 `136` tiles vs FA2 `16`. Prediction matched exactly. At $N=64$ ($T_r=T_c=1$) the two orders are identical in writes (`4,096` each) — the gap only appears once $T_c > 1$.
  - **Negative control**: skipping the final division by $\ell$ in the FA2 order: `max_abs_diff = 12.91`.
  - **Forward shared memory** (formula `kSmemQSize + kSmemKVSize`, `kernel_traits.h:107-109`, `[read, not run]`; the script evaluated it for every active `Flash_fwd_kernel_traits` line): $d=64$ non-dropout `128x128`, 4 warps: `49,152 B` (48 KB); $d=64$ with dropout `128x64`: `32,768 B`. $d=128$ non-dropout: `128x64` (non-sm8x-minor path, i.e. sm80/sm90): `65,536 B` (64 KB); `128x32` on sm86/sm89 non-causal: `49,152 B`; `64x64` on sm86/sm89 causal: `49,152 B`.
  - **Backward shared memory** — the launch template (`flash_bwd_launch_template.h:95`) uses `kSmemSize1colblock`, defined at `kernel_traits.h:287-290` as `kSmemQdOSize + (!Is_V_in_regs ? KV + dS + P : max(KV, KV/2 + dS + P))`, with `kSmemQdOSize` counting 3 copies of the `Q`/`dO` tile (2 if `No_double_buffer`). I hand-checked the first one: $d=64$, `128x128`: $49{,}152 + 32{,}768 + 32{,}768 + 32{,}768 = 147{,}456$ B (144 KB) — agrees with the script. All four active configs: $d=64$ `128x128x8warps` `147,456 B`; $d=64$ `64x128`, V in registers `73,728 B`; $d=128$ `64x128` `147,456 B`; $d=128$ `64x64`, V in registers `81,920 B`.
  - **Which backward tile is picked depends on the device**: `flash_bwd_launch_template.h` branches on `cudaDevAttrMaxSharedMemoryPerBlockOptin` (e.g. for $d=64$: `>= 144 KB` picks `128x128` with `Is_V_in_regs=false`; otherwise `64x128` with `Is_V_in_regs=true`). I read these branches on `k3` (text only).
- **What it showed**: In FA1's loop order the output tile for each row block is written and re-read once per KV block ($T_c$ times); in FA2's order it stays on chip across the inner loop and is written once, in the same arithmetic result. The shared-memory numbers are a formula evaluation of header text, not a compiled kernel's measured usage. Two of my own claims were off: (a) my prediction wrote `kNWarps=8` for $d=128$ forward — the header says `4`; the `64 KB` value stands because it comes from the sizes, not the warp count. (b) the script's first backward shared-memory numbers did not follow the formula at `kernel_traits.h:287-290`; I found that by reading the header, fixed the script and re-ran (the corrected output is the file linked above). Forward numbers had been right.

## 4. How it works

(What each piece is, as I came to understand it. Every statement cites an experiment; paper text I read is cited by file.)

**Standard attention** materialises $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$ and $\mathbf{P} = \mathrm{softmax}(\mathbf{S})$, each $N \times N$, writes both to main memory and reads them back (E4: $4$ HBM accesses per $(q,k)$ pair — $\mathbf{S}$ write/read, $\mathbf{P}$ write/read — hand-counted at $N=4$, $d=2$: `96` elements).

**Online softmax** (E5). For a query row, keys are consumed in blocks. After block $j$ the state is a running max $m$, running sum $\ell$ and a running output equal to the exact softmax attention over the keys seen so far (error `<= 2.2e-16` at every block). When a new block has a larger max, the old $\ell$ and old output are multiplied by $\exp(m_{old} - m_{new})$; leaving that factor out gave `3.1e-01` error. This identity is what lets blocks be processed one at a time.

**FA1 forward** (`2205.14135/src/algo.tex`, Algorithm 1, read; E4, E8). Outer loop over KV blocks $j$, inner loop over Q blocks $i$; each inner step loads $\mathbf{Q}_i, \mathbf{O}_i, \ell_i, m_i$, updates, and writes $\mathbf{O}_i, \ell_i, m_i$ back. Block sizes are set from the SRAM size: $B_c = \lceil M/4d \rceil$, $B_r = \min(B_c, d)$. Matrix traffic is $(3T_c + 2)Nd$ elements (E4). It is exact to `2.4e-15` over 194 configurations (E4). It moves less than standard attention only above a SRAM size: at $N=4096$, $d=64$ the crossover was between $M = 12{,}032$ and $12{,}545$ elements (depending on whether the $\mathbf{O}$-init and $(m,\ell)$ traffic is counted), with $5.2\times$ less traffic at $M=65{,}536$ (E4).

**FA2 forward** (`2307.08691/src/algo.tex`, "Forward pass", read; E5, E7, E8). Loops swapped: outer over Q blocks, inner over KV blocks. The accumulator is kept un-normalised and divided by $\ell$ once at the end; only $L = m + \log\ell$ is stored instead of $(m,\ell)$. Measured effect of the swap: $\mathbf{O}$ written once per row block rather than $T_c$ times, with zero reads of $\mathbf{O}$ (E8, `65,536` vs `1,048,576` elements at $N=1024$). Dropping the final divide breaks the result (E8 control, `12.9`). Causal masking in the paper skips blocks entirely above the diagonal (E8 counts: `136` of `256` tiles for FA1 order, causal $N=1024$).

**Backward with recomputation** (E6, E3). Forward saves $\mathbf{O}$ and $L$ only. Backward precomputes $D_i = \mathrm{rowsum}(d\mathbf{O}_i \circ \mathbf{O}_i)$, then per tile recomputes $\mathbf{S}_{ij}$ and $\mathbf{P}_{ij} = \exp(\mathbf{S}_{ij} - L_i)$ and forms $d\mathbf{S} = \mathbf{P} \circ (d\mathbf{P} - D_i)$. Gradients match `float64` autograd to `~1e-14` (72 configs). Without $D_i$, $\mathbf{dQ}$ and $\mathbf{dK}$ fail but $\mathbf{dV}$ is still exact (E6, E3) — $D_i$ only enters $d\mathbf{S}$. Counted matmul FLOPs: $10N^2d$ vs $8N^2d$ for standard (`1.25x`); saved state `135,168 B` vs `2,097,152 B` at $N=1024$, $d=64$ (E6).

**Split-KV** (E7). Disjoint KV chunks each give $(\mathbf{O}^{(s)}, L^{(s)})$; $L = \mathrm{logsumexp}_s L^{(s)}$ and $\mathbf{O} = \sum_s e^{L^{(s)} - L}\mathbf{O}^{(s)}$. Exact for equal, prime, uneven, empty and fully-masked chunks, provided $L^{(s)} = -\infty$ is guarded (without the guard: NaN).

**The repository's Triton kernels** (E2, E3). `flash_attn_triton.py` implements forward/backward; in `_fwd_kernel` the per-row scale goes through a global `TMP` buffer (a store then immediate load, marked `# BUG` in the source), measured as extra loads/stores in E3. Backward with `SEQUENCE_PARALLEL=False` reloads and stores the $\mathbf{dQ}$ tile each inner step; `True` uses `tl.atomic_add` on $\mathbf{dQ}$. Both run on CPU in Triton's interpreter after one-line `tl.dot` edits (E2, E3).

**The CUDA kernels** (E8, read not run). Tile sizes and shared-memory use are chosen per head dimension, and for backward per device's opt-in shared-memory limit; sizes in E8.

## 5. What surprised me and what I got wrong

- **Triton version drift (E2)**: I predicted the repo's `_fwd_kernel` would need zero edits; Triton 3.8 had dropped `trans_b` from `tl.dot`, so it needed 1 line (plus 4 in backward).
- **`bfloat16` in the interpreter (E1)**: output of `tl.dot` was garbage (`8.9e+10`); `float32`/`float16` were fine. I did not diagnose the cause.
- **Arithmetic slips in my own predictions**: E3 (`52` written, formula gives `56`); E4's crossover was off by one tile step for the all-in count; E8 `kNWarps=8` for $d=128$ forward (header: `4`).
- **A wrong number that was reproducible (E8 backward)**: the script gave the same wrong backward shared-memory numbers on every run. Reproducing a script's output shows stability, not correctness; I found it only by reading the formula in `kernel_traits.h` and doing one case by hand.
- **E3 hook counts element counts** include masked padding in the interpreter (e.g. $N=113$ gives the same counts as $N=128$), so they are tile-level counts, not bytes the hardware would move.

## 6. Authors say / code does / I observed

| Claim | Where | What I observed |
|---|---|---|
| Exact attention, tiled, online softmax | FA1 abstract, Algorithm 1 | exact to `2.4e-15` (E4); my implementation of it, not the authors' CUDA |
| Fewer HBM reads/writes than standard | FA1 abstract | in element counts from my counter: yes above $M \approx 12$K elements at $N=4096$, $d=64$; more below that (E4). Wall-clock: not measured (no GPU) |
| Backward recomputes $\mathbf{S},\mathbf{P}$ from $\mathbf{O}$ and $L$ | FA1 `algo.tex` | exact vs autograd (E6); `+25%` matmul FLOPs counted |
| FA2 avoids rescaling at every step; stores only $L$ | FA2 `algo.tex` | un-normalised accumulator exact (E5/E8); $\mathbf{O}$ writes `16x` fewer at $N=1024$ (E8) |
| FA2 is 2-3x faster | FA2 abstract | not tested (needs GPU) |
| Causal masking skips about half the blocks | FA2 `algo.tex` | `136/256` tiles at $N=1024$ in my loop-order counts (E8) |
| Kernel tile / smem choices | `flash_fwd_launch_template.h`, `kernel_traits.h` | formula evaluated from header text (E8), not measured on a device |

## 7. Not run, and why. Unknowns

- CUDA C++ kernels (`csrc/flash_attn/`, `hopper/`, `flash_attn/cute/`): not run — `k3` has no GPU and no `nvcc`. Every speed claim in the papers (wall-clock, TFLOPs/s, occupancy) is untested here.
- Shared-memory sizes in E8 are computed from header formulas; whether the compiled kernel allocates exactly these bytes is unchecked.
- `bfloat16` interpreter garbage in E1: cause unknown.
- FA3 (`hopper/`) and the `cute/` version: not read.
- Sections of the FA1/FA2 papers beyond `abstract.tex`, FA1 `algo.tex` (Algorithm 1 and surrounding text) and FA2 `algo.tex` (forward pass and causal paragraph): not read in this study.

## 8. Sources and Sources read

- **Paper 1**: Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135v2 (retrieved 2026-10-26).
- **Paper 2**: Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691v1 (retrieved 2026-10-26).
- **Repository**: `https://github.com/Dao-AILab/flash-attention.git` at commit `98c1cee0ed59307833841cdbf567fe336ceec13a` (retrieved 2026-10-26).

### Sources read
- `2205.14135/src/abstract.tex` (lines 1–15, full)
- `2205.14135/src/algo.tex` (lines 40–135: tiling, recomputation, kernel fusion, Algorithm 1)
- `2307.08691/src/abstract.tex` (lines 1–30, full)
- `2307.08691/src/algo.tex` (lines 1–110: forward pass tweaks, Algorithm of the forward pass, causal masking start)
- Repository `csrc/flash_attn/src/kernel_traits.h` lines 103–111 and 262–292; `flash_fwd_launch_template.h` lines 203–278; `flash_bwd_launch_template.h` lines 140–256 (selected branches), on `k3`.
- Repository `flash_attn/flash_attn_triton.py` and `flash_attn_triton_og.py`: read by the first agent wave and exercised by E2/E3; I did not re-read them line by line.

## 9. Files

- Context file: `C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\docs\flashattention-context.md`
- Predictions log: `C:\personal\oss\clover-framework-hybrid\clover-kimi-k3\research\flashattention\work\predictions.txt`

## 10. Independent check (2026-10-26, by me, after the experiments)

- **Re-runs on `k3`** (same scripts, output compared line by line with the stored result files): E1, E3, E4, E5, E6, E7: identical. E2: identical except the two `diff` timestamps and one number, `patched_flash_attn_triton_og` fp16 `max_abs_diff` `1.542e-03` vs stored `1.595e-03`; that test's input at `e2_triton_fwd.py:61` uses `torch.randn` before any seed is set, so a different draw is expected (both within the `2e-3` prediction). E8 re-run produced the corrected backward numbers (stored file replaced). Six of the eight re-runs matched line for line; E2 differed only as stated; E8 changed on purpose.
- **By-hand formula checks (separate Python, not the study scripts)**: $E_{\mathrm{std}} = 4Nd + 4N^2 = 68{,}157{,}440$ at $N=4096$, $d=64$; $(3T_c+2)Nd$ gives `13,107,200` at $M=65{,}536$ ($T_c=16$, ratio `5.2`), `9,175,040` at $M=98{,}304$ ($T_c=11$, `7.428571`), and the crossover values `0.977444` / `1.000000` / `1.023622` at $M = 12{,}032 / 12{,}288 / 12{,}289$; saved state `2,097,152 B` vs `135,168 B` (`15.52x`) at $N=1024$; FA1 $\mathbf{O}$ tiles `256` non-causal and $1+2+\dots+16 = 136$ causal. All agree with E4, E6, E8.
- **By-hand source check**: backward `kSmemSize1colblock` for $d=64$, `128x128`, 8 warps, from the formula at `kernel_traits.h:287-290`: $49{,}152 + 32{,}768 + 32{,}768 + 32{,}768 = 147{,}456$ B; $d=128$ `64x128`: $49{,}152 + 65{,}536 + 16{,}384 + 16{,}384 = 147{,}456$ B. Both agree with the corrected E8.
- **What this does not cover**: any GPU timing; the CUDA kernels as compiled; the papers' experiments sections.
