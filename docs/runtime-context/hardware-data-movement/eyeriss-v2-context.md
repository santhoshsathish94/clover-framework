# Eyeriss v2 and Hierarchical Mesh NoC — Context

Offered, not imposed. Date: 2026-10-26. Direction owner: Santhosh (the creator). Study: `eyeriss-v2`.

## 1. Status

- **E1 (Repo inventory, lint, Verilator/iverilog build & run of all RTL and testbenches, Python golden check):** done
- **E2 (Custom testbenches driving PE 1D row conv and Router all 11 modes + priority/idle behavior):** done
- **E3 (Single-cluster vs 4-cluster 2D convolution, corrupted-weight negative control, problem-size & cluster-count scaling):** done
- **E4 (Does the repository's PE change cycles, MAC enables or SPad reads when data are zero?):** done (PE-level; the first-wave cluster-level plan below was never run)
- **E5 (CSC encoder/decoder and sparse-PE processing order, written from the paper text; round trip, break-even density, emulation vs dense reference):** done
- **E6 (Toy model of the paper's flexible-network argument, with a cross-check against E3 counts):** done; it is my model of the argument, not the authors' analytical model

## 2. What this is

Authors say (Abstract and Section II of Chen, Yang, Emer, Sze, *Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices*, IEEE JETCAS 2019, arXiv:1807.07928v2):
- Eyeriss v2 is a DNN accelerator architecture designed for compact DNNs (widely varying layer shapes and low data reuse, such as MobileNet depthwise and pointwise layers or fully-connected layers at batch size 1) and sparse DNNs (pruned weights and ReLU activations).
- Authors say (Section II, Table II, Section III): It organizes 192 processing elements (PEs) and 192 KB of global buffer (GLB) into an $8 \times 2$ array of 16 clusters ($3 \times 4 = 12$ PEs and 12 KB GLB per cluster), connected by a two-level **Hierarchical Mesh Network (HM-NoC)**: an all-to-all network inside each 12-PE cluster at the lower level and a 2D circuit-switched mesh of router clusters at the top level, configurable per data type (`iact`, `weight`, `psum`) into unicast (high-bandwidth), broadcast (high-reuse), grouped-multicast, or interleaved-multicast modes.
- Authors say (Section IV, Table IV): Each sparse PE uses a 7-stage pipeline, 5 local scratchpads (SPads), and SIMD width 2 (384 MACs total, 8-bit `iact`/`weight`, 20-bit `psum`) to process both weights and input activations directly in Compressed Sparse Column (CSC) format (12-bit `(4b count, 8b data)` pairs plus 7-bit weight column and 4-bit activation segment address pointers), skipping zero activations and zero weights.
- The studied repository (`karthisugumar/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2`, commit `4afea5de159673569c90924d8f3288ec0e09aa07`) is a UCSD CSE 240D Fall 2019 graduate project implementing a scaled-down ($2 \times 2$ clusters of $3 \times 3$ or $5 \times 5$ PEs, 16-bit dense) SystemVerilog prototype of the HM-NoC and Row-Stationary PE cluster targeting Xilinx Vivado.

## 3. Experiments

### E1: Repository inventory, lint, Verilator and iverilog build/run of all testbenches, and Python golden check

- **Question:** From a fresh copy of `karthisugumar/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2`, what is the exact file list, line count, license, and commit hash; which of the 23 unique RTL modules and 25 unique testbenches lint/compile unmodified under `verilator 5.020` and `iverilog 12.0`; what exact Vivado-specific or IEEE-1800 errors prevent the others from building; what is the smallest fix in a fresh copy that makes every testbench build and run; and do their printed outputs match an independent Python golden calculation?
- **Prediction (2026-10-26 12:00 UTC, written before run in `work/predictions.txt`):**
  1. Unmodified under `verilator --lint-only -Wno-lint`: 16 of 23 unique RTL modules will pass; 7 (`PE_cluster.sv`, `router_iact.sv`, `router_weight.sv`, `router_network4.sv`, `HMNoC_cluster.sv`, `HMNoC_cluster_west.sv`, `HMNoC_cluster_east.sv`, `HMNoC_top.sv` — wait: let's check exact count out of 23 unique RTL modules: 15 or 16 pass, 7 or 8 fail) will fail due to: (a) formal parameter shadowing (`input logic [3:0] X_dim, Y_dim` in `PE_cluster.sv:115-118`), (b) mixed blocking `=` and non-blocking `<=` assignments (`BLKANDNBLK`) in `router_iact.sv` and `router_weight.sv`, (c) Vivado hierarchical input port connections `.out_port(other_inst.in_port)` (`ASSIGNIN`) in `router_network4.sv` and `HMNoC_*.sv`, and (d) undeclared port typo `pwest_data_o_wght` in `HMNoC_cluster_east.sv:314`.
  2. Unmodified testbenches under `verilator --binary --timing -Wno-lint`: 7 of 25 unique testbenches (`adder_tb`, `MAC_tb`, `switch_tb`, `SPad_tb`, `glb_iact_tb`, `PE_tb`, `router_glb_tb`) will compile unmodified. Under `iverilog -g2012`, `PE.sv` will fail unmodified because of nested `/*` inside a block comment at line 134.
  3. In `PE_tb`, Iterations 1, 2, 3 will print `14`, `44`, `74` (matching 1D row dot products of `w=[1,2,3]` with `a=[1,2,3]`, `[6,7,8]`, `[11,12,13]`), and then `PE_tb` will hang at `wait(compute_done == 1)` on Iteration 4 because `PE.sv:125` checks `if (iter == act_size - kernel_size + 1)` (`iter == 3`), resets `iter <= 0`, and returns to `IDLE` without asserting `compute_done`.
  4. In Phase 1 `router_cluster_tb`, the output will be `(44, 44, 44)` on Iterations 1–2 and `(20, 27, 27)` on Iteration 3 instead of `(45, 45, 45)` because `router_weight.sv` and `router_iact.sv` increment `r_addr_glb_*` in `READ_GLB` one cycle before asserting `load_en_spad` in `WRITE_SPAD` (skipping index 0), and `act_5x5.txt` has only 20 lines of `1` instead of 25. In Phase 2 `router_cluster_pe_cluster_tb`, the output will match the exact 2D valid convolution `[63, 72, 81, 108, 117, 126, 153, 162, 171]` in 112 cycles.
- **Run:**
  - `research/eyeriss-v2/scripts/e1_a_unmodified_audit.sh` -> `research/eyeriss-v2/results/e1_a_unmodified_audit.out` (`FINISHED rc=0`)
  - `research/eyeriss-v2/scripts/e1_b_ported_and_golden.sh` (calling `work/e1_patch_tbs.py` and `work/e1_golden_check.py`) -> `research/eyeriss-v2/results/e1_b_ported_and_golden.out` (`FINISHED rc=0`)
- **Observation (`results/e1_a_unmodified_audit.out`, `results/e1_b_ported_and_golden.out`):**
  1. **Commit, license, and file inventory:** Pinned commit is `4afea5de159673569c90924d8f3288ec0e09aa07` (`2019-12-13 19:53:44 -0800`, `Final Push`), clean worktree (`0 modified`), **no `LICENSE` or `COPYING` file**. The repo tracks 107 files: 43 `.sv` files under `rtl/` (27 unique RTL module files + 16 MD5-identical copies in `rtl/misc/` and `rtl/phase_1..3/`), 46 `.sv` files under `testbench/` (25 unique testbenches + 3 scratch `.sv` modules + 18 MD5-identical copies in `testbench/phase_1..3/`), 5 text vector files (`act_5x5.txt` 19 newlines/20 lines, `act_7x7.txt` 49 lines, `kernel_3x3.txt` 8 newlines/9 lines, `kernel_5x5.txt` 25 lines, `psum_3x3.txt` 0 bytes), 5 Vivado synthesis files (`constraints_1.xdc` 6 lines, `HMNoC_cluster_utilization_synth.rpt` 195 lines, `HMNoC_top_utilization_synth.rpt` 195 lines, `HMNoC_cluster.vds` 1,449 lines, `HMNoC_top.vds` 16,430 lines), `README.md` (37 lines), `.gitattributes` (2 lines), 1 PDF report (785,195 B), and 12 binary PNG files (not viewed).
  2. **Unmodified RTL lint (`verilator --lint-only -Wno-lint`):** **17 of 27 unique RTL module files pass unmodified** (`FA`, `HA`, `MAC`, `mux2`, `mux_4x1`, `lookup_mux4`, `switch`, `SPad`, `PE`, `glb_iact`, `glb_psum`, `glb_weight`, `GLB_cluster`, `router_psum`, `router`, `router_east`, `router_cluster`). **10 fail unmodified** with exact IEEE-1800 errors:
     - `circular_buffer.sv:82`: `PROCASSWIRE` (`output wire [DATA_WIDTH-1:0] data_o` assigned in `always @(posedge clk)`).
     - `PE_cluster.sv:116`: `psum_adder` function arguments `input logic [3:0] X_dim, Y_dim` shadow module parameters `X_dim, Y_dim` used in `[0 : X_dim*Y_dim-1]`.
     - `router_iact.sv:42,46`, `router_weight.sv:43`, `phase_1/router_cluster.sv:53,56`: `BLKANDNBLK` (mixing `= 0` in `IDLE` with `<=` in sequential `always @(posedge clk)`).
     - `router_network4.sv:121`, `HMNoC_cluster_west.sv:253`, `HMNoC_cluster_east.sv:253` (plus typo `pwest_data_o_wght` at line 314), `HMNoC_top.sv:177`: `ASSIGNIN` from Vivado-specific hierarchical port connections `.out_port(other_inst.in_port)`.
     - `HMNoC_cluster.sv:110`: references `router_cluster_0.read_req_glb_iact` (Phase 1 port name) while resolving `rtl/router_cluster.sv` (Phase 2).
  3. **Unmodified testbenches (`verilator --binary --timing -Wno-lint` and `iverilog -g2012 -Y .sv`):**
     - Under Verilator, **7 of 25 unique testbenches build unmodified**: `adder_tb`, `MAC_tb`, `switch_tb`, `SPad_tb`, `glb_iact_tb`, `PE_tb`, `router_glb_tb`. The remaining 18 fail unmodified due to `ASSIGNIN` hierarchical port connections (11 testbenches), `PE_cluster.sv` parameter shadowing (`PE_cluster_tb`), 1-element unpacked array `[NUM_GLB_IACT-1:0]` passed to scalar port (`GLB_cluster_tb`, `HMNoC_1_tb`), non-existent `.PSUM_READ_ADDR`/`.PSUM_LOAD_ADDR` parameters (`router_weight_tb`, `router_iact_tb`, `router_psum_tb`), and commented-out `ready` ports (`router_east_tb`).
     - Under `iverilog -g2012 -Y .sv`, **4 testbenches build and run unmodified** (`adder_tb`, `MAC_tb`, `SPad_tb`, `glb_iact_tb`); `PE_tb` and `PE_cluster_tb` fail unmodified at `PE.sv:136` (`error: Possible nested comment` from `/*` inside a block comment), while `switch_tb` and `router_glb_tb` fail on `lookup_mux4.sv:30` / `router_east.sv:97` (`unique if` / empty port syntax not supported by `iverilog 12.0`).
  4. **Smallest fixes in a fresh copy (`sim_fixed`) and Python golden verification (`25 / 25` testbenches built and run):**
     - After applying minimal IEEE-1800 wiring/syntax fixes in a fresh copy (`/root/dm-study/eyeriss-v2/w2/sim_fixed`) with zero FSM or datapath logic changes, **all 25 of 25 unique testbenches compile and run (`PASS_CNT = 25 / 25`)**.
     - **Surprise on `PE_tb` (prediction partially missed):** We predicted `PE_tb` would print `[14, 44, 74]` for Iterations 1–3 and hang on Iteration 4 because `PE.sv:125` resets `iter <= 0; state <= IDLE;` without asserting `compute_done` when `iter == 3`. In reality, `PE_tb` completed all 5 iterations and printed **`[14, 44, 74, 14, 44]`**! Inspection of timestamps explained why: `PE_tb` has a 20 ns clock period (`#10` low, `#10` high) and asserts `start = 1; #25; start = 0;` aligned to a rising clock edge. Because `#25` spans **two** rising clock edges (`t` and `t+20`), on Iteration 4 the first rising edge sees `iter == 3` and resets `iter <= 0` in `IDLE`, and the second rising edge (20 ns later, while `start` is still `1` for 5 more ns) immediately launches `iter = 0` (`14`), after which Iteration 5 computes `iter = 1` (`44`).
     - **Golden convolution checks (`work/e1_golden_check.py`):**
       - `PE_cluster_tb` prints `[501, 456, 411]`, `[726, 681, 636]`, `[951, 906, 861]` — matching exact 2D valid convolution `[[411,456,501],[636,681,726],[861,906,951]]` with reversed columns per row because `PE_cluster.sv` declares `output logic [15:0] pe_out[0:X_dim-1]` while `PE_cluster_tb.sv` binds `logic [15:0] pe_out[X_dim-1:0]`.
       - `router_cluster_pe_cluster_tb` reverses the column index back (`pe_out[X_dim-i-1]`) and prints `[63, 72, 81, 108, 117, 126, 153, 162, 171]` in **112 cycles**, matching the exact 2D valid convolution of `1..25` with a $3\times 3$ all-ones kernel.
       - `router_cluster_tb` (Phase 1) prints `[44, 44, 44, 44, 44, 44, 20, 27, 27]` in **115 total cycles** (**33 compute cycles**) instead of `[45]*9`. Our Python bug model reproduced `[44, 44, 44, 44, 44, 44, 20, 27, 27]` exactly: `router_weight.sv` and `router_iact.sv` increment `r_addr_glb_*` in `READ_GLB` one cycle before asserting `load_en_spad` in `WRITE_SPAD` (skipping index 0, so weights loaded are `[2..9, 0]` with sum 44), and `act_5x5.txt` has only 20 lines of `1` instead of 25 (leaving 19 ones and 6 zeros after skipping index 0).
       - `router_pe_4_clusters_tb` completes in **98 cycles** and computes 4 parallel convolutions whose final row outputs (`west0=[4755,4530,4305], west1=[1086,1041,996], east0=[951,906,861], east1=[1041,996,951]`) match Python golden values exactly.
- **What it showed:** The repository ships 27 unique RTL module files and 25 unique testbenches with no license file at commit `4afea5de159673569c90924d8f3288ec0e09aa07`. Seven testbenches build unmodified in Verilator; replacing Vivado-only hierarchical input port connections (`.port(inst.in)`), fixing parameter shadowing in `PE_cluster.sv`, and fixing `BLKANDNBLK` in `router_iact/weight.sv` allows all 25 testbenches to build and run in a clean copy. Phase 2/3 convolution outputs match exact 2D convolution, while Phase 1's `[44,44,44, 44,44,44, 20,27,27]` is fully explained by a 1-cycle address increment before `WRITE_SPAD` plus 5 missing lines in `act_5x5.txt`.

---

### E2: Driving the PE and the Router with custom testbenches

- **Question:** When we drive `PE.sv` (`kernel_size=3, act_size=5`) with our own testbench (`work/tb_e2_pe_row.sv`) using distinct weights `Row0=[10,20,30], Row1=[1,2,3], Row2=[4,5,6]` and activations `Row0=[2,3,4,5,6], Row1=[7,8,9,10,11], Row2=[12,13,14,15,16], Row3=[1]*5, Row4=[2]*5` and a strict 1-cycle `start` pulse, how many cycles does each phase take, what exact SPad addresses are read on every cycle, what outputs are produced, and what happens on the 4th 1-cycle `start` pulse? And when we drive `router.sv` (`work/tb_e2_router_modes.sv`) across all 16 values of `router_mode[3:0]` (`0..15`), each of the 4 input ports individually, simultaneous multi-port inputs, and all-disabled inputs, which output ports receive each word, with what latency, and what does each configuration do?
- **Prediction (2026-10-26 12:20 UTC, written before run in `work/predictions.txt`):**
  1. **PE (`tb_e2_pe_row.sv`):**
     - `LOAD_W` takes **10 clock cycles** (`1 IDLE->LOAD_W + 9 LOAD_W`) to store `mem[0..8]`; `LOAD_A` takes **26 clock cycles** (`1 IDLE->LOAD_A + 25 LOAD_A`) to store `mem[100..124]` and pre-reads `r_addr=0` on completion.
     - Each 1D row convolution (`iter = 0, 1, 2`) takes **11 clock cycles** from the rising edge where `start=1` is sampled in `IDLE` to the rising edge where `compute_done` becomes `1` (`1 IDLE + 3 * (READ_W + READ_A + COMPUTE) + 1 WRITE = 11` cycles = $3K + 2$ cycles for kernel width $K=3$).
     - Because `PE.sv` computes `r_addr <= A_READ_ADDR + filt_count + iter*act_size`, a single PE with `W_READ_ADDR=0, A_READ_ADDR=100` steps vertically across activation rows (`100..102`, `105..107`, `110..112`) against weight row 0 (`[10, 20, 30]`), outputting **`200`**, **`500`**, and **`800`** on `iter = 0, 1, 2` (stored at `SPad mem[500..502]`).
     - With a strict 1-cycle `start` pulse (unlike `PE_tb`'s `#25` 2-edge pulse), the 4th `start` pulse (`iter == 3`) will NOT start a computation: `PE` will stay in `IDLE`, reset `iter <= 0`, keep `compute_done = 0`, and then the 5th 1-cycle `start` pulse will compute `iter = 0` (`200`) in 11 cycles.
     - Negative control: corrupting `w[1]` from `20` to `21` changes outputs to `[203, 508, 813] != [200, 500, 800]`.
  2. **Router (`tb_e2_router_modes.sv`):**
     - Latency is **0 clock cycles** (combinational `always_comb`).
     - Destination enable mask `(N,S,W,E)` across `router_mode = 0..15`: `0(ALL)=1111` (broadcast), `1(NORTH)=1000`, `2(SOUTH)=0100`, `3(WEST)=0010`, `4(EAST)=0001` (4 unicast modes), `5(EASTNORTH)=1001`, `6(EASTSOUTH)=0101`, `7(EASTWEST)=0011`, `8(WESTNORTH)=1010`, `9(WESTSOUTH)=0110`, `10(WESTEAST)=0011` (6 two-destination multicast modes, where `7` and `10` are identical horizontal multicast), and `11..15(default)=0000` (all disabled).
     - Priority when multiple `*_enable_i` are `1`: `north (0x1111) > south (0x2222) > west (0x3333) > east (0x4444)`.
     - When all `*_enable_i == 0`, `data_out` becomes `16'd10101` (`0x2775`), while `*_enable_o` remains `1` on the mode's active destination ports.
- **Run:**
  - `research/eyeriss-v2/scripts/e2_pe_and_router.sh` (building & running `work/tb_e2_pe_row.sv` under both Verilator 5.020 and iverilog 12.0, and `work/tb_e2_router_modes.sv` under Verilator 5.020) -> `research/eyeriss-v2/results/e2_pe_and_router.out` (`FINISHED rc=0`)
- **Observation (`results/e2_pe_and_router.out`):**
  1. **PE (`tb_e2_pe_row.sv`) — exact hit on all cycle counts, addresses, outputs, and 1-cycle `start` wrap behavior:**
     - `LOAD_W` took **10 clock cycles** (`LOAD_W_cycles=10`) to write 9 weights into `SPad mem[0..8]`; `LOAD_A` took **26 clock cycles** (`LOAD_A_cycles=26`) to write 25 activations into `SPad mem[100..124]` and pre-loaded `r_addr=0` (`r_data=10`) on the final cycle.
     - Each 1D row convolution (`iter = 0, 1, 2`) took **11 clock cycles** (`1 IDLE + 3 * (READ_W + READ_A + COMPUTE) + 1 WRITE = 11` cycles, i.e., **3 cycles per MAC** plus 2 overhead cycles):
       - `ITER 0` (`cyc 39..48`): read `r_addr = 0, 100, 1, 101, 2, 102` (`w=[10,20,30]`, `a=[2,3,4]`), accumulated `psum_reg = 20 -> 80 -> 200`, wrote `spad_psum[500] = 200`, `pe_out = 200`, `cycles = 11`.
       - `ITER 1` (`cyc 51..60`): read `w=[10,20,30]`, `a=[7,8,9]` (`r_addr = 105..107`), accumulated `70 -> 230 -> 500`, wrote `spad_psum[501] = 500`, `pe_out = 500`, `cycles = 11`.
       - `ITER 2` (`cyc 63..72`): read `w=[10,20,30]`, `a=[12,13,14]` (`r_addr = 110..112`), accumulated `120 -> 380 -> 800`, wrote `spad_psum[502] = 800`, `pe_out = 800`, `cycles = 11`.
     - On `PULSE_4` (a strict 1-cycle `start` pulse when `iter == 3 == act_size - kernel_size + 1`), `PE` transitioned `iter_before=3 -> iter_after=0, state=0 (IDLE), compute_done=0` in 1 cycle without computing; then `PULSE_5` computed `iter = 0` (`pe_out = 200`) in `11` cycles. Both `verilator` and `iverilog` produced identical output (`diff lines: 0`).
     - Negative control (`corrupt_w1=1`, changing `w[1]` from `20` to `21`): produced `[203, 508, 813] != [200, 500, 800]`, caught by the golden assertion.
  2. **Router (`tb_e2_router_modes.sv`) — exact hit on all 16 modes & idle behavior; surprise on multi-input priority due to `unique if`:**
     - Across all 16 `router_mode` values (`0..15`) and all 4 single-source inputs (`0x1111, 0x2222, 0x3333, 0x4444`), routing is purely combinational (**0 clock cycles** latency) with exact destination enable masks `(N,S,W,E)`:
       - **Broadcast (1 mode):** `0:ALL` -> `1111` (all 4 output ports receive `data_out`).
       - **Unicast (4 modes):** `1:NORTH` -> `1000`, `2:SOUTH` -> `0100`, `3:WEST` -> `0010`, `4:EAST` -> `0001` (only the selected direction receives `data_out` and `enable_o=1`).
       - **Two-destination Multicast (6 modes):** `5:EASTNORTH` -> `1001`, `6:EASTSOUTH` -> `0101`, `7:EASTWEST` -> `0011` (horizontal multicast), `8:WESTNORTH` -> `1010`, `9:WESTSOUTH` -> `0110`, `10:WESTEAST` -> `0011` (identical to `7:EASTWEST`). Note there is no `NORTHSOUTH` (vertical-only multicast without East/West local cluster delivery) mode in `router.sv`: vertical forwarding to a neighbor cluster always simultaneously enables either `EAST` (`EASTNORTH`/`EASTSOUTH`) or `WEST` (`WESTNORTH`/`WESTSOUTH`).
       - **Disabled (5 modes):** `11..15` (`default`) -> `0000`.
     - **Idle behavior (`in_en=0000`):** When all 4 input enables are `0`, `data_out` is driven to `16'd10101` (`0x2775`) while `*_enable_o` STILL remains `1111` (in `ALL` mode) because `*_enable_o` in `router.sv` depends only on `router_mode`, never on `*_enable_i`.
     - **Surprise on simultaneous multi-input enables (`in_en=1111, 0111, 0011`):** We predicted `north > south > west > east` priority from reading the `if (north_enable_i) ... else if (south_enable_i) ...` chain in `router.sv:90-99`. In reality, Verilator output `0x4444` (`EAST`) for all four combinations (`1111`, `0111`, `0011`, `0001`)! Re-reading `router.sv:90` showed why: the author wrote `unique if(north_enable_i)`. Under IEEE 1800-2017 Section 12.4.2, `unique if` asserts that the conditions are mutually exclusive (parallel case), allowing the compiler to evaluate branches in parallel/reverse order rather than as a priority encoder.
- **What it showed:** `PE.sv` is a non-pipelined 7-state FSM taking $K^2+1$ cycles to load weights, $H^2+1$ cycles to load activations, and $3K+2$ cycles per 1D row dot product (3 cycles/MAC: `READ_W -> READ_A -> COMPUTE`). `router.sv` is a 0-cycle combinational crossbar with 1 broadcast mode, 4 unicast modes, and 6 two-port multicast modes (`7` and `10` identical, no pure `NORTHSOUTH` mode), whose `*_enable_o` outputs are unconditionally asserted by `router_mode` even when all input enables are 0, and whose `unique if` input selector assumes at most one `*_enable_i` is active at a time.

---

### E3: One cluster and four clusters running a convolution, problem-size scaling, and cluster-count scaling

- **Question:** How do the repository's 1-cluster (`router_cluster_pe_cluster_tb`, `HMNoC_cluster_west_tb`) and 4-cluster (`router_pe_4_clusters_tb`) convolution testbenches compare in cycle count, words transferred per cluster, and MACs performed; does a corrupted weight fail the golden check; how do cycles change when we vary problem size (`3x3` on `5x5`, `3x3` on `6x6`, `3x3` on `7x7`, and `5x5` on `7x7`) and when we run a fixed 4-channel convolution workload across `1`, `2`, and `4` clusters?
- **Prediction (2026-10-26 12:35 UTC, written before run in `work/predictions.txt`):**
  1. **Repo testbenches (`3x3` kernel on `5x5` act vs `5x5` kernel on `7x7` act):**
     - `router_cluster_pe_cluster_tb` (1 cluster, $3\times 3$ on $5\times 5$) takes **112 cycles** (`35` cycles writing 9 weights + 25 iacts into GLB, `38` cycles loading 9 weights + 25 iacts from GLB through `router_cluster` into the 9 PEs, and `39` cycles for 3 compute iterations + inter-iteration waits), performing $9\text{ PEs} \times 3\text{ MACs/iter} \times 3\text{ iters} = \mathbf{81\text{ MACs}}$. `HMNoC_cluster_west_tb` has the exact same datapath and will also take **112 cycles** and produce the same `[63, 72, 81, 108, 117, 126, 153, 162, 171]`.
     - `router_pe_4_clusters_tb` (4 clusters in parallel, $3\times 3$ on $5\times 5$, no GLB write stage) takes **98 cycles**: `10` idle + `9` weight-broadcast cycles (9 weight words sent once into `west_0` and received simultaneously by all 4 clusters = 36 cluster-words delivered in 9 cycles) + `10` idle + `25` iact-unicast cycles (4 distinct 25-word activation maps injected in parallel into `west_0, west_1, east_0, east_1` = 100 cluster-words delivered in 25 cycles) + `5` idle + `39` compute cycles (all 4 clusters run 81 MACs each in lockstep = **324 MACs** total).
     - For `5x5` kernel on `7x7` act ($25\text{ PEs/cluster}$): 1-cluster (`router_cluster_pe_cluster_5x5_tb`) takes **251 cycles** (`75` GLB write + `78` SPad load + `98` compute/wait, **375 MACs**), while 4-cluster (`router_pe_4_clusters_5x5_tb`) takes **197 cycles** (`10` idle + `25` weight-broadcast + `10` idle + `49` iact-unicast + `5` idle + `98` compute/wait, **1,500 MACs** across 4 clusters).
  2. **Corrupted-weight negative control:** Corrupting a single weight `w[4]` (center of $3\times 3$ kernel) by $+1$ shifts every output element `(r, c)` by $+a[r+1, c+1]$ and causes the Python golden check to fail immediately.
  3. **Controlled scaling experiments (`work/tb_e3_scaling.sv`):**
     - **Problem-size scaling on 1 cluster (no testbench idle padding, 1-cycle `start` gap):** For a $K\times K$ kernel on an $H\times H$ activation map with $E = H - K + 1$ and $E\times K$ PEs, active cycles are exactly:
       $$C_{\text{active}}(K, H) = (K^2 + 1)_{\text{load\_w}} + (H^2 + 1)_{\text{load\_a}} + E \cdot (3K + 3)_{\text{compute}}.$$
       Specifically:
       - $(K=3, H=5, E=3)$: $10 + 26 + 3\times 12 = \mathbf{72\text{ cycles}}$ (`81` MACs).
       - $(K=3, H=6, E=4)$: $10 + 37 + 4\times 12 = \mathbf{95\text{ cycles}}$ (`144` MACs).
       - $(K=3, H=7, E=5)$: $10 + 50 + 5\times 12 = \mathbf{120\text{ cycles}}$ (`225` MACs).
       - $(K=5, H=7, E=3)$: $26 + 50 + 3\times 18 = \mathbf{130\text{ cycles}}$ (`375` MACs).
     - **Cluster-count scaling on a fixed 4-channel workload ($K=3, H=5$, 1 shared $3\times 3$ weight kernel + 4 distinct $5\times 5$ activation maps = 324 MACs total):**
       With weights loaded/broadcast once (`10` cycles) and each pass loading `26` iact cycles + `36` compute cycles (`62` cycles/pass):
       - **1 cluster (4 serial passes):** $10 + 4 \times 62 = \mathbf{258\text{ cycles}}$.
       - **2 clusters (2 parallel passes):** $10 + 2 \times 62 = \mathbf{134\text{ cycles}}$ ($1.93\times$ speedup).
       - **4 clusters (1 parallel pass):** $10 + 1 \times 62 = \mathbf{72\text{ cycles}}$ ($3.58\times$ speedup).
- **Run:**
  - `research/eyeriss-v2/scripts/e3_clusters_and_scaling.sh` (running repo 1-cluster and 4-cluster testbenches plus `work/tb_e3_scaling.sv` and Python golden verification) -> `research/eyeriss-v2/results/e3_clusters_and_scaling.out` (`FINISHED rc=0`)
- **Observation (`results/e3_clusters_and_scaling.out`):**
  1. **Why 4-cluster repo testbenches take fewer cycles than 1-cluster repo testbenches (`98` vs `112` on $3\times 3$, `197` vs `251` on $5\times 5$):**
     - `router_cluster_pe_cluster_tb` (1 cluster, $3\times 3$ kernel on $5\times 5$ act) takes **112 cycles** (`router_cluster_pe_cluster_5x5_tb` takes **251 cycles**), whereas `router_pe_4_clusters_tb` (4 clusters in parallel, $3\times 3$ on $5\times 5$) takes **98 cycles** (`router_pe_4_clusters_5x5_tb` takes **197 cycles**).
     - The 14-cycle difference on $3\times 3$ (`112` vs `98`) and 54-cycle difference on $5\times 5$ (`251` vs `197`) comes from how the testbenches feed data:
       - In `router_cluster_pe_cluster_tb` (1 cluster), the testbench first spends **35 cycles** writing 9 weights and 25 activations into `GLB_cluster_0`, then **38 cycles** reading those 9 weights and 25 activations out of `GLB_cluster_0` through `router_cluster_0` into `pe_cluster_0`, and **39 cycles** computing 3 rows (`81` MACs total).
       - In `router_pe_4_clusters_tb` (4 clusters), `GLB_cluster` is omitted: the testbench drives `network_wght` and `network_iact` directly (`10` idle + **9 cycles** broadcasting 9 weights from `west_0` to all 4 clusters simultaneously = 36 weight words delivered in 9 cycles + `10` idle + **25 cycles** unicasting 4 distinct 25-word activation streams into `west_0, west_1, east_0, east_1` in parallel = 100 activation words delivered in 25 cycles + `5` idle + **39 compute cycles**). All 4 clusters execute 81 MACs in lockstep (**324 MACs** total in 39 compute cycles; **1,500 MACs** total in `router_pe_4_clusters_5x5_tb`).
  2. **Three surprises uncovered during E3:**
     - **Surprise 1 (`HMNoC_cluster_west_tb.sv`):** We predicted `HMNoC_cluster_west_tb` would take 112 cycles and print `[63..171]`. In reality (`HMNoC_cluster_west_tb: cycles=NONE`), reading `HMNoC_cluster_west_tb.sv:515-545` showed the testbench ends right after loading activations into the PE SPad — it never asserts `start = 1`, never waits for `compute_done`, and has no `$stop` or `$finish`.
     - **Surprise 2 (`PE.sv` multi-pass state retention between passes on the same cluster):** We predicted 1-cluster (4 passes) and 2-cluster (2 passes) would take `258` and `134` cycles by loading weights once (`10` cycles) and running `LOAD_A` (`26` cycles) + compute (`36` cycles) on subsequent passes. When we ran that in `tb_e3_scaling.sv`, `pass=1` hung! Tracing `PE.sv` revealed two stateful bugs when running back-to-back passes without `reset`: (a) `PE.sv:146` (`else if (load_en_act)`) transitions `IDLE -> LOAD_A` **without** resetting `filt_count <= 0` (unlike `load_en_wght` at line 144), so after a compute pass leaves `filt_count == kernel_size (3)`, a subsequent `LOAD_A` starts at `filt_count = 3` and exits 3 cycles early; and (b) after $E=3$ compute iterations, `iter` sits at `3` (`act_size - kernel_size + 1`), requiring a 1-cycle `start=1` pulse in `IDLE` (`PE.sv:125`) to wrap `iter` back to `0` plus a 10-cycle weight reload to reset `filt_count <= 0`. With those two hardware requirements accounted for (`72` cycles on Pass 0 and `1 + 10 + 26 + 36 = 73` cycles on each subsequent pass), measured active cycles are:
       - **1 cluster (4 passes):** $72 + 3 \times 73 = \mathbf{291\text{ cycles}}$ (`324` MACs).
       - **2 clusters (2 passes):** $72 + 1 \times 73 = \mathbf{145\text{ cycles}}$ ($2.01\times$ speedup over 1 cluster).
       - **4 clusters (1 pass):** $\mathbf{72\text{ cycles}}$ ($4.04\times$ speedup over 1 cluster).
     - **Surprise 3 (`61329` on un-driven clusters in `N=1` and `N=2` modes):** When `n_clusters=1` or `2`, the un-driven clusters (`w1, e1`) printed `[61329, 61329, 61329]`. This cross-confirmed E2's router finding: when a cluster's iact router has `enable_i = 0`, `router.sv:99` drives `data_out = 10101` during all 25 cycles of `LOAD_A`, and $45 \times 10101 = 454,545 \equiv \mathbf{61329} \pmod{65536}$!
  3. **Problem-size scaling on 1 cluster (`SIZE_SUMMARY` in `results/e3_clusters_and_scaling.out`) — 100% match with formula $C_{\text{active}}(K, H) = (K^2+1) + (H^2+1) + E(3K+3)$:**
     - $(K=3, H=5, E=3, 9\text{ PEs})$: `load_w=10, load_a=26, compute=36, total=72` cycles (`81` MACs), all $3\times 3$ outputs match Python golden.
     - $(K=3, H=6, E=4, 12\text{ PEs})$: `load_w=10, load_a=37, compute=48, total=95` cycles (`144` MACs), all $4\times 4$ outputs match Python golden.
     - $(K=3, H=7, E=5, 15\text{ PEs})$: `load_w=10, load_a=50, compute=60, total=120` cycles (`225` MACs), all $5\times 5$ outputs match Python golden.
     - $(K=5, H=7, E=3, 15\text{ PEs})$: `load_w=26, load_a=50, compute=54, total=130` cycles (`375` MACs), all $3\times 3$ outputs match Python golden.
  4. **Negative control (`corrupt_w4=1`, changing center weight $w[4]$ from $5$ to $6$):** Shifts `w0` row 2 from `[4305, 4530, 4755]` to `[4390, 4620, 4850]` ($+[85, 90, 95] = 5 \times [17, 18, 19]$), matching the corrupted golden formula and failing the normal golden check.
- **What it showed:** Across 4 clusters, broadcasting shared weights (`ALL` + `EASTSOUTH` + `WEST` + `EAST`) while unicasting 4 independent activation streams delivers $4 \times 9 = 36$ weight words in 10 active cycles and $4 \times 25 = 100$ activation words in 26 active cycles, completing 4 convolutions (324 MACs) in 72 active cycles vs 145 cycles on 2 clusters and 291 cycles on 1 cluster ($4.04\times$ speedup).

---

### E4: What is dense and what is not in the RTL (zero weights and activations)

*Note: the question and prediction below are the first-wave plan at the `PE_cluster` level; it was not run. What I ran is a PE-level version (see "As actually run" at the end of this section). The cluster is made of these PEs, and `PE.sv`'s FSM is the only place a zero could change timing.*

- **Question:** Does the repository's RTL contain any zero-gating, zero-skipping, or compressed-format logic? When we drive `PE_cluster` (`3x3` PEs, $3\times 3$ kernel on $5\times 5$ activations = 81 scheduled MACs) across four sparsity regimes — (1) 0% zeros (dense), (2) ~55% zero weights (5/9 zero) and 52% zero activations (13/25 zero), (3) 89% zero weights (8/9 zero) and 80% zero activations (20/25 zero), and (4) 100% zeros (all weights and activations zero) — how many actual MAC clock enables (`mac_en == 1`), how many non-zero operand MACs (`mac_en && (filt_in_reg != 0) && (act_in_reg != 0)`), and how many load and compute cycles occur in each regime?
- **Prediction (2026-10-26 12:50 UTC, written before run in `work/predictions.txt`):**
  1. Searching all `.sv` files in `rtl/` for sparse/compression/zero-gating logic (`csc`, `count`, `skip`, `sparse`, `compress`, or operand `== 0` checks in `PE.sv` / `MAC.sv`) will find **none**: the RTL is 100% dense.
  2. Across all four sparsity regimes (`R1: 0% zeros`, `R2: ~50% zeros`, `R3: ~85% zeros`, `R4: 100% zeros`):
     - Scheduled MACs = **81**.
     - Actual hardware `mac_en == 1` pulses across the 9 PEs = **81** in every regime (`81, 81, 81, 81`), even when non-zero operand pairs `(w != 0 && a != 0)` drop from `81` (`R1`) to `16` (`R2`), `2` (`R3`), and `0` (`R4`).
     - Load cycles (`LOAD_W = 10`, `LOAD_A = 26`) and compute cycles (`36` active cycles) = **72 total active cycles** in every regime (`72, 72, 72, 72`), showing 0 cycles skipped and 0 MACs clock-gated on zero data.
- **Run:** not run at cluster level.
- **Observation:** none at cluster level.
- **What it showed:** nothing at cluster level.

**As actually run (PE level, by me):**
- **Question:** With the repository's unmodified `PE.sv` (the same driver as E2: `kernel_size=3`, `act_size=5`, strict 1-cycle `start`), does anything change per 1-D row iteration when weights or activations are zero?
- **Prediction** (recorded in `work/predictions.txt` before the run): every pattern gives `11` cycles, `3` posedges with `mac_en=1`, `6` posedges with `read_en=1` in `READ_W`/`READ_A`, and `pe_out` equal to the dot product; the count of non-zero pairs varies from 3 to 0 while the three RTL counters do not; no CSC/skip/SIMD code in `rtl/`; 1 SPad, 1 MAC, 16-bit.
- **Run:** [e4_zero_handling.sh](../../../research/hardware-data-movement/eyeriss-v2/scripts/e4_zero_handling.sh) with [tb_e4_pe_zero.sv](../../../research/hardware-data-movement/eyeriss-v2/work/tb_e4_pe_zero.sv) -> [e4_zero_handling.out](../../../research/hardware-data-movement/eyeriss-v2/results/e4_zero_handling.out).
- **Observation:** six data patterns (dense; `w=[0,20,0]`; `w=[0,0,0]`; every other activation zero; all activations zero; `w=[0,20,0]` with every other activation zero), three iterations each: in all 18 iterations `cycles=11`, `mac_en=3`, `spad_reads=6`, and `pe_out` equals the expected dot product (e.g. `200/500/800` dense, `60/160/260` for `w=[0,20,0]`, `0` for all-zero weights). The number of pairs where both operands are non-zero was `3`, `1`, `0`, `2/1/2`, `0`, `0/1/0` for the same iterations. The grep over `rtl/` found no CSC, run-length, skip or SIMD code; the four hits were counter comparisons (`filt_count == 0` selecting the accumulator input at `PE.sv:195`, `psum_count == 0` in `router_psum.sv:108`), which I read. `PE.sv` has `1` SPad instance, `1` MAC instance, `DATA_WIDTH = 16`.
- **What it showed:** The repository's PE is dense: cycles, MAC enables and SPad reads are independent of the data (`11 / 3 / 6` per row), while the number of useful (non-zero, non-zero) pairs ranges from 3 to 0. That is the difference between this PE and what the paper describes (Section IV: only non-zero iacts are read and only the matching non-zero weights are fetched); this is about the community repository's scope (the README describes a scaled-down dense RS and HM-NoC prototype), not about the paper's chip.

---

### E5: The compressed-sparse-column format and the sparse PE's processing order, from the paper text

- **Question:** Section IV of the paper describes CSC coding (4-bit count + 8-bit data per non-zero, address vectors per column / per segment) and a PE that reads a non-zero iact first, then only the non-zero weights of the matching column. If I implement that from the text, (1) does the format round-trip exactly and where does it stop paying off against 8-bit dense storage; (2) does a PE that processes in this order give the same psums as a dense reference, and how do MACs, SIMD-2 cycles and reads vary with the two densities?
- **Prediction** (in `work/predictions.txt`; note there that the file entry was written just after the run because my first write failed — the text had been composed before): round trip exact over 400 random sets and a changed count changes the decode; break-even density for a $32\times15$ weight matrix $d^*=(8\cdot32\cdot15-7\cdot16)/(12\cdot32\cdot15)=0.647$ with the empirical crossing within `0.02`; a single non-zero at the end of a 256-long segment needs `15` padding entries + `1` data entry; the emulation equals dense in every cell; MACs/dense within `0.03` of $d_w d_i$; SIMD-2 cycles between `MACs/2` and `MACs/2 + (non-zero visits)/2`.
- **What is mine, not the paper's** (full list in the script header): count restarts at each segment; a gap above 15 is bridged with padding entries (count 15, data 0, advancing 16 positions); iact flatten order `c*U+u`; weight column `c*S+s`; the emulation has $U=1$ only.
- **Run:** `python work/e5_csc.py` -> [e5_csc.out](../../../research/hardware-data-movement/eyeriss-v2/results/e5_csc.out) (local; pure numpy, fixed seed `20261026`). Code: [e5_csc.py](../../../research/hardware-data-movement/eyeriss-v2/work/e5_csc.py).
- **Observation:**
  1. 400 random segment sets decode exactly; changing one count changes the decode.
  2. Compression of a $32\times15$ weight matrix (200 random matrices per density; dense `3,840` bits): ratio `0.070` (density 0.02), `0.193` (0.1), `0.479` (0.3), `0.785` (0.5), `0.928` (0.6), `0.999` (0.647), `1.084` (0.7), `1.529` (1.0). Empirical crossing of ratio 1.0 at density `0.6474`; formula `0.6472`. Below density 0.1 the measured ratio is above the no-padding formula (e.g. `0.070` vs `0.059` at 0.02) because of padding entries (mean `3`-`5` per matrix at densities 0.02-0.1; `0` from 0.3 up) — I had predicted padding would be rare there.
  3. A 256-long segment with one non-zero at the end: `16` entries (`15` padding), decodes exactly, `206` bits vs `2,048` dense (the formula floor(255/16) gave 15).
  4. Sparse-PE emulation ($M_0=32$, $C_0=4$, $S=3$, $W=40$, 38 windows, dense `14,592` MACs, 5 seeds per cell, 12 cells): output equals the dense reference in all 60 runs; MACs/dense tracks $d_w d_i$ (e.g. `0.3056` vs `0.300`, `0.1562` vs `0.150`; the largest gap was `0.032` at $d_w=1.0, d_i=0.6$ — one cell outside my `0.03` guess, with only 5 seeds); SIMD-2 cycles lie between `MACs/2` and `MACs/2 + visits/2` in every cell, e.g. `3,815` for `7,372` MACs at $d_w=0.5, d_i=1.0$. At $d_w=0.1$ the odd-column padding of SIMD-2 matters most (`927` cycles vs `844` = MACs/2 at $d_i=1.0$); weight reads per MAC `1.000` down to `1.107` at low $d_w$ (padding entries are read); iacts dropped because their weight column was empty appear only at $d_w=0.1$ (`8.8` and `2.2` per run at $d_i=0.6$ and $0.3$).
  5. Moving one non-zero weight by one row changes the emulation output.
- **What it showed:** In the format as I read it, a 12-bit pair per non-zero beats 8-bit dense only below about 65% density for this matrix shape (address vector included), the count field's 15-position limit costs extra entries once columns are very sparse, and processing in the order the paper describes is exact while the MAC count follows the product of the two densities. Whether the authors' count-overflow rule and segment layout match mine is not known from the text.

---

### E6: The paper's argument for a flexible two-level network, as arithmetic

- **Question:** The paper argues that a network with fixed multicast patterns (1-D horizontal weight multicast, 1-D vertical iact multicast) leaves PEs idle when a layer lacks output-channel or position parallelism (depth-wise, fully connected at batch 1), and that HM-NoC's modes avoid this. In a toy model with the assumptions listed in the script header, what utilisation does a rigid $12\times16$ array reach on six layer types, what does a flexible mapping reach, and what does the 24-bit port width impose?
- **What this is:** my model of the argument (Section III, `paper_main.tex` lines 297-330); the authors' own analytical model (mapping search with bandwidth limits) was not available to me, so nothing here reproduces their Fig. 14. Layer shapes are illustrative numbers I chose.
- **Prediction** (in `work/predictions.txt`, written before the run): rigid utilisation `100%` (CONV large, few channels, pointwise), `8.3%` (depth-wise), `8.3%` (FC batch 1), `33.3%` (FC batch 4); flexible `100%` everywhere uncapped; with a 144 words-per-cycle cap, `75%` for depth-wise and FC batch 1.
- **Run:** `python work/e6_noc_model.py` -> [e6_noc_model.out](../../../research/hardware-data-movement/eyeriss-v2/results/e6_noc_model.out). Code: [e6_noc_model.py](../../../research/hardware-data-movement/eyeriss-v2/work/e6_noc_model.py).
- **Observation:** all predicted values came out as predicted (table in the output file). The flexible mapping's implied modes are: CONV and pointwise `12x16` (iact multicast $\times12$ / weight multicast $\times16$), depth-wise `1x192` (iacts unicast, weights broadcast), FC batch 1 `192x1` (iacts broadcast, weights unicast), FC batch 4 `48x4`. Source words per cycle needed: `28` for CONV, `193` for depth-wise or FC batch 1. Cross-check against E3: weights `9` injected $\times$ fan-out `4` = `36` delivered; iacts `100` $\times$ `1` = `100`. **These two E3 numbers were computed from the testbench's schedule, not counted by an RTL counter**, so the cross-check shows consistency with the schedule only.
- **What it showed:** The arithmetic of the argument holds in the toy: a fixed multicast shape reaches 8.3% on depth-wise and FC batch 1 where a free choice of (channels $\times$ positions) reaches 100%, at the price of up to 193 source words per cycle, which exceeds the 144 that I took from the port counts. Because the modes in the model were chosen from the paper's own examples, it checks arithmetic, not the paper's claim about the chip.

## 4. How it works

(From my own reading of `PE.sv`, `router.sv`, the README and Sections III-IV of the paper, and from E1-E6. The repository is a different, much smaller thing than the chip in the paper; the table at the end of this section lists the differences.)

**The row-stationary PE in the repository** (`PE.sv`, read in full). One scratchpad (`SPad`) holds weights from address `0`, activations from `100`, partial sums from `500`. A 7-state FSM (`IDLE, READ_W, READ_A, COMPUTE, WRITE, LOAD_W, LOAD_A`) runs one 1-D row convolution: for each of `kernel_size` taps it reads a weight, reads an activation, multiplies and accumulates (3 cycles per MAC), then writes the sum; `iter` selects the activation row (`A_READ_ADDR + iter*act_size`); after `act_size-kernel_size+1` iterations a further `start` only resets `iter`. Measured (E2): `LOAD_W` `10` cycles for 9 weights, `LOAD_A` `26` for 25 activations, `11` cycles per row ($3K+2$). Two behaviours found by running: the 4th `start` after three rows is consumed to reset `iter`, and `LOAD_A` does not reset `filt_count`, so a second pass needs a weight reload first (E3).

**The cluster** (E1, E3): a $3\times3$ or $5\times5$ grid of these PEs with a `router_cluster`, GLB banks and, in the 4-cluster version, `router_network4`. The convolution matched exact 2D convolution on all runs (E1, E3); cycles for one cluster follow $C(K,H)=(K^2+1)+(H^2+1)+E(3K+3)$ (E3: `72`, `95`, `120`, `130`).

**The router** (`router.sv`, read in full; E2). A combinational 4-port crossover (`N,S,W,E`): `router_mode` selects broadcast (`ALL`), one of four unicasts, or one of six two-destination multicasts; the output enables depend only on the mode, and data come from the first enabled input via a `unique if` (E2: with several inputs enabled, Verilator selected the `EAST` input; I did not test another simulator). With no input enabled the data output is the constant `10101` (E2), which showed up as `61329` in un-driven clusters in E3.

**HM-NoC in the paper** (Section III, read): a separate network per data type; inside a 12-PE cluster an all-to-all network, between clusters a 2-D mesh; circuit-switched, routing mode set by configuration bits; four modes (high bandwidth = unicast, high reuse = broadcast, grouped-multicast, interleaved-multicast); 24-bit ports carry three 8-bit values; iact routers have 4 source and 4 destination ports, weight routers 2 and 2, psum routers 3 and 3 (psum accumulation only vertically).

**Sparse PE in the paper** (Section IV, read; E5 implements the order): CSC with a 4-bit count and an address vector; iacts in segments of $C_0\times U$, weights in columns of $M_0$; the PE reads an iact first and then only the weights of the matching column, so the only MACs are (non-zero, non-zero) pairs; 7 pipeline stages, 5 SPads, SIMD 2.

| | Paper (chip) | Repository (E1-E4) |
|---|---|---|
| PE | 7-stage pipeline, 5 SPads, SIMD 2, CSC on both operands | 7-state FSM, 1 SPad, 1 MAC, dense; 3 cycles per MAC (E2, E4) |
| Data | 8-bit iact/weight, 20-bit psum | 16-bit |
| Cluster | $3\times4$ PEs, $8\times2$ clusters | $3\times3$ or $5\times5$ PEs, 4 clusters |
| Network | configurable modes incl. grouped and interleaved multicast, ready/enable handshake | 11 fixed router modes (E2), ready signals commented out in the router |

## 5. What surprised me and what I got wrong
- **`PE_tb` did not hang (E1)**: I (first wave) predicted a hang at iteration 4; the testbench's 25 ns `start` spans two clock edges, so the 4th iteration wraps `iter` and the second edge starts a new one; outputs `[14, 44, 74, 14, 44]`.
- **Router priority (E2)**: predicted `north > south > west > east`; Verilator selected `EAST` for all multi-input cases because the source uses `unique if`. The explanation (uniqueness assertion; behaviour undefined when violated) is the standard reading; the specific choice of `EAST` is one simulator's.
- **`HMNoC_cluster_west_tb` never computes (E3)**: it ends after loading, with no `start` and no `$finish`.
- **Multi-pass on one cluster (E3)**: `PE.sv` keeps `filt_count` and `iter` between passes; I predicted `258` cycles for the 1-cluster 4-pass case and measured `291` after allowing for that.
- **Phase 1 router bug (E1)**: `[44,44,44,...]` instead of `45`: address increment one cycle before write (skips index 0) plus a stimulus file with 5 fewer ones than expected.
- **My own E5 guesses**: padding entries are not rare at low density, and one cell of the sweep was outside my `0.03` tolerance (statistical, 5 seeds).
- **The cluster-level E4 was never run**; I replaced it with a PE-level one rather than leave an unfilled plan.
- **E3's `36` and `100` words** (delivered per cluster) are schedule arithmetic, not counters (E6).

## 6. Authors say / code does / I observed
| Claim | Where | What I observed |
|---|---|---|
| Flexible two-level network: all-to-all in a cluster, mesh between clusters; unicast / broadcast / grouped / interleaved multicast | Paper Section III | the repository's router has broadcast, 4 unicast and 6 two-port multicast modes with 0-cycle combinational latency (E2); 4 clusters ran broadcast weights + unicast iacts and gave the exact convolutions (E3). Grouped and interleaved multicast across a $8\times2$ array: not present |
| Row-stationary PE with SPads | Paper Section IV | one SPad, FSM, 3 cycles per MAC (E2); exact vs golden (E1, E3) |
| Both operands kept compressed and zeros skipped; SIMD 2 | Paper Section IV | none of it in the repository (E4: cycles, MAC enables and reads independent of data); my CSC emulation (E5) shows the order is exact and MACs follow the product of densities |
| 12-bit count-data pair; best count width 4 bits | Paper Section IV | break-even density `0.647` for a $32\times15$ matrix under my overflow rule (E5); the 4-bit choice itself not tested (needs their data) |
| Performance scales linearly 256 to 1024 PEs, >85% of linear at 16384 PEs | Paper Section III-D | not tested: their analytical model was not available; the repository's 1/2/4-cluster runs gave `291/145/72` cycles on a small fixed workload (E3), which is a different measurement |
| NoC <3% of area, 6-10% of energy | Paper Section V | not examined (the repository ships Vivado reports I did not read) |
| Repository: scaled-down RS + HM-NoC prototype | README | consistent with E1-E4 |

## 7. Not run, and why. Unknowns
- Energy, power, area, throughput of the chip: not examined. The repository ships synthesis files (`.vds`, utilisation `.rpt`) and a PDF report; I did not read them.
- The paper's analytical mapper and its scalability figure: not available; E6 is a toy.
- Grouped and interleaved multicast, the full $8\times2$ cluster array, the psum NoC across clusters: not exercised.
- SIMD, 7-stage pipeline, 5-SPad PE: no RTL available; E5 emulates the processing order in Python only.
- The router's behaviour with several enabled inputs in Vivado or iverilog: not tested.
- The `.png` figures in the repository were not viewed.

## 8. Sources and Sources read
- Paper: Y.-H. Chen, T.-J. Yang, J. Emer, V. Sze, *Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices*, IEEE JETCAS 2019, arXiv:1807.07928v2.
- Repository: `karthisugumar/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2`, commit `4afea5de159673569c90924d8f3288ec0e09aa07` (2019-12-13), no license file.

### Sources read
- By me: `paper_main.tex` lines 297-556 (Section III network, Section IV sparse PE and SIMD); `rtl/PE.sv` and `rtl/router.sv` in full; `README.md` in full; the PE-level testbenches `tb_e2_pe_row.sv` and `tb_e4_pe_zero.sv`, and the patch steps of the E1b/E2/E3 scripts.
- By the first agent (not re-read by me in full): the rest of the paper; the other RTL files and testbenches.

## 9. Files
- Context file: `clover-kimi-k3/docs/eyeriss-v2-context.md`
- Work: `clover-kimi-k3/research/eyeriss-v2/{scripts,work,results}/` (new: `tb_e4_pe_zero.sv`, `e4_zero_handling.sh`, `e5_csc.py`, `e6_noc_model.py`)
- Predictions log: `clover-kimi-k3/research/eyeriss-v2/work/predictions.txt`

## 10. Independent check (2026-10-26, by me)
- **Re-runs on `k3`** in a chain (`E1a`, `E1b`, `E2`, `E3`; each script recreates its own work directory from a clean clone, no in-place edit of the clone): `E1a` (199 lines), `E2` (118), `E3` (51) identical to the stored outputs; `E1b` 54 of 58 lines identical, 4 differ only in what text the one-line summary captured for `glb_iact_tb` (iverilog) and `SPad_tb` (verilator) (pass/fail fields are the same). MD5 of the work files on `k3` equals the local copies.
- Source claims I confirmed by reading: `PE.sv` 7-state enum, `iter == act_size-kernel_size+1` wrap in `IDLE`, `LOAD_A` entry without `filt_count <= 0`, one `SPad` and one `MAC`; `router.sv` `unique if` and mode table.
- **Not covered**: anything about the chip's energy, area or throughput; behaviour outside Verilator/iverilog 12.
