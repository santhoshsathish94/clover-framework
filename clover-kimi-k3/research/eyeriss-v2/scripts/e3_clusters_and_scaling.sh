#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/eyeriss-v2/w2"
ORIG_REPO="/root/dm-study/eyeriss-v2/repo/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2"
E3_DIR="$W2/e3_work"

rm -rf "$E3_DIR"
cp -r "$ORIG_REPO" "$E3_DIR"

# Apply minimal IEEE-1800 fixes to fresh copy
/root/ncm-cpu/venv/bin/python - << 'PYEOF'
import pathlib

fixed = pathlib.Path("/root/dm-study/eyeriss-v2/w2/e3_work")
for p in [fixed / "rtl/SPad.sv", fixed / "rtl/phase_1/SPad.sv"]:
    p.write_text(p.read_text().replace('$display("\\t\\t\\t\\t\\t Current Status:', 'if (0) $display("\\t\\t\\t\\t\\t Current Status:'))

for p in [fixed / "rtl/PE_cluster.sv", fixed / "rtl/phase_1/PE_cluster.sv"]:
    t = p.read_text()
    t = t.replace(
        "input logic [DATA_WIDTH-1:0] psum_out[0 : X_dim*Y_dim-1],\n\t\t\t\t\tinput logic [3:0] X_dim,\n\t\t\t\t\tinput logic [3:0] Y_dim",
        "input logic [DATA_WIDTH-1:0] psum_arr[0 : X_dim*Y_dim-1],\n\t\t\t\t\tinput logic [3:0] x_idx,\n\t\t\t\t\tinput logic [3:0] y_len"
    )
    t = t.replace("for(int i=0; i<Y_dim; i++) begin\n\t\t\t\t\t\tpsum_adder = psum_adder + psum_out[Y_dim*X_dim+i];",
                  "for(int i=0; i<y_len; i++) begin\n\t\t\t\t\t\tpsum_adder = psum_adder + psum_arr[y_len*x_idx+i];")
    p.write_text(t)

# Copy fixed router_network4.sv from sim_fixed or write clean wires
rn4 = pathlib.Path("/root/dm-study/eyeriss-v2/w2/sim_fixed/rtl/router_network4.sv").read_text()
(fixed / "rtl/router_network4.sv").write_text(rn4)
for fn in ["testbench/router_cluster_pe_cluster_tb.sv", "testbench/router_cluster_pe_cluster_5x5_tb.sv",
           "testbench/HMNoC_cluster_west_tb.sv", "testbench/router_pe_4_clusters_tb.sv", "testbench/router_pe_4_clusters_5x5_tb.sv",
           "rtl/HMNoC_cluster_west.sv"]:
    (fixed / fn).write_text((pathlib.Path("/root/dm-study/eyeriss-v2/w2/sim_fixed") / fn).read_text())
PYEOF

echo "=== 1. REPOSITORY 1-CLUSTER VS 4-CLUSTER TESTBENCHES (FRESH COPY) ==="
cd "$E3_DIR/testbench"
for tb in router_cluster_pe_cluster_tb router_cluster_pe_cluster_5x5_tb HMNoC_cluster_west_tb router_pe_4_clusters_tb router_pe_4_clusters_5x5_tb; do
  bdir="$E3_DIR/build_$tb"
  verilator --binary --timing -j 6 -Wno-fatal -Wno-lint -Wno-UNOPTFLAT -Wno-MULTIDRIVEN -I"$E3_DIR/rtl" --Mdir "$bdir" --top-module "$tb" "$E3_DIR/testbench/${tb}.sv" > /dev/null 2>&1
  stdbuf -oL timeout 2s "$bdir/V$tb" > "$E3_DIR/${tb}.log" 2>&1 || true
  cyc=$(grep -E "Total #cycles taken:" "$E3_DIR/${tb}.log" | awk '{print $NF}' || true)
  all4=$(grep -E "ALL4_LAST_ITER:" "$E3_DIR/${tb}.log" || true)
  echo "$tb: cycles=${cyc:-NONE(no_compute_or_finish)} $all4"
done

echo ""
echo "=== 2. CUSTOM PROBLEM-SIZE & CLUSTER-COUNT SCALING TESTBENCH (tb_e3_scaling.sv) ==="
cp "$W2/tb_e3_scaling.sv" "$E3_DIR/"
verilator --binary --timing -j 6 -Wno-fatal -Wno-lint -Wno-UNOPTFLAT -Wno-MULTIDRIVEN -I"$E3_DIR/rtl" --Mdir "$E3_DIR/build_scale" --top-module tb_e3_scaling "$E3_DIR/tb_e3_scaling.sv" > /dev/null 2>&1
stdbuf -oL timeout 5s "$E3_DIR/build_scale/Vtb_e3_scaling" > "$E3_DIR/tb_e3_scaling.log" 2>&1
cat "$E3_DIR/tb_e3_scaling.log"

echo ""
echo "=== 3. PYTHON GOLDEN CONVOLUTION & SCALING VERIFICATION ==="
/root/ncm-cpu/venv/bin/python - << 'PYEOF'
import pathlib, re
import numpy as np

e3 = pathlib.Path("/root/dm-study/eyeriss-v2/w2/e3_work")
log = (e3 / "tb_e3_scaling.log").read_text()

def conv2d(act, ker):
    H, W = act.shape
    R, S = ker.shape
    out = np.zeros((H - R + 1, W - S + 1), dtype=int)
    for r in range(H - R + 1):
        for c in range(W - S + 1):
            out[r, c] = int(np.sum(act[r:r+R, c:c+S] * ker))
    return out

# Check all 4 problem sizes against Python golden and active-cycle formula
for K, H, exp_tot in [(3, 5, 72), (3, 6, 95), (3, 7, 120), (5, 7, 130)]:
    E = H - K + 1
    ker = np.arange(1, K*K + 1).reshape(K, K)
    act = np.arange(1, H*H + 1).reshape(H, H)
    gold = conv2d(act, ker)
    for r in range(E):
        m = re.search(rf"SIZE\(K={K},H={H}\) row={r} out=\[([0-9,]+)\]", log)
        rtl_row = [int(x) for x in m.group(1).split(",")]
        assert rtl_row == gold[r].tolist(), f"Mismatch K={K},H={H},row={r}: {rtl_row} vs {gold[r].tolist()}"
    ms = re.search(rf"SIZE_SUMMARY K={K} H={H} E={E} PEs={E*K}: load_w=(\d+) load_a=(\d+) compute=(\d+) total=(\d+)", log)
    tot = int(ms.group(4))
    print(f"Verified (K={K}, H={H}): all {E}x{E} outputs match Python golden; total_active_cycles={tot} (predicted {exp_tot})")
    assert tot == exp_tot

# Check 1, 2, 4 cluster scaling and negative control
ker3 = np.arange(1, 10).reshape(3, 3)
act5 = np.arange(1, 26).reshape(5, 5)
g_ch = [
    conv2d(act5 * 5, ker3)[2].tolist(),
    conv2d(act5 + 3, ker3)[2].tolist(),
    conv2d(act5, ker3)[2].tolist(),
    conv2d(act5 + 2, ker3)[2].tolist(),
]

for n_cl, exp_c in [(1, 291), (2, 145), (4, 72)]:
    mc = re.search(rf"CLUST_SUMMARY N={n_cl} corrupt=0 passes=\d+ active_cycles=(\d+)", log)
    c_got = int(mc.group(1))
    print(f"Verified Cluster Scaling N={n_cl}: active_cycles={c_got} (expected {exp_c} with PE.sv iter-wrap + filt_count reset)")
    assert c_got == exp_c

# Verify negative control (corrupt_w4=1 on N=4)
m_norm = re.search(r"CLUST\(N=4,corrupt=0\) pass=0 row2: w0=\[([0-9,]+)\] w1=\[([0-9,]+)\] e0=\[([0-9,]+)\] e1=\[([0-9,]+)\]", log)
m_corr = re.search(r"CLUST\(N=4,corrupt=1\) pass=0 row2: w0=\[([0-9,]+)\] w1=\[([0-9,]+)\] e0=\[([0-9,]+)\] e1=\[([0-9,]+)\]", log)
norm_w0 = [int(x) for x in m_norm.group(1).split(",")]
corr_w0 = [int(x) for x in m_corr.group(1).split(",")]
ker3_corr = ker3.copy()
ker3_corr[1, 1] = 6
exp_corr_w0 = conv2d(act5 * 5, ker3_corr)[2].tolist()
print(f"Negative control (corrupt w[4]=5->6): normal_w0={norm_w0} corrupt_w0={corr_w0} expected_corrupt={exp_corr_w0} fails_normal={corr_w0 != norm_w0}")
assert norm_w0 == g_ch[0]
assert corr_w0 == exp_corr_w0 and corr_w0 != norm_w0
print("ALL E3 CHECKS PASSED.")
PYEOF
