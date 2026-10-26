#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/eyeriss-v2/w2"
ORIG_REPO="/root/dm-study/eyeriss-v2/repo/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2"
CLEAN="$W2/clean_repo"
FIXED="$W2/sim_fixed"

rm -rf "$CLEAN" "$FIXED" "$W2/build_fixed"
cp -r "$ORIG_REPO" "$CLEAN"
cp -r "$ORIG_REPO" "$FIXED"
mkdir -p "$W2/build_fixed"

echo "=== 1. UNMODIFIED IVERILOG -g2012 -Y .sv CHECK ==="
for tb in adder_tb MAC_tb switch_tb SPad_tb glb_iact_tb PE_tb PE_cluster_tb router_glb_tb; do
  rm -f "$W2/build_unmod/${tb}.vvp"
  iout=$(iverilog -g2012 -Y .sv -y "$CLEAN/rtl" -s "$tb" -o "$W2/build_unmod/${tb}.vvp" "$CLEAN/testbench/${tb}.sv" 2>&1 || true)
  if [ -z "$iout" ] && [ -f "$W2/build_unmod/${tb}.vvp" ]; then
    rout=$(timeout 1s vvp "$W2/build_unmod/${tb}.vvp" 2>&1 | grep -v -E "Current Status|mem\[|psum:" | tr -s '\n' ' ' | cut -c1-140 || true)
    printf "iverilog %-18s | BUILD: PASS | RUN: %s\n" "$tb" "$rout"
  else
    err=$(echo "$iout" | head -n 2 | tr '\n' ' ')
    printf "iverilog %-18s | BUILD: FAIL | ERR: %s\n" "$tb" "$err"
  fi
done

echo ""
echo "=== 2. UNMODIFIED VERILATOR OUTPUT OF THE 7 PASSING TESTBENCHES ==="
for tb in adder_tb MAC_tb switch_tb SPad_tb glb_iact_tb PE_tb router_glb_tb; do
  cd "$CLEAN/testbench"
  rout=$(timeout 1s "$W2/build_unmod/$tb/V$tb" 2>&1 | grep -v -E "Current Status|mem\[|psum:" | tr -s '\n' ' ' | cut -c1-160 || true)
  printf "verilator %-17s | OUTPUT: %s\n" "$tb" "$rout"
done

echo ""
echo "=== 3. APPLY MINIMAL IEEE-1800 FIXES IN FRESH COPY ($FIXED) ==="
/root/ncm-cpu/venv/bin/python - << 'PYEOF'
import pathlib

fixed = pathlib.Path("/root/dm-study/eyeriss-v2/w2/sim_fixed")

# A. circular_buffer.sv: output wire -> output logic
for p in [fixed / "rtl/circular_buffer.sv", fixed / "rtl/misc/circular_buffer.sv"]:
    p.write_text(p.read_text().replace("output wire [DATA_WIDTH-1:0] data_o", "output logic [DATA_WIDTH-1:0] data_o"))

# B. SPad.sv: silence per-cycle $display dump so testbench output is readable; PE.sv: fix nested /* comment for iverilog
for p in [fixed / "rtl/SPad.sv", fixed / "rtl/phase_1/SPad.sv"]:
    p.write_text(p.read_text().replace('$display("\\t\\t\\t\\t\\t Current Status:', 'if (0) $display("\\t\\t\\t\\t\\t Current Status:'))

for p in [fixed / "rtl/PE.sv", fixed / "rtl/phase_1/PE.sv"]:
    t = p.read_text()
    t = t.replace("/* \t\t\t\t\t\tif(load_en) begin", "/* if(load_en) begin")
    t = t.replace("//***Loading of weights starts at index 0***", "// Loading of weights starts at index 0")
    p.write_text(t)

# C. PE_cluster.sv: rename formal args X_dim, Y_dim in psum_adder to x_idx, y_len
for p in [fixed / "rtl/PE_cluster.sv", fixed / "rtl/phase_1/PE_cluster.sv"]:
    t = p.read_text()
    t = t.replace(
        "input logic [DATA_WIDTH-1:0] psum_out[0 : X_dim*Y_dim-1],\n\t\t\t\t\tinput logic [3:0] X_dim,\n\t\t\t\t\tinput logic [3:0] Y_dim",
        "input logic [DATA_WIDTH-1:0] psum_arr[0 : X_dim*Y_dim-1],\n\t\t\t\t\tinput logic [3:0] x_idx,\n\t\t\t\t\tinput logic [3:0] y_len"
    )
    t = t.replace("for(int i=0; i<Y_dim; i++) begin\n\t\t\t\t\t\tpsum_adder = psum_adder + psum_out[Y_dim*X_dim+i];",
                  "for(int i=0; i<y_len; i++) begin\n\t\t\t\t\t\tpsum_adder = psum_adder + psum_arr[y_len*x_idx+i];")
    p.write_text(t)

# D. router_iact.sv & router_weight.sv: fix BLKANDNBLK (= 0 -> <= 0)
for fn in ["rtl/router_iact.sv", "rtl/phase_1/router_iact.sv", "rtl/router_weight.sv", "rtl/phase_1/router_weight.sv",
           "testbench/router_act.sv", "testbench/router_iact.sv", "testbench/router_weight.sv"]:
    p = fixed / fn
    t = p.read_text()
    t = t.replace("read_req_glb_iact = 0;", "read_req_glb_iact <= 0;")
    t = t.replace("load_en_spad = 0;", "load_en_spad <= 0;")
    t = t.replace("read_req_glb_wght = 0;", "read_req_glb_wght <= 0;")
    p.write_text(t)

# E. router_network4.sv: replace hierarchical .port(other_inst.in_port) with explicit wires
rn4 = """`timescale 1ns / 1ps
module router_network4 #(parameter DATA_WIDTH = 16) (
    input [3:0] router_mode_west_0,
    input [DATA_WIDTH-1:0] west_data_i_west_0,
    input west_enable_i_west_0,
    output logic [DATA_WIDTH-1:0] west_data_o_west_0,
    output logic west_enable_o_west_0,
    input [3:0] router_mode_west_1,
    input [DATA_WIDTH-1:0] west_data_i_west_1,
    input west_enable_i_west_1,
    output logic [DATA_WIDTH-1:0] west_data_o_west_1,
    output logic west_enable_o_west_1,
    input [3:0] router_mode_east_0,
    input [DATA_WIDTH-1:0] east_data_i_east_0,
    input east_enable_i_east_0,
    output logic [DATA_WIDTH-1:0] east_data_o_east_0,
    output logic east_enable_o_east_0,
    input [3:0] router_mode_east_1,
    input [DATA_WIDTH-1:0] east_data_i_east_1,
    input east_enable_i_east_1,
    output logic [DATA_WIDTH-1:0] east_data_o_east_1,
    output logic east_enable_o_east_1
);
    logic [DATA_WIDTH-1:0] w0_s_do, w1_n_do, e0_s_do, e1_n_do;
    logic [DATA_WIDTH-1:0] w0_e_do, e0_w_do, w1_e_do, e1_w_do;
    logic w0_s_eo, w1_n_eo, e0_s_eo, e1_n_eo;
    logic w0_e_eo, e0_w_eo, w1_e_eo, e1_w_eo;
    router #(.DATA_WIDTH(DATA_WIDTH)) router_west_0 (
        .router_mode(router_mode_west_0),
        .north_data_i('0), .north_enable_i(1'b0), .north_data_o(), .north_enable_o(),
        .south_data_i(w1_n_do), .south_enable_i(w1_n_eo), .south_data_o(w0_s_do), .south_enable_o(w0_s_eo),
        .west_data_i(west_data_i_west_0), .west_enable_i(west_enable_i_west_0), .west_data_o(west_data_o_west_0), .west_enable_o(west_enable_o_west_0),
        .east_data_i(e0_w_do), .east_enable_i(e0_w_eo), .east_data_o(w0_e_do), .east_enable_o(w0_e_eo)
    );
    router #(.DATA_WIDTH(DATA_WIDTH)) router_west_1 (
        .router_mode(router_mode_west_1),
        .north_data_i(w0_s_do), .north_enable_i(w0_s_eo), .north_data_o(w1_n_do), .north_enable_o(w1_n_eo),
        .south_data_i('0), .south_enable_i(1'b0), .south_data_o(), .south_enable_o(),
        .west_data_i(west_data_i_west_1), .west_enable_i(west_enable_i_west_1), .west_data_o(west_data_o_west_1), .west_enable_o(west_enable_o_west_1),
        .east_data_i(e1_w_do), .east_enable_i(e1_w_eo), .east_data_o(w1_e_do), .east_enable_o(w1_e_eo)
    );
    router #(.DATA_WIDTH(DATA_WIDTH)) router_east_0 (
        .router_mode(router_mode_east_0),
        .north_data_i('0), .north_enable_i(1'b0), .north_data_o(), .north_enable_o(),
        .south_data_i(e1_n_do), .south_enable_i(e1_n_eo), .south_data_o(e0_s_do), .south_enable_o(e0_s_eo),
        .west_data_i(w0_e_do), .west_enable_i(w0_e_eo), .west_data_o(e0_w_do), .west_enable_o(e0_w_eo),
        .east_data_i(east_data_i_east_0), .east_enable_i(east_enable_i_east_0), .east_data_o(east_data_o_east_0), .east_enable_o(east_enable_o_east_0)
    );
    router #(.DATA_WIDTH(DATA_WIDTH)) router_east_1 (
        .router_mode(router_mode_east_1),
        .north_data_i(e0_s_do), .north_enable_i(e0_s_eo), .north_data_o(e1_n_do), .north_enable_o(e1_n_eo),
        .south_data_i('0), .south_enable_i(1'b0), .south_data_o(), .south_enable_o(),
        .west_data_i(w1_e_do), .west_enable_i(w1_e_eo), .west_data_o(e1_w_do), .west_enable_o(e1_w_eo),
        .east_data_i(east_data_i_east_1), .east_enable_i(east_enable_i_east_1), .east_data_o(east_data_o_east_1), .east_enable_o(east_enable_o_east_1)
    );
endmodule
"""
(fixed / "rtl/router_network4.sv").write_text(rn4)
(fixed / "rtl/phase_3/router_network4.sv").write_text(rn4)
print("RTL fixes applied.")
PYEOF

/root/ncm-cpu/venv/bin/python "$W2/e1_patch_tbs.py"

echo ""
echo "=== 4. BUILD & RUN ALL 25 UNIQUE TESTBENCHES IN FRESH FIXED COPY ==="
TBS=(
  adder_tb MAC_tb switch_tb SPad_tb glb_iact_tb GLB_cluster_tb
  PE_tb PE_cluster_tb HMNoC_1_tb
  router_weight_tb router_iact_tb router_psum_tb
  router_cluster_tb router_cluster_5x5_tb
  router_east_tb router_glb_tb router_cluster_new_tb
  router_cluster_pe_cluster_tb router_cluster_pe_cluster_5x5_tb
  HMNoC_cluster_west_tb
  router_pe_4_clusters_tb router_pe_4_clusters_5x5_tb
  router_broadcast_tb router_multicast_tb router_unicast_tb
)

PASS_CNT=0
for tb in "${TBS[@]}"; do
  bdir="$W2/build_fixed/$tb"
  mkdir -p "$bdir"
  inc="-I$FIXED/rtl -I$FIXED/rtl/misc"
  if [[ "$tb" == "router_cluster_tb" || "$tb" == "router_cluster_5x5_tb" ]]; then
    inc="-I$FIXED/rtl/phase_1 -I$FIXED/rtl -I$FIXED/rtl/misc"
  fi
  vout=$(verilator --binary --timing -j 6 -Wno-fatal -Wno-lint -Wno-UNOPTFLAT -Wno-MULTIDRIVEN $inc --Mdir "$bdir" --top-module "$tb" "$FIXED/testbench/${tb}.sv" 2>&1 || true)
  if [ -x "$bdir/V$tb" ]; then
    PASS_CNT=$((PASS_CNT + 1))
    cd "$FIXED/testbench"
    rout=$(stdbuf -oL timeout 1s "$bdir/V$tb" 2>&1 || true)
    echo "$rout" > "$W2/build_fixed/${tb}.run.log"
    rline=$(echo "$rout" | tr -s '\n' ' ' | sed 's/  */ /g' | cut -c1-150)
    printf "%-32s | BUILD: PASS | RUN: %s\n" "$tb" "$rline"
  else
    err=$(echo "$vout" | grep -E "%Error" | head -n 2 | tr '\n' ' ')
    printf "%-32s | BUILD: FAIL | ERR: %s\n" "$tb" "$err"
  fi
done
echo "Total testbenches built & run in sim_fixed: $PASS_CNT / ${#TBS[@]}"

echo ""
echo "=== 5. PYTHON GOLDEN CONVOLUTION CHECK ON TESTBENCH OUTPUTS ==="
/root/ncm-cpu/venv/bin/python "$W2/e1_golden_check.py"

