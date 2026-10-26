#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/eyeriss-v2/w2"
ORIG_REPO="/root/dm-study/eyeriss-v2/repo/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2"
CLEAN="$W2/clean_repo"

mkdir -p "$W2"
rm -rf "$CLEAN" "$W2/build_unmod"
cp -r "$ORIG_REPO" "$CLEAN"
mkdir -p "$W2/build_unmod"

echo "=== 1. REPO PINNED COMMIT, LICENSE, AND FILE STATS ==="
cd "$CLEAN"
echo "commit: $(git rev-parse HEAD)"
echo "date:   $(git log -1 --format='%ci (%s)')"
echo "status: $(git status --short | wc -l) modified"
if ls LICENSE* COPYING* 2>/dev/null; then
  echo "license: found"
else
  echo "license: NONE (no LICENSE or COPYING file in repository)"
fi

echo ""
echo "=== 2. TRACKED TEXT/SV FILES LINE COUNTS & MD5 ==="
git ls-files | while read -r f; do
  case "$f" in
    *.png|*.PNG|*.pdf)
      sz=$(stat -c%s "$f")
      m=$(md5sum "$f" | awk '{print $1}')
      printf "%-48s %8s lines %8d B  %s\n" "$f" "BINARY" "$sz" "${m:0:12}"
      ;;
    *)
      lc=$(wc -l < "$f")
      sz=$(stat -c%s "$f")
      m=$(md5sum "$f" | awk '{print $1}')
      printf "%-48s %8d lines %8d B  %s\n" "$f" "$lc" "$sz" "${m:0:12}"
      ;;
  esac
done

echo ""
echo "=== 3. UNMODIFIED RTL MODULE LINT (23 UNIQUE MODULES) ==="
RTL_MODULES=(
  FA.sv HA.sv MAC.sv mux2.sv mux_4x1.sv lookup_mux4.sv switch.sv
  circular_buffer.sv SPad.sv PE.sv PE_cluster.sv
  glb_iact.sv glb_psum.sv glb_weight.sv GLB_cluster.sv
  router_iact.sv router_weight.sv router_psum.sv
  router.sv router_east.sv router_cluster.sv router_network4.sv
  HMNoC_cluster.sv HMNoC_cluster_west.sv HMNoC_cluster_east.sv HMNoC_top.sv
)
# Note: router_cluster exists in rtl/ (Phase 2/3) and rtl/phase_1/router_cluster.sv (Phase 1)
for m in "${RTL_MODULES[@]}"; do
  top="${m%.sv}"
  out=$(verilator --lint-only -Wno-lint -I"$CLEAN/rtl" --top-module "$top" "$CLEAN/rtl/$m" 2>&1 || true)
  if [ -z "$out" ]; then
    v_res="PASS"
    v_err="-"
  else
    v_res="FAIL"
    v_err=$(echo "$out" | grep -E "%Error" | head -n 2 | tr '\n' ' ')
  fi
  printf "%-24s | Verilator: %-4s | %s\n" "$m" "$v_res" "$v_err"
done
# Also lint phase_1/router_cluster.sv
out=$(verilator --lint-only -Wno-lint -I"$CLEAN/rtl/phase_1" --top-module router_cluster "$CLEAN/rtl/phase_1/router_cluster.sv" 2>&1 || true)
v_res=$([ -z "$out" ] && echo "PASS" || echo "FAIL")
v_err=$(echo "$out" | grep -E "%Error" | head -n 2 | tr '\n' ' ')
printf "%-24s | Verilator: %-4s | %s\n" "phase_1/router_cluster.sv" "$v_res" "$v_err"

echo ""
echo "=== 4. UNMODIFIED TESTBENCH BUILD & RUN (25 UNIQUE TESTBENCHES) ==="
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

for tb in "${TBS[@]}"; do
  bdir="$W2/build_unmod/$tb"
  mkdir -p "$bdir"
  # Select include dir (Phase 1 tbs use phase_1/router_cluster.sv if router_cluster_tb)
  inc="-I$CLEAN/rtl"
  if [[ "$tb" == "router_cluster_tb" || "$tb" == "router_cluster_5x5_tb" ]]; then
    inc="-I$CLEAN/rtl/phase_1 -I$CLEAN/rtl"
  fi
  vout=$(verilator --binary --timing -Wno-lint $inc --Mdir "$bdir" --top-module "$tb" "$CLEAN/testbench/${tb}.sv" 2>&1 || true)
  if [ -x "$bdir/V$tb" ]; then
    cd "$CLEAN/testbench"
    rout=$(timeout 2s "$bdir/V$tb" 2>&1 || true)
    rline=$(echo "$rout" | tr -s '\n' ' ' | sed 's/  */ /g' | cut -c1-140)
    printf "%-32s | BUILD: PASS | RUN: %s\n" "$tb" "$rline"
  else
    err=$(echo "$vout" | grep -E "%Error" | head -n 2 | tr '\n' ' ')
    printf "%-32s | BUILD: FAIL | ERR: %s\n" "$tb" "$err"
  fi
done

echo ""
echo "=== 5. UNMODIFIED IVERILOG -g2012 CHECK ON KEY TESTBENCHES ==="
for tb in adder_tb MAC_tb switch_tb SPad_tb glb_iact_tb PE_tb PE_cluster_tb; do
  iout=$(iverilog -g2012 -y "$CLEAN/rtl" -s "$tb" -o "$W2/build_unmod/${tb}.vvp" "$CLEAN/testbench/${tb}.sv" 2>&1 || true)
  if [ -z "$iout" ] && [ -f "$W2/build_unmod/${tb}.vvp" ]; then
    rout=$(timeout 1s vvp "$W2/build_unmod/${tb}.vvp" 2>&1 || true)
    rline=$(echo "$rout" | tr -s '\n' ' ' | sed 's/  */ /g' | cut -c1-120)
    printf "iverilog %-23s | BUILD: PASS | RUN: %s\n" "$tb" "$rline"
  else
    err=$(echo "$iout" | head -n 2 | tr '\n' ' ')
    printf "iverilog %-23s | BUILD: FAIL | ERR: %s\n" "$tb" "$err"
  fi
done
