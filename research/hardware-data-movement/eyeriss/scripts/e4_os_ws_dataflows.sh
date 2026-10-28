#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e4"
rm -rf "${W2_DIR}"
mkdir -p "${W2_DIR}"

cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src"
cp "${STUDY_DIR}/w2_work/tb_os_conv2d_e4.sv" "${W2_DIR}/tb_os_conv2d_e4.sv"
cp "${STUDY_DIR}/w2_work/tb_ws_conv2d_e4.sv" "${W2_DIR}/tb_ws_conv2d_e4.sv"
cp "${STUDY_DIR}/w2_work/verify_e3_golden.py" "${W2_DIR}/verify_e3_golden.py"

LEAF_NAMES=(
    BufferController.v buffer.v clkgate.v counter.v
    custom_counter.v demultiplier.v dff.v Disabling_mech.v
    FA.v fifo_buffer.v filter_counter.v
    iact_SPadController.v ifmap_counter_nbit.v ifmap_fifo.v
    muffer.v multiplier.v mux2x1.v PE_Controller.v
    psum_addr_counter.v psum_controller.v psum_write_addr_counter.v
    Reg_SPad_ifmap.v reg_SPad.v SPadController.v SRAM_SPad.v
    zero_buffer.v PE.v
)
SRC_FILES=()
for name in "${LEAF_NAMES[@]}"; do
    SRC_FILES+=("${W2_DIR}/src/${name}")
done

run_df() {
    local tb_mod="$1" name="$2" R="$3" H="$4" W="$5" C="$6" M="$7" SP="$8" CORR="$9" flag="${10:-}"
    iverilog -g2012 -I "${W2_DIR}/src" \
        -P${tb_mod}.R=${R} -P${tb_mod}.H=${H} -P${tb_mod}.W=${W} \
        -P${tb_mod}.C_IN=${C} -P${tb_mod}.M_OUT=${M} \
        -P${tb_mod}.SPARSE=${SP} -P${tb_mod}.CORRUPT_WEIGHT=${CORR} \
        -o "${W2_DIR}/${name}.vvp" "${SRC_FILES[@]}" "${W2_DIR}/${tb_mod}.sv" 2>&1 | grep -vE "warning:|Padding|Pruning|vvp\.tgt sorry" || true
    vvp "${W2_DIR}/${name}.vvp" > "${W2_DIR}/${name}.log"
    /root/ncm-cpu/venv/bin/python "${W2_DIR}/verify_e3_golden.py" "${W2_DIR}/${name}.log" ${flag}
}

echo "=== 0. OS ON UNMODIFIED counter.v (EXPOSES count==3'd7 HARDCODED WRAP AT 8 WEIGHTS) ==="
run_df "tb_os_conv2d_e4" "os_unmod_wrap_bug" 3 5 5 1 1 0 0 "--expect-fail"

echo ""
echo "=== 1. WEIGHT-STATIONARY (WS) ON UNMODIFIED REPO PEs vs PYTHON GOLDEN ==="
run_df "tb_ws_conv2d_e4" "ws_3x3_5x5_dense"   3 5 5 1 1 0 0
run_df "tb_ws_conv2d_e4" "ws_3x3_5x5_sparse"  3 5 5 1 1 1 0
run_df "tb_ws_conv2d_e4" "ws_3x3_5x5_corrupt" 3 5 5 1 1 1 1 "--expect-fail"
run_df "tb_ws_conv2d_e4" "ws_3x3_7x7_dense"   3 7 7 1 1 0 0
run_df "tb_ws_conv2d_e4" "ws_3x3_5x5_c2"      3 5 5 2 1 0 0
run_df "tb_ws_conv2d_e4" "ws_3x3_5x5_m2"      3 5 5 1 2 0 0

echo ""
echo "=== 2. OUTPUT-STATIONARY (OS) WITH counter.v MAX-1 FIX IN work/src vs PYTHON GOLDEN ==="
# Fix hardcoded 3'd7 wrap in work copy of src/counter.v so filter_write_counter can address >8 weights
sed -i "s/if (count == 3'd7)/if (count == MAX - 1)/" "${W2_DIR}/src/counter.v"
run_df "tb_os_conv2d_e4" "os_3x3_5x5_dense"   3 5 5 1 1 0 0
run_df "tb_os_conv2d_e4" "os_3x3_5x5_sparse"  3 5 5 1 1 1 0
run_df "tb_os_conv2d_e4" "os_3x3_5x5_corrupt" 3 5 5 1 1 1 1 "--expect-fail"
run_df "tb_os_conv2d_e4" "os_3x3_7x7_dense"   3 7 7 1 1 0 0
run_df "tb_os_conv2d_e4" "os_3x3_5x5_c2"      3 5 5 2 1 0 0
run_df "tb_os_conv2d_e4" "os_3x3_5x5_m2"      3 5 5 1 2 0 0
