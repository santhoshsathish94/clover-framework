#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e3"
rm -rf "${W2_DIR}"
mkdir -p "${W2_DIR}"

cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src"
cp "${STUDY_DIR}/w2_work/tb_rs_conv2d_e3.sv" "${W2_DIR}/tb_rs_conv2d_e3.sv"
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

run_case() {
    local name="$1" R="$2" H="$3" W="$4" C="$5" M="$6" SP="$7" CORR="$8" flag="${9:-}"
    iverilog -g2012 -I "${W2_DIR}/src" \
        -Ptb_rs_conv2d_e3.R=${R} -Ptb_rs_conv2d_e3.H=${H} -Ptb_rs_conv2d_e3.W=${W} \
        -Ptb_rs_conv2d_e3.C_IN=${C} -Ptb_rs_conv2d_e3.M_OUT=${M} \
        -Ptb_rs_conv2d_e3.SPARSE=${SP} -Ptb_rs_conv2d_e3.CORRUPT_WEIGHT=${CORR} \
        -o "${W2_DIR}/${name}.vvp" "${SRC_FILES[@]}" "${W2_DIR}/tb_rs_conv2d_e3.sv" 2>&1 | grep -vE "warning:|Padding|Pruning|vvp\.tgt sorry" || true
    vvp "${W2_DIR}/${name}.vvp" > "${W2_DIR}/${name}.log"
    /root/ncm-cpu/venv/bin/python "${W2_DIR}/verify_e3_golden.py" "${W2_DIR}/${name}.log" ${flag}
}

echo "=== E3 ROW-STATIONARY 2D CONVOLUTION ON REPO PEs vs PYTHON GOLDEN ==="
run_case "rs_3x3_5x5_dense"   3 5 5 1 1 0 0
run_case "rs_3x3_5x5_sparse"  3 5 5 1 1 1 0
run_case "rs_3x3_5x5_corrupt" 3 5 5 1 1 1 1 "--expect-fail"
run_case "rs_5x5_7x7_dense"   5 7 7 1 1 0 0
run_case "rs_3x3_7x7_dense"   3 7 7 1 1 0 0
run_case "rs_3x3_8x8_dense"   3 8 8 1 1 0 0
run_case "rs_3x3_5x5_c2"      3 5 5 2 1 0 0
run_case "rs_3x3_5x5_m2"      3 5 5 1 2 0 0

echo ""
echo "=== SAMPLE PIXEL DUMP (rs_3x3_5x5_sparse vs rs_3x3_5x5_corrupt) ==="
grep "^OUT " "${W2_DIR}/rs_3x3_5x5_sparse.log" | tr '\n' ' '; echo ""
grep "^OUT " "${W2_DIR}/rs_3x3_5x5_corrupt.log" | tr '\n' ' '; echo ""
