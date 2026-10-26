#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e2"
rm -rf "${W2_DIR}"
mkdir -p "${W2_DIR}"

cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src_unmod"
cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src_wrapped"
cp "${STUDY_DIR}/w2_work/tb_pe_e2.sv" "${W2_DIR}/tb_pe_e2.sv"

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

UNMOD_FILES=()
WRAPPED_FILES=()
for name in "${LEAF_NAMES[@]}"; do
    UNMOD_FILES+=("${W2_DIR}/src_unmod/${name}")
    WRAPPED_FILES+=("${W2_DIR}/src_wrapped/${name}")
done

echo "=== 1. RUN CASES 1, 1b, 2, 3, 4 ON UNMODIFIED SHIPPED PE.v ==="
iverilog -g2012 -I "${W2_DIR}/src_unmod" -o "${W2_DIR}/tb_pe_unmod.vvp" \
    "${UNMOD_FILES[@]}" "${W2_DIR}/tb_pe_e2.sv" 2>/dev/null
for c in 1 2 3 4 5; do
    vvp "${W2_DIR}/tb_pe_unmod.vvp" +CASE=$c | grep -v '\$finish'
done

echo ""
echo "=== 2. RUN CASES 3 & 4 ON WORK COPY WITH COUNTER WRAP + RESET FIXES ==="
# Fix 1: wrap filter_read_counter modulo filter_size (0..filter_size-1)
sed -i 's/\.max_count(filter_size),/.max_count(filter_size - 1),/' "${W2_DIR}/src_wrapped/PE.v"
sed -i 's/if (!at_max)/if (at_max) count <= 0; else/' "${W2_DIR}/src_wrapped/custom_counter.v"
# Fix 2: reset loc_en and read_addr_change in psum_addr_counter.v and mul_valid in multiplier.v
sed -i 's/if (!rstn) begin \/\/ Reset/if (!rstn) begin loc_en <= 0; read_addr_change <= 0;/' "${W2_DIR}/src_wrapped/psum_addr_counter.v"
sed -i 's/mul_ready = 2'\''b00;/mul_ready = 2'\''b00; mul_valid = 1'\''b0;/' "${W2_DIR}/src_wrapped/multiplier.v"

iverilog -g2012 -I "${W2_DIR}/src_wrapped" -o "${W2_DIR}/tb_pe_wrapped.vvp" \
    "${WRAPPED_FILES[@]}" "${W2_DIR}/tb_pe_e2.sv" 2>/dev/null
for c in 1 3 4 5; do
    vvp "${W2_DIR}/tb_pe_wrapped.vvp" +CASE=$c | grep -v '\$finish'
done
