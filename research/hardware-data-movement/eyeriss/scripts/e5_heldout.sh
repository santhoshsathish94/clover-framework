#!/usr/bin/env bash
# E5 held-out RTL runs: shapes that were NOT used when the count model (work/e5_model.py) was written.
# Same testbenches, same unmodified src/PE.v; OS uses the same one-line counter.v fix as E4 section 2 (applied to a copy).
set -uo pipefail
STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e5"
rm -rf "${W2_DIR}"; mkdir -p "${W2_DIR}"
cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src"
for f in tb_rs_conv2d_e3.sv tb_ws_conv2d_e4.sv tb_os_conv2d_e4.sv verify_e3_golden.py; do cp "${STUDY_DIR}/w2_work/$f" "${W2_DIR}/$f"; done

LEAF_NAMES=(BufferController.v buffer.v clkgate.v counter.v custom_counter.v demultiplier.v dff.v Disabling_mech.v FA.v fifo_buffer.v filter_counter.v iact_SPadController.v ifmap_counter_nbit.v ifmap_fifo.v muffer.v multiplier.v mux2x1.v PE_Controller.v psum_addr_counter.v psum_controller.v psum_write_addr_counter.v Reg_SPad_ifmap.v reg_SPad.v SPadController.v SRAM_SPad.v zero_buffer.v PE.v)
SRC_FILES=(); for n in "${LEAF_NAMES[@]}"; do SRC_FILES+=("${W2_DIR}/src/${n}"); done

# OS needs the counter.v fix (hard-coded wrap at 8) -- same sed as E4, on the copy only
sed -i "s/if (count == 3'd7)/if (count == MAX - 1)/" "${W2_DIR}/src/counter.v"

run_df() {
    local tb="$1" name="$2" R="$3" H="$4" W="$5" C="$6" M="$7"
    iverilog -g2012 -I "${W2_DIR}/src" -P${tb}.R=${R} -P${tb}.H=${H} -P${tb}.W=${W} -P${tb}.C_IN=${C} -P${tb}.M_OUT=${M} \
        -P${tb}.SPARSE=0 -P${tb}.CORRUPT_WEIGHT=0 -o "${W2_DIR}/${name}.vvp" "${SRC_FILES[@]}" "${W2_DIR}/${tb}.sv" 2>&1 | grep -vE "warning:|Padding|Pruning|vvp\.tgt sorry" || true
    timeout 200 vvp "${W2_DIR}/${name}.vvp" > "${W2_DIR}/${name}.log" 2>&1
    /root/ncm-cpu/venv/bin/python "${W2_DIR}/verify_e3_golden.py" "${W2_DIR}/${name}.log" || echo "VERIFY-FAILED ${name}"
}

echo "=== RS held-out (tb_rs_conv2d_e3) ==="
run_df tb_rs_conv2d_e3 rs_a 3 6 6 2 2
run_df tb_rs_conv2d_e3 rs_b 4 7 7 1 1
run_df tb_rs_conv2d_e3 rs_c 3 6 5 1 1
run_df tb_rs_conv2d_e3 rs_d 2 5 5 1 1
echo "=== WS held-out (tb_ws_conv2d_e4) ==="
run_df tb_ws_conv2d_e4 ws_a 3 6 6 2 2
run_df tb_ws_conv2d_e4 ws_b 4 7 7 1 1
run_df tb_ws_conv2d_e4 ws_c 3 6 5 1 1
run_df tb_ws_conv2d_e4 ws_d 2 5 5 1 1
echo "=== OS held-out (tb_os_conv2d_e4) ==="
run_df tb_os_conv2d_e4 os_a 3 6 6 2 2
run_df tb_os_conv2d_e4 os_b 4 7 7 1 1
run_df tb_os_conv2d_e4 os_c 3 6 5 1 1
run_df tb_os_conv2d_e4 os_d 2 5 5 1 1
