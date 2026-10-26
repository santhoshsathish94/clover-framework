#!/usr/bin/env bash
# E6: how far is PE_Cluster.v from compiling against the PE.v that ships in the same repository?
# Uses MY stand-ins (hw_config_standin.vh, Mux_standin.v) for the two things the repository does not contain.
STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e6"
rm -rf "${W2_DIR}"; mkdir -p "${W2_DIR}"
cp -r "${STUDY_DIR}/repo/Eyeriss/src" "${W2_DIR}/src"
cp "${STUDY_DIR}/w2_work/hw_config_standin.vh" "${W2_DIR}/src/hw_config.vh"
cp "${STUDY_DIR}/w2_work/Mux_standin.v" "${W2_DIR}/Mux_standin.v"
cd "${W2_DIR}"

LEAVES="BufferController.v buffer.v clkgate.v counter.v custom_counter.v demultiplier.v dff.v Disabling_mech.v FA.v fifo_buffer.v filter_counter.v iact_SPadController.v ifmap_counter_nbit.v ifmap_fifo.v muffer.v multiplier.v mux2x1.v PE_Controller.v psum_addr_counter.v psum_controller.v psum_write_addr_counter.v Reg_SPad_ifmap.v reg_SPad.v SPadController.v SRAM_SPad.v zero_buffer.v PE.v"
FILES=""
for f in $LEAVES; do FILES="$FILES src/$f"; done

echo "=== A. PE.v ports as shipped (declaration lines) ==="
awk '/^module PE[ #(]/,/\);/' src/PE.v | grep -E "^\s*(input|output)" | sed 's/^\s*//'

echo
echo "=== B. PE_Cluster.v named ports used in the PE instance ==="
awk '/PE xx \(/,/\);/' src/PE_Cluster.v | grep -oE "\.[a-z_]+\(" | sort -u | tr '\n' ' '; echo

echo
echo "=== C. iverilog: PE_Cluster_Ctrl.v alone (with stand-in header) ==="
iverilog -g2012 -I src -o ctrl.vvp src/PE_Cluster_Ctrl.v 2>&1 | head -20; echo "rc=${PIPESTATUS[0]}"

echo
echo "=== D. iverilog: PE_Cluster.v + PE.v + leaves + Mux stand-in (all errors) ==="
iverilog -g2012 -I src -o cl.vvp $FILES Mux_standin.v src/PE_Cluster_Ctrl.v src/PE_Cluster.v 2>&1 | grep -v "warning:" | head -60; echo "rc=${PIPESTATUS[0]}"

echo
echo "=== E. count of errors by kind ==="
iverilog -g2012 -I src -o cl.vvp $FILES Mux_standin.v src/PE_Cluster_Ctrl.v src/PE_Cluster.v 2>&1 | grep -v "warning:" | sed -E 's/^[^:]+:[0-9]+: //' | sed -E 's/xx\.|port [^ ]+//' | sort | uniq -c | sort -rn | head -20

echo
echo "=== F. iact_enables concatenation in PE_Cluster.v (repeat check) ==="
grep -n "assign iact_enables" src/PE_Cluster.v
