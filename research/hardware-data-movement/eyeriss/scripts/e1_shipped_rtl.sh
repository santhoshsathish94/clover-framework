#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="/root/dm-study/eyeriss"
W2_DIR="${STUDY_DIR}/w2/e1"
rm -rf "${W2_DIR}"
mkdir -p "${W2_DIR}"

# Clean up unused round-1 pylib if present to save disk space
if [ -d "${STUDY_DIR}/pylib" ]; then
    rm -rf "${STUDY_DIR}/pylib"
fi

echo "=== 1. REPO PROVENANCE, LICENSE, DISK, AND FILE INVENTORY ==="
cp -r "${STUDY_DIR}/repo/Eyeriss" "${W2_DIR}/repo_copy"
cd "${W2_DIR}/repo_copy"
echo "git_head: $(git rev-parse HEAD)"
echo "git_branch_status: $(git status -sb | tr '\n' ' ')"
echo "git_remote: $(git remote -v | head -n 1)"
echo "git_log_last3:"
git log -n 3 --oneline
echo "license_files: $(find . -maxdepth 2 -iname '*licen*' -o -iname '*copying*' || true)"
grep -i "license" README.md || echo "No license line in README.md"
echo "slice_disk_usage: $(du -sh "${STUDY_DIR}" | awk '{print $1}')"

echo ""
echo "=== 2. FILE LIST WITH LINE COUNTS AND BYTE SIZES ==="
find README.md sim src -type f | sort | while read -r f; do
    lines=$(wc -l < "$f")
    bytes=$(wc -c < "$f")
    printf "%-36s %6d lines %8d bytes\n" "$f" "$lines" "$bytes"
done

echo ""
echo "=== 3. INDIVIDUAL AND LIBRARY COMPILATION CHECK (IVERILOG 12.0) ==="
mkdir -p "${W2_DIR}/build_check"
# All leaf modules in src/ (needed because filenames are lowercase e.g. buffer.v while module is Buffer)
LEAF_SRC=(
    src/BufferController.v src/buffer.v src/clkgate.v src/counter.v
    src/custom_counter.v src/demultiplier.v src/dff.v src/Disabling_mech.v
    src/example.v src/FA.v src/fifo_buffer.v src/filter_counter.v
    src/iact_SPadController.v src/ifmap_counter_nbit.v src/ifmap_fifo.v
    src/muffer.v src/multiplier.v src/mux2x1.v src/PE_Controller.v
    src/psum_addr_counter.v src/psum_controller.v src/psum_write_addr_counter.v
    src/Reg_SPad_ifmap.v src/reg_SPad.v src/SPadController.v src/SRAM_SPad.v
    src/zero_buffer.v
)
COMPILABLE_SRC=()
for f in src/*.v src/*.sv; do
    base=$(basename "$f")
    if [ ! -s "$f" ]; then
        echo "EMPTY_FILE: $f (0 bytes, no module)"
        continue
    fi
    # Determine module name inside file
    modname=$(grep -E '^[[:space:]]*module[[:space:]]+' "$f" | head -n 1 | sed -E 's/^[[:space:]]*module[[:space:]]+([A-Za-z0-9_]+).*/\1/')
    EXTRA_DEPS=()
    case "$base" in
        fifo_buffer.v) EXTRA_DEPS=(src/buffer.v) ;;
        ifmap_fifo.v)  EXTRA_DEPS=(src/muffer.v) ;;
        PE.v)          EXTRA_DEPS=("${LEAF_SRC[@]}") ;;
        PARRAY.v|Array_of_PEs.sv) EXTRA_DEPS=("${LEAF_SRC[@]}" src/PE.v) ;;
    esac
    set +e
    out=$(iverilog -g2012 -I src -s "$modname" -o "${W2_DIR}/build_check/${base}.vvp" "$f" "${EXTRA_DEPS[@]}" 2>&1)
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        if [ -n "$out" ]; then
            uniq_warn=$(echo "$out" | sort -u)
            wcnt=$(echo "$out" | grep -c "warning:" || true)
            echo "COMPILES_WITH_WARN: $f (top=$modname, ${wcnt} warnings, unique below)"
            echo "$uniq_warn" | sed 's/^/  WARN: /'
        else
            echo "COMPILES_CLEAN: $f (top=$modname)"
        fi
        COMPILABLE_SRC+=("$f")
    else
        echo "FAILS_COMPILE: $f (rc=$rc)"
        echo "$out" | head -n 15 | sed 's/^/  ERR: /'
    fi
done

echo ""
echo "=== 3b. CASE-SENSITIVITY CHECK WITH iverilog -y src ==="
for f in src/fifo_buffer.v src/ifmap_fifo.v src/PE.v; do
    set +e
    yout=$(iverilog -g2012 -I src -y src -o /dev/null "$f" 2>&1)
    yrc=$?
    set -e
    echo "iverilog -y src $f -> rc=$yrc ($(echo "$yout" | grep "missing:" -A 10 | tr '\n' ' '))"
done

echo ""
echo "=== 4. VERILATOR LINT CHECK ON KEY MODULES ==="
set +e
verilator --lint-only -Wall -Wno-DECLFILENAME -Isrc --top-module PE src/PE.v "${LEAF_SRC[@]}" 2>&1 | grep -E "%Error|%Warning-WIDTH|%Warning-UNDRIVEN|%Warning-IMPLICIT|%Warning-PINMISSING" | head -n 35
echo "--- PE_Cluster.v verilator check ---"
verilator --lint-only -Isrc src/PE_Cluster.v 2>&1 | head -n 15
set -e

echo ""
echo "=== 5. SHIPPED TESTBENCH (sim/PE_tb.sv) BUILD AND RUN ==="
set +e
tb_compile_out=$(iverilog -g2012 -I src -o "${W2_DIR}/build_check/pe_tb_shipped.vvp" "${COMPILABLE_SRC[@]}" sim/PE_tb.sv 2>&1)
tb_rc=$?
set -e
echo "PE_tb compile rc=${tb_rc} (unique warnings below):"
if [ -n "$tb_compile_out" ]; then
    echo "$tb_compile_out" | sort -u | sed 's/^/  WARN: /'
fi
vvp "${W2_DIR}/build_check/pe_tb_shipped.vvp" > "${W2_DIR}/pe_tb_shipped.out" 2>&1
echo "--- PE_tb.sv shipped stdout ---"
cat "${W2_DIR}/pe_tb_shipped.out"

echo ""
echo "=== 6. NON-INTRUSIVE TRACE OF UNMODIFIED PE_tb.sv + PE.v ==="
mkdir -p "${W2_DIR}/trace_work"
cp -r src sim "${W2_DIR}/trace_work/"
cat > "${W2_DIR}/trace_work/pe_probe.sv" << 'EOF'
`timescale 1ns/1ns
module pe_probe;
    integer idx;
    always @(posedge PE_tb.clk) begin
        if (PE_tb.dut.iact_counter_en)
            $display("[T=%0t] IACT_SPAD_WRITE: addr=%0d data=%0d", $time, PE_tb.dut.ifmap_write_cout, PE_tb.dut.iact_buffer_read_data);
        if (PE_tb.dut.weight_counter_en)
            $display("[T=%0t] WEIGHT_SPAD_WRITE: addr=%0d data=%0d", $time, PE_tb.dut.filter_write_cout, PE_tb.dut.weight_buffer_read_data);
        if (PE_tb.dut.pipe_en) begin
            $display("[T=%0t] PIPE: iread=%0d(%0d) wread=%0d(%0d) zf=%b | ri=%0d rw=%0d | mul=%0d mv=%b | paddr=%0d pdis=%b ptomux=%0d sum=%0d | pmux=%h reg_pmux=%h pwen=%b out=%h",
                $time,
                PE_tb.dut.ifmap_read_cout, PE_tb.dut.ifmap_spad_read_data,
                PE_tb.dut.filter_read_cout, PE_tb.dut.filter_spad_read_data,
                PE_tb.dut.zero_flag,
                PE_tb.dut.reg_iact_data, PE_tb.dut.reg_filter_data,
                PE_tb.dut.mult_data_out, PE_tb.dut.mul_out_valid,
                PE_tb.dut.psum_write_addr_fr, PE_tb.dut.psum_zero_flag,
                PE_tb.dut.psum_tomux, PE_tb.dut.sum_adder_out,
                PE_tb.dut.psum_mux_out, PE_tb.dut.reg_out_psum_mux,
                PE_tb.dut.psum_write_cnt_en_reg, PE_tb.data_psum_out);
        end
    end
    initial begin
        #378;
        $display("=== FINAL INTERNAL STATE OF PE_tb.dut AT T=378 ===");
        for (idx = 0; idx < 12; idx = idx + 1)
            $display("  ifmap_spad.regs[%0d] = %0d", idx, PE_tb.dut.ifmap_spad.regs[idx]);
        for (idx = 0; idx < 8; idx = idx + 1)
            $display("  filter_spad.mem[%0d] = %0d", idx, PE_tb.dut.filter_spad.mem[idx]);
        for (idx = 0; idx < 6; idx = idx + 1)
            $display("  psum_spad.regs[%0d]  = %0d (hex %h)", idx, PE_tb.dut.psum_spad.regs[idx], PE_tb.dut.psum_spad.regs[idx]);
        for (idx = 0; idx < 6; idx = idx + 1)
            $display("  output_psum_fifo.mem[%0d] = %h", idx, PE_tb.dut.output_psum_fifo.buffer.memory[idx]);
        $display("  psum_mux_out = %h, reg_out_psum_mux = %h",
            PE_tb.dut.psum_mux_out, PE_tb.dut.reg_out_psum_mux);
    end
endmodule
EOF

iverilog -g2012 -I "${W2_DIR}/trace_work/src" -o "${W2_DIR}/trace_work/pe_tb_probe.vvp" \
    "${COMPILABLE_SRC[@]}" sim/PE_tb.sv "${W2_DIR}/trace_work/pe_probe.sv" 2>/dev/null
vvp "${W2_DIR}/trace_work/pe_tb_probe.vvp" > "${W2_DIR}/trace_work/pe_tb_probe.out"
grep -E "IACT_SPAD_WRITE|WEIGHT_SPAD_WRITE|PIPE:|FINAL INTERNAL|spad|output_psum_fifo|psum_mux_out" "${W2_DIR}/trace_work/pe_tb_probe.out"

echo ""
echo "=== 7. REPAIR TRACE COPY IN work/ (DRIVE psum_mux_out & ISOLATE GUT) ==="
cp src/PE.v "${W2_DIR}/trace_work/PE_fixed.v"
sed -i 's/wire \[PSUM_SPAD_WIDTH-1:0\] psum_mux_out;/wire [PSUM_SPAD_WIDTH-1:0] psum_mux_out = psum_tomux;/' "${W2_DIR}/trace_work/PE_fixed.v"
sed -i 's/parameter PSUM_PAR_READ = 4/parameter PSUM_PAR_READ = 1/' "${W2_DIR}/trace_work/PE_fixed.v"

cp sim/PE_tb.sv "${W2_DIR}/trace_work/PE_tb_fixed.sv"
sed -i '/PEGRID GUT/,/);/ s/\.data_psum_out(data_psum_out)/\.data_psum_out()/' "${W2_DIR}/trace_work/PE_tb_fixed.sv"
sed -i '/PEGRID GUT/,/);/ s/\.iact_buffer_ready(iact_buffer_ready)/\.iact_buffer_ready()/' "${W2_DIR}/trace_work/PE_tb_fixed.sv"
sed -i '/PEGRID GUT/,/);/ s/\.weight_buffer_ready(weight_buffer_ready)/\.weight_buffer_ready()/' "${W2_DIR}/trace_work/PE_tb_fixed.sv"
sed -i '/PEGRID GUT/,/);/ s/\.psum_out_valid(psum_out_valid)/\.psum_out_valid()/' "${W2_DIR}/trace_work/PE_tb_fixed.sv"

OTHER_SRC=()
for f in "${COMPILABLE_SRC[@]}"; do
    if [ "$f" != "src/PE.v" ]; then
        OTHER_SRC+=("$f")
    fi
done
iverilog -g2012 -I src -o "${W2_DIR}/trace_work/pe_tb_fixed.vvp" "${OTHER_SRC[@]}" "${W2_DIR}/trace_work/PE_fixed.v" "${W2_DIR}/trace_work/PE_tb_fixed.sv" 2>/dev/null
vvp "${W2_DIR}/trace_work/pe_tb_fixed.vvp"
