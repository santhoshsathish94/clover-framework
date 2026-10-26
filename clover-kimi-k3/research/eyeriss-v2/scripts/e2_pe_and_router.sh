#!/usr/bin/env bash
set -euo pipefail

W2="/root/dm-study/eyeriss-v2/w2"
ORIG_REPO="/root/dm-study/eyeriss-v2/repo/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2"
E2_DIR="$W2/e2_work"

rm -rf "$E2_DIR"
mkdir -p "$E2_DIR"
cp -r "$ORIG_REPO/rtl" "$E2_DIR/rtl"

# Silence SPad per-cycle dump and fix nested comment in PE.sv in the fresh copy
sed -i 's/\$display("\\t\\t\\t\\t\\t Current Status:/if (0) \$display("\\t\\t\\t\\t\\t Current Status:/' "$E2_DIR/rtl/SPad.sv"
sed -i 's/\/\/\*\*\*Loading of weights starts at index 0\*\*\*/\/\/ Loading of weights starts at index 0/' "$E2_DIR/rtl/PE.sv"

cp "$W2/tb_e2_pe_row.sv" "$E2_DIR/"
cp "$W2/tb_e2_router_modes.sv" "$E2_DIR/"

echo "=== 1. BUILD & RUN tb_e2_pe_row (Verilator & iverilog) ==="
verilator --binary --timing -j 6 -Wno-lint -I"$E2_DIR/rtl" --Mdir "$E2_DIR/v_pe" --top-module tb_e2_pe_row "$E2_DIR/tb_e2_pe_row.sv" > /dev/null
"$E2_DIR/v_pe/Vtb_e2_pe_row" > "$E2_DIR/pe_verilator.log"
cat "$E2_DIR/pe_verilator.log"

iverilog -g2012 -Y .sv -y "$E2_DIR/rtl" -s tb_e2_pe_row -o "$E2_DIR/pe_iv.vvp" "$E2_DIR/tb_e2_pe_row.sv"
vvp "$E2_DIR/pe_iv.vvp" > "$E2_DIR/pe_iverilog.log"
echo "iverilog vs verilator PE output diff lines: $(diff -u <(grep -E '^(===|ITER|PULSE)' "$E2_DIR/pe_verilator.log") <(grep -E '^(===|ITER|PULSE)' "$E2_DIR/pe_iverilog.log") | wc -l)"

echo ""
echo "=== 2. BUILD & RUN tb_e2_router_modes (Verilator) ==="
verilator --binary --timing -j 6 -Wno-lint -I"$E2_DIR/rtl" --Mdir "$E2_DIR/v_router" --top-module tb_e2_router_modes "$E2_DIR/tb_e2_router_modes.sv" > /dev/null
"$E2_DIR/v_router/Vtb_e2_router_modes" > "$E2_DIR/router_verilator.log"
cat "$E2_DIR/router_verilator.log"

echo ""
echo "=== 3. PYTHON HAND-CALCULATION VERIFICATION & NEGATIVE CONTROLS ==="
/root/ncm-cpu/venv/bin/python - << 'PYEOF'
import pathlib, re

e2 = pathlib.Path("/root/dm-study/eyeriss-v2/w2/e2_work")
pe_log = (e2 / "pe_verilator.log").read_text()
rt_log = (e2 / "router_verilator.log").read_text()

# Verify PE normal vs corrupted run
iters = re.findall(r"ITER (\d+): pe_out=(\d+) spad_psum\[(\d+)\]=(\d+) cycles=(\d+)", pe_log)
assert len(iters) == 6, f"Expected 6 ITER lines, got {len(iters)}"
normal_out = [int(iters[i][1]) for i in range(3)]
normal_spad = [int(iters[i][3]) for i in range(3)]
normal_cyc = [int(iters[i][4]) for i in range(3)]
corrupt_out = [int(iters[i][1]) for i in range(3, 6)]

# Hand-computed expected values:
# w_row0 = [10, 20, 30]; act rows = [2,3,4], [7,8,9], [12,13,14]
exp_normal = [10*2 + 20*3 + 30*4, 10*7 + 20*8 + 30*9, 10*12 + 20*13 + 30*14]
exp_corrupt = [10*2 + 21*3 + 30*4, 10*7 + 21*8 + 30*9, 10*12 + 21*13 + 30*14]
print(f"PE normal:  got={normal_out} spad={normal_spad} cycles={normal_cyc} | expected={exp_normal}")
print(f"PE corrupt: got={corrupt_out} | expected_corrupt={exp_corrupt} | fails_normal_check={corrupt_out != exp_normal}")
assert normal_out == exp_normal == [200, 500, 800]
assert normal_spad == [200, 500, 800]
assert corrupt_out == exp_corrupt == [203, 508, 813]
assert corrupt_out != exp_normal

# Verify router enable masks across all 16 modes
exp_masks = {
    0: "1111", 1: "1000", 2: "0100", 3: "0010", 4: "0001",
    5: "1001", 6: "0101", 7: "0011", 8: "1010", 9: "0110", 10: "0011",
    11: "0000", 12: "0000", 13: "0000", 14: "0000", 15: "0000"
}
got_masks = {}
for m_str, mask in re.findall(r"MODE (\d+):\S+\s+\|\s+en_mask\(N,S,W,E\)=(\d{4})", rt_log):
    got_masks[int(m_str)] = mask
print(f"Router masks match all 16 modes: {got_masks == exp_masks}")
assert got_masks == exp_masks
print("ALL E2 CHECKS PASSED.")
PYEOF
