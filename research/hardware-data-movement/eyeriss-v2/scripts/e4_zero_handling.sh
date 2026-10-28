#!/usr/bin/env bash
# E4 (Eyeriss v2): does the repository's PE.sv change cycles / MAC enables / SPad reads when data are zero?
# Same two cosmetic patches as E2 (silence SPad dump; remove nested comment), on a fresh copy of rtl/.
set -euo pipefail
W2="/root/dm-study/eyeriss-v2/w2"
ORIG_REPO="/root/dm-study/eyeriss-v2/repo/CSE240D-Hierarchical_Mesh_NoC-Eyeriss_v2"
E4_DIR="$W2/e4_work"
rm -rf "$E4_DIR"; mkdir -p "$E4_DIR"
cp -r "$ORIG_REPO/rtl" "$E4_DIR/rtl"
sed -i 's/\$display("\\t\\t\\t\\t\\t Current Status:/if (0) \$display("\\t\\t\\t\\t\\t Current Status:/' "$E4_DIR/rtl/SPad.sv"
sed -i 's/\/\/\*\*\*Loading of weights starts at index 0\*\*\*/\/\/ Loading of weights starts at index 0/' "$E4_DIR/rtl/PE.sv"
cp "$W2/tb_e4_pe_zero.sv" "$E4_DIR/"

echo "=== 1. PE with zero weights / zero activations (Verilator) ==="
verilator --binary --timing -j 6 -Wno-lint -I"$E4_DIR/rtl" --Mdir "$E4_DIR/v_zero" --top-module tb_e4_pe_zero "$E4_DIR/tb_e4_pe_zero.sv" > /dev/null
"$E4_DIR/v_zero/Vtb_e4_pe_zero" | grep '^CASE'

echo
echo "=== 2. Does any RTL file mention zero-skipping, CSC, count/run-length, or SIMD? (case-insensitive grep over rtl/) ==="
grep -rniE "csc|compress|run.?length|skip|simd|zero_flag|is_zero|== *0 *\)" "$ORIG_REPO/rtl" --include=*.sv | grep -vE "^[^:]+:[0-9]+:\s*//" | head -20 || true
echo "(end of grep)"

echo
echo "=== 3. PE.sv: number of SPads, MAC instances, data width ==="
grep -cE "^\s*SPad " "$ORIG_REPO/rtl/PE.sv" | sed 's/^/SPad instances in PE.sv: /'
grep -cE "^\s*MAC " "$ORIG_REPO/rtl/PE.sv" | sed 's/^/MAC instances in PE.sv: /'
grep -nE "parameter DATA_WIDTH" "$ORIG_REPO/rtl/PE.sv"
