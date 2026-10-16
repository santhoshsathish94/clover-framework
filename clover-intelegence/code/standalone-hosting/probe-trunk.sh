#!/bin/sh
# What is the trunk, really? Read the source, not my memory of it.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting

echo "===== 1. how ops are named and accounted ====="
grep -n 'OPSTAT\|op_begin\|op_end\|op_record\|OPS\[' $S/clover-one.c | head -30

echo
echo "===== 2. the op name table ====="
grep -n 'static const char \*\|op_names\|OPNAME' $S/clover-one.c | head -20

echo
echo "===== 3. anything called trunk / root weights ====="
grep -n 'trunk\|TRUNK' $S/clover-one.c | head -30

echo
echo "===== 4. the int8 projection op itself ====="
grep -n 'int8\|INT8\|q_int8\|qint8' $S/clover-one.c | head -30

echo
echo "===== 5. per-layer weight slices: how a layer gets its weights ====="
grep -n 'layer_weights\|weights_for_layer\|layer_base\|per_layer' $S/clover-one.c | head -20
