#!/bin/sh
# Step 2: does a longer context cost more time per token?
#
# A single run feeds each generated token back, so context sweeps from the seed
# length upward and one run measures the whole slope. The longest prompt in
# prompts.tsv is used so the sweep reaches as far as the 256-position capacity
# (max_input_tokens + max_output_tokens) allows.
#
# KDA layers hold fixed-size recurrent state and should not grow with context.
# MLA layers attend over every stored position, so SA and the MLA path should.
# Separating those is the point: if the growth is confined to 24 of 93 layers,
# the slope says what long context would actually cost.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
TSV=/opt/clover-k3/prompts.tsv

row=$(sort -t"$(printf '\t')" -k2,2n "$TSV" | tail -1 | cut -f1)
len=$(awk -F'\t' -v r="$row" '$1==r {print $2}' "$TSV")
echo "longest prompt: row $row, $len tokens"

ROW="$row" TOK="${TOK:-100}" BIN="$S/bin/clover-one" OUT=/tmp/ctx sh /tmp/ta2.sh
echo "context run finished"
