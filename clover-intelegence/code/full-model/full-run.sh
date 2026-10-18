#!/bin/bash
# usage: full-run.sh recorded        words = the text the recorded 1000 ids stand for; embeddings and layer 92 output are compared with the pod's chain
#        full-run.sh words COUNT     words = the first COUNT words of text.txt; no recorded values exist, nothing is compared
#        full-run.sh text FILE       the prompt is the bytes of FILE; PF_DECODE=N generates up to N more tokens; PF_CHAT=1|low|high|max asks as a chat message
# needs in /tmp/one-layer: input-stage.c layer0-batch.c layer-batch.c pipeline-full.c ids-to-words.c (host copies), tokfiles/, text.txt, tok-1000.txt,
#   and for `recorded`: refs/embed.f32 (= l1in-1000.s0.f32), refs/final.resid.f32, refs/final.snaps.f32 (the pod chain's layer 92 output, from layer-chain.sh)
set -u
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
# USE_4BIT_TRUNK=1 packs the whole trunk (layers 0 to 92: attention, gates, dense and shared-expert matrices) to 4 bits, for the prompt pass and for decode.
# Experts are already 4-bit and stay as they are. Default (unset or 0) is exact = bit-identical to the pod.
# Not bit-identical; ~29 GB more RAM, so the trunk files are not locked in memory (CLOVER_LOCK_TRUNK=0).
USE_4BIT_TRUNK="${USE_4BIT_TRUNK:-0}"
[ "$USE_4BIT_TRUNK" != "0" ] && CLOVER_LOCK_TRUNK="${CLOVER_LOCK_TRUNK:-0}"
cd $H
F="-std=gnu11 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function"
TOK="-I/root/kimi-k3-in-c/src/tokenizer -I/root/kimi-k3-in-c/third_party"
if [ -z "${PF_NOBUILD:-}" ]; then
sed -E "s/^#define TRANSFORMER_LAYER 2\$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1\$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\(TRANSFORMER_LAYER - 1\)/(2 - 1)/; s/\"operators\/qkv-all\/layer-2\/operator.bin\"/lb_name(0)/; s/\"trunk-2\"/lb_name(1)/; s/\"root-2\"/lb_name(2)/; s/root_open\(root_directory, 2\)/root_open(root_directory, lb_layer_no)/" $D/tansformers/transformer-2/transformer-2.c > tpl-kda.c
sed -E "s/^#define TRANSFORMER_LAYER 3\$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1\$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\(TRANSFORMER_LAYER - 1\)/(3 - 1)/; s/\"operators\/qkv-all\/layer-3\/operator.bin\"/lb_name(0)/; s/\"trunk-3\"/lb_name(1)/; s/\"root-3\"/lb_name(2)/; s/root_open\(root_directory, 3\)/root_open(root_directory, lb_layer_no)/" $D/tansformers/transformer-3/transformer-3.c > tpl-mla.c
fail() { echo "build failed: $1"; grep -E "error" $2 | head -20; exit 1; }
gcc -std=gnu11 -O2 $TOK ids-to-words.c -lm -o ids-to-words 2> build-pw.log || fail ids-to-words build-pw.log
gcc $F -DPIPELINE_ENTRY -O2 $TOK -I$D/server -Dmain=input_stage_main -c input-stage.c -o f-input.o 2> build-f1.log || fail input-stage build-f1.log
gcc $F -DPIPELINE_ENTRY -O2 -funroll-loops -I$D/server -Dmain=layer0_batch_main -c layer0-batch.c -o f-layer0.o 2> build-f2.log || fail layer0 build-f2.log
gcc $F -DPIPELINE_ENTRY -O3 -I$D/common -I$D/tansformers/transformer-1 -Dmain=layer_batch_main -c layer-batch.c -o f-layer1.o 2> build-f3.log || fail layer1 build-f3.log
gcc $F -DLB_GENERIC -DLB_ENTRY -DLB_PFX=lbk -DLB_TEMPLATE='"tpl-kda.c"' -O3 -I$D/common -I. -Dmain=lbk_main_unused -c layer-batch.c -o f-kda.o 2> build-f4.log || fail kda build-f4.log
gcc $F -DLB_GENERIC -DLB_ENTRY -DLB_PFX=lbm -DLB_TEMPLATE='"tpl-mla.c"' -O3 -I$D/common -I. -Dmain=lbm_main_unused -c layer-batch.c -o f-mla.o 2> build-f5.log || fail mla build-f5.log
gcc $F -O2 -c pipeline-full.c -o f-main.o 2> build-f6.log || fail main build-f6.log
# every object keeps only its entry points global, so the several programs built from the same sources cannot collide
objcopy --keep-global-symbol=stage1_setup --keep-global-symbol=stage1_ids --keep-global-symbol=stage1_ids_plain --keep-global-symbol=stage1_id_of --keep-global-symbol=stage1_lookup f-input.o
objcopy --keep-global-symbol=stage2_setup --keep-global-symbol=stage2_run --keep-global-symbol=stage2_step --keep-global-symbol=stage2_stepk --keep-global-symbol=stage2_save_state --keep-global-symbol=stage2_restore_state f-layer0.o
objcopy --keep-global-symbol=stage3_setup --keep-global-symbol=stage3_run --keep-global-symbol=stage3_step --keep-global-symbol=stage3_stepk --keep-global-symbol=stage3_save_state --keep-global-symbol=stage3_restore_state --keep-global-symbol=stage3_set_scratch --keep-global-symbol=stage3_use_buffer --keep-global-symbol=stage3_route_log --keep-global-symbol=stage3_routes --keep-global-symbol=stage3_hits --keep-global-symbol=stage3_set_slots --keep-global-symbol=stage3_cache_stats --keep-global-symbol=stage3_guess_stats --keep-global-symbol=stage3_laps f-layer1.o
for p in lbk:f-kda.o lbm:f-mla.o; do
  pre=${p%%:*}; obj=${p##*:}
  objcopy --keep-global-symbol=${pre}_set_buffer --keep-global-symbol=${pre}_init --keep-global-symbol=${pre}_open --keep-global-symbol=${pre}_run --keep-global-symbol=${pre}_step --keep-global-symbol=${pre}_stepk --keep-global-symbol=${pre}_save_state --keep-global-symbol=${pre}_restore_state --keep-global-symbol=${pre}_set_scratch --keep-global-symbol=${pre}_distinct --keep-global-symbol=${pre}_route_log --keep-global-symbol=${pre}_routes --keep-global-symbol=${pre}_hits --keep-global-symbol=${pre}_set_slots --keep-global-symbol=${pre}_cache_stats --keep-global-symbol=${pre}_guess_stats --keep-global-symbol=${pre}_laps $obj
done
gcc -fopenmp f-main.o f-input.o f-layer0.o f-layer1.o f-kda.o f-mla.o -lm -lpthread -ldl -o pipeline-full 2> build-f7.log || { head -30 build-f7.log; exit 1; }
echo "built; warnings: $(cat build-f?.log | grep -c warning)"
fi
if [ "${1:-recorded}" = "text" ]; then
  WORDS=$2; REFS="-"
elif [ "${1:-recorded}" = "words" ]; then
  python3 -c "
import re,sys
t=open('text.txt',encoding='utf-8').read()
m=list(re.finditer(r'\S+',t))[int(sys.argv[1])-1]
open('words-run.txt','w',encoding='utf-8').write(t[:m.end()])" "${2:-1000}"
  WORDS=words-run.txt; REFS="-"
else
  ./ids-to-words tokfiles tok-1000.txt words-1000.txt; WORDS=words-1000.txt; REFS=refs
fi
ulimit -l unlimited; ulimit -c 0
[ -n "${PF_EXTRA:-}" ] && REFS="-"
echo "PROMPT ($(wc -w < $WORDS) words, $(wc -c < $WORDS) bytes):"; echo "-----"; cat $WORDS; echo; echo "-----"
env -i PATH="$PATH" LB_IO="${LB_IO:-4}" LB_SPEC="${LB_SPEC:-0}" LB_GUESS="${LB_GUESS:-0}" LB_THP="${LB_THP:-1}" LB_EXPK="${LB_EXPK:-1}" USE_4BIT_TRUNK="$USE_4BIT_TRUNK" SPECULATIVE_DRAFT_TOKENS="${SPECULATIVE_DRAFT_TOKENS:-}" CLOVER_LOCK_TRUNK="${CLOVER_LOCK_TRUNK:-}" PF_VERBOSE="${PF_VERBOSE:-}" \
    PF_DECODE="${PF_DECODE:-}" PF_EXTRA="${PF_EXTRA:-}" PF_SAVE_ROWS="${PF_SAVE_ROWS:-}" PF_CMP_ROWS="${PF_CMP_ROWS:-}" PF_SAVE_IDS="${PF_SAVE_IDS:-}" PF_ROUTES="${PF_ROUTES:-}" PF_CHAT="${PF_CHAT:-}" PF_SYSTEM="${PF_SYSTEM:-}" PF_CACHE="${PF_CACHE:-}" \
    PF_SWEEP="${PF_SWEEP:-}" PF_SWEEP_IDS="${PF_SWEEP_IDS:-}" PF_SWEEP_N="${PF_SWEEP_N:-}" LB_SWEEP_EXPK="${LB_SWEEP_EXPK:-0}" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./pipeline-full "$D" $DATA tokfiles $WORDS $REFS
