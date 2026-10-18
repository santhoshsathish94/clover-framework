#!/bin/bash
# Builds the whole-model program (unless BUILD=0) and runs one request through all 93 layers.
#
# usage: run-model.sh recorded        the text the recorded 1000 token ids stand for; embeddings and layer 92 output are compared with the pod's own chain
#        run-model.sh words COUNT     the first COUNT words of inputs/text.txt; nothing is compared
#        run-model.sh text FILE       the prompt is the bytes of FILE
# Settings (environment):
#   USE_4BIT_TRUNK=1        the trunk (layers 0-92: attention, gates, dense and shared-expert matrices) is packed to 4 bits in RAM and used in DECODE steps only.
#                           Not bit-identical to the pod, about 29 GB more RAM (so the trunk files are not locked in memory). Decode is memory-bound, so half
#                           the bytes is faster; the prompt pass is compute-bound, where the 4-bit kernel is slower (1,000 tokens: 355 s against 301 s), so it stays exact.
#   USE_8BIT_EXPERT_ACTIVATIONS=1   prompt pass: the experts take their input rounded to 8 bits (experimental, not bit-identical)
#   USE_8BIT_TRUNK_ACTIVATIONS=1    prompt pass of layers 1-92: the trunk matrices take their input rounded to 8 bits (experimental, not bit-identical)
#   PF_DECODE=N             generate up to N more tokens after the prompt
#   PF_CACHE=C              keep C experts per layer in RAM between decode steps (16 is the usual choice)
#   SPECULATIVE_DRAFT_TOKENS=D   speculative decoding with up to D draft tokens per pass (D at most 15); the tokens are the same as plain decoding
#   PF_CHAT=1|low|high|max  ask the prompt as a chat message (1 = answer at once; low/high/max = think first); PF_SYSTEM adds a system message
#   PF_SAVE_IDS=file        write the generated ids to file (relative paths land in work/results)
#   BUILD=0                 skip the build;   BUILD_ONLY=1   build and stop
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
USE_4BIT_TRUNK="${USE_4BIT_TRUNK:-0}"
[ "$USE_4BIT_TRUNK" != "0" ] && CLOVER_LOCK_TRUNK="${CLOVER_LOCK_TRUNK:-0}"
[ "${USE_8BIT_EXPERT_ACTIVATIONS:-0}" != "0" ] && LB_EXPK=3
[ "${USE_8BIT_TRUNK_ACTIVATIONS:-0}" != "0" ] && LB_A8T=1
SRC="$ENGINE_DIR/src"
F="-std=gnu11 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function"
TOK="-I$TOKENIZER_DIR/src/tokenizer -I$TOKENIZER_DIR/third_party"
cd "$BUILD_DIR"
if [ "${BUILD:-1}" != "0" ] && [ -z "${PF_NOBUILD:-}" ]; then
  # layers 2-92 are built from the deployed pod sources of layer 2 (attention kind KDA) and layer 3 (MLA), with the layer number made a run-time value
  sed -E "s/^#define TRANSFORMER_LAYER 2\$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1\$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\(TRANSFORMER_LAYER - 1\)/(2 - 1)/; s/\"operators\/qkv-all\/layer-2\/operator.bin\"/lb_name(0)/; s/\"trunk-2\"/lb_name(1)/; s/\"root-2\"/lb_name(2)/; s/root_open\(root_directory, 2\)/root_open(root_directory, lb_layer_no)/" $POD_DIR/tansformers/transformer-2/transformer-2.c > template-kda.c
  sed -E "s/^#define TRANSFORMER_LAYER 3\$/#define TRANSFORMER_LAYER lb_layer_no/; s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1\$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/; s/\(TRANSFORMER_LAYER - 1\)/(3 - 1)/; s/\"operators\/qkv-all\/layer-3\/operator.bin\"/lb_name(0)/; s/\"trunk-3\"/lb_name(1)/; s/\"root-3\"/lb_name(2)/; s/root_open\(root_directory, 3\)/root_open(root_directory, lb_layer_no)/" $POD_DIR/tansformers/transformer-3/transformer-3.c > template-mla.c
  fail() { echo "build failed: $1"; grep -E "error" $2 | head -20; exit 1; }
  gcc -std=gnu11 -O2 $TOK $SRC/ids-to-text.c -lm -o ids-to-text 2> build-ids-to-text.log || fail ids-to-text build-ids-to-text.log
  gcc $F -DPIPELINE_ENTRY -O2 $TOK -I$POD_DIR/server -Dmain=words_to_embeddings_main -c $SRC/words-to-embeddings.c -o words-to-embeddings.o 2> build-words-to-embeddings.log || fail words-to-embeddings build-words-to-embeddings.log
  gcc $F -DPIPELINE_ENTRY -O2 -funroll-loops -I$POD_DIR/server -Dmain=layer0_main -c $SRC/layer0.c -o layer0.o 2> build-layer0.log || fail layer0 build-layer0.log
  gcc $F -DPIPELINE_ENTRY -O3 -I$POD_DIR/common -I$POD_DIR/tansformers/transformer-1 -Dmain=layer1_main -c $SRC/layers-1-to-92.c -o layer1.o 2> build-layer1.log || fail layer1 build-layer1.log
  gcc $F -DLB_GENERIC -DLB_ENTRY -DLB_PFX=lbk -DLB_TEMPLATE='"template-kda.c"' -O3 -I$POD_DIR/common -I. -Dmain=kda_main_unused -c $SRC/layers-1-to-92.c -o layers-kda.o 2> build-layers-kda.log || fail layers-kda build-layers-kda.log
  gcc $F -DLB_GENERIC -DLB_ENTRY -DLB_PFX=lbm -DLB_TEMPLATE='"template-mla.c"' -O3 -I$POD_DIR/common -I. -Dmain=mla_main_unused -c $SRC/layers-1-to-92.c -o layers-mla.o 2> build-layers-mla.log || fail layers-mla build-layers-mla.log
  gcc $F -O2 -c $SRC/whole-model.c -o whole-model.o 2> build-whole-model.log || fail whole-model build-whole-model.log
  # every object keeps only its entry points global, so the several programs built from the same sources cannot collide
  objcopy --keep-global-symbol=stage1_setup --keep-global-symbol=stage1_ids --keep-global-symbol=stage1_ids_plain --keep-global-symbol=stage1_id_of --keep-global-symbol=stage1_lookup words-to-embeddings.o
  objcopy --keep-global-symbol=stage2_setup --keep-global-symbol=stage2_run --keep-global-symbol=stage2_step --keep-global-symbol=stage2_stepk --keep-global-symbol=stage2_keep_positions --keep-global-symbol=stage2_use_position layer0.o
  objcopy --keep-global-symbol=stage3_setup --keep-global-symbol=stage3_run --keep-global-symbol=stage3_step --keep-global-symbol=stage3_stepk --keep-global-symbol=stage3_keep_positions --keep-global-symbol=stage3_use_position --keep-global-symbol=stage3_set_scratch --keep-global-symbol=stage3_use_buffer --keep-global-symbol=stage3_route_log --keep-global-symbol=stage3_routes --keep-global-symbol=stage3_hits --keep-global-symbol=stage3_set_slots --keep-global-symbol=stage3_cache_stats --keep-global-symbol=stage3_guess_stats --keep-global-symbol=stage3_laps layer1.o
  for p in lbk:layers-kda.o lbm:layers-mla.o; do
    pre=${p%%:*}; obj=${p##*:}
    objcopy --keep-global-symbol=${pre}_set_buffer --keep-global-symbol=${pre}_init --keep-global-symbol=${pre}_open --keep-global-symbol=${pre}_run --keep-global-symbol=${pre}_step --keep-global-symbol=${pre}_stepk --keep-global-symbol=${pre}_keep_positions --keep-global-symbol=${pre}_use_position --keep-global-symbol=${pre}_set_scratch --keep-global-symbol=${pre}_distinct --keep-global-symbol=${pre}_route_log --keep-global-symbol=${pre}_routes --keep-global-symbol=${pre}_hits --keep-global-symbol=${pre}_set_slots --keep-global-symbol=${pre}_cache_stats --keep-global-symbol=${pre}_guess_stats --keep-global-symbol=${pre}_laps $obj
  done
  gcc -fopenmp whole-model.o words-to-embeddings.o layer0.o layer1.o layers-kda.o layers-mla.o -lm -lpthread -ldl -o whole-model 2> build-link.log || { head -30 build-link.log; exit 1; }
  echo "built; warnings: $(cat build-*.log | grep -c warning)"
fi
[ "${BUILD_ONLY:-0}" = "1" ] && exit 0
cd "$RESULTS_DIR"
if [ "${1:-recorded}" = "text" ]; then
  WORDS=$2; REFS="-"
elif [ "${1:-recorded}" = "words" ]; then
  python3 -c "
import re,sys
t=open(sys.argv[2],encoding='utf-8').read()
m=list(re.finditer(r'\S+',t))[int(sys.argv[1])-1]
open('words-run.txt','w',encoding='utf-8').write(t[:m.end()])" "${2:-1000}" "$INPUTS_DIR/text.txt"
  WORDS=words-run.txt; REFS="-"
else
  "$BUILD_DIR/ids-to-text" "$INPUTS_DIR/tokfiles" "$INPUTS_DIR/tok-1000.txt" words-1000.txt; WORDS=words-1000.txt; REFS="$INPUTS_DIR/refs"
fi
ulimit -l unlimited; ulimit -c 0
[ -n "${PF_EXTRA:-}" ] && REFS="-"
echo "PROMPT ($(wc -w < $WORDS) words, $(wc -c < $WORDS) bytes):"; echo "-----"; cat $WORDS; echo; echo "-----"
env -i PATH="$PATH" LB_IO="${LB_IO:-4}" LB_SPEC="${LB_SPEC:-0}" LB_GUESS="${LB_GUESS:-0}" LB_THP="${LB_THP:-1}" LB_EXPK="${LB_EXPK:-1}" LB_A8T="${LB_A8T:-}" USE_4BIT_TRUNK="$USE_4BIT_TRUNK" SPECULATIVE_DRAFT_TOKENS="${SPECULATIVE_DRAFT_TOKENS:-}" CLOVER_LOCK_TRUNK="${CLOVER_LOCK_TRUNK:-}" PF_VERBOSE="${PF_VERBOSE:-}" \
    PF_DECODE="${PF_DECODE:-}" PF_EXTRA="${PF_EXTRA:-}" PF_SAVE_ROWS="${PF_SAVE_ROWS:-}" PF_CMP_ROWS="${PF_CMP_ROWS:-}" PF_SAVE_IDS="${PF_SAVE_IDS:-}" PF_ROUTES="${PF_ROUTES:-}" PF_CHAT="${PF_CHAT:-}" PF_SYSTEM="${PF_SYSTEM:-}" PF_CACHE="${PF_CACHE:-}" \
    PF_SWEEP="${PF_SWEEP:-}" PF_SWEEP_IDS="${PF_SWEEP_IDS:-}" PF_SWEEP_N="${PF_SWEEP_N:-}" LB_SWEEP_EXPK="${LB_SWEEP_EXPK:-0}" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BUILD_DIR/whole-model" "$POD_DIR" "$DATASET_DIR" "$INPUTS_DIR/tokfiles" $WORDS $REFS
