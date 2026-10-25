#!/bin/bash
# Like run-model.sh, but the layers run on the GPUs of this machine (src/code/gpu/backend-gpu.cu in place of the CPU engine): the prompt pass and decoding,
# the experts brought to the cards after each layer's router (card pool -> pinned RAM cache -> drive). The driver (chat mode, DSpark speculative decoding, the
# tail) is the same program as the CPU build's.
#
# usage: run-model-gpu.sh text FILE        the prompt is the bytes of FILE
# Settings (environment): those of run-model.sh that the driver reads (PF_DECODE, PF_CHAT, PF_SYSTEM, PF_SAVE_IDS, PF_LAST_LAYER, PF_SAVE_ROWS, PF_ROUTES,
#   SPECULATIVE_DRAFT_TOKENS, SPECULATIVE_DRAFTER, ...), and K3GPU_CARDS, K3GPU_HOST_GB, K3GPU_RESERVE_MB, K3GPU_READERS, K3GPU_VERBOSE, K3GPU_COPYQ and K3GPU_COPYQ_LIB, K3GPU_SLOTCACHE and K3GPU_SLOTCACHE_LIB (see backend-gpu.cu).
#   BUILD=0 skips the build, BUILD_ONLY=1 builds and stops. CUDA_HOME (default /usr/local/cuda-12.6) names the toolkit.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
CMN="$CODE_DIR/common"; CPU="$CODE_DIR/cpu"; GPU="$CODE_DIR/gpu"
INC="-I$CMN -I$CPU"
F="-std=gnu11 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function"
TOK="-I$TOKENIZER_DIR/src/tokenizer -I$TOKENIZER_DIR/third_party"
BUILD_GPU="${BUILD_GPU_DIR:-$BUILD_DIR/gpu}"
COPYQ_INC=""; [ -d "$CODE_DIR/../rust/copyq/include" ] && COPYQ_INC="-I$CODE_DIR/../rust/copyq/include"   # contract of the Rust copy queue (K3GPU_COPYQ=rust); without it the engine has only the C++ queue
SLOTCACHE_INC=""; [ -d "$CODE_DIR/../rust/slotcache/include" ] && SLOTCACHE_INC="-I$CODE_DIR/../rust/slotcache/include"   # contract of the Rust slot cache (K3GPU_SLOTCACHE=rust); without it the engine has only the C++ cache
ROUTETABLE_INC=""; [ -d "$CODE_DIR/../rust/routetable/include" ] && ROUTETABLE_INC="-I$CODE_DIR/../rust/routetable/include"   # contract of the Rust route table (K3GPU_ROUTETABLE=rust); without it the engine has only the C++ parser
mkdir -p "$BUILD_GPU"
cd "$BUILD_GPU"
if [ "${BUILD:-1}" != "0" ]; then
  fail() { echo "build failed: $1"; grep -E "error" $2 | head -20; exit 1; }
  gcc -std=gnu11 -O2 $TOK $INC $CMN/ids-to-text.c -lm -o ids-to-text 2> build-ids-to-text.log || fail ids-to-text build-ids-to-text.log
  gcc $F -DPIPELINE_ENTRY -O2 $TOK $INC -Dmain=words_to_embeddings_main -c $CMN/words-to-embeddings.c -o words-to-embeddings.o 2> build-words-to-embeddings.log || fail words-to-embeddings build-words-to-embeddings.log
  gcc $F -O2 -c $CMN/whole-model.c -o whole-model.o 2> build-whole-model.log || fail whole-model build-whole-model.log
  gcc $F -O3 $INC -c $CPU/dspark-draft.c -o dspark-draft.o 2> build-dspark-draft.log || fail dspark-draft build-dspark-draft.log
  gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared $INC $CMN/final-stage.c -lm -o $BUILD_GPU/pipeline-stage.so 2> build-tail-library.log || fail tail-library build-tail-library.log
  nvcc -O3 --fmad=false -arch=sm_86 -std=c++17 -DEXPERT_V2 -DINT8_PLANES -I$GPU -I$CMN $COPYQ_INC $SLOTCACHE_INC $ROUTETABLE_INC -c $GPU/backend-gpu.cu -o backend-gpu.o 2> build-backend-gpu.log || fail backend-gpu build-backend-gpu.log
  objcopy --keep-global-symbol=stage1_setup --keep-global-symbol=stage1_ids --keep-global-symbol=stage1_ids_plain --keep-global-symbol=stage1_id_of --keep-global-symbol=stage1_lookup words-to-embeddings.o
  objcopy --keep-global-symbol=dspark_open --keep-global-symbol=dspark_draft --keep-global-symbol=dspark_reset dspark-draft.o
  nvcc -arch=sm_86 whole-model.o words-to-embeddings.o dspark-draft.o backend-gpu.o -lcublas -Xcompiler -fopenmp -ldl -lpthread -lm -o whole-model-gpu 2> build-link.log || { head -30 build-link.log; exit 1; }
  echo "built; warnings: $(cat build-*.log | grep -c warning)"
fi
[ "${BUILD_ONLY:-0}" = "1" ] && exit 0
cd "$RESULTS_DIR"
if [ "${1:-}" != "text" ]; then echo "usage: run-model-gpu.sh text FILE"; exit 2; fi
WORDS=$2; REFS="-"
ulimit -l unlimited; ulimit -c 0
echo "PROMPT ($(wc -w < $WORDS) words, $(wc -c < $WORDS) bytes):"; echo "-----"; cat $WORDS; echo; echo "-----"
env -i PATH="$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64" SPECULATIVE_DRAFT_TOKENS="${SPECULATIVE_DRAFT_TOKENS:-}" SPECULATIVE_DRAFTER="${SPECULATIVE_DRAFTER:-}" SPECULATIVE_FIXED_WIDTH="${SPECULATIVE_FIXED_WIDTH:-}" \
    PF_SESSION_DIR="${PF_SESSION_DIR:-}" PF_CONTEXT="${PF_CONTEXT:-}" PF_CHUNK="${PF_CHUNK:-}" PF_REPLY_RESERVE="${PF_REPLY_RESERVE:-}" PF_INPUT_MAX="${PF_INPUT_MAX:-}" PF_SUMMARY_TOKENS="${PF_SUMMARY_TOKENS:-}" \
    DSPARK_CHECKPOINT="$DSPARK_DIR/model.safetensors" PF_VERBOSE="${PF_VERBOSE:-}" PF_DECODE="${PF_DECODE:-}" PF_EXTRA="${PF_EXTRA:-}" PF_SAVE_ROWS="${PF_SAVE_ROWS:-}" PF_SAVE_IDS="${PF_SAVE_IDS:-}" \
    PF_ROUTES="${PF_ROUTES:-}" PF_CHAT="${PF_CHAT:-}" PF_SYSTEM="${PF_SYSTEM:-}" PF_LAST_LAYER="${PF_LAST_LAYER:-}" MODEL_GPU_ROUTES="${MODEL_GPU_ROUTES:-}" MODEL_GPU_PLANES_MB="${MODEL_GPU_PLANES_MB:-}" \
    K3GPU_CARDS="${K3GPU_CARDS:-}" K3GPU_HOST_GB="${K3GPU_HOST_GB:-}" K3GPU_RESERVE_MB="${K3GPU_RESERVE_MB:-}" K3GPU_READERS="${K3GPU_READERS:-}" K3GPU_VERBOSE="${K3GPU_VERBOSE:-}" K3GPU_POLICY="${K3GPU_POLICY:-}" K3GPU_CHUNKS="${K3GPU_CHUNKS:-}" K3GPU_ROUTECACHE="${K3GPU_ROUTECACHE:-}" K3GPU_LOOKAHEAD="${K3GPU_LOOKAHEAD:-}" K3GPU_TRACE="${K3GPU_TRACE:-}" K3GPU_VQUEUE="${K3GPU_VQUEUE:-}" K3GPU_VQ_INFLIGHT="${K3GPU_VQ_INFLIGHT:-}" K3GPU_VQ_FROM="${K3GPU_VQ_FROM:-}" K3GPU_PREFILL="${K3GPU_PREFILL:-}" K3GPU_LAYOUT="${K3GPU_LAYOUT:-}" \
    K3GPU_COPYQ="${K3GPU_COPYQ:-}" K3GPU_COPYQ_LIB="${K3GPU_COPYQ_LIB:-}" K3GPU_SLOTCACHE="${K3GPU_SLOTCACHE:-}" K3GPU_SLOTCACHE_LIB="${K3GPU_SLOTCACHE_LIB:-}" K3GPU_ROUTETABLE="${K3GPU_ROUTETABLE:-}" K3GPU_ROUTETABLE_LIB="${K3GPU_ROUTETABLE_LIB:-}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}" OMP_PROC_BIND=close OMP_PLACES=cores \
    "$BUILD_GPU/whole-model-gpu" "$BUILD_GPU" "$DATASET_DIR" "$OPEN_WEIGHTS_DIR" $WORDS $REFS
