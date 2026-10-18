#!/bin/bash
# usage: pipeline-run.sh recorded        words = the text the recorded 1000 ids stand for; every stage is compared with the recorded values
#        pipeline-run.sh words COUNT     words = the first COUNT words of text.txt; no recorded values exist, nothing is compared
# needs in /tmp/one-layer: input-stage.c layer0-batch.c layer-batch.c pipeline.c ids-to-words.c (host copies), tokfiles/, text.txt,
#   tok-1000.txt, l1in-1000.s0.f32, l1in-1000.resid.f32, l1ref-1000.f32 (made with LB_SAVE_REF=/tmp/one-layer/l1ref-1000.f32 batch-run.sh 1000 1000)
set -u
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
F="-std=gnu11 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function -DPIPELINE_ENTRY"
TOK="-I/root/kimi-k3-in-c/src/tokenizer -I/root/kimi-k3-in-c/third_party"
gcc -std=gnu11 -O2 $TOK ids-to-words.c -lm -o ids-to-words 2> build-pw.log || { grep -E "error" build-pw.log | head -20; exit 1; }
gcc $F -O2 $TOK -I$D/server -Dmain=input_stage_main -c input-stage.c -o p-input.o 2> build-p1.log || { grep -E "error" build-p1.log | head -20; exit 1; }
gcc $F -O2 -funroll-loops -I$D/server -Dmain=layer0_batch_main -c layer0-batch.c -o p-layer0.o 2> build-p2.log || { grep -E "error" build-p2.log | head -20; exit 1; }
# numeric-table.h defines these six functions (not static) and both stage 1 and stage 2 include it: keep stage 1's copies private
objcopy -L server_input_open -L server_input_close -L server_input -L server_output_open -L server_output_close -L server_output_project p-input.o
gcc $F -O3 -I$D/common -I$D/tansformers/transformer-1 -Dmain=layer_batch_main -c layer-batch.c -o p-layer1.o 2> build-p3.log || { grep -E "error" build-p3.log | head -20; exit 1; }
gcc $F -O2 -c pipeline.c -o p-main.o 2> build-p4.log || { grep -E "error" build-p4.log | head -20; exit 1; }
gcc -fopenmp p-main.o p-input.o p-layer0.o p-layer1.o -lm -lpthread -o pipeline 2> build-p5.log || { head -30 build-p5.log; exit 1; }
REFS="l1in-1000.s0.f32 l1in-1000.resid.f32 l1ref-1000.f32"
if [ "${1:-recorded}" = "words" ]; then
  python3 -c "
import re,sys
t=open('text.txt',encoding='utf-8').read()
m=list(re.finditer(r'\S+',t))[int(sys.argv[1])-1]
open('words-run.txt','w',encoding='utf-8').write(t[:m.end()])" "${2:-1000}"
  WORDS=words-run.txt; REFS="- - -"
else
  ./ids-to-words tokfiles tok-1000.txt words-1000.txt; WORDS=words-1000.txt
fi
ulimit -l unlimited; ulimit -c 0
env -i PATH="$PATH" LB_IO="${LB_IO:-4}" LB_SPEC="${LB_SPEC:-0}" LB_THP="${LB_THP:-1}" LB_EXPK="${LB_EXPK:-1}" OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./pipeline "$D" tokfiles $WORDS $REFS
