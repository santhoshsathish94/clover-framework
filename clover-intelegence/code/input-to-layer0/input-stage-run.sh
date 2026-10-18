#!/bin/bash
# usage: input-stage-run.sh   (needs tok-1000.txt, l1in-1000.s0.f32, text.txt, prompts.tsv in /tmp/one-layer)
set -u
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
cd $H
mkdir -p tokfiles
ln -sf /opt/clover-k3/clover-intelegence/dataset/tiktoken.model tokfiles/tiktoken.model
ln -sf $D/server/bin/configs/tokenizer_config.json tokfiles/tokenizer_config.json
gcc -std=gnu11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
    -I/root/kimi-k3-in-c/src/tokenizer -I/root/kimi-k3-in-c/third_party -I$D/server \
    input-stage.c -lm -o input-stage 2> build-is.log || { grep -E "error" build-is.log | head -20; exit 1; }
ulimit -c 0
./input-stage tokfiles $D/server/bin/dataset/inputs/seed.bin prompts.tsv text.txt l1in-1000.s0.f32 tok-1000.txt real-ids.txt
