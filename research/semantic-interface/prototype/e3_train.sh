#!/usr/bin/env bash
# Experiment 2 (E3), k3: a UD pipeline learned from UD English-EWT r2.18 train (tagger + morphologizer + trainable
# lemmatizer + parser; spaCy CPU "efficiency" config), seeds 1-3 in parallel, dev for early stopping.
# Usage: bash e3_train.sh <venv python> <UD_English-EWT dir> <work dir>
set -e
P=$1; D=$2; W=$3
S=$W/ewt-spacy
mkdir -p $S
for split in train dev test; do
  [ -f $S/en_ewt-ud-$split.spacy ] || $P -m spacy convert $D/en_ewt-ud-$split.conllu $S --converter conllu --n-sents 10 > $S/convert-$split.log
done
ls -la $S
$P -m spacy init config $W/ewt.cfg --lang en --pipeline tagger,morphologizer,trainable_lemmatizer,parser \
  --optimize efficiency --force || $P -m spacy init config $W/ewt.cfg --lang en --pipeline tagger,morphologizer,parser \
  --optimize efficiency --force
for seed in 1 2 3; do
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 ionice -c3 \
    $P -m spacy train $W/ewt.cfg --output $W/ewt-seed$seed --paths.train $S/en_ewt-ud-train.spacy \
    --paths.dev $S/en_ewt-ud-dev.spacy --system.seed $seed > $W/ewt-seed$seed.log 2>&1 &
done
wait
grep -h -E "Saved pipeline|Error" $W/ewt-seed*.log
