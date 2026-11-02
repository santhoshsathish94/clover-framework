#!/usr/bin/env bash
# T3: the Learning Concepts method on Clover's C1, LFM3-Omni-2B S3 (backbone; HybridLink adapter untouched), through
# harness/lfm3_lc.py (released scripts, explicit adaptations listed there). Own data/checkpoint/results roots so the
# released split scripts never touch the Falcon files. GPU phases wait for cards with enough free memory.
# Usage: lfm3_run.sh [phase ...]  phases: words synonyms split train_all sts
set -uo pipefail
W=/workspace/meaning-core
L=$W/runs/lfm3
export HF_HOME=$W/hf CONCEPT_DATA_ROOT=$W/lc-data-lfm3 CONCEPT_RESULTS_ROOT=$W/lfm3-results
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODEL=$W/models/lfm3-omni2b-s3-backbone
SEEDS=(42 43 44)
CW="--max-content-words all"   # released default None crashes for weights > 0 (see lc_run.sh)
mkdir -p $L $CONCEPT_DATA_ROOT $CONCEPT_RESULTS_ROOT
source /venv/main/bin/activate
cd $W/learning-concepts
WR=$W/lfm3_lc.py
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

free_card() {  # prints a card with >= $1 MiB free and fewer than $2 of my processes; waits otherwise.
  # While the Falcon pipeline (lc_run.sh) runs, it trains on cards 0-3 and then evaluates on cards 0-5 (one ~5 GB
  # eval per card) without checking memory: cards 0-3 are reserved, cards 4-5 take at most 2 of my processes
  # (~10 GB) so its eval still fits; any card holding another process needs 9 GB more free than asked.
  local need=$1 maxp=$2
  while true; do
    local reserved="" capped=""
    if pgrep -f "lc_run[.]sh" > /dev/null; then reserved=" 0 1 2 3 "; capped=" 4 5 "; fi
    while IFS=, read -r idx free; do
      idx=${idx// /}; free=${free// /}
      [[ "$reserved" == *" $idx "* ]] && continue
      local mine=0 other=0 cap=$maxp
      [[ "$capped" == *" $idx "* ]] && [ $cap -gt 2 ] && cap=2
      for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$idx" 2>/dev/null); do
        if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q lfm3_lc.py; then mine=$((mine + 1)); else other=$((other + 1)); fi
      done
      local req=$need; [ $other -gt 0 ] && req=$((need + 9000))
      if [ "$free" -ge "$req" ] && [ "$mine" -lt "$cap" ]; then echo "$idx"; return; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    sleep 30
  done
}

words() {
  stamp "words: start"
  python $WR get_content_words --model $MODEL --dataset c4 > $L/words.log 2>&1
  local rc=$?
  stamp "words: exit $rc $(tail -4 $L/words.log | tr '\n' ' ')"
  [ $rc -eq 0 ]
}

synonyms() {
  local N=32 pids=() fail=0
  stamp "synonyms: start ($N shards; each waits for a card with 7 GB free)"
  for i in $(seq 0 $((N - 1))); do
    s=$((i * 10000 / N)); e=$(((i + 1) * 10000 / N))
    g=$(free_card 7000 5)
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 python $WR embedding_synonyms c4 --start $s --end $e --model $MODEL \
      > $L/syn_$i.log 2>&1 &
    pids+=($!)
    stamp "synonyms: shard $i [$s,$e) on card $g"
    sleep 40   # let the process allocate before the next free-memory check
  done
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  stamp "synonyms: done, failed shards: $fail"
  [ $fail -eq 0 ]
}

split() {
  stamp "split: start"
  python data/merge_synonym_parts.py --force > $L/split.log 2>&1
  python data/randomize_synonyms.py --split train test >> $L/split.log 2>&1
  local rc=$?
  stamp "split: exit $rc $(grep -E 'Wrote|wrote' $L/split.log | tr '\n' ' ')"
  [ $rc -eq 0 ]
}

train_preflight() {  # 16 samples, 1 epoch, both loss settings + one STS task, separate roots: before the 12 runs
  stamp "train_preflight: start"
  local P=$W/lfm3-preflight fail=0 pids=()
  rm -rf $P
  for A in "--concept-loss-weight 0.0" "--concept-loss-weight 1.0"; do
    g=$(free_card 16000 1)
    tag="preflight_$(echo $A | tr -d '-' | tr ' ' '_')"
    CUDA_VISIBLE_DEVICES=$g LC_SEED=42 CONCEPT_CHECKPOINT_ROOT=$P OMP_NUM_THREADS=4 python $WR train --model-name $MODEL \
      --dataset c4 --dataset-type embedding $A $CW --max-train-samples 16 --num-train-epochs 1 > $L/$tag.log 2>&1 &
    pids+=($!)
    sleep 60
  done
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  g=$(free_card 9000 3)
  CUDA_VISIBLE_DEVICES=$g CONCEPT_CHECKPOINT_ROOT=$P python $WR eval_mteb --base-model $MODEL --dataset c4 \
    --dataset-type embedding --tasks BIOSSES --no-quantize --only-adapter-weights 1.0 --max-train-samples 16 \
    --csv-output $P/biosses.csv --mteb-output-root $P/mteb > $L/preflight_eval.log 2>&1 || fail=$((fail + 1))
  stamp "train_preflight: failed $fail; adapters $(find $P -name adapter_model.safetensors | wc -l); $(tail -1 $P/biosses.csv 2>/dev/null)"
  [ $fail -eq 0 ]
}

train_all() {
  stamp "train: start"
  local jobs=() pids=() k=0
  for S in "${SEEDS[@]}"; do
    jobs+=("$S|--concept-loss-weight 0.0" "$S|--concept-loss-weight 1.0"
           "$S|--concept-loss-weight 0.25 --randomized-synonyms" "$S|--concept-loss-weight 1.0 --randomized-synonyms")
  done
  for j in "${jobs[@]}"; do
    S=${j%%|*}; A=${j#*|}
    g=$(free_card 16000 1)
    tag="seed${S}_$(echo $A | tr -d '-' | tr ' ' '_')"
    CUDA_VISIBLE_DEVICES=$g LC_SEED=$S CONCEPT_CHECKPOINT_ROOT=$W/lfm3-ckpt/seed$S OMP_NUM_THREADS=4 \
      python $WR train --model-name $MODEL --dataset c4 --dataset-type embedding $A $CW > $L/train_$tag.log 2>&1 &
    pids+=($!)
    stamp "train: launched $tag on card $g"
    sleep 90
  done
  wait
  local ok=0 bad=""
  for f in $L/train_seed*.log; do
    if grep -q "Model saved to" $f; then ok=$((ok + 1)); else bad="$bad $(basename $f .log)"; fi
  done
  stamp "train: done, saved: $ok of ${#jobs[@]}; not saved:${bad:- none}"
  [ $ok -eq ${#jobs[@]} ]
}

sts() {
  stamp "sts: start"
  local pids=() fail=0
  for S in "${SEEDS[@]}"; do
    for R in "" "--randomized-synonyms"; do
      if [ -z "$R" ]; then ws="base,0.0,1.0"; t=sts_seed$S; else ws="0.25,1.0"; t=sts_rand_seed$S; fi
      [ "$S" != "42" ] && [ -z "$R" ] && ws="0.0,1.0"   # base is seed-independent: once
      g=$(free_card 9000 3)
      CUDA_VISIBLE_DEVICES=$g CONCEPT_CHECKPOINT_ROOT=$W/lfm3-ckpt/seed$S python $WR eval_mteb --base-model $MODEL \
        --dataset c4 --dataset-type embedding --tasks sts --no-quantize $R --only-adapter-weights $ws \
        --csv-output $CONCEPT_RESULTS_ROOT/$t.csv --mteb-output-root $CONCEPT_RESULTS_ROOT/mteb_$t > $L/$t.log 2>&1 &
      pids+=($!)
      stamp "sts: $t ($ws) on card $g"
      sleep 40
    done
  done
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  stamp "sts: done, failed evals: $fail; csv: $(ls $CONCEPT_RESULTS_ROOT/*.csv 2>/dev/null | wc -l)"
  [ $fail -eq 0 ]
}

smoke() {  # 3 sequences through words-equivalent data + synonyms, on the smallest free card, before the real phases
  stamp "smoke: start"
  local S=$W/lfm3-smoke; rm -rf $S; mkdir -p $S/c4/lfm3-omni2b-s3-backbone
  head -3 $CONCEPT_DATA_ROOT/c4/lfm3-omni2b-s3-backbone/combined.jsonl > $S/c4/lfm3-omni2b-s3-backbone/combined.jsonl
  g=$(free_card 7000 6)
  CONCEPT_DATA_ROOT=$S CUDA_VISIBLE_DEVICES=$g python $WR embedding_synonyms c4 --start 0 --end 3 --model $MODEL \
    > $L/smoke.log 2>&1
  local rc=$?
  stamp "smoke: exit $rc on card $g; $(python -c "import json,sys; r=[json.loads(l) for l in open('$S/c4/lfm3-omni2b-s3-backbone/embedding/synonyms_0_3.jsonl')]; c=[t for x in r for t in x['content_word_responses']]; print(len(r),'rows',len(c),'targets',sum(1 for t in c if t['synonyms']),'with synonyms; e.g.', [(t['word'], t['synonyms'][:5]) for t in c[:6]])" 2>&1 | tail -1)"
  [ $rc -eq 0 ]
}

phases=("$@"); [ ${#phases[@]} -eq 0 ] && phases=(words smoke synonyms split train_all sts)
for ph in "${phases[@]}"; do
  $ph || { stamp "phase $ph failed: stopping"; exit 1; }
done
stamp "all requested phases finished: ${phases[*]}"
