#!/usr/bin/env bash
# Learning Concepts, Not Tokens (arXiv 2603.29123), reproduced with the released code on Falcon3-1B-Base, C4,
# embedding synonyms. Phases: words -> synonyms (16 shards on 8 cards) -> split -> train (4 conditions x 3 seeds)
# -> STS. Run detached; each phase logs under $W/runs/lc/. Usage: lc_run.sh [phase ...] (default: all)
set -uo pipefail
W=/workspace/meaning-core
L=$W/runs/lc
export HF_HOME=$W/hf CONCEPT_DATA_ROOT=$W/lc-data CONCEPT_RESULTS_ROOT=$W/lc-results
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
MODEL=tiiuae/Falcon3-1B-Base
SEEDS=(42 43 44)
mkdir -p $L $CONCEPT_DATA_ROOT $CONCEPT_RESULTS_ROOT
source /venv/main/bin/activate
cd $W/learning-concepts
stamp() { echo "[$(date -u +%FT%TZ)] $*"; }

words() {
  stamp "words: start"
  python data/get_content_words.py --model $MODEL --dataset c4 > $L/words.log 2>&1
  stamp "words: exit $? $(tail -2 $L/words.log | tr '\n' ' ')"
}

synonyms() {
  # CPU-bound (spaCy parses each candidate substitution); 4 processes per card (~5 GB each)
  local N=32
  stamp "synonyms: start ($N shards, 4 per card)"
  local pids=()
  for i in $(seq 0 $((N - 1))); do
    s=$((i * 10000 / N)); e=$(((i + 1) * 10000 / N))
    # written for transformers 4.x (DynamicCache.from_legacy_cache): run it on the 4.57.6 overlay (tf457/)
    PYTHONPATH=$W/tf457 CUDA_VISIBLE_DEVICES=$((i % 8)) OMP_NUM_THREADS=4 \
      python data/embedding_synonyms.py c4 --start $s --end $e --model $MODEL > $L/syn_$i.log 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  stamp "synonyms: done, failed shards: $fail"
}

split() {
  stamp "split: start"
  python data/merge_synonym_parts.py --force > $L/split.log 2>&1
  python data/randomize_synonyms.py --split train test >> $L/split.log 2>&1
  stamp "split: exit $? $(grep -E 'Wrote|wrote' $L/split.log | tr '\n' ' ')"
}

# The released train.py/eval code targets transformers 4.46-4.57 (eval_strategy + processing_class need >=4.46;
# warmup_ratio and DynamicCache.from_legacy_cache were removed in 5.x), so training and evaluation run on a complete
# 4.x overlay (tf457x: transformers 4.57.6, huggingface_hub 0.36.0, tokenizers 0.22.1, datasets 4.8.5,
# sentence-transformers 5.7.0, fsspec 2026.2.0; torch/peft/accelerate/bitsandbytes/mteb from /venv/main).
# The synonyms phase ran on the smaller tf457 overlay (transformers/tokenizers/hub only), which does not import datasets.
# No released code is edited.
TF4="PYTHONPATH=$W/tf457x"
# Released train.py crashes for every concept-loss weight > 0 with the default max_content_words=None
# (_build_concept_targets: `!= 'all'` is true and no branch sets k -> UnboundLocalError). Its own flag value "all" is
# the documented "all content words" behaviour and keeps the same checkpoint path (all_content_words).
CW="--max-content-words all"

preflight() {
  # 16 samples, 1 epoch, both loss settings, separate checkpoint root: proves the stack runs before the real jobs
  stamp "preflight: start"
  local fail=0 g=5 pids=()
  for A in "--concept-loss-weight 0.0" "--concept-loss-weight 1.0"; do
    tag="preflight_$(echo $A | tr -d '-' | tr ' ' '_')"
    env $TF4 CUDA_VISIBLE_DEVICES=$g LC_SEED=42 CONCEPT_CHECKPOINT_ROOT=$W/lc-preflight OMP_NUM_THREADS=4 \
      python $W/run_train_seed.py --model-name $MODEL --dataset c4 --dataset-type embedding $A $CW \
      --max-train-samples 16 --num-train-epochs 1 > $L/$tag.log 2>&1 &
    pids+=($!); g=$((g + 1))
  done
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  stamp "preflight: failed runs: $fail; adapters: $(find $W/lc-preflight -name adapter_model.safetensors | wc -l)"
  env $TF4 CUDA_VISIBLE_DEVICES=5 CONCEPT_CHECKPOINT_ROOT=$W/lc-preflight python eval/eval_mteb.py --base-model $MODEL \
    --dataset c4 --dataset-type embedding --tasks BIOSSES --only-adapter-weights 1.0 --max-train-samples 16 \
    --csv-output $W/lc-preflight/biosses.csv --mteb-output-root $W/lc-preflight/mteb > $L/preflight_eval.log 2>&1 \
    || fail=$((fail + 1))
  stamp "preflight: eval exit fail=$fail; $(tail -3 $W/lc-preflight/biosses.csv 2>/dev/null | tr '\n' ' ')"
  [ $fail -eq 0 ]
}

train_all() {
  stamp "train: start"
  local jobs=() gpu_pid=() tags=()
  for S in "${SEEDS[@]}"; do
    jobs+=("$S|--concept-loss-weight 0.0" "$S|--concept-loss-weight 1.0"
           "$S|--concept-loss-weight 0.25 --randomized-synonyms" "$S|--concept-loss-weight 1.0 --randomized-synonyms")
  done
  local k=0
  for j in "${jobs[@]}"; do
    S=${j%%|*}; A=${j#*|}; g=$((k % 8))
    if [ -n "${gpu_pid[$g]:-}" ]; then wait ${gpu_pid[$g]}; fi
    tag="seed${S}_$(echo $A | tr -d '-' | tr ' ' '_')"
    env $TF4 CUDA_VISIBLE_DEVICES=$g LC_SEED=$S CONCEPT_CHECKPOINT_ROOT=$W/lc-ckpt/seed$S OMP_NUM_THREADS=4 \
      python $W/run_train_seed.py --model-name $MODEL --dataset c4 --dataset-type embedding $A $CW > $L/train_$tag.log 2>&1 &
    gpu_pid[$g]=$!
    stamp "train: launched $tag on card $g"
    k=$((k + 1))
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
  local g=0 pids=()
  for S in "${SEEDS[@]}"; do
    env $TF4 CUDA_VISIBLE_DEVICES=$g CONCEPT_CHECKPOINT_ROOT=$W/lc-ckpt/seed$S python eval/eval_mteb.py --base-model $MODEL \
      --dataset c4 --dataset-type embedding --tasks sts --only-adapter-weights base,0.0,1.0 \
      --csv-output $CONCEPT_RESULTS_ROOT/sts_seed$S.csv --mteb-output-root $CONCEPT_RESULTS_ROOT/mteb_seed$S \
      > $L/sts_seed$S.log 2>&1 &
    pids+=($!); g=$((g + 1))
    env $TF4 CUDA_VISIBLE_DEVICES=$g CONCEPT_CHECKPOINT_ROOT=$W/lc-ckpt/seed$S python eval/eval_mteb.py --base-model $MODEL \
      --dataset c4 --dataset-type embedding --tasks sts --randomized-synonyms --only-adapter-weights 0.25,1.0 \
      --csv-output $CONCEPT_RESULTS_ROOT/sts_rand_seed$S.csv --mteb-output-root $CONCEPT_RESULTS_ROOT/mteb_rand_seed$S \
      > $L/sts_rand_seed$S.log 2>&1 &
    pids+=($!); g=$((g + 1))
  done
  local fail=0
  for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
  stamp "sts: done, failed evals: $fail; csv: $(ls $CONCEPT_RESULTS_ROOT/*.csv 2>/dev/null | wc -l)"
}

phases=("$@"); [ ${#phases[@]} -eq 0 ] && phases=(words synonyms split train_all sts)
for ph in "${phases[@]}"; do
  $ph || { stamp "phase $ph failed: stopping"; exit 1; }
done
stamp "all requested phases finished: ${phases[*]}"
