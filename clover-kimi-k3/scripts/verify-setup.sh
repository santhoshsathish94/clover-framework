#!/bin/bash
# Verifies a set-up engine by running it: build and wiring in place, the dataset complete, recorded prompts give the recorded token ids, speculative decoding
# (DSpark) gives the same tokens as plain decoding, the recorded 1000-token prompt matches the pod's own chain bit for bit (when its reference values are
# there), and a chat remembers what was said in an earlier turn. One model run at a time. Result: work/results/verify-setup.result, exit status 0 = all passed.
# usage: [VERIFY_PROMPTS="1 4 10"] [VERIFY_CHAT=0] [VERIFY_1000=0] [VERIFY_SPECULATIVE=0] verify-setup.sh
set -u
. "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
OUT="$RESULTS_DIR/verify-setup.result"
VERIFY_PROMPTS="${VERIFY_PROMPTS:-1 4 10}"
FAILED=0
say()  { printf '%s\n' "$*" | tee -a "$OUT"; }
pass() { say "PASS  $*"; }
warn() { say "WARN  $*"; }
fail() { say "FAIL  $*"; FAILED=1; }
: > "$OUT"; say "== verify-setup $(date) engine $ENGINE_DIR dataset $DATASET_DIR"
cd "$RESULTS_DIR"

# 1. built and wired
[ -x "$BUILD_DIR/whole-model" ] && [ -x "$BUILD_DIR/ids-to-text" ] && [ -f "$POD_DIR/tansformers/transformer-93/bin/pipeline-stage.so" ] && pass "program and tail library are built" || fail "not built: run scripts/run-model.sh with BUILD_ONLY=1"
broken=$(find "$POD_DIR/server/bin" "$POD_DIR/tansformers/transformer-1/bin" -xtype l 2>/dev/null | wc -l)
[ -d "$POD_DIR/server/bin/dataset" ] && [ "$broken" = 0 ] && pass "dataset links are in place, none broken" || fail "dataset links missing or broken ($broken broken): run scripts/wire-dataset.sh"
[ $FAILED = 0 ] || { say "stopping: the engine is not built and wired"; exit 1; }

# 2. dataset complete
"$SCRIPTS_DIR/check-dataset-files.sh" 2>&1 | tee -a "$OUT"
[ "${PIPESTATUS[0]}" = 0 ] || { FAILED=1; say "stopping: the dataset is incomplete"; exit 1; }

# 3. recorded prompts, exact path, against the ids in prompts/expected-exact-ids.tsv
for n in $VERIFY_PROMPTS; do
  awk -F'\t' -v n="$n" '$1==n{printf "%s", $4}' "$ENGINE_DIR/prompts/recorded-34-prompts.tsv" > "verify-prompt-$n.txt"
  want=$(awk -F'\t' -v n="$n" '!/^#/ && $1==n{print $2}' "$ENGINE_DIR/prompts/expected-exact-ids.tsv")
  start=$(date +%s)
  PF_DECODE=11 PF_CACHE=16 PF_SAVE_IDS="verify-$n.ids" BUILD=0 "$SCRIPTS_DIR/run-model.sh" text "verify-prompt-$n.txt" > "verify-$n.log" 2>&1
  got=$(tr -d '\n\r ' < "verify-$n.ids" 2>/dev/null)
  text=$("$BUILD_DIR/ids-to-text" "$OPEN_WEIGHTS_DIR" "verify-$n.ids" "verify-$n.out" > /dev/null 2>&1; tr '\n' '~' < "verify-$n.out" 2>/dev/null)
  if [ -n "$want" ] && [ "$got" = "$want" ]; then pass "prompt $n: the 12 token ids equal the recorded ones ($(( $(date +%s) - start )) s): $(head -c 70 <<< "$text")"
  else fail "prompt $n: ids differ. wanted $want, got ${got:-nothing} (see $RESULTS_DIR/verify-$n.log)"; fi
done

# 4. speculative decoding equals plain decoding
if [ "${VERIFY_SPECULATIVE:-1}" = 1 ]; then
  if [ -f "$DSPARK_DIR/model.safetensors" ]; then
    PF_DECODE=16 PF_CACHE=16 PF_SAVE_IDS=verify-plain.ids BUILD=0 "$SCRIPTS_DIR/run-model.sh" text "$ENGINE_DIR/prompts/code-is-prime.txt" > verify-plain.log 2>&1
    SPECULATIVE_DRAFTER=dspark SPECULATIVE_DRAFT_TOKENS=8 PF_DECODE=16 PF_CACHE=16 PF_SAVE_IDS=verify-dspark.ids BUILD=0 "$SCRIPTS_DIR/run-model.sh" text "$ENGINE_DIR/prompts/code-is-prime.txt" > verify-dspark.log 2>&1
    if [ -s verify-plain.ids ] && cmp -s verify-plain.ids verify-dspark.ids; then pass "DSpark speculative decoding gives the same 17 tokens as plain decoding: $(grep -h 'SPECULATIVE DECODING' verify-dspark.log | sed 's/^ *//' | cut -c1-150)"
    else fail "DSpark speculative decoding differs from plain decoding (see verify-plain.log, verify-dspark.log)"; fi
  else warn "no DSpark checkpoint at $DSPARK_DIR: speculative decoding not checked"; fi
fi

# 5. the recorded 1000-token prompt against the pod's own chain
if [ "${VERIFY_1000:-1}" = 1 ]; then
  if [ -f "$INPUTS_DIR/refs/final.snaps.f32" ] && [ -f "$INPUTS_DIR/tok-1000.txt" ]; then
    BUILD=0 "$SCRIPTS_DIR/run-model.sh" recorded > verify-1000.log 2>&1
    line=$(grep -h 'against the pod' verify-1000.log | sed 's/^ *//')
    case "$line" in *"embeddings 1000 of 1000 rows, layer 92 output 1000 of 1000 rows, 8 snapshots 8000 of 8000"*) pass "1000-token prompt: bit-identical to the pod's chain ($(grep -h 'REQUEST' verify-1000.log | grep -o '[0-9.]* s in total'))";;
      *) fail "1000-token prompt differs from the pod's chain: ${line:-no comparison line} (see verify-1000.log)";; esac
  else warn "no recorded reference values (work/inputs/refs): the 1000-token comparison with the pod's chain was skipped"; fi
fi

# 6. a chat that remembers
if [ "${VERIFY_CHAT:-1}" = 1 ]; then
  if [ -f "$DSPARK_DIR/model.safetensors" ]; then
    export PF_CONTEXT=1024 PF_CHUNK=256 PF_REPLY_RESERVE=256 PF_DECODE=40
    if ! "$SCRIPTS_DIR/chat-start.sh" verify "Answer briefly." > chat-start-verify.log 2>&1; then fail "chat: could not start a session: $(cat chat-start-verify.log)"; fi
    for _ in $(seq 1 240); do [ -f "session-verify/ready" ] && break; sleep 5; done
    if [ -f session-verify/ready ]; then
      WAIT=900 "$SCRIPTS_DIR/chat-say.sh" verify "My name is Santhosh and I am building a C inference engine. Answer in one short sentence." > /dev/null
      reply=$(WAIT=900 "$SCRIPTS_DIR/chat-say.sh" verify "What is my name?")
      if grep -qi santhosh <<< "$reply"; then pass "chat: the second turn remembered the name ($(tail -1 session-verify/status.txt | cut -c1-140))"; else fail "chat: the second turn did not remember the name: $(head -c 200 <<< "$reply")"; fi
    else fail "chat: the session did not become ready (see session-verify/engine.log)"; fi
    "$SCRIPTS_DIR/chat-stop.sh" verify > /dev/null
  else warn "no DSpark checkpoint: the chat check was skipped (a chat session needs it)"; fi
fi

say "== $(date): $([ $FAILED = 0 ] && echo 'all checks passed' || echo 'SOME CHECKS FAILED')"
exit $FAILED
