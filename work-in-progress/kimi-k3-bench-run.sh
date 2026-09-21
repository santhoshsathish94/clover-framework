#!/usr/bin/env bash
# Run the kimi-k3-in-c benchmark campaign on a rented Linux box, unattended.
#
# Why this exists: ROADMAP.md item 2 in that project asks for the published
# campaign to be re-run under the replicating harness, because almost every
# figure there is a single sample against a 33% noise floor. The harnesses are
# written; what is missing is someone with hardware. This is that, plus the
# guards needed so a rented machine cannot quietly cost money after the work
# is done.
#
#   HF_TOKEN=hf_...  ./kimi-k3-bench-run.sh
#
# Nothing here is Clover's work. The engine, the harnesses and every published
# measurement are Fareed Khan's; the weights are Moonshot AI's.
set -euo pipefail

# ---- settings -------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-$HOME/k3model}"      # 1.56 TB, can be the slower device
TRUNK_DIR="${TRUNK_DIR:-$HOME/k3trunk}"      # 109 GB, put this on the fastest device
OUT_DIR="${OUT_DIR:-$HOME/k3results}"
REPS="${REPS:-3}"                            # CONTRIBUTING.md: 3 minimum, and report all
SWEEP_GB="${SWEEP_GB:-}"                     # split-sweep budget; derived from RAM if unset
SHUTDOWN_HOURS="${SHUTDOWN_HOURS:-24}"       # runaway-compute guard, NOT a billing guard
REPO="${REPO:-$HOME/kimi-k3-in-c}"

# The workload arm is the reason for the machine. The campaign arm is the by-product
# that happens to be upstream ROADMAP item 2.
RUN_WORKLOAD="${RUN_WORKLOAD:-1}"
RUN_THREADS="${RUN_THREADS:-1}"              # ROADMAP item 3: never swept on this engine
RUN_CAMPAIGN="${RUN_CAMPAIGN:-1}"
GEN_WORK="${GEN_WORK:-64}"                   # beyond the 8-32 upstream measured; deliberate
GEN_THREAD="${GEN_THREAD:-16}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$OUT_DIR/run.log"; }
mkdir -p "$OUT_DIR"

# ---- 0. runaway guard ------------------------------------------------------
# On a rented DEDICATED server this halts the work, not the bill. Hetzner:
# "Costs are incurred from the time the product is made available until the
# contract is cancelled." Billing stops only when the server is cancelled in
# Robot. Powering the OS off leaves an idle machine still accruing hourly.
if command -v shutdown >/dev/null; then
  sudo shutdown -h "+$((SHUTDOWN_HOURS * 60))" "k3 benchmark runaway guard" || true
  log "auto-halt armed for ${SHUTDOWN_HOURS}h (sudo shutdown -c to cancel)"
  log "REMINDER: this does not stop billing. Cancel the server in Robot when done."
fi

# ---- 1. machine preflight, refuse early rather than after 1.56 TB ---------
log "=== preflight ==="
grep -q avx2 /proc/cpuinfo || { log "FATAL: no AVX2, the engine requires AVX2+FMA"; exit 1; }
grep -q fma  /proc/cpuinfo || { log "FATAL: no FMA"; exit 1; }

# MemTotal is KiB, so this is GiB. Upstream reports peak RSS in decimal GB, which is why
# a "~128 GB" preset is 119 GiB and does fit a 128 GiB box. Keep the units visible.
RAM_GIB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
AVAIL_GIB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
log "RAM: ${RAM_GIB} GiB total, ${AVAIL_GIB} GiB available"
[ "$RAM_GIB" -ge 8 ] || { log "FATAL: 8 GB is the floor"; exit 1; }

# Observation, not a prediction: the published 128 GB rung recorded 128.18 GB peak RSS
# (119 GiB) under a 128 GiB cap on a 228 GiB machine. This box has to fit the same rung
# plus the OS inside its whole RAM, so the rung may OOM. The harness records that as a
# legitimate result. Whether it does is for the run to show.
if [ "$RAM_GIB" -lt 160 ]; then
  log "NOTE: ${RAM_GIB} GiB total. The 128 GB rung needs ~119 GiB plus OS headroom, so it"
  log "      may record OOM here. That is a result, not a failure."
fi

# The sweep holds total memory fixed, so its budget must be one the machine can honour.
# 0.90 of MemAvailable rather than an invented constant: the engine itself refuses to
# start when its plan exceeds MemAvailable * 0.95, so this stays just inside its own rule.
if [ -z "$SWEEP_GB" ]; then
  if [ "$RAM_GIB" -ge 160 ]; then SWEEP_GB=128; else SWEEP_GB=$(( AVAIL_GIB * 90 / 100 )); fi
fi
log "split-sweep budget: ${SWEEP_GB} GB"

# The ladder imposes memory ceilings with systemd-run --user. On a fresh
# SSH-only VM that fails with no user session bus, and the harness exits rather
# than silently measuring nothing.
command -v systemd-run >/dev/null || { log "FATAL: systemd-run missing"; exit 1; }
if ! systemd-run --scope --user -q true 2>/dev/null; then
  log "systemd-run --user unavailable, enabling linger"
  sudo loginctl enable-linger "$USER"
  systemd-run --scope --user -q true 2>/dev/null || {
    log "FATAL: still no user session bus. Log in over SSH rather than running via cloud-init."
    exit 1
  }
fi
log "systemd-run --user: ok"

# Storage. Trunk and checkpoint may be separate devices; TUNING.md documents
# that layout and it inverts the trunk-first rule, so record which is which.
mkdir -p "$MODEL_DIR" "$TRUNK_DIR"
MODEL_FREE=$(df -PBG "$MODEL_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
TRUNK_FREE=$(df -PBG "$TRUNK_DIR" | awk 'NR==2{gsub("G","",$4); print $4}')
MODEL_DEV=$(df -P "$MODEL_DIR" | awk 'NR==2{print $1}')
TRUNK_DEV=$(df -P "$TRUNK_DIR" | awk 'NR==2{print $1}')

# TUNING.md reports that on split storage the expert device becomes the bottleneck and
# the trunk-first rule inverts. That was measured by a user with trunk on NVMe and the
# checkpoint on a slower SATA drive; two identical NVMe may behave differently. Record
# which layout this is rather than assuming the finding transfers.
if [ "$MODEL_DEV" = "$TRUNK_DEV" ]; then
  log "storage: SINGLE device ($MODEL_DEV), trunk-first allocation applies"
  [ "$MODEL_FREE" -ge 1680 ] || { log "FATAL: one device needs 1680 GB for both, has ${MODEL_FREE}"; exit 1; }
else
  log "storage: SPLIT, trunk on $TRUNK_DEV, checkpoint on $MODEL_DEV"
  [ "$MODEL_FREE" -ge 1560 ] || { log "FATAL: checkpoint device has only ${MODEL_FREE} GB"; exit 1; }
  [ "$TRUNK_FREE" -ge 110 ]  || { log "FATAL: trunk device has only ${TRUNK_FREE} GB"; exit 1; }
fi

# ---- 2. quieten the machine ----------------------------------------------
# Each of these was observed contaminating a measurement on the upstream project.
log "=== quietening the machine ==="
sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true
sudo systemctl disable apt-daily.timer apt-daily-upgrade.timer 2>/dev/null || true

# A second engine competing for the same disk is one of the contaminants
# BENCHMARKING.md names. An `if` rather than `&& ... || true`, which reads as
# if-then-else and is not.
if pgrep -x k3 >/dev/null; then
  log "FATAL: a k3 process is already running"
  exit 1
fi

# ---- 3. build and prove the engine before downloading anything ------------
log "=== build ==="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git python3 pipx

# The downloader is the `hf` CLI from huggingface_hub 1.x, and it must be new enough to
# have `hf cache verify`, which is what turns the download into a verified artefact
# rather than a hopeful one. Ubuntu 24.04 marks the system interpreter
# externally-managed under PEP 668, so pipx is the supported route, not pip.
if ! command -v hf >/dev/null 2>&1; then
  pipx install huggingface_hub
  pipx ensurepath
  export PATH="$HOME/.local/bin:$PATH"
fi
hf cache verify --help >/dev/null 2>&1 || { log "FATAL: hf CLI too old, needs 'cache verify'"; exit 1; }
[ -d "$REPO" ] || git clone --depth 1 https://github.com/FareedKhan-dev/kimi-k3-in-c.git "$REPO"
cd "$REPO"
make -j >>"$OUT_DIR/build.log" 2>&1
log "built: $(./bin/k3 --version)"

log "=== weightless gates, before committing to a 1.56 TB download ==="
if make test >>"$OUT_DIR/gates.log" 2>&1; then
  log "gates passed"
else
  log "FATAL: gates failed, do not download 1.56 TB against a broken build"
  exit 1
fi

./scripts/k3-doctor.sh 2>&1 | tee "$OUT_DIR/doctor.txt" || true

# Kernel microbenchmarks need no weights at all. This is the cheapest available read on
# whether 16 cores hold up against the 124-core reference box, and it costs minutes
# rather than a 1.56 TB download.
#
# `make bench` compiles and immediately runs, and that first run is not trustworthy: on
# the AX102 it reported 3.7 GFLOP/s for MXFP4 where four later runs gave 137-142. It is
# kept in its own file rather than deleted, because a recurring outlier is itself data.
# The recorded runs are repeated for the same reason the token measurements are: bf16
# spread 86.1-102.0 GFLOP/s across four clean runs, so one sample is not a result.
log "=== kernel microbenchmarks, no weights required ==="
make bench 2>&1 | tee "$OUT_DIR/bench-kernels-warmup.txt" | tail -5
for r in $(seq 1 "$REPS"); do
  log "  bench_kernels rep $r/$REPS"
  { echo "--- rep $r ---"; ./bin/bench_kernels 2>&1; } | tee -a "$OUT_DIR/bench-kernels.txt"
done
tail -20 "$OUT_DIR/bench-kernels.txt"

# ---- 3b. measure the storage before trusting it ----------------------------
# I/O is 41-61% of wall clock across the published ladder, and the largest single source
# of its 33% run-to-run spread was the device rather than scheduling. Measuring it costs
# seconds now and is unrecoverable once 1.56 TB is already written.
# Repeated rather than sampled once: this number sets the storage ceiling that every
# s/token prediction is compared against, so its spread matters more than the kernels'.
log "=== storage bandwidth (O_DIRECT) ==="
for d in "$TRUNK_DIR" "$MODEL_DIR"; do
  probe="$d/.devbw.probe"
  fallocate -l 8G "$probe" 2>/dev/null || dd if=/dev/zero of="$probe" bs=1M count=8192 status=none
  log "--- $d ---"
  for r in $(seq 1 "$REPS"); do
    { echo "--- $d rep $r ---"; python3 tools/devbw.py "$probe" 2>&1; } \
      | tee -a "$OUT_DIR/devbw.txt" | sed 's/^/   /'
  done
  rm -f "$probe"
done

# ---- 4. checkpoint ---------------------------------------------------------
# Resumable: re-running continues rather than restarting. The script verifies
# shard count and exact byte totals, because a partial download does not fail
# loudly, it produces wrong tokens.
if [ -f "$MODEL_DIR/.download-complete" ]; then
  log "checkpoint already present and verified, skipping"
else
  # The repository is public, so no token is needed. If HF_TOKEN happens to be set the
  # hf CLI picks it up itself; the download script never reads, echoes or forwards it,
  # and neither does this one.
  log "=== downloading 1.56 TB, hours not minutes ==="
  ./scripts/download-model.sh "$MODEL_DIR" 2>&1 | tail -30 | tee -a "$OUT_DIR/download.log"
  touch "$MODEL_DIR/.download-complete"
  log "checkpoint verified byte-exact"
fi

if [ -f "$TRUNK_DIR/trunk.bin" ]; then
  log "packed trunk already present, skipping"
else
  log "=== packing the trunk, about four minutes ==="
  ./scripts/pack-trunk.sh "$MODEL_DIR" "$TRUNK_DIR" 2>&1 | tail -10 | tee -a "$OUT_DIR/pack.log"
fi

# ---- 5. one real generation, so there is a token before any benchmarking ---
log "=== proof of life ==="
./bin/k3 "$MODEL_DIR" --trunk "$TRUNK_DIR" --preset auto --tok "$MODEL_DIR" \
  --prompt "The capital of France is" --gen 8 --incremental \
  2>&1 | tee "$OUT_DIR/first-tokens.txt" | tail -20

# Real code, from the engine's own source, so a prompt is deterministic and needs no
# network. Read from a file rather than argv, which re-encodes.
P="$OUT_DIR/prompts"; mkdir -p "$P" "$OUT_DIR/logs"
head -c  120 "$REPO/src/core/k3_ops.c" > "$P/short.txt"
head -c 1200 "$REPO/src/core/k3_ops.c" > "$P/medium.txt"
head -c 4800 "$REPO/src/core/k3_ops.c" > "$P/long.txt"

# ---- 6. measurement helper -------------------------------------------------
TSV="$OUT_DIR/runs.tsv"
[ -f "$TSV" ] || printf 'arm\trep\ts_per_tok\tpeak_rss_gb\tgb_read\tpinned\trequests\tevictions\tids\n' > "$TSV"

# Every repetition is recorded rather than the best one: against a 33% noise floor a
# single sample is not a result. Counts are captured beside the seconds because counts
# are what survive the noise.
measure() {
  local tag=$1; shift
  local ref=""
  for r in $(seq 1 "$REPS"); do
    local lg="$OUT_DIR/logs/$tag.r$r.log" js="$OUT_DIR/logs/$tag.r$r.json"
    log "  $tag  rep $r/$REPS"
    if ! ./bin/k3 "$MODEL_DIR" --trunk "$TRUNK_DIR" --out "$js" "$@" >"$lg" 2>&1; then
      log "  *** $tag rep $r exited non-zero"; tail -4 "$lg" | sed 's/^/     | /'; continue
    fi
    local spt rss gbr pin rq ev ids
    spt=$(grep -oE '[0-9.]+ s/token average'             "$lg" | tail -1 | awk '{print $1}')
    rss=$(grep -oE 'PEAK RSS for the whole run: [0-9.]+'  "$lg" | tail -1 | awk '{print $7}')
    gbr=$(grep -oE 'read [0-9.]+ GB'                      "$lg" | tail -1 | awk '{print $2}')
    pin=$(grep -oE '[0-9]+/93 layers PINNED'              "$lg" | tail -1 | cut -d/ -f1)
    rq=$( grep -oE 'requests +: [0-9]+'                   "$lg" | tail -1 | awk '{print $3}')
    ev=$( grep -oE 'evictions [0-9]+'                     "$lg" | tail -1 | awk '{print $2}')
    ids=$(python3 -c "import json,sys;print(','.join(map(str,json.load(open(sys.argv[1]))['generated_ids'])))" "$js" 2>/dev/null)

    # An unreadable result file leaves ids empty, and empty-vs-empty would "confirm"
    # determinism. That is the vacuous pass the upstream ladder guards against.
    if [ -z "$ids" ]; then log "  *** $tag rep $r produced no generated_ids"; continue; fi
    if [ -z "$ref" ]; then ref="$ids"
    elif [ "$ids" != "$ref" ]; then log "  *** $tag rep $r IDS DIFFER from rep 1, that is a bug"; fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$r" "${spt:--}" "${rss:--}" "${gbr:--}" "${pin:--}" "${rq:--}" "${ev:--}" "$ids" >> "$TSV"
  done
}

# ---- 7. ARM A: the workload, which is the reason for the machine -----------
# Both upstream harnesses run a 5-token prompt at --gen 8, an operating point dominated
# by cold start: the same engine measured 19.21 s/token at 8 tokens and 10.66-11.79 at
# 16-32. Longer prompts and longer generations are the shape real work has, and nobody
# has measured it.
if [ "$RUN_WORKLOAD" = "1" ]; then
  log "=== ARM A: workload shape, gen ${GEN_WORK} ==="
  measure work_short  --preset auto --tok "$MODEL_DIR" --prompt-file "$P/short.txt"  --gen "$GEN_WORK" --incremental
  measure work_medium --preset auto --tok "$MODEL_DIR" --prompt-file "$P/medium.txt" --gen "$GEN_WORK" --incremental
  measure work_long   --preset auto --tok "$MODEL_DIR" --prompt-file "$P/long.txt"   --gen "$GEN_WORK" --incremental

  # Drafting is by n-gram lookup, so repetitive text is the case it should suit best and
  # code is the obvious candidate. Output is the serial greedy decode by construction.
  measure work_spec   --preset auto --tok "$MODEL_DIR" --prompt-file "$P/medium.txt" --gen "$GEN_WORK" --incremental --spec 4

  # Multi-turn: prefill once with --gen 0, then resume. Measured upstream at 3.9x on
  # turn two, never on commodity hardware.
  log "  work_resume: prefilling the shared prefix"
  ./bin/k3 "$MODEL_DIR" --trunk "$TRUNK_DIR" --preset auto --tok "$MODEL_DIR" \
    --prompt-file "$P/medium.txt" --gen 0 --incremental \
    --save-state "$OUT_DIR/turn1.state" --out "$OUT_DIR/logs/prefill.json" \
    > "$OUT_DIR/logs/prefill.log" 2>&1 || log "  *** prefill failed, see logs/prefill.log"
  if [ -f "$OUT_DIR/turn1.state" ]; then
    measure work_resume --preset auto --tok "$MODEL_DIR" --load-state "$OUT_DIR/turn1.state" \
      --prompt-file "$P/short.txt" --gen "$GEN_WORK" --incremental
  else
    log "  *** no saved state, skipping the resume arm"
  fi
fi

# ---- 8. ARM B: thread scaling, ROADMAP item 3 ------------------------------
# "OMP_NUM_THREADS has never been swept on this engine ... on memory-bound workloads
# throughput often declines past a point. Unknown here." On 16 cores against a 124-core
# reference that is the open question, and it costs one loop.
if [ "$RUN_THREADS" = "1" ]; then
  NPROC=$(nproc)
  log "=== ARM B: thread sweep on ${NPROC} cpus, gen ${GEN_THREAD} ==="
  for t in $(printf '%s\n' 4 8 16 32 "$NPROC" | sort -un); do
    [ "$t" -le "$NPROC" ] || continue
    export OMP_NUM_THREADS="$t"
    measure "threads_$t" --preset auto --tok "$MODEL_DIR" \
      --prompt-file "$P/short.txt" --gen "$GEN_THREAD" --incremental
  done
  unset OMP_NUM_THREADS
fi

# ---- 9. ARM C: the published campaign, replicated --------------------------
if [ "$RUN_CAMPAIGN" = "1" ]; then
  log "=== ARM C: memory ladder, ${REPS} reps per rung ==="
  ./benchmarks/memory-ladder.sh "$MODEL_DIR" "$TRUNK_DIR" "$OUT_DIR/ladder" "$REPS" \
    2>&1 | tee -a "$OUT_DIR/ladder.log" || log "ladder exited non-zero, see log"

  # Argument order is <model> <trunk> <out> [total_gb] [reps]. Passing reps in the fourth
  # position would silently set the memory budget to 3 GB and sweep a machine that is not
  # the one under test.
  log "=== ARM C: split sweep at ${SWEEP_GB} GB, ${REPS} reps per point ==="
  ./benchmarks/split-sweep.sh "$MODEL_DIR" "$TRUNK_DIR" "$OUT_DIR/split" "$SWEEP_GB" "$REPS" \
    2>&1 | tee -a "$OUT_DIR/split.log" || log "split sweep exited non-zero, see log"
fi

# ---- 10. collect -----------------------------------------------------------
# docs/results/<machine>/ is the upstream path for third-party hardware results and the
# Jetson writeup is the template: provenance travels with the numbers, or they are not
# reproducible by anyone else.
log "=== collecting ==="
{
  echo "date       : $(date -uIseconds)"
  echo "engine     : $(./bin/k3 --version 2>&1 | head -1)"
  echo "commit     : $(git -C "$REPO" rev-parse HEAD)"
  echo "cpu        : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
  echo "cpus       : $(nproc)"
  echo "ram_total  : $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo)"
  echo "trunk_fs   : $(df -PT "$TRUNK_DIR" | awk 'NR==2{print $1, $2, $7}')"
  echo "model_fs   : $(df -PT "$MODEL_DIR" | awk 'NR==2{print $1, $2, $7}')"
  echo "kernel     : $(uname -sr)"
  echo "compiler   : $(gcc --version | head -1)"
  echo "page_cache : not dropped between runs"
} | tee "$OUT_DIR/machine.txt"

tar czf "$HOME/k3results.tgz" -C "$HOME" "$(basename "$OUT_DIR")"
log "results  : $HOME/k3results.tgz"
log "per-run  : $TSV"
log "DONE. Copy the tarball off, then CANCEL THE SERVER IN ROBOT. Halting the OS does"
log "      not stop billing: the contract runs until it is cancelled."
