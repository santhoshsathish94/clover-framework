#!/bin/bash
# Sets up clover-kimi-k3 on a new Linux machine, start to finish, and verifies it: checks the machine, installs the tools, clones the code from GitHub,
# downloads the precomputed dataset from Hugging Face and the original open weights files the engine reads (4.4 GiB) from moonshotai/Kimi-K3, checks the files against
# src/scripts/dataset-files.tsv and src/scripts/open-weights-files.tsv, builds the program, fetches the DSpark
# draft model and runs src/scripts/verify-setup.sh (recorded prompts must give the recorded token ids, speculative decoding must equal plain decoding, a chat must remember).
# It can be run again at any point: every step skips what is already in place, and the download resumes.
#
# usage: [VARIABLE=value ...] set-up-new-machine.sh [STEP ...]        (no step = all of them, in this order:)
#   preflight  packages  clone  download  open-weights  check-dataset  inputs  build  dspark  verify
#
# Variables (all optional):
#   INSTALL_DIR     /opt/clover-k3        the code goes to $INSTALL_DIR/clover-framework (only the clover-kimi-k3 folder is checked out), linked as $INSTALL_DIR/clover-kimi-k3
#   REPO_URL        https://github.com/santhoshsathish94/clover-framework.git
#   REPO_REF        main                  a branch, tag or commit
#   HF_REPO         santhoshsathish94/clover-kimi-k3     HF_REPO_TYPE dataset    HF_REVISION main    HF_TOKEN (needed for a private repo)
#   DATASET_DIR     $INSTALL_DIR/datasets/computed  (put it on the big NVMe filesystem; about 1.4 TB); $INSTALL_DIR/clover-kimi-k3/src/datasets/computed is linked to it
#   OPEN_WEIGHTS_REPO moonshotai/Kimi-K3  OPEN_WEIGHTS_REVISION (pinned commit, see below)  OPEN_WEIGHTS_DIR $INSTALL_DIR/datasets/third-party/kimi-k3-open-weights; linked into src/datasets/third-party
#   DSPARK_DIR      $INSTALL_DIR/datasets/third-party/kimi-k3-dspark (linked into src/datasets/third-party)
#   SKIP_PACKAGES=1 do not run apt-get;  SKIP_DOWNLOAD=1 use the dataset as it is;  SKIP_DSPARK=1;  HF_WORKERS=8
#   VERIFY_PROMPTS  "1 4 10"   which of the 34 recorded prompts verify-setup.sh runs
# Exit status 0 only if no step failed. The report is written to $INSTALL_DIR/set-up-new-machine.report.
set -u
INSTALL_DIR="${INSTALL_DIR:-/opt/clover-k3}"
REPO_URL="${REPO_URL:-https://github.com/santhoshsathish94/clover-framework.git}"
REPO_REF="${REPO_REF:-main}"
HF_REPO="${HF_REPO:-santhoshsathish94/clover-kimi-k3}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
HF_REVISION="${HF_REVISION:-main}"
HF_WORKERS="${HF_WORKERS:-8}"
CLONE="$INSTALL_DIR/clover-framework"
ENGINE_DIR="$INSTALL_DIR/clover-kimi-k3"
SRC="$ENGINE_DIR/src"
export DATASET_DIR="${DATASET_DIR:-$INSTALL_DIR/datasets/computed}"
OPEN_WEIGHTS_REPO="${OPEN_WEIGHTS_REPO:-moonshotai/Kimi-K3}"
OPEN_WEIGHTS_REVISION="${OPEN_WEIGHTS_REVISION:-f831ab66814297da540d832a5235f8e904f29d06}"
export OPEN_WEIGHTS_DIR="${OPEN_WEIGHTS_DIR:-$INSTALL_DIR/datasets/third-party/kimi-k3-open-weights}"
export DSPARK_DIR="${DSPARK_DIR:-$INSTALL_DIR/datasets/third-party/kimi-k3-dspark}"
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(preflight packages clone download open-weights check-dataset inputs build dspark verify)
REPORT="$INSTALL_DIR/set-up-new-machine.report"
mkdir -p "$INSTALL_DIR"; echo "== $(date) set-up-new-machine.sh ${STEPS[*]}" >> "$REPORT"
FAILED=0

say()  { printf '%s\n' "$*" | tee -a "$REPORT"; }
pass() { say "PASS  $*"; }
warn() { say "WARN  $*"; }
fail() { say "FAIL  $*"; FAILED=1; }
# Links a data folder kept elsewhere (a big filesystem) to where the engine looks for it, under src/datasets.
link_data() {
  local target="$1" link="$2"
  mkdir -p "$target" "$(dirname "$link")"
  if [ -L "$link" ] || [ ! -e "$link" ]; then ln -sfn "$target" "$link"
  elif [ -d "$link" ] && [ -z "$(ls -A "$link")" ]; then rmdir "$link"; ln -sfn "$target" "$link"
  else say "keeping what is already at $link"; fi
}
need_engine() { [ -f "$SRC/scripts/settings.sh" ] || { fail "the code is not at $ENGINE_DIR yet (run the clone step first)"; return 1; }; }

step_preflight() {
  say "== preflight: is this machine one the engine can run on?"
  [ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] && pass "Linux x86_64" || fail "needs Linux on x86_64, this is $(uname -s) $(uname -m)"
  local flags; flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
  for f in avx2; do case " $flags " in *" $f "*) pass "cpu has $f";; *) fail "cpu lacks $f (the kernels need AVX2 at least)";; esac; done
  for f in avx512f avx512bw avx512_vnni; do case " $flags " in *" $f "*) pass "cpu has $f";; *) warn "cpu lacks $f: the engine falls back to AVX2 or scalar code; speed was only measured with AVX-512";; esac; done
  local ram_gb; ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$ram_gb" -ge 120 ]; then pass "RAM ${ram_gb} GB"; elif [ "$ram_gb" -ge 64 ]; then warn "RAM ${ram_gb} GB: the engine was measured on 124 GB (trunk locked in RAM about 55 GB, expert cache 26 GB, attention cache by context); expect swapping or lock failures with the defaults"; else fail "RAM ${ram_gb} GB is too small"; fi
  mkdir -p "$(dirname "$DATASET_DIR")" 2>/dev/null
  local probe_dir="$DATASET_DIR"; [ -d "$probe_dir" ] || probe_dir="$(dirname "$DATASET_DIR")"
  local free_gb; free_gb=$(df -P --block-size=1G "$probe_dir" | awk 'NR==2 {print $4}')
  local have_gb=0; [ -d "$DATASET_DIR" ] && have_gb=$(du -s --block-size=1G "$DATASET_DIR" 2>/dev/null | awk '{print $1}')
  local need_gb="${NEED_GB:-}"
  if [ -z "$need_gb" ]; then   # what scripts/dataset-files.tsv lists; 1500 if the list is not here
    need_gb=$(awk -F'\t' '!/^#/ && NF>=2 {s+=$2} END{ if (s>0) printf "%d", s/1073741824; else print 1500 }' "$(dirname "${BASH_SOURCE[0]}")/dataset-files.tsv" 2>/dev/null || echo 1500)
  fi
  if [ $have_gb -ge $need_gb ] || [ $free_gb -ge $(( (need_gb - have_gb) * 103 / 100 )) ]; then pass "disk: ${free_gb} GB free on the dataset filesystem, ${have_gb} GB already there (about ${need_gb} GB in all)"; else fail "disk: ${free_gb} GB free and ${have_gb} GB present on the dataset filesystem, about ${need_gb} GB needed in all, 3% more than the missing part must be free (set DATASET_DIR to a bigger filesystem)"; fi
  local probe="$probe_dir/.odirect-probe"
  if dd if=/dev/zero of="$probe" bs=1M count=4 oflag=direct status=none 2>/dev/null; then pass "O_DIRECT works on the dataset filesystem"; else fail "O_DIRECT does not work on $probe_dir (the experts are read with O_DIRECT; use a local NVMe filesystem, not tmpfs or a network share)"; fi
  rm -f "$probe"
  if [ "$(id -u)" = 0 ]; then pass "running as root (can raise the memory-lock limit the engine needs)"; else warn "not root: the engine locks the trunk in RAM (ulimit -l unlimited); as a normal user set memlock to unlimited in /etc/security/limits.d and in the service that runs this"; fi
  local thp; thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo unknown)
  case "$thp" in *"[never]"*) warn "transparent huge pages are off ($thp): the expert buffers ask for them with madvise; set it to madvise or always";; *) pass "transparent huge pages: $thp";; esac
  local maps; maps=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)
  [ "$maps" -ge 262144 ] && pass "vm.max_map_count $maps" || warn "vm.max_map_count is $maps; the engine maps many files, raise it to 1048576"
  for t in git curl python3 gcc objcopy awk sha256sum; do command -v $t > /dev/null && pass "tool: $t" || warn "tool missing: $t (the packages step installs it on Debian and Ubuntu)"; done
  if command -v gcc > /dev/null; then
    printf '#include <omp.h>\nint main(void){return omp_get_max_threads()<1;}\n' > "$INSTALL_DIR/.omp-probe.c"
    gcc -std=gnu11 -march=native -fopenmp "$INSTALL_DIR/.omp-probe.c" -o "$INSTALL_DIR/.omp-probe" 2>/dev/null && "$INSTALL_DIR/.omp-probe" && pass "gcc builds an OpenMP program with -march=native ($(gcc -dumpversion))" || fail "gcc cannot build an OpenMP program with -march=native"
    rm -f "$INSTALL_DIR/.omp-probe.c" "$INSTALL_DIR/.omp-probe"
  fi
}

step_packages() {
  say "== packages"
  if [ "${SKIP_PACKAGES:-0}" = 1 ]; then warn "skipped (SKIP_PACKAGES=1)"; else
    if command -v apt-get > /dev/null && [ "$(id -u)" = 0 ]; then
      DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential binutils git curl ca-certificates python3 python3-venv python3-pip > /dev/null \
        && pass "apt packages installed" || fail "apt-get install failed"
    else warn "not Debian/Ubuntu or not root: install build-essential, binutils, git, curl, python3, python3-venv yourself"; fi
  fi
  if [ ! -x "$INSTALL_DIR/venv/bin/hf" ]; then
    python3 -m venv "$INSTALL_DIR/venv" && "$INSTALL_DIR/venv/bin/pip" -q install huggingface_hub hf_transfer \
      && pass "Hugging Face command line installed in $INSTALL_DIR/venv" || fail "could not install huggingface_hub (python3 -m venv and pip needed)"
  else pass "Hugging Face command line already in $INSTALL_DIR/venv"; fi
}

step_clone() {
  say "== clone: $REPO_URL @ $REPO_REF (only clover-kimi-k3)"
  command -v git > /dev/null || { fail "git is missing"; return; }
  if [ ! -d "$CLONE/.git" ]; then
    git clone -q --filter=blob:none --no-checkout "$REPO_URL" "$CLONE" || { fail "git clone failed"; return; }
  fi
  ( cd "$CLONE" || exit 1
    if [ -f .git/clover-checked-out ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then say "local changes in $CLONE, not touching them"; exit 2; fi
    git sparse-checkout init --cone > /dev/null 2>&1; git sparse-checkout set clover-kimi-k3 > /dev/null 2>&1
    git fetch -q --filter=blob:none origin "$REPO_REF" && git checkout -q -f FETCH_HEAD && touch .git/clover-checked-out ) \
    && pass "checked out $(git -C "$CLONE" rev-parse --short HEAD) ($REPO_REF)" || { fail "could not check out $REPO_REF in $CLONE"; return; }
  if [ -L "$ENGINE_DIR" ] || [ ! -e "$ENGINE_DIR" ]; then ln -sfn "$CLONE/clover-kimi-k3" "$ENGINE_DIR"; fi
  link_data "$DATASET_DIR" "$CLONE/clover-kimi-k3/src/datasets/computed"
  link_data "$OPEN_WEIGHTS_DIR" "$CLONE/clover-kimi-k3/src/datasets/third-party/kimi-k3-open-weights"
  link_data "$DSPARK_DIR" "$CLONE/clover-kimi-k3/src/datasets/third-party/kimi-k3-dspark"
  [ -f "$SRC/code/whole-model.c" ] && pass "engine folder $ENGINE_DIR" || fail "$ENGINE_DIR has no src/code/whole-model.c"
  chmod +x "$SRC"/scripts/*.sh 2>/dev/null
}

step_download() {
  say "== download: $HF_REPO_TYPE $HF_REPO @ $HF_REVISION -> $DATASET_DIR"
  if [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then warn "skipped (SKIP_DOWNLOAD=1)"; return; fi
  local hf="$INSTALL_DIR/venv/bin/hf"; [ -x "$hf" ] || hf=$(command -v hf || command -v huggingface-cli || true)
  [ -n "$hf" ] || { fail "no Hugging Face command line (run the packages step)"; return; }
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "https://huggingface.co/api/${HF_REPO_TYPE}s/$HF_REPO")
  [ "$code" = 200 ] || { fail "Hugging Face answered $code for $HF_REPO_TYPE $HF_REPO: the repo does not exist yet, or it is private and HF_TOKEN is not set"; return; }
  mkdir -p "$DATASET_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf" download "$HF_REPO" --repo-type "$HF_REPO_TYPE" --revision "$HF_REVISION" --local-dir "$DATASET_DIR" --max-workers "$HF_WORKERS" \
    --include 'expert-*/*' --include 'trunk-*/*' --include 'engine-inputs/*' \
    && pass "download finished (re-run to resume if it was interrupted)" || fail "download failed or was interrupted: run this step again, it resumes"
}

step_open_weights() {
  say "== open-weights: $OPEN_WEIGHTS_REPO @ ${OPEN_WEIGHTS_REVISION:0:12} -> $OPEN_WEIGHTS_DIR (shard 94, tiktoken.model, tokenizer_config.json)"
  if [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then warn "skipped (SKIP_DOWNLOAD=1)"; return; fi
  local hf="$INSTALL_DIR/venv/bin/hf"; [ -x "$hf" ] || hf=$(command -v hf || command -v huggingface-cli || true)
  [ -n "$hf" ] || { fail "no Hugging Face command line (run the packages step)"; return; }
  mkdir -p "$OPEN_WEIGHTS_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 "$hf" download "$OPEN_WEIGHTS_REPO" model-00094-of-000096.safetensors tiktoken.model tokenizer_config.json --revision "$OPEN_WEIGHTS_REVISION" --local-dir "$OPEN_WEIGHTS_DIR" \
    && pass "open weights files downloaded (sizes and sha256 are checked in the check-dataset step)" || fail "download of the open weights files failed: run this step again, it resumes"
}

step_check_dataset() {
  say "== check-dataset: every file the engine opens, present and the right size"
  need_engine || return
  DATASET_DIR="$DATASET_DIR" "$SRC/scripts/check-dataset-files.sh" 2>&1 | tee -a "$REPORT"
  [ "${PIPESTATUS[0]}" = 0 ] || FAILED=1
}
step_inputs() {
  say "== inputs: recorded prompt and reference values"
  need_engine || return
  local in="$SRC/work/inputs"; mkdir -p "$in"
  cp -f "$SRC"/prompts/recorded-1000-token-prompt/text.txt "$SRC"/prompts/recorded-1000-token-prompt/tok-1000.txt "$in/" 2>/dev/null
  if [ -d "$DATASET_DIR/engine-inputs/refs" ]; then rm -rf "$in/refs"; cp -r "$DATASET_DIR/engine-inputs/refs" "$in/refs" && pass "recorded pod chain values copied (the 1000-token check can run)"
  else warn "no engine-inputs/refs in the dataset: the 1000-token check against the pod's own chain will be skipped"; fi
}

step_build() {
  say "== build"
  need_engine || return
  ( cd "$SRC" && DATASET_DIR="$DATASET_DIR" BUILD_ONLY=1 scripts/run-model.sh text prompts/code-is-prime.txt ) 2>&1 | tail -15 | tee -a "$REPORT"
  [ -x "$SRC/work/build/whole-model" ] && [ -f "$SRC/work/build/pipeline-stage.so" ] && pass "program and tail library built" || fail "build did not produce src/work/build/whole-model and pipeline-stage.so"
}

step_dspark() {
  say "== dspark: the draft model (public checkpoint, 7.1 GB)"
  need_engine || return
  if [ "${SKIP_DSPARK:-0}" = 1 ]; then warn "skipped (SKIP_DSPARK=1)"; return; fi
  local dir="$DSPARK_DIR"
  if [ -f "$dir/model.safetensors" ] && [ "$(stat -c %s "$dir/model.safetensors")" -gt 7000000000 ]; then pass "already at $dir"; return; fi
  DSPARK_DIR="$dir" "$SRC/scripts/download-dspark-checkpoint.sh"
  [ -f "$dir/model.safetensors" ] && [ "$(stat -c %s "$dir/model.safetensors")" -gt 7000000000 ] && pass "downloaded to $dir" || fail "the checkpoint is not complete in $dir (see $dir/fetch.log)"
}

step_verify() {
  say "== verify: run the engine"
  need_engine || return
  DATASET_DIR="$DATASET_DIR" VERIFY_PROMPTS="${VERIFY_PROMPTS:-1 4 10}" "$SRC/scripts/verify-setup.sh" 2>&1 | tee -a "$REPORT"
  [ "${PIPESTATUS[0]}" = 0 ] && pass "verify-setup.sh passed" || fail "verify-setup.sh failed"
}

for s in "${STEPS[@]}"; do
  case "$s" in
    preflight) step_preflight;; packages) step_packages;; clone) step_clone;; download) step_download;; open-weights) step_open_weights;; check-dataset) step_check_dataset;;
    inputs) step_inputs;; build) step_build;; dspark) step_dspark;; verify) step_verify;;
    *) fail "unknown step $s";;
  esac
  [ $FAILED = 0 ] || { say "stopping after the first failed step ($s); fix it and run again"; break; }
done
say "== $(date): $([ $FAILED = 0 ] && echo 'all steps passed' || echo 'FAILED') (report: $REPORT)"
exit $FAILED
