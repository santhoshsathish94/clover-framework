#!/bin/bash
# The whole box-side verification in one go (inside ~/rustmig/engcfg/verify): rebuild everything, unit tests, both smoke tests, full differential run, sanitizer arm, plumbing checks, hashes.
# Usage: ./all.sh [OUTDIR]     (default out/final). Everything printed also goes to OUTDIR/all.log. Needs `export PATH=$HOME/.cargo/bin:$PATH` for cargo.
export PATH=$HOME/.cargo/bin:$PATH
cd "$(dirname "$0")"
OUT=${1:-out/final}
mkdir -p "$OUT"
{
echo "=== $(date) on $(hostname), $(g++ --version | head -1), $(cargo --version)"
echo "=== reference build + smoke"; (cd ../ref && ./build.sh 2>&1 | tail -3 && ./smoke_ref) || echo "REF BUILD/SMOKE FAILED"
echo "=== rust build (release) + tests + smoke"; (cd .. && cargo build --release 2>&1 | tail -3 && cargo test 2>&1 | grep -E "^test result|FAILED|panicked" ; gcc -O1 -std=gnu11 -Wall -Wextra -I include -o smoke_rust smoke_rust.c -L target/release -lengcfg -Wl,-rpath,$PWD/target/release && ./smoke_rust) || echo "RUST BUILD/TEST/SMOKE FAILED"
echo "=== driver build"; ./build.sh 2>&1 | tail -3
echo "=== hashes"; ./md5.sh
echo "=== differential run (full)"; D=$(OUT=$OUT/diff ./run.sh); echo "$D"; cat "$D/stdout.txt"; ./count.sh "$D" | tail -12
echo "=== sanitizer arm (full)"; A=$(OUT=$OUT/asan ./asan.sh); echo "$A"; grep -E "^(E[0-9]:|TOTAL|exit)" "$A/stdout.txt"; cat "$A/san-summary.txt"
echo "=== plumbing checks (--flip N: the N-th comparison of a unit sees a perturbed candidate; every scenario must FAIL with a located report)"
for s in "E1 7" "E2 100" "E3 300" "E4 5" "E5 50" "E6 20" "E7 5" "E9 4"; do set -- $s
  P=$(OUT=$OUT/flip-$1 ./run.sh "" --only $1 --scale 0.02 --flip $2)
  echo "--- --only $1 --flip $2: $(grep -E '^TOTAL' "$P/stdout.txt")"; f=$(ls "$P"/fail/*.txt 2>/dev/null | head -1); [ -n "$f" ] && head -3 "$f" | cut -c1-200
done
} 2>&1 | tee "$OUT/all.log"
