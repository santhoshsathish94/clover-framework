#!/bin/bash
# cargo test --release of the given mutants (one at a time, nice): output out/mut/cargotest/<name>.txt and one summary line in out/mut/cargotest.txt
# usage: cargo_test.sh NAME...   (run on the box in ~/rustmig/loader/verify)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p out/mut/cargotest
for n in "$@"; do
    d=work/mut/$n
    [ -d "$d" ] || { echo "$n: no dir" >> out/mut/cargotest.txt; continue; }
    (cd "$d" && nice -n 10 timeout 600 cargo test --release -- --test-threads=4 > "$OLDPWD/out/mut/cargotest/$n.txt" 2>&1)
    rc=$?
    res=$(grep -E "^test result" out/mut/cargotest/$n.txt | head -1)
    failed=$(grep -E "^test .* FAILED|^    [a-z_0-9]+$" out/mut/cargotest/$n.txt | grep -E "FAILED" | sed 's/^test //; s/ \.\.\. FAILED//' | tr '\n' ' ')
    echo "$n rc=$rc | ${res:-no test result line (build error or abort)} | failing: ${failed:-none}" >> out/mut/cargotest.txt
done
