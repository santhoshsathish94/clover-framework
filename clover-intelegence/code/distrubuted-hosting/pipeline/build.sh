#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
CODE=$(cd .. && pwd)
mkdir -p bin
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math pipeline.c -lm -ldl -o bin/pipeline
for stage in server normalization; do
    define=SERVER_NO_MAIN
    if [[ "$stage" == normalization ]]; then define=NORMALIZATION_NO_MAIN; fi
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math -fPIC -shared \
        "-D$define" "$CODE/$stage/$stage.c" -lm -o "$CODE/$stage/bin/pipeline-stage.so"
done
for layer in $(seq 1 92); do
    path="$CODE/tansformers/transformer-$layer"
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math -fPIC -shared \
        -DTRANSFORMER_NO_MAIN "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so"
done
printf 'PASS: coordinator and 94 existing stage modules compiled without source changes\n'