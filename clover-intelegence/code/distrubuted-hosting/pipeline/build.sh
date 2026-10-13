#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd -- "$(dirname -- "$0")"
CODE=$(cd .. && pwd)
mkdir -p bin
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math -fopenmp pipeline.c -lm -ldl -lpthread -o bin/pipeline-serve
# Every run script invokes pipeline-serve; pipeline is kept as an alias so the two
# names cannot drift into different builds again.
cp -f bin/pipeline-serve bin/pipeline
# The server carries final normalisation now, so normalization/ builds no pipeline stage.
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function -pedantic -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
    -DSERVER_NO_MAIN "$CODE/server/server.c" -lm -o "$CODE/server/bin/pipeline-stage.so"
# Layer 93 owns final normalisation, the head and the post back to the caller.
mkdir -p "$CODE/tansformers/transformer-93/bin"
gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
    -I"$CODE/common" \
    "$CODE/tansformers/transformer-93/transformer-93.c" -lm -o "$CODE/tansformers/transformer-93/bin/pipeline-stage.so"
for layer in $(seq 1 92); do
    path="$CODE/tansformers/transformer-$layer"
    # The shared reader carries entry points this consumer does not call; everything else stays fatal.
    gcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -Wno-unused-function -ffp-contract=off -fno-fast-math -fopenmp -fPIC -shared \
        -I"$CODE/common" -I"$path" -DTRANSFORMER_NO_MAIN "$path/transformer-$layer.c" -lm -o "$path/bin/pipeline-stage.so"
done
printf 'PASS: coordinator, server stage, layer 93 and 92 layer stages compiled\n'