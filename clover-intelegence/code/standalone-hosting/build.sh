#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --tests ) ]] || { printf 'Use build.sh [--tests]\n' >&2; exit 2; }
test -f bin/configs/generation.json
mkdir -p bin
if [[ ! -e bin/dataset && ! -L bin/dataset ]]; then
    ln -s ../../../dataset bin/dataset
fi
[[ -L bin/dataset && "$(readlink -f bin/dataset)" == "$(realpath ../../dataset)" ]]
SLOTS=${CLOVER_NPOS_SLOTS:-8}
gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS clover-one.c -lm -o bin/clover-one
if [[ ${1:-} == --tests ]]; then
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS test-datasets.c -lm -o bin/test-datasets
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS test-boundaries.c -lm -o bin/test-boundaries
    gcc -O2 -std=c11 -Wall -Wextra -Werror -pedantic test-generation.c -o bin/test-generation
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-live-root.c -lm -o bin/test-live-root
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS test-incremental.c -lm -o bin/test-incremental
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-head-cache.c -lm -o bin/test-head-cache
    gcc -O3 -Wall -Wextra -Werror -pedantic test-decode.c -o bin/test-decode
    gcc -std=c11 -O2 test-pipeline-config.c -o bin/test-pipeline-config
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-root-phases.c -lm -o bin/test-root-phases
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-expert-pipeline.c -lm -lpthread -o bin/test-expert-pipeline
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-pipeline-integrity.c -lm -lpthread -o bin/test-pipeline-integrity
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS profile-stages.c -lm -lpthread -o bin/profile-stages
    gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic test-expert-results.c -o bin/test-expert-results
    gcc -std=c11 -O2 test-result-config.c -o bin/test-result-config
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS test-stage-overlap.c -lm -lpthread -o bin/test-stage-overlap
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS -DCLOVER_DEFER_SHARED clover-one.c -lm -lpthread -o bin/clover-one-before-overlap
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp test-cross-layer-prefetch.c -lm -lpthread -o bin/test-cross-layer-prefetch
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS test-cross-layer-model.c -lm -lpthread -o bin/test-cross-layer-model
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=$SLOTS -DCLOVER_NO_CROSS_LAYER_PREFETCH clover-one.c -lm -lpthread -o bin/clover-one-without-cross-layer
fi
printf 'PASS: standalone compiled; no dataset preparation or inference\n'