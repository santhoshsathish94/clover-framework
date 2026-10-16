#!/bin/sh
# Measurement build only. K3_LSTAT is declared in the engine but never opened,
# so the per-layer vector instrument is dead code; this wires it and lifts the guard.
set -e
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
SRC=/tmp/clover-one-lstat.c

sed -e 's/"K3_STAGE", "K3_LSTAT", /"K3_STAGE", /' \
    -e 's|^    prefix_options();|    prefix_options();\n    { const char *lp = getenv("K3_LSTAT"); if (lp \&\& *lp) lstat_fp = fopen(lp, "w"); }|' \
    -e 's|fclose(lstat_fp); lstat_fp = NULL;|fflush(lstat_fp);|' \
    $S/clover-one.c > $SRC

echo "forbidden entry removed: $(grep -c '"K3_LSTAT"' $SRC) remaining"
echo "opener wired: $(grep -c 'lstat_fp = fopen' $SRC)"
echo "kept open across steps: $(grep -c 'fflush(lstat_fp)' $SRC)"

gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp \
    -DNPOS_SLOTS=8 -I$S $SRC -lm -o /tmp/clover-one-lstat 2>/tmp/lstat-build.err \
  && echo BUILD_OK || { echo BUILD_FAIL; tail -8 /tmp/lstat-build.err; exit 1; }
