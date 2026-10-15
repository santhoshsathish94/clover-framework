#!/usr/bin/env bash
# Build top-k variants of the engine WITHOUT touching the source tree.
#
# WHY A COPY AND NOT -DTOPK
# clover-one.c has a bare `#define TOPK 16`, not `#ifndef TOPK`. Passing -DTOPK=8 on the
# command line is silently overridden by the file's own definition -- gcc warns and
# carries on with 16. That would have been the fifth silent no-op of this session: a
# full k-sweep producing identical results and looking like "k makes no difference".
# So the define is rewritten in a COPY under /tmp and built with -I pointing back at the
# real directory for the headers. The source tree is not modified.
#
# WHY THE COMPILE-TIME CONSTANT IS A FAITHFUL TEST
# k3-data-problem.md measured "load only the top k of the 16 selected experts,
# renormalizing over those kept". clover-one.c does exactly that, at clover-one.c:2732-2747:
# an insertion scan keeps the TOPK highest score[e]+gbias[e], then
#     for (j < TOPK) ssum += wts_all[t][j];  iv = 1/ssum;  wts_all[t][j] *= iv;
# so the normaliser is the sum over the kept set. Lowering TOPK selects the top k and
# renormalises over those k, which is the recorded experiment rather than a variant of it.
#
# A CONTROL IS BUILT AT k=16 FROM THE SAME COPIED SOURCE, so any difference is k and not
# the copying, the include path, or a different compiler invocation.
set -eu

S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
OUT=/tmp/topk
mkdir -p "$OUT"

for k in 16 12 8 6 4; do
    sed "s/^#define TOPK   16$/#define TOPK   $k/" "$S/clover-one.c" > "$OUT/clover-one-k$k.c"
    if ! grep -q "^#define TOPK   $k$" "$OUT/clover-one-k$k.c"; then
        echo "*** sed did not take for k=$k -- the define does not match. Aborting."
        grep -n "define TOPK" "$OUT/clover-one-k$k.c" | head -3
        exit 1
    fi
    gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp -DNPOS_SLOTS=8 \
        -I"$S" "$OUT/clover-one-k$k.c" -lm -o "$OUT/clover-one-k$k" 2>"$OUT/build-k$k.err"
    printf "built k=%-3s %s\n" "$k" "$(ls -la "$OUT/clover-one-k$k" | awk '{print $5" bytes"}')"
done

echo
echo "sizes should differ where TOPK sizes static arrays:"
md5sum "$OUT"/clover-one-k*
