#!/bin/bash
# Rebuilds every mutant with the toolchain of today (mutants/mk_mutants.py, unchanged) and checks that the build did not touch the independent agent's records: the diffs it rewrites are compared
# with the ones it wrote before, md5.txt (which mk_mutants.py appends to) is restored. Output: verify/v2work/mk_all.log, the new md5 of every .so in verify/v2work/md5-so-now.txt.
# usage (box, ~/rustmig/loader/verify): mutants/mk_all.sh [NAME...]
cd "$(dirname "$0")/.."
mkdir -p v2work/diffs-before
[ -f v2work/md5.txt.orig ] || cp -p mutants/md5.txt v2work/md5.txt.orig
for f in mutants/M*.diff; do [ -f v2work/diffs-before/$(basename $f) ] || cp -p $f v2work/diffs-before/; done
export PATH=$HOME/.cargo/bin:$PATH
{
    echo "== $(date -u +%FT%TZ) rustc $(rustc --version) cargo $(cargo --version)"
    python3 mutants/mk_mutants.py "$@"
    echo "== diffs changed by the rebuild (expected none):"
    for f in mutants/M*.diff; do cmp -s $f v2work/diffs-before/$(basename $f) || echo "CHANGED $f"; done
    cp -p v2work/md5.txt.orig mutants/md5.txt
    echo "== md5.txt restored: $(md5sum < mutants/md5.txt | cut -c1-8) (before $(md5sum < v2work/md5.txt.orig | cut -c1-8))"
    md5sum work/so/*.so > v2work/md5-so-now.txt
    echo "== $(ls work/so/*.so | wc -l) .so files, $(du -sh work | cut -f1) in work/ , done $(date -u +%FT%TZ)"
} > v2work/mk_all.log 2>&1
