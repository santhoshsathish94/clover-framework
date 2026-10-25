#!/bin/bash
# Packs the audit trail of the driver-v2 campaign for copying to the repo (verify/mutants/v2out/): the gzipped units.log of every run, the summaries of the control runs, the progress log, the md5 of the
# rebuilt .so files and the log of the rebuild, the canonical-stability result. usage (box, ~/rustmig/loader/verify): mutants/v2_pack.sh   -> v2work/v2out.tgz
cd "$(dirname "$0")/.."
rm -rf v2work/pack; mkdir -p v2work/pack/units v2work/pack/fail v2work/pack/ctl
cp out/mut2/units/* v2work/pack/units/
cp out/mut2/ctl-* v2work/pack/ctl/
[ -d out/mut2/fail ] && cp out/mut2/fail/* v2work/pack/fail/
cp out/mut2/progress.txt v2work/pack/progress.txt
cp v2work/md5-so-now.txt v2work/mk_all.log v2work/pack/
[ -f v2work/canon/result.txt ] && gzip -c v2work/canon/result.txt > v2work/pack/canon-result.txt.gz
cp v2work/canon.log v2work/pack/
tar -czf v2work/v2out.tgz -C v2work/pack .
ls -l v2work/v2out.tgz; tar -tzf v2work/v2out.tgz | wc -l
