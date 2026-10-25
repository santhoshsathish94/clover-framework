#!/bin/bash
# Driver v2: keeps what must outlive the scratch: copies the small result files of the runs from out/ into v2/results/ (the raw per-unit files are not kept), then prints the sizes.
# usage (box, ~/rustmig/poolinit/verify): bash v2/keep_results.sh
cd "$(dirname "$0")/.."
R=v2/results; mkdir -p $R
cp out/v2mut/results.tsv $R/results-3060.tsv
cp out/v2mut/analysis.log $R/analysis-3060.log
cp out/v2mut/progress.txt $R/progress-3060.txt
cp out/v2big/results.tsv $R/results-20x-equivalents.tsv
cp out/v2big/analysis.log $R/analysis-20x-equivalents.log
cp out/v2big/progress.txt $R/progress-20x-equivalents.txt
cp out/v2/campaign.log out/v2/ctl-1.txt out/v2/ctl-2.txt out/v2/ctl-3.txt out/v2/ctl-20x.txt out/v2/build.txt out/v2/regen.txt out/v2/probes.txt out/v2/final.txt $R/
for n in ctl-1 ctl-2 ctl-3 ctl-20x ctl-stress ctl-last; do [ -f out/$n/summary.txt ] && cp out/$n/summary.txt $R/summary-$n.txt; done
du -sh v2 v2/logs v2/results | tr '\n' ' '; ls $R | wc -l
