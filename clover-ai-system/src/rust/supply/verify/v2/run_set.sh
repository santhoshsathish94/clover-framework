#!/bin/bash
# usage: run_set.sh NAME...   Runs the independent verifier's mutants/run_all.sh (unchanged) with every process capped at 3 GB of address space (ulimit -v), from verify/.
# Output as in run_all.sh: out/mut/<name>/ (units.log, summary.txt, *.res, *.diff kept; raw per-unit logs deleted after each run), out/mut/progress.txt (a line per run, disk check every 5 runs, stops below 40 GB free).
cd "$(dirname "$0")/.."
ulimit -v "${VLIMIT_KB:-3145728}"
exec mutants/run_all.sh "$@"
