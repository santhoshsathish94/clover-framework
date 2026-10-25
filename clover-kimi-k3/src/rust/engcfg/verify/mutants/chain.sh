#!/bin/bash
# Second phase, one step at a time (no two driver runs at once): the two extra mutants, cargo test of the uncaught mutants, then the event probes.
export PATH=$HOME/.cargo/bin:$PATH
cd ~/rustmig/engcfg/verify/mutants || exit 1
MK_ONLY="M139 M140" python3 mk_mutants.py
./run_all.sh M139 M140
DEBUG=1 ./run_tests.sh M41 M42 M43 M82 M83 M84 M95 M98 M99 M114 M138
python3 mk_probes.py > /dev/null
./run_probes.sh P41 P42 P43 P82 P83 P84 P95 P92 P98 P99 P100 P101 P102 P103 P104 P105 P114 P48 P138 P106 P107 P108 P109 P110 P112 P113 P115
echo CHAIN-DONE > ../out/chain.done
