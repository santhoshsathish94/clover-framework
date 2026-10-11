#!/bin/bash
# The same capture, sliced one position at a time, so "the residual always wins"
# can be checked against the position it was actually measured at.
for p in 0 1 2 3 5 9; do
    echo "##### position $p"
    POSITIONS=$p python3 /tmp/argmax-positions.py 2>&1 \
        | grep -E 'always won|^  residual|^  snapshot'
done
