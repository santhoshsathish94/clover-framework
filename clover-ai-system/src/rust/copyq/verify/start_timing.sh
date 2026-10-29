#!/bin/bash
cd "$(dirname "$0")"
echo "--- M4 S1 err"; cat out/M4/S1-1.err | head -5; echo "--- M4 S1 txt lines"; wc -l out/M4/S1-1.txt
echo "--- run1 M4"; cat out/run1/M4/S1-1.err | head -5
bash launch.sh timing.sh
