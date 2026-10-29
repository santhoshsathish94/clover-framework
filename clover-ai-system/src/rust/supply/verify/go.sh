#!/bin/bash
# usage: go.sh NAME [run.sh options...]   Builds the driver and starts run.sh NAME in the background (output in out/NAME.stdout, summary in out/NAME/summary.txt when done); returns at once.
cd "$(dirname "$0")"
mkdir -p out
./build.sh > out/go.build.txt 2>&1 || { cat out/go.build.txt; exit 1; }
rm -rf "out/$1"
nohup ./run.sh "$@" > "out/$1.stdout" 2>&1 &
echo "started $1 (pid $!)"
