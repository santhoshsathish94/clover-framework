#!/bin/bash
# usage: launch.sh script.sh [args] : runs it detached, logs to out/<script>.nohup
cd "$(dirname "$0")"
s=$1; shift
setsid nohup bash "$s" "$@" > "out/$s.nohup" 2>&1 < /dev/null &
sleep 1; echo "launched $s pid $!"
