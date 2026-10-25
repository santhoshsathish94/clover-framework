#!/bin/bash
# usage: waitfor.sh LOGNAME WORD SECONDS
cd "$(dirname "$0")"
timeout ${3:-280} bash -c "until grep -qx $2 out/$1.log; do sleep 5; done"; echo "wait exit $?"; tail -n 12 out/$1.log | cut -c1-230
