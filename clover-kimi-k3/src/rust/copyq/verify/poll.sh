#!/bin/bash
# usage: poll.sh LOGNAME [LINES]
cd "$(dirname "$0")"
tail -n ${2:-40} out/$1.log
