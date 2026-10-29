#!/bin/bash
# How many write(2, ...) system calls does each library make for the same ec_print_summary call? (strace -f on tools hand; the first child is the reference, the second the candidate)
# Usage: emit-writes.sh CAND.so   -> prints "pid count" per child (reference first)
cd "$(dirname "$0")"
R=$HOME/rustmig/engcfg/ref/libengcfg_ref.so
T=$(mktemp /tmp/eg-strace.XXXXXX)
strace -f -e trace=write -o "$T" ./hand "$R" "$1" summary 10 5 3 2 17547264 3 0.0005 > /dev/null 2>&1
grep 'write(2,' "$T" | awk '{c[$1]++} END {for (p in c) print p, c[p]}' | sort -n
grep 'write(2,' "$T" | cut -c1-110 | head -14
rm -f "$T"
