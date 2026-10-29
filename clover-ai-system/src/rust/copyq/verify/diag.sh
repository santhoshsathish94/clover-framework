#!/bin/bash
cd "$(dirname "$0")"
date +%T; ps -eo pid,etime,pcpu,rss,args | grep -E "diff_driver" | grep -v grep | cut -c1-140
ls -la out/rust/S2-1.* out/cpp/S2-1.* 2>&1 | head; head -c 500 out/rust/S2-1.err; tail -3 out/rust/S2-1.txt | cut -c1-200
tail -3 out/matrix.log | cut -c1-200
