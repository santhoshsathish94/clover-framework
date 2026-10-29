#!/bin/bash
cd "$(dirname "$0")"
rm -rf /dev/shm/copyq_verify data/rec8.bin
rmdir data 2>/dev/null
du -sh . out mutants 2>/dev/null; ls /dev/shm | head; free -g | head -2
md5sum ../ref/copyq_ref.cu ../ref/libcopyq_ref.so ../target/release/libcopyq.so ../include/copyq.h ../src/*.rs | diff - out/md5-before.txt && echo UNTOUCHED-FILES-UNCHANGED
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | wc -l
