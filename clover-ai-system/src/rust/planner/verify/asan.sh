#!/bin/bash
# Rebuilds the C++ reference planner under -fsanitize=address,undefined in a SEPARATE dir (ref-asan/; ../ref is only read), then runs the whole driver against it.
# usage: asan.sh NAME [pl_driver run options...]
cd "$(dirname "$0")"
mkdir -p ref-asan
g++ -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -std=c++17 -fPIC -shared -Wall -Wextra -I../include -I../../copyq/include -I../../slotcache/include -I../../routetable/include \
    ../ref/planner_ref.cpp -o ref-asan/libplanner_ref_asan.so || exit 1
md5sum ref-asan/libplanner_ref_asan.so
LIBASAN=$(g++ -print-file-name=libasan.so)
REF=$PWD/ref-asan/libplanner_ref_asan.so
export REF
exec ./run.sh "$1" --preload "$LIBASAN" "${@:2}"
