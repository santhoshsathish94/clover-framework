#!/bin/bash
cd "$(dirname "$0")"
grep -E "^(HOLD|SCENARIO)" out/cpp/S3-1.txt out/rust/S3-1.txt
grep -E "^J " out/cpp/S3-1.txt | head -12 | cut -c1-260
echo ---S5; cat out/cpp/S5-1.txt; echo; cat out/rust/S5-1.txt
cat out/cpp/S3-1.err | head -5
