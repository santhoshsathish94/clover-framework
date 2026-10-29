#!/bin/bash
# Archives run 1 (driver with the wrong S3 wb expectation), rebuilds the driver, relaunches the whole matrix.
cd "$(dirname "$0")"
mkdir -p out/run1 && cp -r out/cpp out/rust out/M1 out/M2 out/M2v out/M3 out/M4 out/matrix.log out/run1/ 2>&1 | head -3
bash build.sh 2>&1 | head -20
bash launch.sh matrix.sh
