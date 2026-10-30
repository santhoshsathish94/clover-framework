#!/usr/bin/env bash
# ==============================================================================
# Clover AI System — One-Command Samsung Galaxy S23 (`Snapdragon 8 Gen 2`)
# On-Device Installer & Launcher for `LFM3-1.7B Omni` + All 6 Clover Layers
# (`clover-ai-system/src/models/lfm3-1.7b/install-s23-lfm3-omni.sh`)
# ==============================================================================
# Runs inside Termux (Android 14 / One UI 6+) on Samsung Galaxy S23 (`SM-S911B/U`)
# or any `aarch64` Snapdragon 8 Gen 2 device (`8 GB LPDDR5X RAM`, `UFS 4.0`):
#   1. Installs `clang` (`ARMv9.2-A`), `libomp`, `python`, `ffmpeg`, `git`.
#   2. Compiles `liblfm3_cpu_engine.so` with `-O3 -march=armv8.2-a+fp16+dotprod`
#      and pins OpenMP to the 5 out-of-order performance cores (`3..7`:
#      `1x Cortex-X3 @ 3.36 GHz + 2x Cortex-A715 + 2x Cortex-A710`), skipping
#      the 3 in-order `Cortex-A510` cores (`0..2`).
#   3. Builds / verifies the 24-layer 4-bit planar group-64 SwiGLU FFN trunks
#      (`ffn.4bit.bin`, `0.55 GiB` total across all 24 layers; total 7-layer
#      on-device footprint = `1.71 GiB` inside 8 GB RAM).
#   4. Launches the local Clover 7-Layer AI Server on `http://127.0.0.1:8765`
#      so Samsung Internet / Chrome on your S23 or the Clover Web UI connects
#      directly at zero network latency (`http://127.0.0.1:8765`).
# ==============================================================================
set -euo pipefail

echo "[Clover-S23] Detecting CPU architecture and Snapdragon 8 Gen 2 core topology..."
ARCH="$(uname -m || echo unknown)"
echo "[Clover-S23] Architecture: ${ARCH}"

# Detect Termux vs standard Linux aarch64/x86_64
if command -v pkg >/dev/null 2>&1; then
    echo "[Clover-S23] Termux environment detected on Android. Installing native packages..."
    pkg update -y
    pkg install -y clang libomp python ffmpeg git rust
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Choose compiler flags tailored for Snapdragon 8 Gen 2 (Cortex-X3 / A715 / A710) on aarch64
CC_BIN="${CC:-gcc}"
if command -v clang >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    CC_BIN="clang"
fi

if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
    MARCH_FLAGS="-O3 -march=armv8.2-a+fp16+dotprod -fopenmp -fPIC -shared"
    echo "[Clover-S23] Compiling liblfm3_cpu_engine.so with ARMv9/NEON flags: ${MARCH_FLAGS}"
else
    MARCH_FLAGS="-O3 -march=native -fopenmp -fPIC -shared"
    echo "[Clover-S23] Compiling liblfm3_cpu_engine.so with host flags: ${MARCH_FLAGS}"
fi

${CC_BIN} ${MARCH_FLAGS} "${SCRIPT_DIR}/lfm3_cpu_engine.c" -lm -ldl -o "${SCRIPT_DIR}/liblfm3_cpu_engine.so"
echo "[Clover-S23] Built ${SCRIPT_DIR}/liblfm3_cpu_engine.so"

# Pin OpenMP to 5 threads for Snapdragon 8 Gen 2 performance cores (X3 + 2xA715 + 2xA710)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-5}"
export CLOVER_S23_PIN_BIG_CORES=1
export CLOVER_LFM3_USE_4BIT_FFN=1

if [[ "${1:-}" == "--verify-only" ]]; then
    python3 "${SCRIPT_DIR}/lfm3_omni_s23_mobile.py"
    exit 0
fi

echo "[Clover-S23] Starting 7-Layer Clover LFM3-Omni Mobile Server on http://127.0.0.1:8765 ..."
exec python3 "${REPO_ROOT}/clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py" --host 127.0.0.1 --port 8765
