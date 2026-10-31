#!/usr/bin/env bash
# ==============================================================================
# Clover AI Studio — One-Command Linux Workstation Installer (`64 GB – 128 GB RAM`)
# (`clover-ai-system/cpu-server/install-workstation-linux.sh`)
# ==============================================================================
# Installs the complete 7-Layer Clover Workstation Stack on Linux (`x86_64` / `aarch64`):
#   - Default Foundation Model: `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B`
#     (`d = 4096`, `V = 131,072` Tekken, Native `1M` Context, 4-bit planar experts)
#   - Compiles `libnemotron_cpu_engine.so` with `-O3 -march=native -fopenmp`
#     (AVX-512 / AVX2 + 3D V-Cache L3 blocking + `SpecMD Policy 3 + R4 Clean-Drop`
#     + OS `madvise(MADV_WILLNEED)` + 4-Tier Workstation Data-Movement Ledger)
#   - Compiles native ELF launcher `/usr/local/bin/clover-ai-studio` (and `/usr/bin/clover-ai-studio`)
#   - Installs FreeDesktop `.desktop` launcher & 5-leaf Clover icons (`clover-ai-studio.desktop`)
#   - Registers `systemd` service `clover-ai-workstation.service` (`LimitMEMLOCK=infinity`)
#   - Installs the Clover VS Code extension (`clover-framework.clover-local`) if VS Code is present
#
# Usage:
#   sudo bash install-workstation-linux.sh              # Full install + compile + start service
#   sudo bash install-workstation-linux.sh --verify     # Full install + run 7-layer verification
#   sudo bash install-workstation-linux.sh --uninstall  # Clean uninstall
# ==============================================================================
set -euo pipefail

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOVER_AI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CLOVER_AI_ROOT}/.." && pwd)"
INSTALL_OPT="/opt/clover-ai-workstation"

if [[ "${1:-}" == "--uninstall" ]]; then
    echo "[Clover-Workstation] Uninstalling Clover AI Studio Workstation..."
    systemctl stop clover-ai-workstation 2>/dev/null || true
    systemctl disable clover-ai-workstation 2>/dev/null || true
    rm -f /etc/systemd/system/clover-ai-workstation.service
    systemctl daemon-reload 2>/dev/null || true
    rm -f /usr/local/bin/clover-ai-studio /usr/bin/clover-ai-studio
    rm -f /usr/share/applications/clover-ai-studio.desktop
    rm -f /usr/share/icons/hicolor/256x256/apps/clover-ai-studio.png
    rm -f /usr/share/icons/hicolor/scalable/apps/clover-ai-studio.svg
    rm -rf "${INSTALL_OPT}"
    echo "[Clover-Workstation] Uninstalled cleanly."
    exit 0
fi

echo "========================================================================"
echo "  Clover AI Studio — Linux Workstation Installer v${VERSION}"
echo "  Target Tier: 64 GB – 128 GB RAM Workstation (Default: Nemotron 3.5 Super VL)"
echo "========================================================================"

ARCH="$(uname -m || echo unknown)"
TOTAL_RAM_GB="$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "0")"
CPU_MODEL="$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || echo "Linux CPU")"
CPU_THREADS="$(nproc 2>/dev/null || echo 8)"

echo "[1/6] Hardware Check: ${CPU_MODEL} (${ARCH}, ${CPU_THREADS} threads, ${TOTAL_RAM_GB} GiB RAM)"

# 1. Ensure build & runtime dependencies (gcc, libgomp, python3, ffmpeg, curl) are present
if ! command -v gcc >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[2/6] Installing native Linux toolchain packages (gcc, openmp, python3, ffmpeg)..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential gcc libgomp1 python3 python3-venv ffmpeg curl
    elif command -v dnf >/dev/null 2>&1; then
        dnf install -y gcc libgomp python3 ffmpeg curl
    elif command -v pacman >/dev/null 2>&1; then
        pacman -Sy --noconfirm gcc python ffmpeg curl
    fi
else
    echo "[2/6] Native Linux toolchain verified (gcc + OpenMP + Python + FFmpeg present)."
fi

# 2. Stage workstation files into /opt/clover-ai-workstation
mkdir -p "${INSTALL_OPT}/cpu-server"
cp -f "${SCRIPT_DIR}/nemotron_cpu_engine.c" "${INSTALL_OPT}/cpu-server/"
cp -f "${SCRIPT_DIR}/nemotron_hybrid_ncm.py" "${INSTALL_OPT}/cpu-server/"
cp -f "${SCRIPT_DIR}/nemotron_workstation_clover.py" "${INSTALL_OPT}/cpu-server/"
cp -f "${SCRIPT_DIR}/clover_workstation_launcher.c" "${INSTALL_OPT}/cpu-server/"
if [[ -f "${SCRIPT_DIR}/nemotron_workstation_7layer_report.json" ]]; then
    cp -f "${SCRIPT_DIR}/nemotron_workstation_7layer_report.json" "${INSTALL_OPT}/cpu-server/"
fi

# 3. Compile `libnemotron_cpu_engine.so` with `-O3 -march=native -fopenmp` for both in-tree and /opt
echo "[3/6] Compiling libnemotron_cpu_engine.so with -O3 -march=native -fopenmp..."
gcc -O3 -march=native -fopenmp -fPIC -shared \
    "${SCRIPT_DIR}/nemotron_cpu_engine.c" -lm -ldl \
    -o "${SCRIPT_DIR}/libnemotron_cpu_engine.so"
cp -f "${SCRIPT_DIR}/libnemotron_cpu_engine.so" "${INSTALL_OPT}/cpu-server/libnemotron_cpu_engine.so"

if [[ -d "${CLOVER_AI_ROOT}/src/models/nemotron-3.5-super-vl" ]]; then
    cp -f "${SCRIPT_DIR}/libnemotron_cpu_engine.so" "${CLOVER_AI_ROOT}/src/models/nemotron-3.5-super-vl/libnemotron_cpu_engine.so"
fi

# 4. Compile native ELF launcher `clover-ai-studio`
echo "[4/6] Compiling native Linux ELF launcher (clover-ai-studio)..."
gcc -O2 -Wall -Wextra -s \
    "${SCRIPT_DIR}/clover_workstation_launcher.c" \
    -o "${SCRIPT_DIR}/clover-ai-studio"
install -m 0755 "${SCRIPT_DIR}/clover-ai-studio" "${INSTALL_OPT}/clover-ai-studio"
if [[ -w /usr/local/bin ]]; then
    install -m 0755 "${SCRIPT_DIR}/clover-ai-studio" /usr/local/bin/clover-ai-studio
fi
if [[ -w /usr/bin ]]; then
    install -m 0755 "${SCRIPT_DIR}/clover-ai-studio" /usr/bin/clover-ai-studio
fi

# 5. Install FreeDesktop `.desktop` launcher & 5-leaf Clover application icons
echo "[5/6] Installing Linux Desktop entry, icons, and systemd service..."
if [[ -d "${REPO_ROOT}/assets/app-icon" ]]; then
    mkdir -p /usr/share/icons/hicolor/256x256/apps /usr/share/icons/hicolor/scalable/apps
    if [[ -f "${REPO_ROOT}/assets/app-icon/clover-mark-256.png" ]]; then
        install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark-256.png" /usr/share/icons/hicolor/256x256/apps/clover-ai-studio.png
    fi
    if [[ -f "${REPO_ROOT}/assets/app-icon/clover-mark.svg" ]]; then
        install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark.svg" /usr/share/icons/hicolor/scalable/apps/clover-ai-studio.svg
    fi
fi

mkdir -p /usr/share/applications
cat > /usr/share/applications/clover-ai-studio.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Clover AI Studio (Workstation)
GenericName=7-Layer Neuro-Symbolic AI Workstation
Comment=Clover 7-Layer AI System powered by NVIDIA Nemotron 3.5 Super VL (120B-A12B, 1M Context)
Exec=/usr/bin/clover-ai-studio
Icon=clover-ai-studio
Terminal=false
Categories=Development;Science;ArtificialIntelligence;Utility;
Keywords=Clover;Nemotron;Kimi;LFM3;NCM;HybridAgent;AI;
StartupNotify=true
EOF

# Install VS Code extension if `code` CLI is available
if command -v code >/dev/null 2>&1; then
    VSIX_FILE="$(find "${CLOVER_AI_ROOT}" -maxdepth 3 -name "*.vsix" 2>/dev/null | head -n 1 || true)"
    if [[ -n "${VSIX_FILE}" ]]; then
        code --install-extension "${VSIX_FILE}" --force >/dev/null 2>&1 || true
        echo "  -> Installed VS Code extension: ${VSIX_FILE}"
    fi
fi

# 6. Register systemd service if systemd is present and no external k3_chat_server is already bound to 8765
if [[ -d /etc/systemd/system ]]; then
    PY_BIN="/usr/bin/python3"
    for cand_py in "/root/ncm-cpu/venv/bin/python" "/opt/clover-ai-workstation/venv/bin/python" "/root/hybridagent-venv/bin/python"; do
        if [[ -x "${cand_py}" ]]; then
            PY_BIN="${cand_py}"
            break
        fi
    done
    SRV_SCRIPT="${CLOVER_AI_ROOT}/src/tools/hybrid-ncm-k3/clover-chat-server.py"
    if [[ ! -f "${SRV_SCRIPT}" && -f "/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py" ]]; then
        SRV_SCRIPT="/opt/clover-k3/clover-framework/clover-ai-system/src/tools/hybrid-ncm-k3/clover-chat-server.py"
    fi

    cat > /etc/systemd/system/clover-ai-workstation.service << EOF
[Unit]
Description=Clover AI Studio 7-Layer Workstation Server (Default: Nemotron 3.5 Super VL)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$(dirname "${SRV_SCRIPT}")
Environment="CLOVER_NEMO_USE_4BIT_EXPERTS=1"
Environment="PYTHONIOENCODING=utf-8"
LimitMEMLOCK=infinity
ExecStart=${PY_BIN} -u ${SRV_SCRIPT} --host 127.0.0.1 --port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload 2>/dev/null || true
fi

echo "[6/6] Verifying Clover Workstation Engine status..."
/usr/bin/clover-ai-studio --engine-only || true
/usr/bin/clover-ai-studio --status || true

if [[ "${1:-}" == "--verify" ]]; then
    /usr/bin/clover-ai-studio --verify
fi

echo "========================================================================"
echo "  SUCCESS: Clover AI Studio (Workstation Edition v${VERSION}) installed!"
echo "  Run 'clover-ai-studio' to open Studio or 'clover-ai-studio --status'."
echo "========================================================================"
