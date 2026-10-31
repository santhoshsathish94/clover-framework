#!/usr/bin/env bash
# ==============================================================================
# Builds the Native Linux Workstation Packages for Clover AI Studio (`cpu-server/`):
#   1. `clover-ai-studio`                                 (Compiled ELF x86_64 launcher)
#   2. `clover-ai-studio-workstation_1.0.0_amd64.deb`     (Debian / Ubuntu .deb package)
#   3. `CloverAIStudio-Workstation-Linux-x86_64.run`      (Universal Linux self-extracting .run installer)
# ==============================================================================
set -euo pipefail

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLOVER_AI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${CLOVER_AI_ROOT}/.." && pwd)"
BUILD_ROOT="/tmp/clover-workstation-pkg-build"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}/deb/DEBIAN" \
         "${BUILD_ROOT}/deb/usr/bin" \
         "${BUILD_ROOT}/deb/usr/share/applications" \
         "${BUILD_ROOT}/deb/usr/share/icons/hicolor/256x256/apps" \
         "${BUILD_ROOT}/deb/usr/share/icons/hicolor/scalable/apps" \
         "${BUILD_ROOT}/deb/opt/clover-ai-workstation/cpu-server" \
         "${BUILD_ROOT}/run_payload/cpu-server" \
         "${BUILD_ROOT}/run_payload/assets/app-icon"

echo "[1/4] Compiling native ELF launcher (clover-ai-studio) and libnemotron_cpu_engine.so..."
gcc -O2 -Wall -Wextra -s "${SCRIPT_DIR}/clover_workstation_launcher.c" -o "${SCRIPT_DIR}/clover-ai-studio"
gcc -O3 -march=native -fopenmp -fPIC -shared "${SCRIPT_DIR}/nemotron_cpu_engine.c" -lm -ldl -o "${SCRIPT_DIR}/libnemotron_cpu_engine.so"

# Stage files into .deb tree
install -m 0755 "${SCRIPT_DIR}/clover-ai-studio" "${BUILD_ROOT}/deb/usr/bin/clover-ai-studio"
install -m 0755 "${SCRIPT_DIR}/install-workstation-linux.sh" "${BUILD_ROOT}/deb/opt/clover-ai-workstation/cpu-server/install-workstation-linux.sh"
for f in nemotron_cpu_engine.c nemotron_hybrid_ncm.py nemotron_workstation_clover.py clover_workstation_launcher.c README.md; do
    if [[ -f "${SCRIPT_DIR}/${f}" ]]; then
        install -m 0644 "${SCRIPT_DIR}/${f}" "${BUILD_ROOT}/deb/opt/clover-ai-workstation/cpu-server/${f}"
    fi
done

if [[ -f "${REPO_ROOT}/assets/app-icon/clover-mark-256.png" ]]; then
    install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark-256.png" "${BUILD_ROOT}/deb/usr/share/icons/hicolor/256x256/apps/clover-ai-studio.png"
    install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark-256.png" "${BUILD_ROOT}/run_payload/assets/app-icon/clover-mark-256.png"
fi
if [[ -f "${REPO_ROOT}/assets/app-icon/clover-mark.svg" ]]; then
    install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark.svg" "${BUILD_ROOT}/deb/usr/share/icons/hicolor/scalable/apps/clover-ai-studio.svg"
    install -m 0644 "${REPO_ROOT}/assets/app-icon/clover-mark.svg" "${BUILD_ROOT}/run_payload/assets/app-icon/clover-mark.svg"
fi

cat > "${BUILD_ROOT}/deb/usr/share/applications/clover-ai-studio.desktop" << 'EOF'
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

cat > "${BUILD_ROOT}/deb/DEBIAN/control" << EOF
Package: clover-ai-studio-workstation
Version: ${VERSION}
Section: science
Priority: optional
Architecture: amd64
Depends: libc6 (>= 2.31), libgomp1, gcc, python3
Maintainer: Clover Framework <santhoshsathish94@gmail.com>
Homepage: https://cloverframework.com
Description: Clover AI Studio — 7-Layer Linux Workstation Edition (Nemotron 3.5 Super VL)
 Complete 7-layer neuro-symbolic, compiler-verified, memory-accumulating
 workstation runtime engineered for 64 GB - 128 GB Linux workstations with
 NVIDIA Nemotron 3.5 Super VL (120B-A12B, 131K Tekken vocab, 1M context)
 as default. Includes the native C/OpenMP/AVX-512 engine, 8-GPU trained
 HybridAgent + 11-level NCM renderer, Memory Graph, Ontology GraphRAG,
 Generative Compiler, and SpecMD R4 Clean-Drop slot cache.
EOF

cat > "${BUILD_ROOT}/deb/DEBIAN/postinst" << 'EOF'
#!/usr/bin/env bash
set -e
if [[ -f /opt/clover-ai-workstation/cpu-server/nemotron_cpu_engine.c ]] && command -v gcc >/dev/null 2>&1; then
    gcc -O3 -march=native -fopenmp -fPIC -shared \
        /opt/clover-ai-workstation/cpu-server/nemotron_cpu_engine.c -lm -ldl \
        -o /opt/clover-ai-workstation/cpu-server/libnemotron_cpu_engine.so 2>/dev/null || true
    if [[ -d /opt/clover-k3/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl ]]; then
        cp -f /opt/clover-ai-workstation/cpu-server/libnemotron_cpu_engine.so \
              /opt/clover-k3/clover-framework/clover-ai-system/src/models/nemotron-3.5-super-vl/libnemotron_cpu_engine.so || true
    fi
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "${BUILD_ROOT}/deb/DEBIAN/postinst"

cat > "${BUILD_ROOT}/deb/DEBIAN/prerm" << 'EOF'
#!/usr/bin/env bash
set -e
rm -f /opt/clover-ai-workstation/cpu-server/libnemotron_cpu_engine.so
exit 0
EOF
chmod 0755 "${BUILD_ROOT}/deb/DEBIAN/prerm"

DEB_OUT="${SCRIPT_DIR}/clover-ai-studio-workstation_${VERSION}_amd64.deb"
echo "[2/4] Building Debian/Ubuntu package: ${DEB_OUT} ..."
dpkg-deb --root-owner-group --build "${BUILD_ROOT}/deb" "${DEB_OUT}"

echo "[3/4] Building Universal Linux self-extracting installer: CloverAIStudio-Workstation-Linux-x86_64.run ..."
cp -f "${SCRIPT_DIR}/clover-ai-studio" \
      "${SCRIPT_DIR}/clover_workstation_launcher.c" \
      "${SCRIPT_DIR}/nemotron_cpu_engine.c" \
      "${SCRIPT_DIR}/nemotron_hybrid_ncm.py" \
      "${SCRIPT_DIR}/nemotron_workstation_clover.py" \
      "${SCRIPT_DIR}/install-workstation-linux.sh" \
      "${SCRIPT_DIR}/README.md" \
      "${BUILD_ROOT}/run_payload/cpu-server/"

TAR_PAYLOAD="${BUILD_ROOT}/payload.tar.gz"
tar -C "${BUILD_ROOT}/run_payload" -czf "${TAR_PAYLOAD}" .

RUN_OUT="${SCRIPT_DIR}/CloverAIStudio-Workstation-Linux-x86_64.run"
cat > "${RUN_OUT}" << 'HEADER_EOF'
#!/usr/bin/env bash
# ==============================================================================
# Clover AI Studio — Universal Self-Extracting Linux Workstation Installer (.run)
# Usage:
#   sudo ./CloverAIStudio-Workstation-Linux-x86_64.run
#   sudo ./CloverAIStudio-Workstation-Linux-x86_64.run --verify
#   sudo ./CloverAIStudio-Workstation-Linux-x86_64.run --uninstall
# ==============================================================================
set -euo pipefail

if [[ "${1:-}" == "--uninstall" ]]; then
    if [[ -f /opt/clover-ai-workstation/cpu-server/install-workstation-linux.sh ]]; then
        exec bash /opt/clover-ai-workstation/cpu-server/install-workstation-linux.sh --uninstall
    fi
fi

TMP_EXTRACT="$(mktemp -d /tmp/clover-workstation-setup.XXXXXX)"
trap 'rm -rf "${TMP_EXTRACT}"' EXIT

ARCHIVE_LINE="$(awk '/^__CLOVER_PAYLOAD_BELOW__$/ {print NR + 1; exit 0; }' "$0")"
tail -n +"${ARCHIVE_LINE}" "$0" | tar -xz -C "${TMP_EXTRACT}"

bash "${TMP_EXTRACT}/cpu-server/install-workstation-linux.sh" "$@"
exit 0
__CLOVER_PAYLOAD_BELOW__
HEADER_EOF

cat "${TAR_PAYLOAD}" >> "${RUN_OUT}"
chmod 0755 "${RUN_OUT}"

echo "[4/4] Verifying built packages..."
dpkg-deb --info "${DEB_OUT}"
ls -lh "${SCRIPT_DIR}/clover-ai-studio" "${DEB_OUT}" "${RUN_OUT}"
rm -rf "${BUILD_ROOT}"
echo "SUCCESS: Built Linux Workstation packages in ${SCRIPT_DIR}"
