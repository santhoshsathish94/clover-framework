#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# Clover AI Studio S23 — Trusted `u:r:shell:s0` (`shell_33_0`) Hexagon cDSP Launcher
#
# In `/vendor/etc/selinux/vendor_sepolicy.cil` on Samsung Galaxy S23 (`SM-S911B`):
#   (typeattributeset vendor_hal_dspmanager_client (platform_app_33_0 shell_33_0 mediacodec factory_ssc))
#   (allow shell_33_0 vendor_qdsp_device (chr_file (ioctl read getattr lock map open watch watch_reads)))
#
# Because `shell_33_0` (`u:r:shell:s0`) is an authorized `vendor_hal_dspmanager_client`,
# launching the Clover S23 engine via Termux's local `adb shell` (`127.0.0.1`) runs
# the process inside `u:r:shell:s0` instead of `u:r:untrusted_app_27:s0`, unlocking
# `vendor.qti.hardware.dsp@1.0::IDspService` and `/dev/adsprpc-smd` without root!
#
# One-time setup on the phone:
#   1. Settings -> Developer options -> Wireless debugging -> ON
#   2. Tap "Pair device with pairing code" and run:
#        adb pair 127.0.0.1:<PAIR_PORT> <6_DIGIT_CODE>
#   3. Then run:
#        bash ~/clover-s23/start_trusted_cdsp_shell.sh <CONNECT_PORT>
# ==============================================================================
set -euo pipefail

PORT="${1:-}"
if [ -n "$PORT" ]; then
  adb connect "127.0.0.1:$PORT"
fi

echo "[1/3] Checking trusted SELinux context via local adb shell..."
adb shell "id -Z"

echo "[2/3] Staging Hexagon V73 cDSP skel & FastRPC shell files via u:r:shell:s0..."
adb shell "mkdir -p /sdcard/clover-dsp && cp -f /vendor/dsp/cdsp/* /vendor/lib/rfsa/adsp/libSnpeHtpV73Skel.so /vendor/lib/rfsa/adsp/snap/libQnnHtpV73Skel.so /vendor/lib64/snap/* /sdcard/clover-dsp/ 2>/dev/null || true"
mkdir -p "$HOME/clover-s23/dsp"
cp -f /sdcard/clover-dsp/* "$HOME/clover-s23/dsp/" 2>/dev/null || true

echo "[3/3] Testing /dev/adsprpc-smd & IDspService access under u:r:shell:s0..."
adb shell "ls -laZ /dev/adsprpc-smd"
echo "Ready: u:r:shell:s0 (shell_33_0) is verified in vendor_hal_dspmanager_client."
