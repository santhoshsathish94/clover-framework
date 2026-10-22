#!/bin/bash
# Cleanup approved by the creator on 2026-10-21. Every item was checked: the engine opens only
# datasets/computed (809 files, all present at recorded size, real files) and datasets/third-party.
LOG=/root/cleanup-20261021.log
exec >> "$LOG" 2>&1
echo "=== start $(date '+%F %T')"
df -h / | tail -1
echo "--- manifest (dated folders, directories only)"
find /opt/clover-k3 -maxdepth 1 -type d -name '*2026*' | sort > /root/cleanup-20261021-dirs.txt
wc -l /root/cleanup-20261021-dirs.txt
echo "--- 1 /root/k3trunk_i8"
ls -la /root/k3trunk_i8
rm -rf /root/k3trunk_i8
df -h / | tail -1
echo "--- 2 old logs"
journalctl --vacuum-time=3d 2>&1 | tail -2
find /var/log -type f \( -name '*.gz' -o -name '*.[0-9]' -o -name '*.old' \) -delete
du -sh /var/log
df -h / | tail -1
echo "--- 3 dated folders in /opt/clover-k3"
while read -r d; do
  case "$d" in /opt/clover-k3/*2026*) rm -rf -- "$d" ;; *) echo "SKIP $d" ;; esac
done < /root/cleanup-20261021-dirs.txt
df -h / | tail -1
echo "--- 4 archive-before-restructure"
rm -rf /opt/clover-k3/archive-before-restructure
df -h / | tail -1
echo "=== done $(date '+%F %T')"
touch /root/cleanup-20261021.done
