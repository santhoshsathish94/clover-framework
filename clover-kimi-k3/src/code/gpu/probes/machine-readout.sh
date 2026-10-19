#!/bin/bash
# Reads the real figures of a machine before anything is built for it: CPUs, RAM, GPUs and the links between them, NVMe, network, limits.
# Read only: it changes nothing and needs no root; a tool that is missing is reported, not an error.
# usage: [OUT=file] machine-readout.sh [DIRECTORY_ON_THE_DATA_DISK]       (the report goes to OUT, default ./machine-readout-HOST-DATE.txt, and to the screen)
set -u
DATA_DIR="${1:-$PWD}"
HOST="$(hostname 2>/dev/null || echo host)"
OUT="${OUT:-./machine-readout-$HOST-$(date +%Y%m%d-%H%M).txt}"
have() { command -v "$1" > /dev/null 2>&1; }
section() { printf '\n== %s\n' "$1"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 || printf '(exit status %s)\n' "$?"; }
{
echo "machine read-out $(date -Is) on $HOST; data directory $DATA_DIR"

section "system"
run uname -srm
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release

section "cpu"
if have lscpu; then lscpu | grep -E '^(Architecture|Model name|Socket|Thread|Core|CPU\(s\)|NUMA node|L1d|L2|L3|CPU max MHz|CPU MHz)'; else echo "lscpu not found"; fi
echo "vector and matrix instruction sets found:"; grep -o -E 'avx512[a-z_0-9]*|avx2|amx[a-z_0-9]*|fma|f16c' /proc/cpuinfo 2> /dev/null | sort -u | tr '\n' ' '; echo
have numactl && run numactl --hardware

section "memory"
if have free; then run free -b; fi
grep -E '^(MemTotal|MemAvailable|HugePages_Total|Hugepagesize)' /proc/meminfo 2> /dev/null
[ -r /sys/kernel/mm/transparent_hugepage/enabled ] && echo "transparent huge pages: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"

section "gpu"
if have nvidia-smi; then
  run nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.memory,power.limit,driver_version --format=csv
  run nvidia-smi topo -m
  run nvidia-smi -q -d MEMORY
else echo "no nvidia-smi: no NVIDIA driver on this machine (or it is not on PATH)"; fi
if have nvcc; then run nvcc --version; else echo "nvcc not found (no CUDA toolkit installed, or not on PATH)"; fi
have lspci && { echo '$ lspci (display, NVMe and network devices)'; lspci 2> /dev/null | grep -i -E 'nvidia|vga|3d controller|non-volatile|ethernet' ; }

section "storage"
if have lsblk; then run lsblk -b -o NAME,SIZE,TYPE,ROTA,MODEL,MOUNTPOINT; fi
if have df; then run df -B1 "$DATA_DIR"; fi
have nvme && run nvme list
mount 2> /dev/null | grep -E ' on (/|/mnt[^ ]*|/data[^ ]*|/srv[^ ]*) ' | head -5

section "network"
have ip && run ip -br link
for n in /sys/class/net/*; do [ -r "$n/speed" ] && [ "$(basename "$n")" != lo ] && echo "$(basename "$n"): $(cat "$n/speed" 2> /dev/null) Mbit/s"; done

section "limits"
echo "locked memory (ulimit -l): $(ulimit -l)"; echo "open files (ulimit -n): $(ulimit -n)"

section "summary, to compare with what the provider listed"
echo "listed: Ubuntu 24.04 Server; 8 x RTX 3090 ('VRAM per card: 192 GB'); 2 x Intel, 24 cores at 4.0 GHz; 384 GB RAM; '2000 NVMe'; 10 Gbps port"
if have nvidia-smi; then
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null | awk '{s += $1} END {printf "%.1f", s / 1024}')
  echo "read:   $n GPU(s), $(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | sort -u | tr '\n' ';') $mem GiB of video memory in all"
else echo "read:   no GPU readable"; fi
echo "read:   $(grep -m1 -i 'model name' /proc/cpuinfo 2> /dev/null | cut -d: -f2- | sed 's/^ *//'); $(have lscpu && lscpu | awk -F: '/^Socket\(s\)/ {gsub(/ /, "", $2); print $2}') socket(s); $(grep -c '^processor' /proc/cpuinfo 2> /dev/null) hardware threads; $(awk '/^MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo 2> /dev/null) RAM"
have df && echo "read:   $(df -B1 --output=size,avail "$DATA_DIR" 2> /dev/null | tail -1 | awk '{printf "data disk %.0f GiB, %.0f GiB free", $1 / 1073741824, $2 / 1073741824}')"
} 2>&1 | tee "$OUT"
echo
echo "report written to $OUT"
