#!/usr/bin/env bash
# Both stacks side by side on k3, observed. Writes only under /opt/clover-k3/core-observe.
# Each stack runs as `nobody` in a sandboxed systemd unit: read-only system, /root and /home hidden, writes only to its
# own state folders, its own CPUs (framework 8-15, laptop 24-31), nice 19, idle I/O, 127.0.0.1 only.
# Usage: k3.sh setup | start | stop | status
set -euo pipefail
O=/opt/clover-k3/core-observe
F=$O/framework/clover-ai-system
L=$O/laptop-code/clover
PY=$O/venv/bin/python
FDEV=$(python3 -c 'import platform,re; print(re.sub(r"[^a-zA-Z0-9_-]+","-",(platform.node() or "s23").lower()).strip("-") or "s23")')
LDEV=$(python3 -c 'import re,socket; print(re.sub(r"[^a-z0-9-]+","-",socket.gethostname().lower()).strip("-") or "device")')
ENVS=(-E PYTHONDONTWRITEBYTECODE=1 -E PYTHONIOENCODING=utf-8 -E HOME=$O/home -E HF_HOME=$O/home/hf
      -E XDG_CACHE_HOME=$O/home/cache -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 -E LLAMA_ARG_THREADS=8
      -E GIT_AUTHOR_NAME=clover-observe -E GIT_AUTHOR_EMAIL=observe@localhost
      -E GIT_COMMITTER_NAME=clover-observe -E GIT_COMMITTER_EMAIL=observe@localhost)
SANDBOX=(--uid=nobody --gid=nogroup -p Nice=19 -p IOSchedulingClass=idle -p MemoryMax=20G -p ProtectSystem=strict
         -p ProtectHome=yes -p PrivateTmp=yes -p NoNewPrivileges=yes -p ReadWritePaths=$O/traces
         -p ReadWritePaths=$O/logs -p ReadWritePaths=$O/home --working-directory=$O/harness)

setup() {
  [ -x "$PY" ] || nice -n 19 cp -a /root/ncm-cpu/venv $O/venv   # same versions as the laptop's environment
  # the framework's C1 engine, built as the runtime builds it on x86_64 (compile_s23_lfm3_engine_so)
  (cd $F/mobile && { [ liblfm3_cpu_engine.so -nt lfm3_cpu_engine.c ] || nice -n 19 gcc -O3 -march=native -fopenmp \
      -fPIC -shared lfm3_cpu_engine.c -lm -ldl -o liblfm3_cpu_engine.so; })
  mkdir -p $O/home $O/traces $O/logs $O/files/reports $O/laptop-base/clover-ai-studio/config/$LDEV
  # device settings: CPU engine; shell, ssh and file changes denied; everything else at each stack's defaults
  echo '{"schema_version": 3, "engine": "cpu"}' > $O/fw-base/s23_studio_settings.json
  echo '{"internet": "allow", "code": "allow", "shell": "deny", "ssh": "deny", "files": "allow", "edit_files": "deny",
 "memory": "allow", "vscode": "deny", "media": "deny", "voice_camera": "deny"}' > $O/fw-base/s23_studio_permissions.json
  echo '{"engine": "cpu"}' > $O/laptop-base/clover-ai-studio/config/$LDEV/settings.json
  echo '{"internet": "allow", "code": "allow", "shell": "deny", "files": "allow", "edit_files": "deny",
 "memory": "allow", "vscode": "deny", "media": "deny"}' > $O/laptop-base/clover-ai-studio/config/$LDEV/permissions.json
  cat > $O/files/meeting-notes.txt <<'EOF'
Project sync, 28 October 2026
- Launch moved from 7 November to 14 November 2026.
- Budget approved: 42,000 EUR for the pilot.
- Priya owns the data migration; Tom owns the release notes.
- Next sync: 4 November 2026, 10:00.
EOF
  for i in 1 2 3; do echo "old report $i" > $O/files/reports/old-report-$i.txt; done
  # writable only where each stack keeps its state; model files and code stay root-owned and read-only
  chown nobody:nogroup $O/fw-base $O/laptop-base
  chown -R nobody:nogroup $O/home $O/traces $O/logs $O/laptop-base/clover-ai-studio $F/src/tools/hybrid-ncm-k3/memory_tree
  chown nobody:nogroup $O/fw-base/s23_studio_settings.json $O/fw-base/s23_studio_permissions.json
  echo "SETUP_OK framework-device=$FDEV laptop-device=$LDEV"
}

start() {
  systemd-run --unit=clover-observe-fw --description="Clover core observation: framework runtime" "${SANDBOX[@]}" \
    -p CPUAffinity=8-15 -p ReadWritePaths=$O/fw-base -p ReadWritePaths=$F/src/tools/hybrid-ncm-k3/memory_tree \
    -p StandardOutput=append:$O/logs/framework.log -p StandardError=append:$O/logs/framework.log "${ENVS[@]}" \
    $PY observe_framework.py --framework $F --base $O/fw-base --port 8791 --trace $O/traces/framework.jsonl
  systemd-run --unit=clover-observe-laptop --description="Clover core observation: laptop stack" "${SANDBOX[@]}" \
    -p CPUAffinity=24-31 -p ReadWritePaths=$O/laptop-base \
    -p StandardOutput=append:$O/logs/laptop.log -p StandardError=append:$O/logs/laptop.log "${ENVS[@]}" \
    $PY observe_laptop.py --clover $L --base $O/laptop-base --port 8792 --trace $O/traces/laptop.jsonl
}

stop() { systemctl stop clover-observe-fw clover-observe-laptop 2>/dev/null || true; }

status() {
  systemctl --no-pager --lines=0 status clover-observe-fw clover-observe-laptop 2>&1 | grep -E "^(●|  *Active|  *Main PID|  *Memory|  *CPU)" || true
  ss -ltnp | grep -E "127.0.0.1:(8791|8792) " || true
}

"${1:-status}"
