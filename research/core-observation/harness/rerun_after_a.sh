#!/usr/bin/env bash
# After-A final pass: the final code from an empty memory, as the first observation started. The previous after-A
# pass's framework memory is moved to archive/after-a-pass1 (kept). Trace and turns go to traces/after-a-final/.
# Stage first: scp mobile/s23_on_device_clover.py, mobile/clover_git_memory_repo.py and harness/report_turns.py to
# k3:/tmp/co-after-a/ (contention.py, c1_threads.py, bad_media.py and overlap.py are expected in $O/harness).
set -euo pipefail
O=/opt/clover-k3/core-observe
F=$O/framework/clover-ai-system
PY=$O/venv/bin/python
STAGE=/tmp/co-after-a
OUT=$O/traces/after-a-final
ARCH=$O/archive/after-a-pass1
systemctl stop clover-observe-fw 2>/dev/null || true
systemctl reset-failed clover-observe-fw 2>/dev/null || true
mkdir -p $ARCH/fw-base
for d in clover-memory-repo cache; do
  if [ -e $O/fw-base/$d ]; then
    if [ -e $ARCH/fw-base/$d ]; then echo "ARCHIVE_EXISTS $ARCH/fw-base/$d: not overwriting"; exit 1; fi
    mv $O/fw-base/$d $ARCH/fw-base/$d
  fi
done
install -m 644 -o root -g root $STAGE/s23_on_device_clover.py $F/mobile/s23_on_device_clover.py
install -m 644 -o root -g root $STAGE/clover_git_memory_repo.py $F/mobile/clover_git_memory_repo.py
install -m 644 -o root -g root $STAGE/report_turns.py $O/harness/report_turns.py
mkdir -p $OUT && chown nobody:nogroup $OUT
sha256sum $F/mobile/s23_on_device_clover.py $F/mobile/clover_git_memory_repo.py | sed "s|$O/||"
ENVS=(-E PYTHONDONTWRITEBYTECODE=1 -E PYTHONIOENCODING=utf-8 -E HOME=$O/home -E HF_HOME=$O/home/hf
      -E XDG_CACHE_HOME=$O/home/cache -E HF_HUB_OFFLINE=1 -E TRANSFORMERS_OFFLINE=1 -E LLAMA_ARG_THREADS=8
      -E GIT_AUTHOR_NAME=clover-observe -E GIT_AUTHOR_EMAIL=observe@localhost
      -E GIT_COMMITTER_NAME=clover-observe -E GIT_COMMITTER_EMAIL=observe@localhost)
SANDBOX=(--uid=nobody --gid=nogroup -p Nice=19 -p IOSchedulingClass=idle -p MemoryMax=20G -p ProtectSystem=strict
         -p ProtectHome=yes -p PrivateTmp=yes -p NoNewPrivileges=yes -p ReadWritePaths=$O/traces
         -p ReadWritePaths=$O/logs -p ReadWritePaths=$O/home --working-directory=$O/harness)
LOG=$O/logs/framework-after-a-final.log
: > $LOG && chown nobody:nogroup $LOG
systemd-run --unit=clover-observe-fw --description="Clover core observation: framework runtime (after A, final)" \
  "${SANDBOX[@]}" -p CPUAffinity=8-15 -p ReadWritePaths=$O/fw-base -p ReadWritePaths=$F/src/tools/hybrid-ncm-k3/memory_tree \
  -p StandardOutput=append:$LOG -p StandardError=append:$LOG "${ENVS[@]}" \
  $PY observe_framework.py --framework $F --base $O/fw-base --port 8791 --trace $OUT/framework.jsonl
for i in $(seq 1 120); do
  grep -q "listening on http://127.0.0.1:8791" $LOG && break
  systemctl is-active --quiet clover-observe-fw || { echo "UNIT_FAILED"; tail -40 $LOG; exit 1; }
  sleep 5
done
grep -q "listening on http://127.0.0.1:8791" $LOG || { echo "NOT_LISTENING"; tail -40 $LOG; exit 1; }
cd $O/harness
setpriv --reuid=nobody --regid=nogroup --clear-groups env HOME=$O/home $PY drive.py --stack framework --port 8791 \
  --out $OUT/framework-turns.jsonl --files $O/files > $O/logs/drive-framework-after-a-final.log 2>&1 || echo "DRIVE_EXIT $?"
$PY summarize.py $OUT framework > $OUT/framework-summary.md 2>&1 || echo "SUMMARY_FAILED"
echo "== C1 spans on different threads"; { $PY overlap.py $OUT/framework.jsonl C1 | grep "^overlapping"; } || true
echo "== C1 calls by thread"; $PY c1_threads.py $OUT/framework.jsonl | head -1 || true
echo "== startup modules"; { grep -m1 '"phase": "ready"' $OUT/framework.jsonl | $PY -c "import json,sys; e=json.loads(sys.stdin.read()); m=e.get('modules') or {}; [print(' ', k, (m.get(k) if isinstance(m, dict) else m)) for k in ('clover_cycle','clover_git_memory_repo','semantics')]"; } || true
echo "== turns"; $PY report_turns.py $OUT/framework-turns.jsonl || true
echo "== malformed photo"; setpriv --reuid=nobody --regid=nogroup --clear-groups env HOME=$O/home $PY bad_media.py 8791 > $OUT/bad-media.json 2>&1 || true; cat $OUT/bad-media.json
echo "== tool steps"; grep -a '_check_tool_permission", "phase": "return"\|"fn": "CloverMemoryGitRepo.read_file", "phase": "\(return\|error\)"' $OUT/framework.jsonl \
  | $PY -c "import json,sys; [print(' ', json.loads(l)['turn'], json.loads(l)['fn'].split('.')[-1], json.dumps(json.loads(l).get('args'), ensure_ascii=False)[:160], '->', json.dumps(json.loads(l).get('result') or json.loads(l).get('error'), ensure_ascii=False)[:200]) for l in sys.stdin]" || true
echo "FINAL_DONE"
