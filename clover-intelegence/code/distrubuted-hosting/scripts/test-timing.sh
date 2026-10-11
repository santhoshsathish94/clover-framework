#!/bin/bash
# One request, alone on the box, with a real receiver. Running it alone matters:
# concurrent requests share cores, so per-layer figures from a loaded run would be
# measuring contention rather than the layer.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/timing-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="
pkill -f '[b]in/pipeline-serve'; pkill -f '[r]eceiver.py'
rm -f /tmp/m.in /tmp/m.out /tmp/m.err /tmp/callback-body.json
mkfifo /tmp/m.in

cat > /tmp/receiver.py <<'PY'
import http.server
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        open('/tmp/callback-body.json', 'wb').write(body)
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', 8099), Handler).serve_forever()
PY
setsid nohup python3 /tmp/receiver.py >/dev/null 2>&1 </dev/null &
sleep 2

(
    exec 9<>/tmp/m.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1 CLOVER_TIMING=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/m.in >/tmp/m.out 2>/tmp/m.err
) &
for _ in $(seq 1 300); do grep -q READY /tmp/m.out 2>/dev/null && break; sleep 1; done
echo "fleet ready"

printf 'SUBMIT http://127.0.0.1:8099/done 2 1008 10484 318 15383 387\n' > /tmp/m.in
for _ in $(seq 1 900); do grep -q '^REQUEST' /tmp/m.out 2>/dev/null && break; sleep 2; done
sleep 3

echo "--- stdout ---"
cat /tmp/m.out
echo "--- delivered body ---"
cat /tmp/callback-body.json 2>/dev/null
echo
echo "--- timing rows ---"
grep '^TIMING' /tmp/m.err | sed 's/^TIMING [0-9]* //'
pkill -f '[b]in/pipeline-serve'; pkill -f '[r]eceiver.py'
echo "=== finished $(date -u +%FT%TZ) ==="
