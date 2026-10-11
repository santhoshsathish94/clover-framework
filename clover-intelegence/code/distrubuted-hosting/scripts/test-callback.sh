#!/bin/bash
# Two things this has to show, not assert by construction:
#   1. layer 93 doing the tail produces the same tokens the server's tail produced
#   2. a submission with a callback is acknowledged at once and the answer arrives
#      at the address, without the caller reading anything back off the connection
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
LOG=/tmp/callback-test.log
: > "$LOG"
exec >>"$LOG" 2>&1

echo "=== started $(date -u +%FT%TZ) ==="

pkill -f bin/pipeline-serve 2>/dev/null
rm -f /tmp/c.in /tmp/c.out /tmp/c.err /tmp/callback-body.json /tmp/receiver.log
mkfifo /tmp/c.in

cat > /tmp/receiver.py <<'PY'
import http.server, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        open('/tmp/callback-body.json', 'wb').write(body)
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', 8099), Handler).serve_forever()
PY
setsid nohup python3 /tmp/receiver.py > /tmp/receiver.log 2>&1 < /dev/null &
sleep 2

(
    exec 9<>/tmp/c.in
    ulimit -l unlimited
    export OMP_NUM_THREADS=8 OMP_PROC_BIND=false CLOVER_LOCK_TRUNK=1
    cd "$CODE/pipeline" && exec ./bin/pipeline-serve "$CODE" serve </tmp/c.in >/tmp/c.out 2>/tmp/c.err
) &

for _ in $(seq 1 300); do grep -q READY /tmp/c.out 2>/dev/null && break; sleep 1; done
echo "fleet ready at $(date -u +%T)"

# Same two prompts as the end-to-end gate, so the text can be compared directly.
SUBMITTED=$(date +%s)
printf 'SUBMIT http://127.0.0.1:8099/done 2 91019 25528\n' > /tmp/c.in
for _ in $(seq 1 100); do grep -q '^ACCEPTED' /tmp/c.out 2>/dev/null && break; sleep 0.1; done
ACKED=$(date +%s)
echo "acknowledged after $((ACKED - SUBMITTED))s (the caller is free at this point)"
grep '^ACCEPTED' /tmp/c.out || echo "NO ACCEPTED LINE"

for _ in $(seq 1 600); do [ -s /tmp/callback-body.json ] && break; sleep 2; done
echo "--- posted body ---"
cat /tmp/callback-body.json 2>/dev/null || echo "(nothing arrived)"
echo
echo "--- coordinator stdout ---"
cat /tmp/c.out
echo "--- coordinator stderr tail ---"
tail -5 /tmp/c.err

pkill -f bin/pipeline-serve 2>/dev/null
pkill -f /tmp/receiver.py 2>/dev/null
echo "=== finished $(date -u +%FT%TZ) ==="
