#!/bin/sh
# Does a second process buy throughput, or is one already saturating the machine?
# K3_PAYLOAD_RAM=0 so the 75.9 GB payload is a shared read-only mapping; with it on
# each process would want its own private copy and two cannot fit in 124 GB.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
A='{"input_ids":[1008,10484,318,15383,387],"max_new_tokens":8}'
B='{"input_ids":[1008,10484,318,10417,387],"max_new_tokens":8}'

echo "=== memory before ==="
free -g | head -2

echo
echo "=== sequential: two requests, one process ==="
T0=$(date +%s.%N)
printf '%s\n%s\n' "$A" "$B" | K3_PAYLOAD_RAM=0 "$S/run.sh" >/tmp/seq.out 2>/dev/null
T1=$(date +%s.%N)
echo "wall $(echo "$T1-$T0" | bc) s"

echo
echo "=== concurrent: two processes, one request each ==="
T0=$(date +%s.%N)
printf '%s\n' "$A" | K3_PAYLOAD_RAM=0 "$S/run.sh" >/tmp/con1.out 2>/dev/null &
P1=$!
printf '%s\n' "$B" | K3_PAYLOAD_RAM=0 "$S/run.sh" >/tmp/con2.out 2>/dev/null &
P2=$!
wait $P1 $P2
T1=$(date +%s.%N)
echo "wall $(echo "$T1-$T0" | bc) s"

echo
echo "=== memory after ==="
free -g | head -2
echo
echo "tokens sequential : $(grep -o '"token":[0-9]*' /tmp/seq.out | cut -d: -f2 | tr '\n' ' ')"
echo "tokens proc 1     : $(grep -o '"token":[0-9]*' /tmp/con1.out | cut -d: -f2 | tr '\n' ' ')"
echo "tokens proc 2     : $(grep -o '"token":[0-9]*' /tmp/con2.out | cut -d: -f2 | tr '\n' ' ')"
