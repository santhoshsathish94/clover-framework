"""Prompt-length sweep report.

Requests are split where the reported position resets. A prompt longer than NPOS_SLOTS
is processed in more than one pass and emits a step for each, so a fixed steps-per-request
assumption misaligns the table.
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep.err"
steps = []
for line in open(path):
    if "STEP_JSON" not in line:
        continue
    get = lambda key: float(re.search(r'"' + key + r'":([\d.]+)', line).group(1))
    steps.append((int(get("position")), get("seconds"), get("expert_read_bytes")))

requests = []
current = []
for entry in steps:
    if current and entry[0] < current[-1][0]:
        requests.append(current)
        current = []
    current.append(entry)
if current:
    requests.append(current)

print("in  steps  first step  mean later  total s  readGB  first pos")
previous = 0.0
for index, request in enumerate(requests):
    later = [s[1] for s in request[1:]]
    read = request[-1][2] - previous
    previous = request[-1][2]
    print("%2d  %5d  %10.2f  %10.2f  %7.1f  %6.1f  %9d" % (
        index + 2, len(request), request[0][1],
        sum(later) / len(later) if later else 0.0,
        sum(s[1] for s in request), read / 1e9, request[0][0]))
