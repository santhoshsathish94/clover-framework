"""Show the raw position sequence around the 8/9 boundary, grouped by DONE_JSON."""
import re
import sys

err = open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep.err").read()
out = open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/sweep.out").read()

done = [(int(a), int(b)) for a, b in
        re.findall(r'"input_tokens":(\d+),"output_tokens":(\d+)', out)]
positions = [int(x) for x in re.findall(r'STEP_JSON \{"position":(\d+)', err)]
seconds = [float(x) for x in
           re.findall(r'STEP_JSON \{"position":\d+,"token":\d+,"seconds":([\d.]+)', err)]

print("DONE_JSON reports:", done[:4], "...", done[-2:] if done else "")
print("total steps:", len(positions), " total requests:", len(done))

at = 0
for index, (nin, nout) in enumerate(done):
    span = positions[at:at + nout]
    span_s = seconds[at:at + nout]
    if nin in (7, 8, 9, 10, 15):
        print("in=%-2d out=%-2d first positions %s   first secs %s" % (
            nin, nout, span[:4], ["%.2f" % v for v in span_s[:4]]))
    at += nout
