#!/bin/sh
# Is a token's computed 7168 vector a property of the token, or of its prefix?
#
# The proposal is to precompute, for each of the 163,840 vocabulary entries, the
# 7168 vector it becomes after all 92 layers, store it beside the embedding, and
# skip the layers at run time. That works if and only if the vector depends on
# the token alone.
#
# K3_DUMPRES writes resid[t] for every position after the final layer, so this
# is directly observable: run the same token behind two different prefixes and
# compare. No argument required.
#
#   A: 1008 10484 318 15383 387   ("The capital of France is")  -> 387 at position 4
#   B: 15383 387                  (short prefix)                -> 387 at position 1
#   C: 1008 10484 318 15383 387   (repeat of A)                 -> control
#
# C exists because a difference between A and B means nothing unless A repeats
# itself exactly; otherwise run-to-run noise would read as context dependence.
set -u
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting

run() {
    name=$1; ids=$2
    echo "{\"input_ids\":[$ids],\"max_new_tokens\":1}" > /tmp/dr-$name.json
    env -i PATH="$PATH" \
        CLOVER_DATASET=$S/bin/dataset CLOVER_CONFIG=$S/bin/configs/generation.json \
        K3_RESIDENT_WARMUP=0 K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE=0 \
        K3_DUMPRES=/tmp/dr-$name.bin \
        OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
        $S/bin/clover-one $S/bin/dataset < /tmp/dr-$name.json > /tmp/dr-$name.out 2>/dev/null
    echo "$name: $(ls -la /tmp/dr-$name.bin 2>/dev/null | awk '{print $5}') bytes, token $(grep -o '"token":[0-9]*' /tmp/dr-$name.out | head -1)"
}

run A "1008,10484,318,15383,387"
run B "15383,387"
run C "1008,10484,318,15383,387"

python3 - <<'PY'
import struct
E = 7168
def vec(path, pos):
    with open(path, "rb") as f:
        f.seek(pos * E * 4)
        return struct.unpack("<%df" % E, f.read(E * 4))

def cmp(a, b):
    import math
    d = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    dot = sum(x * y for x, y in zip(a, b))
    return d, d / max(na, 1e-30), dot / max(na * nb, 1e-30)

a4 = vec("/tmp/dr-A.bin", 4)    # token 387, prefix of 4
b1 = vec("/tmp/dr-B.bin", 1)    # token 387, prefix of 1
c4 = vec("/tmp/dr-C.bin", 4)    # token 387, same prefix as A

d, rel, cos = cmp(a4, c4)
print("\ncontrol  A vs C (identical input): dist %.6g  rel %.6g  cos %.9f" % (d, rel, cos))
d, rel, cos = cmp(a4, b1)
print("test     A vs B (same token, different prefix): dist %.6g  rel %.6g  cos %.6f" % (d, rel, cos))

print()
if cos > 0.9999:
    print("The vector is a property of the TOKEN. A precomputed column would work.")
else:
    print("The vector is a property of the PREFIX, not the token. The same token")
    print("after a different prefix lands somewhere else, so one stored vector")
    print("per vocabulary entry cannot represent it.")
PY
