"""LRU is pathological for a cyclic sweep. What would other policies give?

Access pattern is layers 1..92 once per prompt, then the next prompt starts at layer 1
again. With a cache smaller than the working set, LRU evicts exactly what is needed next.
"""
import glob
import random

ROOT = "/opt/clover-k3/clover-intelegence/dataset"
EXPERT_BYTES = 17547264


def load(name):
    path = sorted(glob.glob("%s/root-*/observations/%s/routes.tsv" % (ROOT, name)))[0]
    order = []
    for line in open(path):
        cells = line.split()
        if len(cells) >= 4:
            order.extend((int(cells[0]), int(e)) for e in cells[3:])
    return order


def simulate(order, capacity, policy):
    index, queue, hits = set(), [], 0
    random.seed(1)
    for key in order:
        if key in index:
            hits += 1
            continue
        if len(index) >= capacity:
            if policy == "lru":
                index.discard(queue.pop(0))
            elif policy == "mru":
                index.discard(queue.pop())
            else:
                victim = random.randrange(len(queue))
                index.discard(queue[victim])
                queue[victim] = queue[-1]
                queue.pop()
        index.add(key)
        queue.append(key)
    return 100.0 * hits / len(order)


france, japan = load("france"), load("japan")
mixed = france + japan + france + japan

print("working set per prompt: %d experts, %.1f GB" % (
    len(set(france)), len(set(france)) * EXPERT_BYTES / 1e9))
print("sequence: france, japan, france, japan  (%d accesses)" % len(mixed))
print()
print("capacity        GB     LRU     MRU  random")
for capacity in (512, 1472, 2944, 4096):
    print("%8d  %7.1f  %5.1f%%  %5.1f%%  %5.1f%%" % (
        capacity, capacity * EXPERT_BYTES / 1e9,
        simulate(mixed, capacity, "lru"),
        simulate(mixed, capacity, "mru"),
        simulate(mixed, capacity, "random")))
print()
print("bytes read per prompt after the first, at 2944 experts (51.7 GB):")
for policy in ("lru", "mru", "random"):
    hit = simulate(mixed, 2944, policy)
    print("  %-6s hit %5.1f%%  ->  %5.1f GB read per prompt" % (
        policy, hit, 99.7 * (1 - hit / 100.0)))
