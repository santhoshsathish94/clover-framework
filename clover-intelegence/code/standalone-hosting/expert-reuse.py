"""Would an expert cache help? Measured from the captured routes.

Each routes.tsv holds the whole run: layer, position, token, then the 16 selected expert
ids. One file is the complete record, so read one per prompt and take the layer from the
row, not from the directory name.
"""
import glob

ROOT = "/opt/clover-k3/clover-intelegence/dataset"
EXPERT_BYTES = 17547264


def load(name):
    path = sorted(glob.glob("%s/root-*/observations/%s/routes.tsv" % (ROOT, name)))[0]
    seen, order, rows = set(), [], 0
    for line in open(path):
        cells = line.split()
        if len(cells) < 4:
            continue
        rows += 1
        layer = int(cells[0])
        for expert in cells[3:]:
            key = (layer, int(expert))
            order.append(key)
            seen.add(key)
    return seen, order, rows, path


france, france_order, rows, path = load("france")
japan, japan_order, _, _ = load("japan")
shared = france & japan

print("source: %s" % path.split("dataset/")[1])
print("rows %d  (layers x positions)   accesses %d  (16 per row)" % (rows, len(france_order)))
print()
print("distinct (layer,expert) france : %5d   %6.1f GB" % (
    len(france), len(france) * EXPERT_BYTES / 1e9))
print("distinct (layer,expert) japan  : %5d   %6.1f GB" % (
    len(japan), len(japan) * EXPERT_BYTES / 1e9))
print("shared                         : %5d   %6.1f GB" % (
    len(shared), len(shared) * EXPERT_BYTES / 1e9))
print("union                          : %5d   %6.1f GB" % (
    len(france | japan), len(france | japan) * EXPERT_BYTES / 1e9))
print()
print("overlap shared/union           : %.1f%%" % (100.0 * len(shared) / len(france | japan)))
print("a cache holding france serves  : %.1f%% of japan's distinct experts"
      % (100.0 * len(shared) / len(japan)))
print()
print("accesses %d over %d distinct -> %.2f uses per expert within one prompt"
      % (len(france_order), len(france), len(france_order) / float(len(france))))
print()


def lru_hits(order, capacity):
    cache, index, hits = [], set(), 0
    for key in order:
        if key in index:
            hits += 1
            cache.remove(key)
        else:
            index.add(key)
        cache.append(key)
        if len(cache) > capacity:
            index.discard(cache.pop(0))
    return hits


print("LRU within one france pass:")
for capacity in (16, 64, 160, 512, 1472, 2944):
    print("  capacity %5d experts  %6.1f GB  hit %5.1f%%" % (
        capacity, capacity * EXPERT_BYTES / 1e9,
        100.0 * lru_hits(france_order, capacity) / len(france_order)))
print()
print("france then japan back to back, LRU:")
both = france_order + japan_order
for capacity in (512, 1472, 2944):
    print("  capacity %5d experts  %6.1f GB  hit %5.1f%%" % (
        capacity, capacity * EXPERT_BYTES / 1e9,
        100.0 * lru_hits(both, capacity) / len(both)))
