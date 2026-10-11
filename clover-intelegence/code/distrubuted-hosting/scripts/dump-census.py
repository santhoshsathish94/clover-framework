#!/usr/bin/env python3
"""What is actually in the vector dump: the first and last headers, and a census."""
import collections
import struct

PATH = "/tmp/vec-all.bin"

owners = collections.Counter()
locals_seen = collections.Counter()
positions = collections.Counter()
first = []
total = 0

with open(PATH, "rb") as handle:
    while True:
        header = handle.read(16)
        if len(header) < 16:
            break
        owner, local, position, count = struct.unpack("<4i", header)
        if count <= 0 or count > 1 << 20:
            print(f"bad count {count} at record {total}")
            break
        skipped = handle.seek(4 * count, 1)
        total += 1
        owners[owner] += 1
        locals_seen[local] += 1
        positions[position] += 1
        if total <= 12:
            first.append((owner, local, position, count))

print(f"records: {total}")
print(f"distinct owners: {len(owners)}  min {min(owners)} max {max(owners)}")
print(f"distinct positions: {sorted(positions)}")
print(f"records per owner: min {min(owners.values())} max {max(owners.values())}")
print()
print("first 12 headers (owner, local, position, count):")
for row in first:
    print("  ", row)
print()
print(f"local==0 records: {locals_seen.get(0, 0)}")
print(f"local==120 records: {locals_seen.get(120, 0)}")
print()
missing = [o for o in range(1, 93) if o not in owners]
print(f"owners 1..92 missing from the dump: {missing}")
