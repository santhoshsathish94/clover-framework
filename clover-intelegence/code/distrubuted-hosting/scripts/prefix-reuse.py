#!/usr/bin/env python3
"""Does a shared prefix produce identical state?

france is 1008 10484 318 15383 387
japan  is 1008 10484 318 10417 387

They agree on positions 0, 1, 2 and diverge at position 3. If the computation depends
only on the tokens seen so far, every layer's residual hash must match exactly for
positions 0-2 and must differ from position 3 onward. If it does not match, something
other than the prefix is feeding the result and caching is not available.

This reads traces already captured; it computes nothing new.
"""
import collections


def load(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            owner, local, position, _nanos, digest, _detail = parts
            owner, local, position = int(owner), int(local), int(position)
            if local == 120 and position > 0:
                position -= 1
            rows[(owner, local, position)] = digest
    return rows


def main():
    a = load("/tmp/trace-france.csv")
    b = load("/tmp/trace-japan.csv")
    shared = sorted(set(a) & set(b))
    print(f"france records {len(a)}, japan records {len(b)}, comparable {len(shared)}")

    by_position = collections.defaultdict(lambda: [0, 0])
    for key in shared:
        owner, local, position = key
        entry = by_position[position]
        entry[0] += 1
        if a[key] == b[key]:
            entry[1] += 1

    print()
    print("position   records   identical   share")
    for position in sorted(by_position):
        total, same = by_position[position]
        print(f"{position:8d}  {total:8d}  {same:10d}   {same/total:6.3f}")

    print()
    print("first owner at which the two diverge, per position")
    for position in sorted(by_position):
        first = None
        for owner in range(1, 94):
            keys = [k for k in shared if k[0] == owner and k[2] == position]
            if not keys:
                continue
            if any(a[k] != b[k] for k in keys):
                first = owner
                break
        print(f"  position {position}: {'identical at every owner' if first is None else f'diverges at owner {first}'}")

    print()
    print("=== what fraction of the whole request is prefix-identical? ===")
    total = len(shared)
    same = sum(1 for k in shared if a[k] == b[k])
    print(f"{same}/{total} stage records identical = {same/total:.3f}")


if __name__ == "__main__":
    main()
