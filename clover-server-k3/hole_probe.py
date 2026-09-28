#!/usr/bin/env python3
"""Read-only. Does the set of holes in a shard equal the set of expert ranges
the index names for that layer?

Nothing may be written back until that is true, because a hole the index does
not know about is data that cannot be restored, and a range the index names
that is NOT a hole is data that must not be overwritten.
"""
import os
import struct
import sys

IDX = "/opt/clover-k3/build/eqidx.bin"


def load_index(path):
    b = open(path, "rb").read()
    c = 0
    assert b[0:4] == b"K3EQ", "bad magic"
    c = 4
    nl, n_slots = struct.unpack_from("<ii", b, c); c += 8
    c += nl * n_slots * 32                       # slots
    (n_model,) = struct.unpack_from("<i", b, c); c += 4
    c += n_model * 32                            # mrec
    (n_files,) = struct.unpack_from("<i", b, c); c += 4
    files = []
    for _ in range(n_files):
        (ln,) = struct.unpack_from("<i", b, c); c += 4
        files.append(b[c:c + ln].decode()); c += ln
    (n_erec,) = struct.unpack_from("<i", b, c); c += 4
    erec = []
    for _ in range(n_erec):
        layer, expert, which, kind, fid = struct.unpack_from("<iiiii", b, c); c += 20
        off, nbytes = struct.unpack_from("<qq", b, c); c += 16
        d0, d1 = struct.unpack_from("<ii", b, c); c += 8
        erec.append((layer, expert, which, kind, fid, off, nbytes))
    return nl, files, erec


def holes(path):
    """Every hole in the file, via SEEK_HOLE / SEEK_DATA."""
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    out, pos = [], 0
    try:
        while pos < size:
            try:
                h = os.lseek(fd, pos, os.SEEK_HOLE)
            except OSError:
                break
            if h >= size:
                break
            try:
                d = os.lseek(fd, h, os.SEEK_DATA)
            except OSError:
                d = size
            out.append((h, d - h))
            pos = d
    finally:
        os.close(fd)
    return out


def merge(rs):
    rs = sorted(rs)
    out = []
    for a, n in rs:
        if out and a <= out[-1][0] + out[-1][1]:
            end = max(out[-1][0] + out[-1][1], a + n)
            out[-1] = (out[-1][0], end - out[-1][0])
        else:
            out.append((a, n))
    return out


def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 92
    nl, files, erec = load_index(IDX)

    mine = [r for r in erec if r[0] == layer]
    fids = sorted(set(r[4] for r in mine))
    print("layer %d: %d expert tensors across file_ids %s" % (layer, len(mine), fids))
    if len(fids) != 1:
        print("  layer spans several shards; handling that is not implemented")
        return 1
    path = files[fids[0]]
    print("  shard: %s" % path)

    st = os.stat(path)
    print("  apparent %.2f GB   actual %.2f GB" % (st.st_size / 1e9, st.st_blocks * 512 / 1e9))

    want = merge([(r[5], r[6]) for r in mine])
    have = merge(holes(path))
    wbytes = sum(n for _, n in want)
    hbytes = sum(n for _, n in have)
    print()
    print("  expert ranges in index : %5d ranges, %.2f GB" % (len(want), wbytes / 1e9))
    print("  holes in the shard     : %5d ranges, %.2f GB" % (len(have), hbytes / 1e9))

    # a hole not covered by the index is unrecoverable; report it loudly
    def covered(a, n, rs):
        for ra, rn in rs:
            if ra <= a and a + n <= ra + rn:
                return True
        return False

    bad = [(a, n) for a, n in have if not covered(a, n, want)]
    extra = [(a, n) for a, n in want if not covered(a, n, have)]
    print()
    print("  holes NOT named by the index : %d  (%.2f GB)  <- unrecoverable if > 0"
          % (len(bad), sum(n for _, n in bad) / 1e9))
    print("  index ranges that are NOT holes: %d  (%.2f GB)  <- must not be overwritten"
          % (len(extra), sum(n for _, n in extra) / 1e9))
    for a, n in bad[:5]:
        print("     unnamed hole at %d len %d" % (a, n))
    for a, n in extra[:5]:
        print("     index range already has data at %d len %d" % (a, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
