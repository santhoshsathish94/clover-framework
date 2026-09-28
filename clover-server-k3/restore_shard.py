#!/usr/bin/env python3
"""Put the expert bytes back into the checkpoint shard they were punched out of.

The shards still carry their original lengths and eqidx.bin still names every
expert tensor's offset, so the holes can be filled in place from the SQLite
store. Nothing is deleted here - reclaiming the store is a separate, explicit
step, taken only after this has verified.

Refuses to write over anything that is not a hole. Verifies every tensor by
reading it back and comparing sha256 against the blob it came from.

  ./restore_shard.py --layer 92              dry run
  ./restore_shard.py --layer 92 --apply      write, then verify
"""
import argparse
import hashlib
import os
import sqlite3
import struct
import sys

CH = 8 << 20


def load_index(path):
    b = open(path, "rb").read()
    assert b[0:4] == b"K3EQ", "bad magic"
    c = 4
    nl, n_slots = struct.unpack_from("<ii", b, c); c += 8
    c += nl * n_slots * 32
    (n_model,) = struct.unpack_from("<i", b, c); c += 4
    c += n_model * 32
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
        c += 8   # d0, d1
        erec.append((layer, expert, which, kind, fid, off, nbytes))
    return files, erec


def is_all_zero(fd, off, n):
    got = 0
    while got < n:
        b = os.pread(fd, min(CH, n - got), off + got)
        if not b:
            return False
        if b.count(0) != len(b):
            return False
        got += len(b)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--index", default="/opt/clover-k3/build/eqidx.bin")
    ap.add_argument("--db", default="/srv/k3/db")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    files, erec = load_index(a.index)
    mine = sorted([r for r in erec if r[0] == a.layer], key=lambda r: r[5])
    if not mine:
        print("layer %d has no expert records" % a.layer); return 2
    fids = set(r[4] for r in mine)
    if len(fids) != 1:
        print("layer spans several shards, not handled"); return 2
    shard = files[fids.pop()]
    dbp = os.path.join(a.db, "expert", "L%02d.db" % a.layer)
    if not os.path.exists(dbp):
        print("no store for layer %d at %s" % (a.layer, dbp)); return 2

    total = sum(r[6] for r in mine)
    st = os.stat(shard)
    print("layer %d" % a.layer)
    print("  shard   : %s" % shard)
    print("            apparent %.2f GB   actual %.2f GB" %
          (st.st_size / 1e9, st.st_blocks * 512 / 1e9))
    print("  store   : %s (%.2f GB)" % (dbp, os.path.getsize(dbp) / 1e9))
    print("  tensors : %d, %d bytes (%.2f GB)" % (len(mine), total, total / 1e9))
    print("  mode    : %s" % ("APPLY - will write into the shard" if a.apply else "dry run"))

    con = sqlite3.connect("file:%s?immutable=1" % dbp, uri=True)
    fd = os.open(shard, os.O_RDWR if a.apply else os.O_RDONLY)
    written = skipped = 0
    wbytes = 0
    try:
        for (_, e, w, k, _fid, off, nb) in mine:
            rowid = e * 6 + w * 2 + k
            n = con.execute("select length(data) from part_data where id=?",
                            (rowid,)).fetchone()
            if n is None:
                print("  MISSING row %d (expert %d which %d kind %d)" % (rowid, e, w, k))
                return 1
            if n[0] != nb:
                print("  LENGTH MISMATCH row %d: db %d, index %d" % (rowid, n[0], nb))
                return 1

            cur = os.pread(fd, min(nb, 4096), off)
            zero = (cur.count(0) == len(cur)) and is_all_zero(fd, off, nb)
            if not zero:
                blob = con.blobopen("part_data", "data", rowid, readonly=True)
                same = True
                o = 0
                while o < nb and same:
                    ln = min(CH, nb - o)
                    same = os.pread(fd, ln, off + o) == blob[o:o + ln]
                    o += ln
                blob.close()
                if same:
                    skipped += 1
                    continue
                print("  REFUSING: offset %d len %d is neither a hole nor a match" % (off, nb))
                return 1

            if a.apply:
                blob = con.blobopen("part_data", "data", rowid, readonly=True)
                o = 0
                while o < nb:
                    ln = min(CH, nb - o)
                    got = os.pwrite(fd, blob[o:o + ln], off + o)
                    if got != ln:
                        print("  short write at %d" % (off + o)); return 1
                    o += ln
                blob.close()
            written += 1
            wbytes += nb
    finally:
        os.close(fd)
        con.close()

    print("  %d tensors %s, %d already present, %.2f GB" %
          (written, "written" if a.apply else "would be written", skipped, wbytes / 1e9))
    if not a.apply:
        return 0

    os.sync()
    print("  verifying every tensor against the store ...")
    con = sqlite3.connect("file:%s?immutable=1" % dbp, uri=True)
    fd = os.open(shard, os.O_RDONLY)
    bad = 0
    try:
        for (_, e, w, k, _fid, off, nb) in mine:
            rowid = e * 6 + w * 2 + k
            blob = con.blobopen("part_data", "data", rowid, readonly=True)
            h1, h2 = hashlib.sha256(), hashlib.sha256()
            o = 0
            while o < nb:
                ln = min(CH, nb - o)
                h1.update(blob[o:o + ln])
                h2.update(os.pread(fd, ln, off + o))
                o += ln
            blob.close()
            if h1.digest() != h2.digest():
                bad += 1
                print("  MISMATCH expert %d which %d kind %d at %d" % (e, w, k, off))
                if bad > 5:
                    break
    finally:
        os.close(fd)
        con.close()

    st = os.stat(shard)
    print("  shard now: apparent %.2f GB   actual %.2f GB" %
          (st.st_size / 1e9, st.st_blocks * 512 / 1e9))
    if bad:
        print("  FAILED: %d tensors do not match" % bad)
        return 1
    print("  all %d tensors verified byte-for-byte" % len(mine))
    return 0


if __name__ == "__main__":
    sys.exit(main())
