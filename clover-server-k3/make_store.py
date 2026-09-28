#!/usr/bin/env python3
"""Build one layer's expert store.

The layout is not a choice, it is what the bytes already are: probe_experts.py
established that an expert's six ranges form a single contiguous run of
17,547,264 B, that all 896 experts of a layer are identical in size, and that
a whole layer is one contiguous run in one file. So a row is an expert, and
building a store is a sequential read.

  expert(id INTEGER PRIMARY KEY, data BLOB)   896 rows, one per expert
  meta(k, v)                                  layer, sizes, source, sha256

The six sub-ranges live at fixed offsets inside the blob, recorded in meta, so
a reader that wants only gate.w does not have to fetch the whole expert.

Every blob is read back and compared against a second read of the source
range. A store that matches only itself proves nothing.

usage: make_store.py <layer> [more layers...]
"""
import collections
import hashlib
import json
import os
import sqlite3
import struct
import sys
import time

INDEX = os.environ.get("K3_INDEX")
STORES = os.environ.get("K3_STORES", "/srv/k3/stores")
PAGE = int(os.environ.get("K3_PAGE", "65536"))
CHUNK = 64 << 20


def read_index(path):
    b = open(path, "rb").read()
    if b[:4] != b"K3EQ":
        sys.exit("bad index magic")
    c = 4
    nlay, nslot = struct.unpack_from("<ii", b, c); c += 8
    c += nlay * nslot * 32
    nmodel, = struct.unpack_from("<i", b, c); c += 4
    c += nmodel * 32
    nfiles, = struct.unpack_from("<i", b, c); c += 4
    files = []
    for _ in range(nfiles):
        ln, = struct.unpack_from("<i", b, c); c += 4
        files.append(b[c:c + ln].decode()); c += ln
    nerec, = struct.unpack_from("<i", b, c); c += 4
    per = collections.defaultdict(list)
    for _ in range(nerec):
        L, e, which, kind, fid = struct.unpack_from("<iiiii", b, c); c += 20
        off, nb = struct.unpack_from("<qq", b, c); c += 16
        d0, d1 = struct.unpack_from("<ii", b, c); c += 8
        per[L].append((e, which, kind, fid, off, nb, d0, d1))
    return files, per


def build(L, files, recs):
    byexp = collections.defaultdict(list)
    for r in recs:
        byexp[r[0]].append(r)
    nexp = len(byexp)

    fids = set(r[3] for r in recs)
    if len(fids) != 1:
        sys.exit("L%d spans %d files; this builder assumes one" % (L, len(fids)))
    src = files[fids.pop()]

    # the sub-range offsets inside an expert blob, and the proof they tile it
    ref = sorted(byexp[min(byexp)], key=lambda r: r[4])
    base0 = ref[0][4]
    parts, cur = [], 0
    for e, which, kind, fid, off, nb, d0, d1 in ref:
        if off - base0 != cur:
            sys.exit("L%d expert %d is not contiguous" % (L, e))
        parts.append({"which": which, "kind": kind, "off": cur,
                      "nbytes": nb, "d0": d0, "d1": d1})
        cur += nb
    esz = cur

    order = sorted(byexp, key=lambda e: min(r[4] for r in byexp[e]))
    lo = min(r[4] for r in recs)
    hi = max(r[4] + r[5] for r in recs)
    if hi - lo != esz * nexp:
        sys.exit("L%d experts do not tile [%d,%d)" % (L, lo, hi))
    scrambled = sum(1 for i, e in enumerate(order) if i != e)

    os.makedirs(STORES, exist_ok=True)
    out = os.path.join(STORES, "L%02d.db" % L)
    if os.path.exists(out):
        os.remove(out)
    db = sqlite3.connect(out)
    db.execute("PRAGMA page_size = %d" % PAGE)
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("CREATE TABLE expert (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
    db.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    t0 = time.time()
    h_src = hashlib.sha256()
    with open(src, "rb") as f:
        f.seek(lo)
        for e in order:
            blob = f.read(esz)
            if len(blob) != esz:
                sys.exit("L%d expert %d short read" % (L, e))
            h_src.update(blob)
            db.execute("INSERT INTO expert (id, data) VALUES (?, ?)",
                       (e, sqlite3.Binary(blob)))
    meta = {"layer": L, "nexpert": nexp, "expert_bytes": esz,
            "source": src, "source_off": lo, "source_bytes": esz * nexp,
            "sha256": h_src.hexdigest(), "page_size": PAGE,
            "scrambled": scrambled, "parts": json.dumps(parts)}
    db.executemany("INSERT INTO meta (k, v) VALUES (?, ?)",
                   [(k, str(v)) for k, v in meta.items()])
    db.commit()
    wrote = time.time() - t0

    # read every blob back and compare against a second read of the source.
    # File order, not id order: the two only agree if the ids happen to be
    # laid out in order, and in this checkpoint they are not.
    t1 = time.time()
    h_db = hashlib.sha256()
    n = 0
    for e in order:
        row = db.execute("SELECT data FROM expert WHERE id = ?", (e,)).fetchone()
        if row is None:
            sys.exit("L%d expert %d missing from the store" % (L, e))
        h_db.update(row[0])
        n += 1
    db.close()
    if n != nexp:
        sys.exit("L%d store has %d rows, expected %d" % (L, n, nexp))

    h_again = hashlib.sha256()
    with open(src, "rb") as f:
        f.seek(lo)
        left = esz * nexp
        while left:
            b = f.read(min(CHUNK, left))
            if not b:
                sys.exit("L%d short re-read" % L)
            h_again.update(b)
            left -= len(b)
    verify = time.time() - t1

    ok = (h_db.hexdigest() == h_again.hexdigest() == h_src.hexdigest())
    sz = os.path.getsize(out)
    print("  L%-3d %d experts x %d B = %.2f GB  ->  store %.2f GB (+%.2f%%)"
          % (L, nexp, esz, esz * nexp / 1e9, sz / 1e9,
             100.0 * (sz - esz * nexp) / (esz * nexp)))
    print("       write %.1f s   verify %.1f s   sha256 %s   %s"
          % (wrote, verify, h_db.hexdigest()[:16], "MATCH" if ok else "MISMATCH"))
    print("       expert ids out of file order: %d of %d" % (scrambled, nexp))
    if not ok:
        sys.exit("L%d store does not match the source range" % L)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: make_store.py <layer> [layer...]")
    if not INDEX:
        sys.exit("K3_INDEX is not set")
    files, per = read_index(INDEX)
    for a in sys.argv[1:]:
        L = int(a)
        if L not in per:
            sys.exit("layer %d has no expert records" % L)
        build(L, files, per[L])


main()
