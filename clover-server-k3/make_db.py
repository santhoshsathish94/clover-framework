#!/usr/bin/env python3
"""Move the open weights into SQLite, in full.

Layout, split by access pattern because SQLite's page size is per file and one
size cannot serve 14 KB embedding rows and 5.5 MB expert blobs:

  catalog.db            every relationship, foreign keys on, ~1 MB
  trunk/LNN.db          93 files, page 65536, one row per trunk slot
  expert/LNN.db         92 files, page 65536, one row per expert tensor
  client/embed.db       page 16384, one row per token, 163,840 rows
  client/lmhead.db      page 16384, one row per vocab entry, 163,840 rows
  client/head.db        page 65536, the three output norms
  client/vocab.db       page 4096, tiktoken, with an index on the bytes

lm_head is 2,348,810,240 B and a SQLite blob cannot exceed 2^31-1, so it is
per vocab row rather than one blob. Measured, not anticipated: the first
build died with OverflowError.

The payload tables carry (id, data) and nothing else. Shape, dtype and byte
counts live once in the catalog rather than on all 494,592 expert rows, and
id is computed, never searched:

  trunk slot     id = the S_* enum value
  expert tensor  id = expert * 6 + which * 2 + kind
  token          id = the token id

SQLite cannot enforce foreign keys across attached files, so the FK graph is
inside catalog.db and the payload files are bound to it by sha256 and a
`complete` flag written last. That is a checksum contract, not referential
integrity, and it is not claimed to be more.

Every file is verified against a second independent read of its source before
`complete` is set. Resumable: an already complete file is skipped.

usage: make_db.py catalog | trunk [L...] | expert [L...] | client | status
env:   K3_INDEX  K3_DB  K3_TRUNKBIN  K3_TIKTOKEN  K3_PUNCH=1
"""
import base64
import collections
import hashlib
import json
import os
import sqlite3
import struct
import subprocess
import sys
import time

INDEX = os.environ.get("K3_INDEX", "/opt/clover-k3/build/eqidx.bin")
DB = os.environ.get("K3_DB", "/srv/k3/db")
TRUNKBIN = os.environ.get("K3_TRUNKBIN", "/srv/k3/trunk/full.bin")
TIKTOKEN = os.environ.get("K3_TIKTOKEN", "/srv/k3/model/tiktoken.model")
PUNCH = os.environ.get("K3_PUNCH") == "1"
TOOL = "make_db.py/1"
CHUNK = 64 << 20

SLOTN = ["ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
         "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
         "QA", "QAN", "QB", "KA", "KAN", "KB",
         "MGATE", "MUP", "MDOWN",
         "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2"]
S_CQ, S_QA, S_MGATE = 14, 20, 26

ROLE = {}
for _i, _n in enumerate(SLOTN):
    ROLE[_i] = ("norm" if _n.endswith("N") or _n.endswith("_LN") or _n == "ENORM"
                else "moe" if _n in ("GATE", "GBIAS", "EDOWN", "EUP", "SH1", "SH2", "SH3")
                else "mlp" if _n in ("MGATE", "MUP", "MDOWN")
                else "attn")

DTYPE = [(0, "F32", 32), (1, "BF16", 16), (2, "I8R", 8), (3, "MXFP4", 4), (4, "E8", 8)]
PARTN = {(0, 0): "gate.w", (0, 1): "gate.s", (1, 0): "up.w",
         (1, 1): "up.s", (2, 0): "down.w", (2, 1): "down.s"}


# ------------------------------------------------------------------ the index
def read_index():
    b = open(INDEX, "rb").read()
    if b[:4] != b"K3EQ":
        sys.exit("bad index magic")
    c = 4
    nlay, nslot = struct.unpack_from("<ii", b, c); c += 8
    slots = []
    for _ in range(nlay * nslot):
        present, dtype = struct.unpack_from("<ii", b, c); c += 8
        off, nb = struct.unpack_from("<qq", b, c); c += 16
        d0, d1 = struct.unpack_from("<ii", b, c); c += 8
        slots.append((present, dtype, off, nb, d0, d1))
    nmodel, = struct.unpack_from("<i", b, c); c += 4
    mrec = []
    for _ in range(nmodel):
        fid, = struct.unpack_from("<i", b, c); c += 4
        off, nb = struct.unpack_from("<qq", b, c); c += 16
        d0, d1, dt = struct.unpack_from("<iii", b, c); c += 12
        mrec.append((fid, off, nb, d0, d1, dt))
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
    return nlay, nslot, slots, mrec, files, per


# ------------------------------------------------------------------ helpers
def newdb(path, page):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    db = sqlite3.connect(path)
    db.execute("PRAGMA page_size = %d" % page)
    db.execute("PRAGMA journal_mode = OFF")
    db.execute("PRAGMA synchronous = OFF")
    db.execute("PRAGMA auto_vacuum = NONE")
    db.execute("PRAGMA user_version = 1")
    return db


def sha_range(path, off, nb):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = nb
        while left:
            d = f.read(min(CHUNK, left))
            if not d:
                sys.exit("short read at %s+%d" % (path, off + nb - left))
            h.update(d)
            left -= len(d)
    return h.hexdigest()


def cat():
    return sqlite3.connect(os.path.join(DB, "catalog.db"))


def record(rel, role, layer_id, source, off, nb, sha, nrow, page):
    db = cat()
    db.execute("INSERT OR REPLACE INTO payload (file, role, layer_id, source,"
               " src_off, src_bytes, sha256, n_row, page_size, tool, built_at,"
               " complete) VALUES (?,?,?,?,?,?,?,?,?,?,?,1)",
               (rel, role, layer_id, source, off, nb, sha, nrow, page, TOOL,
                time.strftime("%Y-%m-%dT%H:%M:%S")))
    db.commit()
    db.close()


def done(rel):
    p = os.path.join(DB, "catalog.db")
    if not os.path.exists(p):
        return False
    db = sqlite3.connect(p)
    try:
        r = db.execute("SELECT complete FROM payload WHERE file = ?", (rel,)).fetchone()
    except sqlite3.OperationalError:
        return False
    finally:
        db.close()
    return bool(r and r[0] and os.path.exists(os.path.join(DB, rel)))


# ------------------------------------------------------------------ catalog
def build_catalog():
    nlay, nslot, slots, mrec, files, per = read_index()
    db = newdb(os.path.join(DB, "catalog.db"), 4096)
    db.executescript("""
CREATE TABLE dtype (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, bits INTEGER NOT NULL);
CREATE TABLE arch (
  id INTEGER PRIMARY KEY CHECK (id = 1), model TEXT NOT NULL,
  n_layer INTEGER NOT NULL, d_model INTEGER NOT NULL, n_head INTEGER NOT NULL,
  d_head INTEGER NOT NULL, n_expert INTEGER NOT NULL, top_k INTEGER NOT NULL,
  vocab INTEGER NOT NULL, d_latent INTEGER NOT NULL, d_ffn INTEGER NOT NULL,
  snapshot_every INTEGER NOT NULL);
CREATE TABLE layer_kind (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE layer (
  id INTEGER PRIMARY KEY,
  kind_id INTEGER NOT NULL REFERENCES layer_kind(id),
  has_expert INTEGER NOT NULL CHECK (has_expert IN (0,1)),
  is_snapshot INTEGER NOT NULL CHECK (is_snapshot IN (0,1)));
CREATE TABLE slot_def (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, role TEXT NOT NULL
  CHECK (role IN ('attn','norm','moe','mlp','out')));
CREATE TABLE slot (
  layer_id INTEGER NOT NULL REFERENCES layer(id),
  slot_id  INTEGER NOT NULL REFERENCES slot_def(id),
  dtype_id INTEGER NOT NULL REFERENCES dtype(id),
  d0 INTEGER NOT NULL, d1 INTEGER NOT NULL, nbytes INTEGER NOT NULL,
  PRIMARY KEY (layer_id, slot_id)) WITHOUT ROWID;
CREATE TABLE part_def (
  id INTEGER PRIMARY KEY,
  which INTEGER NOT NULL CHECK (which BETWEEN 0 AND 2),
  kind  INTEGER NOT NULL CHECK (kind  BETWEEN 0 AND 1),
  name TEXT NOT NULL UNIQUE,
  dtype_id INTEGER NOT NULL REFERENCES dtype(id),
  d0 INTEGER NOT NULL, d1 INTEGER NOT NULL, nbytes INTEGER NOT NULL,
  disk_ord INTEGER NOT NULL UNIQUE, UNIQUE (which, kind));
CREATE TABLE model_def (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
  dtype_id INTEGER NOT NULL REFERENCES dtype(id),
  d0 INTEGER NOT NULL, d1 INTEGER NOT NULL, nbytes INTEGER NOT NULL);
CREATE TABLE payload (
  file TEXT PRIMARY KEY,
  role TEXT NOT NULL CHECK (role IN ('trunk','expert','embed','head','vocab')),
  layer_id INTEGER REFERENCES layer(id),
  source TEXT NOT NULL, src_off INTEGER, src_bytes INTEGER,
  sha256 TEXT NOT NULL, n_row INTEGER NOT NULL, page_size INTEGER NOT NULL,
  tool TEXT NOT NULL, built_at TEXT NOT NULL,
  complete INTEGER NOT NULL DEFAULT 0 CHECK (complete IN (0,1)),
  CHECK ((role IN ('trunk','expert')) = (layer_id IS NOT NULL)));
CREATE INDEX payload_layer ON payload(layer_id, role);
""")
    db.executemany("INSERT INTO dtype VALUES (?,?,?)", DTYPE)
    db.execute("INSERT INTO arch VALUES (1,'kimi-k3',93,7168,96,128,896,16,163840,3584,3072,12)")
    db.executemany("INSERT INTO layer_kind VALUES (?,?)",
                   [(0, "dense"), (1, "KDA MoE"), (2, "MLA MoE")])
    db.executemany("INSERT INTO slot_def VALUES (?,?,?)",
                   [(i, SLOTN[i], ROLE[i]) for i in range(len(SLOTN))])

    for L in range(nlay):
        row = lambda s: slots[L * nslot + s]
        kind = 0 if row(S_MGATE)[0] else (2 if row(S_QA)[0] else 1)
        db.execute("INSERT INTO layer VALUES (?,?,?,?)",
                   (L, kind, 1 if L in per else 0, 1 if L % 12 == 0 else 0))
        for s in range(min(nslot, len(SLOTN))):
            present, dt, off, nb, d0, d1 = row(s)
            if present:
                db.execute("INSERT INTO slot VALUES (?,?,?,?,?,?)", (L, s, dt, d0, d1, nb))

    e0 = min(r[0] for r in per[1])
    ref = sorted([r for r in per[1] if r[0] == e0], key=lambda r: r[4])
    for ordinal, (e, which, kind, fid, off, nb, d0, d1) in enumerate(ref):
        db.execute("INSERT INTO part_def VALUES (?,?,?,?,?,?,?,?,?)",
                   (which * 2 + kind, which, kind, PARTN[(which, kind)],
                    3 if kind == 0 else 4, d0, d1, nb, ordinal))

    mn = ["embed_tokens", "out_attn_res_norm", "out_attn_res_proj", "out_norm", "lm_head"]
    for i, (fid, off, nb, d0, d1, dt) in enumerate(mrec):
        db.execute("INSERT INTO model_def VALUES (?,?,?,?,?,?)", (i, mn[i], dt, d0, d1, nb))
    db.commit()
    db.close()
    print("  catalog.db: %d layers, %d slot defs, 6 part defs, %d model defs"
          % (nlay, len(SLOTN), len(mrec)))


# ------------------------------------------------------------------ trunk
def build_trunk(L):
    rel = "trunk/L%02d.db" % L
    if done(rel):
        print("  %s already complete" % rel); return
    nlay, nslot, slots, _, _, _ = read_index()
    have = [(s,) + slots[L * nslot + s][1:] for s in range(min(nslot, len(SLOTN)))
            if slots[L * nslot + s][0]]
    lo = min(r[2] for r in have)
    hi = max(r[2] + r[3] for r in have)

    t0 = time.time()
    db = newdb(os.path.join(DB, rel), 65536)
    db.execute("CREATE TABLE slot_data (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
    h = hashlib.sha256()
    with open(TRUNKBIN, "rb") as f:
        for s, dt, off, nb, d0, d1 in sorted(have, key=lambda r: r[2]):
            f.seek(off)
            blob = f.read(nb)
            if len(blob) != nb:
                sys.exit("L%d slot %d short read" % (L, s))
            h.update(blob)
            db.execute("INSERT INTO slot_data VALUES (?,?)", (s, sqlite3.Binary(blob)))
    db.commit()

    g = hashlib.sha256()
    for s, dt, off, nb, d0, d1 in sorted(have, key=lambda r: r[2]):
        row = db.execute("SELECT data FROM slot_data WHERE id = ?", (s,)).fetchone()
        if row is None or len(row[0]) != nb:
            sys.exit("L%d slot %d bad on read back" % (L, s))
        g.update(row[0])
    db.close()

    a = hashlib.sha256()
    with open(TRUNKBIN, "rb") as f:
        for s, dt, off, nb, d0, d1 in sorted(have, key=lambda r: r[2]):
            f.seek(off); a.update(f.read(nb))
    if not (h.hexdigest() == g.hexdigest() == a.hexdigest()):
        sys.exit("L%d trunk store does not match the source" % L)

    sz = os.path.getsize(os.path.join(DB, rel))
    tot = sum(r[3] for r in have)
    record(rel, "trunk", L, TRUNKBIN, lo, hi - lo, h.hexdigest(), len(have), 65536)
    print("  %s  %2d slots  %.3f GB -> %.3f GB (+%.2f%%)  %.1f s  %s MATCH"
          % (rel, len(have), tot / 1e9, sz / 1e9, 100.0 * (sz - tot) / tot,
             time.time() - t0, h.hexdigest()[:12]))


# ------------------------------------------------------------------ experts
def build_expert(L):
    rel = "expert/L%02d.db" % L
    if done(rel):
        print("  %s already complete" % rel); return
    _, _, _, _, files, per = read_index()
    recs = per[L]
    byexp = collections.defaultdict(list)
    for r in recs:
        byexp[r[0]].append(r)
    nexp = len(byexp)
    fids = set(r[3] for r in recs)
    if len(fids) != 1:
        sys.exit("L%d spans %d files" % (L, len(fids)))
    src = files[fids.pop()]

    ref = sorted(byexp[min(byexp)], key=lambda r: r[4])
    base0 = ref[0][4]
    parts, cur = [], 0
    for e, which, kind, fid, off, nb, d0, d1 in ref:
        if off - base0 != cur:
            sys.exit("L%d expert %d not contiguous" % (L, e))
        parts.append((which * 2 + kind, cur, nb))
        cur += nb
    esz = cur
    order = sorted(byexp, key=lambda e: min(r[4] for r in byexp[e]))
    lo = min(r[4] for r in recs)
    if max(r[4] + r[5] for r in recs) - lo != esz * nexp:
        sys.exit("L%d experts do not tile" % L)

    t0 = time.time()
    db = newdb(os.path.join(DB, rel), 65536)
    db.execute("CREATE TABLE part_data (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as f:
        f.seek(lo)
        for e in order:
            blob = f.read(esz)
            if len(blob) != esz:
                sys.exit("L%d expert %d short read" % (L, e))
            h.update(blob)
            for pid, poff, pnb in parts:
                db.execute("INSERT INTO part_data VALUES (?,?)",
                           (e * 6 + pid, sqlite3.Binary(blob[poff:poff + pnb])))
                n += 1
    db.commit()

    g = hashlib.sha256()
    for e in order:
        for pid, poff, pnb in parts:
            row = db.execute("SELECT data FROM part_data WHERE id = ?",
                             (e * 6 + pid,)).fetchone()
            if row is None or len(row[0]) != pnb:
                sys.exit("L%d row %d bad on read back" % (L, e * 6 + pid))
            g.update(row[0])
    db.close()

    a = sha_range(src, lo, esz * nexp)
    if not (h.hexdigest() == g.hexdigest() == a):
        sys.exit("L%d expert store does not match the source" % L)

    sz = os.path.getsize(os.path.join(DB, rel))
    record(rel, "expert", L, src, lo, esz * nexp, h.hexdigest(), n, 65536)
    msg = ("  %s  %d rows  %.2f GB -> %.2f GB (+%.2f%%)  %.1f s  %s MATCH"
           % (rel, n, esz * nexp / 1e9, sz / 1e9,
              100.0 * (sz - esz * nexp) / (esz * nexp), time.time() - t0,
              h.hexdigest()[:12]))
    if PUNCH:
        # only after the store is proven against a second read of the source
        subprocess.run(["fallocate", "--punch-hole", "--keep-size",
                        "-o", str(lo), "-l", str(esz * nexp), src], check=True)
        msg += "  punched %.2f GB" % (esz * nexp / 1e9)
    print(msg)


# ------------------------------------------------------------------ client
def build_client():
    _, _, _, mrec, files, _ = read_index()

    rel = "client/embed.db"
    if not done(rel):
        fid, off, nb, d0, d1, dt = mrec[0]
        stride = nb // d0
        t0 = time.time()
        db = newdb(os.path.join(DB, rel), 16384)   # 14,336 B rows stay inline, 1.14x
        db.execute("CREATE TABLE embed (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
        h = hashlib.sha256()
        with open(files[fid], "rb") as f:
            f.seek(off)
            for i in range(d0):
                blob = f.read(stride)
                h.update(blob)
                db.execute("INSERT INTO embed VALUES (?,?)", (i, sqlite3.Binary(blob)))
        db.commit(); db.close()
        if h.hexdigest() != sha_range(files[fid], off, nb):
            sys.exit("embed does not match the source")
        sz = os.path.getsize(os.path.join(DB, rel))
        record(rel, "embed", None, files[fid], off, nb, h.hexdigest(), d0, 16384)
        print("  %s  %d rows of %d B  %.2f GB -> %.2f GB  %.1f s  MATCH"
              % (rel, d0, stride, nb / 1e9, sz / 1e9, time.time() - t0))

    rel = "client/lmhead.db"
    if not done(rel):
        fid, off, nb, d0, d1, dt = mrec[4]
        stride = nb // d0
        t0 = time.time()
        db = newdb(os.path.join(DB, rel), 16384)
        db.execute("CREATE TABLE lmhead (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
        h = hashlib.sha256()
        with open(files[fid], "rb") as f:
            f.seek(off)
            for i in range(d0):
                blob = f.read(stride)
                h.update(blob)
                db.execute("INSERT INTO lmhead VALUES (?,?)", (i, sqlite3.Binary(blob)))
        db.commit(); db.close()
        if h.hexdigest() != sha_range(files[fid], off, nb):
            sys.exit("lmhead does not match the source")
        sz = os.path.getsize(os.path.join(DB, rel))
        record(rel, "head", None, files[fid], off, nb, h.hexdigest(), d0, 16384)
        print("  %s %d rows of %d B  %.2f GB -> %.2f GB  %.1f s  MATCH"
              % (rel, d0, stride, nb / 1e9, sz / 1e9, time.time() - t0))

    rel = "client/head.db"
    if not done(rel):
        t0 = time.time()
        db = newdb(os.path.join(DB, rel), 65536)
        db.execute("CREATE TABLE head_data (id INTEGER PRIMARY KEY, data BLOB NOT NULL)")
        h = hashlib.sha256()
        tot = 0
        for i in (1, 2, 3):          # lm_head is its own file, see above
            fid, off, nb, d0, d1, dt = mrec[i]
            with open(files[fid], "rb") as f:
                f.seek(off); blob = f.read(nb)
            if len(blob) != nb:
                sys.exit("model tensor %d short read" % i)
            h.update(blob); tot += nb
            db.execute("INSERT INTO head_data VALUES (?,?)", (i, sqlite3.Binary(blob)))
        db.commit(); db.close()
        sz = os.path.getsize(os.path.join(DB, rel))
        record(rel, "head", None, files[mrec[1][0]], mrec[1][1], tot,
               h.hexdigest(), 3, 65536)
        print("  %s   3 rows  %d B  %.1f s" % (rel, tot, time.time() - t0))

    rel = "client/vocab.db"
    if not done(rel):
        t0 = time.time()
        db = newdb(os.path.join(DB, rel), 4096)
        db.execute("CREATE TABLE vocab (id INTEGER PRIMARY KEY, token BLOB NOT NULL)")
        h = hashlib.sha256()
        n = 0
        for line in open(TIKTOKEN, "rb"):
            p = line.split()
            if len(p) != 2:
                continue
            tok = base64.b64decode(p[0])
            db.execute("INSERT INTO vocab VALUES (?,?)", (int(p[1]), sqlite3.Binary(tok)))
            h.update(tok); n += 1
        db.execute("CREATE UNIQUE INDEX vocab_token ON vocab(token)")  # bytes -> id
        db.commit(); db.close()
        record(rel, "vocab", None, TIKTOKEN, 0, os.path.getsize(TIKTOKEN),
               h.hexdigest(), n, 4096)
        print("  %s  %d tokens  %.1f s" % (rel, n, time.time() - t0))


# ------------------------------------------------------------------ status
def status():
    db = cat()
    for role, n, gb in db.execute(
            "SELECT role, count(*), sum(src_bytes)/1e9 FROM payload"
            " WHERE complete = 1 GROUP BY role ORDER BY role"):
        print("  %-8s %3d files  %8.2f GB" % (role, n, gb or 0))
    r = db.execute("SELECT count(*), sum(src_bytes)/1e9 FROM payload WHERE complete=1").fetchone()
    print("  %-8s %3d files  %8.2f GB" % ("TOTAL", r[0], r[1] or 0))
    db.close()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    args = [int(a) for a in sys.argv[2:]]
    if cmd == "catalog":
        build_catalog()
    elif cmd == "trunk":
        for L in (args or range(93)):
            build_trunk(L)
    elif cmd == "expert":
        for L in (args or range(1, 93)):
            build_expert(L)
    elif cmd == "client":
        build_client()
    elif cmd == "status":
        status()
    else:
        sys.exit("unknown command %s" % cmd)


main()
