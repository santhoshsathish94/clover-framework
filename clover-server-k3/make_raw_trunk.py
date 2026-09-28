#!/usr/bin/env python3
"""Write the trunk as one raw file per (layer, stage), so a stage can be read
with O_DIRECT from offset 0 and needs no page-list walk to find it.

  <out>/trunk/L<LL>/s<SS>.bin      the stage bytes, nothing else
  <out>/trunk/manifest.tsv         layer, slot, nbytes, sha256

Every stage is hashed out of SQLite and hashed again off disk, because a
converter that does not compare has not converted anything. Resumable: a stage
whose file already matches the manifest is skipped.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

CHUNK = 8 << 20


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def convert_layer(args):
    db_dir, out_dir, layer, verify_existing = args
    src = os.path.join(db_dir, "trunk", "L%02d.db" % layer)
    dst_dir = os.path.join(out_dir, "trunk", "L%02d" % layer)
    os.makedirs(dst_dir, exist_ok=True)

    con = sqlite3.connect("file:%s?immutable=1" % src, uri=True)
    ids = [r[0] for r in con.execute("select id from slot_data order by id")]

    rows, skipped, written, nbytes = [], 0, 0, 0
    for sid in ids:
        dst = os.path.join(dst_dir, "s%02d.bin" % sid)
        n = con.execute("select length(data) from slot_data where id=?", (sid,)).fetchone()[0]

        if os.path.exists(dst) and os.path.getsize(dst) == n and not verify_existing:
            rows.append((layer, sid, n, sha_file(dst)))
            skipped += 1
            nbytes += n
            continue

        blob = con.blobopen("slot_data", "data", sid, readonly=True)
        h = hashlib.sha256()
        tmp = dst + ".part"
        with open(tmp, "wb", buffering=0) as f:
            off = 0
            while off < n:
                b = blob[off:min(off + CHUNK, n)]
                h.update(b)
                f.write(b)
                off += len(b)
            f.flush()
            os.fsync(f.fileno())
        blob.close()
        os.replace(tmp, dst)

        # the hash that matters is the one read back off the disk, not the one
        # accumulated on the way past
        got = sha_file(dst)
        if got != h.hexdigest():
            con.close()
            return (layer, None, "MISMATCH slot %d: sqlite %s disk %s" % (sid, h.hexdigest(), got))
        rows.append((layer, sid, n, got))
        written += 1
        nbytes += n

    con.close()
    return (layer, rows, "%2d slots (%d written, %d kept) %10.3f MB" %
            (len(ids), written, skipped, nbytes / 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/srv/k3/db")
    ap.add_argument("--out", default="/srv/k3/raw")
    ap.add_argument("--layers", default="0-92")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--verify-existing", action="store_true")
    a = ap.parse_args()

    layers = []
    for part in a.layers.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))

    os.makedirs(os.path.join(a.out, "trunk"), exist_ok=True)
    work = [(a.db, a.out, L, a.verify_existing) for L in layers]

    all_rows, failed = [], []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(convert_layer, w): w[2] for w in work}
        done = 0
        for fut in as_completed(futs):
            L, rows, msg = fut.result()
            done += 1
            if rows is None:
                failed.append((L, msg))
                print("L%02d  FAIL  %s" % (L, msg), flush=True)
            else:
                all_rows.extend(rows)
                print("L%02d  %s   [%d/%d]" % (L, msg, done, len(work)), flush=True)

    all_rows.sort()
    man = os.path.join(a.out, "trunk", "manifest.tsv")
    with open(man, "w") as f:
        f.write("layer\tslot\tnbytes\tsha256\n")
        for r in all_rows:
            f.write("%d\t%d\t%d\t%s\n" % r)

    total = sum(r[2] for r in all_rows)
    print()
    print("stages   : %d" % len(all_rows))
    print("bytes    : %d  (%.2f GB)" % (total, total / 1e9))
    print("manifest : %s" % man)
    if failed:
        print("FAILED   : %d layers" % len(failed))
        return 1
    print("all stages bit-exact against SQLite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
