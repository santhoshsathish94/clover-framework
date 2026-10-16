#!/usr/bin/env python3
"""Verify the 4-bit dataset is complete and actually differs, before running it.

The build wrote 7.8 GB but produced no log, so its own account of itself is
missing. Checking the artifact is the only honest way to know it finished: the
modified layers must exist as real directories with group files byte-identical
in SIZE to the originals (the container is unchanged) but different in CONTENT
(the weights are not). A file that matches in both is a layer that was never
rewritten, and one that differs in size would break the index.

Also confirms the untouched layers are still symlinks to the original dataset,
so a partial build cannot quietly masquerade as a full one.

usage: verify-fourbit.py DIR [layer ...]
"""
import hashlib
import os
import sys

SRC = "/opt/clover-k3/clover-intelegence/dataset"


def digest(path, limit=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ds4bit"
    want = [int(a) for a in sys.argv[2:]] or list(range(0, 93, 6))

    modified, unchanged, missing, badsize = [], [], [], []
    for layer in range(93):
        d = os.path.join(out, "trunk-%d" % layer)
        s = os.path.join(SRC, "trunk-%d" % layer)
        if not os.path.exists(d):
            missing.append(layer)
            continue
        is_real = os.path.isdir(d) and not os.path.islink(d)
        if layer in want and not is_real:
            missing.append(layer)
            continue
        if layer not in want:
            if is_real:
                badsize.append(("unexpected real dir", layer))
            continue
        names = ["common.bin", "attention.bin" if layer else "kda.bin",
                 "routed-shared.bin" if layer else "dense.bin"]
        diff = False
        for nm in names:
            dp, sp = os.path.join(d, nm), os.path.join(s, nm)
            if not os.path.exists(dp) or not os.path.exists(sp):
                missing.append(layer)
                diff = None
                break
            if os.path.getsize(dp) != os.path.getsize(sp):
                badsize.append((layer, nm, os.path.getsize(dp), os.path.getsize(sp)))
                diff = None
                break
            if digest(dp) != digest(sp):
                diff = True
        if diff is True:
            modified.append(layer)
        elif diff is False:
            unchanged.append(layer)

    print("requested %d layers: %s" % (len(want), want))
    print("MODIFIED  %d: %s" % (len(modified), modified))
    if unchanged:
        print("UNCHANGED %d: %s   <- rewritten with identical bytes, or skipped"
              % (len(unchanged), unchanged))
    if missing:
        print("MISSING   %d: %s" % (len(missing), sorted(set(missing))))
    if badsize:
        print("SIZE/SHAPE PROBLEMS: %s" % badsize[:5])

    links = sum(1 for l in range(93)
                if os.path.islink(os.path.join(out, "trunk-%d" % l)))
    print("untouched layers still symlinked: %d" % links)

    ok = len(modified) == len(want) and not missing and not badsize
    print("\n%s" % ("COMPLETE - safe to run" if ok
                    else "INCOMPLETE - do not draw conclusions from a run on this"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
