#!/usr/bin/env python3
"""Writes a copy of the trunk folders without the records the engine never reads.

usage: prune-trunk.py SOURCE_DATASET OUTPUT_DIR [LAYER ...]      (no layer: 1 to 92)

For every trunk-N it writes index.bin, common.bin, attention.bin and routed-shared.bin to OUTPUT_DIR/trunk-N. A record the layer reads
(the transformer_needs list of src/code/layer-kda.c and src/code/layer-mla.c) is copied byte for byte; every other record stays in the index
with length 0. The Q, K and V weights are among the dropped ones: trunk-N/operator.bin holds them (it is hard-linked into the output). A group file with nothing dropped is
hard-linked instead of copied. The source is only read. trunk-0 (layer 0) is not touched here.
"""
import hashlib, os, struct, sys

KDA_NEEDS = {5, 6, 7, 11, 12, 13, 18, 19} | set(range(29, 40))
MLA_NEEDS = {5, 6, 7} | set(range(29, 40))
GROUPS = ("common.bin", "attention.bin", "routed-shared.bin")
HEADER, RECORD = 1168, 72

def prune(src, dst, layer):
    index = bytearray(open(os.path.join(src, "index.bin"), "rb").read())
    if index[:8] != b"K3TRK001" or struct.unpack_from("<I", index, 12)[0] != layer:
        raise SystemExit(f"trunk-{layer}: not a trunk index for this layer")
    count = struct.unpack_from("<I", index, 16)[0]
    if len(index) != HEADER + count * RECORD:
        raise SystemExit(f"trunk-{layer}: index size does not match its record count")
    needs = {31: KDA_NEEDS, 24: MLA_NEEDS}.get(count)
    if needs is None:
        raise SystemExit(f"trunk-{layer}: {count} records, expected 31 (KDA) or 24 (MLA)")
    lengths_old = struct.unpack_from("<3Q", index, 24)
    records = []
    for position in range(count):
        base = HEADER + position * RECORD
        rid, group, kind, rows, cols, n = struct.unpack_from("<6I", index, base)
        offset, length = struct.unpack_from("<2Q", index, base + 24)
        records.append([base, rid, group, offset, length])
    os.makedirs(dst, exist_ok=True)
    lengths_new, dropped = [], 0
    for group, name in enumerate(GROUPS):
        mine = [r for r in records if r[2] == group]
        cursor = 0
        if any(r[1] not in needs for r in mine):
            out_path = os.path.join(dst, name)
            with open(os.path.join(src, name), "rb") as source, open(out_path + ".tmp", "wb") as out:
                if os.fstat(source.fileno()).st_size != lengths_old[group]:
                    raise SystemExit(f"trunk-{layer}/{name}: size differs from the index")
                for r in mine:
                    base, rid, _, offset, length = r
                    if r[3] != cursor_old(mine, r):
                        raise SystemExit(f"trunk-{layer}/{name}: records are not contiguous")
                    keep = rid in needs
                    if keep:
                        source.seek(offset)
                        data = source.read(length)
                        if len(data) != length:
                            raise SystemExit(f"trunk-{layer}/{name}: short read")
                        out.write(data)
                    else:
                        dropped += length
                    new_length = length if keep else 0
                    struct.pack_into("<2Q", index, base + 24, cursor, new_length)
                    cursor += new_length
            os.replace(out_path + ".tmp", out_path)
            if cursor == 0:   # nothing left of this group: no file (the engine does not open an empty group)
                os.remove(out_path)
        else:
            for r in mine:
                cursor += r[4]
            target = os.path.join(dst, name)
            if os.path.exists(target):
                os.remove(target)
            os.link(os.path.join(src, name), target)
        lengths_new.append(cursor)
    struct.pack_into("<3Q", index, 24, *lengths_new)
    operator = os.path.join(src, "operator.bin")
    if os.path.exists(operator):
        target = os.path.join(dst, "operator.bin")
        if os.path.exists(target):
            os.remove(target)
        os.link(operator, target)
    with open(os.path.join(dst, "index.bin"), "wb") as out:
        out.write(index)
    return dropped, sum(lengths_old), sum(lengths_new)

def cursor_old(mine, record):
    position = 0
    for r in mine:
        if r is record:
            return position
        position += r[4]

def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    source, output = sys.argv[1], sys.argv[2]
    layers = [int(x) for x in sys.argv[3:]] or list(range(1, 93))
    total_old = total_new = 0
    for layer in layers:
        dropped, old, new = prune(os.path.join(source, f"trunk-{layer}"), os.path.join(output, f"trunk-{layer}"), layer)
        total_old += old
        total_new += new
        print(f"trunk-{layer}: dropped {dropped} bytes, {old} -> {new}", flush=True)
    print(f"{len(layers)} layers: {total_old} -> {total_new} bytes ({total_old / 2**30:.2f} -> {total_new / 2**30:.2f} GiB)")

if __name__ == "__main__":
    main()
