#!/usr/bin/env python3
"""Pull records out of a vectors dump into a flat f32 file, in the order named.

head-walk.c reads consecutive 7168 float vectors from a raw file, because that is what
tail-walk produced. The traced dumps are framed records, so feeding a dump straight to
the head reads the header as data. This writes the bodies only, in the order asked for,
so the head can be pointed at any step of the chain.

usage: pull-f32.py DUMP OUT local:pos [local:pos ...]
"""
import struct
import sys


def find(path, local, position):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(16)
            if len(header) < 16:
                return None
            _, rec_local, rec_position, count = struct.unpack("<4i", header)
            if count <= 0 or count > 1 << 22:
                return None
            body_bytes = 4 * count
            if rec_local == local and rec_position == position:
                body = handle.read(body_bytes)
                return body if len(body) == body_bytes else None
            handle.seek(body_bytes, 1)


def main():
    if len(sys.argv) < 4:
        print(__doc__.strip())
        return 1
    dump, out = sys.argv[1], sys.argv[2]
    with open(out, "wb") as sink:
        for index, item in enumerate(sys.argv[3:]):
            local, _, position = item.partition(":")
            body = find(dump, int(local), int(position or 0))
            if body is None:
                print("record %s not found" % item)
                return 1
            sink.write(body)
            print("slot %d <- local %s position %s, %d floats"
                  % (index, local, position or 0, len(body) // 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
