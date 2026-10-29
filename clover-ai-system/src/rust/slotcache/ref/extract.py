#!/usr/bin/env python3
"""Pull the real SlotCache text out of the engine's decode.cuh, byte for byte (CRLF kept), into one include file.

Two blocks are taken, in file order, and concatenated with nothing added:
  A  from the line starting "// Slots with replacement of the least recently used entry" to the line starting
     "static int (*g_nu_fn)(int key) = nullptr;"  (the comments, g_policy / g_chunks and g_nu_fn)
  B  from the line "struct SlotCache {" to the first following line that is exactly "};"
Exits 1 with a message, writing nothing, if a start marker does not match exactly one line, an end marker is missing, the
block is implausibly long, the braces do not balance or close before the end line, or expected members are absent.
Any previous --out is removed first so a failed run cannot leave a stale file behind.
Usage: python extract.py --src <src/code/gpu dir> --out <file>
"""
import argparse, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def fail(msg):
    sys.stderr.write("extract.py: FAIL: " + msg + "\n")
    sys.exit(1)


def read_lines(path):
    if not os.path.isfile(path):
        fail("source file not found: " + path)
    with open(path, "rb") as f:
        parts = f.read().split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def text(l):
    return l.rstrip(b"\r\n")


def find_one(lines, pred, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    if len(hits) != 1:
        fail("expected exactly one line matching %s, found %d" % (what, len(hits)))
    return hits[0]


def first_after(lines, start, pred, what):
    for i in range(start, len(lines)):
        if pred(lines[i]):
            return i
    fail("end marker not found after line %d: %s" % (start + 1, what))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "code", "gpu")))
    ap.add_argument("--out", default=os.path.join(HERE, "slotcache_extracted.inc"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        os.remove(a.out)
    path = os.path.join(a.src, "decode.cuh")
    lines = read_lines(path)

    # block A
    a0 = find_one(lines, lambda l: l.startswith(b"// Slots with replacement of the least recently used entry"), "block A start comment")
    a1 = first_after(lines, a0, lambda l: l.startswith(b"static int (*g_nu_fn)(int key) = nullptr;"), "g_nu_fn declaration")
    if a1 - a0 > 20:
        fail("block A spans %d lines (expected about 7); source layout changed?" % (a1 - a0 + 1))
    A = b"".join(lines[a0:a1 + 1])
    if sum(1 for l in lines[a0:a1 + 1] if l.startswith(b"static int g_policy = ")) != 1:
        fail("block A does not contain exactly one 'static int g_policy = ' line")

    # block B
    b0 = find_one(lines, lambda l: l.startswith(b"struct SlotCache {"), "'struct SlotCache {'")
    b1 = first_after(lines, b0, lambda l: text(l) == b"};", "'};'")
    if b1 - b0 > 150:
        fail("block B spans %d lines (expected about 78); source layout changed?" % (b1 - b0 + 1))
    depth = 0
    for i in range(b0, b1 + 1):
        depth += lines[i].count(b"{") - lines[i].count(b"}")
        if depth < 0 or (depth == 0 and i != b1):
            fail("braces of struct SlotCache do not balance as one block (depth %d at source line %d)" % (depth, i + 1))
    if depth != 0:
        fail("braces of struct SlotCache unbalanced at its end line (depth %d)" % depth)
    B = b"".join(lines[b0:b1 + 1])
    for need in (b"std::list<int> lru;", b"std::unordered_map<int, Ent> map;", b"std::set<std::pair<int, int>> by_next;",
                 b"std::vector<unsigned char *> spare;", b"unsigned char *find(int key)", b"void touch(int key, float w)",
                 b"bool worth(int key) const", b"unsigned char *acquire(int key,", b"unsigned char *try_acquire(int key,",
                 b"unsigned char *remove(int key)", b"void release(unsigned char *p)", b"bool belady() const"):
        if need not in B:
            fail("struct SlotCache block lacks expected text %r (source changed?)" % need)

    data = A + B
    with open(a.out, "wb") as f:
        f.write(data)
    with open(a.out, "rb") as f:
        back = f.read()
    if back != data:
        fail("self-check: written file differs from extracted bytes")
    h = lambda b: hashlib.sha256(b).hexdigest()
    with open(path, "rb") as f:
        src_all = f.read()
    print("source  %s sha256 %s" % (path, h(src_all)))
    print("block A decode.cuh lines %d-%d (%d bytes) sha256 %s" % (a0 + 1, a1 + 1, len(A), h(A)))
    print("block B decode.cuh lines %d-%d (%d bytes) sha256 %s" % (b0 + 1, b1 + 1, len(B), h(B)))
    print("wrote %s (%d bytes, CRLF lines %d of %d)" % (a.out, len(data), data.count(b"\r\n"), data.count(b"\n")))
    print("extracted text sha256 %s" % h(data))


if __name__ == "__main__":
    main()
