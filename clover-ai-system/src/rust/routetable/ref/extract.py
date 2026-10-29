#!/usr/bin/env python3
"""Pull the real route-table text out of the engine's backend-gpu.cu, byte for byte (CRLF kept), into one include file.

Three blocks are taken, in file order, and concatenated with nothing added:
  A  from the line starting "struct CallSet {" to the line starting "static std::unordered_map<int, size_t> g_use_ptr;"
     (CallSet, g_rc, g_rc_pos/g_rc_planned/g_pend_cap, g_rc_cur, g_uses, g_use_ptr, with the comments between)
  B  from the comment line starting "// next call after the current one that needs the expert" (it must be the line directly
     above) through rc_nu: from the line "static int rc_nu(int key)" to the first following line that is exactly "}"
  C  rc_load: from the line "static void rc_load(const char *path)" to the first following line that is exactly "}"
Exits 1 with a message, writing nothing, if a start marker does not match exactly one line, an end marker is missing, a block is
implausibly long, braces do not balance or close before the end line, or expected text is absent. Any previous --out is removed
first so a failed run cannot leave a stale file behind.
Usage: python extract.py --src <src/code/gpu dir> --out <file>
"""
import argparse, hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMENT = b"// next call after the current one that needs the expert"


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


def check_balance(lines, b0, b1, what):
    depth = 0
    opened = False
    for i in range(b0, b1 + 1):
        depth += lines[i].count(b"{") - lines[i].count(b"}")
        opened = opened or b"{" in lines[i]
        if depth < 0 or (opened and depth == 0 and i != b1 and what != "block A"):
            fail("braces of %s do not balance as one block (depth %d at source line %d)" % (what, depth, i + 1))
    if depth != 0:
        fail("braces of %s unbalanced at its end line (depth %d)" % (what, depth))


def function_block(lines, sig, what, max_len):
    b0 = find_one(lines, lambda l: text(l) == sig, "'%s'" % sig.decode())
    b1 = first_after(lines, b0, lambda l: text(l) == b"}", "'}' closing " + what)
    if b1 - b0 + 1 > max_len:
        fail("%s spans %d lines (expected at most %d); source layout changed?" % (what, b1 - b0 + 1, max_len))
    check_balance(lines, b0, b1, what)
    return b0, b1


def need(block, items, what):
    for n in items:
        if n not in block:
            fail("%s lacks expected text %r (source changed?)" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "code", "gpu")))
    ap.add_argument("--out", default=os.path.join(HERE, "routetable_extracted.inc"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        os.remove(a.out)
    path = os.path.join(a.src, "backend-gpu.cu")
    lines = read_lines(path)

    # block A
    a0 = find_one(lines, lambda l: l.startswith(b"struct CallSet {"), "'struct CallSet {'")
    a1 = first_after(lines, a0, lambda l: l.startswith(b"static std::unordered_map<int, size_t> g_use_ptr;"), "g_use_ptr declaration")
    if a1 - a0 + 1 > 12:
        fail("block A spans %d lines (expected 6); source layout changed?" % (a1 - a0 + 1))
    check_balance(lines, a0, a1, "block A")
    A = b"".join(lines[a0:a1 + 1])
    need(A, [b"struct CallSet { int layer; std::vector<int> experts; std::vector<int> counts; };", b"static std::vector<CallSet> g_rc;",
             b"static long g_rc_cur = -1;", b"static std::unordered_map<int, std::vector<int>> g_uses;"], "block A")

    # block B
    b0, b1 = function_block(lines, b"static int rc_nu(int key)", "rc_nu", 20)
    if not lines[b0 - 1].startswith(COMMENT):
        fail("the line above 'static int rc_nu(int key)' is not the expected comment (source line %d)" % b0)
    b0 -= 1
    B = b"".join(lines[b0:b1 + 1])
    need(B, [b"g_uses.find(key)", b"g_use_ptr[key]", b"l[p] <= g_rc_cur", b"return 1000000000;"], "block B")

    # block C
    c0, c1 = function_block(lines, b"static void rc_load(const char *path)", "rc_load", 40)
    C = b"".join(lines[c0:c1 + 1])
    need(C, [b"getline(&line, &cap, f)", b"strtol(p, &end, 10)", b"ekey(g_rc[i].layer, e)", b"cannot read route cache %s\\n"], "block C")

    if not (a0 < b0 and b1 < c0):
        fail("blocks are not in the expected file order A < B < C")

    data = A + B + C
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
    print("block A backend-gpu.cu lines %d-%d (%d bytes) sha256 %s" % (a0 + 1, a1 + 1, len(A), h(A)))
    print("block B backend-gpu.cu lines %d-%d (%d bytes) sha256 %s" % (b0 + 1, b1 + 1, len(B), h(B)))
    print("block C backend-gpu.cu lines %d-%d (%d bytes) sha256 %s" % (c0 + 1, c1 + 1, len(C), h(C)))
    print("wrote %s (%d bytes, CRLF lines %d of %d)" % (a.out, len(data), data.count(b"\r\n"), data.count(b"\n")))
    print("extracted text sha256 %s" % h(data))


if __name__ == "__main__":
    main()
