#!/usr/bin/env python3
"""Pull the real supply text (pool_ensure and what only it needs) out of the engine as COMMITTED at 4d8d891, byte for byte, into two include files.

The source is `git show 4d8d891:clover-kimi-k3/src/code/gpu/<file>` (the working tree is being edited by someone else and is never read for the text). The committed blobs have
LF line endings (0 CRLF); a Windows checkout with autocrlf shows CRLF, so a working-tree hash differs from the blob hash by design. Each blob's size, line count and sha256 are checked
against the constants below before anything is cut; a different blob exits 1.

supply_types.inc (decode.cuh, file order, nothing added):
  T1  the line starting "struct ExpertStats {"              (the per-pool counters; sy_stats of supply.h has this layout)
  T2  the line "enum { MAX_READERS = 32 };"
supply_extracted.inc (decode.cuh, file order, nothing added):
  X1  run_parallel: the comment line starting "// runs the jobs on up to nthreads threads" through the first following line exactly "}"
  X2  the line starting "static double wall_now() {"
  X3  the line starting "static unsigned long g_wb_skipped = 0;"
  X4  pool_ensure: the comment lines directly above "static void pool_ensure(" (starting at "// makes the given experts of `layer` resident") through the first following line exactly "}"
Checked, NOT taken (the harness adapts them; the text is recorded so the ADAPTED notes stay anchored): kernels.cuh CHECK macro, moe.cuh enum with EXPERT_RAW and EXPERTS.
Exits 1 with a message, writing nothing, if a blob differs, a start marker does not match exactly one line, an end marker is missing, a block is implausibly long, braces (counted per
character; comments and strings are not parsed) do not balance or close early, expected text is missing, blocks overlap or are out of file order, or the independent line-range
slice (EXPECTED_RANGES) differs. Any previous output is removed first so a failed run cannot leave a stale file.
Usage: python extract.py [--repo DIR] [--commit 4d8d891] [--outdir DIR]
"""
import argparse, hashlib, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))   # .../clover-framework (the git root)
PREFIX = "clover-kimi-k3/src/code/gpu/"

# (bytes, lines, sha256) of the committed blobs at 4d8d891
BLOBS = {
    "decode.cuh": (45669, 779, "fc7682e9bc6a043af4ebe59b1a7cb27f750ff7e6cde4dcb8beeaed5294955dee"),
    "kernels.cuh": (29623, 533, "0ce0a33edd8ffde2a0f2fc3dbff8b2028cae07ba210874660a123271dde6277a"),
    "moe.cuh": (24905, 444, "4af5c825ebadaf33c15f8d13be7f1a45f4e800b92e476f5de1cd2bfdb271d28f"),
}
# independent second slice: 1-based inclusive line ranges of decode.cuh found when this script was written (checked against the markers' result)
EXPECTED_RANGES = {"T1": (437, 437), "T2": (439, 439), "X1": (473, 484), "X2": (487, 487), "X3": (495, 495), "X4": (536, 620)}


def fail(msg):
    sys.stderr.write("extract.py: FAIL: " + msg + "\n")
    sys.exit(1)


def git_blob(repo, commit, name):
    p = subprocess.run(["git", "-C", repo, "show", "%s:%s%s" % (commit, PREFIX, name)], capture_output=True)
    if p.returncode != 0:
        fail("git show %s:%s%s failed: %s" % (commit, PREFIX, name, p.stderr.decode("utf-8", "replace").strip()))
    return p.stdout


def dir_file(src, name):
    path = os.path.join(src, name)
    if not os.path.isfile(path):
        fail("source file not found: " + path)
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")   # a checkout may have CRLF; the committed text is LF


def split_lines(data):
    parts = data.split(b"\n")
    lines = [x + b"\n" for x in parts[:-1]]
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


def starts(prefix):
    return lambda l: l.startswith(prefix)


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
        if depth < 0 or (opened and depth == 0 and i != b1):
            fail("braces of %s do not balance as one block (depth %d at source line %d)" % (what, depth, i + 1))
    if depth != 0:
        fail("braces of %s unbalanced at its end line (depth %d)" % (what, depth))


def need(block, items, what):
    for n in items:
        if n not in block:
            fail("%s lacks expected text %r (source changed?)" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--commit", default="4d8d891")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--print-ranges", action="store_true", help="print the line ranges found (to fill EXPECTED_RANGES) and exit")
    ap.add_argument("--src", default=None, help="read decode.cuh, kernels.cuh, moe.cuh from this directory (CRLF read as LF) instead of git: to check a working tree or, with the switch below, a control")
    ap.add_argument("--control-skip-blob-check", action="store_true", help="negative controls only: do not compare the files with the committed blobs' hashes")
    a = ap.parse_args()

    out_types = os.path.join(a.outdir, "supply_types.inc")
    out_ext = os.path.join(a.outdir, "supply_extracted.inc")
    for p in (out_types, out_ext):
        if os.path.exists(p):
            os.remove(p)

    h = lambda b: hashlib.sha256(b).hexdigest()
    blobs = {}
    for name, (size, nlines, sha) in BLOBS.items():
        d = dir_file(a.src, name) if a.src else git_blob(a.repo, a.commit, name)
        if not a.control_skip_blob_check and (len(d) != size or d.count(b"\n") != nlines or h(d) != sha):
            fail("blob %s at %s differs from the one this script was written for (size %d lines %d sha256 %s)" % (name, a.src or a.commit, len(d), d.count(b"\n"), h(d)))
        blobs[name] = d
    dec = split_lines(blobs["decode.cuh"])

    blocks = []   # (name, first index, last index, bytes)

    def take(name, b0, b1, items, balance=True):
        if balance:
            check_balance(dec, b0, b1, name)
        data = b"".join(dec[b0:b1 + 1])
        need(data, items, name)
        blocks.append((name, b0, b1, data))

    t1 = find_one(dec, starts(b"struct ExpertStats {"), "'struct ExpertStats {'")
    take("T1", t1, t1, [b"unsigned long needed = 0, hit_card = 0, hit_host = 0, from_drive = 0, written_back = 0;", b"double t_read = 0, t_copy = 0, t_ensure = 0;"], balance=False)
    t2 = find_one(dec, lambda l: text(l) == b"enum { MAX_READERS = 32 };", "'enum { MAX_READERS = 32 };'")
    take("T2", t2, t2, [], balance=False)

    xs = find_one(dec, starts(b"static void run_parallel(std::vector<std::function<void(int)>> &jobs, int nthreads)"), "run_parallel signature")
    x1a = xs - 1
    if not dec[x1a].startswith(b"// runs the jobs on up to nthreads threads"):
        fail("the comment line above run_parallel is not the expected one")
    x1b = first_after(dec, xs, lambda l: text(l) == b"}", "'}' closing run_parallel")
    if x1b - x1a + 1 > 20:
        fail("run_parallel spans %d lines (expected 12)" % (x1b - x1a + 1))
    take("X1", x1a, x1b, [b"std::atomic<size_t> next(0);", b"th.emplace_back(work, i);", b"work(0);", b"for (auto &t : th) t.join();"])

    x2 = find_one(dec, starts(b"static double wall_now() {"), "wall_now line")
    take("X2", x2, x2, [b"CLOCK_MONOTONIC"], balance=False)
    x3 = find_one(dec, starts(b"static unsigned long g_wb_skipped = 0;"), "g_wb_skipped line")
    take("X3", x3, x3, [], balance=False)

    ps = find_one(dec, starts(b"static void pool_ensure(CardPool &pool, HostCache &hc, int layer, int fd, const std::vector<int> &experts, const int *counts, int readers)"), "pool_ensure signature")
    x4a = ps
    while x4a > 0 and dec[x4a - 1].startswith(b"//"):
        x4a -= 1
    if not dec[x4a].startswith(b"// makes the given experts of `layer` resident in the card's pool"):
        fail("the comment block above pool_ensure does not start as expected")
    x4b = first_after(dec, ps, lambda l: text(l) == b"}", "'}' closing pool_ensure")
    if x4b - x4a + 1 > 100:
        fail("pool_ensure block spans %d lines (expected 85)" % (x4b - x4a + 1))
    take("X4", x4a, x4b, [
        b"if (g_vq) CHECK(cudaStreamSynchronize(0)); else CHECK(cudaDeviceSynchronize());",
        b"unsigned char *p = pool.c.find(key);",
        b"if (hc.c.find(key)) { pend_take(key); n.host = hc.c.remove(key); held.push_back(n.host); pool.st.hit_host++; }",
        b"if (!hc.cap_slots || g_prefill) break;",
        b"if (!hc.c.worth(v.key)) { g_wb_skipped++; continue; }",
        b"unsigned char *h = hc.c.acquire(v.key, g_pend_keys, v.heat, &dk, &dh);",
        b"CHECK(cudaEventRecord(pool.written_back, pool.copy));",
        b"int nchunk = g_chunks > 0 ? g_chunks : (n_drive >= pool_readers || n_drive == 0) ? 1 : std::min(8, (pool_readers + n_drive - 1) / n_drive);",
        b"const size_t chunk = ((((size_t)EXPERT_RAW + nchunk - 1) / nchunk) + 4095) / 4096 * 4096;",
        b"reading expert %d of layer %d failed: %s\\n",
        b"CHECK(cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0));",
        b"run_parallel(jobs, pool_readers);",
        b"for (unsigned char *h : held) hc.c.release(h);",
    ])

    # order of blocks in the file
    for x, y in zip(blocks, blocks[1:]):
        if not x[2] < y[1]:
            fail("blocks %s and %s overlap or are out of file order" % (x[0], y[0]))

    if a.print_ranges:
        for n, b0, b1, d in blocks:
            print(n, b0 + 1, b1 + 1)
        return

    # independent second slice by line ranges
    for n, b0, b1, d in blocks:
        lo, hi = EXPECTED_RANGES[n]
        if lo is None:
            fail("EXPECTED_RANGES has no entry for %s" % n)
        if (b0 + 1, b1 + 1) != (lo, hi):
            fail("block %s found at lines %d-%d, expected %d-%d" % (n, b0 + 1, b1 + 1, lo, hi))
        if b"".join(dec[lo - 1:hi]) != d:
            fail("independent slice of block %s differs" % n)

    # checked-not-taken text of the two adaptation sources
    kern = split_lines(blobs["kernels.cuh"])
    k = find_one(kern, starts(b"#define CHECK(x) do { cudaError_t e_ = (x);"), "CHECK macro")
    ktxt = text(kern[k])
    if ktxt != b'#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)':
        fail("kernels.cuh CHECK macro text differs from the one the harness was adapted from")
    moe = split_lines(blobs["moe.cuh"])
    m = find_one(moe, starts(b"enum { LATENT = 3584, EXPERT = 3072, SHARED = 6144, EXPERT_RAW = 17547264,"), "moe.cuh enum line")
    if b"EXPERTS = 896" not in moe[m]:
        fail("moe.cuh enum does not contain EXPERTS = 896")

    types = b"".join(b[3] for b in blocks if b[0].startswith("T"))
    ext = b"".join(b[3] for b in blocks if b[0].startswith("X"))
    for p, d in ((out_types, types), (out_ext, ext)):
        with open(p, "wb") as f:
            f.write(d)
        with open(p, "rb") as f:
            if f.read() != d:
                fail("self-check: written file %s differs from extracted bytes" % p)
    for name, (size, nlines, sha) in BLOBS.items():
        print("source  %s%s @ %s: %d bytes, %d lines, sha256 %s" % (PREFIX, name, a.commit, size, nlines, sha))
    for n, b0, b1, d in blocks:
        print("block %s decode.cuh lines %d-%d (%d bytes) sha256 %s" % (n, b0 + 1, b1 + 1, len(d), h(d)))
    print("checked, not taken: kernels.cuh:%d CHECK macro; moe.cuh:%d enum (EXPERT_RAW, EXPERTS)" % (k + 1, m + 1))
    print("wrote %s (%d bytes, CRLF lines %d of %d) sha256 %s" % (out_types, len(types), types.count(b"\r\n"), types.count(b"\n"), h(types)))
    print("wrote %s (%d bytes, CRLF lines %d of %d) sha256 %s" % (out_ext, len(ext), ext.count(b"\r\n"), ext.count(b"\n"), h(ext)))


if __name__ == "__main__":
    main()
