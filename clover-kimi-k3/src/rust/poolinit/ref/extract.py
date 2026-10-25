#!/usr/bin/env python3
"""Pull the real pool start-up text out of the engine's decode.cuh AS COMMITTED (git blob of a revision, default 4d8d891), byte for byte, into include files.

The source of truth is the committed blob, not the working tree: the working-tree files are edited by other people, and with core.autocrlf=true a Windows checkout has CRLF while the
blob has LF (git ls-files --eol: i/lf w/crlf). The blob bytes are what is extracted and hashed here (LF). The hash of the same text with CRLF line ends (what a Windows checkout
holds) is printed too, and the working tree is compared only as information (never a failure).

Two blocks of decode.cuh are written, nothing added, nothing changed:
  D0  the line starting "static double wall_now_()" (one line, the clock the original init uses)                       -> poolinit_wall.inc
  D1  from the line starting "struct HostCache {" through the first line exactly "};" after the line starting "struct CardPool {"
      (struct HostCache with init and CHUNK_SLOTS, struct ExpertStats, enum MAX_READERS, struct CardPool with init)     -> poolinit_extracted.inc
Because D1 is contiguous, a "#line 403 "decode.cuh"" in front of its include makes __LINE__ of every CHECK in it the line of the original file.
Also read, NOT written (they are replaced by the harness; the checks prove the replaced text is what the harness assumes):
  X1  kernels.cuh   the CHECK macro (exact text)
  X2  moe.cuh       the enum line with EXPERT_RAW and EXPERTS (the real constants)
The line numbers of the ten CHECK( calls inside D1 are asserted (they are the numbers the Rust crate and the header carry).
Exits 1 with a message, writing nothing, if a marker matches 0 or >1 lines, an end marker is missing, a block is implausibly long, braces do not balance (counted per character;
comments and strings are not parsed), expected text is absent, a CHECK site moved, or the macro / constants differ. Any previous output is removed first.
Usage: python extract.py [--repo DIR] [--rev REV] [--prefix PATH_IN_REPO] --outdir DIR
"""
import argparse, hashlib, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_CHECK_LINES = [428, 429, 454, 455, 459, 461, 463, 464, 466, 467]   # decode.cuh@4d8d891
EXPECTED_CHECK_MACRO = (b'#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)')


def fail(msg):
    sys.stderr.write("extract.py: FAIL: " + msg + "\n")
    sys.exit(1)


def git_show(repo, rev, path):
    r = subprocess.run(["git", "-C", repo, "show", "%s:%s" % (rev, path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        fail("git show %s:%s failed: %s" % (rev, path, r.stderr.decode("utf-8", "replace").strip()))
    return r.stdout


def git_out(repo, *args):
    r = subprocess.run(["git", "-C", repo] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        fail("git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    return r.stdout.decode().strip()


def split_lines(data):
    parts = data.split(b"\n")
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
    for i in range(b0, b1 + 1):
        depth += lines[i].count(b"{") - lines[i].count(b"}")
        if depth < 0:
            fail("braces of %s go negative at source line %d" % (what, i + 1))
    if depth != 0:
        fail("braces of %s unbalanced at the end line (depth %d)" % (what, depth))


def need(block, items, what):
    for n in items:
        if n not in block:
            fail("%s lacks expected text %r (source changed?)" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--rev", default="4d8d891")
    ap.add_argument("--prefix", default="clover-kimi-k3/src/code/gpu")
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()

    repo = a.repo or git_out(HERE, "rev-parse", "--show-toplevel")
    out_wall = os.path.join(a.outdir, "poolinit_wall.inc")
    out_main = os.path.join(a.outdir, "poolinit_extracted.inc")
    for p in (out_wall, out_main):
        if os.path.exists(p):
            os.remove(p)
    commit = git_out(repo, "rev-parse", a.rev + "^{commit}")
    blob_id = lambda f: git_out(repo, "rev-parse", "%s:%s/%s" % (a.rev, a.prefix, f))
    dec_raw = git_show(repo, a.rev, a.prefix + "/decode.cuh")
    ker_raw = git_show(repo, a.rev, a.prefix + "/kernels.cuh")
    moe_raw = git_show(repo, a.rev, a.prefix + "/moe.cuh")
    if b"\r" in dec_raw:
        fail("the committed decode.cuh contains CR characters; this script assumes an LF blob (recheck git ls-files --eol)")
    dec, ker, moe = split_lines(dec_raw), split_lines(ker_raw), split_lines(moe_raw)

    # D0
    d0 = find_one(dec, lambda l: l.startswith(b"static double wall_now_()"), "'static double wall_now_()'")
    D0 = dec[d0]
    need(D0, [b"clock_gettime(CLOCK_MONOTONIC", b"return t.tv_sec + 1e-9 * t.tv_nsec;"], "D0")

    # D1
    d1a = find_one(dec, lambda l: l.startswith(b"struct HostCache {"), "'struct HostCache {'")
    d1p = find_one(dec, lambda l: l.startswith(b"struct CardPool {"), "'struct CardPool {'")
    if not d1a < d1p or d1p - d1a > 60:
        fail("struct HostCache and struct CardPool are not close together in file order (lines %d, %d)" % (d1a + 1, d1p + 1))
    d1b = first_after(dec, d1p, lambda l: text(l) == b"};", "'};' closing struct CardPool")
    if d1b - d1a + 1 > 90:
        fail("block D1 spans %d lines (expected 69); source layout changed?" % (d1b - d1a + 1))
    check_balance(dec, d1a, d1b, "D1")
    D1 = b"".join(dec[d1a:d1b + 1])
    need(D1, [b"enum { CHUNK_SLOTS = 32 };", b"struct ExpertStats { unsigned long needed = 0, hit_card = 0, hit_host = 0, from_drive = 0, written_back = 0; double t_read = 0, t_copy = 0, t_ensure = 0; };",
              b"enum { MAX_READERS = 32 };", b"cap_slots = bytes / EXPERT_RAW; ready = true;", b"std::atomic<size_t> next(0);", b"for (int t = 1; t < 16; t++) th.emplace_back(touch);",
              b"madvise(p, chunk_bytes, MADV_HUGEPAGE);", b"memset(p, 0, chunk_bytes);", b"for (size_t i = n_chunks; i-- > 0;)", b"for (int s = CHUNK_SLOTS; s-- > 0;) c.add_spare(base[i] + (size_t)s * EXPERT_RAW);",
              b"cap_slots = n_chunks * CHUNK_SLOTS;", b'"K3GPU: host expert cache of %zu slots (%.1f GB) pinned in %.1f s\\n"', b'"host expert cache: cannot map %zu bytes\\n"',
              b"dev = device; readers = std::min(nreaders, (int)MAX_READERS);", b"slots = (int)(usable / EXPERT_RAW);", b'"card %d: only %d expert slots fit after the reserve\\n"',
              b"for (int i = slots; i-- > 0;) c.add_spare(base + (size_t)i * EXPERT_RAW);", b"CHECK(cudaMalloc(&d_eptr, EXPERTS * sizeof(void *)));", b"h_eptr.assign(EXPERTS, nullptr);"], "D1")
    n_semi = sum(1 for l in dec[d1a:d1b + 1] if text(l) == b"};")
    if n_semi != 2:
        fail("block D1 has %d lines exactly '};' (expected 2: HostCache, CardPool)" % n_semi)
    sites = [i + 1 for i in range(d1a, d1b + 1) if b"CHECK(" in dec[i]]
    if sites != EXPECTED_CHECK_LINES:
        fail("CHECK sites are at lines %s, the header and the Rust crate carry %s" % (sites, EXPECTED_CHECK_LINES))
    if any(dec[i - 1].count(b"CHECK(") != 1 for i in sites):
        fail("a CHECK line holds more than one CHECK")

    # X1, X2
    x1 = [text(l) for l in ker if l.startswith(b"#define CHECK(x)")]
    if len(x1) != 1 or x1[0] != EXPECTED_CHECK_MACRO:
        fail("kernels.cuh CHECK macro is not the expected text: %r" % x1)
    x2 = [text(l) for l in moe if b"EXPERT_RAW = " in l and b"EXPERTS = " in l]
    if len(x2) != 1 or b"EXPERT_RAW = 17547264" not in x2[0] or b"EXPERTS = 896" not in x2[0]:
        fail("moe.cuh constants are not EXPERT_RAW = 17547264, EXPERTS = 896: %r" % x2)

    with open(out_wall, "wb") as f:
        f.write(D0)
    with open(out_main, "wb") as f:
        f.write(D1)
    for p, d in ((out_wall, D0), (out_main, D1)):
        with open(p, "rb") as f:
            if f.read() != d:
                fail("self-check: written file %s differs from the extracted bytes" % p)

    h = lambda b: hashlib.sha256(b).hexdigest()
    crlf = lambda b: b.replace(b"\n", b"\r\n")
    print("repo %s" % repo)
    print("revision %s -> commit %s" % (a.rev, commit))
    for name, raw in (("decode.cuh", dec_raw), ("kernels.cuh", ker_raw), ("moe.cuh", moe_raw)):
        print("blob    %s/%s git-object %s sha256 %s (%d bytes, %d LF, %d CR)" % (a.prefix, name, blob_id(name), h(raw), len(raw), raw.count(b"\n"), raw.count(b"\r")))
    print("block D0 decode.cuh line %d (%d bytes) sha256 %s  crlf-form sha256 %s" % (d0 + 1, len(D0), h(D0), h(crlf(D0))))
    print("block D1 decode.cuh lines %d-%d (%d bytes, %d lines) sha256 %s  crlf-form sha256 %s" % (d1a + 1, d1b + 1, len(D1), d1b - d1a + 1, h(D1), h(crlf(D1))))
    print("CHECK sites in D1 (file lines): %s" % sites)
    print("kernels.cuh CHECK macro: exact expected text, sha256 %s" % h(x1[0]))
    print("moe.cuh constants line: %s" % x2[0].decode())
    wt = os.path.join(repo, a.prefix, "decode.cuh")
    if os.path.isfile(wt):
        with open(wt, "rb") as f:
            w = f.read().replace(b"\r\n", b"\n")
        same_d1 = D1 in w
        print("working tree %s (CRLF folded): block D1 %s; whole file %s the blob (information only)" % (wt, "present unchanged" if same_d1 else "NOT found unchanged", "equals" if w == dec_raw else "differs from"))
    print("wrote %s and %s" % (out_wall, out_main))


if __name__ == "__main__":
    main()
