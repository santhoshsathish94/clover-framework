#!/usr/bin/env python3
"""Pull the real loader text out of the engine, byte for byte, into two include files for loader_ref.cpp.

Source of truth: the engine as COMMITTED at 4d8d891 (the working tree is being edited by someone else). Default mode reads the blobs with
`git show <rev>:clover-kimi-k3/src/code/gpu/<file>` (bytes, no text conversion); `--src <dir>` reads the same file names from a directory instead
(for tests with a changed copy). Line endings are whatever the source has (the committed blobs are LF; a Windows working tree is CRLF): bytes are copied, never converted.

loader_a.inc  (placed before the harness stubs of open_layer):
  K1  kernels.cuh     the line starting "#define CHECK(x) do {"
  B1  backend-gpu.cu  the line starting "enum { MAX_CARDS = 8, N_LAYERS = 92 };"
  B2  backend-gpu.cu  from the line starting "struct Loader {" through loader_wait: the first line exactly "}" after the line starting "static void loader_wait()"
                      (Loader, g_loader, pending_mu, pending_cv, g_pending, loader_submit, loader_wait, with the comments between)
loader_b.inc:
  B3  backend-gpu.cu  open_layer: from the line starting "static Handle *open_layer(" to the first following line exactly "}",
                      EXCEPT the five lines of the state group (h->mla = h->y.mla; ... } else kda_state_alloc(h->kda);), which are checked (exact text, exactly once)
                      and replaced by ONE line `loader_state_alloc(h, positions);` (ADAPTED (6) in loader_ref.cpp / ref/NOTES.md).
NOT taken: configure, card_of, ensure_work, layer_load_*, kda_state_alloc, forward, any CUDA part, Handle, g_handles, g_slab.
Exits 1 with a message, writing nothing, if a marker matches 0 or more than one line, an end marker is missing, a block is implausibly long, braces do not balance
(counted per character; comments and strings are not parsed), a block closes early, expected text is absent, or a sha256 differs from the pinned value (--no-pin to explore
another version). A previous output is removed first. Prints sha256 of every source and block and of the two outputs.
Usage: python extract.py [--git <repo> --rev 4d8d891 | --src <dir>] [--outdir <dir>] [--no-pin]
"""
import argparse, hashlib, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SUBDIR = "clover-kimi-k3/src/code/gpu/"

# sha256 of the sources and blocks as committed at 4d8d891 (pinned after the first reading; extract.py fails if they change)
PIN_SOURCE = {
    "backend-gpu.cu": "cc1950a108900a62f42f9e278012753ecbe64f2928e7b95fe6958e7b9207c3f3",
    "kernels.cuh": "0ce0a33edd8ffde2a0f2fc3dbff8b2028cae07ba210874660a123271dde6277a",
}
PIN_BLOCK = {
    "K1": "1089b6325c70ffc523ea4e77fcaa64d7acec360f2d3143e324681d0c1684f917",
    "B1": "33e69051834b3f9675f3168c811f37c3080cb5d8f74ab95d5c967f2984b5e3d2",
    "B2": "2515fce16d02ef490084654d34b53cf95fc61118a2e6e526411d56e81bcfd667",
    "B3": "f53ed3f08a4f0e7ea96b02a2d5f7b6732ef88b188466c329a42d022c1f48bbcc",
}
PIN_OUT = {"loader_a.inc": "71ac541871b3c8850c3227d771c6a82fac9cde956a7ab5ea5852662f20f0daa5",
           "loader_b.inc": "167676f25a65e8fa41da3eaca8fc636102ac1f18892a86ff4ec0aa61826d41d2"}


def fail(msg):
    sys.stderr.write("extract.py: FAIL: " + msg + "\n")
    sys.exit(1)


def split_lines(data):
    parts = data.split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def text(l):
    return l.rstrip(b"\r\n")


def eol_of(l):
    return l[len(text(l)):]


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


def check_balance(lines, b0, b1, what, closes_early_ok):
    depth = 0
    opened = False
    for i in range(b0, b1 + 1):
        depth += lines[i].count(b"{") - lines[i].count(b"}")
        opened = opened or b"{" in lines[i]
        if depth < 0 or (not closes_early_ok and opened and depth == 0 and i != b1):
            fail("braces of %s do not balance as one block (depth %d at source line %d)" % (what, depth, i + 1))
    if depth != 0:
        fail("braces of %s unbalanced at its end line (depth %d)" % (what, depth))


def need(block, items, what):
    for n in items:
        if n not in block:
            fail("%s lacks expected text %r (source changed?)" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..")))
    ap.add_argument("--rev", default="4d8d891")
    ap.add_argument("--src", default=None)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args()

    outs = [os.path.join(a.outdir, n) for n in ("loader_a.inc", "loader_b.inc")]
    for o in outs:
        if os.path.exists(o):
            os.remove(o)

    def source(name):
        if a.src:
            p = os.path.join(a.src, name)
            if not os.path.isfile(p):
                fail("source file not found: " + p)
            with open(p, "rb") as f:
                return f.read()
        try:
            return subprocess.check_output(["git", "-C", a.git, "show", "%s:%s%s" % (a.rev, REPO_SUBDIR, name)], stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, OSError) as e:
            fail("git show %s:%s%s failed: %s" % (a.rev, REPO_SUBDIR, name, e))

    h = lambda b: hashlib.sha256(b).hexdigest()
    src_bytes = {n: source(n) for n in ("backend-gpu.cu", "kernels.cuh")}
    for n, d in src_bytes.items():
        print("source  %s sha256 %s (%d bytes)" % (n, h(d), len(d)))
        if not a.no_pin and h(d) != PIN_SOURCE[n]:
            fail("source %s sha256 %s differs from the pinned %s" % (n, h(d), PIN_SOURCE[n]))
    back = split_lines(src_bytes["backend-gpu.cu"])
    ker = split_lines(src_bytes["kernels.cuh"])

    blocks = []   # (name, file, first, last, bytes)

    def take(name, fname, lines, b0, b1, items, closes_early_ok=False, balance=True):
        if balance:
            check_balance(lines, b0, b1, name, closes_early_ok)
        data = b"".join(lines[b0:b1 + 1])
        need(data, items, name)
        blocks.append([name, fname, b0, b1, data])
        return data

    k1 = find_one(ker, starts(b"#define CHECK(x) do {"), "CHECK macro")
    take("K1", "kernels.cuh", ker, k1, k1, [b"cudaGetErrorString(e_)", b"exit(1)"], balance=False)

    b1 = find_one(back, starts(b"enum { MAX_CARDS = 8, N_LAYERS = 92 };"), "enum MAX_CARDS")
    take("B1", "backend-gpu.cu", back, b1, b1, [], balance=False)

    b2a = find_one(back, starts(b"struct Loader {"), "'struct Loader {'")
    b2s = find_one(back, starts(b"static void loader_wait()"), "'static void loader_wait()'")
    if not b2a < b2s:
        fail("struct Loader is not before loader_wait")
    b2b = first_after(back, b2s, lambda l: text(l) == b"}", "'}' closing loader_wait")
    if b2b - b2a + 1 > 60:
        fail("block B2 spans %d lines (expected about 38); source layout changed?" % (b2b - b2a + 1))
    B2 = take("B2", "backend-gpu.cu", back, b2a, b2b,
              [b"static Loader *g_loader[MAX_CARDS];", b"static int g_pending = 0;", b"static void loader_submit(int card, std::function<void()> job)",
               b"if (!g_loader[card]) g_loader[card] = new Loader;", b"CHECK(cudaSetDevice(card));", b"l.th.detach();", b"pending_cv().notify_all();",
               b"pending_cv().wait(u, []() { return g_pending == 0; });"], closes_early_ok=True)
    if sum(1 for l in B2.split(b"\n") if l.rstrip(b"\r") == b"}") != 2:
        fail("block B2 does not contain exactly two top-level '}' lines (loader_submit, loader_wait)")

    b3a = find_one(back, starts(b"static Handle *open_layer("), "'static Handle *open_layer('")
    b3b = first_after(back, b3a, lambda l: text(l) == b"}", "'}' closing open_layer")
    if b3b - b3a + 1 > 60:
        fail("block B3 spans %d lines (expected about 38); source layout changed?" % (b3b - b3a + 1))
    if not b2b < b3a:
        fail("open_layer is not after loader_wait")
    B3 = take("B3", "backend-gpu.cu", back, b3a, b3b,
              [b"configure();", b"h->L = (int)L; h->card = card_of((int)L); h->k_in = (int)k;", b"if (L <= (unsigned)N_LAYERS) g_handles[L] = h;",
               b"loader_submit(h->card, [h, ds, L, positions]() {", b"CHECK(cudaSetDevice(h->card));", b"ensure_work(h->card);", b"layer_load_trunk(ds.c_str(), (int)L, h->y);",
               b"if (L >= 1) layer_load_expert_tables(ds.c_str(), (int)L, h->y);", b"h->fd = open(path, O_RDONLY | O_DIRECT);", b"if (h->fd < 0) h->fd = open(path, O_RDONLY);",
               b"fprintf(stderr, \"cannot open %s\\n\", path); exit(1);", b"CHECK(cudaHostAlloc((void **)&g_slab[L], (size_t)4 * EXPERT_RAW, cudaHostAllocDefault));",
               b"if (rd <= 0) { fprintf(stderr, \"prefill: cannot read a record of layer %u\\n\", L); exit(1); }", b"return h;"], closes_early_ok=True)

    # The state group inside open_layer: five consecutive lines, exact text, exactly once; replaced by one line.
    b3_lines = split_lines(B3)
    group = [b"        h->mla = h->y.mla;",
             b"        if (h->mla) {",
             b"            h->mc.cap = positions;",
             b"            CHECK(cudaMalloc(&h->mc.kv, (size_t)positions * MLA_EXP * 4)); CHECK(cudaMalloc(&h->mc.lat, (size_t)positions * MLA_KVLAT * 4));",
             b"        } else kda_state_alloc(h->kda);"]
    hits = [i for i in range(len(b3_lines) - len(group) + 1) if [text(l) for l in b3_lines[i:i + len(group)]] == group]
    if len(hits) != 1:
        fail("the five-line state group of open_layer was found %d times (expected exactly once); source changed?" % len(hits))
    g0 = hits[0]
    replaced = b"".join(b3_lines[g0:g0 + len(group)])
    new_line = b"        loader_state_alloc(h, positions);   // ADAPTED (6): replaces the five lines h->mla = h->y.mla ... kda_state_alloc(h->kda); see NOTES.md" + eol_of(b3_lines[g0 + len(group) - 1])
    b3_adapted = b"".join(b3_lines[:g0]) + new_line + b"".join(b3_lines[g0 + len(group):])
    if b3_adapted.replace(new_line, replaced) != B3:
        fail("self-check: the adapted open_layer does not reduce to the extracted block")

    out_a = b"".join(b[4] for b in blocks if b[0] in ("K1", "B1", "B2"))
    out_b = b3_adapted
    if not a.no_pin:
        for name, ref in PIN_BLOCK.items():
            got = h([b for b in blocks if b[0] == name][0][4])
            if got != ref:
                fail("block %s sha256 %s differs from the pinned %s" % (name, got, ref))
        for fn, data in (("loader_a.inc", out_a), ("loader_b.inc", out_b)):
            if h(data) != PIN_OUT[fn]:
                fail("%s sha256 %s differs from the pinned %s" % (fn, h(data), PIN_OUT[fn]))
    for o, data in zip(outs, (out_a, out_b)):
        with open(o, "wb") as f:
            f.write(data)
        with open(o, "rb") as f:
            if f.read() != data:
                fail("self-check: written file %s differs from the extracted bytes" % o)
    for name, fname, b0, b1_, d in blocks:
        print("block %s %s lines %d-%d (%d bytes) sha256 %s" % (name, fname, b0 + 1, b1_ + 1, len(d), h(d)))
    print("state group lines %d-%d of B3 (source lines %d-%d, %d bytes) sha256 %s" % (g0 + 1, g0 + len(group), b3a + g0 + 1, b3a + g0 + len(group), len(replaced), h(replaced)))
    for o, data in zip(outs, (out_a, out_b)):
        print("output  %s (%d bytes, %d lines) sha256 %s md5 %s" % (os.path.basename(o), len(data), data.count(b"\n"), h(data), hashlib.md5(data).hexdigest()))


if __name__ == "__main__":
    main()
