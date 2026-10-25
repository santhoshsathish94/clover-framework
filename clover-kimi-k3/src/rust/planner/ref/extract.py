#!/usr/bin/env python3
"""Pull the real planner text out of the engine's backend-gpu.cu and decode.cuh, byte for byte (CRLF kept), into one include file.

Ten blocks are taken and concatenated, with nothing added, in this order (decode.cuh first because PfJob needs Pending and Arrival):
  D1  decode.cuh   from the line starting "struct Pending {" to the line starting "static double g_t_pwait = 0;"
                   (Pending, g_pend, g_pend_keys, g_wb_skipped, g_t_pwait, g_pf_used, g_pf_waited, with the comment lines between)
  D2  decode.cuh   pend_take: from the line starting "static void pend_take(int key)" to the first following line that is exactly "}"
  D3  decode.cuh   from the line starting "struct Arrival {" through arr_wait (first line exactly "}" after "static void arr_wait(int key)")
                   (Arrival, g_arr, g_t_vwait, g_vq_waited, arr_wait)
  B1  backend-gpu.cu  the line starting "static size_t g_rc_pos = 0, g_rc_planned = 0, g_pend_cap = 0;"
  B2  backend-gpu.cu  the line starting "static int g_lookahead = 0; static bool g_rc_ok = false;"
  B3  backend-gpu.cu  the line starting "static unsigned long g_pf_planned = 0, g_pf_skipped = 0;"
  B4  backend-gpu.cu  from the line starting "struct Rel {" through the first line exactly "};" after the line starting "struct PfJob {"
  B5  backend-gpu.cu  rc_plan: from "static void rc_plan(size_t pos)" to the first following line exactly "}"
  B6  backend-gpu.cu  from the line starting "static std::vector<std::shared_ptr<Rel>> g_rel;" through rc_plan_vq (first line exactly "}"
                      after "static void rc_plan_vq(size_t pos, const std::vector<int> &cur)"): g_rel, g_wb_keys, g_vq_cap, g_vq_from, g_t_plan,
                      g_vq_ram/g_vq_drive/g_vq_wb/g_vq_nowb, vq_sweep, vq_issue, rc_plan_vq, with the comments between
  B7  backend-gpu.cu  route_hook: from "static void route_hook(int L, const std::vector<int> &uniq)" to the first following line exactly "}"
NOT taken: force_routes, rc_load, rc_nu, pool_ensure, pf_enqueue, print_summary, configure, SlotCacheSel, any CUDA part.
Exits 1 with a message, writing nothing, if a start marker does not match exactly one line, an end marker is missing, a block is implausibly long, braces
do not balance (counted per character; comments and strings are not parsed), a function block closes early, or expected text is absent. Any previous --out is
removed first so a failed run cannot leave a stale file behind. Prints sha256 of every block, of the extracted text and of both source files.
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


def function_end(lines, sig_start, what, max_len):
    b1 = first_after(lines, sig_start, lambda l: text(l) == b"}", "'}' closing " + what)
    if b1 - sig_start + 1 > max_len:
        fail("%s spans %d lines (expected at most %d); source layout changed?" % (what, b1 - sig_start + 1, max_len))
    return b1


def need(block, items, what):
    for n in items:
        if n not in block:
            fail("%s lacks expected text %r (source changed?)" % (what, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "code", "gpu")))
    ap.add_argument("--out", default=os.path.join(HERE, "planner_extracted.inc"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        os.remove(a.out)
    p_back = os.path.join(a.src, "backend-gpu.cu")
    p_dec = os.path.join(a.src, "decode.cuh")
    back = read_lines(p_back)
    dec = read_lines(p_dec)

    blocks = []   # (name, file name, first line index, last line index, bytes)

    def take(name, fname, lines, b0, b1, items, closes_early_ok=False, balance=True):
        if balance:
            check_balance(lines, b0, b1, name, closes_early_ok)
        data = b"".join(lines[b0:b1 + 1])
        need(data, items, name)
        blocks.append((name, fname, b0, b1, data))
        return data

    # ---- decode.cuh
    d1a = find_one(dec, starts(b"struct Pending {"), "'struct Pending {'")
    d1b = find_one(dec, starts(b"static double g_t_pwait = 0;"), "'static double g_t_pwait = 0;'")
    if not d1a < d1b or d1b - d1a > 8:
        fail("block D1 is not a short span in file order (lines %d-%d)" % (d1a + 1, d1b + 1))
    take("D1", "decode.cuh", dec, d1a, d1b,
         [b"static std::unordered_map<int, Pending *> g_pend;", b"static std::unordered_set<int> g_pend_keys;", b"static unsigned long g_wb_skipped = 0;",
          b"g_pf_used = 0, g_pf_waited = 0;"], closes_early_ok=True)

    d2a = find_one(dec, starts(b"static void pend_take(int key)"), "'static void pend_take(int key)'")
    d2b = function_end(dec, d2a, "pend_take", 20)
    take("D2", "decode.cuh", dec, d2a, d2b, [b"g_pend.find(key)", b"g_t_pwait += wall_now() - t0; g_pf_waited++;", b"g_pf_used++; g_pend.erase(f); g_pend_keys.erase(key); delete p;"])

    d3a = find_one(dec, starts(b"struct Arrival {"), "'struct Arrival {'")
    d3s = find_one(dec, starts(b"static void arr_wait(int key)"), "'static void arr_wait(int key)'")
    d3b = function_end(dec, d3s, "arr_wait", 20)
    if not d3a < d3s or d3s - d3a > 6:
        fail("struct Arrival and arr_wait are not close together in file order")
    take("D3", "decode.cuh", dec, d3a, d3b, [b"static std::unordered_map<int, std::shared_ptr<Arrival>> g_arr;", b"g_t_vwait = 0; static unsigned long g_vq_waited = 0;",
                                              b"g_arr.find(key)", b"g_t_vwait += wall_now() - t0; g_vq_waited++;"], closes_early_ok=True)

    # ---- backend-gpu.cu
    b1 = find_one(back, starts(b"static size_t g_rc_pos = 0, g_rc_planned = 0, g_pend_cap = 0;"), "g_rc_pos line")
    take("B1", "backend-gpu.cu", back, b1, b1, [], balance=False)
    b2 = find_one(back, starts(b"static int g_lookahead = 0; static bool g_rc_ok = false;"), "g_lookahead line")
    take("B2", "backend-gpu.cu", back, b2, b2, [], balance=False)
    b3 = find_one(back, starts(b"static unsigned long g_pf_planned = 0, g_pf_skipped = 0;"), "g_pf_planned line")
    take("B3", "backend-gpu.cu", back, b3, b3, [], balance=False)

    b4a = find_one(back, starts(b"struct Rel {"), "'struct Rel {'")
    b4p = find_one(back, starts(b"struct PfJob {"), "'struct PfJob {'")
    if not b4a < b4p or b4p - b4a > 3:
        fail("struct Rel and struct PfJob are not adjacent in file order")
    b4b = first_after(back, b4p, lambda l: text(l) == b"};", "'};' closing struct PfJob")
    if b4b - b4a > 12:
        fail("block B4 spans %d lines (expected 6); source layout changed?" % (b4b - b4a + 1))
    take("B4", "backend-gpu.cu", back, b4a, b4b, [b"struct Rel { std::atomic<int> state; unsigned char *slot;", b"std::shared_ptr<Arrival> arr; std::shared_ptr<Rel> rel;",
                                                   b"int kind = 0, dev = 0;"], closes_early_ok=True)

    b5a = find_one(back, starts(b"static void rc_plan(size_t pos)"), "'static void rc_plan(size_t pos)'")
    b5b = function_end(back, b5a, "rc_plan", 40)
    take("B5", "backend-gpu.cu", back, b5a, b5b, [b"g_rc_planned = i + 1;", b"g_pf_skipped++;", b"g_host.c.worth(key)", b"pf_submit(j);"])

    b6a = find_one(back, starts(b"static std::vector<std::shared_ptr<Rel>> g_rel;"), "'static std::vector<std::shared_ptr<Rel>> g_rel;'")
    s_sweep = find_one(back, lambda l: text(l) == b"static void vq_sweep()", "'static void vq_sweep()'")
    s_issue = find_one(back, starts(b"static bool vq_issue(CardPool &pool, Handle *hd, int key, int e, int cand_nu, const std::unordered_set<int> &prot)"), "vq_issue signature")
    b6s = find_one(back, starts(b"static void rc_plan_vq(size_t pos, const std::vector<int> &cur)"), "rc_plan_vq signature")
    if not (b6a < s_sweep < s_issue < b6s):
        fail("g_rel, vq_sweep, vq_issue, rc_plan_vq are not in the expected file order")
    b6b = function_end(back, b6s, "rc_plan_vq", 50)
    if b6b - b6a + 1 > 90:
        fail("block B6 spans %d lines (expected 78); source layout changed?" % (b6b - b6a + 1))
    B6 = take("B6", "backend-gpu.cu", back, b6a, b6b,
              [b"static std::vector<int> g_wb_keys;", b"static size_t g_vq_cap = 256;", b"static long g_vq_from = 0;", b"static double g_t_plan = 0;",
               b"g_vq_ram = 0, g_vq_drive = 0, g_vq_wb = 0, g_vq_nowb = 0;", b"g_host.c.release(g_rel[i]->slot)", b"pool.c.try_acquire(key, cand_nu, prot, 1.0f, &vk, &vh)",
               b"j.host_src = g_host.c.remove(key); j.kind = 2;", b"g_host.cap_slots * 8 / 10", b"j.arr = std::make_shared<Arrival>(); g_arr[key] = j.arr;",
               b"i == pos ? (int)pos : rc_nu_sel(key)"], closes_early_ok=True)
    if sum(1 for l in B6.split(b"\n") if l.rstrip(b"\r") == b"}") != 3:
        fail("block B6 does not contain exactly three top-level functions (vq_sweep, vq_issue, rc_plan_vq)")

    b7a = find_one(back, starts(b"static void route_hook(int L, const std::vector<int> &uniq)"), "'static void route_hook(...)'")
    b7b = function_end(back, b7a, "route_hook", 20)
    take("B7", "backend-gpu.cu", back, b7a, b7b, [b"rc_set_cur((long)g_rc_pos);", b"does not match the cache; look-ahead stopped", b"slotcache_set_next_use(nullptr);",
                                                    b"rc_plan_vq(g_rc_pos, uniq); else rc_plan(g_rc_pos);", b"g_rc_pos++;"])

    # order of blocks inside each file must be the file order (the include keeps it)
    for fname in ("decode.cuh", "backend-gpu.cu"):
        pos = [(b[2], b[3], b[0]) for b in blocks if b[1] == fname]
        for x, y in zip(pos, pos[1:]):
            if not x[1] < y[0]:
                fail("blocks %s and %s overlap or are out of file order in %s" % (x[2], y[2], fname))

    data = b"".join(b[4] for b in blocks)
    with open(a.out, "wb") as f:
        f.write(data)
    with open(a.out, "rb") as f:
        back_read = f.read()
    if back_read != data:
        fail("self-check: written file differs from extracted bytes")
    h = lambda b: hashlib.sha256(b).hexdigest()
    for p in (p_dec, p_back):
        with open(p, "rb") as f:
            print("source  %s sha256 %s" % (p, h(f.read())))
    for name, fname, b0, b1_, d in blocks:
        print("block %s %s lines %d-%d (%d bytes) sha256 %s" % (name, fname, b0 + 1, b1_ + 1, len(d), h(d)))
    print("wrote %s (%d bytes, CRLF lines %d of %d)" % (a.out, len(data), data.count(b"\r\n"), data.count(b"\n")))
    print("extracted text sha256 %s" % h(data))


if __name__ == "__main__":
    main()
