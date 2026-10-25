#!/usr/bin/env python3
"""Negative controls of extract.py (Windows or anywhere with python and git): each case must make extract.py exit 1 and write no output file.
Cases: (1) a marker renamed (run with --no-pin so the failure is the marker, not the hash), (2) the state group edited (--no-pin), (3) an unrelated byte changed with the pins on
(source sha256 mismatch), (4) a missing source directory, (5) the pinned run on the real blobs succeeds. The shipped .inc files are never touched (outputs go to a temp dir).
Usage: python extract_controls.py
"""
import os, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
SUB = "clover-kimi-k3/src/code/gpu/"


def blob(name):
    return subprocess.check_output(["git", "-C", REPO, "show", "4d8d891:" + SUB + name])


def run(args, outdir):
    p = subprocess.run([sys.executable, os.path.join(HERE, "extract.py"), "--outdir", outdir] + args, capture_output=True, text=True)
    left = sorted(os.listdir(outdir))
    return p.returncode, (p.stderr.strip() or p.stdout.strip().splitlines()[-1]), left


def main():
    bad = 0
    tmp = tempfile.mkdtemp(prefix="extract_controls_")
    try:
        srcs = {n: blob(n) for n in ("backend-gpu.cu", "kernels.cuh")}

        def case(name, mutate, extra, want_rc):
            nonlocal bad
            d = os.path.join(tmp, name); os.makedirs(os.path.join(d, "src")); os.makedirs(os.path.join(d, "out"))
            for n, b in srcs.items():
                m = mutate(n, b)
                assert m is not None
                with open(os.path.join(d, "src", n), "wb") as f:
                    f.write(m)
            changed = any(mutate(n, b) != b for n, b in srcs.items())
            rc, msg, left = run(["--src", os.path.join(d, "src")] + extra, os.path.join(d, "out"))
            ok = rc == want_rc and (rc == 0 or left == [])
            bad += 0 if ok else 1
            print("%-34s source changed=%-5s rc=%d outputs=%s %s | %s" % (name, changed, rc, left, "OK" if ok else "UNEXPECTED", msg))

        def rep(a, b):
            def f(n, data):
                if n != "backend-gpu.cu":
                    return data
                assert data.count(a) == 1, a
                return data.replace(a, b)
            return f

        case("marker_loader_wait_renamed", rep(b"static void loader_wait()", b"static void loader_waitX()"), ["--no-pin"], 1)
        case("marker_loader_struct_renamed", rep(b"struct Loader {", b"struct LoaderX {"), ["--no-pin"], 1)
        case("state_group_line_edited", rep(b"        h->mla = h->y.mla;", b"        h->mla = !h->y.mla;"), ["--no-pin"], 1)
        case("pread_check_changed", rep(b"if (rd <= 0) {", b"if (rd < 0) {"), ["--no-pin"], 1)
        case("one_byte_changed_pins_on", rep(b"cannot open %s", b"cannot open  %s"), [], 1)
        case("untouched_copy_pins_on", lambda n, b: b, [], 0)
        # a missing source directory
        out = os.path.join(tmp, "missing_out"); os.makedirs(out)
        rc, msg, left = run(["--src", os.path.join(tmp, "does-not-exist")], out)
        ok = rc == 1 and left == []
        bad += 0 if ok else 1
        print("%-34s rc=%d outputs=%s %s | %s" % ("missing_source_dir", rc, left, "OK" if ok else "UNEXPECTED", msg))
        # the real run from git (default mode)
        out = os.path.join(tmp, "git_out"); os.makedirs(out)
        rc, msg, left = run([], out)
        ok = rc == 0 and left == ["loader_a.inc", "loader_b.inc"]
        bad += 0 if ok else 1
        print("%-34s rc=%d outputs=%s %s" % ("git_mode_pinned", rc, left, "OK" if ok else "UNEXPECTED"))
        for fn in ("loader_a.inc", "loader_b.inc"):
            same = open(os.path.join(out, fn), "rb").read() == open(os.path.join(HERE, fn), "rb").read()
            bad += 0 if same else 1
            print("%-34s equals shipped %s: %s" % ("git_mode_pinned", fn, same))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("extract controls:", "ALL AS EXPECTED" if bad == 0 else "%d UNEXPECTED" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
