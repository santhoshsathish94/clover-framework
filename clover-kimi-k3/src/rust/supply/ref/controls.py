#!/usr/bin/env python3
"""Negative controls of extract.py (Windows or anywhere with python and git). Each control gives the extractor a source that differs from the committed text in one way and expects
exit status 1 and NO output file; the positive control (an unchanged copy read with --src) must exit 0 and write byte-identical files. Writes only to a temp dir.
Usage: python controls.py [--repo DIR]"""
import argparse, hashlib, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "clover-kimi-k3/src/code/gpu/"


def blob(repo, name):
    p = subprocess.run(["git", "-C", repo, "show", "4d8d891:" + PREFIX + name], capture_output=True)
    if p.returncode != 0:
        sys.exit("git show failed: " + p.stderr.decode())
    return p.stdout


def run(args, outdir):
    p = subprocess.run([sys.executable, os.path.join(HERE, "extract.py"), "--outdir", outdir] + args, capture_output=True)
    wrote = [f for f in ("supply_types.inc", "supply_extracted.inc") if os.path.exists(os.path.join(outdir, f))]
    msg = (p.stderr.decode(errors="replace").strip().splitlines() or [""])[-1]
    return p.returncode, wrote, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..")))
    a = ap.parse_args()
    names = ["decode.cuh", "kernels.cuh", "moe.cuh"]
    orig = {n: blob(a.repo, n) for n in names}
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        def case(label, edits, extra, expect_ok=False):
            nonlocal bad
            dname = "".join(ch if ch.isalnum() else "_" for ch in label)
            src = os.path.join(tmp, dname, "src"); out = os.path.join(tmp, dname, "out")
            os.makedirs(src); os.makedirs(out)
            for n in names:
                d = orig[n]
                for (fn, old, new) in edits:
                    if fn == n:
                        assert old in d, "control %s: text to replace not found: %r" % (label, old)
                        d = d.replace(old, new, 1)
                with open(os.path.join(src, n), "wb") as f:
                    f.write(d)
            # a stale output from an earlier run must be removed by the extractor
            with open(os.path.join(out, "supply_types.inc"), "wb") as f:
                f.write(b"stale")
            rc, wrote, msg = run(["--src", src] + extra, out)
            ok = (rc == 0 and len(wrote) == 2) if expect_ok else (rc == 1 and not wrote)
            print("%-4s %-34s exit %d, files written: %s | %s" % ("OK" if ok else "BAD", label, rc, wrote or "none", msg[:150]))
            if not ok:
                bad += 1
            return out
        skip = ["--control-skip-blob-check"]
        out = case("positive: unchanged copy", [], [], expect_ok=True)
        for n in ("supply_types.inc", "supply_extracted.inc"):
            with open(os.path.join(out, n), "rb") as f, open(os.path.join(HERE, n), "rb") as g:
                same = f.read() == g.read()
            print("%-4s %-34s %s identical to the shipped file" % ("OK" if same else "BAD", "positive: " + n, "is" if same else "is NOT"))
            bad += 0 if same else 1
        case("blob hash: one comment byte changed", [("decode.cuh", b"// runs the jobs on up to nthreads threads", b"// runs the jobs on up to nthreads thread")], [])
        case("marker: pool_ensure renamed", [("decode.cuh", b"static void pool_ensure(CardPool", b"static void pool_ensureX(CardPool")], skip)
        case("marker: pool_ensure defined twice", [("decode.cuh", b"static void kda_attention_state(", b"static void pool_ensure(CardPool &pool, HostCache &hc, int layer, int fd, const std::vector<int> &experts, const int *counts, int readers) {}\nstatic void kda_attention_state(")], skip)
        case("text: write-back threshold line removed", [("decode.cuh", b"if (!hc.c.worth(v.key)) { g_wb_skipped++; continue; }", b"if (!hc.c.worth(v.key)) { continue; }")], skip)
        case("text: chunk arithmetic changed", [("decode.cuh", b"(pool_readers + n_drive - 1) / n_drive", b"(pool_readers + n_drive) / n_drive")], skip)
        case("braces: run_parallel loses its last brace", [("decode.cuh", b"    for (auto &t : th) t.join();\n}\n", b"    for (auto &t : th) t.join();\n\n")], skip)
        case("end marker: pool_ensure lost, next function too", [("decode.cuh", b"pool.st.t_read += t1 - t0; pool.st.t_copy += wall_now() - t1;\n}\n", b"pool.st.t_read += t1 - t0; pool.st.t_copy += wall_now() - t1;\n")], skip)
        case("adaptation source: CHECK macro changed", [("kernels.cuh", b"cudaGetErrorString(e_)); exit(1); } } while (0)", b"cudaGetErrorString(e_)); abort(); } } while (0)")], skip)
        case("adaptation source: EXPERTS changed", [("moe.cuh", b"EXPERTS = 896", b"EXPERTS = 897")], skip)
    # sources that cannot be read at all
    with tempfile.TemporaryDirectory() as tmp:
        for label, args in (("source: nonexistent --src dir", ["--src", os.path.join(tmp, "nope")]), ("source: nonexistent --repo", ["--repo", os.path.join(tmp, "nope")]), ("source: another commit (7cbb532)", ["--commit", "7cbb532"])):
            rc, wrote, msg = run(args + (["--repo", a.repo] if "--repo" not in args else []), tmp)
            ok = rc == 1 and not wrote
            print("%-4s %-34s exit %d, files written: %s | %s" % ("OK" if ok else "BAD", label, rc, wrote or "none", msg[:150]))
            bad += 0 if ok else 1
    print("controls: %s" % ("all as expected" if bad == 0 else "%d BAD" % bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
