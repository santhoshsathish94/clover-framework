#!/usr/bin/env python3
# Extra controls of this pass, NOT part of the independent verifier's 97 mutants: same method as mutants/mk_mutants.py (copy of the crate in verify/work/mut/<name>, exactly one textual change that
# must occur exactly once, cargo build --release), names X1.. . usage: mk_extra.py [name ...]   (diffs go to v2/logs/<name>.diff, nothing under mutants/ is written)
import difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/supply")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/v2/logs")
SCINC = os.path.join(HOME, "rustmig/slotcache/include")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])
E = "ensure.rs"
X = [
    ("X1", "job file offset: e * raw as an UNSIGNED 32-bit product (a wrong but positive offset once e * raw >= 2^32; equal below)", E,
     "off: (n.e as i64).wrapping_mul(raw as i64).wrapping_add(co as i64)", "off: ((n.e as u32).wrapping_mul(raw as u32) as i64).wrapping_add(co as i64)"),
    ("X2", "pread loop: a short read is accepted as the whole read (loop ends after the first successful pread)", E,
     "            got += r as usize;\n", "            got = j.len;\n"),
    ("X3", "run_parallel: calling thread numbered 1 (tid 0 never used when more than one thread runs)", E,
     "        work(c, 0);\n    });", "        work(c, if n > 1 { 1 } else { 0 });\n    });"),
]
if __name__ == "__main__":
    only = sys.argv[1:]
    os.makedirs(WORK, exist_ok=True); os.makedirs(DIFFS, exist_ok=True)
    for name, desc, fn, old, new in X:
        if only and name not in only:
            continue
        d = os.path.join(WORK, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for f in ("Cargo.toml", "Cargo.lock"):
            shutil.copy(os.path.join(SRC, f), d)
        shutil.copytree(os.path.join(SRC, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(SRC, "include"), os.path.join(d, "include"))
        shutil.copytree(SCINC, os.path.join(d, "slotcache-include"))
        p = os.path.join(d, "src", fn)
        t0 = open(p, newline="").read()
        if t0.count(old) != 1:
            print(name, "NOT APPLIED (count %d)" % t0.count(old), desc, flush=True)
            continue
        t1 = t0.replace(old, new)
        open(p, "w", newline="").write(t1)
        diff = "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/" + fn, "b/src/" + fn, n=3))
        open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s: %s\n%s" % (name, desc, diff))
        r = subprocess.run(["nice", "-n", "10", "cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
        so = os.path.join(d, "target/release/libsupply.so")
        if r.returncode != 0 or not os.path.exists(so):
            print(name, "BUILD FAILED", desc); print((r.stdout + r.stderr)[-1200:], flush=True)
            continue
        print(name, hashlib.md5(open(so, "rb").read()).hexdigest(), desc, flush=True)
