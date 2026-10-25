#!/usr/bin/env python3
# Probes for the mutants the driver did not catch (independent verifier): the mutant PLUS a line appended to /dev/shm/ldprobe-<probe>.txt whenever the changed path runs,
# or, where the mutation only matters under a condition, whenever that condition holds ("pid count tag"; count = n-th hit of that tag in that process; written for n <= 50 and at powers of 10).
# usage: mk_probes.py [PROBE ...]     builds work/mut/<probe>/ and work/so/<probe>.so (then: TAG=probe run_mut.sh <probe>..., probe_sum.py)
import hashlib, os, shutil, subprocess, sys
import mk_mutants as mm

PR = '''
#[allow(dead_code)]
pub fn pr(tag: &'static str) {
    use std::io::Write;
    static CNT: Mutex<Vec<(&'static str, usize)>> = Mutex::new(Vec::new());
    let n = {
        let mut v = CNT.lock().unwrap_or_else(|e| e.into_inner());
        match v.iter_mut().find(|e| e.0 == tag) {
            Some(e) => {
                e.1 += 1;
                e.1
            }
            None => {
                v.push((tag, 1));
                1
            }
        }
    };
    if n <= 50 || n == 100 || n == 1000 || n == 10000 || n == 100000 {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("/dev/shm/ldprobe-@NAME@.txt") {
            let line = format!("{} {} {}\\n", std::process::id(), n, tag);
            let _ = f.write_all(line.as_bytes());
        }
    }
}
'''
Q, J, L = mm.Q, mm.J, mm.L
P = {
    "PM05": ("M05", [(Q, "            *lock(&self.pending) -= 1;\n",
                      "            {\n                let mut pg = lock(&self.pending);\n                if *pg == 0 {\n                    pr(\"counter decremented before it was incremented (wraps)\");\n                }\n                *pg = pg.wrapping_sub(1);\n            }\n")]),
    "PM14": ("M14", [(Q, "            let g = lock(&self.pending);\n            drop(self.pending_cv.wait(g)",
                      "            let g = lock(&self.pending);\n            if *g == 0 {\n                pr(\"window hit: count already 0 when the waiter is about to sleep\");\n            }\n            drop(self.pending_cv.wait(g)")]),
    "PM25": ("M25", [(Q, "        if !started {\n            g.started = true;\n",
                      "        if !started {\n            if g.started {\n                pr(\"double start: another submitter started the card between the check and the lock\");\n            }\n            g.started = true;\n")]),
    "PM50": ("M50", [(J, "    if fd >= 0 {\n        set_fd(j.h_fd, fd);\n    }\n",
                      "    if fd >= 0 {\n        set_fd(j.h_fd, fd);\n    } else {\n        crate::queue::pr(\"first open failed: its -1 is not stored (fallback follows)\");\n    }\n")]),
    "PM58": ("M58", [(J, "let off = ((r as u32).wrapping_mul(raw as u32) as i64).wrapping_add(got as i64);",
                      "let off = ((r as u32).wrapping_mul(raw as u32) as i64).wrapping_add(got as i64);\n                crate::queue::pr(\"offset computed\");\n                if off != (r as i64).wrapping_mul(raw as i64).wrapping_add(got as i64) {\n                    crate::queue::pr(\"32-bit offset DIFFERS from the 64-bit one\");\n                }")]),
    "PM59": ("M59", [(J, "let bytes = (4u32.wrapping_mul(raw as u32)) as usize;",
                      "let bytes = (4u32.wrapping_mul(raw as u32)) as usize;\n        crate::queue::pr(\"slab size computed\");\n        if bytes != 4usize.wrapping_mul(raw) {\n            crate::queue::pr(\"32-bit slab size DIFFERS from the 64-bit one\");\n        }")]),
    "PM51": ("M51", [(J, "    if fd <= 0 {\n        let mut m = b\"cannot open \"",
                      "    if fd == 0 {\n        crate::queue::pr(\"open returned fd 0\");\n    }\n    if fd <= 0 {\n        let mut m = b\"cannot open \"")]),
    "PM87": ("M87", [(L, "    if STATE.get().is_some() {\n        return; // the first configuration is kept\n    }\n    if cfg.is_null() {",
                      "    if STATE.get().is_some() {\n        crate::queue::pr(\"ld_init after init\");\n        if cfg.is_null() {\n            crate::queue::pr(\"ld_init(NULL) after init\");\n        }\n        return; // the first configuration is kept\n    }\n    if cfg.is_null() {")]),
    "PM92": ("M92", [(L, "    die(check_slab(st.cfg.prefill, r.layer));\n    die(check_card(r.card));\n",
                      "    crate::queue::pr(\"guards evaluated\");\n    if check_slab(st.cfg.prefill, r.layer).is_err() && check_card(r.card).is_err() {\n        crate::queue::pr(\"both guards fail\");\n    }\n    die(check_slab(st.cfg.prefill, r.layer));\n    die(check_card(r.card));\n")]),
}

def build(pname, base, extra):
    edits = [e for n, d, e in mm.M if n == base][0] + extra
    d = os.path.join(mm.WORK, pname)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(mm.SRC, f), d)
    for sub in ("src", "include"):
        shutil.copytree(os.path.join(mm.SRC, sub), os.path.join(d, sub))
    for fn in sorted(set(e[0] for e in edits)):
        p = os.path.join(d, "src", fn)
        t = open(p, newline="").read()
        for f2, old, new in edits:
            if f2 != fn:
                continue
            o, n = mm.crlf(old, t), mm.crlf(new, t)
            if t.count(o) != 1:
                print(pname, "NOT APPLIED (count %d) %r" % (t.count(o), old[:60])); return
            t = t.replace(o, n)
        open(p, "w", newline="").write(t)
    p = os.path.join(d, "src", "queue.rs")
    t = open(p, newline="").read()
    t += mm.crlf(PR.replace("@NAME@", pname), t)
    open(p, "w", newline="").write(t)
    r = subprocess.run(["nice", "-n", "10", "cargo", "build", "--release"], cwd=d, env=mm.env, capture_output=True, text=True, timeout=900)
    so = os.path.join(d, "target/release/libloader.so")
    if r.returncode != 0 or not os.path.exists(so):
        print(pname, "BUILD FAILED"); print((r.stdout + r.stderr)[-1500:]); return
    shutil.copy(so, os.path.join(mm.SODIR, pname + ".so"))
    print(pname, hashlib.md5(open(so, "rb").read()).hexdigest(), "probe of", base, flush=True)

if __name__ == "__main__":
    for pname, (base, extra) in P.items():
        if sys.argv[1:] and pname not in sys.argv[1:]:
            continue
        build(pname, base, extra)
