#!/usr/bin/env python3
# Instrumented copies of M06 and M08 (probes, not mutants of the real crate): each appends one line per event to out/mut/probe_events.txt so the driver run shows
# whether the changed code path is ever reached. Builds ~/rustmig/planner/verify/work/mut/<name>/ like mk_mutants.py.
import os, shutil, subprocess, hashlib, difflib

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/planner")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/mutants")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])
EV = '''pub fn ev(tag: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("/home/usergpu/rustmig/planner/verify/out/mut/probe_events.txt") {
        let _ = f.write_all(format!("{} {}\\n", tag, std::process::id()).as_bytes());
    }
}

'''
HOOK = "pub fn write_stderr(bytes: &[u8]) {"
P = {
 "M06p": [
  (HOOK, EV + HOOK),
  ("            self.pend_take(key);\n            j.host_src = unsafe { (self.ops.sc_remove)(self.w.host.cache, key) };",
   "            j.host_src = unsafe { (self.ops.sc_remove)(self.w.host.cache, key) };\n"
   "            ev(match self.pend.get(key).map(|p| p.state.load(Ordering::Acquire)) { None => \"M06 absent\", Some(0) => \"M06 present_state0\", Some(_) => \"M06 present_done\" });\n"
   "            self.pend_take(key);")],
 "M08p": [
  (HOOK, EV + HOOK),
  ("            if !done(&old.state) {\n                self.dead_pend.push(old);\n            }",
   "            if !done(&old.state) {\n                ev(\"M08 dropped_unfinished_pending\");\n            }\n            drop(old);")],
}
for name, edits in P.items():
    d = os.path.join(WORK, name)
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)
    for f in ("Cargo.toml", "Cargo.lock"): shutil.copy(os.path.join(SRC, f), d)
    for sub in ("src", "include", "copyq-include", "slotcache-include", "routetable-include"): shutil.copytree(os.path.join(SRC, sub), os.path.join(d, sub))
    p = os.path.join(d, "src/planner.rs"); t0 = open(p).read(); t1 = t0
    for old, new in edits:
        assert t1.count(old) == 1, (name, old[:40])
        t1 = t1.replace(old, new)
    open(p, "w").write(t1)
    open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s: instrumented probe\n%s" % (name, "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/planner.rs", "b/src/planner.rs", n=3))))
    r = subprocess.run(["cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
    so = os.path.join(d, "target/release/libplanner.so")
    print(name, "BUILD FAILED\n" + (r.stdout + r.stderr)[-1500:] if r.returncode else hashlib.md5(open(so, "rb").read()).hexdigest(), flush=True)
