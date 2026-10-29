#!/usr/bin/env python3
# Builds the mutant crates for the differential-driver check (agent C2). The REAL crate is only read.
# For each mutant: ~/rustmig/planner/verify/work/mut/<name>/ (Cargo.toml, Cargo.lock, src/, include/ and the three sibling include dirs), a unified diff in
# ~/rustmig/planner/verify/mutants/<name>.diff, then `cargo build --release` there. The first 21 mutations are read from ~/rustmig/planner-mut.py (the unit-test campaign).
import ast, difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/planner")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/mutants")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])

def old_mutations():
    tree = ast.parse(open(os.path.join(HOME, "rustmig/planner-mut.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "M":
            return ast.literal_eval(node.value)
    raise SystemExit("M list not found")

# (name, description, file, old, new, expected occurrences of old)
MUT = []
for desc, fn, old, new in old_mutations():
    MUT.append((desc.split()[0], desc, fn, old, new, 1))

TRY = "(self.ops.sc_try_acquire)(pool.cache, key, cand_nu, prot.as_ptr()"
OFF = "j.off = (e as i64).wrapping_mul(self.cfg.expert_raw as i64);"
OFF32 = "j.off = (e.wrapping_mul(self.cfg.expert_raw as i32)) as i64;"
MUT += [
    ("MA", "vq_issue: try_acquire gets (cand_nu, key) instead of (key, cand_nu)", "planner.rs",
     TRY, "(self.ops.sc_try_acquire)(pool.cache, cand_nu, key, prot.as_ptr()", 1),
    ("MB", "pend_take counts pf_used also when there is no pending entry", "planner.rs",
     "let Some(p) = self.pend.get(key) else { return };\n        if !done(&p.state) {\n            let t0 = Instant::now();\n            spin_until_set(&p.state);",
     "let Some(p) = self.pend.get(key) else {\n            self.st.pf_used += 1;\n            return;\n        };\n        if !done(&p.state) {\n            let t0 = Instant::now();\n            spin_until_set(&p.state);", 1),
    ("MC", "card-pool prot list drops its last element", "planner.rs",
     "let prot: Vec<i32> = cur.iter().map(|&e| self.ekey(pos_layer, e)).collect();",
     "let mut prot: Vec<i32> = cur.iter().map(|&e| self.ekey(pos_layer, e)).collect();\n        prot.pop();", 1),
    ("MD", "job off = e * expert_raw as a 32-bit multiply (both job builders)", "planner.rs", OFF, OFF32, 2),
    ("MD1", "job off as a 32-bit multiply, read-ahead job (kind 0) only", "planner.rs",
     "j.fd = fd;\n        " + OFF + "\n        j.slot = slot;", "j.fd = fd;\n        " + OFF32 + "\n        j.slot = slot;", 1),
    ("MD2", "job off as a 32-bit multiply, card-queue job (kind 1/2) only", "planner.rs",
     "j.fd = fd;\n        " + OFF + "\n        j.layer = layer;", "j.fd = fd;\n        " + OFF32 + "\n        j.layer = layer;", 1),
]
MUT.insert(0, ("M00", "control: unmodified copy", "planner.rs", "", "", 0))

only = sys.argv[1:]
os.makedirs(WORK, exist_ok=True); os.makedirs(DIFFS, exist_ok=True)
for name, desc, fn, old, new, cnt in MUT:
    if only and name not in only:
        continue
    d = os.path.join(WORK, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(SRC, f), d)
    for sub in ("src", "include", "copyq-include", "slotcache-include", "routetable-include"):
        shutil.copytree(os.path.join(SRC, sub), os.path.join(d, sub))
    p = os.path.join(d, "src", fn)
    t0 = open(p).read()
    if cnt:
        if t0.count(old) != cnt:
            print(name, "NOT APPLIED (count %d, wanted %d)" % (t0.count(old), cnt), flush=True)
            continue
        t1 = t0.replace(old, new)
        open(p, "w").write(t1)
    else:
        t1 = t0
    diff = "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/" + fn, "b/src/" + fn, n=3))
    open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s: %s\n%s" % (name, desc, diff))
    r = subprocess.run(["cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
    so = os.path.join(d, "target/release/libplanner.so")
    if r.returncode != 0 or not os.path.exists(so):
        print(name, "BUILD FAILED"); print((r.stdout + r.stderr)[-800:], flush=True)
        continue
    print(name, hashlib.md5(open(so, "rb").read()).hexdigest(), desc, flush=True)
