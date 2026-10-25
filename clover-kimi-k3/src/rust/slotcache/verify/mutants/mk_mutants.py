#!/usr/bin/env python3
# Creates the mutant crate copies M0..M16 under ~/rustmig/slotcache/verify/work/mut/ from the real crate (read only).
# Each mutation is an exact-text replacement that must match exactly once. M0 is an unmodified copy (control for the build's reproducibility).
import os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
REAL = os.path.join(HOME, "rustmig/slotcache")
ROOT = os.path.join(REAL, "verify/work/mut")
real_src = open(os.path.join(REAL, "src/lib.rs"), newline="").read()

M = {}
M["M1"] = [("self.nodes[i].heat * 0.8f32 + w", "self.nodes[i].heat * 0.9f32 + w")]
M["M2"] = [("if POLICY.load(Ordering::SeqCst) != 0 { 48 } else { 1 }", "if POLICY.load(Ordering::SeqCst) != 0 { 47 } else { 1 }")]
M["M3"] = [("n.heat < best_heat", "n.heat <= best_heat")]
M["M4"] = [("for &(_, k) in self.by_next.iter().rev() {", "for &(_, k) in self.by_next.iter() {")]
M["M5"] = [("nu_call(key) < largest", "nu_call(key) <= largest")]
M["M6"] = [("if nu > cand_nu {", "if nu >= cand_nu {")]
M["M7"] = [("            self.nodes[i].nu = nu;\n            self.by_next.insert((nu, key));\n", "            self.nodes[i].nu = nu;\n")]
M["M8"] = [("        let Some(&i) = self.map.get(&key) else { return std::ptr::null_mut() };\n        self.detach(i, key)\n",
            "        let Some(&i) = self.map.get(&key) else { return std::ptr::null_mut() };\n        let slot = self.nodes[i].slot;\n        self.unlink(i);\n        self.free.push(i);\n        self.map.remove(&key);\n        slot\n")]
M["M9"] = [("            for &(_, k) in self.by_next.iter().rev() {\n                if !prot.contains(&k) {\n                    victim = k;\n                    break;\n                }\n            }\n",
            "            if let Some(&(_, k)) = self.by_next.iter().next_back() {\n                victim = k;\n            }\n")]
M["M10"] = [("        self.unlink(i);\n        self.push_front(i);\n        self.nodes[i].slot\n", "        self.nodes[i].slot\n")]
M["M11"] = [("        if let Some(s) = self.spare.pop() {\n            p = s;\n", "        if !self.spare.is_empty() {\n            p = self.spare.remove(0);\n")]
M["M12"] = [("self.nodes[i].heat = self.nodes[i].heat * 0.8f32 + w;", "self.nodes[i].heat = self.nodes[i].heat.mul_add(0.8f32, w);")]
M["M13"] = [("        let mut nu = 0;\n        if belady() {\n            nu = nu_call(key);\n            self.by_next.insert((nu, key));\n        }\n", "        let nu = 0;\n")]
M["M14"] = [("                    seen += 1;\n                }\n                it = n.prev;\n", "                }\n                seen += 1;\n                it = n.prev;\n")]
M["M15"] = [("(too few slots)\\n\"", "(too few slotz)\\n\"")]
M["M16"] = [("POLICY.load(Ordering::SeqCst) == 2 && NU_FN", "POLICY.load(Ordering::SeqCst) >= 2 && NU_FN")]
M["M0"] = []

os.makedirs(ROOT, exist_ok=True)
for name in sorted(M, key=lambda s: int(s[1:])):
    d = os.path.join(ROOT, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "src"))
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(REAL, f), os.path.join(d, f))
    shutil.copy(os.path.join(REAL, "src/tests.rs"), os.path.join(d, "src/tests.rs"))
    s = real_src
    for old, new in M[name]:
        n = s.count(old)
        if n != 1:
            sys.exit("%s: pattern matched %d times: %r" % (name, n, old[:60]))
        s = s.replace(old, new)
    open(os.path.join(d, "src/lib.rs"), "w", newline="").write(s)
    with open(os.path.join(ROOT, name + ".diff"), "w") as out:
        subprocess.run(["diff", "-u", "--label", "a/src/lib.rs", "--label", "b/src/lib.rs", os.path.join(REAL, "src/lib.rs"), os.path.join(d, "src/lib.rs")], stdout=out)
    print(name, "created")
