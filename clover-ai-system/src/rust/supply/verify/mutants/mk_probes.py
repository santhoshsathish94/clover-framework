#!/usr/bin/env python3
# Probes for the mutants the driver did not catch (independent verifier, 2026-10-25). A probe = the mutant plus lines that append one line "<pid> <tag>" to the file named by $SUPPLY_PROBE
# whenever the changed path runs (so "the path is reached" and "the state where original and mutant differ is reached" can be counted over the whole driver set).
# Probe crates are built in ~/rustmig/supply/verify/work/mut/<probe>/ like the mutants; the diff (mutant + probe lines, against the real file) goes to mutants/<probe>.diff.
# usage: mk_probes.py [probe ...]
import difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/supply")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/mutants")
SCINC = os.path.join(HOME, "rustmig/slotcache/include")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])
E = "ensure.rs"

PROBE_FN = '''pub static WB_SKIPPED: AtomicU64 = AtomicU64::new(0);

pub fn probe(tag: &str) {
    use std::io::Write;
    if let Ok(p) = std::env::var("SUPPLY_PROBE") {
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let line = format!("{} {}\n", std::process::id(), tag);
            let _ = f.write_all(line.as_bytes());
        }
    }
}'''
FN_OLD = "pub static WB_SKIPPED: AtomicU64 = AtomicU64::new(0);"

CHUNK_ANCHOR = "    let nchunk: c_int = if cfg_chunks > 0 {"
Q8 = '''    if cfg_chunks <= 0 && n_drive < pool_readers && n_drive > 0 {
        let q = (pool_readers + n_drive - 1) / n_drive;
        if q >= 8 {
            let r = |n: usize| (raw.wrapping_add(n).wrapping_sub(1) / n).wrapping_add(4095) / 4096 * 4096;
            probe(&format!("q8 q={} raw={} chunk8={} chunk7={} chunk9={}", q, raw, r(8), r(7), r(9)));
        }
    }
'''
P = []  # (name, base mutant, description, [(file, old, new)])
P.append(("P16", "M16", "vk sentinel -2 (instead of the mutant's 0) and a tag after sc_acquire: was *vkey written?", [
    (E, "let mut vk: c_int = -1;", "let mut vk: c_int = -2;"),
    (E, "p = (ops.sc_acquire)(pool.cache, key, prot.as_ptr(), prot.len(), heat, &mut vk, &mut vh);",
        "p = (ops.sc_acquire)(pool.cache, key, prot.as_ptr(), prot.len(), heat, &mut vk, &mut vh);\n            probe(if vk == -2 { \"vk_unwritten\" } else if vk == -1 { \"vk_written_minus1\" } else { \"vk_written_victim\" });")]))
P.append(("P26", "M26", "M26 (continue instead of break) + a tag at the continue, with the number of victims", [
    (E, "if host.cap_slots == 0 || prefill {\n                break;\n            }",
        "if host.cap_slots == 0 || prefill {\n                probe(if victims.len() > 1 { \"continue_victims_gt1\" } else { \"continue_victims_1\" });\n                continue;\n            }")]))
P.append(("P41", "M41", "M41 (n_drive > pool_readers) + a tag when n_drive == pool_readers > 0, with the nchunk the other branch would give", [
    (E, "} else if n_drive >= pool_readers || n_drive == 0 {", "} else if n_drive > pool_readers || n_drive == 0 {"),
    (E, CHUNK_ANCHOR, "    if cfg_chunks <= 0 && n_drive == pool_readers && n_drive > 0 {\n        probe(&format!(\"eq nd={} alt_nchunk={}\", n_drive, std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)));\n    }\n" + CHUNK_ANCHOR)]))
P.append(("P42", "M42", "M42 (min 7) + a tag when the computed ratio is >= 8 with the chunk sizes 8 / 7 / 9 chunks would give", [
    (E, "std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)", "std::cmp::min(7, (pool_readers + n_drive - 1) / n_drive)"),
    (E, CHUNK_ANCHOR, Q8 + CHUNK_ANCHOR)]))
P.append(("P43", "M43", "M43 (min 9) + the same tag as P42", [
    (E, "std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)", "std::cmp::min(9, (pool_readers + n_drive - 1) / n_drive)"),
    (E, CHUNK_ANCHOR, Q8 + CHUNK_ANCHOR)]))
P.append(("P52", "M52", "M52 (32-bit e * raw) + a tag per job: does the 32-bit product differ from the 64-bit one?", [
    (E, "jobs.push(Job { dst: n.dst.wrapping_add(co), off: (n.e as i64).wrapping_mul(raw as i64).wrapping_add(co as i64), e: n.e, co, len });",
        "{\n                    let full = (n.e as i64).wrapping_mul(raw as i64);\n                    let narrow = n.e.wrapping_mul(raw as i32) as i64;\n                    probe(&format!(\"off e={} raw={} full={} narrow_equal={}\", n.e, raw, full, full == narrow));\n                }\n                jobs.push(Job { dst: n.dst.wrapping_add(co), off: (n.e.wrapping_mul(raw as i32) as i64).wrapping_add(co as i64), e: n.e, co, len });")]))
P.append(("P57", "M57", "M57 (.max(1)) + a tag before run_parallel when the unclamped pool_readers is < 1", [
    (E, "let pool_readers: c_int = std::cmp::min(readers, pool.readers);", "let pool_readers: c_int = std::cmp::min(readers, pool.readers).max(1);"),
    (E, "        let t0 = Instant::now();\n        run_parallel(&ctx, pool_readers);",
        "        if std::cmp::min(readers, pool.readers) < 1 {\n            probe(&format!(\"pr_le0 jobs={}\", ctx.jobs.len()));\n        }\n        let t0 = Instant::now();\n        run_parallel(&ctx, pool_readers);")]))
P.append(("P62", "M62", "M62 (got = r) + tags: a short read, and a loop iteration with got > 0", [
    (E, "got += r as usize;",
        "if (r as usize) < j.len - got {\n                probe(&format!(\"short r={} want={} got={}\", r, j.len - got, got));\n            }\n            if got > 0 {\n                probe(&format!(\"iteration_with_got_gt0 got={} r={} len={}\", got, r, j.len));\n            }\n            got = r as usize;")]))
P.append(("P67", "M67", "M67 (exit 2) + a tag in fatal_bytes", [
    (E, "    write_stderr(msg);\n    std::process::exit(1)\n", "    probe(\"fatal_bytes\");\n    write_stderr(msg);\n    std::process::exit(2)\n")]))
P.append(("P74", "M74", "M74 (no exit after check returns) + tags: status != 0 handed to check; check returned with status != 0", [
    (E, "unsafe { (ops.check)(status, site.as_ptr() as *const c_char) };\n    if status != 0 {\n        std::process::exit(1); // ops.check must not return",
        "if status != 0 {\n        probe(\"nonzero_status_to_check\");\n    }\n    unsafe { (ops.check)(status, site.as_ptr() as *const c_char) };\n    if status != 0 {\n        probe(\"check_returned_with_nonzero_status\");\n    }\n    if status != 0 && false {\n        std::process::exit(1); // ops.check must not return")]))
P.append(("P76", "M76", "M76 (one thread fewer) + a tag with n and the job count at every run_parallel", [
    (E, "for i in 1..n {", "for i in 1..(n - 1).max(1) {"),
    (E, "    std::thread::scope(|s| {", "    probe(&format!(\"run_parallel n={} jobs={}\", n, c.jobs.len()));\n    std::thread::scope(|s| {")]))
P.append(("P90", "M90", "M90 (NULL RAM cache not worth a slot) + a tag at every host_worth call", [
    (E, "        host.cache.is_null() || (self.ops.sc_worth)(host.cache, key) != 0",
        "        probe(if host.cache.is_null() { \"worth_null_cache\" } else { \"worth_nonnull_cache\" });\n        !host.cache.is_null() && (self.ops.sc_worth)(host.cache, key) != 0")]))
P.append(("P92", "M92", "M92 (no NULL guard in pool_find) + a tag at every pool_find call", [
    (E, "        if pool.cache.is_null() {\n            null_mut()\n        } else {\n            (self.ops.sc_find)(pool.cache, key)\n        }\n",
        "        probe(if pool.cache.is_null() { \"pool_find_null_cache\" } else { \"pool_find_nonnull_cache\" });\n        (self.ops.sc_find)(pool.cache, key)\n")]))
P.append(("P93", "M93", "M93 (counter not reset in Supply::new) + a tag with the counter's value at every init", [
    (E, "        WB_SKIPPED.store(0, Ordering::SeqCst);\n", "        probe(&format!(\"init prev_wbs={}\", WB_SKIPPED.load(Ordering::SeqCst)));\n")]))
P.append(("P97", "M97", "M97 (n == 0 with non-NULL experts) + a tag at every sy_ensure with n and NULL-ness", [
    ("lib.rs", "let e: &[c_int] = if n == 0 || experts.is_null() { &[] } else { std::slice::from_raw_parts(experts, n) };",
              "ensure::probe(&format!(\"sy_ensure n={} experts_null={}\", n, experts.is_null()));\n    let e: &[c_int] = if experts.is_null() { &[] } else { std::slice::from_raw_parts(experts, n) };")]))

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, base, desc, edits in P:
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
        orig = {}; cur = {}
        ok = True
        for fn, old, new in [(E, FN_OLD, PROBE_FN)] + edits:
            p = os.path.join(d, "src", fn)
            if fn not in cur:
                orig[fn] = open(p, newline="").read(); cur[fn] = orig[fn]
            if cur[fn].count(old) != 1:
                print(name, "NOT APPLIED (count %d)" % cur[fn].count(old), old[:60].replace("\n", "\\n"), flush=True); ok = False; break
            cur[fn] = cur[fn].replace(old, new)
        if not ok:
            continue
        diff = ""
        for fn in cur:
            open(os.path.join(d, "src", fn), "w", newline="").write(cur[fn])
            diff += "".join(difflib.unified_diff(orig[fn].splitlines(True), cur[fn].splitlines(True), "a/src/" + fn, "b/src/" + fn, n=3))
        open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s (%s + probe): %s\n%s" % (name, base, desc, diff))
        r = subprocess.run(["nice", "-n", "10", "cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
        so = os.path.join(d, "target/release/libsupply.so")
        if r.returncode != 0 or not os.path.exists(so):
            print(name, "BUILD FAILED", desc); print((r.stdout + r.stderr)[-1500:], flush=True)
            continue
        print(name, hashlib.md5(open(so, "rb").read()).hexdigest(), desc, flush=True)
