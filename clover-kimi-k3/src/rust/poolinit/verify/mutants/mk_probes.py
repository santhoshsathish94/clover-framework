#!/usr/bin/env python3
# Probe copies of NOT-CAUGHT mutants (and one instrumented copy of the ORIGINAL): the same single change plus a line appended to the file named by the environment
# variable PI_PROBE whenever the changed path runs (sys::probe("<tag> ...")). Built exactly like the mutants (mk_mutants.build_one), names <mutant>p; M00c = the original
# with a probe that tells which thread maps each chunk. Used by run_probes.sh / probe_count.py.
# usage: mk_probes.py [name ...]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mk_mutants as mm
from mk_mutants import E, H, P, A, L, S

MUTS = {n: (d, ed) for n, d, ed in mm.MUT}

PROBEFN = '''pub fn probe(tag: &str) {
    if let Ok(p) = std::env::var("PI_PROBE") {
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(p) {
            let line = format!("{} {}\\n", std::process::id(), tag);
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// seconds of clock `id` (0 = REALTIME, 1 = MONOTONIC), same arithmetic as wall_now
pub fn clk(id: c_int) -> f64 {
    let mut t = Timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `t` is a valid Timespec.
    unsafe { clock_gettime(id, &mut t) };
    t.tv_sec as f64 + 1e-9 * t.tv_nsec as f64
}

'''
PF = E(S, "pub fn write_stderr(msg: &[u8]) {", PROBEFN + "pub fn write_stderr(msg: &[u8]) {")
SUMM = "write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());"

def based(name, extra, replace=None):
    """the mutant's edits, with the (old,new) pairs in `replace` {old_new_text: probe_new_text} substituted, plus extra edits, plus the probe function."""
    d, ed = MUTS[name]
    out = []
    for f, old, new, cnt in ed:
        if replace and (f, old) in replace:
            new = replace[(f, old)]
        out.append((f, old, new, cnt))
    return (name + "p", d + " [+probe]", out + extra + [PF])

PROBES = [
    ("M00c", "ORIGINAL + probe: which thread maps each chunk (caller or helper), n_chunks, chunk number, address", [
        E(H, "let t0 = sys::wall_now();", "let t0 = sys::wall_now();\n    let caller_id = std::thread::current().id();"),
        E(H, "base[i].store(p as usize, Ordering::SeqCst);", "base[i].store(p as usize, Ordering::SeqCst);\n            sys::probe(&format!(\"map caller={} n_chunks={} i={} addr={:x}\", std::thread::current().id() == caller_id, n_chunks, i, p as usize));"),
        PF]),
    based("M03", [], {(H, "        touch();\n    });"): "        sys::probe(&format!(\"M03 caller-skips-touch n_chunks={}\", n_chunks));\n    });"}),
    ("M11p", "M11 [+probe]: orig seconds vs mutant seconds", [
        E(H, SUMM, "let t1 = sys::wall_now();\n    sys::probe(&format!(\"M11 orig_s={:.4} mut_s={:.4}\", sys::wall_now() - t0, sys::wall_now() - t1));\n    write_stderr(summary_line(cap, raw, sys::wall_now() - t1).as_bytes());"), PF]),
    ("M12p", "M12 [+probe]: orig seconds vs mutant seconds (x1000)", [
        E(H, SUMM, "let s = sys::wall_now() - t0;\n    sys::probe(&format!(\"M12 orig_s={:.4} mut_s={:.4}\", s, s * 1000.0));\n    write_stderr(summary_line(cap, raw, s * 1000.0).as_bytes());"), PF]),
    ("M13p", "M13 [+probe]: orig seconds vs mutant seconds (0.0)", [
        E(H, SUMM, "let s = sys::wall_now() - t0;\n    sys::probe(&format!(\"M13 orig_s={:.4} mut_s={:.4}\", s, 0.0));\n    write_stderr(summary_line(cap, raw, 0.0).as_bytes());"), PF]),
    ("M14p", "M14 [+probe]: elapsed on CLOCK_REALTIME vs CLOCK_MONOTONIC around the whole call", [
        E(H, "let t0 = sys::wall_now();", "let t0 = sys::wall_now();\n    let (r0, m0) = (sys::clk(0), sys::clk(1));"),
        E(H, SUMM, "sys::probe(&format!(\"M14 d_real={:.6} d_mono={:.6}\", sys::clk(0) - r0, sys::clk(1) - m0));\n    write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());"), PF]),
    based("M33", [], {(H, "for b in base.iter() {"): "sys::probe(&format!(\"M33 both-reversed n_chunks={}\", n_chunks));\n        for b in base.iter().rev() {"}),
    based("M46", [], {(H, "base[i].store(p as usize, Ordering::SeqCst);"): "base[n_chunks - 1 - i].store(p as usize, Ordering::SeqCst);\n            sys::probe(&format!(\"M46 reversed-index i={} n_chunks={}\", i, n_chunks));"}),
    ("M47p", "M47 [+probe]: every chunk store / load that is Relaxed in the mutant", [
        E(H, "Ordering::SeqCst", "Ordering::Relaxed", 4),
        E(H, "base[i].store(p as usize, Ordering::SeqCst);", "base[i].store(p as usize, Ordering::SeqCst);\n            sys::probe(\"M47 relaxed-store\");"), PF]),
    ("M48p", "M48 [+probe]: final cap computed both ways", [
        E(H, "let cap = n_chunks.wrapping_mul(PI_CHUNK_SLOTS);", "let cap = n_chunks.wrapping_mul(PI_CHUNK_SLOTS);\n    sys::probe(&format!(\"M48 cap0={} orig={} mut={} equal={}\", cap0, cap, cap0 & !(PI_CHUNK_SLOTS - 1), cap == (cap0 & !(PI_CHUNK_SLOTS - 1))));"), PF]),
    ("M49p", "M49 [+probe]: chunk_bytes computed both ways", [
        E(H, "PI_CHUNK_SLOTS.wrapping_mul(expert_raw), cap_slots", "{ crate::sys::probe(&format!(\"M49 raw={} equal={}\", expert_raw, PI_CHUNK_SLOTS.wrapping_mul(expert_raw) == (expert_raw << 5))); expert_raw << 5 }, cap_slots"), PF]),
    ("M57p", "M57 [+probe]: second error_string call in check()", [
        E(S, "m.extend_from_slice(cstr_bytes(error_string(code)));", "let _ = error_string(code);\n        probe(&format!(\"M57 error_string-twice code={}\", code));\n        m.extend_from_slice(cstr_bytes(error_string(code)));"), PF]),
    ("M58p", "M58 [+probe]: prefix written in a separate write call", [
        E(S, 'm.extend_from_slice(format!(":{}: ", line).as_bytes());', 'm.extend_from_slice(format!(":{}: ", line).as_bytes());\n        probe("M58 two-writes");\n        write_stderr(&m);\n        m.clear();'), PF]),
    ("M59p", "M59 [+probe]: exit through std::process::exit", [
        E(S, "    unsafe { exit(1) }", "    probe(\"M59 process-exit\");\n    std::process::exit(1)"), PF]),
    ("M67p", "M67 [+probe]: usable computed with >= ; tags the only inputs where > and >= could differ", [
        E(P, "pub fn slots_for(free_bytes: usize, reserve_bytes: usize, expert_raw: usize) -> c_int {", "pub fn slots_for(free_bytes: usize, reserve_bytes: usize, expert_raw: usize) -> c_int {\n    crate::sys::probe(&format!(\"M67 call free_eq_reserve={} free_lt_reserve={}\", free_bytes == reserve_bytes, free_bytes < reserve_bytes));"),
        E(P, "if free_bytes > reserve_bytes {", "if free_bytes >= reserve_bytes {"), PF]),
    ("M68p", "M68 [+probe]: usable = saturating_sub; tags calls with free < reserve", [
        E(P, "pub fn slots_for(free_bytes: usize, reserve_bytes: usize, expert_raw: usize) -> c_int {", "pub fn slots_for(free_bytes: usize, reserve_bytes: usize, expert_raw: usize) -> c_int {\n    crate::sys::probe(&format!(\"M68 call free_lt_reserve={}\", free_bytes < reserve_bytes));"),
        E(P, "let usable = if free_bytes > reserve_bytes { free_bytes - reserve_bytes } else { 0 };", "let usable = free_bytes.saturating_sub(reserve_bytes);"), PF]),
    ("M86p", "M86 [+probe]: d_eptr size with zero-extended experts; tags calls with experts < 0", [
        E(P, "cfg.experts as isize as usize", "{ crate::sys::probe(&format!(\"M86 experts={} negative={}\", cfg.experts, cfg.experts < 0)); cfg.experts as u32 as usize }"), PF]),
]

# registration order of the chunks seen from outside: address of each registered chunk, in the order the host_register calls are made (original, M46, M33)
REGP = "sys::probe(&format!(\"reg k={} n_chunks={} addr={:x}\", reg_k, n_chunks, b.load(Ordering::SeqCst)));"
EXTRA = [
    ("M00r", "ORIGINAL + probe: address of each chunk in registration order", [
        E(H, "for b in base.iter() {", "for (reg_k, b) in base.iter().enumerate() {\n            " + REGP), PF]),
    ("M46r", "M46 [+probe]: address of each chunk in registration order", [
        E(H, "base[i].store(p as usize, Ordering::SeqCst);", "base[n_chunks - 1 - i].store(p as usize, Ordering::SeqCst);"),
        E(H, "for b in base.iter() {", "for (reg_k, b) in base.iter().enumerate() {\n            " + REGP), PF]),
    ("M33r", "M33 [+probe]: address of each chunk in registration order", [
        E(H, "for b in base.iter() {", "for (reg_k, b) in base.iter().rev().enumerate() {\n            " + REGP),
        E(H, "for b in base.iter().rev() {", "for b in base.iter() {"), PF]),
]
PROBES += EXTRA

if __name__ == "__main__":
    only = sys.argv[1:]
    os.makedirs(mm.WORK, exist_ok=True); os.makedirs(mm.DIFFS, exist_ok=True)
    for name, desc, edits in PROBES:
        if only and name not in only:
            continue
        md5 = mm.build_one(name, desc, edits)
        if md5:
            print("%s %s %s" % (name, md5, desc), flush=True)
