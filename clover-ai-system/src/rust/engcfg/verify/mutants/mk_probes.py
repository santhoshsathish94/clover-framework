#!/usr/bin/env python3
# Run on the box after mk_mutants.py: builds "probe" copies = a mutant's src/lib.rs plus one line that APPENDS A TAG to a file each time the changed path runs with an input on which it can matter
# (verify/work/probe/<P>/ ; events in verify/out/probe/<P>.ev). Same driver run as the mutant; the number of lines in the .ev file says whether the driver ever reached the distinguishing input.
# Usage: mk_probes.py [P41 P42 ...]   (default: all)
import os, shutil, sys

HOME = os.path.expanduser("~")
REAL = os.path.join(HOME, "rustmig/engcfg")
WORK = os.path.join(REAL, "verify/work")
EVDIR = os.path.join(REAL, "verify/out/probe")
EVFN = '''
fn ev(tag: &str) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("%s/%s.ev") {
        let _ = f.write_all(tag.as_bytes());
        let _ = f.write_all(b"\\n");
    }
}
'''
P = []
def probe(name, base, desc, *pairs):
    P.append((name, base, desc, list(pairs)))

N = "    if neg {\n        if over || acc > (1u64 << 63) {"
probe("P41", "M41", "event when the value is exactly 2^63 with a minus sign (the only input on which `>=` vs `>` is executed differently)",
      ("    if neg {\n        if over || acc >= (1u64 << 63) {", "    if neg {\n        if !over && acc == (1u64 << 63) { ev(\"neg_exact_2^63\"); }\n        if over || acc >= (1u64 << 63) {"))
probe("P42", "M42", "event when the value is exactly i64::MAX without a minus sign",
      (N, "    if !neg && !over && acc == i64::MAX as u64 { ev(\"pos_exact_i64max\"); }\n" + N))
probe("P43", "M43", "event when digits remain after the accumulator overflowed (the digits the mutant no longer consumes)",
      (N, "    if over && i < s.len() && s[i].is_ascii_digit() { ev(\"digits_left_after_overflow\"); }\n" + N))
probe("P82", "M82", "event when `layer - 1` overflows i32", ("let d = layer - 1;", "if layer.checked_sub(1).is_none() { ev(\"layer-1_overflows\"); }\n        let d = layer - 1;"))
probe("P83", "M83", "event when `layer * cards` overflows i32", ("(layer * cards) / 93)", "({ if layer.checked_mul(cards).is_none() { ev(\"layer*cards_overflows\"); } layer * cards }) / 93)"))
probe("P84", "M84", "event when the removed explicit check would have fired", ("/* no explicit check */", "ev(\"trap_input_reached\");"))
probe("P103", "M103", "event when top is negative (the only case in which zero- and sign-extension differ)", ("t.wrapping_mul(top as u32 as usize)", "{ if top < 0 { ev(\"top_negative\"); } t }.wrapping_mul(top as u32 as usize)"))
probe("P104", "M104", "event when a fill counter is negative", ("fill[t] as u32 as usize)", "{ if fill[t] < 0 { ev(\"fill_negative\"); } fill[t] as u32 } as usize)"))
probe("P105", "M105", "event when n < 0 reaches the dealing", ("    let mut fill = vec![0i32; n as usize];", "    if n < 0 { ev(\"n_negative_reaches_dealing\"); }\n    let mut fill = vec![0i32; n as usize];"))
probe("P114", "M114", "event every time the changed expression is evaluated", ("s.t_calls[i] * 1000.0 / s.n_calls[i]", "{ ev(\"ms_expression_evaluated\"); s.t_calls[i] * 1000.0 / s.n_calls[i] }"))
probe("P95", "M95", "event when a call has more counts than experts", ("if c.n_counts < c.n_experts {", "if c.n_counts > c.n_experts { ev(\"more_counts_than_experts\"); }\n    if c.n_counts < c.n_experts {"))
probe("P92", "M92", "event when the fits check is the only thing rejecting the call (some count > n, total == n*top)",
      ("if total != (n as i64) * (top as i64) {", "if !fits && total == (n as i64) * (top as i64) { ev(\"fits_false_total_ok\"); }\n    if total != (n as i64) * (top as i64) {"))
probe("P98", "M98", "event when the sum of counts leaves the i32 range",
      ("let mut total: i32 = 0;", "let mut total: i32 = 0;\n    let mut tt: i64 = 0;"), ("total = total.wrapping_add(k);", "total = total.wrapping_add(k);\n        tt += k as i64;"),
      ("    if !fits || (total as i64)", "    if tt > i32::MAX as i64 || tt < i32::MIN as i64 { ev(\"sum_outside_i32\"); }\n    if !fits || (total as i64)"))
probe("P99", "M99", "event when n * top does not fit i32", ("total != n.wrapping_mul(top) as i64", "{ if (n as i64) * (top as i64) != n.wrapping_mul(top) as i64 { ev(\"ntop_outside_i32\"); } total != n.wrapping_mul(top) as i64 }"))
probe("P100", "M100", "event when a fill counter reaches 255", ("fill[t] += 1;", "if fill[t] == 255 { ev(\"fill_reaches_255\"); }\n            fill[t] += 1;"))
probe("P101", "M101", "event when a fill counter reaches 65535", ("fill[t] += 1;", "if fill[t] == 65535 { ev(\"fill_reaches_65535\"); }\n            fill[t] += 1;"))
probe("P102", "M102", "event when the dealing counter reaches 2^31", ("s = s.wrapping_add(1);", "if s == i32::MAX { ev(\"s_reaches_2^31\"); }\n            s = s.wrapping_add(1);"))
probe("P48", "M48", "event when 6x wraps in 64 bits (cap_slots >= 2^64/6)", ("cap_slots.wrapping_mul(3) / 5", "{ if cap_slots.checked_mul(6).is_none() { ev(\"pend_cap_product_wraps\"); } cap_slots.wrapping_mul(3) / 5 }"))
probe("P138", "M138", "event on each emit call (the changed path)", ("    let h = b.len() / 2;\n", "    ev(\"emit_called\");\n    let h = b.len() / 2;\n"))
# fp rewrites: event when the rewritten expression gives a different double ("value") or a different %.1f text ("print") from the original expression
def fp(name, base, old, orig, new, vars_):
    probe(name, base, "events: rewritten expression differs from the original in value / in its %.1f text",
          (old, "{ let o: f64 = %s; let a: f64 = %s; if o != a { ev(\"value_differs\"); if format!(\"{:.1}\", o) != format!(\"{:.1}\", a) { ev(\"print_differs\"); } } ev(\"evaluated\"); a }" % (orig, new)))
fp("P106", "M106", "hcf / needf * 100.0", "100.0 * hcf / needf", "hcf / needf * 100.0", "")
fp("P107", "M107", "100.0 * (hhf / needf)", "100.0 * hhf / needf", "100.0 * (hhf / needf)", "")
fp("P108", "M108", "drf / needf * 100.0", "100.0 * drf / needf", "drf / needf * 100.0", "")
fp("P109", "M109", "drf / 1e9 * (s.expert_raw as f64)", "drf * (s.expert_raw as f64) / 1e9", "drf / 1e9 * (s.expert_raw as f64)", "")
fp("P110", "M110", "drf * (s.expert_raw as f64) * 1e-9", "drf * (s.expert_raw as f64) / 1e9", "drf * (s.expert_raw as f64) * 1e-9", "")
fp("P112", "M112", "s.t_calls[i] / s.n_calls[i] * 1000.0", "1000.0 * s.t_calls[i] / s.n_calls[i]", "s.t_calls[i] / s.n_calls[i] * 1000.0", "")
fp("P113", "M113", "1000.0 * (s.t_calls[i] / s.n_calls[i])", "1000.0 * s.t_calls[i] / s.n_calls[i]", "1000.0 * (s.t_calls[i] / s.n_calls[i])", "")
fp("P115", "M115", "s.t_calls[i] / (s.n_calls[i] / 1000.0)", "1000.0 * s.t_calls[i] / s.n_calls[i]", "s.t_calls[i] / (s.n_calls[i] / 1000.0)", "")

os.makedirs(EVDIR, exist_ok=True)
want = sys.argv[1:]
ok = True
for name, base, desc, pairs in P:
    if want and name not in want:
        continue
    src = open(os.path.join(WORK, "mut", base, "src/lib.rs"), "rb").read().decode()
    nl = "\r\n" if "\r\n" in src else "\n"
    d = os.path.join(WORK, "probe", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "src"))
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(WORK, "mut", base, f), os.path.join(d, f))
    shutil.copy(os.path.join(WORK, "mut", base, "src/tests.rs"), os.path.join(d, "src/tests.rs"))
    out = src
    for old, new in pairs:
        old = old.replace("\n", nl); new = new.replace("\n", nl)
        if out.count(old) != 1:
            print("ERROR", name, "pattern occurs", out.count(old), "times:", repr(old)); ok = False
        out = out.replace(old, new)
    out += (EVFN % (EVDIR, name)).replace("\n", nl)
    open(os.path.join(d, "src/lib.rs"), "wb").write(out.encode())
    print(name, "on", base, "-", desc)
sys.exit(0 if ok else 1)
