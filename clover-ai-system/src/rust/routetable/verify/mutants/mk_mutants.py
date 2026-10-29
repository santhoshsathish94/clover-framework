#!/usr/bin/env python3
# Run on the box: creates verify/work/mut/<name>/{Cargo.toml,Cargo.lock,src/,include/} copies of the real crate with ONE edit in src/lib.rs each,
# and verify/work/mut/diffs/<name>.diff (unified diff against the real src/lib.rs). The real crate is only read.
import difflib, os, shutil, sys

HOME = os.path.expanduser("~")
REAL = os.path.join(HOME, "rustmig/routetable")
WORK = os.path.join(REAL, "verify/work/mut")
src = open(os.path.join(REAL, "src/lib.rs"), "rb").read().decode()

M = []
def mut(name, desc, old, new):
    M.append((name, desc, old, new))

mut("M0", "control: unmodified copy", None, None)
mut("M1", "next-use skip `<=` cur -> `<` cur",
    "while *p < l.len() && (l[*p] as i64) <= cur {", "while *p < l.len() && (l[*p] as i64) < cur {")
mut("M2", "sentinel 1000000000 -> 999999999",
    "const EXHAUSTED: i32 = 1_000_000_000;", "const EXHAUSTED: i32 = 999_999_999;")
mut("M3", "per-key pointer reset to 0 on every rt_next_use call",
    "Some((l, p)) => {\n                while", "Some((l, p)) => {\n                *p = 0;\n                while")
mut("M4", "rt_next_use inserts an (empty list, pointer 0) entry for an unknown key",
    "None => EXHAUSTED,", "None => {\n                self.uses.insert(key, (Vec::new(), 0));\n                EXHAUSTED\n            }")
mut("M5", "a call that lists an expert twice contributes its number once to the uses list",
    "uses.entry(ekey(c.layer, experts, e)).or_insert_with(|| (Vec::new(), 0)).0.push(i as i32);",
    "{\n                let v = &mut uses.entry(ekey(c.layer, experts, e)).or_insert_with(|| (Vec::new(), 0)).0;\n                if v.last() != Some(&(i as i32)) {\n                    v.push(i as i32);\n                }\n            }")
mut("M6", "key = e * experts + layer (operands swapped)",
    "layer.wrapping_mul(experts).wrapping_add(e)", "e.wrapping_mul(experts).wrapping_add(layer)")
mut("M7", "expert loop also skips tabs between numbers",
    "while at(s, p) == b' ' {", "while matches!(at(s, p), b' ' | b'\\t') {")
mut("M8", "expert loop also starts an expert at '+'/'-' followed by a digit (safe variant, see M8L)",
    "if !at(s, p).is_ascii_digit() {\n            break;",
    "if !(at(s, p).is_ascii_digit() || (matches!(at(s, p), b'+' | b'-') && at(s, p + 1).is_ascii_digit())) {\n            break;")
mut("M8L", "literal M8: '+'/'-' alone start an expert (no progress on a bare sign: infinite loop; run under ulimit -v)",
    "if !at(s, p).is_ascii_digit() {\n            break;",
    "if !(at(s, p).is_ascii_digit() || matches!(at(s, p), b'+' | b'-')) {\n            break;")
mut("M9", "the second number (ntok) is not skipped",
    "    p = strtol(s, p).1;\n    loop {", "    loop {")
mut("M10", "empty lines (just \"\\n\") are dropped instead of becoming calls",
    "data.split_inclusive(|&b| b == b'\\n').map(parse_call).collect()",
    "data.split_inclusive(|&b| b == b'\\n').filter(|l| !l.starts_with(b\"\\n\")).map(parse_call).collect()")
mut("M11", "a last line without a trailing newline is dropped",
    "data.split_inclusive(|&b| b == b'\\n').map(parse_call).collect()",
    "let mut lines: Vec<&[u8]> = data.split_inclusive(|&b| b == b'\\n').collect();\n    if lines.last().map_or(false, |l| !l.ends_with(b\"\\n\")) {\n        lines.pop();\n    }\n    lines.into_iter().map(parse_call).collect()")
mut("M12", "a colon with no digits after it records no count (instead of 0)",
    "            call.counts.push(c as i32);\n            p = q;",
    "            if q != p + 1 {\n                call.counts.push(c as i32);\n            }\n            p = q;")
mut("M13", "a count is kept only while counts and experts are aligned (counts so far + 1 == experts so far)",
    "            call.counts.push(c as i32);\n            p = q;",
    "            if call.counts.len() + 1 == call.experts.len() {\n                call.counts.push(c as i32);\n            }\n            p = q;")
mut("M14", "strtol: on u64 overflow the accumulator wraps instead of saturating",
    "None => overflow = true,", "None => acc = acc.wrapping_mul(10).wrapping_add(d),")
mut("M15", "rt_set_current stores cur truncated to i32",
    "(*t).cur = cur as i64;", "(*t).cur = cur as i32 as i64;")
mut("M16", "missing-file message loses its trailing newline",
    "    let _ = e.write_all(path);\n    let _ = e.write_all(b\"\\n\");", "    let _ = e.write_all(path);")
mut("M17", "missing-file exit status 1 -> 2",
    "std::process::exit(1)", "std::process::exit(2)")
mut("M18", "rt_layer returns the layer of call i+1 when it exists",
    "t.calls[i].layer\n}", "t.calls.get(i + 1).unwrap_or(&t.calls[i]).layer\n}")
mut("M19", "strtol leading-whitespace set loses vertical tab (0x0b) and form feed (0x0c)",
    "b' ' | b'\\t' | b'\\n' | 0x0b | 0x0c | b'\\r'", "b' ' | b'\\t' | b'\\n' | b'\\r'")

os.makedirs(os.path.join(WORK, "diffs"), exist_ok=True)
lines_real = src.splitlines(keepends=True)
ok = True
for name, desc, old, new in M:
    d = os.path.join(WORK, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "src"))
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(REAL, f), os.path.join(d, f))
    shutil.copytree(os.path.join(REAL, "include"), os.path.join(d, "include"))
    shutil.copy(os.path.join(REAL, "src/tests.rs"), os.path.join(d, "src/tests.rs"))
    out = src
    if old is not None:
        if src.count(old) != 1:
            print("ERROR", name, "pattern occurs", src.count(old), "times"); ok = False; continue
        out = src.replace(old, new)
    open(os.path.join(d, "src/lib.rs"), "wb").write(out.encode())
    diff = list(difflib.unified_diff(lines_real, out.splitlines(keepends=True), "src/lib.rs (real)", f"{name}/src/lib.rs", n=3))
    hunks = sum(1 for l in diff if l.startswith("@@"))
    open(os.path.join(WORK, "diffs", name + ".diff"), "w", newline="\n").write("".join(diff))
    open(os.path.join(WORK, "diffs", name + ".txt"), "w", newline="\n").write(desc + "\n")
    print(name, "hunks", hunks, "-", desc)
sys.exit(0 if ok else 1)
