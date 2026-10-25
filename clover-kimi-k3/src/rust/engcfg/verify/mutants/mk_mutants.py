#!/usr/bin/env python3
# Run on the box: creates verify/work/mut/<name>/{Cargo.toml,Cargo.lock,src/,include/} copies of the real crate with ONE change in src/lib.rs each
# (a change may be several textual replacements when they are one idea, e.g. a type change), plus verify/work/mut/diffs/<name>.{diff,txt}.
# The real crate is only read. Every pattern must occur exactly once in the real src/lib.rs.
import difflib, os, shutil, sys

HOME = os.path.expanduser("~")
REAL = os.path.join(HOME, "rustmig/engcfg")
WORK = os.path.join(REAL, "verify/work/mut")
src = open(os.path.join(REAL, "src/lib.rs"), "rb").read().decode()
NL = "\r\n" if "\r\n" in src else "\n"   # the crate files are byte-identical to the Windows ones (CRLF); patterns below are written with \n

M = []
def mut(desc, *pairs):
    M.append((desc, list(pairs)))

# ---------------------------------------------------------------- defaults
mut("default K3GPU_CARDS 8 -> 7", ('ei(c"K3GPU_CARDS", 8)', 'ei(c"K3GPU_CARDS", 7)'))
mut("default K3GPU_READERS 16 -> 15", ('ei(c"K3GPU_READERS", 16)', 'ei(c"K3GPU_READERS", 15)'))
mut("default K3GPU_RESERVE_MB 2500 -> 2501", ('ei(c"K3GPU_RESERVE_MB", 2500)', 'ei(c"K3GPU_RESERVE_MB", 2501)'))
mut("default K3GPU_POLICY 1 -> 0", ('ei(c"K3GPU_POLICY", 1)', 'ei(c"K3GPU_POLICY", 0)'))
mut("default K3GPU_HOST_GB 150 -> 151", ('ei(c"K3GPU_HOST_GB", 150)', 'ei(c"K3GPU_HOST_GB", 151)'))
mut("default K3GPU_LOOKAHEAD 92 -> 93", ('ei(c"K3GPU_LOOKAHEAD", 92)', 'ei(c"K3GPU_LOOKAHEAD", 93)'))
mut("default K3GPU_VQ_INFLIGHT 256 -> 255", ('ei(c"K3GPU_VQ_INFLIGHT", 256)', 'ei(c"K3GPU_VQ_INFLIGHT", 255)'))
mut("default K3GPU_VERBOSE 0 -> 1", ('ei(c"K3GPU_VERBOSE", 0)', 'ei(c"K3GPU_VERBOSE", 1)'))
mut("default K3GPU_CHUNKS 0 -> 1", ('ei(c"K3GPU_CHUNKS", 0)', 'ei(c"K3GPU_CHUNKS", 1)'))
mut("default K3GPU_LAYOUT 0 -> 1", ('ei(c"K3GPU_LAYOUT", 0)', 'ei(c"K3GPU_LAYOUT", 1)'))
mut("K3GPU_PREFILL: `!= 0` -> `> 0` (negative values are off)", ('(ei(c"K3GPU_PREFILL", 0) != 0) as c_int', '(ei(c"K3GPU_PREFILL", 0) > 0) as c_int'))
mut("K3GPU_PREFILL: value passed through instead of `!= 0` (2 stays 2)", ('(ei(c"K3GPU_PREFILL", 0) != 0) as c_int', 'ei(c"K3GPU_PREFILL", 0)'))
mut("variable name typo: K3GPU_CHUNKS -> K3GPU_CHUNK", ('c"K3GPU_CHUNKS"', 'c"K3GPU_CHUNK"'))
mut("reads K3GPU_ROUTECACHE's lookahead from K3GPU_LOOKAHEAD even without a route cache (lookahead field initialised from the variable)",
    ('        routecache: ptr::null(),\n        lookahead: 0,\n', '        routecache: ptr::null(),\n        lookahead: ei(c"K3GPU_LOOKAHEAD", 92),\n'))
mut("route cache counts as set when the variable is empty", ('if !unset_or_empty(rc) {', 'if !rc.is_null() {'))

# ---------------------------------------------------------------- shifts, widths, clamps
mut("reserve shift << 20 -> << 30", ('as i64 as usize) << 20;', 'as i64 as usize) << 30;'))
mut("host_bytes shift << 30 -> << 20", ('(host_gb as i64 as usize) << 30;', '(host_gb as i64 as usize) << 20;'))
mut("reserve computed in 32 bits (mb << 20 as i32, then widened)",
    ('let reserve = (ei(c"K3GPU_RESERVE_MB", 2500) as i64 as usize) << 20;', 'let reserve = ei(c"K3GPU_RESERVE_MB", 2500).wrapping_shl(20) as i64 as usize;'))
mut("host_bytes computed in 32 bits (gb << 30 as i32, then widened)",
    ('let host_bytes = (host_gb as i64 as usize) << 30;', 'let host_bytes = host_gb.wrapping_shl(30) as i64 as usize;'))
mut("reserve zero-extends the int (as u32) instead of sign-extending", ('let reserve = (ei(c"K3GPU_RESERVE_MB", 2500) as i64 as usize) << 20;', 'let reserve = (ei(c"K3GPU_RESERVE_MB", 2500) as u32 as usize) << 20;'))
mut("host_bytes zero-extends host_gb", ('(host_gb as i64 as usize) << 30;', '(host_gb as u32 as usize) << 30;'))
mut("vq_cap zero-extends", ('ei(c"K3GPU_VQ_INFLIGHT", 256) as i64 as usize', 'ei(c"K3GPU_VQ_INFLIGHT", 256) as u32 as usize'))
mut("vq_from zero-extends", ('ei(c"K3GPU_VQ_FROM", 0) as c_long', 'ei(c"K3GPU_VQ_FROM", 0) as u32 as c_long'))
mut("vqueue no longer requires policy 2", (' && policy == 2) as c_int', ') as c_int'))
mut("vqueue: `!= 0` -> `== 1`", ('ei(c"K3GPU_VQUEUE", 0) != 0 &&', 'ei(c"K3GPU_VQUEUE", 0) == 1 &&'))
mut("vqueue requires policy >= 2 instead of == 2", ('policy == 2) as c_int', 'policy >= 2) as c_int'))
mut("cards: min with devices dropped", ('.min(devices)', ''))
mut("cards: clamp to EC_MAX_CARDS dropped", ('.min(EC_MAX_CARDS as i32)', ''))
mut("cards: clamp to 9 instead of 8", ('.min(EC_MAX_CARDS as i32)', '.min(9)'))
mut("cards: floor of 1 added (.max(1))", ('.min(EC_MAX_CARDS as i32);', '.min(EC_MAX_CARDS as i32).max(1);'))

# ---------------------------------------------------------------- env_int / atoi
mut("env_int: empty value counts as set (atoi(\"\") = 0)", ('if unset_or_empty(v) {\n        dflt', 'if v.is_null() {\n        dflt'))
mut("env_int: a parsed 0 falls back to the default", ('atoi(unsafe { cbytes(v) })', '{ let r = atoi(unsafe { cbytes(v) }); if r == 0 { dflt } else { r } }'))
mut("atoi: leading whitespace set loses VT and FF", ("b' ' | b'\\t' | b'\\n' | 0x0b | 0x0c | b'\\r'", "b' ' | b'\\t' | b'\\n' | b'\\r'"))
mut("atoi: leading whitespace set also skips 0xa0", ("b' ' | b'\\t' | b'\\n' | 0x0b | 0x0c | b'\\r'", "b' ' | b'\\t' | b'\\n' | 0x0b | 0x0c | b'\\r' | 0xa0"))
mut("atoi: '+' sign not accepted", ("(s[i] == b'-' || s[i] == b'+') {\n        neg", "(s[i] == b'-') {\n        neg"))
mut("atoi: several signs accepted (`if` -> `while`, last one wins)", ("if i < s.len() && (s[i] == b'-' || s[i] == b'+') {", "while i < s.len() && (s[i] == b'-' || s[i] == b'+') {"))
mut("atoi: result saturated to the int range instead of the low 32 bits of the long", ('strtol10(s) as i32', 'strtol10(s).clamp(i32::MIN as i64, i32::MAX as i64) as i32'))
mut("atoi: accumulator wraps on u64 overflow instead of saturating", ('None => over = true,', 'None => acc = acc.wrapping_mul(10).wrapping_add((s[i] - b\'0\') as u64),'))
mut("atoi: negative overflow saturates at LONG_MIN+1", ('if over || acc > (1u64 << 63) {\n            i64::MIN', 'if over || acc > (1u64 << 63) {\n            i64::MIN + 1'))
mut("atoi: positive overflow saturates at LONG_MAX-1", ('{\n        i64::MAX\n    } else {', '{\n        i64::MAX - 1\n    } else {'))
mut("atoi: negative bound `> 2^63` -> `>= 2^63` (-9223372036854775808 takes the saturating branch)", ('acc > (1u64 << 63)', 'acc >= (1u64 << 63)'))
mut("atoi: positive bound `> i64::MAX` -> `>= i64::MAX` (9223372036854775807 takes the saturating branch)", ('acc > i64::MAX as u64', 'acc >= i64::MAX as u64'))
mut("atoi: digit loop stops at the first overflow (the rest of the digits are not consumed; only the value is used afterwards)", ('while i < s.len() && s[i].is_ascii_digit() {', 'while i < s.len() && s[i].is_ascii_digit() && !over {'))

# ---------------------------------------------------------------- pend_cap
mut("pend_cap: 6/10 -> 5/10", ('cap_slots.wrapping_mul(6) / 10', 'cap_slots.wrapping_mul(5) / 10'))
mut("pend_cap: divide first (cap / 10 * 6)", ('cap_slots.wrapping_mul(6) / 10', 'cap_slots / 10 * 6'))
mut("pend_cap: 128-bit product, no wrap", ('cap_slots.wrapping_mul(6) / 10', '((cap_slots as u128 * 6) / 10) as usize'))
mut("pend_cap: 32-bit arithmetic", ('cap_slots.wrapping_mul(6) / 10', '((cap_slots as u32).wrapping_mul(6) / 10) as usize'))
mut("pend_cap: 3/5 instead of 6/10 (same value unless the product wraps)", ('cap_slots.wrapping_mul(6) / 10', 'cap_slots.wrapping_mul(3) / 5'))

# ---------------------------------------------------------------- fatal / selectors / prefill / trace
mut("fatal exit status 1 -> 2", ('emit(b);\n    std::process::exit(1)', 'emit(b);\n    std::process::exit(2)'))
mut("fatal leaves through _exit (no atexit handlers) instead of exit", ('emit(b);\n    std::process::exit(1)', 'emit(b);\n    extern "C" { fn _exit(c: c_int) -> !; }\n    unsafe { _exit(1) }'))
mut("selector: `rust` matched case-insensitively", ('if m_bytes == b"rust" {', 'if m_bytes.eq_ignore_ascii_case(b"rust") {'))
mut("selector: trailing whitespace ignored (`rust `)", ('if m_bytes == b"rust" {', 'if m_bytes.trim_ascii_end() == b"rust" {'))
mut("selector: `cpp` matched case-insensitively", ('let is_cpp = empty || m_bytes == b"cpp";', 'let is_cpp = empty || m_bytes.eq_ignore_ascii_case(b"cpp");'))
mut("selector: unset/empty no longer means cpp", ('let is_cpp = empty || m_bytes == b"cpp";', 'let is_cpp = m_bytes == b"cpp";'))
mut("selector: build without the header accepts `rust` as cpp", ('    if !is_cpp {\n        return SelOutcome::Fatal(\n            Line::new()', '    if !is_cpp && m_bytes != b"rust" {\n        return SelOutcome::Fatal(\n            Line::new()'))
mut("selector: built flag inverted in ec_selector_mode", ('selector_decide(&ProcessEnv, component, built != 0)', 'selector_decide(&ProcessEnv, component, built == 0)'))
mut("selector lib: empty K3GPU_<NAME>_LIB counts as set", ('if unset_or_empty(l) {\n        si.deflib.as_ptr()', 'if l.is_null() {\n        si.deflib.as_ptr()'))
mut("selector lib: default of the slot cache libslotcache.so -> libslotcache.so.1", ('deflib: c"libslotcache.so" }', 'deflib: c"libslotcache.so.1" }'))
mut("selector announce text `rust (` -> `rust: (`", ('.s(" rust (").cs(lib)', '.s(" rust: (").cs(lib)'))
mut("selector: fatal message of a built engine `must be cpp or rust, not` -> `must be cpp or rust:`", ('.s(" must be cpp or rust, not ")', '.s(" must be cpp or rust: ")'))
mut("selector: component variable names of COPYQ and ROUTETABLE swapped", ('var: c"K3GPU_COPYQ", libvar', 'var: c"K3GPU_ROUTETABLE", libvar'))
mut("prefill: `calls == 0` -> `calls <= 1`", ('if calls == 0 {\n        fatal(PREFILL_NEEDS);', 'if calls <= 1 {\n        fatal(PREFILL_NEEDS);'))
mut("prefill: the needs-route-cache case writes the message but does not exit", ('fatal(PREFILL_NEEDS);', 'emit(PREFILL_NEEDS);'))
mut("prefill: ON text `4 records` -> `4 record`", ('expert bytes are 4 records per layer', 'expert bytes are 4 record per layer'))
mut("trace: fopen mode \"w\" -> \"a\"", ('fopen(tr, c"w".as_ptr())', 'fopen(tr, c"a".as_ptr())'))
mut("trace: empty K3GPU_TRACE counts as set", ('if unset_or_empty(tr) {\n        return ptr::null_mut();', 'if tr.is_null() {\n        return ptr::null_mut();'))
mut("trace: failure message `cannot write` -> `cannot open`", ('.s("cannot write ")', '.s("cannot open ")'))
mut("trace: failure returns NULL instead of exiting", ('fatal(&trace_error_line(tr));', 'emit(&trace_error_line(tr)); return ptr::null_mut();'))

# ---------------------------------------------------------------- messages
mut("route cache line: `look-ahead` -> `lookahead`", ('.s(", look-ahead ")', '.s(", lookahead ")'))
mut("route cache line: card queues on/off swapped", ('.s(if s.vqueue != 0 { "on" } else { "off" })', '.s(if s.vqueue != 0 { "off" } else { "on" })'))
mut("route cache line: call count printed as 32 bits", ('.n(calls)\n        .s(" calls from ")', '.n(calls as u32)\n        .s(" calls from ")'))
mut("route cache line: vq_cap printed as 32 bits", ('.n(s.vq_cap)\n        .s(" in flight)\\n")', '.n(s.vq_cap as u32)\n        .s(" in flight)\\n")'))
mut("start line: readers and host_gb swapped", ('.n(s.host_gb).s(" GB, ").n(s.readers)', '.n(s.readers).s(" GB, ").n(s.host_gb)'))

# ---------------------------------------------------------------- card_of
mut("card_of layout 1: (layer - 1) % cards -> layer % cards", ('return d % cards;', 'return layer % cards;'))
mut("card_of layout 1: rem_euclid instead of the C remainder", ('return d % cards;', 'return d.rem_euclid(cards);'))
mut("card_of layout 1: `layer == 0` -> `layer <= 0`", ('        if layer == 0 {\n            return 0;\n        }\n        let d', '        if layer <= 0 {\n            return 0;\n        }\n        let d'))
mut("card_of: layout test `== 1` -> `!= 0`", ('    if layout == 1 {\n        if layer == 0 {', '    if layout != 0 {\n        if layer == 0 {'))
mut("card_of grouped: divisor 93 -> 92", ('layer.wrapping_mul(cards) / 93)', 'layer.wrapping_mul(cards) / 92)'))
mut("card_of grouped: the min with cards - 1 dropped", ('cards.wrapping_sub(1).min(layer.wrapping_mul(cards) / 93)', 'layer.wrapping_mul(cards) / 93'))
mut("card_of grouped: the layer 0 shortcut dropped", ('    if layer == 0 {\n        0\n    } else {\n        cards.wrapping_sub(1).min(layer.wrapping_mul(cards) / 93)\n    }', '    cards.wrapping_sub(1).min(layer.wrapping_mul(cards) / 93)'))
mut("card_of grouped: product in 64 bits (no int wrap)", ('layer.wrapping_mul(cards) / 93)', '((layer as i64 * cards as i64) / 93) as i32)'))
mut("card_of: `layer.wrapping_sub(1)` -> `layer - 1` (release profile: no overflow check, so wraps all the same)", ('let d = layer.wrapping_sub(1);', 'let d = layer - 1;'))
mut("card_of: `layer.wrapping_mul(cards)` -> `layer * cards` (same, release profile)", ('layer.wrapping_mul(cards) / 93)', '(layer * cards) / 93)'))
mut("card_of: explicit trap check removed (the Rust `%` itself panics on 0 and MIN % -1)", ('std::process::abort(); // the original traps (SIGFPE) on exactly these inputs', '/* no explicit check */'))

# ---------------------------------------------------------------- force_routes
mut("force: experts dealt in reverse order", ('for i in 0..experts.len() {\n        for _ in 0..counts[i] {', 'for i in (0..experts.len()).rev() {\n        for _ in 0..counts[i] {'))
mut("force: dealing counter s restarts for every expert", ('for i in 0..experts.len() {\n        for _ in 0..counts[i] {', 'for i in 0..experts.len() {\n        s = 0;\n        for _ in 0..counts[i] {'))
mut("force: token index (s + 1) % n", ('let t = (s % n as i64) as usize;', 'let t = ((s + 1) % n as i64) as usize;'))
mut("force: selected index = fill * top + token (operands swapped)", ('let idx = t.wrapping_mul(top as isize as usize).wrapping_add(fill[t] as isize as usize);', 'let idx = (fill[t] as isize as usize).wrapping_mul(top as isize as usize).wrapping_add(t);'))
mut("force: a count of 0 or less still deals once (counts[i].max(1))", ('for _ in 0..counts[i] {', 'for _ in 0..counts[i].max(1) {'))
mut("force: sum test `total != n*top` -> `total > n*top`", ('total != (n as i64) * (top as i64)', 'total > (n as i64) * (top as i64)'))
mut("force: `k > n` -> `k >= n`", ('if k > n {', 'if k >= n {'))
mut("force: the `fits` (count > n) check removed", ('if !fits || total != (n as i64) * (top as i64) {', 'if total != (n as i64) * (top as i64) {'))
mut("force: layer mismatch not checked", ('if pos >= size || (*cur).layer != l {', 'if pos >= size {'))
mut("force: `pos >= size` -> `pos > size` (reads cur at pos == size)", ('if pos >= size || (*cur).layer != l {', 'if pos > size || (*cur).layer != l {'))
mut("force: counts check `!=` -> `<` (more counts than experts accepted)", ('if c.n_counts != c.n_experts {', 'if c.n_counts < c.n_experts {'))
mut("force: the warning is written on every rejected call", ('if !WARNED.swap(true, Ordering::Relaxed) {', 'if { WARNED.swap(true, Ordering::Relaxed); true } {'))
mut("force: warning text `(layer ` -> `(L `", ('.s(" (layer ")', '.s(" (L ")'))
mut("force: total accumulated in i32 (wrapping)", ('let mut total: i64 = 0;', 'let mut total: i32 = 0;'), ('total += k as i64;', 'total = total.wrapping_add(k);'), ('if !fits || total != (n as i64)', 'if !fits || (total as i64) != (n as i64)'))
mut("force: n * top computed in i32 (wrapping)", ('total != (n as i64) * (top as i64)', 'total != n.wrapping_mul(top) as i64'))
mut("force: fill counters are u8", ('let mut fill = vec![0i32; n as usize];', 'let mut fill = vec![0u8; n as usize];'))
mut("force: fill counters are u16", ('let mut fill = vec![0i32; n as usize];', 'let mut fill = vec![0u16; n as usize];'))
mut("force: dealing counter s is i32 (wrapping)", ('let mut s: i64 = 0;', 'let mut s: i32 = 0;'), ('let t = (s % n as i64) as usize;', 'let t = (s as i64 % n as i64) as usize;'), ('            s += 1;', '            s = s.wrapping_add(1);'))
mut("force: top zero-extended in the index (top as u32)", ('t.wrapping_mul(top as isize as usize)', 't.wrapping_mul(top as u32 as usize)'))
mut("force: fill value zero-extended in the index (as u32)", ('fill[t] as isize as usize)', 'fill[t] as u32 as usize)'))
mut("force: explicit n < 0 abort removed (vec![0; n as usize] then aborts on capacity)", ('    if n < 0 {\n        std::process::abort(); // std::vector<int> fill(n, 0) throws std::length_error in the original\n    }\n', ''))

# ---------------------------------------------------------------- print_summary
mut("summary: pct of cards `100.0 * hc / need` -> `hc / need * 100.0`", ('100.0 * hcf / needf', 'hcf / needf * 100.0'))
mut("summary: pct of host `100.0 * hh / need` -> `100.0 * (hh / need)`", ('100.0 * hhf / needf', '100.0 * (hhf / needf)'))
mut("summary: pct of drive `100.0 * dr / need` -> `dr / need * 100.0`", ('100.0 * drf / needf', 'drf / needf * 100.0'))
mut("summary: GB `dr * raw / 1e9` -> `dr / 1e9 * raw`", ('drf * (s.expert_raw as f64) / 1e9', 'drf / 1e9 * (s.expert_raw as f64)'))
mut("summary: GB `/ 1e9` -> `* 1e-9`", ('drf * (s.expert_raw as f64) / 1e9', 'drf * (s.expert_raw as f64) * 1e-9'))
mut("summary: GB unit 1e9 -> 2^30", ('/ 1e9', '/ 1073741824.0'))
mut("summary: ms each `1000 * t / n` -> `t / n * 1000`", ('1000.0 * s.t_calls[i] / s.n_calls[i]', 's.t_calls[i] / s.n_calls[i] * 1000.0'))
mut("summary: ms each `1000 * t / n` -> `1000 * (t / n)`", ('1000.0 * s.t_calls[i] / s.n_calls[i]', '1000.0 * (s.t_calls[i] / s.n_calls[i])'))
mut("summary: ms each `1000 * t / n` -> `t * 1000 / n` (commutative: bit-identical)", ('1000.0 * s.t_calls[i] / s.n_calls[i]', 's.t_calls[i] * 1000.0 / s.n_calls[i]'))
mut("summary: ms each `1000 * t / n` -> `t / (n / 1000)`", ('1000.0 * s.t_calls[i] / s.n_calls[i]', 's.t_calls[i] / (s.n_calls[i] / 1000.0)'))
mut("summary: read/copy seconds summed in reverse card order", ('    for k in &s.card[..cards] {\n        if k.ready == 0 {', '    for k in s.card[..cards].iter().rev() {\n        if k.ready == 0 {'))
mut("summary: sums of not-ready cards included in need (ready test dropped in the first loop)", ('        if k.ready == 0 {\n            continue;\n        }\n', ''))
mut("summary: need sum saturates instead of wrapping", ('need = need.wrapping_add(k.needed);', 'need = need.saturating_add(k.needed);'))
mut("summary: `need == 0` -> `need <= 1`", ('if need == 0 {\n        return out;', 'if need <= 1 {\n        return out;'))
mut("summary: per-card line numbered c + 1", ('.s("K3GPU card ")\n                    .n(c)', '.s("K3GPU card ")\n                    .n(c + 1)'))
mut("summary: per-card lines also for cards that are not ready", ('        if k.ready != 0 {\n            out.push(', '        {\n            out.push('))
mut("summary: experts line counts of card and host swapped", ('.n(hc)\n            .s(" already on the card (")', '.n(hh)\n            .s(" already on the card (")'))
mut("summary: pct on the card %.1f -> %.2f", ('.f(1, 100.0 * hcf / needf)', '.f(3, 100.0 * hcf / needf)'))
mut("summary: pct denominator = hc + hh + dr instead of need", ('100.0 * hcf / needf', '100.0 * hcf / ((hc.wrapping_add(hh).wrapping_add(dr)) as f64)'))
mut("summary: route cache line condition rc_calls != 0 -> rc_pos != 0", ('if s.rc_calls != 0 {', 'if s.rc_pos != 0 {'))
mut("summary: look-ahead on/stopped swapped", ('if s.rc_ok != 0 { "on to the end" } else { "stopped" }', 'if s.rc_ok != 0 { "stopped" } else { "on to the end" }'))
mut("summary: queue line condition `vq != 0` -> `vq == 1`", ('if s.vq != 0 {', 'if s.vq == 1 {'))
mut("summary: queue total saturates instead of wrapping", ('s.vq_ram.wrapping_add(s.vq_drive)', 's.vq_ram.saturating_add(s.vq_drive)'))
mut("summary: call class test `!= 0.0` -> `> 0.0`", ('if s.n_calls[i] != 0.0 {', 'if s.n_calls[i] > 0.0 {'))
mut("summary: call count %.0f -> %.1f", ('.f(0, s.n_calls[i])', '.f(1, s.n_calls[i])'))
mut("summary: total seconds %.3f -> %.2f", ('.f(3, s.t_calls[i])', '.f(1, s.t_calls[i])'))
mut("summary: route table line rc_calls printed as 32 bits", ('.n(s.rc_calls).s(" calls, ")', '.n(s.rc_calls as u32).s(" calls, ")'))
mut("summary: cards > EC_MAX_CARDS check `>` -> `>=`", ('if s.cards > EC_MAX_CARDS as c_int {', 'if s.cards >= EC_MAX_CARDS as c_int {'))
mut("summary: cards < 0 not clamped to 0", ('let cards = s.cards.max(0) as usize;', 'let cards = s.cards as usize;'))
mut("summary: copy queue cpp/rust words swapped", ('let which = |b: c_int| if b != 0 { "rust" } else { "cpp" };', 'let which = |b: c_int| if b != 0 { "cpp" } else { "rust" };'))

# ---------------------------------------------------------------- output plumbing
mut("doubles formatted by Rust format! instead of libc snprintf", ('    fn f(&mut self, prec: u8, v: f64) -> &mut Line {\n', '    fn f(&mut self, prec: u8, v: f64) -> &mut Line {\n        if true { return self.s(&format!("{:.*}", prec as usize, v)); }\n'))
mut("doubles: the formatted text truncated at 63 bytes", ('let len = len.max(0) as usize;', 'let len = (len.max(0) as usize).min(63);'))
mut("emit: each line written with two write calls (halves)", ('fn emit(b: &[u8]) {\n    let _ = std::io::stderr().write_all(b);', 'fn emit(b: &[u8]) {\n    let h = b.len() / 2;\n    let _ = std::io::stderr().write_all(&b[..h]);\n    let _ = std::io::stderr().write_all(&b[h..]);'))
# added after the first campaign run had started (numbering of M01..M138 unchanged; MK_ONLY="M139 M140" builds only these)
mut("atoi: strict parse, trailing junk after the digits gives 0", ('    if neg {\n        if over ||', '    if i < s.len() {\n        return 0;\n    }\n    if neg {\n        if over ||'))
mut("atoi: a 0x / 0X prefix is skipped (so 0x10 gives 10, not 0)", ('    let mut acc: u64 = 0;\n    let mut over = false;\n', '    if i + 1 < s.len() && s[i] == b\'0\' && (s[i + 1] == b\'x\' || s[i + 1] == b\'X\') {\n        i += 2;\n    }\n    let mut acc: u64 = 0;\n    let mut over = false;\n'))

os.makedirs(os.path.join(WORK, "diffs"), exist_ok=True)
lines_real = src.splitlines(keepends=True)
ok = True
# M00 = control: unmodified copy
entries = [("M00", "control: unmodified copy of src/lib.rs", [])]
for i, (desc, pairs) in enumerate(M, start=1):
    entries.append(("M%02d" % i, desc, pairs))
for name, desc, pairs in entries:
    if os.environ.get("MK_ONLY") and name not in os.environ["MK_ONLY"].split():
        continue
    d = os.path.join(WORK, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(os.path.join(d, "src"))
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(REAL, f), os.path.join(d, f))
    shutil.copytree(os.path.join(REAL, "include"), os.path.join(d, "include"))
    shutil.copy(os.path.join(REAL, "src/tests.rs"), os.path.join(d, "src/tests.rs"))
    out = src
    bad = False
    for old, new in pairs:
        old = old.replace("\n", NL); new = new.replace("\n", NL)
        if src.count(old) != 1:
            print("ERROR", name, "pattern occurs", src.count(old), "times:", repr(old)); ok = False; bad = True
        out = out.replace(old, new)
    if bad:
        continue
    open(os.path.join(d, "src/lib.rs"), "wb").write(out.encode())
    diff = list(difflib.unified_diff(lines_real, out.splitlines(keepends=True), "src/lib.rs (real)", f"{name}/src/lib.rs", n=2))
    hunks = sum(1 for l in diff if l.startswith("@@"))
    open(os.path.join(WORK, "diffs", name + ".diff"), "w", newline="\n").write("".join(diff))
    open(os.path.join(WORK, "diffs", name + ".txt"), "w", newline="\n").write(desc + "\n")
    print(name, "hunks", hunks, "-", desc)
print("TOTAL", len(entries), "mutants incl. control")
sys.exit(0 if ok else 1)
