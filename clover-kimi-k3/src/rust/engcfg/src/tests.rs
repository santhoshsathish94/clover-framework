//! Unit tests. Expected values are derived by hand from the original text (backend-gpu.cu / decode.cuh at 4d8d891) and from include/engcfg.h, not from running the C++ reference.
//! Tests that exit or write to the real stderr run the test binary again as a child (env ENGCFG_CHILD) and check status and bytes.

use super::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::mem::{offset_of, size_of};
use std::os::unix::ffi::OsStrExt;
use std::process::Command;

struct MapEnv(HashMap<String, CString>);

impl MapEnv {
    fn new(pairs: &[(&str, &str)]) -> MapEnv {
        MapEnv(pairs.iter().map(|(k, v)| (k.to_string(), CString::new(*v).unwrap())).collect())
    }
}

impl EnvSource for MapEnv {
    fn get(&self, name: &CStr) -> *const c_char {
        self.0.get(name.to_str().unwrap()).map_or(ptr::null(), |c| c.as_ptr())
    }
}

fn s(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

// ---------------------------------------------------------------------------------------------------------------------------------- atoi / env_int

#[test]
fn atoi_whitespace_sign_digits() {
    assert_eq!(atoi(b"12"), 12);
    assert_eq!(atoi(b""), 0);
    assert_eq!(atoi(b" \t\n\x0b\x0c\r 7"), 7);
    assert_eq!(atoi(b"+5"), 5);
    assert_eq!(atoi(b"-5"), -5);
    assert_eq!(atoi(b"+-5"), 0);
    assert_eq!(atoi(b"- 5"), 0);
    assert_eq!(atoi(b"0x10"), 0);
    assert_eq!(atoi(b"007"), 7);
    assert_eq!(atoi(b"12abc"), 12);
    assert_eq!(atoi(b"abc"), 0);
    assert_eq!(atoi(b"1 2"), 1);
    assert_eq!(atoi(b"\xa0 5"), 0, "0xa0 is not C-locale whitespace");
}

#[test]
fn atoi_cast_keeps_low_32_bits() {
    assert_eq!(atoi(b"2147483647"), i32::MAX);
    assert_eq!(atoi(b"2147483648"), i32::MIN);
    assert_eq!(atoi(b"4294967295"), -1);
    assert_eq!(atoi(b"4294967296"), 0);
    assert_eq!(atoi(b"4294967297"), 1);
    assert_eq!(atoi(b"-2147483648"), i32::MIN);
    assert_eq!(atoi(b"-2147483649"), i32::MAX); // 0x7fffffff
    assert_eq!(atoi(b"99999999999"), 1_215_752_191); // 23 * 2^32 + 1215752191
}

#[test]
fn atoi_saturates_like_strtol() {
    assert_eq!(strtol10(b"9223372036854775807"), i64::MAX);
    assert_eq!(strtol10(b"9223372036854775808"), i64::MAX);
    assert_eq!(strtol10(b"-9223372036854775808"), i64::MIN);
    assert_eq!(strtol10(b"-9223372036854775809"), i64::MIN);
    assert_eq!(strtol10(b"18446744073709551616"), i64::MAX); // 2^64
    assert_eq!(strtol10(b"-18446744073709551616"), i64::MIN);
    assert_eq!(strtol10(b"99999999999999999999999"), i64::MAX);
    assert_eq!(atoi(b"9223372036854775808"), -1); // (int)LONG_MAX
    assert_eq!(atoi(b"-9223372036854775809"), 0); // (int)LONG_MIN
    assert_eq!(atoi(b"00000000000000000000000000000000000000000000000000000000000000000005"), 5, "leading zeros never overflow");
}

#[test]
fn env_int_unset_and_empty_give_the_default_everything_else_parses() {
    let e = MapEnv::new(&[("EMPTY", ""), ("ZERO", "0"), ("X", "x"), ("NEG", "-1"), ("SP", "  9")]);
    assert_eq!(env_int(&e, c"UNSET", 42), 42);
    assert_eq!(env_int(&e, c"EMPTY", -7), -7);
    assert_eq!(env_int(&e, c"ZERO", 42), 0, "0 is a value, not the default");
    assert_eq!(env_int(&e, c"X", 42), 0, "no digits gives 0, not the default");
    assert_eq!(env_int(&e, c"NEG", 42), -1);
    assert_eq!(env_int(&e, c"SP", 42), 9);
}

// ---------------------------------------------------------------------------------------------------------------------------------- settings

#[test]
fn defaults() {
    let e = MapEnv::new(&[]);
    let t = settings_from(&e, 4);
    assert_eq!((t.cards, t.readers, t.verbose), (4, 16, 0));
    assert_eq!(t.reserve, 2_621_440_000); // 2500 << 20
    assert_eq!((t.policy, t.chunks, t.host_gb), (1, 0, 150));
    assert_eq!(t.host_bytes, 161_061_273_600); // 150 << 30
    assert!(t.routecache.is_null());
    assert_eq!((t.lookahead, t.vqueue, t.vq_cap, t.vq_from), (0, 0, 256, 0));
    assert_eq!((t.layout, t.prefill), (0, 0));
}

#[test]
fn cards_is_min_of_env_devices_and_eight() {
    let c = |env: &[(&str, &str)], dev| settings_from(&MapEnv::new(env), dev).cards;
    assert_eq!(c(&[], 16), 8);
    assert_eq!(c(&[("K3GPU_CARDS", "20")], 16), 8);
    assert_eq!(c(&[("K3GPU_CARDS", "3")], 16), 3);
    assert_eq!(c(&[("K3GPU_CARDS", "3")], 2), 2);
    assert_eq!(c(&[], 0), 0);
    assert_eq!(c(&[], -3), -3);
    assert_eq!(c(&[("K3GPU_CARDS", "-5")], 8), -5);
    assert_eq!(c(&[("K3GPU_CARDS", "")], 8), 8, "empty = default 8");
    assert_eq!(c(&[("K3GPU_CARDS", "x")], 8), 0, "no digits = 0");
    assert_eq!(c(&[("K3GPU_CARDS", "4294967299")], 8), 3, "low 32 bits");
}

#[test]
fn reserve_and_host_bytes_sign_extend_then_shift() {
    let g = |env: &[(&str, &str)]| settings_from(&MapEnv::new(env), 8);
    assert_eq!(g(&[("K3GPU_RESERVE_MB", "-1")]).reserve, 18_446_744_073_708_503_040); // 2^64 - 2^20
    assert_eq!(g(&[("K3GPU_RESERVE_MB", "2147483647")]).reserve, 2_251_799_812_636_672); // 2^51 - 2^20
    assert_eq!(g(&[("K3GPU_RESERVE_MB", "2147483648")]).reserve, 18_444_492_273_895_866_368); // (size_t)INT_MIN << 20 = 2^64 - 2^51
    assert_eq!(g(&[("K3GPU_RESERVE_MB", "4294967296")]).reserve, 0);
    assert_eq!(g(&[("K3GPU_RESERVE_MB", "0x10")]).reserve, 0);
    assert_eq!(g(&[("K3GPU_HOST_GB", "-1")]).host_bytes, 18_446_744_072_635_809_792); // 2^64 - 2^30
    assert_eq!(g(&[("K3GPU_HOST_GB", "-1")]).host_gb, -1);
    assert_eq!(g(&[("K3GPU_HOST_GB", "2147483647")]).host_bytes, 2_305_843_008_139_952_128); // 2^61 - 2^30
    assert_eq!(g(&[("K3GPU_HOST_GB", "0")]).host_bytes, 0);
    assert_eq!(g(&[("K3GPU_HOST_GB", "+5")]).host_bytes, 5 << 30);
}

#[test]
fn plain_ints() {
    let t = settings_from(&MapEnv::new(&[("K3GPU_READERS", "0x10"), ("K3GPU_VERBOSE", " 3"), ("K3GPU_POLICY", "-2"), ("K3GPU_CHUNKS", "9999999999"), ("K3GPU_LAYOUT", "7")]), 8);
    assert_eq!(t.readers, 0);
    assert_eq!(t.verbose, 3);
    assert_eq!(t.policy, -2);
    assert_eq!(t.chunks, 1_410_065_407, "9999999999 = 2*2^32 + 1410065407");
    assert_eq!(t.layout, 7, "layout is kept as read; only the value 1 is special in card_of");
}

#[test]
fn prefill_is_nonzero() {
    let p = |v: &str| settings_from(&MapEnv::new(&[("K3GPU_PREFILL", v)]), 8).prefill;
    assert_eq!((p("0"), p(""), p("0x1"), p("x")), (0, 0, 0, 0));
    assert_eq!((p("1"), p("-1"), p(" 2"), p("7")), (1, 1, 1, 1));
    assert_eq!(p("4294967296"), 0, "low 32 bits are zero");
}

#[test]
fn route_cache_fields_only_with_a_nonempty_route_cache() {
    let all = [("K3GPU_POLICY", "2"), ("K3GPU_VQUEUE", "3"), ("K3GPU_VQ_INFLIGHT", "-1"), ("K3GPU_VQ_FROM", "92"), ("K3GPU_LOOKAHEAD", "10")];
    let t = settings_from(&MapEnv::new(&all), 8);
    assert!(t.routecache.is_null());
    assert_eq!((t.lookahead, t.vqueue, t.vq_cap, t.vq_from), (0, 0, 256, 0), "without K3GPU_ROUTECACHE the four are never read");
    let mut with = all.to_vec();
    with.push(("K3GPU_ROUTECACHE", ""));
    let t = settings_from(&MapEnv::new(&with), 8);
    assert!(t.routecache.is_null() && t.vqueue == 0 && t.lookahead == 0, "empty = unset");
    with.pop();
    with.push(("K3GPU_ROUTECACHE", "/x/y"));
    let env = MapEnv::new(&with);
    let t = settings_from(&env, 8);
    assert_eq!(unsafe { cbytes(t.routecache) }, b"/x/y");
    assert_eq!(t.routecache, env.get(c"K3GPU_ROUTECACHE"), "a pointer into the environment, not a copy");
    assert_eq!((t.lookahead, t.vqueue, t.vq_cap, t.vq_from), (10, 1, usize::MAX, 92));
}

#[test]
fn vqueue_needs_policy_2_and_a_nonzero_value() {
    let q = |pol: &str, vq: &str| settings_from(&MapEnv::new(&[("K3GPU_ROUTECACHE", "r"), ("K3GPU_POLICY", pol), ("K3GPU_VQUEUE", vq)]), 8).vqueue;
    assert_eq!((q("2", "1"), q("2", "-1"), q("2", " 5"), q("2", "0"), q("2", "x"), q("2", "")), (1, 1, 1, 0, 0, 0));
    assert_eq!((q("1", "1"), q("0", "1"), q("3", "1"), q("", "1")), (0, 0, 0, 0), "only policy 2; empty policy = default 1");
}

#[test]
fn route_cache_defaults_and_wide_values() {
    let g = |env: &[(&str, &str)]| {
        let mut v = vec![("K3GPU_ROUTECACHE", "r")];
        v.extend_from_slice(env);
        settings_from(&MapEnv::new(&v), 8)
    };
    let t = g(&[]);
    assert_eq!((t.lookahead, t.vqueue, t.vq_cap, t.vq_from), (92, 0, 256, 0));
    assert_eq!(g(&[("K3GPU_VQ_INFLIGHT", "4294967296")]).vq_cap, 0);
    assert_eq!(g(&[("K3GPU_VQ_INFLIGHT", "2147483648")]).vq_cap, 0xFFFF_FFFF_8000_0000, "(size_t)INT_MIN");
    assert_eq!(g(&[("K3GPU_VQ_FROM", "-7")]).vq_from, -7);
    assert_eq!(g(&[("K3GPU_VQ_FROM", "2147483648")]).vq_from, -2_147_483_648, "(long)(int)");
    assert_eq!(g(&[("K3GPU_LOOKAHEAD", "x")]).lookahead, 0);
    assert_eq!(g(&[("K3GPU_LOOKAHEAD", "")]).lookahead, 92);
}

#[test]
fn pend_cap_is_six_tenths() {
    assert_eq!(ec_pend_cap(1000), 600);
    assert_eq!(ec_pend_cap(7), 4); // 42 / 10
    assert_eq!(ec_pend_cap(0), 0);
    assert_eq!(ec_pend_cap(9), 5); // 54 / 10
    assert_eq!(ec_pend_cap(usize::MAX), 1_844_674_407_370_955_161, "6 * (2^64 - 1) wraps to 2^64 - 6 = 18446744073709551610, / 10");
}

// ---------------------------------------------------------------------------------------------------------------------------------- messages

fn base_settings() -> EcSettings {
    settings_from(&MapEnv::new(&[]), 8)
}

#[test]
fn route_cache_line_text() {
    let mut st = base_settings();
    let p = CString::new("/r.trace").unwrap();
    st.routecache = p.as_ptr();
    st.lookahead = 92;
    st.vqueue = 1;
    st.vq_cap = 256;
    assert_eq!(
        s(&route_cache_line(&st, 5612, 1234)),
        "K3GPU route cache: 5612 calls from /r.trace, look-ahead 92 calls, at most 1234 experts read ahead, card queues on (at most 256 in flight)\n"
    );
    st.vqueue = 0;
    st.lookahead = -3;
    st.vq_cap = usize::MAX;
    assert_eq!(
        s(&route_cache_line(&st, 0, 0)),
        "K3GPU route cache: 0 calls from /r.trace, look-ahead -3 calls, at most 0 experts read ahead, card queues off (at most 18446744073709551615 in flight)\n"
    );
    st.routecache = ptr::null();
    assert!(s(&route_cache_line(&st, 1, 1)).contains("calls from (null),"), "glibc prints NULL as (null)");
}

#[test]
fn route_cache_line_keeps_non_utf8_path_bytes() {
    let mut st = base_settings();
    let p = CString::new(vec![b'/', 0xFF, 0xFE, b'x']).unwrap();
    st.routecache = p.as_ptr();
    let l = route_cache_line(&st, 1, 2);
    let at = l.windows(4).position(|w| w == [b'/', 0xFF, 0xFE, b'x']);
    assert!(at.is_some());
}

#[test]
fn start_line_text() {
    let mut st = base_settings();
    assert_eq!(s(&start_line(&st)), "K3GPU: 8 cards, host expert cache 150 GB, 16 drive readers\n");
    st.cards = -1;
    st.host_gb = 0;
    st.readers = 0;
    assert_eq!(s(&start_line(&st)), "K3GPU: -1 cards, host expert cache 0 GB, 0 drive readers\n");
}

#[test]
fn fixed_prefill_texts() {
    assert_eq!(s(PREFILL_NEEDS), "K3GPU_PREFILL needs K3GPU_ROUTECACHE\n");
    assert_eq!(
        s(PREFILL_ON),
        "K3GPU PREFILL ON: timing only. No drive is read, expert bytes are 4 records per layer, routes are taken from the route cache; outputs are not the model's.\n"
    );
}

// ---------------------------------------------------------------------------------------------------------------------------------- selectors

fn decide(env: &[(&str, &str)], comp: c_int, built: bool) -> String {
    match selector_decide(&MapEnv::new(env), comp, built) {
        SelOutcome::Cpp => format!("cpp|{}", s(&selector_cpp_line(comp))),
        SelOutcome::Rust => "rust".to_string(),
        SelOutcome::Fatal(m) => format!("fatal|{}", s(&m)),
    }
}

#[test]
fn selector_built() {
    let cq = |v: Option<&str>| match v {
        Some(v) => decide(&[("K3GPU_COPYQ", v)], 0, true),
        None => decide(&[], 0, true),
    };
    assert_eq!(cq(None), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(cq(Some("")), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(cq(Some("cpp")), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(cq(Some("rust")), "rust");
    assert_eq!(cq(Some("Rust")), "fatal|K3GPU_COPYQ must be cpp or rust, not Rust\n");
    assert_eq!(cq(Some("rust ")), "fatal|K3GPU_COPYQ must be cpp or rust, not rust \n");
    assert_eq!(cq(Some(" cpp")), "fatal|K3GPU_COPYQ must be cpp or rust, not  cpp\n");
    assert_eq!(cq(Some("cpp\n")), "fatal|K3GPU_COPYQ must be cpp or rust, not cpp\n\n");
    assert_eq!(cq(Some("1")), "fatal|K3GPU_COPYQ must be cpp or rust, not 1\n");
    assert_eq!(decide(&[("K3GPU_SLOTCACHE", "go")], 1, true), "fatal|K3GPU_SLOTCACHE must be cpp or rust, not go\n");
    assert_eq!(decide(&[("K3GPU_SLOTCACHE", "rust")], 1, true), "rust");
    assert_eq!(decide(&[], 1, true), "cpp|K3GPU: slot cache cpp\n");
    assert_eq!(decide(&[("K3GPU_ROUTETABLE", "x")], 2, true), "fatal|K3GPU_ROUTETABLE must be cpp or rust, not x\n");
    assert_eq!(decide(&[("K3GPU_ROUTETABLE", "cpp")], 2, true), "cpp|K3GPU: route table cpp\n");
    assert_eq!(decide(&[("K3GPU_COPYQ", "rust"), ("K3GPU_SLOTCACHE", "bad")], 1, true), "fatal|K3GPU_SLOTCACHE must be cpp or rust, not bad\n", "each component reads its own variable");
}

#[test]
fn selector_not_built() {
    assert_eq!(decide(&[], 0, false), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(decide(&[("K3GPU_COPYQ", "")], 0, false), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(decide(&[("K3GPU_COPYQ", "cpp")], 0, false), "cpp|K3GPU: copy queue cpp\n");
    assert_eq!(
        decide(&[("K3GPU_COPYQ", "rust")], 0, false),
        "fatal|K3GPU_COPYQ=rust: this build has no copyq.h, only the C++ queue (build with -I src/rust/copyq/include)\n"
    );
    assert_eq!(
        decide(&[("K3GPU_SLOTCACHE", "Cpp")], 1, false),
        "fatal|K3GPU_SLOTCACHE=Cpp: this build has no slotcache.h, only the C++ cache (build with -I src/rust/slotcache/include)\n"
    );
    assert_eq!(
        decide(&[("K3GPU_ROUTETABLE", "rust")], 2, false),
        "fatal|K3GPU_ROUTETABLE=rust: this build has no routetable.h, only the C++ parser (build with -I src/rust/routetable/include)\n"
    );
}

#[test]
fn selector_lib_and_announce() {
    let e = MapEnv::new(&[("K3GPU_SLOTCACHE_LIB", "/tmp/x.so"), ("K3GPU_ROUTETABLE_LIB", "")]);
    assert_eq!(unsafe { cbytes(selector_lib_from(&e, 1)) }, b"/tmp/x.so");
    assert_eq!(unsafe { cbytes(selector_lib_from(&e, 2)) }, b"libroutetable.so", "empty = default");
    assert_eq!(unsafe { cbytes(selector_lib_from(&e, 0)) }, b"libcopyq.so");
    assert_eq!(unsafe { cbytes(selector_lib_from(&MapEnv::new(&[]), 1)) }, b"libslotcache.so");
    let lib = CString::new("/a b/libcq.so").unwrap();
    assert_eq!(s(&selector_announce_line(0, lib.as_ptr())), "K3GPU: copy queue rust (/a b/libcq.so)\n");
    assert_eq!(s(&selector_announce_line(1, lib.as_ptr())), "K3GPU: slot cache rust (/a b/libcq.so)\n");
    assert_eq!(s(&selector_announce_line(2, ptr::null())), "K3GPU: route table rust ((null))\n");
}

// ---------------------------------------------------------------------------------------------------------------------------------- card_of

#[test]
fn card_of_layout_0() {
    // L * cards / 93, capped at cards - 1 (C division truncates toward zero)
    assert_eq!([0, 1, 12, 13, 50, 92].map(|l| card_of(0, 8, l)), [0, 0, 1, 1, 4, 7]); // 0 | 8/93 | 96/93 | 104/93 | 400/93 | 736/93
    assert_eq!([0, 30, 31, 46, 47, 62, 63, 92].map(|l| card_of(0, 3, l)), [0, 0, 1, 1, 1, 2, 2, 2]); // 90, 93, 138, 141, 186, 189, 276 / 93
    assert_eq!(card_of(0, 8, 93), 7, "744/93 = 8, capped");
    assert_eq!(card_of(0, 8, 1000), 7);
    assert_eq!(card_of(0, 8, -1), 0, "-8/93 truncates to 0");
    assert_eq!(card_of(0, 8, -94), -8, "-752/93 = -8");
    assert_eq!(card_of(0, 1, 50), 0);
    assert_eq!(card_of(0, 0, 5), -1, "min(-1, 0)");
    assert_eq!(card_of(0, 0, 0), 0);
    assert_eq!(card_of(2, 8, 50), 4, "layout 2 is not 1: the grouped layout");
    assert_eq!(card_of(-1, 8, 50), 4);
    assert_eq!(card_of(0, 8, i32::MAX), 0, "2147483647 * 8 wraps to -8, -8/93 = 0");
}

#[test]
fn card_of_layout_1() {
    assert_eq!([0, 1, 2, 8, 9, 10, 92].map(|l| card_of(1, 8, l)), [0, 0, 1, 7, 0, 1, 3]); // (L-1) % 8: 0, 1, 7, 8%8, 9%8, 91%8
    assert_eq!([0, 1, 2, 3, 4, 5].map(|l| card_of(1, 3, l)), [0, 0, 1, 2, 0, 1]);
    assert_eq!(card_of(1, 1, 77), 0);
    assert_eq!(card_of(1, 8, -5), -6, "(-6) % 8 keeps the dividend's sign");
    assert_eq!(card_of(1, 8, i32::MIN), 7, "MIN - 1 wraps to 2147483647, % 8 = 7");
    assert_eq!(card_of(1, -1, 5), 0);
    assert_eq!(card_of(1, -3, 5), 1, "4 % -3 = 1");
}

// ---------------------------------------------------------------------------------------------------------------------------------- force_routes

fn call<'a>(layer: i32, e: &'a [i32], c: &'a [i32]) -> EcCall {
    EcCall { layer, experts: e.as_ptr(), n_experts: e.len(), counts: c.as_ptr(), n_counts: c.len() }
}

#[test]
fn force_routes_deals_round_robin() {
    // top 2, n 3: counts 3,2,1 -> s = 0,1,2 / 3,4 / 5 -> tokens 0,1,2 / 0,1 / 2
    let e = [10, 20, 30];
    let c = [3, 2, 1];
    let k = call(7, &e, &c);
    let mut sel = [-1i32; 8];
    unsafe { force_routes(0, 1, &k, 7, 3, sel.as_mut_ptr(), 2) };
    assert_eq!(sel, [10, 20, 10, 20, 10, 30, -1, -1]);
}

#[test]
fn force_routes_one_expert_per_token_slot() {
    // top 3, n 2: counts 2,2,2 -> experts 5,6,7 each to both tokens: s = 0,1 / 2,3 / 4,5 -> t = 0,1 / 0,1 / 0,1
    let e = [5, 6, 7];
    let c = [2, 2, 2];
    let mut sel = [0i32; 6];
    unsafe { force_routes(4, 9, &call(1, &e, &c), 1, 2, sel.as_mut_ptr(), 3) };
    assert_eq!(sel, [5, 6, 7, 5, 6, 7]);
}

#[test]
fn force_routes_s_runs_on_across_experts() {
    // top 2, n 3: counts 1,3,2: s = 0 / 1,2,3 / 4,5 -> expert 1 -> t0; expert 2 -> t1,t2,t0; expert 3 -> t1,t2
    // token 0: [1, 2], token 1: [2, 3], token 2: [2, 3]
    let e = [1, 2, 3];
    let c = [1, 3, 2];
    let mut sel = [0i32; 6];
    unsafe { force_routes(0, 1, &call(9, &e, &c), 9, 3, sel.as_mut_ptr(), 2) };
    assert_eq!(sel, [1, 2, 2, 3, 2, 3]);
}

#[test]
fn force_routes_exits_leave_sel_alone() {
    let e = [1, 2];
    let c = [3, 3];
    let mut sel = [-9i32; 6];
    let p = sel.as_mut_ptr();
    unsafe {
        force_routes(1, 1, ptr::null(), 3, 3, p, 2); // pos == size
        force_routes(0, 1, &call(3, &e, &c), 4, 3, p, 2); // layer differs
        force_routes(0, 1, &call(3, &e, &c[..1]), 3, 3, p, 2); // counts shorter than experts
        force_routes(0, 1, &call(3, &e, &c), 3, 4, p, 2); // total 6 != 4 * 2
        force_routes(0, 1, &call(3, &e, &[4, 2]), 3, 3, p, 2); // count 4 > n 3
        force_routes(0, 1, &call(3, &[], &[]), 3, 3, p, 2); // empty call: total 0 != 6
    }
    assert_eq!(sel, [-9; 6]);
}

#[test]
fn force_routes_n_zero_and_empty_calls() {
    let mut sel = [-1i32; 2];
    unsafe {
        force_routes(0, 1, &call(2, &[], &[]), 2, 0, sel.as_mut_ptr(), 16); // total 0 == 0 * 16, nothing to deal
        force_routes(0, 1, &call(2, &[7], &[0]), 2, 0, sel.as_mut_ptr(), 16); // count 0 is not > 0, total 0
    }
    assert_eq!(sel, [-1, -1]);
}

#[test]
fn force_routes_negative_count_deals_nothing_for_it() {
    // top 1, n 2: counts 2,-1,1 sum 2 = 2*1 and none > 2, so the checks pass. Expert 1 (count 2): s=0 -> token 0, sel[0]; s=1 -> token 1, sel[1*1+0] = sel[1].
    // The -1 deals nothing. Expert 3 (count 1): s=2 -> token 0 again, fill[0] = 1, so sel[0*1+1] = sel[1]: a second slot for token 0 that is token 1's slot (no bounds check, as the original).
    let e = [1, 2, 3];
    let c = [2, -1, 1];
    let mut sel = [0i32; 4];
    unsafe { force_routes(0, 1, &call(0, &e, &c), 0, 2, sel.as_mut_ptr(), 1) };
    assert_eq!(sel, [1, 3, 0, 0]);
}

#[test]
fn force_routes_full_width() {
    // the real shape: top 16, n 5 tokens, 16 experts each with count 5 -> every token gets all 16 experts in order
    let e: Vec<i32> = (100..116).collect();
    let c = vec![5i32; 16];
    let mut sel = vec![-1i32; 5 * 16];
    unsafe { force_routes(3, 4, &call(12, &e, &c), 12, 5, sel.as_mut_ptr(), 16) };
    for t in 0..5 {
        assert_eq!(&sel[t * 16..t * 16 + 16], &e[..]);
    }
}

#[test]
fn warn_text() {
    assert_eq!(s(&warn_line(0, 8, "does not match the route cache")), "K3GPU prefill: call 0 (layer 8): does not match the route cache; the computed routes are used from here\n");
    assert_eq!(s(&warn_line(18446744073709551615, -1, "x")), "K3GPU prefill: call 18446744073709551615 (layer -1): x; the computed routes are used from here\n");
}

// ---------------------------------------------------------------------------------------------------------------------------------- doubles through libc

fn fd(prec: u8, v: f64) -> String {
    s(&Line::new().f(prec, v).done())
}

#[test]
fn libc_formats_halves_and_specials() {
    // exact binary values: 0.25 and 2.5 are ties and go to the even digit; 0.05 is above, 0.15 and 0.35 below their half; 0.45 above
    assert_eq!([0.25, 0.75, 1.25, 0.05, 0.15, 0.35, 0.45].map(|v| fd(1, v)), ["0.2", "0.8", "1.2", "0.1", "0.1", "0.3", "0.5"].map(String::from));
    assert_eq!([0.5, 1.5, 2.5, 3.5, 0.4999999999999999].map(|v| fd(0, v)), ["0", "2", "2", "4", "0"].map(String::from));
    assert_eq!([0.0005, 0.0015, 0.0625, 0.0].map(|v| fd(3, v)), ["0.001", "0.002", "0.062", "0.000"].map(String::from)); // 0.0005 above its half, 0.0015 above; 0.0625 exact tie -> even
    assert_eq!(fd(1, f64::NAN), "nan");
    assert_eq!(fd(1, -f64::NAN), "-nan");
    assert_eq!(fd(1, f64::INFINITY), "inf");
    assert_eq!(fd(3, f64::NEG_INFINITY), "-inf");
    assert_eq!(fd(1, -0.0), "-0.0");
    assert_eq!(fd(0, -0.0), "-0");
    assert_eq!(fd(1, -0.04), "-0.0");
    assert_eq!(fd(1, 1e22), "10000000000000000000000.0");
    assert_eq!(fd(1, f64::MIN_POSITIVE), "0.0");
    assert_eq!(fd(3, 5e-324), "0.000");
    let big = fd(1, 1e300);
    assert_eq!(big.len(), 303);
    assert!(big.starts_with('1') && big.ends_with(".0"));
    assert_eq!(fd(1, f64::MAX).len(), 311, "309 digits + '.0'");
}

// ---------------------------------------------------------------------------------------------------------------------------------- summary

fn texts(s_: &EcSummary) -> Vec<String> {
    summary_lines(s_).iter().map(|l| s(l)).collect()
}

fn small() -> EcSummary {
    let mut m = EcSummary::default();
    m.cards = 2;
    m.expert_raw = 17_547_264;
    m.card[0] = EcCardStats { ready: 1, slots: 100, needed: 10, hit_card: 5, hit_host: 3, from_drive: 2, written_back: 1, t_read: 1.25, t_copy: 0.5 };
    m.cq_jobs = [1, 2, 3];
    m.sc_calls = [4, 5, 6];
    m.rt_nu_calls = 7;
    m.n_calls[0] = 3.0;
    m.t_calls[0] = 0.0005;
    m
}

#[test]
fn summary_small_case_by_hand() {
    // 5/10 -> 50.0, 3/10 -> 30.0, 2/10 -> 20.0; GB = 2 * 17547264 / 1e9 = 0.035 -> 0.0; reads 1.25 -> 1.2 (tie to even), copy 0.5;
    // call line: 3, 0.0005 -> 0.001 (the double is a little above), 1000 * 0.0005 / 3 = 0.1666.. -> 0.2
    assert_eq!(
        texts(&small()),
        vec![
            "K3GPU copy queue cpp: jobs sent 1 drive->RAM, 2 drive->card, 3 RAM->card\n",
            "K3GPU slot cache cpp: 4 acquire, 5 try_acquire, 6 find\n",
            "K3GPU route table cpp: 0 calls, 7 next_use calls\n",
            "K3GPU experts: 10 needed, 5 already on the card (50.0%), 3 from the host cache (30.0%), 2 from the drive (20.0%, 0.0 GB); drive reads 1.2 s, host to card 0.5 s\n",
            "K3GPU card 0: 100 expert slots, 10 needed, 5 on the card, 3 host, 2 drive, 1 written back to RAM\n",
            "K3GPU prompt-pass layer calls: 3, 0.001 s in all, 0.2 ms each\n",
        ]
    );
}

#[test]
fn summary_three_lines_only_when_nothing_was_needed() {
    let mut m = small();
    m.card[0].needed = 0;
    m.copyq_rust = 1;
    m.slotcache_rust = 9;
    m.routetable_rust = -1;
    m.rc_calls = 5612;
    m.vq = 1;
    let t = texts(&m);
    assert_eq!(t.len(), 3);
    assert_eq!(t[0], "K3GPU copy queue rust: jobs sent 1 drive->RAM, 2 drive->card, 3 RAM->card\n");
    assert_eq!(t[1], "K3GPU slot cache rust: 4 acquire, 5 try_acquire, 6 find\n");
    assert_eq!(t[2], "K3GPU route table rust: 5612 calls, 7 next_use calls\n");
}

#[test]
fn summary_counts_only_ready_cards_below_cards() {
    let mut m = small();
    m.card[1] = EcCardStats { ready: 0, slots: 5, needed: 99, ..Default::default() };
    m.card[5] = EcCardStats { ready: 1, slots: 5, needed: 99, ..Default::default() };
    assert_eq!(texts(&m).len(), 6, "card 1 not ready, card 5 beyond `cards`");
    m.cards = 6;
    let t = texts(&m);
    assert!(t[3].starts_with("K3GPU experts: 109 needed, 5 already on the card (4.6%)"), "{}", t[3]); // 100 * 5 / 109 = 4.587
    assert_eq!(t[5], "K3GPU card 5: 5 expert slots, 99 needed, 0 on the card, 0 host, 0 drive, 0 written back to RAM\n");
    m.cards = 0;
    assert_eq!(texts(&m).len(), 3, "no card is read");
    m.cards = -4;
    assert_eq!(texts(&m).len(), 3);
}

#[test]
fn summary_need_wraps_modulo_2_64() {
    let mut m = small();
    m.card[0].needed = u64::MAX;
    m.card[1] = EcCardStats { ready: 1, needed: 1, ..Default::default() };
    assert_eq!(texts(&m).len(), 3, "u64::MAX + 1 wraps to 0: the original returns after the three lines");
}

#[test]
fn summary_route_cache_and_queue_lines() {
    let mut m = small();
    m.rc_calls = 5612;
    m.rc_pos = 300;
    m.rc_ok = 1;
    m.pf_planned = 11;
    m.pf_skipped = 12;
    m.pf_used = 13;
    m.pf_waited = 14;
    m.t_pwait = 2.25;
    m.wb_skipped = 15;
    m.vq = 1;
    m.vq_ram = u64::MAX;
    m.vq_drive = 2;
    m.vq_wb = 3;
    m.vq_nowb = 4;
    m.vq_waited = 5;
    m.t_vwait = 0.05;
    m.t_plan = 99.95;
    let t = texts(&m);
    assert_eq!(t.len(), 8);
    assert_eq!(t[5], "K3GPU route cache: 300 of 5612 calls seen, look-ahead on to the end; 11 experts read ahead, 12 already resident, 13 taken from read-ahead, 14 of them waited for (2.2 s); 15 card victims not kept in RAM\n");
    assert_eq!(t[6], "K3GPU card queues: 1 experts brought ahead (18446744073709551615 from RAM, 2 from the drive), 3 replaced experts copied to RAM, 4 dropped; 5 waited for (0.1 s); planning in the hook 100.0 s\n");
    m.rc_ok = 0;
    assert!(texts(&m)[5].contains("look-ahead stopped;"));
    m.rc_calls = 0;
    assert!(!texts(&m).iter().any(|l| l.contains("route cache")), "no table, no route cache line, even with rc_pos set");
}

#[test]
fn summary_call_classes() {
    let mut m = small();
    m.n_calls = [0.0, -0.0, 4.0];
    m.t_calls = [1.0, 1.0, 2.0];
    let t = texts(&m);
    assert_eq!(t.len(), 6, "0 and -0 are false; one class line");
    assert_eq!(t[5], "K3GPU multi-token layer calls: 4, 2.000 s in all, 500.0 ms each\n");
    m.n_calls = [f64::NAN, f64::INFINITY, 0.5];
    m.t_calls = [1.0, 3.0, 0.25];
    let t = texts(&m);
    assert_eq!(t[5], "K3GPU prompt-pass layer calls: nan, 1.000 s in all, nan ms each\n");
    assert_eq!(t[6], "K3GPU single-token layer calls: inf, 3.000 s in all, 0.0 ms each\n");
    assert_eq!(t[7], "K3GPU multi-token layer calls: 0, 0.250 s in all, 500.0 ms each\n", "%.0f of 0.5 is 0, 1000 * 0.25 / 0.5 = 500");
}

#[test]
fn summary_fields_expert_raw_and_large_counts() {
    let mut m = small();
    m.expert_raw = 1_000_000_000;
    m.card[0].from_drive = 1_000_000_000_000; // 1e12 * 1e9 / 1e9 = 1e12 GB
    m.card[0].needed = 1_000_000_000_000;
    m.card[0].hit_card = 0;
    m.card[0].hit_host = 0;
    m.card[0].t_read = 1e22;
    let t = texts(&m);
    assert_eq!(
        t[3],
        "K3GPU experts: 1000000000000 needed, 0 already on the card (0.0%), 0 from the host cache (0.0%), 1000000000000 from the drive (100.0%, 1000000000000.0 GB); drive reads 10000000000000000000000.0 s, host to card 0.5 s\n"
    );
}

// ---------------------------------------------------------------------------------------------------------------------------------- ABI layout (offsets computed by hand for LP64; the C side checks the same numbers)

#[test]
fn abi_layout() {
    assert_eq!(size_of::<EcSettings>(), 88);
    assert_eq!(
        [
            offset_of!(EcSettings, cards), offset_of!(EcSettings, readers), offset_of!(EcSettings, verbose), offset_of!(EcSettings, reserve), offset_of!(EcSettings, policy),
            offset_of!(EcSettings, chunks), offset_of!(EcSettings, host_gb), offset_of!(EcSettings, host_bytes), offset_of!(EcSettings, routecache), offset_of!(EcSettings, lookahead),
            offset_of!(EcSettings, vqueue), offset_of!(EcSettings, vq_cap), offset_of!(EcSettings, vq_from), offset_of!(EcSettings, layout), offset_of!(EcSettings, prefill)
        ],
        [0, 4, 8, 16, 24, 28, 32, 40, 48, 56, 60, 64, 72, 80, 84]
    );
    assert_eq!(size_of::<EcCall>(), 40);
    assert_eq!([offset_of!(EcCall, layer), offset_of!(EcCall, experts), offset_of!(EcCall, n_experts), offset_of!(EcCall, counts), offset_of!(EcCall, n_counts)], [0, 8, 16, 24, 32]);
    assert_eq!(size_of::<EcCardStats>(), 64);
    assert_eq!(
        [
            offset_of!(EcCardStats, ready), offset_of!(EcCardStats, slots), offset_of!(EcCardStats, needed), offset_of!(EcCardStats, hit_card), offset_of!(EcCardStats, hit_host),
            offset_of!(EcCardStats, from_drive), offset_of!(EcCardStats, written_back), offset_of!(EcCardStats, t_read), offset_of!(EcCardStats, t_copy)
        ],
        [0, 4, 8, 16, 24, 32, 40, 48, 56]
    );
    assert_eq!(size_of::<EcSummary>(), 784);
    assert_eq!(
        [
            offset_of!(EcSummary, cards), offset_of!(EcSummary, expert_raw), offset_of!(EcSummary, card), offset_of!(EcSummary, copyq_rust), offset_of!(EcSummary, cq_jobs),
            offset_of!(EcSummary, slotcache_rust), offset_of!(EcSummary, sc_calls), offset_of!(EcSummary, routetable_rust), offset_of!(EcSummary, rc_calls), offset_of!(EcSummary, rt_nu_calls),
            offset_of!(EcSummary, rc_pos), offset_of!(EcSummary, rc_ok), offset_of!(EcSummary, pf_planned), offset_of!(EcSummary, pf_skipped), offset_of!(EcSummary, pf_used),
            offset_of!(EcSummary, pf_waited), offset_of!(EcSummary, t_pwait), offset_of!(EcSummary, wb_skipped), offset_of!(EcSummary, vq), offset_of!(EcSummary, vq_ram),
            offset_of!(EcSummary, vq_drive), offset_of!(EcSummary, vq_wb), offset_of!(EcSummary, vq_nowb), offset_of!(EcSummary, vq_waited), offset_of!(EcSummary, t_vwait),
            offset_of!(EcSummary, t_plan), offset_of!(EcSummary, n_calls), offset_of!(EcSummary, t_calls)
        ],
        [0, 4, 8, 520, 528, 552, 560, 584, 592, 600, 608, 616, 624, 632, 640, 648, 656, 664, 672, 680, 688, 696, 704, 712, 720, 728, 736, 760]
    );
}

// ---------------------------------------------------------------------------------------------------------------------------------- exits and stderr (child processes)

fn run_child(test: &str, envs: &[(&str, &[u8])]) -> (Option<i32>, Vec<u8>) {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args([test, "--exact", "--nocapture", "--test-threads=1"]).env("ENGCFG_CHILD", "1");
    for (k, v) in envs {
        cmd.env(k, std::ffi::OsStr::from_bytes(v));
    }
    let out = cmd.output().unwrap();
    (out.status.code(), out.stderr)
}

fn in_child() -> bool {
    std::env::var_os("ENGCFG_CHILD").is_some()
}

#[test]
fn child_trace_unset() {
    if !in_child() {
        return;
    }
    assert!(ec_open_trace().is_null());
}

#[test]
fn child_trace_dev_null() {
    if !in_child() {
        return;
    }
    assert!(!ec_open_trace().is_null());
}

#[test]
fn child_trace_unwritable() {
    if !in_child() {
        return;
    }
    ec_open_trace();
    std::process::exit(99);
}

#[test]
fn trace_open_and_failure() {
    let (code, err) = run_child("tests::child_trace_unset", &[]);
    assert_eq!((code, err.as_slice()), (Some(0), &b""[..]));
    let (code, err) = run_child("tests::child_trace_unset", &[("K3GPU_TRACE", b"")]);
    assert_eq!((code, err.as_slice()), (Some(0), &b""[..]), "empty = unset");
    let (code, err) = run_child("tests::child_trace_dev_null", &[("K3GPU_TRACE", b"/dev/null")]);
    assert_eq!((code, err.as_slice()), (Some(0), &b""[..]));
    let (code, err) = run_child("tests::child_trace_unwritable", &[("K3GPU_TRACE", b"/nonexistent_dir_engcfg/x")]);
    assert_eq!((code, err), (Some(1), b"cannot write /nonexistent_dir_engcfg/x\n".to_vec()));
    let (code, err) = run_child("tests::child_trace_unwritable", &[("K3GPU_TRACE", b"/nonexistent_\xff\xfe/x")]);
    assert_eq!((code, err), (Some(1), b"cannot write /nonexistent_\xff\xfe/x\n".to_vec()), "raw bytes, not lossy UTF-8");
}

#[test]
fn child_prefill_empty() {
    if !in_child() {
        return;
    }
    ec_prefill_start(0);
    std::process::exit(99);
}

#[test]
fn child_prefill_some() {
    if !in_child() {
        return;
    }
    ec_prefill_start(1);
}

#[test]
fn prefill_start_exit_and_line() {
    let (code, err) = run_child("tests::child_prefill_empty", &[]);
    assert_eq!((code, s(&err)), (Some(1), "K3GPU_PREFILL needs K3GPU_ROUTECACHE\n".to_string()));
    let (code, err) = run_child("tests::child_prefill_some", &[]);
    assert_eq!((code, err), (Some(0), PREFILL_ON.to_vec()));
}

#[test]
fn child_selector_fatal() {
    if !in_child() {
        return;
    }
    ec_selector_mode(0, 1);
    std::process::exit(99);
}

#[test]
fn child_selector_cpp() {
    if !in_child() {
        return;
    }
    std::process::exit(20 + ec_selector_mode(1, 1));
}

#[test]
fn selector_exit_through_the_c_abi() {
    let (code, err) = run_child("tests::child_selector_fatal", &[("K3GPU_COPYQ", b"Rust")]);
    assert_eq!((code, s(&err)), (Some(1), "K3GPU_COPYQ must be cpp or rust, not Rust\n".to_string()));
    let (code, err) = run_child("tests::child_selector_fatal", &[("K3GPU_COPYQ", b"ru\xffst")]);
    assert_eq!((code, err), (Some(1), b"K3GPU_COPYQ must be cpp or rust, not ru\xffst\n".to_vec()));
    let (code, err) = run_child("tests::child_selector_cpp", &[]);
    assert_eq!((code, s(&err)), (Some(20), "K3GPU: slot cache cpp\n".to_string()));
    let (code, err) = run_child("tests::child_selector_cpp", &[("K3GPU_SLOTCACHE", b"rust")]);
    assert_eq!((code, err.as_slice()), (Some(21), &b""[..]));
}

#[test]
fn child_force_twice() {
    if !in_child() {
        return;
    }
    let e = [1, 2];
    let c = [3, 3];
    let mut sel = [0i32; 6];
    let p = sel.as_mut_ptr();
    unsafe {
        force_routes(0, 1, &call(3, &e, &c), 4, 3, p, 2); // layer differs: the one message
        force_routes(2, 2, ptr::null(), 4, 3, p, 2); // silent
        force_routes(0, 1, &call(3, &e, &c[..1]), 3, 3, p, 2); // silent
    }
}

fn run_child_signal(test: &str) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    let out = Command::new(std::env::current_exe().unwrap()).args([test, "--exact", "--nocapture", "--test-threads=1"]).env("ENGCFG_CHILD", "1").output().unwrap();
    out.status.signal()
}

#[test]
fn child_abort_summary_cards_9() {
    if !in_child() {
        return;
    }
    let mut m = small();
    m.cards = 9;
    unsafe { ec_print_summary(&m) };
    std::process::exit(99);
}

#[test]
fn child_abort_card_of_zero_cards() {
    if !in_child() {
        return;
    }
    std::process::exit(card_of(1, 0, 5));
}

#[test]
fn child_abort_card_of_min_over_minus_one() {
    if !in_child() {
        return;
    }
    std::process::exit(card_of(1, -1, i32::MIN.wrapping_add(1)));
}

#[test]
fn child_abort_force_negative_n() {
    if !in_child() {
        return;
    }
    // n = -1, top 16: counts must sum to -16 with none > -1: sixteen counts of -1
    let e = [0i32; 16];
    let c = [-1i32; 16];
    let mut sel = [0i32; 1];
    unsafe { force_routes(0, 1, &call(0, &e, &c), 0, -1, sel.as_mut_ptr(), 16) };
    std::process::exit(99);
}

#[test]
fn out_of_contract_inputs_abort_with_sigabrt() {
    for t in ["tests::child_abort_summary_cards_9", "tests::child_abort_card_of_zero_cards", "tests::child_abort_card_of_min_over_minus_one", "tests::child_abort_force_negative_n"] {
        assert_eq!(run_child_signal(t), Some(6), "{t}");
    }
}

#[test]
fn warns_once_per_process() {
    let (code, err) = run_child("tests::child_force_twice", &[]);
    assert_eq!((code, s(&err)), (Some(0), "K3GPU prefill: call 0 (layer 4): does not match the route cache; the computed routes are used from here\n".to_string()));
}
