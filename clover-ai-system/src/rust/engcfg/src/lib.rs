//! engcfg: Rust implementation of include/engcfg.h (the engine's start-up settings, selector parsing, card_of, force_routes and the exit summary).
//! Behaviour reference: backend-gpu.cu / decode.cuh as committed at 4d8d891; see NOTES.md for the map from original to code.
//! Messages are built as bytes (strings the original prints with %s stay raw bytes) and written to fd 2 in one write per original line.
//! Doubles are formatted by the C library's snprintf, not by a Rust formatter: the original uses printf, and libc gives the same digits, rounding of halves,
//! "inf"/"nan"/"-nan" spelling and negative zero by construction.

use std::ffi::{c_char, c_int, c_long, c_ulong, c_void, CStr};
use std::io::Write;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

extern "C" {
    fn getenv(name: *const c_char) -> *mut c_char;
    fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> c_int;
    fn fopen(path: *const c_char, mode: *const c_char) -> *mut c_void;
}

pub const EC_MAX_CARDS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EcSettings {
    pub cards: c_int,
    pub readers: c_int,
    pub verbose: c_int,
    pub reserve: usize,
    pub policy: c_int,
    pub chunks: c_int,
    pub host_gb: c_int,
    pub host_bytes: usize,
    pub routecache: *const c_char,
    pub lookahead: c_int,
    pub vqueue: c_int,
    pub vq_cap: usize,
    pub vq_from: c_long,
    pub layout: c_int,
    pub prefill: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EcCall {
    pub layer: c_int,
    pub experts: *const c_int,
    pub n_experts: usize,
    pub counts: *const c_int,
    pub n_counts: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EcCardStats {
    pub ready: c_int,
    pub slots: c_int,
    pub needed: c_ulong,
    pub hit_card: c_ulong,
    pub hit_host: c_ulong,
    pub from_drive: c_ulong,
    pub written_back: c_ulong,
    pub t_read: f64,
    pub t_copy: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct EcSummary {
    pub cards: c_int,
    pub expert_raw: c_int,
    pub card: [EcCardStats; EC_MAX_CARDS],
    pub copyq_rust: c_int,
    pub cq_jobs: [c_ulong; 3],
    pub slotcache_rust: c_int,
    pub sc_calls: [c_ulong; 3],
    pub routetable_rust: c_int,
    pub rc_calls: usize,
    pub rt_nu_calls: c_ulong,
    pub rc_pos: usize,
    pub rc_ok: c_int,
    pub pf_planned: c_ulong,
    pub pf_skipped: c_ulong,
    pub pf_used: c_ulong,
    pub pf_waited: c_ulong,
    pub t_pwait: f64,
    pub wb_skipped: c_ulong,
    pub vq: c_int,
    pub vq_ram: c_ulong,
    pub vq_drive: c_ulong,
    pub vq_wb: c_ulong,
    pub vq_nowb: c_ulong,
    pub vq_waited: c_ulong,
    pub t_vwait: f64,
    pub t_plan: f64,
    pub n_calls: [f64; 3],
    pub t_calls: [f64; 3],
}

// ---------------------------------------------------------------------------------------------------------------------------------- environment

/// Where variables come from. The C ABI reads the process environment; unit tests use a map.
pub trait EnvSource {
    fn get(&self, name: &CStr) -> *const c_char;
}

struct ProcessEnv;
impl EnvSource for ProcessEnv {
    fn get(&self, name: &CStr) -> *const c_char {
        unsafe { getenv(name.as_ptr()) }
    }
}

/// glibc strtol(s, NULL, 10) on the bytes of a C string (no NUL inside): C-locale whitespace, one sign, decimal digits, saturation at LONG_MAX / LONG_MIN, 0 when no digit.
fn strtol10(s: &[u8]) -> i64 {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut acc: u64 = 0;
    let mut over = false;
    while i < s.len() && s[i].is_ascii_digit() {
        match acc.checked_mul(10).and_then(|v| v.checked_add((s[i] - b'0') as u64)) {
            Some(v) => acc = v,
            None => over = true,
        }
        i += 1;
    }
    if neg {
        if over || acc > (1u64 << 63) {
            i64::MIN
        } else {
            (acc as i64).wrapping_neg()
        }
    } else if over || acc > i64::MAX as u64 {
        i64::MAX
    } else {
        acc as i64
    }
}

/// glibc atoi: (int)strtol(s, NULL, 10); the cast keeps the low 32 bits.
pub fn atoi(s: &[u8]) -> i32 {
    strtol10(s) as i32
}

unsafe fn cbytes<'a>(p: *const c_char) -> &'a [u8] {
    CStr::from_ptr(p).to_bytes()
}

fn unset_or_empty(p: *const c_char) -> bool {
    p.is_null() || unsafe { *p } == 0
}

fn env_int<E: EnvSource + ?Sized>(e: &E, name: &CStr, dflt: i32) -> i32 {
    let v = e.get(name);
    if unset_or_empty(v) {
        dflt
    } else {
        atoi(unsafe { cbytes(v) })
    }
}

fn settings_from<E: EnvSource + ?Sized>(e: &E, devices: i32) -> EcSettings {
    let ei = |n: &CStr, d: i32| env_int(e, n, d);
    let cards = ei(c"K3GPU_CARDS", 8).min(devices).min(EC_MAX_CARDS as i32);
    let readers = ei(c"K3GPU_READERS", 16);
    let verbose = ei(c"K3GPU_VERBOSE", 0);
    let reserve = (ei(c"K3GPU_RESERVE_MB", 2500) as i64 as usize) << 20;
    let policy = ei(c"K3GPU_POLICY", 1);
    let chunks = ei(c"K3GPU_CHUNKS", 0);
    let host_gb = ei(c"K3GPU_HOST_GB", 150);
    let host_bytes = (host_gb as i64 as usize) << 30;
    let layout = ei(c"K3GPU_LAYOUT", 0);
    let prefill = (ei(c"K3GPU_PREFILL", 0) != 0) as c_int;
    let rc = e.get(c"K3GPU_ROUTECACHE");
    let mut s = EcSettings {
        cards,
        readers,
        verbose,
        reserve,
        policy,
        chunks,
        host_gb,
        host_bytes,
        routecache: ptr::null(),
        lookahead: 0,
        vqueue: 0,
        vq_cap: 256,
        vq_from: 0,
        layout,
        prefill,
    };
    if !unset_or_empty(rc) {
        s.routecache = rc;
        s.lookahead = ei(c"K3GPU_LOOKAHEAD", 92);
        s.vqueue = (ei(c"K3GPU_VQUEUE", 0) != 0 && policy == 2) as c_int;
        s.vq_cap = ei(c"K3GPU_VQ_INFLIGHT", 256) as i64 as usize;
        s.vq_from = ei(c"K3GPU_VQ_FROM", 0) as c_long;
    }
    s
}

// ---------------------------------------------------------------------------------------------------------------------------------- output

fn emit(b: &[u8]) {
    let _ = std::io::stderr().write_all(b);
}

fn fatal(b: &[u8]) -> ! {
    emit(b);
    std::process::exit(1)
}

/// One output line, as bytes.
struct Line(Vec<u8>);

impl Line {
    fn new() -> Line {
        Line(Vec::with_capacity(256))
    }
    fn s(&mut self, t: &str) -> &mut Line {
        self.0.extend_from_slice(t.as_bytes());
        self
    }
    fn b(&mut self, t: &[u8]) -> &mut Line {
        self.0.extend_from_slice(t);
        self
    }
    /// %s of a C string; glibc prints a NULL pointer as "(null)".
    fn cs(&mut self, p: *const c_char) -> &mut Line {
        if p.is_null() {
            self.s("(null)")
        } else {
            self.b(unsafe { cbytes(p) })
        }
    }
    /// %d / %zu / %lu (all decimal)
    fn n(&mut self, v: impl std::fmt::Display) -> &mut Line {
        self.s(&v.to_string())
    }
    /// %.0f, %.1f or %.3f through the C library.
    fn f(&mut self, prec: u8, v: f64) -> &mut Line {
        let fmt: &CStr = match prec {
            0 => c"%.0f",
            1 => c"%.1f",
            3 => c"%.3f",
            _ => unreachable!(),
        };
        let len = unsafe { snprintf(ptr::null_mut(), 0, fmt.as_ptr(), v) };
        let len = len.max(0) as usize;
        let mut buf = vec![0u8; len + 1];
        unsafe { snprintf(buf.as_mut_ptr() as *mut c_char, len + 1, fmt.as_ptr(), v) };
        buf.truncate(len);
        self.b(&buf)
    }
    fn done(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------- messages of configure()

fn route_cache_line(s: &EcSettings, calls: usize, pend_cap: usize) -> Vec<u8> {
    Line::new()
        .s("K3GPU route cache: ")
        .n(calls)
        .s(" calls from ")
        .cs(s.routecache)
        .s(", look-ahead ")
        .n(s.lookahead)
        .s(" calls, at most ")
        .n(pend_cap)
        .s(" experts read ahead, card queues ")
        .s(if s.vqueue != 0 { "on" } else { "off" })
        .s(" (at most ")
        .n(s.vq_cap)
        .s(" in flight)\n")
        .done()
}

const PREFILL_NEEDS: &[u8] = b"K3GPU_PREFILL needs K3GPU_ROUTECACHE\n";
const PREFILL_ON: &[u8] = b"K3GPU PREFILL ON: timing only. No drive is read, expert bytes are 4 records per layer, routes are taken from the route cache; outputs are not the model's.\n";

fn start_line(s: &EcSettings) -> Vec<u8> {
    Line::new().s("K3GPU: ").n(s.cards).s(" cards, host expert cache ").n(s.host_gb).s(" GB, ").n(s.readers).s(" drive readers\n").done()
}

fn trace_error_line(path: *const c_char) -> Vec<u8> {
    Line::new().s("cannot write ").cs(path).s("\n").done()
}

// ---------------------------------------------------------------------------------------------------------------------------------- selectors

struct SelInfo {
    name: &'static str,
    label: &'static str,
    hdr: &'static str,
    what: &'static str,
    dir: &'static str,
    var: &'static CStr,
    libvar: &'static CStr,
    deflib: &'static CStr,
}

static SELS: [SelInfo; 3] = [
    SelInfo { name: "COPYQ", label: "copy queue", hdr: "copyq.h", what: "the C++ queue", dir: "copyq", var: c"K3GPU_COPYQ", libvar: c"K3GPU_COPYQ_LIB", deflib: c"libcopyq.so" },
    SelInfo { name: "SLOTCACHE", label: "slot cache", hdr: "slotcache.h", what: "the C++ cache", dir: "slotcache", var: c"K3GPU_SLOTCACHE", libvar: c"K3GPU_SLOTCACHE_LIB", deflib: c"libslotcache.so" },
    SelInfo { name: "ROUTETABLE", label: "route table", hdr: "routetable.h", what: "the C++ parser", dir: "routetable", var: c"K3GPU_ROUTETABLE", libvar: c"K3GPU_ROUTETABLE_LIB", deflib: c"libroutetable.so" },
];

fn sel_info(component: c_int) -> &'static SelInfo {
    match component {
        0..=2 => &SELS[component as usize],
        _ => std::process::abort(),
    }
}

enum SelOutcome {
    Cpp,
    Rust,
    Fatal(Vec<u8>),
}

/// The decision of copyq_select / slotcache_select / routetable_select before any library is touched: the bytes to print for the cpp case are returned by `selector_cpp_line`.
fn selector_decide<E: EnvSource + ?Sized>(e: &E, component: c_int, built: bool) -> SelOutcome {
    let si = sel_info(component);
    let m = e.get(si.var);
    let empty = unset_or_empty(m);
    let m_bytes: &[u8] = if m.is_null() { b"" } else { unsafe { cbytes(m) } };
    let is_cpp = empty || m_bytes == b"cpp";
    if built {
        if is_cpp {
            return SelOutcome::Cpp;
        }
        if m_bytes == b"rust" {
            return SelOutcome::Rust;
        }
        return SelOutcome::Fatal(Line::new().s("K3GPU_").s(si.name).s(" must be cpp or rust, not ").b(m_bytes).s("\n").done());
    }
    if !is_cpp {
        return SelOutcome::Fatal(
            Line::new()
                .s("K3GPU_")
                .s(si.name)
                .s("=")
                .b(m_bytes)
                .s(": this build has no ")
                .s(si.hdr)
                .s(", only ")
                .s(si.what)
                .s(" (build with -I src/rust/")
                .s(si.dir)
                .s("/include)\n")
                .done(),
        );
    }
    SelOutcome::Cpp
}

fn selector_cpp_line(component: c_int) -> Vec<u8> {
    Line::new().s("K3GPU: ").s(sel_info(component).label).s(" cpp\n").done()
}

fn selector_lib_from<E: EnvSource + ?Sized>(e: &E, component: c_int) -> *const c_char {
    let si = sel_info(component);
    let l = e.get(si.libvar);
    if unset_or_empty(l) {
        si.deflib.as_ptr()
    } else {
        l
    }
}

fn selector_announce_line(component: c_int, lib: *const c_char) -> Vec<u8> {
    Line::new().s("K3GPU: ").s(sel_info(component).label).s(" rust (").cs(lib).s(")\n").done()
}

// ---------------------------------------------------------------------------------------------------------------------------------- card_of

fn card_of(layout: i32, cards: i32, layer: i32) -> i32 {
    if layout == 1 {
        if layer == 0 {
            return 0;
        }
        let d = layer.wrapping_sub(1);
        if cards == 0 || (cards == -1 && d == i32::MIN) {
            std::process::abort(); // the original traps (SIGFPE) on exactly these inputs
        }
        return d % cards;
    }
    if layer == 0 {
        0
    } else {
        cards.wrapping_sub(1).min(layer.wrapping_mul(cards) / 93)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------- force_routes

static WARNED: AtomicBool = AtomicBool::new(false);

fn warn_line(pos: usize, l: i32, why: &str) -> Vec<u8> {
    Line::new().s("K3GPU prefill: call ").n(pos).s(" (layer ").n(l).s("): ").s(why).s("; the computed routes are used from here\n").done()
}

unsafe fn slice_or_empty<'a>(p: *const c_int, n: usize) -> &'a [c_int] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

/// Safety: `cur` is readable when pos < size; `sel` is writable at every index the dealing produces (as in the original, nothing is bounds-checked).
unsafe fn force_routes(pos: usize, size: usize, cur: *const EcCall, l: i32, n: i32, sel: *mut i32, top: i32) {
    let warn = |why: &str| {
        if !WARNED.swap(true, Ordering::Relaxed) {
            emit(&warn_line(pos, l, why));
        }
    };
    if pos >= size || (*cur).layer != l {
        warn("does not match the route cache");
        return;
    }
    let c = &*cur;
    if c.n_counts != c.n_experts {
        warn("the route cache has no counts");
        return;
    }
    let experts = slice_or_empty(c.experts, c.n_experts);
    let counts = slice_or_empty(c.counts, c.n_counts);
    let mut total: i64 = 0;
    let mut fits = true;
    for &k in counts {
        total += k as i64;
        if k > n {
            fits = false;
        }
    }
    if !fits || total != (n as i64) * (top as i64) {
        warn("counts do not fit this call");
        return;
    }
    if n < 0 {
        std::process::abort(); // std::vector<int> fill(n, 0) throws std::length_error in the original
    }
    let mut fill = vec![0i32; n as usize];
    let mut s: i64 = 0;
    for i in 0..experts.len() {
        for _ in 0..counts[i] {
            let t = (s % n as i64) as usize;
            let idx = t.wrapping_mul(top as isize as usize).wrapping_add(fill[t] as isize as usize);
            *sel.wrapping_add(idx) = experts[i];
            fill[t] += 1;
            s += 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------- print_summary

fn summary_lines(s: &EcSummary) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    if s.cards > EC_MAX_CARDS as c_int {
        std::process::abort();
    }
    let cards = s.cards.max(0) as usize;
    let (mut need, mut hc, mut hh, mut dr): (u64, u64, u64, u64) = (0, 0, 0, 0);
    let (mut tr, mut tc) = (0f64, 0f64);
    for k in &s.card[..cards] {
        if k.ready == 0 {
            continue;
        }
        need = need.wrapping_add(k.needed);
        hc = hc.wrapping_add(k.hit_card);
        hh = hh.wrapping_add(k.hit_host);
        dr = dr.wrapping_add(k.from_drive);
        tr += k.t_read;
        tc += k.t_copy;
    }
    let which = |b: c_int| if b != 0 { "rust" } else { "cpp" };
    out.push(
        Line::new()
            .s("K3GPU copy queue ")
            .s(which(s.copyq_rust))
            .s(": jobs sent ")
            .n(s.cq_jobs[0])
            .s(" drive->RAM, ")
            .n(s.cq_jobs[1])
            .s(" drive->card, ")
            .n(s.cq_jobs[2])
            .s(" RAM->card\n")
            .done(),
    );
    out.push(
        Line::new()
            .s("K3GPU slot cache ")
            .s(which(s.slotcache_rust))
            .s(": ")
            .n(s.sc_calls[0])
            .s(" acquire, ")
            .n(s.sc_calls[1])
            .s(" try_acquire, ")
            .n(s.sc_calls[2])
            .s(" find\n")
            .done(),
    );
    out.push(Line::new().s("K3GPU route table ").s(which(s.routetable_rust)).s(": ").n(s.rc_calls).s(" calls, ").n(s.rt_nu_calls).s(" next_use calls\n").done());
    if need == 0 {
        return out;
    }
    let (needf, hcf, hhf, drf) = (need as f64, hc as f64, hh as f64, dr as f64);
    out.push(
        Line::new()
            .s("K3GPU experts: ")
            .n(need)
            .s(" needed, ")
            .n(hc)
            .s(" already on the card (")
            .f(1, 100.0 * hcf / needf)
            .s("%), ")
            .n(hh)
            .s(" from the host cache (")
            .f(1, 100.0 * hhf / needf)
            .s("%), ")
            .n(dr)
            .s(" from the drive (")
            .f(1, 100.0 * drf / needf)
            .s("%, ")
            .f(1, drf * (s.expert_raw as f64) / 1e9)
            .s(" GB); drive reads ")
            .f(1, tr)
            .s(" s, host to card ")
            .f(1, tc)
            .s(" s\n")
            .done(),
    );
    for (c, k) in s.card[..cards].iter().enumerate() {
        if k.ready != 0 {
            out.push(
                Line::new()
                    .s("K3GPU card ")
                    .n(c)
                    .s(": ")
                    .n(k.slots)
                    .s(" expert slots, ")
                    .n(k.needed)
                    .s(" needed, ")
                    .n(k.hit_card)
                    .s(" on the card, ")
                    .n(k.hit_host)
                    .s(" host, ")
                    .n(k.from_drive)
                    .s(" drive, ")
                    .n(k.written_back)
                    .s(" written back to RAM\n")
                    .done(),
            );
        }
    }
    if s.rc_calls != 0 {
        out.push(
            Line::new()
                .s("K3GPU route cache: ")
                .n(s.rc_pos)
                .s(" of ")
                .n(s.rc_calls)
                .s(" calls seen, look-ahead ")
                .s(if s.rc_ok != 0 { "on to the end" } else { "stopped" })
                .s("; ")
                .n(s.pf_planned)
                .s(" experts read ahead, ")
                .n(s.pf_skipped)
                .s(" already resident, ")
                .n(s.pf_used)
                .s(" taken from read-ahead, ")
                .n(s.pf_waited)
                .s(" of them waited for (")
                .f(1, s.t_pwait)
                .s(" s); ")
                .n(s.wb_skipped)
                .s(" card victims not kept in RAM\n")
                .done(),
        );
    }
    if s.vq != 0 {
        out.push(
            Line::new()
                .s("K3GPU card queues: ")
                .n(s.vq_ram.wrapping_add(s.vq_drive))
                .s(" experts brought ahead (")
                .n(s.vq_ram)
                .s(" from RAM, ")
                .n(s.vq_drive)
                .s(" from the drive), ")
                .n(s.vq_wb)
                .s(" replaced experts copied to RAM, ")
                .n(s.vq_nowb)
                .s(" dropped; ")
                .n(s.vq_waited)
                .s(" waited for (")
                .f(1, s.t_vwait)
                .s(" s); planning in the hook ")
                .f(1, s.t_plan)
                .s(" s\n")
                .done(),
        );
    }
    let names = ["prompt-pass layer calls", "single-token layer calls", "multi-token layer calls"];
    for i in 0..3 {
        if s.n_calls[i] != 0.0 {
            out.push(
                Line::new()
                    .s("K3GPU ")
                    .s(names[i])
                    .s(": ")
                    .f(0, s.n_calls[i])
                    .s(", ")
                    .f(3, s.t_calls[i])
                    .s(" s in all, ")
                    .f(1, 1000.0 * s.t_calls[i] / s.n_calls[i])
                    .s(" ms each\n")
                    .done(),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------- C ABI

/// # Safety
/// `name` is a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ec_env_int(name: *const c_char, dflt: c_int) -> c_int {
    env_int(&ProcessEnv, CStr::from_ptr(name), dflt)
}

/// # Safety
/// `out` is writable.
#[no_mangle]
pub unsafe extern "C" fn ec_settings_from_env(out: *mut EcSettings, devices: c_int) {
    *out = settings_from(&ProcessEnv, devices);
}

#[no_mangle]
pub extern "C" fn ec_pend_cap(cap_slots: usize) -> usize {
    cap_slots.wrapping_mul(6) / 10
}

#[no_mangle]
pub extern "C" fn ec_open_trace() -> *mut c_void {
    let tr = ProcessEnv.get(c"K3GPU_TRACE");
    if unset_or_empty(tr) {
        return ptr::null_mut();
    }
    let f = unsafe { fopen(tr, c"w".as_ptr()) };
    if f.is_null() {
        fatal(&trace_error_line(tr));
    }
    f
}

/// # Safety
/// `s` is readable; `s->routecache` is NULL or a C string.
#[no_mangle]
pub unsafe extern "C" fn ec_print_route_cache(s: *const EcSettings, calls: usize, pend_cap: usize) {
    emit(&route_cache_line(&*s, calls, pend_cap));
}

#[no_mangle]
pub extern "C" fn ec_prefill_start(calls: usize) {
    if calls == 0 {
        fatal(PREFILL_NEEDS);
    }
    emit(PREFILL_ON);
}

/// # Safety
/// `s` is readable.
#[no_mangle]
pub unsafe extern "C" fn ec_print_start_line(s: *const EcSettings) {
    emit(&start_line(&*s));
}

#[no_mangle]
pub extern "C" fn ec_selector_mode(component: c_int, built: c_int) -> c_int {
    match selector_decide(&ProcessEnv, component, built != 0) {
        SelOutcome::Cpp => {
            emit(&selector_cpp_line(component));
            0
        }
        SelOutcome::Rust => 1,
        SelOutcome::Fatal(msg) => fatal(&msg),
    }
}

#[no_mangle]
pub extern "C" fn ec_selector_lib(component: c_int) -> *const c_char {
    selector_lib_from(&ProcessEnv, component)
}

/// # Safety
/// `lib` is NULL or a C string.
#[no_mangle]
pub unsafe extern "C" fn ec_selector_announce_rust(component: c_int, lib: *const c_char) {
    emit(&selector_announce_line(component, lib));
}

#[no_mangle]
pub extern "C" fn ec_card_of(layout: c_int, cards: c_int, layer: c_int) -> c_int {
    card_of(layout, cards, layer)
}

/// # Safety
/// See include/engcfg.h: `cur` is readable when pos < size; `sel` has room for every write of the dealing.
#[no_mangle]
pub unsafe extern "C" fn ec_force_routes(pos: usize, size: usize, cur: *const EcCall, l: c_int, n: c_int, sel: *mut c_int, top: c_int) {
    force_routes(pos, size, cur, l, n, sel, top);
}

/// # Safety
/// `s` is readable.
#[no_mangle]
pub unsafe extern "C" fn ec_print_summary(s: *const EcSummary) {
    for line in summary_lines(&*s) {
        emit(&line);
    }
}

#[cfg(test)]
mod tests;
