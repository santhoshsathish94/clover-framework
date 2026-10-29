//! Unit tests. Expectations are derived by hand from decode.cuh@4d8d891 (derivations in the comments). Fake ops record every call; the memory of the host cache is real (mmap) with
//! 64 KB "experts" (one 32-slot chunk = 2 MiB). Paths that end the process run the test binary again as a child (`child_entry`, selected by the POOLINIT_CHILD environment variable).
use crate::abi::*;
use crate::{host, pi_host_init, pi_pool_init, pool};
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const RAW: usize = 65536;
const SENTINEL: usize = 0xA5A5_A5A5_A5A5_A5A5;

thread_local! {
    static LOG: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static SPARES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static REG: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static ERR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static CALLS: Cell<u32> = const { Cell::new(0) };
    static FAIL_AT: Cell<u32> = const { Cell::new(0) };
    static FREE: Cell<usize> = const { Cell::new(0) };
    static DARENA: Cell<u64> = const { Cell::new(0) };
    static HARENA: Cell<u64> = const { Cell::new(0) };
    static HND: Cell<u64> = const { Cell::new(0) };
}
// counters read by the child's atexit handler (thread-locals with destructors are gone by then)
static G_EV: AtomicUsize = AtomicUsize::new(0);
static G_SP: AtomicUsize = AtomicUsize::new(0);
static G_POOL_OUT: AtomicUsize = AtomicUsize::new(0);
static G_HOST_OUT: AtomicUsize = AtomicUsize::new(0);

fn reset(free: usize) {
    LOG.with(|l| l.borrow_mut().clear());
    SPARES.with(|l| l.borrow_mut().clear());
    REG.with(|l| l.borrow_mut().clear());
    CALLS.with(|c| c.set(0));
    FAIL_AT.with(|c| c.set(0));
    FREE.with(|c| c.set(free));
    DARENA.with(|c| c.set(0x7000_0000_0000));
    HARENA.with(|c| c.set(0x7e00_0000_0000));
    HND.with(|c| c.set(0x1000));
    G_EV.store(0, Ordering::SeqCst);
    G_SP.store(0, Ordering::SeqCst);
}
fn log(s: String) {
    LOG.with(|l| l.borrow_mut().push(s));
    G_EV.fetch_add(1, Ordering::SeqCst);
}
fn maybe_fail() -> c_int {
    let n = CALLS.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    if n == FAIL_AT.with(|c| c.get()) {
        2
    } else {
        0
    }
}
unsafe extern "C" fn f_set_device(d: c_int) -> c_int {
    log(format!("set_device {}", d));
    maybe_fail()
}
unsafe extern "C" fn f_mem_get_info(f: *mut usize, t: *mut usize) -> c_int {
    log("mem_get_info".into());
    unsafe {
        *f = FREE.with(|c| c.get());
        *t = 1usize << 30;
    }
    maybe_fail()
}
unsafe extern "C" fn f_device_malloc(p: *mut *mut c_void, n: usize) -> c_int {
    let a = DARENA.with(|c| c.get());
    log(format!("malloc {} -> {:x}", n, a));
    unsafe { *p = a as usize as *mut c_void };
    DARENA.with(|c| c.set(a + ((n as u64 + 511) & !511)));
    maybe_fail()
}
unsafe extern "C" fn f_host_register(p: *mut c_void, n: usize, fl: c_uint) -> c_int {
    log(format!("register {} {}", n, fl));
    REG.with(|r| r.borrow_mut().push(p as usize));
    maybe_fail()
}
unsafe extern "C" fn f_host_alloc(p: *mut *mut c_void, n: usize, fl: c_uint) -> c_int {
    let a = HARENA.with(|c| c.get());
    log(format!("host_alloc {} {} -> {:x}", n, fl, a));
    unsafe { *p = a as usize as *mut c_void };
    HARENA.with(|c| c.set(a + n as u64));
    maybe_fail()
}
unsafe extern "C" fn f_stream_create(p: *mut *mut c_void, fl: c_uint) -> c_int {
    let h = HND.with(|c| c.get());
    log(format!("stream {} -> {:x}", fl, h));
    unsafe { *p = h as usize as *mut c_void };
    HND.with(|c| c.set(h + 0x10));
    maybe_fail()
}
unsafe extern "C" fn f_event_create(p: *mut *mut c_void, fl: c_uint) -> c_int {
    let h = HND.with(|c| c.get());
    log(format!("event {} -> {:x}", fl, h));
    unsafe { *p = h as usize as *mut c_void };
    HND.with(|c| c.set(h + 0x10));
    maybe_fail()
}
unsafe extern "C" fn f_error_string(c: c_int) -> *const c_char {
    ERR.with(|e| {
        *e.borrow_mut() = Some(CString::new(format!("fake error {}", c)).unwrap());
        e.borrow().as_ref().unwrap().as_ptr()
    })
}
unsafe extern "C" fn f_add_spare(_c: *mut sc_cache, p: *mut u8) {
    log("spare".into());
    SPARES.with(|s| s.borrow_mut().push(p as usize));
    G_SP.fetch_add(1, Ordering::SeqCst);
}
fn ops() -> pi_ops {
    pi_ops {
        set_device: Some(f_set_device),
        mem_get_info: Some(f_mem_get_info),
        device_malloc: Some(f_device_malloc),
        host_register: Some(f_host_register),
        host_alloc: Some(f_host_alloc),
        stream_create: Some(f_stream_create),
        event_create: Some(f_event_create),
        error_string: Some(f_error_string),
        sc_add_spare: Some(f_add_spare),
    }
}
fn cache() -> *mut sc_cache {
    1usize as *mut sc_cache
}
fn events() -> Vec<String> {
    LOG.with(|l| l.borrow().clone())
}
fn spares() -> Vec<usize> {
    SPARES.with(|l| l.borrow().clone())
}
fn stats_pattern() -> pi_expert_stats {
    pi_expert_stats { needed: 7, hit_card: 8, hit_host: 9, from_drive: 10, written_back: 11, t_read: 1.5, t_copy: 2.5, t_ensure: 3.5 }
}
/// The caller's zero-initialised struct, with the entries init must not write set to a pattern.
fn fresh_out() -> pi_pool_out {
    let mut o: pi_pool_out = unsafe { std::mem::zeroed() };
    o.rstream = [SENTINEL as *mut c_void; PI_MAX_READERS];
    o.bounce = [SENTINEL as *mut u8; PI_MAX_READERS];
    o.st = stats_pattern();
    o
}
fn pool_cfg(dev: c_int, reserve: usize, readers: c_int, raw: usize, experts: c_int, file: *const c_char) -> pi_pool_cfg {
    pi_pool_cfg { device: dev, reserve_bytes: reserve, nreaders: readers, expert_raw: raw, experts, cache: cache(), src_file: file }
}
fn host_cfg(bytes: usize, raw: usize, file: *const c_char) -> pi_host_cfg {
    pi_host_cfg { bytes, expert_raw: raw, cache: cache(), src_file: file }
}
fn run_pool(cfg: &pi_pool_cfg, out: &mut pi_pool_out) {
    let o = ops();
    unsafe { pi_pool_init(&o, cfg, out) }
}
fn run_host(cfg: &pi_host_cfg, out: &mut pi_host_out) {
    let o = ops();
    unsafe { pi_host_init(&o, cfg, out) }
}

// ---------------------------------------------------------------- child processes
extern "C" {
    fn atexit(cb: extern "C" fn()) -> c_int;
}
extern "C" fn dump_at_exit() {
    use std::io::Write;
    let (ev, sp) = (G_EV.load(Ordering::SeqCst), G_SP.load(Ordering::SeqCst));
    let mut e = std::io::stderr();
    let p = G_POOL_OUT.load(Ordering::SeqCst) as *const pi_pool_out;
    if !p.is_null() {
        let o = unsafe { &*p };
        let _ = write!(e, "ATEXIT dev={} slots={} readers={} ready={} ev={} sp={}\n", o.dev, o.slots, o.readers, o.ready, ev, sp);
    }
    let h = G_HOST_OUT.load(Ordering::SeqCst) as *const pi_host_out;
    if !h.is_null() {
        let o = unsafe { &*h };
        let _ = write!(e, "ATEXIT cap={} ready={} ev={} sp={}\n", o.cap_slots, o.ready, ev, sp);
    }
}
/// Runs the scenario `spec` in a child process (this test binary, one test); returns (exit code, stderr).
fn child(spec: &str) -> (Option<i32>, String) {
    let exe = std::env::current_exe().unwrap();
    let o = Command::new(exe).args(["--exact", "tests::child_entry", "--nocapture", "--test-threads=1"]).env("POOLINIT_CHILD", spec).output().unwrap();
    (o.status.code(), String::from_utf8_lossy(&o.stderr).into_owned())
}
/// spec = "pool <free> <reserve> <readers> <dev> <fail_at> <file|-> <raw> <experts>", "host <bytes> <fail_at> <file|-> <raw>", "null <which>"
#[test]
fn child_entry() {
    let spec = match std::env::var("POOLINIT_CHILD") {
        Ok(s) => s,
        Err(_) => return,
    };
    let a: Vec<&str> = spec.split(' ').collect();
    let file = |s: &str| -> Option<CString> { if s == "-" { None } else { Some(CString::new(s).unwrap()) } };
    match a[0] {
        "pool" => {
            reset(a[1].parse().unwrap());
            FAIL_AT.with(|c| c.set(a[5].parse().unwrap()));
            let f = file(a[6]);
            let cfg = pool_cfg(a[4].parse().unwrap(), a[2].parse().unwrap(), a[3].parse().unwrap(), a[7].parse().unwrap(), a[8].parse().unwrap(), f.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()));
            let mut out = Box::new(fresh_out());
            G_POOL_OUT.store(&*out as *const pi_pool_out as usize, Ordering::SeqCst);
            unsafe { atexit(dump_at_exit) };
            run_pool(&cfg, &mut out);
            eprintln!("RETURNED");
        }
        "host" => {
            reset(0);
            FAIL_AT.with(|c| c.set(a[2].parse().unwrap()));
            let f = file(a[3]);
            let cfg = host_cfg(a[1].parse().unwrap(), a[4].parse().unwrap(), f.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()));
            let mut out = Box::new(pi_host_out { cap_slots: 0, ready: 0 });
            G_HOST_OUT.store(&*out as *const pi_host_out as usize, Ordering::SeqCst);
            unsafe { atexit(dump_at_exit) };
            run_host(&cfg, &mut out);
            eprintln!("RETURNED");
        }
        "null" => {
            let o = ops();
            let pc = pool_cfg(0, 0, 1, RAW, 4, std::ptr::null());
            let hc = host_cfg(RAW, RAW, std::ptr::null());
            let mut po = fresh_out();
            let mut ho = pi_host_out { cap_slots: 0, ready: 0 };
            unsafe {
                match a[1] {
                    "pool_ops" => pi_pool_init(std::ptr::null(), &pc, &mut po),
                    "pool_cfg" => pi_pool_init(&o, std::ptr::null(), &mut po),
                    "pool_out" => pi_pool_init(&o, &pc, std::ptr::null_mut()),
                    "host_ops" => pi_host_init(std::ptr::null(), &hc, &mut ho),
                    "host_cfg" => pi_host_init(&o, std::ptr::null(), &mut ho),
                    "host_out" => pi_host_init(&o, &hc, std::ptr::null_mut()),
                    "pool_fn" => {
                        let mut o2 = o;
                        o2.host_alloc = None;
                        pi_pool_init(&o2, &pc, &mut po)
                    }
                    "host_fn" => {
                        let mut o2 = o;
                        o2.sc_add_spare = None;
                        pi_host_init(&o2, &hc, &mut ho)
                    }
                    "pool_raw0" => pi_pool_init(&o, &pool_cfg(0, 0, 1, 0, 4, std::ptr::null()), &mut po),
                    "host_raw0" => pi_host_init(&o, &host_cfg(RAW, 0, std::ptr::null()), &mut ho),
                    other => panic!("unknown null case {}", other),
                }
            }
            eprintln!("RETURNED");
        }
        other => panic!("unknown child scenario {}", other),
    }
}

// ---------------------------------------------------------------- card pool
/// Free = reserve (1 MiB) + 100 experts + 5000 bytes: usable = 100*65536 + 5000, slots = 100 (usable / raw = 100.07). Three readers.
/// Calls: set_device 3, mem_get_info, malloc 100*65536 = 6553600 at the arena start 0x700000000000 (the arena moves on by the size rounded up to 512: 6553600 = 12800*512, so 0x640000),
/// 100 spares base+99*raw down to base, stream (copy) 0x1000, then per reader stream (0x1010, 0x1020, 0x1030) and a 64 KB host_alloc (0x7e0000000000, +0x10000, +0x20000), event 0x1040
/// (flag 2), malloc 10*8 = 80 at 0x700000640000.
#[test]
fn pool_calls_and_out_fields() {
    reset((1 << 20) + 100 * RAW + 5000);
    let mut out = fresh_out();
    run_pool(&pool_cfg(3, 1 << 20, 3, RAW, 10, std::ptr::null()), &mut out);
    let ev = events();
    assert_eq!(ev.len(), 112);
    assert_eq!(&ev[0], "set_device 3");
    assert_eq!(&ev[1], "mem_get_info");
    assert_eq!(&ev[2], "malloc 6553600 -> 700000000000");
    assert!(ev[3..103].iter().all(|e| e == "spare"));
    let want: Vec<&str> = vec![
        "stream 1 -> 1000",
        "stream 1 -> 1010",
        "host_alloc 65536 0 -> 7e0000000000",
        "stream 1 -> 1020",
        "host_alloc 65536 0 -> 7e0000010000",
        "stream 1 -> 1030",
        "host_alloc 65536 0 -> 7e0000020000",
        "event 2 -> 1040",
        "malloc 80 -> 700000640000",
    ];
    assert_eq!(ev[103..].iter().map(|s| s.as_str()).collect::<Vec<_>>(), want);
    let sp = spares();
    assert_eq!(sp.len(), 100);
    for (j, a) in sp.iter().enumerate() {
        assert_eq!(*a as u64, 0x7000_0000_0000u64 + (99 - j as u64) * RAW as u64);
    }
    assert_eq!((out.dev, out.slots, out.readers, out.ready), (3, 100, 3, 1));
    assert_eq!(out.base as usize, 0x7000_0000_0000);
    assert_eq!(out.copy as usize, 0x1000);
    assert_eq!([out.rstream[0] as usize, out.rstream[1] as usize, out.rstream[2] as usize], [0x1010, 0x1020, 0x1030]);
    assert_eq!([out.bounce[0] as usize, out.bounce[1] as usize, out.bounce[2] as usize], [0x7e00_0000_0000, 0x7e00_0001_0000, 0x7e00_0002_0000]);
    assert_eq!(out.written_back as usize, 0x1040);
    assert_eq!(out.d_eptr as usize, 0x7000_0064_0000);
    // entries from `readers` on and the stats were not written
    assert!(out.rstream[3..].iter().all(|p| *p as usize == SENTINEL) && out.bounce[3..].iter().all(|p| *p as usize == SENTINEL));
    assert_eq!(out.st, stats_pattern());
}

#[test]
fn pool_slot_count_boundaries() {
    let reserve = 1usize << 20;
    assert_eq!(pool::slots_for(reserve + 64 * RAW, reserve, RAW), 64); // exactly 64 slots: accepted
    assert_eq!(pool::slots_for(reserve + 64 * RAW - 1, reserve, RAW), 63); // one byte less: 63, refused
    assert_eq!(pool::slots_for(reserve + 64 * RAW + RAW - 1, reserve, RAW), 64);
    assert_eq!(pool::slots_for(reserve, reserve, RAW), 0);
    assert_eq!(pool::slots_for(reserve - 1, reserve, RAW), 0); // free below the reserve: usable 0, no wrap
    assert_eq!(pool::slots_for(0, 5, RAW), 0);
    assert_eq!(pool::slots_for(100 * RAW, 0, RAW), 100);
    // the cast to int wraps like C's: usable 2^31 + 5 with raw 1 -> -2147483643; 2^34 experts of 64 KB -> 0
    assert_eq!(pool::slots_for((1usize << 31) + 5, 0, 1), i32::MIN + 5);
    assert_eq!(pool::slots_for((1usize << 34) * RAW, 0, RAW), 0);
    assert_eq!(pool::refusal_message(5, 63), "card 5: only 63 expert slots fit after the reserve\n");
    assert_eq!(pool::refusal_message(0, -2147483643), "card 0: only -2147483643 expert slots fit after the reserve\n");
}

/// 40 readers become 32: calls = 3 (set_device, mem_get_info, malloc) + 64 spares + 1 stream + 32*2 + event + malloc = 134; a negative count makes none: 3 + 64 + 1 + 2 = 70 calls.
#[test]
fn pool_readers_clamped_and_negative() {
    reset(64 * RAW);
    let mut out = fresh_out();
    run_pool(&pool_cfg(0, 0, 40, RAW, 896, std::ptr::null()), &mut out);
    let ev = events();
    assert_eq!((out.slots, out.readers, out.ready), (64, 32, 1));
    assert_eq!(ev.len(), 134);
    assert_eq!(&ev[2], "malloc 4194304 -> 700000000000"); // 64 * 65536
    assert_eq!(&ev[133], "malloc 7168 -> 700000400000"); // 896 * 8, after the 0x400000 slab
    assert!(out.rstream[31] as usize != SENTINEL && out.bounce[31] as usize != SENTINEL);
    reset(64 * RAW + 1);
    let mut out = fresh_out();
    run_pool(&pool_cfg(0, 0, -2, RAW, 4, std::ptr::null()), &mut out);
    assert_eq!((out.readers, out.ready), (-2, 1));
    assert_eq!(events().len(), 70);
    assert!(out.rstream.iter().all(|p| *p as usize == SENTINEL));
}

/// The engine's constants through the fake ops (nothing is touched): raw 17547264 (= 512 * 34272), 896 experts, reserve 2500 MiB = 2621440000, free 22000000000:
/// usable = 19378560000, / 17547264 = 1104.35 -> 1104 slots; slab = 1104 * 17547264 = 19372179456 bytes; 16 readers: calls = 3 + 1104 + 1 + 32 + 2 = 1142; d_eptr table 896 * 8 = 7168 bytes.
#[test]
fn pool_with_real_constants() {
    const REAL_RAW: usize = 17547264;
    reset(22_000_000_000);
    let mut out = fresh_out();
    run_pool(&pool_cfg(2, 2500 << 20, 16, REAL_RAW, 896, std::ptr::null()), &mut out);
    let ev = events();
    assert_eq!(out.slots, 1104);
    assert_eq!(ev.len(), 1142);
    assert_eq!(&ev[2], "malloc 19372179456 -> 700000000000");
    assert_eq!(&ev[3 + 1104 + 1 + 1], "host_alloc 17547264 0 -> 7e0000000000");
    assert_eq!(ev[ev.len() - 1], format!("malloc 7168 -> {:x}", 0x7000_0000_0000u64 + 19372179456u64));
    let sp = spares();
    assert_eq!((sp[0] as u64, sp[1103] as u64), (0x7000_0000_0000u64 + 1103 * REAL_RAW as u64, 0x7000_0000_0000u64));
    assert_eq!(pool::slots_for(22_000_000_000, 2500 << 20, REAL_RAW), 1104);
}

#[test]
fn pool_refusals_and_failed_checks_end_the_process() {
    // 63 slots: the text of the original, exit 1, the out struct already holds dev 3 and slots 63 (ready 0); only set_device and mem_get_info were called
    let (code, err) = child(&format!("pool {} {} 3 3 0 - {} 10", (1usize << 20) + 63 * RAW + 65535, 1usize << 20, RAW));
    assert_eq!(code, Some(1));
    assert_eq!(err, "card 3: only 63 expert slots fit after the reserve\nATEXIT dev=3 slots=63 readers=3 ready=0 ev=2 sp=0\n");
    let (code, err) = child(&format!("pool 12345 {} 1 5 0 - {} 10", 1usize << 20, RAW));
    assert_eq!(code, Some(1));
    assert_eq!(err, "card 5: only 0 expert slots fit after the reserve\nATEXIT dev=5 slots=0 readers=1 ready=0 ev=2 sp=0\n");
    // the k-th CUDA call fails with code 2 (spares are not CUDA calls): the CHECK line of decode.cuh@4d8d891, "file:line: fake error 2"; slots is stored once mem_get_info returned
    // (k >= 3), 100 spares are added after the slab call (k >= 4); events = k up to the slab call, 100 + k after it
    let lines = [454, 455, 459, 461, 463, 464, 463, 464, 463, 464, 466, 467];
    for k in 1..=12usize {
        let (code, err) = child(&format!("pool {} 0 3 1 {} x/decode.cuh {} 10", 100 * RAW, k, RAW));
        let want = format!(
            "x/decode.cuh:{}: fake error 2\nATEXIT dev=1 slots={} readers=3 ready=0 ev={} sp={}\n",
            lines[k - 1],
            if k >= 3 { 100 } else { 0 },
            if k <= 3 { k } else { 100 + k },
            if k >= 4 { 100 } else { 0 }
        );
        assert_eq!((code, err), (Some(1), want), "failing call {}", k);
    }
    // no src_file: "decode.cuh"
    let (code, err) = child(&format!("pool {} 0 3 1 1 - {} 10", 100 * RAW, RAW));
    assert_eq!((code, err.as_str()), (Some(1), "decode.cuh:454: fake error 2\nATEXIT dev=1 slots=0 readers=3 ready=0 ev=1 sp=0\n"));
}

// ---------------------------------------------------------------- host cache
/// bytes = 3*32*raw + 100*raw: cap_slots 196, n_chunks 6 (6*32 = 192; 4 slots dropped), chunk_bytes 32*65536 = 2097152. Calls: set_device 0, 6 registers (2097152, flag 1), 192 spares: the first
/// is chunk 5 slot 31, the last chunk 0 slot 0, where chunk i is the i-th registered region. The memory is real: page aligned, distinct, zero.
#[test]
fn host_calls_spare_order_and_memory() {
    reset(0);
    let mut out = pi_host_out { cap_slots: 0, ready: 0 };
    run_host(&host_cfg(3 * 32 * RAW + 100 * RAW, RAW, std::ptr::null()), &mut out);
    assert_eq!((out.cap_slots, out.ready), (192, 1));
    let ev = events();
    assert_eq!(ev.len(), 1 + 6 + 192);
    assert_eq!(&ev[0], "set_device 0");
    assert!(ev[1..7].iter().all(|e| e == "register 2097152 1"));
    assert!(ev[7..].iter().all(|e| e == "spare"));
    let reg = REG.with(|r| r.borrow().clone());
    assert_eq!(reg.len(), 6);
    for (i, a) in reg.iter().enumerate() {
        assert_eq!(a % 4096, 0);
        for b in &reg[..i] {
            assert!(*a + 2097152 <= *b || *b + 2097152 <= *a, "regions overlap");
        }
        let words = unsafe { std::slice::from_raw_parts(*a as *const u64, 2097152 / 8) };
        assert!(words.iter().all(|w| *w == 0));
    }
    let sp = spares();
    assert_eq!(sp.len(), 192);
    for (k, a) in sp.iter().enumerate() {
        assert_eq!(*a, reg[5 - k / 32] + (31 - k % 32) * RAW);
    }
}

#[test]
fn host_zero_chunks_and_tiny_budgets() {
    for bytes in [31 * RAW + 5, RAW - 1, 0, 32 * RAW - 1] {
        reset(0);
        let mut out = pi_host_out { cap_slots: 99, ready: 0 };
        run_host(&host_cfg(bytes, RAW, std::ptr::null()), &mut out);
        assert_eq!((out.cap_slots, out.ready), (0, 1), "bytes {}", bytes);
        assert_eq!(events(), vec!["set_device 0".to_string()]); // the device is still selected; nothing registered, no spare
    }
    // exactly one chunk: 32 slots, chunk 0 slot 31 first
    reset(0);
    let mut out = pi_host_out { cap_slots: 0, ready: 0 };
    run_host(&host_cfg(32 * RAW, RAW, std::ptr::null()), &mut out);
    assert_eq!(out.cap_slots, 32);
    assert_eq!(events().len(), 1 + 1 + 32);
    let reg = REG.with(|r| r.borrow().clone());
    let sp = spares();
    assert_eq!((sp[0], sp[31]), (reg[0] + 31 * RAW, reg[0]));
}

/// The real constants, arithmetic only (a real chunk is 561 MB): 150 GiB = 161061273600 bytes / 17547264 = 9178.7 -> 9178 slots; 9178 / 32 = 286 chunks (286*32 = 9152);
/// chunk = 32 * 17547264 = 561512448 bytes; 9152 * 17547264 = 160592560128 -> "160.6 GB".
#[test]
fn host_plan_and_summary_with_real_constants() {
    assert_eq!(host::plan(150usize << 30, 17547264), (9178, 561512448, 286));
    assert_eq!(host::plan(32 * 17547264, 17547264), (32, 561512448, 1));
    assert_eq!(host::plan(32 * 17547264 - 1, 17547264), (31, 561512448, 0));
    assert_eq!(host::summary_line(9152, 17547264, 3.24), "K3GPU: host expert cache of 9152 slots (160.6 GB) pinned in 3.2 s\n");
    assert_eq!(host::summary_line(0, 17547264, 0.04), "K3GPU: host expert cache of 0 slots (0.0 GB) pinned in 0.0 s\n");
    // the real engine with an empty budget (bytes < 32 experts): through the fake ops, nothing mapped
    reset(0);
    let mut out = pi_host_out { cap_slots: 1, ready: 0 };
    run_host(&host_cfg(31 * 17547264, 17547264, std::ptr::null()), &mut out);
    assert_eq!(out.cap_slots, 0);
}

#[test]
fn host_failures_end_the_process() {
    let bytes = 3 * 32 * RAW + 100 * RAW;
    // first CUDA call (set_device 0): line 428; cap_slots is still 196 (stored first, trimmed to 192 at the end), ready 1
    let (code, err) = child(&format!("host {} 1 x/decode.cuh {}", bytes, RAW));
    assert_eq!((code, err.as_str()), (Some(1), "x/decode.cuh:428: fake error 2\nATEXIT cap=196 ready=1 ev=1 sp=0\n"));
    // the third register (4th call): line 429, 4 events, no spare yet
    let (code, err) = child(&format!("host {} 4 x/decode.cuh {}", bytes, RAW));
    assert_eq!((code, err.as_str()), (Some(1), "x/decode.cuh:429: fake error 2\nATEXIT cap=196 ready=1 ev=4 sp=0\n"));
    let (code, err) = child(&format!("host {} 1 - {}", bytes, RAW));
    assert_eq!((code, err.as_str()), (Some(1), "decode.cuh:428: fake error 2\nATEXIT cap=196 ready=1 ev=1 sp=0\n"));
}

#[test]
fn null_arguments_and_zero_expert_raw_are_refused_first() {
    let cases = [
        ("pool_ops", "poolinit: ops is NULL\n"),
        ("pool_cfg", "poolinit: cfg is NULL\n"),
        ("pool_out", "poolinit: out is NULL\n"),
        ("host_ops", "poolinit: ops is NULL\n"),
        ("host_cfg", "poolinit: cfg is NULL\n"),
        ("host_out", "poolinit: out is NULL\n"),
        ("pool_fn", "poolinit: ops->host_alloc is NULL\n"),
        ("host_fn", "poolinit: ops->sc_add_spare is NULL\n"),
        ("pool_raw0", "poolinit: expert_raw is 0\n"),
        ("host_raw0", "poolinit: expert_raw is 0\n"),
    ];
    for (c, msg) in cases {
        let (code, err) = child(&format!("null {}", c));
        assert_eq!((code, err.as_str()), (Some(1), msg), "case {}", c);
    }
}

// ---------------------------------------------------------------- formatting and layout
extern "C" {
    fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> c_int;
}
fn c_fmt1(x: f64) -> String {
    let mut b = [0 as c_char; 64];
    let n = unsafe { snprintf(b.as_mut_ptr(), 64, c"%.1f".as_ptr(), x) };
    b[..n as usize].iter().map(|c| *c as u8 as char).collect()
}
#[test]
fn one_decimal_formatting_equals_c_printf() {
    let mut v: Vec<f64> = vec![0.0, 0.04, 0.05, 0.06, 0.15, 0.25, 0.35, 0.45, 0.75, 1.25, 2.5, 160.592560128, 0.0126, 1e9 / 4.0, 12.35, 99.95, 0.95, 3.0, 7.0 / 8.0];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..20000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push((s >> 11) as f64 / 9007199254740992.0 * 1000.0);
        v.push(((s >> 40) as f64) / 8.0); // many exact ties at one decimal (multiples of 1/8)
        v.push(((s >> 40) as f64) / 64.0);
    }
    for x in v {
        assert_eq!(format!("{:.1}", x), c_fmt1(x), "value {:e}", x);
    }
    // the sizes the engine prints: cap_slots * (double)raw / 1e9
    for slots in [0usize, 32, 192, 9152, 1104] {
        let g = slots as f64 * (17547264usize as f64) / 1e9;
        assert_eq!(format!("{:.1}", g), c_fmt1(g));
    }
}

#[test]
fn struct_layout_equals_the_c_header() {
    use std::mem::{offset_of, size_of};
    let dir = std::env::temp_dir().join(format!("poolinit_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("layout.c");
    let exe = dir.join("layout");
    std::fs::write(
        &c,
        r#"#include <stdio.h>
#include <stddef.h>
#include "poolinit.h"
#define O(t, f) printf(" %zu", offsetof(t, f))
int main(void) {
  printf("sizes %zu %zu %zu %zu %zu %zu\n", sizeof(pi_ops), sizeof(pi_expert_stats), sizeof(pi_host_cfg), sizeof(pi_host_out), sizeof(pi_pool_cfg), sizeof(pi_pool_out));
  printf("ops"); O(pi_ops, set_device); O(pi_ops, mem_get_info); O(pi_ops, device_malloc); O(pi_ops, host_register); O(pi_ops, host_alloc); O(pi_ops, stream_create); O(pi_ops, event_create); O(pi_ops, error_string); O(pi_ops, sc_add_spare); printf("\n");
  printf("stats"); O(pi_expert_stats, needed); O(pi_expert_stats, hit_card); O(pi_expert_stats, hit_host); O(pi_expert_stats, from_drive); O(pi_expert_stats, written_back); O(pi_expert_stats, t_read); O(pi_expert_stats, t_copy); O(pi_expert_stats, t_ensure); printf("\n");
  printf("hcfg"); O(pi_host_cfg, bytes); O(pi_host_cfg, expert_raw); O(pi_host_cfg, cache); O(pi_host_cfg, src_file); printf("\n");
  printf("hout"); O(pi_host_out, cap_slots); O(pi_host_out, ready); printf("\n");
  printf("pcfg"); O(pi_pool_cfg, device); O(pi_pool_cfg, reserve_bytes); O(pi_pool_cfg, nreaders); O(pi_pool_cfg, expert_raw); O(pi_pool_cfg, experts); O(pi_pool_cfg, cache); O(pi_pool_cfg, src_file); printf("\n");
  printf("pout"); O(pi_pool_out, dev); O(pi_pool_out, slots); O(pi_pool_out, readers); O(pi_pool_out, ready); O(pi_pool_out, base); O(pi_pool_out, copy); O(pi_pool_out, rstream); O(pi_pool_out, bounce); O(pi_pool_out, written_back); O(pi_pool_out, d_eptr); O(pi_pool_out, st); printf("\n");
  printf("consts %d %d %u %u %u %u\n", PI_CHUNK_SLOTS, PI_MAX_READERS, PI_HOST_ALLOC_DEFAULT, PI_HOST_REGISTER_PORTABLE, PI_STREAM_NON_BLOCKING, PI_EVENT_DISABLE_TIMING);
  return 0;
}
"#,
    )
    .unwrap();
    let m = env!("CARGO_MANIFEST_DIR");
    let inc = format!("{}/include", m);
    let sc: Vec<String> = [format!("{}/../slotcache/include", m), format!("{}/slotcache-include", m)].into_iter().filter(|p| std::path::Path::new(p).is_dir()).collect();
    assert!(!sc.is_empty(), "slotcache.h not found next to the crate (../slotcache/include)");
    let st = Command::new("cc").args(["-O0", "-I", &inc, "-I", &sc[0], "-o"]).arg(&exe).arg(&c).status().expect("cc");
    assert!(st.success());
    let got = String::from_utf8(Command::new(&exe).output().unwrap().stdout).unwrap();
    let want = format!(
        "sizes {} {} {} {} {} {}\nops {} {} {} {} {} {} {} {} {}\nstats {} {} {} {} {} {} {} {}\nhcfg {} {} {} {}\nhout {} {}\npcfg {} {} {} {} {} {} {}\npout {} {} {} {} {} {} {} {} {} {} {}\nconsts {} {} {} {} {} {}\n",
        size_of::<pi_ops>(), size_of::<pi_expert_stats>(), size_of::<pi_host_cfg>(), size_of::<pi_host_out>(), size_of::<pi_pool_cfg>(), size_of::<pi_pool_out>(),
        offset_of!(pi_ops, set_device), offset_of!(pi_ops, mem_get_info), offset_of!(pi_ops, device_malloc), offset_of!(pi_ops, host_register), offset_of!(pi_ops, host_alloc),
        offset_of!(pi_ops, stream_create), offset_of!(pi_ops, event_create), offset_of!(pi_ops, error_string), offset_of!(pi_ops, sc_add_spare),
        offset_of!(pi_expert_stats, needed), offset_of!(pi_expert_stats, hit_card), offset_of!(pi_expert_stats, hit_host), offset_of!(pi_expert_stats, from_drive),
        offset_of!(pi_expert_stats, written_back), offset_of!(pi_expert_stats, t_read), offset_of!(pi_expert_stats, t_copy), offset_of!(pi_expert_stats, t_ensure),
        offset_of!(pi_host_cfg, bytes), offset_of!(pi_host_cfg, expert_raw), offset_of!(pi_host_cfg, cache), offset_of!(pi_host_cfg, src_file),
        offset_of!(pi_host_out, cap_slots), offset_of!(pi_host_out, ready),
        offset_of!(pi_pool_cfg, device), offset_of!(pi_pool_cfg, reserve_bytes), offset_of!(pi_pool_cfg, nreaders), offset_of!(pi_pool_cfg, expert_raw),
        offset_of!(pi_pool_cfg, experts), offset_of!(pi_pool_cfg, cache), offset_of!(pi_pool_cfg, src_file),
        offset_of!(pi_pool_out, dev), offset_of!(pi_pool_out, slots), offset_of!(pi_pool_out, readers), offset_of!(pi_pool_out, ready), offset_of!(pi_pool_out, base),
        offset_of!(pi_pool_out, copy), offset_of!(pi_pool_out, rstream), offset_of!(pi_pool_out, bounce), offset_of!(pi_pool_out, written_back),
        offset_of!(pi_pool_out, d_eptr), offset_of!(pi_pool_out, st),
        PI_CHUNK_SLOTS, PI_MAX_READERS, PI_HOST_ALLOC_DEFAULT, PI_HOST_REGISTER_PORTABLE, PI_STREAM_NON_BLOCKING, PI_EVENT_DISABLE_TIMING
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(got, want);
}
