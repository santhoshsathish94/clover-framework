//! Unit tests. Expectations are derived by hand from the original text of pool_ensure (decode.cuh at 4d8d891, lines 536-620) and from include/supply.h; the fake ops table records every call in one
//! ordered log (same notation as ref/smoke_ref.c: DS device_sync, SS(s) stream_sync, SD(dev) set_device, D2H/H2D(dst,src,n,stream), ER event_record, WE wait_event, kX check of site X,
//! PF/PT/PA pool find/touch/acquire, HF/HW/HR/HL host find/worth/remove/release, HA host_acquire, PN pend_take, AW arr_wait; Pn pool slot n, Hn host slot n, Bn bounce n, S+off slab).
//! The fakes share one global world, so the tests run one at a time (SERIAL).
use crate::abi::*;
use crate::ensure::{chunking, site, write_stderr, Supply, WB_SKIPPED};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::os::unix::io::AsRawFd;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

const RAW: usize = 10000;
const NEXP: usize = 16;
const LAYER: i32 = 5; // key = 5 * 16 + e = 80 + e

static SERIAL: Mutex<()> = Mutex::new(());
static POOLC: u8 = 0;
static HOSTC: u8 = 0;
fn pool_cache() -> *mut sc_cache {
    &POOLC as *const u8 as *mut sc_cache
}
fn host_cache() -> *mut sc_cache {
    &HOSTC as *const u8 as *mut sc_cache
}

#[derive(Default)]
struct Fk {
    held: Vec<(i32, usize)>,
    acq: Vec<(usize, i32, f32)>,
    ia: usize,
    notworth: Vec<i32>,
}

struct World {
    log: String,
    pm: Vec<Vec<u8>>,
    hm: Vec<Vec<u8>>,
    bm: Vec<Vec<u8>>,
    sm: Vec<u8>,
    pool: Fk,
    host: Fk,
    recs: Vec<(usize, usize, usize)>, // reader copies: (dst address, bytes, stream index)
    inject: c_int,
    check_returns: bool,
    copy_s: Box<u8>,
    rs: Vec<Box<u8>>,
    ev: Box<u8>,
}

static W: Mutex<Option<World>> = Mutex::new(None);

fn w<R>(f: impl FnOnce(&mut World) -> R) -> R {
    let mut g = W.lock().unwrap_or_else(|e| e.into_inner());
    f(g.as_mut().expect("world"))
}

impl World {
    fn nm(&self, a: usize) -> String {
        if a == 0 {
            return "-".into();
        }
        for (tag, v) in [("P", &self.pm), ("H", &self.hm), ("B", &self.bm)] {
            for (i, s) in v.iter().enumerate() {
                let b = s.as_ptr() as usize;
                if a >= b && a < b + RAW {
                    return if a == b { format!("{}{}", tag, i) } else { format!("{}{}+{}", tag, i, a - b) };
                }
            }
        }
        let b = self.sm.as_ptr() as usize;
        if a >= b && a < b + self.sm.len() {
            return format!("S+{}", a - b);
        }
        "?".into()
    }
    fn sn(&self, a: usize) -> String {
        if a == 0 {
            return "0".into();
        }
        if a == &*self.copy_s as *const u8 as usize {
            return "c".into();
        }
        for (i, s) in self.rs.iter().enumerate() {
            if a == &**s as *const u8 as usize {
                return format!("r{}", i);
            }
        }
        "?".into()
    }
    fn stream_idx(&self, a: usize) -> usize {
        self.rs.iter().position(|s| a == &**s as *const u8 as usize).unwrap_or(99)
    }
    fn fk(&mut self, c: *const sc_cache) -> (&mut Fk, &'static str) {
        if c == pool_cache() as *const sc_cache {
            (&mut self.pool, "P")
        } else {
            (&mut self.host, "H")
        }
    }
}

unsafe extern "C" fn f_device_sync() -> c_int {
    w(|x| x.log.push_str("DS;"));
    0
}
unsafe extern "C" fn f_stream_sync(s: *mut c_void) -> c_int {
    w(|x| {
        let n = x.sn(s as usize);
        x.log.push_str(&format!("SS({});", n));
    });
    0
}
unsafe extern "C" fn f_set_device(d: c_int) -> c_int {
    w(|x| {
        x.log.push_str(&format!("SD({});", d));
        if x.inject != 0 && d == 5 {
            x.inject
        } else {
            0
        }
    })
}
unsafe extern "C" fn f_d2h(dst: *mut c_void, src: *const c_void, n: usize, s: *mut c_void) -> c_int {
    w(|x| {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n);
        let l = format!("D2H({},{},{},{});", x.nm(dst as usize), x.nm(src as usize), n, x.sn(s as usize));
        x.log.push_str(&l);
    });
    0
}
unsafe extern "C" fn f_h2d(dst: *mut c_void, src: *const c_void, n: usize, s: *mut c_void) -> c_int {
    w(|x| {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n);
        let l = format!("H2D({},{},{},{});", x.nm(dst as usize), x.nm(src as usize), n, x.sn(s as usize));
        x.log.push_str(&l);
        if s as usize != &*x.copy_s as *const u8 as usize {
            let si = x.stream_idx(s as usize);
            x.recs.push((dst as usize, n, si));
            // a reader copy must come from the bounce slot (or slab) of the same thread index as its stream
            if si < 4 && x.nm(src as usize) != format!("B{}", si) && !x.nm(src as usize).starts_with("S+") {
                x.log.push_str("BAD-BOUNCE;");
            }
        }
    });
    0
}
unsafe extern "C" fn f_event_record(_e: *mut c_void, s: *mut c_void) -> c_int {
    w(|x| {
        let n = x.sn(s as usize);
        x.log.push_str(&format!("ER(ev,{});", n));
    });
    0
}
unsafe extern "C" fn f_wait_event(s: *mut c_void, _e: *mut c_void, _f: c_uint) -> c_int {
    w(|x| {
        let n = x.sn(s as usize);
        x.log.push_str(&format!("WE({},ev);", n));
    });
    0
}
const SITES: [(&[u8], &str); 11] = [
    (site::STREAM_SYNC_0, "k0"),
    (site::DEVICE_SYNC, "kD"),
    (site::SET_DEVICE_POOL, "kS"),
    (site::D2H, "kW"),
    (site::EVENT_RECORD, "kE"),
    (site::H2D_MAIN, "kH"),
    (site::SET_DEVICE_JOB, "kd"),
    (site::WAIT_EVENT, "kw"),
    (site::H2D_JOB, "kc"),
    (site::SYNC_JOB, "ks"),
    (site::SYNC_COPY, "kC"),
];
unsafe extern "C" fn f_check(status: c_int, expr: *const c_char) {
    let e = std::ffi::CStr::from_ptr(expr).to_bytes_with_nul();
    let code = SITES.iter().find(|(s, _)| *s == e).map(|(_, c)| *c).unwrap_or("k?");
    let (ret, msg) = w(|x| {
        if status != 0 {
            x.log.push_str(&format!("{}!{};", code, status));
            (x.check_returns, format!("cuda error {} at {}\n", status, String::from_utf8_lossy(&e[..e.len() - 1])))
        } else {
            x.log.push_str(&format!("{};", code));
            (true, String::new())
        }
    });
    if status != 0 && !ret {
        write_stderr(msg.as_bytes());
        std::process::exit(1);
    }
}
unsafe extern "C" fn f_find(c: *mut sc_cache, key: c_int) -> *mut u8 {
    w(|x| {
        let (f, tag) = x.fk(c);
        let r = f.held.iter().find(|(k, _)| *k == key).map(|(_, s)| *s).unwrap_or(0);
        let n = x.nm(r);
        x.log.push_str(&format!("{}F{}>{};", tag, key, n));
        r as *mut u8
    })
}
unsafe extern "C" fn f_touch(c: *mut sc_cache, key: c_int, wt: f32) {
    w(|x| {
        let (_, tag) = x.fk(c);
        x.log.push_str(&format!("{}T{},{};", tag, key, wt));
    });
}
unsafe extern "C" fn f_worth(c: *const sc_cache, key: c_int) -> c_int {
    w(|x| {
        let (f, tag) = x.fk(c);
        let r = !f.notworth.contains(&key);
        x.log.push_str(&format!("{}W{}={};", tag, key, r as i32));
        r as c_int
    })
}
unsafe extern "C" fn f_acquire(c: *mut sc_cache, key: c_int, prot: *const c_int, np: usize, h0: f32, vk: *mut c_int, vh: *mut f32) -> *mut u8 {
    let mut p: Vec<i32> = std::slice::from_raw_parts(prot, np).to_vec();
    p.sort_unstable();
    let list = p.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(",");
    w(|x| {
        let (f, tag) = x.fk(c);
        let (slot, k, h) = f.acq[f.ia];
        f.ia += 1;
        f.held.push((key, slot));
        *vk = k;
        if k >= 0 {
            *vh = h;
        }
        let n = x.nm(slot);
        x.log.push_str(&format!("{}A{},{},[{}]>{},{};", tag, key, h0, list, n, k));
        slot as *mut u8
    })
}
unsafe extern "C" fn f_remove(c: *mut sc_cache, key: c_int) -> *mut u8 {
    w(|x| {
        let (f, tag) = x.fk(c);
        let r = match f.held.iter().position(|(k, _)| *k == key) {
            Some(i) => f.held.remove(i).1,
            None => 0,
        };
        let n = x.nm(r);
        x.log.push_str(&format!("{}R{}>{};", tag, key, n));
        r as *mut u8
    })
}
unsafe extern "C" fn f_release(_c: *mut sc_cache, s: *mut u8) {
    w(|x| {
        let n = x.nm(s as usize);
        x.log.push_str(&format!("HL({});", n));
    });
}
unsafe extern "C" fn f_pend_take(key: c_int) {
    w(|x| x.log.push_str(&format!("PN{};", key)));
}
unsafe extern "C" fn f_arr_wait(key: c_int) {
    w(|x| x.log.push_str(&format!("AW{};", key)));
}
unsafe extern "C" fn f_host_acquire(key: c_int, heat: f32, vk: *mut c_int, vh: *mut f32) -> *mut u8 {
    w(|x| {
        let f = &mut x.host;
        let (slot, k, h) = f.acq[f.ia];
        f.ia += 1;
        *vk = k;
        if k >= 0 {
            *vh = h;
        }
        let n = x.nm(slot);
        x.log.push_str(&format!("HA{},{}>{},{};", key, heat, n, k));
        slot as *mut u8
    })
}

fn ops() -> sy_ops {
    sy_ops {
        device_sync: f_device_sync,
        stream_sync: f_stream_sync,
        set_device: f_set_device,
        memcpy_d2h_async: f_d2h,
        memcpy_h2d_async: f_h2d,
        event_record: f_event_record,
        stream_wait_event: f_wait_event,
        check: f_check,
        sc_find: f_find,
        sc_touch: f_touch,
        sc_worth: f_worth,
        sc_acquire: f_acquire,
        sc_remove: f_remove,
        sc_release: f_release,
        pend_take: f_pend_take,
        arr_wait: f_arr_wait,
        host_acquire: f_host_acquire,
    }
}

fn pattern(i: usize) -> u8 {
    ((i.wrapping_mul(131)) + (i >> 7)) as u8
}
fn slab_pattern(i: usize) -> u8 {
    ((i.wrapping_mul(17)) + (i >> 5) + 3) as u8
}

/// A test world: the fake caches' memory, a supply, a pool, a host cache, a file of NEXP experts.
struct Env {
    sup: Supply,
    pool: sy_pool,
    host: sy_host,
    eptr: Vec<*const u8>,
    counts: Vec<c_int>,
    slab: Vec<*const u8>,
    file: Vec<u8>,
}

fn setup_full(experts: usize, vq: bool, prefill: bool, chunks: i32, cap: usize, host_null: bool, pool_readers: i32) -> Env {
    let file: Vec<u8> = (0..NEXP * RAW).map(pattern).collect();
    let mut pm: Vec<Vec<u8>> = (0..8).map(|i| vec![0xA0 + i as u8; RAW]).collect();
    let hm: Vec<Vec<u8>> = (0..8).map(|i| vec![0xB0 + i as u8; RAW]).collect();
    let bm: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; RAW]).collect();
    let sm: Vec<u8> = (0..4 * RAW).map(slab_pattern).collect();
    pm.shrink_to_fit();
    let wd = World {
        log: String::new(),
        pm,
        hm,
        bm,
        sm,
        pool: Fk::default(),
        host: Fk::default(),
        recs: vec![],
        inject: 0,
        check_returns: false,
        copy_s: Box::new(0),
        rs: (0..4).map(|_| Box::new(0u8)).collect(),
        ev: Box::new(0),
    };
    let mut pool = sy_pool {
        dev: 5,
        readers: pool_readers,
        cache: pool_cache(),
        copy: &*wd.copy_s as *const u8 as *mut c_void,
        rstream: [std::ptr::null_mut(); SY_MAX_READERS],
        bounce: [std::ptr::null_mut(); SY_MAX_READERS],
        written_back: &*wd.ev as *const u8 as *mut c_void,
        h_eptr: std::ptr::null_mut(),
        st: sy_stats::default(),
    };
    for i in 0..4 {
        pool.rstream[i] = &*wd.rs[i] as *const u8 as *mut c_void;
        pool.bounce[i] = wd.bm[i].as_ptr() as *mut u8;
    }
    let mut eptr: Vec<*const u8> = vec![std::ptr::null(); experts];
    pool.h_eptr = eptr.as_mut_ptr();
    let mut slab: Vec<*const u8> = vec![std::ptr::null(); 93];
    slab[LAYER as usize] = wd.sm.as_ptr();
    let cfg = sy_config { experts: experts as c_int, expert_raw: RAW, vq: vq as c_int, prefill: prefill as c_int, chunks, slab: slab.as_ptr() };
    *W.lock().unwrap_or_else(|e| e.into_inner()) = Some(wd);
    Env {
        sup: Supply::new(ops(), cfg),
        pool,
        host: sy_host { cache: if host_null { std::ptr::null_mut() } else { host_cache() }, cap_slots: cap },
        eptr,
        counts: vec![1; experts],
        slab,
        file,
    }
}

fn setup(vq: bool, prefill: bool, chunks: i32, cap: usize, host_null: bool) -> Env {
    setup_full(NEXP, vq, prefill, chunks, cap, host_null, 4)
}

impl Env {
    fn run(&mut self, fd: c_int, experts: &[i32], readers: c_int) {
        let _ = &self.slab;
        // SAFETY: the world's memory outlives the call; the fakes are thread-safe.
        unsafe { self.sup.ensure(&mut self.pool, &self.host, LAYER, fd, experts, self.counts.as_ptr(), readers) };
    }
    fn pm(&self, i: usize) -> usize {
        w(|x| x.pm[i].as_ptr() as usize)
    }
    fn pool_acq(&self, slot: usize, vk: i32, vh: f32) {
        let a = w(|x| x.pm[slot].as_ptr() as usize);
        w(|x| x.pool.acq.push((a, vk, vh)));
    }
    fn host_acq(&self, slot: usize, vk: i32, vh: f32) {
        let a = w(|x| x.hm[slot].as_ptr() as usize);
        w(|x| x.host.acq.push((a, vk, vh)));
    }
    fn pool_has(&self, key: i32, slot: usize) {
        let a = w(|x| x.pm[slot].as_ptr() as usize);
        w(|x| x.pool.held.push((key, a)));
    }
    fn host_has(&self, key: i32, slot: usize) {
        let a = w(|x| x.hm[slot].as_ptr() as usize);
        w(|x| x.host.held.push((key, a)));
    }
    fn notworth(&self, key: i32) {
        w(|x| x.host.notworth.push(key));
    }
    fn log(&self) -> String {
        w(|x| x.log.clone())
    }
    fn pm_eq(&self, slot: usize, e: usize) -> bool {
        w(|x| x.pm[slot][..] == self.file[e * RAW..(e + 1) * RAW])
    }
    fn stats(&self) -> (u64, u64, u64, u64, u64, u64) {
        let s = &self.pool.st;
        (s.needed, s.hit_card, s.hit_host, s.from_drive, s.written_back, WB_SKIPPED.load(Ordering::SeqCst))
    }
}

fn job(slot: &str, src: &str, n: usize, s: &str) -> String {
    format!("SD(5);kd;WE({s},ev);kw;H2D({slot},{src},{n},{s});kc;SS({s});ks;")
}

fn tmpfile(bytes: &[u8]) -> std::fs::File {
    use std::io::Write;
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let p = std::env::temp_dir().join(format!("supply-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    let mut f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&p).unwrap();
    f.write_all(bytes).unwrap();
    std::fs::remove_file(&p).unwrap();
    f
}

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------------------------- card hits

#[test]
fn all_hits_without_card_queue() {
    let _g = serial();
    let mut e = setup(false, false, 0, 4, false);
    e.pool_has(82, 0);
    e.pool_has(85, 1);
    e.counts[2] = 3;
    e.counts[5] = 1;
    let fd = tmpfile(&e.file.clone());
    e.run(fd.as_raw_fd(), &[2, 5], 4);
    assert_eq!(e.log(), "DS;kD;PF82>P0;PT82,3;PF85>P1;PT85,1;");
    assert_eq!(e.eptr[2] as usize, e.pm(0));
    assert_eq!(e.eptr[5] as usize, e.pm(1));
    assert_eq!(e.stats(), (2, 2, 0, 0, 0, 0));
    assert!(e.pool.st.t_ensure > 0.0 && e.pool.st.t_read == 0.0 && e.pool.st.t_copy == 0.0);
}

#[test]
fn all_hits_with_card_queue_wait_for_the_arrival_before_the_touch() {
    let _g = serial();
    let mut e = setup(true, false, 0, 4, false);
    e.pool_has(82, 0);
    e.pool_has(85, 1);
    e.counts[2] = 3;
    e.counts[5] = 1;
    e.run(-1, &[2, 5], 4);
    assert_eq!(e.log(), "SS(0);k0;PF82>P0;AW82;PT82,3;PF85>P1;AW85;PT85,1;");
    assert_eq!(e.stats(), (2, 2, 0, 0, 0, 0));
}

#[test]
fn empty_call_does_only_the_first_sync() {
    let _g = serial();
    let mut e = setup(false, false, 0, 4, false);
    e.run(-1, &[], 4);
    assert_eq!(e.log(), "DS;kD;");
    assert_eq!(e.stats(), (0, 0, 0, 0, 0, 0));
    assert!(e.pool.st.t_ensure > 0.0);
}

#[test]
fn hit_key_uses_the_real_constants_layer_92_expert_895() {
    let _g = serial();
    let mut e = setup_full(896, false, false, 0, 4, false, 4);
    // ekey(92, 895) = 92 * 896 + 895 = 83327
    e.pool_has(83327, 0);
    e.counts[895] = 2;
    let (l, ex) = (92, [895]);
    unsafe { e.sup.ensure(&mut e.pool, &e.host, l, -1, &ex, e.counts.as_ptr(), 1) };
    assert_eq!(e.log(), "DS;kD;PF83327>P0;PT83327,2;");
}

// ---------------------------------------------------------------------------------------------- RAM, drive, mix

#[test]
fn all_from_ram_in_need_order_and_released_in_held_order() {
    let _g = serial();
    let mut e = setup(false, false, 0, 4, false);
    e.pool_acq(0, -1, 0.0);
    e.pool_acq(1, -1, 0.0);
    e.host_has(83, 0);
    e.host_has(81, 3);
    e.counts[3] = 2;
    e.counts[1] = 7;
    e.run(-1, &[3, 1], 1);
    assert_eq!(
        e.log(),
        "DS;kD;PF83>-;PA83,2,[81,83]>P0,-1;PF81>-;PA81,7,[81,83]>P1,-1;HF83>H0;PN83;HR83>H0;HF81>H3;PN81;HR81>H3;SD(5);kS;ER(ev,c);kE;\
         H2D(P0,H0,10000,c);kH;H2D(P1,H3,10000,c);kH;SS(c);kC;HL(H0);HL(H3);"
    );
    assert_eq!(e.stats(), (2, 0, 2, 0, 0, 0));
}

#[test]
fn all_from_drive_one_reader_runs_in_expert_order_on_the_calling_thread() {
    let _g = serial();
    let mut e = setup(false, false, 0, 4, false);
    for s in 0..3 {
        e.pool_acq(s, -1, 0.0);
    }
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[0, 7, 9], 1);
    let want = format!(
        "DS;kD;PF80>-;PA80,1,[80,87,89]>P0,-1;PF87>-;PA87,1,[80,87,89]>P1,-1;PF89>-;PA89,1,[80,87,89]>P2,-1;HF80>-;HF87>-;HF89>-;SD(5);kS;ER(ev,c);kE;{}{}{}SS(c);kC;",
        job("P0", "B0", 10000, "r0"),
        job("P1", "B0", 10000, "r0"),
        job("P2", "B0", 10000, "r0")
    );
    assert_eq!(e.log(), want);
    assert!(e.pm_eq(0, 0) && e.pm_eq(1, 7) && e.pm_eq(2, 9));
    assert_eq!(e.stats(), (3, 0, 0, 3, 0, 0));
    assert!(e.pool.st.t_read > 0.0 && e.pool.st.t_copy > 0.0 && e.pool.st.t_ensure >= e.pool.st.t_read + e.pool.st.t_copy);
}

#[test]
fn mix_with_two_victims_one_written_back_one_not_worth() {
    let _g = serial();
    let mut e = setup(false, false, 0, 2, false);
    e.pool_has(81, 0);
    e.pool_acq(1, 200, 1.5);
    e.pool_acq(2, 201, 0.5);
    e.pool_acq(3, -1, 0.0);
    e.host_has(82, 0);
    w(|x| x.hm[0].copy_from_slice(&e.file[2 * RAW..3 * RAW]));
    e.notworth(201);
    e.host_acq(1, 300, 2.0);
    e.counts[1] = 1;
    e.counts[2] = 2;
    e.counts[3] = 3;
    e.counts[4] = 4;
    w(|x| x.pm[1].fill(0x5A));
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[1, 2, 3, 4], 1);
    let want = format!(
        "DS;kD;PF81>P0;PT81,1;PF82>-;PA82,2,[81,82,83,84]>P1,200;PF83>-;PA83,3,[81,82,83,84]>P2,201;PF84>-;PA84,4,[81,82,83,84]>P3,-1;\
         HF82>H0;PN82;HR82>H0;HF83>-;HF84>-;SD(5);kS;HW200=1;HA200,1.5>H1,300;D2H(H1,P1,10000,c);kW;HW201=0;ER(ev,c);kE;H2D(P1,H0,10000,c);kH;{}{}SS(c);kC;HL(H0);",
        job("P2", "B0", 10000, "r0"),
        job("P3", "B0", 10000, "r0")
    );
    assert_eq!(e.log(), want);
    assert!(w(|x| x.hm[1].iter().all(|b| *b == 0x5A)), "the victim's bytes were copied down into the RAM slot");
    assert!(e.pm_eq(1, 2) && e.pm_eq(2, 3) && e.pm_eq(3, 4));
    assert_eq!(e.stats(), (4, 1, 1, 2, 1, 1));
}

#[test]
fn every_victim_not_worth_still_records_the_event() {
    let _g = serial();
    let mut e = setup(false, false, 0, 2, false);
    e.pool_acq(0, 200, 1.0);
    e.pool_acq(1, 201, 1.0);
    e.notworth(200);
    e.notworth(201);
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[1, 2], 1);
    let want = format!(
        "DS;kD;PF81>-;PA81,1,[81,82]>P0,200;PF82>-;PA82,1,[81,82]>P1,201;HF81>-;HF82>-;SD(5);kS;HW200=0;HW201=0;ER(ev,c);kE;{}{}SS(c);kC;",
        job("P0", "B0", 10000, "r0"),
        job("P1", "B0", 10000, "r0")
    );
    assert_eq!(e.log(), want);
    assert_eq!(e.stats(), (2, 0, 0, 2, 0, 2));
}

#[test]
fn host_capacity_zero_asks_nothing_of_the_ram_cache_but_find() {
    let _g = serial();
    let mut e = setup(false, false, 0, 0, false);
    e.pool_acq(0, 200, 1.0);
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[1], 1);
    assert_eq!(e.log(), format!("DS;kD;PF81>-;PA81,1,[81]>P0,200;HF81>-;SD(5);kS;ER(ev,c);kE;{}SS(c);kC;", job("P0", "B0", 10000, "r0")));
    assert_eq!(e.stats(), (1, 0, 0, 1, 0, 0));
}

#[test]
fn null_host_cache_is_never_called() {
    let _g = serial();
    let mut e = setup(false, false, 0, 0, true);
    e.pool_acq(0, 200, 1.0);
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[1], 1);
    assert_eq!(e.log(), format!("DS;kD;PF81>-;PA81,1,[81]>P0,200;SD(5);kS;ER(ev,c);kE;{}SS(c);kC;", job("P0", "B0", 10000, "r0")));
}

#[test]
fn duplicate_expert_is_a_card_hit_the_second_time() {
    let _g = serial();
    let mut e = setup(true, false, 0, 4, false);
    e.pool_acq(0, -1, 0.0);
    e.counts[4] = 4;
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[4, 4], 1);
    assert_eq!(e.log(), format!("SS(0);k0;PF84>-;PA84,4,[84]>P0,-1;PF84>P0;AW84;PT84,4;HF84>-;SD(5);kS;ER(ev,c);kE;{}SS(c);kC;", job("P0", "B0", 10000, "r0")));
    assert_eq!(e.stats(), (2, 1, 0, 1, 0, 0));
}

#[test]
fn card_queue_waits_for_the_victim_before_it_is_copied_down() {
    let _g = serial();
    let mut e = setup(true, false, 0, 2, false);
    e.pool_has(81, 0);
    e.pool_acq(1, 77, 2.5);
    e.host_acq(2, -1, 0.0);
    w(|x| x.pm[1].fill(0x77));
    e.counts[1] = 6;
    e.counts[2] = 9;
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[1, 2], 1);
    let want = format!(
        "SS(0);k0;PF81>P0;AW81;PT81,6;PF82>-;PA82,9,[81,82]>P1,77;AW77;HF82>-;SD(5);kS;HW77=1;HA77,2.5>H2,-1;D2H(H2,P1,10000,c);kW;ER(ev,c);kE;{}SS(c);kC;",
        job("P1", "B0", 10000, "r0")
    );
    assert_eq!(e.log(), want);
    assert!(w(|x| x.hm[2].iter().all(|b| *b == 0x77)));
    assert_eq!(e.stats(), (2, 1, 0, 1, 1, 0));
}

// ---------------------------------------------------------------------------------------------- prefill

#[test]
fn prefill_reads_nothing_from_the_drive_writes_nothing_back_and_sources_from_the_slab() {
    let _g = serial();
    let mut e = setup(false, true, 0, 2, false);
    e.pool_acq(0, 300, 1.0);
    e.pool_acq(1, -1, 0.0);
    e.host_has(86, 1);
    e.run(-1, &[6, 13], 1);
    let want = format!(
        "DS;kD;PF86>-;PA86,1,[86,93]>P0,300;PF93>-;PA93,1,[86,93]>P1,-1;HF86>H1;PN86;HR86>H1;HF93>-;SD(5);kS;ER(ev,c);kE;H2D(P0,S+20000,10000,c);kH;{}SS(c);kC;HL(H1);",
        job("P1", "S+10000", 10000, "r0")
    );
    assert_eq!(e.log(), want);
    let sm: Vec<u8> = w(|x| x.sm.clone());
    assert!(w(|x| x.pm[0][..] == sm[2 * RAW..3 * RAW]) && w(|x| x.pm[1][..] == sm[RAW..2 * RAW]));
    assert_eq!(e.stats(), (2, 0, 1, 1, 0, 0));
}

#[test]
fn prefill_chunk_sources_continue_inside_the_slab_record() {
    let _g = serial();
    let mut e = setup(false, true, 3, 2, false);
    e.pool_acq(0, -1, 0.0);
    e.run(-1, &[7], 1); // e & 3 = 3: record 3 at slab offset 30000; 3 chunks of 4096, 4096, 1808
    let want = format!(
        "DS;kD;PF87>-;PA87,1,[87]>P0,-1;HF87>-;SD(5);kS;ER(ev,c);kE;{}{}{}SS(c);kC;",
        job("P0", "S+30000", 4096, "r0"),
        job("P0+4096", "S+34096", 4096, "r0"),
        job("P0+8192", "S+38192", 1808, "r0")
    );
    assert_eq!(e.log(), want);
}

// ---------------------------------------------------------------------------------------------- chunking

#[test]
fn chunking_table_derived_by_hand() {
    // (raw, cfg.chunks, n_drive, pool_readers) -> (nchunk, chunk)
    // raw 17547264 = 4284 * 4096 (the real record): 3 experts on 16 readers -> min(8, (16+3-1)/3 = 6) = 6 chunks of 714 blocks = 2924544 bytes.
    assert_eq!(chunking(17547264, 0, 3, 16), (6, 2924544));
    // 1 expert on 16 readers -> min(8, 16) = 8 chunks; ceil(raw/8) = 2193408 -> 536 blocks (2193408 + 4095 = 2197503; / 4096 = 536) = 2195456 bytes
    assert_eq!(chunking(17547264, 0, 1, 16), (8, 2195456));
    // as many experts as readers, or more: one chunk, raw is already a multiple of 4096
    assert_eq!(chunking(17547264, 0, 16, 16), (1, 17547264));
    assert_eq!(chunking(17547264, 0, 40, 16), (1, 17547264));
    // nothing to read
    assert_eq!(chunking(17547264, 0, 0, 16), (1, 17547264));
    // fewer readers than experts
    assert_eq!(chunking(17547264, 0, 5, 2), (1, 17547264));
    // 3 experts, 4 readers -> 2; 5 experts, 7 readers -> (7+5-1)/5 = 2; 1 expert, 3 readers -> 3
    assert_eq!(chunking(17547264, 0, 3, 4).0, 2);
    assert_eq!(chunking(17547264, 0, 5, 7).0, 2);
    assert_eq!(chunking(17547264, 0, 1, 3).0, 3);
    // no readers at all (0 or negative): n_drive >= pool_readers, one chunk
    assert_eq!(chunking(17547264, 0, 2, 0).0, 1);
    assert_eq!(chunking(17547264, 0, 2, -3).0, 1);
    // forced: any value > 0 wins over the computed one, 0 and negative do not
    assert_eq!(chunking(17547264, 5, 16, 16).0, 5);
    assert_eq!(chunking(17547264, 0, 1, 16).0, 8);
    // raw 10000: 2 chunks -> ceil(10000/2) = 5000 -> (5000 + 4095) / 4096 = 2 blocks = 8192; 3 chunks -> 3334 -> 1 block = 4096; 4 chunks -> 2500 -> 4096; 1 chunk -> 10000 -> 3 blocks = 12288
    assert_eq!(chunking(10000, 2, 1, 1), (2, 8192));
    assert_eq!(chunking(10000, 3, 1, 1), (3, 4096));
    assert_eq!(chunking(10000, 4, 1, 1), (4, 4096));
    assert_eq!(chunking(10000, 1, 1, 1), (1, 12288));
    // raw 4096 in 8 chunks: ceil = 512 -> 1 block = 4096 (the first chunk covers the record)
    assert_eq!(chunking(4096, 8, 1, 1), (8, 4096));
}

#[test]
fn forced_chunks_stop_at_the_end_of_the_record() {
    let _g = serial();
    for (chunks, n) in [(3, 3usize), (4, 3), (1000, 3)] {
        let mut e = setup(false, false, chunks, 4, false);
        e.pool_acq(0, -1, 0.0);
        let f = tmpfile(&e.file.clone());
        e.run(f.as_raw_fd(), &[2], 1);
        let want = format!(
            "DS;kD;PF82>-;PA82,1,[82]>P0,-1;HF82>-;SD(5);kS;ER(ev,c);kE;{}{}{}SS(c);kC;",
            job("P0", "B0", 4096, "r0"),
            job("P0+4096", "B0", 4096, "r0"),
            job("P0+8192", "B0", 1808, "r0")
        );
        assert_eq!(e.log(), want, "chunks={} gives {} jobs", chunks, n);
        assert!(e.pm_eq(0, 2));
    }
}

#[test]
fn one_chunk_when_the_record_fits_the_first() {
    let _g = serial();
    // raw 10000, chunks forced to 1 gives chunk 12288: a single job of 10000 bytes
    let mut e = setup(false, false, 1, 4, false);
    e.pool_acq(0, -1, 0.0);
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[2], 1);
    assert_eq!(e.log(), format!("DS;kD;PF82>-;PA82,1,[82]>P0,-1;HF82>-;SD(5);kS;ER(ev,c);kE;{}SS(c);kC;", job("P0", "B0", 10000, "r0")));
}

// ---------------------------------------------------------------------------------------------- readers and threads

#[test]
fn zero_or_negative_readers_still_run_every_job_on_the_calling_thread() {
    let _g = serial();
    for readers in [0, -3] {
        let mut e = setup(false, false, 0, 4, false);
        e.pool_acq(0, -1, 0.0);
        e.pool_acq(1, -1, 0.0);
        let f = tmpfile(&e.file.clone());
        e.run(f.as_raw_fd(), &[0, 1], readers);
        let want = format!(
            "DS;kD;PF80>-;PA80,1,[80,81]>P0,-1;PF81>-;PA81,1,[80,81]>P1,-1;HF80>-;HF81>-;SD(5);kS;ER(ev,c);kE;{}{}SS(c);kC;",
            job("P0", "B0", 10000, "r0"),
            job("P1", "B0", 10000, "r0")
        );
        assert_eq!(e.log(), want, "readers={}", readers);
    }
}

#[test]
fn the_pool_reader_count_caps_the_argument() {
    let _g = serial();
    // readers 16 but pool.readers 1: one thread, 3 experts >= 1 reader -> 1 chunk each, run in list order
    let mut e = setup_full(NEXP, false, false, 0, 4, false, 1);
    for s in 0..3 {
        e.pool_acq(s, -1, 0.0);
    }
    let f = tmpfile(&e.file.clone());
    e.run(f.as_raw_fd(), &[0, 1, 2], 16);
    let want = format!(
        "DS;kD;PF80>-;PA80,1,[80,81,82]>P0,-1;PF81>-;PA81,1,[80,81,82]>P1,-1;PF82>-;PA82,1,[80,81,82]>P2,-1;HF80>-;HF81>-;HF82>-;SD(5);kS;ER(ev,c);kE;{}{}{}SS(c);kC;",
        job("P0", "B0", 10000, "r0"),
        job("P1", "B0", 10000, "r0"),
        job("P2", "B0", 10000, "r0")
    );
    assert_eq!(e.log(), want);
}

#[test]
fn real_threads_each_chunk_once_streams_match_bounce_slots() {
    let _g = serial();
    for _round in 0..20 {
        // readers 16, pool.readers 4 -> 4 reader threads; 2 experts -> nchunk = min(8, (4 + 2 - 1) / 2) = 2, chunk 8192 -> 4 jobs
        let mut e = setup_full(NEXP, false, false, 0, 4, false, 4);
        e.pool_acq(0, -1, 0.0);
        e.pool_acq(1, -1, 0.0);
        let f = tmpfile(&e.file.clone());
        e.run(f.as_raw_fd(), &[0, 1], 16);
        let (p0, p1) = (e.pm(0), e.pm(1));
        let mut got: Vec<(usize, usize)> = w(|x| x.recs.iter().map(|r| (r.0, r.1)).collect());
        got.sort();
        let mut want = vec![(p0, 8192), (p0 + 8192, 1808), (p1, 8192), (p1 + 8192, 1808)];
        want.sort();
        assert_eq!(got, want);
        let l = e.log();
        assert!(!l.contains("BAD-BOUNCE"), "{}", l);
        assert!(l.starts_with("DS;kD;PF80>-;PA80,1,[80,81]>P0,-1;PF81>-;PA81,1,[80,81]>P1,-1;HF80>-;HF81>-;SD(5);kS;ER(ev,c);kE;"));
        assert!(l.ends_with("SS(c);kC;"));
        assert_eq!(l.matches("SD(5);").count(), 5, "the main thread's set_device and one per reader job");
        assert!(e.pm_eq(0, 0) && e.pm_eq(1, 1));
        let used: Vec<usize> = w(|x| x.recs.iter().map(|r| r.2).collect());
        assert!(used.iter().all(|&s| s < 4));
    }
}

#[test]
fn eight_experts_on_four_threads_all_arrive() {
    let _g = serial();
    for _round in 0..20 {
        let mut e = setup_full(NEXP, false, false, 0, 4, false, 4);
        for s in 0..8 {
            e.pool_acq(s, -1, 0.0);
        }
        let f = tmpfile(&e.file.clone());
        e.run(f.as_raw_fd(), &[0, 1, 2, 3, 4, 5, 6, 7], 4);
        assert!((0..8).all(|i| e.pm_eq(i, i)));
        assert_eq!(w(|x| x.recs.len()), 8);
        assert!(!e.log().contains("BAD-BOUNCE"));
        assert_eq!(e.stats(), (8, 0, 0, 8, 0, 0));
    }
}

// ---------------------------------------------------------------------------------------------- fatal paths (a child process: they exit(1))

extern "C" {
    fn atexit(cb: extern "C" fn()) -> c_int;
}
extern "C" fn dump_log() {
    let l = W.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|x| x.log.clone()).unwrap_or_default();
    write_stderr(format!("LOG:{}", l).as_bytes());
}

/// Run by `run_child` in a child process with SUPPLY_CHILD=<scenario>; does nothing in a normal test run.
#[test]
fn child_scenario() {
    let name = match std::env::var("SUPPLY_CHILD") {
        Ok(n) => n,
        Err(_) => return,
    };
    let mut e = setup(false, false, 0, 4, false);
    if name == "before_init" {
        // the exported function with no sy_init in this process
        let mut pool = e.pool;
        let mut host = e.host;
        unsafe { crate::sy_ensure(&mut pool, &mut host, LAYER, -1, [1].as_ptr(), 1, e.counts.as_ptr(), 1) };
        return;
    }
    e.pool_acq(0, -1, 0.0);
    e.pool_acq(1, -1, 0.0);
    unsafe { atexit(dump_log) };
    match name.as_str() {
        "eof" => {
            let f = tmpfile(&e.file.clone()[..45000]);
            e.run(f.as_raw_fd(), &[1, 4], 1);
        }
        "badfd" => e.run(-1, &[1], 1),
        "cuda_error" => {
            w(|x| x.inject = 700);
            let f = tmpfile(&e.file.clone());
            e.run(f.as_raw_fd(), &[1], 1)
        }
        "check_returns" => {
            w(|x| {
                x.inject = 700;
                x.check_returns = true;
            });
            let f = tmpfile(&e.file.clone());
            e.run(f.as_raw_fd(), &[1], 1)
        }
        _ => panic!("unknown scenario"),
    }
}

fn run_child(name: &str) -> (Option<i32>, String) {
    let exe = std::env::current_exe().unwrap();
    let o = std::process::Command::new(exe).args(["--exact", "tests::child_scenario", "--nocapture", "--test-threads=1"]).env("SUPPLY_CHILD", name).output().unwrap();
    (o.status.code(), String::from_utf8_lossy(&o.stderr).to_string())
}

#[test]
fn fatal_paths_message_status_and_the_calls_before_them() {
    let _g = serial();
    let head = "DS;kD;PF81>-;PA81,1,[81]>P0,-1;HF81>-;SD(5);kS;ER(ev,c);kE;";
    // EOF in the second job (expert 4: 5000 of its 10000 bytes exist); the first job (expert 1) completed
    let (st, err) = run_child("eof");
    assert_eq!(st, Some(1));
    let want = format!(
        "reading expert 4 of layer 5 failed: end of file\nLOG:DS;kD;PF81>-;PA81,1,[81,84]>P0,-1;PF84>-;PA84,1,[81,84]>P1,-1;HF81>-;HF84>-;SD(5);kS;ER(ev,c);kE;{}",
        job("P0", "B0", 10000, "r0")
    );
    assert_eq!(err, want);
    let (st, err) = run_child("badfd");
    assert_eq!(st, Some(1));
    assert_eq!(err, format!("reading expert 1 of layer 5 failed: Bad file descriptor\nLOG:{}", head));
    let (st, err) = run_child("cuda_error");
    assert_eq!(st, Some(1));
    assert_eq!(err, "cuda error 700 at cudaSetDevice(pool.dev)\nLOG:DS;kD;PF81>-;PA81,1,[81]>P0,-1;HF81>-;SD(5);kS!700;");
    let (st, err) = run_child("check_returns");
    assert_eq!(st, Some(1));
    assert_eq!(err, "LOG:DS;kD;PF81>-;PA81,1,[81]>P0,-1;HF81>-;SD(5);kS!700;");
    let (st, err) = run_child("before_init");
    assert_eq!(st, Some(1));
    assert_eq!(err, "supply: sy_ensure before sy_init\n");
}

// ---------------------------------------------------------------------------------------------- the exported C functions

#[test]
fn exported_functions_end_to_end() {
    let _g = serial();
    let mut e = setup(false, false, 0, 2, false);
    let ops = ops();
    let cfg = sy_config { experts: NEXP as c_int, expert_raw: RAW, vq: 0, prefill: 0, chunks: 0, slab: e.slab.as_ptr() };
    // a victim that is not worth a slot counts in the library-wide counter; sy_init resets it
    e.pool_acq(0, 201, 1.0);
    e.notworth(201);
    let f = tmpfile(&e.file.clone());
    unsafe {
        crate::sy_init(&ops, &cfg);
        crate::sy_ensure(&mut e.pool, &mut e.host, LAYER, f.as_raw_fd(), [1].as_ptr(), 1, e.counts.as_ptr(), 1);
    }
    assert_eq!(crate::sy_wb_skipped(), 1);
    assert_eq!(e.log(), format!("DS;kD;PF81>-;PA81,1,[81]>P0,201;HF81>-;SD(5);kS;HW201=0;ER(ev,c);kE;{}SS(c);kC;", job("P0", "B0", 10000, "r0")));
    assert!(e.pm_eq(0, 1));
    assert_eq!(e.pool.st.needed, 1);
    // n = 0 with a NULL list
    w(|x| x.log.clear());
    unsafe { crate::sy_ensure(&mut e.pool, &mut e.host, LAYER, f.as_raw_fd(), std::ptr::null(), 0, e.counts.as_ptr(), 1) };
    assert_eq!(e.log(), "DS;kD;");
    // a second sy_init resets the counter
    unsafe { crate::sy_init(&ops, &cfg) };
    assert_eq!(crate::sy_wb_skipped(), 0);
}

// ---------------------------------------------------------------------------------------------- layout against a real C compile of the header

#[test]
fn struct_layout_matches_a_c_compile_of_the_header() {
    use std::mem::{offset_of, size_of};
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let inc = dir.join("include");
    let sc = [dir.join("../slotcache/include"), dir.join("slotcache-include")].into_iter().find(|p| p.join("slotcache.h").exists()).expect("slotcache.h not found");
    let src = format!(
        "#include <stdio.h>\n#include <stddef.h>\n#include \"supply.h\"\nint main(void){{\nprintf(\"%zu %zu %zu %zu %zu\\n\", sizeof(sy_ops), sizeof(sy_config), sizeof(sy_stats), sizeof(sy_pool), sizeof(sy_host));\n\
         printf(\"%zu %zu %zu %zu %zu\\n\", offsetof(sy_ops, check), offsetof(sy_ops, host_acquire), offsetof(sy_config, expert_raw), offsetof(sy_config, chunks), offsetof(sy_config, slab));\n\
         printf(\"%zu %zu %zu %zu %zu %zu %zu\\n\", offsetof(sy_pool, cache), offsetof(sy_pool, copy), offsetof(sy_pool, rstream), offsetof(sy_pool, bounce), offsetof(sy_pool, written_back), offsetof(sy_pool, h_eptr), offsetof(sy_pool, st));\n\
         printf(\"%zu %zu %zu %zu %zu\\n\", offsetof(sy_stats, hit_card), offsetof(sy_stats, written_back), offsetof(sy_stats, t_read), offsetof(sy_stats, t_ensure), offsetof(sy_host, cap_slots));\n\
         printf(\"%d\\n\", SY_MAX_READERS);\nreturn 0;}}\n"
    );
    let tmp = std::env::temp_dir().join(format!("supply-layout-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("l.c"), src).unwrap();
    let cc = std::process::Command::new("cc")
        .arg("-I").arg(&inc).arg("-I").arg(&sc).arg("-o").arg(tmp.join("l")).arg(tmp.join("l.c"))
        .output()
        .expect("cc");
    assert!(cc.status.success(), "{}", String::from_utf8_lossy(&cc.stderr));
    let out = std::process::Command::new(tmp.join("l")).output().unwrap();
    let got = String::from_utf8_lossy(&out.stdout).to_string();
    let _ = std::fs::remove_dir_all(&tmp);
    let want = format!(
        "{} {} {} {} {}\n{} {} {} {} {}\n{} {} {} {} {} {} {}\n{} {} {} {} {}\n{}\n",
        size_of::<sy_ops>(), size_of::<sy_config>(), size_of::<sy_stats>(), size_of::<sy_pool>(), size_of::<sy_host>(),
        offset_of!(sy_ops, check), offset_of!(sy_ops, host_acquire), offset_of!(sy_config, expert_raw), offset_of!(sy_config, chunks), offset_of!(sy_config, slab),
        offset_of!(sy_pool, cache), offset_of!(sy_pool, copy), offset_of!(sy_pool, rstream), offset_of!(sy_pool, bounce), offset_of!(sy_pool, written_back), offset_of!(sy_pool, h_eptr), offset_of!(sy_pool, st),
        offset_of!(sy_stats, hit_card), offset_of!(sy_stats, written_back), offset_of!(sy_stats, t_read), offset_of!(sy_stats, t_ensure), offset_of!(sy_host, cap_slots),
        SY_MAX_READERS
    );
    assert_eq!(got, want);
}
