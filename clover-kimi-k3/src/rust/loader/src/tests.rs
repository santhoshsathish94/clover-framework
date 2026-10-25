//! Unit tests. Every expectation is derived by hand from the original text (backend-gpu.cu 538-572, 574-610 at 4d8d891) and from loader.h, not from running this crate.
use crate::abi::{ld_callbacks, ld_config, ld_layer};
use crate::job::{expert_path, run, Cb, LayerJob};
use crate::queue::Loader;
use crate::sys::{Sys, O_DIRECT, O_RDONLY};
use crate::{check_card, check_slab, validate_config};
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------- recording callbacks
struct Rec {
    v: Mutex<Vec<String>>,
    slab_base: usize,
}
fn rec<'a>(ctx: *mut c_void) -> &'a Rec {
    // SAFETY: tests pass the address of a `Rec` that outlives every call.
    unsafe { &*(ctx as *const Rec) }
}
fn text(p: *const c_char) -> String {
    // SAFETY: the job passes a NUL-terminated string.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}
unsafe extern "C" fn t_set_device(ctx: *mut c_void, card: c_int) {
    rec(ctx).v.lock().unwrap().push(format!("set_device {}", card));
}
unsafe extern "C" fn t_ensure_work(ctx: *mut c_void, card: c_int) {
    rec(ctx).v.lock().unwrap().push(format!("ensure_work {}", card));
}
unsafe extern "C" fn t_trunk(ctx: *mut c_void, h: *mut c_void, ds: *const c_char, l: c_int) {
    // SAFETY: tests pass the address of a live u32 as the handle.
    let hv = unsafe { *(h as *const u32) };
    rec(ctx).v.lock().unwrap().push(format!("trunk L{} h={} ds={}", l, hv, text(ds)));
}
unsafe extern "C" fn t_tables(ctx: *mut c_void, h: *mut c_void, ds: *const c_char, l: c_int) {
    // SAFETY: as above.
    let hv = unsafe { *(h as *const u32) };
    rec(ctx).v.lock().unwrap().push(format!("tables L{} h={} ds={}", l, hv, text(ds)));
}
unsafe extern "C" fn t_state(ctx: *mut c_void, h: *mut c_void, positions: c_uint) {
    // SAFETY: as above.
    let hv = unsafe { *(h as *const u32) };
    rec(ctx).v.lock().unwrap().push(format!("state h={} positions={}", hv, positions));
}
unsafe extern "C" fn t_host_alloc(ctx: *mut c_void, dst: *mut *mut u8, bytes: usize) {
    let r = rec(ctx);
    let slot = (dst as usize - r.slab_base) / std::mem::size_of::<*mut u8>();
    let mut b = vec![0xEEu8; bytes.max(1)];
    // SAFETY: dst is the address of a slab table entry the test owns.
    unsafe { *dst = b.as_mut_ptr() };
    std::mem::forget(b);
    r.v.lock().unwrap().push(format!("host_alloc slot={} bytes={}", slot, bytes));
}

fn cb_for(r: &Rec) -> Cb {
    Cb {
        ctx: r as *const Rec as *mut c_void,
        set_device: t_set_device,
        ensure_work: t_ensure_work,
        load_trunk: t_trunk,
        load_expert_tables: t_tables,
        alloc_state: t_state,
        host_alloc: Some(t_host_alloc),
    }
}

// ---------------------------------------------------------------- fake libc
struct FakeSys {
    opens: Mutex<Vec<(Vec<u8>, c_int)>>,
    open_results: Mutex<VecDeque<c_int>>,
    data: Vec<u8>,
    virtual_len: Option<usize>, // a file of this many bytes whose content is not materialised: reads return lengths but copy nothing
    chunk: usize,
    fail_call: Option<usize>,
    preads: Mutex<Vec<(c_int, usize, i64)>>,
}
impl FakeSys {
    fn new(open_results: &[c_int], data: Vec<u8>, chunk: usize) -> FakeSys {
        FakeSys { opens: Mutex::new(vec![]), open_results: Mutex::new(open_results.iter().copied().collect()), data, virtual_len: None, chunk, fail_call: None, preads: Mutex::new(vec![]) }
    }
}
impl Sys for FakeSys {
    fn open(&self, path: &CStr, flags: c_int) -> c_int {
        self.opens.lock().unwrap().push((path.to_bytes().to_vec(), flags));
        self.open_results.lock().unwrap().pop_front().expect("unscripted open")
    }
    fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> isize {
        let mut p = self.preads.lock().unwrap();
        let idx = p.len();
        p.push((fd, len, off));
        if self.fail_call == Some(idx) {
            return -1;
        }
        if let Some(total) = self.virtual_len {
            return if off < 0 || off as usize >= total { 0 } else { len.min(self.chunk).min(total - off as usize) as isize };
        }
        if off < 0 || off as usize >= self.data.len() {
            return 0;
        }
        let n = len.min(self.chunk).min(self.data.len() - off as usize);
        // SAFETY: the job passes a buffer writable for `len` >= n bytes.
        unsafe { std::ptr::copy_nonoverlapping(self.data.as_ptr().add(off as usize), buf, n) };
        n as isize
    }
}

struct Fix {
    rec: Box<Rec>,
    slab: Box<[*mut u8; 93]>,
    fd_cell: Box<c_int>,
    handle: Box<u32>,
}
impl Fix {
    fn new() -> Fix {
        let slab = Box::new([std::ptr::null_mut::<u8>(); 93]);
        let base = slab.as_ptr() as usize;
        Fix { rec: Box::new(Rec { v: Mutex::new(vec![]), slab_base: base }), slab, fd_cell: Box::new(-77), handle: Box::new(4242) }
    }
    fn job(&mut self, ds: &str, layer: u32, card: c_int, positions: u32, prefill: bool, raw: usize) -> LayerJob {
        LayerJob {
            cb: cb_for(&self.rec),
            handle: &*self.handle as *const u32 as *mut c_void,
            ds: CString::new(ds).unwrap(),
            layer,
            positions,
            card,
            h_fd: &mut *self.fd_cell as *mut c_int,
            expert_raw: raw,
            prefill,
            slab: self.slab.as_mut_ptr(),
        }
    }
    fn log(&self) -> Vec<String> {
        self.rec.v.lock().unwrap().clone()
    }
}

// ---------------------------------------------------------------- job: callback order and files
#[test]
fn layer0_calls_in_order_and_opens_nothing() {
    // original: CHECK(cudaSetDevice) ; ensure_work ; layer_load_trunk ; (L >= 1 only: tables) ; state group ; (L >= 1 only: file) -> layer 0 has 4 events, no open, fd untouched
    let mut f = Fix::new();
    let sys = FakeSys::new(&[], vec![], 1);
    let j = f.job("/d", 0, 2, 9, true, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    assert_eq!(f.log(), vec!["set_device 2", "ensure_work 2", "trunk L0 h=4242 ds=/d", "state h=4242 positions=9"]);
    assert!(sys.opens.lock().unwrap().is_empty());
    assert_eq!(*f.fd_cell, -77);
    assert!(f.slab[0].is_null());
}

#[test]
fn layer3_without_prefill_opens_direct_first_and_reads_nothing() {
    // L >= 1: tables after the trunk, state after the tables, then open(path, O_RDONLY | O_DIRECT); no prefill -> no host_alloc, no pread
    let mut f = Fix::new();
    let sys = FakeSys::new(&[11], vec![], 1);
    let j = f.job("/data/set", 3, 1, 5, false, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    assert_eq!(
        f.log(),
        vec!["set_device 1", "ensure_work 1", "trunk L3 h=4242 ds=/data/set", "tables L3 h=4242 ds=/data/set", "state h=4242 positions=5"]
    );
    assert_eq!(*sys.opens.lock().unwrap(), vec![(b"/data/set/expert-3/experts.direct".to_vec(), O_RDONLY | O_DIRECT)]);
    assert_eq!(*f.fd_cell, 11);
    assert!(sys.preads.lock().unwrap().is_empty());
    assert!(f.slab.iter().all(|p| p.is_null()));
}

#[test]
fn direct_open_refused_falls_back_to_plain_open() {
    // `if (h->fd < 0) h->fd = open(path, O_RDONLY)`: any failure of the first open triggers the second, which has no O_DIRECT
    let mut f = Fix::new();
    let sys = FakeSys::new(&[-1, 12], vec![], 1);
    let j = f.job("/d", 3, 0, 5, false, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    let o = sys.opens.lock().unwrap().clone();
    assert_eq!(o.len(), 2);
    assert_eq!(o[0].1, O_RDONLY | O_DIRECT);
    assert_eq!(o[1].1, O_RDONLY);
    assert_eq!(o[0].0, o[1].0);
    assert_eq!(*f.fd_cell, 12);
}

#[test]
fn both_opens_failing_is_fatal_with_the_exact_text_and_fd_minus_one() {
    // `fprintf(stderr, "cannot open %s\n", path); exit(1);`; h->fd holds the last (negative) result
    let mut f = Fix::new();
    let sys = FakeSys::new(&[-1, -1], vec![], 1);
    let j = f.job("/nowhere", 3, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Err(b"cannot open /nowhere/expert-3/experts.direct\n".to_vec()));
    assert_eq!(*f.fd_cell, -1);
    // the failing open happens before host_alloc: no slab event
    assert!(!f.log().iter().any(|l| l.starts_with("host_alloc")));
}

#[test]
fn open_succeeding_with_fd_zero_counts_as_success() {
    // `h->fd < 0` is the test: fd 0 is a valid result
    let mut f = Fix::new();
    let sys = FakeSys::new(&[0], vec![], 1);
    let j = f.job("/d", 1, 0, 5, false, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    assert_eq!(*f.fd_cell, 0);
    assert_eq!(sys.opens.lock().unwrap().len(), 1);
}

#[test]
fn path_text_and_cut_at_4095_bytes_like_snprintf() {
    assert_eq!(expert_path(b"/d", 7), b"/d/expert-7/experts.direct".to_vec());
    assert_eq!(expert_path(b"/d", 4294967295), b"/d/expert-4294967295/experts.direct".to_vec());
    // "/expert-12/experts.direct" is 8 + 2 + 15 = 25 bytes. ds of 4070 -> 4095 bytes exactly: unchanged
    let ds = vec![b'x'; 4070];
    let p = expert_path(&ds, 12);
    assert_eq!(p.len(), 4095);
    assert!(p.ends_with(b"/expert-12/experts.direct"));
    // ds of 4071 -> 4096 bytes: the last byte is cut
    let p = expert_path(&vec![b'x'; 4071], 12);
    assert_eq!(p.len(), 4095);
    assert!(p.ends_with(b"/expert-12/experts.direc"));
    // ds of 4080 -> cut inside the suffix after 15 of its 25 bytes
    let p = expert_path(&vec![b'x'; 4080], 12);
    assert_eq!(p.len(), 4095);
    assert!(p.ends_with(b"xxx/expert-12/expe") && !p.ends_with(b"/expert-12/exper"));
    // ds longer than 4095: only x remain
    assert_eq!(expert_path(&vec![b'x'; 5000], 12), vec![b'x'; 4095]);
}

#[test]
fn cut_path_is_what_is_opened_and_what_the_message_prints() {
    let mut f = Fix::new();
    let ds = "y".repeat(5000);
    let sys = FakeSys::new(&[-1, -1], vec![], 1);
    let j = f.job(&ds, 12, 0, 1, false, 10);
    let mut want = b"cannot open ".to_vec();
    want.extend(vec![b'y'; 4095]);
    want.push(b'\n');
    assert_eq!(run(&sys, &j), Err(want));
    assert!(sys.opens.lock().unwrap().iter().all(|(p, _)| p.len() == 4095));
}

#[test]
fn prefill_reads_four_records_into_the_slab_with_one_pread_per_record() {
    // layer 5, expert_raw 10, a file of 5 records (the fifth is not read), the OS gives whole reads:
    // host_alloc(&slab[5], 40); pread(fd, slab5 + r*10 + 0, 10, r*10) for r = 0..3
    let mut f = Fix::new();
    let data: Vec<u8> = (0..50u8).collect();
    let sys = FakeSys::new(&[11], data.clone(), 1000);
    let j = f.job("/d", 5, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    assert_eq!(f.log().last().unwrap(), "host_alloc slot=5 bytes=40");
    assert_eq!(*sys.preads.lock().unwrap(), vec![(11, 10, 0), (11, 10, 10), (11, 10, 20), (11, 10, 30)]);
    // SAFETY: slab[5] points at the 40-byte block host_alloc made.
    let got = unsafe { std::slice::from_raw_parts(f.slab[5], 40) };
    assert_eq!(got, &data[..40]);
}

#[test]
fn prefill_short_reads_continue_at_the_right_offsets() {
    // the OS returns 3 bytes per call: per record the remaining length and offset are (10-g, r*10+g) for g = 0, 3, 6, 9 (3+3+3+1 = 10)
    let mut f = Fix::new();
    let data: Vec<u8> = (0..50u8).map(|i| i.wrapping_mul(7)).collect();
    let sys = FakeSys::new(&[11], data.clone(), 3);
    let j = f.job("/d", 5, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Ok(()));
    let mut want = vec![];
    for r in 0..4i64 {
        for g in [0i64, 3, 6, 9] {
            want.push((11, (10 - g) as usize, r * 10 + g));
        }
    }
    assert_eq!(*sys.preads.lock().unwrap(), want);
    // SAFETY: as above.
    let got = unsafe { std::slice::from_raw_parts(f.slab[5], 40) };
    assert_eq!(got, &data[..40]);
}

#[test]
fn prefill_end_of_file_inside_a_record_is_fatal() {
    // file of 25 bytes: record 0 and 1 whole; record 2 gets 5 bytes (pread(10 @ 20) -> 5), then pread(5 @ 25) -> 0 = end of file -> message with the layer, no further read
    let mut f = Fix::new();
    let sys = FakeSys::new(&[11], (0..25u8).collect(), 1000);
    let j = f.job("/d", 5, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Err(b"prefill: cannot read a record of layer 5\n".to_vec()));
    assert_eq!(*sys.preads.lock().unwrap(), vec![(11, 10, 0), (11, 10, 10), (11, 10, 20), (11, 5, 25)]);
}

#[test]
fn prefill_empty_file_fails_on_the_first_read() {
    let mut f = Fix::new();
    let sys = FakeSys::new(&[11], vec![], 1000);
    let j = f.job("/d", 17, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Err(b"prefill: cannot read a record of layer 17\n".to_vec()));
    assert_eq!(*sys.preads.lock().unwrap(), vec![(11, 10, 0)]);
}

#[test]
fn prefill_read_error_is_fatal_without_retry() {
    // `rd <= 0`: -1 (any errno, EINTR included) is the same message and the loop does not try again
    let mut f = Fix::new();
    let mut sys = FakeSys::new(&[11], (0..50u8).collect(), 1000);
    sys.fail_call = Some(1);
    let j = f.job("/d", 5, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Err(b"prefill: cannot read a record of layer 5\n".to_vec()));
    assert_eq!(sys.preads.lock().unwrap().len(), 2);
}

#[test]
fn prefill_layer_text_is_unsigned_decimal() {
    let mut f = Fix::new();
    let sys = FakeSys::new(&[11], vec![], 1000);
    let j = f.job("/d", 92, 0, 5, true, 10);
    assert_eq!(run(&sys, &j), Err(b"prefill: cannot read a record of layer 92\n".to_vec()));
    assert!(f.log().iter().any(|l| l == "host_alloc slot=92 bytes=40"));
}

#[test]
fn prefill_with_expert_raw_zero_allocates_zero_and_reads_nothing() {
    // `while (got < 0)` never runs; the original still calls cudaHostAlloc(4 * 0)
    let mut f = Fix::new();
    let sys = FakeSys::new(&[11], vec![], 1000);
    let j = f.job("/d", 5, 0, 5, true, 0);
    assert_eq!(run(&sys, &j), Ok(()));
    assert!(f.log().iter().any(|l| l == "host_alloc slot=5 bytes=0"));
    assert!(sys.preads.lock().unwrap().is_empty());
}

#[test]
fn prefill_with_the_engines_record_size_uses_record_multiples_as_offsets() {
    // expert_raw 17547264: host_alloc(4 * 17547264 = 70189056); record r is read at r * 17547264 = 0, 17547264, 35094528, 52641792 (a virtual file: lengths only)
    let mut f = Fix::new();
    let mut sys = FakeSys::new(&[11], vec![], usize::MAX);
    sys.virtual_len = Some(5 * 17_547_264);
    let j = f.job("/d", 5, 0, 5, true, 17_547_264);
    assert_eq!(run(&sys, &j), Ok(()));
    assert_eq!(f.log().last().unwrap(), "host_alloc slot=5 bytes=70189056");
    assert_eq!(*sys.preads.lock().unwrap(), vec![(11, 17_547_264, 0), (11, 17_547_264, 17_547_264), (11, 17_547_264, 35_094_528), (11, 17_547_264, 52_641_792)]);
}

// ---------------------------------------------------------------- guards and configuration
fn cfg_with(prefill: c_int, slab: *mut *mut u8, handles: *mut *mut c_void, rec: &Rec) -> ld_config {
    ld_config {
        cb: ld_callbacks {
            ctx: rec as *const Rec as *mut c_void,
            set_device: Some(t_set_device),
            ensure_work: Some(t_ensure_work),
            load_trunk: Some(t_trunk),
            load_expert_tables: Some(t_tables),
            alloc_state: Some(t_state),
            host_alloc: Some(t_host_alloc),
        },
        expert_raw: 10,
        prefill,
        slab,
        handles,
    }
}

#[test]
fn guards_have_the_header_texts() {
    assert_eq!(check_card(0), Ok(()));
    assert_eq!(check_card(7), Ok(()));
    assert_eq!(check_card(8), Err("loader: card 8 out of range (0..7)\n".to_string()));
    assert_eq!(check_card(-1), Err("loader: card -1 out of range (0..7)\n".to_string()));
    assert_eq!(check_slab(true, 92), Ok(()));
    assert_eq!(check_slab(true, 93), Err("loader: layer 93 has no slab slot (0..92)\n".to_string()));
    assert_eq!(check_slab(false, 93), Ok(()));
    assert_eq!(check_slab(true, 4_000_000_000), Err("loader: layer 4000000000 has no slab slot (0..92)\n".to_string()));
}

#[test]
fn validate_config_rejects_each_missing_required_item_in_order() {
    let f = Fix::new();
    let mut slab = [std::ptr::null_mut::<u8>(); 93];
    let mut handles = [std::ptr::null_mut::<c_void>(); 93];
    let good = cfg_with(1, slab.as_mut_ptr(), handles.as_mut_ptr(), &f.rec);
    assert!(validate_config(&good).is_ok());
    let msg = |c: ld_config| validate_config(&c).err().unwrap();
    let mut c = good;
    c.cb.set_device = None;
    assert_eq!(msg(c), "loader: set_device is NULL\n");
    c = good;
    c.cb.ensure_work = None;
    assert_eq!(msg(c), "loader: ensure_work is NULL\n");
    c = good;
    c.cb.load_trunk = None;
    assert_eq!(msg(c), "loader: load_trunk is NULL\n");
    c = good;
    c.cb.load_expert_tables = None;
    assert_eq!(msg(c), "loader: load_expert_tables is NULL\n");
    c = good;
    c.cb.alloc_state = None;
    assert_eq!(msg(c), "loader: alloc_state is NULL\n");
    c = good;
    c.handles = std::ptr::null_mut();
    assert_eq!(msg(c), "loader: handles is NULL\n");
    c = good;
    c.cb.host_alloc = None;
    assert_eq!(msg(c), "loader: host_alloc is NULL\n");
    c = good;
    c.slab = std::ptr::null_mut();
    assert_eq!(msg(c), "loader: slab is NULL\n");
    // without prefill host_alloc and slab are not required
    c = good;
    c.prefill = 0;
    c.cb.host_alloc = None;
    c.slab = std::ptr::null_mut();
    assert!(validate_config(&c).is_ok());
}

// ---------------------------------------------------------------- queue
type Starts = Arc<Mutex<Vec<(usize, ThreadId)>>>;
fn loader() -> (&'static Loader, Starts) {
    let starts: Starts = Arc::new(Mutex::new(vec![]));
    let s2 = Arc::clone(&starts);
    (Loader::new(Box::new(move |card| s2.lock().unwrap().push((card, std::thread::current().id())))), starts)
}
fn wait_until(what: &str, mut f: impl FnMut() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(10), "timed out waiting for {}", what);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn no_thread_before_the_first_submit_and_one_start_hook_per_card() {
    let (l, starts) = loader();
    l.wait(); // nothing pending: returns at once
    assert!(starts.lock().unwrap().is_empty());
    for _ in 0..5 {
        l.submit(3, Box::new(|| {}));
    }
    l.wait();
    l.submit(3, Box::new(|| {}));
    l.wait();
    let s = starts.lock().unwrap();
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].0, 3);
}

#[test]
fn start_hook_runs_before_the_first_job_on_the_card_thread() {
    // the hook and the job record into the same log
    let order = Arc::new(Mutex::new(vec![]));
    let o2 = Arc::clone(&order);
    let o3 = Arc::clone(&order);
    let l = Loader::new(Box::new(move |card| o3.lock().unwrap().push(format!("start {}", card))));
    l.submit(2, Box::new(move || o2.lock().unwrap().push("job".to_string())));
    l.wait();
    assert_eq!(*order.lock().unwrap(), vec!["start 2".to_string(), "job".to_string()]);
}

#[test]
fn jobs_of_one_card_run_in_submission_order_on_one_thread() {
    let (l, starts) = loader();
    let seen = Arc::new(Mutex::new(vec![]));
    for i in 0..200usize {
        let s = Arc::clone(&seen);
        l.submit(1, Box::new(move || s.lock().unwrap().push((i, std::thread::current().id()))));
    }
    l.wait();
    let s = seen.lock().unwrap();
    assert_eq!(s.iter().map(|x| x.0).collect::<Vec<_>>(), (0..200).collect::<Vec<_>>());
    assert!(s.iter().all(|x| x.1 == s[0].1));
    assert_eq!(starts.lock().unwrap().len(), 1);
    assert_eq!(starts.lock().unwrap()[0].1, s[0].1, "the start hook runs on the thread that runs the jobs");
}

#[test]
fn cards_have_distinct_threads() {
    let (l, starts) = loader();
    for card in 0..8usize {
        l.submit(card, Box::new(|| {}));
    }
    l.wait();
    let s = starts.lock().unwrap();
    assert_eq!(s.len(), 8);
    let mut ids: Vec<_> = s.iter().map(|x| format!("{:?}", x.1)).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 8);
    let mut cards: Vec<_> = s.iter().map(|x| x.0).collect();
    cards.sort();
    assert_eq!(cards, (0..8).collect::<Vec<_>>());
}

#[test]
fn jobs_of_different_cards_run_at_the_same_time() {
    // 4 jobs wait for each other: they can only all pass if the 4 cards' threads run at once
    let (l, _s) = loader();
    let arrived = Arc::new(AtomicUsize::new(0));
    let broken = Arc::new(AtomicBool::new(false));
    for card in 0..4usize {
        let a = Arc::clone(&arrived);
        let b = Arc::clone(&broken);
        l.submit(
            card,
            Box::new(move || {
                a.fetch_add(1, Ordering::SeqCst);
                let t0 = Instant::now();
                while a.load(Ordering::SeqCst) < 4 {
                    if t0.elapsed() > Duration::from_secs(10) {
                        b.store(true, Ordering::SeqCst);
                        return;
                    }
                    std::thread::yield_now();
                }
            }),
        );
    }
    l.wait();
    assert!(!broken.load(Ordering::SeqCst));
    assert_eq!(arrived.load(Ordering::SeqCst), 4);
}

#[test]
fn jobs_of_one_card_never_overlap() {
    let (l, _s) = loader();
    let running = Arc::new(AtomicUsize::new(0));
    let max = Arc::new(AtomicUsize::new(0));
    for _ in 0..50 {
        let (r, m) = (Arc::clone(&running), Arc::clone(&max));
        l.submit(
            6,
            Box::new(move || {
                let now = r.fetch_add(1, Ordering::SeqCst) + 1;
                m.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_micros(100));
                r.fetch_sub(1, Ordering::SeqCst);
            }),
        );
    }
    l.wait();
    assert_eq!(max.load(Ordering::SeqCst), 1);
}

#[test]
fn wait_blocks_until_the_job_finished_and_returns_with_the_count_at_zero() {
    let (l, _s) = loader();
    let gate = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let (g, d) = (Arc::clone(&gate), Arc::clone(&done));
    l.submit(
        0,
        Box::new(move || {
            while !g.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            d.store(true, Ordering::SeqCst);
        }),
    );
    assert_eq!(l.pending(), 1);
    let returned = Arc::new(AtomicBool::new(false));
    let r2 = Arc::clone(&returned);
    let w = std::thread::spawn(move || {
        l.wait();
        r2.store(true, Ordering::SeqCst);
    });
    std::thread::sleep(Duration::from_millis(100));
    assert!(!returned.load(Ordering::SeqCst), "wait returned while the job was blocked");
    gate.store(true, Ordering::SeqCst);
    w.join().unwrap();
    assert!(done.load(Ordering::SeqCst), "wait returned before the job finished");
    assert_eq!(l.pending(), 0);
}

#[test]
fn a_job_submitted_by_another_thread_during_the_wait_is_waited_for() {
    // A (gated) is pending, a waiter waits, a third thread submits B (gated) on another card, then A is released: pending is 1, so the waiter must still wait for B
    let (l, _s) = loader();
    let (ga, gb) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (a_done, b_done) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    {
        let (g, d) = (Arc::clone(&ga), Arc::clone(&a_done));
        l.submit(0, Box::new(move || {
            while !g.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            d.store(true, Ordering::SeqCst);
        }));
    }
    let returned = Arc::new(AtomicBool::new(false));
    let r2 = Arc::clone(&returned);
    let w = std::thread::spawn(move || {
        l.wait();
        r2.store(true, Ordering::SeqCst);
    });
    std::thread::sleep(Duration::from_millis(50));
    {
        let (g, d) = (Arc::clone(&gb), Arc::clone(&b_done));
        let s = std::thread::spawn(move || {
            l.submit(1, Box::new(move || {
                while !g.load(Ordering::SeqCst) {
                    std::thread::yield_now();
                }
                d.store(true, Ordering::SeqCst);
            }));
        });
        s.join().unwrap();
    }
    assert_eq!(l.pending(), 2);
    ga.store(true, Ordering::SeqCst);
    wait_until("A done", || a_done.load(Ordering::SeqCst));
    wait_until("pending 1", || l.pending() == 1);
    std::thread::sleep(Duration::from_millis(100));
    assert!(!returned.load(Ordering::SeqCst), "wait returned with B still pending");
    gb.store(true, Ordering::SeqCst);
    w.join().unwrap();
    assert!(b_done.load(Ordering::SeqCst));
    assert_eq!(l.pending(), 0);
}

#[test]
fn a_job_that_submits_a_job_keeps_the_count_above_zero() {
    // the parent submits the child (other card, then its own card) before it returns: the wait covers parent and both children
    let (l, _s) = loader();
    let log = Arc::new(Mutex::new(vec![]));
    let lg = Arc::clone(&log);
    l.submit(
        2,
        Box::new(move || {
            let (a, b) = (Arc::clone(&lg), Arc::clone(&lg));
            l.submit(5, Box::new(move || a.lock().unwrap().push("child other card")));
            l.submit(2, Box::new(move || b.lock().unwrap().push("child same card")));
            std::thread::sleep(Duration::from_millis(20));
            lg.lock().unwrap().push("parent");
        }),
    );
    l.wait();
    let v = log.lock().unwrap().clone();
    assert_eq!(v.len(), 3);
    // the same-card child can only run after the parent returned
    let pi = v.iter().position(|x| *x == "parent").unwrap();
    let si = v.iter().position(|x| *x == "child same card").unwrap();
    assert!(pi < si);
    assert!(v.contains(&"child other card"));
    assert_eq!(l.pending(), 0);
}

#[test]
fn several_waiters_all_return() {
    let (l, _s) = loader();
    let gate = Arc::new(AtomicBool::new(false));
    let g = Arc::clone(&gate);
    l.submit(0, Box::new(move || {
        while !g.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
    }));
    let n = Arc::new(AtomicUsize::new(0));
    let ws: Vec<_> = (0..4).map(|_| {
        let n = Arc::clone(&n);
        std::thread::spawn(move || {
            l.wait();
            n.fetch_add(1, Ordering::SeqCst);
        })
    }).collect();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(n.load(Ordering::SeqCst), 0);
    gate.store(true, Ordering::SeqCst);
    for w in ws {
        w.join().unwrap();
    }
    assert_eq!(n.load(Ordering::SeqCst), 4);
}

#[test]
fn concurrent_first_submits_to_one_card_make_one_thread_and_keep_each_submitters_order() {
    // the original's unlocked `if (!g_loader[card]) g_loader[card] = new Loader` is the race this exercises; here: one start hook, per-submitter order kept
    for _ in 0..50 {
        let (l, starts) = loader();
        let seen = Arc::new(Mutex::new(vec![]));
        let ts: Vec<_> = (0..4usize).map(|t| {
            let s = Arc::clone(&seen);
            std::thread::spawn(move || {
                for i in 0..10usize {
                    let s2 = Arc::clone(&s);
                    l.submit(4, Box::new(move || s2.lock().unwrap().push((t, i))));
                }
            })
        }).collect();
        for t in ts {
            t.join().unwrap();
        }
        l.wait();
        assert_eq!(starts.lock().unwrap().len(), 1);
        let s = seen.lock().unwrap();
        assert_eq!(s.len(), 40);
        for t in 0..4usize {
            let mine: Vec<usize> = s.iter().filter(|x| x.0 == t).map(|x| x.1).collect();
            assert_eq!(mine, (0..10).collect::<Vec<_>>());
        }
    }
}

// ---------------------------------------------------------------- C ABI: layout against the real header, and the exported functions
#[test]
fn struct_layout_and_o_direct_match_the_c_header() {
    use std::mem::{offset_of, size_of};
    let cb = ["ctx", "set_device", "ensure_work", "load_trunk", "load_expert_tables", "alloc_state", "host_alloc"];
    let cfg = ["cb", "expert_raw", "prefill", "slab", "handles"];
    let lay = ["handle", "dataset", "layer", "k", "positions", "card", "h_L", "h_card", "h_k_in", "h_fd"];
    let mut c = String::from("#define _GNU_SOURCE\n#include <stdio.h>\n#include <stddef.h>\n#include <fcntl.h>\n#include \"loader.h\"\nint main(void){\n");
    c += "printf(\"cb.size=%zu\\n\", sizeof(ld_callbacks));\nprintf(\"cfg.size=%zu\\n\", sizeof(ld_config));\nprintf(\"lay.size=%zu\\n\", sizeof(ld_layer));\n";
    c += "printf(\"O_DIRECT=%d\\n\", (int)O_DIRECT);\nprintf(\"off_t=%zu\\n\", sizeof(off_t));\n";
    for f in cb { c += &format!("printf(\"cb.{f}=%zu\\n\", offsetof(ld_callbacks, {f}));\n"); }
    for f in cfg { c += &format!("printf(\"cfg.{f}=%zu\\n\", offsetof(ld_config, {f}));\n"); }
    for f in lay { c += &format!("printf(\"lay.{f}=%zu\\n\", offsetof(ld_layer, {f}));\n"); }
    c += "return 0;}\n";

    use crate::abi::{ld_callbacks as C1, ld_config as C2, ld_layer as C3};
    let mut want: Vec<(String, usize)> = vec![
        ("cb.size".into(), size_of::<C1>()),
        ("cfg.size".into(), size_of::<C2>()),
        ("lay.size".into(), size_of::<C3>()),
        ("O_DIRECT".into(), O_DIRECT as usize),
        ("off_t".into(), 8),
    ];
    macro_rules! offs {
        ($t:ty, $c:literal, [$($f:ident),*]) => { $( want.push((format!("{}.{}", $c, stringify!($f)), offset_of!($t, $f))); )* };
    }
    offs!(C1, "cb", [ctx, set_device, ensure_work, load_trunk, load_expert_tables, alloc_state, host_alloc]);
    offs!(C2, "cfg", [cb, expert_raw, prefill, slab, handles]);
    offs!(C3, "lay", [handle, dataset, layer, k, positions, card, h_L, h_card, h_k_in, h_fd]);

    let dir = std::env::temp_dir().join(format!("loader_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src = dir.join("layout.c");
    let exe = dir.join("layout");
    std::fs::write(&src, c).expect("write C source");
    let inc = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let st = std::process::Command::new("cc").arg("-I").arg(inc).arg(&src).arg("-o").arg(&exe).status().expect("a C compiler named `cc` is required for this test");
    assert!(st.success(), "C compile of loader.h failed");
    let out = std::process::Command::new(&exe).output().expect("run layout probe");
    let got: Vec<(String, usize)> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.parse().unwrap_or(usize::MAX))))
        .collect();
    assert_eq!(got.len(), want.len());
    for w in &want {
        assert!(got.contains(w), "layout mismatch for {}: rust {} vs C {:?}", w.0, w.1, got.iter().find(|g| g.0 == w.0));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn exported_functions_end_to_end() {
    // the one test that uses the process-wide state (ld_init) and real files; prefill is off here (the process-wide state is configured once)
    use crate::{ld_init, ld_open_layer, ld_submit, ld_wait};
    use std::os::unix::fs::FileExt;
    let dir = std::env::temp_dir().join(format!("loader_e2e_{}", std::process::id()));
    std::fs::create_dir_all(dir.join("expert-2")).unwrap();
    let data: Vec<u8> = (0..8192u32).map(|i| (i * 13 + 5) as u8).collect();
    std::fs::write(dir.join("expert-2/experts.direct"), &data).unwrap();
    // the process-wide state keeps these addresses for the rest of the process: leaked
    let rec: &'static Rec = Box::leak(Box::new(Rec { v: Mutex::new(vec![]), slab_base: 0 }));
    let slab: &'static mut [*mut u8; 93] = Box::leak(Box::new([std::ptr::null_mut::<u8>(); 93]));
    let handles: &'static mut [*mut c_void; 93] = Box::leak(Box::new([std::ptr::null_mut::<c_void>(); 93]));
    let other: &'static mut [*mut c_void; 93] = Box::leak(Box::new([std::ptr::null_mut::<c_void>(); 93]));
    let c1 = cfg_with(0, slab.as_mut_ptr(), handles.as_mut_ptr(), rec);
    // ld_wait before ld_init returns at once
    ld_wait();
    // SAFETY: valid config; everything it points to is leaked.
    unsafe { ld_init(&c1) };
    // a second ld_init with another table is ignored: the first configuration is kept
    let c2 = cfg_with(0, slab.as_mut_ptr(), other.as_mut_ptr(), rec);
    // SAFETY: as above.
    unsafe { ld_init(&c2) };

    static HITS: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "C" fn job(arg: *mut c_void) {
        HITS.fetch_add(arg as usize, Ordering::SeqCst);
    }
    // SAFETY: job and the integer argument need no memory.
    unsafe {
        ld_submit(3, Some(job), 5 as *mut c_void);
        ld_submit(3, Some(job), 7 as *mut c_void);
    }
    ld_wait();
    assert_eq!(HITS.load(Ordering::SeqCst), 12);

    let ds = CString::new(dir.to_str().unwrap()).unwrap();
    let mut hv = 99u32;
    let (mut l, mut card, mut k, mut fd) = (-1, -1, -1, -5);
    let req = ld_layer { handle: &mut hv as *mut u32 as *mut c_void, dataset: ds.as_ptr(), layer: 2, k: 4, positions: 11, card: 6, h_L: &mut l, h_card: &mut card, h_k_in: &mut k, h_fd: &mut fd };
    // SAFETY: valid request; hv, l, card, k, fd, handles live until after ld_wait.
    unsafe { ld_open_layer(&req) };
    // bookkeeping is done on return
    assert_eq!((l, card, k), (2, 6, 4));
    assert_eq!(handles[2], &mut hv as *mut u32 as *mut c_void);
    assert!(other[2].is_null());
    assert!(handles.iter().enumerate().all(|(i, p)| i == 2 || p.is_null()));
    ld_wait();
    assert!(fd >= 0);
    // the fd is the experts file: aligned 4096-byte reads work for O_DIRECT or plain
    // SAFETY: fd was just opened by the job and is owned by nobody else; ManuallyDrop keeps it open.
    let file = std::mem::ManuallyDrop::new(unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) });
    let mut buf = vec![0u8; 8192 + 4096];
    let off = (4096 - (buf.as_ptr() as usize % 4096)) % 4096;
    let n = file.read_at(&mut buf[off..off + 4096], 4096).unwrap();
    assert_eq!(n, 4096);
    assert_eq!(&buf[off..off + 4096], &data[4096..8192]);
    let log = rec.v.lock().unwrap().clone();
    let want: Vec<String> = vec![
        "set_device 3".into(),
        "set_device 6".into(),
        "set_device 6".into(),
        "ensure_work 6".into(),
        format!("trunk L2 h=99 ds={}", dir.display()),
        format!("tables L2 h=99 ds={}", dir.display()),
        "state h=99 positions=11".into(),
    ];
    assert_eq!(log, want);
    assert!(slab.iter().all(|p| p.is_null()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fatal_child() {
    // not a test by itself: the parent below re-runs this binary with LOADER_TEST_FATAL=1 to see the real exit status and stderr text
    if std::env::var("LOADER_TEST_FATAL").is_ok() {
        crate::sys::fatal_msg(b"prefill: cannot read a record of layer 7\n");
    }
}

#[test]
fn fatal_message_goes_to_stderr_and_exits_with_status_1() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe).args(["--exact", "tests::fatal_child", "--nocapture", "--test-threads=1"]).env("LOADER_TEST_FATAL", "1").output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(out.stderr, b"prefill: cannot read a record of layer 7\n");
}
