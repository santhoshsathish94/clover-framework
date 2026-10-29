//! Unit tests of the planner against a FAKE ops table written here: caches answer from sets and scripts, the route table is a Vec, enqueue records the jobs.
//! The fake state is thread-local (every test runs on its own thread, `make` resets it), the expected logs were derived by hand from the original (backend-gpu.cu, decode.cuh).
use super::*;
use crate::dense::DenseMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::ptr::{null, null_mut};
use std::sync::Mutex;

const HOST: usize = 0x1000;
const HID: usize = 9; // cache id of the host cache in the logs; pools are 0..7
const NONE_NU: i32 = 1_000_000_000;

fn host_ptr() -> *mut sc_cache {
    HOST as *mut sc_cache
}
fn pool_ptr(card: usize) -> *mut sc_cache {
    (0x2000 + card * 0x10) as *mut sc_cache
}
fn cid(c: *const sc_cache) -> usize {
    let a = c as usize;
    if a == HOST {
        HID
    } else {
        (a - 0x2000) / 0x10
    }
}
/// address of the n-th slot the fake acquire / try_acquire hands out
fn slot_addr(n: usize) -> usize {
    0x10000 + 16 * n
}
fn removed_addr(key: i32) -> usize {
    0x5000 + key as usize
}

#[derive(Debug, Clone, PartialEq)]
enum Ev {
    SetNu(bool),
    Contains(usize, i32),
    Worth(usize, i32),
    Acq(usize, i32, Vec<i32>, f32),
    TryAcq(usize, i32, i32, Vec<i32>),
    Remove(usize, i32),
    Release(usize, usize),
    SetCur(i64),
    NextUse(i32),
    Enq(usize),
}
use Ev::*;

#[derive(Default)]
struct Fake {
    log: Vec<Ev>,
    held: HashMap<usize, HashSet<i32>>,
    worth_no: HashSet<i32>,
    try_no: HashSet<i32>,
    victims: HashMap<(usize, i32), (i32, f32)>,
    nslot: usize,
    nu: HashMap<i32, i32>,
    calls: Vec<(i32, Vec<i32>)>,
    jobs: Vec<cq_job>,
    enqueue_sleep_ms: u64,
}

thread_local! { static F: RefCell<Fake> = RefCell::new(Fake::default()); }

fn f<R>(g: impl FnOnce(&mut Fake) -> R) -> R {
    F.with(|x| g(&mut x.borrow_mut()))
}
fn reset() {
    F.with(|x| *x.borrow_mut() = Fake::default());
}
fn log() -> Vec<Ev> {
    f(|x| x.log.clone())
}
fn clear_log() {
    f(|x| x.log.clear());
}
fn job(i: usize) -> cq_job {
    f(|x| x.jobs[i])
}
fn njobs() -> usize {
    f(|x| x.jobs.len())
}
fn hold(id: usize, key: i32) {
    f(|x| {
        x.held.entry(id).or_default().insert(key);
    });
}
fn add_calls(c: &[(i32, &[i32])]) {
    f(|x| x.calls.extend(c.iter().map(|(l, e)| (*l, e.to_vec()))));
}
fn set1(p: *mut c_int) {
    unsafe { (*(p as *const AtomicI32)).store(1, Ordering::Release) };
}
fn is_set(p: *mut c_int) -> bool {
    unsafe { (*(p as *const AtomicI32)).load(Ordering::Acquire) == 1 }
}

unsafe fn prot_vec(prot: *const c_int, n: usize) -> Vec<i32> {
    let mut v = if n == 0 { Vec::new() } else { std::slice::from_raw_parts(prot, n).to_vec() };
    v.sort();
    v
}

unsafe extern "C" fn f_set_nu(fnp: sc_next_use_fn) {
    f(|x| x.log.push(SetNu(fnp.is_some())))
}
unsafe extern "C" fn f_contains(c: *const sc_cache, key: c_int) -> c_int {
    f(|x| {
        let id = cid(c);
        x.log.push(Contains(id, key));
        x.held.get(&id).map_or(false, |s| s.contains(&key)) as c_int
    })
}
unsafe extern "C" fn f_worth(c: *const sc_cache, key: c_int) -> c_int {
    f(|x| {
        x.log.push(Worth(cid(c), key));
        !x.worth_no.contains(&key) as c_int
    })
}
fn grant(x: &mut Fake, id: usize, key: i32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
    let slot = slot_addr(x.nslot) as *mut u8;
    x.nslot += 1;
    x.held.entry(id).or_default().insert(key);
    unsafe {
        if let Some(&(vk, vh)) = x.victims.get(&(id, key)) {
            x.held.entry(id).or_default().remove(&vk);
            *vkey = vk;
            *vheat = vh;
        } else {
            *vkey = -1;
        }
    }
    slot
}
unsafe extern "C" fn f_acquire(c: *mut sc_cache, key: c_int, prot: *const c_int, nprot: usize, heat: f32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
    let p = prot_vec(prot, nprot);
    f(|x| {
        let id = cid(c);
        x.log.push(Acq(id, key, p, heat));
        grant(x, id, key, vkey, vheat)
    })
}
unsafe extern "C" fn f_try_acquire(c: *mut sc_cache, key: c_int, cand: c_int, prot: *const c_int, nprot: usize, _heat: f32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
    let p = prot_vec(prot, nprot);
    f(|x| {
        let id = cid(c);
        x.log.push(TryAcq(id, key, cand, p));
        if x.try_no.contains(&key) {
            *vkey = -1;
            return null_mut();
        }
        grant(x, id, key, vkey, vheat)
    })
}
unsafe extern "C" fn f_remove(c: *mut sc_cache, key: c_int) -> *mut u8 {
    f(|x| {
        let id = cid(c);
        x.log.push(Remove(id, key));
        if x.held.entry(id).or_default().remove(&key) {
            removed_addr(key) as *mut u8
        } else {
            null_mut()
        }
    })
}
unsafe extern "C" fn f_release(c: *mut sc_cache, slot: *mut u8) {
    f(|x| x.log.push(Release(cid(c), slot as usize)))
}
unsafe extern "C" fn f_calls(_t: *const rt_table) -> usize {
    f(|x| x.calls.len())
}
unsafe extern "C" fn f_layer(_t: *const rt_table, i: usize) -> c_int {
    f(|x| x.calls[i].0)
}
unsafe extern "C" fn f_experts(_t: *const rt_table, i: usize, n: *mut usize) -> *const c_int {
    f(|x| {
        let e = &x.calls[i].1;
        *n = e.len();
        if e.is_empty() {
            null()
        } else {
            e.as_ptr()
        }
    })
}
unsafe extern "C" fn f_set_cur(_t: *mut rt_table, cur: c_long) {
    f(|x| x.log.push(SetCur(cur as i64)))
}
unsafe extern "C" fn f_next_use(_t: *mut rt_table, key: c_int) -> c_int {
    f(|x| {
        x.log.push(NextUse(key));
        x.nu.get(&key).copied().unwrap_or(NONE_NU)
    })
}
unsafe extern "C" fn f_enqueue(j: *const cq_job) {
    let ms = f(|x| {
        x.jobs.push(*j);
        let n = x.jobs.len() - 1;
        x.log.push(Enq(n));
        x.enqueue_sleep_ms
    });
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

fn fake_ops() -> sp_ops {
    sp_ops {
        sc_set_next_use: f_set_nu,
        sc_contains: f_contains,
        sc_worth: f_worth,
        sc_acquire: f_acquire,
        sc_try_acquire: f_try_acquire,
        sc_remove: f_remove,
        sc_release: f_release,
        table: 0x3000 as *mut rt_table,
        rt_calls: f_calls,
        rt_layer: f_layer,
        rt_experts: f_experts,
        rt_set_current: f_set_cur,
        rt_next_use: f_next_use,
        enqueue: f_enqueue,
    }
}

fn base_cfg() -> sp_config {
    sp_config { experts: 10, expert_raw: 1000, cards: 2, lookahead: 5, pend_cap: 100, vq: 0, vq_cap: 256, vq_from: 0, prefill: 0, rc_ok: 1 }
}

/// layer 1 -> card 0 (dev 5, fd 11), layer 2 -> card 1 (dev 6, fd 12), both pools ready, host cache of 10 slots; the fake is reset
fn make(cfg: sp_config) -> Planner {
    reset();
    let layers: &'static LayerTable = Box::leak(Box::new(LayerTable::new()));
    layers.set(1, 11, 0);
    layers.set(2, 12, 1);
    let mut w = Wiring::UNSET;
    w.pools[0] = Pool { cache: pool_ptr(0), dev: 5, ready: true };
    w.pools[1] = Pool { cache: pool_ptr(1), dev: 6, ready: true };
    w.host = Host { cache: host_ptr(), cap_slots: 10 };
    Planner::new(fake_ops(), cfg, layers, w)
}

fn vq_cfg() -> sp_config {
    sp_config { vq: 1, ..base_cfg() }
}

fn pending_done(slot: *mut u8) -> Box<Pending> {
    Box::new(Pending { state: AtomicI32::new(1), slot })
}
fn pending_open() -> Box<Pending> {
    Box::new(Pending { state: AtomicI32::new(0), slot: null_mut() })
}

// ---------------------------------------------------------------- read-ahead plan

#[test]
fn read_ahead_plan_job_list_and_skipped() {
    let mut p = make(base_cfg());
    // layer 1 keys 10.., layer 2 keys 20..; pool 0 holds 13, the host holds 25
    add_calls(&[(1, &[3, 4]), (2, &[5]), (1, &[3, 6])]);
    hold(0, 13);
    hold(HID, 25);
    p.route_hook(1, &[3, 4]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            // call 0: expert 3 (key 13) is on the card -> skipped; expert 4 (key 14) is read ahead
            Contains(0, 13),
            Contains(0, 14),
            Contains(HID, 14),
            Worth(HID, 14),
            Acq(HID, 14, vec![], 1.0),
            Enq(0),
            // call 1: key 25 is in the host cache (the pool check comes first) -> skipped
            Contains(1, 25),
            Contains(HID, 25),
            // call 2: key 13 skipped again, key 16 read ahead with prot = {14}
            Contains(0, 13),
            Contains(0, 16),
            Contains(HID, 16),
            Worth(HID, 16),
            Acq(HID, 16, vec![14], 1.0),
            Enq(1),
        ]
    );
    let (j0, j1) = (job(0), job(1));
    assert_eq!((j0.kind, j0.dev, j0.layer, j0.fd, j0.off), (0, 0, 0, 11, 4000));
    assert_eq!((j1.kind, j1.dev, j1.layer, j1.fd, j1.off), (0, 0, 0, 11, 6000));
    assert_eq!((j0.slot as usize, j1.slot as usize), (slot_addr(0), slot_addr(1)));
    for j in [j0, j1] {
        assert!(j.dst.is_null() && j.host_src.is_null() && j.victim_host.is_null());
        assert!(j.wb_state.is_null() && j.arr_state.is_null() && j.rel_state.is_null());
    }
    assert_eq!(j0.p_state, p.pend.get(14).unwrap().state.as_ptr());
    assert_eq!(j1.p_state, p.pend.get(16).unwrap().state.as_ptr());
    assert_eq!(p.pend.get(14).unwrap().slot as usize, slot_addr(0));
    let s = p.stats();
    assert_eq!((s.planned, s.skipped, s.planned_to, s.pos, s.pending), (2, 3, 3, 1, 2));
    assert_eq!(s.rc_ok, 1);
    assert!(s.t_plan >= 0.0);
}

#[test]
fn pend_cap_early_return_and_planned_resumption() {
    let mut p = make(sp_config { pend_cap: 2, ..base_cfg() });
    add_calls(&[(1, &[0]), (1, &[1, 2])]);
    p.route_hook(1, &[0]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            Contains(0, 10),
            Contains(HID, 10),
            Worth(HID, 10),
            Acq(HID, 10, vec![], 1.0),
            Enq(0),
            Contains(0, 11),
            Contains(HID, 11),
            Worth(HID, 11),
            Acq(HID, 11, vec![10], 1.0),
            Enq(1),
            // key 12: pending count 2 >= cap 2 -> planned = i (1) and return, no worth() call
            Contains(0, 12),
            Contains(HID, 12),
        ]
    );
    assert_eq!(p.stats().planned_to, 1);
    assert_eq!(p.stats().planned, 2);
    // the first read finished and call 0 took it; planning resumes at max(planned, pos) = 1, key 11 is pending
    set1(job(0).p_state);
    p.pend_take(10);
    assert_eq!((p.stats().pf_used, p.stats().pf_waited, p.stats().pending), (1, 0, 1));
    clear_log();
    p.route_hook(1, &[1, 2]);
    assert_eq!(log(), vec![SetCur(1), Contains(0, 12), Contains(HID, 12), Worth(HID, 12), Acq(HID, 12, vec![11], 1.0), Enq(2)]);
    assert_eq!((p.stats().planned_to, p.stats().planned, p.stats().pos), (2, 3, 2));
}

#[test]
fn worth_false_returns_without_looking_further() {
    let mut p = make(base_cfg());
    add_calls(&[(1, &[0, 1, 2])]);
    f(|x| x.worth_no.insert(11));
    p.route_hook(1, &[0, 1, 2]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            Contains(0, 10),
            Contains(HID, 10),
            Worth(HID, 10),
            Acq(HID, 10, vec![], 1.0),
            Enq(0),
            Contains(0, 11),
            Contains(HID, 11),
            Worth(HID, 11), // false: planned = i (0), return; key 12 is never looked at
        ]
    );
    assert_eq!((p.stats().planned_to, p.stats().planned), (0, 1));
}

#[test]
fn calls_without_handle_or_fd_are_passed_over_and_not_revisited() {
    let mut p = make(base_cfg());
    p.layers.set(1, -1, 0); // layer 1 is open but its file is not yet
    add_calls(&[(3, &[1]), (1, &[2]), (2, &[4]), (1, &[2])]);
    p.route_hook(3, &[1]); // layer 3 has no handle at all
    // i = 0 (no handle) -> planned_to 1; i = 1 (fd < 0) -> 2; i = 2: key 24 planned with fd 12 on card 1 -> 3; i = 3 is layer 1 with fd < 0 -> 4
    assert_eq!(log(), vec![SetCur(0), Contains(1, 24), Contains(HID, 24), Worth(HID, 24), Acq(HID, 24, vec![], 1.0), Enq(0)]);
    assert_eq!(job(0).fd, 12);
    assert_eq!(p.stats().planned_to, 4);
    // the loader thread opens layer 1 later: calls already passed (planned_to = 4) are not planned again
    p.layers.set_fd(1, 21);
    clear_log();
    p.route_hook(1, &[2]);
    assert_eq!(log(), vec![SetCur(1)]);
}

#[test]
fn lookahead_bounds_the_window_and_is_clamped_to_the_table() {
    let mut p = make(sp_config { lookahead: 0, ..base_cfg() });
    add_calls(&[(1, &[0]), (1, &[1]), (1, &[2])]);
    p.route_hook(1, &[0]);
    assert_eq!((p.stats().planned_to, p.stats().planned), (1, 1)); // last = pos + 0 + 1
    let mut q = make(sp_config { lookahead: 1000, ..base_cfg() });
    add_calls(&[(1, &[0]), (1, &[1]), (1, &[2])]);
    q.route_hook(1, &[0]);
    assert_eq!((q.stats().planned_to, q.stats().planned), (3, 3)); // min(calls, ...)
}

#[test]
fn key_and_offset_use_the_configured_experts_and_expert_raw() {
    let mut p = make(sp_config { experts: 896, expert_raw: 17_547_264, ..base_cfg() });
    add_calls(&[(2, &[895, 0])]);
    p.route_hook(2, &[895, 0]);
    let key = 2 * 896 + 895;
    assert!(log().contains(&Acq(HID, key, vec![], 1.0)));
    assert_eq!(job(0).off, 895i64 * 17_547_264); // 15,704,801,280: needs 64 bits
    assert_eq!(job(1).off, 0);
    assert_eq!(job(0).fd, 12);
}

#[test]
fn rc_ok_zero_from_the_start_only_sets_current_and_counts() {
    let mut p = make(sp_config { rc_ok: 0, ..base_cfg() });
    add_calls(&[(1, &[0])]);
    p.route_hook(1, &[0]);
    p.route_hook(7, &[9, 9]);
    assert_eq!(log(), vec![SetCur(0), SetCur(1)]);
    let s = p.stats();
    assert_eq!((s.pos, s.rc_ok, s.planned, s.skipped), (2, 0, 0, 0));
    assert_eq!(s.t_plan, 0.0);
}

#[test]
fn vq_from_selects_the_plan_by_call_number() {
    let mut p = make(sp_config { vq_from: 1, ..vq_cfg() });
    add_calls(&[(1, &[1]), (1, &[2])]);
    p.route_hook(1, &[1]); // pos 0 < vq_from: read-ahead plan, both calls
    assert_eq!(log().iter().filter(|e| matches!(e, TryAcq(..))).count(), 0);
    assert_eq!((p.stats().planned, p.stats().planned_to), (2, 2));
    clear_log();
    p.route_hook(1, &[2]); // pos 1: card-queue plan; key 12 is in the host cache with a read in flight -> issue refuses
    assert_eq!(log(), vec![SetCur(1), Contains(0, 12), Contains(HID, 12)]);
    assert_eq!(p.stats().vq_drive + p.stats().vq_ram, 0);
}

// ---------------------------------------------------------------- mismatch

#[test]
fn mismatch_message_text_is_exact() {
    assert_eq!(mismatch_message(12, 34), b"K3GPU route cache: call 12 (layer 34) does not match the cache; look-ahead stopped\n".to_vec());
}

#[test]
fn mismatch_effects_for_each_cause() {
    // a different expert, a different layer, a prefix of the list, a longer list, a position beyond the table
    let cases: [(i32, &[i32]); 4] = [(1, &[3, 5]), (2, &[3, 4]), (1, &[3]), (1, &[3, 4, 5])];
    for (layer, uniq) in cases {
        let mut p = make(base_cfg());
        add_calls(&[(1, &[3, 4])]);
        p.route_hook(layer, uniq);
        assert_eq!(log(), vec![SetCur(0), SetNu(false)], "layer {layer} uniq {uniq:?}");
        let s = p.stats();
        assert_eq!((s.rc_ok, s.pos, s.planned, s.skipped), (0, 1, 0, 0));
        assert_eq!(s.t_plan, 0.0);
        clear_log();
        p.route_hook(1, &[3, 4]); // from now on only set_current and the increment
        assert_eq!(log(), vec![SetCur(1)]);
        assert_eq!(p.stats().pos, 2);
    }
    let mut p = make(base_cfg());
    add_calls(&[(1, &[3])]);
    p.route_hook(1, &[3]);
    clear_log();
    p.route_hook(1, &[3]); // pos 1 >= calls 1
    assert_eq!(log(), vec![SetCur(1), SetNu(false)]);
    assert_eq!(p.stats().rc_ok, 0);
}

/// Run only by `mismatch_message_reaches_stderr` as a child process.
#[test]
fn mismatch_child() {
    if std::env::var("PLANNER_MISMATCH_CHILD").is_err() {
        return;
    }
    let mut p = make(base_cfg());
    add_calls(&[(1, &[3, 4])]);
    p.route_hook(1, &[3, 5]);
}

#[test]
fn mismatch_message_reaches_stderr() {
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe)
        .args(["--exact", "planner::tests::mismatch_child", "--nocapture", "--test-threads=1"])
        .env("PLANNER_MISMATCH_CHILD", "1")
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("test planner::tests::mismatch_child ... ok"), "child did not run");
    assert_eq!(out.stderr, mismatch_message(0, 1));
}

#[test]
fn t_plan_counts_planning_time_only() {
    let mut p = make(base_cfg());
    add_calls(&[(1, &[0])]);
    f(|x| x.enqueue_sleep_ms = 30);
    p.route_hook(1, &[0]);
    assert!(p.stats().t_plan >= 0.025, "t_plan {}", p.stats().t_plan);
    let before = p.stats().t_plan;
    p.route_hook(1, &[0]); // mismatch (pos 1 beyond the table): no planning time added
    assert_eq!(p.stats().t_plan, before);
}

// ---------------------------------------------------------------- card-queue plan

#[test]
fn vq_plan_in_ram_false_and_true() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[3, 4]), (2, &[5])]);
    hold(HID, 14); // key 14 is in RAM; 13 and 25 are not
    f(|x| x.nu.insert(25, 7));
    p.route_hook(1, &[3, 4]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            // call 0 is the current call: cand = pos (0), no next_use call
            Contains(0, 13),
            Contains(HID, 13),
            TryAcq(0, 13, 0, vec![13, 14]),
            Enq(0),
            Contains(0, 14),
            Contains(HID, 14),
            TryAcq(0, 14, 0, vec![13, 14]),
            Remove(HID, 14),
            Enq(1),
            // call 1: cand = next_use(25), asked before the host cache is looked at
            Contains(1, 25),
            NextUse(25),
            Contains(HID, 25),
            TryAcq(1, 25, 7, vec![13, 14]),
            Enq(2),
        ]
    );
    let (a, b, c) = (job(0), job(1), job(2));
    assert_eq!((a.kind, a.dev, a.fd, a.off, a.layer), (1, 5, 11, 3000, 1));
    assert_eq!(a.dst as usize, slot_addr(0));
    assert!(a.host_src.is_null() && a.victim_host.is_null() && a.p_state.is_null() && a.wb_state.is_null() && a.rel_state.is_null() && a.slot.is_null());
    assert!(!a.arr_state.is_null());
    assert_eq!((b.kind, b.dev, b.fd, b.off, b.layer), (2, 5, 11, 4000, 1));
    assert_eq!((b.dst as usize, b.host_src as usize), (slot_addr(1), removed_addr(14)));
    assert!(!b.arr_state.is_null() && !b.rel_state.is_null() && b.wb_state.is_null() && b.victim_host.is_null());
    assert_eq!((c.kind, c.dev, c.fd, c.off, c.layer), (1, 6, 12, 5000, 2));
    assert!(c.rel_state.is_null());
    assert_eq!(p.rel.len(), 1);
    assert_eq!(p.rel[0].slot as usize, removed_addr(14));
    assert_eq!(p.rel[0].state.as_ptr(), b.rel_state);
    assert_eq!(p.arr.get(14).unwrap().state.as_ptr(), b.arr_state);
    let s = p.stats();
    assert_eq!((s.vq_ram, s.vq_drive, s.vq_wb, s.vq_nowb, s.planned, s.arrivals, s.releases), (1, 2, 0, 0, 0, 3, 1));
}

#[test]
fn vq_issue_refuses_while_the_ram_read_is_in_flight_then_takes_it() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[4]), (1, &[4])]);
    hold(HID, 14);
    let open = pending_open();
    let open_state = open.state.as_ptr();
    p.insert_pend(14, open);
    p.route_hook(1, &[4]);
    // call 0: in RAM with a pending read at state 0 -> vq_issue returns false (full[card 0] = true), the key is pending -> next;
    // call 1 (same key): the card is full, so no issue, the key is pending -> next
    assert_eq!(log(), vec![SetCur(0), Contains(0, 14), Contains(HID, 14), Contains(0, 14)]);
    assert_eq!(njobs(), 0);
    set1(open_state);
    clear_log();
    p.route_hook(1, &[4]);
    // pos 1: state 1 -> issue goes ahead; pend_take frees the entry (no wait), then the key leaves the host cache
    assert_eq!(log(), vec![SetCur(1), Contains(0, 14), Contains(HID, 14), TryAcq(0, 14, 1, vec![14]), Remove(HID, 14), Enq(0)]);
    assert_eq!(job(0).kind, 2);
    let s = p.stats();
    assert_eq!((s.pf_used, s.pf_waited, s.pending, s.vq_ram), (1, 0, 0, 1));
}

/// One call (1,[3]) on card 0; try_acquire hands out victim 19 (heat 3.5) for key 13; `pend_n` dummy pending entries (keys 100..) exist.
fn wb_case(pend_n: usize, victim_worth: bool, prefill: bool, cap: usize) -> (Vec<Ev>, cq_job, Planner) {
    let mut p = make(sp_config { prefill: prefill as c_int, ..vq_cfg() });
    p.w.host.cap_slots = cap;
    add_calls(&[(1, &[3])]);
    f(|x| {
        x.victims.insert((0, 13), (19, 3.5));
        if !victim_worth {
            x.worth_no.insert(19);
        }
    });
    for k in 0..pend_n {
        p.pend.insert(100 + k as i32, pending_done(null_mut()));
    }
    p.route_hook(1, &[3]);
    let (l, j) = (log(), job(0));
    (l, j, p)
}

fn wb_prefix() -> Vec<Ev> {
    vec![SetCur(0), Contains(0, 13), Contains(HID, 13), TryAcq(0, 13, 0, vec![13])]
}

#[test]
fn write_back_rule_eight_tenths_of_the_host_slots() {
    // cap 10: threshold 10 * 8 / 10 = 8; 7 pending entries -> write back
    let (l, j, p) = wb_case(7, true, false, 10);
    let mut want = wb_prefix();
    want.extend([Worth(HID, 19), Acq(HID, 19, (100..107).collect(), 3.5), Enq(0)]);
    assert_eq!(l, want);
    assert_eq!(j.kind, 1);
    assert_eq!(j.victim_host as usize, slot_addr(1));
    assert_eq!(j.wb_state, p.pend.get(19).unwrap().state.as_ptr());
    assert_eq!(p.pend.get(19).unwrap().slot as usize, slot_addr(1));
    assert_eq!(p.wb_keys, vec![19]);
    assert_eq!((p.stats().vq_wb, p.stats().vq_nowb, p.stats().pending), (1, 0, 8));
    // 8 pending entries -> not below the threshold -> dropped
    let (l, j, p) = wb_case(8, true, false, 10);
    let mut want = wb_prefix();
    want.push(Enq(0));
    assert_eq!(l, want);
    assert!(j.victim_host.is_null() && j.wb_state.is_null());
    assert_eq!((p.stats().vq_wb, p.stats().vq_nowb), (0, 1));
    // worth() false for the victim: asked, then dropped
    let (l, _, p) = wb_case(7, false, false, 10);
    let mut want = wb_prefix();
    want.extend([Worth(HID, 19), Enq(0)]);
    assert_eq!(l, want);
    assert_eq!((p.stats().vq_wb, p.stats().vq_nowb), (0, 1));
    // prefill: dropped, worth() not asked
    let (l, _, p) = wb_case(7, true, true, 10);
    let mut want = wb_prefix();
    want.push(Enq(0));
    assert_eq!(l, want);
    assert_eq!(p.stats().vq_nowb, 1);
    // host cache with no slots: dropped, worth() not asked
    let (l, _, p) = wb_case(0, true, false, 0);
    let mut want = wb_prefix();
    want.push(Enq(0));
    assert_eq!(l, want);
    assert_eq!(p.stats().vq_nowb, 1);
    // integer division: cap 13 -> 104 / 10 = 10
    let (_, _, p) = wb_case(9, true, false, 13);
    assert_eq!(p.stats().vq_wb, 1);
    let (_, _, p) = wb_case(10, true, false, 13);
    assert_eq!(p.stats().vq_nowb, 1);
}

#[test]
fn victim_arrival_is_waited_for_and_forgotten_before_the_write_back() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    f(|x| x.victims.insert((0, 13), (19, 3.5)));
    let a = Box::new(Arrival { state: AtomicI32::new(1) });
    p.insert_arr(19, a);
    p.route_hook(1, &[3]);
    assert!(p.arr.get(19).is_none());
    assert!(p.arr.get(13).is_some());
    assert_eq!((p.stats().vq_waited, p.stats().arrivals), (0, 1));
    // the arrival of the victim not finished yet: spin until a thread sets it
    let mut q = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    f(|x| x.victims.insert((0, 13), (19, 3.5)));
    let a = Box::new(Arrival { state: AtomicI32::new(0) });
    let addr = a.state.as_ptr() as usize;
    q.insert_arr(19, a);
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(60));
        unsafe { (*(addr as *const AtomicI32)).store(1, Ordering::Release) };
    });
    q.route_hook(1, &[3]);
    h.join().unwrap();
    assert_eq!(q.stats().vq_waited, 1);
    assert!(q.stats().t_vwait >= 0.05, "t_vwait {}", q.stats().t_vwait);
}

#[test]
fn full_card_stops_issuing_for_that_card_only() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[1, 2, 3]), (2, &[4])]);
    f(|x| x.try_no.insert(11));
    p.route_hook(1, &[1, 2, 3]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            // key 11: the pool refuses -> full[0]; the same key goes to the host read-ahead
            Contains(0, 11),
            Contains(HID, 11),
            TryAcq(0, 11, 0, vec![11, 12, 13]),
            Contains(HID, 11),
            Worth(HID, 11),
            Acq(HID, 11, vec![], 1.0),
            Enq(0),
            // keys 12, 13: card 0 is full, no try_acquire
            Contains(0, 12),
            Contains(HID, 12),
            Worth(HID, 12),
            Acq(HID, 12, vec![11], 1.0),
            Enq(1),
            Contains(0, 13),
            Contains(HID, 13),
            Worth(HID, 13),
            Acq(HID, 13, vec![11, 12], 1.0),
            Enq(2),
            // card 1 is not full
            Contains(1, 24),
            NextUse(24),
            Contains(HID, 24),
            TryAcq(1, 24, NONE_NU, vec![11, 12, 13]),
            Enq(3),
        ]
    );
}

#[test]
fn inflight_limit_counts_unfinished_arrivals_and_stops_issuing_without_marking_full() {
    let mut p = make(sp_config { vq_cap: 2, ..vq_cfg() });
    add_calls(&[(1, &[1, 2])]);
    p.insert_arr(99, Box::new(Arrival { state: AtomicI32::new(0) })); // counts
    p.insert_arr(98, Box::new(Arrival { state: AtomicI32::new(1) })); // does not
    p.route_hook(1, &[1, 2]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            Contains(0, 11),
            Contains(HID, 11),
            TryAcq(0, 11, 0, vec![11, 12]),
            Enq(0), // out 1 -> 2
            Contains(0, 12),
            // out (2) >= cap (2): no issue; the host read-ahead takes it
            Contains(HID, 12),
            Worth(HID, 12),
            Acq(HID, 12, vec![], 1.0),
            Enq(1),
        ]
    );
    assert_eq!(job(0).kind, 1);
    assert_eq!(job(1).kind, 0);
    assert_eq!(p.stats().planned, 1);
}

#[test]
fn vq_plan_skips_keys_held_by_the_pool_and_keys_pending_or_in_ram() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[1, 2, 3])]);
    f(|x| x.try_no.insert(12));
    hold(0, 11); // on the card or on its way
    hold(HID, 13); // in RAM (no pending entry) but try_acquire is fine for 13
    p.route_hook(1, &[1, 2, 3]);
    assert_eq!(
        log(),
        vec![
            SetCur(0),
            Contains(0, 11), // held: next
            Contains(0, 12),
            Contains(HID, 12),
            TryAcq(0, 12, 0, vec![11, 12, 13]), // refused -> full[0]
            Contains(HID, 12),
            Worth(HID, 12),
            Acq(HID, 12, vec![], 1.0),
            Enq(0),
            Contains(0, 13), // card full: host side: key 13 is in RAM -> next
            Contains(HID, 13),
        ]
    );
}

#[test]
fn host_side_of_the_vq_plan_respects_pend_cap_and_worth() {
    let mut p = make(sp_config { pend_cap: 1, ..vq_cfg() });
    add_calls(&[(1, &[1, 2])]);
    f(|x| x.try_no.extend([11, 12]));
    p.route_hook(1, &[1, 2]);
    // key 11: refused by the pool, read ahead (pend 1 = cap); key 12: full card, pend 1 >= 1 -> next WITHOUT worth() and without stopping
    assert_eq!(p.stats().planned, 1);
    assert!(!log().contains(&Worth(HID, 12)));
    assert!(log().ends_with(&[Contains(0, 12), Contains(HID, 12)]));
}

// ---------------------------------------------------------------- sweep

#[test]
fn sweep_removes_by_swapping_with_the_last_element() {
    let mut p = make(vq_cfg());
    let mk = |state: i32, slot: usize| Box::new(Rel { state: AtomicI32::new(state), slot: slot as *mut u8 });
    // states 1,0,1,1 for slots A..D: A done -> D takes its place and is done -> C takes its place and is done -> B stays
    p.rel.extend([mk(1, 0xA), mk(0, 0xB), mk(1, 0xC), mk(1, 0xD)]);
    for k in [10, 12, 13, 14] {
        p.pend.insert(k, if k == 12 { pending_open() } else { pending_done(null_mut()) });
    }
    p.wb_keys = vec![10, 11, 12, 13, 14]; // 11 has no pending entry
    p.vq_sweep();
    assert_eq!(log(), vec![Release(HID, 0xA), Release(HID, 0xD), Release(HID, 0xC)]);
    assert_eq!(p.rel.len(), 1);
    assert_eq!(p.rel[0].slot as usize, 0xB);
    // wb_keys: 10 done -> [14,11,12,13]; 14 done -> [13,11,12]; 13 done -> [12,11]; 12 open -> stays; 11 absent -> removed
    assert_eq!(p.wb_keys, vec![12]);
    assert_eq!(p.pend.len(), 1);
    assert!(p.pend.contains(12));
}

#[test]
fn sweep_runs_before_the_plan_and_releases_to_the_host_cache() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[1])]);
    p.rel.push(Box::new(Rel { state: AtomicI32::new(1), slot: 0xAB as *mut u8 }));
    f(|x| x.try_no.insert(11));
    p.route_hook(1, &[1]);
    assert_eq!(&log()[..3], &[SetCur(0), Release(HID, 0xAB), Contains(0, 11)]);
    assert_eq!(p.stats().releases, 0);
}

// ---------------------------------------------------------------- waiting

#[test]
fn pend_take_spins_until_the_state_is_set() {
    let mut p = make(base_cfg());
    let e = pending_open();
    let addr = e.state.as_ptr() as usize;
    p.insert_pend(5, e);
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(60));
        unsafe { (*(addr as *const AtomicI32)).store(1, Ordering::Release) };
    });
    p.pend_take(5);
    h.join().unwrap();
    let s = p.stats();
    assert_eq!((s.pf_used, s.pf_waited, s.pending), (1, 1, 0));
    assert!(s.t_pwait >= 0.05, "t_pwait {}", s.t_pwait);
    assert!(!p.pend.contains(5));
    // absent key: no-op; finished entry: used but not waited
    p.pend_take(5);
    p.insert_pend(6, pending_done(null_mut()));
    p.pend_take(6);
    let s = p.stats();
    assert_eq!((s.pf_used, s.pf_waited), (2, 1));
}

#[test]
fn arr_wait_spins_forgets_the_entry_first_and_ignores_absent_keys() {
    let mut p = make(base_cfg());
    let a = Box::new(Arrival { state: AtomicI32::new(0) });
    let addr = a.state.as_ptr() as usize;
    p.insert_arr(7, a);
    let h = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(60));
        unsafe { (*(addr as *const AtomicI32)).store(1, Ordering::Release) };
    });
    p.arr_wait(7);
    h.join().unwrap();
    let s = p.stats();
    assert_eq!((s.vq_waited, s.arrivals), (1, 0));
    assert!(s.t_vwait >= 0.05, "t_vwait {}", s.t_vwait);
    p.arr_wait(7);
    p.insert_arr(8, Box::new(Arrival { state: AtomicI32::new(1) }));
    p.arr_wait(8);
    assert_eq!((p.stats().vq_waited, p.stats().arrivals), (1, 0));
}

// ---------------------------------------------------------------- replaced entries

#[test]
fn replaced_pending_entry_is_kept_until_its_state_reads_one() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[3]), (1, &[3])]);
    f(|x| x.victims.insert((0, 13), (19, 3.5)));
    let old = pending_open(); // a read of key 19 still in flight
    let old_addr = old.state.as_ptr() as usize;
    p.insert_pend(19, old);
    p.route_hook(1, &[3]);
    // the write-back entry replaced it: the old one is alive in the graveyard, not freed
    assert_eq!(p.graveyard_len(), (1, 0));
    assert_eq!(p.pend.len(), 1);
    let new_state = p.pend.get(19).unwrap().state.as_ptr() as usize;
    assert_ne!(new_state, old_addr);
    assert_eq!(unsafe { (*(old_addr as *const AtomicI32)).load(Ordering::Acquire) }, 0);
    p.vq_sweep();
    assert_eq!(p.graveyard_len(), (1, 0)); // still 0: stays
    unsafe { (*(old_addr as *const AtomicI32)).store(1, Ordering::Release) };
    p.vq_sweep();
    assert_eq!(p.graveyard_len(), (0, 0));
    // a replaced entry that already reads 1 is freed at once
    let mut q = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    f(|x| x.victims.insert((0, 13), (19, 3.5)));
    q.insert_pend(19, pending_done(null_mut()));
    q.route_hook(1, &[3]);
    assert_eq!(q.graveyard_len(), (0, 0));
}

#[test]
fn replaced_arrival_is_kept_until_its_state_reads_one() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    let old = Box::new(Arrival { state: AtomicI32::new(0) });
    let old_addr = old.state.as_ptr() as usize;
    p.insert_arr(13, old); // an older flight of key 13, unfinished
    p.route_hook(1, &[3]);
    assert_eq!(p.graveyard_len(), (0, 1));
    assert_eq!(p.arr.len(), 1);
    assert_ne!(p.arr.get(13).unwrap().state.as_ptr() as usize, old_addr);
    p.vq_sweep();
    assert_eq!(p.graveyard_len(), (0, 1));
    unsafe { (*(old_addr as *const AtomicI32)).store(1, Ordering::Release) };
    p.vq_sweep();
    assert_eq!(p.graveyard_len(), (0, 0));
    let mut q = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    q.insert_arr(13, Box::new(Arrival { state: AtomicI32::new(1) }));
    q.route_hook(1, &[3]);
    assert_eq!(q.graveyard_len(), (0, 0));
}

#[test]
fn job_state_pointers_are_the_entries_and_the_queue_can_release_them() {
    let mut p = make(vq_cfg());
    add_calls(&[(1, &[3])]);
    hold(HID, 13);
    p.route_hook(1, &[3]);
    let j = job(0);
    assert_eq!(j.kind, 2);
    assert!(!is_set(j.rel_state) && !is_set(j.arr_state));
    // the worker sets rel before arr; the next sweep releases the RAM slot and drops the entry
    set1(j.rel_state);
    set1(j.arr_state);
    assert_eq!(p.arr.get(13).unwrap().state.load(Ordering::Acquire), 1);
    p.vq_sweep();
    assert_eq!(p.stats().releases, 0);
    assert!(log().contains(&Release(HID, removed_addr(13))));
}

// ---------------------------------------------------------------- helpers and layout

#[test]
fn dense_map_swap_remove_keeps_keys_and_index_consistent() {
    let mut m: DenseMap<i32> = DenseMap::new();
    for k in [5, 6, 7, 8] {
        assert!(m.insert(k, k * 10).is_none());
    }
    assert_eq!(m.insert(6, 61), Some(60));
    assert_eq!(m.keys(), &[5, 6, 7, 8]);
    assert_eq!(m.remove(5), Some(50));
    assert_eq!(m.keys(), &[8, 6, 7]); // the last key took its place
    assert_eq!((m.get(8), m.get(6), m.get(7), m.get(5)), (Some(&80), Some(&61), Some(&70), None));
    assert_eq!(m.remove(7), Some(70));
    assert_eq!(m.keys(), &[8, 6]);
    assert_eq!(m.remove(7), None);
    assert_eq!(m.len(), 2);
    assert_eq!(m.values().copied().sum::<i32>(), 141);
    for k in 0..1000 {
        m.insert(k, k);
    }
    for k in (0..1000).step_by(3) {
        m.remove(k);
    }
    assert!(m.keys().iter().all(|&k| m.get(k).is_some()));
    assert_eq!(m.len(), m.keys().len());
}

fn include_dirs() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let c = [
        "include",
        "../copyq/include",
        "../slotcache/include",
        "../routetable/include",
        "copyq-include",
        "slotcache-include",
        "routetable-include",
    ];
    c.iter().map(|d| root.join(d)).filter(|d| d.is_dir()).collect()
}

#[test]
fn struct_layout_matches_the_c_header() {
    use std::mem::{offset_of, size_of};
    macro_rules! offs {
        ($t:ty, $c:literal, [$($f:ident),*]) => { vec![ $( (format!("{}.{}", $c, stringify!($f)), offset_of!($t, $f)) ),* ] };
    }
    let job_f = ["kind", "dev", "fd", "off", "layer", "slot", "p_state", "dst", "host_src", "victim_host", "wb_state", "arr_state", "rel_state"];
    let ops_f = [
        "sc_set_next_use",
        "sc_contains",
        "sc_worth",
        "sc_acquire",
        "sc_try_acquire",
        "sc_remove",
        "sc_release",
        "table",
        "rt_calls",
        "rt_layer",
        "rt_experts",
        "rt_set_current",
        "rt_next_use",
        "enqueue",
    ];
    let cfg_f = ["experts", "expert_raw", "cards", "lookahead", "pend_cap", "vq", "vq_cap", "vq_from", "prefill", "rc_ok"];
    let st_f = [
        "pos", "planned_to", "rc_ok", "planned", "skipped", "vq_ram", "vq_drive", "vq_wb", "vq_nowb", "pf_used", "pf_waited", "vq_waited", "t_pwait", "t_vwait", "t_plan", "pending",
        "arrivals", "releases", "wb_keys",
    ];
    let mut c = String::from("#include <stdio.h>\n#include <stddef.h>\n#include \"planner.h\"\nint main(void){\n");
    for (t, n) in [("cq_job", "job"), ("sp_ops", "ops"), ("sp_config", "cfg"), ("sp_stats", "st")] {
        c += &format!("printf(\"{n}.size=%zu\\n\", sizeof({t}));\n");
    }
    for (t, n, fs) in [("cq_job", "job", &job_f[..]), ("sp_ops", "ops", &ops_f[..]), ("sp_config", "cfg", &cfg_f[..]), ("sp_stats", "st", &st_f[..])] {
        for f in fs {
            c += &format!("printf(\"{n}.{f}=%zu\\n\", offsetof({t}, {f}));\n");
        }
    }
    c += "return 0;}\n";
    let mut want: Vec<(String, usize)> =
        vec![("job.size".into(), size_of::<cq_job>()), ("ops.size".into(), size_of::<sp_ops>()), ("cfg.size".into(), size_of::<sp_config>()), ("st.size".into(), size_of::<sp_stats>())];
    want.extend(offs!(cq_job, "job", [kind, dev, fd, off, layer, slot, p_state, dst, host_src, victim_host, wb_state, arr_state, rel_state]));
    want.extend(offs!(
        sp_ops,
        "ops",
        [sc_set_next_use, sc_contains, sc_worth, sc_acquire, sc_try_acquire, sc_remove, sc_release, table, rt_calls, rt_layer, rt_experts, rt_set_current, rt_next_use, enqueue]
    ));
    want.extend(offs!(sp_config, "cfg", [experts, expert_raw, cards, lookahead, pend_cap, vq, vq_cap, vq_from, prefill, rc_ok]));
    want.extend(offs!(
        sp_stats,
        "st",
        [pos, planned_to, rc_ok, planned, skipped, vq_ram, vq_drive, vq_wb, vq_nowb, pf_used, pf_waited, vq_waited, t_pwait, t_vwait, t_plan, pending, arrivals, releases, wb_keys]
    ));

    let dir = std::env::temp_dir().join(format!("planner_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (src, exe) = (dir.join("layout.c"), dir.join("layout"));
    std::fs::write(&src, c).expect("write C source");
    let mut cmd = Command::new("cc");
    for d in include_dirs() {
        cmd.arg("-I").arg(d);
    }
    let st = cmd.arg(&src).arg("-o").arg(&exe).status().expect("a C compiler named `cc` is required for this test");
    assert!(st.success(), "C compile of planner.h failed");
    let out = Command::new(&exe).output().expect("run layout probe");
    let got: Vec<(String, usize)> =
        String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.parse().unwrap_or(usize::MAX)))).collect();
    assert_eq!(got.len(), want.len());
    for w in &want {
        assert!(got.contains(w), "layout mismatch for {}: rust {} vs C {:?}", w.0, w.1, got.iter().find(|g| g.0 == w.0));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------- the exported C functions (process-wide state: the only test that touches it)

static ABI_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn exported_functions_end_to_end() {
    let _g = ABI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    reset();
    let mut s0 = sp_stats::default();
    unsafe { crate::sp_get_stats(&mut s0) };
    assert_eq!((s0.pos, s0.planned), (0, 0)); // before sp_init: zeros
    // pools and host set BEFORE sp_init are kept
    unsafe {
        crate::sp_pool_set(0, pool_ptr(0), 5, 1);
        crate::sp_pool_set(9, pool_ptr(1), 0, 1); // outside 0..7: ignored
        crate::sp_host_set(host_ptr(), 10);
    }
    let ops = fake_ops();
    let cfg = base_cfg();
    unsafe { crate::sp_init(&ops, &cfg) };
    crate::sp_layer_set(1, -1, 0);
    crate::sp_layer_set(100, 3, 0); // outside 0..92: ignored
    add_calls(&[(1, &[3, 4]), (1, &[5])]);
    unsafe { crate::sp_route_hook(1, [3, 4].as_ptr(), 2) };
    assert_eq!(log(), vec![SetCur(0)]); // layer 1 has fd -1: planned passed over
    crate::sp_layer_set_fd(1, 11); // another thread would do this
    clear_log();
    unsafe { crate::sp_route_hook(1, [5].as_ptr(), 1) };
    assert_eq!(log(), vec![SetCur(1)]); // pos 1, planned_to 2 already: nothing left
    let mut s = sp_stats::default();
    unsafe { crate::sp_get_stats(&mut s) };
    assert_eq!((s.pos, s.planned_to, s.rc_ok, s.pending), (2, 2, 1, 0));
    // sp_host_acquire uses the pending keys as prot: re-init (keeps wiring, resets state), plan one call, then acquire
    unsafe { crate::sp_init(&ops, &cfg) };
    unsafe { crate::sp_get_stats(&mut s) };
    assert_eq!((s.pos, s.planned_to), (0, 0));
    f(|x| x.calls.clear());
    add_calls(&[(1, &[3]), (1, &[4])]);
    clear_log();
    unsafe { crate::sp_route_hook(1, [3].as_ptr(), 1) };
    assert_eq!(njobs(), 2); // two kind-0 jobs: calls 0 and 1 of the window
    clear_log();
    let (mut vk, mut vh) = (0i32, 0f32);
    let slot = unsafe { crate::sp_host_acquire(77, 2.5, &mut vk, &mut vh) };
    assert_eq!(log(), vec![Acq(HID, 77, vec![13, 14], 2.5)]);
    assert_eq!(slot as usize, slot_addr(2));
    assert_eq!(vk, -1);
    // pend_take through the ABI: the first job's read is done
    set1(job(0).p_state);
    crate::sp_pend_take(13);
    crate::sp_pend_take(13);
    crate::sp_arr_wait(13);
    unsafe { crate::sp_get_stats(&mut s) };
    assert_eq!((s.pf_used, s.pf_waited, s.pending, s.planned), (1, 0, 1, 2));
    // a mismatch through the ABI
    unsafe { crate::sp_route_hook(1, [9].as_ptr(), 1) };
    assert_eq!(*log().last().unwrap(), SetNu(false));
    // a card outside 0..7 is ignored (no handle is stored), as is a negative one
    crate::sp_layer_set(2, 5, 9);
    crate::sp_layer_set(3, 5, -1);
    assert!(crate::LAYERS.get(2).is_none() && crate::LAYERS.get(3).is_none());
    crate::sp_layer_set(2, 5, 7);
    assert_eq!(crate::LAYERS.get(2).map(|h| (h.fd, h.card)), Some((5, 7)));
    // sp_get_stats while another call is in flight (the busy flag is held): no abort, the flag stays held, the counters are returned
    let mut s1 = sp_stats::default();
    unsafe { crate::sp_get_stats(&mut s1) };
    assert!(!crate::G.busy.swap(true, std::sync::atomic::Ordering::SeqCst));
    let mut s2 = sp_stats::default();
    unsafe { crate::sp_get_stats(&mut s2) };
    assert!(crate::G.busy.load(std::sync::atomic::Ordering::SeqCst), "sp_get_stats must not release a flag it did not take");
    crate::G.busy.store(false, std::sync::atomic::Ordering::SeqCst);
    assert_eq!((s2.pos, s2.planned, s2.pending, s2.pf_used), (s1.pos, s1.planned, s1.pending, s1.pf_used));
}
