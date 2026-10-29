//! Rust implementation of include/planner.h (the supply planner of the K3GPU route cache). No dependency on the other crates: slot caches, route table and copy queue
//! are reached through the `sp_ops` table given to `sp_init`. One planner per process; every function except `sp_layer_set_fd` is for the main thread only.
#![allow(non_camel_case_types)]

pub mod abi;
mod dense;
pub mod planner;

use abi::{sc_cache, sp_config, sp_ops, sp_stats};
use planner::{fatal, LayerTable, Planner, Pool, Wiring, MAX_CARDS};
use std::cell::UnsafeCell;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicBool, Ordering};

static LAYERS: LayerTable = LayerTable::new();

struct Global {
    /// sp_pool_set / sp_host_set calls made before the first sp_init
    pre: UnsafeCell<Wiring>,
    planner: UnsafeCell<Option<Planner>>,
    busy: AtomicBool,
}

// SAFETY: the contract has one caller thread (the main thread); `busy` turns a concurrent or re-entrant call into a clean exit instead of a data race.
unsafe impl Sync for Global {}

static G: Global = Global { pre: UnsafeCell::new(Wiring::UNSET), planner: UnsafeCell::new(None), busy: AtomicBool::new(false) };

struct Busy;
impl Busy {
    fn enter() -> Busy {
        if G.busy.swap(true, Ordering::Acquire) {
            fatal("planner: concurrent or re-entrant call");
        }
        Busy
    }
}
impl Drop for Busy {
    fn drop(&mut self) {
        G.busy.store(false, Ordering::Release);
    }
}

fn with_planner<R>(name: &str, f: impl FnOnce(&mut Planner) -> R) -> R {
    let _b = Busy::enter();
    // SAFETY: exclusive by `busy`.
    match unsafe { &mut *G.planner.get() } {
        Some(p) => f(p),
        None => fatal(&format!("planner: {} before sp_init", name)),
    }
}

/// # Safety
/// `ops` and `cfg` must point to valid structs; the function pointers, `ops.table` and the caches given later must stay valid for the life of the process.
/// A second call starts a new planning state (counters, pending, arrivals, releases); layers, pools and the host cache set so far are kept. The old entries are leaked, not freed:
/// the copy queue may still write their state.
#[no_mangle]
pub unsafe extern "C" fn sp_init(ops: *const sp_ops, cfg: *const sp_config) {
    if ops.is_null() || cfg.is_null() {
        fatal("planner: sp_init(NULL)");
    }
    let _b = Busy::enter();
    // SAFETY: exclusive by `busy`; non-null checked above.
    let slot = unsafe { &mut *G.planner.get() };
    let w = match slot.take() {
        Some(old) => {
            let w = old.w;
            std::mem::forget(old);
            w
        }
        None => unsafe { *G.pre.get() },
    };
    *slot = Some(Planner::new(unsafe { *ops }, unsafe { *cfg }, &LAYERS, w));
}

#[no_mangle]
pub extern "C" fn sp_layer_set(layer: c_int, fd: c_int, card: c_int) {
    LAYERS.set(layer, fd, card);
}

#[no_mangle]
pub extern "C" fn sp_layer_set_fd(layer: c_int, fd: c_int) {
    LAYERS.set_fd(layer, fd);
}

fn set_wiring(f: impl FnOnce(&mut Wiring)) {
    let _b = Busy::enter();
    // SAFETY: exclusive by `busy`.
    match unsafe { &mut *G.planner.get() } {
        Some(p) => f(&mut p.w),
        None => f(unsafe { &mut *G.pre.get() }),
    }
}

/// # Safety
/// `cache` must stay valid for the life of the process (it is only passed back to `sp_ops`).
#[no_mangle]
pub unsafe extern "C" fn sp_pool_set(card: c_int, cache: *mut sc_cache, dev: c_int, ready: c_int) {
    if card < 0 || card as usize >= MAX_CARDS {
        return; // outside 0..7: ignored (the original indexes g_pool out of bounds)
    }
    set_wiring(|w| w.pools[card as usize] = Pool { cache, dev, ready: ready != 0 });
}

/// # Safety
/// As `sp_pool_set`.
#[no_mangle]
pub unsafe extern "C" fn sp_host_set(cache: *mut sc_cache, cap_slots: usize) {
    set_wiring(|w| w.host = planner::Host { cache, cap_slots });
}

/// # Safety
/// `uniq` must point to `n` ints (it may be NULL when `n` is 0).
#[no_mangle]
pub unsafe extern "C" fn sp_route_hook(layer: c_int, uniq: *const c_int, n: usize) {
    let u: &[i32] = if n == 0 || uniq.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(uniq, n) } };
    with_planner("sp_route_hook", |p| p.route_hook(layer, u));
}

#[no_mangle]
pub extern "C" fn sp_pend_take(key: c_int) {
    with_planner("sp_pend_take", |p| p.pend_take(key));
}

#[no_mangle]
pub extern "C" fn sp_arr_wait(key: c_int) {
    with_planner("sp_arr_wait", |p| p.arr_wait(key));
}

/// # Safety
/// `vkey` and `vheat` must be valid for writes (they are handed to `sc_acquire`).
#[no_mangle]
pub unsafe extern "C" fn sp_host_acquire(key: c_int, heat: f32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
    with_planner("sp_host_acquire", |p| p.host_acquire(key, heat, vkey, vheat))
}

/// # Safety
/// `out` must be valid for a write of `sp_stats`. Before `sp_init` it receives zeros. The engine calls this from an atexit handler; if another planner call is still in flight
/// (a copy-queue worker called exit() while the main thread spins in sp_arr_wait), the counters are read as they are, without taking the flag, and may be mid-update.
#[no_mangle]
pub unsafe extern "C" fn sp_get_stats(out: *mut sp_stats) {
    if out.is_null() {
        fatal("planner: sp_get_stats(NULL)");
    }
    let owned = !G.busy.swap(true, Ordering::Acquire);
    // SAFETY: exclusive when `owned`; otherwise a plain read of counters for a final report, see above.
    let s = match unsafe { &*G.planner.get() } {
        Some(p) => p.stats(),
        None => sp_stats::default(),
    };
    if owned {
        G.busy.store(false, Ordering::Release);
    }
    unsafe { out.write(s) };
}
