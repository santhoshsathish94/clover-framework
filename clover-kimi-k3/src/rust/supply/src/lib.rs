//! Rust implementation of include/supply.h (the card supply: pool_ensure of the K3GPU decode engine). No dependency on the other crates: the slot caches, the planner and CUDA are reached
//! through the `sy_ops` table given to `sy_init`. One supply per process; `sy_init` and `sy_ensure` are for the main thread only (the reader threads it starts call only the thread-safe ops).
#![allow(non_camel_case_types)]

pub mod abi;
pub mod ensure;

use abi::{sy_config, sy_host, sy_ops, sy_pool};
use ensure::{fatal, Supply, WB_SKIPPED};
use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_ulong};
use std::sync::atomic::{AtomicBool, Ordering};

struct Global {
    supply: UnsafeCell<Option<Supply>>,
    busy: AtomicBool,
}

// SAFETY: the contract has one caller thread; `busy` turns a concurrent or re-entrant call into a clean exit instead of a data race.
unsafe impl Sync for Global {}

static G: Global = Global { supply: UnsafeCell::new(None), busy: AtomicBool::new(false) };

struct Busy;
impl Busy {
    fn enter() -> Busy {
        if G.busy.swap(true, Ordering::Acquire) {
            fatal("supply: concurrent or re-entrant call\n");
        }
        Busy
    }
}
impl Drop for Busy {
    fn drop(&mut self) {
        G.busy.store(false, Ordering::Release);
    }
}

/// # Safety
/// `ops` and `cfg` must point to valid structs; the function pointers and `cfg.slab` must stay valid for the life of the process. A second call replaces both and resets the g_wb_skipped counter.
#[no_mangle]
pub unsafe extern "C" fn sy_init(ops: *const sy_ops, cfg: *const sy_config) {
    if ops.is_null() || cfg.is_null() {
        fatal("supply: sy_init(NULL)\n");
    }
    let _b = Busy::enter();
    // SAFETY: exclusive by `busy`; non-null checked above.
    *G.supply.get() = Some(Supply::new(*ops, *cfg));
}

/// # Safety
/// As supply.h: `pool`, `host` valid; `experts` points to `n` ints (it may be NULL when `n` is 0), each in 0..cfg.experts; `counts` and `pool.h_eptr` have cfg.experts entries; the caches, streams, event,
/// bounce slots (one per reader thread), slab table and `fd` stay valid while the call runs.
#[no_mangle]
pub unsafe extern "C" fn sy_ensure(pool: *mut sy_pool, host: *mut sy_host, layer: c_int, fd: c_int, experts: *const c_int, n: usize, counts: *const c_int, readers: c_int) {
    let _b = Busy::enter();
    // SAFETY: exclusive by `busy`.
    let s = match &*G.supply.get() {
        Some(s) => *s,
        None => fatal("supply: sy_ensure before sy_init\n"),
    };
    let e: &[c_int] = if n == 0 || experts.is_null() { &[] } else { std::slice::from_raw_parts(experts, n) };
    s.ensure(pool, host, layer, fd, e, counts, readers);
}

#[no_mangle]
pub extern "C" fn sy_wb_skipped() -> c_ulong {
    WB_SKIPPED.load(Ordering::SeqCst) as c_ulong
}

#[cfg(test)]
mod tests;
