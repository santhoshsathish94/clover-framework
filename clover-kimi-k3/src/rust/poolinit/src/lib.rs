//! Rust implementation of include/poolinit.h: the start-up of the pinned RAM cache (`pi_host_init`) and of one card's expert pool (`pi_pool_init`).
//! No state between calls, no dependency on the other crates: CUDA and the slot caches are reached through the `pi_ops` table.
#![allow(non_camel_case_types)]

pub mod abi;
pub mod host;
pub mod pool;
pub mod sys;
#[cfg(test)]
mod tests;

use abi::{pi_host_cfg, pi_host_out, pi_ops, pi_pool_cfg, pi_pool_out};
use sys::bad;

macro_rules! need_ptr {
    ($p:expr, $name:expr) => {
        if $p.is_null() {
            bad(concat!($name, " is NULL"));
        }
    };
}
macro_rules! need_fn {
    ($ops:expr, $f:ident) => {
        if $ops.$f.is_none() {
            bad(concat!("ops->", stringify!($f), " is NULL"));
        }
    };
}

/// # Safety
/// `ops`, `cfg` and `out` must be NULL (refused with a message) or valid; the function pointers in `ops` must stay valid for the call and be called from the caller's thread only;
/// `cfg.cache` is only handed to `ops.sc_add_spare`; `cfg.src_file` is NULL or a C string.
#[no_mangle]
pub unsafe extern "C" fn pi_host_init(ops: *const pi_ops, cfg: *const pi_host_cfg, out: *mut pi_host_out) {
    need_ptr!(ops, "ops");
    need_ptr!(cfg, "cfg");
    need_ptr!(out, "out");
    // SAFETY: non-null checked above; copied once, as the reference does.
    let (ops, cfg) = unsafe { (*ops, *cfg) };
    need_fn!(ops, set_device);
    need_fn!(ops, host_register);
    need_fn!(ops, error_string);
    need_fn!(ops, sc_add_spare);
    if cfg.expert_raw == 0 {
        bad("expert_raw is 0");
    }
    // SAFETY: by the contract.
    unsafe { host::host_init(&ops, &cfg, out) }
}

/// # Safety
/// As `pi_host_init`.
#[no_mangle]
pub unsafe extern "C" fn pi_pool_init(ops: *const pi_ops, cfg: *const pi_pool_cfg, out: *mut pi_pool_out) {
    need_ptr!(ops, "ops");
    need_ptr!(cfg, "cfg");
    need_ptr!(out, "out");
    // SAFETY: non-null checked above.
    let (ops, cfg) = unsafe { (*ops, *cfg) };
    need_fn!(ops, set_device);
    need_fn!(ops, mem_get_info);
    need_fn!(ops, device_malloc);
    need_fn!(ops, host_alloc);
    need_fn!(ops, stream_create);
    need_fn!(ops, event_create);
    need_fn!(ops, error_string);
    need_fn!(ops, sc_add_spare);
    if cfg.expert_raw == 0 {
        bad("expert_raw is 0");
    }
    // SAFETY: by the contract.
    unsafe { pool::pool_init(&ops, &cfg, out) }
}
