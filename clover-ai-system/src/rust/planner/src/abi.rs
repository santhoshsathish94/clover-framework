//! C types of include/planner.h and of the three headers it includes (only the parts the planner touches). Field order and types are the headers'.
#![allow(non_camel_case_types)]

use std::os::raw::{c_int, c_long, c_ulong};

#[repr(C)]
pub struct sc_cache {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct rt_table {
    _opaque: [u8; 0],
}

pub type sc_next_use_fn = Option<extern "C" fn(c_int) -> c_int>;

/// copyq.h `cq_job`. `off` is `off_t` (64-bit on LP64 Linux, the only target).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct cq_job {
    pub kind: c_int,
    pub dev: c_int,
    pub fd: c_int,
    pub off: i64,
    pub layer: c_int,
    pub slot: *mut u8,
    pub p_state: *mut c_int,
    pub dst: *mut u8,
    pub host_src: *const u8,
    pub victim_host: *mut u8,
    pub wb_state: *mut c_int,
    pub arr_state: *mut c_int,
    pub rel_state: *mut c_int,
}

impl cq_job {
    pub const ZERO: cq_job = cq_job {
        kind: 0,
        dev: 0,
        fd: 0,
        off: 0,
        layer: 0,
        slot: std::ptr::null_mut(),
        p_state: std::ptr::null_mut(),
        dst: std::ptr::null_mut(),
        host_src: std::ptr::null(),
        victim_host: std::ptr::null_mut(),
        wb_state: std::ptr::null_mut(),
        arr_state: std::ptr::null_mut(),
        rel_state: std::ptr::null_mut(),
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sp_ops {
    pub sc_set_next_use: unsafe extern "C" fn(sc_next_use_fn),
    pub sc_contains: unsafe extern "C" fn(*const sc_cache, c_int) -> c_int,
    pub sc_worth: unsafe extern "C" fn(*const sc_cache, c_int) -> c_int,
    pub sc_acquire: unsafe extern "C" fn(*mut sc_cache, c_int, *const c_int, usize, f32, *mut c_int, *mut f32) -> *mut u8,
    pub sc_try_acquire: unsafe extern "C" fn(*mut sc_cache, c_int, c_int, *const c_int, usize, f32, *mut c_int, *mut f32) -> *mut u8,
    pub sc_remove: unsafe extern "C" fn(*mut sc_cache, c_int) -> *mut u8,
    pub sc_release: unsafe extern "C" fn(*mut sc_cache, *mut u8),
    pub table: *mut rt_table,
    pub rt_calls: unsafe extern "C" fn(*const rt_table) -> usize,
    pub rt_layer: unsafe extern "C" fn(*const rt_table, usize) -> c_int,
    pub rt_experts: unsafe extern "C" fn(*const rt_table, usize, *mut usize) -> *const c_int,
    pub rt_set_current: unsafe extern "C" fn(*mut rt_table, c_long),
    pub rt_next_use: unsafe extern "C" fn(*mut rt_table, c_int) -> c_int,
    pub enqueue: unsafe extern "C" fn(*const cq_job),
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sp_config {
    pub experts: c_int,
    pub expert_raw: usize,
    pub cards: c_int,
    pub lookahead: c_int,
    pub pend_cap: usize,
    pub vq: c_int,
    pub vq_cap: usize,
    pub vq_from: c_long,
    pub prefill: c_int,
    pub rc_ok: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct sp_stats {
    pub pos: usize,
    pub planned_to: usize,
    pub rc_ok: c_int,
    pub planned: c_ulong,
    pub skipped: c_ulong,
    pub vq_ram: c_ulong,
    pub vq_drive: c_ulong,
    pub vq_wb: c_ulong,
    pub vq_nowb: c_ulong,
    pub pf_used: c_ulong,
    pub pf_waited: c_ulong,
    pub vq_waited: c_ulong,
    pub t_pwait: f64,
    pub t_vwait: f64,
    pub t_plan: f64,
    pub pending: usize,
    pub arrivals: usize,
    pub releases: usize,
    pub wb_keys: usize,
}
