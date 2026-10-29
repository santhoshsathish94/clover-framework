//! C types of include/supply.h (and of slotcache.h's opaque `sc_cache`). Field order and types are the header's.
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_ulong, c_void};

pub const SY_MAX_READERS: usize = 32;

#[repr(C)]
pub struct sc_cache {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sy_ops {
    pub device_sync: unsafe extern "C" fn() -> c_int,
    pub stream_sync: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub set_device: unsafe extern "C" fn(c_int) -> c_int,
    pub memcpy_d2h_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> c_int,
    pub memcpy_h2d_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, *mut c_void) -> c_int,
    pub event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    pub stream_wait_event: unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint) -> c_int,
    pub check: unsafe extern "C" fn(c_int, *const c_char),
    pub sc_find: unsafe extern "C" fn(*mut sc_cache, c_int) -> *mut u8,
    pub sc_touch: unsafe extern "C" fn(*mut sc_cache, c_int, f32),
    pub sc_worth: unsafe extern "C" fn(*const sc_cache, c_int) -> c_int,
    pub sc_acquire: unsafe extern "C" fn(*mut sc_cache, c_int, *const c_int, usize, f32, *mut c_int, *mut f32) -> *mut u8,
    pub sc_remove: unsafe extern "C" fn(*mut sc_cache, c_int) -> *mut u8,
    pub sc_release: unsafe extern "C" fn(*mut sc_cache, *mut u8),
    pub pend_take: unsafe extern "C" fn(c_int),
    pub arr_wait: unsafe extern "C" fn(c_int),
    pub host_acquire: unsafe extern "C" fn(c_int, f32, *mut c_int, *mut f32) -> *mut u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sy_config {
    pub experts: c_int,
    pub expert_raw: usize,
    pub vq: c_int,
    pub prefill: c_int,
    pub chunks: c_int,
    pub slab: *const *const u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct sy_stats {
    pub needed: c_ulong,
    pub hit_card: c_ulong,
    pub hit_host: c_ulong,
    pub from_drive: c_ulong,
    pub written_back: c_ulong,
    pub t_read: f64,
    pub t_copy: f64,
    pub t_ensure: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sy_pool {
    pub dev: c_int,
    pub readers: c_int,
    pub cache: *mut sc_cache,
    pub copy: *mut c_void,
    pub rstream: [*mut c_void; SY_MAX_READERS],
    pub bounce: [*mut u8; SY_MAX_READERS],
    pub written_back: *mut c_void,
    pub h_eptr: *mut *const u8,
    pub st: sy_stats,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct sy_host {
    pub cache: *mut sc_cache,
    pub cap_slots: usize,
}
