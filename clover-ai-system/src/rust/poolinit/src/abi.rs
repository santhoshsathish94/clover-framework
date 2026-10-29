//! `#[repr(C)]` mirrors of include/poolinit.h, in the header's field order. LP64 Linux (`size_t` = usize, `unsigned long` = u64).
#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_ulong, c_void};

pub const PI_CHUNK_SLOTS: usize = 32;
pub const PI_MAX_READERS: usize = 32;
pub const PI_HOST_ALLOC_DEFAULT: c_uint = 0x00;
pub const PI_HOST_REGISTER_PORTABLE: c_uint = 0x01;
pub const PI_STREAM_NON_BLOCKING: c_uint = 0x01;
pub const PI_EVENT_DISABLE_TIMING: c_uint = 0x02;

/// Opaque `sc_cache` of slotcache.h: only ever passed back to `sc_add_spare`.
#[repr(C)]
pub struct sc_cache {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pi_ops {
    pub set_device: Option<unsafe extern "C" fn(device: c_int) -> c_int>,
    pub mem_get_info: Option<unsafe extern "C" fn(free_bytes: *mut usize, total_bytes: *mut usize) -> c_int>,
    pub device_malloc: Option<unsafe extern "C" fn(ptr: *mut *mut c_void, bytes: usize) -> c_int>,
    pub host_register: Option<unsafe extern "C" fn(ptr: *mut c_void, bytes: usize, flags: c_uint) -> c_int>,
    pub host_alloc: Option<unsafe extern "C" fn(ptr: *mut *mut c_void, bytes: usize, flags: c_uint) -> c_int>,
    pub stream_create: Option<unsafe extern "C" fn(stream: *mut *mut c_void, flags: c_uint) -> c_int>,
    pub event_create: Option<unsafe extern "C" fn(event: *mut *mut c_void, flags: c_uint) -> c_int>,
    pub error_string: Option<unsafe extern "C" fn(code: c_int) -> *const c_char>,
    pub sc_add_spare: Option<unsafe extern "C" fn(cache: *mut sc_cache, slot: *mut u8)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct pi_expert_stats {
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
pub struct pi_host_cfg {
    pub bytes: usize,
    pub expert_raw: usize,
    pub cache: *mut sc_cache,
    pub src_file: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct pi_host_out {
    pub cap_slots: usize,
    pub ready: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pi_pool_cfg {
    pub device: c_int,
    pub reserve_bytes: usize,
    pub nreaders: c_int,
    pub expert_raw: usize,
    pub experts: c_int,
    pub cache: *mut sc_cache,
    pub src_file: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pi_pool_out {
    pub dev: c_int,
    pub slots: c_int,
    pub readers: c_int,
    pub ready: c_int,
    pub base: *mut u8,
    pub copy: *mut c_void,
    pub rstream: [*mut c_void; PI_MAX_READERS],
    pub bounce: [*mut u8; PI_MAX_READERS],
    pub written_back: *mut c_void,
    pub d_eptr: *const *const u8,
    pub st: pi_expert_stats,
}
