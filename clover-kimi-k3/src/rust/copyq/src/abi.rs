//! `#[repr(C)]` mirror of include/copyq.h, field for field.
use std::os::raw::{c_int, c_uchar};

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct cq_job {
    pub kind: c_int,
    pub dev: c_int,
    pub fd: c_int,
    pub off: i64, // off_t on LP64 Linux
    pub layer: c_int,
    pub slot: *mut c_uchar,
    pub p_state: *mut c_int,
    pub dst: *mut c_uchar,
    pub host_src: *const c_uchar,
    pub victim_host: *mut c_uchar,
    pub wb_state: *mut c_int,
    pub arr_state: *mut c_int,
    pub rel_state: *mut c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct cq_config {
    pub expert_raw: usize,
    pub readers: c_int,
    pub cards: c_int,
    pub prefill: c_int,
    pub slab: *const *const c_uchar,
    pub act_hold: *const c_int,
}
