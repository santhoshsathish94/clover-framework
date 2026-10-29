//! `#[repr(C)]` mirrors of the structs in include/loader.h (field order and types as in the header; checked against a real `cc` compile of the header by a unit test).
#![allow(non_camel_case_types, non_snake_case)]
use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type cb_card = unsafe extern "C" fn(*mut c_void, c_int);
pub type cb_layer = unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char, c_int);
pub type cb_state = unsafe extern "C" fn(*mut c_void, *mut c_void, c_uint);
pub type cb_alloc = unsafe extern "C" fn(*mut c_void, *mut *mut u8, usize);
pub type job_fn = unsafe extern "C" fn(*mut c_void);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ld_callbacks {
    pub ctx: *mut c_void,
    pub set_device: Option<cb_card>,
    pub ensure_work: Option<cb_card>,
    pub load_trunk: Option<cb_layer>,
    pub load_expert_tables: Option<cb_layer>,
    pub alloc_state: Option<cb_state>,
    pub host_alloc: Option<cb_alloc>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ld_config {
    pub cb: ld_callbacks,
    pub expert_raw: usize,
    pub prefill: c_int,
    pub slab: *mut *mut u8,
    pub handles: *mut *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ld_layer {
    pub handle: *mut c_void,
    pub dataset: *const c_char,
    pub layer: c_uint,
    pub k: c_uint,
    pub positions: c_uint,
    pub card: c_int,
    pub h_L: *mut c_int,
    pub h_card: *mut c_int,
    pub h_k_in: *mut c_int,
    pub h_fd: *mut c_int,
}
