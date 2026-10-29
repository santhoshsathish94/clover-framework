//! The body of an `ld_open_layer` job: the lambda passed to `loader_submit` by `open_layer` (backend-gpu.cu 580-608 at 4d8d891), with the CUDA / model calls as callbacks.
//! `run` returns the text of a fatal message instead of exiting, so every fatal path can be tested; the caller prints it and exits.
use crate::abi::{cb_alloc, cb_card, cb_layer, cb_state};
use crate::sys::{Sys, O_DIRECT, O_RDONLY};
use std::ffi::CString;
use std::os::raw::{c_int, c_void};

/// The callbacks of loader.h, all resolved (a missing required one is rejected by `ld_init`). `host_alloc` is only required with prefill.
#[derive(Clone, Copy)]
pub struct Cb {
    pub ctx: *mut c_void,
    pub set_device: cb_card,
    pub ensure_work: cb_card,
    pub load_trunk: cb_layer,
    pub load_expert_tables: cb_layer,
    pub alloc_state: cb_state,
    pub host_alloc: Option<cb_alloc>,
}
// SAFETY: the header says the callbacks are called from loader threads and must be thread-safe, and that ctx stays valid for the life of the process.
unsafe impl Send for Cb {}
unsafe impl Sync for Cb {}

pub struct LayerJob {
    pub cb: Cb,
    pub handle: *mut c_void,
    pub ds: CString,
    pub layer: u32,
    pub positions: u32,
    pub card: c_int,
    pub h_fd: *mut c_int,
    pub expert_raw: usize,
    pub prefill: bool,
    pub slab: *mut *mut u8,
}
// SAFETY: the raw pointers are the caller's (header: valid for the life of the process; the job is the only writer of *h_fd and slab[layer] until ld_wait returns).
unsafe impl Send for LayerJob {}

/// snprintf(path, 4096, "%s/expert-%u/experts.direct", ds, L): the result is cut to 4095 bytes.
pub fn expert_path(ds: &[u8], layer: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(ds.len() + 32);
    p.extend_from_slice(ds);
    p.extend_from_slice(format!("/expert-{}/experts.direct", layer).as_bytes());
    p.truncate(4095);
    p
}

fn set_fd(p: *mut c_int, v: c_int) {
    if !p.is_null() {
        // SAFETY: header: h_fd is the address of the Handle's int, valid for the life of the process; this job is its only writer until ld_wait returns.
        unsafe { *p = v }
    }
}

/// Steps 1-7 of the header, in order. Err(message) = what the original prints before exit(1).
pub fn run<S: Sys>(sys: &S, j: &LayerJob) -> Result<(), Vec<u8>> {
    let cb = &j.cb;
    let dataset = j.ds.as_ptr();
    // SAFETY (all callback calls): header: ctx and the callbacks are valid for the life of the process; `handle` is the caller's pointer; `dataset` is NUL-terminated and lives as long as `j`.
    unsafe {
        (cb.set_device)(cb.ctx, j.card);
        (cb.ensure_work)(cb.ctx, j.card);
        (cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer as c_int);
        if j.layer >= 1 {
            (cb.load_expert_tables)(cb.ctx, j.handle, dataset, j.layer as c_int);
        }
        (cb.alloc_state)(cb.ctx, j.handle, j.positions);
    }
    if j.layer < 1 {
        return Ok(());
    }
    let path = expert_path(j.ds.as_bytes(), j.layer);
    // No interior NUL: `ds` is a CString and the path only appends text to it.
    let cpath = match CString::new(path.clone()) {
        Ok(c) => c,
        Err(_) => return Err(b"loader: path contains a NUL\n".to_vec()),
    };
    let mut fd = sys.open(&cpath, O_RDONLY | O_DIRECT);
    set_fd(j.h_fd, fd);
    if fd < 0 {
        fd = sys.open(&cpath, O_RDONLY);
        set_fd(j.h_fd, fd);
    }
    if fd < 0 {
        let mut m = b"cannot open ".to_vec();
        m.extend_from_slice(&path);
        m.push(b'\n');
        return Err(m);
    }
    if j.prefill {
        let raw = j.expert_raw;
        let layer = j.layer as usize;
        let bytes = 4usize.wrapping_mul(raw);
        match j.cb.host_alloc {
            // SAFETY: slab is the caller's table of 93 pointers and layer <= 92 (checked by ld_open_layer); host_alloc stores the new block into slab[layer].
            Some(f) => unsafe { f(cb.ctx, j.slab.add(layer), bytes) },
            None => return Err(b"loader: host_alloc is NULL\n".to_vec()),
        }
        for r in 0..4usize {
            let mut got = 0usize;
            while got < raw {
                let off = (r as i64).wrapping_mul(raw as i64).wrapping_add(got as i64);
                // SAFETY: slab[layer] was just filled by host_alloc; r * raw + got < 4 * raw is inside the block.
                let buf = unsafe { (*j.slab.add(layer)).wrapping_add(r.wrapping_mul(raw)).wrapping_add(got) };
                let rd = sys.pread(fd, buf, raw - got, off);
                if rd <= 0 {
                    return Err(format!("prefill: cannot read a record of layer {}\n", j.layer).into_bytes());
                }
                got += rd as usize;
            }
        }
    }
    Ok(())
}
