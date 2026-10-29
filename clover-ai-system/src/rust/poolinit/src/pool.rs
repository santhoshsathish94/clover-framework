//! CardPool::init (decode.cuh:451-470 at 4d8d891): slot count from the free memory, the slab, the spares, the streams, the bounce slots, the event, the pointer table.
use crate::abi::{pi_ops, pi_pool_cfg, pi_pool_out, PI_EVENT_DISABLE_TIMING, PI_HOST_ALLOC_DEFAULT, PI_MAX_READERS, PI_STREAM_NON_BLOCKING};
use crate::sys::{self, check, exit1, write_stderr};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr::addr_of_mut;

// CHECK sites of decode.cuh@4d8d891 (ref/extract.py asserts the line numbers of all ten CHECK calls of the extracted block, host and pool)
const LINE_SET_DEVICE: u32 = 454;
const LINE_MEM_GET_INFO: u32 = 455;
const LINE_MALLOC_SLAB: u32 = 459;
const LINE_STREAM_COPY: u32 = 461;
const LINE_STREAM_READER: u32 = 463;
const LINE_HOST_ALLOC: u32 = 464;
const LINE_EVENT: u32 = 466;
const LINE_MALLOC_EPTR: u32 = 467;
/// The smallest pool the original accepts.
pub const MIN_SLOTS: c_int = 64;

/// `usable = f > reserve ? f - reserve : 0; slots = (int)(usable / EXPERT_RAW)`; the cast wraps like C's.
pub fn slots_for(free_bytes: usize, reserve_bytes: usize, expert_raw: usize) -> c_int {
    let usable = if free_bytes > reserve_bytes { free_bytes - reserve_bytes } else { 0 };
    (usable / expert_raw) as c_int
}

/// The refusal text of the original: "card %d: only %d expert slots fit after the reserve\n".
pub fn refusal_message(dev: c_int, slots: c_int) -> String {
    format!("card {}: only {} expert slots fit after the reserve\n", dev, slots)
}

/// # Safety
/// `ops` must hold valid functions (checked non-NULL by the caller); `out` must be valid for writes. The ops get the address of the out field itself
/// (as the original passes `&member`), and are called from this thread only.
pub unsafe fn pool_init(ops: &pi_ops, cfg: &pi_pool_cfg, out: *mut pi_pool_out) {
    let raw = cfg.expert_raw;
    let file: *const c_char = cfg.src_file;
    let (set_device, mem_get_info, device_malloc, host_alloc, stream_create, event_create, error_string, add_spare) = match (
        ops.set_device,
        ops.mem_get_info,
        ops.device_malloc,
        ops.host_alloc,
        ops.stream_create,
        ops.event_create,
        ops.error_string,
        ops.sc_add_spare,
    ) {
        (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f), Some(g), Some(h)) => (a, b, c, d, e, f, g, h),
        _ => sys::bad("ops incomplete"),
    };
    // SAFETY: `out` valid by the caller; every pointer given to an op is the address of a field of `*out` or of a local.
    unsafe {
        (*out).dev = cfg.device;
        (*out).readers = std::cmp::min(cfg.nreaders, PI_MAX_READERS as c_int);
        check(set_device(cfg.device), error_string, file, LINE_SET_DEVICE);
        let (mut f, mut t) = (0usize, 0usize);
        check(mem_get_info(&mut f, &mut t), error_string, file, LINE_MEM_GET_INFO);
        let slots = slots_for(f, cfg.reserve_bytes, raw);
        (*out).slots = slots;
        if slots < MIN_SLOTS {
            write_stderr(refusal_message(cfg.device, slots).as_bytes());
            exit1();
        }
        check(device_malloc(addr_of_mut!((*out).base) as *mut *mut c_void, (slots as usize).wrapping_mul(raw)), error_string, file, LINE_MALLOC_SLAB);
        let base = (*out).base;
        for i in (0..slots).rev() {
            add_spare(cfg.cache, base.wrapping_add((i as usize).wrapping_mul(raw)));
        }
        check(stream_create(addr_of_mut!((*out).copy), PI_STREAM_NON_BLOCKING), error_string, file, LINE_STREAM_COPY);
        let readers = (*out).readers;
        for i in 0..readers {
            let i = i as usize;
            check(stream_create(addr_of_mut!((*out).rstream[i]), PI_STREAM_NON_BLOCKING), error_string, file, LINE_STREAM_READER);
            check(host_alloc(addr_of_mut!((*out).bounce[i]) as *mut *mut c_void, raw, PI_HOST_ALLOC_DEFAULT), error_string, file, LINE_HOST_ALLOC);
        }
        check(event_create(addr_of_mut!((*out).written_back), PI_EVENT_DISABLE_TIMING), error_string, file, LINE_EVENT);
        // `EXPERTS * sizeof(void *)`: an int converted to size_t (sign-extended), times 8
        check(
            device_malloc(addr_of_mut!((*out).d_eptr) as *mut *mut c_void, (cfg.experts as isize as usize).wrapping_mul(std::mem::size_of::<*const u8>())),
            error_string,
            file,
            LINE_MALLOC_EPTR,
        );
        (*out).ready = 1;
    }
}
