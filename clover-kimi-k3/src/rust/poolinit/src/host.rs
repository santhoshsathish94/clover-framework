//! HostCache::init (decode.cuh:403-435 at 4d8d891): map the chunks with 16 workers, register them, add the spares, write the summary line.
use crate::abi::{pi_host_cfg, pi_host_out, pi_ops, PI_CHUNK_SLOTS, PI_HOST_REGISTER_PORTABLE};
use crate::sys::{self, check, exit1, madvise, mmap, write_stderr};
use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

// CHECK sites of decode.cuh@4d8d891 (extract.py asserts these line numbers)
const LINE_SET_DEVICE: u32 = 428;
const LINE_REGISTER: u32 = 429;
/// `std::thread` workers besides the calling thread (`for (int t = 1; t < 16; t++)`).
const HELPERS: usize = 15;

/// `cap_slots = bytes / EXPERT_RAW`, `chunk_bytes = (size_t)CHUNK_SLOTS * EXPERT_RAW`, `n_chunks = cap_slots / CHUNK_SLOTS` (size_t arithmetic, wrapping).
pub fn plan(bytes: usize, expert_raw: usize) -> (usize, usize, usize) {
    let cap_slots = bytes / expert_raw;
    (cap_slots, PI_CHUNK_SLOTS.wrapping_mul(expert_raw), cap_slots / PI_CHUNK_SLOTS)
}

/// The summary line, with the C `printf` conversions `%zu`, `%.1f`, `%.1f`.
pub fn summary_line(cap_slots: usize, expert_raw: usize, seconds: f64) -> String {
    format!(
        "K3GPU: host expert cache of {} slots ({:.1} GB) pinned in {:.1} s\n",
        cap_slots,
        cap_slots as f64 * (expert_raw as f64) / 1e9,
        seconds
    )
}

/// The ops are called from this thread only; the helpers do mmap, madvise and the zeroing.
///
/// # Safety
/// `ops` must hold valid functions (checked non-NULL by the caller), `out` must be valid for writes, `file` is cfg.src_file.
pub unsafe fn host_init(ops: &pi_ops, cfg: &pi_host_cfg, out: *mut pi_host_out) {
    let raw = cfg.expert_raw;
    let file: *const c_char = cfg.src_file;
    let (set_device, host_register, error_string, add_spare) = match (ops.set_device, ops.host_register, ops.error_string, ops.sc_add_spare) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => sys::bad("ops incomplete"),
    };
    let (cap0, chunk_bytes, n_chunks) = plan(cfg.bytes, raw);
    // SAFETY: `out` valid by the caller; stored now, as the original assigns the members first.
    unsafe {
        (*out).cap_slots = cap0;
        (*out).ready = 1;
    }
    let base: Vec<AtomicUsize> = (0..n_chunks).map(|_| AtomicUsize::new(0)).collect();
    let next = AtomicUsize::new(0);
    let t0 = sys::wall_now();
    let touch = || loop {
        let i = next.fetch_add(1, Ordering::SeqCst);
        if i >= n_chunks {
            return;
        }
        // SAFETY: anonymous private mapping of `chunk_bytes`; only written inside the mapping.
        unsafe {
            let p = mmap(std::ptr::null_mut(), chunk_bytes, sys::PROT_READ_WRITE, sys::MAP_PRIVATE_ANONYMOUS, -1, 0);
            if p == sys::MAP_FAILED {
                write_stderr(format!("host expert cache: cannot map {} bytes\n", chunk_bytes).as_bytes());
                exit1();
            }
            madvise(p, chunk_bytes, sys::MADV_HUGEPAGE);
            std::ptr::write_bytes(p as *mut u8, 0, chunk_bytes);
            base[i].store(p as usize, Ordering::SeqCst);
        }
    };
    std::thread::scope(|s| {
        for _ in 0..HELPERS {
            if std::thread::Builder::new().spawn_scoped(s, &touch).is_err() {
                write_stderr(b"poolinit: cannot start a helper thread\n");
                std::process::abort();
            }
        }
        touch();
    });
    // SAFETY: ops are valid functions; the pointers handed to them are mappings made above.
    unsafe {
        check(set_device(0), error_string, file, LINE_SET_DEVICE);
        for b in base.iter() {
            check(host_register(b.load(Ordering::SeqCst) as *mut c_void, chunk_bytes, PI_HOST_REGISTER_PORTABLE), error_string, file, LINE_REGISTER);
        }
        for b in base.iter().rev() {
            let p = b.load(Ordering::SeqCst) as *mut u8;
            for s in (0..PI_CHUNK_SLOTS).rev() {
                add_spare(cfg.cache, p.wrapping_add(s.wrapping_mul(raw)));
            }
        }
    }
    let cap = n_chunks.wrapping_mul(PI_CHUNK_SLOTS);
    // SAFETY: `out` valid by the caller.
    unsafe { (*out).cap_slots = cap };
    write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());
}
