//! The few system calls the original makes directly (mmap, madvise, clock_gettime, exit) and the message / failure helpers.
//! Messages are written with one `write_all` on stderr and errors ignored, like `fprintf(stderr, ...)`; never `eprintln!` (it panics on a write error).
use std::io::Write;
use std::os::raw::{c_char, c_int, c_void};

pub const PROT_READ_WRITE: c_int = 0x1 | 0x2;
pub const MAP_PRIVATE_ANONYMOUS: c_int = 0x02 | 0x20; // MAP_PRIVATE | MAP_ANONYMOUS (Linux)
pub const MADV_HUGEPAGE: c_int = 14; // Linux
pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
pub const CLOCK_MONOTONIC: c_int = 1;

#[repr(C)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

extern "C" {
    pub fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    pub fn clock_gettime(clk: c_int, ts: *mut Timespec) -> c_int;
    pub fn exit(code: c_int) -> !;
}

/// `wall_now_()` of decode.cuh:202: `t.tv_sec + 1e-9 * t.tv_nsec` in doubles.
pub fn wall_now() -> f64 {
    let mut t = Timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `t` is a valid Timespec.
    unsafe { clock_gettime(CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as f64 + 1e-9 * t.tv_nsec as f64
}

pub fn write_stderr(msg: &[u8]) {
    let _ = std::io::stderr().write_all(msg);
}

/// The process ends here, as `exit(1)` does in the original (atexit handlers run).
pub fn exit1() -> ! {
    // SAFETY: plain C exit.
    unsafe { exit(1) }
}

/// A message of this crate that the original does not have: "poolinit: <what>\n", exit(1).
pub fn bad(what: &str) -> ! {
    write_stderr(format!("poolinit: {}\n", what).as_bytes());
    exit1()
}

/// `fprintf("%s", p)` of a C string: the bytes up to the NUL, or "(null)" for a NULL pointer (glibc).
///
/// # Safety
/// `p` must be NULL or point to a NUL-terminated string.
pub unsafe fn cstr_bytes<'a>(p: *const c_char) -> &'a [u8] {
    if p.is_null() {
        b"(null)"
    } else {
        // SAFETY: by the caller.
        unsafe { std::ffi::CStr::from_ptr(p).to_bytes() }
    }
}

/// The CHECK macro (kernels.cuh:17): on a nonzero code, "<file>:<line>: <error_string(code)>\n" and exit(1). `file` is cfg.src_file or "decode.cuh".
///
/// # Safety
/// `error_string` must return NULL or a valid C string; `file` as for `cstr_bytes`.
pub unsafe fn check(code: c_int, error_string: unsafe extern "C" fn(c_int) -> *const c_char, file: *const c_char, line: u32) {
    if code == 0 {
        return;
    }
    let mut m: Vec<u8> = Vec::with_capacity(96);
    // SAFETY: by the caller.
    unsafe {
        if file.is_null() {
            m.extend_from_slice(b"decode.cuh");
        } else {
            m.extend_from_slice(cstr_bytes(file));
        }
        m.extend_from_slice(format!(":{}: ", line).as_bytes());
        m.extend_from_slice(cstr_bytes(error_string(code)));
    }
    m.push(b'\n');
    write_stderr(&m);
    exit1()
}
