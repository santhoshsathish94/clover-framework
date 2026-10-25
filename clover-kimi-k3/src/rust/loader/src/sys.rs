//! The few libc calls the job makes, behind a trait so the job's logic can be tested without files, plus the fatal exit. No dependencies: the functions are declared here.
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};

extern "C" {
    fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize; // off_t is 64 bits on the supported targets (LP64 Linux)
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn exit(code: c_int) -> !;
}

pub const O_RDONLY: c_int = 0;
#[cfg(target_arch = "x86_64")]
pub const O_DIRECT: c_int = 0o40000;
#[cfg(target_arch = "aarch64")]
pub const O_DIRECT: c_int = 0o200000;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("O_DIRECT is only known for x86_64 and aarch64 Linux");

/// fprintf(stderr, ...) followed by exit(1): the message goes out first, then the C library's exit(), so atexit handlers run and stdio is flushed (as the original's exit(1)).
pub fn fatal_msg(msg: &[u8]) -> ! {
    let mut done = 0;
    while done < msg.len() {
        // SAFETY: fd 2 and a live buffer of msg.len() - done bytes.
        let n = unsafe { write(2, msg[done..].as_ptr() as *const c_void, msg.len() - done) };
        if n <= 0 {
            break; // fprintf ignores a failed write too
        }
        done += n as usize;
    }
    // SAFETY: libc exit; never returns.
    unsafe { exit(1) }
}

pub trait Sys: Send + Sync + 'static {
    fn open(&self, path: &CStr, flags: c_int) -> c_int;
    /// One pread(2): bytes read (0 = end of file) or -1.
    fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> isize;
}

pub struct RealSys;

impl Sys for RealSys {
    fn open(&self, path: &CStr, flags: c_int) -> c_int {
        // SAFETY: `path` is NUL-terminated; open takes no mode argument without O_CREAT.
        unsafe { open(path.as_ptr(), flags) }
    }
    fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> isize {
        // SAFETY: the job passes a buffer it was told (header) is writable for `len` bytes: slab[layer] + r * expert_raw + got inside the 4 * expert_raw block host_alloc made.
        unsafe { pread(fd, buf as *mut c_void, len, off) }
    }
}
