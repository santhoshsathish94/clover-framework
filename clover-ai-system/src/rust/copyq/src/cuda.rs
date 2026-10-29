//! Real backend: CUDA driver API through cudarc's raw `result`/`sys` layer plus pread(2). The safe `CudaSlice`/`CudaStream` types are not used, so cudarc's
//! event tracking never sees these cross-context raw pointers.
use crate::engine::{Backend, MAX_CARDS};
use cudarc::driver::result::{self, stream::StreamKind};
use cudarc::driver::{sys, CudaContext};
use std::os::raw::{c_int, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

extern "C" {
    fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize; // glibc, LP64: off_t = i64
}

pub struct CudaBackend {
    ctxs: Mutex<[Option<Arc<CudaContext>>; MAX_CARDS]>,
}

pub struct CudaWs {
    ctx: [Option<Arc<CudaContext>>; MAX_CARDS],
    stream: [sys::CUstream; MAX_CARDS],
    bounce: *mut u8,
}

impl CudaBackend {
    pub fn new() -> Self {
        CudaBackend { ctxs: Mutex::new(std::array::from_fn(|_| None)) }
    }
    fn context(&self, dev: usize) -> Result<Arc<CudaContext>, String> {
        let mut g = self.ctxs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = &g[dev] {
            return Ok(Arc::clone(c));
        }
        let c = CudaContext::new(dev).map_err(|e| e.to_string())?; // retains the primary context, the one the C++ runtime's cudaMalloc uses
        g[dev] = Some(Arc::clone(&c));
        Ok(c)
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CudaBackend {
    type Ws = CudaWs;

    fn new_ws(&self) -> CudaWs {
        CudaWs { ctx: std::array::from_fn(|_| None), stream: [ptr::null_mut(); MAX_CARDS], bounce: ptr::null_mut() }
    }

    fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> Result<usize, i32> {
        // SAFETY: the engine passes a writable buffer of at least `len` bytes (slot, or the bounce buffer of EXPERT_RAW bytes).
        let r = unsafe { pread(fd, buf as *mut c_void, len, off) };
        if r < 0 { Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0)) } else { Ok(r as usize) }
    }

    fn set_device(&self, ws: &mut CudaWs, dev: usize) -> Result<(), String> {
        let ctx = match &ws.ctx[dev] {
            Some(c) => Arc::clone(c),
            None => {
                let c = self.context(dev)?;
                ws.ctx[dev] = Some(Arc::clone(&c));
                c
            }
        };
        ctx.bind_to_thread().map_err(|e| e.to_string())?; // raw driver calls below need the context current on this thread
        if ws.stream[dev].is_null() {
            ws.stream[dev] = result::stream::create(StreamKind::NonBlocking).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn d2h(&self, ws: &mut CudaWs, dev: usize, host: *mut u8, dev_ptr: *const u8, n: usize) -> Result<(), String> {
        let s = ws.stream[dev];
        // SAFETY: caller guarantees `dev_ptr` is a live device allocation of n bytes on `dev` and `host` writable for n bytes; context is current (set_device).
        unsafe { sys::cuMemcpyDtoHAsync_v2(host as *mut c_void, dev_ptr as usize as sys::CUdeviceptr, n, s) }.result().map_err(|e| e.to_string())?;
        // SAFETY: `s` was created by this thread in set_device and is never destroyed.
        unsafe { result::stream::synchronize(s) }.map_err(|e| e.to_string())
    }

    fn bounce(&self, ws: &mut CudaWs, n: usize) -> Result<*mut u8, String> {
        if ws.bounce.is_null() {
            // SAFETY: plain pinned allocation (flags 0 = cudaHostAllocDefault); the context is current (set_device); never freed, like the original.
            ws.bounce = unsafe { result::malloc_host(n, 0) }.map_err(|e| e.to_string())? as *mut u8;
        }
        Ok(ws.bounce)
    }

    fn h2d(&self, ws: &mut CudaWs, dev: usize, dst: *mut u8, src: *const u8, n: usize) -> Result<(), String> {
        let s = ws.stream[dev];
        // SAFETY: caller guarantees `dst` is a live device allocation of n bytes on `dev` and `src` readable for n bytes until the later synchronise; context is current.
        unsafe { sys::cuMemcpyHtoDAsync_v2(dst as usize as sys::CUdeviceptr, src as *const c_void, n, s) }.result().map_err(|e| e.to_string())
    }

    fn sync(&self, ws: &mut CudaWs, dev: usize) -> Result<(), String> {
        // SAFETY: the stream was created by this thread in set_device and is never destroyed.
        unsafe { result::stream::synchronize(ws.stream[dev]) }.map_err(|e| e.to_string())
    }
}
