//! The card supply proper: pool_ensure of decode.cuh (committed at 4d8d891, lines 536-620) with run_parallel (473-484). Every CUDA call, cache call and planner call goes through
//! `sy_ops`, in the order the original makes it (see ORDER OF EFFECTS in include/supply.h).
use crate::abi::*;
use std::io::Write;
use std::os::raw::{c_char, c_int, c_ulong, c_void};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// g_wb_skipped: card victims that were not worth a RAM slot.
pub static WB_SKIPPED: AtomicU64 = AtomicU64::new(0);

/// SpecMD `access_group` Hit-First ordering + Red-Blue Clean-Input `R4` Host-Shadow mode:
/// Enabled in production via `sy_set_hit_first(1)` (while the contract test suite leaves it `false`).
pub static HIT_FIRST_MODE: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn pread(fd: c_int, buf: *mut c_void, count: usize, offset: i64) -> isize;
    fn strerror(errnum: c_int) -> *const c_char;
    fn __errno_location() -> *mut c_int;
}

pub fn write_stderr(bytes: &[u8]) {
    let _ = std::io::stderr().write_all(bytes);
}

pub fn fatal_bytes(msg: &[u8]) -> ! {
    write_stderr(msg);
    std::process::exit(1)
}

pub fn fatal(msg: &str) -> ! {
    fatal_bytes(msg.as_bytes())
}

/// The texts handed to `ops.check`: the argument of the original's CHECK(...), as the preprocessor would stringize it (NUL-terminated for C).
pub mod site {
    pub const STREAM_SYNC_0: &[u8] = b"cudaStreamSynchronize(0)\0";
    pub const DEVICE_SYNC: &[u8] = b"cudaDeviceSynchronize()\0";
    pub const SET_DEVICE_POOL: &[u8] = b"cudaSetDevice(pool.dev)\0";
    pub const D2H: &[u8] = b"cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy)\0";
    pub const EVENT_RECORD: &[u8] = b"cudaEventRecord(pool.written_back, pool.copy)\0";
    pub const H2D_MAIN: &[u8] = b"cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy)\0";
    pub const SET_DEVICE_JOB: &[u8] = b"cudaSetDevice(dev)\0";
    pub const WAIT_EVENT: &[u8] = b"cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0)\0";
    pub const H2D_JOB: &[u8] = b"cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid])\0";
    pub const SYNC_JOB: &[u8] = b"cudaStreamSynchronize(pool.rstream[tid])\0";
    pub const SYNC_COPY: &[u8] = b"cudaStreamSynchronize(pool.copy)\0";
}

fn do_check(ops: &sy_ops, status: c_int, site: &'static [u8]) {
    // SAFETY: `site` is NUL-terminated; the function pointer is the caller's.
    unsafe { (ops.check)(status, site.as_ptr() as *const c_char) };
    if status != 0 {
        std::process::exit(1); // ops.check must not return on a non-zero status; if it does, stop here as the original's CHECK would have
    }
}

#[derive(Clone, Copy)]
pub struct Supply {
    pub ops: sy_ops,
    pub cfg: sy_config,
}

struct Need {
    e: c_int,
    dst: *mut u8,
    host: *mut u8,
}

struct Victim {
    key: c_int,
    heat: f32,
    src: *mut u8,
}

/// One reader job: a chunk of one expert.
struct Job {
    dst: *mut u8,
    host_shadow: *mut u8,
    off: i64,
    e: c_int,
    co: usize,
    len: usize,
}

/// What the reader jobs share. Read-only while the threads run.
struct Ctx {
    ops: sy_ops,
    prefill: bool,
    raw: usize,
    layer: c_int,
    fd: c_int,
    dev: c_int,
    rstream: [*mut c_void; SY_MAX_READERS],
    bounce: [*mut u8; SY_MAX_READERS],
    written_back: *mut c_void,
    slab: *const *const u8,
    jobs: Vec<Job>,
    next: AtomicUsize,
}

// SAFETY: the raw pointers are the caller's streams, event, bounce slots and slab table, which the contract keeps valid for the call; the ops that use them are thread-safe by contract.
unsafe impl Sync for Ctx {}

/// One job on the thread numbered `tid` (pool_ensure's lambda).
unsafe fn run_job(c: &Ctx, j: &Job, tid: usize) {
    let mut b: *mut u8 = if !j.host_shadow.is_null() { j.host_shadow } else { c.bounce[tid] };
    if c.prefill {
        b = (*c.slab.offset(c.layer as isize)).wrapping_add(((j.e & 3) as usize).wrapping_mul(c.raw)).wrapping_add(j.co) as *mut u8;
    } else {
        let mut got = 0usize;
        while got < j.len {
            let r = pread(c.fd, b.add(got) as *mut c_void, j.len - got, j.off.wrapping_add(got as i64));
            if r <= 0 {
                let why: Vec<u8> = if r < 0 {
                    let errno = *__errno_location();
                    std::ffi::CStr::from_ptr(strerror(errno)).to_bytes().to_vec()
                } else {
                    b"end of file".to_vec()
                };
                let mut m = format!("reading expert {} of layer {} failed: ", j.e, c.layer).into_bytes();
                m.extend_from_slice(&why);
                m.push(b'\n');
                fatal_bytes(&m);
            }
            got += r as usize;
        }
    }
    do_check(&c.ops, (c.ops.set_device)(c.dev), site::SET_DEVICE_JOB);
    do_check(&c.ops, (c.ops.stream_wait_event)(c.rstream[tid], c.written_back, 0), site::WAIT_EVENT);
    do_check(&c.ops, (c.ops.memcpy_h2d_async)(j.dst as *mut c_void, b as *const c_void, j.len, c.rstream[tid]), site::H2D_JOB);
    do_check(&c.ops, (c.ops.stream_sync)(c.rstream[tid]), site::SYNC_JOB);
}

/// The number of chunks per expert and the chunk size: a call with few experts cannot fill the drive with whole-expert reads, so each expert is split among the idle readers (at most 8 chunks);
/// `cfg_chunks` > 0 overrides the count. The size is rounded up to 4096.
pub fn chunking(raw: usize, cfg_chunks: c_int, n_drive: c_int, pool_readers: c_int) -> (c_int, usize) {
    let nchunk: c_int = if cfg_chunks > 0 {
        cfg_chunks
    } else if n_drive >= pool_readers || n_drive == 0 {
        1
    } else {
        std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)
    };
    let chunk: usize = (raw.wrapping_add(nchunk as usize).wrapping_sub(1) / nchunk as usize).wrapping_add(4095) / 4096 * 4096;
    (nchunk, chunk)
}

fn work(c: &Ctx, tid: usize) {
    loop {
        let i = c.next.fetch_add(1, Ordering::SeqCst);
        if i >= c.jobs.len() {
            return;
        }
        // SAFETY: see `Ctx`.
        unsafe { run_job(c, &c.jobs[i], tid) };
    }
}

/// run_parallel: jobs on up to `nthreads` threads, the calling thread being thread 0; threads 1.. are created now and joined before it returns.
fn run_parallel(c: &Ctx, nthreads: c_int) {
    if c.jobs.is_empty() {
        return;
    }
    let n: c_int = std::cmp::min(nthreads, c.jobs.len() as c_int);
    std::thread::scope(|s| {
        for i in 1..n {
            s.spawn(move || work(c, i as usize));
        }
        work(c, 0);
    });
}

impl Supply {
    pub fn new(ops: sy_ops, cfg: sy_config) -> Supply {
        WB_SKIPPED.store(0, Ordering::SeqCst);
        Supply { ops, cfg }
    }

    fn ekey(&self, layer: c_int, e: c_int) -> c_int {
        layer.wrapping_mul(self.cfg.experts).wrapping_add(e)
    }

    unsafe fn pool_find(&self, pool: &sy_pool, key: c_int) -> *mut u8 {
        if pool.cache.is_null() {
            null_mut()
        } else {
            (self.ops.sc_find)(pool.cache, key)
        }
    }

    unsafe fn host_find(&self, host: &sy_host, key: c_int) -> *mut u8 {
        if host.cache.is_null() {
            null_mut()
        } else {
            (self.ops.sc_find)(host.cache, key)
        }
    }

    unsafe fn host_worth(&self, host: &sy_host, key: c_int) -> bool {
        host.cache.is_null() || (self.ops.sc_worth)(host.cache, key) != 0
    }

    unsafe fn host_remove(&self, host: &sy_host, key: c_int) -> *mut u8 {
        if host.cache.is_null() {
            null_mut()
        } else {
            (self.ops.sc_remove)(host.cache, key)
        }
    }

    /// pool_ensure(pool, host, layer, fd, experts, counts, readers).
    ///
    /// # Safety
    /// As supply.h: `pool`, `host`, `counts`, `pool.h_eptr` (indexed by every expert), the caches, streams, event, bounce slots and the slab table must be valid; `experts` must be in 0..cfg.experts.
    pub unsafe fn ensure(&self, pool: *mut sy_pool, host: *const sy_host, layer: c_int, fd: c_int, experts: &[c_int], counts: *const c_int, readers: c_int) {
        let t_start = Instant::now();
        self.ensure_inner(&mut *pool, &*host, layer, fd, experts, counts, readers);
        (*pool).st.t_ensure += t_start.elapsed().as_secs_f64();
    }

    unsafe fn ensure_inner(&self, pool: &mut sy_pool, host: &sy_host, layer: c_int, fd: c_int, experts: &[c_int], counts: *const c_int, readers: c_int) {
        let ops = &self.ops;
        let vq = self.cfg.vq != 0;
        let prefill = self.cfg.prefill != 0;
        let raw = self.cfg.expert_raw;
        if vq {
            do_check(ops, (ops.stream_sync)(null_mut()), site::STREAM_SYNC_0);
        } else {
            do_check(ops, (ops.device_sync)(), site::DEVICE_SYNC);
        }
        // the distinct keys of this call: the original's unordered_set, given to the cache as an array (membership only)
        let mut prot: Vec<c_int> = experts.iter().map(|&e| self.ekey(layer, e)).collect();
        prot.sort_unstable();
        prot.dedup();
        let mut need: Vec<Need> = Vec::new();
        let mut victims: Vec<Victim> = Vec::new();
        pool.st.needed = pool.st.needed.wrapping_add(experts.len() as c_ulong);
        let hf_mode = HIT_FIRST_MODE.load(Ordering::Relaxed);
        if hf_mode {
            // SpecMD `access_group` Two-Pass Hit-First Ordering inside `libsupply.so`:
            // Pass 1: find & touch ALL resident CardPool hits before any miss calls `sc_acquire`!
            let mut misses: Vec<c_int> = Vec::new();
            for &e in experts {
                let key = self.ekey(layer, e);
                let p = self.pool_find(pool, key);
                let heat = *counts.offset(e as isize) as f32;
                if !p.is_null() {
                    if vq {
                        (ops.arr_wait)(key);
                    }
                    if !pool.cache.is_null() {
                        (ops.sc_touch)(pool.cache, key, heat);
                    }
                    *pool.h_eptr.offset(e as isize) = p as *const u8;
                    pool.st.hit_card = pool.st.hit_card.wrapping_add(1);
                } else {
                    misses.push(e);
                }
            }
            // Pass 2: acquire slots for misses
            for e in misses {
                let key = self.ekey(layer, e);
                let p_dup = self.pool_find(pool, key);
                let heat = *counts.offset(e as isize) as f32;
                if !p_dup.is_null() {
                    if vq {
                        (ops.arr_wait)(key);
                    }
                    if !pool.cache.is_null() {
                        (ops.sc_touch)(pool.cache, key, heat);
                    }
                    *pool.h_eptr.offset(e as isize) = p_dup as *const u8;
                    pool.st.hit_card = pool.st.hit_card.wrapping_add(1);
                    continue;
                }
                let mut vk: c_int = -1;
                let mut vh: f32 = 0.0;
                let p = (ops.sc_acquire)(pool.cache, key, prot.as_ptr(), prot.len(), heat, &mut vk, &mut vh);
                if vk >= 0 {
                    if vq {
                        (ops.arr_wait)(vk);
                    }
                    victims.push(Victim { key: vk, heat: vh, src: p });
                }
                *pool.h_eptr.offset(e as isize) = p as *const u8;
                need.push(Need { e, dst: p, host: null_mut() });
            }
        } else {
            for &e in experts {
                let key = self.ekey(layer, e);
                let mut p = self.pool_find(pool, key);
                let heat = *counts.offset(e as isize) as f32;
                if !p.is_null() {
                    if vq {
                        (ops.arr_wait)(key);
                    }
                    if !pool.cache.is_null() {
                        (ops.sc_touch)(pool.cache, key, heat);
                    }
                    *pool.h_eptr.offset(e as isize) = p as *const u8;
                    pool.st.hit_card = pool.st.hit_card.wrapping_add(1);
                    continue;
                }
                let mut vk: c_int = -1;
                let mut vh: f32 = 0.0;
                p = (ops.sc_acquire)(pool.cache, key, prot.as_ptr(), prot.len(), heat, &mut vk, &mut vh);
                if vk >= 0 {
                    if vq {
                        (ops.arr_wait)(vk);
                    }
                    victims.push(Victim { key: vk, heat: vh, src: p });
                }
                *pool.h_eptr.offset(e as isize) = p as *const u8;
                need.push(Need { e, dst: p, host: null_mut() });
            }
        }
        if need.is_empty() {
            return;
        }
        // 1. experts found in RAM leave the RAM cache; their slots are held until the copies are done
        let mut held: Vec<*mut u8> = Vec::new();
        for n in need.iter_mut() {
            let key = self.ekey(layer, n.e);
            if !self.host_find(host, key).is_null() {
                (ops.pend_take)(key);
                n.host = self.host_remove(host, key);
                held.push(n.host);
                pool.st.hit_host = pool.st.hit_host.wrapping_add(1);
            }
        }
        do_check(ops, (ops.set_device)(pool.dev), site::SET_DEVICE_POOL);
        // 2. the experts the card gives up are copied down into the RAM cache
        for v in &victims {
            if host.cap_slots == 0 || prefill {
                break;
            }
            if !self.host_worth(host, v.key) {
                WB_SKIPPED.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            let mut dk: c_int = -1;
            let mut dh: f32 = 0.0;
            let h = (ops.host_acquire)(v.key, v.heat, &mut dk, &mut dh);
            do_check(ops, (ops.memcpy_d2h_async)(h as *mut c_void, v.src as *const c_void, raw, pool.copy), site::D2H);
            pool.st.written_back = pool.st.written_back.wrapping_add(1);
        }
        do_check(ops, (ops.event_record)(pool.written_back, pool.copy), site::EVENT_RECORD);
        // 3. RAM -> card, queued behind the copies down on the same stream
        for n in &need {
            if !n.host.is_null() {
                let src: *const u8 = if prefill {
                    (*self.cfg.slab.offset(layer as isize)).wrapping_add(((n.e & 3) as usize).wrapping_mul(raw))
                } else {
                    n.host as *const u8
                };
                do_check(ops, (ops.memcpy_h2d_async)(n.dst as *mut c_void, src as *const c_void, raw, pool.copy), site::H2D_MAIN);
            }
        }
        // 4. drive -> pinned slot -> card, one chunk at a time per reading thread
        let n_drive: c_int = need.iter().filter(|n| n.host.is_null()).count() as c_int;
        let pool_readers: c_int = std::cmp::min(readers, pool.readers);
        let (nchunk, chunk) = chunking(raw, self.cfg.chunks, n_drive, pool_readers);
        let mut jobs: Vec<Job> = Vec::new();
        for n in &need {
            if !n.host.is_null() {
                continue;
            }
            pool.st.from_drive = pool.st.from_drive.wrapping_add(1);
            // In HIT_FIRST_MODE (production Policy 3), allocate a HostCache shadow slot for each NVMe read
            // so the parallel reader threads pread directly into pinned HostCache RAM and H2D from there!
            // Then every future CardPool eviction of this expert is a 0-byte R4 drop (`written_back = 0` from call 1!).
            let mut shadow_base: *mut u8 = null_mut();
            if hf_mode && host.cap_slots > 0 && !prefill {
                let key = self.ekey(layer, n.e);
                let heat = *counts.offset(n.e as isize) as f32;
                let mut dk: c_int = -1;
                let mut dh: f32 = 0.0;
                shadow_base = (ops.host_acquire)(key, heat, &mut dk, &mut dh);
            }
            for k in 0..nchunk {
                let co = (k as usize).wrapping_mul(chunk);
                if co >= raw {
                    break;
                }
                let len = std::cmp::min(chunk, raw - co);
                let h_sh = if shadow_base.is_null() { null_mut() } else { shadow_base.wrapping_add(co) };
                jobs.push(Job { dst: n.dst.wrapping_add(co), host_shadow: h_sh, off: (n.e as i64).wrapping_mul(raw as i64).wrapping_add(co as i64), e: n.e, co, len });
            }
        }
        let ctx = Ctx {
            ops: *ops,
            prefill,
            raw,
            layer,
            fd,
            dev: pool.dev,
            rstream: pool.rstream,
            bounce: pool.bounce,
            written_back: pool.written_back,
            slab: self.cfg.slab,
            jobs,
            next: AtomicUsize::new(0),
        };
        let t0 = Instant::now();
        run_parallel(&ctx, pool_readers);
        let t1 = Instant::now();
        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);
        for &h in &held {
            (ops.sc_release)(host.cache, h);
        }
        pool.st.t_read += t1.duration_since(t0).as_secs_f64();
        pool.st.t_copy += t1.elapsed().as_secs_f64();
    }
}
