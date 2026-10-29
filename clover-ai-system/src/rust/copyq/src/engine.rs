//! Queue and worker logic of the per-card copy queue. Behaviour reference: pf_enqueue() in backend-gpu.cu (lines 61-129) and act_gate() in decode.cuh.
//! All CUDA and drive access goes through the small `Backend` trait so that queue order, lazy thread start and flag order can be unit-tested without a GPU.
use crate::abi::{cq_config, cq_job};
use std::collections::VecDeque;
use std::io::Write;
use std::os::raw::c_int;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

pub const MAX_CARDS: usize = 8; // backend-gpu.cu:30
pub const N_SLAB: usize = 93; // decode.cuh:390 `g_slab[93]`

/// What the workers need from the outside world. `Ws` is the state of one worker thread (bounce buffer, one stream per device); it is created inside that thread.
pub trait Backend: Send + Sync + 'static {
    type Ws;
    fn new_ws(&self) -> Self::Ws;
    /// One pread(2): Ok(bytes read, 0 = end of file) or Err(errno).
    fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> Result<usize, i32>;
    /// cudaSetDevice + lazy creation of this thread's non-blocking stream for `dev`.
    fn set_device(&self, ws: &mut Self::Ws, dev: usize) -> Result<(), String>;
    /// Async device -> host copy of `n` bytes followed by a stream synchronise.
    fn d2h(&self, ws: &mut Self::Ws, dev: usize, host: *mut u8, dev_ptr: *const u8, n: usize) -> Result<(), String>;
    /// This thread's pinned bounce buffer of `n` bytes, allocated on first use.
    fn bounce(&self, ws: &mut Self::Ws, n: usize) -> Result<*mut u8, String>;
    /// Async host -> device copy on this thread's stream for `dev` (no synchronise).
    fn h2d(&self, ws: &mut Self::Ws, dev: usize, dst: *mut u8, src: *const u8, n: usize) -> Result<(), String>;
    /// Synchronise this thread's stream for `dev`.
    fn sync(&self, ws: &mut Self::Ws, dev: usize) -> Result<(), String>;
}

/// A job as handed to the queue; the raw pointers belong to the caller.
pub struct Job(pub cq_job);
// SAFETY: the header makes the caller guarantee that every pointer in a job stays valid until its flag is set and may be used from any thread.
unsafe impl Send for Job {}

/// Plain copy of cq_config; pointers are kept as addresses so the struct is Send + Sync.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub expert_raw: usize,
    pub readers: i32,
    pub cards: i32,
    pub prefill: bool,
    pub slab: usize,
    pub act_hold: usize,
}

impl Cfg {
    pub fn from_c(c: &cq_config) -> Cfg {
        Cfg { expert_raw: c.expert_raw, readers: c.readers, cards: c.cards, prefill: c.prefill != 0, slab: c.slab as usize, act_hold: c.act_hold as usize }
    }
}

struct QState {
    q: VecDeque<Job>,
    started: bool,
}
struct Queue {
    st: Mutex<QState>,
    cv: Condvar,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner()) // no code path panics while holding a queue lock; poisoning cannot lose data
}

pub fn fatal(msg: &str) -> ! {
    // eprintln! would panic on a failed stderr write and leave callers spinning; fprintf in the original ignores it.
    let _ = writeln!(std::io::stderr(), "{}", msg);
    std::process::exit(1)
}

macro_rules! chk {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => fatal(&format!("{}:{}: {}", file!(), line!(), e)),
        }
    };
}

/// Release store of 1 into a caller-owned atomic int (backend-gpu.cu:94,112,121,122 `state.store(1, release)`).
fn store1(p: *mut c_int) {
    if p.is_null() {
        return;
    }
    // SAFETY: the header says every *_state is the address of a live, 4-byte aligned atomic int owned by the caller.
    unsafe { AtomicI32::from_ptr(p) }.store(1, Ordering::Release);
}

pub struct Engine<B: Backend> {
    cfg: Cfg,
    be: B,
    queues: Vec<Queue>,
}

impl<B: Backend> Engine<B> {
    pub fn new(cfg: Cfg, be: B) -> Arc<Self> {
        if cfg.expert_raw == 0 {
            fatal("copyq: expert_raw must be non-zero");
        }
        let queues = (0..=MAX_CARDS).map(|_| Queue { st: Mutex::new(QState { q: VecDeque::new(), started: false }), cv: Condvar::new() }).collect();
        Arc::new(Engine { cfg, be, queues })
    }

    /// Threads a queue gets when its first job arrives (backend-gpu.cu:72).
    pub fn threads_for(cfg: &Cfg, qi: usize) -> i32 {
        if qi == MAX_CARDS { cfg.readers } else { std::cmp::max(2, cfg.readers / std::cmp::max(1, cfg.cards)) }
    }

    /// pf_enqueue (backend-gpu.cu:66-125).
    pub fn enqueue(self: &Arc<Self>, j: &cq_job) {
        let qi = if j.kind == 0 { MAX_CARDS } else { j.dev as usize };
        if j.kind != 0 && !(0..MAX_CARDS as c_int).contains(&j.dev) {
            fatal(&format!("copyq: job for card {} (queues exist for 0..{})", j.dev, MAX_CARDS - 1));
        }
        let q = &self.queues[qi];
        let mut g = lock(&q.st);
        if !g.started {
            g.started = true;
            for _ in 0..Self::threads_for(&self.cfg, qi) {
                let eng = Arc::clone(self);
                if let Err(e) = std::thread::Builder::new().stack_size(8 << 20).spawn(move || eng.worker(qi)) {
                    fatal(&format!("copyq: cannot start a reading thread: {}", e));
                }
            }
        }
        g.q.push_back(Job(*j));
        q.cv.notify_one();
    }

    fn worker(&self, qi: usize) {
        let mut ws = self.be.new_ws();
        let q = &self.queues[qi];
        loop {
            let job = {
                let mut g = lock(&q.st);
                while g.q.is_empty() {
                    g = q.cv.wait(g).unwrap_or_else(|e| e.into_inner());
                }
                g.q.pop_front()
            };
            if let Some(Job(jb)) = job {
                self.run(&mut ws, &jb);
            }
        }
    }

    fn read_full(&self, fd: c_int, buf: *mut u8, off: i64, what: &str) {
        let raw = self.cfg.expert_raw;
        let mut got = 0usize;
        while got < raw {
            match self.be.pread(fd, buf.wrapping_add(got), raw - got, off + got as i64) {
                Ok(n) if n > 0 => got += n,
                Ok(_) => fatal(&format!("{}: pread failed: end of file", what)),
                Err(errno) => {
                    let s = std::io::Error::from_raw_os_error(errno).to_string();
                    let s = s.split(" (os error").next().unwrap_or("").to_string();
                    fatal(&format!("{}: pread failed: {}", what, s))
                }
            }
        }
    }

    /// act_gate (decode.cuh:392).
    fn act_gate(&self) {
        if self.cfg.act_hold == 0 {
            return;
        }
        // SAFETY: the header says act_hold is the address of a live atomic int for the whole run.
        let hold = unsafe { AtomicI32::from_ptr(self.cfg.act_hold as *mut i32) };
        while hold.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
        }
    }

    /// `g_slab[layer] + ((off / EXPERT_RAW) & 3) * EXPERT_RAW` (backend-gpu.cu:104, 117).
    fn slab_src(&self, layer: c_int, off: i64) -> *const u8 {
        if self.cfg.slab == 0 || layer < 0 || layer as usize >= N_SLAB {
            fatal(&format!("copyq: prefill source: layer {} outside the slab table", layer));
        }
        // SAFETY: the header says slab points to N_SLAB pointers; layer was range-checked above.
        let base = unsafe { *(self.cfg.slab as *const *const u8).add(layer as usize) };
        let raw = self.cfg.expert_raw;
        base.wrapping_add(((off / raw as i64) & 3) as usize * raw)
    }

    /// Body of the worker loop for one job (backend-gpu.cu:82-123).
    fn run(&self, ws: &mut B::Ws, jb: &cq_job) {
        let raw = self.cfg.expert_raw;
        if jb.kind == 0 {
            if !self.cfg.prefill {
                self.read_full(jb.fd, jb.slot, jb.off, "read-ahead");
            }
            store1(jb.p_state);
            return;
        }
        let dev = jb.dev as usize; // range checked in enqueue
        chk!(self.be.set_device(ws, dev));
        if !jb.victim_host.is_null() {
            chk!(self.be.d2h(ws, dev, jb.victim_host, jb.dst, raw));
            store1(jb.wb_state); // before the upload, as the original
        }
        if jb.kind == 1 {
            let src: *const u8 = if self.cfg.prefill {
                self.slab_src(jb.layer, jb.off)
            } else {
                let b = chk!(self.be.bounce(ws, raw));
                self.read_full(jb.fd, b, jb.off, "card read-ahead");
                b
            };
            self.act_gate();
            chk!(self.be.h2d(ws, dev, jb.dst, src, raw));
        } else {
            self.act_gate();
            let src = if self.cfg.prefill { self.slab_src(jb.layer, jb.off) } else { jb.host_src };
            chk!(self.be.h2d(ws, dev, jb.dst, src, raw));
        }
        chk!(self.be.sync(ws, dev));
        store1(jb.rel_state); // before arr_state, as the original
        store1(jb.arr_state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::{Duration, Instant};

    const RAW: usize = 4096;

    #[derive(Default)]
    struct Stub {
        new_ws: AtomicUsize,
        h2d_calls: AtomicUsize,
        pread_calls: AtomicUsize,
        pread_fds: Mutex<Vec<i32>>,
        log: Mutex<Vec<String>>,
        flags: Mutex<Option<[usize; 3]>>, // wb, rel, arr addresses to snapshot into the log
    }
    struct StubWs {
        bounce: Vec<u8>,
    }
    impl Stub {
        fn snap(&self, what: &str) {
            let f = *lock(&self.flags);
            let s = match f {
                // SAFETY: test-owned atomics that outlive the job.
                Some(a) => a.iter().map(|&p| unsafe { AtomicI32::from_ptr(p as *mut i32) }.load(Ordering::SeqCst).to_string()).collect::<Vec<_>>().join(""),
                None => String::new(),
            };
            lock(&self.log).push(format!("{} {}", what, s));
        }
    }
    impl Backend for Stub {
        type Ws = StubWs;
        fn new_ws(&self) -> StubWs {
            self.new_ws.fetch_add(1, Ordering::SeqCst);
            StubWs { bounce: Vec::new() }
        }
        fn pread(&self, fd: c_int, buf: *mut u8, len: usize, off: i64) -> Result<usize, i32> {
            self.pread_calls.fetch_add(1, Ordering::SeqCst);
            lock(&self.pread_fds).push(fd);
            let n = len.min(1000); // short reads exercise the loop
            for i in 0..n {
                // SAFETY: the engine passes a buffer of at least `len` bytes.
                unsafe { *buf.add(i) = (off as usize + i) as u8 };
            }
            Ok(n)
        }
        fn set_device(&self, _: &mut StubWs, _: usize) -> Result<(), String> {
            self.snap("set_device");
            Ok(())
        }
        fn d2h(&self, _: &mut StubWs, _: usize, host: *mut u8, dev_ptr: *const u8, n: usize) -> Result<(), String> {
            self.snap("d2h");
            // SAFETY: test buffers of RAW bytes.
            unsafe { std::ptr::copy_nonoverlapping(dev_ptr, host, n) };
            Ok(())
        }
        fn bounce(&self, ws: &mut StubWs, n: usize) -> Result<*mut u8, String> {
            if ws.bounce.is_empty() {
                self.snap("bounce-alloc");
                ws.bounce = vec![0; n];
            }
            Ok(ws.bounce.as_mut_ptr())
        }
        fn h2d(&self, _: &mut StubWs, _: usize, dst: *mut u8, src: *const u8, n: usize) -> Result<(), String> {
            self.snap("h2d");
            self.h2d_calls.fetch_add(1, Ordering::SeqCst);
            // SAFETY: test buffers of RAW bytes.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, n) };
            Ok(())
        }
        fn sync(&self, _: &mut StubWs, _: usize) -> Result<(), String> {
            self.snap("sync");
            Ok(())
        }
    }

    fn cfg(readers: i32, cards: i32) -> Cfg {
        Cfg { expert_raw: RAW, readers, cards, prefill: false, slab: 0, act_hold: 0 }
    }
    fn job() -> cq_job {
        cq_job {
            kind: 0, dev: 0, fd: 0, off: 0, layer: 0,
            slot: std::ptr::null_mut(), p_state: std::ptr::null_mut(), dst: std::ptr::null_mut(), host_src: std::ptr::null(),
            victim_host: std::ptr::null_mut(), wb_state: std::ptr::null_mut(), arr_state: std::ptr::null_mut(), rel_state: std::ptr::null_mut(),
        }
    }
    fn wait_for(cond: impl Fn() -> bool) -> bool {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(10) {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }
    fn flag() -> Arc<AtomicI32> {
        Arc::new(AtomicI32::new(0))
    }

    #[test]
    fn thread_counts_follow_the_original() {
        assert_eq!(Engine::<Stub>::threads_for(&cfg(16, 8), MAX_CARDS), 16);
        assert_eq!(Engine::<Stub>::threads_for(&cfg(16, 8), 3), 2);
        assert_eq!(Engine::<Stub>::threads_for(&cfg(16, 2), 3), 8);
        assert_eq!(Engine::<Stub>::threads_for(&cfg(16, 0), 3), 16); // max(1, cards)
        assert_eq!(Engine::<Stub>::threads_for(&cfg(1, 1), 0), 2); // at least 2 per card queue
    }

    #[test]
    fn threads_start_lazily_and_only_once_per_queue() {
        let eng = Engine::new(cfg(4, 2), Stub::default()); // card queue: max(2, 4/2) = 2, drive queue: 4
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(eng.be.new_ws.load(Ordering::SeqCst), 0, "no thread before the first job");
        let buf = vec![0u8; RAW];
        let p = flag();
        let mut j = job();
        j.slot = buf.as_ptr() as *mut u8;
        j.p_state = p.as_ptr();
        eng.enqueue(&j);
        assert!(wait_for(|| p.load(Ordering::SeqCst) == 1));
        assert!(wait_for(|| eng.be.new_ws.load(Ordering::SeqCst) == 4));
        let p2 = flag();
        j.p_state = p2.as_ptr();
        eng.enqueue(&j);
        assert!(wait_for(|| p2.load(Ordering::SeqCst) == 1));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(eng.be.new_ws.load(Ordering::SeqCst), 4, "second job to the same queue starts no thread");
        // a card-1 job starts that queue's 2 threads; card 0 stays unstarted
        let dst = vec![0u8; RAW];
        let src = vec![0u8; RAW];
        let a = flag();
        let mut k = job();
        k.kind = 2; k.dev = 1; k.dst = dst.as_ptr() as *mut u8; k.host_src = src.as_ptr(); k.arr_state = a.as_ptr();
        eng.enqueue(&k);
        assert!(wait_for(|| a.load(Ordering::SeqCst) == 1));
        assert!(wait_for(|| eng.be.new_ws.load(Ordering::SeqCst) == 6));
        assert!(!lock(&eng.queues[0].st).started);
    }

    #[test]
    fn single_reader_serves_a_queue_in_fifo_order() {
        let eng = Engine::new(cfg(1, 1), Stub::default()); // drive queue has exactly 1 thread -> completion order = queue order
        let bufs: Vec<Vec<u8>> = (0..20).map(|_| vec![0u8; RAW]).collect();
        let flags: Vec<Arc<AtomicI32>> = (0..20).map(|_| flag()).collect();
        for i in 0..20 {
            let mut j = job();
            j.fd = i as i32; j.slot = bufs[i].as_ptr() as *mut u8; j.p_state = flags[i].as_ptr();
            eng.enqueue(&j);
        }
        assert!(wait_for(|| flags.iter().all(|f| f.load(Ordering::SeqCst) == 1)));
        let fds = lock(&eng.be.pread_fds).clone();
        // each job reads RAW bytes in 1000-byte pieces: 5 calls per job, in job order
        let per_job: Vec<i32> = fds.chunks(5).map(|c| c[0]).collect();
        assert_eq!(per_job, (0..20).collect::<Vec<i32>>());
        assert!(fds.chunks(5).all(|c| c.len() == 5 && c.iter().all(|&x| x == c[0])));
        assert_eq!(bufs[7][1001], 1001usize as u8, "bytes land at the right offset (partial reads continue where they stopped)");
    }

    #[test]
    fn prefill_skips_the_read_but_sets_the_flag() {
        let mut c = cfg(1, 1);
        c.prefill = true;
        let eng = Engine::new(c, Stub::default());
        let p = flag();
        let mut j = job();
        j.slot = std::ptr::null_mut();
        j.p_state = p.as_ptr();
        eng.enqueue(&j);
        assert!(wait_for(|| p.load(Ordering::SeqCst) == 1));
        assert_eq!(eng.be.pread_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn prefill_source_is_slab_layer_plus_record_mod_4() {
        let mut slab_bufs: Vec<Vec<u8>> = (0..N_SLAB).map(|_| vec![0u8; RAW * 4]).collect();
        for (l, b) in slab_bufs.iter_mut().enumerate() {
            for r in 0..4 {
                for x in &mut b[r * RAW..(r + 1) * RAW] {
                    *x = (l * 4 + r) as u8;
                }
            }
        }
        let slab: Vec<*const u8> = slab_bufs.iter().map(|b| b.as_ptr()).collect();
        let mut c = cfg(2, 1);
        c.prefill = true;
        c.slab = slab.as_ptr() as usize;
        let eng = Engine::new(c, Stub::default());
        for (kind, layer, rec) in [(1, 5, 6i64), (2, 17, 3), (1, 92, 4)] {
            let dst = vec![0u8; RAW];
            let a = flag();
            let mut j = job();
            j.kind = kind; j.dev = 0; j.layer = layer; j.off = rec * RAW as i64; j.dst = dst.as_ptr() as *mut u8; j.arr_state = a.as_ptr();
            eng.enqueue(&j);
            assert!(wait_for(|| a.load(Ordering::SeqCst) == 1));
            assert_eq!(dst[0], (layer as usize * 4 + (rec & 3) as usize) as u8);
            assert_eq!(dst[RAW - 1], dst[0]);
        }
        assert_eq!(eng.be.pread_calls.load(Ordering::SeqCst), 0, "no drive read with prefill");
        assert!(!lock(&eng.be.log).iter().any(|l| l.starts_with("bounce-alloc")), "no bounce buffer with prefill");
    }

    #[test]
    fn kind1_order_victim_then_bounce_read_then_upload_then_flags() {
        let eng = Engine::new(cfg(2, 1), Stub::default());
        let (wb, rel, arr) = (flag(), flag(), flag());
        *lock(&eng.be.flags) = Some([wb.as_ptr() as usize, rel.as_ptr() as usize, arr.as_ptr() as usize]);
        let dst = vec![0xAAu8; RAW];
        let victim = vec![0u8; RAW];
        let mut j = job();
        j.kind = 1; j.dev = 0; j.fd = 9; j.off = 0; j.dst = dst.as_ptr() as *mut u8; j.victim_host = victim.as_ptr() as *mut u8;
        j.wb_state = wb.as_ptr(); j.rel_state = rel.as_ptr(); j.arr_state = arr.as_ptr();
        // watcher: arr may only be seen as 1 when rel is already 1
        let (r2, a2) = (rel.clone(), arr.clone());
        let watcher = std::thread::spawn(move || {
            while a2.load(Ordering::SeqCst) == 0 {
                std::hint::spin_loop();
            }
            r2.load(Ordering::SeqCst)
        });
        eng.enqueue(&j);
        assert_eq!(watcher.join().unwrap_or(0), 1, "rel_state is set before arr_state");
        let log = lock(&eng.be.log).clone();
        // flags snapshot is wb,rel,arr
        assert_eq!(log, vec!["set_device 000", "d2h 000", "bounce-alloc 100", "h2d 100", "sync 100"]);
        assert!(victim.iter().all(|&b| b == 0xAA), "victim got the card's old bytes");
        assert_eq!(dst[1001], (1001usize) as u8, "card got the drive bytes");
    }

    #[test]
    fn kind2_without_victim_and_null_rel() {
        let eng = Engine::new(cfg(2, 1), Stub::default());
        let src: Vec<u8> = (0..RAW).map(|i| (i * 3) as u8).collect();
        let dst = vec![0u8; RAW];
        let a = flag();
        let mut j = job();
        j.kind = 2; j.dst = dst.as_ptr() as *mut u8; j.host_src = src.as_ptr(); j.arr_state = a.as_ptr();
        eng.enqueue(&j);
        assert!(wait_for(|| a.load(Ordering::SeqCst) == 1));
        assert_eq!(dst, src);
        let log = lock(&eng.be.log).clone();
        assert!(!log.iter().any(|l| l.starts_with("d2h")));
    }

    #[test]
    fn act_hold_blocks_the_upload_until_released() {
        let hold = flag();
        let mut c = cfg(2, 1);
        c.act_hold = hold.as_ptr() as usize;
        hold.store(1, Ordering::SeqCst);
        let eng = Engine::new(c, Stub::default());
        let src = vec![1u8; RAW];
        let dst = vec![0u8; RAW];
        let a = flag();
        let mut j = job();
        j.kind = 2; j.dst = dst.as_ptr() as *mut u8; j.host_src = src.as_ptr(); j.arr_state = a.as_ptr();
        eng.enqueue(&j);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(eng.be.h2d_calls.load(Ordering::SeqCst), 0, "held");
        assert_eq!(a.load(Ordering::SeqCst), 0);
        hold.store(0, Ordering::SeqCst);
        assert!(wait_for(|| a.load(Ordering::SeqCst) == 1));
        assert_eq!(eng.be.h2d_calls.load(Ordering::SeqCst), 1);
    }
}
