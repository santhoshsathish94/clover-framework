//! Rust implementation of include/loader.h (`ld_init`, `ld_submit`, `ld_wait`, `ld_open_layer`).
pub mod abi;
pub mod job;
pub mod queue;
pub mod sys;
#[cfg(test)]
mod tests;

use abi::{job_fn, ld_config, ld_layer};
use job::{Cb, LayerJob};
use queue::{Loader, MAX_CARDS};
use std::ffi::CStr;
use std::os::raw::{c_int, c_void};
use sys::{fatal_msg, RealSys};

pub const N_LAYERS: u32 = 92; // backend-gpu.cu:31; the engine's g_handles and g_slab have N_LAYERS + 1 = 93 entries

/// ld_init's configuration, with the pointers kept as addresses so the struct is Send + Sync.
pub struct Config {
    pub cb: Cb,
    pub expert_raw: usize,
    pub prefill: bool,
    pub slab: usize,
    pub handles: usize,
}

struct State {
    cfg: Config,
    loader: &'static Loader,
}
static STATE: std::sync::OnceLock<State> = std::sync::OnceLock::new();

impl Cb {
    fn start_thread(&self, card: usize) {
        // SAFETY: header: set_device is valid for the life of the process and thread-safe.
        unsafe { (self.set_device)(self.ctx, card as c_int) }
    }
}

/// The checks of ld_init, in the reference's order; Err = the message before exit(1).
pub fn validate_config(c: &ld_config) -> Result<Config, String> {
    let n = |what: &str| format!("loader: {} is NULL\n", what);
    let set_device = c.cb.set_device.ok_or_else(|| n("set_device"))?;
    let ensure_work = c.cb.ensure_work.ok_or_else(|| n("ensure_work"))?;
    let load_trunk = c.cb.load_trunk.ok_or_else(|| n("load_trunk"))?;
    let load_expert_tables = c.cb.load_expert_tables.ok_or_else(|| n("load_expert_tables"))?;
    let alloc_state = c.cb.alloc_state.ok_or_else(|| n("alloc_state"))?;
    if c.handles.is_null() {
        return Err(n("handles"));
    }
    if c.prefill != 0 && c.cb.host_alloc.is_none() {
        return Err(n("host_alloc"));
    }
    if c.prefill != 0 && c.slab.is_null() {
        return Err(n("slab"));
    }
    Ok(Config {
        cb: Cb { ctx: c.cb.ctx, set_device, ensure_work, load_trunk, load_expert_tables, alloc_state, host_alloc: c.cb.host_alloc },
        expert_raw: c.expert_raw,
        prefill: c.prefill != 0,
        slab: c.slab as usize,
        handles: c.handles as usize,
    })
}

pub fn check_card(card: c_int) -> Result<(), String> {
    if card < 0 || card as usize >= MAX_CARDS {
        return Err(format!("loader: card {} out of range (0..{})\n", card, MAX_CARDS - 1));
    }
    Ok(())
}

pub fn check_slab(prefill: bool, layer: u32) -> Result<(), String> {
    if prefill && layer >= 1 && layer > N_LAYERS {
        return Err(format!("loader: layer {} has no slab slot (0..{})\n", layer, N_LAYERS));
    }
    Ok(())
}

fn die(m: Result<(), String>) {
    if let Err(m) = m {
        fatal_msg(m.as_bytes());
    }
}

/// Thread-safe job argument.
struct Arg(*mut c_void);
// SAFETY: the header makes the caller keep the argument valid and usable from the card's thread.
unsafe impl Send for Arg {}
impl Arg {
    fn get(&self) -> *mut c_void {
        self.0
    }
}

fn state(function: &str) -> &'static State {
    match STATE.get() {
        Some(s) => s,
        None => fatal_msg(format!("loader: {} before ld_init\n", function).as_bytes()),
    }
}

/// # Safety
/// `cfg` must point to a valid `ld_config` whose callbacks, ctx and tables stay valid for the life of the process.
#[no_mangle]
pub unsafe extern "C" fn ld_init(cfg: *const ld_config) {
    if cfg.is_null() {
        fatal_msg(b"loader: cfg is NULL\n");
    }
    if STATE.get().is_some() {
        return; // the first configuration is kept
    }
    // SAFETY: non-null checked above; caller guarantees validity.
    let c = unsafe { &*cfg };
    let conf = match validate_config(c) {
        Ok(v) => v,
        Err(m) => fatal_msg(m.as_bytes()),
    };
    let cb = conf.cb;
    let loader = Loader::new(Box::new(move |card| cb.start_thread(card)));
    let _ = STATE.set(State { cfg: conf, loader });
}

/// # Safety
/// `fn_` (if not NULL) must be callable with `arg` from the card's thread; both stay valid until the job has run.
#[no_mangle]
pub unsafe extern "C" fn ld_submit(card: c_int, fn_: Option<job_fn>, arg: *mut c_void) {
    let st = state("ld_submit");
    let f = match fn_ {
        Some(f) => f,
        None => fatal_msg(b"loader: job is NULL\n"),
    };
    die(check_card(card));
    let arg = Arg(arg);
    // SAFETY (inside the job): the caller's contract above.
    st.loader.submit(card as usize, Box::new(move || unsafe { f(arg.get()) }));
}

#[no_mangle]
pub extern "C" fn ld_wait() {
    if let Some(s) = STATE.get() {
        s.loader.wait();
    } // before ld_init nothing can be pending
}

fn put(p: *mut c_int, v: c_int) {
    if !p.is_null() {
        // SAFETY: header: the h_* pointers are addresses of the Handle's ints, valid for the life of the process.
        unsafe { *p = v }
    }
}

/// # Safety
/// `req` must point to a valid `ld_layer`; see loader.h for what the caller keeps valid.
#[no_mangle]
pub unsafe extern "C" fn ld_open_layer(req: *const ld_layer) {
    let st = state("ld_open_layer");
    if req.is_null() {
        fatal_msg(b"loader: req is NULL\n");
    }
    // SAFETY: non-null checked above; caller guarantees validity.
    let r = unsafe { &*req };
    if r.dataset.is_null() {
        fatal_msg(b"loader: dataset is NULL\n");
    }
    die(check_card(r.card));
    die(check_slab(st.cfg.prefill, r.layer));
    // bookkeeping, in the original's order
    put(r.h_L, r.layer as c_int);
    put(r.h_card, r.card);
    put(r.h_k_in, r.k as c_int);
    if r.layer <= N_LAYERS {
        // SAFETY: handles is the caller's table of N_LAYERS + 1 pointers (header) and layer <= N_LAYERS.
        unsafe { *(st.cfg.handles as *mut *mut c_void).add(r.layer as usize) = r.handle }
    }
    // SAFETY: dataset is non-null and NUL-terminated (header); copied here, as `const std::string ds(dataset)` does.
    let ds = unsafe { CStr::from_ptr(r.dataset) }.to_owned();
    let job = LayerJob {
        cb: st.cfg.cb,
        handle: r.handle,
        ds,
        layer: r.layer,
        positions: r.positions,
        card: r.card,
        h_fd: r.h_fd,
        expert_raw: st.cfg.expert_raw,
        prefill: st.cfg.prefill,
        slab: st.cfg.slab as *mut *mut u8,
    };
    st.loader.submit(
        r.card as usize,
        Box::new(move || {
            if let Err(m) = job::run(&RealSys, &job) {
                fatal_msg(&m);
            }
        }),
    );
}
