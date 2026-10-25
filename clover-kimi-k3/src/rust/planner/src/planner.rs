//! The planner proper. Behaviour reference: rc_plan, vq_sweep, vq_issue, rc_plan_vq, route_hook, pf_submit_rust (backend-gpu.cu) and pend_take, arr_wait (decode.cuh).
//! Every call out to the slot caches, the route table and the copy queue goes through `sp_ops`, in the order the original makes it.
use crate::abi::*;
use crate::dense::DenseMap;
use std::io::Write;
use std::os::raw::{c_int, c_long};
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

pub const MAX_CARDS: usize = 8;
pub const LAYER_SLOTS: usize = 93; // layers 0..92

pub fn write_stderr(bytes: &[u8]) {
    let _ = std::io::stderr().write_all(bytes);
}

pub fn fatal(msg: &str) -> ! {
    write_stderr(msg.as_bytes());
    write_stderr(b"\n");
    std::process::exit(1)
}

pub fn mismatch_message(pos: usize, layer: i32) -> Vec<u8> {
    format!("K3GPU route cache: call {} (layer {}) does not match the cache; look-ahead stopped\n", pos, layer).into_bytes()
}

// ---- the layers' handles: fd may be set by another thread (sp_layer_set_fd), so every field is atomic and lives outside the planner state

pub struct LayerSlot {
    present: AtomicI32,
    fd: AtomicI32,
    card: AtomicI32,
}

pub struct LayerTable {
    slots: [LayerSlot; LAYER_SLOTS],
}

#[derive(Clone, Copy)]
pub struct Handle {
    pub fd: i32,
    pub card: usize,
}

impl LayerTable {
    pub const fn new() -> Self {
        const S: LayerSlot = LayerSlot { present: AtomicI32::new(0), fd: AtomicI32::new(-1), card: AtomicI32::new(0) };
        LayerTable { slots: [S; LAYER_SLOTS] }
    }
    fn slot(&self, layer: i32) -> Option<&LayerSlot> {
        if layer < 0 {
            None
        } else {
            self.slots.get(layer as usize)
        }
    }
    /// A layer outside 0..92, or a card outside 0..7, is ignored (the original indexes g_handles / g_pool out of bounds there).
    pub fn set(&self, layer: i32, fd: i32, card: i32) {
        if card < 0 || card as usize >= MAX_CARDS {
            return;
        }
        if let Some(s) = self.slot(layer) {
            s.card.store(card, Ordering::Relaxed);
            s.fd.store(fd, Ordering::Release);
            s.present.store(1, Ordering::Release);
        }
    }
    pub fn set_fd(&self, layer: i32, fd: i32) {
        if let Some(s) = self.slot(layer) {
            s.fd.store(fd, Ordering::Release);
        }
    }
    pub fn get(&self, layer: i32) -> Option<Handle> {
        let s = self.slot(layer)?;
        if s.present.load(Ordering::Acquire) == 0 {
            return None;
        }
        Some(Handle { fd: s.fd.load(Ordering::Acquire), card: s.card.load(Ordering::Relaxed) as usize })
    }
}

// ---- entries the copy queue's workers write: the address of `state` goes into the job, the planner frees an entry only after it read 1 with acquire ordering

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[repr(C)]
#[allow(dead_code)]
pub struct Pending {
    pub state: AtomicI32,
    pub slot: *mut u8,
}
pub struct Arrival {
    pub state: AtomicI32,
}
#[allow(dead_code)]
pub struct Rel {
    pub state: AtomicI32,
    pub slot: *mut u8,
}

fn done(state: &AtomicI32) -> bool {
    state.load(Ordering::Acquire) == 1
}

fn spin_until_set(state: &AtomicI32) {
    while state.load(Ordering::Acquire) == 0 {
        std::thread::yield_now();
    }
}

// ---- what sp_pool_set / sp_host_set describe

#[derive(Clone, Copy)]
pub struct Pool {
    pub cache: *mut sc_cache,
    pub dev: i32,
    pub ready: bool,
}

#[derive(Clone, Copy)]
pub struct Host {
    pub cache: *mut sc_cache,
    pub cap_slots: usize,
}

#[derive(Clone, Copy)]
pub struct Wiring {
    pub pools: [Pool; MAX_CARDS],
    pub host: Host,
}

impl Wiring {
    pub const UNSET: Wiring = Wiring {
        pools: [Pool { cache: std::ptr::null_mut(), dev: 0, ready: false }; MAX_CARDS],
        host: Host { cache: std::ptr::null_mut(), cap_slots: 0 },
    };
}

pub struct Planner {
    ops: sp_ops,
    cfg: sp_config,
    layers: &'static LayerTable,
    pub w: Wiring,
    /// pos, planned_to, rc_ok and the counters; sizes are filled in by `stats()`
    st: sp_stats,
    pend: DenseMap<Box<Pending>>, // g_pend and g_pend_keys: the keys are the protected set
    arr: DenseMap<Box<Arrival>>,  // g_arr
    rel: Vec<Box<Rel>>,           // g_rel
    wb_keys: Vec<i32>,            // g_wb_keys
    // entries replaced in their map before their state was 1: the queue may still write them, so they are kept and freed at a later sweep once the state reads 1
    dead_pend: Vec<Box<Pending>>,
    dead_arr: Vec<Box<Arrival>>,
}

impl Planner {
    pub fn new(ops: sp_ops, cfg: sp_config, layers: &'static LayerTable, w: Wiring) -> Planner {
        let st = sp_stats { rc_ok: (cfg.rc_ok != 0) as c_int, ..sp_stats::default() };
        Planner { ops, cfg, layers, w, st, pend: DenseMap::new(), arr: DenseMap::new(), rel: Vec::new(), wb_keys: Vec::new(), dead_pend: Vec::new(), dead_arr: Vec::new() }
    }

    pub fn stats(&self) -> sp_stats {
        let mut s = self.st;
        s.pending = self.pend.len();
        s.arrivals = self.arr.len();
        s.releases = self.rel.len();
        s.wb_keys = self.wb_keys.len();
        s
    }

    /// Entries kept because they were replaced before their state was 1: (pending, arrivals).
    pub fn graveyard_len(&self) -> (usize, usize) {
        (self.dead_pend.len(), self.dead_arr.len())
    }

    // ---- thin wrappers over the ops table

    fn ekey(&self, layer: i32, e: i32) -> i32 {
        layer.wrapping_mul(self.cfg.experts).wrapping_add(e)
    }
    fn calls(&self) -> usize {
        unsafe { (self.ops.rt_calls)(self.ops.table) }
    }
    fn call_layer(&self, i: usize) -> i32 {
        unsafe { (self.ops.rt_layer)(self.ops.table, i) }
    }
    /// The experts of call `i`; the table keeps them alive until rt_free.
    fn call_experts<'a>(&self, i: usize) -> &'a [i32] {
        let mut n: usize = 0;
        let p = unsafe { (self.ops.rt_experts)(self.ops.table, i, &mut n) };
        if n == 0 || p.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(p, n) }
        }
    }
    fn host_contains(&self, key: i32) -> bool {
        unsafe { (self.ops.sc_contains)(self.w.host.cache, key) != 0 }
    }
    fn host_worth(&self, key: i32) -> bool {
        unsafe { (self.ops.sc_worth)(self.w.host.cache, key) != 0 }
    }
    fn pool_contains(&self, pool: &Pool, key: i32) -> bool {
        unsafe { (self.ops.sc_contains)(pool.cache, key) != 0 }
    }
    /// host.acquire(key, prot = all pending keys, heat, ...)
    fn host_acquire_raw(&self, key: i32, heat: f32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
        let prot = self.pend.keys();
        unsafe { (self.ops.sc_acquire)(self.w.host.cache, key, prot.as_ptr(), prot.len(), heat, vkey, vheat) }
    }
    fn enqueue(&self, job: &cq_job) {
        unsafe { (self.ops.enqueue)(job) }
    }

    fn insert_pend(&mut self, key: i32, p: Box<Pending>) {
        if let Some(old) = self.pend.insert(key, p) {
            if !done(&old.state) {
                self.dead_pend.push(old);
            }
        }
    }
    fn insert_arr(&mut self, key: i32, a: Box<Arrival>) {
        if let Some(old) = self.arr.insert(key, a) {
            if !done(&old.state) {
                self.dead_arr.push(old);
            }
        }
    }

    // ---- pend_take, arr_wait

    pub fn pend_take(&mut self, key: i32) {
        let Some(p) = self.pend.get(key) else { return };
        if !done(&p.state) {
            let t0 = Instant::now();
            spin_until_set(&p.state);
            self.st.t_pwait += t0.elapsed().as_secs_f64();
            self.st.pf_waited += 1;
        }
        self.st.pf_used += 1;
        self.pend.remove(key);
    }

    pub fn arr_wait(&mut self, key: i32) {
        let Some(a) = self.arr.remove(key) else { return };
        if !done(&a.state) {
            let t0 = Instant::now();
            spin_until_set(&a.state);
            self.st.t_vwait += t0.elapsed().as_secs_f64();
            self.st.vq_waited += 1;
        }
    }

    pub fn host_acquire(&self, key: i32, heat: f32, vkey: *mut c_int, vheat: *mut f32) -> *mut u8 {
        self.host_acquire_raw(key, heat, vkey, vheat)
    }

    // ---- route_hook

    pub fn route_hook(&mut self, layer: i32, uniq: &[i32]) {
        unsafe { (self.ops.rt_set_current)(self.ops.table, self.st.pos as c_long) };
        if self.st.rc_ok != 0 {
            let pos = self.st.pos;
            if pos >= self.calls() || self.call_layer(pos) != layer || self.call_experts(pos) != uniq {
                write_stderr(&mismatch_message(pos, layer));
                self.st.rc_ok = 0;
                unsafe { (self.ops.sc_set_next_use)(None) };
            } else {
                let t0 = Instant::now();
                if self.cfg.vq != 0 && (pos as c_long) >= self.cfg.vq_from {
                    self.rc_plan_vq(pos, uniq);
                } else {
                    self.rc_plan(pos);
                }
                self.st.t_plan += t0.elapsed().as_secs_f64();
            }
        }
        self.st.pos += 1;
    }

    fn last_call(&self, pos: usize) -> usize {
        self.calls().min(pos.wrapping_add(self.cfg.lookahead as usize).wrapping_add(1))
    }

    /// The read-ahead into the RAM cache: slot from the host cache, pending entry, job kind 0.
    fn host_read_ahead(&mut self, key: i32, e: i32, fd: i32) {
        let mut vk: c_int = -1;
        let mut vh: f32 = 0.0;
        let slot = self.host_acquire_raw(key, 1.0, &mut vk, &mut vh);
        let p = Box::new(Pending { state: AtomicI32::new(0), slot });
        let p_state = p.state.as_ptr();
        self.insert_pend(key, p);
        self.st.planned += 1;
        let mut j = cq_job::ZERO;
        j.fd = fd;
        j.off = (e as i64).wrapping_mul(self.cfg.expert_raw as i64);
        j.slot = slot;
        j.p_state = p_state;
        self.enqueue(&j);
    }

    // ---- rc_plan

    fn rc_plan(&mut self, pos: usize) {
        let last = self.last_call(pos);
        let mut i = self.st.planned_to.max(pos);
        while i < last {
            let layer = self.call_layer(i);
            let hd = match self.layers.get(layer) {
                Some(h) if h.fd >= 0 => h,
                _ => {
                    self.st.planned_to = i + 1;
                    i += 1;
                    continue;
                }
            };
            let pool = self.w.pools[hd.card];
            for &e in self.call_experts(i) {
                let key = self.ekey(layer, e);
                if self.pend.contains(key) {
                    continue;
                }
                if (pool.ready && self.pool_contains(&pool, key)) || self.host_contains(key) {
                    self.st.skipped += 1;
                    continue;
                }
                if self.pend.len() >= self.cfg.pend_cap {
                    self.st.planned_to = i;
                    return;
                }
                if !self.host_worth(key) {
                    self.st.planned_to = i;
                    return;
                }
                self.host_read_ahead(key, e, hd.fd);
            }
            self.st.planned_to = i + 1;
            i += 1;
        }
    }

    // ---- the card queues

    fn vq_sweep(&mut self) {
        let mut i = 0;
        while i < self.rel.len() {
            if done(&self.rel[i].state) {
                let slot = self.rel[i].slot;
                unsafe { (self.ops.sc_release)(self.w.host.cache, slot) };
                self.rel.swap_remove(i);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.wb_keys.len() {
            let key = self.wb_keys[i];
            match self.pend.get(key) {
                None => {
                    self.wb_keys.swap_remove(i);
                }
                Some(p) if done(&p.state) => {
                    self.pend.remove(key);
                    self.wb_keys.swap_remove(i);
                }
                Some(_) => i += 1,
            }
        }
        self.dead_pend.retain(|p| !done(&p.state));
        self.dead_arr.retain(|a| !done(&a.state));
    }

    /// false if the pool takes nothing more for this expert now
    fn vq_issue(&mut self, pool: &Pool, layer: i32, fd: i32, key: i32, e: i32, cand_nu: i32, prot: &[i32]) -> bool {
        let in_ram = self.host_contains(key);
        if in_ram {
            if let Some(p) = self.pend.get(key) {
                if p.state.load(Ordering::Acquire) == 0 {
                    return false;
                }
            }
        }
        let mut vk: c_int = -1;
        let mut vh: f32 = 0.0;
        let dst = unsafe { (self.ops.sc_try_acquire)(pool.cache, key, cand_nu, prot.as_ptr(), prot.len(), 1.0, &mut vk, &mut vh) };
        if dst.is_null() {
            return false;
        }
        let mut j = cq_job::ZERO;
        j.dev = pool.dev;
        j.dst = dst;
        j.fd = fd;
        j.off = (e as i64).wrapping_mul(self.cfg.expert_raw as i64);
        j.layer = layer;
        if in_ram {
            self.pend_take(key);
            j.host_src = unsafe { (self.ops.sc_remove)(self.w.host.cache, key) };
            j.kind = 2;
            let r = Box::new(Rel { state: AtomicI32::new(0), slot: j.host_src as *mut u8 });
            j.rel_state = r.state.as_ptr();
            self.rel.push(r);
            self.st.vq_ram += 1;
        } else {
            j.kind = 1;
            self.st.vq_drive += 1;
        }
        if vk >= 0 {
            self.arr_wait(vk); // the held expert was itself still on its way
            let cap = self.w.host.cap_slots;
            if self.cfg.prefill == 0 && cap != 0 && self.pend.len() < cap.wrapping_mul(8) / 10 && self.host_worth(vk) {
                let mut dk: c_int = -1;
                let mut dh: f32 = 0.0;
                j.victim_host = self.host_acquire_raw(vk, vh, &mut dk, &mut dh);
                let wb = Box::new(Pending { state: AtomicI32::new(0), slot: j.victim_host });
                j.wb_state = wb.state.as_ptr();
                self.insert_pend(vk, wb);
                self.wb_keys.push(vk);
                self.st.vq_wb += 1;
            } else {
                self.st.vq_nowb += 1;
            }
        }
        let a = Box::new(Arrival { state: AtomicI32::new(0) });
        j.arr_state = a.state.as_ptr();
        self.insert_arr(key, a);
        self.enqueue(&j);
        true
    }

    fn rc_plan_vq(&mut self, pos: usize, cur: &[i32]) {
        self.vq_sweep();
        let pos_layer = self.call_layer(pos);
        let prot: Vec<i32> = cur.iter().map(|&e| self.ekey(pos_layer, e)).collect();
        let mut out = self.arr.values().filter(|a| a.state.load(Ordering::Acquire) == 0).count();
        let mut full = [false; MAX_CARDS];
        let last = self.last_call(pos);
        for i in pos..last {
            let layer = self.call_layer(i);
            let hd = match self.layers.get(layer) {
                Some(h) if h.fd >= 0 => h,
                _ => continue,
            };
            let pool = self.w.pools[hd.card];
            for &e in self.call_experts(i) {
                let key = self.ekey(layer, e);
                if pool.ready && self.pool_contains(&pool, key) {
                    continue; // on the card or on its way
                }
                if pool.ready && !full[hd.card] && out < self.cfg.vq_cap {
                    let cand = if i == pos { pos as c_int } else { unsafe { (self.ops.rt_next_use)(self.ops.table, key) } };
                    if self.vq_issue(&pool, layer, hd.fd, key, e, cand, &prot) {
                        out += 1;
                        continue;
                    }
                    full[hd.card] = true;
                }
                if self.pend.contains(key) || self.host_contains(key) {
                    continue;
                }
                if self.pend.len() >= self.cfg.pend_cap || !self.host_worth(key) {
                    continue;
                }
                self.host_read_ahead(key, e, hd.fd);
            }
        }
    }
}
