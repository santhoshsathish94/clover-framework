//! Rust implementation of include/slotcache.h. Behaviour reference: `struct SlotCache` and the globals `g_policy` / `g_nu_fn` in src/code/gpu/decode.cuh.
//! Not thread-safe per cache (as the contract says); the process-wide policy and next-use function are atomics.
#![allow(non_camel_case_types)]

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

#[cfg(test)]
mod tests;

pub type sc_next_use_fn = Option<extern "C" fn(i32) -> i32>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct sc_entry {
    pub key: i32,
    pub slot: *mut u8,
    pub heat: f32,
    pub nu: i32,
}

const FATAL_MSG: &[u8] = b"expert cache: every slot is in use by one call (too few slots)\n";

static POLICY: AtomicI32 = AtomicI32::new(1);
// 0 = unset, else an `extern "C" fn(i32) -> i32` address
static NU_FN: AtomicUsize = AtomicUsize::new(0);

fn belady() -> bool {
    POLICY.load(Ordering::SeqCst) == 2 && NU_FN.load(Ordering::SeqCst) != 0
}

fn nu_call(key: i32) -> i32 {
    let v = NU_FN.load(Ordering::SeqCst);
    if v == 0 {
        // the original would call a null pointer here; only reachable if a callback unsets itself mid-operation
        return 0;
    }
    // SAFETY: a non-zero value was stored by sc_set_next_use from a valid `extern "C" fn(i32) -> i32`.
    let f: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute::<usize, extern "C" fn(i32) -> i32>(v) };
    f(key)
}

// eprintln! can panic on a closed stderr; write_all errors are ignored.
fn fatal() -> ! {
    let _ = std::io::stderr().write_all(FATAL_MSG);
    std::process::exit(1)
}

const NIL: usize = usize::MAX;

struct Node {
    key: i32,
    slot: *mut u8,
    heat: f32,
    nu: i32,
    prev: usize,
    next: usize,
}

pub struct sc_cache {
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize, // most recently used
    tail: usize, // least recently used
    map: HashMap<i32, usize>,
    by_next: BTreeSet<(i32, i32)>,
    spare: Vec<*mut u8>,
    prot_bits: Vec<u64>, // membership bitmap for a large `prot`, allocated on first use, all zero between calls
}

// `prot` above this many keys, all in [0, PROT_BITS), is tested through the bitmap: the heat rule asks about up to 48 + |prot| keys per acquire, the original's hash set answers in O(1)
const PROT_LINEAR_MAX: usize = 32;
const PROT_BITS: usize = 1 << 20;

struct Acquired {
    slot: *mut u8,
    vkey: i32,
    vheat: Option<f32>,
}

impl sc_cache {
    fn new() -> Self {
        sc_cache { nodes: Vec::new(), free: Vec::new(), head: NIL, tail: NIL, map: HashMap::new(), by_next: BTreeSet::new(), spare: Vec::new(), prot_bits: Vec::new() }
    }

    // true if `prot` is now in the bitmap (the caller must call prot_clear with the same list)
    fn prot_prepare(&mut self, prot: &[i32]) -> bool {
        if prot.len() <= PROT_LINEAR_MAX || !prot.iter().all(|&k| k >= 0 && (k as usize) < PROT_BITS) {
            return false;
        }
        if self.prot_bits.is_empty() {
            self.prot_bits = vec![0u64; PROT_BITS / 64];
        }
        for &k in prot {
            self.prot_bits[(k as usize) >> 6] |= 1u64 << (k & 63);
        }
        true
    }

    fn prot_clear(&mut self, prot: &[i32]) {
        for &k in prot {
            self.prot_bits[(k as usize) >> 6] = 0;
        }
    }

    fn is_prot(&self, prot: &[i32], bitmap: bool, k: i32) -> bool {
        if bitmap {
            // every key of `prot` is in range, so an out-of-range k is not in it
            k >= 0 && (k as usize) < PROT_BITS && (self.prot_bits[(k as usize) >> 6] >> (k & 63)) & 1 != 0
        } else {
            prot.contains(&k)
        }
    }

    fn unlink(&mut self, i: usize) {
        let (p, n) = (self.nodes[i].prev, self.nodes[i].next);
        if p != NIL { self.nodes[p].next = n } else { self.head = n }
        if n != NIL { self.nodes[n].prev = p } else { self.tail = p }
    }

    fn push_front(&mut self, i: usize) {
        self.nodes[i].prev = NIL;
        self.nodes[i].next = self.head;
        if self.head != NIL { self.nodes[self.head].prev = i } else { self.tail = i }
        self.head = i;
    }

    fn alloc(&mut self, n: Node) -> usize {
        match self.free.pop() {
            Some(i) => {
                self.nodes[i] = n;
                i
            }
            None => {
                self.nodes.push(n);
                self.nodes.len() - 1
            }
        }
    }

    fn find(&mut self, key: i32) -> *mut u8 {
        let Some(&i) = self.map.get(&key) else { return std::ptr::null_mut() };
        self.unlink(i);
        self.push_front(i);
        self.nodes[i].slot
    }

    fn touch(&mut self, key: i32, w: f32) {
        let Some(&i) = self.map.get(&key) else { return };
        if belady() {
            let old = self.nodes[i].nu;
            self.by_next.remove(&(old, key));
            let nu = nu_call(key);
            self.nodes[i].nu = nu;
            self.by_next.insert((nu, key));
        } else {
            self.nodes[i].heat = self.nodes[i].heat * 0.8f32 + w;
        }
    }

    fn worth(&self, key: i32) -> bool {
        if !belady() || !self.spare.is_empty() || self.by_next.is_empty() {
            return true;
        }
        let largest = self.by_next.last().map(|p| p.0).unwrap_or(0);
        nu_call(key) < largest
    }

    // removes the node `i` of `key` from list, map and (if present) by_next; returns its slot
    fn detach(&mut self, i: usize, key: i32) -> *mut u8 {
        let (slot, nu) = (self.nodes[i].slot, self.nodes[i].nu);
        self.by_next.remove(&(nu, key));
        self.unlink(i);
        self.free.push(i);
        self.map.remove(&key);
        slot
    }

    fn acquire(&mut self, key: i32, prot: &[i32], heat0: f32) -> Acquired {
        let bitmap = self.prot_prepare(prot);
        let a = self.acquire_inner(key, prot, bitmap, heat0);
        if bitmap {
            self.prot_clear(prot);
        }
        a
    }

    fn acquire_inner(&mut self, key: i32, prot: &[i32], bitmap: bool, heat0: f32) -> Acquired {
        let mut vkey = -1;
        let mut vheat = None;
        let p;
        if let Some(s) = self.spare.pop() {
            p = s;
        } else if belady() {
            // largest (next use, key) first; the original's sentinel is -1, so a negative key is never a victim
            let mut victim = -1;
            for &(_, k) in self.by_next.iter().rev() {
                if !self.is_prot(prot, bitmap, k) {
                    victim = k;
                    break;
                }
            }
            if victim < 0 {
                fatal();
            }
            let i = *self.map.get(&victim).expect("by_next key not in map (outside the contract)");
            vkey = victim;
            vheat = Some(self.nodes[i].heat);
            p = self.detach(i, victim);
        } else {
            let limit = if POLICY.load(Ordering::SeqCst) != 0 { 48 } else { 1 };
            let mut best = NIL;
            let mut best_heat = 0f32;
            let mut seen = 0;
            let mut it = self.tail;
            while it != NIL && seen < limit {
                let n = &self.nodes[it];
                if !self.is_prot(prot, bitmap, n.key) {
                    // strict <: on a tie the earlier one in the scan stays
                    if best == NIL || n.heat < best_heat {
                        best = it;
                        best_heat = n.heat;
                    }
                    seen += 1;
                }
                it = n.prev;
            }
            if best == NIL {
                fatal();
            }
            let victim = self.nodes[best].key;
            vkey = victim;
            vheat = Some(self.nodes[best].heat);
            p = self.detach(best, victim); // erases {stored nu, victim} from by_next; a no-op when it is not there
        }
        // outside the contract (key already held): drop the old entry so the structure stays consistent
        if let Some(&old) = self.map.get(&key) {
            self.detach(old, key);
        }
        let mut nu = 0;
        if belady() {
            nu = nu_call(key);
            self.by_next.insert((nu, key));
        }
        let i = self.alloc(Node { key, slot: p, heat: heat0, nu, prev: NIL, next: NIL });
        self.push_front(i);
        self.map.insert(key, i);
        Acquired { slot: p, vkey, vheat }
    }

    fn try_acquire(&mut self, key: i32, cand_nu: i32, prot: &[i32], heat0: f32) -> Option<Acquired> {
        let bitmap = self.prot_prepare(prot);
        let r = self.try_acquire_inner(key, cand_nu, prot, bitmap, heat0);
        if bitmap {
            self.prot_clear(prot);
        }
        r
    }

    fn try_acquire_inner(&mut self, key: i32, cand_nu: i32, prot: &[i32], bitmap: bool, heat0: f32) -> Option<Acquired> {
        if self.spare.is_empty() {
            // looks at by_next whether or not belady() is true; acquire() decides again
            let mut victim = -1;
            for &(nu, k) in self.by_next.iter().rev() {
                if !self.is_prot(prot, bitmap, k) {
                    if nu > cand_nu {
                        victim = k;
                    }
                    break;
                }
            }
            if victim < 0 {
                return None;
            }
        }
        Some(self.acquire_inner(key, prot, bitmap, heat0))
    }

    fn remove(&mut self, key: i32) -> *mut u8 {
        let Some(&i) = self.map.get(&key) else { return std::ptr::null_mut() };
        self.detach(i, key)
    }
}

/// # Safety
/// `prot` must be NULL or point to `n` readable ints.
unsafe fn prot_slice<'a>(prot: *const i32, n: usize) -> &'a [i32] {
    if prot.is_null() || n == 0 {
        &[]
    } else {
        // SAFETY: caller contract
        unsafe { std::slice::from_raw_parts(prot, n) }
    }
}

#[no_mangle]
pub extern "C" fn sc_set_policy(policy: i32) {
    POLICY.store(policy, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn sc_set_next_use(f: sc_next_use_fn) {
    let v = match f {
        Some(f) => f as usize,
        None => 0,
    };
    NU_FN.store(v, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn sc_new() -> *mut sc_cache {
    Box::into_raw(Box::new(sc_cache::new()))
}

/// # Safety
/// `c` must be NULL or a pointer from `sc_new` not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sc_free(c: *mut sc_cache) {
    if !c.is_null() {
        // SAFETY: caller contract
        drop(unsafe { Box::from_raw(c) });
    }
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_add_spare(c: *mut sc_cache, slot: *mut u8) {
    // SAFETY: caller contract
    unsafe { &mut *c }.spare.push(slot);
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_release(c: *mut sc_cache, slot: *mut u8) {
    // SAFETY: caller contract
    unsafe { &mut *c }.spare.push(slot);
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_contains(c: *const sc_cache, key: i32) -> i32 {
    // SAFETY: caller contract
    unsafe { &*c }.map.contains_key(&key) as i32
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_size(c: *const sc_cache) -> usize {
    // SAFETY: caller contract
    unsafe { &*c }.map.len()
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_find(c: *mut sc_cache, key: i32) -> *mut u8 {
    // SAFETY: caller contract
    unsafe { &mut *c }.find(key)
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_touch(c: *mut sc_cache, key: i32, w: f32) {
    // SAFETY: caller contract
    unsafe { &mut *c }.touch(key, w)
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_worth(c: *const sc_cache, key: i32) -> i32 {
    // SAFETY: caller contract
    unsafe { &*c }.worth(key) as i32
}

unsafe fn report(a: &Acquired, vkey: *mut i32, vheat: *mut f32) {
    // SAFETY: the contract passes writable pointers; NULL is tolerated and skipped.
    unsafe {
        if !vkey.is_null() {
            *vkey = a.vkey;
        }
        if let Some(h) = a.vheat {
            if !vheat.is_null() {
                *vheat = h;
            }
        }
    }
}

/// # Safety
/// `c` live; `prot` points to `nprot` ints (or NULL); `vkey` / `vheat` writable.
#[no_mangle]
pub unsafe extern "C" fn sc_acquire(c: *mut sc_cache, key: i32, prot: *const i32, nprot: usize, heat0: f32, vkey: *mut i32, vheat: *mut f32) -> *mut u8 {
    // SAFETY: caller contract
    let (cache, prot) = unsafe { (&mut *c, prot_slice(prot, nprot)) };
    let a = cache.acquire(key, prot, heat0);
    // SAFETY: caller contract
    unsafe { report(&a, vkey, vheat) };
    a.slot
}

/// # Safety
/// As `sc_acquire`.
#[no_mangle]
pub unsafe extern "C" fn sc_try_acquire(
    c: *mut sc_cache, key: i32, cand_nu: i32, prot: *const i32, nprot: usize, heat0: f32, vkey: *mut i32, vheat: *mut f32,
) -> *mut u8 {
    // SAFETY: caller contract
    let (cache, prot) = unsafe { (&mut *c, prot_slice(prot, nprot)) };
    // SAFETY: caller contract
    unsafe {
        if !vkey.is_null() {
            *vkey = -1;
        }
    }
    match cache.try_acquire(key, cand_nu, prot, heat0) {
        Some(a) => {
            // SAFETY: caller contract
            unsafe { report(&a, vkey, vheat) };
            a.slot
        }
        None => std::ptr::null_mut(),
    }
}

/// # Safety
/// `c` must be a live cache from `sc_new`.
#[no_mangle]
pub unsafe extern "C" fn sc_remove(c: *mut sc_cache, key: i32) -> *mut u8 {
    // SAFETY: caller contract
    unsafe { &mut *c }.remove(key)
}

/// # Safety
/// `c` live; `out` writable for `cap` records (may be NULL when cap is 0).
#[no_mangle]
pub unsafe extern "C" fn sc_snapshot_lru(c: *const sc_cache, out: *mut sc_entry, cap: usize) -> usize {
    // SAFETY: caller contract
    let c = unsafe { &*c };
    let (mut i, mut n) = (c.head, 0usize);
    while i != NIL {
        if n < cap {
            let nd = &c.nodes[i];
            // SAFETY: n < cap
            unsafe { out.add(n).write(sc_entry { key: nd.key, slot: nd.slot, heat: nd.heat, nu: nd.nu }) };
        }
        n += 1;
        i = c.nodes[i].next;
    }
    n
}

/// # Safety
/// `c` live; `out` writable for `cap` pointers (may be NULL when cap is 0).
#[no_mangle]
pub unsafe extern "C" fn sc_snapshot_spare(c: *const sc_cache, out: *mut *mut u8, cap: usize) -> usize {
    // SAFETY: caller contract
    let c = unsafe { &*c };
    for (n, &s) in c.spare.iter().take(cap).enumerate() {
        // SAFETY: n < cap
        unsafe { out.add(n).write(s) };
    }
    c.spare.len()
}

/// # Safety
/// `c` live; `nu_out` and `key_out` writable for `cap` ints each (may be NULL when cap is 0).
#[no_mangle]
pub unsafe extern "C" fn sc_snapshot_next(c: *const sc_cache, nu_out: *mut i32, key_out: *mut i32, cap: usize) -> usize {
    // SAFETY: caller contract
    let c = unsafe { &*c };
    for (n, &(nu, k)) in c.by_next.iter().take(cap).enumerate() {
        // SAFETY: n < cap
        unsafe {
            nu_out.add(n).write(nu);
            key_out.add(n).write(k);
        }
    }
    c.by_next.len()
}
