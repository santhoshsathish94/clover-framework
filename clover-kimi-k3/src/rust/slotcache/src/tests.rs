//! Hand-derived tests. The policy and next-use function are process-wide, so every test holds LOCK through `Env`.
use super::*;
use std::process::Command;
use std::sync::atomic::AtomicUsize;
use std::sync::{Mutex, MutexGuard};

static LOCK: Mutex<()> = Mutex::new(());
static NU: [AtomicI32; 64] = [const { AtomicI32::new(i32::MAX) }; 64];
static NU_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "C" fn nu_script(key: i32) -> i32 {
    NU_CALLS.fetch_add(1, Ordering::SeqCst);
    NU[key as usize].load(Ordering::SeqCst)
}

fn nu_set(key: usize, v: i32) {
    NU[key].store(v, Ordering::SeqCst);
}

fn calls() -> usize {
    NU_CALLS.load(Ordering::SeqCst)
}

struct Env(#[allow(dead_code)] MutexGuard<'static, ()>);

fn reset(policy: i32) {
    sc_set_policy(policy);
    sc_set_next_use(None);
    for n in NU.iter() {
        n.store(i32::MAX, Ordering::SeqCst);
    }
    NU_CALLS.store(0, Ordering::SeqCst);
}

impl Env {
    fn new(policy: i32) -> Env {
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset(policy);
        Env(g)
    }
}

impl Drop for Env {
    fn drop(&mut self) {
        reset(1);
    }
}

fn use_fn() {
    sc_set_next_use(Some(nu_script));
}

fn s(n: usize) -> *mut u8 {
    (0x1000 + 0x100 * n) as *mut u8 // never dereferenced
}
const A: usize = 1;
const B: usize = 2;
const C: usize = 3;

const SENTINEL_BITS: u32 = 0x7fc0_1234;

struct Acq {
    slot: *mut u8,
    vkey: i32,
    vheat: Option<f32>,
}

struct T(*mut sc_cache);

impl Drop for T {
    fn drop(&mut self) {
        unsafe { sc_free(self.0) }
    }
}

impl T {
    fn new() -> T {
        T(sc_new())
    }
    fn spares(&self, order: &[usize]) {
        for &n in order {
            unsafe { sc_add_spare(self.0, s(n)) }
        }
    }
    fn acq(&self, key: i32, prot: &[i32], heat0: f32) -> Acq {
        let mut vk = 12345;
        let mut vh = f32::from_bits(SENTINEL_BITS);
        let p = unsafe { sc_acquire(self.0, key, prot.as_ptr(), prot.len(), heat0, &mut vk, &mut vh) };
        Acq { slot: p, vkey: vk, vheat: (vh.to_bits() != SENTINEL_BITS).then_some(vh) }
    }
    fn try_acq(&self, key: i32, cand: i32, prot: &[i32], heat0: f32) -> Acq {
        let mut vk = 12345;
        let mut vh = f32::from_bits(SENTINEL_BITS);
        let p = unsafe { sc_try_acquire(self.0, key, cand, prot.as_ptr(), prot.len(), heat0, &mut vk, &mut vh) };
        Acq { slot: p, vkey: vk, vheat: (vh.to_bits() != SENTINEL_BITS).then_some(vh) }
    }
    fn find(&self, k: i32) -> *mut u8 {
        unsafe { sc_find(self.0, k) }
    }
    fn touch(&self, k: i32, w: f32) {
        unsafe { sc_touch(self.0, k, w) }
    }
    fn worth(&self, k: i32) -> i32 {
        unsafe { sc_worth(self.0, k) }
    }
    fn remove(&self, k: i32) -> *mut u8 {
        unsafe { sc_remove(self.0, k) }
    }
    fn contains(&self, k: i32) -> i32 {
        unsafe { sc_contains(self.0, k) }
    }
    fn size(&self) -> usize {
        unsafe { sc_size(self.0) }
    }
    fn lru(&self) -> Vec<sc_entry> {
        let n = unsafe { sc_snapshot_lru(self.0, std::ptr::null_mut(), 0) };
        let mut v = vec![sc_entry { key: 0, slot: std::ptr::null_mut(), heat: 0.0, nu: 0 }; n];
        assert_eq!(unsafe { sc_snapshot_lru(self.0, v.as_mut_ptr(), n) }, n);
        v
    }
    fn keys(&self) -> Vec<i32> {
        self.lru().iter().map(|e| e.key).collect()
    }
    fn entry(&self, k: i32) -> sc_entry {
        *self.lru().iter().find(|e| e.key == k).expect("key not held")
    }
    fn spare_list(&self) -> Vec<*mut u8> {
        let n = unsafe { sc_snapshot_spare(self.0, std::ptr::null_mut(), 0) };
        let mut v = vec![std::ptr::null_mut(); n];
        assert_eq!(unsafe { sc_snapshot_spare(self.0, v.as_mut_ptr(), n) }, n);
        v
    }
    fn next_set(&self) -> Vec<(i32, i32)> {
        let n = unsafe { sc_snapshot_next(self.0, std::ptr::null_mut(), std::ptr::null_mut(), 0) };
        let (mut nu, mut k) = (vec![0; n], vec![0; n]);
        assert_eq!(unsafe { sc_snapshot_next(self.0, nu.as_mut_ptr(), k.as_mut_ptr(), n) }, n);
        nu.into_iter().zip(k).collect()
    }
}

fn heat_after(h: f32, w: f32) -> f32 {
    h * 0.8f32 + w
}

#[test]
fn policy0_evicts_least_recently_used() {
    let _e = Env::new(0);
    let c = T::new();
    c.spares(&[C, B, A]); // last in, first out: key 1 gets A, 2 gets B, 3 gets C
    let a1 = c.acq(1, &[], 1.0);
    assert_eq!((a1.slot, a1.vkey, a1.vheat), (s(A), -1, None));
    assert_eq!(c.acq(2, &[], 1.0).slot, s(B));
    assert_eq!(c.acq(3, &[], 1.0).slot, s(C));
    assert_eq!(c.keys(), [3, 2, 1]);
    assert_eq!((c.size(), c.contains(2), c.contains(9)), (3, 1, 0));
    assert!(c.spare_list().is_empty());
    assert_eq!(c.find(1), s(A));
    assert_eq!(c.keys(), [1, 3, 2]);
    assert!(c.find(99).is_null());
    assert_eq!(c.keys(), [1, 3, 2]);
    let a4 = c.acq(4, &[], 1.0);
    assert_eq!((a4.slot, a4.vkey, a4.vheat), (s(B), 2, Some(1.0)));
    assert_eq!(c.keys(), [4, 1, 3]);
    assert_eq!((c.size(), c.contains(2), c.contains(4)), (3, 0, 1));
    assert!(c.next_set().is_empty());
    assert_eq!(c.entry(4).nu, 0);
}

#[test]
fn policy0_ignores_heat_and_prot_does_not_count_toward_the_window_of_one() {
    let _e = Env::new(0);
    let c = T::new();
    c.spares(&[C, B, A]);
    c.acq(1, &[], 9.0);
    c.acq(2, &[], 1.0);
    c.acq(3, &[], 5.0); // lru: 3 2 1
    let a4 = c.acq(4, &[], 2.0);
    assert_eq!((a4.vkey, a4.vheat, a4.slot), (1, Some(9.0), s(A))); // hottest, but least recently used
    assert_eq!(c.keys(), [4, 3, 2]);
    let a5 = c.acq(5, &[2, 2, 2], 2.0); // tail 2 is prot (duplicates allowed): skipped, then 3
    assert_eq!((a5.vkey, a5.vheat), (3, Some(5.0)));
    assert_eq!(c.keys(), [5, 4, 2]);
}

#[test]
fn policy1_lowest_heat_ties_keep_earlier_in_scan_and_prot_is_skipped() {
    let _e = Env::new(1);
    let c = T::new();
    c.spares(&[5, 4, 3, 2, 1]); // keys 10..14 get slots 1..5
    for (k, h) in [(10, 4.0), (11, 2.0), (12, 2.0), (13, 9.0), (14, 9.0)] {
        c.acq(k, &[], h);
    }
    assert_eq!(c.keys(), [14, 13, 12, 11, 10]);
    assert!(c.spare_list().is_empty());
    let a = c.acq(20, &[], 6.0); // scan from the tail: 10(4) 11(2) 12(2): 12 is not strictly lower, 11 stays
    assert_eq!((a.vkey, a.vheat), (11, Some(2.0)));
    assert_eq!(a.slot, s(2)); // 11 was acquired second and got slot 2
    assert_eq!(c.keys(), [20, 14, 13, 12, 10]);
    let b = c.acq(21, &[12], 1.0); // without prot the victim would be 12 (2.0); with it, 10 (4.0)
    assert_eq!((b.vkey, b.vheat), (10, Some(4.0)));
    assert_eq!(c.keys(), [21, 20, 14, 13, 12]);
}

#[test]
fn policy1_touch_is_single_precision_heat_decay_and_changes_the_victim() {
    let _e = Env::new(1);
    let c = T::new();
    c.spares(&[B, A]);
    c.acq(1, &[], 1.0); // tail, lowest heat before the touches
    c.acq(2, &[], 3.0);
    c.touch(1, 5.0);
    c.touch(2, 0.0);
    c.touch(77, 1.0); // absent: no effect
    assert_eq!(c.entry(1).heat.to_bits(), heat_after(1.0, 5.0).to_bits());
    assert!((c.entry(1).heat - 5.8).abs() < 1e-6);
    assert_eq!(c.entry(2).heat.to_bits(), (3.0f32 * 0.8f32).to_bits()); // 2.4000000953674316
    assert_eq!(c.keys(), [2, 1]); // touch does not reorder the lru list
    let a = c.acq(3, &[], 1.0);
    assert_eq!(a.vkey, 2);
    assert_eq!(a.vheat.map(f32::to_bits), Some((3.0f32 * 0.8f32).to_bits()));
}

#[test]
fn policy1_window_is_48_entries_not_in_prot() {
    let _e = Env::new(1);
    let c = T::new();
    let n = 52usize;
    for i in (0..n).rev() {
        unsafe { sc_add_spare(c.0, s(i)) };
    }
    // key i acquired in order 0..52: key 0 is the least recently used
    for k in 0..n as i32 {
        let h = match k {
            51 => 0.1,
            50 => 2.0,
            47 => 3.0,
            _ => 10.0,
        };
        c.acq(k, &[], h);
    }
    // no prot: the window is keys 0..48, so key 47 (3.0) wins over the colder 50 and 51 outside it
    let a = c.acq(100, &[], 10.0);
    assert_eq!((a.vkey, a.vheat), (47, Some(3.0)));
    // from the tail now: 0..=46, 48..=51, 100. With 0,1,2 in prot the window is 3..=46, 48..=51 (48 keys, 100 is outside)
    // and 51 is reachable only because the prot entries were not counted toward the 48.
    let b = c.acq(101, &[0, 1, 2], 10.0);
    assert_eq!((b.vkey, b.vheat), (51, Some(0.1))); // 51 is inside the window only because prot entries were not counted
}

#[test]
fn belady_scripted_victims_and_ordered_set() {
    let _e = Env::new(2);
    use_fn();
    for (k, v) in [(1, 10), (2, 30), (3, 20), (4, 15), (5, 5)] {
        nu_set(k, v);
    }
    let c = T::new();
    c.spares(&[C, B, A]);
    c.acq(1, &[], 1.0);
    c.acq(2, &[], 2.0);
    c.acq(3, &[], 3.0);
    assert_eq!(c.next_set(), [(10, 1), (20, 3), (30, 2)]);
    assert_eq!(c.entry(2).nu, 30);
    let a = c.acq(4, &[], 0.5);
    assert_eq!((a.vkey, a.vheat, a.slot), (2, Some(2.0), s(B)));
    assert_eq!(c.next_set(), [(10, 1), (15, 4), (20, 3)]);
    assert_eq!((c.entry(4).nu, c.entry(4).heat), (15, 0.5));
    assert_eq!(c.keys(), [4, 3, 1]);
    let b = c.acq(5, &[3], 0.5); // largest (20,3) is prot: skipped, victim is (15,4)
    assert_eq!((b.vkey, b.vheat), (4, Some(0.5)));
    assert_eq!(c.next_set(), [(5, 5), (10, 1), (20, 3)]);
    nu_set(1, 25);
    c.touch(1, 99.0); // belady: replaces the next use, heat untouched
    assert_eq!(c.next_set(), [(5, 5), (20, 3), (25, 1)]);
    assert_eq!((c.entry(1).nu, c.entry(1).heat), (25, 1.0));
}

#[test]
fn belady_equal_next_use_takes_the_larger_key() {
    let _e = Env::new(2);
    use_fn();
    nu_set(7, 50);
    nu_set(8, 50);
    nu_set(9, 1);
    let c = T::new();
    c.spares(&[B, A]);
    c.acq(7, &[], 1.0);
    c.acq(8, &[], 1.0);
    assert_eq!(c.next_set(), [(50, 7), (50, 8)]);
    let a = c.acq(9, &[], 1.0);
    assert_eq!(a.vkey, 8);
    assert_eq!(c.next_set(), [(1, 9), (50, 7)]);
}

#[test]
fn worth_in_all_four_cases() {
    let _e = Env::new(2);
    use_fn();
    nu_set(1, 10);
    nu_set(2, 30);
    let c = T::new();
    c.spares(&[B, A]);
    // (c) belady, no spare, ordered set empty: yes, without calling the function
    let empty = T::new();
    assert_eq!((empty.worth(5), calls()), (1, 0));
    c.acq(1, &[], 1.0);
    c.acq(2, &[], 1.0);
    // (d) belady, no spare, set non-empty: next_use(key) < largest stored (30), strictly
    nu_set(20, 29);
    nu_set(21, 30);
    nu_set(22, 31);
    assert_eq!((c.worth(20), c.worth(21), c.worth(22)), (1, 0, 0));
    // (b) a spare exists: yes, function not called
    c.spares(&[C]);
    let before = calls();
    nu_set(23, 99);
    assert_eq!((c.worth(23), calls()), (1, before));
    nu_set(3, 5);
    unsafe { sc_acquire(c.0, 3, std::ptr::null(), 0, 1.0, &mut 0, &mut 0.0) }; // uses the spare again; set is (5,3) (10,1) (30,2)
    nu_set(24, 99);
    assert_eq!(c.worth(24), 0);
    // (a) not belady: policy 1 with the same state, or policy 2 with no function
    sc_set_policy(1);
    let before = calls();
    assert_eq!((c.worth(24), calls()), (1, before));
    sc_set_policy(2);
    sc_set_next_use(None);
    assert_eq!(c.worth(24), 1);
}

#[test]
fn try_acquire_accepts_and_refuses() {
    let _e = Env::new(2);
    use_fn();
    for (k, v) in [(1, 10), (2, 40), (3, 5)] {
        nu_set(k, v);
    }
    let c = T::new();
    c.spares(&[B, A]);
    c.acq(1, &[], 1.0);
    c.acq(2, &[], 2.0);
    let (lru0, next0) = (c.keys(), c.next_set());
    // largest (40,2) is needed at 40, candidate at 50: refuse, nothing changes, vkey written as -1
    let r = c.try_acq(3, 50, &[], 1.0);
    assert!(r.slot.is_null() && r.vkey == -1 && r.vheat.is_none());
    assert_eq!((c.keys(), c.next_set()), (lru0, next0));
    // strictly later is required: cand 40 refuses, 39 accepts
    assert!(c.try_acq(3, 40, &[], 1.0).slot.is_null());
    let r = c.try_acq(3, 39, &[], 1.0);
    assert_eq!((r.vkey, r.vheat, r.slot), (2, Some(2.0), s(B)));
    assert_eq!(c.next_set(), [(5, 3), (10, 1)]);
    // prot: first non-prot from the top decides, the scan does not continue past it
    nu_set(4, 1);
    let r = c.try_acq(4, 10, &[1], 1.0); // top is (10,1) prot; next (5,3): 5 > 10 false -> refuse
    assert!(r.slot.is_null() && r.vkey == -1);
    let r = c.try_acq(4, 4, &[1], 1.0); // (5,3) > 4 -> accept, victim 3
    assert_eq!(r.vkey, 3);
    // a spare is always accepted
    c.spares(&[C]);
    let r = c.try_acq(9, 1_000_000, &[], 1.0);
    assert_eq!((r.slot, r.vkey), (s(C), -1));
}

#[test]
fn try_acquire_with_nothing_to_replace_refuses_and_precheck_ignores_belady() {
    let _e = Env::new(2);
    use_fn();
    let c = T::new();
    let r = c.try_acq(1, 5, &[], 1.0); // no spare, empty set
    assert!(r.slot.is_null() && r.vkey == -1);
    assert_eq!(c.size(), 0);

    // pre-check passes on by_next although policy is now 1; acquire() then decides with the heat rule (victim 1, not belady's 2)
    nu_set(1, 10);
    nu_set(2, 40);
    nu_set(3, 0);
    let c = T::new();
    c.spares(&[B, A]);
    c.acq(1, &[], 1.0); // tail
    c.acq(2, &[], 7.0);
    sc_set_policy(1);
    let r = c.try_acq(3, 30, &[], 2.0); // by_next top (40,2): 40 > 30 passes
    assert_eq!((r.vkey, r.vheat, r.slot), (1, Some(1.0), s(A)));
    assert_eq!(c.next_set(), [(40, 2)]); // (10,1) erased by the heat-rule removal
    assert_eq!(c.entry(3).nu, 0);
}

#[test]
fn remove_returns_slot_without_making_it_spare() {
    let _e = Env::new(2);
    use_fn();
    nu_set(1, 10);
    nu_set(2, 20);
    let c = T::new();
    c.spares(&[B, A]);
    c.acq(1, &[], 1.0);
    c.acq(2, &[], 1.0);
    assert!(c.remove(99).is_null());
    assert_eq!(c.remove(1), s(A));
    assert_eq!((c.size(), c.contains(1), c.keys()), (1, 0, vec![2]));
    assert!(c.spare_list().is_empty());
    assert_eq!(c.next_set(), [(20, 2)]);
    assert!(c.remove(1).is_null());
    assert_eq!(c.remove(2), s(B));
    assert_eq!((c.size(), c.next_set().len()), (0, 0));
    unsafe { sc_release(c.0, s(A)) }; // release == add_spare
    unsafe { sc_add_spare(c.0, s(B)) };
    assert_eq!(c.spare_list(), [s(A), s(B)]); // last record is handed out next
}

#[test]
fn find_and_touch_of_absent_key_change_nothing() {
    let _e = Env::new(2);
    use_fn();
    let c = T::new();
    c.spares(&[A]);
    nu_set(1, 3);
    c.acq(1, &[], 1.0);
    let before = (c.keys(), c.next_set(), calls());
    assert!(c.find(5).is_null());
    c.touch(5, 1.0);
    assert_eq!((c.keys(), c.next_set(), calls()), before);
}

#[test]
fn stale_by_next_after_next_use_function_is_unset() {
    let _e = Env::new(2);
    use_fn();
    for (k, v) in [(1, 10), (2, 30), (3, 20)] {
        nu_set(k, v);
    }
    let c = T::new();
    c.spares(&[C, B, A]);
    c.acq(1, &[], 5.0);
    c.acq(2, &[], 3.0);
    c.acq(3, &[], 4.0); // lru 3 2 1
    assert_eq!(c.next_set(), [(10, 1), (20, 3), (30, 2)]);
    sc_set_next_use(None);
    let n0 = calls();
    // touch: heat rule, stored next use and the set untouched
    c.touch(1, 2.0);
    assert_eq!(c.entry(1).heat.to_bits(), heat_after(5.0, 2.0).to_bits()); // 6.0
    assert_eq!(c.entry(1).nu, 10);
    assert_eq!(c.next_set().len(), 3);
    assert_eq!(c.worth(7), 1);
    // acquire: policy 2 without function = heat rule; victim 2 (heat 3.0); its pair {30,2} is erased from the stale set
    let a = c.acq(4, &[], 1.0);
    assert_eq!((a.vkey, a.vheat, a.slot), (2, Some(3.0), s(B)));
    assert_eq!(c.next_set(), [(10, 1), (20, 3)]);
    assert_eq!((c.entry(4).nu, c.keys()), (0, vec![4, 3, 1]));
    // remove erases {stored nu, key}; key 4 was never in the set (no-op)
    assert_eq!(c.remove(3), s(C));
    assert_eq!(c.next_set(), [(10, 1)]);
    assert!(!c.remove(4).is_null());
    assert_eq!((c.next_set(), c.size()), (vec![(10, 1)], 1));
    assert_eq!(calls(), n0); // the function was never called while unset
    // function back: touch replaces the stored value 10 by next_use(1)
    use_fn();
    nu_set(1, 77);
    c.touch(1, 0.0);
    assert_eq!(c.next_set(), [(77, 1)]);
    assert_eq!(c.entry(1).nu, 77);
    sc_set_next_use(None);
    assert_eq!(c.remove(1), s(A));
    assert!(c.next_set().is_empty());
}

#[test]
fn entry_acquired_while_unset_is_invisible_to_belady_until_touched() {
    let _e = Env::new(2);
    let c = T::new();
    c.spares(&[A]);
    c.acq(1, &[], 2.0); // no function: stored nu 0, nothing in the set
    use_fn();
    nu_set(1, 9);
    assert!(c.next_set().is_empty());
    assert_eq!(c.worth(1), 1); // no spare, but the set is empty
    let r = c.try_acq(2, 5, &[], 1.0);
    assert!(r.slot.is_null());
    c.touch(1, 0.0); // erase {0,1} is a no-op, then insert {9,1}
    assert_eq!(c.next_set(), [(9, 1)]);
    assert_eq!(c.entry(1).nu, 9);
}

#[test]
fn slab_reuse_after_many_evictions_keeps_the_list_consistent() {
    let _e = Env::new(0);
    let c = T::new();
    c.spares(&[3, 2, 1]);
    for k in 0..200 {
        c.acq(k, &[], 1.0);
        if k % 7 == 0 {
            c.find(k - 1);
        }
        if k % 11 == 0 && c.size() > 1 {
            let victim = c.keys()[1];
            let p = c.remove(victim);
            assert!(!p.is_null());
            unsafe { sc_release(c.0, p) };
        }
        assert_eq!(c.keys().len(), c.size());
    }
}

#[test]
fn struct_layout_matches_the_c_header() {
    use std::mem::{offset_of, size_of};
    let c = "#include <stdio.h>\n#include <stddef.h>\n#include \"slotcache.h\"\nint main(void){\n\
        printf(\"size=%zu\\n\", sizeof(sc_entry));\n\
        printf(\"key=%zu\\n\", offsetof(sc_entry, key));\n\
        printf(\"slot=%zu\\n\", offsetof(sc_entry, slot));\n\
        printf(\"heat=%zu\\n\", offsetof(sc_entry, heat));\n\
        printf(\"nu=%zu\\n\", offsetof(sc_entry, nu));\nreturn 0;}\n";
    let want = [
        ("size", size_of::<sc_entry>()),
        ("key", offset_of!(sc_entry, key)),
        ("slot", offset_of!(sc_entry, slot)),
        ("heat", offset_of!(sc_entry, heat)),
        ("nu", offset_of!(sc_entry, nu)),
    ];
    let dir = std::env::temp_dir().join(format!("slotcache_layout_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let (src, exe) = (dir.join("layout.c"), dir.join("layout"));
    std::fs::write(&src, c).expect("write C source");
    let inc = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let st = match Command::new("cc").arg("-I").arg(inc).arg(&src).arg("-o").arg(&exe).status() {
        Ok(st) => st,
        Err(e) => {
            println!("SKIPPED struct_layout_matches_the_c_header: no C compiler `cc` ({e})");
            eprintln!("SKIPPED struct_layout_matches_the_c_header: no C compiler `cc` ({e})");
            return;
        }
    };
    assert!(st.success(), "C compile of slotcache.h failed");
    let out = Command::new(&exe).output().expect("run layout probe");
    let text = String::from_utf8_lossy(&out.stdout);
    let got: Vec<(String, usize)> = text.lines().filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.parse().unwrap_or(usize::MAX)))).collect();
    assert_eq!(got.len(), want.len(), "probe output: {text}");
    for ((gk, gv), (wk, wv)) in got.iter().zip(want.iter()) {
        assert_eq!((gk.as_str(), *gv), (*wk, *wv));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// Child half of the fatal-path test: runs only when the parent sets the env var, and never returns.
#[test]
fn fatal_child() {
    let Ok(mode) = std::env::var("SLOTCACHE_FATAL_CHILD") else { return };
    let _e = Env::new(1);
    let c = T::new();
    match mode.as_str() {
        "lru" | "heat" => {
            sc_set_policy(if mode == "lru" { 0 } else { 1 });
            c.spares(&[A]);
            c.acq(1, &[], 1.0);
            c.acq(2, &[1], 1.0);
        }
        "belady" => {
            sc_set_policy(2);
            use_fn();
            nu_set(1, 10);
            c.spares(&[A]);
            c.acq(1, &[], 1.0);
            c.acq(2, &[1], 1.0);
        }
        "empty" => {
            c.acq(1, &[], 1.0);
        }
        "belady_empty" => {
            sc_set_policy(2);
            use_fn();
            c.acq(1, &[], 1.0);
        }
        other => panic!("unknown mode {other}"),
    }
    panic!("acquire returned instead of exiting (mode {mode})");
}

#[test]
fn fatal_path_exits_1_with_the_exact_message() {
    let exe = std::env::current_exe().expect("current_exe");
    for mode in ["lru", "heat", "belady", "empty", "belady_empty"] {
        let out = Command::new(&exe)
            .args(["--exact", "tests::fatal_child", "--nocapture", "--test-threads=1"])
            .env("SLOTCACHE_FATAL_CHILD", mode)
            .output()
            .expect("spawn child");
        assert_eq!(out.status.code(), Some(1), "mode {mode}: status {:?}, stderr {}", out.status, String::from_utf8_lossy(&out.stderr));
        assert_eq!(out.stderr, FATAL_MSG, "mode {mode}: stderr {:?}", String::from_utf8_lossy(&out.stderr));
    }
}


// prot above PROT_LINEAR_MAX keys goes through the bitmap when every key is in range; the result must equal the plain scan, and the bitmap must be left all zero
#[test]
fn large_prot_bitmap_equals_linear_scan() {
    let _e = Env::new(1);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut used_bitmap = 0;
    for _ in 0..300 {
        let (mut a, mut b) = (sc_cache::new(), sc_cache::new());
        let n = 60 + (next() % 200) as i32;
        for k in 0..n {
            let h = (next() % 50) as f32 / 3.0;
            let sl = s(k as usize + 1);
            a.spare.push(sl);
            b.spare.push(sl);
            let (ra, rb) = (a.acquire(k, &[], h), b.acquire_inner(k, &[], false, h));
            assert_eq!((ra.slot, ra.vkey), (rb.slot, rb.vkey));
        }
        let m = 33 + (next() % 100) as usize;
        let mut prot: Vec<i32> = (0..m).map(|_| n / 2 + (next() % (n as u64 / 2 + 10)) as i32).collect();
        if next() % 4 == 0 {
            let odd = [1 << 20, (1 << 20) - 1, -3, i32::MAX, i32::MIN, 1 << 25];
            let at = (next() % m as u64) as usize;
            prot[at] = odd[(next() % 6) as usize];
        }
        let eligible = prot.iter().all(|&k| k >= 0 && (k as usize) < PROT_BITS);
        let ra = a.acquire(n, &prot, 1.0);
        let rb = b.acquire_inner(n, &prot, false, 1.0);
        assert_eq!((ra.slot, ra.vkey, ra.vheat.map(f32::to_bits)), (rb.slot, rb.vkey, rb.vheat.map(f32::to_bits)));
        assert!(a.prot_bits.iter().all(|&w| w == 0), "bitmap not cleared");
        if eligible {
            assert_eq!(a.prot_bits.len(), PROT_BITS / 64, "bitmap path not taken");
            used_bitmap += 1;
        }
    }
    assert!(used_bitmap > 100, "only {used_bitmap} of 300 rounds used the bitmap");
}

#[test]
fn large_prot_bitmap_equals_linear_scan_belady() {
    let _e = Env::new(2);
    use_fn();
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..300 {
        for k in 0..64 {
            nu_set(k, (next() % 1000) as i32);
        }
        let (mut a, mut b) = (sc_cache::new(), sc_cache::new());
        for k in 0..60 {
            a.spare.push(s(k + 1));
            b.spare.push(s(k + 1));
            a.acquire(k as i32, &[], 1.0);
            b.acquire_inner(k as i32, &[], false, 1.0);
        }
        let m = 33 + (next() % 12) as usize;
        let prot: Vec<i32> = (0..m).map(|_| 20 + (next() % 44) as i32).collect();
        let cand = (next() % 1000) as i32;
        let (ta, tb) = (a.try_acquire(60, cand, &prot, 1.0), b.try_acquire_inner(60, cand, &prot, false, 1.0));
        assert_eq!(ta.is_some(), tb.is_some());
        if let (Some(ra), Some(rb)) = (ta, tb) {
            assert_eq!((ra.slot, ra.vkey), (rb.slot, rb.vkey));
        }
        let (ra, rb) = (a.acquire(61, &prot, 1.0), b.acquire_inner(61, &prot, false, 1.0));
        assert_eq!((ra.slot, ra.vkey), (rb.slot, rb.vkey));
        assert!(a.prot_bits.iter().all(|&w| w == 0), "bitmap not cleared");
    }
}