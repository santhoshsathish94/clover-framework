//! Map from i32 key to value with a dense key vector: the keys can be handed out as a slice (the planner's `prot`) without building it per call.
//! Removal is swap-remove, so the order of `keys()` is not stable; callers use it for membership only.
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct IntHasher(u64);

impl Hasher for IntHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_i32(&mut self, i: i32) {
        let h = (i as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
}

pub struct DenseMap<V> {
    keys: Vec<i32>,
    vals: Vec<V>,
    idx: HashMap<i32, usize, BuildHasherDefault<IntHasher>>,
}

impl<V> DenseMap<V> {
    pub fn new() -> Self {
        DenseMap { keys: Vec::new(), vals: Vec::new(), idx: HashMap::default() }
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn contains(&self, key: i32) -> bool {
        self.idx.contains_key(&key)
    }
    pub fn get(&self, key: i32) -> Option<&V> {
        self.idx.get(&key).map(|&i| &self.vals[i])
    }
    pub fn keys(&self) -> &[i32] {
        &self.keys
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.vals.iter()
    }
    /// Stores `v` under `key`; returns the value it replaced, if any (the key keeps its place in `keys()`).
    pub fn insert(&mut self, key: i32, v: V) -> Option<V> {
        if let Some(&i) = self.idx.get(&key) {
            return Some(std::mem::replace(&mut self.vals[i], v));
        }
        self.idx.insert(key, self.keys.len());
        self.keys.push(key);
        self.vals.push(v);
        None
    }
    pub fn remove(&mut self, key: i32) -> Option<V> {
        let i = self.idx.remove(&key)?;
        self.keys.swap_remove(i);
        let v = self.vals.swap_remove(i);
        if i < self.keys.len() {
            self.idx.insert(self.keys[i], i);
        }
        Some(v)
    }
}
