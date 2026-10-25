//! Rust implementation of include/routetable.h. Behaviour reference: `struct CallSet`, `g_rc`, `g_uses`, `g_use_ptr`, `g_rc_cur`, `rc_nu` and `rc_load` in src/code/gpu/backend-gpu.cu.
//! Not thread-safe (one caller thread, as the contract says).
#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::ffi::{CStr, OsStr};
use std::io::{Read, Write};
use std::os::raw::{c_char, c_int, c_long};
use std::os::unix::ffi::OsStrExt;

#[cfg(test)]
mod tests;

const EXHAUSTED: i32 = 1_000_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub layer: i32,
    pub experts: Vec<i32>,
    pub counts: Vec<i32>,
}

#[allow(non_camel_case_types)]
pub struct rt_table {
    calls: Vec<Call>,
    // key -> (ascending call numbers, forward-only pointer into them); an entry exists only for keys that occur in some call
    uses: HashMap<i32, (Vec<i32>, usize)>,
    cur: i64,
}

// C strtol(s, &end, 10) on a NUL-terminated string where `at` yields 0 past the slice: returns (value, index of end). No digits -> (0, pos) (nothing consumed).
// Saturates at LONG_MAX / LONG_MIN (64-bit long); the caller casts to i32 like the original's (int) cast (low 32 bits).
pub fn strtol(s: &[u8], pos: usize) -> (i64, usize) {
    let mut i = pos;
    while matches!(at(s, i), b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    match at(s, i) {
        b'-' => {
            neg = true;
            i += 1;
        }
        b'+' => i += 1,
        _ => {}
    }
    if !at(s, i).is_ascii_digit() {
        return (0, pos);
    }
    let mut acc: u64 = 0;
    let mut overflow = false;
    while at(s, i).is_ascii_digit() {
        let d = (at(s, i) - b'0') as u64;
        if !overflow {
            match acc.checked_mul(10).and_then(|v| v.checked_add(d)) {
                Some(v) => acc = v,
                None => overflow = true,
            }
        }
        i += 1;
    }
    let v = if neg {
        if overflow || acc > (1u64 << 63) {
            i64::MIN
        } else {
            (acc as i64).wrapping_neg()
        }
    } else if overflow || acc > i64::MAX as u64 {
        i64::MAX
    } else {
        acc as i64
    };
    (v, i)
}

fn at(s: &[u8], i: usize) -> u8 {
    s.get(i).copied().unwrap_or(0)
}

// One getline() line (including its '\n', if any) parsed like rc_load's loop body. The line is cut at the first NUL, as the C string functions see it.
pub fn parse_call(line: &[u8]) -> Call {
    let s = match line.iter().position(|&b| b == 0) {
        Some(n) => &line[..n],
        None => line,
    };
    let (layer, mut p) = strtol(s, 0);
    let mut call = Call { layer: layer as i32, experts: Vec::new(), counts: Vec::new() };
    p = strtol(s, p).1;
    loop {
        while at(s, p) == b' ' {
            p += 1;
        }
        if !at(s, p).is_ascii_digit() {
            break;
        }
        let (e, q) = strtol(s, p);
        call.experts.push(e as i32);
        p = q;
        if at(s, p) == b':' {
            let (c, q) = strtol(s, p + 1);
            call.counts.push(c as i32);
            p = q;
        }
    }
    call
}

// Lines as getline returns them: split after each '\n'; the last line may lack it; an empty input has none.
pub fn parse_data(data: &[u8]) -> Vec<Call> {
    data.split_inclusive(|&b| b == b'\n').map(parse_call).collect()
}

// C int: layer * experts + e; signed overflow is UB in C, here it wraps (what the original does on x86-64 gcc without -ftrapv).
fn ekey(layer: i32, experts: i32, e: i32) -> i32 {
    layer.wrapping_mul(experts).wrapping_add(e)
}

impl rt_table {
    pub fn from_calls(calls: Vec<Call>, experts: i32) -> rt_table {
        let mut uses: HashMap<i32, (Vec<i32>, usize)> = HashMap::new();
        for (i, c) in calls.iter().enumerate() {
            for &e in &c.experts {
                uses.entry(ekey(c.layer, experts, e)).or_insert_with(|| (Vec::new(), 0)).0.push(i as i32);
            }
        }
        rt_table { calls, uses, cur: -1 }
    }

    pub fn next_use(&mut self, key: i32) -> i32 {
        let cur = self.cur;
        match self.uses.get_mut(&key) {
            None => EXHAUSTED,
            Some((l, p)) => {
                while *p < l.len() && (l[*p] as i64) <= cur {
                    *p += 1;
                }
                if *p < l.len() {
                    l[*p]
                } else {
                    EXHAUSTED
                }
            }
        }
    }
}

// fopen succeeds on a directory and getline then fails, so a read error ends the data (what was read before it is kept); only a failed open is fatal.
fn read_route_file(path: &[u8]) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(OsStr::from_bytes(path)).ok()?;
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    Some(buf)
}

// eprintln! can panic on a closed stderr; write errors are ignored.
fn fatal_cannot_read(path: &[u8]) -> ! {
    let mut e = std::io::stderr();
    let _ = e.write_all(b"cannot read route cache ");
    let _ = e.write_all(path);
    let _ = e.write_all(b"\n");
    std::process::exit(1)
}

/// # Safety
/// `path` must point to a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn rt_load(path: *const c_char, experts: c_int) -> *mut rt_table {
    let p = CStr::from_ptr(path).to_bytes();
    let data = match read_route_file(p) {
        Some(d) => d,
        None => fatal_cannot_read(p),
    };
    Box::into_raw(Box::new(rt_table::from_calls(parse_data(&data), experts)))
}

/// # Safety
/// `t` must be NULL or a pointer returned by `rt_load` that has not been freed.
#[no_mangle]
pub unsafe extern "C" fn rt_free(t: *mut rt_table) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// # Safety
/// `t` must be a live table.
#[no_mangle]
pub unsafe extern "C" fn rt_calls(t: *const rt_table) -> usize {
    let t: &rt_table = &*t;
    t.calls.len()
}

/// # Safety
/// `t` must be a live table and `i < rt_calls(t)` (otherwise the process aborts on the index check).
#[no_mangle]
pub unsafe extern "C" fn rt_layer(t: *const rt_table, i: usize) -> c_int {
    let t: &rt_table = &*t;
    t.calls[i].layer
}

unsafe fn list(v: &[i32], n: *mut usize) -> *const i32 {
    if !n.is_null() {
        *n = v.len();
    }
    if v.is_empty() {
        std::ptr::null()
    } else {
        v.as_ptr()
    }
}

/// # Safety
/// `t` must be a live table, `i < rt_calls(t)`, `n` NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn rt_experts(t: *const rt_table, i: usize, n: *mut usize) -> *const c_int {
    let t: &rt_table = &*t;
    list(&t.calls[i].experts, n)
}

/// # Safety
/// `t` must be a live table, `i < rt_calls(t)`, `n` NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn rt_counts(t: *const rt_table, i: usize, n: *mut usize) -> *const c_int {
    let t: &rt_table = &*t;
    list(&t.calls[i].counts, n)
}

/// # Safety
/// `t` must be a live table.
#[no_mangle]
pub unsafe extern "C" fn rt_set_current(t: *mut rt_table, cur: c_long) {
    (*t).cur = cur as i64;
}

/// # Safety
/// `t` must be a live table.
#[no_mangle]
pub unsafe extern "C" fn rt_next_use(t: *mut rt_table, key: c_int) -> c_int {
    let t: &mut rt_table = &mut *t;
    t.next_use(key)
}
