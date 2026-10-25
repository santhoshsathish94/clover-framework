#!/usr/bin/env python3
# Builds the mutant crates for the differential-driver check of poolinit (independent verifier). The REAL crate is only read.
# Per mutant: ~/rustmig/poolinit/verify/work/mut/<name>/ (Cargo.toml, Cargo.lock, src/, include/, slotcache-include/), exactly ONE logical change in src/*.rs,
# a unified diff in ~/rustmig/poolinit/verify/mutants/<name>.diff, then `cargo build --release` there. Edits are located in the ORIGINAL text (each `old` must occur
# exactly `cnt` times there), so two edits of one mutant never see each other's output.
# usage: mk_mutants.py [name ...]       (no names = all)      output: one line per mutant "name md5 description", also verify/mutants/md5list.txt
import difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/poolinit")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/mutants")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])

def E(f, old, new, cnt=1):
    return (f, old, new, cnt)

H, P, A, L, S = "host.rs", "pool.rs", "abi.rs", "lib.rs", "sys.rs"
MADV = "madvise(p, chunk_bytes, sys::MADV_HUGEPAGE);"
ZERO = "std::ptr::write_bytes(p as *mut u8, 0, chunk_bytes);"
SPARE_LOOP_CHUNKS = "for b in base.iter().rev() {"
REG_LOOP = "for b in base.iter() {"
COPY_LINE = "check(stream_create(addr_of_mut!((*out).copy), PI_STREAM_NON_BLOCKING), error_string, file, LINE_STREAM_COPY);"
RS_LINE = "check(stream_create(addr_of_mut!((*out).rstream[i]), PI_STREAM_NON_BLOCKING), error_string, file, LINE_STREAM_READER);"
BN_LINE = "check(host_alloc(addr_of_mut!((*out).bounce[i]) as *mut *mut c_void, raw, PI_HOST_ALLOC_DEFAULT), error_string, file, LINE_HOST_ALLOC);"
READERS = "(*out).readers = std::cmp::min(cfg.nreaders, PI_MAX_READERS as c_int);"
REFUSAL = "write_stderr(refusal_message(cfg.device, slots).as_bytes());"

MUT = [
    ("M00", "control: unmodified copy", []),
    # ---------------------------------------------------------------- host: threads and chunk arithmetic
    ("M01", "host: 16 helper threads instead of 15", [E(H, "const HELPERS: usize = 15;", "const HELPERS: usize = 16;")]),
    ("M02", "host: 14 helper threads instead of 15", [E(H, "const HELPERS: usize = 15;", "const HELPERS: usize = 14;")]),
    ("M03", "host: the calling thread does not map (touch() only by the 15 helpers)", [E(H, "        touch();\n    });", "    });")]),
    ("M04", "host plan: chunk_bytes = (CHUNK_SLOTS - 1) * raw", [E(H, "PI_CHUNK_SLOTS.wrapping_mul(expert_raw), cap_slots", "(PI_CHUNK_SLOTS - 1).wrapping_mul(expert_raw), cap_slots")]),
    ("M05", "host plan: n_chunks rounds up", [E(H, "cap_slots / PI_CHUNK_SLOTS)\n}", "(cap_slots + PI_CHUNK_SLOTS - 1) / PI_CHUNK_SLOTS)\n}")]),
    ("M06", "host plan: n_chunks one too few (saturating_sub(1))", [E(H, "cap_slots / PI_CHUNK_SLOTS)\n}", "(cap_slots / PI_CHUNK_SLOTS).saturating_sub(1))\n}")]),
    ("M07", "host: final out.cap_slots stored untrimmed (cap0)", [E(H, "unsafe { (*out).cap_slots = cap };", "unsafe { (*out).cap_slots = cap0 };")]),
    ("M08", "host: summary line prints the untrimmed cap_slots (cap0)", [E(H, "write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());", "write_stderr(summary_line(cap0, raw, sys::wall_now() - t0).as_bytes());")]),
    ("M09", "host summary: GB divisor 2^30 instead of 1e9", [E(H, "cap_slots as f64 * (expert_raw as f64) / 1e9,", "cap_slots as f64 * (expert_raw as f64) / 1073741824.0,")]),
    ("M10", "host summary: GB truncated (floor) to one decimal instead of rounded", [E(H, "cap_slots as f64 * (expert_raw as f64) / 1e9,", "(cap_slots as f64 * (expert_raw as f64) / 1e9 * 10.0).floor() / 10.0,")]),
    ("M11", "host summary: seconds measured from after the mapping (t1 just before the line)", [E(H, "write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());", "let t1 = sys::wall_now();\n    write_stderr(summary_line(cap, raw, sys::wall_now() - t1).as_bytes());")]),
    ("M12", "host summary: seconds in milliseconds", [E(H, "summary_line(cap, raw, sys::wall_now() - t0)", "summary_line(cap, raw, (sys::wall_now() - t0) * 1000.0)")]),
    ("M13", "host summary: seconds always 0.0", [E(H, "summary_line(cap, raw, sys::wall_now() - t0)", "summary_line(cap, raw, 0.0 * (sys::wall_now() - t0))")]),
    ("M14", "sys: wall_now uses CLOCK_REALTIME (0) instead of CLOCK_MONOTONIC", [E(S, "pub const CLOCK_MONOTONIC: c_int = 1;", "pub const CLOCK_MONOTONIC: c_int = 0;")]),
    ("M15", "sys: wall_now nanoseconds scaled 1e-6 instead of 1e-9", [E(S, "t.tv_sec as f64 + 1e-9 * t.tv_nsec as f64", "t.tv_sec as f64 + 1e-6 * t.tv_nsec as f64")]),
    # ---------------------------------------------------------------- host: mmap / madvise / zeroing
    ("M16", "host: madvise after the zeroing instead of before", [E(H, MADV + "\n            " + ZERO, ZERO + "\n            " + MADV)]),
    ("M17", "host: madvise removed", [E(H, MADV + "\n            " + ZERO, ZERO)]),
    ("M18", "host: madvise advice 15 (MADV_NOHUGEPAGE) instead of 14", [E(H, MADV, "madvise(p, chunk_bytes, 15);")]),
    ("M19", "host: madvise length chunk_bytes - 4096", [E(H, MADV, "madvise(p, chunk_bytes - 4096, sys::MADV_HUGEPAGE);")]),
    ("M20", "host: zeroing one byte short (chunk_bytes - 1)", [E(H, ZERO, "std::ptr::write_bytes(p as *mut u8, 0, chunk_bytes - 1);")]),
    ("M21", "host: zeroing skipped for chunk numbers >= 8", [E(H, ZERO, "if i < 8 {\n                std::ptr::write_bytes(p as *mut u8, 0, chunk_bytes);\n            }")]),
    ("M22", "host: zeroing removed", [E(H, ZERO, "")]),
    ("M23", "host: mmap flags | MAP_NORESERVE", [E(H, "sys::MAP_PRIVATE_ANONYMOUS, -1, 0)", "sys::MAP_PRIVATE_ANONYMOUS | 0x4000, -1, 0)")]),
    ("M24", "sys: mmap prot adds PROT_EXEC", [E(S, "pub const PROT_READ_WRITE: c_int = 0x1 | 0x2;", "pub const PROT_READ_WRITE: c_int = 0x1 | 0x2 | 0x4;")]),
    ("M25", "host: 'cannot map' message prints n_chunks instead of chunk_bytes", [E(H, 'cannot map {} bytes\\n", chunk_bytes)', 'cannot map {} bytes\\n", n_chunks)')]),
    ("M26", "host: 'cannot map' exits with status 2", [E(H, "                exit1();\n", "                sys::exit(2);\n")]),
    ("M27", "host: 'cannot map' message without newline", [E(H, 'cannot map {} bytes\\n"', 'cannot map {} bytes"')]),
    ("M28", "sys: MAP_FAILED is NULL (a failed mmap is not detected)", [E(S, "pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;", "pub const MAP_FAILED: *mut c_void = std::ptr::null_mut();")]),
    ("M61", "host: mmap offset 4096", [E(H, "sys::MAP_PRIVATE_ANONYMOUS, -1, 0)", "sys::MAP_PRIVATE_ANONYMOUS, -1, 4096)")]),
    ("M62", "host: mmap fd 0 instead of -1", [E(H, "sys::MAP_PRIVATE_ANONYMOUS, -1, 0)", "sys::MAP_PRIVATE_ANONYMOUS, 0, 0)")]),
    # ---------------------------------------------------------------- host: ops
    ("M29", "host: set_device(1) instead of 0", [E(H, "check(set_device(0),", "check(set_device(1),")]),
    ("M30", "host: host_register flags 0 instead of Portable", [E(H, "chunk_bytes, PI_HOST_REGISTER_PORTABLE), error_string", "chunk_bytes, 0), error_string")]),
    ("M31", "host: the last chunk is not registered", [E(H, REG_LOOP, "for b in base.iter().take(n_chunks.saturating_sub(1)) {")]),
    ("M32", "host: chunks registered in descending order (spares unchanged)", [E(H, REG_LOOP, "for b in base.iter().rev() {")]),
    ("M33", "host: registration descending AND spare chunk order ascending (both reversed)", [E(H, REG_LOOP, "for b in base.iter().rev() {"), E(H, SPARE_LOOP_CHUNKS, "for b in base.iter() {")]),
    ("M34", "host: spares within a chunk ascending (slot 0 first)", [E(H, "for s in (0..PI_CHUNK_SLOTS).rev() {", "for s in 0..PI_CHUNK_SLOTS {")]),
    ("M35", "host: spares chunk order ascending (slots descending)", [E(H, SPARE_LOOP_CHUNKS, "for b in base.iter() {")]),
    ("M36", "host: 31 spares per chunk (slot 0 dropped)", [E(H, "for s in (0..PI_CHUNK_SLOTS).rev() {", "for s in (1..PI_CHUNK_SLOTS).rev() {")]),
    ("M37", "host: spare address stride raw - 1", [E(H, "p.wrapping_add(s.wrapping_mul(raw))", "p.wrapping_add(s.wrapping_mul(raw - 1))")]),
    ("M38", "host: spares go to a NULL cache instead of cfg.cache", [E(H, "add_spare(cfg.cache, p.wrapping_add", "add_spare(std::ptr::null_mut(), p.wrapping_add")]),
    ("M39", "host: out.ready never set", [E(H, "        (*out).cap_slots = cap0;\n        (*out).ready = 1;\n    }", "        (*out).cap_slots = cap0;\n    }")]),
    ("M40", "host: out.ready stored last (after the summary line) instead of first", [E(H, "        (*out).cap_slots = cap0;\n        (*out).ready = 1;\n    }", "        (*out).cap_slots = cap0;\n    }"), E(H, "write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());\n}", "write_stderr(summary_line(cap, raw, sys::wall_now() - t0).as_bytes());\n    unsafe { (*out).ready = 1 };\n}")]),
    ("M41", "host: out.cap_slots not stored at the start (only at the end)", [E(H, "        (*out).cap_slots = cap0;\n        (*out).ready = 1;", "        (*out).ready = 1;")]),
    # ---------------------------------------------------------------- host: workers
    ("M42", "host: helper threads detached, never joined (std::thread::spawn, leaked table)", [
        E(H, "let base: Vec<AtomicUsize> = (0..n_chunks).map(|_| AtomicUsize::new(0)).collect();\n    let next = AtomicUsize::new(0);",
             "let base: &'static Vec<AtomicUsize> = Box::leak(Box::new((0..n_chunks).map(|_| AtomicUsize::new(0)).collect::<Vec<AtomicUsize>>()));\n    let next: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));"),
        E(H, "let touch = || loop {", "let touch = move || loop {"),
        E(H, "    std::thread::scope(|s| {\n        for _ in 0..HELPERS {\n            if std::thread::Builder::new().spawn_scoped(s, &touch).is_err() {\n                write_stderr(b\"poolinit: cannot start a helper thread\\n\");\n                std::process::abort();\n            }\n        }\n        touch();\n    });",
             "    for _ in 0..HELPERS {\n        std::thread::spawn(touch);\n    }\n    touch();")]),
    ("M43", "host: chunk counter advances by 2 (every second chunk skipped)", [E(H, "next.fetch_add(1, Ordering::SeqCst)", "next.fetch_add(2, Ordering::SeqCst)")]),
    ("M44", "host: worker loop bound i > n_chunks (one extra chunk)", [E(H, "if i >= n_chunks {", "if i > n_chunks {")]),
    ("M45", "host: worker loop bound i + 1 >= n_chunks (last chunk never mapped)", [E(H, "if i >= n_chunks {", "if i + 1 >= n_chunks {")]),
    ("M46", "host: chunk table written in reverse index (base[n-1-i])", [E(H, "base[i].store(p as usize, Ordering::SeqCst);", "base[n_chunks - 1 - i].store(p as usize, Ordering::SeqCst);")]),
    ("M47", "host: all atomics Relaxed instead of SeqCst", [E(H, "Ordering::SeqCst", "Ordering::Relaxed", 4)]),
    ("M48", "host: final cap = cap0 & !31 instead of n_chunks * 32", [E(H, "let cap = n_chunks.wrapping_mul(PI_CHUNK_SLOTS);", "let cap = cap0 & !(PI_CHUNK_SLOTS - 1);")]),
    ("M49", "host plan: chunk_bytes = raw << 5 instead of 32 * raw", [E(H, "PI_CHUNK_SLOTS.wrapping_mul(expert_raw), cap_slots", "expert_raw << 5, cap_slots")]),
    # ---------------------------------------------------------------- check / messages / exit (sys.rs, host.rs line numbers)
    ("M50", "host: CHECK line of set_device 429 instead of 428", [E(H, "const LINE_SET_DEVICE: u32 = 428;", "const LINE_SET_DEVICE: u32 = 429;")]),
    ("M51", "host: CHECK line of host_register 428 instead of 429", [E(H, "const LINE_REGISTER: u32 = 429;", "const LINE_REGISTER: u32 = 428;")]),
    ("M52", "sys: default file name 'decode.cu' when src_file is NULL", [E(S, 'm.extend_from_slice(b"decode.cuh");', 'm.extend_from_slice(b"decode.cu");')]),
    ("M53", "sys: check() ignores negative codes (code <= 0 is success)", [E(S, "    if code == 0 {", "    if code <= 0 {")]),
    ("M54", "sys: NULL error_string prints '' instead of '(null)'", [E(S, 'b"(null)"', 'b""')]),
    ("M55", "sys: check() message without trailing newline", [E(S, "    m.push(b'\\n');\n", "")]),
    ("M56", "sys: check() exits with status 2", [E(S, "    write_stderr(&m);\n    exit1()", "    write_stderr(&m);\n    unsafe { exit(2) }")]),
    ("M57", "sys: check() calls error_string twice", [E(S, "m.extend_from_slice(cstr_bytes(error_string(code)));", "let _ = error_string(code);\n        m.extend_from_slice(cstr_bytes(error_string(code)));")]),
    ("M58", "sys: check() writes the 'file:line: ' prefix and the text in two write calls", [E(S, 'm.extend_from_slice(format!(":{}: ", line).as_bytes());', 'm.extend_from_slice(format!(":{}: ", line).as_bytes());\n        write_stderr(&m);\n        m.clear();')]),
    ("M59", "sys: exit1() via std::process::exit(1)", [E(S, "    unsafe { exit(1) }", "    std::process::exit(1)")]),
    ("M60", "sys: exit1() via _exit(1) (atexit handlers skipped)", [E(S, "    unsafe { exit(1) }", "    unsafe {\n        extern \"C\" {\n            fn _exit(c: c_int) -> !;\n        }\n        _exit(1)\n    }")]),
    # ---------------------------------------------------------------- pool
    ("M63", "pool: MIN_SLOTS 63", [E(P, "pub const MIN_SLOTS: c_int = 64;", "pub const MIN_SLOTS: c_int = 63;")]),
    ("M64", "pool: refusal when slots <= 64 (probe C of the author)", [E(P, "if slots < MIN_SLOTS {", "if slots <= MIN_SLOTS {")]),
    ("M65", "pool: MIN_SLOTS 65", [E(P, "pub const MIN_SLOTS: c_int = 64;", "pub const MIN_SLOTS: c_int = 65;")]),
    ("M66", "pool: the threshold compares as u32 (negative slot counts not refused)", [E(P, "if slots < MIN_SLOTS {", "if (slots as u32) < (MIN_SLOTS as u32) {")]),
    ("M67", "pool: usable test free >= reserve instead of >", [E(P, "if free_bytes > reserve_bytes {", "if free_bytes >= reserve_bytes {")]),
    ("M68", "pool: usable = free.saturating_sub(reserve)", [E(P, "let usable = if free_bytes > reserve_bytes { free_bytes - reserve_bytes } else { 0 };", "let usable = free_bytes.saturating_sub(reserve_bytes);")]),
    ("M69", "pool: usable = free.wrapping_sub(reserve) (no underflow guard)", [E(P, "let usable = if free_bytes > reserve_bytes { free_bytes - reserve_bytes } else { 0 };", "let usable = free_bytes.wrapping_sub(reserve_bytes);")]),
    ("M70", "pool: slot count saturates at INT_MAX instead of wrapping in the cast", [E(P, "(usable / expert_raw) as c_int", "(usable / expert_raw).min(c_int::MAX as usize) as c_int")]),
    ("M71", "pool: slot count rounds up", [E(P, "(usable / expert_raw) as c_int", "((usable + expert_raw - 1) / expert_raw) as c_int")]),
    ("M72", "pool: slot count computed with u32 operands", [E(P, "(usable / expert_raw) as c_int", "((usable as u32) / (expert_raw as u32)) as c_int")]),
    ("M73", "pool: slab size (slots - 1) * raw", [E(P, "(slots as usize).wrapping_mul(raw)", "((slots as usize) - 1).wrapping_mul(raw)")]),
    ("M74", "pool: slab size computed as a 32-bit multiply", [E(P, "(slots as usize).wrapping_mul(raw)", "((slots as u32).wrapping_mul(raw as u32)) as usize")]),
    ("M75", "pool: spares ascending (slot 0 added first)", [E(P, "for i in (0..slots).rev() {", "for i in 0..slots {")]),
    ("M76", "pool: slot 0 never added as a spare", [E(P, "for i in (0..slots).rev() {", "for i in (1..slots).rev() {")]),
    ("M77", "pool: spare address computed with a 32-bit multiply", [E(P, "base.wrapping_add((i as usize).wrapping_mul(raw))", "base.wrapping_add(((i as i32).wrapping_mul(raw as i32)) as usize)")]),
    ("M78", "pool: copy stream flags 0 instead of NonBlocking", [E(P, "check(stream_create(addr_of_mut!((*out).copy), PI_STREAM_NON_BLOCKING)", "check(stream_create(addr_of_mut!((*out).copy), 0)")]),
    ("M79", "pool: reader stream flags 0 instead of NonBlocking", [E(P, "rstream[i]), PI_STREAM_NON_BLOCKING)", "rstream[i]), 0)")]),
    ("M80", "pool: copy stream created after the reader loop", [E(P, "        " + COPY_LINE + "\n", ""), E(P, "        check(event_create(addr_of_mut!((*out).written_back)", "        " + COPY_LINE + "\n        check(event_create(addr_of_mut!((*out).written_back)")]),
    ("M81", "pool: bounce slot allocated before the reader stream", [E(P, "            " + RS_LINE + "\n            " + BN_LINE, "            " + BN_LINE + "\n            " + RS_LINE)]),
    ("M82", "pool: bounce host_alloc flags 1 instead of Default", [E(P, "raw, PI_HOST_ALLOC_DEFAULT)", "raw, 1)")]),
    ("M83", "pool: bounce size raw - 1", [E(P, "as *mut *mut c_void, raw, PI_HOST_ALLOC_DEFAULT)", "as *mut *mut c_void, raw - 1, PI_HOST_ALLOC_DEFAULT)")]),
    ("M84", "pool: event flags 0 instead of DisableTiming", [E(P, "PI_EVENT_DISABLE_TIMING), error_string", "0), error_string")]),
    ("M85", "pool: d_eptr size experts * 4", [E(P, "std::mem::size_of::<*const u8>()", "4")]),
    ("M86", "pool: d_eptr size uses experts zero-extended (negative experts)", [E(P, "cfg.experts as isize as usize", "cfg.experts as u32 as usize")]),
    ("M87", "pool: readers clamp 31", [E(P, "std::cmp::min(cfg.nreaders, PI_MAX_READERS as c_int)", "std::cmp::min(cfg.nreaders, 31)")]),
    ("M88", "pool: readers clamp 33", [E(P, "std::cmp::min(cfg.nreaders, PI_MAX_READERS as c_int)", "std::cmp::min(cfg.nreaders, 33)")]),
    ("M89", "pool: negative readers clamped to 0 in out.readers", [E(P, READERS, READERS[:-1] + ".max(0);")]),
    ("M90", "pool: reader loop inclusive (0..=readers)", [E(P, "for i in 0..readers {", "for i in 0..=readers {")]),
    ("M91", "pool: reader loop 0..readers-1", [E(P, "for i in 0..readers {", "for i in 0..readers - 1 {")]),
    ("M92", "pool: out.slots stored after the refusal test", [E(P, "        (*out).slots = slots;\n        if slots < MIN_SLOTS {\n            " + REFUSAL + "\n            exit1();\n        }", "        if slots < MIN_SLOTS {\n            " + REFUSAL + "\n            exit1();\n        }\n        (*out).slots = slots;")]),
    ("M93", "pool: out.ready also stored first", [E(P, "        (*out).dev = cfg.device;", "        (*out).ready = 1;\n        (*out).dev = cfg.device;")]),
    ("M94", "pool: out.dev never stored", [E(P, "        (*out).dev = cfg.device;\n", "")]),
    ("M95", "pool: set_device(0) instead of cfg.device", [E(P, "check(set_device(cfg.device),", "check(set_device(0),")]),
    ("M96", "pool: refusal message arguments swapped (slots, dev)", [E(P, "refusal_message(cfg.device, slots)", "refusal_message(slots, cfg.device)")]),
    ("M97", "pool: refusal message without newline", [E(P, 'format!("card {}: only {} expert slots fit after the reserve\\n"', 'format!("card {}: only {} expert slots fit after the reserve"')]),
    ("M98", "pool: refusal exits with status 2", [E(P, REFUSAL + "\n            exit1();", REFUSAL + "\n            sys::exit(2);")]),
    ("M99", "pool: refusal does not exit (falls through)", [E(P, REFUSAL + "\n            exit1();", REFUSAL)]),
    ("M100", "pool: CHECK line of mem_get_info 456 instead of 455", [E(P, "const LINE_MEM_GET_INFO: u32 = 455;", "const LINE_MEM_GET_INFO: u32 = 456;")]),
    ("M101", "pool: CHECK line of the bounce host_alloc 463 instead of 464", [E(P, "const LINE_HOST_ALLOC: u32 = 464;", "const LINE_HOST_ALLOC: u32 = 463;")]),
    ("M102", "pool: CHECK line of the event 467 instead of 466", [E(P, "const LINE_EVENT: u32 = 466;", "const LINE_EVENT: u32 = 467;")]),
    ("M103", "pool: CHECK line of the d_eptr malloc 466 instead of 467", [E(P, "const LINE_MALLOC_EPTR: u32 = 467;", "const LINE_MALLOC_EPTR: u32 = 466;")]),
    ("M104", "pool: CHECK line of the reader stream 464 instead of 463", [E(P, "const LINE_STREAM_READER: u32 = 463;", "const LINE_STREAM_READER: u32 = 464;")]),
    ("M105", "pool: CHECK line of the slab malloc 460 instead of 459", [E(P, "const LINE_MALLOC_SLAB: u32 = 459;", "const LINE_MALLOC_SLAB: u32 = 460;")]),
    ("M106", "pool: CHECK line of the copy stream 462 instead of 461", [E(P, "const LINE_STREAM_COPY: u32 = 461;", "const LINE_STREAM_COPY: u32 = 462;")]),
    ("M107", "pool: CHECK line of set_device 455 instead of 454", [E(P, "const LINE_SET_DEVICE: u32 = 454;", "const LINE_SET_DEVICE: u32 = 455;")]),
    ("M108", "pool: mem_get_info arguments swapped (&mut t, &mut f)", [E(P, "check(mem_get_info(&mut f, &mut t)", "check(mem_get_info(&mut t, &mut f)")]),
    ("M109", "pool: spares go to a NULL cache instead of cfg.cache", [E(P, "add_spare(cfg.cache, base.wrapping_add(", "add_spare(std::ptr::null_mut(), base.wrapping_add(")]),
    ("M110", "pool: out.readers stored after mem_get_info instead of first", [E(P, "        " + READERS + "\n", ""), E(P, "        let slots = slots_for(f, cfg.reserve_bytes, raw);", "        " + READERS + "\n        let slots = slots_for(f, cfg.reserve_bytes, raw);")]),
    ("M111", "pool: reader loop bound is the unclamped cfg.nreaders", [E(P, "let readers = (*out).readers;", "let readers = cfg.nreaders;")]),
    ("M112", "pool: every reader stream stored into rstream[0]", [E(P, "check(stream_create(addr_of_mut!((*out).rstream[i])", "check(stream_create(addr_of_mut!((*out).rstream[0])")]),
    # ---------------------------------------------------------------- abi / lib
    ("M113", "abi: PI_CHUNK_SLOTS 31", [E(A, "pub const PI_CHUNK_SLOTS: usize = 32;", "pub const PI_CHUNK_SLOTS: usize = 31;")]),
    ("M114", "abi: PI_MAX_READERS 31", [E(A, "pub const PI_MAX_READERS: usize = 32;", "pub const PI_MAX_READERS: usize = 31;")]),
    ("M115", "abi: PI_HOST_REGISTER_PORTABLE 0x03", [E(A, "pub const PI_HOST_REGISTER_PORTABLE: c_uint = 0x01;", "pub const PI_HOST_REGISTER_PORTABLE: c_uint = 0x03;")]),
    ("M116", "abi: PI_EVENT_DISABLE_TIMING 0x03", [E(A, "pub const PI_EVENT_DISABLE_TIMING: c_uint = 0x02;", "pub const PI_EVENT_DISABLE_TIMING: c_uint = 0x03;")]),
    ("M117", "abi: pi_expert_stats fields reordered (needed last, t_read first)", [E(A, "    pub needed: c_ulong,\n    pub hit_card: c_ulong,\n    pub hit_host: c_ulong,\n    pub from_drive: c_ulong,\n    pub written_back: c_ulong,\n    pub t_read: f64,", "    pub t_read: f64,\n    pub hit_card: c_ulong,\n    pub hit_host: c_ulong,\n    pub from_drive: c_ulong,\n    pub written_back: c_ulong,\n    pub needed: c_ulong,")]),
    ("M118", "abi: pi_pool_cfg nreaders and experts swapped", [E(A, "    pub nreaders: c_int,\n    pub expert_raw: usize,\n    pub experts: c_int,", "    pub experts: c_int,\n    pub expert_raw: usize,\n    pub nreaders: c_int,")]),
    ("M119", "abi: pi_host_cfg bytes and expert_raw swapped", [E(A, "    pub bytes: usize,\n    pub expert_raw: usize,\n    pub cache: *mut sc_cache,", "    pub expert_raw: usize,\n    pub bytes: usize,\n    pub cache: *mut sc_cache,")]),
    ("M120", "lib: pi_pool_init does not refuse expert_raw == 0", [E(L, "    need_fn!(ops, sc_add_spare);\n    if cfg.expert_raw == 0 {\n        bad(\"expert_raw is 0\");\n    }\n    // SAFETY: by the contract.\n    unsafe { pool::pool_init", "    need_fn!(ops, sc_add_spare);\n    // SAFETY: by the contract.\n    unsafe { pool::pool_init")]),
    ("M121", "lib: pi_pool_init does not check ops->host_alloc (falls to the generic 'ops incomplete')", [E(L, "    need_fn!(ops, host_alloc);\n", "")]),
    ("M122", "lib: message 'expert_raw is zero' instead of 'expert_raw is 0' (both entry points)", [E(L, 'bad("expert_raw is 0")', 'bad("expert_raw is zero")', 2)]),
]

def build_one(name, desc, edits):
    d = os.path.join(WORK, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in ("Cargo.toml", "Cargo.lock"):
        shutil.copy(os.path.join(SRC, f), d)
    shutil.copytree(os.path.join(SRC, "src"), os.path.join(d, "src"))
    shutil.copytree(os.path.join(SRC, "include"), os.path.join(d, "include"))
    shutil.copytree(os.path.join(HOME, "rustmig/slotcache/include"), os.path.join(d, "slotcache-include"))
    byfile = {}
    for f, old, new, cnt in edits:
        byfile.setdefault(f, []).append((old, new, cnt))
    diff = ""
    for f, es in byfile.items():
        p = os.path.join(d, "src", f)
        t0 = open(p, newline="").read()
        pos = []
        for old, new, cnt in es:
            n = t0.count(old)
            if n != cnt:
                print(name, "NOT APPLIED: %r occurs %d times in %s, wanted %d" % (old[:60], n, f, cnt), flush=True)
                return None
            i = -1
            for _ in range(cnt):
                i = t0.index(old, i + 1)
                pos.append((i, len(old), new))
        t1 = t0
        for i, l, new in sorted(pos, reverse=True):
            t1 = t1[:i] + new + t1[i + l:]
        open(p, "w", newline="").write(t1)
        diff += "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/" + f, "b/src/" + f, n=3))
    open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s: %s\n%s" % (name, desc, diff))
    r = subprocess.run(["cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
    so = os.path.join(d, "target/release/libpoolinit.so")
    if r.returncode != 0 or not os.path.exists(so):
        print(name, "BUILD FAILED"); print((r.stdout + r.stderr)[-1500:], flush=True)
        return None
    return hashlib.md5(open(so, "rb").read()).hexdigest()

if __name__ == "__main__":
    only = sys.argv[1:]
    os.makedirs(WORK, exist_ok=True); os.makedirs(DIFFS, exist_ok=True)
    mdl = os.path.join(DIFFS, "md5list.txt")
    lines = {}
    if os.path.exists(mdl):
        for l in open(mdl):
            k = l.split(" ", 1)[0]
            lines[k] = l.rstrip("\n")
    names = [m[0] for m in MUT]
    assert len(names) == len(set(names)), "duplicate mutant name"
    for name, desc, edits in MUT:
        if only and name not in only:
            continue
        md5 = build_one(name, desc, edits)
        if md5:
            lines[name] = "%s %s %s" % (name, md5, desc)
            print(lines[name], flush=True)
    with open(mdl, "w") as f:
        for k in sorted(lines, key=lambda s: (len(s), s)):
            f.write(lines[k] + "\n")
