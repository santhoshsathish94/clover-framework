#!/usr/bin/env python3
# Builds the mutant crates for the differential-driver check of the `supply` crate (independent verifier, 2026-10-25). The REAL crate is only read.
# Each mutant: ~/rustmig/supply/verify/work/mut/<name>/ (Cargo.toml, Cargo.lock, src/, include/, slotcache-include/), exactly ONE textual change in one src/*.rs file,
# a unified diff in ~/rustmig/supply/verify/mutants/<name>.diff (first line = description), then `cargo build --release` there. usage: mk_mutants.py [name ...]
import difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/supply")
WORK = os.path.join(SRC, "verify/work/mut")
DIFFS = os.path.join(SRC, "verify/mutants")
SCINC = os.path.join(HOME, "rustmig/slotcache/include")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])

E = "ensure.rs"
# (description, file, old, new)   -- each `old` must occur exactly once in the real file
L = []
def m(desc, old, new, f=E):
    L.append((desc, f, old, new))

# ---- step 0 / prot / needed
m("vq: first sync is device_sync, not stream_sync(NULL) (condition inverted)",
  "        if vq {\n            do_check(ops, (ops.stream_sync)(null_mut()), site::STREAM_SYNC_0);",
  "        if !vq {\n            do_check(ops, (ops.stream_sync)(null_mut()), site::STREAM_SYNC_0);")
m("vq: first sync is stream_sync(pool.copy) instead of stream_sync(NULL)",
  "(ops.stream_sync)(null_mut()), site::STREAM_SYNC_0", "(ops.stream_sync)(pool.copy), site::STREAM_SYNC_0")
m("prot not de-duplicated (dedup removed)", "        prot.dedup();\n", "")
m("prot keys without the layer (raw expert numbers)",
  "experts.iter().map(|&e| self.ekey(layer, e)).collect();", "experts.iter().map(|&e| e).collect();")
m("prot not sorted (sort removed; dedup then only drops adjacent duplicates)", "        prot.sort_unstable();\n", "")
m("needed counts the distinct experts, not the experts of the call",
  "pool.st.needed = pool.st.needed.wrapping_add(experts.len() as c_ulong);", "pool.st.needed = pool.st.needed.wrapping_add(prot.len() as c_ulong);")
# ---- card hit
m("card hit: sc_touch before arr_wait (order swapped)",
  "                if vq {\n                    (ops.arr_wait)(key);\n                }\n                if !pool.cache.is_null() {\n                    (ops.sc_touch)(pool.cache, key, heat);\n                }\n",
  "                if !pool.cache.is_null() {\n                    (ops.sc_touch)(pool.cache, key, heat);\n                }\n                if vq {\n                    (ops.arr_wait)(key);\n                }\n")
m("card hit: arr_wait missing", "                if vq {\n                    (ops.arr_wait)(key);\n                }\n                if !pool.cache.is_null() {", "                if !pool.cache.is_null() {")
m("card hit: hit_card not counted", "pool.st.hit_card = pool.st.hit_card.wrapping_add(1);", "pool.st.hit_card = pool.st.hit_card;")
m("heat = counts[e] + 1", "let heat = *counts.offset(e as isize) as f32;", "let heat = (*counts.offset(e as isize) + 1) as f32;")
m("card hit: sc_touch gets the expert number instead of the key (equal only for layer 0)", "(ops.sc_touch)(pool.cache, key, heat);", "(ops.sc_touch)(pool.cache, e, heat);")
m("card hit: h_eptr[e] not set",
  "                *pool.h_eptr.offset(e as isize) = p as *const u8;\n                pool.st.hit_card", "                pool.st.hit_card")
# ---- miss / victims
m("victim test vk > 0 instead of vk >= 0 (victim key 0 ignored)", "if vk >= 0 {", "if vk > 0 {")
m("victim: arr_wait on the new key instead of the victim key", "(ops.arr_wait)(vk);", "(ops.arr_wait)(key);")
m("victim heat taken from the new expert (heat) instead of vh", "victims.push(Victim { key: vk, heat: vh, src: p });", "victims.push(Victim { key: vk, heat: heat, src: p });")
m("vk initialised to 0 instead of -1 (slot cache writes *vkey on every path)", "let mut vk: c_int = -1;", "let mut vk: c_int = 0;")
m("sc_acquire gets nprot - 1", "prot.as_ptr(), prot.len(), heat, &mut vk", "prot.as_ptr(), prot.len().saturating_sub(1), heat, &mut vk")
m("early return when nothing is needed removed", "if need.is_empty() {\n            return;\n        }", "if false {\n            return;\n        }")
# ---- step 1 (RAM -> held)
m("hit_host not counted", "pool.st.hit_host = pool.st.hit_host.wrapping_add(1);", "pool.st.hit_host = pool.st.hit_host;")
m("pend_take after sc_remove (order swapped)",
  "(ops.pend_take)(key);\n                n.host = self.host_remove(host, key);", "n.host = self.host_remove(host, key);\n                (ops.pend_take)(key);")
m("held slot not remembered (never released)", "                held.push(n.host);\n", "")
m("held slots released in reverse order", "for &h in &held {", "for &h in held.iter().rev() {")
m("pend_take gets the expert number instead of the key (equal only for layer 0)", "(ops.pend_take)(key);", "(ops.pend_take)(n.e);")
m("pend_take missing", "                (ops.pend_take)(key);\n", "")
# ---- step 2 (write-back)
m("write-back stop condition cap_slots == 0 && prefill (|| -> &&)", "if host.cap_slots == 0 || prefill {", "if host.cap_slots == 0 && prefill {")
m("write-back: leave the loop with continue instead of break", "if host.cap_slots == 0 || prefill {\n                break;\n            }", "if host.cap_slots == 0 || prefill {\n                continue;\n            }")
m("write-back: worth never asked (always worth)", "if !self.host_worth(host, v.key) {", "if false {")
m("wb_skipped not counted", "WB_SKIPPED.fetch_add(1, Ordering::SeqCst);", "WB_SKIPPED.fetch_add(0, Ordering::SeqCst);")
m("not worth: no continue (the victim is written back anyway)", "WB_SKIPPED.fetch_add(1, Ordering::SeqCst);\n                continue;", "WB_SKIPPED.fetch_add(1, Ordering::SeqCst);")
m("host_acquire gets heat 0 instead of the victim's heat", "(ops.host_acquire)(v.key, v.heat, &mut dk, &mut dh)", "(ops.host_acquire)(v.key, 0.0, &mut dk, &mut dh)")
m("D2H copy on rstream[0] instead of pool.copy", "raw, pool.copy), site::D2H", "raw, pool.rstream[0]), site::D2H")
m("D2H copies raw - 1 bytes", "v.src as *const c_void, raw, pool.copy), site::D2H", "v.src as *const c_void, raw - 1, pool.copy), site::D2H")
m("written_back not counted", "pool.st.written_back = pool.st.written_back.wrapping_add(1);", "pool.st.written_back = pool.st.written_back;")
# ---- step 3 (event, RAM -> card)
m("event recorded only when the card gave up a victim", "do_check(ops, (ops.event_record)(pool.written_back, pool.copy), site::EVENT_RECORD);",
  "if !victims.is_empty() {\n            do_check(ops, (ops.event_record)(pool.written_back, pool.copy), site::EVENT_RECORD);\n        }")
m("event_record arguments swapped (stream, event)", "(ops.event_record)(pool.written_back, pool.copy)", "(ops.event_record)(pool.copy, pool.written_back)")
m("prefill main copy: slab record index e & 7 instead of e & 3", "((n.e & 3) as usize).wrapping_mul(raw))", "((n.e & 7) as usize).wrapping_mul(raw))")
m("prefill main copy: slab[0] instead of slab[layer]", "(*self.cfg.slab.offset(layer as isize))", "(*self.cfg.slab.offset(0))")
m("main H2D copies raw - 1 bytes", "src as *const c_void, raw, pool.copy), site::H2D_MAIN", "src as *const c_void, raw - 1, pool.copy), site::H2D_MAIN")
m("set_device(pool.dev) missing before the write-backs", "        do_check(ops, (ops.set_device)(pool.dev), site::SET_DEVICE_POOL);\n", "")
# ---- step 4 chunk arithmetic / jobs
m("n_drive = all needs (RAM hits counted too)", "let n_drive: c_int = need.iter().filter(|n| n.host.is_null()).count() as c_int;", "let n_drive: c_int = need.len() as c_int;")
m("nchunk: n_drive > pool_readers instead of >=", "} else if n_drive >= pool_readers || n_drive == 0 {", "} else if n_drive > pool_readers || n_drive == 0 {")
m("nchunk: min(7, ..) instead of min(8, ..)", "std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)", "std::cmp::min(7, (pool_readers + n_drive - 1) / n_drive)")
m("nchunk: min(9, ..) instead of min(8, ..)", "std::cmp::min(8, (pool_readers + n_drive - 1) / n_drive)", "std::cmp::min(9, (pool_readers + n_drive - 1) / n_drive)")
m("nchunk: floor+1 instead of ceil, (readers + n_drive) / n_drive", "(pool_readers + n_drive - 1) / n_drive", "(pool_readers + n_drive) / n_drive")
m("cfg.chunks > 1 instead of > 0 (forced 1 ignored)", "let nchunk: c_int = if cfg_chunks > 0 {", "let nchunk: c_int = if cfg_chunks > 1 {")
m("chunk rounding +4096 instead of +4095", ".wrapping_add(4095) / 4096 * 4096;", ".wrapping_add(4096) / 4096 * 4096;")
m("chunk not rounded to 4096", ").wrapping_add(4095) / 4096 * 4096;", ");")
m("chunk: raw + nchunk instead of raw + nchunk - 1", "raw.wrapping_add(nchunk as usize).wrapping_sub(1) / nchunk as usize", "raw.wrapping_add(nchunk as usize) / nchunk as usize")
m("chunk loop stops at co > raw instead of co >= raw (zero-length job)", "if co >= raw {", "if co > raw {")
m("job len = chunk (not clamped to the end of the record)", "let len = std::cmp::min(chunk, raw - co);", "let len = chunk;")
m("from_drive counted per job instead of per expert",
  "            pool.st.from_drive = pool.st.from_drive.wrapping_add(1);\n            for k in 0..nchunk {\n                let co = (k as usize).wrapping_mul(chunk);\n                if co >= raw {\n                    break;\n                }\n",
  "            for k in 0..nchunk {\n                let co = (k as usize).wrapping_mul(chunk);\n                if co >= raw {\n                    break;\n                }\n                pool.st.from_drive = pool.st.from_drive.wrapping_add(1);\n")
m("job file offset: e * raw as a 32-bit multiply", "off: (n.e as i64).wrapping_mul(raw as i64).wrapping_add(co as i64)", "off: (n.e.wrapping_mul(raw as i32) as i64).wrapping_add(co as i64)")
m("job file offset without the chunk offset", ".wrapping_mul(raw as i64).wrapping_add(co as i64), e: n.e", ".wrapping_mul(raw as i64), e: n.e")
m("job destination without the chunk offset", "Job { dst: n.dst.wrapping_add(co),", "Job { dst: n.dst,")
m("pool_readers = readers (pool.readers ignored)", "let pool_readers: c_int = std::cmp::min(readers, pool.readers);", "let pool_readers: c_int = readers;")
m("pool_readers = pool.readers (readers argument ignored)", "let pool_readers: c_int = std::cmp::min(readers, pool.readers);", "let pool_readers: c_int = pool.readers;")
m("pool_readers clamped to >= 1", "let pool_readers: c_int = std::cmp::min(readers, pool.readers);", "let pool_readers: c_int = std::cmp::min(readers, pool.readers).max(1);")
# ---- reader job
m("job prefill: slab record index e & 7", "((j.e & 3) as usize).wrapping_mul(c.raw)", "((j.e & 7) as usize).wrapping_mul(c.raw)")
m("job prefill: chunk offset not added to the slab address", ".wrapping_mul(c.raw)).wrapping_add(j.co) as *mut u8;", ".wrapping_mul(c.raw)) as *mut u8;")
m("job prefill: slab[0] instead of slab[layer]", "(*c.slab.offset(c.layer as isize))", "(*c.slab.offset(0))")
m("job reads into bounce[0] whatever the thread", "let mut b: *mut u8 = c.bounce[tid];", "let mut b: *mut u8 = c.bounce[0];")
m("pread loop: got = r instead of got += r", "got += r as usize;", "got = r as usize;")
m("pread loop: offset without got", "j.off.wrapping_add(got as i64)", "j.off")
m("pread: r < 0 is fatal, r == 0 is not (spins on EOF)", "if r <= 0 {", "if r < 0 {")
m("EOF text 'EOF' instead of 'end of file'", 'b"end of file".to_vec()', 'b"EOF".to_vec()')
m("fatal text: expert and layer swapped", 'format!("reading expert {} of layer {} failed: ", j.e, c.layer)', 'format!("reading expert {} of layer {} failed: ", c.layer, j.e)')
m("fatal exit status 2 instead of 1", "    write_stderr(msg);\n    std::process::exit(1)\n", "    write_stderr(msg);\n    std::process::exit(2)\n")
m("job: set_device(0) instead of set_device(dev)", "(c.ops.set_device)(c.dev)", "(c.ops.set_device)(0)")
m("job: stream_wait_event flags 1 instead of 0", "(c.ops.stream_wait_event)(c.rstream[tid], c.written_back, 0)", "(c.ops.stream_wait_event)(c.rstream[tid], c.written_back, 1)")
m("job: stream_wait_event on rstream[0] whatever the thread", "(c.ops.stream_wait_event)(c.rstream[tid], c.written_back, 0)", "(c.ops.stream_wait_event)(c.rstream[0], c.written_back, 0)")
m("job: H2D on rstream[0] whatever the thread", "j.len, c.rstream[tid]), site::H2D_JOB", "j.len, c.rstream[0]), site::H2D_JOB")
m("job: final sync on the default stream (NULL)", "(c.ops.stream_sync)(c.rstream[tid])", "(c.ops.stream_sync)(null_mut())")
m("check site text of the job set_device is 'cudaSetDevice(pool.dev)'", 'b"cudaSetDevice(dev)\\0"', 'b"cudaSetDevice(pool.dev)\\0"')
# ---- check
m("do_check: no exit when check returns with a non-zero status", "    if status != 0 {\n        std::process::exit(1); // ops.check must not return", "    if status != 0 && false {\n        std::process::exit(1); // ops.check must not return")
m("do_check: ops.check gets status 0 always", "unsafe { (ops.check)(status, site.as_ptr() as *const c_char) };", "unsafe { (ops.check)(0, site.as_ptr() as *const c_char) };")
# ---- threads
m("run_parallel: one thread fewer is spawned", "for i in 1..n {", "for i in 1..(n - 1).max(1) {")
m("run_parallel: spawned thread numbered i - 1 (tid 0 used twice)", "s.spawn(move || work(c, i as usize));", "s.spawn(move || work(c, (i - 1) as usize));")
m("run_parallel: nthreads not capped by the job count", "let n: c_int = std::cmp::min(nthreads, c.jobs.len() as c_int);", "let n: c_int = nthreads;")
m("work: jobs taken in reverse list order", "unsafe { run_job(c, &c.jobs[i], tid) };", "unsafe { run_job(c, &c.jobs[c.jobs.len() - 1 - i], tid) };")
m("work: end test i > len instead of i >= len", "if i >= c.jobs.len() {", "if i > c.jobs.len() {")
m("job counter fetch_add with Ordering::Relaxed", "c.next.fetch_add(1, Ordering::SeqCst)", "c.next.fetch_add(1, Ordering::Relaxed)")
# ---- end of call
m("final stream_sync(pool.copy) missing", "        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);\n", "")
m("final stream_sync(pool.copy) before the reader jobs",
  "        run_parallel(&ctx, pool_readers);\n        let t1 = Instant::now();\n        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);\n",
  "        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);\n        run_parallel(&ctx, pool_readers);\n        let t1 = Instant::now();\n")
m("held slots released before the final stream_sync",
  "        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);\n        for &h in &held {\n            (ops.sc_release)(host.cache, h);\n        }\n",
  "        for &h in &held {\n            (ops.sc_release)(host.cache, h);\n        }\n        do_check(ops, (ops.stream_sync)(pool.copy), site::SYNC_COPY);\n")
m("t_read = time since t0 (includes the copy sync)", "pool.st.t_read += t1.duration_since(t0).as_secs_f64();", "pool.st.t_read += t0.elapsed().as_secs_f64();")
m("t_copy gets the read interval (t1 - t0)", "pool.st.t_copy += t1.elapsed().as_secs_f64();", "pool.st.t_copy += t1.duration_since(t0).as_secs_f64();")
m("t_read assigned (=) instead of added", "pool.st.t_read += t1.duration_since(t0).as_secs_f64();", "pool.st.t_read = t1.duration_since(t0).as_secs_f64();")
m("t_ensure assigned (=) instead of added", "(*pool).st.t_ensure += t_start.elapsed().as_secs_f64();", "(*pool).st.t_ensure = t_start.elapsed().as_secs_f64();")
# ---- keys, helpers, init
m("ekey uses the constant 896 instead of cfg.experts", "layer.wrapping_mul(self.cfg.experts)", "layer.wrapping_mul(896)")
m("host_worth: a NULL RAM cache is not worth a slot", "host.cache.is_null() || (self.ops.sc_worth)(host.cache, key) != 0", "!host.cache.is_null() && (self.ops.sc_worth)(host.cache, key) != 0")
m("host_find: no NULL guard", "        if host.cache.is_null() {\n            null_mut()\n        } else {\n            (self.ops.sc_find)(host.cache, key)\n        }\n", "        (self.ops.sc_find)(host.cache, key)\n")
m("pool_find: no NULL guard", "        if pool.cache.is_null() {\n            null_mut()\n        } else {\n            (self.ops.sc_find)(pool.cache, key)\n        }\n", "        (self.ops.sc_find)(pool.cache, key)\n")
m("Supply::new does not reset the g_wb_skipped counter", "        WB_SKIPPED.store(0, Ordering::SeqCst);\n", "")
m("abi: sy_stats hit_card and hit_host swapped in the struct", "    pub hit_card: c_ulong,\n    pub hit_host: c_ulong,\n", "    pub hit_host: c_ulong,\n    pub hit_card: c_ulong,\n", "abi.rs")
m("abi: sy_config vq and prefill swapped", "    pub vq: c_int,\n    pub prefill: c_int,\n", "    pub prefill: c_int,\n    pub vq: c_int,\n", "abi.rs")
m("lib: sy_wb_skipped reads the counter with an offset (+1)", "WB_SKIPPED.load(Ordering::SeqCst) as c_ulong", "(WB_SKIPPED.load(Ordering::SeqCst) + 1) as c_ulong", "lib.rs")
m("lib: n == 0 treated as a one-element list when experts is non-NULL", "if n == 0 || experts.is_null() { &[] }", "if experts.is_null() { &[] }", "lib.rs")

MUT = [("M00", "control: unmodified copy", E, "", "")]
for i, (desc, f, old, new) in enumerate(L, 1):
    MUT.append(("M%02d" % i, desc, f, old, new))

if __name__ == "__main__":
    only = sys.argv[1:]
    os.makedirs(WORK, exist_ok=True); os.makedirs(DIFFS, exist_ok=True)
    for name, desc, fn, old, new in MUT:
        if only and name not in only:
            continue
        d = os.path.join(WORK, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for f in ("Cargo.toml", "Cargo.lock"):
            shutil.copy(os.path.join(SRC, f), d)
        shutil.copytree(os.path.join(SRC, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(SRC, "include"), os.path.join(d, "include"))
        shutil.copytree(SCINC, os.path.join(d, "slotcache-include"))
        p = os.path.join(d, "src", fn)
        t0 = open(p, newline="").read()
        if old:
            if t0.count(old) != 1:
                print(name, "NOT APPLIED (count %d)" % t0.count(old), desc, flush=True)
                continue
            t1 = t0.replace(old, new)
            open(p, "w", newline="").write(t1)
        else:
            t1 = t0
        diff = "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/" + fn, "b/src/" + fn, n=3))
        open(os.path.join(DIFFS, name + ".diff"), "w").write("# %s: %s\n%s" % (name, desc, diff))
        r = subprocess.run(["nice", "-n", "10", "cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
        so = os.path.join(d, "target/release/libsupply.so")
        if r.returncode != 0 or not os.path.exists(so):
            print(name, "BUILD FAILED", desc); print((r.stdout + r.stderr)[-1200:], flush=True)
            continue
        print(name, hashlib.md5(open(so, "rb").read()).hexdigest(), desc, flush=True)
