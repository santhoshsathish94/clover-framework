#!/usr/bin/env python3
# Builds the mutant crates for the differential-driver check of the loader crate (independent verifier, 2026-10-25). The REAL crate is only read.
# For each mutant: ~/rustmig/loader/verify/work/mut/<name>/ (Cargo.toml, Cargo.lock, src/, include/), a diff in ~/rustmig/loader/verify/mutants/<name>.diff (first line = description),
# `cargo build --release` there, the .so copied to ~/rustmig/loader/verify/work/so/<name>.so and its md5 appended to ~/rustmig/loader/verify/mutants/md5.txt.
# usage: mk_mutants.py [--list] [NAME ...]      (no names = all)
import difflib, hashlib, os, shutil, subprocess, sys

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "rustmig/loader")
WORK = os.path.join(SRC, "verify/work/mut")
SODIR = os.path.join(SRC, "verify/work/so")
DIFFS = os.path.join(SRC, "verify/mutants")
env = dict(os.environ, PATH=os.path.join(HOME, ".cargo/bin") + ":" + os.environ["PATH"])

Q, J, L, S = "queue.rs", "job.rs", "lib.rs", "sys.rs"

SUBMIT_HEAD = "        *lock(&self.pending) += 1;\n        let c = &self.cards[card];\n        let mut g = lock(&c.st);\n"
WORKER_JOB = "            job();\n            *lock(&self.pending) -= 1;\n            self.pending_cv.notify_all();\n"
WAIT = "        let mut p = lock(&self.pending);\n        while *p != 0 {\n            p = self.pending_cv.wait(p).unwrap_or_else(|e| e.into_inner());\n        }\n"
POP = "            let job = {\n                let mut g = lock(&c.st);\n                loop {\n                    if let Some(j) = g.q.pop_front() {\n                        break j;\n                    }\n                    g = c.cv.wait(g).unwrap_or_else(|e| e.into_inner());\n                }\n            };\n"
CALLS = ("        (cb.set_device)(cb.ctx, j.card);\n        (cb.ensure_work)(cb.ctx, j.card);\n        (cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer as c_int);\n"
         "        if j.layer >= 1 {\n            (cb.load_expert_tables)(cb.ctx, j.handle, dataset, j.layer as c_int);\n        }\n        (cb.alloc_state)(cb.ctx, j.handle, j.positions);\n")
BOOK = ("    // bookkeeping, in the original's order\n    put(r.h_L, r.layer as c_int);\n    put(r.h_card, r.card);\n    put(r.h_k_in, r.k as c_int);\n    if r.layer <= N_LAYERS {\n"
        "        // SAFETY: handles is the caller's table of N_LAYERS + 1 pointers (header) and layer <= N_LAYERS.\n"
        "        unsafe { *(st.cfg.handles as *mut *mut c_void).add(r.layer as usize) = r.handle }\n    }\n")
BOOK_BODY = BOOK.split("\n", 1)[1]
SUBMIT_END = "        }),\n    );\n}\n"
OPEN1 = "    let mut fd = sys.open(&cpath, O_RDONLY | O_DIRECT);\n    set_fd(j.h_fd, fd);\n"
FALLBACK = "    if fd < 0 {\n        fd = sys.open(&cpath, O_RDONLY);\n        set_fd(j.h_fd, fd);\n    }\n"
CHK = "        ld_done = 1;\n"

# (name, description, [(file, old, new), ...])   one logical change each; several hunks only where the bug needs them (stated in the description)
M = [
    # ---- queue.rs: per-card FIFO, threads, counter, wait
    ("M01", "per-card FIFO broken: LIFO (pop_back)", [(Q, "if let Some(j) = g.q.pop_front() {", "if let Some(j) = g.q.pop_back() {")]),
    ("M02", "job stealing: an idle worker pops a job of another card's queue (try_lock) before it sleeps",
     [(Q, "                    if let Some(j) = g.q.pop_front() {\n                        break j;\n                    }\n                    g = c.cv.wait(g)",
          "                    if let Some(j) = g.q.pop_front() {\n                        break j;\n                    }\n"
          "                    if let Some(j) = (0..MAX_CARDS).filter(|&o| o != card).find_map(|o| self.cards[o].st.try_lock().ok().and_then(|mut x| x.q.pop_front())) {\n                        break j;\n                    }\n"
          "                    g = c.cv.wait(g)")]),
    ("M03", "one shared queue and one thread for all cards (queue 0, worker(0))",
     [(Q, SUBMIT_HEAD, SUBMIT_HEAD.replace("self.cards[card]", "self.cards[0]")), (Q, "spawn(move || me.worker(card))", "spawn(move || me.worker(0))")]),
    ("M04", "a thread per job: `started` is never set", [(Q, "            g.started = true;\n", "")]),
    ("M05", "pending++ after the push and after the card lock is released (racy window)",
     [(Q, SUBMIT_HEAD, SUBMIT_HEAD.replace("        *lock(&self.pending) += 1;\n", "")),
      (Q, "        g.q.push_back(job);\n        c.cv.notify_one();\n    }", "        g.q.push_back(job);\n        c.cv.notify_one();\n        drop(g);\n        *lock(&self.pending) += 1;\n    }")]),
    ("M06", "pending++ after the thread start, just before the push (card lock still held)",
     [(Q, SUBMIT_HEAD, SUBMIT_HEAD.replace("        *lock(&self.pending) += 1;\n", "")),
      (Q, "        g.q.push_back(job);\n        c.cv.notify_one();\n    }", "        *lock(&self.pending) += 1;\n        g.q.push_back(job);\n        c.cv.notify_one();\n    }")]),
    ("M07", "pending-- before the job runs (count lowered before the job ended)",
     [(Q, WORKER_JOB, "            *lock(&self.pending) -= 1;\n            self.pending_cv.notify_all();\n            job();\n")]),
    ("M08", "pending-- never executed (notify only)", [(Q, "            *lock(&self.pending) -= 1;\n", "")]),
    ("M09", "no notify after the decrement (lost wake-up of ld_wait)", [(Q, "            self.pending_cv.notify_all();\n", "")]),
    ("M10", "notify_all -> notify_one after the decrement", [(Q, "self.pending_cv.notify_all();", "self.pending_cv.notify_one();")]),
    ("M11", "notify_all before the decrement (wrong order, wake-up can be lost)",
     [(Q, WORKER_JOB, "            job();\n            self.pending_cv.notify_all();\n            *lock(&self.pending) -= 1;\n")]),
    ("M12", "wait returns early: while count > 1", [(Q, "while *p != 0 {", "while *p > 1 {")]),
    ("M13", "wait: `if` instead of `while` (one wake-up ends the wait)",
     [(Q, WAIT, "        let mut p = lock(&self.pending);\n        if *p != 0 {\n            p = self.pending_cv.wait(p).unwrap_or_else(|e| e.into_inner());\n        }\n        drop(p);\n")]),
    ("M14", "wait: count checked, lock released, then wait without re-check (lost wake-up window)",
     [(Q, WAIT, "        loop {\n            if *lock(&self.pending) == 0 {\n                return;\n            }\n            let g = lock(&self.pending);\n            drop(self.pending_cv.wait(g).unwrap_or_else(|e| e.into_inner()));\n        }\n")]),
    ("M15", "as M14 with a 300 us sleep between the check and the wait (window widened, shows the driver's sensitivity)",
     [(Q, WAIT, "        loop {\n            if *lock(&self.pending) == 0 {\n                return;\n            }\n            std::thread::sleep(std::time::Duration::from_micros(300));\n            let g = lock(&self.pending);\n            drop(self.pending_cv.wait(g).unwrap_or_else(|e| e.into_inner()));\n        }\n")]),
    ("M16", "no notify_one after the push in submit (worker is only woken by its own first look)", [(Q, "        c.cv.notify_one();\n", "")]),
    ("M17", "submit: notify_one -> notify_all (one waiter per card, expected equivalent)", [(Q, "c.cv.notify_one();", "c.cv.notify_all();")]),
    ("M18", "a job runs while the card's queue lock is held",
     [(Q, POP + "            job();\n",
          "            let g = {\n                let mut g = lock(&c.st);\n                loop {\n                    if let Some(j) = g.q.pop_front() {\n                        j();\n                        break g;\n                    }\n                    g = c.cv.wait(g).unwrap_or_else(|e| e.into_inner());\n                }\n            };\n            drop(g);\n")]),
    ("M19", "card 7 uses card 6's queue and thread (cards[card.min(6)])", [(Q, SUBMIT_HEAD, SUBMIT_HEAD.replace("self.cards[card]", "self.cards[card.min(6)]"))]),
    ("M20", "thread start hook (set_device) missing", [(Q, "        (self.on_start)(card);\n", "")]),
    ("M21", "thread start hook gets card 0 instead of card", [(Q, "(self.on_start)(card);", "(self.on_start)(0);")]),
    ("M22", "thread start hook runs on the submitting thread (before the spawn), not on the card's thread",
     [(Q, "        (self.on_start)(card);\n", ""), (Q, "            let me: &'static Loader = self;\n", "            (self.on_start)(card);\n            let me: &'static Loader = self;\n")]),
    ("M23", "threads started eagerly: ld_init submits a no-op to every card", [(L, "    let _ = STATE.set(State { cfg: conf, loader });", "    for c in 0..MAX_CARDS {\n        loader.submit(c, Box::new(|| {}));\n    }\n    let _ = STATE.set(State { cfg: conf, loader });")]),
    ("M24", "thread joined in submit (never detached): the submitting call blocks", [(Q, "spawn(move || me.worker(card)) {", "spawn(move || me.worker(card)).map(|h| { let _ = h.join(); }) {")]),
    ("M25", "first-submit guard removed: `started` is read outside the lock and set later (the original's check-then-act)",
     [(Q, "        let mut g = lock(&c.st);\n        if !g.started {\n", "        let started = lock(&c.st).started;\n        let mut g = lock(&c.st);\n        if !started {\n")]),
    ("M26", "as M25 with a 100 us sleep between the check and the set (window widened)",
     [(Q, "        let mut g = lock(&c.st);\n        if !g.started {\n", "        let started = lock(&c.st).started;\n        std::thread::sleep(std::time::Duration::from_micros(100));\n        let mut g = lock(&c.st);\n        if !started {\n")]),
    ("M27", "thread stack size left at the default (2 MB) instead of 8 MB", [(Q, ".stack_size(8 << 20)", "")]),
    # ---- job.rs: callback order and arguments
    ("M30", "set_device after ensure_work", [(J, CALLS, CALLS.replace("        (cb.set_device)(cb.ctx, j.card);\n        (cb.ensure_work)(cb.ctx, j.card);\n", "        (cb.ensure_work)(cb.ctx, j.card);\n        (cb.set_device)(cb.ctx, j.card);\n"))]),
    ("M31", "expert tables loaded before the trunk",
     [(J, CALLS, "        (cb.set_device)(cb.ctx, j.card);\n        (cb.ensure_work)(cb.ctx, j.card);\n        if j.layer >= 1 {\n            (cb.load_expert_tables)(cb.ctx, j.handle, dataset, j.layer as c_int);\n        }\n        (cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer as c_int);\n        (cb.alloc_state)(cb.ctx, j.handle, j.positions);\n")]),
    ("M32", "alloc_state before the trunk",
     [(J, CALLS, "        (cb.set_device)(cb.ctx, j.card);\n        (cb.ensure_work)(cb.ctx, j.card);\n        (cb.alloc_state)(cb.ctx, j.handle, j.positions);\n        (cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer as c_int);\n        if j.layer >= 1 {\n            (cb.load_expert_tables)(cb.ctx, j.handle, dataset, j.layer as c_int);\n        }\n")]),
    ("M33", "alloc_state after the experts file is opened (layer >= 1), before the prefill",
     [(J, "        (cb.alloc_state)(cb.ctx, j.handle, j.positions);\n    }\n    if j.layer < 1 {\n        return Ok(());\n    }\n",
          "        if j.layer < 1 {\n            (cb.alloc_state)(cb.ctx, j.handle, j.positions);\n        }\n    }\n    if j.layer < 1 {\n        return Ok(());\n    }\n"),
      (J, "    if j.prefill {\n        let raw = j.expert_raw;\n", "    // SAFETY: as above.\n    unsafe { (cb.alloc_state)(cb.ctx, j.handle, j.positions) };\n    if j.prefill {\n        let raw = j.expert_raw;\n")]),
    ("M34", "expert tables also loaded for layer 0", [(J, "        if j.layer >= 1 {\n            (cb.load_expert_tables)", "        if true {\n            (cb.load_expert_tables)")]),
    ("M35", "expert tables skipped for layer 1 (`> 1`)", [(J, "        if j.layer >= 1 {\n            (cb.load_expert_tables)", "        if j.layer > 1 {\n            (cb.load_expert_tables)")]),
    ("M36", "layer 1 treated like layer 0 (no file, no prefill): `< 2`", [(J, "    if j.layer < 1 {\n        return Ok(());", "    if j.layer < 2 {\n        return Ok(());")]),
    ("M37", "alloc_state receives layer instead of positions", [(J, "(cb.alloc_state)(cb.ctx, j.handle, j.positions);\n    }", "(cb.alloc_state)(cb.ctx, j.handle, j.layer);\n    }")]),
    ("M38", "ensure_work receives card 0", [(J, "(cb.ensure_work)(cb.ctx, j.card);", "(cb.ensure_work)(cb.ctx, 0);")]),
    ("M39", "load_trunk receives layer - 1", [(J, "(cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer as c_int);", "(cb.load_trunk)(cb.ctx, j.handle, dataset, j.layer.saturating_sub(1) as c_int);")]),
    ("M40", "set_device receives a NULL ctx", [(J, "(cb.set_device)(cb.ctx, j.card);", "(cb.set_device)(std::ptr::null_mut(), j.card);")]),
    # ---- job.rs: path, opens
    ("M41", "path cut at 4096 bytes instead of 4095", [(J, "p.truncate(4095);", "p.truncate(4096);")]),
    ("M42", "path cut at 4094 bytes", [(J, "p.truncate(4095);", "p.truncate(4094);")]),
    ("M43", "path not cut at all", [(J, "    p.truncate(4095);\n", "")]),
    ("M44", "file name experts.dat instead of experts.direct", [(J, '"/expert-{}/experts.direct"', '"/expert-{}/experts.dat"')]),
    ("M45", "O_DIRECT fallback removed (no second open)", [(J, FALLBACK, "")]),
    ("M46", "fallback only when the first open failed with EINVAL (errno 22)", [(J, "    if fd < 0 {\n        fd = sys.open(&cpath, O_RDONLY);", "    if fd < 0 && std::io::Error::last_os_error().raw_os_error() == Some(22) {\n        fd = sys.open(&cpath, O_RDONLY);")]),
    ("M47", "first open without O_DIRECT", [(J, "let mut fd = sys.open(&cpath, O_RDONLY | O_DIRECT);", "let mut fd = sys.open(&cpath, O_RDONLY);")]),
    ("M48", "fallback open also uses O_DIRECT", [(J, "        fd = sys.open(&cpath, O_RDONLY);", "        fd = sys.open(&cpath, O_RDONLY | O_DIRECT);")]),
    ("M49", "result of the fallback open not stored in *h_fd", [(J, "        fd = sys.open(&cpath, O_RDONLY);\n        set_fd(j.h_fd, fd);\n", "        fd = sys.open(&cpath, O_RDONLY);\n")]),
    ("M50", "result of the first open stored only if >= 0 (the -1 is never stored before the fallback; expected equivalent)", [(J, OPEN1, "    let mut fd = sys.open(&cpath, O_RDONLY | O_DIRECT);\n    if fd >= 0 {\n        set_fd(j.h_fd, fd);\n    }\n")]),
    ("M51", "fd 0 treated as a failed open (`fd <= 0`)", [(J, "    if fd < 0 {\n        let mut m = b\"cannot open \"", "    if fd <= 0 {\n        let mut m = b\"cannot open \"")]),
    ("M52", "message `cannot open` has no trailing newline", [(J, "        m.push(b'\\n');\n", "")]),
    ("M53", "message text `cannot open file `", [(J, 'b"cannot open "', 'b"cannot open file "')]),
    # ---- job.rs: prefill
    ("M54", "slab block of 3 records (3 * raw)", [(J, "let bytes = 4usize.wrapping_mul(raw);", "let bytes = 3usize.wrapping_mul(raw);")]),
    ("M55", "slab block of 5 records (5 * raw)", [(J, "let bytes = 4usize.wrapping_mul(raw);", "let bytes = 5usize.wrapping_mul(raw);")]),
    ("M56", "3 records read instead of 4", [(J, "for r in 0..4usize {", "for r in 0..3usize {")]),
    ("M57", "5 records read instead of 4", [(J, "for r in 0..4usize {", "for r in 0..5usize {")]),
    ("M58", "record offset r * raw computed in 32 bits (u32 multiply)", [(J, "let off = (r as i64).wrapping_mul(raw as i64).wrapping_add(got as i64);", "let off = ((r as u32).wrapping_mul(raw as u32) as i64).wrapping_add(got as i64);")]),
    ("M59", "slab size 4 * raw computed in 32 bits", [(J, "let bytes = 4usize.wrapping_mul(raw);", "let bytes = (4u32.wrapping_mul(raw as u32)) as usize;")]),
    ("M60", "pread offset does not advance with `got`", [(J, "let off = (r as i64).wrapping_mul(raw as i64).wrapping_add(got as i64);", "let off = (r as i64).wrapping_mul(raw as i64);")]),
    ("M61", "pread buffer does not advance with `got`", [(J, ".wrapping_add(r.wrapping_mul(raw)).wrapping_add(got) }", ".wrapping_add(r.wrapping_mul(raw)) }")]),
    ("M62", "pread length is raw instead of raw - got", [(J, "sys.pread(fd, buf, raw - got, off)", "sys.pread(fd, buf, raw, off)")]),
    ("M63", "got = rd instead of got += rd", [(J, "got += rd as usize;", "got = rd as usize;")]),
    ("M64", "no read loop: one pread per record (`if got < raw`)", [(J, "while got < raw {", "if got < raw {")]),
    ("M65", "end of file not detected: `rd < 0`", [(J, "if rd <= 0 {", "if rd < 0 {")]),
    ("M66", "read error not detected: `rd == 0`", [(J, "if rd <= 0 {", "if rd == 0 {")]),
    ("M67", "EINTR retried (unbounded)", [(J, "                if rd <= 0 {", "                if rd < 0 && std::io::Error::last_os_error().raw_os_error() == Some(4) {\n                    continue;\n                }\n                if rd <= 0 {")]),
    ("M68", "prefill message text `a record` -> `record`", [(J, "cannot read a record of layer", "cannot read record of layer")]),
    ("M69", "host_alloc stores into slab[layer - 1]", [(J, "f(cb.ctx, j.slab.add(layer), bytes)", "f(cb.ctx, j.slab.add(layer - 1), bytes)")]),
    # ---- lib.rs
    ("M70", "handles[layer] stored only for layer < 92", [(L, "    if r.layer <= N_LAYERS {\n        // SAFETY: handles", "    if r.layer < N_LAYERS {\n        // SAFETY: handles")]),
    ("M71", "handles[layer] stored for layer <= 93 (writes past the table)", [(L, "    if r.layer <= N_LAYERS {\n        // SAFETY: handles", "    if r.layer <= N_LAYERS + 1 {\n        // SAFETY: handles")]),
    ("M72", "slab-slot check `layer >= 92` (92 refused with prefill)", [(L, "layer >= 1 && layer > N_LAYERS", "layer >= 1 && layer >= N_LAYERS")]),
    ("M73", "slab-slot check `layer > 93` (93 passes with prefill)", [(L, "layer >= 1 && layer > N_LAYERS", "layer >= 1 && layer > N_LAYERS + 1")]),
    ("M74", "N_LAYERS = 93", [(L, "pub const N_LAYERS: u32 = 92;", "pub const N_LAYERS: u32 = 93;")]),
    ("M75", "slab-slot check without `layer >= 1 &&` (implied by layer > 92; expected equivalent)", [(L, "layer >= 1 && layer > N_LAYERS", "layer > N_LAYERS")]),
    ("M76", "negative card not rejected (`card < 0 ||` removed)", [(L, "if card < 0 || card as usize >= MAX_CARDS {", "if card as usize >= MAX_CARDS {")]),
    ("M77", "card 8 not rejected (`> MAX_CARDS`)", [(L, "card as usize >= MAX_CARDS {", "card as usize > MAX_CARDS {")]),
    ("M78", "*h_L receives the card", [(L, "put(r.h_L, r.layer as c_int);", "put(r.h_L, r.card);")]),
    ("M79", "*h_card receives the layer", [(L, "put(r.h_card, r.card);", "put(r.h_card, r.layer as c_int);")]),
    ("M80", "*h_k_in receives positions", [(L, "put(r.h_k_in, r.k as c_int);", "put(r.h_k_in, r.positions as c_int);")]),
    ("M81", "*h_k_in saturated to i32::MAX", [(L, "put(r.h_k_in, r.k as c_int);", "put(r.h_k_in, r.k.min(i32::MAX as u32) as c_int);")]),
    ("M82", "bookkeeping (*h_L, *h_card, *h_k_in, handles[layer]) written after the job is queued (before the call returns)",
     [(L, BOOK, ""), (L, SUBMIT_END, "        }),\n    );\n" + BOOK_BODY + "}\n")]),
    ("M83", "as M82 with a 2 ms sleep before the writes (window widened; the callbacks never read the fields in the driver)",
     [(L, BOOK, ""), (L, SUBMIT_END, "        }),\n    );\n    std::thread::sleep(std::time::Duration::from_millis(2));\n" + BOOK_BODY + "}\n")]),
    ("M84", "the job reads *h_card when it runs, not the card captured at the call",
     [(L, "    let job = LayerJob {\n", "    let hc = r.h_card as usize;\n    let job = LayerJob {\n"),
      (L, "        Box::new(move || {\n            if let Err(m) = job::run(&RealSys, &job) {", "        Box::new(move || {\n            let mut job = job;\n            if hc != 0 {\n                // SAFETY: mutant\n                job.card = unsafe { *(hc as *const c_int) };\n            }\n            if let Err(m) = job::run(&RealSys, &job) {")]),
    ("M85", "the job re-reads the caller's dataset string when it runs (not the copy made at the call)",
     [(L, "    let job = LayerJob {\n", "    let dsp = r.dataset as usize;\n    let job = LayerJob {\n"),
      (L, "        Box::new(move || {\n            if let Err(m) = job::run(&RealSys, &job) {", "        Box::new(move || {\n            let mut job = job;\n            // SAFETY: mutant\n            job.ds = unsafe { CStr::from_ptr(dsp as *const std::os::raw::c_char) }.to_owned();\n            if let Err(m) = job::run(&RealSys, &job) {")]),
    ("M86", "second ld_init not ignored early (the check removed)", [(L, "    if STATE.get().is_some() {\n        return; // the first configuration is kept\n    }\n", "")]),
    ("M87", "ld_init: already-initialised check before the NULL check",
     [(L, "    if cfg.is_null() {\n        fatal_msg(b\"loader: cfg is NULL\\n\");\n    }\n    if STATE.get().is_some() {\n        return; // the first configuration is kept\n    }\n",
          "    if STATE.get().is_some() {\n        return; // the first configuration is kept\n    }\n    if cfg.is_null() {\n        fatal_msg(b\"loader: cfg is NULL\\n\");\n    }\n")]),
    ("M88", "ld_wait before ld_init is fatal", [(L, "    if let Some(s) = STATE.get() {\n        s.loader.wait();\n    } // before ld_init nothing can be pending\n", "    state(\"ld_wait\").loader.wait();\n")]),
    ("M89", "ld_submit with a NULL job is ignored silently", [(L, "None => fatal_msg(b\"loader: job is NULL\\n\"),", "None => return,")]),
    ("M90", "ld_submit does not check the card", [(L, "    die(check_card(card));\n    let arg = Arg(arg);", "    let arg = Arg(arg);")]),
    ("M91", "ld_open_layer does not check the card", [(L, "    die(check_card(r.card));\n", "")]),
    ("M92", "ld_open_layer: slab-slot check before the card check (guard order swapped)", [(L, "    die(check_card(r.card));\n    die(check_slab(st.cfg.prefill, r.layer));\n", "    die(check_slab(st.cfg.prefill, r.layer));\n    die(check_card(r.card));\n")]),
    ("M93", "expert_raw fixed at 4096", [(L, "expert_raw: st.cfg.expert_raw,", "expert_raw: 4096,")]),
    ("M94", "prefill always on in the job", [(L, "prefill: st.cfg.prefill,", "prefill: true,")]),
    ("M95", "handles[layer] receives NULL instead of the handle", [(L, "add(r.layer as usize) = r.handle }", "add(r.layer as usize) = std::ptr::null_mut() }")]),
    ("M96", "ld_init does not require `handles`", [(L, "    if c.handles.is_null() {\n        return Err(n(\"handles\"));\n    }\n", "")]),
    ("M97", "ld_init does not require host_alloc with prefill", [(L, "    if c.prefill != 0 && c.cb.host_alloc.is_none() {\n        return Err(n(\"host_alloc\"));\n    }\n", "")]),
    # ---- sys.rs
    ("M98", "fatal exit status 2", [(S, "unsafe { exit(1) }", "unsafe { exit(2) }")]),
    ("M99", "fatal exit with _exit (no atexit handlers, no stdio flush)", [(S, "    fn exit(code: c_int) -> !;", "    #[link_name = \"_exit\"]\n    fn exit(code: c_int) -> !;")]),
    ("MA1", "fatal message written to stdout (fd 1)", [(S, "write(2, msg[done..]", "write(1, msg[done..]")]),
    ("MA2", "O_DIRECT constant of aarch64 (0o200000) on x86_64", [(S, "pub const O_DIRECT: c_int = 0o40000;", "pub const O_DIRECT: c_int = 0o200000;")]),
    # ---- added after the first campaign: widened-window twins of the racy mutants (the window is real, only its natural width is the question)
    ("M11w", "as M11 with a 200 us sleep between the notify and the decrement (window widened)",
     [(Q, WORKER_JOB, "            job();\n            self.pending_cv.notify_all();\n            std::thread::sleep(std::time::Duration::from_micros(200));\n            *lock(&self.pending) -= 1;\n")]),
    ("M05w", "as M05 with a 200 us sleep between the push and the pending++ (window widened)",
     [(Q, SUBMIT_HEAD, SUBMIT_HEAD.replace("        *lock(&self.pending) += 1;\n", "")),
      (Q, "        g.q.push_back(job);\n        c.cv.notify_one();\n    }", "        g.q.push_back(job);\n        c.cv.notify_one();\n        drop(g);\n        std::thread::sleep(std::time::Duration::from_micros(200));\n        *lock(&self.pending) += 1;\n    }")]),
]
M.insert(0, ("M00", "control: unmodified copy", []))


def crlf(s, like):
    return s.replace("\n", "\r\n") if "\r\n" in like else s


def main():
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        for n, d, e in M:
            print(n, d)
        return
    os.makedirs(WORK, exist_ok=True); os.makedirs(SODIR, exist_ok=True); os.makedirs(DIFFS, exist_ok=True)
    md5f = os.path.join(DIFFS, "md5.txt")
    for name, desc, edits in M:
        if args and name not in args:
            continue
        d = os.path.join(WORK, name)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        for f in ("Cargo.toml", "Cargo.lock"):
            shutil.copy(os.path.join(SRC, f), d)
        for sub in ("src", "include"):
            shutil.copytree(os.path.join(SRC, sub), os.path.join(d, sub))
        diff = ""
        bad = False
        for fn in sorted(set(e[0] for e in edits)):
            p = os.path.join(d, "src", fn)
            t0 = open(p, newline="").read()
            t1 = t0
            for f2, old, new in edits:
                if f2 != fn:
                    continue
                o, n = crlf(old, t0), crlf(new, t0)
                if t1.count(o) != 1:
                    print(name, "NOT APPLIED (count %d) %r" % (t1.count(o), old[:60]), flush=True); bad = True; break
                t1 = t1.replace(o, n)
            if bad:
                break
            open(p, "w", newline="").write(t1)
            diff += "".join(difflib.unified_diff(t0.splitlines(True), t1.splitlines(True), "a/src/" + fn, "b/src/" + fn, n=3))
        if bad:
            continue
        open(os.path.join(DIFFS, name + ".diff"), "w", newline="").write("# %s: %s\n%s" % (name, desc, diff.replace("\r", "")))
        r = subprocess.run(["nice", "-n", "10", "cargo", "build", "--release"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
        so = os.path.join(d, "target/release/libloader.so")
        if r.returncode != 0 or not os.path.exists(so):
            print(name, "BUILD FAILED"); print((r.stdout + r.stderr)[-1200:], flush=True)
            continue
        shutil.copy(so, os.path.join(SODIR, name + ".so"))
        h = hashlib.md5(open(so, "rb").read()).hexdigest()
        with open(md5f, "a") as f:
            f.write("%s  %s  %s\n" % (h, name, desc))
        warn = r.stderr.count("warning")
        print(name, h, ("warnings=%d " % warn if warn else "") + desc, flush=True)


if __name__ == "__main__":
    main()