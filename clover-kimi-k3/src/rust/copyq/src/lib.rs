//! Rust implementation of include/copyq.h (`cq_init`, `cq_enqueue`).
pub mod abi;
pub mod cuda;
pub mod engine;

use abi::{cq_config, cq_job};
use cuda::CudaBackend;
use engine::{fatal, Cfg, Engine};
use std::sync::{Arc, OnceLock};

static ENGINE: OnceLock<Arc<Engine<CudaBackend>>> = OnceLock::new();

/// # Safety
/// `cfg` must point to a valid `cq_config` whose pointers (slab, act_hold) stay valid for the life of the process.
#[no_mangle]
pub unsafe extern "C" fn cq_init(cfg: *const cq_config) {
    if cfg.is_null() {
        fatal("copyq: cq_init(NULL)");
    }
    // SAFETY: non-null checked above; caller guarantees validity.
    let c = unsafe { *cfg };
    // The engine's EXPERT_RAW is a compile-time constant; the reference rejects any other value and so does this.
    if c.expert_raw != 17_547_264 {
        fatal(&format!("cq_init: expert_raw {} != compiled EXPERT_RAW 17547264", c.expert_raw));
    }
    let _ = ENGINE.set(Engine::new(Cfg::from_c(&c), CudaBackend::new())); // a second call keeps the first configuration
}

/// # Safety
/// `job` must point to a valid `cq_job`; every pointer inside it must stay valid until the job's flag is set.
#[no_mangle]
pub unsafe extern "C" fn cq_enqueue(job: *const cq_job) {
    if job.is_null() {
        fatal("copyq: cq_enqueue(NULL)");
    }
    match ENGINE.get() {
        // SAFETY: non-null checked above; caller guarantees validity.
        Some(e) => e.enqueue(unsafe { &*job }),
        None => fatal("copyq: cq_enqueue before cq_init"),
    }
}

#[cfg(test)]
mod layout_tests {
    use super::abi::{cq_config, cq_job};
    use std::mem::{offset_of, size_of};
    use std::process::Command;

    macro_rules! offs {
        ($t:ty, $c:literal, [$($f:ident),*]) => { vec![ $( (format!("{}.{}", $c, stringify!($f)), offset_of!($t, $f)) ),* ] };
    }

    #[test]
    fn struct_layout_matches_the_c_header() {
        let job = ["kind", "dev", "fd", "off", "layer", "slot", "p_state", "dst", "host_src", "victim_host", "wb_state", "arr_state", "rel_state"];
        let cfg = ["expert_raw", "readers", "cards", "prefill", "slab", "act_hold"];
        let mut c = String::from("#include <stdio.h>\n#include <stddef.h>\n#include \"copyq.h\"\nint main(void){\n");
        c += "printf(\"job.size=%zu\\n\", sizeof(cq_job));\nprintf(\"cfg.size=%zu\\n\", sizeof(cq_config));\n";
        for f in job { c += &format!("printf(\"job.{f}=%zu\\n\", offsetof(cq_job, {f}));\n"); }
        for f in cfg { c += &format!("printf(\"cfg.{f}=%zu\\n\", offsetof(cq_config, {f}));\n"); }
        c += "return 0;}\n";

        let mut want: Vec<(String, usize)> = vec![("job.size".into(), size_of::<cq_job>()), ("cfg.size".into(), size_of::<cq_config>())];
        want.extend(offs!(cq_job, "job", [kind, dev, fd, off, layer, slot, p_state, dst, host_src, victim_host, wb_state, arr_state, rel_state]));
        want.extend(offs!(cq_config, "cfg", [expert_raw, readers, cards, prefill, slab, act_hold]));

        let dir = std::env::temp_dir().join(format!("copyq_layout_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let src = dir.join("layout.c");
        let exe = dir.join("layout");
        std::fs::write(&src, c).expect("write C source");
        let inc = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
        let st = Command::new("cc").arg("-I").arg(inc).arg(&src).arg("-o").arg(&exe).status().expect("a C compiler named `cc` is required for this test");
        assert!(st.success(), "C compile of copyq.h failed");
        let out = Command::new(&exe).output().expect("run layout probe");
        let got: Vec<(String, usize)> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.parse().unwrap_or(usize::MAX))))
            .collect();
        assert_eq!(got.len(), want.len());
        for w in &want {
            assert!(got.contains(w), "layout mismatch for {}: rust {} vs C {:?}", w.0, w.1, got.iter().find(|g| g.0 == w.0));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
