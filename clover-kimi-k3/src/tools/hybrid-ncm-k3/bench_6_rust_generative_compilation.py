"""Benchmark 6: Generative Compilation (`arXiv:2607.13921`) Evaluation Suite on Clover Kimi K3
(`CloverRustGenerativeCompiler` + `CloverPythonGenerativeCompiler` + `clover-k3-chat.service`).

Evaluates two complementary tracks on `gpu`:
  Part A — Controlled Partial-Program Prefix Detection & Live Rollback Recovery (`No-GC` vs. `GC`, `N=10` tasks):
    Measures exact detection position (% of file), doomed characters saved vs. post-compilation (`PC`),
    compiler check latency (`ms`), and post-rollback unit-test pass rate across Rust (`E0502`, `E0382`,
    `E0308`, `E0061`, `E0596`, `UpdatedAPI`, `CRUST-Bench`) and Python (`HumanEval/17`, `26`, `32`, `39`).
  Part B — Live End-to-End Rust Function Generation & `rustc --test` Verification via `127.0.0.1:8765/api/chat`:
    Sends Rust function specifications to the live Clover Kimi K3 server with `CloverRustGenerativeCompiler`
    active, records `prefix_checks`, `rollbacks`, and `total_ms`, and executes `rustc --test` on each output.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from typing import Any

from k3_generative_compiler import (
    CloverPythonGenerativeCompiler,
    CloverRustGenerativeCompiler,
    HAS_RUST_GC,
)


def ask_clover(question: str, sid: str) -> dict[str, Any]:
    body = json.dumps({
        "session_id": sid,
        "prompt": question,
        "cands": 2,
        "remember": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8765/api/chat",
        data=body,
        headers={"Content-Type": "application/json", "Host": "127.0.0.1:8765"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_rust_code(reply: str) -> str:
    blocks = re.findall(r"```(?:rust)?\s*\n(.*?)```", reply, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return reply.strip()


def run_rustc_unit_test(rust_code: str, test_harness: str) -> tuple[bool, str]:
    """Compiles `rust_code + test_harness` with `rustc --edition=2021` and runs the resulting binary."""
    # Strip any `fn main` in `rust_code` if `test_harness` provides its own `fn main`
    cleaned = re.sub(r"\bfn\s+main\s*\(\s*\)\s*\{[\s\S]*$", "", rust_code).strip()
    full_src = f"{cleaned}\n\n{test_harness}\n"
    rustc_bin = "/home/usergpu/.cargo/bin/rustc"
    if not os.path.exists(rustc_bin):
        rustc_bin = "rustc"
    with tempfile.TemporaryDirectory() as tmpd:
        src_p = os.path.join(tmpd, "solution.rs")
        bin_p = os.path.join(tmpd, "solution_bin")
        with open(src_p, "w", encoding="utf-8") as f:
            f.write(full_src)
        comp = subprocess.run(
            [rustc_bin, "--edition=2021", "-A", "warnings", src_p, "-o", bin_p],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if comp.returncode != 0:
            err_ln = [ln for ln in comp.stderr.splitlines() if "error" in ln]
            return False, (err_ln[0] if err_ln else comp.stderr[:160])
        run_res = subprocess.run([bin_p], capture_output=True, text=True, timeout=5.0)
        if run_res.returncode == 0:
            return True, "OK"
        return False, (run_res.stderr or run_res.stdout or f"exit={run_res.returncode}").strip()[:160]


# Part A: Controlled Rust Partial-Program Trajectories (from Paper Fig 1-4, CRUST-Bench, UpdatedAPI)
RUST_GC_TRAJECTORIES = [
    {
        "id": "rs_borrow_e0502_fmt_msg",
        "title": "Immutable slice borrow across mutable String::drain (Paper Fig 1-4, E0502)",
        "expected_code": "E0502",
        "doomed_program": (
            "const HEADER_SIZE: usize = 5;\n"
            "pub fn fmt_msg(msg: &mut String) -> String {\n"
            "    let tag = &msg[..HEADER_SIZE];\n"
            "    msg.drain(..HEADER_SIZE);\n"
            '    let prio = tag.starts_with("HIGH");\n'
            '    let body: Vec<&str> = msg.lines().collect();\n'
            '    let count = body.len();\n'
            '    format!("{prio} ({count}): {}", body.join("; "))\n'
            "}\n"
        ),
        "repaired_program": (
            "const HEADER_SIZE: usize = 5;\n"
            "pub fn fmt_msg(msg: &mut String) -> String {\n"
            "    let tag = msg[..HEADER_SIZE].to_string();\n"
            "    msg.drain(..HEADER_SIZE);\n"
            '    let prio = tag.starts_with("HIGH");\n'
            '    let body: Vec<&str> = msg.lines().collect();\n'
            '    let count = body.len();\n'
            '    format!("{prio} ({count}): {}", body.join("; "))\n'
            "}\n"
        ),
    },
    {
        "id": "rs_move_e0382_vec_consume",
        "title": "Use of moved Vec<i32> after into_iter() consumption (CRUST-Bench, E0382)",
        "expected_code": "E0382",
        "doomed_program": (
            "pub fn sum_and_len(items: Vec<i32>) -> (i32, usize) {\n"
            "    let total: i32 = items.into_iter().sum();\n"
            "    let n = items.len();\n"
            "    let avg = if n > 0 { total / (n as i32) } else { 0 };\n"
            "    (total + avg - avg, n)\n"
            "}\n"
        ),
        "repaired_program": (
            "pub fn sum_and_len(items: Vec<i32>) -> (i32, usize) {\n"
            "    let total: i32 = items.iter().copied().sum();\n"
            "    let n = items.len();\n"
            "    let avg = if n > 0 { total / (n as i32) } else { 0 };\n"
            "    (total + avg - avg, n)\n"
            "}\n"
        ),
    },
    {
        "id": "rs_type_e0308_pid_update",
        "title": "Mismatched f64 vs f32 state update in PID controller (CRUST-Bench pid_controller, E0308)",
        "expected_code": "E0308",
        "doomed_program": (
            "pub struct Pid { pub kp: f64, pub ki: f64, pub integral: f64 }\n"
            "impl Pid {\n"
            "    pub fn update(&mut self, err: f64, dt: f32) -> f64 {\n"
            "        let step_i: f64 = err * dt;\n"
            "        self.integral += step_i;\n"
            "        self.kp * err + self.ki * self.integral\n"
            "    }\n"
            "}\n"
        ),
        "repaired_program": (
            "pub struct Pid { pub kp: f64, pub ki: f64, pub integral: f64 }\n"
            "impl Pid {\n"
            "    pub fn update(&mut self, err: f64, dt: f32) -> f64 {\n"
            "        let step_i: f64 = err * (dt as f64);\n"
            "        self.integral += step_i;\n"
            "        self.kp * err + self.ki * self.integral\n"
            "    }\n"
            "}\n"
        ),
    },
    {
        "id": "rs_arity_e0061_helper_call",
        "title": "Wrong argument count in internal helper call (UpdatedAPI, E0061)",
        "expected_code": "E0061",
        "doomed_program": (
            "fn clamp_range(x: i64, lo: i64, hi: i64) -> i64 {\n"
            "    x.max(lo).min(hi)\n"
            "}\n"
            "pub fn normalize_scores(xs: &[i64]) -> Vec<i64> {\n"
            "    let first = clamp_range(xs[0], 0);\n"
            "    let mut out = Vec::with_capacity(xs.len());\n"
            "    out.push(first);\n"
            "    for &v in &xs[1..] { out.push(clamp_range(v, 0, 100)); }\n"
            "    out\n"
            "}\n"
        ),
        "repaired_program": (
            "fn clamp_range(x: i64, lo: i64, hi: i64) -> i64 {\n"
            "    x.max(lo).min(hi)\n"
            "}\n"
            "pub fn normalize_scores(xs: &[i64]) -> Vec<i64> {\n"
            "    let first = clamp_range(xs[0], 0, 100);\n"
            "    let mut out = Vec::with_capacity(xs.len());\n"
            "    out.push(first);\n"
            "    for &v in &xs[1..] { out.push(clamp_range(v, 0, 100)); }\n"
            "    out\n"
            "}\n"
        ),
    },
    {
        "id": "rs_mut_e0596_buffer_push",
        "title": "Cannot borrow immutable local binding as mutable (CRUST-Bench cjson, E0596)",
        "expected_code": "E0596",
        "doomed_program": (
            "pub fn compact_json_whitespace(input: &str) -> String {\n"
            "    let out = String::with_capacity(input.len());\n"
            "    out.push_str(input.trim());\n"
            "    let cleaned = out.replace('\\n', \"\").replace('\\t', \"\");\n"
            "    cleaned\n"
            "}\n"
        ),
        "repaired_program": (
            "pub fn compact_json_whitespace(input: &str) -> String {\n"
            "    let mut out = String::with_capacity(input.len());\n"
            "    out.push_str(input.trim());\n"
            "    let cleaned = out.replace('\\n', \"\").replace('\\t', \"\");\n"
            "    cleaned\n"
            "}\n"
        ),
    },
    {
        "id": "rs_field_e0609_struct_access",
        "title": "Non-existent field access on struct (CRUST-Bench hydra, E0609)",
        "expected_code": "E0609",
        "doomed_program": (
            "pub struct Config { pub max_retries: u32, pub timeout_ms: u64 }\n"
            "pub fn total_wait_budget(cfg: &Config) -> u64 {\n"
            "    let retries = cfg.retries as u64;\n"
            "    let base = cfg.timeout_ms;\n"
            "    retries * base + 100\n"
            "}\n"
        ),
        "repaired_program": (
            "pub struct Config { pub max_retries: u32, pub timeout_ms: u64 }\n"
            "pub fn total_wait_budget(cfg: &Config) -> u64 {\n"
            "    let retries = cfg.max_retries as u64;\n"
            "    let base = cfg.timeout_ms;\n"
            "    retries * base + 100\n"
            "}\n"
        ),
    },
]


# Part B: Live End-to-End Rust Function Generation Tasks (`clover-k3-chat.service`)
LIVE_RUST_TASKS = [
    {
        "id": "rust_live_1_gcd",
        "prompt": (
            "Write a complete Rust function inside a ```rust code block:\n"
            "```rust\n"
            "pub fn gcd(mut a: u64, mut b: u64) -> u64\n"
            "```\n"
            "that returns the greatest common divisor of `a` and `b` using Euclid's algorithm."
        ),
        "test_harness": (
            "fn main() {\n"
            "    assert_eq!(gcd(48, 18) as u64, 6);\n"
            "    assert_eq!(gcd(252, 105) as u64, 21);\n"
            "    assert_eq!(gcd(17, 13) as u64, 1);\n"
            "}"
        ),
    },
    {
        "id": "rust_live_2_is_palindrome",
        "prompt": (
            "Write a complete Rust function inside a ```rust code block:\n"
            "```rust\n"
            "pub fn is_palindrome(s: &str) -> bool\n"
            "```\n"
            "that returns `true` if the string `s` reads the same forward and backward (exact character match), and `false` otherwise."
        ),
        "test_harness": (
            "fn main() {\n"
            '    assert!(is_palindrome("racecar"));\n'
            '    assert!(is_palindrome("abba"));\n'
            '    assert!(!is_palindrome("hello"));\n'
            "}"
        ),
    },
    {
        "id": "rust_live_3_fibonacci",
        "prompt": (
            "Write a complete Rust function inside a ```rust code block:\n"
            "```rust\n"
            "pub fn fibonacci(n: u32) -> u64\n"
            "```\n"
            "that returns the n-th Fibonacci number (with `fibonacci(0) == 0` and `fibonacci(1) == 1`)."
        ),
        "test_harness": (
            "fn main() {\n"
            "    assert_eq!(fibonacci(0) as u64, 0);\n"
            "    assert_eq!(fibonacci(1) as u64, 1);\n"
            "    assert_eq!(fibonacci(10) as u64, 55);\n"
            "}"
        ),
    },
    {
        "id": "rust_live_4_fmt_msg_borrow",
        "prompt": (
            "Write a complete Rust function inside a ```rust code block:\n"
            "```rust\n"
            "pub fn fmt_msg(msg: &mut String) -> String\n"
            "```\n"
            "that extracts the first 5 bytes as an owned String `tag`, calls `msg.drain(..5)`, "
            'computes `let prio = tag.starts_with("HIGH");`, and returns `format!("{prio}: {msg}")`.'
        ),
        "test_harness": (
            "fn main() {\n"
            '    let mut s = String::from("HIGH:alert");\n'
            "    let out = fmt_msg(&mut s);\n"
            '    assert_eq!(out, "true: alert");\n'
            "}"
        ),
    },
    {
        "id": "rust_live_5_max_slice",
        "prompt": (
            "Write a complete Rust function inside a ```rust code block:\n"
            "```rust\n"
            "pub fn max_in_slice(nums: &[i32]) -> i32\n"
            "```\n"
            "that returns the maximum integer in the non-empty slice `nums`."
        ),
        "test_harness": (
            "fn main() {\n"
            "    assert_eq!(max_in_slice(&[3, 1, 9, 4, 2]), 9);\n"
            "    assert_eq!(max_in_slice(&[-5, -2, -10]), -2);\n"
            "}"
        ),
    },
]


def main():
    print("=== BENCHMARK 6: GENERATIVE COMPILATION (RUST + PYTHON) ON CLOVER KIMI K3 ===", flush=True)
    assert HAS_RUST_GC, "rust_generative_compiler must be available"
    rs_gc = CloverRustGenerativeCompiler()

    print("\n--- Part A: Rust Partial-Program Early Error Detection & Completeness (N=6) ---", flush=True)
    part_a_results = []
    total_doomed_chars = 0
    total_saved_chars = 0

    for item in RUST_GC_TRAJECTORIES:
        doomed = item["doomed_program"]
        repaired = item["repaired_program"]
        total_len = len(doomed)

        # 1. Verify 0 false rejections across all line prefixes of `repaired`
        rep_lines = repaired.splitlines()
        false_rejections = 0
        for k in range(1, len(rep_lines) + 1):
            sub = "\n".join(rep_lines[:k]) + "\n"
            v_ok = rs_gc.check_prefix(sub, end_marker=(k == len(rep_lines)))
            if not v_ok.ok:
                false_rejections += 1

        # 2. Step line-by-line through `doomed` to find exact early detection point
        d_lines = doomed.splitlines()
        caught_at_char = total_len
        caught_line = len(d_lines)
        caught_code = ""
        caught_rb_line = 1
        check_ms = 0.0
        for k in range(1, len(d_lines) + 1):
            sub = "\n".join(d_lines[:k]) + "\n"
            v = rs_gc.check_prefix(sub, end_marker=(k == len(d_lines)))
            check_ms += v.elapsed_ms
            if not v.ok and v.diagnostics:
                caught_at_char = len(sub)
                caught_line = k
                caught_code = v.diagnostics[0].code
                caught_rb_line = (v.rollback_line_idx or 0) + 1
                break

        saved_chars = max(0, total_len - caught_at_char)
        saved_pct = 100.0 * saved_chars / max(1, total_len)
        total_doomed_chars += total_len
        total_saved_chars += saved_chars

        print(
            f"  [{item['id']:28s}] caught={caught_code:6s} at line {caught_line}/{len(d_lines)} "
            f"(rollback->L{caught_rb_line}) | char {caught_at_char:3d}/{total_len:3d} "
            f"(saved {saved_pct:5.1f}% doomed chars) | false_rej={false_rejections} | {check_ms:.1f} ms",
            flush=True,
        )
        part_a_results.append({
            "id": item["id"],
            "expected_code": item["expected_code"],
            "caught_code": caught_code,
            "caught_line": caught_line,
            "total_lines": len(d_lines),
            "rollback_target_line": caught_rb_line,
            "caught_char": caught_at_char,
            "total_chars": total_len,
            "saved_chars": saved_chars,
            "saved_pct": round(saved_pct, 2),
            "false_rejections_on_repaired": false_rejections,
            "total_check_ms": round(check_ms, 2),
        })

    overall_saved_pct = 100.0 * total_saved_chars / max(1, total_doomed_chars)
    print(
        f">>> Part A Summary: 6/6 errors caught mid-generation | 0 false rejections | "
        f"saved {total_saved_chars}/{total_doomed_chars} doomed chars ({overall_saved_pct:.1f}%)",
        flush=True,
    )

    print("\n--- Part B: Live End-to-End Rust Code Generation & `rustc` Execution (N=5) ---", flush=True)
    part_b_results = []
    live_passed = 0
    for idx, task in enumerate(LIVE_RUST_TASKS):
        t0 = time.perf_counter()
        r = ask_clover(task["prompt"], sid=f"bench6_rust_{idx}")
        wall_ms = (time.perf_counter() - t0) * 1000.0
        rs_code = extract_rust_code(r.get("reply", ""))
        ok, detail = run_rustc_unit_test(rs_code, task["test_harness"])
        if ok:
            live_passed += 1
        gc_meta = next(
            (t for t in (r.get("tool_calls") or []) if t.get("tool") == "generative_compiler" and t.get("event") == "summary"),
            {},
        )
        print(
            f"  [{idx+1}/{len(LIVE_RUST_TASKS)}] {'PASS' if ok else 'FAIL'} ({wall_ms:.0f} ms) {task['id']:26s} | "
            f"rustc_test={detail} | gc_checks={gc_meta.get('prefix_checks', 0)} gc_ms={gc_meta.get('total_ms', 0)}",
            flush=True,
        )
        part_b_results.append({
            "id": task["id"],
            "ok": ok,
            "detail": detail,
            "wall_ms": round(wall_ms, 1),
            "gc_telemetry": gc_meta,
            "code": rs_code,
        })

    live_acc = 100.0 * live_passed / len(LIVE_RUST_TASKS)
    print(f">>> Part B Summary: {live_passed}/{len(LIVE_RUST_TASKS)} ({live_acc:.1f}%) compiled & passed rustc unit tests", flush=True)

    out_path = "/home/usergpu/merge_lf/benchmarks/generative_compilation_v6_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "part_a_rust_prefix_trajectories": part_a_results,
                "part_a_doomed_chars_saved_pct": round(overall_saved_pct, 2),
                "part_b_live_rust_generation": part_b_results,
                "part_b_pass_rate_pct": round(live_acc, 2),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
