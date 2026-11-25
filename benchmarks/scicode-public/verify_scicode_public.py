#!/usr/bin/env python3
"""Standalone, reproducible public evaluator for Clover's SciCode (`AA-SciCode` 288-subproblem) results.

Anyone can reproduce and verify every score independently in 3 commands:
  1. Clone official `scicode-bench/SciCode` & download `problems_test.jsonl` + `test_data.h5`
  2. Verify SHA-256 hashes of the official benchmark files (`test_data.h5` & `problems_test.jsonl`)
  3. Run:
       python3 benchmarks/scicode-public/verify_scicode_public.py \
         --code-dir benchmarks/scicode-public/submissions/clover_system_kimi_k31/with_background \
         --problems /path/to/problems_test.jsonl \
         --h5 /path/to/test_data.h5
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OFFICIAL_PROBLEMS_TEST_SHA256 = "38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d"
OFFICIAL_TEST_DATA_H5_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
STARTER_STEPS = {("13", 6), ("62", 1), ("76", 3)}  # Official SciCode v1.0.1 provided starter steps (291 - 3 = 288)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _grade_one_step(args: tuple[str, str, str, list[str], str, str, int]) -> dict:
    prob_id, step_num_idx, step_id, test_cases, code_text, h5_path, timeout_s = args
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"scicode_eval_{step_id}_") as td:
        script = Path(td) / f"{step_id}.py"
        body = (
            "import scipy.integrate\n"
            "if not hasattr(scipy.integrate, 'simps'):\n"
            "    scipy.integrate.simps = scipy.integrate.simpson\n"
            + code_text
            + "\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n"
            + f"targets = process_hdf5_to_tuple({step_id!r}, {len(test_cases)}, {h5_path!r})\n"
        )
        for idx, tc in enumerate(test_cases):
            body += f"target = targets[{idx}]\n{tc}\n"
        script.write_text(body, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=td,
            )
            passed = proc.returncode == 0
            err = "" if passed else (proc.stderr or proc.stdout or "")[-600:]
            status = "pass" if passed else "fail"
        except subprocess.TimeoutExpired:
            passed = False
            err = f"TimeoutExpired ({timeout_s}s)"
            status = "timeout"
    return {
        "problem_id": prob_id,
        "step_idx": int(step_num_idx),
        "step_id": step_id,
        "passed": passed,
        "status": status,
        "duration_s": round(time.perf_counter() - t0, 3),
        "error_tail": err,
    }


def evaluate_submission(
    code_dir: Path,
    problems_jsonl: Path,
    h5_file: Path,
    workers: int = 16,
    timeout_s: int = 300,
    verify_hashes: bool = True,
) -> dict:
    p_sha = sha256_file(problems_jsonl)
    h5_sha = sha256_file(h5_file)
    if verify_hashes:
        if p_sha != OFFICIAL_PROBLEMS_TEST_SHA256:
            raise RuntimeError(f"problems_test.jsonl SHA-256 mismatch: {p_sha}")
        if h5_sha != OFFICIAL_TEST_DATA_H5_SHA256:
            raise RuntimeError(f"test_data.h5 SHA-256 mismatch: {h5_sha}")

    problems = [json.loads(line) for line in problems_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = []
    missing_files = []
    prob_step_counts: dict[str, int] = {}

    for prob in problems:
        pid = str(prob["problem_id"])
        graded_in_prob = 0
        for idx, sub in enumerate(prob["sub_steps"], start=1):
            if (pid, idx) in STARTER_STEPS:
                continue
            graded_in_prob += 1
            step_id = str(sub["step_number"])
            py_file = code_dir / f"{pid}.{idx}.py"
            if not py_file.is_file():
                missing_files.append(f"{pid}.{idx}.py")
                code_text = "raise RuntimeError('missing submission file')\n"
            else:
                code_text = py_file.read_text(encoding="utf-8")
            tasks.append((pid, str(idx), step_id, list(sub["test_cases"]), code_text, str(h5_file), timeout_s))
        prob_step_counts[pid] = graded_in_prob

    t_start = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_grade_one_step, tasks):
            results.append(res)

    results.sort(key=lambda r: (int(r["problem_id"]), r["step_idx"]))
    passed_steps = sum(1 for r in results if r["passed"])
    total_steps = len(results)

    by_prob: dict[str, int] = {}
    for r in results:
        if r["passed"]:
            by_prob[r["problem_id"]] = by_prob.get(r["problem_id"], 0) + 1
    solved_main = sorted(
        [int(pid) for pid, cnt in prob_step_counts.items() if cnt > 0 and by_prob.get(pid, 0) == cnt]
    )

    return {
        "benchmark": "SciCode (AA-SciCode test split, with_background=True, v1.0.1)",
        "code_dir": str(code_dir),
        "problems_test_sha256": p_sha,
        "test_data_h5_sha256": h5_sha,
        "subproblems_passed": passed_steps,
        "subproblems_total": total_steps,
        "subproblem_pass_pct": round(100.0 * passed_steps / max(1, total_steps), 2),
        "main_problems_solved": len(solved_main),
        "main_problems_total": len(problems),
        "main_problem_pass_pct": round(100.0 * len(solved_main) / max(1, len(problems)), 2),
        "solved_main_problem_ids": solved_main,
        "missing_files": missing_files,
        "eval_wall_time_s": round(time.perf_counter() - t_start, 2),
        "per_step_results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code-dir", type=Path, required=True)
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-hash-check", action="store_true")
    args = ap.parse_args()

    rep = evaluate_submission(
        code_dir=args.code_dir,
        problems_jsonl=args.problems,
        h5_file=args.h5,
        workers=args.workers,
        timeout_s=args.timeout,
        verify_hashes=not args.no_hash_check,
    )
    print(
        f"SCICODE_VERIFY: subproblems={rep['subproblems_passed']}/{rep['subproblems_total']} "
        f"({rep['subproblem_pass_pct']:.2f}%) | main_problems={rep['main_problems_solved']}/{rep['main_problems_total']} "
        f"({rep['main_problem_pass_pct']:.2f}%) | wall={rep['eval_wall_time_s']}s"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
