#!/usr/bin/env python3
"""
Stratified 4-Tier Lean 4 `RiemannHypothesis` + Verifier SWE Benchmark Extractor
(`benchmarks/riemann-lean4/extract_rh_tasks.py`).

Extracts reproducible, anti-leakage-audited benchmark tasks directly from Levent Alpöge's
175,187-file Lean 4 `RiemannHypothesis` archive (`/mnt/nvme1/clover-rh-bench/rh-src/lean/src`):

  - **Tier 1 (`tier1_spec_roundtrip`):** Cryptic machine-generated Lean theorem signature ->
    natural-language spec -> regenerated Lean proposition checked via kernel `Iff` typecheck.
  - **Tier 2 (`tier2_leaf_tactic_repair`):** Standalone `import Mathlib`-only leaf modules from
    `Lemmata/` (binned into `short_1_6`, `medium_7_14`, and `hard_15_28` proof lines) where:
      * Every sibling theorem/lemma AFTER the target in the file is stripped (`0` sibling leakage),
      * Only the target's `:= by ...` body is replaced, locked by `prefix_sha256`,
      * Ground-truth proof is optionally verified to compile in `< 15s` under `lean_oracle`.
  - **Tier 3 (`tier3_egm_subgoal_audit`):** Paired *valid* vs *subtly falsified* theorem/subgoal
    decompositions derived from real RH `Formalism/` & `Lemmata/` statements (e.g., dropping
    `s ≠ 1` pole exclusion, `χ ≠ 1` non-principal condition, or `0 < σ` positivity bound) to
    measure whether `Elephant + Goldfish` catches unprovable subgoals *before* proving.
  - **Tier 4 (`tier4_verifier_swe`):** Companion verifier systems-engineering tasks (Python/C)
    with deterministic unit-test assertions (topological `.lean` DAG cycle/ready-queue scheduler,
    nested Lean comment-stripping static gate, exact prime/character arithmetic).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)", re.MULTILINE)
_DECL_START_RE = re.compile(r"^((?:private\s+|protected\s+)?(?:theorem|lemma)\s+([A-Za-z0-9_']+))", re.MULTILINE)


def parse_standalone_mathlib_theorems(lean_file: Path) -> list[dict[str, Any]]:
    """Return theorems/lemmas in files whose ONLY imports are `Mathlib` / `Mathlib.*`."""
    try:
        src = lean_file.read_text(encoding="utf-8")
    except OSError:
        return []
    imports = _IMPORT_RE.findall(src)
    if not imports or any(not (imp == "Mathlib" or imp.startswith("Mathlib.")) for imp in imports):
        return []
    # Skip files using custom elaborators / macros (`kexactB`, `kcopy`) so standard Mathlib tactics apply
    if "kexactB" in src or "kcopy" in src:
        return []

    matches = list(_DECL_START_RE.finditer(src))
    if not matches:
        return []

    # Find trailing `end <Namespace>` lines if a `namespace <N>` opened before the theorem
    ns_open = re.findall(r"^\s*namespace\s+([A-Za-z0-9_.]+)", src[: matches[0].start()], flags=re.MULTILINE)
    closing_suffix = "\n".join(f"end {ns}" for ns in reversed(ns_open))

    results: list[dict[str, Any]] = []
    for idx, m in enumerate(matches):
        # Only take the FIRST theorem in a file so zero later helper lemmas are needed, and zero sibling proofs leak
        if idx > 0:
            break
        decl_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(src)
        chunk = src[m.start() : decl_end]
        by_pos = chunk.find(":= by\n")
        if by_pos == -1:
            continue
        prefix = src[: m.start()] + chunk[: by_pos + len(":= by\n")]
        raw_proof = chunk[by_pos + len(":= by\n") :]
        # Strip trailing `end Namespace` from raw_proof
        proof_lines: list[str] = []
        for ln in raw_proof.splitlines():
            if re.match(r"^\s*end\s+[A-Za-z0-9_.]+\s*$", ln):
                break
            proof_lines.append(ln)
        while proof_lines and not proof_lines[-1].strip():
            proof_lines.pop()
        non_empty = [ln for ln in proof_lines if ln.strip() and not ln.strip().startswith("--")]
        if not (1 <= len(non_empty) <= 28):
            continue

        n_lines = len(non_empty)
        if n_lines <= 6:
            length_bin = "short_1_6"
        elif n_lines <= 14:
            length_bin = "medium_7_14"
        else:
            length_bin = "hard_15_28"

        results.append({
            "thm_name": m.group(2),
            "immutable_prefix": prefix,
            "immutable_suffix": closing_suffix,
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "ground_truth_tactics": "\n".join(proof_lines),
            "proof_line_count": n_lines,
            "length_bin": length_bin,
        })
    return results


def build_tier3_egm_audit_tasks() -> list[dict[str, Any]]:
    """Paired valid vs subtly-unprovable subgoal decompositions from RH `Statements.lean` & `ZetaRealNonvanish.lean`."""
    return [
        {
            "task_id": "rh_t3_egm_01_valid_dirichlet_nonvanish",
            "tier": "tier3_egm_subgoal_audit",
            "title": "Audit valid Dirichlet L-function non-vanishing subgoal on the critical half-plane (`Statements.lean`)",
            "proposed_lean_subgoal": (
                "∀ (N : ℕ) [NeZero N] (χ : DirichletCharacter ℂ N) (s : ℂ),\n"
                "  (χ ≠ 1 ∨ s ≠ 1) → 1 ≤ s.re → DirichletCharacter.LFunction χ s ≠ 0"
            ),
            "is_provable": True,
            "ground_truth_flaw": "",
        },
        {
            "task_id": "rh_t3_egm_02_trivial_zeros_missing",
            "tier": "tier3_egm_subgoal_audit",
            "title": "Audit subgoal signature for RiemannZetaZerosOnLine reduction",
            "proposed_lean_subgoal": (
                "∀ s : ℂ, riemannZeta s = 0 → s ≠ 1 → s.re = 1 / 2"
            ),
            "is_provable": False,
            "ground_truth_flaw": (
                "False on trivial zeros `s = -2, -4, -6, ...`! Missing `(¬∃ n : ℕ, s = -2 * ((n : ℂ) + 1))` "
                "(or `0 < s.re`), since `riemannZeta (-2) = 0` has `(-2 : ℂ).re = -2 ≠ 1 / 2`."
            ),
        },
        {
            "task_id": "rh_t3_egm_03_valid_landau_nonneg",
            "tier": "tier3_egm_subgoal_audit",
            "title": "Audit Landau Mellin abscissa continuation subgoal (`LandauLemma`)",
            "proposed_lean_subgoal": (
                "∀ (H : ℝ → ℝ) (a b : ℝ), a < b → AEStronglyMeasurable H (volume.restrict (Ioi 1)) →\n"
                "  (∀ x : ℝ, 1 < x → 0 ≤ H x) →\n"
                "  (∀ σ : ℝ, b < σ → IntegrableOn (fun x : ℝ => H x * x ^ (-σ - 1)) (Ioi 1)) →\n"
                "  ∀ Φ : ℂ → ℂ, (∀ s : ℂ, b < s.re → AnalyticAt ℂ Φ s) →\n"
                "  (∀ σ : ℝ, a < σ → σ ≤ b → AnalyticAt ℂ Φ σ) →\n"
                "  (∀ s : ℂ, b < s.re → Φ s = ∫ x in Ioi (1:ℝ), (H x : ℂ) * (x : ℂ) ^ (-s - 1)) →\n"
                "  ∀ σ : ℝ, a < σ → IntegrableOn (fun x : ℝ => H x * x ^ (-σ - 1)) (Ioi 1)"
            ),
            "is_provable": True,
            "ground_truth_flaw": "",
        },
        {
            "task_id": "rh_t3_egm_04_landau_missing_nonneg",
            "tier": "tier3_egm_subgoal_audit",
            "title": "Audit Landau Mellin continuation subgoal WITHOUT non-negativity of H(x)",
            "proposed_lean_subgoal": (
                "∀ (H : ℝ → ℝ) (a b : ℝ), a < b → AEStronglyMeasurable H (volume.restrict (Ioi 1)) →\n"
                "  (∀ σ : ℝ, b < σ → IntegrableOn (fun x : ℝ => H x * x ^ (-σ - 1)) (Ioi 1)) →\n"
                "  ∀ Φ : ℂ → ℂ, (∀ s : ℂ, a < s.re → AnalyticAt ℂ Φ s) →\n"
                "  (∀ s : ℂ, b < s.re → Φ s = ∫ x in Ioi (1:ℝ), (H x : ℂ) * (x : ℂ) ^ (-s - 1)) →\n"
                "  ∀ σ : ℝ, a < σ → IntegrableOn (fun x : ℝ => H x * x ^ (-σ - 1)) (Ioi 1)"
            ),
            "is_provable": False,
            "ground_truth_flaw": (
                "Unprovable / false! Dropping `∀ x > 1, 0 ≤ H x` breaks Landau's theorem: oscillating functions like "
                "`H(x) = sin(x)` have analytic Mellin continuations past their abscissa of absolute convergence without being Lebesgue `IntegrableOn`!"
            ),
        },
    ]


def build_tier4_swe_tasks() -> list[dict[str, Any]]:
    """Companion verifier systems-engineering tasks with exact executable assertions."""
    return [
        {
            "task_id": "rh_t4_swe_01_dag_ready_scheduler",
            "tier": "tier4_verifier_swe",
            "title": "Implement lock-free topological Lean module batch scheduler (`ready_batches`)",
            "prompt": (
                "Write a self-contained Python function `ready_batches(deps: dict[str, set[str]]) -> list[list[str]]` "
                "that takes a mapping from Lean module name to the set of internal module names it imports (all keys in `deps` "
                "or external Mathlib modules not in `deps`, which should be ignored as already built), raises `ValueError` "
                "if any import cycle exists among modules in `deps`, and otherwise returns deterministic lexicographically "
                "sorted topological waves where wave `k` contains every remaining module whose internal dependencies appeared "
                "in waves `< k`. Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert ready_batches({'Main': {'Statements', 'Mathlib'}, 'Statements': {'Mathlib'}}) == [['Statements'], ['Main']]\n"
                "assert ready_batches({'C': {'A', 'B'}, 'B': {'A'}, 'A': set(), 'D': {'A'}}) == [['A'], ['B', 'D'], ['C']]\n"
                "try:\n"
                "    ready_batches({'A': {'B'}, 'B': {'A'}})\n"
                "    raise AssertionError('Expected ValueError on cycle')\n"
                "except ValueError:\n"
                "    pass\n"
            ),
        },
        {
            "task_id": "rh_t4_swe_02_nested_lean_comment_scanner",
            "tier": "tier4_verifier_swe",
            "title": "Scan Lean 4 source for forbidden tokens ignoring nested `/- ... -/` comments & strings",
            "prompt": (
                "Write a Python function `scan_lean_ax(src: str) -> list[str]` that strips Lean 4 line comments (`-- ...` to EOL), "
                "ARBITRARILY NESTED block comments (`/- ... /- nested -/ ... -/`), character literals (`'\"'`, `'\\''`, `'a'`), "
                "and string literals (`\"...\"` with `\\\"` escapes), and returns a sorted list of unique forbidden whole-word "
                "identifiers from `{'sorry', 'admit', 'axiom', 'unsafe', 'skipKernelTC'}` present in the remaining code. "
                "Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert scan_lean_ax('-- sorry\\n/- outer /- sorry -/ axiom -/\\ndef s := \"admit\"\\ntheorem t : True := trivial') == []\n"
                "assert scan_lean_ax(\"have c : Char := '\\\"'\\ntheorem bad : False := by sorry\") == ['sorry']\n"
                "assert scan_lean_ax('/- a /- b -/ -/ axiom bad : False') == ['axiom']\n"
            ),
        },
        {
            "task_id": "rh_t4_swe_03_lcp_prefix_cache_matcher",
            "tier": "tier4_verifier_swe",
            "title": "Implement Longest-Common-Prefix cache planner for truncatable KV vs recurrent SSM engines",
            "prompt": (
                "Write a Python function `plan_prefill(cached_ids: list[int], new_ids: list[int], supports_truncate: bool) -> tuple[str, int, list[int]]` "
                "for an LLM engine where at least 1 token must be fed to produce fresh logits unless `cached_ids == new_ids` (in which case return "
                "`('reuse_exact', len(new_ids), [])`). Otherwise let `lcp` be the longest common prefix of `cached_ids` and `new_ids` capped at `len(new_ids) - 1`. "
                "If `supports_truncate` is True: if `lcp < len(cached_ids)` return `('truncate_and_feed', lcp, new_ids[lcp:])`, else return `('extend', lcp, new_ids[lcp:])`. "
                "If `supports_truncate` is False (recurrent Mamba/BD-GDN): if `cached_ids` is non-empty and `lcp == len(cached_ids)` return `('extend', lcp, new_ids[lcp:])`, "
                "otherwise return `('reset_and_feed', 0, list(new_ids))`. Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert plan_prefill([1, 2, 3], [1, 2, 3], False) == ('reuse_exact', 3, [])\n"
                "assert plan_prefill([1, 2, 3], [1, 2, 3, 4, 5], False) == ('extend', 3, [4, 5])\n"
                "assert plan_prefill([1, 2, 9], [1, 2, 3, 4], False) == ('reset_and_feed', 0, [1, 2, 3, 4])\n"
                "assert plan_prefill([1, 2, 9], [1, 2, 3, 4], True) == ('truncate_and_feed', 2, [3, 4])\n"
                "assert plan_prefill([1, 2, 3], [1, 2], True) == ('truncate_and_feed', 1, [2])\n"
            ),
        },
        {
            "task_id": "rh_t4_swe_04_mxfp4_e2m1_block_decoder",
            "tier": "tier4_verifier_swe",
            "title": "Implement exact OCP MXFP4 (`E2M1` + `E8M0` scale) 32-element block decoder",
            "prompt": (
                "Write a Python function `decode_mxfp4_block(packed16: bytes, scale_e8m0: int) -> list[float]` that decodes 16 packed bytes "
                "(32 4-bit nibbles: low nibble `b & 0xF` first, then high nibble `b >> 4` for each byte `b` in `packed16`) scaled by an OCP `E8M0` "
                "shared exponent `scale_e8m0` (`0..254`, where scale factor is `2.0 ** (scale_e8m0 - 127)`, and `scale_e8m0 == 255` raises `ValueError('NaN scale')`). "
                "Each 4-bit `E2M1` nibble `n` has sign bit `(n >> 3) & 1`, 2-bit exponent `e = (n >> 1) & 3`, and 1-bit mantissa `m = n & 1`: "
                "magnitude is `0.5 * m` when `e == 0` (subnormal/zero: `0.0` or `0.5`), and `(1.0 + 0.5 * m) * (2.0 ** (e - 1))` when `e > 0` "
                "(giving magnitudes `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`), negated if sign bit is 1. Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "b = bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] + [0x00]*8)\n"
                "out = decode_mxfp4_block(b, 127)\n"
                "assert out[:16] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]\n"
                "assert decode_mxfp4_block(bytes([0x27] + [0x00]*15), 129)[:2] == [24.0, 4.0]\n"
                "try:\n"
                "    decode_mxfp4_block(bytes(16), 255)\n"
                "    raise AssertionError('Expected ValueError')\n"
                "except ValueError:\n"
                "    pass\n"
            ),
        },
    ]


def build_tier5_re_tasks(re_dir: str = "/mnt/nvme1/clover-rh-bench/re_targets") -> list[dict[str, Any]]:
    """Software & binary reverse-engineering tasks (`tier5_reverse_engineering`) backed by real compiled/minified targets."""
    return [
        {
            "task_id": "rh_t5_re_01_minified_ipc_preload",
            "tier": "tier5_reverse_engineering",
            "target_kind": "javascript_app",
            "target_path": f"{re_dir}/ipc_app",
            "title": "Reverse-engineer obfuscated Electron preload IPC frame signer (`sign_ipc_frame`)",
            "prompt": (
                "Target Electron preload application at `/mnt/nvme1/clover-rh-bench/re_targets/ipc_app` "
                "(raw hex of `preload.min.js`: "
                "`636f6e7374205f4d3d7b636c69703a276e783a63623a7632272c7661756c743a276e783a766c3a7632272c73796e633a276e783a73793a7632277d3b` ... "
                "use reverse-engineering evidence if available). Implement an exact Python replica "
                "`sign_ipc_frame(channel: str, seq: int, payload: str) -> dict` returning `{'ch': mapped_channel, 'tag': hex8}` "
                "(raising `PermissionError` if `channel` is not in the allow-map). Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert sign_ipc_frame('clip', 7, 'hello') == {'ch': 'nx:cb:v2', 'tag': '9b0637d6'}\n"
                "assert sign_ipc_frame('vault', 65538, 'k=1') == sign_ipc_frame('vault', 2, 'k=1')\n"
                "try:\n"
                "    sign_ipc_frame('admin', 1, 'x')\n"
                "    raise AssertionError('Expected PermissionError')\n"
                "except PermissionError:\n"
                "    pass\n"
            ),
        },
        {
            "task_id": "rh_t5_re_02_stripped_elf_mixer",
            "tier": "tier5_reverse_engineering",
            "target_kind": "elf_x86_64",
            "target_path": f"{re_dir}/libmystery_mix.so",
            "title": "Reconstruct 64-bit bijective mixer `clover_mix64(x)` & inverse from stripped ELF binary",
            "prompt": (
                "Target stripped x86-64 shared library `/mnt/nvme1/clover-rh-bench/re_targets/libmystery_mix.so` exports `clover_mix64` "
                "(`.text` machine bytes: `48 89 f8 48 c1 ef 1e 48 31 f8 48 b9 b9 e5 e4 1c 6d 47 58 bf 48 0f af c1 48 89 c2 48 c1 ea 1b "
                "48 31 d0 48 b9 eb 11 31 13 bb 49 d0 94 48 0f af c1 48 89 c2 48 c1 ea 1f 48 31 d0 c3`). "
                "Write `clover_mix64(x: int) -> int` (64-bit unsigned) AND its exact inverse `clover_unmix64(y: int) -> int` such that "
                "`clover_unmix64(clover_mix64(x)) == x` for all 64-bit unsigned `x`. Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert clover_mix64(0) == 0\n"
                "assert clover_mix64(1) == 0x5692161d100b05e5\n"
                "for v in [0, 1, 42, 0xdeadbeefcafebabe, (1<<64)-1]:\n"
                "    y = clover_mix64(v)\n"
                "    assert 0 <= y < (1 << 64)\n"
                "    assert clover_unmix64(y) == v\n"
            ),
        },
        {
            "task_id": "rh_t5_re_03_binary_tile_container",
            "tier": "tier5_reverse_engineering",
            "target_kind": "elf_x86_64",
            "target_path": f"{re_dir}/libtile_unpack.so",
            "title": "Recover packed 4-bit binary container format from stripped `libtile_unpack.so`",
            "prompt": (
                "Target stripped x86-64 library `/mnt/nvme1/clover-rh-bench/re_targets/libtile_unpack.so` implements "
                "`int unpack_rec(const uint8_t *buf, size_t len, float *out)` (inspect binary strings/disassembly evidence if provided; "
                "header starts with 4-byte magic `'CLV4'`). Implement `unpack_rec(buf: bytes) -> list[float]` in Python, raising `ValueError` "
                "on invalid magic, odd count, count > 1024, or truncated buffer. Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "import struct\n"
                "blob = b'CLV4' + struct.pack('<hhh', 4, 128, -256) + bytes([(10 << 4) | 8, (6 << 4) | 15])\n"
                "assert unpack_rec(blob) == [-1.0, 0.0, 2.5, -2.0]\n"
                "try:\n"
                "    unpack_rec(b'BAD!' + bytes(10))\n"
                "    raise AssertionError('Expected ValueError')\n"
                "except ValueError:\n"
                "    pass\n"
            ),
        },
        {
            "task_id": "rh_t5_re_04_protocol_state_machine",
            "tier": "tier5_reverse_engineering",
            "target_kind": "javascript_app",
            "target_path": f"{re_dir}/fsm_app",
            "title": "Reconstruct retry/backoff FSM from obfuscated JS transport bundle",
            "prompt": (
                "Target JS bundle `/mnt/nvme1/clover-rh-bench/re_targets/fsm_app/router.min.js` implements a transport retry state machine "
                "(use reverse-engineering evidence if provided; states `'IDLE'`, `'READY'`, `'RETRY'`, `'HALT'`, events `'ok'`, `'rate'`, `'auth'`, `'timeout'`). "
                "Implement an exact Python replica `step_fsm(events: list[str]) -> tuple[str, int]` returning `(final_state, total_backoff_ms)`. "
                "Return ONLY Python code inside ```python ... ```."
            ),
            "test_code": (
                "assert step_fsm(['timeout', 'ok', 'rate', 'rate', 'ok']) == ('READY', 600)\n"
                "assert step_fsm(['ok', 'timeout', 'timeout', 'timeout', 'ok']) == ('HALT', 1500)\n"
                "assert step_fsm(['rate', 'rate', 'rate', 'rate', 'ok']) == ('HALT', 3000)\n"
            ),
        },
    ]


def extract_benchmark_suite(
    rh_src_dir: Path,
    out_jsonl: Path,
    per_bin: int = 8,
    verify_ground_truth: bool = False,
) -> dict[str, Any]:
    from lean_oracle import verify_lean_module

    lemmata_dir = rh_src_dir / "Lemmata"
    search_root = lemmata_dir if lemmata_dir.is_dir() else rh_src_dir

    bins: dict[str, list[dict[str, Any]]] = {"short_1_6": [], "medium_7_14": [], "hard_15_28": []}
    scanned_files = 0
    cand_files = sorted(search_root.rglob("*.lean"))

    for fp in cand_files:
        if all(len(v) >= per_bin for v in bins.values()):
            break
        scanned_files += 1
        thms = parse_standalone_mathlib_theorems(fp)
        for thm in thms:
            bname = thm["length_bin"]
            if len(bins[bname]) >= per_bin:
                continue
            rel_mod = fp.relative_to(rh_src_dir).as_posix()
            if verify_ground_truth:
                chk = verify_lean_module(
                    thm["immutable_prefix"],
                    thm["ground_truth_tactics"],
                    lean_project_dir=rh_src_dir.parent,
                    immutable_suffix=thm["immutable_suffix"],
                    expected_prefix_sha256=thm["prefix_sha256"],
                    timeout_s=30.0,
                )
                if not chk.passed:
                    continue
                gt_ms = chk.elapsed_ms
            else:
                gt_ms = None

            idx_in_bin = len(bins[bname]) + 1
            bins[bname].append({
                "task_id": f"rh_t2_{bname}_{idx_in_bin:02d}",
                "tier": "tier2_leaf_tactic_repair",
                "length_bin": bname,
                "module_relpath": rel_mod,
                "thm_name": thm["thm_name"],
                "proof_line_count": thm["proof_line_count"],
                "immutable_prefix": thm["immutable_prefix"],
                "immutable_suffix": thm["immutable_suffix"],
                "prefix_sha256": thm["prefix_sha256"],
                "ground_truth_tactics": thm["ground_truth_tactics"],
                "ground_truth_verify_ms": gt_ms,
            })

    all_tasks: list[dict[str, Any]] = []
    # Tier 1 tasks derived from the first 2 standalone theorem statements
    t2_flat = bins["short_1_6"] + bins["medium_7_14"] + bins["hard_15_28"]
    for i, t2 in enumerate(t2_flat[:2], start=1):
        all_tasks.append({
            "task_id": f"rh_t1_spec_{i:02d}",
            "tier": "tier1_spec_roundtrip",
            "module_relpath": t2["module_relpath"],
            "thm_name": t2["thm_name"],
            "lean_statement_prefix": t2["immutable_prefix"],
        })
    all_tasks.extend(t2_flat)
    all_tasks.extend(build_tier3_egm_audit_tasks())
    all_tasks.extend(build_tier4_swe_tasks())
    all_tasks.extend(build_tier5_re_tasks())

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    suite_sha = hashlib.sha256(out_jsonl.read_bytes()).hexdigest()
    return {
        "scanned_lean_files": scanned_files,
        "total_tasks": len(all_tasks),
        "tier1_count": sum(1 for x in all_tasks if x["tier"] == "tier1_spec_roundtrip"),
        "tier2_short": len(bins["short_1_6"]),
        "tier2_medium": len(bins["medium_7_14"]),
        "tier2_hard": len(bins["hard_15_28"]),
        "tier3_egm_count": len(build_tier3_egm_audit_tasks()),
        "tier4_swe_count": len(build_tier4_swe_tasks()),
        "tier5_re_count": len(build_tier5_re_tasks()),
        "suite_sha256": suite_sha,
        "output_jsonl": str(out_jsonl),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rh-src", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh-src/lean/src"))
    ap.add_argument("--out", type=Path, default=Path("/mnt/nvme1/clover-rh-bench/rh_bench_tasks.jsonl"))
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--verify-gt", action="store_true")
    args = ap.parse_args()
    rep = extract_benchmark_suite(args.rh_src, args.out, per_bin=args.per_bin, verify_ground_truth=args.verify_gt)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
