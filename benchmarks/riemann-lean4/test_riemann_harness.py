#!/usr/bin/env python3
"""Unit tests for `benchmarks/riemann-lean4/` (anti-leakage, header-tamper lock, nested comment stripping, MCP tools)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from extract_rh_tasks import build_tier3_egm_audit_tasks, build_tier4_swe_tasks, parse_standalone_mathlib_theorems
from lean_oracle import (
    assemble_candidate_lean,
    extract_clean_tactic_block,
    find_forbidden_constructs,
    verify_printed_axioms,
)
from mcp_clover_tools import AntiLeakagePremiseIndex, eval_exact_math
from run_ablation import run_python_sandbox


class TestRiemannLeanHarness(unittest.TestCase):
    def test_nested_comment_and_string_forbidden_scanner(self) -> None:
        safe = "-- sorry\n/- outer /- nested sorry & axiom -/ still comment -/\ndef msg := \"admit skipKernelTC\"\ntheorem ok : True := trivial"
        self.assertEqual(find_forbidden_constructs(safe), ())
        self.assertEqual(find_forbidden_constructs("theorem bad : False := by sorry"), ("sorry",))
        self.assertEqual(find_forbidden_constructs("/- closed -/ axiom cheat : False"), ("axiom",))

    def test_immutable_header_lock_blocks_tampering_and_sorry(self) -> None:
        prefix = "import Mathlib\n\ntheorem t : 1 + 1 = 2 := by\n"
        sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()

        # Tampered prefix rejected
        src_bad, err_bad = assemble_candidate_lean("import Mathlib\ntheorem t : True := by\n", "trivial", expected_prefix_sha256=sha)
        self.assertIsNone(src_bad)
        assert err_bad is not None
        self.assertEqual(err_bad.stage, "header_tamper")

        # Forbidden `sorry` rejected before touching Lean process
        src_s, err_s = assemble_candidate_lean(prefix, "```lean\nsorry\n```", expected_prefix_sha256=sha)
        self.assertIsNone(src_s)
        assert err_s is not None
        self.assertEqual(err_s.stage, "static_gate")
        self.assertIn("sorry", err_s.forbidden_hits)

        # Valid tactic block assembled cleanly (even when model echoes header up to `:= by`)
        src_ok, err_ok = assemble_candidate_lean(prefix, "```lean\ntheorem t : 1 + 1 = 2 := by\n  rfl\n```", expected_prefix_sha256=sha)
        self.assertIsNone(err_ok)
        assert src_ok is not None
        self.assertTrue(src_ok.startswith(prefix))
        self.assertIn("  rfl", src_ok)

    def test_anti_leakage_sibling_stripping_and_premise_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src" / "Formalism").mkdir(parents=True)
            f1 = root / "src" / "Formalism" / "TargetMod.lean"
            f1.write_text(
                "import Mathlib\nnamespace Foo\ntheorem target_lem (n : Nat) : n = n := by\n  rfl\n"
                "theorem secret_sibling (n : Nat) : n + 0 = n := by\n  simp\nend Foo\n",
                encoding="utf-8",
            )
            f2 = root / "src" / "Formalism" / "AllowedMod.lean"
            f2.write_text(
                "import Mathlib\ntheorem allowed_helper (m : Nat) : m + 0 = m := by\n  omega\n",
                encoding="utf-8",
            )
            parsed = parse_standalone_mathlib_theorems(f1)
            self.assertEqual(len(parsed), 1)
            self.assertNotIn("secret_sibling", parsed[0]["immutable_prefix"])
            self.assertNotIn("secret_sibling", parsed[0]["ground_truth_tactics"])
            self.assertEqual(parsed[0]["immutable_suffix"], "end Foo")

            idx = AntiLeakagePremiseIndex(root)
            hits = idx.search("Nat m + 0", excluded_module_substr="TargetMod", excluded_lemma="target_lem")
            mods = [h["module"] for h in hits]
            self.assertTrue(any("AllowedMod" in m for m in mods))
            self.assertFalse(any("TargetMod" in m for m in mods))

    def test_mcp_exact_math_and_tier3_tier4_definitions(self) -> None:
        from extract_rh_tasks import build_tier5_re_tasks
        self.assertEqual(eval_exact_math("pow(2, 127) - 1"), "170141183460469231731687303715884105727")
        self.assertEqual(eval_exact_math("is_prime(7919)"), "True")
        self.assertEqual(len(build_tier3_egm_audit_tasks()), 4)
        self.assertEqual(len(build_tier4_swe_tasks()), 4)
        self.assertEqual(len(build_tier5_re_tasks()), 4)

    def test_mcp_stdio_roundtrip(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(HERE / "mcp_clover_tools.py")],
            input='{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"xcruncher_exact_math","arguments":{"expression":"gcd(240820, 1024)"}}}\n',
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("s2_lean_premise_search", proc.stdout)
        self.assertIn('"4"', proc.stdout)

    def test_opus55_audit_regressions_issues_1_to_7(self) -> None:
        # Issue 1: `decide +native`, `'\"'` char-literal smuggling, and `#print axioms` check
        self.assertTrue(len(find_forbidden_constructs("decide +native")) > 0)
        self.assertIn("native_decide", find_forbidden_constructs("have c : Char := '\"'\n  native_decide\n  -- \""))
        ok_ax, _ = verify_printed_axioms("'goal1' depends on axioms: [propext, Classical.choice, Quot.sound]", "goal1")
        self.assertTrue(ok_ax)
        bad_ax, err_ax = verify_printed_axioms("'goal1' depends on axioms: [propext, Lean.ofReduceBool]", "goal1")
        self.assertFalse(bad_ax)
        self.assertIn("Lean.ofReduceBool", err_ax)
        # Multi-line wrapped `#print axioms` output (`_private...` long names) MUST reject `sorryAx` on line 2+
        wrapped_ax = "'_private.0.Bl.goal1' depends on axioms: [propext,\n sorryAx,\n Classical.choice,\n Quot.sound]"
        bad_wrap, err_wrap = verify_printed_axioms(wrapped_ax, "goal1")
        self.assertFalse(bad_wrap)
        self.assertIn("sorryAx", err_wrap)

        # Tier 3 balance check: must contain exactly 2 provable and 2 unprovable tasks
        t3 = build_tier3_egm_audit_tasks()
        self.assertEqual(sum(1 for x in t3 if x["is_provable"]), 2)
        self.assertEqual(sum(1 for x in t3 if not x["is_provable"]), 2)

        # Issue 2: `run_tac`, `elab`, `macro`, `#eval` rejected
        for snippet in ('run_tac pure ()', 'elab "x" : tactic => pure ()', 'macro "y" : tactic => `(tactic| rfl)', '#eval 1 + 1'):
            self.assertTrue(len(find_forbidden_constructs(snippet)) > 0, f"Expected rejection for: {snippet}")

        # Issue 3: `extract_clean_tactic_block` MUST preserve nested `have h : ... := by`!
        nested = "  have h : n + 0 = n := by\n    simp\n  exact h"
        self.assertEqual(extract_clean_tactic_block(nested).strip(), nested.strip())
        echoed = "```lean\ntheorem foo (n : Nat) : n + 0 = n := by\n  have h : n + 0 = n := by\n    simp\n  exact h\n```"
        self.assertIn("have h : n + 0 = n := by", extract_clean_tactic_block(echoed))

        # Issue 5: Sandboxed SWE runner survives `SystemExit(1)` AND rejects `sys.exit(0)` bypass before asserts!
        ok_s, _ = run_python_sandbox("def f(x): return x + 1\n", timeout_s=5.0, test_code="assert f(2) == 3\n")
        self.assertTrue(ok_s)
        ok_exit, _ = run_python_sandbox("raise SystemExit(1)\n", timeout_s=5.0, test_code="assert True\n")
        self.assertFalse(ok_exit)
        ok_exit0, _ = run_python_sandbox("import sys\nsys.exit(0)\n", timeout_s=5.0, test_code="assert False, 'must not pass'\n")
        self.assertFalse(ok_exit0)

        # Issue 6: Pattern-matching theorems without `:=` must NOT leak `| 0 => ...` proof arms
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src" / "Formalism").mkdir(parents=True)
            (root / "src" / "Formalism" / "PatMod.lean").write_text(
                "import Mathlib\ntheorem pat_thm : (n : Nat) -> n + 0 = n\n  | 0 => rfl\n  | k + 1 => by simp [secret_body_token]\n",
                encoding="utf-8",
            )
            idx = AntiLeakagePremiseIndex(root)
            hits = idx.search("pat_thm Nat")
            self.assertEqual(len(hits), 1)
            self.assertNotIn("secret_body_token", hits[0]["signature"])

        # Issue 7: `factorial(2000)` (> 4300 digits) summarizes cleanly without Python 3.11+ str limit crash, and unbounded pow raises
        fac_s = eval_exact_math("factorial(2000)")
        self.assertIn("bit_length=", fac_s)
        with self.assertRaises(ValueError):
            eval_exact_math("pow(10, 10000000)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
