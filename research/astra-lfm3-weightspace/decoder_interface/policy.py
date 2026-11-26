from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
import types
import unittest


POLICIES = ("stock", "no_forced_copy", "no_copy", "raw_restricted", "raw_full")


def copy_variant(original, mode):
    if mode not in ("no_forced_copy", "no_copy"):
        raise ValueError("unsupported copy policy")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "eff_scale" for target in node.targets)
    ]
    expected = ast.parse("soft_scale if has_semi else (soft_scale + 65.0)", mode="eval").body
    if len(assignments) != 1 or ast.dump(assignments[0].value) != ast.dump(expected):
        raise ValueError("frozen copy-scale expression does not match the experiment")
    expression = "soft_scale" if mode == "no_forced_copy" else "soft_scale * 0.0"
    assignments[0].value = ast.parse(expression, mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, f"<decoder-policy:{mode}>", "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


def greedy_neural_decode(logits, allowed_ids, tokenizer):
    import numpy as np

    logits = np.asarray(logits)
    allowed_ids = np.asarray(allowed_ids)
    if logits.ndim != 2 or allowed_ids.ndim != 1 or logits.shape[1] != len(allowed_ids):
        raise ValueError("neural logits/vocabulary shape mismatch")
    if not np.isfinite(logits).all() or not len(allowed_ids):
        raise ValueError("empty vocabulary or nonfinite logits")
    token_ids = []
    for column in np.argmax(logits, axis=-1).tolist():
        token_id = int(allowed_ids[column])
        if token_id == tokenizer.eos_token_id:
            break
        if token_id != tokenizer.pad_token_id:
            token_ids.append(token_id)
    return tokenizer.decode(token_ids).strip(), token_ids


def install_policy(runtime, name):
    if name not in POLICIES:
        raise ValueError(name)
    if getattr(runtime, "_decoder_probe_policy", None) is not None:
        raise ValueError("policy already installed on this runtime")
    runtime._decoder_probe_policy = name
    runtime._decoder_probe_trace = []
    original_pointer = runtime._pointer_decode_from_zhat.__func__
    if name in ("no_forced_copy", "no_copy"):
        pointer = copy_variant(original_pointer, name)
        runtime._pointer_decode_from_zhat = types.MethodType(pointer, runtime)
    elif name.startswith("raw_"):
        def raw_pointer(self, z_hat, z_rh_v, z_bpe_v, pids, tail_ids, clean, precomputed=None):
            if precomputed is None or tail_ids:
                raise ValueError("raw diagnostic requires a complete precomputed window")
            logits = precomputed[2]
            allowed_ids = precomputed[3]
            text, token_ids = greedy_neural_decode(logits, allowed_ids, self.tok)
            self._decoder_probe_trace.append({
                "allowed_ids": len(allowed_ids),
                "positions": int(logits.shape[0]),
                "token_ids": token_ids,
                "policy": name,
            })
            return text, False

        runtime._pointer_decode_from_zhat = types.MethodType(raw_pointer, runtime)
        if name == "raw_full":
            def full_vocabulary(self, plan_str):
                import numpy as np

                return np.arange(self.vocab_size, dtype=np.int64)

            runtime._build_allowed_vocab_ids = types.MethodType(full_vocabulary, runtime)
    return runtime


class CopyProbe:
    def scale(self, soft_scale, has_semi):
        eff_scale = soft_scale if has_semi else (soft_scale + 65.0)
        return eff_scale


class PolicyTests(unittest.TestCase):
    def test_stock_still_has_forced_bonus(self):
        probe = CopyProbe()
        changed = types.MethodType(copy_variant(CopyProbe.scale, "no_forced_copy"), probe)
        self.assertEqual(changed(3.0, False), 3.0)
        self.assertEqual(probe.scale(3.0, False), 68.0)
        self.assertEqual(changed(3.0, True), probe.scale(3.0, True))

    def test_no_copy_removes_both_terms(self):
        probe = CopyProbe()
        changed = types.MethodType(copy_variant(CopyProbe.scale, "no_copy"), probe)
        self.assertEqual(changed(3.0, False), 0.0)
        self.assertEqual(changed(3.0, True), 0.0)

    def test_unexpected_expression_rejected(self):
        def different(self, soft_scale, has_semi):
            eff_scale = soft_scale + 7.0
            return eff_scale

        with self.assertRaisesRegex(ValueError, "expression"):
            copy_variant(different, "no_copy")

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            copy_variant(CopyProbe.scale, "unregistered")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy tests run in the isolated K3 venv")
    def test_raw_can_emit_non_input_token(self):
        class Tokenizer:
            eos_token_id = 2
            pad_token_id = 0

            def decode(self, ids):
                return " ".join({1: "question", 3: "answer"}[number] for number in ids)

        text, ids = greedy_neural_decode([[0.0, 9.0, -2.0], [0.0, -2.0, 9.0]], [1, 3, 2], Tokenizer())
        self.assertEqual(text, "answer")
        self.assertEqual(ids, [3])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy tests run in the isolated K3 venv")
    def test_raw_rejects_invalid_logit_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            greedy_neural_decode([[1.0, 2.0]], [0], object())


if __name__ == "__main__":
    unittest.main()