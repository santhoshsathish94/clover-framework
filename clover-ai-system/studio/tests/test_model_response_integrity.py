"""Offline response-integrity checks. Recording providers are test doubles, not live-model evidence."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import clover_cycle
import image_gen
import jepa_engine
import k3_host_engine
import lfm3_layers
import omni_c2_c3
import omni_native
import server


class RecordingTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 7, 1

    def decode(self, token_ids, **kwargs):
        words = {1: "red", 2: "green", 3: "blue", 7: ""}
        return " ".join(words.get(int(token_id), "") for token_id in token_ids)


class ModelResponseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="clover-response-integrity-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def omni(self, text):
        engine = object.__new__(omni_native.OmniNative)
        engine.base_dir = engine.media_dir = self.base
        engine.lock = threading.Lock()
        engine.stage, engine._extra = {"name": "idle"}, {}
        engine.workdir, engine.workname = None, ""
        engine.perms = SimpleNamespace(get=lambda: {"code": "deny"})
        engine._chat = Mock(return_value=(text, [], {
            "prompt_tokens": 12, "reused_tokens": 0, "new_tokens": 10 if text else 0,
            "prefill_s": 0.01, "decode_s": 0.01, "finish": "stop",
        }))
        return engine

    def test_missing_image_models_never_create_media(self):
        output = self.base / "media"
        for preset in ("sd15-lcm", "sd15", "unknown-model"):
            with self.subTest(preset=preset):
                with self.assertRaisesRegex(RuntimeError, "image model unavailable"):
                    image_gen.ImageGen(self.base, preset).generate("A yellow teapot.", output)
        self.assertFalse(output.exists())

    def test_missing_selection_never_uses_previous_model(self):
        with patch.object(server.S, "omni_engine", None), patch.object(server.S, "engine", object()):
            self.assertIsNone(server.engine_for("omni"))
            with patch.object(server, "model_catalog", return_value=[]):
                with self.assertRaisesRegex(ValueError, "Unknown or unavailable model"):
                    server.engine_for("unknown-model")

    def test_capability_question_calls_model(self):
        text = "The selected model's own response."
        engine = self.omni(text)
        result = engine.run_turn([], "Can you generate images?", tools=False, system="SYS")
        self.assertEqual(engine._chat.call_count, 1)
        self.assertEqual(result["reply"], text)
        self.assertEqual(result["generation"][0]["new_tokens"], 10)

    def test_wrong_arithmetic_is_not_replaced(self):
        engine = self.omni("390")
        result = engine.run_turn([], "What is 17 * 23?", tools=False, system="SYS")
        self.assertEqual(engine._chat.call_count, 3)
        self.assertEqual(result["reply"], "390")
        self.assertTrue(any(layer.get("layer") == "outcome" and layer.get("passed") is False
                            for layer in result["layers"]))

    def test_empty_answer_and_provider_errors_propagate(self):
        engine = self.omni("")
        with self.assertRaisesRegex(RuntimeError, "returned no answer"):
            engine.run_turn([], "Say hello.", tools=False, system="SYS")
        engine._chat = Mock(side_effect=TypeError("provider failure"))
        with self.assertRaisesRegex(TypeError, "provider failure"):
            engine.run_turn([], "Say hello.", tools=False, system="SYS")
        self.assertEqual(engine._chat.call_count, 1)

    def test_c1_requires_selected_inference_provider(self):
        engine = self.omni("unused")
        other = Mock(side_effect=AssertionError("different model used"))
        engine.local_omni = SimpleNamespace(_c1_server=other)
        with self.assertRaisesRegex(RuntimeError, "C1 inference is unavailable"):
            engine._c1_server()
        other.assert_not_called()

    def test_decoder_keeps_predicted_tokens(self):
        runtime = object.__new__(omni_c2_c3.OmniC2C3Runtime)
        runtime.tok, runtime.vocab_size = RecordingTokenizer(), 8
        runtime.repeatable_grammar_ids = set()
        text, token_ids, fallback = runtime._decode_window_aligned(
            np.array([[0., 0., 100.], [0., 0., 100.]]), np.array([1, 2, 3]), [1, 2])
        self.assertEqual((text, token_ids, fallback), ("blue blue", [3, 3], False))
        with self.assertRaisesRegex(RuntimeError, "no decoded text"):
            runtime._decode_window_aligned(np.array([[100.]]), np.array([7]), [1, 2])
        runtime.lfm3_emb_mm = None
        with self.assertRaisesRegex(RuntimeError, "embeddings are unavailable"):
            runtime.token_embeddings_2048([1])

    def test_missing_neural_stages_never_pass_text_through(self):
        kit = object.__new__(lfm3_layers.LayerKit)
        kit._runtime_for = Mock(return_value=None)
        for operation in (
            lambda: kit.c1_infer("Original text."),
            lambda: kit.c2_plan_state("Question.", c1_state={"raw_text": "Original text."}),
            lambda: kit.c3_decode({"stage": "c2_plan", "pids_primary": [1]}),
            lambda: kit.ncm_render("Original text."),
        ):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                operation()
        with self.assertRaisesRegex(ValueError, "real C2 plan"):
            kit.c3_decode({"stage": "c2_plan", "raw_text": "Not a real plan."})

    def test_model_paths_stay_in_selected_installation(self):
        for model in ("omni", "clover-35b-a3b", "nemotron-3.5-super-vl", "kimi-k3"):
            with self.subTest(model=model):
                self.assertTrue(omni_c2_c3.precomputed_dir(self.base, model).is_relative_to(self.base))
                self.assertTrue(omni_c2_c3.missing_c2_c3_files(self.base, model))

    def test_jepa_never_generates_replacement_weights(self):
        with patch.object(jepa_engine.CloverJEPARuntime, "_resolve_trunk_dir", return_value=self.base), \
                patch.object(jepa_engine, "_build_default_weights", side_effect=AssertionError("random weights used")):
            with self.assertRaises(FileNotFoundError):
                jepa_engine.CloverJEPARuntime(self.base)
            (self.base / "index.bin").touch()
            with patch.object(jepa_engine, "read_clvrtrk1", side_effect=RuntimeError("corrupt checkpoint")):
                with self.assertRaisesRegex(RuntimeError, "corrupt checkpoint"):
                    jepa_engine.CloverJEPARuntime(self.base)
            with patch.object(jepa_engine, "read_clvrtrk1", return_value={}):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    jepa_engine.CloverJEPARuntime(self.base)

    def test_hosted_worker_is_not_replaced_or_relabeled(self):
        backing = SimpleNamespace(display_model_name="Omni", run_turn=Mock(side_effect=AssertionError("Omni used")))
        engine = k3_host_engine.K3HostEngine("nemotron-3.5-super-vl", backing, self.base,
                                           SimpleNamespace(get=lambda: {"code": "deny"}))
        text = "I am LFM3-Omni-2B, built by Liquid AI."
        engine._call_worker = Mock(return_value={"reply": text})
        for web in (False, True):
            with patch.object(k3_host_engine.paths, "WEB", web):
                result = engine.run_turn([], "What model are you?", tools=False)
            self.assertEqual(result["reply"], text)
            self.assertIsNone(result["generation"][0]["new_tokens"])
            self.assertFalse(any(layer.get("layer") == "ncm" for layer in result["layers"]))
        self.assertEqual(engine._call_worker.call_count, 2)
        self.assertEqual(backing.display_model_name, "Omni")
        backing.run_turn.assert_not_called()
        engine._call_worker = Mock(side_effect=RuntimeError("worker unavailable"))
        with self.assertRaisesRegex(RuntimeError, "worker unavailable"):
            engine.run_turn([], "Say hello.", tools=False)

    def test_failed_image_tool_prevents_substitute_text(self):
        events = []
        engine = SimpleNamespace(base_dir=self.base, media_dir=self.base / "media",
                                 execute_tool=Mock(return_value=(True, "Error: image model unavailable", "allowed")))
        call = {"id": "image-test", "name": "generate_image", "arguments": {"prompt": "A teapot."}}
        with self.assertRaisesRegex(RuntimeError, "Image generation failed"):
            omni_native.execute_s3_pre_calls(engine, call, [], emit=events.append)
        self.assertTrue(any(event.get("type") == "tool_result" and event["worked"] is False for event in events))

    def test_retry_returns_exact_model_candidate(self):
        candidates = iter(["Model working. ANSWER: 30", "Other working. ANSWER: 30"])
        answer, info = clover_cycle.retry_cycle(
            "And how much would 30 pens cost?", "30 x $0.67 = $20.10.", "Try again",
            rewrite=lambda instruction: "New response.", solve=lambda index, note=None: next(candidates))
        self.assertEqual(answer, "Model working. ANSWER: 30")
        self.assertEqual(info["solutions"], [30, 30])
        with self.assertRaisesRegex(RuntimeError, "no answer during retry"):
            clover_cycle.retry_cycle("Write a short story.", "An earlier story.", "Try again.",
                                     rewrite=lambda instruction: "")


if __name__ == "__main__":
    unittest.main(verbosity=2)