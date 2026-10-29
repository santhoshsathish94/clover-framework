"""Unit tests for the hyphenated complete-English Clover Object Model (`clover-object-model.py`)."""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "clover-ai-system" / "src" / "tools" / "hybrid-ncm-k3"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from clover_hyphen_runtime import load_hyphenated_module
from clover_request_pipeline import (
    CloverModelBackendFactory,
    CloverPlanWindowSplitter,
    CloverRequestValidator,
)

_IMPL_SUITE = load_hyphenated_module(
    Path(__file__).with_name("test-clover-object-model.py"),
    module_alias="test_clover_object_model_hyphen_suite",
)
TestCloverObjectModel = _IMPL_SUITE.TestCloverObjectModel


class TestCloverRequestPipeline(unittest.TestCase):
    """Verifies SOLID Factory, Validator, and PlanWindowSplitter patterns in `clover_request_pipeline.py`."""

    def test_01_model_backend_factory_resolves_all_three_models_and_null_inputs(self) -> None:
        self.assertEqual(CloverModelBackendFactory.resolve_canonical_model_key(None), "nemotron-3.5-super-vl")
        self.assertEqual(CloverModelBackendFactory.resolve_canonical_model_key(""), "nemotron-3.5-super-vl")
        self.assertEqual(
            CloverModelBackendFactory.resolve_canonical_model_key("clover-nemotron-3.5-super-vl"),
            "nemotron-3.5-super-vl",
        )
        self.assertEqual(
            CloverModelBackendFactory.resolve_canonical_model_key("lfm3-1.7b"),
            "lfm3-1.7b",
        )
        self.assertEqual(
            CloverModelBackendFactory.resolve_canonical_model_key("clover-kimi-k3"),
            "kimi-k3",
        )
        models_list = CloverModelBackendFactory.build_openai_models_list(1700000000, 1048576, {})
        model_ids = [m["id"] for m in models_list]
        self.assertIn("nemotron-3.5-super-vl", model_ids)
        self.assertIn("lfm3-1.7b", model_ids)
        self.assertIn("clover-kimi-k3", model_ids)

    def test_02_request_validator_guards_sticky_output_type_and_null_fields(self) -> None:
        # A conversational question sent while the Video/Image tab was left selected should resolve to 'text'
        val_q = CloverRequestValidator.validate_chat_payload(
            {
                "session_id": None,
                "model": "nemotron-3.5-super-vl",
                "output_type": "video",
                "prompt": "A deeper dive into the quantum realm and looking at the cosmos through the quantum perspective",
            }
        )
        self.assertEqual(val_q.session_id, "default")
        self.assertEqual(val_q.output_type, "text")

        # An explicit image request with output_type='image' should stay 'image'
        val_img = CloverRequestValidator.validate_chat_payload(
            {
                "session_id": "s1",
                "model": "lfm3-1.7b",
                "output_type": "image",
                "prompt": "create an image of clover",
            }
        )
        self.assertEqual(val_img.model_key, "lfm3-1.7b")
        self.assertEqual(val_img.output_type, "image")

    def test_03_plan_window_splitter_never_truncates_overlong_sentences(self) -> None:
        class _DummyTok:
            def encode(self, txt: str) -> list[int]:
                return list(range(len(txt.split())))

        long_plan = (
            "\nVisual Details: Synthesized at 1280x720 HD resolution featuring multi-layer ridged mountain peaks, "
            "golden atmospheric scattering, and rippled alpine lake reflections, available in PNG, JPG, WebP, and BMP formats."
        )
        split_wins = CloverPlanWindowSplitter.split_overlong_windows([long_plan], _DummyTok(), max_bpe=12)
        self.assertGreater(len(split_wins), 1)
        self.assertTrue(split_wins[0].startswith("\nVisual Details:"))
        self.assertTrue(split_wins[-1].endswith("BMP formats."))


if __name__ == "__main__":
    unittest.main()
