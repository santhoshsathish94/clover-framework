#!/usr/bin/env python3
"""Offline unit & C11 native-library tests for `hg4_engine.py` (`Clover HG4-128`)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

STUDIO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDIO / "clover"))

import hg4_engine  # noqa: E402


class TestHg4EngineOffline(unittest.TestCase):
    def test_native_lib_loads_and_reports_simd_isa(self):
        lib = hg4_engine.get_hg4_lib()
        self.assertIsNotNone(lib, "liblfm3_hg4_engine failed to load")
        isa = lib.lfm3_hg4_isa_name().decode("utf-8")
        self.assertIn("HG4-128", isa)

    def test_chat_template_matches_lfm3_omni_jinja(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate expression",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "defer_loading": True,
                "function": {"name": "hidden_tool", "description": "Deferred"},
            },
        ]
        msgs = [
            {"role": "system", "content": "You are Clover."},
            {"role": "user", "content": "Compute 6 * 7."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "calculate", "arguments": '{"expression": "6 * 7"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
        rendered = hg4_engine.render_lfm3_chat(msgs, tools=tools, add_generation_prompt=True)
        self.assertTrue(rendered.startswith("<|startoftext|><|im_start|>system\nYou are Clover.\n\n<tools>\n"))
        self.assertIn('"name": "calculate"', rendered)
        self.assertNotIn("hidden_tool", rendered)
        self.assertIn("<tool_call>\n<function=calculate>\n<parameter=expression>\n6 * 7\n</parameter>\n</function>\n</tool_call>", rendered)
        self.assertIn("<|im_start|>tool\n<tool_response>\n42\n</tool_response><|im_end|>\n", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think></think>"))


if __name__ == "__main__":
    unittest.main()
