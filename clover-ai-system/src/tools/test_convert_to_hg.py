#!/usr/bin/env python3
"""Unit tests for the universal Clover HG4-16R / K31H converter (`convert_to_hg.py`)."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import convert_to_hg as chg  # noqa: E402


class TestConvertToHG(unittest.TestCase):
    def test_odd_non_multiple_shapes_and_outlier_parity(self) -> None:
        rng = np.random.Generator(np.random.PCG64(7))
        # Neither R=173 (%16!=0) nor K=300 (%128!=0) is block-aligned
        w = rng.standard_normal((173, 300), dtype=np.float32) * 0.03
        w[:, [5, 199, 299]] *= 35.0
        blob, meta = chg.pack_matrix_hg(w, name="odd.proj.weight", bits=4, use_ldlq=True)
        self.assertEqual(meta["orig_shape"], [173, 300])
        self.assertEqual(meta["padded_shape"], [176, 384])

        x = rng.standard_normal((19, 300), dtype=np.float32)
        y_hg = chg.forward_hg_matmul(x, blob)
        self.assertEqual(y_hg.shape, (19, 173))

        v = chg.verify_matrix_parity(w, blob, seed=99)
        self.assertGreater(v["hg_cosine"], 0.996)
        self.assertGreater(v["hg_sqnr_db"], v["naive_int4_sqnr_db"] + 5.0)

    def test_mxfp4_and_fp8_safetensors_end_to_end_k31h(self) -> None:
        rng = np.random.Generator(np.random.PCG64(123))
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "mxfp4_src"
            out = Path(tmp) / "mxfp4_hg"
            src.mkdir()

            # Build a 3D MXFP4 expert tensor [4 experts, R=64, K=128] + FP8 trunk [64, 128]
            pk = rng.integers(0, 256, size=(4, 64, 64), dtype=np.uint8)
            sc = np.full((4, 64, 4), 125, dtype=np.uint8)  # 2^(125-127) = 0.25
            fp8 = rng.integers(0, 120, size=(64, 128), dtype=np.uint8)  # finite positive FP8_E4M3

            raw_pk = pk.tobytes()
            raw_sc = sc.tobytes()
            raw_fp8 = fp8.tobytes()
            hdr = {
                "model.layers.0.self_attn.o_proj.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [64, 128],
                    "data_offsets": [0, len(raw_fp8)],
                },
                "model.layers.0.mlp.experts.up_proj.weight_packed": {
                    "dtype": "U8",
                    "shape": [4, 64, 64],
                    "data_offsets": [len(raw_fp8), len(raw_fp8) + len(raw_pk)],
                },
                "model.layers.0.mlp.experts.up_proj.weight_scale": {
                    "dtype": "U8",
                    "shape": [4, 64, 4],
                    "data_offsets": [len(raw_fp8) + len(raw_pk), len(raw_fp8) + len(raw_pk) + len(raw_sc)],
                },
            }
            hb = json.dumps(hdr).encode("utf-8")
            hb += b" " * ((8 - len(hb) % 8) % 8)
            with (src / "model.safetensors").open("wb") as fh:
                fh.write(struct.pack("<Q", len(hb)))
                fh.write(hb)
                fh.write(raw_fp8)
                fh.write(raw_pk)
                fh.write(raw_sc)

            rep = chg.convert_checkpoint(
                str(src),
                out,
                bits=4,
                moe_cold_bits=2,
                hot_experts=2,
                verify=True,
            )
            self.assertEqual(rep["quantized_tensors"], 5)  # 1 FP8 trunk + 4 MXFP4 experts (2 hot HG4 + 2 cold HG2)
            tiers = [t["tier"] for t in rep["tensors"] if "tier" in t]
            self.assertEqual(tiers.count("hg4_hot"), 3)
            self.assertEqual(tiers.count("hg2_cold"), 2)
            self.assertGreater(rep["verification_summary"]["mean_hg_cosine"], 0.96)

    def test_self_test_entrypoint(self) -> None:
        self.assertEqual(chg.main(["--self-test"]), 0)


if __name__ == "__main__":
    unittest.main()
