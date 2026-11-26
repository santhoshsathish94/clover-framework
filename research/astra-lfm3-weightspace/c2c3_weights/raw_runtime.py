from __future__ import annotations

import argparse
import json
import os
import sys
import unittest
from pathlib import Path


def initialize_paths(root):
    experiment = root / "c2c3_weights"
    os.environ["HF_HOME"] = str(experiment / "cache" / "huggingface")
    os.environ["HF_MODULES_CACHE"] = str(experiment / "cache" / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "4"
    for directory in (experiment / "vendor", experiment / "vendor" / "ncm"):
        sys.path.insert(0, str(directory))
    return experiment


def setup_metadata(root, source):
    sys.path.insert(0, str(root / "composite"))
    from prepare import physical_copy, write_json

    experiment = root / "c2c3_weights"
    destination = experiment / "c2"
    destination.mkdir(exist_ok=False)
    records = []
    for path in sorted(source.iterdir()):
        if path.is_file() and (path.suffix in (".json", ".jsonl", ".py", ".jinja") or path.name == "LICENSE"):
            records.append(physical_copy(path, destination / path.name, experiment))
    for filename, copied in (("model.safetensors", "0-model.safetensors"), ("adapter.pt", "1-adapter.pt")):
        records.append(physical_copy(experiment / "inputs" / copied, destination / filename, experiment))
    write_json(experiment / "RUNTIME_INPUTS.json", {"files": records})
    print(json.dumps({"c2_runtime_files": len(records), "physical_copies": True}), flush=True)


def load_c2(root, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hybridlink import HybridLink

    experiment = initialize_paths(root)
    checkpoint = experiment / "c2"
    backbone, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa", output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise ValueError(f"C2 loading error: {key}: {info[key]}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    if backbone.get_input_embeddings().num_embeddings != 128000:
        raise ValueError("C2 embedding size changed")
    link = HybridLink(2048).to(dtype=torch.bfloat16)
    state = torch.load(adapter or checkpoint / "adapter.pt", map_location="cpu", weights_only=True)
    link.load_state_dict(state, strict=True)
    return backbone.eval().requires_grad_(False), link.eval().requires_grad_(False), tokenizer


def load_c3(root, checkpoint=None):
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf
    from lfm3_clover import Lfm3NativeHybridLink, Lfm3ComboBridge, Lfm3ExpertBridge, Lfm3NCMRenderer, _TokenizerShell
    from ncm_runtime import LexicalPointerDecoder
    from models.vqvae import VQVAE
    from models.next_concept import NextConceptTransformer

    experiment = initialize_paths(root)
    source = checkpoint or experiment / "inputs" / "2-lfm3_ncm_core_bf16.pt"
    state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    config = OmegaConf.load(experiment / "vendor" / "config.yaml")
    vq_config = dict(config.tokenizer.vqvae)
    vq_config.pop("checkpoint_path", None)
    vq_config["in_vocab_size"] = 128000
    vqvae = VQVAE(**vq_config).eval()
    vq_state = dict(state["vqvae_core"])
    pre = root / "composite" / "inputs" / "reference" / "data" / "omni" / "precomputed" / "ncm"
    for name, filename in (("encoder.wte.weight", "vqvae_wte.bf16.bin"), ("decoder.output_layer.weight", "vqvae_dec.bf16.bin")):
        vq_state[name] = torch.from_file(str(pre / filename), shared=False, size=128000 * 384, dtype=torch.bfloat16).reshape(128000, 384)
    initialized = vqvae.state_dict()
    omitted = set(initialized) - set(vq_state)
    expected_ema = {f"vq.{kind}_{index}" for kind in ("ema_cluster_size", "ema_w") for index in range(len(vqvae.vq.levels))}
    if omitted != expected_ema or not omitted <= set(dict(vqvae.named_buffers())):
        raise ValueError(f"unexpected missing VQ inference state: {sorted(omitted)}")
    vq_state.update({name: initialized[name] for name in omitted})
    vqvae.load_state_dict(vq_state, strict=True)
    config.model.prefix_len = 96
    config.model.predict_level0 = True
    config.model.prefix_pad_token_id = 124893
    net = NextConceptTransformer(vqvae_local=vqvae, config=config.model).eval()
    net.load_state_dict(state["net"], strict=True)
    pointer = LexicalPointerDecoder().eval()
    pointer.load_state_dict(state["ptr_dec"], strict=True)
    native = Lfm3NativeHybridLink().eval()
    native.load_state_dict(state["lfm3_native_hybrid_link"], strict=True)
    combo = Lfm3ComboBridge().eval()
    combo.load_state_dict(state["lfm3_combo_bridge"], strict=True)
    expert = Lfm3ExpertBridge().eval()
    expert.load_state_dict(state["lfm3_expert_bridge"], strict=True)
    modules = {"vqvae": vqvae, "net": net, "pointer": pointer, "native": native, "combo": combo, "expert": expert}
    for module in modules.values():
        module.eval().requires_grad_(False)
    return modules


def least_squares_gain(predicted, target):
    import numpy as np

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("incompatible gain-fitting vectors")
    denominator = float(np.sum(predicted * predicted))
    if denominator <= 0:
        raise ValueError("zero prediction energy")
    return max(0.0, float(np.sum(predicted * target)) / denominator)


class GainTests(unittest.TestCase):
    def test_known_gain(self):
        self.assertAlmostEqual(least_squares_gain([2, 4, 6], [1, 2, 3]), 0.5)

    def test_zero_energy_rejected(self):
        with self.assertRaisesRegex(ValueError, "energy"):
            least_squares_gain([0, 0], [1, 2])

    def test_no_negative_gain(self):
        self.assertEqual(least_squares_gain([1, 2], [-1, -2]), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--setup-metadata", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.root and args.setup_metadata:
        setup_metadata(args.root, args.setup_metadata)
    elif args.root and args.smoke:
        initialize_paths(args.root)
        import torch

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        backbone, link, tokenizer = load_c2(args.root)
        token_ids = tokenizer.encode("The train arrives at noon.", add_special_tokens=False)
        result = backbone(torch.tensor([token_ids]), output_hidden_states=True, use_cache=False)
        latent = link(result.hidden_states[-1][:, -1])
        modules = load_c3(args.root)
        print(json.dumps({
            "c2_loaded": True, "c2_state_shape": list(latent.shape),
            "c3_modules_loaded": list(modules), "c3_vocab": modules["vqvae"].in_vocab_size,
            "c3_prefix": modules["net"].prefix_len, "c3_levels": list(modules["vqvae"].vq.levels),
            "c2_latent_rms": float(latent.float().square().mean().sqrt()),
            "c2_embedding_rms": float(backbone.get_input_embeddings()(torch.tensor(token_ids)).float().square().mean().sqrt()),
        }), flush=True)
    else:
        parser.error("choose --self-test or --root with --setup-metadata/--smoke")


if __name__ == "__main__":
    main()