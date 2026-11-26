from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")


def assign_complete(module, state):
    result = module.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise ValueError(f"unexpected checkpoint tensors: {result.unexpected_keys}")
    if hasattr(module, "tie_weights"):
        module.tie_weights()
    loaded = {tensor.data_ptr() for tensor in state.values()}
    parameters = dict(module.named_parameters(remove_duplicate=False))
    for name, parameter in parameters.items():
        if parameter.is_meta or parameter.data_ptr() not in loaded:
            raise ValueError(f"unassigned model parameter: {name}")
    for name in result.missing_keys:
        if name not in parameters or parameters[name].data_ptr() not in loaded:
            raise ValueError(f"missing checkpoint tensor: {name}")
    if any(buffer.is_meta for buffer in module.buffers()):
        raise ValueError("unassigned model buffer")
    return module.eval().requires_grad_(False)


class Bundle:
    def __init__(self, package: Path):
        from safetensors import safe_open

        self.package = package.resolve(strict=True)
        self.handle = safe_open(self.package / "model.safetensors", framework="pt", device="cpu")
        self.keys = tuple(self.handle.keys())

    def tensor(self, name):
        return self.handle.get_tensor(name)

    def state(self, prefix):
        prefix += "."
        return {name[len(prefix):]: self.tensor(name) for name in self.keys if name.startswith(prefix)}

    def group(self, path):
        return {name: tensor.float().numpy() for name, tensor in self.state("c2c3." + path).items()}

    def table(self, name):
        import torch

        return self.tensor("c2c3." + name).view(torch.uint16).numpy()


def load_c2c3(bundle: Bundle):
    import numpy as np
    import reference_c2c3 as reference

    class BundledC2C3(reference.OmniC2C3Runtime):
        def __init__(self):
            started = time.perf_counter()
            self.base_dir = bundle.package
            self.pre = bundle.package
            self.lock = threading.Lock()
            self.tok = reference.Lfm3PureBpeTokenizer(bundle.package / "c2_tokenizer.json")
            self.vocab_size = reference.V_LFM3
            self.plan_t = bundle.group("hybridagent/plan-trunk-0")
            self.has_hl2048 = "hl2048.proj1_fused.weight" in self.plan_t
            self.bridge_t = bundle.group("bridges/bridge-trunk-0")
            self.vq_trunk = bundle.group("ncm/vqvae-trunk-0")
            self.vq_experts = [bundle.group(f"ncm/vqvae-expert-{index}") for index in range(11)]
            self.vq_wte_mm = bundle.table("vq_wte")
            self.vq_dec_mm = bundle.table("vq_dec")
            self.lfm3_emb_mm = bundle.table("lfm3_embedding")
            self.ncm_global = bundle.group("ncm/ncm-global-0")
            self._ncm_blocks_all = [bundle.group(f"ncm/ncm-trunk-{index}") for index in range(12)]
            self.ncm_blocks = [self._ncm_blocks_all[index] for index in (0, 5, 11)]
            self.ptr_global = bundle.group("ncm/ptr-global-0")
            self.ptr_blocks = [bundle.group(f"ncm/ptr-trunk-{index}") for index in range(3)]
            self.rope_cos, self.rope_sin = reference.precompute_rope_cos_sin(max_pos=4096, head_dim=64, base=100000.0)
            level_ids = [index for index, level in enumerate(reference.VQ_LEVELS[1:]) for _ in range(level)]
            self.level_pos_embed = self.ncm_global["transformer.wle.weight"][np.array(level_ids, dtype=np.int64)][None, :, :]
            self.grammar_ids = {self.tok.eos_token_id}
            self.repeatable_grammar_ids = {self.tok.eos_token_id}
            for word in reference.GRAMMAR_CONNECTOR_WORDS:
                for variant in (word, " " + word):
                    for token_id in self.tok.encode(variant, allow_special=False):
                        if 0 <= token_id < self.vocab_size:
                            self.grammar_ids.add(token_id)
                            if word.lower() in (
                                "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from",
                                "as", "is", "are", "was", "were", ".", ",", ";", ":", "-", "/", "//", "://", "_", "=", "&", "(", ")", "[", "]",
                            ):
                                self.repeatable_grammar_ids.add(token_id)
            self.setup_ms = round((time.perf_counter() - started) * 1000.0, 1)

    return BundledC2C3()


def load_native(bundle: Bundle):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.processing_utils import ProcessorMixin

    metadata = bundle.package / "omni"
    config = AutoConfig.from_pretrained(metadata, trust_remote_code=True, local_files_only=True)
    with init_empty_weights(include_buffers=False):
        native = AutoModelForMultimodalLM.from_config(config, trust_remote_code=True, attn_implementation="sdpa")
    assign_complete(native, bundle.state("c1"))
    codec_class = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecModel", metadata, local_files_only=True)
    codec_config = codec_class.config_class.from_json_file(metadata / "audio_tokenizer" / "config.json")
    with init_empty_weights(include_buffers=False):
        codec = codec_class(codec_config)
    codec = assign_complete(codec, bundle.state("codec")).float()
    processor_class = get_class_from_dynamic_module("processing_lfm3_omni.Lfm3OmniProcessor", metadata, local_files_only=True)

    class BundledProcessor(processor_class):
        @classmethod
        def get_processor_dict(cls, pretrained_model_name_or_path, **kwargs):
            result, kwargs = ProcessorMixin.get_processor_dict.__func__(cls, pretrained_model_name_or_path, **kwargs)
            result.pop("audio_tokenizer_subfolder", None)
            result["audio_tokenizer"] = codec
            return result, kwargs

    processor = BundledProcessor.from_pretrained(metadata, trust_remote_code=True, local_files_only=True)
    if processor.audio_tokenizer is not codec:
        raise ValueError("processor did not retain the bundled codec")
    processor.tokenizer.padding_side = "left"
    return native, processor, codec


class CompositeModel:
    def __init__(self, package: Path):
        self.bundle = Bundle(package)
        self.native, self.processor, self.codec = load_native(self.bundle)
        self.c2c3 = load_c2c3(self.bundle)

    def states(self, token_ids):
        import torch

        with torch.inference_mode():
            inputs = torch.tensor([token_ids], dtype=torch.long)
            result = self.native.model.language_model(input_ids=inputs, use_cache=False, return_dict=True)
            return result.last_hidden_state[0].float().numpy()

    def compose(self, text, prefix_ids=()):
        c1, c1_info = self.c2c3.c1_infer_states(text, self.states, prefix_ids=prefix_ids, model_name="bundled LFM3-Omni-2B")
        c2, c2_info = self.c2c3.c2_plan(text, c1_state=c1, jepa_arch="none")
        answer, c3_info = self.c2c3.c3_decode(c2, full_ncm=True)
        return answer, {"c1": c1_info, "c2": c2_info, "c3": c3_info}, c1, c2


class LoaderTests(unittest.TestCase):
    def test_missing_parameter_rejected(self):
        import torch

        module = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "unassigned"):
            assign_complete(module, {"weight": torch.ones(2, 2)})

    def test_complete_assignment(self):
        import torch

        module = torch.nn.Linear(2, 2)
        source = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
        assign_complete(module, source)
        self.assertEqual(module.weight.data_ptr(), source["weight"].data_ptr())
        self.assertFalse(module.weight.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__])
    elif args.package and args.smoke:
        import torch

        experiment = args.package.resolve().parent
        os.environ["HF_MODULES_CACHE"] = str(experiment / "cache" / "modules")
        os.environ["HF_HOME"] = str(experiment / "cache" / "huggingface")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        model = CompositeModel(args.package)
        token_ids = model.processor.tokenizer.encode("The result is 42.", add_special_tokens=False)
        states = model.states(token_ids)
        if states.shape != (len(token_ids), 2048):
            raise ValueError("wrong real C1 state shape")
        print(json.dumps({"native_and_codec_loaded": True, "c1_state_shape": list(states.shape), "c2_tensors": len(model.c2c3.plan_t), "ncm_blocks": len(model.c2c3._all_ncm_blocks())}), flush=True)
    else:
        parser.error("choose --self-test or --package with --smoke")


if __name__ == "__main__":
    main()