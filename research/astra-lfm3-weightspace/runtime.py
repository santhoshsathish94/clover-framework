import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import unittest


LAYER_INDICES = [0, 2, 4, 11, 17, 23]
GENERATION_SETTINGS = {
    "do_sample": False, "temperature": 1.0, "top_p": 1.0,
    "top_k": 0, "repetition_penalty": 1.0,
    "audio_temperature": 0.0, "audio_repetition_penalty": 1.2,
    "return_dict_in_generate": True,
}


def score_letter(text):
    match = re.match(r"^\s*(?:\(([ABCD])\)|([ABCD]))(?:[\s.,:)]|$)", text)
    return next((group for group in match.groups() if group), None) if match else None


def array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def last_vector(value):
    return value.reshape(-1, value.shape[-1])[-1]


def video_clock(frame_count):
    return {"fps": 2.0, "total_num_frames": frame_count, "duration": frame_count / 2.0,
            "frames_indices": list(range(frame_count))}


class OmniRuntime:
    def __init__(self, root, checkpoint=None, threads=8):
        self.root = Path(root)
        self.checkpoint = Path(checkpoint or self.root / "original")
        os.environ["HF_HOME"] = str(self.root / "cache" / "huggingface")
        os.environ["HF_MODULES_CACHE"] = str(self.root / "cache" / "modules")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = str(self.root / "cache" / "numba")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        torch.manual_seed(20261111)
        self.model, self.loading_info = AutoModelForMultimodalLM.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, attn_implementation="sdpa", output_loading_info=True,
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if self.loading_info.get(key):
                raise RuntimeError(f"Checkpoint loading issue: {key}: {self.loading_info[key]}")
        self.model.eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True, local_files_only=True,
        )
        self.processor.tokenizer.padding_side = "left"
        self.codec = self.processor.audio_tokenizer.float().eval().requires_grad_(False)
        if self.model.lm_head.weight.data_ptr() != self.model.get_input_embeddings().weight.data_ptr():
            raise RuntimeError("Declared text input/output weight tying did not load")
        self.letter_ids = []
        for letter in "ABCD":
            encoded = self.processor.tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise RuntimeError(f"MCQ letter is not one token: {letter}")
            self.letter_ids.append(encoded[0])
        self.capture = {}
        self.handles = []
        self.add_hooks()
        self.parameter_versions = {name: parameter._version for name, parameter in self.model.named_parameters()}

    def add_hooks(self):
        torch = self.torch

        def capture_hook(name):
            def hook(module, inputs, output):
                if name in self.capture:
                    return
                value = output
                if not isinstance(value, torch.Tensor):
                    value = value.last_hidden_state if hasattr(value, "last_hidden_state") else value[0]
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"Unsupported activation output at {name}")
                rows = value.reshape(-1, value.shape[-1])
                self.capture[name] = rows[-1].detach().float().cpu().numpy().copy()
                self.capture[name + ".mean"] = rows.detach().float().mean(0).cpu().numpy().copy()
            return hook

        for index in LAYER_INDICES:
            layer = self.model.model.language_model.layers[index]
            self.handles.append(layer.register_forward_hook(capture_hook(f"layer_{index}")))
        for name in ("audio_encoder", "audio_projector", "vision_tower", "multi_modal_projector"):
            module = getattr(self.model.model, name)
            self.handles.append(module.register_forward_hook(capture_hook(name)))
        self.handles.append(self.model.depth_decoder.model.register_forward_hook(capture_hook("speech_depth")))

        def final_norm_hook(module, inputs, output):
            if "final_norm_input" in self.capture:
                return
            vector = inputs[0][0, -1].detach().float()
            normalized = vector * torch.rsqrt(vector.square().mean() + module.variance_epsilon)
            self.capture["final_norm_input"] = vector.cpu().numpy().copy()
            self.capture["final_norm_unit"] = normalized.to(inputs[0].dtype).float().cpu().numpy().copy()
            self.capture["final_hidden"] = output[0, -1].detach().float().cpu().numpy().copy()

        def head_hook(module, inputs, output):
            if "letter_logits" not in self.capture:
                self.capture["letter_logits"] = last_vector(output)[self.letter_ids].detach().float().cpu().numpy().copy()

        self.handles.append(self.model.model.language_model.embedding_norm.register_forward_hook(final_norm_hook))
        self.handles.append(self.model.lm_head.register_forward_hook(head_hook))

    def census(self):
        aliases = {}
        parameters = []
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
            parameters.append({"name": name, "shape": list(parameter.shape),
                               "dtype": str(parameter.dtype), "numel": parameter.numel()})
        return {
            "python": sys.version, "prefix": sys.prefix,
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "numpy", "safetensors", "torchvision",
                "librosa", "soundfile", "torchcodec", "accelerate",
            )},
            "parameters": parameters,
            "aliases": [names for names in aliases.values() if len(names) > 1],
            "buffers": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                        for name, value in self.model.named_buffers()],
            "loading_info": {key: value for key, value in self.loading_info.items() if key != "conversion_errors"},
            "generation_settings": GENERATION_SETTINGS,
            "precision": {"model": str(next(self.model.parameters()).dtype),
                          "codec": str(next(self.codec.parameters()).dtype),
                          "attention": "CPU SDPA"},
            "codec_unique_parameters": sum(parameter.numel() for parameter in self.codec.parameters()),
            "codec_parameters": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                                  "numel": value.numel()} for name, value in self.codec.named_parameters()],
            "codec_buffers": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                               "numel": value.numel()} for name, value in self.codec.named_buffers()],
        }

    def inputs_for(self, case):
        import numpy as np
        import soundfile as sf
        from PIL import Image

        content = []
        media_kwargs = {}
        if case.get("image"):
            content.append({"type": "image"})
            media_kwargs["images"] = [Image.open(self.root / "fixtures" / case["image"]).convert("RGB")]
        if case.get("video"):
            content.append({"type": "video"})
            frames = np.load(self.root / "fixtures" / case["video"], allow_pickle=False)
            media_kwargs["videos"] = [frames]
            media_kwargs["video_metadata"] = [video_clock(len(frames))]
            media_kwargs["do_sample_frames"] = False
        if case.get("audio"):
            content.append({"type": "audio"})
            waveform, rate = sf.read(self.root / "fixtures" / case["audio"], dtype="float32")
            if rate != 16_000 or waveform.ndim != 1:
                raise RuntimeError("Frozen input audio must be mono 16 kHz")
            media_kwargs["audio"] = [waveform]
            media_kwargs["sampling_rate"] = rate
        if case.get("prompt"):
            content.append({"type": "text", "text": case["prompt"]})
        messages = []
        if case.get("system"):
            messages.append({"role": "system", "content": case["system"]})
        messages.append({"role": "user", "content": content})
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], return_tensors="pt", **media_kwargs)
        return {key: value for key, value in inputs.items() if isinstance(value, self.torch.Tensor)}

    def run_case(self, case, output_dir):
        import numpy as np
        import soundfile as sf

        self.capture = {}
        self.torch.manual_seed(20261111)
        inputs = self.inputs_for(case)
        input_digest = hashlib.sha256()
        input_shapes = {}
        for key, value in sorted(inputs.items()):
            input_shapes[key] = list(value.shape)
            input_digest.update(key.encode())
            input_digest.update(value.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        start = time.monotonic()
        max_new_tokens = case.get("max_new_tokens", 24)
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            generation_mode=case.get("generation_mode", "text"), **GENERATION_SETTINGS,
        )
        tokens = output.sequences[0, inputs["input_ids"].shape[1]:int(output.sequence_lengths[0])].tolist()
        text = self.processor.decode(tokens, skip_special_tokens=True).strip()
        result = {
            "id": case["id"], "modality": case["modality"], "split": case.get("split", "pilot"),
            "input_sha256": input_digest.hexdigest(), "input_shapes": input_shapes,
            "text": text, "tokens": tokens, "generated_slots": len(tokens),
            "max_new_tokens": max_new_tokens, "hit_token_limit": len(tokens) >= max_new_tokens,
            "generation_s": time.monotonic() - start,
        }
        if "gold_letter" in case:
            prediction = score_letter(text)
            result.update({"prediction": prediction, "gold": case["gold_letter"],
                           "correct": prediction == case["gold_letter"]})
            if "letter_logits" in self.capture:
                logits = self.capture["letter_logits"]
                gold_index = "ABCD".index(case["gold_letter"])
                alternatives = [index for index in range(4) if index != gold_index]
                competitor = max(alternatives, key=lambda index: logits[index])
                result.update({
                    "letter_logits": logits.tolist(), "competitor": "ABCD"[competitor],
                    "restricted_prediction": "ABCD"[int(np.argmax(logits))],
                    "gold_margin": float(logits[gold_index] - logits[competitor]),
                })
        elif "gold_text" in case:
            normalized = lambda value: re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()
            result["correct"] = normalized(text) == normalized(case["gold_text"])
            result["gold"] = case["gold_text"]
        audio_frames = int(output.audio_lengths[0]) if output.audio_lengths is not None else 0
        if audio_frames:
            codes = output.audio_codes[:1, :, :audio_frames]
            waveform = self.codec.decode(codes).audio_values[0].squeeze(0).detach().float().cpu().numpy()
            if not np.isfinite(waveform).all():
                raise RuntimeError("Codec emitted nonfinite waveform")
            path = Path(output_dir) / f"{case['id']}.wav"
            if path.exists():
                raise FileExistsError(path)
            sf.write(path, waveform, 24_000, subtype="FLOAT")
            result["speech"] = {
                "frames": audio_frames, "samples": len(waveform), "sample_rate": 24_000,
                "duration_s": len(waveform) / 24_000,
                "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                "peak_abs": float(np.max(np.abs(waveform))),
                "clipped_fraction": float(np.mean(np.abs(waveform) >= 1)),
                "codes_sha256": array_hash(codes.cpu().numpy()),
                "waveform_sha256": array_hash(waveform),
                "qualification": "Waveform integrity only; intelligibility requires an independent check.",
            }
        result["activation_keys"] = list(self.capture)
        for name, parameter in self.model.named_parameters():
            if parameter._version != self.parameter_versions[name]:
                raise RuntimeError(f"Read-only inference changed a parameter: {name}")
        return result, self.capture.copy()


def smoke(root, run_name="runtime-smoke"):
    import numpy as np
    from survey import verify_original

    root = Path(root)
    verify_original(root)
    if Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run_name must be one directory name")
    output_dir = root / "reports" / run_name
    output_dir.mkdir(exist_ok=False)
    runtime = OmniRuntime(root)
    (output_dir / "runtime-census.json").write_text(json.dumps(runtime.census(), indent=2, default=str))
    case = {"id": "pilot-addition", "modality": "text", "prompt": "What is 7 plus 8? Answer with only the number.",
            "gold_text": "15", "max_new_tokens": 16}
    observed, activations = runtime.run_case(case, output_dir)
    np.savez_compressed(output_dir / "activations.npz", **activations)
    for handle in runtime.handles:
        handle.remove()
    runtime.handles = []
    unhooked, _ = runtime.run_case(case, output_dir)
    if observed["tokens"] != unhooked["tokens"]:
        raise RuntimeError("Observation hooks changed generated tokens")
    result = {"with_hooks": observed, "without_hooks": unhooked,
              "hooks_preserve_token_ids": True, "source_hashes_verified_after": False}
    verify_original(root)
    result["source_hashes_verified_after"] = True
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print("RUNTIME_SMOKE_COMPLETE " + json.dumps(result), flush=True)


class RuntimeTests(unittest.TestCase):
    def test_answer_letter_forms(self):
        for text in ("B", "B.", "B, because the total is 7", "(B)"):
            self.assertEqual(score_letter(text), "B")

    def test_echo_is_not_an_answer(self):
        for text in ("Answer with A or B", "Because", "The options are A and B", ""):
            self.assertIsNone(score_letter(text))

    def test_head_shapes(self):
        import torch

        vector = torch.arange(8)
        self.assertTrue(torch.equal(last_vector(vector), vector))
        self.assertTrue(torch.equal(last_vector(vector.reshape(1, 1, 8)), vector))
        batched = torch.arange(24).reshape(2, 3, 4)
        self.assertTrue(torch.equal(last_vector(batched), torch.tensor([20, 21, 22, 23])))

    def test_video_clock_matches_decoded_frames(self):
        metadata = video_clock(8)
        self.assertEqual(metadata["frames_indices"], list(range(8)))
        self.assertEqual(metadata["total_num_frames"], 8)
        self.assertEqual(metadata["duration"], 4.0)
        self.assertEqual(metadata["fps"], 2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-name", default="runtime-smoke")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["runtime.py"], verbosity=2)
    elif args.root:
        smoke(args.root, args.run_name)
    else:
        parser.error("--root or --self-test required")