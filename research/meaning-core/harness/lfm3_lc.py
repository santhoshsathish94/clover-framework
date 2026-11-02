"""Run a released Learning Concepts script on LFM3-Omni-2B S3 (Clover's C1), with explicit adaptations and no released
file edited. LFM3's custom code is written for transformers 5.x, so this runs on /venv/main (5.22).

Adaptations (each one forced by an observation, not by preference):
 1. BOS. LFM3's tokenizer adds no BOS (its chat template writes <|startoftext|> itself). Without BOS the model copies
    earlier words instead of predicting ("The doctor examined the" -> " doctor" 0.99; with BOS " patient" 0.69):
    results/lfm3_probe_nobos.json vs lfm3_probe_bos.json. A post-processor adds BOS whenever add_special_tokens=True
    (the released code's default for sequences) and never for add_special_tokens=False (its word/synonym encodes).
 2. trust_remote_code=True on every model load (eval_mteb.py loads without it).
 3. transformers 5.x: TrainingArguments(warmup_ratio=r) becomes warmup_steps=r (5.x reads a float < 1 as a ratio),
    patched on the released train module's own name.
 4. embedding_synonyms.py builds each candidate's contextual state from a prefix slice of the full-sequence KV cache
    via DynamicCache.from_legacy_cache. LFM3's cache also holds conv states (a 6-position window), so that slice cannot
    be built. Instead: one prefix forward with a cache, the cache repeated across the candidates, one batched 1-token
    step. Equal to per-candidate full forwards up to bf16 rounding (cos >= 0.99984; near-tied candidates can swap):
    harness/lfm3_cache_test.py.
 5. No 4-bit: LFM3 runs in bf16 (the released default is QLoRA nf4 with fp16 compute). LoRA r/alpha/targets
    (q/k/v/o_proj), lr, schedule, epochs, losses: the released values.
 Seed: LC_SEED (default 42), as harness/run_train_seed.py.

Usage (from the learning-concepts folder):
  python lfm3_lc.py get_content_words|embedding_synonyms|train|eval_mteb <the script's own arguments>"""
import dataclasses
import os
import runpy
import sys

import torch
import transformers
from tokenizers import processors

LC = os.getcwd()
sys.path.insert(0, LC)

_auto_tok = transformers.AutoTokenizer.from_pretrained


def _tokenizer_with_bos(*args, **kwargs):
    kwargs.setdefault("trust_remote_code", True)
    tok = _auto_tok(*args, **kwargs)
    bos = tok.bos_token
    bos_id = tok.convert_tokens_to_ids(bos)
    tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A", pair=f"{bos} $A $B:1", special_tokens=[(bos, bos_id)])
    assert tok("a b").input_ids[0] == bos_id and bos_id not in tok.encode(" a", add_special_tokens=False)
    return tok


transformers.AutoTokenizer.from_pretrained = _tokenizer_with_bos

_auto_model = transformers.AutoModelForCausalLM.from_pretrained


def _model_lfm3(*args, **kwargs):
    kwargs["trust_remote_code"] = True
    if kwargs.get("quantization_config") is not None:
        raise RuntimeError("4-bit requested; the LFM3 runs are bf16 (set by this wrapper's config)")
    kwargs.pop("quantization_config", None)
    dtype = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", None) or torch.bfloat16
    kwargs["dtype"] = dtype
    return _auto_model(*args, **kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _model_lfm3

_training_args = transformers.TrainingArguments


def _training_arguments(*args, **kwargs):
    if "warmup_ratio" in kwargs:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
    return _training_args(*args, **kwargs)


import conceptlib.config as cfg  # noqa: E402  (released)

_config = cfg.TrainingConfig
cfg.TrainingConfig = lambda: dataclasses.replace(_config(), use_4bit=False, seed=int(os.environ.get("LC_SEED", 42)))


def repeat_cache(cache, n):
    """Repeat a filled LFM3 cache n times along the batch: attention layers with their own method, conv layers
    (LinearAttentionLayer has none in transformers 5.22) by repeating their state tensors."""
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
            continue
        for states in (layer.conv_states, layer.recurrent_states):
            for k, v in states.items():
                if v is not None:
                    states[k] = v.repeat_interleave(n, dim=0)


def run_embedding_synonyms(argv):
    sys.path.insert(0, os.path.join(LC, "data"))
    import embedding_synonyms as es  # released module; its __main__ block is replicated below

    analyze = es.EmbeddingSynonymExtractor.analyze_sequence

    def analyze_sequence(self, row, save_path, topk_save_path):
        self._lfm3_ids = torch.tensor([row["input_ids"]], device=self.model.device, dtype=torch.long)
        return analyze(self, row, save_path, topk_save_path)

    def candidate_contextual_reps(self, past_kv, idx, cand_ids):
        n = cand_ids.shape[0]
        dev = cand_ids.device
        with torch.inference_mode():
            cache = self.model(input_ids=self._lfm3_ids[:, :idx], use_cache=True).past_key_values
            repeat_cache(cache, n)
            out = self.model(input_ids=cand_ids.unsqueeze(-1),
                             attention_mask=torch.ones(n, idx + 1, device=dev, dtype=torch.long),
                             position_ids=torch.full((n, 1), idx, device=dev, dtype=torch.long),
                             past_key_values=cache, output_hidden_states=True, use_cache=True)
        return out.hidden_states[-1][:, 0, :].float()

    es.EmbeddingSynonymExtractor.analyze_sequence = analyze_sequence
    es.EmbeddingSynonymExtractor._candidate_contextual_reps = candidate_contextual_reps
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=str, default="c4", nargs="?")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    a = p.parse_args(argv)
    es.run(a.dataset, a.start, a.end, model=a.model)


if __name__ == "__main__":
    script, rest = sys.argv[1], sys.argv[2:]
    if script == "embedding_synonyms":
        run_embedding_synonyms(rest)
    elif script == "train":
        import train  # released
        # train.py's `from transformers import (..., TrainingArguments, Trainer, ...)` returns the original class even
        # after transformers.TrainingArguments is replaced (lazy module; reproduced), so patch train's own name
        train.TrainingArguments = _training_arguments
        sys.argv = ["train.py"] + rest
        train.main()
    elif script in ("get_content_words", "eval_mteb"):
        path = os.path.join(LC, "data" if script == "get_content_words" else "eval", script + ".py")
        sys.argv = [path] + rest
        runpy.run_path(path, run_name="__main__")
    else:
        raise SystemExit(f"unknown script {script}")
