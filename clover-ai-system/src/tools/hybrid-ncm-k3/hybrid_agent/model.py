"""HybridAgentModel: a causal-LM backbone paired with HybridLink.

Provides the anchor-token registration, HybridLink application, checkpoint save/load
and a loader that accepts either a local directory or a Hugging Face repo id.
Shared by HybridSFT, HybridRL and inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .constants import ANCHOR_TOKENS
from .hybridlink import HYBRIDLINK_CONFIG, HybridLink

HYBRIDLINK_WEIGHTS = "adapter.pt"
HYBRIDLINK_CONFIG_FILE = "adapter_config.json"


def resolve_model_dir(path_or_repo: str | os.PathLike, revision: str | None = None) -> Path:
    """Local directory as is; otherwise download the Hugging Face repo snapshot."""
    path = Path(path_or_repo)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(path_or_repo), revision=revision))


def add_anchor_tokens(tokenizer) -> int:
    """Register `<latent>` / `</latent>` as special tokens; returns how many were added."""
    known = set(getattr(tokenizer, "all_special_tokens", None) or [])
    missing = [tok for tok in ANCHOR_TOKENS if tok not in known]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return len(missing)


class HybridAgentModel(nn.Module):
    def __init__(self, backbone: nn.Module, hybridlink: HybridLink | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.get_input_embeddings().weight.shape[-1]
        self.hybridlink = hybridlink if hybridlink is not None else HybridLink(hidden_size)
        self.hybridlink.to(dtype=next(backbone.parameters()).dtype)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | os.PathLike,
        *,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        require_hybridlink: bool = False,
        revision: str | None = None,
    ) -> "HybridAgentModel":
        """Load the backbone and, when present, the HybridLink weights of a checkpoint."""
        local = resolve_model_dir(path_or_repo, revision)
        backbone = AutoModelForCausalLM.from_pretrained(
            str(local), dtype=dtype, trust_remote_code=trust_remote_code
        )
        model = cls(backbone)
        weights = local / HYBRIDLINK_WEIGHTS
        if weights.is_file():
            config_file = local / HYBRIDLINK_CONFIG_FILE
            if config_file.is_file():
                tag = json.loads(config_file.read_text(encoding="utf-8")).get("adapter_type")
                if tag != HYBRIDLINK_CONFIG["adapter_type"]:
                    raise ValueError(f"unsupported HybridLink config {tag!r} in {config_file}")
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
            model.hybridlink.load_state_dict(state, strict=True)
        elif require_hybridlink:
            raise FileNotFoundError(f"{HYBRIDLINK_WEIGHTS} not found in {local}")
        return model

    def run_hybridlink(
        self, hidden_states: torch.Tensor, output_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """R(h): next-step latent inputs from last-layer hidden states."""
        link_dtype = next(self.hybridlink.parameters()).dtype
        x = hidden_states if hidden_states.dtype == link_dtype else hidden_states.to(link_dtype)
        out = self.hybridlink(x)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def resize_to_tokenizer(self, tokenizer) -> None:
        """Match the embedding table to the tokenizer (needed once for a fresh base model)."""
        if self.backbone.get_input_embeddings().num_embeddings != len(tokenizer):
            self.backbone.resize_token_embeddings(len(tokenizer))

    def save_pretrained(self, output_dir: str | os.PathLike, tokenizer=None, state_dict=None) -> None:
        """Write the backbone (HF format), tokenizer and HybridLink weights into one directory."""
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
        link_sd = {k[len("hybridlink."):]: v for k, v in state_dict.items() if k.startswith("hybridlink.")}
        self.backbone.save_pretrained(output_dir, state_dict=backbone_sd)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(link_sd, os.path.join(output_dir, HYBRIDLINK_WEIGHTS))
        with open(os.path.join(output_dir, HYBRIDLINK_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump(HYBRIDLINK_CONFIG, f, indent=2)


def load_hybrid_agent(
    path_or_repo: str | os.PathLike,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
    require_hybridlink: bool = True,
    revision: str | None = None,
):
    """Return `(model, tokenizer)` ready for generation from a local dir or HF repo id."""
    local = resolve_model_dir(path_or_repo, revision)
    tokenizer = AutoTokenizer.from_pretrained(str(local), trust_remote_code=trust_remote_code)
    add_anchor_tokens(tokenizer)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = HybridAgentModel.from_pretrained(
        local, dtype=dtype, trust_remote_code=trust_remote_code,
        require_hybridlink=require_hybridlink, revision=revision,
    )
    model.resize_to_tokenizer(tokenizer)
    if device is not None:
        model.to(device)
    model.eval()
    return model, tokenizer
