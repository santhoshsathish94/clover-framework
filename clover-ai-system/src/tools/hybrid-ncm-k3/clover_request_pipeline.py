#!/usr/bin/env python3
"""
Clover Request Pipeline — SOLID, Factory, Adapter, and Validator Patterns
(`clover-ai-system/src/tools/hybrid-ncm-k3/clover_request_pipeline.py`).

Applies traditional software engineering standards (SOLID principles, Factory Pattern,
Strategy/Adapter Pattern, Defensive Null Checks, and Structured Exception Handling)
to request validation, model resolution, intent classification, window splitting,
and conversational completeness verification across `/api/chat` and `/v1/chat/completions`.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# 1. CloverModelBackendFactory (Factory Pattern — Single Responsibility & Open/Closed)
# ============================================================================

CANONICAL_MODELS = ("nemotron-3.5-super-vl", "lfm3-1.7b", "kimi-k3")
DEFAULT_CANONICAL_MODEL = "nemotron-3.5-super-vl"


class CloverModelBackendFactory:
    """
    Central Factory for resolving model identifiers/aliases and building OpenAI-compatible
    and Clover Studio capability descriptors across all three switchable Layer-1 models:
      - `nemotron-3.5-super-vl` (NVIDIA Nemotron 3.5 Super VL 120B-A12B, Default)
      - `lfm3-1.7b`             (Liquid AI LFM3-1.7B 24-Layer Tri-Hybrid)
      - `kimi-k3`               (MoonshotAI Kimi K3 93-Layer 2.78T MoE)
    """

    @staticmethod
    def resolve_canonical_model_key(
        requested_model: str | None,
        default_model: str = DEFAULT_CANONICAL_MODEL,
    ) -> str:
        """Null-safe resolution of any model ID or alias to its canonical backend key."""
        if not requested_model or not isinstance(requested_model, str):
            return default_model if default_model in CANONICAL_MODELS else DEFAULT_CANONICAL_MODEL

        raw = requested_model.strip().lower()
        if not raw:
            return default_model if default_model in CANONICAL_MODELS else DEFAULT_CANONICAL_MODEL

        if "nemotron" in raw or "super-vl" in raw or "120b" in raw:
            return "nemotron-3.5-super-vl"
        if "lfm3" in raw or "liquid" in raw or "1.7b" in raw:
            return "lfm3-1.7b"
        if "kimi" in raw or "k3" in raw:
            return "kimi-k3"
        if raw in CANONICAL_MODELS:
            return raw
        return default_model if default_model in CANONICAL_MODELS else DEFAULT_CANONICAL_MODEL

    @staticmethod
    def build_openai_models_list(
        created_timestamp: int,
        max_context_tokens: int,
        capabilities_catalog: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Builds the `/v1/models` OpenAI-compatible model list for VS Code and CLI clients."""
        cat = capabilities_catalog if isinstance(capabilities_catalog, dict) else {}
        shared_caps = {"toolCalling": True, "vision": True, "streaming": True}
        return [
            {
                "id": "nemotron-3.5-super-vl",
                "name": "Clover — Nemotron 3.5 Super VL (120B-A12B)",
                "object": "model",
                "created": int(created_timestamp),
                "owned_by": "clover-framework",
                "context_length": int(max_context_tokens),
                "max_input_tokens": 1032576,
                "max_output_tokens": 16000,
                "capabilities": cat.get("nemotron-3.5-super-vl", shared_caps),
            },
            {
                "id": "lfm3-1.7b",
                "name": "Clover — Liquid LFM3-1.7B (24L Tri-Hybrid)",
                "object": "model",
                "created": int(created_timestamp),
                "owned_by": "clover-framework",
                "context_length": int(max_context_tokens),
                "max_input_tokens": 1032576,
                "max_output_tokens": 16000,
                "capabilities": cat.get("lfm3-1.7b", shared_caps),
            },
            {
                "id": "clover-kimi-k3",
                "name": "Clover — Kimi K3 (2.78T MoE)",
                "object": "model",
                "created": int(created_timestamp),
                "owned_by": "clover-framework",
                "context_length": int(max_context_tokens),
                "max_input_tokens": 1032576,
                "max_output_tokens": 16000,
                "capabilities": cat.get("kimi-k3", shared_caps),
            },
            {
                "id": "kimi-k3",
                "name": "Clover — Kimi K3 (2.78T MoE)",
                "object": "model",
                "created": int(created_timestamp),
                "owned_by": "clover-framework",
                "context_length": int(max_context_tokens),
                "max_input_tokens": 1032576,
                "max_output_tokens": 16000,
                "capabilities": cat.get("kimi-k3", shared_caps),
            },
        ]


# ============================================================================
# 2. CloverRequestValidator (Validator Pattern — Null Safety & Input Sanitization)
# ============================================================================

@dataclass
class ValidatedChatRequest:
    """Immutable, null-safe validated chat request object."""
    session_id: str
    prompt: str
    model_key: str
    output_type: str
    cands: int
    remember: bool
    image_base64: str | None = None
    non_image_attachments: list[dict[str, Any]] = field(default_factory=list)
    client_history: list[dict[str, Any]] = field(default_factory=list)
    custom_max_context_tokens: int | None = None


class CloverRequestValidator:
    """
    Validates, sanitizes, and normalizes incoming HTTP JSON payloads for `/api/chat`
    and `/v1/chat/completions` with comprehensive null checks and exception safety.
    """

    _QUESTION_OR_CONVERSATIONAL_RE = re.compile(
        r"^(?:what|why|how|who|when|where|which|can\s+you|could\s+you|would\s+you|should\s+i|"
        r"do\s+you|did\s+you|did\s+we|is\s+there|are\s+there|is\s+it|explain|describe|tell\s+me|"
        r"compare|summarize|analyze|help\s+me|solve|calculate|write\s+a\s+(?:function|program|script|paper)|"
        r"def\s+[a-zA-Z_]|pub\s+fn\s+[a-zA-Z_]|hi\b|hello\b|hey\b|thanks\b|thank\s+you\b)",
        flags=re.IGNORECASE,
    )

    _EXPLICIT_MEDIA_KEYWORD_RE = re.compile(
        r"\b(?:image|picture|photo|illustration|artwork|drawing|wallpaper|poster|icon|logo|"
        r"draw|paint|sketch|video|animation|movie|motion\s+clip|gif|mp4|webm|"
        r"audio|voice|speech|sound|soundscape|chime|recording|mp3|wav|flac|ogg)\b",
        flags=re.IGNORECASE,
    )

    @classmethod
    def resolve_effective_output_type(cls, requested_output_type: Any, prompt: str) -> str:
        """
        Prevents sticky UI media tabs (`image`/`video`/`audio`) from hijacking conversational
        questions or code/math queries that contain no media intent.
        """
        raw_out = str(requested_output_type or "text").strip().lower()
        if raw_out not in ("text", "image", "video", "audio"):
            return "text"
        if raw_out == "text":
            return "text"

        clean_p = (prompt or "").strip()
        if not clean_p:
            return raw_out

        # If the prompt explicitly mentions a media keyword, honor the selected tab
        if cls._EXPLICIT_MEDIA_KEYWORD_RE.search(clean_p):
            return raw_out

        # If the prompt ends with '?' or starts with a conversational/code question prefix
        # and has no media keyword, route as 'text' so questions are answered as text!
        if clean_p.endswith("?") or cls._QUESTION_OR_CONVERSATIONAL_RE.search(clean_p):
            return "text"

        # If the prompt is a long multi-clause analytical topic (>= 10 words) without media keywords
        # such as "A deeper dive into the quantum realm and looking at the cosmos through the quantum perspective",
        # treat as 'text' unless it starts with 'generate/create/make/render/draw/animate'
        if len(clean_p.split()) >= 10 and not re.match(
            r"^(?:please\s+)?(?:generate|create|make|render|draw|paint|animate|synthesize|show)\b",
            clean_p,
            flags=re.IGNORECASE,
        ):
            return "text"

        return raw_out

    @classmethod
    def validate_chat_payload(
        cls,
        payload: Any,
        default_cands: int = 1,
        max_session_context_tokens: int = 1048576,
    ) -> ValidatedChatRequest:
        """Validates `/api/chat` JSON payload with defensive null checks on every field."""
        if not isinstance(payload, dict):
            payload = {}

        raw_sid = payload.get("session_id")
        session_id = str(raw_sid).strip()[:96] if raw_sid is not None and str(raw_sid).strip() else "default"

        raw_prompt = payload.get("prompt")
        prompt = str(raw_prompt).strip() if raw_prompt is not None else ""

        raw_img = payload.get("image_base64")
        image_b64 = str(raw_img).strip() if isinstance(raw_img, str) and raw_img.strip() else None

        raw_atts = payload.get("attachments")
        attachments = raw_atts if isinstance(raw_atts, list) else []
        non_img_attachments: list[dict[str, Any]] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            m_type = str(att.get("type") or "").strip().lower()
            d_b64 = str(att.get("data_b64") or "").strip()
            if (
                not image_b64
                and d_b64
                and (
                    m_type.startswith("image/")
                    or (m_type.startswith("video/") and d_b64.startswith("data:image/"))
                )
            ):
                image_b64 = d_b64
                if m_type.startswith("video/"):
                    non_img_attachments.append(att)
            else:
                non_img_attachments.append(att)

        try:
            cands = max(1, min(16, int(payload.get("cands", default_cands))))
        except (ValueError, TypeError):
            cands = max(1, int(default_cands))

        remember_val = payload.get("remember", True)
        remember = bool(remember_val) if remember_val is not None else True

        model_key = CloverModelBackendFactory.resolve_canonical_model_key(
            payload.get("model"),
            default_model=DEFAULT_CANONICAL_MODEL,
        )
        output_type = cls.resolve_effective_output_type(payload.get("output_type"), prompt)

        raw_hist = payload.get("history") or payload.get("supplied_history")
        client_history: list[dict[str, Any]] = []
        if isinstance(raw_hist, list):
            for h_item in raw_hist[-24:]:
                if isinstance(h_item, dict):
                    u_txt = str(h_item.get("user") or "").strip()
                    r_txt = str(h_item.get("reply") or "").strip()
                    if u_txt or r_txt:
                        client_history.append({
                            "user": u_txt,
                            "reply": r_txt,
                            "plans": list(h_item.get("plans") or []) if isinstance(h_item.get("plans"), list) else [],
                        })

        custom_max_ctx: int | None = None
        if payload.get("max_context_tokens") is not None:
            try:
                custom_max_ctx = max(256, min(max_session_context_tokens, int(payload["max_context_tokens"])))
            except (ValueError, TypeError):
                custom_max_ctx = None

        return ValidatedChatRequest(
            session_id=session_id,
            prompt=prompt,
            model_key=model_key,
            output_type=output_type,
            cands=cands,
            remember=remember,
            image_base64=image_b64,
            non_image_attachments=non_img_attachments,
            client_history=client_history,
            custom_max_context_tokens=custom_max_ctx,
        )


# ============================================================================
# 3. CloverPlanWindowSplitter (Prevents `[:P_BPE]` Mid-Sentence Truncation)
# ============================================================================

class CloverPlanWindowSplitter:
    """
    Splits any planned response window whose token count in the active tokenizer exceeds
    `max_bpe` (`38 < P_BPE = 48`) along clause or word boundaries so `[W, P_BPE]` tensor
    packing never truncates the tail of a sentence.
    """

    @staticmethod
    def split_overlong_windows(plans: list[str] | None, tok: Any, max_bpe: int = 38) -> list[str]:
        if not plans or not isinstance(plans, list):
            return ["Hello! How can I help you today?"]
        if tok is None or not hasattr(tok, "encode"):
            return [str(p) for p in plans if p is not None and str(p).strip()]

        expanded: list[str] = []
        for raw_w in plans:
            if raw_w is None:
                continue
            raw_str = str(raw_w)
            if not raw_str.strip():
                continue
            has_leading_nl = raw_str.startswith("\n")
            cp = raw_str.lstrip("\n").strip()
            if cp.startswith("```") or ("####" in cp and cp.startswith("Final Answer:")):
                expanded.append(("\n" + cp) if has_leading_nl else cp)
                continue
            try:
                tok_ids = tok.encode(cp)
            except Exception:
                tok_ids = cp.split()
            if len(tok_ids) <= max_bpe:
                expanded.append(("\n" + cp) if has_leading_nl else cp)
                continue

            words = cp.split()
            chunks: list[str] = []
            cur_words: list[str] = []
            for w in words:
                cand = " ".join(cur_words + [w])
                try:
                    c_len = len(tok.encode(cand))
                except Exception:
                    c_len = len(cand.split())
                if cur_words and c_len > max_bpe:
                    chunks.append(" ".join(cur_words))
                    cur_words = [w]
                else:
                    cur_words.append(w)
            if cur_words:
                chunks.append(" ".join(cur_words))

            for c_idx, ch in enumerate(chunks):
                if c_idx == 0 and has_leading_nl:
                    expanded.append("\n" + ch)
                else:
                    expanded.append(ch)

        return expanded if expanded else ["Hello! How can I help you today?"]
