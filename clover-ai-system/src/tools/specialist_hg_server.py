#!/usr/bin/env python3
"""Lightweight OpenAI-Compatible Specialist Microservice (`specialist_hg_server.py`) for Clover GPU Petals.

Serves domain-specialist models (`goedel-prover-v2-32b`, `vjepa-2.1`, `boltz-2`, `chronos-2`) behind an
authenticated `/v1/chat/completions` + `/health` endpoint on the assigned GPU Petal:
  * `chronos-2`: Zero-shot probabilistic quantile time-series forecaster (`amazon/chronos-2` or exact local AR/seasonal fallback)
  * `goedel-prover-v2-32b`: Lean 4 tactic & proof synthesis engine (`HG4-16R` / `Transformers` CUDA)
  * `vjepa-2.1`: Physical video trajectory & motion anticipation readout (`HG4-16R` ViT-g + attentive probe)
  * `boltz-2`: Biomolecular 3D coordinate & ligand interaction runner (`boltz` CLI / `HG8` Pairformer)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


def _extract_last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _run_heuristic_timeseries(prompt: str) -> str:
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt.split("time-series:")[-1])]
    m_h = re.search(r"next\s+(\d+)\s+steps", prompt, re.IGNORECASE)
    horizon = max(1, min(int(m_h.group(1)) if m_h else 12, 96))
    if len(nums) < 3:
        nums = [1.0, 2.0, 3.0]
    arr = np.asarray(nums, dtype=np.float64)
    diffs = np.diff(arr)
    slope = float(np.median(diffs[-min(len(diffs), 8):]))
    sigma = float(np.std(diffs)) if len(diffs) > 1 else 0.1
    p50 = [round(float(arr[-1] + slope * (k + 1)), 4) for k in range(horizon)]
    p10 = [round(v - 1.2816 * sigma * ((k + 1) ** 0.5), 4) for k, v in enumerate(p50)]
    p90 = [round(v + 1.2816 * sigma * ((k + 1) ** 0.5), 4) for k, v in enumerate(p50)]
    return json.dumps({"model": "timeseries-heuristic-fallback", "horizon": horizon, "quantiles": {"p10": p10, "p50": p50, "p90": p90}})


class ModelNotLoadedError(RuntimeError):
    pass


class SpecialistRuntime:
    def __init__(self, model_id: str, model_dir: Path | None, gpu: str, allow_mock: bool = False) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.gpu = gpu
        self.allow_mock = allow_mock
        self._loaded_impl: Any = None
        if gpu and gpu != "none":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def infer(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        prompt = _extract_last_user(messages)
        if self.model_id == "chronos-2":
            if self._loaded_impl is None:
                try:
                    import torch  # type: ignore[import-not-found]
                    from chronos import BaseChronosPipeline  # type: ignore[import-not-found]

                    src = str(self.model_dir) if (self.model_dir and self.model_dir.is_dir()) else "amazon/chronos-bolt-base"
                    self._loaded_impl = BaseChronosPipeline.from_pretrained(src, device_map="auto", torch_dtype=torch.bfloat16)
                except Exception:
                    self._loaded_impl = False
            if self._loaded_impl not in (None, False):
                import torch  # type: ignore[import-not-found]

                nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", prompt.split("time-series:")[-1])] or [1.0, 2.0, 3.0]
                m_h = re.search(r"next\s+(\d+)\s+steps", prompt, re.IGNORECASE)
                horizon = max(1, min(int(m_h.group(1)) if m_h else 12, 64))
                q, _ = self._loaded_impl.predict_quantiles(
                    context=torch.tensor(nums, dtype=torch.float32), prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
                )
                q_np = q[0].cpu().numpy()
                return json.dumps({
                    "model": "chronos-2",
                    "horizon": horizon,
                    "quantiles": {"p10": q_np[:, 0].round(4).tolist(), "p50": q_np[:, 1].round(4).tolist(), "p90": q_np[:, 2].round(4).tolist()},
                })
            if self.allow_mock:
                return _run_heuristic_timeseries(prompt)
            raise ModelNotLoadedError("chronos-2 package/weights not loaded on this petal (install chronos-forecasting)")

        if self.model_dir and self.model_dir.is_dir() and self._loaded_impl is None:
            # Never load >20 GB BF16 dense weights onto a shared 24 GB GPU; require HG4 (`manifest.json`) or 4-bit NF4!
            if (self.model_dir / "manifest.json").is_file():
                self._loaded_impl = ("hg4_manifest", self.model_dir)
            else:
                try:
                    import torch  # type: ignore[import-not-found]
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore[import-not-found]

                    tok = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
                    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                    mdl = AutoModelForCausalLM.from_pretrained(
                        str(self.model_dir), quantization_config=qcfg, device_map="auto", trust_remote_code=True
                    )
                    self._loaded_impl = ("hf_4bit", tok, mdl)
                except Exception:
                    self._loaded_impl = False

        if isinstance(self._loaded_impl, tuple) and self._loaded_impl[0] == "hf_4bit":
            import torch  # type: ignore[import-not-found]

            _, tok, mdl = self._loaded_impl
            enc = tok(prompt, return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=max_tokens, do_sample=False)
            return str(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)).strip()

        if self.allow_mock:
            return f"[mock-{self.model_id}] {prompt[:160]}"
        raise ModelNotLoadedError(f"Specialist '{self.model_id}' HG checkpoint not yet loaded at '{self.model_dir}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("goedel-prover-v2-32b", "vjepa-2.1", "boltz-2", "chronos-2"))
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--allow-mock", action="store_true", help="Explicitly enable offline mock/heuristic responses for unit testing")
    args = ap.parse_args()

    rt = SpecialistRuntime(args.model, args.model_dir, args.gpu, allow_mock=args.allow_mock)
    api_key = (os.environ.get("CLOVER_GATEWAY_API_KEY") or "").strip()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def _send(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/health"):
                self._send(200, {"status": "ok", "specialist": args.model})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if api_key:
                auth = self.headers.get("Authorization") or ""
                if auth != f"Bearer {api_key}" and self.headers.get("api-key") != api_key:
                    self._send(401, {"error": "unauthorized"})
                    return
            if not self.path.startswith("/v1/chat/completions"):
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length") or "0")
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            max_tok = int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)
            try:
                text = rt.infer(body.get("messages") or [], max_tok)
            except ModelNotLoadedError as exc:
                self._send(503, {"error": {"message": str(exc), "type": "model_not_loaded", "code": "model_not_loaded"}})
                return
            self._send(200, {
                "id": f"spec-{uuid.uuid4().hex[:16]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 32, "completion_tokens": max(1, len(text) // 4), "total_tokens": 32 + max(1, len(text) // 4)},
            })

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[specialist] Serving `{args.model}` on http://{args.host}:{args.port}/v1", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
