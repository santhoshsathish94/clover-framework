#!/usr/bin/env python3
"""Verification suite for Clover Studio's 3-Tier Image Generation Gateway (`image_gen.py`)."""
import base64
import http.server
import json
import shutil
import struct
import sys
import tempfile
import threading
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parent / "clover"
if str(CLOVER) not in sys.path:
    sys.path.insert(0, str(CLOVER))

import image_gen


def check(cond, msg):
    if not cond:
        print("FAIL", msg)
        raise AssertionError(msg)
    print("PASS", msg)


def _make_rgba_png(w: int, h: int, rgba=(20, 120, 220, 255)) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgba) * w for _ in range(h))

    def _chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="clover_img_gw_"))
    try:
        media = tmp / "media"

        # 1. Missing generative models MUST raise ImageBackendUnavailableError (no silent canvas substitution!)
        for preset in ("sd15-lcm", "sd15", "auto", "gpt-image-1"):
            raised = False
            try:
                image_gen.ImageGen(tmp, preset).generate("a photorealistic temple beside a lotus pond", media)
            except image_gen.ImageBackendUnavailableError as exc:
                raised = exc.failure_layer == "backend_capability"
            check(raised and not media.exists(), f"{preset} raises ImageBackendUnavailableError without silent canvas PNG")

        # 2. Explicit `diagram` preset produces non-generative provenance
        diag = image_gen.ImageGen(tmp, "diagram").generate("system architecture diagram", media, size=512, transparent=True)
        check(diag["is_generative"] is False and diag["backend_tier"] == "procedural_diagram"
              and diag["transparent"] is True and (media / diag["file"]).is_file(),
              f"diagram preset sets honest non-generative provenance: {diag}")

        # 3. High-Quality Images API tier (generation + transparent + multipart reference edit)
        recorded: list[dict] = []
        png_1024 = _make_rgba_png(64, 64, (40, 180, 90, 0))

        class _ImgHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                ln = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(ln)
                recorded.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "ctype": self.headers.get("Content-Type", ""),
                    "body": body,
                })
                payload = json.dumps({"data": [{"b64_json": base64.b64encode(png_1024).decode("ascii")}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        httpd = http.server.HTTPServer(("127.0.0.1", 0), _ImgHandler)
        port = httpd.server_address[1]
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            cfg_dir = tmp / "config"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "image_provider.json").write_text(json.dumps({
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "test-local-key",
                "model": "gpt-image-1",
            }), encoding="utf-8")

            caps = image_gen.capabilities(tmp)
            check(caps["high_quality_api"]["available"] is True and caps["local_diffusion"]["sd15_lcm_available"] is False,
                  f"capabilities() reports honest tier availability: {caps}")

            r_gen = image_gen.ImageGen(tmp, "auto").generate(
                "a futuristic minimalist coffee machine product render", media, transparent=True
            )
            req0 = json.loads(recorded[-1]["body"].decode("utf-8"))
            check(recorded[-1]["path"] == "/v1/images/generations"
                  and req0.get("background") == "transparent"
                  and r_gen["backend_tier"] == "high_quality"
                  and r_gen["is_generative"] is True
                  and r_gen["width"] == 64,
                  f"auto routes to configured High-Quality API with transparent background: {r_gen}")

            ref_file = tmp / "ref_input.png"
            ref_file.write_bytes(_make_rgba_png(32, 32, (200, 50, 50, 255)))
            r_edit = image_gen.ImageGen(tmp, "gpt-image-1").edit(
                "add warm studio rim lighting without changing the subject", ref_file, media
            )
            check(recorded[-1]["path"] == "/v1/images/edits"
                  and "multipart/form-data" in recorded[-1]["ctype"]
                  and b"ref_input.png" in recorded[-1]["body"]
                  and r_edit["operation"] == "edit"
                  and r_edit["backend_tier"] == "high_quality",
                  f"reference-guided image edit sends multipart /v1/images/edits: {r_edit}")
        finally:
            httpd.shutdown()
            httpd.server_close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
