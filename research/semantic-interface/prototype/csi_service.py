"""CSI as a local side service for Clover's runtime (Experiment 1f decision: the routing use and the claim check, off by
default). Loopback only. POST /analyze with {"text": "..."} (Content-Type application/json) -> the CSI analysis
(ARCHITECTURE.md section 2); GET /health -> {"ok": true, "csi_version", "parser"}. The runtime calls it with a short
timeout and behaves as without CSI when it does not answer (semantics.csi_reader).
Usage: CSI_PARSER=en_core_web_trf CSI_MORPH=<morph-verb-v1.1> CSI_DICT=<english-wordnet-2025.xml.gz> \
       python csi_service.py [--port 8790]"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi import CSI_VERSION, analyze  # noqa: E402
from csi.analyze import PARSER  # noqa: E402

MAX_BODY = 64 * 1024
MAX_TEXT = 4000  # characters; a longer message is not read (the runtime then behaves as without CSI)
ONE_AT_A_TIME = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local(self):
        """Only requests addressed to this machine by name (no DNS rebinding through a browser)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "local only"})
        if self.path == "/health":
            return self._send(200, {"ok": True, "csi_version": CSI_VERSION, "parser": PARSER})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._local():
            return self._send(403, {"error": "local only"})
        if self.path != "/analyze":
            return self._send(404, {"error": "not found"})
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return self._send(415, {"error": "application/json only"})  # a web page cannot send this without asking
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._send(413 if n > MAX_BODY else 400, {"error": "body size"})
        try:
            text = json.loads(self.rfile.read(n).decode("utf-8")).get("text")
        except (ValueError, AttributeError, UnicodeDecodeError):
            return self._send(400, {"error": "not JSON"})
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            return self._send(400, {"error": "text"})
        try:
            with ONE_AT_A_TIME:
                out = analyze(text)
        except Exception:  # noqa: BLE001 - the caller treats any failure as "not read"
            return self._send(500, {"error": "analysis failed"})
        self._send(200, out)

    def log_message(self, fmt, *args):  # messages are private: nothing is logged
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    analyze("Please search the web.")  # load the parser and the lemmatiser before the first request
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"CSI {CSI_VERSION} ({PARSER}) on http://127.0.0.1:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
