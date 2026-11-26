"""OpenAI-compatible `/v1` + Copilot Fusion + Elephant-Goldfish gateway for the native C11 `HG4-16R` engines.

    python gateway.py                      # http://127.0.0.1:8790/v1  (native engines, mock fallback)
    python gateway.py --mode real          # native engines only
    python gateway.py --mode mock          # deterministic offline mock engines

Endpoints:
    GET  /health, /v1/health
    GET  /v1/models, /v1/models/{id}
    GET  /v1/telemetry
    POST /v1/chat/completions   (stream false/true, tools, tool_choice, stop, temperature, max_(completion_)tokens)
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends import build_backends  # noqa: E402
from clover_4part_agent import Clover4PartAgent  # noqa: E402
from clover_telemetry_ledger import CloverMetricsLedger  # noqa: E402
from clover_token_socket import StepHardwareMeter, TokenSocketServer  # noqa: E402
from cluster_scheduler_mcp2 import (  # noqa: E402
    HOP_CTX,
    MCP2_PROTOCOL_VERSION,
    ClusterBatchQueue,
    Mcp2MeshServer,
    PetalRegistry,
    ReplicaPoolBackend,
)
from fusion_router import Completion, FusionRouter, RouterError  # noqa: E402

DEFAULT_PORT = 8790
MAX_BODY = 64 * 1024 * 1024
VERSION = "2.0"


def _completion_json(c: Completion, cid: str, created: int) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": c.content if (c.content or not c.tool_calls) else None}
    if c.tool_calls:
        msg["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in c.tool_calls]
    return {
        "id": cid, "object": "chat.completion", "created": created, "model": c.model,
        "system_fingerprint": f"clover-hg4-fusion-{VERSION}",
        "choices": [{"index": 0, "message": msg, "finish_reason": c.finish_reason, "logprobs": None}],
        "usage": c.usage,
        "x_clover_telemetry": c.telemetry,
    }


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "CloverHG4FusionGateway/" + VERSION
    protocol_version = "HTTP/1.1"
    router: FusionRouter
    batch_queue: ClusterBatchQueue
    mcp2: Mcp2MeshServer
    petal_registry: PetalRegistry
    api_key: str | None = None
    quiet = False
    socket_timeout_s: float = float(os.environ.get("CLOVER_GATEWAY_SOCKET_TIMEOUT", "30.0"))

    def setup(self) -> None:
        super().setup()
        if self.socket_timeout_s and self.socket_timeout_s > 0:
            self.connection.settimeout(self.socket_timeout_s)

    # -- plumbing ----------------------------------------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        if not self.quiet:
            sys.stderr.write("[gateway] %s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization, Content-Type, OpenAI-Organization, X-Request-Id, X-Clover-Hop, Mcp-Method, Mcp-Name, Mcp-Priority, If-None-Match",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, obj: Any, status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
        data = b"" if obj is None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        if obj is not None:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for hk, hv in (extra_headers or {}).items():
            self.send_header(hk, hv)
        self._cors()
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _error(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        self._send_json({"error": {"message": message, "type": code, "code": code, "param": None}}, status)

    def _drain_and_error(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        try:
            cl = int(self.headers.get("Content-Length") or 0)
            if 0 < cl <= 65536:
                _ = self.rfile.read(cl)
            elif cl > 65536:
                self.close_connection = True
        except Exception:
            self.close_connection = True
        self._error(status, message, code)

    def _authorized(self) -> bool:
        if not self.api_key:
            return True
        key_b = self.api_key.encode("utf-8")
        auth = (self.headers.get("Authorization") or "").encode("utf-8")
        hdr_k = (self.headers.get("api-key") or "").encode("utf-8")
        return hmac.compare_digest(auth, b"Bearer " + key_b) or hmac.compare_digest(hdr_k, key_b)

    def _path(self) -> str:
        p = self.path.split("?", 1)[0].rstrip("/")
        return p or "/"

    # -- verbs -------------------------------------------------------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        p = self._path()
        if p in ("/health", "/v1/health"):
            self._send_json({
                "status": "ok", "version": VERSION,
                "backends": {k: {"kind": b.kind, "family": b.family} for k, b in self.router.backends.items()},
            })
            return
        if not self._authorized():
            self._error(401, "invalid API key", "authentication_error")
            return
        if p in ("/v1/models", "/models"):
            self._send_json({"object": "list", "data": self.router.list_models()})
        elif p.startswith("/v1/models/"):
            mid = p[len("/v1/models/"):]
            for m in self.router.list_models():
                if m["id"] == mid:
                    self._send_json(m)
                    return
            self._error(404, f"model '{mid}' not found", "model_not_found")
        elif p in ("/v1/telemetry", "/telemetry"):
            self._send_json(self.router.telemetry.snapshot())
        elif p in ("/v1/clover/metrics", "/clover/metrics"):
            qs = urllib.parse.parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            btag = (qs.get("benchmark_tag") or [""])[0].strip() or None
            reg_inst: dict[str, list[str]] = {}
            for k, b in self.router.backends.items():
                reg_inst[k] = list(getattr(b, "endpoints", None) or ["local"])
            self._send_json(
                self.metrics_ledger.snapshot(
                    registered_models=list(self.router.backends.keys()),
                    registered_tools=[t["name"] for t in self.mcp2._full_tools_list],
                    registered_instances=reg_inst,
                    benchmark_tag=btag,
                )
            )
        elif p in ("/v1/queue", "/queue"):
            pools = {
                k: b.snapshot() for k, b in self.router.backends.items() if isinstance(b, ReplicaPoolBackend)
            }
            self._send_json({"queue": self.batch_queue.snapshot(), "replica_pools": pools})
        elif p in ("/v1/cluster/petals", "/cluster/petals"):
            self._send_json(self.petal_registry.snapshot())
        elif p in ("/v1/hw", "/hw"):
            self._send_json(StepHardwareMeter().finish(1, 0, 0))
        elif p in ("/mcp", "/v1/mcp"):
            self._send_json({
                "protocol": "mcp",
                "protocolVersion": MCP2_PROTOCOL_VERSION,
                "stateless": True,
                "node": self.mcp2.node_id,
                "catalogEtag": self.mcp2.catalog_etag,
                "endpoints": {"jsonrpc": "/mcp", "batchCompletions": "/v1/chat/completions/batch", "queue": "/v1/queue"},
            }, extra_headers={"Mcp-Protocol-Version": MCP2_PROTOCOL_VERSION})
        else:
            self._error(404, f"unknown endpoint {p}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        p = self._path()
        if not self._authorized():
            self._drain_and_error(401, "invalid API key", "authentication_error")
            return
        if p in ("/v1/cache/reset", "/cache/reset"):
            self._send_json(self.router.reset_caches())
            return
        if p in ("/v1/clover/metrics/reset", "/clover/metrics/reset"):
            self._send_json(self.metrics_ledger.reset())
            return
        if p in ("/v1/cluster/register", "/cluster/register"):
            try:
                n_reg = int(self.headers.get("Content-Length") or 0)
                if n_reg > MAX_BODY:
                    self._drain_and_error(413, "oversized petal registration")
                    return
                reg_body = json.loads(self.rfile.read(n_reg).decode("utf-8")) if 0 < n_reg <= MAX_BODY else {}
                out_reg = self.petal_registry.register(reg_body)
                self._send_json(out_reg)
            except (ValueError, UnicodeDecodeError) as exc:
                self._error(400, f"invalid petal registration: {exc}")
            return
        if p in ("/mcp", "/v1/mcp"):
            try:
                n_mcp = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n_mcp = 0
            if n_mcp <= 0 or n_mcp > MAX_BODY:
                self._drain_and_error(400 if n_mcp <= 0 else 413, "missing or oversized MCP body")
                return
            t_mcp0 = time.perf_counter()
            raw_mcp = self.rfile.read(n_mcp)
            status, extra_hdrs, resp_obj = self.mcp2.handle_http(raw_mcp, dict(self.headers.items()))
            try:
                req_mcp = json.loads(raw_mcp.decode("utf-8"))
                if isinstance(req_mcp, dict) and req_mcp.get("method") == "tools/call":
                    t_nm = str((req_mcp.get("params") or {}).get("name") or self.headers.get("Mcp-Name") or "").strip()
                    if t_nm:
                        res_m = (resp_obj or {}).get("result") or {}
                        ok_m = status == 200 and not res_m.get("isError") and "error" not in (resp_obj or {})
                        self.metrics_ledger.record_tool(t_nm, ok_m, (time.perf_counter() - t_mcp0) * 1000.0)
            except Exception:
                pass
            self._send_json(resp_obj, status=status, extra_headers=extra_hdrs)
            return
        if p in ("/v1/chat/completions/batch", "/chat/completions/batch"):
            try:
                n_b = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n_b = 0
            if n_b <= 0 or n_b > MAX_BODY:
                self._drain_and_error(400 if n_b <= 0 else 413, "missing or oversized batch body")
                return
            try:
                bbody = json.loads(self.rfile.read(n_b).decode("utf-8"))
                reqs = bbody.get("requests") if isinstance(bbody, dict) else bbody
                if not isinstance(reqs, list):
                    raise ValueError("expected `{\"requests\": [...]}` or a JSON list")
                prio = int(bbody.get("priority", 2)) if isinstance(bbody, dict) else 2
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                self._error(400, f"invalid batch JSON: {exc}")
                return
            is_hop = bool(self.headers.get("X-Clover-Hop"))
            batch_out = self.batch_queue.submit_batch(reqs, priority=prio, local_only_hop=is_hop)
            now_ts = int(time.time())
            for item in batch_out["results"]:
                if "completion" in item:
                    item["completion"] = _completion_json(item["completion"], "chatcmpl-" + uuid.uuid4().hex[:24], now_ts)
            self._send_json(batch_out)
            return
        if p in ("/v1/clover/solve", "/clover/solve"):
            try:
                n_s = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n_s = 0
            if n_s <= 0 or n_s > MAX_BODY:
                self._drain_and_error(400 if n_s <= 0 else 413, "missing or oversized solve body")
                return
            try:
                sbody = json.loads(self.rfile.read(n_s).decode("utf-8"))
                if not isinstance(sbody, dict):
                    raise ValueError("body must be a JSON object")
                res_solve = self.agent4.solve(sbody)
                self._send_json(res_solve)
            except (ValueError, UnicodeDecodeError) as exc:
                self._error(400, f"invalid solve request: {exc}")
            except RouterError as exc:
                self._error(exc.status, str(exc), exc.code)
            except Exception as exc:
                self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            return
        if p not in ("/v1/chat/completions", "/chat/completions"):
            self._drain_and_error(404, f"unknown endpoint {p}", "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._drain_and_error(400 if n <= 0 else 413, "missing or oversized request body")
            return
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            prio_chk = int(body.get("x_clover_priority", 1))
        except (ValueError, TypeError, UnicodeDecodeError) as exc:
            self._error(400, f"invalid JSON body: {exc}")
            return
        body["x_clover_priority"] = prio_chk
        if body.get("n") not in (None, 1):
            self._error(400, "only n=1 is supported")
            return
        if body.get("stream"):
            self._stream(body)
        else:
            self._complete(body)

    def _complete(self, body: dict[str, Any]) -> None:
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        prio = int(body.get("x_clover_priority", 1))
        is_hop = bool(self.headers.get("X-Clover-Hop"))
        t_c0 = time.perf_counter()
        try:
            c, q_meta = self.batch_queue.submit(body, priority=prio, local_only_hop=is_hop)
            c.telemetry["queue"] = q_meta
            wall_ms = (time.perf_counter() - t_c0) * 1000.0
            served_m = str(c.telemetry.get("served_by") or c.model)
            inst_id = f"{served_m}@{c.telemetry.get('replica_url') or 'local'}"
            tok_s = self.metrics_ledger.record_completion(
                model=served_m,
                prompt_tokens=int(c.usage.get("prompt_tokens") or 0),
                completion_tokens=int(c.usage.get("completion_tokens") or 0),
                wall_ms=wall_ms,
                instance_id=inst_id,
                benchmark_tag=str(body.get("x_clover_benchmark_tag") or ""),
            )
            c.telemetry["decode_tok_per_s"] = tok_s
            c.telemetry["wall_ms"] = round(wall_ms, 2)
        except RouterError as exc:
            self._error(exc.status, str(exc), exc.code)
            return
        except RuntimeError as exc:
            code = 503 if "X-Clover-Hop" in str(exc) else 500
            self._error(code, str(exc), "service_unavailable" if code == 503 else "server_error")
            return
        except Exception as exc:  # engine failure
            self._error(500, f"{type(exc).__name__}: {exc}", "server_error")
            return
        self._send_json(_completion_json(c, cid, int(time.time())))

    def _stream(self, body: dict[str, Any]) -> None:
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = str(body.get("model") or "")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        state = {"headers": False, "role": False, "broken": False}

        def write(obj: Any) -> None:
            if state["broken"]:
                raise ConnectionAbortedError("client disconnected")
            payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
            try:
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError as exc:
                state["broken"] = True
                raise ConnectionAbortedError("client disconnected") from exc

        def start() -> None:
            if state["headers"]:
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()
            state["headers"] = True

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "system_fingerprint": f"clover-hg4-fusion-{VERSION}",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
            d.update(extra)
            return d

        def role_once() -> None:
            start()
            if not state["role"]:
                state["role"] = True
                write(chunk({"role": "assistant", "content": ""}))

        def on_delta(text: str) -> None:
            role_once()
            if text:
                write(chunk({"content": text}))

        HOP_CTX.local_only = bool(self.headers.get("X-Clover-Hop"))
        try:
            c = self.router.complete(body, on_delta=on_delta)
        except ConnectionAbortedError:
            HOP_CTX.local_only = False
            self.close_connection = True
            return
        except RouterError as exc:
            HOP_CTX.local_only = False
            if not state["headers"]:
                self._error(exc.status, str(exc), exc.code)
                return
            write({"error": {"message": str(exc), "type": exc.code}})
            write("[DONE]")
            self.close_connection = True
            return
        except Exception as exc:
            HOP_CTX.local_only = False
            if not state["headers"]:
                code = 503 if "X-Clover-Hop" in str(exc) else 500
                self._error(code, f"{type(exc).__name__}: {exc}", "server_error")
                return
            try:
                write({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}})
                write("[DONE]")
            except ConnectionAbortedError:
                pass
            self.close_connection = True
            return
        HOP_CTX.local_only = False
        try:
            role_once()
            for i, tc in enumerate(c.tool_calls):
                write(chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                             "function": {"name": tc["function"]["name"], "arguments": ""}}]}))
                args = tc["function"]["arguments"]
                for k in range(0, len(args), 256):
                    write(chunk({"tool_calls": [{"index": i, "function": {"arguments": args[k:k + 256]}}]}))
            write(chunk({}, c.finish_reason, x_clover_telemetry=c.telemetry))
            if include_usage:
                write({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [], "usage": c.usage})
            write("[DONE]")
        except ConnectionAbortedError:
            pass
        self.close_connection = True


def create_server(router: FusionRouter, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                  api_key: str | None = None, quiet: bool = False,
                  batch_window_ms: float | None = None) -> ThreadingHTTPServer:
    eff_key = (api_key if api_key is not None else (os.environ.get("CLOVER_GATEWAY_API_KEY") or "")).strip() or None
    if host not in ("127.0.0.1", "localhost", "::1") and not eff_key and os.environ.get("CLOVER_ALLOW_UNAUTH_PUBLIC_GATEWAY") != "1":
        raise ValueError(
            f"Refusing to bind non-loopback host {host!r} without --api-key or CLOVER_GATEWAY_API_KEY "
            "(set CLOVER_ALLOW_UNAUTH_PUBLIC_GATEWAY=1 only on an isolated private VLAN)."
        )
    bqueue = ClusterBatchQueue(router.complete, batch_window_ms=batch_window_ms)
    mcp2 = Mcp2MeshServer(router, bqueue)
    preg = PetalRegistry(router, mcp2)
    mledger = CloverMetricsLedger()
    agent4 = Clover4PartAgent(bqueue, mcp2, metrics_ledger=mledger)
    handler = type(
        "BoundGatewayHandler",
        (GatewayHandler,),
        {
            "router": router,
            "batch_queue": bqueue,
            "mcp2": mcp2,
            "petal_registry": preg,
            "metrics_ledger": mledger,
            "agent4": agent4,
            "api_key": eff_key,
            "quiet": quiet,
        },
    )
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    orig_close = srv.server_close

    def _close() -> None:
        mcp2.close()
        bqueue.close()
        orig_close()

    srv.server_close = _close  # type: ignore[method-assign]
    return srv


def serve_in_thread(router: FusionRouter, host: str = "127.0.0.1", port: int = 0,
                    quiet: bool = True) -> tuple[ThreadingHTTPServer, threading.Thread]:
    srv = create_server(router, host, port, quiet=quiet)
    th = threading.Thread(target=srv.serve_forever, name="hg4-fusion-gateway", daemon=True)
    th.start()
    return srv, th


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Clover HG4 Fusion Gateway (OpenAI /v1 + Copilot Fusion + EGM)")
    ap.add_argument("--host", default=os.environ.get("CLOVER_GATEWAY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLOVER_GATEWAY_PORT", DEFAULT_PORT)))
    ap.add_argument(
        "--token-socket-port",
        type=int,
        default=int(os.environ.get("CLOVER_TOKEN_SOCKET_PORT", "0")),
        help="Optional dedicated binary TCP_NODELAY token socket port (e.g. 10100 for tsock:// peer hops)",
    )
    ap.add_argument("--mode", choices=("auto", "real", "mock"), default=os.environ.get("CLOVER_GATEWAY_MODE", "auto"))
    ap.add_argument("--api-key", default=os.environ.get("CLOVER_GATEWAY_API_KEY") or None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    backends = build_backends(a.mode, log=lambda s: print(s, flush=True))
    if not backends:
        print("no backends available", file=sys.stderr)
        return 2
    router = FusionRouter(backends)
    tsock_srv = None
    if a.token_socket_port > 0:
        tsock_srv = TokenSocketServer(router, host=a.host, port=a.token_socket_port, secret=a.api_key)
        print(f"[gateway] Binary TokenSocket listening on tsock://{a.host}:{tsock_srv.bound_port}", flush=True)
    srv = create_server(router, a.host, a.port, api_key=a.api_key, quiet=a.quiet)
    print(f"[gateway] Clover HG4 Fusion Gateway {VERSION} on http://{a.host}:{srv.server_address[1]}/v1 "
          f"models={[m['id'] for m in router.list_models()]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if tsock_srv is not None:
            tsock_srv.close()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
