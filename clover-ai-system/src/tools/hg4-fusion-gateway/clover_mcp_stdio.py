#!/usr/bin/env python3
"""Zero-Dependency Stdio <-> Stateless HTTP MCP 2.0 Bridge (`clover_mcp_stdio.py`).

Lets any stdio-based MCP client (VS Code Copilot `mcp.json`, Claude Code, OpenCode, Cursor)
connect seamlessly to a local or remote Clover MCP 2.0 server (`http://127.0.0.1:8790/mcp`
or `https://cloverframework.com/mcp`) with zero third-party Python packages required.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    mcp_url = (os.environ.get("CLOVER_MCP_URL") or "http://127.0.0.1:8790/mcp").strip()
    api_key = (os.environ.get("CLOVER_GATEWAY_API_KEY") or "").strip()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            rpc = json.loads(line)
        except ValueError as exc:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Invalid JSON: {exc}"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        # JSON-RPC notifications (no `id`, e.g., `notifications/initialized`) require no response
        if isinstance(rpc, dict) and "id" not in rpc:
            continue

        headers = {"Content-Type": "application/json", "Mcp-Protocol-Version": "2026-07-28"}
        if isinstance(rpc, dict):
            if rpc.get("method"):
                headers["Mcp-Method"] = str(rpc["method"])
            params = rpc.get("params") if isinstance(rpc.get("params"), dict) else {}
            if params.get("name"):
                headers["Mcp-Name"] = str(params["name"])
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(mcp_url, data=json.dumps(rpc).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                body = resp.read().decode("utf-8").strip()
                if body:
                    # Normalize multi-line JSON onto one line as required by MCP stdio framing
                    parsed = json.loads(body)
                    sys.stdout.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
        except urllib.error.HTTPError as hexc:
            raw_err = hexc.read().decode("utf-8", errors="ignore").strip()
            req_id = rpc.get("id") if isinstance(rpc, dict) else None
            try:
                parsed_err = json.loads(raw_err)
                if isinstance(parsed_err, dict) and parsed_err.get("jsonrpc") == "2.0" and "id" in parsed_err:
                    out_rpc = parsed_err
                else:
                    err_msg = (
                        ((parsed_err.get("error") or {}).get("message") if isinstance(parsed_err.get("error"), dict) else parsed_err.get("error"))
                        if isinstance(parsed_err, dict)
                        else raw_err[:240]
                    )
                    out_rpc = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": f"HTTP {hexc.code}: {err_msg}"}}
                sys.stdout.write(json.dumps(out_rpc, ensure_ascii=False) + "\n")
            except ValueError:
                sys.stdout.write(
                    json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": f"HTTP {hexc.code}: {raw_err[:240]}"}}) + "\n"
                )
            sys.stdout.flush()
        except Exception as exc:
            req_id = rpc.get("id") if isinstance(rpc, dict) else None
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32001, "message": f"Clover MCP bridge error: {exc}"}}) + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
