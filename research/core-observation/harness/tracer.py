"""Records what each Clover layer receives and hands on, without changing it.

wrap() replaces a function or method with one that calls the original with the same arguments and returns the same
result (exceptions pass through unchanged; generators are passed through item by item). Each call writes JSON lines:
{"layer", "fn", "phase": call|return|error|stream_end, "args"/"result", "dur_ms", "turn", "thread", ...}.
A turn is one POST to a chat endpoint; work on other threads is recorded with that thread's name.
"""
import dataclasses
import functools
import inspect
import json
import os
import threading
import time

_lock = threading.Lock()
_out = None
_t0 = time.perf_counter()
_seq = 0
_state = {"turn": 0, "path": None}
decoders = {}  # "ids" -> callable(list[int]) -> str, set by a stack's launcher once its tokenizer exists


def open_trace(path):
    global _out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _out = open(path, "a", encoding="utf-8", buffering=1)


def emit(rec):
    global _seq
    with _lock:
        _seq += 1
        rec = dict(rec, seq=_seq, t_ms=round((time.perf_counter() - _t0) * 1000, 1), wall=round(time.time(), 3),
                   turn=_state["turn"], thread=threading.current_thread().name)
        if _out is not None:
            _out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def brief(x, depth=0, limit=700):
    """A bounded, JSON-ready picture of a value (strings cut with head and tail kept, arrays as shape and norm)."""
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return x if len(x) <= limit else {"len": len(x), "head": x[:limit * 2 // 3], "tail": x[-limit // 3:]}
    if isinstance(x, (bytes, bytearray, memoryview)):
        return {"bytes": len(x)}
    tn = type(x).__name__
    if tn in ("ndarray", "memmap"):
        d = {"array": list(x.shape), "dtype": str(x.dtype)}
        if 0 < x.size <= 10_000_000 and x.dtype.kind in "fiu":
            import numpy as np
            d["norm"] = round(float(np.linalg.norm(np.asarray(x, dtype=np.float64).ravel())), 4)
        return d
    if tn in ("Tensor", "Parameter"):
        return {"tensor": list(x.shape), "dtype": str(x.dtype)}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: brief(getattr(x, f.name), depth + 1, limit) for f in dataclasses.fields(x) if f.repr}
    if isinstance(x, dict):
        if depth >= 4:
            return {"keys": [str(k) for k in list(x)[:30]]}
        return {str(k): brief(v, depth + 1, limit) for k, v in list(x.items())[:60]}
    if isinstance(x, (list, tuple, set, frozenset)):
        seq = list(x)
        if seq and len(seq) > 16 and all(isinstance(i, int) and not isinstance(i, bool) for i in seq[:64]):
            return {"ints": len(seq), "head": seq[:12], "tail": seq[-6:]}
        if depth >= 4:
            return {"len": len(seq)}
        out = [brief(i, depth + 1, limit) for i in seq[:16]]
        return out + [{"more": len(seq) - 16}] if len(seq) > 16 else out
    if inspect.isgenerator(x):
        return {"generator": tn}
    if callable(x):
        return {"callable": getattr(x, "__qualname__", tn)}
    return {"type": tn, "repr": repr(x)[:240]}


def bound_args(fn, a, kw):
    try:
        ba = inspect.signature(fn).bind_partial(*a, **kw)
        items = [(k, v) for k, v in ba.arguments.items() if k not in ("self", "cls")]
    except (TypeError, ValueError):
        items = [(f"arg{i}", v) for i, v in enumerate(a)] + list(kw.items())
    return items


def _passthrough(gen, layer, label, t_call):
    n, t_first, text, tool_names, last = 0, None, [], [], None
    try:
        for item in gen:
            n += 1
            if t_first is None:
                t_first = time.perf_counter()
            last = item
            if isinstance(item, dict):
                for ch in item.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        text.append(d["content"])
                    for tc in d.get("tool_calls") or []:
                        name = (tc.get("function") or {}).get("name")
                        if name:
                            tool_names.append(name)
            yield item
    finally:
        gen.close()
        now = time.perf_counter()
        emit({"layer": layer, "fn": label, "phase": "stream_end", "items": n,
              "first_item_ms": round((t_first - t_call) * 1000, 1) if t_first else None,
              "dur_ms": round((now - t_call) * 1000, 1), "text": brief("".join(text), limit=6000),
              "tool_call_names": tool_names,
              "timings": brief(last.get("timings")) if isinstance(last, dict) else None})


def wrap(owner, name, layer, args=None, result=None, after=None, label=None):
    """Wrap owner.name (a module function, plain method, staticmethod or classmethod) in place.
    args(items) / result(value) / after(items, value) customise what is recorded; they never change values."""
    raw = inspect.getattr_static(owner, name)
    kind = "static" if isinstance(raw, staticmethod) else "class" if isinstance(raw, classmethod) else "plain"
    fn = raw.__func__ if kind != "plain" else raw
    label = label or f"{getattr(owner, '__name__', type(owner).__name__)}.{name}"

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        items = bound_args(fn, a, kw)
        try:
            rec_args = args(items) if args else {k: brief(v) for k, v in items}
        except Exception as exc:  # recording must never break the call
            rec_args = {"_record_error": repr(exc)[:200]}
        emit({"layer": layer, "fn": label, "phase": "call", "args": rec_args})
        t = time.perf_counter()
        try:
            res = fn(*a, **kw)
        except BaseException as exc:
            emit({"layer": layer, "fn": label, "phase": "error", "dur_ms": round((time.perf_counter() - t) * 1000, 2),
                  "error": f"{type(exc).__name__}: {exc}"[:800]})
            raise
        if inspect.isgenerator(res):
            return _passthrough(res, layer, label, t)
        rec = {"layer": layer, "fn": label, "phase": "return", "dur_ms": round((time.perf_counter() - t) * 1000, 2)}
        try:
            rec["result"] = result(res) if result else brief(res)
            if after:
                rec["after"] = after(items, res)
        except Exception as exc:
            rec["_record_error"] = repr(exc)[:200]
        emit(rec)
        return res

    setattr(owner, name, staticmethod(wrapper) if kind == "static" else classmethod(wrapper) if kind == "class"
            else wrapper)
    return wrapper


def mark_turns(chat_paths=("/api/chat",)):
    """Each POST to a chat endpoint starts a new turn number (requests are sent one at a time by the driver)."""
    import http.server
    original = http.server.BaseHTTPRequestHandler.parse_request

    def parse_request(self):
        ok = original(self)
        if ok and self.command == "POST" and any(self.path.startswith(p) for p in chat_paths):
            with _lock:
                _state["turn"] += 1
                _state["path"] = self.path
            emit({"layer": "input", "fn": "http", "phase": "request", "method": self.command, "path": self.path})
        return ok

    http.server.BaseHTTPRequestHandler.parse_request = parse_request


def module_files(names):
    import sys
    return {n: getattr(sys.modules.get(n), "__file__", None) for n in names}
