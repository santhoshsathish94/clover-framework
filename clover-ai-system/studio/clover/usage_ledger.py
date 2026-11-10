"""Real token and time use, from the engine's own counts: the `usage` of a /v1 reply, and a ledger with one line per
VS Code step and per turn (<base>/logs/usage/<device>-<yyyy-mm-dd>.jsonl).

Each model call's stats come from the engine (llama-server's timings, or the C engine's own count):
prompt_tokens (all of them, reused included), reused_tokens, new_tokens, prefill_s, decode_s.
"""
import json
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()


def totals(generations):
    """The sum of the engine's stats over the model calls behind one reply."""
    t = {"model_calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "prefill_s": 0.0,
         "decode_s": 0.0}
    for g in generations or ():
        if not isinstance(g, dict):
            continue
        t["model_calls"] += 1
        t["prompt_tokens"] += int(g.get("prompt_tokens") or 0)
        t["cached_tokens"] += int(g.get("reused_tokens") or 0)
        t["completion_tokens"] += int(g.get("new_tokens") or 0)
        t["prefill_s"] += float(g.get("prefill_s") or 0)
        t["decode_s"] += float(g.get("decode_s") or 0)
    t["prefill_s"], t["decode_s"] = round(t["prefill_s"], 2), round(t["decode_s"], 2)
    return t


def openai_usage(t):
    """OpenAI's `usage` for totals `t`; None when no model call was counted."""
    if not t or not t.get("model_calls"):
        return None
    return {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
            "total_tokens": t["prompt_tokens"] + t["completion_tokens"],
            "prompt_tokens_details": {"cached_tokens": t["cached_tokens"]}}


class Ledger:
    def __init__(self, base_dir, device):
        self.dir = Path(base_dir) / "logs" / "usage"
        self.device = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(device or "device"))

    @property
    def path(self):
        return self.dir / f"{self.device}-{time.strftime('%Y-%m-%d')}.jsonl"

    def record(self, kind, t, **fields):
        """Append one line; a ledger that cannot be written never stops a reply."""
        rec = dict({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}, **(t or {}),
                   **{k: v for k, v in fields.items() if v is not None})
        try:
            with _LOCK:
                self.dir.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            return None
        return rec

    def read(self, days=1):
        """The records of the last `days` ledger files, oldest first."""
        out = []
        for p in sorted(self.dir.glob(f"{self.device}-*.jsonl"))[-max(1, int(days)):]:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out
