#!/usr/bin/env python3
"""
S2 · Relationships & Turn-Level Memory Index (`memory_index.py`).

Ported from `laptop/clover/memory_index.py` (chosen by experiment on `LongMemEval_S` 500q and `LoCoMo` 1,531q):
  1. **Turn-level units (`sessions/<month>/<chat>.jsonl#<n>:<role>`)**:
     - Indexing individual user/assistant messages instead of whole rounds jumped 1,000-token evidence recall
       from `0.601` to `0.758` (`exp A`).
  2. **Idle Background `v3` Notes (`growth/notes/<device>/<month>.jsonl`)**:
     - While the Studio is idle, writes concise `USER:` / `ASSISTANT:` notes (`NOTE_PROMPT` `v3`) for each
       completed exchange (`exp B`: `+0.078` recall gain when fused via RRF).
  3. **BM25 + Dense Vector Reciprocal Rank Fusion (`RRF`, `k=60`)**:
     - Ranks turn texts and `v3` exchange notes using BM25 + `BAAI/bge-small-en-v1.5` (or deterministic 384-d
       hashed subword/token embedding when `bge-small-en-v1.5` is not downloaded on the phone).
     - Excludes answers the user marked not right or retried (`repo.negative_turns()`).
  4. **Dated Second-System-Message Injection (`F6`, `HEAD`)**:
     - Formats retrieved turns in chronological order with timestamps (`[YYYY/MM/DD (Day) HH:MM]`) under `HEAD`
       (`exp G5`: preserves `38/65` on general tasks vs `37/65` without memory while lifting memory QA from `0.04` to `0.60`).
"""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

EMBED_MODEL, EMBED_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
HEAD = "From your earlier conversations with this user, dated (they may not be relevant to the current message):"
BUDGET, PER_UNIT = 1000, 700
IDLE_NOTE_S = 8
NOTE_PROMPT = (
    "Conversation from {date}:\n{text}\n\n"
    "Write notes about the USER from this conversation, for a memory that will help answer their later "
    "questions.\nFormat, one note per line:\n"
    "USER: <something the user said about themselves: an event, plan, possession, preference, person, number "
    "or date>\n"
    "ASSISTANT: <a specific item the assistant gave the user: a name, number, place or recommendation>\n"
    "Rules: only what is written above; never general knowledge; do not describe the conversation or repeat "
    "its date; at most 3 lines. If there is nothing like that, write NONE."
)
NOTE_LINE = re.compile(r"^(USER|ASSISTANT):\s*(.+)$")

WORD = re.compile(r"[a-z0-9]+")
STOP = set(
    """a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with would you
your yours yourself yourselves user assistant also could may might must shall like get got one two us let s t don ll re
ve m d im""".split()
)


def toks(text: str) -> list[str]:
    return [w for w in WORD.findall((text or "").lower()) if w not in STOP and len(w) > 1]


def ntok(text: str) -> int:
    return int(len(WORD.findall((text or "").lower())) * 1.3) + 1


def bm25(docs: list[list[str]], query: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    n = len(docs)
    lens = np.array([len(d) for d in docs], dtype=np.float64)
    avg = lens.mean() if n else 1.0
    df = collections.Counter(w for d in docs for w in set(d))
    tf = [collections.Counter(d) for d in docs]
    s = np.zeros(n)
    for w in set(query):
        if w not in df:
            continue
        idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
        f = np.array([t.get(w, 0) for t in tf], dtype=np.float64)
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens / avg))
    return s


def key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def ranks_of(s: np.ndarray) -> np.ndarray:
    r = np.empty(len(s), dtype=np.int64)
    r[np.argsort(-s, kind="stable")] = np.arange(len(s))
    return r


def rrf(*scores: np.ndarray, k: int = 60) -> np.ndarray:
    return sum(1.0 / (k + ranks_of(s)) for s in scores)


def clean_note(notes: str) -> str:
    return "\n".join(
        f"{m.group(1)}: {m.group(2).strip()}"
        for m in (NOTE_LINE.match(x.strip()) for x in (notes or "").splitlines())
        if m
    )


def block_lines(units: list[dict[str, Any]], order: list[int], budget: int = BUDGET, per_unit: int = PER_UNIT) -> tuple[str, list[int]]:
    picked: list[int] = []
    used = 0
    for i in order:
        if used >= budget:
            break
        picked.append(i)
        used += min(ntok(units[i]["text"]), per_unit)
    picked.sort(key=lambda i: units[i]["pos"])
    lines: list[str] = []
    for i in picked:
        role, _, text = units[i]["text"].partition(": ")
        words = text.split()
        cut = " ".join(words[: int(per_unit / 1.3)]) + (" ..." if len(words) > per_unit / 1.3 else "")
        lines.append(f"[{units[i]['date']}] {'Clover' if role == 'assistant' else 'user'}: {cut}")
    return "\n".join(lines), picked


def fmt_date(at: str) -> str:
    try:
        s = str(at or "").replace("Z", "+00:00")
        t = datetime.datetime.fromisoformat(s)
    except ValueError:
        return str(at)[:16]
    return t.strftime("%Y/%m/%d (%a) %H:%M")


class Embedder:
    """`bge-small-en-v1.5` CLS-pooled normalized 384-d embedder when present on disk."""

    def __init__(self, model_dir: Path, threads: int = 2):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()

    def encode(self, texts: list[str], query: bool = False, bs: int = 32, max_len: int = 512) -> np.ndarray:
        out = []
        pre = QUERY_PREFIX if query else ""
        with self.torch.inference_mode():
            for i in range(0, len(texts), bs):
                enc = self.tok(
                    [pre + t for t in texts[i : i + bs]],
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                h = self.model(**enc).last_hidden_state
                out.append(self.torch.nn.functional.normalize(h[:, 0], dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class SubwordHashEmbedder:
    """
    Zero-dependency 384-d L2-normalized word + character-ngram embedder used on mobile (`s23`)
    when `data/embed/bge-small-en-v1.5` has not been downloaded yet, so `MemoryIndex` works offline out-of-the-box.
    """

    def encode(self, texts: list[str], query: bool = False, bs: int = 32, max_len: int = 512) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for row_idx, txt in enumerate(texts):
            words = toks(txt)
            vec = out[row_idx]
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % 384
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                vec[idx] += sign * (2.0 if any(c.isdigit() for c in w) else 1.0)
                if len(w) >= 4:
                    for k in range(len(w) - 2):
                        tri = w[k : k + 3]
                        ht = int(hashlib.md5(tri.encode("utf-8")).hexdigest()[:8], 16)
                        vec[ht % 384] += (0.35 if ((ht >> 8) & 1) == 0 else -0.35)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
        return out


class MemoryIndex:
    """
    S2 Turn-Level Memory Index over `S1` (`sessions/<month>/<chat>.jsonl`, `growth/notes/`, `outcomes/`).
    """

    def __init__(self, base: Path, repo: Any, is_idle: Any = None, note_writer: Any = None, embedder: Any = None, start: bool = True, recall: bool = True):
        self.base = Path(base)
        self.repo = repo
        self.model_dir = self.base / "data" / "embed" / "bge-small-en-v1.5"
        self.cache_dir = self.base / "cache" / "memory-index"
        self.is_idle = is_idle or (lambda: False)
        self.note_writer = note_writer
        self.want_recall = bool(recall)
        self.lock = threading.RLock()
        self._enc = threading.Lock()
        self._refresh_lock = threading.Lock()
        self.emb = embedder
        self.error: str | None = None
        self.state = "ready" if embedder is not None else "off"
        self.units: list[dict[str, Any]] = []
        self.vec: dict[str, np.ndarray] = {}
        self.notes: dict[tuple[str, int], str] = {}
        self.negative: set[tuple[str, int]] = set()
        self.answered: set[tuple[str, int]] = set()
        self._files: dict[Path, Any] = {}
        self._dirty = False
        self._wake = threading.Event()
        if self.emb is None and self.want_recall:
            self._load_embedder()
        if start:
            threading.Thread(target=self._run, daemon=True, name="s2-memory-index").start()

    def describe(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "embedder": type(self.emb).__name__ if self.emb is not None else None,
                "error": self.error,
                "messages": len(self.units),
                "exchange_notes": len(self.notes),
                "notes_with_content": sum(1 for v in self.notes.values() if v),
            }

    def wake(self) -> None:
        self._wake.set()

    def set_recall(self, on: bool) -> None:
        self.want_recall = bool(on)
        if not on and self.emb is None:
            self.state = "off"
        self.wake()

    def _load_embedder(self) -> None:
        self.state = "loading"
        try:
            if (self.model_dir / "model.safetensors").is_file():
                self.emb = Embedder(self.model_dir)
            else:
                self.emb = SubwordHashEmbedder()
            self._load_cache()
            self.state = "ready"
        except Exception as exc:
            self.emb = SubwordHashEmbedder()
            self.state = "ready"
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        while True:
            try:
                if self.want_recall and self.emb is None and self.state != "error":
                    self._load_embedder()
                self.refresh()
                self.write_notes()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._wake.wait(30)
            self._wake.clear()

    def _load_cache(self) -> None:
        try:
            keys = json.loads((self.cache_dir / "keys.json").read_text(encoding="utf-8"))
            vecs = np.load(self.cache_dir / "vectors.npy")
        except (OSError, ValueError):
            return
        if len(keys) == len(vecs):
            self.vec.update(zip(keys, vecs))

    def _save_cache(self) -> None:
        with self.lock:
            if not self._dirty or not self.vec:
                return
            ks = list(self.vec)
            arr = np.stack([self.vec[k] for k in ks])
            self._dirty = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / "vectors.tmp.npy", arr)
        (self.cache_dir / "keys.tmp.json").write_text(json.dumps(ks), encoding="utf-8")
        os.replace(self.cache_dir / "vectors.tmp.npy", self.cache_dir / "vectors.npy")
        os.replace(self.cache_dir / "keys.tmp.json", self.cache_dir / "keys.json")

    def _ensure(self, texts: list[str]) -> None:
        if self.emb is None:
            return
        todo = sorted({t for t in texts if key(t) not in self.vec}, key=len)
        if todo:
            with self._enc:
                vecs = self.emb.encode(todo)
            with self.lock:
                self.vec.update(zip(map(key, todo), vecs))
                self._dirty = True

    def refresh(self, save: bool = True) -> None:
        with self._refresh_lock:
            self._refresh()
        if save:
            self._save_cache()

    def _refresh(self) -> None:
        units: list[dict[str, Any]] = []
        files: dict[Path, Any] = {}
        for sid, fp in sorted(self.repo.session_files().items()):
            try:
                st = fp.stat()
            except OSError:
                continue
            sig = (st.st_mtime_ns, st.st_size)
            old = self._files.get(fp)
            if old and old[0] == sig:
                files[fp] = old
                units += old[1]
                continue
            rel = fp.relative_to(self.repo.dir).as_posix()
            mine: list[dict[str, Any]] = []
            for m in self.repo.messages(sid):
                role, text, n = m.get("role"), str(m.get("text") or "").strip(), m.get("n")
                if role not in ("user", "clover") or not text or not isinstance(n, int):
                    continue
                who = "user" if role == "user" else "assistant"
                mine.append({
                    "ref": f"{rel}#{n}:{role}",
                    "sid": sid,
                    "n": n,
                    "role": role,
                    "at": m.get("at") or "",
                    "date": fmt_date(m.get("at") or ""),
                    "text": f"{who}: {text}",
                })
            files[fp] = (sig, mine)
            units += mine
        units.sort(key=lambda u: (u["at"], u["n"], u["role"] != "user"))
        for i, u in enumerate(units):
            u["pos"] = i
        notes, negative = self.repo.notes(), self.repo.negative_turns()
        texts = [u["text"] for u in units] + [x for x in notes.values() if x] + ["(nothing)"]
        self._ensure(texts)
        answered = {(u["sid"], u["n"]) for u in units if u["role"] == "clover"}
        with self.lock:
            self.units, self._files, self.notes, self.negative, self.answered = units, files, notes, negative, answered
            keep = set(map(key, texts))
            if len(keep) < len(self.vec):
                self.vec = {k: v for k, v in self.vec.items() if k in keep}
                self._dirty = True

    def recall(self, sid: str, n: int, query: str, history_turns: int = 0, budget: int = BUDGET) -> tuple[str | None, dict[str, Any] | None]:
        """
        Retrieves earlier turns from `sessions/` ranked by BM25 + dense vector RRF (plus `v3` exchange notes when present),
        excluding turns already in the active chat window and answers the user marked not right (`self.negative`).
        """
        if self.state != "ready" or not self.want_recall:
            return None, {"layer": "recall", "skipped": self.state}
        qt = toks(query)
        if not qt:
            return None, None
        t0 = time.perf_counter()
        self.refresh(save=False)
        with self.lock:
            cands = [
                u for u in self.units
                if not (u["sid"] == sid and u["n"] >= n - history_turns)
                and not (u["role"] == "clover" and (u["sid"], u["n"]) in self.negative)
            ]
            notes = [
                self.notes.get((u["sid"], u["n"]), None if (u["sid"], u["n"]) in self.answered else "")
                for u in cands
            ]
        if not cands:
            return None, None
        self._ensure([u["text"] for u in cands])
        with self._enc:
            qv = self.emb.encode([query[:2000]], query=True)[0]
        with self.lock:
            texts = [u["text"] for u in cands]
            e = np.stack([self.vec[key(t)] for t in texts])
            lists = [bm25([toks(t) for t in texts], qt), e @ qv]
            noted = all(x is not None for x in notes)
            if noted:
                lists += [
                    bm25([toks(x) for x in notes], qt),
                    np.stack([self.vec[key(x or "(nothing)")] for x in notes]) @ qv,
                ]
            text, picked = block_lines(cands, list(np.argsort(-rrf(*lists), kind="stable")), budget=budget)
            refs = [cands[i]["ref"] for i in picked]
        info = {
            "layer": "recall",
            "refs": refs,
            "chats": len({r.split("#")[0] for r in refs}),
            "candidates": len(cands),
            "with_notes": noted,
            "best_cosine": round(float((e @ qv).max()), 4),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return HEAD + "\n" + text, info

    def pending_notes(self) -> list[tuple[str, int, str, str, str]]:
        cur = self.repo.current
        with self.lock:
            turns: dict[tuple[str, int], dict[str, Any]] = collections.OrderedDict()
            for u in self.units:
                turns.setdefault((u["sid"], u["n"]), {})[u["role"]] = u
            return [
                (
                    k[0],
                    k[1],
                    v["user"]["text"].partition(": ")[2],
                    v["clover"]["text"].partition(": ")[2],
                    v["user"]["at"],
                )
                for k, v in turns.items()
                if "user" in v and "clover" in v and k not in self.notes and k != cur
            ]

    def write_notes(self, limit: int | None = None) -> int:
        if self.note_writer is None:
            return 0
        done = 0
        try:
            for sid, n, utext, atext, at in self.pending_notes()[:limit]:
                if not self.is_idle():
                    break
                prompt = NOTE_PROMPT.format(date=fmt_date(at), text=f"user: {utext}\nassistant: {atext}"[:2800])
                raw = self.note_writer([{"role": "user", "content": prompt}], 90)
                if raw is None:
                    break
                note = clean_note(raw)
                if not self.repo.add_note(sid, n, note):
                    continue
                if note:
                    self._ensure([note])
                with self.lock:
                    self.notes[(sid, n)] = note
                done += 1
        finally:
            if done:
                self.repo.schedule(f"Notes for {done} exchange{'s' if done > 1 else ''}")
                self._save_cache()
        return done
