"""Layer 2 (relationships): what earlier conversations hold that the current message needs. Chosen by experiment
(scratch-agents/lfm3-hybridagent/TODO.md step 19 and memory-research/, on LongMemEval and LoCoMo):
- units are single messages, keyed by their text and by the note the Omni writes about their exchange (exp A at turn
  level, exp B), ranked by BM25 and bge-small-en-v1.5, fused with reciprocal rank fusion; no graph filter in the
  ranking (exp F: no spectral filter transferred between the two benchmarks);
- the retrieved messages reach the model as a second system message right before the user's message, as background
  that may not be relevant (exp G5: 38/65 general tasks with unrelated memory vs 37/65 without; memory questions 0.60
  vs 0.04). No relevance check or gate: both lost answers or let unrelated memory through (exp G2-G4).
Nodes refer to Layer 1 records (sessions/<month>/<chat>.jsonl#<n>:<role>); their relations come from Layer 1 when the
index refreshes (the chat and turn of each message, outcomes that judged an answer); only the notes are new, kept in
growth/notes/. Vectors are derived data in CloverAI\\cache\\memory-index (rebuilt if deleted)."""
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

import numpy as np

EMBED_MODEL, EMBED_REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # exp_a.SPEC (from the model card)
HEAD = "From your earlier conversations with this user, dated (they may not be relevant to the current message):"
BUDGET, PER_UNIT = 1000, 700  # tokens of memory per turn and per message (exp D/G5)
IDLE_NOTE_S = 8
# memory-research/exp_b_notes.py PROMPTS["v3"]: v4 copied less but put the assistant's words under USER more often
NOTE_PROMPT = ("Conversation from {date}:\n{text}\n\n"
               "Write notes about the USER from this conversation, for a memory that will help answer their later "
               "questions.\nFormat, one note per line:\n"
               "USER: <something the user said about themselves: an event, plan, possession, preference, person, number "
               "or date>\n"
               "ASSISTANT: <a specific item the assistant gave the user: a name, number, place or recommendation>\n"
               "Rules: only what is written above; never general knowledge; do not describe the conversation or repeat "
               "its date; at most 3 lines. If there is nothing like that, write NONE.")
NOTE_LINE = re.compile(r"^(USER|ASSISTANT):\s*(.+)$")

# --- memory-research/lme.py and exp_a.py, verbatim ---
WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where which while who whom why will with would you
your yours yourself yourselves user assistant also could may might must shall like get got one two us let s t don ll re
ve m d im""".split())


def toks(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1]


def ntok(text):
    return int(len(WORD.findall(text.lower())) * 1.3) + 1


def bm25(docs, query, k1=1.2, b=0.75):
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


def key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ranks_of(s):
    r = np.empty(len(s), dtype=np.int64)
    r[np.argsort(-s, kind="stable")] = np.arange(len(s))
    return r


def rrf(*scores, k=60):
    return sum(1.0 / (k + ranks_of(s)) for s in scores)


# --- memory-research/exp_b_eval.py clean() and exp_g2.py block(), the latter without its heading ---
def clean_note(notes):
    return "\n".join(f"{m.group(1)}: {m.group(2).strip()}" for m in (NOTE_LINE.match(x.strip()) for x in notes.splitlines())
                     if m)


def block_lines(units, order, budget=BUDGET, per_unit=PER_UNIT):
    """Messages in rank order until the budget is spent, shown in conversation order with their dates.
    -> (text, picked indices)"""
    picked, used = [], 0
    for i in order:
        if used >= budget:
            break
        picked.append(i)
        used += min(ntok(units[i]["text"]), per_unit)
    picked.sort(key=lambda i: units[i]["pos"])
    lines = []
    for i in picked:
        role, _, text = units[i]["text"].partition(": ")
        words = text.split()
        cut = " ".join(words[:int(per_unit / 1.3)]) + (" ..." if len(words) > per_unit / 1.3 else "")
        lines.append(f"[{units[i]['date']}] {'Clover' if role == 'assistant' else 'user'}: {cut}")
    return "\n".join(lines), picked


def fmt_date(at):
    """ISO time -> LongMemEval's style, "2026/10/31 (Sat) 19:30" (the format the experiments used)."""
    try:
        t = datetime.datetime.fromisoformat(str(at))
    except ValueError:
        return str(at)[:16]
    return t.strftime("%Y/%m/%d (%a) %H:%M")


class SubwordHashEmbedder:
    """Zero-dependency 384-d L2-normalized word + character-trigram embedder used when torch/transformers or the
    bge-small-en-v1.5 weights are not available (such as on Android Chaquopy or lightweight web environments)."""

    def encode(self, texts, query=False, bs=32, max_len=512):
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
                        tri = w[k:k + 3]
                        ht = int(hashlib.md5(tri.encode("utf-8")).hexdigest()[:8], 16)
                        vec[ht % 384] += (0.35 if ((ht >> 8) & 1) == 0 else -0.35)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
        return out


class Embedder:
    """exp_a.Embedder for bge-small-en-v1.5 (CLS pooling, normalized, fp32 on the CPU), from a local folder."""

    def __init__(self, model_dir, threads=2):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModel.from_pretrained(str(model_dir)).eval()

    def encode(self, texts, query=False, bs=32, max_len=512):
        out = []
        pre = QUERY_PREFIX if query else ""
        with self.torch.inference_mode():
            for i in range(0, len(texts), bs):
                enc = self.tok([pre + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=max_len,
                               return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                out.append(self.torch.nn.functional.normalize(h[:, 0], dim=-1).numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class MemoryIndex:
    """The index over Layer 1's conversations. A background thread follows new messages, loads the embedder when recall
    is wanted, and while the Studio is idle has the Omni write a note for each exchange (note_writer: messages,
    max_tokens -> text, or None when it is busy or was interrupted). Notes are written whether or not recall is on."""

    def __init__(self, base, repo, is_idle=None, note_writer=None, embedder=None, start=True, recall=True):
        self.base, self.repo = Path(base), repo
        self.model_dir = self.base / "data" / "embed" / "bge-small-en-v1.5"
        self.cache_dir = self.base / "cache" / "memory-index"
        self.is_idle = is_idle or (lambda: False)
        self.note_writer = note_writer
        self.want_recall = bool(recall)
        self.lock = threading.RLock()
        self._enc = threading.Lock()
        self._refresh_lock = threading.Lock()
        self.emb, self.error = embedder, None
        self.state = "ready" if embedder is not None else "off"
        self.units, self.vec, self.notes, self.negative, self.answered = [], {}, {}, set(), set()
        self._files, self._dirty, self._wake = {}, False, threading.Event()
        if start:
            threading.Thread(target=self._run, daemon=True, name="memory-index").start()

    def describe(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "messages": len(self.units),
                    "exchange_notes": len(self.notes), "notes_with_content": sum(1 for v in self.notes.values() if v)}

    def wake(self):
        self._wake.set()

    def set_recall(self, on):
        self.want_recall = bool(on)
        if not on and self.emb is None:
            self.state = "off"
        self.wake()

    # ---------------- the background thread ----------------
    def _load_embedder(self):
        import paths
        if not (self.model_dir / "model.safetensors").is_file():
            if paths.ANDROID or paths.WEB:
                self.emb = SubwordHashEmbedder()
                self._load_cache()
                self.state = "ready"
                return
            self.state = "needs the embedder: run Clover AI Studio setup again"
            return
        self.state = "loading"
        try:
            self.emb = Embedder(self.model_dir)
            self._load_cache()
            self.state = "ready"
        except Exception as exc:  # fallback to SubwordHashEmbedder when torch/transformers is unavailable
            self.emb = SubwordHashEmbedder()
            self._load_cache()
            self.state, self.error = "ready", None

    def _run(self):
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

    def _load_cache(self):
        try:
            keys = json.loads((self.cache_dir / "keys.json").read_text(encoding="utf-8"))
            vecs = np.load(self.cache_dir / "vectors.npy")
        except (OSError, ValueError):
            return
        if len(keys) == len(vecs):
            self.vec.update(zip(keys, vecs))

    def _save_cache(self):
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

    def _ensure(self, texts):
        if self.emb is None:
            return
        todo = sorted({t for t in texts if key(t) not in self.vec}, key=len)
        if todo:
            with self._enc:
                vecs = self.emb.encode(todo)
            with self.lock:
                self.vec.update(zip(map(key, todo), vecs))
                self._dirty = True

    def refresh(self, save=True):
        """Messages from every chat file (re-read when changed), notes, and answers the user said were not right."""
        with self._refresh_lock:
            self._refresh()
        if save:
            self._save_cache()

    def _refresh(self):
        units, files = [], {}
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
            mine = []
            for m in self.repo.messages(sid):
                role, text, n = m.get("role"), str(m.get("text") or "").strip(), m.get("n")
                if role not in ("user", "clover") or not text or not isinstance(n, int):
                    continue
                who = "user" if role == "user" else "assistant"
                mine.append({"ref": f"{rel}#{n}:{role}", "sid": sid, "n": n, "role": role, "at": m.get("at") or "",
                             "date": fmt_date(m.get("at") or ""), "text": f"{who}: {text}"})
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
            if len(keep) < len(self.vec):  # a deleted chat leaves no vectors behind
                self.vec = {k: v for k, v in self.vec.items() if k in keep}
                self._dirty = True

    # ---------------- retrieval ----------------
    def recall(self, sid, n, query, history_turns=0, force=False):
        """The block for message n of chat sid: earlier messages, leaving out this chat's turns already in the prompt
        (the last history_turns before n, and n itself) and answers the user said were not right.
        -> (block, info) or (None, info)"""
        if force and self.emb is None:
            self._load_embedder()
        if self.state != "ready" or (not self.want_recall and not force):
            return None, {"layer": "recall", "skipped": self.state}
        qt = toks(query)
        if not qt:
            return None, None
        t0 = time.perf_counter()
        self.refresh(save=False)  # feedback given in the previous turn must apply now, not at the next background pass
        with self.lock:
            cands = [u for u in self.units if not (u["sid"] == sid and u["n"] >= n - history_turns)
                     and not (u["role"] == "clover" and (u["sid"], u["n"]) in self.negative)]
            # an exchange that never got an answer has nothing to note
            notes = [self.notes.get((u["sid"], u["n"]), None if (u["sid"], u["n"]) in self.answered else "")
                     for u in cands]
        if not cands:
            return None, None
        self._ensure([u["text"] for u in cands])
        with self._enc:
            qv = self.emb.encode([query[:2000]], query=True)[0]
        with self.lock:
            texts = [u["text"] for u in cands]
            e = np.stack([self.vec[key(t)] for t in texts])
            lists = [bm25([toks(t) for t in texts], qt), e @ qv]
            noted = all(x is not None for x in notes)  # partial notes would rank the unnoted by file order
            if noted:
                lists += [bm25([toks(x) for x in notes], qt), np.stack([self.vec[key(x or "(nothing)")] for x in notes]) @ qv]
            text, picked = block_lines(cands, list(np.argsort(-rrf(*lists), kind="stable")))
            refs = [cands[i]["ref"] for i in picked]
        info = {"layer": "recall", "refs": refs, "chats": len({r.split("#")[0] for r in refs}), "candidates": len(cands),
                "with_notes": noted, "best_cosine": round(float((e @ qv).max()), 4),
                "ms": round((time.perf_counter() - t0) * 1000, 1)}
        return HEAD + "\n" + text, info

    # ---------------- notes, written while the Studio is idle ----------------
    def pending_notes(self):
        """Exchanges (a message and Clover's answer) without a note, oldest first: (sid, n, user, answer, at)."""
        cur = self.repo.current
        with self.lock:
            turns = collections.OrderedDict()
            for u in self.units:
                turns.setdefault((u["sid"], u["n"]), {})[u["role"]] = u
            return [(k[0], k[1], v["user"]["text"].partition(": ")[2], v["clover"]["text"].partition(": ")[2],
                     v["user"]["at"]) for k, v in turns.items()
                    if "user" in v and "clover" in v and k not in self.notes and k != cur]

    def write_notes(self, limit=None):
        if self.note_writer is None:
            return 0
        done = 0
        try:
            for sid, n, utext, atext, at in self.pending_notes()[:limit]:
                if not self.is_idle():
                    break
                prompt = NOTE_PROMPT.format(date=fmt_date(at), text=f"user: {utext}\nassistant: {atext}"[:2800])
                raw = self.note_writer([{"role": "user", "content": prompt}], 90)
                if raw is None:  # busy or interrupted by a new message: later
                    break
                note = clean_note(raw)
                if not self.repo.add_note(sid, n, note):  # the chat was deleted while its note was written
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
