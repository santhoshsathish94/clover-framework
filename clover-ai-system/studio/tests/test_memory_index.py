"""Offline checks of Layer 2 (memory_index.py) on a real repository with a stand-in embedder (no torch, no model): the
messages it indexes, what recall leaves out (this chat's turns in the prompt, answers the user said were not right),
the block's format, notes written only while idle and stopped when the writer is interrupted, note fusion, deleting a
chat, the vector cache, and the engine placing the block right before the user's message.
Usage: python tests/test_memory_index.py"""
import hashlib
import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import clover_repo as R  # noqa: E402
import memory_index as MI  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


class FakeEmbedder:
    """Bag of hashed words, normalized: messages that share words are close (enough to test ranking plumbing)."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, query=False):
        self.calls += 1
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in MI.toks(t):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


ok = True
base = Path(tempfile.mkdtemp())
agents = base / "AGENTS-source.md"
agents.write_text("# AGENTS.md\n", encoding="utf-8")
repo = R.Repo(base, agents)


def turn(sid, text, reply, retry=None):
    n, _ = repo.turn_start(sid, text, retry)
    repo.turn_end(sid, n, {"reply": reply}, 1.0)
    return n


turn("chatA", "My sister Asha moved to Pune last month.", "That is a big move! Pune is lovely.")
turn("chatA", "I adopted a beagle named Biscuit.", "Biscuit is a great name for a beagle.")
turn("chatB", "What is the capital of France?", "The capital of France is Lyon.")
turn("chatB", "That is not right", "Sorry: the capital of France is Paris.", retry="said")
turn("chatC", "Plan a three-day trip to Kyoto for me.", "Day 1: Fushimi Inari; day 2: Arashiyama; day 3: Gion.")

emb = FakeEmbedder()
idle = {"v": True}
written = []


def writer(msgs, max_new):
    written.append(msgs[0]["content"])
    return "USER: has a sister called Asha in Pune\nASSISTANT: said Pune is lovely" if "Asha" in msgs[0]["content"] else "NONE"


idx = MI.MemoryIndex(base, repo, is_idle=lambda: idle["v"], note_writer=writer, embedder=emb, start=False)
idx.refresh()
ok &= check("every user message and answer indexed with a reference into sessions/",
            len(idx.units) == 10 and all(u["ref"].startswith("sessions/") and "#" in u["ref"] for u in idx.units))
ok &= check("an answer the user said was not right is known", ("chatB", 1) in idx.negative)

repo.current = ("chatD", 1)
n, _ = repo.turn_start("chatD", "Where does my sister live now?")
block, info = idx.recall("chatD", n, "Where does my sister live now?", history_turns=0)
ok &= check("recall finds the earlier message about the sister, from another chat",
            block is not None and "Asha moved to Pune" in block and any("chatA" in r for r in info["refs"]))
ok &= check("block format: heading, then dated 'user:' / 'Clover:' lines",
            block.startswith(MI.HEAD + "\n[") and "] user: My sister Asha" in block)
ok &= check("the current message itself is never recalled", not any("chatD" in r for r in info["refs"]))
ok &= check("the rejected answer (Lyon) is left out; the user's correction stays findable",
            not any(r.endswith("#1:clover") and "chatB" in r for r in info["refs"])
            and any(u["ref"].endswith("#2:user") and u["sid"] == "chatB" for u in idx.units))
ok &= check("without a note for every exchange, recall ranks on text alone", info["with_notes"] is False)

repo.turn_end("chatD", n, {"reply": "She lives in Pune."}, 1.0)
idx.refresh()
n2, _ = repo.turn_start("chatD", "And what is my dog called?")
_, info2 = idx.recall("chatD", n2, "And what is my dog called?", history_turns=1)
ok &= check("this chat's turns already in the prompt are left out",
            not any("chatD" in r for r in info2["refs"]) and any("chatA" in r for r in info2["refs"]))
repo.turn_end("chatD", n2, {"reply": "Biscuit."}, 1.0)

idle["v"] = False
idx.refresh()
ok &= check("no notes while the Studio is busy", idx.write_notes() == 0 and not written)
idle["v"] = True
done = idx.write_notes()
notes = repo.notes()
ok &= check("notes written for every answered exchange while idle, kept in growth/notes/",
            done == 7 and len(notes) == 7 and any((repo.dir / "growth" / "notes").rglob("*.jsonl")))
ok &= check("a note keeps only labelled lines; 'NONE' becomes an empty note",
            notes[("chatA", 1)] == "USER: has a sister called Asha in Pune\nASSISTANT: said Pune is lovely"
            and notes[("chatA", 2)] == "")
ok &= check("the note prompt is exp B's v3 prompt with the exchange", written and written[0].startswith("Conversation from ")
            and "Write notes about the USER" in written[0] and "user: My sister Asha" in written[0])

n3, _ = repo.turn_start("chatE", "Which city is Asha in?")
idx.refresh()
_, info3 = idx.recall("chatE", n3, "Which city is Asha in?", history_turns=0)
ok &= check("with a note for every exchange, recall fuses the note keys", info3["with_notes"] is True
            and any("chatA" in r for r in info3["refs"]))
repo.turn_end("chatE", n3, {"reply": "Pune."}, 1.0)

calls = {"n": 0}


def interrupted(msgs, max_new):
    calls["n"] += 1
    return None


idx.note_writer = interrupted
idx.refresh()
before = len(repo.notes())
ok &= check("an interrupted note is not saved and the rest wait", idx.write_notes() == 0 and calls["n"] == 1
            and len(repo.notes()) == before)

turn("chatF", "Which river is the longest?", "The longest river is the Thames.")
idx.refresh()
turn("chatF", "That is not right", "Sorry: it is the Nile.", retry="said")  # feedback after the index's last refresh
nf, _ = repo.turn_start("chatG", "Which river is the longest?")
_, infof = idx.recall("chatG", nf, "Which river is the longest?", history_turns=0)
ok &= check("feedback given since the last background refresh applies at once",
            not any(r.endswith("chatF.jsonl#1:clover") for r in infof["refs"]))
repo.turn_end("chatG", nf, {"reply": "The Nile."}, 1.0)

idx._dirty = True
idx._save_cache()
fresh = MI.MemoryIndex(base, repo, embedder=FakeEmbedder(), start=False)
fresh._load_cache()
ok &= check("vectors cached in cache/memory-index and loaded again",
            (base / "cache" / "memory-index" / "vectors.npy").is_file() and len(fresh.vec) == len(idx.vec))

repo.delete_session("chatA")
idx.refresh()
ok &= check("deleting a chat removes its messages, notes and vectors",
            not any(u["sid"] == "chatA" for u in idx.units) and ("chatA", 1) not in repo.notes()
            and MI.key("user: My sister Asha moved to Pune last month.") not in idx.vec)

turn("chatH", "I play the veena on weekends.", "The veena is a beautiful instrument.")
idx.refresh()


def deleting_writer(msgs, max_new):  # the user deletes the chat while its note is being written
    repo.delete_session("chatH")
    return "USER: plays the veena"


idx.note_writer = deleting_writer
idx.write_notes()
ok &= check("no note is kept for a chat deleted while its note was written",
            ("chatH", 1) not in repo.notes() and ("chatH", 1) not in idx.notes)

# the engine: the block is a system message right before the user's message; the chat's prefix is unchanged
import omni_native  # noqa: E402

eng = object.__new__(omni_native.OmniNative)
eng.lock, eng.stage, eng.media_dir, eng._extra = threading.Lock(), {"name": "idle"}, base, {}
eng.workdir, eng.workname = None, ""
seen = []


def fake_chat(msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None):
    seen.append([dict(m) for m in msgs])
    return "ok", [], {"finish": "stop"}


eng._chat = fake_chat


class Kit:
    def recall_context(self, q, history_turns):
        Kit.args = (q, history_turns)
        return MI.HEAD + "\n[2026/10/31 (Sat) 10:00] user: something", {"layer": "recall", "refs": ["sessions/x#1:user"]}


hist = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
res = eng.run_turn(hist, "question?", tools=False, layers=("recall",), kit=Kit(), system="SYS")
m = seen[0]
ok &= check("engine: system prompt, history, the recall system message, then the user's message",
            [x["role"] for x in m] == ["system", "user", "assistant", "system", "user"]
            and m[0]["content"] == "SYS" and m[3]["content"].startswith(MI.HEAD) and m[4]["content"] == "question?")
ok &= check("engine: recall told how many of this chat's turns are in the prompt", Kit.args == ("question?", 1))
ok &= check("engine: the recall step is logged with the turn", any(x.get("layer") == "recall" for x in res["layers"]))
seen.clear()
eng.run_turn(hist, "question?", tools=False, layers=(), kit=Kit(), system="SYS")
ok &= check("engine: recall off -> no extra message", [x["role"] for x in seen[0]] == ["system", "user", "assistant", "user"])

gate = omni_native.SideGate()
gate.hold()
ok &= check("side gate: held while a turn runs, free after", gate.is_set() and (gate.release() or not gate.is_set()))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
