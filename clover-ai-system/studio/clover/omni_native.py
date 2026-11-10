"""Layer 1 as LFM3-Omni-2B (LiquidAI): text, images, audio and video in; text and speech out; on llama.cpp.

The Omni's language model has LFM3-1.7B's layout but its own weights; it is a non-reasoning model (its card:
`enable_thinking` unsupported; llama-server runs it with `--reasoning off`). llama-server applies the chat template
stored in the GGUF, turns media parts into the model's media tokens (the projector holds the vision and audio
encoders; video needs ffmpeg) and speaks text through /tts. The Studio keeps its own tool loop, permissions and
Clover layers on top, the same way as for LFM3-1.7B (lfm3_native.LFM3Native, whose tools this class reuses).
Media files live in CloverAI\\media (the user clears or backs them up) and reach llama-server as file:// names inside
--media-path.
"""
import datetime
import json
import re
import threading
import time
import uuid
from pathlib import Path

import llamacpp_engine as L
import lfm3_native as N
import clover_cycle as C
import paths
import semantics

QUANTS = {"q4_0": "LFM3-Omni-2B-Q4_0.gguf", "q4_k_m": "LFM3-Omni-2B-Q4_K_M.gguf"}
MMPROJ = "mmproj-LFM3-Omni-2B-Q8_0.gguf"
PART = {"image": "image_url", "audio": "input_audio", "video": "input_video"}
GREEDY = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
          "xtc_probability": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}  # the card: greedy for text output
TOOL_XML = re.compile(r"<tool_call>[\s\S]*?(?:</tool_call>|$)")
LOOP_TAIL = 160
# a forced round (tool_choice required) writes its text first: "which application would you like to install? Could you
# please tell me:" came with a run_powershell call (tests/tool_choice_probe.py)
ASKS_USER = re.compile(r"\?\s*(?:\n|$)|\b(?:could|can|would) you (?:please )?(?:tell|provide|confirm|clarify|share|specify)\b"
                       r"|\bplease (?:tell|provide|confirm|clarify|specify)\b", re.I)
PROMISE = semantics.PROMISE
ANDROID = paths.ANDROID
PLATFORM = "Android phone" if ANDROID else "Windows laptop"


def looping(text):
    """The last 160 characters appeared twice before: greedy decoding is repeating itself (a user's request ran to the
    2,048-token limit in 89 s that way)."""
    return len(text) >= 4 * LOOP_TAIL and text.count(text[-LOOP_TAIL:], 0, len(text) - LOOP_TAIL) >= 2


# a paragraph that closes an answer: "Just let me know what you'd like to do!", "I'm here to help!"
SIGN_OFF = re.compile(r"\blet\s+me\s+know\b|\bfeel\s+free\s+to\b|\bi'?m\s+(?:here|ready)\s+to\s+help\b|\bhappy\s+to\s+help\b|"
                      r"\banything\s+else\b|\bwhat\s+would\s+you\s+like\s+to\s+(?:do|work\s+on)\b|\bready\s+when\s+you\s+are\b", re.I)
TRAILING_Q = re.compile(
    r"\n\n+(?:(?:Would|Do)\s+you\s+(?:like|want|need)\b|What\s+(?:else\s+)?would\s+you\s+like\b|"
    r"Is\s+there\s+anything\s+else\b|Shall\s+I\b|If\s+you(?:'d|\s+would)\s+like[^\n]*\?)"
    r"[\s\S]*\?\s*$", re.I)


def rambling(text):
    """-> where the answer ends, or None. Past its sign-off the Omni wrote one variation after another (\"If you'd like
    me to ..., I can do that too. Just let me know ...!\" six times, 37 s, 2026-11-04), each different enough for
    looping(). Three sign-off paragraphs with a sign-off said twice end the answer after the first; an email's own
    \"let me know\" and the closing line after it are different sentences."""
    offs = [m for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text) if SIGN_OFF.search(m.group(0))]
    if len(offs) < 3:
        return None
    seen = set()
    for m in offs:
        for s in re.split(r"(?<=[.!?])\s+", m.group(0)):
            if SIGN_OFF.search(s):
                key = re.sub(r"[^a-z' ]+", "", s.lower()).strip()
                if key in seen:
                    return offs[0].end()
                seen.add(key)
    return None
# Speech runs on a CPU llama.cpp instance when chat has the Iris Xe. Measured on the i5-1335U (tests/omni_resource_probe,
# omni_prefill_probe): speech 1.4-1.6x real time on 10 CPU threads against 0.8x on the GPU, while the GPU reads a
# 2,700-token Studio prompt at 438 tok/s against 94 on the CPU (writing about equal, 26-28 tok/s). The instance holds
# about 1.6 GB, so it starts on first use and stops after VOICE_IDLE_S without speech.
VOICE_IDLE_S = 600
VOICE_THREADS = 10


def omni_files(base_dir, quant="q4_0"):
    """-> (paths, missing names) for the Omni on llama.cpp."""
    base_dir = Path(base_dir)
    g = base_dir / "data" / "omni" / "gguf"
    paths = {"model": g / QUANTS[quant], "mmproj": g / MMPROJ}
    missing = [p.name for p in paths.values() if not p.is_file()]
    return paths, missing


def media_part(m):
    kind = m["kind"]
    url = "file://" + m["file"]
    if kind == "image":
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": PART[kind], PART[kind]: {"url": url}}


def user_content(text, media, media_dir):
    """OpenAI content for a user message: media first (llama.cpp's advice), then the text."""
    parts = [media_part(m) for m in media or () if (Path(media_dir) / m["file"]).is_file()]
    return parts + [{"type": "text", "text": text}] if parts else text


def speakable(text, limit=1200):
    """Answer text for speech: no code blocks, links or markdown marks."""
    t = re.sub(r"```[\s\S]*?```", " (code shown on screen) ", text or "")
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def speech_parts(text, max_chars=240, first_chars=80, grow=1.25, limit=4000):
    """Speakable text in sentence groups; the next part is synthesized while the current one plays. The first part is
    short so speech starts soon (a 237-character part took 14.6 s before any sound, with the instance start), and each
    part is at most `grow` x the previous one: the CPU speaks 1.2-1.6x faster than real time, so a part longer than
    that is not ready when the previous one ends (45 then 214 characters: 3.9 s of sound, then 9.7 s to make the next)."""
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?;:])\s+", speakable(text, limit)):
        while True:
            cap = min(max_chars, max(first_chars, int(grow * len(parts[-1])))) if parts else first_chars
            if cur and len(cur) + 1 + len(s) > cap:
                parts.append(cur)
                cur = ""
                continue
            if len(s) <= cap:
                cur = (cur + " " + s).strip()
                break
            # a sentence longer than the part: cut at a late comma, else at the last space before the limit
            cut = s.rfind(", ", 0, cap)
            if cut < cap // 2:
                cut = s.rfind(" ", 0, cap)
            cut = cut if cut > cap // 3 else cap
            parts.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
            if not s:
                break
    return [p for p in parts + [cur] if p]


class SideGate:
    """Held while a turn or speech needs the hardware; a side completion (an exchange note) stops when it is held.
    Works as chat_stream's `cancel`."""

    def __init__(self):
        self.n, self.lock = 0, threading.Lock()

    def is_set(self):
        return self.n > 0

    def hold(self):
        with self.lock:
            self.n += 1

    def release(self):
        with self.lock:
            self.n = max(0, self.n - 1)


class OmniNative(N.LFM3Native):
    def __init__(self, base_dir, perms, max_tokens=65536, progress=None, engine="gpu", quant="q4_0",
                 voice_threads=VOICE_THREADS, gpu_name="GPU", threads=None):
        """max_tokens (context), voice_threads, gpu_name and (on a phone) threads come from this machine (hardware.choose)."""
        t0 = time.perf_counter()
        progress = progress or (lambda m: None)
        base_dir = Path(base_dir)
        self.base_dir, self.perms, self.quant = base_dir, perms, quant
        self.voice_threads = int(voice_threads)
        paths, missing = omni_files(base_dir, quant)
        if missing:
            raise FileNotFoundError("missing " + ", ".join(missing) + " (run Clover AI Studio setup again)")
        self.media_dir = base_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        ff_txt = base_dir / "tools" / "ffmpeg" / "BIN.txt"
        self.ffmpeg_dir = ff_txt.read_text(encoding="utf-8").strip() if ff_txt.is_file() else ""
        extra = ["--reasoning", "off", "--media-path", str(self.media_dir), "--video-fps", "2"]
        if ANDROID:
            # Adreno's OpenCL backend lacks ops of the projector's graph: a picture ended llama-server (S23, b12012); on
            # the CPU a picture took 8 s and a voice clip 3 s, and 0.74 GB less GPU memory stays pinned
            extra += ["--no-mmproj-offload"]
        if self.ffmpeg_dir:
            extra += ["--video-ffmpeg-dir", self.ffmpeg_dir]
        kind = "vulkan" if engine == "gpu" else "cpu"
        progress("LFM3-Omni-2B on the " + (gpu_name if engine == "gpu" else "CPU") + " (llama.cpp, 4-bit)")
        self.engine_kind, self.max_tokens, self.cache = engine, int(max_tokens), None
        # speech on its own CPU instance while chat has the laptop's GPU; a phone has no memory for a second copy
        self.separate_voice = engine == "gpu" and not ANDROID
        # a re-sent prompt is processed again from the last checkpoint, which sits at a ubatch end: the same 4,198-token
        # VS Code request reprocessed 2,052 tokens at -ub 2048 and 260 at -ub 256 (k3 CPU 3.2 s -> 0.5 s, laptop Iris Xe
        # Vulkan 5.8 s -> 1.5 s); cold it cost nothing on k3 (5.6 s vs 6.1 s) and 4-8% on the laptop (11.0 s vs 10.6 s);
        # a 1,920-token picture and a voice clip still worked at 256 on both (TokenEconomizer item 3)
        ubatch_sz = 256
        self.gpu = L.LlamaServer(base_dir, kind, max_tokens, ubatch=ubatch_sz, model=paths["model"], mmproj=paths["mmproj"],
                                 extra_args=extra, name=f"omni-{kind}", timeout_s=300, threads=threads)
        props = self.gpu.props()
        mods = props.get("modalities") or {}
        self.inputs = ["text"] + [k for k in ("vision", "audio") if mods.get(k)] + (["video"] if self.ffmpeg_dir and mods.get("vision") else [])
        self.lock = threading.Lock()
        self.stage = {"name": "idle", "since": time.time()}
        self._paths = paths
        self._voice, self._voice_busy, self._voice_used, self._voice_error = None, 0, 0.0, None
        self._voice_lock = threading.Lock()
        self.side_gate = SideGate()
        self._py = None
        self._web = None
        self._extra = {}
        self.workdir, self.workname, self.after_write = None, "", None  # the active workspace (server sets them)
        self.setup_s = round(time.perf_counter() - t0, 2)

    def describe(self):
        return {"mode": "omni", "model": "LFM3-Omni-2B", "engine": self.engine_kind, "max_tokens": self.max_tokens,
                "setup_s": self.setup_s, "tools": [t["function"]["name"] for t in N.STUDIO_TOOLS],
                "layers": [
                    f"C1 LFM3-Omni-2B (llama.cpp {self.gpu.kind}, {self.quant.upper()})",
                    "C2 HybridAgent (HybridLink2048 + Lfm3NativeHybridLink, plan-trunk-0)",
                    "Bridges (Lfm3ComboBridge + Lfm3ExpertBridge, bridge-trunk-0)",
                    "C3 11-level NCM (VQ-VAE + 12L NextConceptTransformer + 3L PointerDecoder)",
                    "S1 Memory & Records · S2 Recall & Ontology · S3 Tools & Compiler · CSI & PPL",
                ],
                "device": self.gpu.device, "weights": f"4-bit ({self.quant.upper()}) + projector Q8_0",
                "backend": "llama.cpp " + self.gpu.kind + " + C2 HybridAgent + Bridges + C3 11-level NCM",
                "inputs": self.inputs, "outputs": ["text", "speech"],
                "speech": (f"CPU, own llama.cpp instance ({self.voice_threads} threads; "
                           + ("running)" if self._voice is not None else "starts when Clover first speaks)")
                           if self.separate_voice else "same instance as chat"),
                "speech_error": self._voice_error,
                "reasoning": False}

    def system_prompt(self, tools_on=True, asked=None):
        now = datetime.datetime.now().astimezone()
        return (f"You are Clover, an assistant running entirely on the user's {PLATFORM} (model: LFM3-Omni-2B). "
                f"You can see images, hear audio and watch videos the user attaches. "
                f"Today is {now.strftime('%A %d %B %Y')}. The user's home folder is {Path.home()}. "
                "Answer directly and completely without asking unnecessary follow-up or clarifying questions. "
                + self.workspace_sentence() + N.tools_sentence(tools_on, asked))

    def transcribe(self, media):
        """The words of the voice messages in `media`, to see whether the user asked for a tool."""
        parts = [media_part(m) for m in media or () if m["kind"] == "audio" and (self.media_dir / m["file"]).is_file()]
        if not parts:
            return ""
        # the probe's ASR form; an instruction after the audio made the Omni answer the question instead
        msgs = [{"role": "system", "content": "Perform ASR in English."}, {"role": "user", "content": parts}]
        with self.lock:
            try:
                return self._chat(msgs, [], 200, {})[0].strip()
            finally:
                self.stage = {"name": "idle", "since": time.time()}

    def tts(self, text, voice="female", lang="en"):
        t = speakable(text)
        if not t:
            raise ValueError("nothing to speak")
        self.side_gate.hold()
        try:
            srv = self._voice_server() if self.separate_voice else None
            if srv is not None:
                try:
                    return srv.tts(t, voice=voice, lang=lang)
                finally:
                    self._voice_done()
            with self.lock:  # one llama-server slot: speech waits for a running answer
                self.stage = {"name": "speaking", "since": time.time()}
                try:
                    return self.gpu.tts(t, voice=voice, lang=lang)
                finally:
                    self.stage = {"name": "idle", "since": time.time()}
        finally:
            self.side_gate.release()

    def side_complete(self, msgs, max_new):
        """A short greedy completion (an exchange note) on the CPU instance, never on the chat's slot, so the chat keeps
        its cached prompt. -> text, or None when there is no CPU instance (CPU engine) or a turn or speech started."""
        if not self.separate_voice or self.side_gate.is_set():
            return None
        srv = self._voice_server()
        if srv is None:
            return None
        try:
            text = ""
            for ev in srv.chat_stream(dict(GREEDY, messages=msgs, max_tokens=int(max_new), cache_prompt=False),
                                      self.side_gate):
                if ev.get("cancelled"):
                    return None
                for ch in ev.get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
            return text
        finally:
            self._voice_done()

    def warm_voice(self):
        """Start the speech instance in the background (a reply that will be spoken is being written)."""
        def run():
            if self._voice_server() is not None:
                self._voice_done()
        if self.separate_voice and self._voice is None:
            threading.Thread(target=run, daemon=True).start()

    def _voice_done(self):
        with self._voice_lock:
            self._voice_busy -= 1
            self._voice_used = time.time()

    def _voice_server(self):
        """The CPU speech instance (started here on first use), marked busy; None -> speak on the main instance."""
        with self._voice_lock:
            if self._voice is not None and self._voice.proc.poll() is not None:
                self._voice = None
            if self._voice is None:
                try:
                    self._voice = L.LlamaServer(self.base_dir, "cpu", 8192, model=self._paths["model"],
                                                mmproj=self._paths["mmproj"], name="omni-voice", timeout_s=120,
                                                extra_args=["--reasoning", "off", "-t", str(self.voice_threads)])
                    self._voice_error = None
                    threading.Thread(target=self._voice_reaper, daemon=True).start()
                except (OSError, RuntimeError, TimeoutError) as exc:
                    self._voice_error = str(exc)
                    return None
            self._voice_busy += 1
            return self._voice

    def _voice_reaper(self):
        while True:
            time.sleep(30)
            with self._voice_lock:
                if self._voice is None:
                    return
                if not self._voice_busy and time.time() - self._voice_used > VOICE_IDLE_S:
                    self._voice.close()
                    self._voice = None
                    return

    def free_voice(self):
        """Stops the voice instance when it is idle (it starts again on the next Listen): ~1.6 GB for a picture."""
        with self._voice_lock:
            if self._voice is None or self._voice_busy:
                return False
            self._voice.close()
            self._voice = None
            return True

    def close(self):
        with self._voice_lock:
            if self._voice is not None:
                self._voice.close()
                self._voice = None
        super().close()

    # ---------------- generation ----------------
    def _chat(self, msgs, schemas, max_new, by_name, on_delta=None, cancel=None, seed=0, tool_choice=None, on_plan=None):
        """One C1 inference pass through /v1/chat/completions -> (content, calls, stats).
        C1 is strictly for inference only; C2 plans and C3 (11-Level NCM) decodes the final answer."""
        eff_max = min(int(max_new), 512) if tool_choice == "required" else int(max_new)
        body = dict(GREEDY, messages=msgs, max_tokens=eff_max, seed=int(seed), cache_prompt=True)
        if schemas:
            body["tools"] = schemas
            if tool_choice:
                body["tool_choice"] = tool_choice
        content, acc, finish, tm = "", {}, None, {}
        planned_tools = set()
        t0 = time.perf_counter()
        t_first = None
        looped, checked, cut = False, 0, None
        self.stage = {"name": "planning" if tool_choice else "C1 inference", "since": time.time()}
        stream = self.gpu.chat_stream(body, cancel)
        try:
            for ev in stream:
                if ev.get("cancelled"):
                    finish = "cancelled"
                    break
                tm = ev.get("timings") or tm
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("tool_calls"):
                        if t_first is None:
                            t_first = time.perf_counter()
                            self.stage = {"name": "planning tool call" if (tool_choice or d.get("tool_calls")) else "C1 inference",
                                          "since": time.time()}
                    if d.get("content"):
                        content += d["content"]
                        if "<function=" in content and on_plan:
                            for fm in re.finditer(r"<function=([^>\s]+)>", content):
                                tname = fm.group(1)
                                if tname not in planned_tools:
                                    planned_tools.add(tname)
                                    self.stage = {"name": f"preparing {tname}", "since": time.time()}
                                    on_plan(tname)
                        if on_delta:
                            on_delta(d["content"], content)
                        if "</tool_call>" in content:
                            finish = "tool_calls"
                            break
                        if len(content) - checked >= 200:
                            checked = len(content)
                            looped = looping(content)
                            cut = rambling(content)
                    for tc in d.get("tool_calls") or []:
                        slot = acc.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
                        if slot["name"] and slot["name"] in by_name and slot["name"] not in planned_tools:
                            planned_tools.add(slot["name"])
                            self.stage = {"name": f"preparing {slot['name']}", "since": time.time()}
                            if on_plan:
                                on_plan(slot["name"])
                    finish = ch.get("finish_reason") or finish
                if finish == "tool_calls" and "</tool_call>" in content:
                    break
                if looped:  # closing the stream stops llama-server; one copy of the repeated text is kept
                    tail = content[-LOOP_TAIL:]
                    content, finish = content[:content.find(tail) + LOOP_TAIL], "stop"
                    break
                if cut is not None:
                    content, finish = content[:cut], "stop"
                    break
        finally:
            stream.close()
        calls = []
        for s in acc.values():
            try:
                args = json.loads(s["arguments"] or "{}")
            except ValueError:
                args = {"_raw": s["arguments"]}
            calls.append({"id": s["id"] or "call_" + uuid.uuid4().hex[:16], "name": s["name"], "arguments": args})
        if not calls and "<tool_call>" in content:  # template parser did not split the call out: parse it here
            calls = N.parse_tool_calls(content, by_name)
            content = TOOL_XML.sub("", content)
        t_end = time.perf_counter()
        reason = {"stop": "stop", "tool_calls": "stop", "length": "length", "cancelled": "cancelled"}.get(finish, "stop")
        if cancel is not None and cancel.is_set():
            reason = "cancelled"
        prompt_ms, pred_ms = tm.get("prompt_ms"), tm.get("predicted_ms")
        prefill_s = (prompt_ms / 1000) if prompt_ms is not None else ((t_first or t_end) - t0)
        decode_s = (pred_ms / 1000) if pred_ms is not None else (t_end - (t_first or t_end))
        n_new = int(tm.get("predicted_n") or 0)
        return content, calls, {"prompt_tokens": int(tm.get("prompt_n") or 0) + int(tm.get("cache_n") or 0),
                                "reused_tokens": int(tm.get("cache_n") or 0), "new_tokens": n_new,
                                "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
                                "decode_tok_s": round(n_new / decode_s, 2) if decode_s > 0 and n_new else None,
                                "finish": reason, **({"looped": True} if looped else {}),
                                **({"rambling": True} if cut is not None else {})}

    def _history(self, history, keep_last_media=False, strip_attached=False, current_query=None, is_follow_up=False):
        if history and current_query and not is_follow_up and not keep_last_media:
            import lfm3_layers
            if not lfm3_layers.needs_recall(current_query):
                q_terms = lfm3_layers.topic_terms(current_query)
                if len(q_terms) >= 1 and not re.search(r"\b(?:it|that|this|those|these|them|they|same|instead|above|earlier|previous|before|more|else|another)\b", current_query, re.I):
                    rel_pairs = []
                    i = 0
                    while i < len(history):
                        m = history[i]
                        if m.get("role") == "user":
                            u_txt = str(m.get("content") or "")
                            a_m = history[i + 1] if (i + 1 < len(history) and history[i + 1].get("role") == "assistant") else None
                            a_txt = str((a_m or {}).get("content") or "")
                            if lfm3_layers.topic_terms(u_txt + " " + a_txt) & q_terms:
                                rel_pairs.append(m)
                                if a_m is not None:
                                    rel_pairs.append(a_m)
                            i += 2 if a_m is not None else 1
                        else:
                            i += 1
                    history = rel_pairs
        out = []
        last_media_idx = -1
        if keep_last_media:
            for idx in range(len(history) - 1, -1, -1):
                if history[idx].get("role") == "user" and history[idx].get("media"):
                    last_media_idx = idx
                    break
        for idx, m in enumerate(history):
            if idx == last_media_idx and m.get("role") == "user" and m.get("media"):
                out.append({"role": "user", "content": user_content(m.get("content") or "", m["media"], self.media_dir)})
            elif m.get("role") == "user" and m.get("media"):
                names = ", ".join(str(x.get("name") or x.get("kind") or "attachment") for x in m["media"] if isinstance(x, dict))
                txt = str(m.get("content") or "").strip()
                out.append({"role": "user", "content": (f"[Attached: {names}]\n{txt}" if names else txt).strip()})
            else:
                item = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id")}
                if strip_attached and item.get("role") == "user" and isinstance(item.get("content"), str):
                    marker = "\n\nUsing the attached content / session context above, answer the user's request clearly and thoroughly:\n"
                    if marker in item["content"]:
                        item["content"] = "[Earlier file attachment omitted]\n" + item["content"].split(marker, 1)[1]
                out.append(item)
        return out

    # ---------------- one Studio turn ----------------
    def run_turn(self, history, user_text, *, tools=True, sampling="general", thinking=True, max_new=2048,
                 max_rounds=8, on_event=None, ask=None, allow=(), cancel=None, seed=0, system=None,
                 layers=(), kit=None, layer_query=None, final_on_limit=False, force_answer=False, extra_tools=(),
                 media=(), only_tools=None, meaning=None):
        """Same contract as LFM3Native.run_turn. `thinking`, `sampling` and `force_answer` do not apply: the Omni
        does not reason and its card asks for greedy text decoding. media: [{"kind", "file"}] for this message.
        meaning: the turn's semantics.Frame (the user's paths, the search limit); None works it out from user_text."""
        emit = on_event or (lambda e: None)
        self.turn_cancel = cancel  # a running command stops when the user stops the turn
        self.turn_paths = list(meaning.paths) if meaning is not None else N.user_paths(user_text)
        self.turn_hosts = N.typed_hosts(meaning.text if meaning is not None else user_text)
        self.turn_unknown = set()
        self.turn_ran = set()
        self.turn_web, self.turn_web_cap = 0, (meaning.web_cap if meaning is not None else
                                               8 if "web_search" in N.asked_tools(user_text) else 3)
        base = [t for t in N.STUDIO_TOOLS if only_tools is None or t["function"]["name"] in only_tools]
        extra_tools = [t for t in extra_tools if only_tools is None or t["schema"]["function"]["name"] in only_tools]
        tools = tools and bool(base or extra_tools)
        self._extra = {t["schema"]["function"]["name"]: t for t in extra_tools} if tools else {}
        schemas = (base + [t["schema"] for t in extra_tools]) if tools else []
        by_name = {t["function"]["name"]: t["function"] for t in schemas}
        query = layer_query or user_text
        m_c1 = getattr(self, "display_model_name", None) or (
            self.describe().get("model") if hasattr(self, "describe") else None
        ) or "LFM3-Omni-2B"
        l_c1 = getattr(self, "display_layers_desc", None) or ""
        import workspaces as WS
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name=m_c1,
            query=query,
            offered_tools=only_tools if only_tools is not None else list(by_name)[:4],
        )
        self.active_op_plan = op_plan
        op_plan.start_stage("context")
        layer_log, blocks = [], []
        for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"), ("workspace", "workspace_context"),
                         ("records", "records_context")):
            if name in layers:
                block, info = getattr(kit, fn)(query)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
                    emit(dict(info, type="layer"))
        ws_active_p = (kit.workspaces.active() or {}).get("path") if (kit is not None and getattr(kit, "workspaces", None) is not None) else None
        l_blk, l_info = WS.local_context_for_paths(self.turn_paths, workdir=self.workdir, active_ws_path=ws_active_p)
        if l_blk:
            blocks.append(l_blk)
            layer_log.append(l_info)
            emit(dict(l_info, type="layer"))
        bridge_info = kit.compute_meaning_bridge(query) if (kit is not None and hasattr(kit, "compute_meaning_bridge")) else None
        if bridge_info:
            b_emit = {k: v for k, v in bridge_info.items() if k != "action_prior_8"}
            layer_log.append(b_emit)
            emit(dict(b_emit, type="layer"))
        if blocks:  # Clover appends its observations after the question
            user_text = user_text + "\n\n" + "\n\n".join(blocks)
            if not getattr(meaning, "caps", ()) and any(x.get("layer") in ("ontology", "memory") for x in layer_log):
                bridged_recs = (bridge_info or {}).get("recommended_tools") or []
                declined_set = set(getattr(meaning, "declined", ()) or ()) if meaning is not None else set()
                has_lrn_or_formula = bool(
                    bridge_info
                    and (
                        any(str(m_id).startswith("LRN-") for m_id in (bridge_info.get("memory_entries") or ()))
                        or any(str(e_id).startswith(("FORMULA:", "STEM_CONCEPT:")) for e_id in (bridge_info.get("ontology_entities") or ()))
                        or bool((bridge_info.get("curiosity") or {}).get("should_observe"))
                    )
                )
                # only tools the route allowed this turn: a note naming plan.md got read_file offered and read unasked in
                # Always, which leaves files out when the message names none (2026-11-09)
                allowed = set(by_name)
                active_bridged = [t for t in bridged_recs if t not in declined_set and t in allowed] if has_lrn_or_formula else []
                if active_bridged and tools:
                    schemas = [t for t in N.STUDIO_TOOLS if t["function"]["name"] in active_bridged]
                    by_name = {t["function"]["name"]: t["function"] for t in schemas}
                    only_tools = active_bridged
                else:
                    schemas = []
                    only_tools = None
        recall = []
        if "recall" in layers and kit is not None:  # background right before the message (memory-research exp G5)
            block, info = kit.recall_context(query, sum(1 for m in history if m.get("role") == "user"))
            if block:
                recall.append({"role": "system", "content": block})
            if info:
                layer_log.append(info)
                if info.get("refs"):
                    emit(dict(info, type="layer"))
        if meaning is not None and getattr(meaning, "guidance", ""):
            # the user's preferred form of answer (personal.py): fixed wording, no user text
            recall.append({"role": "system", "content": meaning.guidance})
        sys_txt = system or self.system_prompt(bool(schemas), only_tools)
        keep_last_media = bool(meaning is not None and getattr(meaning, "seen", ()) and not media)
        has_new_visual = any(m.get("kind") in ("image", "video") for m in media or () if isinstance(m, dict))
        has_new_audio = any(m.get("kind") == "audio" for m in media or () if isinstance(m, dict))
        not_comparing_prior = not re.search(r"\b(?:previous|earlier|first|prior|other|both|compare|same\s+as)\b", query or "", re.I)
        jepa_arch_cfg = getattr(kit, "jepa_arch", "auto") if kit is not None else "auto"
        if kit is not None:
            kit.visual_cue_64 = None
            kit.audio_cue_64 = None
        if has_new_audio and jepa_arch_cfg != "none":
            try:
                import jepa_engine
                for m in media or ():
                    if isinstance(m, dict) and m.get("kind") == "audio" and m.get("file"):
                        aud_fp = Path(self.media_dir) / m["file"]
                        if aud_fp.is_file():
                            ja = jepa_engine.for_base(self.base_dir).inspect_audio_latent(aud_fp, configured_arch=jepa_arch_cfg)
                            if ja.get("enabled"):
                                if kit is not None:
                                    kit.audio_cue_64 = ja.get("z_jepa")
                                info_aud = {
                                    "layer": "jepa_audio",
                                    "arch": ja["arch"],
                                    "file": m["file"],
                                    "acoustic_regime": ja["acoustic_regime"],
                                    "snr_db": ja["snr_db"],
                                    "tf_latent_cos": ja["tf_latent_cos"],
                                    "ms": ja["ms"],
                                }
                                layer_log.append(info_aud)
                                emit(dict(info_aud, type="layer"))
            except Exception:
                pass
        if has_new_visual:
            try:
                import image_inspector
                snap_blocks = []
                for m in media or ():
                    if isinstance(m, dict) and m.get("kind") == "image" and m.get("file"):
                        img_fp = Path(self.media_dir) / m["file"]
                        if img_fp.is_file():
                            rep = image_inspector.inspect_image(img_fp, annotate_dir=Path(self.media_dir), jepa_arch=jepa_arch_cfg)
                            snap_blocks.append(image_inspector.format_snapshot_for_prompt(rep))
                            jv = rep.get("jepa_visual") or {}
                            if kit is not None and jv.get("arch"):
                                try:
                                    import jepa_engine
                                    from PIL import Image
                                    with Image.open(img_fp) as _im:
                                        _jv_full = jepa_engine.for_base(self.base_dir).inspect_visual_latent(_im.convert("RGB"), configured_arch=jepa_arch_cfg)
                                        kit.visual_cue_64 = _jv_full.get("z_jepa")
                                except Exception:
                                    pass
                            info_img = {
                                "layer": "image_analysis",
                                "file": m["file"],
                                "annotated_file": rep.get("annotated_file"),
                                "width": rep["width"],
                                "height": rep["height"],
                                "elements": len(rep.get("elements") or []),
                                "ocr_lines": len(rep.get("ocr_lines") or []),
                                "jepa_arch": jv.get("arch"),
                                "jepa_scene": jv.get("scene_kind"),
                                "ms": rep.get("ms", 0),
                            }
                            layer_log.append(info_img)
                            emit(dict(info_img, type="layer"))
                if snap_blocks:
                    act_hdr = ""
                    try:
                        import device_actions
                        if device_actions.asks_for_screen_or_activity(query) or any(m.get("screen_capture") for m in media or () if isinstance(m, dict)):
                            act = device_actions.read_active_activity(
                                include_clipboard=device_actions.mentions_clipboard(query))
                            parts_a = []
                            if act.get("foreground_window"):
                                parts_a.append(f"- Active Foreground Window: {act['foreground_window']}")
                            if act.get("open_windows"):
                                parts_a.append("- Open Application Windows: " + "; ".join(act["open_windows"][:10]))
                            if act.get("clipboard_preview"):
                                parts_a.append(f"- Clipboard Text Preview: {act['clipboard_preview'][:400]}")
                            if parts_a:
                                act_hdr = "Live Device Activity Context:\n" + "\n".join(parts_a) + "\n\n"
                    except Exception:
                        pass
                    recall.append({
                        "role": "system",
                        "content": (
                            act_hdr
                            + "S3 Playwright-Style Visual & UI Inspection of attached image(s) "
                            "(use these exact OCR texts, [e1..eN] bounding boxes, dimensions, and hex colors together with what you see in the image):\n"
                            + "\n\n".join(snap_blocks)
                        ),
                    })
            except Exception:
                pass
        if has_new_visual and not_comparing_prior:
            past_msgs = []
            user_text = (user_text or "").rstrip() + "\n(Answer strictly from the image/video attached to this message.)"
        else:
            past_msgs = self._history(
                history,
                keep_last_media=keep_last_media,
                strip_attached=bool(media),
                current_query=query,
                is_follow_up=bool(meaning is not None and getattr(meaning, "follow_up", False)),
            )
        msgs = ([{"role": "system", "content": sys_txt}] + past_msgs + recall
                + [{"role": "user", "content": user_content(user_text, media, self.media_dir)}])
        rounds, final, stats_all = [], "", []
        done_calls = {}  # (name, args) -> first result: the Omni repeated identical searches 3 times in a test

        def delta_cb(round_no, kind="thinking"):
            shown = {"n": 0}

            def on_delta(delta, text):
                visible = TOOL_XML.sub("", text)
                if len(visible) > shown["n"]:
                    emit({"type": "token", "round": round_no, "kind": kind, "text": visible[shown["n"]:]})
                    shown["n"] = len(visible)
            return on_delta

        with self.lock:
            try:
                # One-time flow S1 -> S2 -> S3 -> C1 -> C2 -> C3:
                # When S1 (semantics) routes a deterministic S3 tool (web_search, fetch_url with URL,
                # get_current_time, laptop_status, remember), execute S3 once BEFORE C1 so C1 runs in a single pass.
                s3_pre_call = None
                if tools and schemas and only_tools and not (cancel and cancel.is_set()):
                    url_m = re.search(r"https?://\S+", query or "")
                    if "fetch_url" in only_tools and "fetch_url" in by_name and url_m:
                        u = url_m.group(0).rstrip(".,;)'>\"]")
                        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "fetch_url", "arguments": {"url": u}}
                    elif (
                        "web_search" in only_tools
                        and "web_search" in by_name
                        and set(only_tools) <= {"web_search", "fetch_url"}
                        and not url_m
                    ):
                        ws_q = query
                        if meaning is not None and getattr(meaning, "follow_up", False) and past_msgs:
                            prev_u = next(
                                (str(m.get("content") or "") for m in reversed(past_msgs)
                                 if m.get("role") == "user" and isinstance(m.get("content"), str)),
                                "",
                            )
                            if prev_u and len((query or "").split()) <= 6:
                                ws_q = f"{prev_u.strip()} {query.strip()}".strip()[:200]
                        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "web_search", "arguments": {"query": ws_q}}
                    elif list(only_tools) == ["get_current_time"] and "get_current_time" in by_name:
                        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "get_current_time", "arguments": {}}
                    elif list(only_tools) == ["laptop_status"] and "laptop_status" in by_name:
                        s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "laptop_status", "arguments": {}}
                    elif "run_python" in only_tools and "run_python" in by_name and re.search(r"```(?:python|py)\b", query or "", re.I):
                        import lfm3_layers
                        py_code = lfm3_layers.extract_python(query or "")
                        if py_code.strip():
                            s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "run_python", "arguments": {"code": py_code.strip()}}
                    elif list(only_tools) == ["remember"] and "remember" in by_name and meaning is not None and getattr(meaning, "remember", ""):
                        note = str(meaning.remember).strip()
                        if note:
                            s3_pre_call = {
                                "id": "call_" + uuid.uuid4().hex[:16],
                                "name": "remember",
                                "arguments": {"text": note[0].upper() + note[1:] + ("" if note.endswith(".") else ".")},
                            }
                    elif "vscode_open" in only_tools and "vscode_open" in by_name and not semantics.VSCODE.search(query or ""):
                        import device_actions
                        m_app = device_actions.OPEN_APP_ASK.match(query or "")
                        if m_app:
                            tgt = m_app.group(1).strip()
                            if device_actions.is_standalone_app_target(tgt):
                                s3_pre_call = {"id": "call_" + uuid.uuid4().hex[:16], "name": "vscode_open", "arguments": {"target": tgt}}
                    elif (
                        "read_file" in only_tools
                        and "read_file" in by_name
                        and "write_file" not in only_tools
                        and self.turn_paths
                        and self.perms.get().get("files") == "allow"
                    ):
                        cand_p = self._path(self.turn_paths[0])
                        if cand_p.is_file() and not paths.read_blocked(self.base_dir, cand_p):
                            s3_pre_call = {
                                "id": "call_" + uuid.uuid4().hex[:16],
                                "name": "read_file",
                                "arguments": {"path": str(cand_p), "start_line": 1, "end_line": 120},
                            }
                ctx_sources = [x.get("layer") for x in layer_log if isinstance(x, dict) and x.get("layer")]
                op_plan.complete_stage(
                    "context",
                    "Context established (" + (", ".join(dict.fromkeys(ctx_sources)) if ctx_sources else "AGENTS.md & conversation state") + ")",
                )
                op_plan.start_stage(
                    "direction",
                    "Direction set ("
                    + (f"pre-executing {s3_pre_call['name']}" if s3_pre_call else (
                        "tools: " + ", ".join(only_tools[:4]) if only_tools else "direct reasoning & verification"
                    ))
                    + ")",
                )
                op_plan.complete_stage("direction")
                op_plan.start_stage("execution")
                if s3_pre_call is not None:
                        emit({"type": "round", "round": 1})
                        emit({"type": "plan", "round": 1, "stage": "planning", "tools": [s3_pre_call["name"]]})
                        emit({"type": "plan", "round": 1, "stage": "preparing_tool", "tool": s3_pre_call["name"]})
                        emit({"type": "tool_call", "id": s3_pre_call["id"], "name": s3_pre_call["name"], "arguments": s3_pre_call["arguments"]})
                        key = (s3_pre_call["name"], json.dumps(s3_pre_call["arguments"], sort_keys=True))
                        ran, result, why = self.execute_tool(s3_pre_call["name"], s3_pre_call["arguments"], ask, allow)
                        done_calls[key] = result
                        worked, summary = N.tool_worked(s3_pre_call["name"], ran, result, why)
                        emit({
                            "type": "tool_result",
                            "id": s3_pre_call["id"],
                            "name": s3_pre_call["name"],
                            "ran": ran,
                            "worked": worked,
                            "summary": summary,
                            "why": why,
                            "result": result[:3000],
                        })
                        rounds.append({
                            "tool": s3_pre_call["name"],
                            "arguments": s3_pre_call["arguments"],
                            "ran": ran,
                            "worked": worked,
                            "summary": summary,
                            "why": why,
                            "result": result[:2000],
                        })
                        msgs.insert(
                            -1,
                            {
                                "role": "system",
                                "content": (
                                    f"Live {s3_pre_call['name']} results for this request (already executed in S3 — "
                                    f"answer directly from these real results in plain text without calling tools):\n{result}"
                                ),
                            },
                        )
                        msgs[0]["content"] = system or self.system_prompt(False, None)
                        schemas = []
                        only_tools = None

                def _auto_verify_py(code_str):
                    if cancel and cancel.is_set():
                        return False, "", False
                    if self.perms.get().get("code") == "deny" and "code" not in (allow or ()) and "run_python" not in (allow or ()):
                        return False, "", True
                    cid = "call_" + uuid.uuid4().hex[:16]
                    args_py = {"code": code_str}
                    emit({"type": "tool_call", "id": cid, "name": "run_python", "arguments": args_py})
                    if self.perms.get().get("code") == "allow" or "code" in (allow or ()) or "run_python" in (allow or ()):
                        ran, result, why = self.execute_tool("run_python", args_py, ask, allow)
                    else:
                        result = self._run_tool("run_python", {"code": code_str})
                        ran, why = True, "Clover Cycle Outcome sandbox verification"
                    worked, summary = N.tool_worked("run_python", ran, result, why)
                    emit({"type": "tool_result", "id": cid, "name": "run_python", "ran": ran, "worked": worked,
                          "summary": summary, "why": why, "result": result[:3000]})
                    rounds.append({"tool": "run_python", "arguments": args_py, "ran": ran, "worked": worked,
                                   "summary": summary, "why": why, "result": result[:2000]})
                    return ran, result, worked

                first_cycle_findings = None
                c1_first_failed = None
                max_clover_cycles = 3
                cycles_used = 1
                for attempt in range(max_clover_cycles):
                    cycles_used = attempt + 1
                    for r in range(max_rounds):
                        if not (r == 0 and attempt == 0 and s3_pre_call is not None):
                            emit({"type": "round", "round": r + 1})
                        if schemas and getattr(self, "turn_web", 0) >= getattr(self, "turn_web_cap", 99):
                            # past the search limit the Omni kept calling refused searches, then wrote a Python scraper of
                            # invented URLs for 5 minutes (2026-11-01): no tools for the rest of this answer
                            schemas = []
                        if (
                            schemas
                            and r >= 1
                            and set(by_name) <= {"web_search", "fetch_url"}
                            and any(rd["tool"] == "web_search" and rd["ran"] and rd.get("worked") for rd in rounds)
                            and not re.search(r"https?://|www\.|\b(?:compare|corroborate|both|each)\b", query or "", re.I)
                        ):
                            # A single successful web_search already returned the top 5 snippets/headlines; removing
                            # schemas in Round 1 prevents the 2B model from looping across 8 rounds of fetch_url calls.
                            schemas = []
                        if schemas and sum(1 for rd in rounds if rd["why"] == "same call already made in this answer") >= 2:
                            # "good vegetarian restaurants in Jaipur" repeated two searches six times, to the round limit
                            schemas = []
                        # asked for a tool (only_tools set): told to use it, the Omni still answered from memory
                        forced = "required" if only_tools and schemas and r == 0 and attempt == 0 else None
                        if r == 0 and attempt == 0 and only_tools:
                            emit({"type": "plan", "round": r + 1, "stage": "planning", "tools": list(only_tools)})
                        elif r > 0:
                            emit({"type": "plan", "round": r + 1, "stage": "reviewing"})
                        emit({"type": "plan", "round": r + 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
                        # C1 is strictly for inference only: never emit final answer tokens (`kind="answer"`) from C1
                        chat_kw = ({"on_plan": lambda tname, rn=r + 1: emit({"type": "plan", "round": rn, "stage": "preparing_tool", "tool": tname})}
                                   if schemas else {})
                        try:
                            content, calls, st = self._chat(
                                msgs, schemas, max_new, by_name,
                                None, cancel,
                                seed + 100 * attempt + r, forced,
                                **chat_kw,
                            )
                        except TypeError:
                            content, calls, st = self._chat(
                                msgs, schemas, max_new, by_name,
                                None, cancel,
                                seed + 100 * attempt + r, forced,
                            )
                        stats_all.append(st)
                        if forced and calls and ASKS_USER.search(content[-400:]) and any(
                            c["name"] in ("run_powershell", "write_file", "vscode_run", "vscode_install", "vscode_clone", "vscode_open")
                            for c in calls
                        ):
                            calls = []  # the model asks the user which command/app to run: its question is the answer
                        if r == 0 and attempt == 0 and not calls:
                            if "web_search" in by_name and ("web_search" in (only_tools or ()) or semantics.CANNOT.search(content)):
                                calls = [{"id": "call_" + uuid.uuid4().hex[:16], "name": "web_search", "arguments": {"query": query}}]
                                content = ""
                            elif (
                                semantics.CANNOT.search(content)
                                and "web.search" not in getattr(meaning, "declined", ())
                                and not media
                                and not getattr(meaning, "seen", ())
                                and getattr(meaning, "act", "") not in ("creative", "social")
                                and not blocks
                            ):
                                ws_t = next((t for t in N.STUDIO_TOOLS if t["function"]["name"] == "web_search"), None)
                                if ws_t is not None:
                                    schemas.append(ws_t)
                                    by_name["web_search"] = ws_t["function"]
                                    tools = True
                                    calls = [{"id": "call_" + uuid.uuid4().hex[:16], "name": "web_search", "arguments": {"query": query}}]
                                    content = ""
                        if calls and r == 0 and meaning is not None and not getattr(meaning, "follow_up", False) and history:
                            import lfm3_layers
                            q_topics = lfm3_layers.topic_terms(query)
                            prev_u = " ".join(str(m.get("content") or "") for m in history[-4:] if m.get("role") == "user")
                            prev_topics = lfm3_layers.topic_terms(prev_u) - q_topics
                            if len(q_topics) >= 1 and prev_topics:
                                for c in calls:
                                    if c.get("name") == "web_search" and isinstance(c.get("arguments"), dict):
                                        cq = str(c["arguments"].get("query") or "")
                                        ct = lfm3_layers.topic_terms(cq)
                                        if not (ct & q_topics) and (ct & prev_topics):
                                            c["arguments"] = {"query": query}
                        names = [c["name"] for c in calls]
                        if calls and semantics.answered_unasked(meaning, TOOL_XML.sub("", content), names):
                            rt = meaning.runtime
                            used = ", ".join(dict.fromkeys(rt.word(rt.capability(n)) for n in names))
                            layer_log.append({"layer": "semantics", "label": f"answer kept · {used} not used (not asked)"})
                            content, calls = TOOL_XML.sub("", content), []
                        if calls and tools and schemas and st["finish"] != "cancelled":
                            kept_c = "" if (forced or semantics.CANNOT.search(content)) else content
                            msgs.append({"role": "assistant", "content": kept_c, "tool_calls": [
                                {"id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]})
                            for c in calls:
                                emit({"type": "tool_call", "id": c["id"], "name": c["name"], "arguments": c["arguments"]})
                                key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
                                if key in done_calls:
                                    ran, why = False, "same call already made in this answer"
                                    result = ("You already made this exact call in this answer; its result is above. Do not "
                                              "repeat it: answer from the results you have, or try something different.")
                                else:
                                    ran, result, why = self.execute_tool(c["name"], c["arguments"], ask, allow)
                                    done_calls[key] = result
                                worked, summary = N.tool_worked(c["name"], ran, result, why)
                                summary = N.checked_summary(self, ran, summary)
                                emit({"type": "tool_result", "id": c["id"], "name": c["name"], "ran": ran, "worked": worked,
                                      "summary": summary, "why": why, "result": result[:3000]})
                                rounds.append({"tool": c["name"], "arguments": c["arguments"], "ran": ran, "worked": worked,
                                               "summary": summary, "why": why, "result": result[:2000]})
                                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                                op_plan.advance_execution_step(c["name"])
                            continue
                        final = TOOL_XML.sub("", content).strip() if forced else content.strip()
                        web_ok = any(rd["tool"] == "web_search" and rd["ran"] and not rd["result"].startswith("No results") for rd in rounds)
                        refused_after_web = web_ok and (
                            semantics.CANNOT.search(final)
                            or bool(re.search(r"don'?t know where you are|need to know which city|tell me your city|your city or location", final, re.I))
                        )
                        if (semantics.non_answer(final) or refused_after_web) and rounds and \
                                st["finish"] not in ("length", "cancelled") and not (cancel and cancel.is_set()):
                            # with its tools gone the Omni still wrote a call, which is stripped, and the answer was
                            # empty: "(Clover finished without writing an answer.)" in a trip-planning chat (2026-11-01)
                            msgs.append({"role": "user", "content": "Answer now in plain text directly using the exact "
                                         "facts and numbers in the tool results above; do not ask for the user's "
                                         "location or call tools."})
                            lead = "Here is what I can tell you: "
                            msgs.append({"role": "assistant", "content": lead})
                            emit({"type": "round", "round": r + 2})
                            emit({"type": "plan", "round": r + 2, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
                            content, _, st = self._chat(msgs, [], max_new, by_name, None, cancel,
                                                        seed + 100 * attempt + r + 1)
                            stats_all.append(st)
                            final = (lead + content.strip()) if content.strip() else ""
                        if not final:
                            final = {"length": "(Clover ran out of room before answering. Ask for a shorter answer.)",
                                     "cancelled": "(Stopped.)"}.get(st["finish"], "(Clover finished without writing an answer.)")
                        break
                    else:
                        if final_on_limit and not (cancel and cancel.is_set()):
                            msgs.append({"role": "user", "content": "The tool budget for this question is used up. Answer "
                                         "now from the results you already have; do not call more tools."})
                            # an assistant message last is continued by llama-server, so the answer starts as text
                            lead = "From what I found so far: "
                            msgs.append({"role": "assistant", "content": lead})
                            emit({"type": "round", "round": max_rounds + 1})
                            emit({"type": "plan", "round": max_rounds + 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
                            content, _, st = self._chat(msgs, [], max_new, by_name, None, cancel,
                                                        seed + 100 * attempt + max_rounds)
                            stats_all.append(st)
                            final = (lead + content.strip()) if content.strip() else ""
                        last = next((rd for rd in reversed(rounds) if rd["ran"]), None)
                        final = final or (f"(Clover used all {max_rounds} tool rounds without finishing an answer."
                                          + (f" Last {last['tool']} result: {last['result'][:400]}" if last else "") + ")")

                    # ---- Clover Cycle Outcome Stage (`Context -> Direction -> Execution -> Outcome -> Growth`) ----
                    if cancel and cancel.is_set():
                        break
                    op_plan.complete_stage("execution")
                    op_plan.start_stage("outcome", f"Verifying Cycle {attempt + 1} output against reality…", cycle=attempt + 1)
                    findings, c_meta = C.verify_turn_outcome(
                        query,
                        final,
                        rounds,
                        kit=kit,
                        layers=layers,
                        meaning=meaning,
                        execute_python_fn=_auto_verify_py,
                    )
                    comp_info = c_meta.get("compiler_info")
                    if comp_info is not None:
                        layer_log.append(comp_info)
                        if comp_info.get("checked"):
                            if c_meta.get("compiler_ok"):
                                emit({"type": "layer", "layer": "compiler", "checked": True, "ok": True})
                            else:
                                emit({"type": "layer", "layer": "compiler", "checked": True, "ok": False, "feedback": findings[0]})

                    if not findings:
                        op_plan.complete_stage("outcome", C.outcome_label(c_meta["checks"], attempt + 1))
                        layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": True})
                        if first_cycle_findings:
                            only_comp = all(f.startswith("Code compiler check:") for f in first_cycle_findings)
                            emit({"type": "cycle", "cycle": attempt + 1, "stage": "verified", "cycles_used": attempt + 1})
                            layer_log.append({
                                "layer": "cycle",
                                "trigger": "auto",
                                "action": "compiler_repaired" if only_comp else "corrected",
                                "cycles_used": attempt + 1,
                                "first_findings": first_cycle_findings,
                                "final_findings": [],
                                "tools": [rd["tool"] for rd in rounds if rd.get("ran") and rd.get("worked")],
                                "lesson": C.lesson_text(query, [f[:240] for f in first_cycle_findings[:2]], "", passed=True),
                                "lesson_kind": "verified",
                            })
                        break

                    if first_cycle_findings is None:
                        first_cycle_findings = list(findings)
                        if kit is not None and hasattr(kit, "c1_infer"):
                            try:
                                c1_first_failed = kit.c1_infer(final)[0]
                            except Exception:
                                c1_first_failed = None

                    fixed = C.direct_arithmetic_answer(query) if C.only_arithmetic(findings) else None
                    if fixed:  # a plain sum: Clover has the exact value, so it writes the answer itself
                        emit({"type": "cycle", "cycle": attempt + 2, "stage": "retry", "findings": findings})
                        final = fixed
                        emit({"type": "token", "round": 1, "kind": "answer", "text": final})
                        op_plan.complete_stage("outcome", C.outcome_label(c_meta["checks"], attempt + 1))
                        layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": True})
                        layer_log.append({"layer": "cycle", "trigger": "auto", "action": "corrected", "cycles_used": attempt + 1,
                                          "first_findings": list(findings), "final_findings": [], "tools": [],
                                          "lesson": C.lesson_text(query, [findings[0][:240]], "", passed=True),
                                          "lesson_kind": "verified"})
                        emit({"type": "cycle", "cycle": attempt + 1, "stage": "verified", "cycles_used": attempt + 1})
                        break

                    if attempt + 1 < max_clover_cycles:
                        emit({"type": "cycle", "cycle": attempt + 2, "stage": "retry", "findings": findings})
                        op_plan.retry_from_context(attempt + 2, findings[0])
                        if any(f.startswith(("Arithmetic:", "Code execution check:", "Code output check:", "Tool execution check:")) for f in findings):
                            if self.perms.get().get("code") != "deny" or "code" in (allow or ()):
                                for t_cand in N.STUDIO_TOOLS:
                                    t_fn = t_cand["function"]["name"]
                                    if t_fn in ("run_python", "calculate") and t_fn not in by_name:
                                        schemas.append(t_cand)
                                        by_name[t_fn] = t_cand["function"]
                                        tools = True
                        msgs.append({"role": "assistant", "content": final})
                        msgs.append({"role": "user", "content": C.AUTO_CYCLE_FIX.format(
                            findings="\n".join("- " + f for f in findings)
                        )})
                    else:
                        op_plan.complete_stage("outcome", f"Checked across {attempt + 1} cycles (unconfirmed)")
                        layer_log.append({"layer": "outcome", "checks": list(c_meta["checks"]), "passed": False})
                        layer_log.append({
                            "layer": "cycle",
                            "trigger": "auto",
                            "action": "unconfirmed",
                            "cycles_used": attempt + 1,
                            "first_findings": first_cycle_findings or findings,
                            "final_findings": findings,
                        })

                op_plan.start_stage("growth", "Aligning epistemic state & recording learned knowledge…")
                rnd_n = len(stats_all) or 1
                if kit is not None and not (cancel and cancel.is_set()):
                    # 1. Strict Stage C1: Inference ONLY (`c1_infer`)
                    if hasattr(kit, "c1_infer"):
                        try:
                            c1_state, c1_info = kit.c1_infer(final, stats=stats_all, model=m_c1, layers_desc=l_c1)
                        except TypeError:
                            c1_state, c1_info = kit.c1_infer(final, stats=stats_all)
                            c1_info["model"] = m_c1
                            if l_c1:
                                c1_info["layers_desc"] = l_c1
                        layer_log.append(c1_info)
                        emit(dict(c1_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c1_done", "ms": c1_info.get("ms")})
                    else:
                        c1_state = None
                    # 2. Strict Stage C2: Planning ONLY (`HybridAgent` `plan-trunk-0`)
                    if hasattr(kit, "c2_plan_state"):
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_plan", "model": "HybridAgent"})
                        planned_t = [rd["tool"] for rd in rounds if rd.get("ran")] or (
                            list(only_tools) if only_tools is not None else [t["function"]["name"] for t in schemas[:5]]
                        )
                        if c1_first_failed is not None and cycles_used > 1 and hasattr(kit, "jepa_record_outcome"):
                            try:
                                kit.c2_plan_state(query, c1_state=c1_first_failed, offered_tools=planned_t)
                                kit.jepa_record_outcome(c1_obs_state=c1_state, is_baseline=False)
                            except Exception:
                                pass
                        c2_state, c2_info = kit.c2_plan_state(query, c1_state=c1_state, offered_tools=planned_t)
                        layer_log.append(c2_info)
                        emit(dict(c2_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_done", "ms": c2_info.get("ms")})
                        if hasattr(kit, "jepa_record_outcome") and c1_first_failed is None:
                            kit.jepa_record_outcome(c1_obs_state=c1_state, is_baseline=True)
                    else:
                        c2_state = final
                    # 3. Strict Stage C3: Decoding ONLY (`Bridges + 11-Level NCM`)
                    if "ncm" in layers and (hasattr(kit, "c3_decode") or hasattr(kit, "ncm_render")):
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_decode", "model": "11-Level NCM"})
                        final, ncm_info = kit.c3_decode(c2_state) if hasattr(kit, "c3_decode") else kit.ncm_render(final)
                        layer_log.append(ncm_info)
                        emit(dict(ncm_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_done",
                              "ms": round((ncm_info.get("s") or 0) * 1000)})
                if "ontology" in layers and kit is not None:
                    final, changed = kit.grounding_filter(query, final)
                    layer_log.append({"layer": "ontology_grounding", "changed": changed})
                mq = TRAILING_Q.search(final) if len(final) > 60 else None
                if mq and mq.start() >= 40:
                    final = final[:mq.start()].strip()
                if not (cancel and cancel.is_set()):
                    op_plan.complete_stage(
                        "growth",
                        "Growth preserved ("
                        + (f"repaired in {cycles_used} cycles · lesson saved" if first_cycle_findings else "context updated")
                        + ")",
                    )
                    emit({"type": "token", "round": rnd_n, "kind": "answer", "text": final})
            finally:
                self.stage = {"name": "idle", "since": time.time()}
        return {"reply": final, "tool_calls": rounds, "rounds": len(stats_all), "generation": stats_all,
                "reasoning": "", "cancelled": bool(cancel and cancel.is_set()), "layers": layer_log,
                "todo_list": op_plan.todos}

    # ---------------- the Clover cycle, when the user says the answer is not right ----------------
    def retry_turn(self, history, question, previous, feedback, *, media=(), layers=(), kit=None, on_event=None,
                   cancel=None, seed=0, max_new=2048, system=None, ask=None, allow=(), context=""):
        """clover_cycle.retry_cycle on this engine. history: the conversation, ending with `previous` (the answer the
        user says is not right); question: the message that answer is about, media: that message's attachments;
        feedback: what the user said; context: the exchange before a follow-up question (clover_cycle.context_before),
        put before the question in the independent solutions. Tools (web research, Python) go through the user's
        permissions (`ask` pauses for a decision). Returns the run_turn result shape."""
        emit = on_event or (lambda e: None)
        active_model_name = getattr(self, "display_model_name", None) or (
            self.describe().get("model") if hasattr(self, "describe") else None
        ) or "LFM3-Omni-2B"
        op_plan = C.OperatingCyclePlan(
            emit,
            model_name=active_model_name,
            query=f"{question}\n{feedback}",
            offered_tools=["run_python", "web_search"],
        )
        op_plan.start_stage("context", "Re-reading Context: original task, prior attempt & user feedback…")
        layer_log, blocks, stats_all, rounds, used = [], [], [], [], []
        for name, fn in (("memory", "memory_context"), ("ontology", "ontology_context"), ("workspace", "workspace_context"),
                         ("records", "records_context")):
            if name in layers and kit is not None:  # Context gathered again for the question
                block, info = getattr(kit, fn)(question)
                if block:
                    blocks.append(block)
                    layer_log.append(info)
        op_plan.complete_stage("context", "Context established (original question, user correction & learned lessons)")
        op_plan.start_stage("direction", f"Direction set (Clover repair cycle · {active_model_name})")
        op_plan.complete_stage("direction")
        op_plan.start_stage("execution", f"Executing Clover repair cycle with {active_model_name}…")
        sys_txt = system or self.system_prompt(False)
        convo = [{"role": "system", "content": sys_txt}] + self._history(history)
        start = len(convo)
        notes = web = ""

        def tool(name, args, run):
            cid = "call_" + uuid.uuid4().hex[:16]
            emit({"type": "tool_call", "id": cid, "name": name, "arguments": args})
            ran, result, why = run()
            emit({"type": "tool_result", "id": cid, "name": name, "ran": ran, "why": why, "result": result[:3000]})
            rounds.append({"tool": name, "arguments": args, "ran": ran, "why": why, "result": result[:2000]})
            return ran, result

        def rewrite(instruction):  # the model sees the conversation and each earlier attempt of this retry
            if cancel and cancel.is_set():
                return previous
            m_c1 = getattr(self, "display_model_name", None) or (
                self.describe().get("model") if hasattr(self, "describe") else None
            ) or "LFM3-Omni-2B"
            l_c1 = getattr(self, "display_layers_desc", None) or ""
            convo.append({"role": "user", "content": instruction + (notes + web if len(convo) == start else "")})
            emit({"type": "plan", "round": 1, "stage": "c1_inference", "model": m_c1, "layers_desc": l_c1})
            content, _, st = self._chat(convo, [], max_new, {}, None, cancel, seed + 500 + len(convo))
            stats_all.append(st)
            new = TOOL_XML.sub("", content).strip() or previous
            convo.append({"role": "assistant", "content": new})
            return new

        def solve(k, note=None):  # the question alone: the earlier answer is not shown; web results reach solve 2 only
            if cancel and cancel.is_set():
                return ""
            ctx = notes + (web if k == 2 else "")
            m2 = [{"role": "system", "content": sys_txt},
                  {"role": "user", "content": user_content(C.second_solve_prompt(context + question + ctx, question, k, note),
                                                           media, self.media_dir)}]
            content, _, st = self._chat(m2, [], max_new, {}, None, cancel, seed + 900 + k)
            stats_all.append(st)
            return content

        def python_solve(note):
            """A third solution for a number, as a program Clover runs: '' when code may not run, the program only
            writes down a value (no calculation), or it gives no value."""
            if self.perms.get()["code"] == "deny" or (cancel and cancel.is_set()):
                return ""
            emit({"type": "cycle", "stage": "python"})
            m2 = [{"role": "system", "content": sys_txt},
                  {"role": "user", "content": user_content(C.python_solve_prompt(context + question + notes, note), media,
                                                           self.media_dir)}]
            content, _, st = self._chat(m2, [], max_new, {}, None, cancel, seed + 950)
            stats_all.append(st)
            import lfm3_layers
            code = lfm3_layers.extract_python(content)
            if not code or not C.computes(code) or (cancel and cancel.is_set()):
                return ""
            ran, out = tool("run_python", {"code": code}, lambda: self.execute_tool("run_python", {"code": code}, ask, allow))
            value = C.python_answer(out) if ran else None
            if value is None:
                return ""
            used.append("a Python calculation")
            shown = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
            return f"Clover ran this Python program:\n```python\n{code.strip()}\n```\nIt printed: {out.strip()[-300:]}\n\nANSWER: {shown}"

        def code_check(ans):
            if "compiler" in layers and kit is not None:
                fb, info = kit.compiler_check(question, ans)
                layer_log.append(info)
                if fb:
                    return fb
            q_comb = f"{question}\n{feedback}" if feedback else question

            def _retry_verify_py(c_str):
                if self.perms.get().get("code") == "deny" or (cancel and cancel.is_set()):
                    return False, "", True
                ran_p, out_p = tool("run_python", {"code": c_str}, lambda: (True, self._run_tool("run_python", {"code": c_str}), "retry sandbox check"))
                worked_p, _ = N.tool_worked("run_python", ran_p, out_p, "retry sandbox check")
                if ran_p:
                    used.append("a Python calculation")
                return ran_p, out_p, worked_p

            v_findings, _ = C.verify_turn_outcome(
                q_comb, ans, rounds, kit=kit, layers=layers, execute_python_fn=_retry_verify_py
            )
            code_f = [f for f in v_findings if f.startswith(("Code ", "Outcome check:"))]
            return "\n".join(code_f) if code_f else None

        with self.lock:
            try:
                notes = ("\n\n" + "\n\n".join(blocks)) if blocks else ""  # Clover's records, after the request
                if not blocks and C.needs_web(context + question) and not (cancel and cancel.is_set()):
                    q = C.web_query(question)  # Context it did not have: search the web, read the best pages
                    emit({"type": "cycle", "stage": "search"})
                    ran, found = tool("web_search", {"query": q}, lambda: self.research_web(q, ask, allow))
                    if ran and found.startswith("Searched "):  # one vote at most: a page can mislead (measured)
                        web = ("\n\nClover searched the web for this question. Use these results only where they "
                               "answer it; they can be incomplete or wrong:\n" + found)
                        layer_log.append({"layer": "web", "query": q, "chars": len(found)})
                        used.append("a web search")
                answer, info = C.retry_cycle(question, previous, feedback, rewrite, solve=solve, emit=emit, used=used,
                                             program=python_solve,
                                             code_check=code_check)
                op_plan.complete_stage("execution")
                op_plan.start_stage("outcome", "Verifying repaired answer against reality…")
                op_plan.complete_stage("outcome", f"Outcome action: {info.get('action', 'corrected')}")
                op_plan.start_stage("growth", "Recording lesson to Clover records for future turns…")
                info["tools"] = list(dict.fromkeys(used))
                layer_log.append(info)
                rnd_n = len(stats_all) or 1
                if kit is not None and not (cancel and cancel.is_set()):
                    if hasattr(kit, "c1_infer"):
                        c1_prev, _ = kit.c1_infer(previous or question)
                        c1_state, c1_info = kit.c1_infer(answer, stats=stats_all)
                        layer_log.append(c1_info)
                        emit(dict(c1_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c1_done", "ms": c1_info.get("ms")})
                    else:
                        c1_prev, c1_state = None, None
                    if hasattr(kit, "c2_plan_state"):
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_plan", "model": "HybridAgent"})
                        _, _ = kit.c2_plan_state(question, c1_state=c1_prev, offered_tools=info["tools"])
                        if hasattr(kit, "jepa_record_outcome"):
                            j_out = kit.jepa_record_outcome(c1_obs_state=c1_state, is_baseline=False)
                            if isinstance(j_out, dict) and j_out.get("enabled"):
                                layer_log.append({"layer": "jepa_outcome_growth", **j_out})
                                emit({"type": "layer", "layer": "jepa_outcome_growth", **j_out})
                        c2_state, c2_info = kit.c2_plan_state(question, c1_state=c1_state, offered_tools=info["tools"])
                        layer_log.append(c2_info)
                        emit(dict(c2_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "c2_done", "ms": c2_info.get("ms")})
                    else:
                        c2_state = answer
                    if "ncm" in layers and (hasattr(kit, "c3_decode") or hasattr(kit, "ncm_render")):
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_decode", "model": "11-Level NCM"})
                        answer, ncm_info = kit.c3_decode(c2_state) if hasattr(kit, "c3_decode") else kit.ncm_render(answer)
                        layer_log.append(ncm_info)
                        emit(dict(ncm_info, type="layer"))
                        emit({"type": "plan", "round": rnd_n, "stage": "ncm_done",
                              "ms": round((ncm_info.get("s") or 0) * 1000)})
                if "ontology" in layers and kit is not None:
                    answer, changed = kit.grounding_filter(question, answer)
                    layer_log.append({"layer": "ontology_grounding", "changed": changed})
                if not (cancel and cancel.is_set()):
                    op_plan.complete_stage("growth", "Growth preserved · lesson written to Clover records")
                    emit({"type": "token", "round": rnd_n, "kind": "answer", "text": answer})
            finally:
                self.stage = {"name": "idle", "since": time.time()}
        cancelled = bool(cancel and cancel.is_set())
        return {"reply": "(Stopped.)" if cancelled else answer, "tool_calls": rounds, "rounds": len(stats_all),
                "generation": stats_all, "reasoning": "", "cancelled": cancelled, "layers": layer_log,
                "todo_list": op_plan.todos}

    # ---------------- VS Code (OpenAI messages + VS Code's tools) ----------------
    def vscode_step(self, user_prompt, workspace_info, prior_turns, tool_rounds, vs_tools, thinking=True, max_new=1024,
                    media=()):
        schemas, by_name = [], {}
        active_core = N.active_vscode_tools(user_prompt, workspace_info, tool_rounds)
        for t in vs_tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            name = str(fn.get("name") or "").strip()
            if not name or name == "vscode_askQuestions" or name in by_name:
                continue
            f = {"name": name, "description": str(fn.get("description") or "").split("\n")[0][:240],
                 "parameters": N.vscode_tool_params(name, fn.get("parameters") or fn.get("inputSchema")
                                                    or {"type": "object", "properties": {}})}
            by_name[name] = f
            entry = {"type": "function", "function": f}
            if name not in active_core:
                entry["defer_loading"] = True
            schemas.append(entry)
        deferred = [s["function"] for s in schemas if s.get("defer_loading")]
        if deferred:
            st = N._fn("search_tools", "Search the catalog of more VS Code tools by keywords. Returns tools you can then call.",
                       {"query": {"type": "string"}}, ["query"])
            schemas.insert(0, st)
            by_name["search_tools"] = st["function"]
        m_root = re.search(r"-\s*([A-Za-z]:\\[^\r\n]+|/[^\r\n]+)", workspace_info or "")
        root = m_root.group(1).strip() if m_root else ""
        sys_txt = ("You are Clover, a coding assistant inside VS Code, running on the user's laptop (model: LFM3-Omni-2B). "
                   "Inspect the workspace with the tools before answering; never guess file contents. "
                   + (f"The workspace root is {root}; use absolute paths. " if root else "")
                   + ("If none of your tools fits, call search_tools to find one. " if deferred else ""))
        msgs = [{"role": "system", "content": sys_txt}]
        for h in prior_turns[-3:]:
            msgs += [{"role": "user", "content": str(h["user"])[:1500]}, {"role": "assistant", "content": str(h["reply"])[:1500]}]
        ws = f"<workspace_info>\n{workspace_info[:1500]}\n</workspace_info>\n" if workspace_info else ""
        msgs.append({"role": "user", "content": user_content(ws + user_prompt[:6000], media, self.media_dir)})
        for rd in tool_rounds[-8:]:
            cid = rd.get("id") or ("call_" + uuid.uuid4().hex[:12])
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": rd["name"], "arguments": rd.get("arguments") or "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": N._clip(rd.get("output") or "", 5000)})
        trace = []
        nudged, answer = False, ""
        with self.lock:
            try:
                for _ in range(4):
                    content, calls, st = self._chat(msgs, schemas, max_new, by_name)
                    trace.append({"generation": st, "calls": [c["name"] for c in calls], "text": content[-400:]})
                    if calls and calls[0]["name"] == "search_tools":
                        q = str(calls[0]["arguments"].get("query") or "").lower().split()
                        # offered tools too: a search for the Playwright tool it already had found nothing, four times
                        pool = [f for n, f in by_name.items() if n != "search_tools"]
                        scored = sorted(pool, key=lambda f: -sum(w in (f["name"] + " " + f["description"]).lower() for w in q))
                        found = [f["name"] for f in scored[:3] if any(w in (f["name"] + " " + f["description"]).lower() for w in q)]
                        msgs.append({"role": "assistant", "content": content, "tool_calls": [
                            {"id": calls[0]["id"], "type": "function",
                             "function": {"name": "search_tools", "arguments": json.dumps(calls[0]["arguments"])}}]})
                        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content":
                                     f"Found {len(found)} tools:\n" + "\n".join(json.dumps(by_name[n]) for n in found)})
                        continue
                    if calls:
                        c = calls[0]
                        refused = N.unshared_open(c["name"], c["arguments"], "\n".join(
                            [user_prompt] + [str(h.get("user") or "") for h in prior_turns[-3:]]), workspace_info, tool_rounds)
                        if refused:
                            trace[-1]["refused"] = str(c["arguments"].get("url"))[:200]
                            msgs.append({"role": "assistant", "content": content, "tool_calls": [
                                {"id": c["id"], "type": "function",
                                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}]})
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": refused})
                            continue
                        read = N.shared_page_read(c["name"], c["arguments"], user_prompt, workspace_info, by_name)
                        if read:
                            trace[-1]["read_instead"] = True
                            return {"tool_call": {"name": read["name"], "arguments": json.dumps(read["arguments"])}, "trace": trace}
                        c_args = N.repair_vscode_browser_args(c["name"], c["arguments"], workspace_info, tool_rounds)
                        if nudged and N.repeats_round(c["name"], c_args, tool_rounds):
                            return {"content": answer or "(Clover did not finish an answer in time.)", "trace": trace}
                        return {"tool_call": {"name": c["name"], "arguments": json.dumps(c_args)}, "trace": trace}
                    nudge = "" if nudged else N.unfinished_page_steps(user_prompt, workspace_info, tool_rounds, by_name)
                    if nudge:
                        nudged = trace[-1]["nudged"] = True
                        answer = content.strip()
                        msgs += [{"role": "assistant", "content": content}, {"role": "user", "content": nudge}]
                        continue
                    return {"content": content.strip() or "(Clover did not finish an answer in time.)", "trace": trace}
                return {"content": "(Clover kept searching for tools without choosing one.)", "trace": trace}
            finally:
                self.stage = {"name": "idle", "since": time.time()}
