"""Clover layers 3-6 as add-ons to LFM3 alone (Layer 1 + native tools), each one switchable on its own.

Each layer is Clover's own component, unchanged; only the glue to LFM3's chat loop is new:
  memory   (L4) retrieve_relevant_memory_package (its own precision gate) + render_for_k3(compact=True), appended
                after the question with Clover's instruction, as run_agent_planning appends tool observations.
  ontology (L5) CloverOntologyRetriever.retrieve(q, max_entities=4, max_paths=6) + render_evidence, appended the same
                way; CloverSymbolicVerifier.verify_source_grounding on the final reply, as agent_reply does.
                Not used: solve_symbolic_math_constraints and verify_mcq_with_ontology. Their rules match specific test
                questions (house flip, lemon tree, round trip, thin film 600 nm, carrying capacity...): answer keys.
  compiler (L6) CloverPythonGenerativeCompiler.check_prefix(code, end_marker=True, spec_prompt=q) on the Python in the
                final reply (syntax, names, call arity, the prompt's own >>> examples). Its feedback goes back to LFM3
                for up to two repair rounds. Clover checks prefixes while decoding; LFM3's loop checks the finished code.
  ncm      (L3) Lfm3NCMRenderer.render_plans_in_model_vocab(windows of the final reply, cands=1), the renderer Clover's
                chat server uses for LFM3; code blocks pass through unchanged, as in the renderer.
App mode (memory_root given): the user's knowledge is growth/knowledge in Clover's repository: notes they asked Clover
to remember and learnings from outcomes, retrieved with a general gate (Clover's gate also keys on the memory
benchmark's wording); the ontology leaves out the records that were written for benchmark questions (STEM_CONCEPT,
FORMULA, DOC_RECORD).
"""
import ast
import hashlib
import re
import threading
import time
from pathlib import Path

from semantics import FOOD_ASK, FOOD_NOTE, NOT_FOR_ME  # noqa: F401 (the meaning rules are common)

LAYERS = ("memory", "ontology", "compiler", "ncm")
BENCHMARK_ONLY_COLLECTIONS = ("STEM_CONCEPT/", "FORMULA/", "DOC_RECORD/")
LINKS_AND_PATHS = re.compile(r"https?://\S+|www\.\S+|\b[A-Za-z]:\\\S*|(?:~|\.{1,2})?/[\w.-]+(?:/[\w.-]+)+")
MEMORY_CUES = ("remember", "earlier", "last time", "you told me", "i told you", "i said", "my ", "we decided",
               "correction", "recorded", "note")
RECALL_CUE = re.compile(
    r"\b(?:my|mine|i\s+(?:told|said|mentioned|asked|have|own|live|work|like|prefer|adopted|bought|moved)|"
    r"earlier|last\s+time|previous(?:ly)?|before|remember|we\s+(?:talked|discussed|spoke|decided)|"
    r"what\s+did\s+i|where\s+do\s+i|who\s+is\s+my)\b", re.I
)
DEVICE_PHRASE = re.compile(r"\b(?:on|in|with|from|to|for)\s+my\s+(?:phone|mobile|laptop|computer|pc|device|screen)\b", re.I)


def needs_recall(text):
    cleaned = DEVICE_PHRASE.sub(" ", text or "")
    return bool(RECALL_CUE.search(cleaned))
STOP = set("the a an and or of to in on for with is are was were be been it this that these those my your our their "
           "i you we they he she me us them what which who when where why how about from at by as into than then".split())
# words that say nothing about a topic: "much" alone pulled a lesson on pens into "how much free disk space" (2026-11-01)
FILLER = STOP | set("much many would could should does did have has had like just also only some any more most very "
                    "really there here question answer clover checks check found gave give solutions solution "
                    "independent first said user right wrong not".split())
LESSON_TOPIC = re.compile(r'On a question like "([^"]+)"')


def topic_terms(text):
    """Content words of a question or memory item; for a learning, of the question it is about."""
    m = LESSON_TOPIC.search(text or "")
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", (m.group(1) if m else text or "").lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in FILLER}


# who says it does not change a note: "I'm vegetarian and allergic to peanuts." and "The user is vegetarian and allergic
# to peanuts." are one note; "not", "no" and "never" are kept, so a correction is a new note
SPEAKER = STOP | {"user", "user's", "i'm", "im", "am", "i've", "has", "have", "remember", "that"}


def note_words(text):
    words = re.findall(r"[a-z0-9][a-z0-9'\-]*", str(text or "").lower().replace("\u2019", "'"))
    return tuple(w for w in words if w not in SPEAKER)  # in order: "Priya reports to Arun" is not "Arun ... Priya"
MEMORY_INSTRUCTION = (
    "Instruction: Use the persistent memory records [M#], their provenance, and their correction/evolution chains "
    "above. Explicitly present both any superseded initial interpretation and the superseding correction, distinguish "
    "user_stated_belief/hypothesis/unresolved from externally_verified facts, and preserve any unresolved contradictions.")
ONTOLOGY_INSTRUCTION = (
    "Instruction: Ground your reasoning and answer strictly in the verified entity records [E#], relationship paths "
    "[P#], and mathematical/domain constraints above.")
APP_ONTOLOGY_INSTRUCTION = (
    "Instruction: Answer from these records in plain words. Do not mention the records themselves, labels or IDs.")
APP_MEMORY_INSTRUCTION = (
    "Instruction: Use these notes where they answer the question. They record what the user told you and what earlier "
    "answers taught (with how often that worked); only a note marked externally verified was checked, and its cases hold "
    "only for the inputs shown. If one was superseded, use the newer one and say so, and do not repeat what did not work.")
VERIFIED_FUNCTION = re.compile(r"^Verified workspace function (\w+)\(")
PYTHON_BLOCK = re.compile(r"```python\n([\s\S]*?)```")


def _literal(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, type(None)))
            and len(repr(node.value)) <= 40)


def run_verified_calls(entry, q, sandbox=None):
    """[(call, result)] for the question's literal calls of the function an externally verified note recorded, run on
    the verified copy in IsolatedPythonSandbox. Given only the note, the model copied its stored cases for other inputs
    (consciousness step 4)."""
    m = VERIFIED_FUNCTION.match(entry.summary or "")
    block = PYTHON_BLOCK.search(entry.content or "")
    if entry.epistemic_status != "externally_verified" or m is None or block is None:
        return []
    calls = []
    for c in re.finditer(rf"\b{m.group(1)}\s*\(([^()\n#]*)\)", q or ""):
        try:
            node = ast.parse(f"f({c.group(1)})", mode="eval").body
        except SyntaxError:
            continue
        call = f"{m.group(1)}({c.group(1).strip()})"
        if all(map(_literal, node.args)) and all(k.arg and _literal(k.value) for k in node.keywords) and call not in calls:
            calls.append(call)
    calls = calls[:4]
    if not calls:
        return []
    body = "\n".join(ln for ln in block.group(1).splitlines() if not ln.lstrip().startswith(("assert ", "RESULT ")))
    code = (body + "\nRESULT = []\nfor _f in (" + "".join(f"lambda: {c}, " for c in calls) + "):\n"
            "    try:\n        _v = _f()\n        RESULT.append('= ' + repr(float(format(_v, '.10g')) if isinstance(_v, float) else _v))\n"
            "    except Exception as _e:\n        RESULT.append('raises ' + repr(_e))\n")
    try:
        import isolated_sandbox
        ok, results, _ = (sandbox or isolated_sandbox.IsolatedPythonSandbox(timeout_s=3.0)).run(code)
    except Exception:  # no runnable Python for the child process (a phone build): the note stays as it is
        return []
    # a NameError means the verified copy lacks something the module has: its result says nothing about the module
    if not ok or not isinstance(results, list) or len(results) != len(calls) or any("NameError" in str(r) for r in results):
        return []
    return [(c, str(r)[:200]) for c, r in zip(calls, results)]


PY_HINT = re.compile(r"^\s*(?:def|class|import|from|for|while|if|elif|try|with|return|print\(|[A-Za-z_]\w*\s*=(?!=))",
                     re.M)


def extract_python(reply):
    """The Python to check in a reply: ```python / ```py fences, or bare fences that read like Python. A solved Sudoku
    grid in a bare fence was compiled as Python, reported as a syntax error, and the repair turn replaced a correct
    answer with code (Omni ladder step 4)."""
    blocks = re.findall(r"```(python|py)?[ \t]*\n([\s\S]*?)```", reply)
    code = [b for tag, b in blocks if tag or PY_HINT.search(b)]
    if code:
        return max(code, key=len)
    if blocks:  # fenced text, none of it Python
        return ""
    return reply if re.search(r"^\s*def\s+\w+\s*\(", reply, flags=re.M) else ""


def answer_windows(reply):
    """Plan windows for the NCM renderer: code fences stay whole, other non-empty lines are one window each."""
    windows, i, lines = [], 0, reply.splitlines()
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            windows.append("\n".join(lines[i:j + 1]))
            i = j + 1
            continue
        if ln.strip():
            windows.append(ln.strip())
        i += 1
    return windows


class LayerKit:
    def __init__(self, base_dir, memory_root=None):
        self.base_dir = Path(base_dir)
        self.memory_root = Path(memory_root) if memory_root else None
        self._cl = None
        self._memory = None
        self._ontology = None
        self._verifier = None
        self._gc = None
        self._ncm = None
        self._store_lock = threading.Lock()
        self.workspaces = None  # workspaces.Workspaces (app mode): the folder the user works in
        self.records = None  # clover_repo.Repo (app mode): conversations, outcomes and summaries
        self.recall = None  # memory_index.MemoryIndex (app mode): Layer 2 over the conversations
        self.meaning = None  # semantics.Frame of the turn being answered (app mode)
        self.jepa_arch = "auto"  # selectable JEPA architecture: auto | clover_rpre | ijepa | audio_jepa | eb_jepa | mjepa | none
        self.visual_cue_64 = None
        self.audio_cue_64 = None
        self.semantic_cue_64 = None
        self.last_c2_jepa = None
        self.enable_meaning_bridge = True
        self.enable_jepa_action_prior = True
        self.last_memory_package = None
        self.last_ontology_evidence = []
        self.last_meaning_bridge = None
        self._emergent = None
        self._learning_mission = None
        self._curiosity = None

    def curiosity_state(self, storage_path=None):
        """Return the `PersistentCuriosityEngine` (`curiosity_foundation.py`) for this `LayerKit` instance."""
        if self._curiosity is None or storage_path is not None:
            import curiosity_foundation
            if storage_path is not None:
                c_path = Path(storage_path)
            elif self.memory_root is not None:
                c_path = self.memory_root.parent / "curiosity" / "curiosity_state.json"
            else:
                c_path = self.base_dir / "growth" / "curiosity" / "curiosity_state.json"
            self._curiosity = curiosity_foundation.for_path(c_path)
        return self._curiosity

    def emergent_concepts(self):
        """Return the `EmergentConceptEngine` connected to this `LayerKit`'s `CloverMemoryGraphStore` and `CloverOntologyGraph`."""
        if self._emergent is None:
            import emergent_concepts
            if self._ontology is None:
                self._ontology = self.cl.CloverOntologyRetriever()
                self._verifier = self.cl.CloverSymbolicVerifier(self._ontology)
            self._emergent = emergent_concepts.EmergentConceptEngine(
                self.store(), self._ontology.graph, base_dir=self.base_dir
            )
        return self._emergent

    def learning_mission(self, sandbox_dir=None):
        """Return the `WorldLearningMissionEngine` running inside an isolated sandbox directory so production
        knowledge is never modified by unvalidated hypotheses."""
        if self._learning_mission is None or sandbox_dir is not None:
            import world_learning_mission
            self._learning_mission = world_learning_mission.WorldLearningMissionEngine(
                self.base_dir, sandbox_dir=sandbox_dir
            )
        return self._learning_mission

    def workspace_context(self, q):
        if self.workspaces is None or self._skip(q):
            return None, None
        return self.workspaces.context_block(q)

    def records_context(self, q):
        # questions about the conversation itself are where the records matter, so only greetings are skipped
        if self.records is None or self.cl.is_short_greeting_or_ack(q):
            return None, None
        return self.records.context_block(q)

    def recall_context(self, q, history_turns=0):
        """Earlier messages for the current turn (Layer 2); history_turns: this chat's turns already in the prompt."""
        if self.recall is None or self.records is None or self.records.current is None:
            return None, None
        if self._skip(q):
            return None, None
        cued = needs_recall(q)
        if not getattr(self.recall, "want_recall", False) and not cued:
            return None, None
        if not cued:
            if self.meaning is not None and (getattr(self.meaning, "retry", None) or getattr(self.meaning, "asked", None)):
                return None, None
            fmt_words = {"shorter", "longer", "brief", "briefly", "concise", "sentence", "bullet", "paragraph",
                         "detail", "detailed", "simple", "plain", "word", "line", "again", "rewrite", "summarize", "please"}
            if not (topic_terms(q) - fmt_words):
                return None, None
        sid, n = self.records.current
        try:
            return self.recall.recall(sid, n, q, history_turns, force=cued)
        except TypeError:
            return self.recall.recall(sid, n, q, history_turns)

    @property
    def cl(self):
        if self._cl is None:
            import stand_ins
            stand_ins.install(Path(__file__).resolve().parent)
            import clover_loop
            self._cl = clover_loop
        return self._cl

    def _skip(self, q):
        return self.cl.is_short_greeting_or_ack(q) or self.cl.is_conversational_meta_or_confirmation(q)

    def store(self):
        with self._store_lock:
            if self._memory is None:
                if self.memory_root is None:
                    self._memory = self.cl.ensure_default_memory_tree()
                else:
                    self.memory_root.mkdir(parents=True, exist_ok=True)
                    self._memory = self.cl.CloverMemoryGraphStore(self.memory_root, auto_load=True)
            return self._memory

    def _user_memory_package(self, q):
        st = self.store()
        if not st.entries:
            return None
        all_seeds = st._score_seed_entries(q)
        # Exclude unverified hypotheses (`under_review` / `contested` / `hypothesis`) from factual user memory retrieval
        seeds = [
            s for s in all_seeds
            if st.entries.get(s[1]) is not None
            and st.entries[s[1]].status == "active"
            and st.entries[s[1]].epistemic_status != "hypothesis"
        ]
        if not seeds:
            return self._food_notes(q, st)
        score, _, reason = seeds[0]
        strong = any(k in reason for k in ("direct_id(", "concept_exact(", "phrase_match(", "evidence_phrase("))
        cued = any(c in q.lower() for c in MEMORY_CUES)
        if not (strong and score >= 12.0) and not (cued and score >= 8.0) and score < 18.0:
            return self._food_notes(q, st)
        pkg = st.retrieve_context_package(query=q, max_seeds=3, max_depth=2, max_entries=6, token_budget=1100)
        qt = topic_terms(q)
        top_score, top_id, _ = seeds[0]
        seed_scores = {eid: sc for sc, eid, _ in seeds}
        min_overlap = 2 if len(qt) >= 3 else 1
        pkg.items = [
            it for it in pkg.items
            if it.entry.status != "under_review"
            and it.entry.status != "contested"
            and it.entry.epistemic_status != "hypothesis"
            and (
                it.entry.id == top_id
                or getattr(it, "depth", 0) > 0
                or (
                    seed_scores.get(it.entry.id, 0.0) >= 0.35 * top_score
                    and len(qt & topic_terms((it.entry.summary or "") + " " + (it.entry.content or ""))) >= min_overlap
                )
            )
        ]
        return pkg if pkg.items else self._food_notes(q, st)

    def _food_notes(self, q, st):
        """The user's diet and allergy notes for a question about food (they share no word with it)."""
        if re.search(r"\b(?:bill|tip|gst|tax|split|pays?|cost)\b", q or "", re.I):
            return None
        if not (self.meaning.diet if self.meaning is not None else FOOD_ASK.search(q or "") and not NOT_FOR_ME.search(q or "")):
            return None
        ids = [e.id for e in st.entries.values() if "learning" not in e.tags and e.status == "active"
               and FOOD_NOTE.search(e.summary or "")]
        if not ids:
            return None
        pkg = st.retrieve_context_package(query=" ".join(st.entries[i].summary for i in ids[:3]), max_seeds=3,
                                          max_depth=0, max_entries=4, token_budget=600)
        pkg.items = [it for it in pkg.items if it.entry.id in ids]
        return pkg if pkg.items else None

    def memory_context(self, q):
        self.last_memory_package = None
        self.last_meaning_bridge = None
        self.semantic_cue_64 = None
        if self._skip(q):
            return None, None
        t0 = time.perf_counter()
        if self.memory_root is not None:  # "C:\Users\..." matched the word "user" in every note (2026-11-01)
            q = LINKS_AND_PATHS.sub(" ", q)
        pkg = self.cl.retrieve_relevant_memory_package(self.store(), q) if self.memory_root is None else self._user_memory_package(q)
        if pkg is None or not pkg.items:
            return None, None
        self.last_memory_package = pkg
        info = {"layer": "memory", "entries": [it.entry.id for it in pkg.items], "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if self.memory_root is not None:
            # the K3 rendering lists file paths, and LFM3 then went looking for the files (8 tool calls in a test)
            lines = ["Notes from this user's memory, already retrieved for you (no tools needed):"]
            computed = []
            for i, it in enumerate(pkg.items[:4], start=1):
                e = it.entry
                late = f" (superseded by {it.superseded_by_id})" if it.superseded_by_id else ""
                w, d = self.tally(e)
                used = f"; used before: worked {w}, did not work {d}" if w or d else ""
                lines.append(f"[M{i}] {e.summary} — {e.epistemic_status.replace('_', ' ')}, recorded {e.created_at[:10]}"
                             f"{used}{late}")
                ran = [] if it.superseded_by_id else run_verified_calls(e, q)
                if ran:
                    lines.append("     Computed now by running that verified function: " + "; ".join(f"{c} {r}" for c, r in ran))
                    computed += [{"entry": e.id, "call": c, "result": r} for c, r in ran]
            if computed:
                info["computed"] = computed
                info["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return "\n".join(lines) + "\n" + APP_MEMORY_INSTRUCTION, info
        return pkg.render_for_k3(compact=True) + "\n" + MEMORY_INSTRUCTION, info

    # ---- the user's memory (app mode): what they asked Clover to remember, with provenance ----
    def remember(self, text, kind="observation", conversation_id=None):
        import k3_memory_graph as M
        text = " ".join(str(text or "").split())[:2000]
        if len(text) < 3:
            raise ValueError("nothing to remember")
        same = next((e for e in self.store().entries.values() if "learning" not in e.tags and e.status == "active"
                     and note_words(e.content or e.summary) == note_words(text)), None)
        if same is not None:  # "Remember that I'm vegetarian..." twice made two notes that say the same (2026-11-01)
            return same.id
        meta_stop = FILLER | {"remember", "remembered", "earlier", "previous", "previously", "noted", "told", "said", "asked", "agreed", "discussed", "please"}
        words = [w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", text.lower()) if w not in meta_stop]
        tags = list(dict.fromkeys(words))[:8]
        eid = "MEM-USER-" + time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        entry = M.MemoryEntry(
            id=eid, title=text[:80], entry_type=kind if kind in M.VALID_ENTRY_TYPES else "observation",
            epistemic_status="user_stated_belief", summary=text[:400], content=text,
            source_evidence="Stated by the user in a Clover AI Studio chat.",
            provenance=M.MemoryProvenance(source_type="conversation", source_ref=f"studio-chat:{conversation_id or 'unknown'}",
                                          conversation_id=conversation_id, actor="user"),
            concepts=tags[:4], tags=tags)
        self.store().create_entry(entry)
        return eid

    def memories(self):
        out = []
        for e in self.store().entries.values():
            w, d = self.tally(e)
            out.append({"id": e.id, "title": e.title, "summary": e.summary, "created_at": e.created_at,
                        "status": e.status, "type": e.entry_type, "epistemic": e.epistemic_status,
                        "kind": "learning" if "learning" in e.tags else "note", "worked": w, "did_not_work": d})
        return sorted(out, key=lambda d: d["created_at"], reverse=True)

    def forget(self, entry_id):
        st = self.store()
        e = st.entries.get(str(entry_id))
        if e is None or self.memory_root is None:
            return False
        fp = (st.root_dir / e.rel_path).resolve()
        if fp.is_file() and str(fp).startswith(str(st.root_dir)):
            fp.unlink()
        st.rebuild_index()
        return True

    # ---- Growth: learnings from outcomes, recorded at once, then confirmed, contested or superseded by what follows ----
    @staticmethod
    def tally(entry):
        """-> (worked, did not work): outcomes of the answers this entry was used in."""
        acts = [r.action for r in entry.revisions]
        return acts.count("worked"), acts.count("did_not_work")

    def _similar_learning(self, text):
        st = self.store()
        for score, eid, _ in st._score_seed_entries(text) if st.entries else []:
            e = st.entries.get(eid)
            if score >= 18.0 and e is not None and "learning" in e.tags and e.status != "superseded":
                return e
        return None

    def learn(self, text, sid, n, kind="user"):
        """A learning from a correction (the user said an answer was not right; Clover's checks). Related to an
        earlier learning on the same question: it corrects (and supersedes) one that did not work, else refines it.
        -> (id, [changes])"""
        import k3_memory_graph as M
        st = self.store()
        tags = sorted(topic_terms(text))[:6]
        eid = "LRN-" + time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        ref = f"sessions:{sid}#{n}"
        old = self._similar_learning(text)
        rels, rel = [], None
        if old is not None:
            w, d = self.tally(old)
            rel = "corrects" if d > w else "refines"
            rels.append(M.MemoryRelationship(type=rel, target=old.id, source=eid, provenance=ref, confidence=0.6,
                                             note="a later correction on a similar question"))
        entry = M.MemoryEntry(
            id=eid, title=text[:80], entry_type="correction",
            epistemic_status="observed" if kind == "verified" else "user_stated_belief", summary=text[:400],
            content=text,
            source_evidence=("Clover's checks: the answer the user said was not right failed them; the corrected answer "
                             "passed them.") if kind == "verified" else "The user said Clover's answer was not right.",
            provenance=M.MemoryProvenance(source_type="conversation", source_ref=ref, conversation_id=sid,
                                          turn_range=str(n), actor="clover-growth"),
            concepts=tags[:4], tags=tags + ["learning"], relationships=rels)
        st.create_entry(entry)
        changes = [f"learned {eid}"]
        if rel == "corrects":
            st.mark_superseded(old.id, eid, "a later correction on a similar question", actor="clover-growth")
            changes.append(f"{eid} corrects {old.id}, which did not work (superseded)")
        elif rel:
            changes.append(f"{eid} refines {old.id}")
        return eid, changes

    def score_learning(self, eid, worked, ref):
        """An outcome of an answer that used this learning: contested once it fails more often than it works (at least
        twice), holding again when it works as often as it fails. Notes the user stated are not scored. -> change"""
        import k3_memory_graph as M
        st = self.store()
        e = st.entries.get(eid)
        if e is None or "learning" not in e.tags or e.status == "superseded":
            return None
        stamp = M.utc_now_iso()
        e.revisions.append(M.MemoryRevision(timestamp=stamp, actor="clover-growth",
                                            action="worked" if worked else "did_not_work", reason=ref))
        w, d = self.tally(e)
        before = e.status
        if d >= 2 and d > w:
            e.status = "contested"
        elif e.status == "contested" and w >= d:
            e.status = "active"
        if w >= 2 and d == 0 and e.epistemic_status == "user_stated_belief":
            e.epistemic_status = "observed"
        e.updated_at = stamp
        st._save_entry_in_place(e)
        st.rebuild_index()
        try:
            import curiosity_foundation as _CF
            derived = _CF.derive_real_turn_outcome(
                f"{e.title or ''} {e.summary or ''}",
                e.content or e.summary or "",
                user_feedback="worked" if worked else "did_not_work",
                correction_text=ref,
                entry_tally=(w, d),
            )
            self.curiosity_state().record_interaction_outcome(
                derived["query_text"],
                pred_error=derived["pred_error"],
                comp_error=derived["comp_error"],
                strategy_used="memory_graph_recall",
                actor="user_feedback",
                controllable=True,
                unresolved_summary=(f"Regression on {e.id}: {e.summary}" if not worked else None),
                memory_entry_id=e.id,
                user_request=e.title or e.summary,
                model_prediction=e.summary,
                correction=ref,
            )
        except Exception:
            pass
        return (f"{eid} {'worked' if worked else 'did not work'} (worked {w}, did not work {d})"
                + (f", now {'holding' if e.status == 'active' else e.status}" if e.status != before else ""))

    def ontology_context(self, q):
        self.last_ontology_evidence = []
        if self.memory_root is not None:  # app: a link or path is not a topic (github.com/.../clover-framework matched)
            q = LINKS_AND_PATHS.sub(" ", q)
            if not re.search(r"[A-Za-z]{3}", q):
                return None, None
        if self._skip(q):
            return None, None
        if self._ontology is None:
            self._ontology = self.cl.CloverOntologyRetriever()
            self._verifier = self.cl.CloverSymbolicVerifier(self._ontology)
        t0 = time.perf_counter()
        ev = list(self._ontology.retrieve(q, max_entities=4, max_paths=6))
        bridged_from_mem = []
        bridged_entities = []
        if self.enable_meaning_bridge and self.last_memory_package is not None and getattr(self.last_memory_package, "items", None):
            active_mems = [it.entry for it in self.last_memory_package.items if not getattr(it, "superseded_by_id", None)]
            if active_mems:
                mem_terms = []
                for e in active_mems[:3]:
                    raw_m = f"{e.title or ''} {e.summary or ''} {e.content or ''} {' '.join(e.concepts or ())} {' '.join(e.tags or ())}"
                    clean_m = re.sub(r"\bClover(?:'s)?\s+checks?\b", " ", raw_m, flags=re.I)
                    mem_terms.append(clean_m)
                bridge_q = f"{q}\n" + "\n".join(mem_terms)
                ev_bridged = self._ontology.retrieve(bridge_q, max_entities=4, max_paths=6)
                seen_keys = {(item.kind, tuple(item.entities), item.text[:120]) for item in ev}
                for e in active_mems[:3]:
                    if str(e.id).startswith("MEM-CONCEPT-EMERGENT-"):
                        import k3_ontology_graph as _O
                        s_hash = str(e.id).rsplit("-", 1)[-1]
                        c_ref = _O.EntityRef("STEM_CONCEPT", f"STEM-EMERGENT-{s_hash}")
                        c_doc = self._ontology.graph.get(c_ref)
                        if c_doc is not None:
                            ev.append(_O.Evidence(
                                kind="entity",
                                text=_O.render_entity_record(c_ref, c_doc),
                                entities=(c_ref.canonical,),
                                score=0.98,
                            ))
                            if c_ref.canonical not in bridged_entities:
                                bridged_entities.append(c_ref.canonical)
                            for p_em in self._ontology.traversal.paths([c_ref], hops=2, max_paths=4):
                                ev.append(_O.Evidence(
                                    kind="path",
                                    text=p_em.describe(),
                                    entities=tuple(r.canonical for r in p_em.entity_refs),
                                    score=0.94,
                                    provenance=tuple(ed.get("provenance", {}) for ed in p_em.edges),
                                ))
                for b_item in ev_bridged:
                    b_key = (b_item.kind, tuple(b_item.entities), b_item.text[:120])
                    if b_key not in seen_keys and len(ev) < 10:
                        ev.append(b_item)
                        seen_keys.add(b_key)
                        for ent_id in b_item.entities:
                            if ent_id not in bridged_entities:
                                bridged_entities.append(ent_id)
                if bridged_entities or ev_bridged:
                    bridged_from_mem = [e.id for e in active_mems[:3]]
        if self.memory_root is not None:
            ev = [
                e for e in ev
                if not any(
                    str(x).startswith(BENCHMARK_ONLY_COLLECTIONS) and "EMERGENT-" not in str(x)
                    for x in e.entities
                )
            ]
        if not ev:
            return None, None
        self.last_ontology_evidence = list(ev)
        info = {"layer": "ontology", "entities": [e for item in ev for e in item.entities], "n_evidence": len(ev),
                "ms": round((time.perf_counter() - t0) * 1000, 1)}
        if bridged_from_mem:
            info["bridged_from_memory"] = bridged_from_mem
            info["bridged_entities"] = bridged_entities
        rendered = self.cl.render_evidence(ev)
        if self.memory_root is None:
            return "Ontology-Guided Symbolic Evidence & Provenance Paths:\n" + rendered + "\n" + ONTOLOGY_INSTRUCTION, info
        # app: told not to, LFM3-Omni still quoted [E2]/[P5] and raw IDs, so the labels, IDs and paths are left out
        notes = []
        for blk in rendered.split("\n\n"):
            lines = blk.strip().splitlines()
            if not lines or not re.match(r"\[E\d+\]", lines[0]):
                continue
            body = [re.sub(r"^[A-Z_]+:[^:]+:\s*", "", ln).replace("Verified Constraints:", "Key points:") for ln in lines[1:]]
            notes.append("- " + "\n  ".join(b for b in body if b.strip()))
        if not notes:
            return None, None
        return ("Clover's own records on this topic (already retrieved; no tools needed):\n" + "\n".join(notes) + "\n"
                + APP_ONTOLOGY_INSTRUCTION), info

    def grounding_filter(self, q, reply):
        if self._verifier is None:
            self._ontology = self._ontology or self.cl.CloverOntologyRetriever()
            self._verifier = self.cl.CloverSymbolicVerifier(self._ontology)
        out = self._verifier.verify_source_grounding(q, reply)
        return out, out != reply

    def compiler_check(self, q, reply):
        """-> (feedback or None, info). No code in the reply -> nothing to check."""
        code = extract_python(reply)
        if not code.strip():
            return None, {"layer": "compiler", "checked": False}
        if self._gc is None:
            self._gc = self.cl.CloverPythonGenerativeCompiler(allow_undefined=True)
        v = self._gc.check_prefix(code, end_marker=True, spec_prompt=q)
        info = {"layer": "compiler", "checked": True, "ok": bool(v.ok), "ms": v.elapsed_ms,
                "diagnostics": [d.format_concise() for d in v.diagnostics[:3]]}
        return (None if v.ok else v.format_feedback()), info

    def c1_infer(self, raw_reply, stats=None, model=None, layers_desc=None):
        """Strictly C1: extract 2048-d C1 inference representations (`c1_state`, `c1_info`) without planning or decoding."""
        import omni_c2_c3
        m_name = model or getattr(self, "model_name", None) or "LFM3-Omni-2B"
        l_desc = layers_desc or getattr(self, "layers_desc", None) or ""
        rt = omni_c2_c3.for_base(self.base_dir)
        if rt is not None:
            st, info = rt.c1_infer(raw_reply, generation_stats=stats, model_name=m_name)
            info["model"] = m_name
            if l_desc:
                info["layers_desc"] = l_desc
            return st, info
        windows = answer_windows(raw_reply)
        prefill_ms = sum(float(s.get("prefill_s") or 0) * 1000.0 for s in (stats or ()) if isinstance(s, dict))
        gen_tok = sum(int(s.get("new_tokens") or 0) for s in (stats or ()) if isinstance(s, dict))
        n_win = 1 if windows else 0
        state = {"stage": "c1_inference", "raw_text": str(raw_reply or ""), "windows": windows, "window_tokens": 512, "tokens": gen_tok, "ms": round(prefill_ms, 2)}
        info = {"layer": "c1_inference", "model": m_name, "windows": n_win, "window_tokens": 512, "tokens": gen_tok, "ms": round(prefill_ms, 2)}
        if l_desc:
            info["layers_desc"] = l_desc
        return state, info

    @staticmethod
    def _fallback_h2048(text: str):
        import numpy as np
        vec = np.zeros((4, 2048), dtype=np.float32)
        words = re.findall(r"[a-z0-9_]+", str(text or "").lower())
        if not words:
            words = ["empty"]
        for i, w in enumerate(words[:128]):
            h = hashlib.sha256(w.encode("utf-8")).digest()
            row = i % 4
            for b_i in range(0, 24, 3):
                idx = ((h[b_i] << 8) | h[b_i + 1]) % 2048
                sgn = 1.0 if (h[b_i + 2] & 1) == 0 else -1.0
                vec[row, idx] += np.float32(sgn)
        norms = np.linalg.norm(vec, axis=-1, keepdims=True) + np.float32(1e-6)
        return (vec / norms * np.float32(0.5))[None, :, :]

    def compute_meaning_bridge(self, q):
        """Synthesize the `Memory -> Ontology -> HybridAgent/JEPA -> Action` bridge state for the current turn:
          1. Extracts active Memory Graph entries (`self.last_memory_package`) and Ontology Graph evidence (`self.last_ontology_evidence`).
          2. Computes the 8-d action capability prior `action_prior_8` and `recommended_tools` via `jepa_engine.semantic_action_prior_8`.
          3. Projects the bridged symbolic evidence into a 64-d JEPA latent context cue `semantic_cue_64` via `jrt.encode_semantic_bridge_cue`.
        """
        if not self.enable_meaning_bridge:
            self.last_meaning_bridge = None
            self.semantic_cue_64 = None
            return None
        mem_ids, mem_texts = [], []
        if self.last_memory_package is not None and getattr(self.last_memory_package, "items", None):
            for it in self.last_memory_package.items:
                if not getattr(it, "superseded_by_id", None):
                    e = it.entry
                    mem_ids.append(e.id)
                    mem_texts.append(f"{e.title or ''}: {e.summary or ''} ({e.content or ''})")
        ont_entities, ont_texts = [], []
        for item in self.last_ontology_evidence or ():
            for ent_id in getattr(item, "entities", ()):
                if ent_id not in ont_entities:
                    ont_entities.append(ent_id)
            ont_texts.append(str(getattr(item, "text", "") or ""))
        if not mem_texts and not ont_texts:
            self.last_meaning_bridge = None
            self.semantic_cue_64 = None
            return None
        import jepa_engine
        action_prior_8, rec_tools, dim_scores = jepa_engine.semantic_action_prior_8(
            q if (mem_texts or ont_texts) else "",
            memory_texts=mem_texts,
            ontology_texts=ont_texts,
        )
        sem_cue_64 = None
        try:
            jrt = jepa_engine.for_base(self.base_dir)
            bridge_str = "\n".join(mem_texts[:2] + ont_texts[:3])[:1600]
            h_bridge = self._fallback_h2048(bridge_str)
            sem_cue_64 = jrt.encode_semantic_bridge_cue(h_bridge, configured_arch=getattr(self, "jepa_arch", "auto"))
        except Exception:
            sem_cue_64 = None
        self.semantic_cue_64 = sem_cue_64
        curiosity_info = None
        try:
            disc_state = getattr(self.meaning, "discernment", "proceed") if self.meaning is not None else "proceed"
            decl_caps = getattr(self.meaning, "declined", ()) if self.meaning is not None else ()
            perms_map = getattr(self, "permissions_map", None)
            curiosity_info = self.curiosity_state().recommend_strategy_and_voi(
                q or "\n".join(mem_texts[:1]),
                latent_cue_64=sem_cue_64,
                discernment_state=disc_state,
                declined_capabilities=decl_caps,
                permissions=perms_map,
                interactive=True,
            )
            if curiosity_info and curiosity_info.get("should_observe"):
                for st_tool in curiosity_info.get("recommended_studio_tools") or ():
                    if st_tool not in rec_tools and st_tool not in decl_caps:
                        rec_tools.append(st_tool)
                if action_prior_8 is not None:
                    import numpy as np
                    boost_v = curiosity_info.get("belief_action_boost_8")
                    boost = np.asarray(np.zeros(8) if boost_v is None else boost_v, dtype=np.float32)
                    combined_ap = np.asarray(action_prior_8, dtype=np.float32) + np.float32(0.25) * boost
                    norm_ap = float(np.linalg.norm(combined_ap))
                    if norm_ap > 1e-6:
                        action_prior_8 = combined_ap / np.float32(norm_ap)
        except Exception:
            curiosity_info = None
        bridge_info = {
            "layer": "meaning_bridge",
            "memory_entries": mem_ids,
            "ontology_entities": ont_entities,
            "recommended_tools": rec_tools,
            "action_prior_dims": dim_scores,
            "action_prior_8": action_prior_8 if self.enable_jepa_action_prior else None,
            "has_semantic_cue_64": sem_cue_64 is not None,
            "curiosity": {
                "region_id": curiosity_info.get("region_id"),
                "state_curiosity": curiosity_info.get("state_curiosity"),
                "selected_strategy": curiosity_info.get("selected_strategy"),
                "recommended_studio_tools": curiosity_info.get("recommended_studio_tools", []),
                "should_observe": curiosity_info.get("should_observe"),
                "voi": (curiosity_info.get("voi_analysis") or {}).get("value_of_information_voi", 0.0),
            } if isinstance(curiosity_info, dict) else None,
        }
        self.last_meaning_bridge = bridge_info
        return bridge_info

    def run_idle_curiosity_step(
        self,
        *,
        is_idle_fn=None,
        permissions=None,
        budget=None,
        regression_tests=(),
    ):
        """Priority 2 & 3: Execute one bounded curiosity step through `studio_idle()` with explicit resource budgets
        and permissions, closing the strategy-to-action-to-outcome loop."""
        ws_root = None
        if self.workspaces is not None and hasattr(self.workspaces, "active"):
            act_ws = self.workspaces.active()
            if isinstance(act_ws, dict) and act_ws.get("path"):
                ws_root = act_ws["path"]
        if self._ontology is None:
            try:
                self._ontology = self.cl.CloverOntologyRetriever()
                self._verifier = self.cl.CloverSymbolicVerifier(self._ontology)
            except Exception:
                pass
        ont_g = getattr(self._ontology, "graph", None) if self._ontology is not None else None
        res = self.curiosity_state().run_bounded_idle_curiosity_step(
            is_idle_fn=is_idle_fn,
            permissions=permissions if permissions is not None else getattr(self, "permissions_map", None),
            workspace_root=ws_root,
            memory_store=self.store(),
            ontology_graph=ont_g,
            regression_tests=regression_tests,
            budget=budget,
        )
        if res.get("ran") and self.records is not None and hasattr(self.records, "log"):
            ar = res.get("action_result") or {}
            self.records.log(
                None,
                None,
                f"idle_curiosity: strategy={ar.get('strategy')} region={ar.get('region_id')} "
                f"err={ar.get('comp_error_before')}->{ar.get('comp_error_after')} "
                f"({ar.get('verification_result', '')[:120]}) in {res.get('wall_ms')} ms",
            )
            self.records.schedule("Idle curiosity step")
        return res

    def c2_plan_state(self, q, c1_state=None, offered_tools=None):
        """Strictly C2: run HybridAgent continuous `<latent>` planning (`plan-trunk-0`) + selectable JEPA (`eb_jepa`/`clover_rpre`)
        latent trajectory optimization on `c1_state` -> (`c2_state`, `c2_info`)."""
        import numpy as np
        import omni_c2_c3
        if self.enable_meaning_bridge and self.last_meaning_bridge is None and (self.last_memory_package or self.last_ontology_evidence):
            self.compute_meaning_bridge(q)
        cues = [c for c in (self.visual_cue_64, self.audio_cue_64, self.semantic_cue_64 if self.enable_meaning_bridge else None) if c is not None]
        cue_64 = np.mean(np.stack([np.asarray(c, dtype=np.float32).reshape(64) for c in cues], axis=0), axis=0) if cues else None
        act_prior = (
            self.last_meaning_bridge.get("action_prior_8")
            if (self.enable_meaning_bridge and self.enable_jepa_action_prior and isinstance(self.last_meaning_bridge, dict))
            else None
        )
        tools_in = list(offered_tools or [])
        if not tools_in and self.enable_meaning_bridge and isinstance(self.last_meaning_bridge, dict):
            declined = set(getattr(self.meaning, "declined", ()) or ()) if self.meaning is not None else set()
            for rt_name in self.last_meaning_bridge.get("recommended_tools") or ():
                if rt_name not in tools_in and rt_name not in declined:
                    tools_in.append(rt_name)
        rt = omni_c2_c3.for_base(self.base_dir)
        if rt is not None:
            c2_state, c2_info = rt.c2_plan(
                q,
                c1_state=c1_state,
                offered_tools=tools_in,
                jepa_arch=getattr(self, "jepa_arch", "auto"),
                context_cue_64=cue_64,
                action_prior_8=act_prior,
            )
            if isinstance(self.last_meaning_bridge, dict):
                c2_info["meaning_bridge"] = {
                    "memory_entries": self.last_meaning_bridge.get("memory_entries", []),
                    "ontology_entities": self.last_meaning_bridge.get("ontology_entities", []),
                    "recommended_tools": self.last_meaning_bridge.get("recommended_tools", []),
                    "action_prior_dims": self.last_meaning_bridge.get("action_prior_dims", {}),
                }
            self.last_c2_jepa = c2_state.get("jepa_plan")
            return c2_state, c2_info
        raw = str((c1_state or {}).get("raw_text") or "")
        windows = answer_windows(raw)
        jepa_plan = None
        planned_tools = list(tools_in)
        jepa_red = 0.0
        try:
            import jepa_engine
            jrt = jepa_engine.for_base(self.base_dir)
            h_p = (c1_state or {}).get("h_primary")
            if h_p is None:
                h_p = self._fallback_h2048(q or raw)
            r_p = self._fallback_h2048((q or "") + "::plan::" + raw)
            jepa_plan = jrt.plan_c2_latent(
                h_p,
                r_p,
                offered_tools=planned_tools,
                context_cue_64=cue_64,
                action_prior_8=act_prior,
                configured_arch=getattr(self, "jepa_arch", "auto"),
            )
            self.last_c2_jepa = jepa_plan
            if jepa_plan.get("ranked_tools"):
                planned_tools = list(jepa_plan["ranked_tools"])
            jepa_red = float(jepa_plan.get("energy_reduction_pct") or 0.0)
        except Exception:
            pass
        info = {
            "layer": "c2_hybridagent",
            "ran": False,
            "reason": "C2/C3 parts not installed",
            "planned_tools": planned_tools,
            "jepa_energy_reduction_pct": jepa_red,
        }
        if jepa_plan is not None:
            info["jepa_from"] = "hashed words"
        if isinstance(self.last_meaning_bridge, dict):
            info["meaning_bridge"] = {
                "memory_entries": self.last_meaning_bridge.get("memory_entries", []),
                "ontology_entities": self.last_meaning_bridge.get("ontology_entities", []),
                "recommended_tools": self.last_meaning_bridge.get("recommended_tools", []),
                "action_prior_dims": self.last_meaning_bridge.get("action_prior_dims", {}),
            }
        state = {"stage": "c2_plan", "raw_text": raw, "windows": windows, "planned_tools": planned_tools, "jepa_plan": jepa_plan}
        return state, info

    def jepa_record_outcome(self, z_obs_64=None, c1_obs_state=None, is_baseline=False):
        """Flow 5 (`Outcome -> Growth`): update `OutcomeResidualAdapter` from the latest C2 plan and observed outcome.
        When `c1_obs_state` is provided (e.g., the C1 state of a verified or retry-corrected answer), encodes its
        2048-d representation into 64-d JEPA state space as the real observed outcome `z_obs_64`.
        """
        if getattr(self, "jepa_arch", "auto") == "none" or not isinstance(self.last_c2_jepa, dict):
            return None
        try:
            import jepa_engine
            import numpy as np
            z_init = self.last_c2_jepa.get("z_init_64")
            act = self.last_c2_jepa.get("first_action")
            z_plan = self.last_c2_jepa.get("z_planned_64")
            if z_init is None or act is None or z_plan is None:
                return None
            jrt = jepa_engine.for_base(self.base_dir)
            if z_obs_64 is not None:
                z_obs = np.asarray(z_obs_64, dtype=np.float32)
            elif isinstance(c1_obs_state, dict) and "h_primary" in c1_obs_state:
                h_mean_2048 = np.asarray(c1_obs_state["h_primary"], dtype=np.float32).mean(axis=1)
                h_vq = jepa_engine._layer_norm(jepa_engine._linear(h_mean_2048, jrt.w["bridge.lfm3_to_vq.weight"], jrt.w["bridge.lfm3_to_vq.bias"]))
                z_obs = jepa_engine._layer_norm(jepa_engine._linear(h_vq, jrt.w["state_adapter.weight"], jrt.w["state_adapter.bias"]) + jrt.w["state_mod_token"]).squeeze(0)
            else:
                z_obs = np.asarray(z_plan, dtype=np.float32)
            cue_64 = None
            if self.visual_cue_64 is not None and self.audio_cue_64 is not None:
                cue_64 = 0.5 * (self.visual_cue_64 + self.audio_cue_64)
            elif self.visual_cue_64 is not None:
                cue_64 = self.visual_cue_64
            elif self.audio_cue_64 is not None:
                cue_64 = self.audio_cue_64
            return jrt.record_and_adapt_outcome(
                z_init,
                act,
                z_obs,
                context_cue_64=cue_64,
                is_baseline=is_baseline,
                configured_arch=getattr(self, "jepa_arch", "auto"),
            )
        except Exception:
            return None

    def c2_plan(self, q, offered_tools=None, c1_state=None):
        """Run C2 HybridAgent continuous `<latent>` planning (`plan-trunk-0` / `HybridLink2048` + `Lfm3NativeHybridLink`)."""
        _, info = self.c2_plan_state(q, c1_state=c1_state, offered_tools=offered_tools)
        return info

    def c3_decode(self, c2_state, on_window=None):
        """Strictly C3: decode `c2_state` through Bridges + 11-level VQ-VAE + 12L NextConceptTransformer + 3L LexicalPointerDecoder."""
        import omni_c2_c3
        rt = omni_c2_c3.for_base(self.base_dir)
        if rt is not None and isinstance(c2_state, dict) and c2_state.get("stage") == "c2_plan" and "pids_primary" in c2_state:
            return rt.c3_decode(c2_state, on_window=on_window)
        raw = str(c2_state.get("raw_text") if isinstance(c2_state, dict) else (c2_state or ""))
        return self.ncm_render(raw)

    def ncm_render(self, reply):
        if isinstance(reply, dict) and reply.get("stage") == "c2_plan":
            return self.c3_decode(reply)
        windows = answer_windows(reply)
        if not windows:
            return reply, {"layer": "ncm", "windows": 0}
        import omni_c2_c3
        rt = omni_c2_c3.for_base(self.base_dir)
        if rt is not None:
            return rt.render_reply(reply, windows=windows)
        if self._ncm is None and (self.base_dir / "data" / "ncm" / "config.yaml").is_file():
            import lfm3_clover
            self._ncm = lfm3_clover.Lfm3NCMRenderer(self.base_dir / "data", self.base_dir / "engine")
        if self._ncm is not None:
            t0 = time.perf_counter()
            text, ids, meta = self._ncm.render_plans_in_model_vocab(windows, cands=1)
            return text, {"layer": "ncm", "windows": len(windows), "s": round(time.perf_counter() - t0, 1),
                          "c_engine_ms": meta.get("c_engine_ms")}
        return reply, {"layer": "ncm", "ran": False, "reason": "C2/C3 parts not installed", "windows": len(windows)}
