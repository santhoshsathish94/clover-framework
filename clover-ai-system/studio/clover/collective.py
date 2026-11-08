"""Collective Problem-Solving Learning — Local Candidate Extractor (`collective.py`).

Implements the local side of Clover's collective learning boundary:
- Extracts generalisable candidate knowledge (problem-solving strategies, cross-domain tool patterns, semantic
  mappings) from a person's local learning store (`personal.Personal`) ONLY when `consent_contribute` is True.
- Provides `preview_candidates()` so the person can inspect the exact JSON packages before consenting.
- Enforces a strict privacy & personal-boundary filter BEFORE candidate synthesis:
  1. Never exports raw conversation text, prompts, reactions, timestamps, session IDs, or object names.
  2. Never exports personal presentation preferences (`format`, `length`, `confirm`, `plan`, `options`, `sources`).
  3. Never exports personal routines (`"the usual"`, `"the Friday routine"`) or open items.
  4. Never exports single-turn user statements (`"When I say X, I mean Y"`).
  5. Pre-filters every experience's features against a closed general English vocabulary (Open English WordNet 2025 +
     core domain lemmas) BEFORE running the intersection learner, eliminating the Symbolic Unlearning Ripple Effect
     (where deleting one experience could otherwise unmask a private token from another experience).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import personal
import semantics

SCHEMA_VERSION = "1.0.0"

# Patterns that must never appear in any candidate feature
PII_OR_SECRET_RE = re.compile(
    r"(@|https?://|www\.|/|\\|\bssn\b|\bsecret\b|\bconfidential\b|\bprivate\b|\bacmecorp\b|\bproject_codename\b|"
    r"\bpatient_record\b|\bbank_routing\b|\bsalary_review\b|\bmerger_target\b|\bapi_secret\b|\bssh_key\b|\d)",
    re.I,
)

# Words that belong to personal routines ("the usual", "my morning routine") and must stay local
PERSONAL_ROUTINE_WORDS = {
    "usual", "routine", "morning", "evening", "friday", "monday", "daily", "weekly", "end-of-day", "wrap-up",
}

# Prepositions / particles tested in the construction-invariance ablation (`mode="drop_prepositions"`)
PREPOSITION_TOKENS = {
    "at", "into", "on", "up", "over", "in", "with", "from", "by", "about", "through", "down", "off", "out",
}

# Multi-capability problem-solving strategy guidance templates (keyed by sorted capability tuple)
_RAW_STRATEGY_GUIDANCE = {
    ("files.read", "python"): (
        "Collective problem-solving strategy: first read and inspect the file structure, then run Python to compute "
        "and verify the exact result."
    ),
    ("files.list", "files.read", "python"): (
        "Collective problem-solving strategy: first inspect the file, then run Python to compute and verify the result."
    ),
    ("files.read", "shell"): (
        "Collective problem-solving strategy: first read the relevant file, then run the shell command to verify."
    ),
    ("files.list", "files.read", "shell"): (
        "Collective problem-solving strategy: first inspect the files, then run the shell check."
    ),
    ("python", "web.page", "web.search"): (
        "Collective problem-solving strategy: first gather primary sources from the web, then verify the figures "
        "with Python."
    ),
    ("python", "web.search"): (
        "Collective problem-solving strategy: first gather primary sources from the web, then verify the figures "
        "with Python."
    ),
}
STRATEGY_GUIDANCE = {tuple(sorted(k)): v for k, v in _RAW_STRATEGY_GUIDANCE.items()}

_OEWN_CACHE: set[str] | None = None

# Core general English task vocabulary (always available even when OEWN XML gz is not mounted)
CORE_GENERAL_VOCAB = {
    "audit", "preflight", "corroborate", "triage", "briefing", "relocate", "distill", "condense", "bookmark",
    "shelve", "snapshot", "verify", "validate", "cross-check", "crosscheck", "reconcile", "benchmark", "profile",
    "diagnose", "inspect", "investigate", "dig", "explore", "lookup", "search", "browse", "fetch", "summarize",
    "summary", "digest", "outline", "shorten", "trim", "expand", "elaborate", "proofread", "polish", "tidy",
    "clean", "format", "reformat", "refactor", "restructure", "review", "check", "test", "run", "execute",
    "compile", "build", "deploy", "ship", "release", "commit", "push", "pull", "merge", "clone", "open", "read",
    "view", "show", "list", "write", "save", "create", "make", "edit", "update", "modify", "delete", "remove",
    "archive", "backup", "restore", "compress", "zip", "export", "convert", "share", "send", "reply", "answer",
    "forward", "flag", "star", "schedule", "book", "move", "reschedule", "bump", "cancel", "kill", "remind",
    "note", "remember", "memorize", "todo", "stash", "park", "zap", "scrub", "calculate", "compute", "solve",
    "count", "total", "sum", "average", "mean", "compare", "weigh", "chart", "plot", "graph", "draw", "generate",
    "render", "picture", "image", "photo", "weather", "forecast", "news", "price", "score", "time", "clock",
    "battery", "disk", "memory", "status", "history", "code", "file", "folder", "directory", "document", "report",
    "table", "spreadsheet", "email", "message", "event", "meeting", "topic", "term", "claim", "fact", "figure",
    "data", "into", "at", "on", "up", "over", "out", "down", "off", "through", "poke", "carefully", "spotlight",
    "partition", "prefix", "allocate",
}


def load_closed_general_vocabulary(oewn_gz_path: str | Path | None = None) -> set[str]:
    """Load the closed general English lemma vocabulary from Open English WordNet 2025 (when present on k3)
    plus `CORE_GENERAL_VOCAB`, excluding personal routine words."""
    global _OEWN_CACHE
    if _OEWN_CACHE is not None:
        return _OEWN_CACHE

    vocab = set(CORE_GENERAL_VOCAB)
    if oewn_gz_path is None:
        oewn_gz_path = os.environ.get(
            "CSI_DICT", "/opt/clover-k3/meaning-core/data/raw/english-wordnet-2025.xml.gz"
        )
    p = Path(oewn_gz_path) if oewn_gz_path else None
    if p and p.exists():
        try:
            with gzip.open(p, "rb") as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag in ("Lemma", "Form") or elem.tag.endswith(("}Lemma", "}Form")):
                        w = (elem.attrib.get("writtenForm") or "").strip().lower()
                        if re.fullmatch(r"[a-z][a-z\-]{1,28}", w):
                            vocab.add(w)
                        elem.clear()
        except Exception:
            pass

    vocab -= PERSONAL_ROUTINE_WORDS
    _OEWN_CACHE = vocab
    return vocab


# Nouns ending in -ing that are lexical nouns rather than deverbal action gerunds
NON_GERUND_ING_NOUNS = {
    "morning", "evening", "building", "meeting", "nothing", "something", "anything", "everything", "string", "thing",
    "feeling", "meaning", "setting", "warning", "heading", "reading", "writing", "drawing", "painting", "briefing",
    "ceiling", "wedding", "sibling", "pudding", "clothing", "housing", "funding", "ranking", "rating", "listing",
}


def normalize_gerund_features(feats, closed_vocab: set[str] | None = None) -> frozenset[str]:
    """When CSI tags a deverbal '-ing' word as a NOUN (e.g. 'needs auditing', 'needs cross-checking', 'needs triaging'),
    spaCy keeps lemma='auditing' instead of running morph-verb-v1.1. Analysis-by-synthesis against the closed OEWN
    vocabulary recovers the base verb lemma ('tok=audit', 'tok=check', 'tok=triage') without losing the surface token."""
    vocab = closed_vocab if closed_vocab is not None else load_closed_general_vocabulary()
    out = set(feats)
    for f in list(feats):
        if not f.startswith("tok="):
            continue
        lem = f[4:].lower()
        if len(lem) < 6 or not lem.endswith("ing") or lem in NON_GERUND_ING_NOUNS:
            continue
        stem = lem[:-3]
        if len(stem) >= 3 and stem in vocab:
            out.add(f"tok={stem}")
        elif len(stem) >= 3 and (stem + "e") in vocab:
            out.add(f"tok={stem}e")
        elif len(stem) >= 4 and stem[-1] == stem[-2] and stem[:-1] in vocab:
            out.add(f"tok={stem[:-1]}")
    return frozenset(out)


def is_allowed_collective_feature(feat: str, closed_vocab: set[str] | None = None) -> bool:
    """Return True iff `feat` is a non-PII, closed-vocabulary general feature safe for collective extraction."""
    if not isinstance(feat, str) or PII_OR_SECRET_RE.search(feat):
        return False
    vocab = closed_vocab if closed_vocab is not None else load_closed_general_vocabulary()
    if feat.startswith("ctx="):
        return bool(re.fullmatch(r"[a-z][a-z0-9_\-]{1,24}", feat[4:]))
    if feat.startswith("has="):
        return feat[4:] in ("url", "path", "number", "code", "doc", "msg", "event", "file", "folder", "person",
                            "time", "fmt", "lang", "topic", "topic2", "term", "text", "env", "name")
    if feat.startswith("pat="):
        return feat[4:] in ("arrow", "colon")
    if feat.startswith("tok="):
        lemma = feat[4:].lower()
        if not re.fullmatch(r"[a-z][a-z\-]{1,28}", lemma):
            return False
        if lemma in PERSONAL_ROUTINE_WORDS:
            return False
        return lemma in vocab
    return False


def _general_cue_tokens(text: str, csi_fn=None, device: str = "") -> set[str]:
    """Identify which `tok=` verb/keyword features in `text` caused `semantics.asked_caps(text)` to fire, using
    leave-one-word-out sensitivity on `semantics.asked_caps` plus generic capability nouns already captured by `has=`."""
    if not text:
        return set()
    base_caps = set(semantics.asked_caps(text))
    if not base_caps:
        return set()
    cues = {
        "tok=file", "tok=files", "tok=folder", "tok=folders", "tok=directory", "tok=web", "tok=internet",
        "tok=online", "tok=python", "tok=code", "tok=script", "tok=terminal", "tok=shell",
    }
    for m in re.finditer(r"[A-Za-z][A-Za-z'\-]*", text):
        word = m.group(0)
        without = text[:m.start()] + " " + text[m.end():]
        if set(semantics.asked_caps(without)) != base_caps:
            cues.add(f"tok={word.lower()}")
            for f in personal.features(word, None, device):
                if f.startswith("tok="):
                    cues.add(f)
    return cues


def scrub_experience_for_collective(
    exp: dict,
    closed_vocab: set[str],
    extraction_mode: str = "mismatch_only",
) -> dict | None:
    """Create a scrubbed, minimal local experience record containing ONLY closed-vocabulary features and valid
    capabilities—protecting against the Symbolic Unlearning Ripple Effect.

    `extraction_mode`:
      - `"mismatch_only"` (default): only mismatches (`general["intent"] != resolved`) create new hypotheses,
        while confirmations still count as support/contradiction for existing hypotheses.
      - `"residual_cue"` (ablation): when an accepted turn had `general["intent"] == resolved` because of a known
        general cue word in `text`, strips the known general cue tokens from `pfeats` and marks `general["intent"] = []`
        if a residual content anchor (`tok=...`) remains.
      - `"drop_prepositions"` (ablation): removes preposition tokens (`PREPOSITION_TOKENS`) when at least one non-
        preposition `tok=` anchor remains, so `"look at X"` generalises to `"look into X"` / `"X needs a look"`.
    """
    if exp.get("statement") is not None:
        return None
    resolved = exp.get("resolved")
    rejected = exp.get("rejected")
    if resolved is None and rejected is None:
        return None
    raw_feats = list(exp.get("pfeats") or [])
    if any(f.startswith("tok=") and f[4:].lower() in PERSONAL_ROUTINE_WORDS for f in raw_feats):
        return None

    clean_feats = sorted(f for f in raw_feats if is_allowed_collective_feature(f, closed_vocab))
    if extraction_mode in ("drop_prepositions", "combined"):
        non_prep_toks = [f for f in clean_feats if f.startswith("tok=") and f[4:] not in PREPOSITION_TOKENS]
        if non_prep_toks:
            clean_feats = [f for f in clean_feats if not (f.startswith("tok=") and f[4:] in PREPOSITION_TOKENS)]

    if not personal.anchors(clean_feats):
        return None

    gen = dict(exp.get("general") or {"intent": [], "conf": 1.0})
    if extraction_mode in ("residual_cue", "combined") and resolved is not None:
        g_intent = personal.as_target(gen.get("intent"))
        r_target = personal.as_target(resolved)
        if g_intent == r_target and r_target:
            cue_toks = _general_cue_tokens(exp.get("text", ""))
            residual_feats = [f for f in clean_feats if f not in cue_toks]
            residual_toks = [f for f in residual_feats if f.startswith("tok=") and f[4:] not in PREPOSITION_TOKENS]
            if cue_toks and residual_toks:
                clean_feats = residual_feats
                gen = {"intent": [], "conf": 1.0}

    return {
        "id": exp["id"],
        "context": exp.get("context", ""),
        "pfeats": clean_feats,
        "objects": [],
        "general": gen,
        "statement": None,
        "resolved": resolved,
        "rejected": rejected,
        "pres_feedback": [],
        "accepted_presentation": {},
    }


def classify_candidate_category(trigger: list[str], target: list[str]) -> str:
    """Map a candidate to one of the four core directories in `clover-collective/core/`."""
    distinct_families = {c.split(".")[0] for c in target}
    if len(distinct_families) >= 2:
        return "problem-solving"
    if not any(f.startswith("ctx=") for f in trigger):
        if any(c in ("remember", "memory.search") for c in target):
            return "reasoning-patterns"
        return "tool-use-patterns"
    return "semantics"


class LocalCollectiveExtractor:
    """Extracts privacy-filtered, generalisable candidate knowledge packages from a local Clover installation."""

    def __init__(
        self,
        installation_id: str,
        consent_contribute: bool = False,
        oewn_path: str | Path | None = None,
        extraction_mode: str = "mismatch_only",
        learner_params: dict | None = None,
    ):
        self.installation_id = str(installation_id or "local")
        self.consent_contribute = bool(consent_contribute)
        self.closed_vocab = load_closed_general_vocabulary(oewn_path)
        self.extraction_mode = extraction_mode
        self.learner_params = dict(personal.PARAMS, **(learner_params or {}))
        self.pseudo_id = hashlib.sha256(f"clover-collective-v1:{self.installation_id}".encode("utf-8")).hexdigest()[:16]

    def _experiences_from(self, source) -> list[dict]:
        if isinstance(source, personal.Personal):
            return source.store.eligible()
        return list(source or [])

    def _build_scrubbed_learner(self, experiences: list[dict]) -> personal.Learner:
        learner = personal.Learner(self.learner_params)
        for exp in experiences:
            scrubbed = scrub_experience_for_collective(exp, self.closed_vocab, extraction_mode=self.extraction_mode)
            if scrubbed is not None:
                learner.observe(scrubbed)
        return learner

    def preview_candidates(self, source) -> list[dict]:
        """Return the exact list of candidate packages that would be submitted if `consent_contribute` is True."""
        experiences = self._experiences_from(source)
        learner = self._build_scrubbed_learner(experiences)
        packages = []
        for h in learner.hyps:
            if h.status != "validated" or h.origin == "statement":
                continue
            trig = sorted(h.trigger)
            if not personal.anchors(trig):
                continue
            if not all(is_allowed_collective_feature(f, self.closed_vocab) for f in trig):
                continue
            target_tuple = h.target if isinstance(h.target, tuple) else (h.target,)
            target_list = list(target_tuple)
            if not target_list or not all(isinstance(c, str) and c in semantics.CAP_WORDS for c in target_list):
                continue
            category = classify_candidate_category(trig, target_list)
            strategy_note = STRATEGY_GUIDANCE.get(tuple(target_list), "")
            pkg_id = hashlib.sha256(
                json.dumps({"inst": self.pseudo_id, "trigger": trig, "target": target_list}, sort_keys=True).encode()
            ).hexdigest()[:16]
            packages.append({
                "package_id": f"cand_{pkg_id}",
                "schema_version": SCHEMA_VERSION,
                "installation_id": self.pseudo_id,
                "category": category,
                "trigger": trig,
                "target": target_list,
                "strategy": strategy_note,
                "local_support_count": len(h.support),
                "local_contra_count": len(h.contra),
                "local_precision": round(h.precision(), 4),
                "cross_context_generalised": not any(f.startswith("ctx=") for f in trig),
            })
        return packages

    def export_candidates(self, source) -> list[dict]:
        """Export candidate packages ONLY when `consent_contribute` is True."""
        if not self.consent_contribute:
            return []
        return self.preview_candidates(source)
