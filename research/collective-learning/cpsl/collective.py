"""`collective.py` — Local candidate extractor and privacy boundary on each Clover installation.

Implements the local side of the collective learning boundary (`whats-needed-for-clover.txt` §§1, 2, 5, 6):
1. Strict separation between Personal Learning and Collective Candidates:
   - Never exports user preferences (`format`, `length`, `confirm`, `plan`, `options`, `sources`),
   - Never exports personal routines (`"the usual"`, `"the Friday routine"`) or open items,
   - Never exports raw conversation prompts, reactions, timestamps, object names, or session IDs.
2. Pre-Extraction Closed-Vocabulary (OEWN 2025 + CSI) & PII Gate (solving Stage 2 M5 Symbolic Unlearning Ripple Effect):
   - Every feature in an experience is filtered against the closed general vocabulary BEFORE collective candidate
     synthesis, so deleting or excluding a second experience can never unmask a private token (`secret_project_alpha`,
     email, path, SSN, identifier) from a remaining experience.
   - Any trigger containing a token outside the closed general vocabulary or matching a PII/identifier pattern is
     dropped locally and never exported.
3. Consent & Preview (`consent_contribute`):
   - `preview_candidates()` returns the exact inspectable JSON list that would be submitted so the user can review it.
   - `export_candidates()` returns `[]` when `consent_contribute` is False.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ppl import personas as P
from ppl import world as W
from ppl.learner import Learner, anchors, trigger_features

SCHEMA_VERSION = "1.0.0"

# Patterns that must NEVER appear in any candidate feature
PII_OR_SECRET_RE = re.compile(
    r"(@|https?://|www\.|/|\\|\bssn\b|\bsecret\b|\bconfidential\b|\bprivate\b|\bacmecorp\b|\bproject_codename\b|"
    r"\bpatient_record\b|\bbank_routing\b|\bsalary_review\b|\bmerger_target\b|\bapi_secret\b|\bssh_key\b|\d)",
    re.I,
)

# Routine words that are personal by definition and never exported to the collective core
PERSONAL_ROUTINE_WORDS = {
    w.lower()
    for phrase in P.ROUTINE_PHRASES
    for w in re.findall(r"[a-z]+", phrase.lower())
    if w not in ("the", "of")
}

_OEWN_CACHE: set[str] | None = None


def load_closed_general_vocabulary(oewn_gz_path: str | Path | None = None) -> set[str]:
    """Load the closed general English lemma vocabulary from Open English WordNet 2025 (when present on k3)
    plus the general world vocabulary (excluding all proper names, filenames, and private identifiers)."""
    global _OEWN_CACHE
    if _OEWN_CACHE is not None:
        return _OEWN_CACHE

    vocab = set()
    if oewn_gz_path is None:
        oewn_gz_path = os.environ.get(
            "CSI_DICT", "/opt/clover-k3/meaning-core/data/raw/english-wordnet-2025.xml.gz"
        )
    p = Path(oewn_gz_path)
    if p.exists():
        with gzip.open(p, "rb") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if elem.tag == "Form":
                    w = (elem.attrib.get("writtenForm") or "").strip().lower()
                    if re.fullmatch(r"[a-z][a-z\-]{1,28}", w):
                        vocab.add(w)
                    elem.clear()

    # Ensure standard general task verbs/nouns used in the simulation domain (e.g., preflight, ptal, fyi, triage,
    # corroborate, relocate, distill, condense, bookmark, shelve, snapshot, wrap, briefing) are present even in offline unit tests
    domain_general_words = {
        "preflight", "audit", "relocate", "safely", "prepare", "briefing", "build", "together", "triage", "file",
        "corroborate", "claim", "source", "bookmark", "list", "shelve", "wrap", "up", "today", "all", "snapshot",
        "distill", "key", "point", "reading", "condense", "bit", "reader", "tidy", "clean", "look", "kill", "polish",
        "ship", "stash", "long", "red", "test", "answer", "reply", "wait", "mess", "see", "clash", "collide",
        "overlap", "park", "zap", "check", "send", "bump", "format", "reformat", "prettify", "formatter", "run",
        "suite", "commit", "push", "change", "work", "deploy", "release", "revert", "roll", "back", "undo", "last",
        "review", "over", "explain", "walk", "through", "tell", "what", "refactor", "restructure", "summarize",
        "gist", "sum", "proofread", "typo", "shorten", "trim", "cut", "down", "expand", "flesh", "out", "detail",
        "translate", "put", "into", "export", "convert", "save", "share", "with", "email", "write", "off", "archive",
        "away", "flag", "mark", "important", "star", "forward", "pass", "schedule", "set", "book", "move",
        "reschedule", "cancel", "call", "find", "free", "slot", "time", "remind", "reminder", "open", "pull",
        "rename", "trash", "bin", "nuke", "delete", "remove", "zip", "compress", "backup", "search", "information",
        "define", "term", "mean", "compare", "weigh", "against", "note", "todo", "later", "come", "keep", "done",
        "drop", "still", "left", "show", "item", "scrub",
    }
    vocab |= domain_general_words
    # Explicitly exclude personal routine tokens ("usual", "friday", "morning", "weekly")
    vocab -= PERSONAL_ROUTINE_WORDS
    _OEWN_CACHE = vocab
    return vocab


def is_allowed_collective_feature(feat: str, closed_vocab: set[str]) -> bool:
    """Return True iff `feat` is a safe, non-PII, generalisable feature eligible for collective candidate extraction."""
    if not isinstance(feat, str) or PII_OR_SECRET_RE.search(feat):
        return False
    if feat.startswith("ctx="):
        return feat[4:] in W.CONTEXTS
    if feat.startswith("has="):
        return feat[4:] in W.LEXICON
    if feat.startswith("pat="):
        return feat[4:] in ("arrow", "colon")
    if feat.startswith("tok="):
        lemma = feat[4:].lower()
        if not re.fullmatch(r"[a-z][a-z\-]{1,28}", lemma):
            return False
        if lemma in PERSONAL_ROUTINE_WORDS:
            return False
        return lemma in closed_vocab
    return False


def scrub_experience_for_collective(exp: dict, closed_vocab: set[str]) -> dict | None:
    """Create a scrubbed, minimal local experience record containing ONLY closed-vocabulary features and general
    resolved targets—used to build collective candidates immune to the Stage 2 M5 Symbolic Unlearning Ripple Effect.
    Excludes statements, open-item management queries, and personal routines."""
    if exp.get("statement") is not None:
        return None
    resolved = exp.get("resolved")
    if resolved is None:
        return None
    # Convert list target back to tuple or str
    target = tuple(resolved) if isinstance(resolved, (list, tuple)) else resolved
    # Personal 2-step routines triggered by personal routine phrases ("the usual", "the Friday routine") are excluded
    raw_feats = set(exp.get("pfeats") or [])
    if any(f.startswith("tok=") and f[4:] in PERSONAL_ROUTINE_WORDS for f in raw_feats):
        return None
    clean_feats = sorted(f for f in raw_feats if is_allowed_collective_feature(f, closed_vocab))
    if not anchors(clean_feats):
        return None
    return {
        "id": exp["id"],
        "context": exp.get("context", ""),
        "pfeats": clean_feats,
        "objects": [],
        "general": exp.get("general"),
        "statement": None,
        "resolved": resolved,
        "pres_feedback": [],
        "accepted_presentation": {},
    }


def classify_candidate_type(trigger: list[str], target) -> str:
    """Classify a candidate into one of the four core knowledge categories in `whats-needed-for-clover.txt` §3."""
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return "problem-solving"
    t_str = target[0] if isinstance(target, (list, tuple)) else str(target)
    if not any(f.startswith("ctx=") for f in trigger):
        if t_str in ("notes.later", "notes.done"):
            return "reasoning-patterns"
        return "tool-use-patterns"
    return "semantics"


class LocalCollectiveExtractor:
    """Extracts privacy-filtered, generalisable candidate knowledge packages from a local Clover installation."""

    def __init__(self, installation_id: str, consent_contribute: bool = False, oewn_path: str | Path | None = None):
        self.installation_id = installation_id
        self.consent_contribute = bool(consent_contribute)
        self.closed_vocab = load_closed_general_vocabulary(oewn_path)
        self.pseudo_id = hashlib.sha256(f"clover-cpsl-v1:{installation_id}".encode("utf-8")).hexdigest()[:16]

    def _build_scrubbed_learner(self, eligible_experiences: list[dict], personal_csi: bool = True) -> Learner:
        """Replay eligible experiences through a fresh Learner AFTER closed-vocabulary scrubbing so no private token
        can ever enter a candidate trigger (even if other experiences are later deleted)."""
        learner = Learner(dict(K=2, P=0.7, R=2, retire=True, apply_provisional=False, use_ctx=True, C=0.0, mode="full"))
        for exp in eligible_experiences:
            scrubbed = scrub_experience_for_collective(exp, self.closed_vocab)
            if scrubbed is not None:
                learner.observe(scrubbed)
        return learner

    def preview_candidates(self, eligible_experiences: list[dict]) -> list[dict]:
        """Return the exact list of candidate packages that would be submitted if `consent_contribute` is enabled.
        Allows the user to inspect every field before consenting (`whats-needed-for-clover.txt` §5)."""
        learner = self._build_scrubbed_learner(eligible_experiences)
        packages = []
        for h in learner.hyps:
            if h.status != "validated" or h.origin == "statement":
                continue
            trig = sorted(h.trigger)
            if not anchors(trig):
                continue
            if not all(is_allowed_collective_feature(f, self.closed_vocab) for f in trig):
                continue
            target_out = list(h.target) if isinstance(h.target, tuple) else [h.target]
            # Validate all target intents exist in general intent catalog
            if not all(t in W.INTENTS for t in target_out):
                continue
            category = classify_candidate_type(trig, h.target)
            pkg_body = {
                "schema_version": SCHEMA_VERSION,
                "installation_id": self.pseudo_id,
                "category": category,
                "trigger": trig,
                "target": target_out,
                "local_support_count": len(h.support),
                "local_contra_count": len(h.contra),
                "local_precision": round(h.precision(), 4),
                "cross_context_generalised": not any(f.startswith("ctx=") for f in trig),
            }
            pkg_id = hashlib.sha256(
                json.dumps({"inst": self.pseudo_id, "trigger": trig, "target": target_out}, sort_keys=True).encode()
            ).hexdigest()[:16]
            pkg_body["package_id"] = f"cand_{pkg_id}"
            packages.append(pkg_body)
        return packages

    def export_candidates(self, eligible_experiences: list[dict]) -> list[dict]:
        """Export candidate packages to the collective gateway ONLY when `consent_contribute` is True."""
        if not self.consent_contribute:
            return []
        return self.preview_candidates(eligible_experiences)
