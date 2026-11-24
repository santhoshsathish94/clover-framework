"""Clover Studio Capability S4: Symbolic Ontology Deductive Closure & Constraint Verifier.

Distilled cleanly from `clover-ai-system/studio/clover/symbolic-ontology-graph.py`
(`CloverOntologyGraph` + `CloverSymbolicVerifier`):
  1. Parses explicit relational triples (`DEPENDS_ON`, `AFFECTED_BY`, `CWE_MAPS_TO`,
     `EXPLOITED_BY`, `ENTAILS_CONSTRAINT`, `FORBIDS`) and manifest/config flag bindings
     from the user's prompt into an ephemeral per-request directed ontology graph.
  2. Computes deterministic Horn-rule fixed-point deductive closure:
       - `A DEPENDS_ON B & B DEPENDS_ON C => A DEPENDS_ON* C`
       - `A DEPENDS_ON* B & B ENTAILS_CONSTRAINT K => A REQUIRES_CONSTRAINT K`
       - `A DEPENDS_ON* B & B FORBIDS F => A FORBIDS_FLAG F`
       - `A DEPENDS_ON* B & B AFFECTED_BY_ACTIVE V & V CWE_MAPS_TO W & W EXPLOITED_BY X => A EXPOSED_TO X`
         (respecting `patched_in == deployed` non-propagation guards!)
  3. Verifies whether the assistant's `VERDICT: COMPLIANT | VIOLATION` and cited IDs match
     the symbolic deductive closure, triggering 1 LCP repair turn ONLY on contradiction.
"""

from __future__ import annotations

import re
from typing import Any

_RELATIONS = (
    "DEPENDS_ON",
    "AFFECTED_BY",
    "CWE_MAPS_TO",
    "EXPLOITED_BY",
    "ENTAILS_CONSTRAINT",
    "FORBIDS",
)

_TRIPLE_RE = re.compile(
    r"[`'\"]?([A-Z][A-Z0-9_-]{1,40})[`'\"]?\s+"
    r"(DEPENDS_ON|AFFECTED_BY|CWE_MAPS_TO|EXPLOITED_BY|ENTAILS_CONSTRAINT|FORBIDS)\s+"
    r"[`'\"]?([A-Z][A-Z0-9_-]{1,40})[`'\"]?"
    r"(?:\s*\(([^)\n]*)\))?",
)

_PATCH_META_RE = re.compile(
    r"patched_in\s*:\s*[`'\"]?([0-9A-Za-z._-]+)[`'\"]?\s*,\s*deployed\s*:\s*[`'\"]?([0-9A-Za-z._-]+)[`'\"]?",
    re.IGNORECASE,
)

_RULE_CAPEC_FLAG_RE = re.compile(
    r"mapped\s+to\s+[`'\"]?(CAPEC-\d+)[`'\"]?\s+MUST\s+enable\s+[`'\"]?([A-Z][A-Z0-9_-]+)[`'\"]?",
    re.IGNORECASE,
)

_TRUE_VALS = {"true", "enabled", "on", "1", "yes"}
_FALSE_VALS = {"false", "disabled", "off", "0", "no"}


def _parse_scoped_and_global_bindings(prompt_text: str, entities: set[str]) -> tuple[dict[str, dict[str, bool]], dict[str, bool]]:
    """Extract per-entity flag bindings (`SVC: {FLAG: false}`, `for X sets K = false`, `X with K = true`) and global bindings."""
    scoped: dict[str, dict[str, bool]] = {}
    global_flags: dict[str, bool] = {}

    # 1. Dict-style: `SVC-AUTH: {FLAG-SANDBOX-ISOLATE: false}`
    for m in re.finditer(r"([A-Z][A-Z0-9_-]+)\s*:\s*\{([^}]+)\}", prompt_text):
        ent = m.group(1)
        body = m.group(2)
        for kv in re.finditer(r"([A-Z][A-Z0-9_-]+)\s*[:=]\s*(true|false|enabled|disabled)", body, re.IGNORECASE):
            val = kv.group(2).lower() in _TRUE_VALS
            scoped.setdefault(ent, {})[kv.group(1)] = val

    # 2. Clause-style: "`MOD-GATEWAY` with `CST-ALIGN-64B = true`" or "`NODE-CLOUD` with `CST-FP-CONTRACT-OFF = false`"
    for seg in re.split(r"(?:\band\b|[.;\n])", prompt_text):
        ent_hits = [e for e in entities if re.search(r"\b" + re.escape(e) + r"\b", seg)]
        kv_hits = list(re.finditer(r"[`'\"]?([A-Z][A-Z0-9_-]+)[`'\"]?\s*=\s*[`'\"]?(true|false|enabled|disabled)[`'\"]?", seg, re.IGNORECASE))
        if not kv_hits:
            continue
        # Look for explicit `for ENT` or `on ENT` or `ENT with`
        owners = [e for e in ent_hits if not any(e == kv.group(1) for kv in kv_hits)]
        if len(owners) == 1:
            for kv in kv_hits:
                scoped.setdefault(owners[0], {})[kv.group(1)] = kv.group(2).lower() in _TRUE_VALS
        elif not owners:
            # Check if previous sentence half named `Manifest for PIPE-TRAIN sets ...`
            for kv in kv_hits:
                global_flags[kv.group(1)] = kv.group(2).lower() in _TRUE_VALS

    # Also catch `Manifest for X sets A = true and B = false` across `and`
    for line in prompt_text.splitlines():
        m_for = re.search(r"(?:for|on)\s+[`'\"]?([A-Z][A-Z0-9_-]+)[`'\"]?", line, re.IGNORECASE)
        if m_for and m_for.group(1) in entities:
            owner = m_for.group(1)
            for kv in re.finditer(r"[`'\"]?([A-Z][A-Z0-9_-]+)[`'\"]?\s*=\s*[`'\"]?(true|false|enabled|disabled)[`'\"]?", line, re.IGNORECASE):
                scoped.setdefault(owner, {})[kv.group(1)] = kv.group(2).lower() in _TRUE_VALS

    return scoped, global_flags


def evaluate_ontology_closure(prompt_text: str) -> dict[str, Any] | None:
    """Build ephemeral ontology graph, compute Horn deductive closure, and return violations if ontology triples exist."""
    triples = list(_TRIPLE_RE.finditer(prompt_text))
    if len(triples) < 2 or "VERDICT" not in prompt_text.upper():
        return None

    deps: dict[str, set[str]] = {}
    entails: dict[str, set[str]] = {}
    forbids: dict[str, set[str]] = {}
    affected_active: dict[str, set[str]] = {}
    cwe_map: dict[str, set[str]] = {}
    exploited_by: dict[str, set[str]] = {}
    entities: set[str] = set()

    for m in triples:
        subj, rel, obj, meta = m.group(1), m.group(2), m.group(3), (m.group(4) or "")
        entities.add(subj)
        entities.add(obj)
        if rel == "DEPENDS_ON":
            deps.setdefault(subj, set()).add(obj)
        elif rel == "ENTAILS_CONSTRAINT":
            entails.setdefault(subj, set()).add(obj)
        elif rel == "FORBIDS":
            forbids.setdefault(subj, set()).add(obj)
        elif rel == "AFFECTED_BY":
            pm = _PATCH_META_RE.search(meta)
            if pm and pm.group(1) == pm.group(2):
                # Patched version matches deployed version -> vulnerability is inactive
                continue
            affected_active.setdefault(subj, set()).add(obj)
        elif rel == "CWE_MAPS_TO":
            cwe_map.setdefault(subj, set()).add(obj)
        elif rel == "EXPLOITED_BY":
            exploited_by.setdefault(subj, set()).add(obj)

    # Transitive closure of DEPENDS_ON* (reflexive + transitive)
    reach: dict[str, set[str]] = {e: {e} for e in entities}
    changed = True
    while changed:
        changed = False
        for u in list(entities):
            cur = reach[u]
            add: set[str] = set()
            for mid in cur:
                add |= deps.get(mid, set())
            if not add.issubset(cur):
                cur |= add
                changed = True

    # Rule-derived CAPEC -> required mitigation flag
    capec_req_flag: dict[str, str] = {}
    for rm in _RULE_CAPEC_FLAG_RE.finditer(prompt_text):
        capec_req_flag[rm.group(1)] = rm.group(2)

    scoped_flags, global_flags = _parse_scoped_and_global_bindings(prompt_text, entities)

    # Root components: entities that appear on LHS of DEPENDS_ON and have scoped/global config or exposure query
    roots = set(deps.keys()) | set(scoped_flags.keys())
    violations: list[dict[str, str]] = []

    def _lookup_flag(ent: str, flag: str) -> bool | None:
        if ent in scoped_flags and flag in scoped_flags[ent]:
            return scoped_flags[ent][flag]
        if flag in global_flags:
            return global_flags[flag]
        return None

    for root in sorted(roots):
        for target in sorted(reach.get(root, {root})):
            # 1. Transitive ENTAILS_CONSTRAINT
            for cst in sorted(entails.get(target, set())):
                val = _lookup_flag(root, cst)
                if val is False:
                    violations.append({
                        "entity": root,
                        "constraint": cst,
                        "reason": f"{root} ->* {target} ENTAILS_CONSTRAINT {cst}, but {cst}=false",
                    })
            # 2. Transitive FORBIDS
            for fflag in sorted(forbids.get(target, set())):
                val = _lookup_flag(root, fflag)
                if val is True:
                    violations.append({
                        "entity": root,
                        "constraint": fflag,
                        "reason": f"{root} ->* {target} FORBIDS {fflag}, but {fflag}=enabled",
                    })
            # 3. Transitive CVE -> CWE -> CAPEC exposure
            for cve in sorted(affected_active.get(target, set())):
                for cwe in sorted(cwe_map.get(cve, set())):
                    for capec in sorted(exploited_by.get(cwe, set())):
                        req_flag = capec_req_flag.get(capec)
                        if req_flag is not None:
                            val = _lookup_flag(root, req_flag)
                            if val is False:
                                violations.append({
                                    "entity": root,
                                    "constraint": req_flag,
                                    "reason": f"{root} ->* {target} exposed to {cve}->{cwe}->{capec} without {req_flag}",
                                })
                        elif re.search(r"\bexposed\s+to\s+[`'\"]?" + re.escape(capec), prompt_text, re.IGNORECASE):
                            violations.append({
                                "entity": root,
                                "constraint": capec,
                                "reason": f"{root} ->* {target} actively exposed to {capec}",
                            })

    expected_verdict = "VIOLATION" if violations else "COMPLIANT"
    return {
        "triples_count": len(triples),
        "expected_verdict": expected_verdict,
        "violations": violations,
    }


def check_symbolic_ontology(
    messages: list[dict[str, Any]],
    draft_content: str,
    tool_calls: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Run CloverSymbolicVerifier on ontology prompts; return `(repair_prompt_or_None, meta)`."""
    if tool_calls or not messages:
        return None, {"active": False}
    prompt_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            prompt_text = c if isinstance(c, str) else str(c or "")
            break
    if not prompt_text.strip():
        return None, {"active": False}

    closure = evaluate_ontology_closure(prompt_text)
    if closure is None:
        return None, {"active": False}

    cleaned = re.sub(r"<think>.*?</think>", "", draft_content or "", flags=re.DOTALL).strip()
    m = re.search(r"VERDICT\s*:\s*(COMPLIANT|VIOLATION)", cleaned, flags=re.IGNORECASE)
    got_verdict = m.group(1).upper() if m else ""
    exp_verdict = closure["expected_verdict"]

    missing_citations: list[str] = []
    up = cleaned.upper()
    for v in closure["violations"]:
        if v["entity"].upper() not in up and v["constraint"].upper() not in up:
            missing_citations.append(f"{v['entity']}/{v['constraint']}")

    if got_verdict == exp_verdict and not missing_citations:
        return None, {"active": True, "triples": closure["triples_count"], "verdict": exp_verdict, "repaired": False}

    trace = "; ".join(v["reason"] for v in closure["violations"]) or "All transitive obligations satisfied (0 violations)."
    repair = (
        f"Deterministic `CloverSymbolicVerifier` deductive closure found `{exp_verdict}`: {trace}\n"
        f"Re-emit your response starting line 1 with `VERDICT: {exp_verdict}` and citing only the exact violating entity and constraint/flag IDs."
    )
    return repair, {"active": True, "triples": closure["triples_count"], "verdict": exp_verdict, "repaired": True}
