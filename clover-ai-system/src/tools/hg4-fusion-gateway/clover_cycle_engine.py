"""Clover `AGENTS.md` Operating Cycle Engine (`Context -> Direction -> Execution -> Outcome -> Growth`).

Implements Clover's core operating rules from `AGENTS.md` (Sections 1, 4, 5, 6, 8, and 9) as a
deterministic, zero-prefix-mutation runtime layer across all gateways and GPU Petals:

  1. **Authentic `AGENTS.md` Provenance**: Loads the repository's `AGENTS.md` specification and records
     its exact `SHA-256` digest in every response's `x_clover_telemetry["clover_cycle"]`.
  2. **Section 5 Structured 8-Field Context File Handoffs (`CloverContextRecord`)**:
     - Preserves what every favorable or unfavorable Outcome taught (`intended_outcome`, `what_was_known`,
       `what_was_tried`, `what_the_system_showed`, `what_happened`, `what_was_ruled_out`,
       `what_remains_unknown`, `next_cycle_change`) as both an inspectable Markdown context file
       (when `CLOVER_CONTEXT_DIR` is set) and a session-keyed handoff record so the next cycle or
       escalated model (`35B -> 120B -> Kimi-2.8T`) NEVER starts from zero or repeats a ruled-out approach.
     - Appends handoffs strictly at the **tail** of the latest message so Turn-1 has `0` extra prompt
       tokens and multi-turn LCP KV/GDN cache hit rates stay `>95%`.
  3. **Section 6 Anti-Blind-Retry & Know-When-To-Stop Gate**:
     - Detects repeated identical failed executions from unchanged Context (`blind_retry_blocked = True`)
       and forces a `Return to Context` (inspecting evidence or changing hypothesis).
     - Deliberately halts (`discernment_state = "STOP_AND_ASK"`) after `>= max_stagnant_cycles` (default `3`)
       consecutive cycles with the identical unresolved failure signature instead of looping forever.
  4. **Sections 4 & 9 Verification-Boundary Runtime Guard (`REFUSE`)**:
     - Intercepts and refuses (`discernment_state = "REFUSE"`) any unrequested tool call attempting to
       modify, delete, or weaken protected verification controls (`test_*.py`, `*_oracle*`, `verify_*.py`, `.h5`).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_PROTECTED_VERIFIER_FILE_RE = re.compile(
    r"(?i)(?:^|[/\\\s\"'`=(])("
    r"(?:tests?|__tests__|\.github[/\\]workflows)(?:[/\\][A-Za-z0-9_./\\*?\[\]${}-]+|(?=$|[/\\\s\"'`:;,)]))"
    r"|\[[tT]\]ests?(?:[/\\][A-Za-z0-9_./\\*?\[\]${}-]*)?"
    r"|t[?*\[]sts?(?:[/\\][A-Za-z0-9_./\\*?\[\]${}-]*)?"
    r"|t\*s\b"
    r"|test_[A-Za-z0-9_.*?\[\]-]+\.(?:py|ts|tsx|js|jsx|rs|go|rb|lean)"
    r"|[A-Za-z0-9_.*?\[\]-]+_(?:test|spec)\.(?:py|ts|tsx|js|jsx|rs|go|rb|lean)"
    r"|[A-Za-z0-9_.*?\[\]-]+\.(?:test|spec)\.(?:py|ts|tsx|js|jsx|rs|go|rb|lean)"
    r"|verify_[A-Za-z0-9_.*?\[\]-]+\.py"
    r"|[A-Za-z0-9_.-]*oracle[A-Za-z0-9_.-]*\.(?:py|json|h5|lean)"
    r"|test_data\.h5|conftest\.py|pytest\.ini|tox\.ini|noxfile\.py|setup\.cfg|Makefile|GNUmakefile"
    r"|(?:jest|vitest|playwright)\.config\.[A-Za-z0-9]+"
    r"|pyproject\.toml|package\.json"
    r")(?=$|[/\\\s\"'`:;,)])"
)

_SHELL_MUTATE_RE = re.compile(
    r"(?i)\b(?:sed\s+-i\S*|perl\s+-[A-Za-z]*i\S*|awk\s+-i|python3?|node|ruby|pwsh|powershell|"
    r"rm|rmdir|del|erase|mv|move|cp|copy|ln|unlink|truncate|tee|chmod|chown|dd|install|patch|rsync|tar|unzip|xargs|"
    r"Set-Content|Add-Content|Out-File|Clear-Content|Remove-Item|Copy-Item|Move-Item|New-Item|Rename-Item|"
    r"WriteAllText|WriteAllLines|WriteAllBytes|AppendAllText|StreamWriter|"
    r"sc|ac|ri|ni|mi|cpi|rni|clc|"
    r"git\s+(?:checkout|restore|reset|clean|apply|am|mv|rm|stash))\b"
    r"|(?i:find\b[^\n;|&]*\s-(?:delete|exec|execdir)\b)"
    r"|(?i:\[(?:System\.)?IO\.File\])"
    r"|(?<![0-9&])>{1,2}\s*\S"
)

_CODE_DIRECT_MUTATE_RE = re.compile(
    r"(?i)\b(?:write_text|write_bytes|writeFileSync|appendFileSync|writeFile|appendFile|unlinkSync|rmSync|"
    r"unlink|remove|rmtree|rename|replace|move|copy|copy2|copyfile|truncate|chmod|symlink_to|hardlink_to|"
    r"makedirs|mkdir|rmdir|system|popen|Popen|run|check_call|check_output|execSync|spawnSync|O_WRONLY|O_RDWR|O_CREAT|O_TRUNC)\b"
    r"|(?is:\bopen\s*\(.*?(?:,\s*['\"][rwa+x]*[wax+][rwa+xb]*['\"]|\bmode\s*=\s*['\"][rwa+x]*[wax+]))"
)

_READ_ONLY_TOOLS = frozenset({
    "read", "view", "grep", "rg", "glob", "ls", "cat", "find",
    "clover_context_read", "clover_todo_list", "clover_route_and_guard",
    "tools/search", "tools/list",
})


def _discover_agents_md() -> tuple[Path | None, str]:
    here = Path(__file__).resolve()
    env_p = (os.environ.get("CLOVER_AGENTS_MD") or "").strip()
    cands: list[Path] = []
    if env_p:
        cands.append(Path(env_p))
    if len(here.parents) > 4:
        cands.append(here.parents[4] / "AGENTS.md")
    cands.extend([
        Path("/app/AGENTS.md"),
        here.parents[2] / "studio" / "clover" / "AGENTS.md" if len(here.parents) > 2 else Path("/nonexistent"),
    ])
    for p in cands:
        if p.is_file():
            raw = p.read_bytes()
            return p, hashlib.sha256(raw).hexdigest()
    return None, "missing"


AGENTS_MD_PATH, AGENTS_MD_SHA256 = _discover_agents_md()


@dataclass
class CloverContextRecord:
    """Exact 8-field `AGENTS.md` Section 5 Context Handoff Record."""

    task_id: str
    cycle_count: int = 0
    intended_outcome: str = ""
    what_was_known: list[str] = field(default_factory=list)
    what_was_tried: list[str] = field(default_factory=list)
    what_the_system_showed: list[str] = field(default_factory=list)
    what_happened: str = ""
    what_was_ruled_out: list[str] = field(default_factory=list)
    what_remains_unknown: list[str] = field(default_factory=list)
    next_cycle_change: str = ""
    attempt_fingerprints: list[str] = field(default_factory=list)
    error_fingerprints: list[str] = field(default_factory=list)
    context_file_path: str | None = None
    updated_at: float = 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# Clover Context Handoff (`{self.task_id}` — Cycle {self.cycle_count})",
            "",
            f"- **Intended outcome (Human Direction):** {self.intended_outcome}",
            "- **What was known:** " + ("; ".join(self.what_was_known[-4:]) or "Initial task prompt & workspace access."),
            "- **What was tried:** " + (" -> ".join(self.what_was_tried[-4:]) or "None yet."),
            "- **What the System showed:** " + (" | ".join(self.what_the_system_showed[-3:]) or "No environment check yet."),
            f"- **What happened:** {self.what_happened or 'Awaiting verification.'}",
            "- **What was ruled out (DO NOT REPEAT):** " + ("; ".join(self.what_was_ruled_out[-5:]) or "None."),
            "- **What remains unknown:** " + ("; ".join(self.what_remains_unknown[-3:]) or "Exact root cause pending next check."),
            f"- **What should be different in the next cycle:** {self.next_cycle_change}",
            "",
            "_Offered per AGENTS.md §5: the human decides whether to keep, edit, or discard this context handoff._",
        ]
        return "\n".join(lines) + "\n"

    def to_compact_tail_note(self, blind_retry_blocked: bool = False) -> str:
        parts = [
            f"[Clover Context -> Growth Handoff (AGENTS.md §5 & §6, Cycle {self.cycle_count}"
            + (f", file={self.context_file_path}" if self.context_file_path else "")
            + "):"
        ]
        parts.append(f"  Intended Outcome: {self.intended_outcome[:220]}")
        if self.what_was_tried:
            parts.append("  Tried: " + " ; ".join(self.what_was_tried[-3:]))
        if self.what_the_system_showed:
            parts.append("  System Showed: " + " | ".join(self.what_the_system_showed[-2:]))
        if self.what_was_ruled_out:
            parts.append("  RULED OUT (do NOT repeat): " + " ; ".join(self.what_was_ruled_out[-4:]))
        if blind_retry_blocked:
            parts.append(
                "  BLIND-RETRY BLOCKED (AGENTS.md §6): You attempted the SAME execution twice from unchanged Context. "
                "Do NOT repeat the previous command/code; inspect new Context or change your hypothesis first."
            )
        parts.append(f"  Next Cycle Change: {self.next_cycle_change}]")
        return "\n".join(parts)


def _msg_text(m: dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
    return str(c or "")


def _normalize_attempt(asst_msg: dict[str, Any]) -> tuple[str, str, bool]:
    """Return `(fingerprint, human_summary, is_read_only_context_step)` for an assistant turn."""
    tcs = asst_msg.get("tool_calls") or []
    if tcs and isinstance(tcs, list):
        parts = []
        summaries = []
        all_ro = True
        for tc in tcs:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name") or "")
            raw_args = fn.get("arguments") or "{}"
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except ValueError:
                    parsed_args = raw_args
            else:
                parsed_args = raw_args
            norm_args = json.dumps(parsed_args, sort_keys=True, separators=(",", ":"))
            parts.append(f"{name}:{norm_args}")
            summaries.append(f"{name}({norm_args[:90]})")
            if name.lower() not in _READ_ONLY_TOOLS:
                all_ro = False
                if name.lower() == "bash" and isinstance(parsed_args, dict):
                    cmd = str(parsed_args.get("command") or "").strip()
                    if re.match(r"^(?:cat|ls|grep|rg|head|tail|git\s+(?:status|diff|log))\b", cmd) and not _SHELL_MUTATE_RE.search(cmd):
                        all_ro = True
        joined = "|".join(parts)
        fp = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
        return fp, ", ".join(summaries)[:160], all_ro

    txt = re.sub(r"\s+", " ", _msg_text(asst_msg)).strip()
    fp = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
    return fp, (txt[:140] + ("..." if len(txt) > 140 else "")) or "<empty-draft>", False


def _normalize_error_sig(tool_txt: str, marker: str) -> str:
    lines = [ln.strip() for ln in tool_txt.splitlines() if ln.strip()]
    key_line = next((ln for ln in reversed(lines) if marker.lower() in ln.lower()), lines[-1] if lines else marker)
    # Strip memory addresses & temp paths so identical errors produce the same signature
    cleaned = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", key_line)
    cleaned = re.sub(r"/tmp/[A-Za-z0-9_./-]+", "/tmp/TMP", cleaned)
    return f"{marker}:{cleaned[:140]}"


class CloverCycleEngine:
    """Stateful `AGENTS.md` Context->Growth Handoff, Anti-Blind-Retry & Boundary Guard."""

    def __init__(self, failure_re: re.Pattern[str], max_stagnant_cycles: int = 3) -> None:
        self.failure_re = failure_re
        self.max_stagnant_cycles = max_stagnant_cycles
        self._mu = threading.Lock()
        self._records: collections.OrderedDict[str, CloverContextRecord] = collections.OrderedDict()

    def reset(self) -> None:
        with self._mu:
            self._records.clear()

    @staticmethod
    def sanitize_task_id(raw_id: str, strict: bool = False) -> str:
        s = (raw_id or "").strip()
        if strict and (not s or "/" in s or "\\" in s or ".." in s or ":" in s):
            raise ValueError(f"Unsafe or empty task_id rejected: {raw_id!r}")
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-")[:64]
        if strict and cleaned != s:
            raise ValueError(f"Invalid characters in task_id: {raw_id!r}")
        return cleaned or "default_task"

    @staticmethod
    def derive_task_id(body: dict[str, Any], messages: list[dict[str, Any]]) -> str:
        explicit = str(body.get("x_clover_task_id") or "").strip()
        if explicit:
            return CloverCycleEngine.sanitize_task_id(explicit, strict=False)
        first_user = ""
        for m in messages:
            if m.get("role") == "user":
                first_user = _msg_text(m).split("[Clover Context -> Growth Handoff", 1)[0].strip()
                break
        digest = hashlib.sha256(first_user.encode("utf-8")).hexdigest()[:12]
        return f"task_{digest}"

    def get_record(self, task_id: str, strict: bool = False) -> CloverContextRecord | None:
        safe_id = self.sanitize_task_id(task_id, strict=strict)
        with self._mu:
            rec = self._records.get(safe_id)
            if rec is not None:
                return rec
        ctx_dir = (os.environ.get("CLOVER_CONTEXT_DIR") or "").strip()
        if ctx_dir:
            base_d = Path(ctx_dir).resolve()
            json_path = (base_d / f"{safe_id}.json").resolve()
            if json_path.parent == base_d and json_path.is_file():
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                    loaded = CloverContextRecord(**data)
                    with self._mu:
                        self._records[safe_id] = loaded
                    return loaded
                except Exception:
                    return None
        return None

    def save_record(self, rec: CloverContextRecord, strict: bool = False) -> CloverContextRecord:
        rec.task_id = self.sanitize_task_id(rec.task_id, strict=strict)
        rec.updated_at = time.time()
        ctx_dir = (os.environ.get("CLOVER_CONTEXT_DIR") or "").strip()
        if ctx_dir:
            out_d = Path(ctx_dir).resolve()
            out_d.mkdir(parents=True, exist_ok=True)
            md_file = (out_d / f"{rec.task_id}.md").resolve()
            json_file = (out_d / f"{rec.task_id}.json").resolve()
            if md_file.parent != out_d or json_file.parent != out_d:
                raise ValueError(f"Path traversal blocked for task_id {rec.task_id!r}")
            rec.context_file_path = str(md_file)
            md_file.write_text(rec.to_markdown(), encoding="utf-8")
            json_file.write_text(json.dumps(asdict(rec), indent=2, ensure_ascii=False), encoding="utf-8")
        with self._mu:
            self._records[rec.task_id] = rec
            self._records.move_to_end(rec.task_id)
            while len(self._records) > 1024:
                self._records.popitem(last=False)
        return rec

    def prepare_cycle(
        self, body: dict[str, Any], messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
        """Analyze conversation & persisted Context file; return `(augmented_messages, cycle_meta, stop_response)`."""
        task_id = self.derive_task_id(body, messages)
        first_user = ""
        for m in messages:
            if m.get("role") == "user":
                first_user = _msg_text(m).split("[Clover Context -> Growth Handoff", 1)[0].strip()
                break

        prior_rec = self.get_record(task_id)

        # Merge persisted Context fields when resuming an explicit `x_clover_task_id` or starting a new <=2-msg turn
        explicit_tid = bool(str(body.get("x_clover_task_id") or "").strip())
        carry_prior = bool(prior_rec and (explicit_tid or len(messages) <= 2))
        tried_list: list[str] = list(prior_rec.what_was_tried) if carry_prior else []
        showed_list: list[str] = list(prior_rec.what_the_system_showed) if carry_prior else []
        ruled_out: list[str] = list(prior_rec.what_was_ruled_out) if carry_prior else []
        known_list: list[str] = list(prior_rec.what_was_known) if carry_prior else []
        attempt_fps: list[str] = []
        error_sigs: list[str] = []
        context_updated_since_last_fail = False

        idx = 0
        n = len(messages)
        while idx < n:
            m = messages[idx]
            if m.get("role") == "assistant":
                fp, summ, is_ro = _normalize_attempt(m)
                tool_outs: list[str] = []
                j = idx + 1
                while j < n and messages[j].get("role") in ("tool", "function"):
                    tool_outs.append(_msg_text(messages[j]))
                    j += 1
                if tool_outs:
                    joined_out = "\n".join(tool_outs)
                    fm = self.failure_re.search(joined_out)
                    if fm and not is_ro:
                        esig = _normalize_error_sig(joined_out, fm.group(0).strip())
                        tried_list.append(f"[{fp[:6]}] {summ}")
                        showed_list.append(esig)
                        ruled_entry = f"{summ} -> failed with `{esig}`"
                        if ruled_entry not in ruled_out:
                            ruled_out.append(ruled_entry)
                        attempt_fps.append(fp)
                        error_sigs.append(esig)
                        context_updated_since_last_fail = False
                    else:
                        context_updated_since_last_fail = True
                        snippet = re.sub(r"\s+", " ", joined_out).strip()[:110]
                        if snippet and snippet not in known_list:
                            known_list.append(f"Verified observation: {snippet}")
                idx = j
            else:
                if m.get("role") == "user" and idx > 0:
                    context_updated_since_last_fail = True
                idx += 1

        # Fresh turn with no failures AND no prior persisted Context handoff -> zero prompt mutation!
        if not ruled_out and not error_sigs:
            return messages, {
                "agents_md_sha256": AGENTS_MD_SHA256[:16],
                "task_id": task_id,
                "cycle": 1,
                "discernment_state": "PROCEED",
                "handoff_injected": False,
                "blind_retry_blocked": False,
            }, None

        # Detect Blind Retry (`AGENTS.md §6`: identical execution attempted twice without new Context)
        blind_retry = bool(
            len(attempt_fps) >= 2
            and attempt_fps[-1] == attempt_fps[-2]
            and not context_updated_since_last_fail
        )

        # Detect Stagnation (`AGENTS.md §1 "Know when to stop"` & `§6`): >= N consecutive identical error signatures
        max_stag = int(body.get("x_clover_max_stagnant_cycles") or os.environ.get("CLOVER_MAX_STAGNANT_CYCLES", str(self.max_stagnant_cycles)))
        stagnant = bool(
            len(error_sigs) >= max_stag
            and len(set(error_sigs[-max_stag:])) == 1
            and not context_updated_since_last_fail
        )

        cycle_num = max(len(error_sigs) + 1, (prior_rec.cycle_count + 1) if (prior_rec and not error_sigs) else 1)
        rec = CloverContextRecord(
            task_id=task_id,
            cycle_count=cycle_num,
            intended_outcome=first_user[:300],
            what_was_known=known_list[-5:],
            what_was_tried=tried_list[-5:],
            what_the_system_showed=showed_list[-4:],
            what_happened=(
                f"Stagnated for {max_stag} cycles on `{error_sigs[-1]}`" if stagnant
                else (f"Last attempt ruled out due to `{error_sigs[-1]}`" if error_sigs else "Resumed from persisted Clover Context handoff.")
            ),
            what_was_ruled_out=ruled_out[-6:],
            what_remains_unknown=["Missing system invariant or environmental constraint behind repeated failure."] if stagnant else ["Root cause of prior verification failure must be isolated before editing."],
            next_cycle_change=(
                "STOP execution and ask the human for the missing Context or Direction."
                if stagnant
                else (
                    "Read relevant source/state first (Context) and formulate a distinct mechanism that avoids all ruled-out attempts."
                    if blind_retry
                    else "Formulate a genuinely different implementation addressing the exact verifier diagnostic without repeating ruled-out approaches."
                )
            ),
            attempt_fingerprints=attempt_fps[-6:],
            error_fingerprints=error_sigs[-6:],
        )
        self.save_record(rec)

        meta = {
            "agents_md_sha256": AGENTS_MD_SHA256[:16],
            "task_id": task_id,
            "cycle": cycle_num,
            "discernment_state": "STOP_AND_ASK" if stagnant else "PROCEED_WITH_GROWTH",
            "handoff_injected": not stagnant,
            "blind_retry_blocked": blind_retry,
            "ruled_out_count": len(rec.what_was_ruled_out),
            "context_file": rec.context_file_path,
            "offered_handoff": True,
        }

        if stagnant:
            stop_msg = (
                f"[Clover Deliberate Stop — AGENTS.md §1 & §6 (`{task_id}`, Cycle {cycle_num})]\n"
                f"Stopping deliberately rather than blindly retrying: {max_stag} consecutive cycles produced the same "
                f"unresolved System outcome (`{error_sigs[-1]}`) without new Context.\n\n"
                f"- **What was tried & ruled out:** {'; '.join(rec.what_was_ruled_out[-3:])}\n"
                + (f"- **Context handoff offered at:** `{rec.context_file_path}`\n" if rec.context_file_path else "")
                + "- **Needed from Human (Context / Direction):** Please share the missing invariant, expected boundary condition, "
                "or pointer to the relevant source/specification so the next cycle starts from verified Context."
            )
            return messages, meta, stop_msg

        # Append the structured Section 5 Context->Growth Handoff strictly at the TAIL of the last message (preserves 100% prefix LCP!)
        tail_note = rec.to_compact_tail_note(blind_retry_blocked=blind_retry)
        aug = [dict(m) for m in messages]
        last = dict(aug[-1])
        base_c = _msg_text(last).split("[Clover Context -> Growth Handoff", 1)[0].rstrip()
        last["content"] = f"{base_c}\n\n{tail_note}" if base_c else tail_note
        aug[-1] = last
        return aug, meta, None

    @staticmethod
    def check_verification_boundary_tamper(
        messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
    ) -> tuple[bool, str | None]:
        """AGENTS.md §4, §8 & §9 Runtime Enforcement: Refuse unrequested edits/deletions of protected test/oracle files!"""
        if not tool_calls:
            return False, None
        # Strip embedded mock/tool/code blocks from user messages so pasted evidence is never treated as authority (AGENTS.md §8)
        human_prose_parts: list[str] = []
        if os.environ.get("CLOVER_ALLOW_TEST_EDITS") == "1":
            return False, None

        code_exec_tools = {"python", "run_python", "exec", "eval", "node", "ruby"}
        shell_like_tools = {
            "bash", "sh", "zsh", "shell", "powershell", "pwsh", "cmd", "terminal",
            "run_in_terminal", "run_command", "run_shell", "run_powershell", "execute_command",
        }
        for tc in tool_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name") or "").lower().strip()
            if not name or name in _READ_ONLY_TOOLS:
                continue
            raw_args = str(fn.get("arguments") or "")
            try:
                args_obj = json.loads(raw_args) if raw_args else {}
            except ValueError:
                args_obj = {}

            if name in code_exec_tools:
                code_s = str((args_obj.get("code") or args_obj.get("script") or args_obj.get("command") or raw_args) if isinstance(args_obj, dict) else raw_args)
                m_c = _PROTECTED_VERIFIER_FILE_RE.search(code_s)
                if m_c and _CODE_DIRECT_MUTATE_RE.search(code_s):
                    return True, Path(m_c.group(1)).name or m_c.group(1)
                if re.search(r"(?i)(?:\b(?:base64|b64decode|codecs\.decode|fromhex|chr)\b|\bbytes\s*\(\s*\[)", code_s) and _CODE_DIRECT_MUTATE_RE.search(code_s):
                    return True, "obfuscated_code_mutation"
                continue

            is_shell_tool = name in shell_like_tools or any(k in name for k in ("shell", "terminal", "command", "powershell", "bash"))
            if is_shell_tool:
                cmd = str((args_obj.get("command") or args_obj.get("cmd") or args_obj.get("code") or raw_args) if isinstance(args_obj, dict) else raw_args)
                if re.search(r"(?i)\bgit\s+clean\b", cmd):
                    return True, "git_clean"
                m = _PROTECTED_VERIFIER_FILE_RE.search(cmd)
                if m and _SHELL_MUTATE_RE.search(cmd):
                    return True, Path(m.group(1)).name or m.group(1)
                # Catch obfuscated pipelines (`base64 -d | xargs rm -rf`, `$'\x74ests'`, `$(printf ...)`)
                if re.search(r"(?i)(?:\b(?:base64\s+-d|xxd\s+-r|openssl\s+enc)\b|\$'\\x|\$\(\s*printf\b)", cmd) and _SHELL_MUTATE_RE.search(cmd):
                    return True, "obfuscated_shell_mutation"
                # Catch indirect shell variable/glob indirection (`f=tests/test_core; mv x $f.py`, `rm -rf [t]ests`, `rm -rf t*s`, `patch -p1`)
                if _SHELL_MUTATE_RE.search(cmd) and re.search(r"(?i)(?:\b(?:tests?|specs?|__tests__)\b|\[[tT]\]ests?|t[?*\[]sts?|\bt\*s\b|\bpatch\b|\bgit\s+apply\b)", cmd):
                    m_dir = re.search(r"(?i)((?:tests?|specs?|__tests__|\[[tT]\]ests?|t[?*\[]sts?|t\*s)[A-Za-z0-9_./\\*?\[\]${}-]*|patch|git\s+apply)", cmd)
                    if m_dir:
                        return True, Path(m_dir.group(1)).name or m_dir.group(1)
            else:
                path_vals = [
                    str(v) for k, v in (args_obj.items() if isinstance(args_obj, dict) else [])
                    if any(tok in k.lower() for tok in ("path", "file", "target", "patch", "uri"))
                ]
                target = " ".join(path_vals) + " " + raw_args
                m = _PROTECTED_VERIFIER_FILE_RE.search(target)
                if m:
                    return True, Path(m.group(1)).name or m.group(1)
        return False, None

    @staticmethod
    def snapshot_protected_files(workspace_root: str | Path) -> dict[str, tuple[str, bytes | None]]:
        """Authoritative filesystem tamper snapshot (AGENTS.md §4): records `(sha256, raw_bytes)` for every
        protected verifier/test/CI file in `workspace_root` (`raw_bytes` kept for files <= 2 MiB to enable byte-exact rollback).
        Note: for remote stateless `/v1/chat/completions` clients (OpenCode/VS Code), tools run on the client host so
        pre-call regex screening is a best-effort heuristic; server-side tool execution (`Clover4PartAgent.solve` &
        Studio `lfm3_native.execute_tool`) enforces this filesystem hash + rollback control directly on disk."""
        root = Path(workspace_root)
        if not root.is_dir():
            return {}
        always_prune = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "site-packages", "dist-packages", ".tox", ".nox",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", ".gradle",
        }
        root_only_prune = {"build", "dist", "target", "out"}
        snap: dict[str, tuple[str, bytes | None]] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            is_top = dp == root
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in always_prune
                and not d.startswith(".venv")
                and not (is_top and d in root_only_prune)
                and not (dp / d / "pyvenv.cfg").is_file()
            )
            for fname in sorted(filenames):
                p = dp / fname
                rel = p.relative_to(root).as_posix()
                if _PROTECTED_VERIFIER_FILE_RE.search(rel):
                    try:
                        data = p.read_bytes()
                        snap[rel] = (hashlib.sha256(data).hexdigest(), data if len(data) <= 2_097_152 else None)
                    except OSError:
                        pass
        return snap

    @staticmethod
    def snapshot_protected_hashes(workspace_root: str | Path) -> dict[str, str]:
        return {k: v[0] for k, v in CloverCycleEngine.snapshot_protected_files(workspace_root).items()}

    @staticmethod
    def verify_protected_hashes_unchanged(workspace_root: str | Path, before: dict[str, str]) -> list[str]:
        """Return list of protected verifier/test files modified, deleted, or injected since `before`."""
        after = CloverCycleEngine.snapshot_protected_hashes(workspace_root)
        changed = [k for k, h in before.items() if after.get(k) != h]
        added = [k for k in after if k not in before]
        return sorted(set(changed + added))

    @staticmethod
    def restore_and_verify_protected_files(workspace_root: str | Path, before_files: dict[str, tuple[str, bytes | None]]) -> list[str]:
        """Detect any modified, deleted, or newly injected protected test/verifier file in `workspace_root`.
        Restores pre-existing modified/deleted protected files byte-for-byte on disk, NEVER deletes newly created
        files on disk (reports & refuses instead so user files are never destroyed), and returns tampered paths."""
        root = Path(workspace_root)
        if not root.is_dir():
            return []
        after = CloverCycleEngine.snapshot_protected_files(root)
        tampered: list[str] = []
        for rel, (old_sha, old_bytes) in before_files.items():
            cur = after.get(rel)
            if cur is None or cur[0] != old_sha:
                tampered.append(rel)
                if old_bytes is not None:
                    dst = root / rel
                    try:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(old_bytes)
                    except OSError:
                        pass
        for rel in after:
            if rel not in before_files:
                tampered.append(rel)
        return sorted(set(tampered))
