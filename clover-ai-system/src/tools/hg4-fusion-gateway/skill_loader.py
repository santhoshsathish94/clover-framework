"""
Anthropic `agentskills.io` (`SKILL.md`) Compatible Loader for Clover & OpenCode
(`clover-ai-system/src/tools/hg4-fusion-gateway/skill_loader.py`).

Supports the open `SKILL.md` directory standard (`https://github.com/anthropics/skills`
and OpenCode's `.opencode/skills/<name>/SKILL.md`):
  - Parses YAML frontmatter (`name`, `description`) with strict validation,
  - Renders a **deterministic alphabetically sorted** compact skill index (`<= 15` skills)
    so system-prompt prefixes remain 100% cache-stable across turns,
  - Loads full procedural `SKILL.md` bodies on demand without mutating earlier prefix turns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)
_KEBAB_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    body: str
    path: Path


def parse_skill_md(skill_md_path: Path) -> SkillEntry:
    raw = skill_md_path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError(f"{skill_md_path}: missing YAML frontmatter (`--- ... ---`)")
    fm_text, body = m.group(1), m.group(2).strip()
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip("\"'")

    name = meta.get("name", "")
    desc = meta.get("description", "")
    if not name or not _KEBAB_NAME_RE.match(name):
        raise ValueError(f"{skill_md_path}: invalid or missing kebab-case `name`: {name!r}")
    if not desc:
        raise ValueError(f"{skill_md_path}: missing `description` in frontmatter")
    return SkillEntry(name=name, description=desc, body=body, path=skill_md_path)


def discover_skills(skills_root: Path, max_skills: int = 15) -> list[SkillEntry]:
    if not skills_root.is_dir():
        return []
    entries: list[SkillEntry] = []
    for p in sorted(skills_root.glob("*/SKILL.md")):
        entries.append(parse_skill_md(p))
    entries.sort(key=lambda e: e.name)
    return entries[:max_skills]


def format_compact_skill_index(entries: list[SkillEntry]) -> str:
    if not entries:
        return ""
    lines = ["<available_skills>"]
    for e in sorted(entries, key=lambda x: x.name):
        lines.append(f"- {e.name}: {e.description}")
    lines.append("</available_skills>")
    return "\n".join(lines)
