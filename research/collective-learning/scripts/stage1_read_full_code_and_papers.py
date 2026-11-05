#!/usr/bin/env python3
"""Read full code of Voyager SkillManager/Critic/Voyager, Reflexion programming & alfworld, MAML, Swarm Learning,
and extract key algorithmic sections from the downloaded paper texts on k3."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/opt/clover-k3/meaning-core/collective-learning")
REPOS = ROOT / "repos"
PAPERS = ROOT / "papers"
OUT = ROOT / "stage1_code_and_papers_dump.txt"


def main():
    lines = []

    def add_section(title, text):
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        lines.append(text)
        lines.append("")

    # 1. Voyager
    for rel in [
        "voyager/voyager/agents/skill.py",
        "voyager/voyager/agents/critic.py",
        "voyager/voyager/voyager.py",
    ]:
        p = REPOS / rel
        if p.exists():
            add_section(f"REPO FILE: {rel}", p.read_text(errors="replace"))

    # 2. Reflexion
    for rel in [
        "reflexion/programming_runs/reflexion.py",
        "reflexion/alfworld_runs/generate_reflections.py",
        "reflexion/hotpotqa_runs/agents.py",
    ]:
        p = REPOS / rel
        if p.exists():
            add_section(f"REPO FILE: {rel}", p.read_text(errors="replace")[:8000])

    # 3. MAML
    for rel in ["maml/maml.py"]:
        p = REPOS / rel
        if p.exists():
            add_section(f"REPO FILE: {rel}", p.read_text(errors="replace"))

    # 4. Swarm Learning
    for rel in [
        "swarm_learning/lib/src/python-client/swarmlearning/client/swarm.py",
        "swarm_learning/examples/fraud-detection-skl/model/fraud-detection-SGD.py",
    ]:
        p = REPOS / rel
        if p.exists():
            add_section(f"REPO FILE: {rel}", p.read_text(errors="replace")[:8000])

    # 5. Paper algorithmic sections
    for name in [
        "p1_mcmahan17_fedavg.txt",
        "p4_wang23_voyager.txt",
        "p5_shinn23_reflexion.txt",
        "p6_finn17_maml.txt",
        "p10_borkar25_privacy_ripple.txt",
    ]:
        p = PAPERS / name
        if p.exists():
            add_section(f"PAPER TEXT: {name}", p.read_text(errors="replace")[:15000])

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes, {len(lines)} chunks)")


if __name__ == "__main__":
    main()
