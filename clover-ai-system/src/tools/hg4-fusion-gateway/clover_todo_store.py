"""Thread-Safe SQLite + Markdown Todo Store (`clover_todo_store.py`) for Clover's 4-Part Harness.

Enforces step-by-step decomposition (`pending -> in_progress -> done | blocked`) with explicit
prerequisite checking (`depends_on`):
  - Rejects transitioning a todo to `in_progress` or `done` while any `depends_on` prerequisite
    is not yet `done`.
  - Rejects self-dependencies and circular dependency cycles (`A -> B -> A`).
  - Optionally mirrors the live Todo table into `<CLOVER_CONTEXT_DIR>/<task_id>.todos.md` so both
    humans and models can inspect todos as plain Markdown alongside `<task_id>.md`.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

VALID_STATUSES = frozenset({"pending", "in_progress", "done", "blocked"})
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _clean_id(raw: str, fallback: str = "default") -> str:
    s = _SAFE_ID_RE.sub("_", (raw or "").strip()).strip("._-")
    return s[:96] or fallback


class CloverTodoStore:
    """SQLite-backed (or in-memory) Todo + Prerequisite store scoped by `task_id`."""

    def __init__(self, db_path: str | Path | None = None, context_dir: str | Path | None = None) -> None:
        raw_db = str(db_path if db_path is not None else os.environ.get("CLOVER_TODOS_DB", "")).strip()
        raw_ctx = str(context_dir if context_dir is not None else os.environ.get("CLOVER_CONTEXT_DIR", "")).strip()
        self.db_path: str = raw_db if raw_db else ":memory:"
        self.context_dir: Path | None = Path(raw_ctx) if raw_ctx else None
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if self.context_dir is not None:
            self.context_dir.mkdir(parents=True, exist_ok=True)

        self._mu = threading.RLock()
        self._mem_conn: sqlite3.Connection | None = (
            sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        )
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clover_todos (
                    task_id TEXT NOT NULL,
                    todo_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (task_id, todo_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clover_todo_deps (
                    task_id TEXT NOT NULL,
                    todo_id TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    PRIMARY KEY (task_id, todo_id, depends_on)
                )
                """
            )

    @contextlib.contextmanager
    def _conn(self):
        if self._mem_conn is not None:
            with self._mem_conn:
                yield self._mem_conn
        else:
            with contextlib.closing(sqlite3.connect(self.db_path, timeout=15.0)) as c:
                with c:
                    yield c

    def close(self) -> None:
        with self._mu:
            if self._mem_conn is not None:
                self._mem_conn.close()
                self._mem_conn = None

    @staticmethod
    def _would_create_cycle(adj: dict[str, set[str]], src: str, dst: str) -> bool:
        """Return True if adding `src -> dst` (`src` depends on `dst`) creates a cycle."""
        if src == dst:
            return True
        stack = [dst]
        seen: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, ()))
        return False

    def upsert(
        self,
        task_id: str,
        todo_id: str,
        title: str | None = None,
        status: str | None = None,
        description: str | None = None,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        tid = _clean_id(task_id, "default_task")
        item_id = _clean_id(todo_id, "")
        if not item_id:
            raise ValueError("todo_id is required")
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'; must be one of {sorted(VALID_STATUSES)}")

        now = time.time()
        with self._mu:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT title, description, status, created_at FROM clover_todos WHERE task_id=? AND todo_id=?",
                    (tid, item_id),
                ).fetchone()

                cur_title = title.strip() if (title and title.strip()) else (row[0] if row else item_id)
                cur_desc = description if description is not None else (row[1] if row else "")
                target_status = status if status is not None else (row[2] if row else "pending")
                created_at = float(row[3]) if row else now

                if target_status == "blocked" and not cur_desc.strip():
                    raise ValueError("Blocked todos must explain why in `description`")

                # Validate `depends_on` references & cycle-freedom before mutating
                if depends_on is not None:
                    clean_deps = [_clean_id(d, "") for d in depends_on if _clean_id(d, "")]
                    existing_ids = {
                        r[0]
                        for r in conn.execute("SELECT todo_id FROM clover_todos WHERE task_id=?", (tid,)).fetchall()
                    } | {item_id}
                    missing = [d for d in clean_deps if d not in existing_ids]
                    if missing:
                        raise ValueError(f"Unknown prerequisite todo_id(s) in task '{tid}': {missing}")

                    adj: dict[str, set[str]] = {}
                    for u, v in conn.execute(
                        "SELECT todo_id, depends_on FROM clover_todo_deps WHERE task_id=? AND todo_id!=?",
                        (tid, item_id),
                    ).fetchall():
                        adj.setdefault(u, set()).add(v)
                    for dep in clean_deps:
                        if self._would_create_cycle(adj, item_id, dep):
                            raise ValueError(f"Circular todo dependency rejected: '{item_id}' -> '{dep}'")
                        adj.setdefault(item_id, set()).add(dep)
                else:
                    clean_deps = [
                        r[0]
                        for r in conn.execute(
                            "SELECT depends_on FROM clover_todo_deps WHERE task_id=? AND todo_id=? ORDER BY depends_on",
                            (tid, item_id),
                        ).fetchall()
                    ]

                # Prerequisite gate: cannot mark `in_progress` or `done` until all `depends_on` are `done`!
                if target_status in ("in_progress", "done") and clean_deps:
                    placeholders = ",".join("?" for _ in clean_deps)
                    unfinished = [
                        {"todo_id": r[0], "status": r[1]}
                        for r in conn.execute(
                            f"SELECT todo_id, status FROM clover_todos WHERE task_id=? AND todo_id IN ({placeholders}) AND status!='done'",
                            (tid, *clean_deps),
                        ).fetchall()
                    ]
                    if unfinished:
                        raise ValueError(
                            f"Cannot set '{item_id}' to '{target_status}' before prerequisites finish: {unfinished}"
                        )

                conn.execute(
                    """
                    INSERT INTO clover_todos (task_id, todo_id, title, description, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id, todo_id) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        status=excluded.status,
                        updated_at=excluded.updated_at
                    """,
                    (tid, item_id, cur_title, cur_desc, target_status, created_at, now),
                )
                if depends_on is not None:
                    conn.execute("DELETE FROM clover_todo_deps WHERE task_id=? AND todo_id=?", (tid, item_id))
                    for dep in clean_deps:
                        conn.execute(
                            "INSERT OR IGNORE INTO clover_todo_deps (task_id, todo_id, depends_on) VALUES (?, ?, ?)",
                            (tid, item_id, dep),
                        )

            listing = self.list_todos(tid)
            self._mirror_markdown(tid, listing)
        return {
            "task_id": tid,
            "updated": item_id,
            "status": target_status,
            "depends_on": clean_deps,
            "ready_todos": listing["ready_todo_ids"],
            "todos_file": listing.get("todos_file"),
        }

    def list_todos(self, task_id: str) -> dict[str, Any]:
        tid = _clean_id(task_id, "default_task")
        with self._mu, self._conn() as conn:
            rows = conn.execute(
                "SELECT todo_id, title, description, status, updated_at FROM clover_todos WHERE task_id=? ORDER BY created_at ASC, todo_id ASC",
                (tid,),
            ).fetchall()
            dep_rows = conn.execute(
                "SELECT todo_id, depends_on FROM clover_todo_deps WHERE task_id=? ORDER BY depends_on ASC",
                (tid,),
            ).fetchall()

        deps_map: dict[str, list[str]] = {}
        for u, v in dep_rows:
            deps_map.setdefault(u, []).append(v)

        status_by_id = {r[0]: r[3] for r in rows}
        items: list[dict[str, Any]] = []
        ready_ids: list[str] = []
        for todo_id, title, desc, st, upd in rows:
            dlist = deps_map.get(todo_id, [])
            is_ready = st == "pending" and all(status_by_id.get(d) == "done" for d in dlist)
            if is_ready:
                ready_ids.append(todo_id)
            items.append({
                "todo_id": todo_id,
                "title": title,
                "description": desc,
                "status": st,
                "depends_on": dlist,
                "ready": is_ready,
                "updated_at": round(float(upd), 3),
            })

        counts = {s: sum(1 for it in items if it["status"] == s) for s in sorted(VALID_STATUSES)}
        md_file = str(self.context_dir / f"{tid}.todos.md") if self.context_dir is not None else None
        return {
            "task_id": tid,
            "total": len(items),
            "counts": counts,
            "all_done": bool(items) and counts["done"] == len(items),
            "ready_todo_ids": ready_ids,
            "todos": items,
            "todos_file": md_file,
        }

    def _mirror_markdown(self, tid: str, listing: dict[str, Any]) -> None:
        if self.context_dir is None:
            return
        import tempfile

        badges = {"done": "[x]", "in_progress": "[~]", "blocked": "[!]", "pending": "[ ]"}
        lines = [f"# Clover Todo List (`{tid}`)", ""]
        for it in listing["todos"]:
            dep_note = f" *(depends on: {', '.join(it['depends_on'])})*" if it["depends_on"] else ""
            desc_note = f" — {it['description']}" if it["description"] else ""
            lines.append(f"- {badges.get(it['status'], '[ ]')} **`{it['todo_id']}`** ({it['status']}): {it['title']}{dep_note}{desc_note}")
        out_p = self.context_dir / f"{tid}.todos.md"
        with tempfile.NamedTemporaryFile("w", dir=str(self.context_dir), prefix=f".{tid}.", suffix=".tmp", delete=False, encoding="utf-8") as tf:
            tf.write("\n".join(lines) + "\n")
            tmp_p = Path(tf.name)
        os.replace(tmp_p, out_p)
