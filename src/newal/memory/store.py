"""SQLite-backed persistence for code chunks and cross-session notes.

The notes table is the part Codex and Claude Code do not have by default: facts
the assistant learned about *this* repo survive process exit and get replayed
into the system prompt on the next run.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    content     TEXT NOT NULL,
    mtime       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

CREATE TABLE IF NOT EXISTS files (
    path   TEXT PRIMARY KEY,
    mtime  REAL NOT NULL,
    size   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic);
"""


@dataclass
class Chunk:
    id: str
    path: str
    start_line: int
    end_line: int
    content: str

    def cite(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass
class Note:
    id: int
    topic: str
    content: str
    created_at: float


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # WAL so an indexing pass and a read from the agent loop do not block.
        self._conn.execute("PRAGMA journal_mode=WAL")
        with closing(self._conn.cursor()) as cursor:
            cursor.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- file bookkeeping -----------------------------------------------------

    def file_is_current(self, path: str, mtime: float, size: int) -> bool:
        row = self._conn.execute(
            "SELECT mtime, size FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row is not None and row["mtime"] == mtime and row["size"] == size

    def indexed_paths(self) -> set[str]:
        return {row["path"] for row in self._conn.execute("SELECT path FROM files")}

    def replace_file(self, path: str, mtime: float, size: int, chunks: list[Chunk]) -> None:
        """Atomically swap in a file's chunks."""
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, path, start_line, end_line, content, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                [(c.id, c.path, c.start_line, c.end_line, c.content, mtime) for c in chunks],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO files (path, mtime, size) VALUES (?, ?, ?)",
                (path, mtime, size),
            )

    def forget_file(self, path: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self._conn.execute("DELETE FROM files WHERE path = ?", (path,))

    # ---- chunk access ---------------------------------------------------------

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT id, path, start_line, end_line, content FROM chunks"
        ).fetchall()
        return [Chunk(**dict(row)) for row in rows]

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, path, start_line, end_line, content FROM chunks "
            f"WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {row["id"]: Chunk(**dict(row)) for row in rows}
        return [by_id[i] for i in ids if i in by_id]  # preserve ranking order

    def chunk_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # ---- durable notes --------------------------------------------------------

    def add_note(self, topic: str, content: str) -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO notes (topic, content, created_at) VALUES (?, ?, ?)",
                (topic, content, time.time()),
            )
        return int(cursor.lastrowid)

    def recent_notes(self, limit: int = 20) -> list[Note]:
        rows = self._conn.execute(
            "SELECT id, topic, content, created_at FROM notes "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [Note(**dict(row)) for row in rows]

    def search_notes(self, query: str, limit: int = 10) -> list[Note]:
        rows = self._conn.execute(
            "SELECT id, topic, content, created_at FROM notes "
            "WHERE topic LIKE ? OR content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [Note(**dict(row)) for row in rows]

    def clear_notes(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM notes")
