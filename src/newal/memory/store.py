"""SQLite-backed persistence for code chunks and cross-session notes.

The notes table is the part Codex and Claude Code do not have by default: facts
the assistant learned about *this* repo survive process exit and get replayed
into the system prompt on the next run.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Bumped whenever the chunk/file tables change shape. The index is a
#: rebuildable cache, so a mismatch drops and re-derives it rather than
#: attempting a migration. Notes are user data and always survive.
SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    content     TEXT NOT NULL,
    mtime       REAL NOT NULL,
    embedding   BLOB
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

-- Observed routing outcomes, used as training exemplars. Like notes, this is
-- learned data rather than a derived cache, so it survives a schema rebuild.
CREATE TABLE IF NOT EXISTS route_outcomes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt      TEXT NOT NULL,
    label       TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    escalated   INTEGER NOT NULL,
    verify_ok   INTEGER,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_route_label ON route_outcomes(label);
"""


@dataclass
class Chunk:
    id: str
    path: str
    start_line: int
    end_line: int
    content: str
    embedding: list[float] | None = None

    def cite(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def pack_embedding(vector: list[float] | None) -> bytes | None:
    """Serialise an embedding as float32 bytes."""
    if not vector:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


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
        self._apply_schema_version()

    def _apply_schema_version(self) -> None:
        """Drop the derived index if it was built by an older schema."""
        found = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if found == SCHEMA_VERSION:
            return
        if found != 0:
            log.info("memory schema %s -> %s; rebuilding index", found, SCHEMA_VERSION)
            with self._conn:
                # Notes are user data and deliberately untouched.
                self._conn.execute("DROP TABLE IF EXISTS chunks")
                self._conn.execute("DROP TABLE IF EXISTS files")
            with closing(self._conn.cursor()) as cursor:
                cursor.executescript(SCHEMA)
        with self._conn:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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
                "(id, path, start_line, end_line, content, mtime, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.path,
                        c.start_line,
                        c.end_line,
                        c.content,
                        mtime,
                        pack_embedding(c.embedding),
                    )
                    for c in chunks
                ],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO files (path, mtime, size) VALUES (?, ?, ?)",
                (path, mtime, size),
            )

    def set_embeddings(self, vectors: dict[str, list[float]]) -> None:
        """Attach embeddings to chunks that were indexed without them."""
        if not vectors:
            return
        with self._conn:
            self._conn.executemany(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                [(pack_embedding(vec), chunk_id) for chunk_id, vec in vectors.items()],
            )

    def chunks_missing_embeddings(self, limit: int | None = None) -> list[Chunk]:
        sql = (
            "SELECT id, path, start_line, end_line, content FROM chunks "
            "WHERE embedding IS NULL"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Chunk(**dict(row)) for row in self._conn.execute(sql).fetchall()]

    def embedded_chunks(self) -> list[Chunk]:
        """Every chunk that has an embedding, for dense search."""
        rows = self._conn.execute(
            "SELECT id, path, start_line, end_line, content, embedding FROM chunks "
            "WHERE embedding IS NOT NULL"
        ).fetchall()
        return [
            Chunk(
                id=row["id"],
                path=row["path"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                content=row["content"],
                embedding=unpack_embedding(row["embedding"]),
            )
            for row in rows
        ]

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

    def chunks_without_embeddings_count(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL"
            ).fetchone()[0]
        )

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

    # ---- routing outcomes -----------------------------------------------------

    def record_route_outcome(
        self,
        prompt: str,
        label: str,
        *,
        model_key: str,
        escalated: bool,
        verify_ok: bool | None,
    ) -> int:
        """Store one observed routing outcome as a future training exemplar."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO route_outcomes "
                "(prompt, label, model_key, escalated, verify_ok, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    prompt.strip()[:2000],
                    label,
                    model_key,
                    int(escalated),
                    None if verify_ok is None else int(verify_ok),
                    time.time(),
                ),
            )
        return int(cursor.lastrowid)

    def routing_exemplars(self, limit: int = 200) -> list[tuple[str, str]]:
        """Recent (prompt, label) pairs, newest first.

        Recency-ordered rather than balanced: a project's routing needs drift as
        the codebase does, and the newest evidence describes it best.
        """
        rows = self._conn.execute(
            "SELECT prompt, label FROM route_outcomes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(row["prompt"], row["label"]) for row in rows]

    def routing_outcome_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT label, COUNT(*) AS n FROM route_outcomes GROUP BY label"
        ).fetchall()
        return {row["label"]: int(row["n"]) for row in rows}

    def clear_route_outcomes(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM route_outcomes")
