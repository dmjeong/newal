"""Session transcripts.

One JSONL file per session, one object per turn. Written for a person to read
back later -- what was asked, which models answered, what changed on disk, and
whether verification agreed.

Attachments are redacted before writing. A single video turn carries dozens of
base64 frames; storing them would balloon the file by megabytes per turn and
put user screenshots on disk under a name nobody expects.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REDACTED_IMAGE = "<image redacted>"


def redact_content(content: Any) -> Any:
    """Strip base64 payloads from message content, keeping the structure.

    Shared with the training-data capture: anything persisted from a message
    history has to go through here first.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    parts: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            parts.append(part)
            continue
        if part.get("type") == "image_url":
            parts.append({"type": "image_url", "image_url": {"url": REDACTED_IMAGE}})
        else:
            parts.append(part)
    return parts


def redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy a message list with every attachment payload replaced."""
    redacted: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        if "content" in copied:
            copied["content"] = redact_content(copied["content"])
        redacted.append(copied)
    return redacted


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class TranscriptWriter:
    """Appends one JSON object per turn to a per-session file."""

    directory: Path
    session_id: str

    @classmethod
    def create(cls, directory: str | Path, session_id: str | None = None) -> TranscriptWriter:
        return cls(Path(directory).expanduser(), session_id or new_session_id())

    @property
    def path(self) -> Path:
        return self.directory / f"session-{self.session_id}.jsonl"

    def record(
        self,
        *,
        prompt: str,
        response: str,
        routes: list[str] | None = None,
        attachments: list[str] | None = None,
        files_written: list[str] | None = None,
        verification: tuple[str | None, bool] | None = None,
        usage_by_model: dict[str, int] | None = None,
        steps: int = 0,
        escalations: int = 0,
    ) -> Path | None:
        """Append one turn. Never raises -- a logging failure must not end a session."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response,
            "steps": steps,
            "escalations": escalations,
            "routes": routes or [],
            "attachments": attachments or [],
            "files_written": files_written or [],
            "usage_by_model": usage_by_model or {},
        }
        if verification is not None:
            command, passed = verification
            entry["verification"] = {"command": command, "passed": passed}

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("could not write transcript to %s: %s", self.path, exc)
            return None
        return self.path

    def read_all(self) -> list[dict[str, Any]]:
        """Read this session's turns back. Malformed lines are skipped."""
        if not self.path.is_file():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed transcript line in %s", self.path)
        return entries
