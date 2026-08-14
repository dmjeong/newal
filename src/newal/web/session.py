"""One browser session bound to one agent.

The agent loop is synchronous and blocking, so a turn runs on a worker thread
while the request handler streams whatever the loop emits. Events cross that
boundary through an asyncio queue, and shell approvals cross it in the other
direction through a threading event.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent import Agent, Toolbox, TranscriptWriter
from ..config import Config
from ..media import Attachment, AttachmentError, prepare_attachments
from ..memory import RepoIndex, build_index
from ..models import ModelPool

log = logging.getLogger(__name__)

#: How long a queued shell approval waits before it is treated as declined.
#: Long enough to walk away from the screen, short enough that a closed tab
#: does not pin a worker thread forever.
APPROVAL_TIMEOUT_S = 300.0


@dataclass
class PendingAttachment:
    """An uploaded file waiting to be sent with the next message."""

    id: str
    path: Path
    kind: str
    summary: str = ""


@dataclass
class Session:
    """Everything one browser tab needs, and nothing shared between tabs."""

    config: Config
    agent: Agent
    pool: ModelPool
    index: RepoIndex | None
    transcript: TranscriptWriter | None
    upload_dir: Path
    attachments: dict[str, PendingAttachment] = field(default_factory=dict)

    #: Set while a turn is running, so a second request can be refused rather
    #: than corrupting the agent's message history.
    busy: bool = False
    #: Overridable so a test never has to wait out the real timeout.
    approval_timeout_s: float = APPROVAL_TIMEOUT_S
    _queue: asyncio.Queue | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _approvals: dict[str, threading.Event] = field(default_factory=dict)
    _answers: dict[str, bool] = field(default_factory=dict)

    # ---- event plumbing -------------------------------------------------------

    def bind(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        self._loop = loop
        self._queue = queue

    def emit(self, kind: str, payload: Any) -> None:
        """Push an event from the worker thread onto the streaming queue."""
        if self._loop is None or self._queue is None:
            return
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, {"kind": kind, "payload": payload}
            )
        except RuntimeError:  # loop already closed, client went away
            log.debug("dropping %s event, event loop is gone", kind)

    # ---- shell approval -------------------------------------------------------

    def request_approval(self, tool_name: str, detail: str) -> bool:
        """Block the worker thread until the browser answers, or time out.

        Auto-allowing here would be a real downgrade from the terminal, where
        the same policy prompts. A tab that closes mid-prompt declines.
        """
        request_id = uuid.uuid4().hex[:12]
        gate = threading.Event()
        self._approvals[request_id] = gate

        self.emit(
            "approval",
            {"id": request_id, "tool": tool_name, "detail": detail[:2000]},
        )
        answered = gate.wait(timeout=self.approval_timeout_s)
        self._approvals.pop(request_id, None)
        allowed = self._answers.pop(request_id, False)

        if not answered:
            log.info("approval %s timed out; declining", request_id)
        return bool(answered and allowed)

    def answer_approval(self, request_id: str, allow: bool) -> bool:
        gate = self._approvals.get(request_id)
        if gate is None:
            return False
        self._answers[request_id] = allow
        gate.set()
        return True

    # ---- attachments ----------------------------------------------------------

    def prepare_pending(self) -> list[Attachment]:
        """Turn uploaded files into model-ready parts, then clear the queue."""
        paths = [item.path for item in self.attachments.values()]
        self.attachments.clear()
        if not paths:
            return []
        try:
            return prepare_attachments(paths, self.config.media)
        except (AttachmentError, FileNotFoundError, RuntimeError) as exc:
            self.emit("error", {"message": f"attachment failed: {exc}"})
            return []

    def close(self) -> None:
        self.pool.close()


def build_session(config: Config, *, autostart: bool = True) -> Session:
    """Start the pool, open the index, and assemble an agent for the browser."""
    root = config.workspace_path()
    config.runtime.autostart = autostart

    pool = ModelPool(config)

    index: RepoIndex | None = None
    if config.memory.enabled:
        index = build_index(root, config.memory)
        index.attach_models(embedder=pool.embedder, reranker=pool.reranker)
        index.refresh()

    learned = (
        index.store.routing_exemplars(config.router.max_learned_exemplars)
        if index
        else []
    )
    pool.build_classifier(learned)

    upload_dir = root / ".newal" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    transcript: TranscriptWriter | None = None
    if config.ui.save_transcripts:
        directory = Path(config.ui.transcript_dir)
        transcript = TranscriptWriter.create(
            directory if directory.is_absolute() else root / directory
        )

    session = Session(
        config=config,
        agent=None,  # type: ignore[arg-type]
        pool=pool,
        index=index,
        transcript=transcript,
        upload_dir=upload_dir,
    )

    toolbox = Toolbox(config.tools, index=index, approve=session.request_approval)
    session.agent = Agent(
        config,
        pool,
        toolbox,
        index=index,
        on_event=lambda kind, text: session.emit(kind, {"text": text}),
    )
    return session
