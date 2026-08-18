"""HTTP front end.

Deliberately small: the browser talks to the same `Agent` the terminal drives,
so this module is transport and nothing else. No business logic lives here.

Streaming uses server-sent events rather than websockets. The traffic is one
directional -- the agent narrates, the browser listens -- and SSE reconnects on
its own, needs no handshake, and survives a proxy that mangles upgrades.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import Config
from ..media import is_image_path, is_video_path
from ..training import FORMATS, export
from ..training.plan import TrainingPlan, render_script
from . import settings as settings_module
from .session import PendingAttachment, Session, build_session

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
#: Cap on a single upload. A minute of 1080p video is comfortably under this,
#: and it keeps a stray file from filling the disk.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(config: Config, *, autostart: bool = True) -> FastAPI:
    """Build the app around one already-configured session."""
    state: dict[str, Session] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Building the session starts model servers, so it must not run at
        # import time -- only when something actually serves the app.
        state["session"] = build_session(config, autostart=autostart)
        try:
            yield
        finally:
            session = state.pop("session", None)
            if session is not None:
                session.close()

    app = FastAPI(
        title="newal",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    def _session() -> Session:
        session = state.get("session")
        if session is None:  # pragma: no cover - only before startup
            raise HTTPException(503, "session is not ready")
        return session

    # ---- state ----------------------------------------------------------------

    @app.get("/api/state")
    def get_state() -> JSONResponse:
        session = _session()
        counts = session.index.store.training_counts() if session.index else {}
        return JSONResponse(
            {
                "version": __version__,
                "workspace": str(session.config.workspace_path()),
                "models": session.pool.describe(),
                # Empty in the normal case. Populated when a member's server was
                # not reachable at startup, so the page can say so instead of
                # letting the user find out by sending a message.
                "unavailable": session.pool.unavailable,
                "router": {
                    "strategy": session.config.router.strategy,
                    "mode": session.config.router.mode,
                    "classifier": session.pool.router.classifier is not None,
                },
                "shell_policy": session.config.tools.shell_policy,
                "usage": {
                    key: usage.total_tokens
                    for key, usage in session.agent.usage_by_model.items()
                },
                "captured": counts,
                "attachments": [
                    {"id": a.id, "summary": a.summary, "kind": a.kind}
                    for a in session.attachments.values()
                ],
                "busy": session.busy,
            }
        )

    @app.post("/api/reset")
    def reset() -> JSONResponse:
        session = _session()
        if session.busy:
            raise HTTPException(409, "a turn is still running")
        session.agent.reset()
        session.attachments.clear()
        return JSONResponse({"ok": True})

    # ---- attachments ----------------------------------------------------------

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> JSONResponse:
        session = _session()
        name = Path(file.filename or "upload").name

        if not (is_image_path(name) or is_video_path(name)):
            raise HTTPException(
                415, f"{name}: only images and video can be attached"
            )

        attachment_id = uuid.uuid4().hex[:12]
        target = session.upload_dir / f"{attachment_id}-{name}"

        written = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, "file is larger than 512 MB")
                handle.write(chunk)

        kind = "video" if is_video_path(name) else "image"
        session.attachments[attachment_id] = PendingAttachment(
            id=attachment_id, path=target, kind=kind, summary=f"{name} ({written:,} B)"
        )
        return JSONResponse({"id": attachment_id, "kind": kind, "name": name})

    @app.delete("/api/upload/{attachment_id}")
    def drop_upload(attachment_id: str) -> JSONResponse:
        session = _session()
        item = session.attachments.pop(attachment_id, None)
        if item is not None:
            item.path.unlink(missing_ok=True)
        return JSONResponse({"ok": item is not None})

    # ---- approvals ------------------------------------------------------------

    @app.post("/api/approve")
    def approve(request_id: str = Form(...), allow: bool = Form(...)) -> JSONResponse:
        answered = _session().answer_approval(request_id, allow)
        if not answered:
            raise HTTPException(404, "no pending approval with that id")
        return JSONResponse({"ok": True})

    # ---- the turn -------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(message: str = Form(...)) -> StreamingResponse:
        session = _session()
        if session.busy:
            raise HTTPException(409, "a turn is still running")
        if not message.strip():
            raise HTTPException(400, "message is empty")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        session.bind(loop, queue)
        session.busy = True

        def run_turn() -> None:
            try:
                attachments = session.prepare_pending()
                for attachment in attachments:
                    session.emit("attachment", {"summary": attachment.summary})

                result = session.agent.run(message, attachments)

                if session.transcript is not None:
                    verification = None
                    if result.verification and not result.verification.skipped:
                        verification = (
                            result.verification.command,
                            result.verification.passed,
                        )
                    session.transcript.record(
                        prompt=message,
                        response=result.text,
                        routes=result.routes,
                        attachments=[a.source for a in attachments],
                        files_written=result.files_written,
                        verification=verification,
                        usage_by_model={
                            k: u.total_tokens for k, u in result.usage_by_model.items()
                        },
                        steps=result.steps,
                        escalations=result.escalations,
                    )

                session.emit(
                    "done",
                    {
                        "text": result.text,
                        "steps": result.steps,
                        "escalations": result.escalations,
                        "routes": result.routes,
                        "files_written": result.files_written,
                        "tokens": result.usage.total_tokens,
                        "verification": (
                            None
                            if result.verification is None
                            or result.verification.skipped
                            else {
                                "command": result.verification.command,
                                "passed": result.verification.passed,
                            }
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - report, never hang the stream
                log.exception("turn failed")
                session.emit("error", {"message": f"{type(exc).__name__}: {exc}"})
            finally:
                session.busy = False
                session.emit("end", {})

        async def stream():
            worker = asyncio.get_running_loop().run_in_executor(None, run_turn)
            try:
                while True:
                    event = await queue.get()
                    if event["kind"] == "end":
                        break
                    yield _sse(event)
            finally:
                await worker
            yield _sse({"kind": "end", "payload": {}})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- settings (normal mode) -----------------------------------------------

    @app.get("/api/settings")
    def read_settings() -> JSONResponse:
        return JSONResponse({"settings": settings_module.describe(_session().config)})

    @app.put("/api/settings")
    async def write_settings(request: Request) -> JSONResponse:
        session = _session()
        if session.busy:
            raise HTTPException(409, "a turn is still running")

        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "expected a JSON object of path -> value")

        result = settings_module.apply(session.config, payload)
        if result.errors:
            raise HTTPException(422, "; ".join(result.errors))

        # The router caches thresholds taken at construction, so refresh the
        # ones it holds rather than leaving the panel and the pool disagreeing.
        router = session.pool.router
        router.strategy = session.config.router.strategy
        router.escalate_threshold = session.config.router.escalate_threshold
        router.uncertainty_band = session.config.router.uncertainty_band
        router.thinking_mode = session.config.router.thinking.mode
        router.thinking_threshold = session.config.router.thinking.threshold

        return JSONResponse(
            {
                "changed": result.changed,
                "saved_to": result.saved_to,
                "settings": settings_module.describe(session.config),
            }
        )

    # ---- training mode --------------------------------------------------------

    @app.get("/training")
    def training_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "training.html")

    @app.get("/api/training/summary")
    def training_summary() -> JSONResponse:
        session = _session()
        if session.index is None:
            raise HTTPException(409, "memory is disabled, so nothing is captured")

        counts = session.index.store.training_counts()
        pool = list(session.config.enabled_models(task="generate").items())
        cheapest = min(pool, key=lambda item: item[1].tier)[1].id if pool else ""
        return JSONResponse(
            {
                "counts": counts,
                "capture_enabled": session.config.training.enabled,
                "db_path": str(session.index.store.db_path),
                "candidate_models": [spec.id for _, spec in sorted(pool, key=lambda i: i[1].tier)],
                "suggested_base_model": cheapest,
                "defaults": TrainingPlan().to_dict(),
                "formats": list(FORMATS),
            }
        )

    @app.get("/api/training/dataset/{fmt}")
    def download_dataset(fmt: str, include_unverified: bool = False) -> FileResponse:
        session = _session()
        if session.index is None:
            raise HTTPException(409, "memory is disabled, so nothing is captured")
        if fmt not in FORMATS:
            raise HTTPException(400, f"format must be one of {', '.join(FORMATS)}")

        target = session.upload_dir.parent / "datasets" / f"{fmt}.jsonl"
        written = export(
            session.index.store, fmt, target, include_unverified=include_unverified
        )
        if written == 0:
            raise HTTPException(404, f"no {fmt} samples captured yet")
        return FileResponse(target, filename=f"newal-{fmt}.jsonl", media_type="application/jsonl")

    @app.post("/api/training/plan")
    async def build_plan(request: Request) -> JSONResponse:
        session = _session()
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "expected a JSON object")

        try:
            plan = TrainingPlan.from_dict(payload)
        except TypeError as exc:
            raise HTTPException(400, str(exc)) from exc

        errors = plan.validate()
        if errors:
            raise HTTPException(422, "; ".join(errors))

        counts = session.index.store.training_counts() if session.index else {}
        available = counts.get(
            "turns_clean_first_try" if plan.task == "sft" else "repair_pairs", 0
        )
        dataset_name = f"newal-{plan.task}.jsonl"
        return JSONResponse(
            {
                "plan": plan.to_dict(),
                "warnings": plan.warnings(available),
                "samples": available,
                "dataset": dataset_name,
                "script_name": f"train_{plan.task}.py",
                "script": render_script(plan, dataset_name),
            }
        )

    @app.post("/api/training/clear")
    def clear_training() -> JSONResponse:
        session = _session()
        if session.index is None:
            raise HTTPException(409, "memory is disabled")
        if session.busy:
            raise HTTPException(409, "a turn is still running")
        session.index.store.clear_training_data()
        return JSONResponse({"ok": True, "counts": session.index.store.training_counts()})

    # ---- static ---------------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    mimetypes.add_type("text/javascript", ".js")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
