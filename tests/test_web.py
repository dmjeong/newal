"""The HTTP front end.

The web layer is transport only, so these tests check the transport: that a
turn streams, that uploads are gated, that a shell approval crosses the thread
boundary in both directions, and that a second turn cannot start on top of a
running one.
"""

from __future__ import annotations

import json
import threading

import pytest

from newal.backends.base import Completion, ToolCall, Usage
from newal.config import load_config
from newal.models import Router

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from newal.web.app import create_app  # noqa: E402


class _Backend:
    def __init__(self, script):
        self.script, self.calls = script, 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return self.script(self.calls)

    def stream(self, *args, **kwargs):
        yield ""

    def close(self):
        return None


class _Pool:
    def __init__(self, config, backend):
        self.config = config
        generators = config.enabled_models(task="generate")
        self.router = Router(tiers=[(k, v.tier) for k, v in generators.items()])
        self._backend = backend

    def for_route(self, route):
        return self._backend

    def build_classifier(self, learned=None):
        return None

    def describe(self):
        return ["heavy: fake (generate) [tier 3]"]

    @property
    def embedder(self):
        return None

    @property
    def reranker(self):
        return None

    def close(self):
        return None


def _client(tmp_path, monkeypatch, script, *, shell_policy="deny"):
    config = load_config(
        use_env=False,
        overrides={
            "tools": {"workspace_root": str(tmp_path), "shell_policy": shell_policy},
            "agent": {"plan_first": False, "verify": False},
            "memory": {"db_path": str(tmp_path / ".newal" / "m.db")},
            "ui": {"save_transcripts": False},
        },
    )
    monkeypatch.setattr(
        "newal.web.session.ModelPool", lambda cfg, **kw: _Pool(cfg, _Backend(script))
    )
    return TestClient(create_app(config, autostart=False))


def _events(response) -> list[dict]:
    out = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def _reply(text="완료했습니다."):
    return lambda n: Completion(text=text, usage=Usage(10, 5), model_key="heavy")


# ---- state -------------------------------------------------------------------


def test_state_reports_the_pool_and_policy(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        state = client.get("/api/state").json()

    assert state["models"] == ["heavy: fake (generate) [tier 3]"]
    assert state["shell_policy"] == "deny"
    assert state["router"]["mode"]
    assert state["busy"] is False


def test_index_page_is_served(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        page = client.get("/")
    assert page.status_code == 200
    assert "newal" in page.text
    # The UI must not need a bundler; it loads a plain script.
    assert "/static/app.js" in page.text


# ---- a turn ------------------------------------------------------------------


def test_chat_streams_events_and_finishes(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply("고쳤습니다.")) as client:
        response = client.post("/api/chat", data={"message": "add() 고쳐줘"})

    assert response.status_code == 200
    kinds = [e["kind"] for e in _events(response)]
    assert "done" in kinds
    assert kinds[-1] == "end"

    done = next(e for e in _events(response) if e["kind"] == "done")
    assert done["payload"]["text"] == "고쳤습니다."
    assert done["payload"]["tokens"] == 15


def test_empty_message_is_rejected(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        assert client.post("/api/chat", data={"message": "   "}).status_code == 400


def test_backend_failure_is_reported_not_hung(tmp_path, monkeypatch):
    def explode(n):
        raise RuntimeError("model is down")

    with _client(tmp_path, monkeypatch, explode) as client:
        response = client.post("/api/chat", data={"message": "안녕"})

    events = _events(response)
    assert any(e["kind"] == "error" for e in events)
    assert events[-1]["kind"] == "end", "the stream must still terminate"


def test_reset_clears_the_conversation(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        client.post("/api/chat", data={"message": "첫 질문"})
        assert client.post("/api/reset").json() == {"ok": True}


# ---- uploads -----------------------------------------------------------------


def test_image_upload_is_accepted(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post(
            "/api/upload", files={"file": ("shot.png", b"\x89PNG fake", "image/png")}
        )
        assert response.status_code == 200
        assert response.json()["kind"] == "image"

        listed = client.get("/api/state").json()["attachments"]
        assert len(listed) == 1


def test_video_upload_is_recognised(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post(
            "/api/upload", files={"file": ("clip.mp4", b"fake", "video/mp4")}
        )
        assert response.json()["kind"] == "video"


def test_other_file_types_are_refused(tmp_path, monkeypatch):
    """The agent reads the repo through its tools; uploads are media only."""
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post(
            "/api/upload", files={"file": ("secrets.env", b"TOKEN=1", "text/plain")}
        )
    assert response.status_code == 415


def test_dropping_an_upload_removes_the_file(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        attachment_id = client.post(
            "/api/upload", files={"file": ("a.png", b"x", "image/png")}
        ).json()["id"]

        stored = list((tmp_path / ".newal" / "uploads").iterdir())
        assert len(stored) == 1

        assert client.delete(f"/api/upload/{attachment_id}").json() == {"ok": True}
        assert list((tmp_path / ".newal" / "uploads").iterdir()) == []


# ---- shell approval ----------------------------------------------------------


def _bare_session(tmp_path):
    """A Session with no pool, for exercising the approval handshake alone."""
    from newal.web.session import Session

    return Session(
        config=load_config(use_env=False),
        agent=None,
        pool=None,
        index=None,
        transcript=None,
        upload_dir=tmp_path,
        approval_timeout_s=2.0,
    )


def test_approval_blocks_until_the_browser_answers(tmp_path):
    session = _bare_session(tmp_path)
    result = {}

    worker = threading.Thread(
        target=lambda: result.update(allowed=session.request_approval("run_shell", "ls"))
    )
    worker.start()

    # Wait for the agent thread to register the request, then answer it.
    for _ in range(200):
        if session._approvals:
            break
        threading.Event().wait(0.01)

    request_id = next(iter(session._approvals))
    assert session.answer_approval(request_id, True)
    worker.join(timeout=3)

    assert result["allowed"] is True


def test_a_declined_approval_returns_false(tmp_path):
    session = _bare_session(tmp_path)
    result = {}

    worker = threading.Thread(
        target=lambda: result.update(allowed=session.request_approval("run_shell", "rm x"))
    )
    worker.start()
    for _ in range(200):
        if session._approvals:
            break
        threading.Event().wait(0.01)

    session.answer_approval(next(iter(session._approvals)), False)
    worker.join(timeout=3)

    assert result["allowed"] is False


def test_an_unanswered_approval_times_out_as_a_refusal(tmp_path):
    """A closed tab must not leave the command permitted, or the thread pinned."""
    session = _bare_session(tmp_path)
    session.approval_timeout_s = 0.2

    assert session.request_approval("run_shell", "curl evil.example") is False
    assert session._approvals == {}, "the pending request must be cleaned up"


def test_answering_an_unknown_approval_is_a_404(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post(
            "/api/approve", data={"request_id": "nope", "allow": "true"}
        )
    assert response.status_code == 404


def test_shell_stays_blocked_when_the_policy_says_deny(tmp_path, monkeypatch):
    """With deny, run_shell is not even offered to the model."""
    def script(call: int):
        if call == 1:
            return Completion(
                tool_calls=[ToolCall("c1", "run_shell", {"command": "echo hi"})],
                usage=Usage(1, 1),
                model_key="heavy",
            )
        return Completion(text="못 했습니다.", usage=Usage(1, 1), model_key="heavy")

    with _client(tmp_path, monkeypatch, script, shell_policy="deny") as client:
        response = client.post("/api/chat", data={"message": "echo 해줘"})

    events = _events(response)
    assert not any(e["kind"] == "approval" for e in events)
    assert events[-1]["kind"] == "end"
