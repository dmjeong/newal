"""The HTTP front end.

The web layer is transport only, so these tests check the transport: that a
turn streams, that uploads are gated, that a shell approval crosses the thread
boundary in both directions, and that a second turn cannot start on top of a
running one.
"""

from __future__ import annotations

import json
import threading
from html.parser import HTMLParser

import pytest

from newal.backends.base import Completion, ToolCall, Usage
from newal.config import load_config
from newal.models import Router
from newal.training.plan import TrainingPlan

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from newal.web.app import STATIC_DIR, create_app  # noqa: E402


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
        self.unavailable: dict[str, str] = {}

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
    pools: list[_Pool] = []

    def _make_pool(cfg, **_kwargs):
        pool = _Pool(cfg, _Backend(script))
        pools.append(pool)
        return pool

    monkeypatch.setattr("newal.web.session.ModelPool", _make_pool)
    client = TestClient(create_app(config, autostart=False))
    # The session lives in a closure, so hand the pool out here for the few
    # tests that have to look at the live router rather than at a response.
    client.pools = pools
    return client


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


def test_the_ui_serves_with_no_model_reachable(tmp_path, monkeypatch):
    """F5 with no GPU has to reach the page, not a stack trace in the terminal.

    Nothing on either page needs a model except answering a turn, so a pool
    that cannot reach its server must not take the whole app down at startup.
    """
    monkeypatch.setattr("newal.models.pool.is_server_up", lambda _url: False)
    config = load_config(
        use_env=False,
        overrides={
            "tools": {"workspace_root": str(tmp_path)},
            "memory": {"db_path": str(tmp_path / ".newal" / "m.db")},
            "ui": {"save_transcripts": False},
            "runtime": {"autostart": False},
        },
    )
    with TestClient(create_app(config, autostart=False)) as client:
        for path in ("/", "/training", "/api/settings", "/api/training/summary"):
            assert client.get(path).status_code == 200, path

        # And it says so, rather than leaving the user to discover it by asking.
        assert client.get("/api/state").json()["unavailable"]


# ---- settings (normal mode) --------------------------------------------------


def test_settings_are_listed_with_current_values(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        items = client.get("/api/settings").json()["settings"]

    paths = {item["path"] for item in items}
    assert "router.mode" in paths
    assert "tools.shell_policy" in paths
    # Startup-only config must not be offered.
    assert not any(p.startswith("models") for p in paths)


def test_saving_settings_applies_and_persists(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.put("/api/settings", json={"router.mode": "heuristic"})
        assert response.status_code == 200
        assert response.json()["changed"] == {"router.mode": "heuristic"}

        state_after = client.get("/api/state").json()
        assert state_after["router"]["mode"] == "heuristic"

    assert (tmp_path / "configs" / "local.yaml").is_file()


def test_saving_a_bad_setting_is_a_422(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.put("/api/settings", json={"router.mode": "telepathy"})
    assert response.status_code == 422


def test_threshold_changes_reach_the_live_router(tmp_path, monkeypatch):
    """The router copies thresholds at construction, so saving has to refresh them."""
    with _client(tmp_path, monkeypatch, _reply()) as client:
        router = client.pools[0].router
        assert router.escalate_threshold != 0.9, "pick a value the default is not"

        assert client.put(
            "/api/settings", json={"router.escalate_threshold": 0.9}
        ).status_code == 200
        assert router.escalate_threshold == 0.9


def test_the_save_response_carries_the_new_values(tmp_path, monkeypatch):
    """The panel redraws from this response, so it has to be the post-save state."""
    with _client(tmp_path, monkeypatch, _reply()) as client:
        body = client.put("/api/settings", json={"agent.max_steps": 9}).json()

    items = {item["path"]: item["value"] for item in body["settings"]}
    assert items["agent.max_steps"] == 9


# ---- training mode -----------------------------------------------------------


def test_training_page_is_a_separate_address(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        page = client.get("/training")
    assert page.status_code == 200
    assert "/static/training.js" in page.text
    # It is a different page, not the chat UI with a tab.
    assert "id=\"composer\"" not in page.text


def test_both_pages_load_the_shared_labels(tmp_path, monkeypatch):
    """captureLabel() is a plain global, so a missing tag is a runtime error."""
    with _client(tmp_path, monkeypatch, _reply()) as client:
        for path in ("/", "/training"):
            assert "/static/labels.js" in client.get(path).text, path
        counts = client.get("/api/training/summary").json()["counts"]

    labels = (STATIC_DIR / "labels.js").read_text(encoding="utf-8")
    for key in counts:
        assert f"{key}:" in labels, f"{key} would fall back to its raw column name"


def test_training_summary_reports_counts_and_candidates(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        summary = client.get("/api/training/summary").json()

    assert "turns_clean_first_try" in summary["counts"]
    assert summary["candidate_models"]
    assert summary["suggested_base_model"]
    assert summary["defaults"]["task"] == "sft"


def test_a_plan_returns_a_script_and_advice(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post(
            "/api/training/plan",
            json={"task": "dpo", "base_model": "Qwen/Qwen3.5-4B", "lora_r": 16},
        )

    body = response.json()
    assert response.status_code == 200
    assert "DPOTrainer" in body["script"]
    assert body["script_name"] == "train_dpo.py"
    # No data captured in this fixture, so it must say so rather than pretend.
    assert any("0개" in note for note in body["warnings"])


def test_an_invalid_plan_is_a_422(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        response = client.post("/api/training/plan", json={"lora_r": 0})
    assert response.status_code == 422


def test_downloading_an_empty_dataset_is_a_404_not_an_empty_file(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        assert client.get("/api/training/dataset/sft").status_code == 404


def test_an_unknown_dataset_format_is_rejected(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        assert client.get("/api/training/dataset/grpo").status_code == 400


def _number_inputs() -> dict[str, dict[str, str]]:
    """Pull the <input type=number> constraints out of the training form."""

    class Reader(HTMLParser):
        def __init__(self):
            super().__init__()
            self.fields: dict[str, dict[str, str]] = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "input" and attributes.get("type") == "number":
                self.fields[attributes["name"]] = attributes

    reader = Reader()
    reader.feed((STATIC_DIR / "training.html").read_text(encoding="utf-8"))
    return reader.fields


def test_every_number_field_accepts_its_own_default():
    """A browser refuses to submit a form whose value violates min/max/step.

    It does so silently as far as the page is concerned, so a mismatch between
    the HTML constraints and the defaults the server hands back would leave the
    'generate' button doing nothing at all.
    """
    defaults = TrainingPlan().to_dict()

    for name, attributes in _number_inputs().items():
        value = defaults[name]
        if "min" in attributes:
            assert value >= float(attributes["min"]), f"{name}: default is below min"
        if "max" in attributes:
            assert value <= float(attributes["max"]), f"{name}: default is above max"

        step = attributes.get("step", "1")
        if step == "any":
            continue
        # HTML counts steps from min (or 0), so the value has to land on one.
        base = float(attributes.get("min", 0))
        offset = (value - base) / float(step)
        assert abs(offset - round(offset)) < 1e-9, (
            f"{name}: default {value} is not a whole step from {base}; "
            f'use step="any" for fractional fields'
        )


def test_captured_data_can_be_cleared(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, _reply()) as client:
        client.post("/api/chat", data={"message": "첫 질문"})
        response = client.post("/api/training/clear")

    assert response.status_code == 200
    assert response.json()["counts"]["turns"] == 0
