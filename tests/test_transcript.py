"""Session transcript writing and attachment redaction."""

from __future__ import annotations

import json

from newal.agent.transcript import (
    REDACTED_IMAGE,
    TranscriptWriter,
    new_session_id,
    redact_content,
    redact_messages,
)

BIG_IMAGE = "data:image/jpeg;base64," + "A" * 50_000


# ---- redaction ---------------------------------------------------------------


def test_plain_string_content_is_untouched():
    assert redact_content("just text") == "just text"


def test_image_payload_is_replaced():
    parts = redact_content(
        [{"type": "image_url", "image_url": {"url": BIG_IMAGE}}]
    )
    assert parts[0]["image_url"]["url"] == REDACTED_IMAGE


def test_text_parts_survive_alongside_images():
    parts = redact_content(
        [
            {"type": "text", "text": "[t=1.50s]"},
            {"type": "image_url", "image_url": {"url": BIG_IMAGE}},
            {"type": "text", "text": "what is wrong here?"},
        ]
    )
    assert parts[0]["text"] == "[t=1.50s]"
    assert parts[2]["text"] == "what is wrong here?"
    assert BIG_IMAGE not in json.dumps(parts)


def test_redaction_does_not_mutate_the_original():
    original = [{"type": "image_url", "image_url": {"url": BIG_IMAGE}}]
    redact_content(original)
    assert original[0]["image_url"]["url"] == BIG_IMAGE


def test_redact_messages_keeps_roles_and_tool_calls():
    messages = [
        {"role": "system", "content": "you are newal"},
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": BIG_IMAGE}}],
        },
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    redacted = redact_messages(messages)

    assert [m["role"] for m in redacted] == ["system", "user", "assistant", "tool"]
    assert redacted[2]["tool_calls"] == [{"id": "c1"}]
    assert BIG_IMAGE not in json.dumps(redacted)
    # Original history must stay usable for the next model call.
    assert messages[1]["content"][0]["image_url"]["url"] == BIG_IMAGE


# ---- writing -----------------------------------------------------------------


def test_session_id_is_filesystem_safe():
    session = new_session_id()
    assert not set(session) & set(r'/\:*?"<>|')


def test_record_creates_the_directory_and_file(tmp_path):
    writer = TranscriptWriter.create(tmp_path / "nested" / "transcripts", "test")
    writer.record(prompt="hello", response="hi")

    assert writer.path.is_file()
    assert writer.path.name == "session-test.jsonl"


def test_each_turn_appends_one_line(tmp_path):
    writer = TranscriptWriter.create(tmp_path, "test")
    writer.record(prompt="first", response="1")
    writer.record(prompt="second", response="2")

    entries = writer.read_all()
    assert [e["prompt"] for e in entries] == ["first", "second"]


def test_recorded_turn_carries_the_useful_fields(tmp_path):
    writer = TranscriptWriter.create(tmp_path, "test")
    writer.record(
        prompt="fix add()",
        response="done",
        routes=["light (code, direct, score=0.35)"],
        attachments=["/tmp/bug.mp4"],
        files_written=["app.py"],
        verification=("python -m pytest -q", True),
        usage_by_model={"heavy": 74, "light": 30},
        steps=4,
        escalations=1,
    )

    entry = writer.read_all()[0]
    assert entry["files_written"] == ["app.py"]
    assert entry["verification"] == {"command": "python -m pytest -q", "passed": True}
    assert entry["usage_by_model"]["heavy"] == 74
    assert entry["escalations"] == 1
    assert isinstance(entry["timestamp"], float)


def test_verification_is_omitted_when_it_did_not_run(tmp_path):
    writer = TranscriptWriter.create(tmp_path, "test")
    writer.record(prompt="what does this do?", response="reads a file")
    assert "verification" not in writer.read_all()[0]


def test_korean_text_is_not_escaped(tmp_path):
    writer = TranscriptWriter.create(tmp_path, "test")
    writer.record(prompt="변수 이름 바꿔줘", response="바꿨습니다")

    raw = writer.path.read_text(encoding="utf-8")
    assert "변수 이름 바꿔줘" in raw  # ensure_ascii=False, readable on disk


def test_write_failure_is_swallowed(tmp_path):
    """A transcript problem must never end a working session."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file", encoding="utf-8")

    writer = TranscriptWriter.create(blocker, "test")
    assert writer.record(prompt="hello", response="hi") is None


def test_reading_a_missing_transcript_returns_empty(tmp_path):
    assert TranscriptWriter.create(tmp_path, "absent").read_all() == []


def test_malformed_lines_are_skipped(tmp_path):
    writer = TranscriptWriter.create(tmp_path, "test")
    writer.record(prompt="good", response="ok")
    with writer.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    entries = writer.read_all()
    assert len(entries) == 1
    assert entries[0]["prompt"] == "good"
