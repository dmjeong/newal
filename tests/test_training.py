"""Fine-tuning data capture and export."""

from __future__ import annotations

import json

import pytest

from newal.memory.store import MemoryStore
from newal.training.export import export, iter_dpo, iter_routing, iter_sft

BIG_IMAGE = "data:image/jpeg;base64," + "A" * 10_000


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def _turn(store, **overrides):
    payload = {
        "session_id": "s1",
        "prompt": "fix add()",
        "messages": [
            {"role": "user", "content": "fix add()"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "edit_file", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "edited"},
            {"role": "assistant", "content": "Fixed the sign error."},
        ],
        "model_key": "light",
        "verify_ok": True,
        "escalated": False,
        "had_attachments": False,
    }
    payload.update(overrides)
    return store.record_turn(**payload)


# ---- capture -----------------------------------------------------------------


def test_turn_round_trips_through_sqlite(store):
    _turn(store)
    turns = store.turns()
    assert len(turns) == 1
    assert turns[0]["messages"][0]["content"] == "fix add()"


def test_training_counts_report_each_table(store):
    _turn(store)
    _turn(store, verify_ok=False)
    _turn(store, had_attachments=True)
    store.record_repair_pair(
        session_id="s1", prompt="p", context=[], rejected=[{"role": "assistant"}],
        chosen=[{"role": "assistant"}], failure_output="boom", verify_command="pytest",
    )

    counts = store.training_counts()
    assert counts["turns"] == 3
    assert counts["turns_verified"] == 2
    assert counts["turns_with_attachments"] == 1
    assert counts["repair_pairs"] == 1


def test_verified_only_filter(store):
    _turn(store, verify_ok=True)
    _turn(store, verify_ok=False)
    _turn(store, verify_ok=None)
    assert len(store.turns(verified_only=True)) == 1
    assert len(store.turns()) == 3


def test_clearing_training_data_keeps_notes_and_routing(store):
    _turn(store)
    store.add_note("build", "run pytest")
    store.record_route_outcome(
        "p", "cheap", model_key="light", escalated=False, verify_ok=True
    )

    store.clear_training_data()

    assert store.training_counts()["turns"] == 0
    assert len(store.recent_notes()) == 1
    assert store.training_counts()["route_outcomes"] == 1


# ---- SFT export --------------------------------------------------------------


def test_sft_exports_only_verified_turns_by_default(store):
    _turn(store, verify_ok=True)
    _turn(store, verify_ok=False)
    assert len(list(iter_sft(store))) == 1


def test_sft_can_include_unverified(store):
    _turn(store, verify_ok=True)
    _turn(store, verify_ok=False)
    assert len(list(iter_sft(store, include_unverified=True))) == 2


def test_sft_skips_turns_whose_images_were_redacted(store):
    """Training on a redacted image would teach answering about unseen pictures."""
    _turn(
        store,
        verify_ok=True,
        had_attachments=True,
        messages=[
            {
                "role": "user",
                "content": [{"type": "image_url",
                             "image_url": {"url": "<image redacted>"}}],
            },
            {"role": "assistant", "content": "the button is misaligned"},
        ],
    )
    assert list(iter_sft(store)) == []


def test_sft_skips_a_turn_with_no_assistant_message(store):
    _turn(store, messages=[{"role": "user", "content": "hello"}])
    assert list(iter_sft(store)) == []


def test_sft_preserves_tool_calls(store):
    _turn(store)
    sample = next(iter(iter_sft(store)))
    assistant = [m for m in sample["messages"] if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "edit_file"


def test_sft_replaces_null_content_with_empty_string(store):
    """A tool-only assistant turn has null content; chat templates need a string."""
    _turn(store)
    sample = next(iter(iter_sft(store)))
    assert all(m["content"] is not None for m in sample["messages"])


def test_sft_drops_orphaned_tool_results(store):
    _turn(
        store,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result with no id"},
            {"role": "assistant", "content": "done"},
        ],
    )
    sample = next(iter(iter_sft(store)))
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]


def test_sft_drops_unknown_roles(store):
    _turn(
        store,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "internal"},
            {"role": "assistant", "content": "done"},
        ],
    )
    sample = next(iter(iter_sft(store)))
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]


# ---- DPO export --------------------------------------------------------------


def _pair(store, **overrides):
    payload = {
        "session_id": "s1",
        "prompt": "fix add()",
        "context": [{"role": "user", "content": "fix add()"}],
        "rejected": [{"role": "assistant", "content": "return a * b"}],
        "chosen": [{"role": "assistant", "content": "return a + b"}],
        "failure_output": "assert 6 == 5",
        "verify_command": "pytest -q",
    }
    payload.update(overrides)
    return store.record_repair_pair(**payload)


def test_dpo_pair_has_the_three_required_keys(store):
    _pair(store)
    sample = next(iter(iter_dpo(store)))
    assert set(sample) == {"prompt", "chosen", "rejected"}


def test_dpo_chosen_is_the_repair_that_passed(store):
    _pair(store)
    sample = next(iter(iter_dpo(store)))
    assert sample["chosen"][0]["content"] == "return a + b"
    assert sample["rejected"][0]["content"] == "return a * b"


def test_dpo_prompt_is_the_original_context_not_the_failure_message(store):
    """The lesson is 'produce the passing version', not 'react to this error'."""
    _pair(store)
    sample = next(iter(iter_dpo(store)))
    rendered = json.dumps(sample["prompt"])
    assert "fix add()" in rendered
    assert "assert 6 == 5" not in rendered


def test_dpo_skips_pairs_with_an_empty_side(store):
    _pair(store, chosen=[])
    assert list(iter_dpo(store)) == []


def test_dpo_skips_pairs_whose_context_had_an_image(store):
    _pair(
        store,
        context=[
            {"role": "user",
             "content": [{"type": "image_url", "image_url": {"url": "<image redacted>"}}]}
        ],
    )
    assert list(iter_dpo(store)) == []


# ---- routing export ----------------------------------------------------------


def test_routing_export_shape(store):
    store.record_route_outcome(
        "변수 이름 바꿔줘", "strong", model_key="light", escalated=True, verify_ok=False
    )
    sample = next(iter(iter_routing(store)))
    assert sample == {"text": "변수 이름 바꿔줘", "label": "strong"}


# ---- file writing ------------------------------------------------------------


def test_export_writes_jsonl(store, tmp_path):
    _turn(store)
    _turn(store)
    out = tmp_path / "nested" / "sft.jsonl"

    written = export(store, "sft", out)
    assert written == 2

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "messages" in json.loads(lines[0])


def test_export_keeps_korean_readable(store, tmp_path):
    store.record_route_outcome(
        "오타 고쳐줘", "cheap", model_key="light", escalated=False, verify_ok=True
    )
    out = tmp_path / "routing.jsonl"
    export(store, "routing", out)
    assert "오타 고쳐줘" in out.read_text(encoding="utf-8")


def test_export_rejects_an_unknown_format(store, tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        export(store, "grpo", tmp_path / "x.jsonl")


def test_export_of_an_empty_store_writes_an_empty_file(store, tmp_path):
    out = tmp_path / "empty.jsonl"
    assert export(store, "dpo", out) == 0
    assert out.read_text(encoding="utf-8") == ""


# ---- repaired turns are DPO material, not SFT material -----------------------


def test_repaired_turns_are_excluded_from_sft(store):
    """Their history still contains the attempt the tests rejected."""
    _turn(store, verify_ok=True, repaired=True)
    assert list(iter_sft(store)) == []


def test_repaired_turns_can_be_opted_back_in(store):
    _turn(store, verify_ok=True, repaired=True)
    assert len(list(iter_sft(store, include_repaired=True))) == 1


def test_clean_first_try_turns_still_export(store):
    _turn(store, verify_ok=True, repaired=False)
    _turn(store, verify_ok=True, repaired=True)
    assert len(list(iter_sft(store))) == 1


def test_counts_separate_clean_first_try_turns(store):
    _turn(store, verify_ok=True, repaired=False)
    _turn(store, verify_ok=True, repaired=True)
    _turn(store, verify_ok=True, repaired=False, had_attachments=True)

    counts = store.training_counts()
    assert counts["turns_verified"] == 3
    assert counts["turns_clean_first_try"] == 1
