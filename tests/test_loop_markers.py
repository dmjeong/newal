"""Capture markers must survive history trimming.

`_attempt_start` and `_repair_start` are absolute indices into the message
list, and `_trim_history` drops entries from the front of that list. If the
markers are not moved with it, the captured turn and the preference pair
describe the wrong messages -- and because nothing raises, the wrong data is
stored silently.
"""

from __future__ import annotations

from newal.agent.loop import MARKER_LOST, Agent, _shift_marker
from newal.agent.tools import Toolbox
from newal.backends.base import Completion, Usage
from newal.config import load_config
from newal.memory import build_index
from newal.models import Router

# ---- the marker arithmetic ---------------------------------------------------


def test_marker_slides_by_the_number_dropped():
    # Trimming keeps messages[0] and removes 3 after it, so index 10 -> 7.
    assert _shift_marker(10, 3) == 7


def test_marker_is_unchanged_when_nothing_was_dropped():
    assert _shift_marker(10, 0) == 10


def test_marker_is_lost_when_its_own_message_was_dropped():
    """Clamping to another message would claim a slice that is not the attempt."""
    assert _shift_marker(2, 5) == MARKER_LOST


def test_first_surviving_index_is_kept():
    # dropped=3 removes original indices 1..3, so 4 is the first survivor.
    assert _shift_marker(4, 3) == 1
    assert _shift_marker(3, 3) == MARKER_LOST


def test_a_lost_marker_stays_lost():
    assert _shift_marker(MARKER_LOST, 2) == MARKER_LOST


# ---- the same thing through the real Agent -----------------------------------


class _Backend:
    def __init__(self, replies):
        self.replies, self.calls = replies, 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        index = min(self.calls - 1, len(self.replies) - 1)
        return self.replies[index]

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

    def describe(self):
        return []

    def close(self):
        return None


# _trim_history computes: context_length - max_output_tokens - CONTEXT_SAFETY_MARGIN
# and gives up when that is <= 0. With the 4096 margin, a "small" context_length
# disables trimming entirely instead of forcing it -- which is how two of these
# tests originally passed while exercising nothing. This leaves ~390 tokens of
# room, so a few hundred characters of history is enough to trigger a trim.
TRIMMING_CONTEXT = 4500


def _agent(tmp_path, *, context_length: int, replies):
    config = load_config(
        use_env=False,
        overrides={
            "tools": {"workspace_root": str(tmp_path)},
            "agent": {"plan_first": False, "verify": False},
            "generation": {"context_length": context_length, "max_output_tokens": 16},
            "memory": {"db_path": str(tmp_path / ".newal" / "m.db")},
        },
    )
    index = build_index(tmp_path, config.memory)
    pool = _Pool(config, _Backend(replies))
    return Agent(config, pool, Toolbox(config.tools, index=index), index=index), index


def test_capture_is_correct_when_no_trimming_happens(tmp_path):
    agent, index = _agent(
        tmp_path,
        context_length=1_000_000,
        replies=[Completion(text="done", usage=Usage(1, 1), model_key="heavy")],
    )
    agent.run("첫 번째 질문")

    turns = index.store.turns()
    assert len(turns) == 1
    # The stored slice is the user prompt plus the assistant's reply.
    assert [m["role"] for m in turns[0]["messages"]] == ["user", "assistant"]
    assert turns[0]["messages"][-1]["content"] == "done"


def test_trimming_does_not_corrupt_the_captured_turn(tmp_path):
    """Across enough turns the history gets trimmed mid-conversation."""
    agent, index = _agent(
        tmp_path,
        context_length=TRIMMING_CONTEXT,
        replies=[Completion(text="답변 " + "가" * 200, usage=Usage(1, 1), model_key="heavy")],
    )
    for number in range(8):
        agent.run(f"질문 {number} " + "나" * 200)

    # Guard against the test going vacuous: the history must actually have been
    # trimmed, otherwise this proves nothing about marker shifting.
    assert len(agent.messages) < 17, "expected trimming to drop earlier turns"

    turns = index.store.turns()
    assert turns, "trimming should not stop capture entirely"
    for turn in turns:
        roles = [m["role"] for m in turn["messages"]]
        # Whatever survived must still be a user prompt followed by assistant
        # work -- never the system prompt or a stranded fragment.
        assert roles[0] == "user"
        assert "system" not in roles
        assert "assistant" in roles


def test_markers_move_with_the_messages_they_point_at(tmp_path):
    agent, _ = _agent(
        tmp_path,
        context_length=TRIMMING_CONTEXT,
        replies=[Completion(text="ok", usage=Usage(1, 1), model_key="heavy")],
    )
    agent.messages.extend(
        {"role": "user", "content": f"filler {i} " + "가" * 120} for i in range(12)
    )
    agent._attempt_start = 11
    agent._repair_start = 12
    marked_attempt = agent.messages[11]
    marked_repair = agent.messages[12]

    before = len(agent.messages)
    agent._trim_history()
    assert len(agent.messages) < before, "expected a trim"

    # The markers must still point at the very same message objects.
    assert agent._attempt_start > MARKER_LOST
    assert agent.messages[agent._attempt_start] is marked_attempt
    assert agent.messages[agent._repair_start] is marked_repair


def test_a_trimmed_away_boundary_skips_capture_instead_of_storing_junk(tmp_path):
    agent, index = _agent(
        tmp_path,
        context_length=TRIMMING_CONTEXT,
        replies=[Completion(text="ok", usage=Usage(1, 1), model_key="heavy")],
    )
    agent.messages.extend(
        {"role": "user", "content": f"filler {i} " + "다" * 120} for i in range(20)
    )
    # Point the marker at a message trimming is certain to remove.
    agent._attempt_start = 2

    before = len(agent.messages)
    agent._trim_history()
    assert len(agent.messages) < before, "expected a trim"

    assert agent._attempt_start == MARKER_LOST

    agent._record_turn("질문", _result(), had_attachments=False)
    assert index.store.training_counts()["turns"] == 0


def _result():
    from newal.agent.loop import AgentResult

    return AgentResult(text="ok", steps=1)
