"""Export captured interactions as fine-tuning datasets.

Three formats, all JSONL, all in the shapes TRL's trainers accept directly:

``sft``
    ``{"messages": [...]}`` -- verified turns. What the assistant did when the
    project's own tests agreed it was right.

``dpo``
    ``{"prompt": [...], "chosen": [...], "rejected": [...]}`` -- built from the
    verify-and-repair loop. The attempt that failed the tests is the rejected
    completion, the repair that passed them is the chosen one.

``routing``
    ``{"text": ..., "label": "cheap"|"strong"}`` -- the router's own training
    set, for replacing the kNN classifier with a trained one.

What makes this data unusual is the labelling. There is no human annotator and
no LLM judge anywhere in it: every label comes from running the user's test
suite. That is the same verifiable-reward signal RLVR methods look for, except
it is produced as a by-product of ordinary work on a real repository.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

Format = Literal["sft", "dpo", "routing"]
FORMATS: tuple[Format, ...] = ("sft", "dpo", "routing")

# Roles TRL's conversational format understands. Anything else (a bare tool
# result with no id, say) would break the chat template at training time.
CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})


@dataclass
class ExportStats:
    written: int = 0
    skipped: int = 0
    reasons: dict[str, int] | None = None

    def skip(self, reason: str) -> None:
        self.skipped += 1
        if self.reasons is None:
            self.reasons = {}
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def _clean_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise one stored message into the conversational format."""
    role = message.get("role")
    if role not in CHAT_ROLES:
        return None

    cleaned: dict[str, Any] = {"role": role}
    content = message.get("content")
    # An assistant turn that only called tools has null content; the trainer
    # wants a string there, and the tool_calls carry the actual output.
    cleaned["content"] = "" if content is None else content

    if role == "assistant" and message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    if role == "tool":
        if not message.get("tool_call_id"):
            return None  # orphaned tool result -- unusable in a chat template
        cleaned["tool_call_id"] = message["tool_call_id"]
        if message.get("name"):
            cleaned["name"] = message["name"]
    return cleaned


def _clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = [_clean_message(m) for m in messages]
    return [m for m in cleaned if m is not None]


def _has_redacted_image(messages: list[dict[str, Any]]) -> bool:
    """True if any message references an attachment we no longer have."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def iter_sft(
    store: Any, *, include_unverified: bool = False, include_repaired: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield SFT samples from captured turns.

    Turns that needed a repair are excluded by default. Their message history
    still contains the attempt the tests rejected, plus the internal prompt that
    fed the failure back, so training on them would teach the model to get it
    wrong first. Those turns are exported as DPO pairs instead, which is the
    shape that can actually use a failure.
    """
    stats = ExportStats()
    turns = store.turns(
        verified_only=not include_unverified,
        exclude_repaired=not include_repaired,
    )

    for turn in turns:
        messages = turn["messages"]
        # A turn whose images were redacted would teach the model to answer
        # confidently about pictures it cannot see. Never train on those.
        if turn.get("had_attachments") or _has_redacted_image(messages):
            stats.skip("attachment redacted")
            continue

        cleaned = _clean_messages(messages)
        if len([m for m in cleaned if m["role"] == "assistant"]) == 0:
            stats.skip("no assistant turn")
            continue

        stats.written += 1
        yield {"messages": cleaned}

    log.info("sft export: %s written, %s skipped (%s)", stats.written, stats.skipped,
             stats.reasons or {})


def iter_dpo(store: Any) -> Iterator[dict[str, Any]]:
    """Yield DPO preference pairs from the verify-and-repair loop.

    Note what this pair is and is not. The repair actually happened *after* the
    model was shown the test failure, so chosen and rejected were not sampled
    from the same prefix. The pair is deliberately reconstructed against the
    original context: the lesson to learn is "given this task, produce the
    version that passes", not "given this failure message, fix it".
    """
    for pair in store.repair_pairs():
        prompt = _clean_messages(pair["context"])
        chosen = _clean_messages(pair["chosen"])
        rejected = _clean_messages(pair["rejected"])

        if not chosen or not rejected:
            continue
        if _has_redacted_image(pair["context"]):
            continue

        yield {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


def iter_routing(store: Any) -> Iterator[dict[str, Any]]:
    """Yield the router's classification set."""
    for prompt, label in store.routing_exemplars(limit=1_000_000):
        if prompt.strip():
            yield {"text": prompt, "label": label}


def export(
    store: Any,
    fmt: Format,
    destination: str | Path,
    *,
    include_unverified: bool = False,
) -> int:
    """Write a dataset to ``destination`` as JSONL. Returns the sample count."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {', '.join(FORMATS)}")

    if fmt == "sft":
        rows = iter_sft(store, include_unverified=include_unverified)
    elif fmt == "dpo":
        rows = iter_dpo(store)
    else:
        rows = iter_routing(store)

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
