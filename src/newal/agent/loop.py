"""The agent loop: plan, act with tools, then verify and repair."""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..backends.base import Backend, Completion, ToolCall, Usage
from ..config import Config
from ..media import Attachment
from ..memory import RepoIndex
from .prompts import PLAN_PROMPT, VERIFY_PROMPT, VIDEO_HINT, build_system_prompt, load_project_doc
from .tools import Toolbox, ToolResult
from .verifier import VerificationResult, run_verification

log = logging.getLogger(__name__)

EventKind = Literal[
    "phase", "thinking", "assistant", "tool_call", "tool_result", "verify", "warning"
]
EventCallback = Callable[[EventKind, str], None]

# Leave room for the reply plus a safety margin when trimming history.
CONTEXT_SAFETY_MARGIN = 4096
# Rough chars-per-token for mixed prose and code. Only used for trimming
# decisions, so a coarse estimate is fine.
CHARS_PER_TOKEN = 3.5


@dataclass
class AgentResult:
    text: str
    steps: int
    usage: Usage = field(default_factory=Usage)
    files_written: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    plan: str | None = None
    hit_step_limit: bool = False


class Agent:
    """Stateful conversation over a workspace.

    One instance per session: it keeps the message history so follow-up turns
    reuse everything already established.
    """

    def __init__(
        self,
        config: Config,
        backend: Backend,
        toolbox: Toolbox,
        *,
        index: RepoIndex | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.toolbox = toolbox
        self.index = index
        self._emit = on_event or (lambda _kind, _text: None)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        self.usage = Usage()

    def _system_prompt(self) -> str:
        notes: list[str] = []
        if self.index is not None:
            notes = [
                f"[{note.topic}] {note.content}"
                for note in self.index.store.recent_notes(limit=15)
            ]
        return build_system_prompt(
            self.toolbox.workspace.root,
            platform.platform(),
            notes=notes,
            project_doc=load_project_doc(self.toolbox.workspace.root),
        )

    # ---- public entry point ---------------------------------------------------

    def run(
        self, user_text: str, attachments: list[Attachment] | None = None
    ) -> AgentResult:
        attachments = attachments or []
        self.messages.append(self._user_message(user_text, attachments))
        self.toolbox.files_written.clear()

        plan: str | None = None
        if self.config.agent.plan_first:
            plan = self._draft_plan()

        result = self._act()
        result.plan = plan

        if self.config.agent.verify and self.toolbox.files_written:
            result.verification = self._verify_and_repair(result)

        result.files_written = sorted(self.toolbox.files_written)
        result.usage = self.usage
        return result

    # ---- phases ---------------------------------------------------------------

    def _draft_plan(self) -> str | None:
        """Ask for a plan without granting write tools."""
        self._emit("phase", "planning")
        read_only = [
            schema
            for schema in self.toolbox.schemas()
            if schema["function"]["name"] in {"read_file", "list_dir", "grep", "search_memory"}
        ]
        probe = self.messages + [{"role": "user", "content": PLAN_PROMPT}]

        try:
            completion = self.backend.complete(probe, tools=read_only, enable_thinking=True)
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            log.warning("planning turn failed, continuing without a plan: %s", exc)
            self._emit("warning", f"planning skipped: {exc}")
            return None

        self.usage = self.usage + completion.usage

        # The planner may call read-only tools; let it finish those before we
        # take its plan text.
        steps = 0
        while completion.wants_tools and steps < 8:
            probe.append(_assistant_message(completion))
            for call in completion.tool_calls:
                self._emit("tool_call", f"{call.name} {_short_args(call)}")
                result = self.toolbox.call(call.name, call.arguments)
                probe.append(_tool_message(call, result))
            completion = self.backend.complete(probe, tools=read_only, enable_thinking=True)
            self.usage = self.usage + completion.usage
            steps += 1

        plan = completion.text.strip()
        if plan:
            self._emit("assistant", plan)
            # Feed the plan back as context for the acting phase.
            self.messages.append({"role": "assistant", "content": plan})
        return plan or None

    def _act(self) -> AgentResult:
        """Run the tool-calling loop until the model produces a final answer."""
        self._emit("phase", "working")
        tools = self.toolbox.schemas()
        steps = 0

        while steps < self.config.agent.max_steps:
            self._trim_history()
            completion = self.backend.complete(self.messages, tools=tools)
            self.usage = self.usage + completion.usage
            steps += 1

            if completion.reasoning and self.config.ui.show_thinking:
                self._emit("thinking", completion.reasoning)

            if not completion.wants_tools:
                text = completion.text.strip()
                self.messages.append({"role": "assistant", "content": text})
                self._emit("assistant", text)
                return AgentResult(text=text, steps=steps)

            self.messages.append(_assistant_message(completion))
            if completion.text.strip():
                self._emit("assistant", completion.text.strip())

            for call in completion.tool_calls:
                self._emit("tool_call", f"{call.name} {_short_args(call)}")
                result = self.toolbox.call(call.name, call.arguments)
                self._emit("tool_result", _preview(result))
                self.messages.append(_tool_message(call, result))

        self._emit("warning", f"stopped after {steps} steps (agent.max_steps)")
        return AgentResult(
            text=(
                f"I stopped after {steps} tool steps without reaching a conclusion. "
                "Raise agent.max_steps, or narrow the request into smaller pieces."
            ),
            steps=steps,
            hit_step_limit=True,
        )

    def _verify_and_repair(self, result: AgentResult) -> VerificationResult:
        """Run the project's tests; on failure, hand the output back for repair."""
        self._emit("phase", "verifying")
        verification = run_verification(
            self.toolbox, configured_command=self.config.agent.verify_command
        )
        if verification.skipped or verification.passed:
            self._emit("verify", verification.output if verification.skipped else "passed")
            return verification

        for attempt in range(1, self.config.agent.max_verify_retries + 1):
            self._emit("verify", f"failed (repair attempt {attempt})")
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{VERIFY_PROMPT}\n\n"
                        f"`{verification.command}` failed:\n\n{verification.output}"
                    ),
                }
            )
            repair = self._act()
            result.text = repair.text
            result.steps += repair.steps

            verification = run_verification(
                self.toolbox, configured_command=self.config.agent.verify_command
            )
            if verification.passed:
                self._emit("verify", f"passed after {attempt} repair attempt(s)")
                return verification

        self._emit("verify", "still failing after all repair attempts")
        return verification

    # ---- message construction -------------------------------------------------

    def _user_message(self, text: str, attachments: list[Attachment]) -> dict[str, Any]:
        if not attachments:
            return {"role": "user", "content": text}

        parts: list[dict[str, Any]] = []
        for attachment in attachments:
            parts.append({"type": "text", "text": f"[attached {attachment.kind}: "
                                                  f"{attachment.summary}]"})
            parts.extend(attachment.parts)
        if any(a.kind == "video" for a in attachments):
            parts.append({"type": "text", "text": VIDEO_HINT})
        parts.append({"type": "text", "text": text})
        return {"role": "user", "content": parts}

    def _trim_history(self) -> None:
        """Drop the oldest turns when the transcript approaches the context limit.

        The system message and the two most recent turns are always kept, and a
        tool result is never separated from the assistant message that asked
        for it -- an orphaned tool message is a hard API error.
        """
        limit = self.config.model.context_length - self.config.model.max_output_tokens
        limit -= CONTEXT_SAFETY_MARGIN
        if limit <= 0 or self._estimated_tokens() <= limit:
            return

        head, body = self.messages[:1], self.messages[1:]
        while len(body) > 4 and self._estimated_tokens() > limit:
            # Drop from the front, taking any tool messages that follow with it.
            body.pop(0)
            while body and body[0].get("role") == "tool":
                body.pop(0)
            self.messages = head + body

        if self._estimated_tokens() > limit:
            self._emit(
                "warning",
                "conversation still exceeds the context window after trimming; "
                "start a fresh session with /reset",
            )

    def _estimated_tokens(self) -> int:
        total = 0
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        total += len(part.get("text", ""))
                    else:
                        # An encoded image is huge as a string but costs its
                        # measured visual tokens, not its base64 length.
                        total += 300 * int(CHARS_PER_TOKEN)
            for call in message.get("tool_calls", []) or []:
                total += len(str(call))
        return int(total / CHARS_PER_TOKEN)

    def reset(self) -> None:
        """Clear the conversation, keeping the workspace and memory."""
        self.messages = [{"role": "system", "content": self._system_prompt()}]
        self.usage = Usage()


# ---- message helpers ----------------------------------------------------------


def _assistant_message(completion: Completion) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": completion.text or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _json_dumps(call.arguments)},
            }
            for call in completion.tool_calls
        ],
    }


def _tool_message(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    prefix = "ERROR: " if result.is_error else ""
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": prefix + result.content,
    }


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _short_args(call: ToolCall, limit: int = 100) -> str:
    rendered = ", ".join(
        f"{key}={_truncate(str(value), 40)}" for key, value in call.arguments.items()
    )
    return _truncate(rendered, limit)


def _preview(result: ToolResult, limit: int = 240) -> str:
    first = result.content.strip().splitlines()
    head = first[0] if first else ""
    suffix = f" (+{len(first) - 1} lines)" if len(first) > 1 else ""
    return _truncate(head, limit) + suffix


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."
