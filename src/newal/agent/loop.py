"""The agent loop: route, plan, act with tools, then verify and repair.

Every model call goes through the router, so the tier and the thinking mode are
decided from what is actually happening -- how the request reads, whether tools
have been failing, whether verification already rejected the work -- rather than
being fixed in config.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..backends.base import Completion, ToolCall, Usage
from ..config import Config
from ..media import Attachment
from ..memory import RepoIndex
from ..models import ModelPool, Role, Route, RouteSignals, label_from_outcome
from .prompts import PLAN_PROMPT, VERIFY_PROMPT, VIDEO_HINT, build_system_prompt, load_project_doc
from .tools import Toolbox, ToolResult
from .verifier import VerificationResult, run_verification

log = logging.getLogger(__name__)

EventKind = Literal[
    "phase", "route", "thinking", "assistant", "tool_call", "tool_result", "verify", "warning"
]
EventCallback = Callable[[EventKind, str], None]

# Leave room for the reply plus a safety margin when trimming history.
CONTEXT_SAFETY_MARGIN = 4096
# Rough chars-per-token for mixed prose and code. Only used for trimming
# decisions, so a coarse estimate is fine.
CHARS_PER_TOKEN = 3.5
# Read-only tools the planning phase is allowed to touch.
PLANNING_TOOLS = frozenset({"read_file", "list_dir", "grep", "search_memory"})


@dataclass
class AgentResult:
    text: str
    steps: int
    usage: Usage = field(default_factory=Usage)
    usage_by_model: dict[str, Usage] = field(default_factory=dict)
    files_written: list[str] = field(default_factory=list)
    verification: VerificationResult | None = None
    plan: str | None = None
    routes: list[str] = field(default_factory=list)
    escalations: int = 0
    hit_step_limit: bool = False


class Agent:
    """Stateful conversation over a workspace, backed by a pool of models."""

    def __init__(
        self,
        config: Config,
        pool: ModelPool,
        toolbox: Toolbox,
        *,
        index: RepoIndex | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.config = config
        self.pool = pool
        self.toolbox = toolbox
        self.index = index
        self._emit = on_event or (lambda _kind, _text: None)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        self.usage = Usage()
        self.usage_by_model: dict[str, Usage] = {}

        # Per-turn routing state.
        self._prompt = ""
        self._has_attachments = False
        self._has_video = False
        self._tool_errors = 0
        self._verification_failed = False
        self._escalations = 0
        self._routes: list[str] = []
        # The tier the turn actually started on, which is what the outcome
        # label is about -- later escalations are the outcome, not the choice.
        self._first_route: Route | None = None

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

    # ---- routing --------------------------------------------------------------

    def _signals(self, *, attempt: int = 0) -> RouteSignals:
        return RouteSignals(
            prompt=self._prompt,
            has_attachments=self._has_attachments,
            has_video=self._has_video,
            attempt=attempt,
            tool_errors=self._tool_errors,
            verification_failed=self._verification_failed,
            files_touched=len(self.toolbox.files_written),
        )

    def _call(
        self,
        role: Role,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        route: Route | None = None,
        attempt: int = 0,
    ) -> tuple[Completion, Route]:
        """Route and execute one model call, recording usage against its model."""
        if route is None:
            route = self.pool.router.route(role, self._signals(attempt=attempt))

        self._routes.append(route.describe())
        if self._first_route is None:
            self._first_route = route
        if self.config.router.explain:
            self._emit("route", f"{route.describe()} :: {'; '.join(route.reasons)}")

        backend = self.pool.for_route(route)
        completion = backend.complete(
            messages, tools=tools, enable_thinking=route.enable_thinking
        )

        key = completion.model_key or route.model_key
        self.usage = self.usage + completion.usage
        self.usage_by_model[key] = self.usage_by_model.get(key, Usage()) + completion.usage
        return completion, route

    # ---- public entry point ---------------------------------------------------

    def run(
        self, user_text: str, attachments: list[Attachment] | None = None
    ) -> AgentResult:
        attachments = attachments or []

        self._prompt = user_text
        self._has_attachments = bool(attachments)
        self._has_video = any(a.kind == "video" for a in attachments)
        self._tool_errors = 0
        self._verification_failed = False
        self._escalations = 0
        self._routes = []
        self._first_route = None

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
        result.usage_by_model = dict(self.usage_by_model)
        result.routes = list(self._routes)
        result.escalations = self._escalations

        self._record_routing_outcome(user_text, result)
        return result

    def _record_routing_outcome(self, prompt: str, result: AgentResult) -> None:
        """Turn what happened into a labelled exemplar for future routing.

        This is the part a hosted router cannot do: the label comes from
        execution on *this* repository, so over time the classifier learns which
        requests here actually need the strong model.
        """
        if self.index is None or not self.config.router.learn_from_outcomes:
            return
        if self._first_route is None or not prompt.strip():
            return

        verify_ok: bool | None = None
        if result.verification is not None and not result.verification.skipped:
            verify_ok = result.verification.passed

        label = label_from_outcome(
            routed_to_strong=self._first_route.model_key == self.pool.router.strongest,
            escalated=result.escalations > 0,
            verification_failed=verify_ok is False,
        )
        if label is None:
            return

        try:
            self.index.store.record_route_outcome(
                prompt,
                label,
                model_key=self._first_route.model_key,
                escalated=result.escalations > 0,
                verify_ok=verify_ok,
            )
        except Exception as exc:  # noqa: BLE001 - learning is never load-bearing
            log.warning("could not record routing outcome: %s", exc)

    # ---- phases ---------------------------------------------------------------

    def _draft_plan(self) -> str | None:
        """Ask for a plan without granting write tools."""
        self._emit("phase", "planning")
        read_only = [
            schema
            for schema in self.toolbox.schemas()
            if schema["function"]["name"] in PLANNING_TOOLS
        ]
        probe = self.messages + [{"role": "user", "content": PLAN_PROMPT}]

        try:
            completion, route = self._call(Role.PLAN, probe, tools=read_only)
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            log.warning("planning turn failed, continuing without a plan: %s", exc)
            self._emit("warning", f"planning skipped: {exc}")
            return None

        # The planner may call read-only tools; let it finish those before we
        # take its plan text.
        steps = 0
        while completion.wants_tools and steps < 8:
            probe.append(_assistant_message(completion))
            for call in completion.tool_calls:
                self._emit("tool_call", f"{call.name} {_short_args(call)}")
                result = self.toolbox.call(call.name, call.arguments)
                probe.append(_tool_message(call, result))
            completion, route = self._call(Role.PLAN, probe, tools=read_only, route=route)
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
        route: Route | None = None

        while steps < self.config.agent.max_steps:
            self._trim_history()

            role = Role.VISION if self._has_attachments and steps == 0 else Role.CODE
            completion, route = self._call(role, self.messages, tools=tools, route=route)
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

            errors_before = self._tool_errors
            for call in completion.tool_calls:
                self._emit("tool_call", f"{call.name} {_short_args(call)}")
                result = self.toolbox.call(call.name, call.arguments)
                if result.is_error:
                    self._tool_errors += 1
                self._emit("tool_result", _preview(result))
                self.messages.append(_tool_message(call, result))

            route = self._maybe_escalate(route, errors_before)

        self._emit("warning", f"stopped after {steps} steps (agent.max_steps)")
        return AgentResult(
            text=(
                f"I stopped after {steps} tool steps without reaching a conclusion. "
                "Raise agent.max_steps, or narrow the request into smaller pieces."
            ),
            steps=steps,
            hit_step_limit=True,
        )

    def _maybe_escalate(self, route: Route | None, errors_before: int) -> Route | None:
        """Move up a tier when tools keep failing at the current one.

        Repeated tool errors are the cheap in-loop analogue of a failed test:
        evidence from execution, not from a second model's opinion.
        """
        if route is None or self._tool_errors == errors_before:
            return route
        if self._escalations >= self.config.router.max_escalations:
            return route

        stronger = self.pool.router.escalate(route, f"{self._tool_errors} tool error(s)")
        if stronger is None:
            return route

        self._escalations += 1
        self._emit("route", f"escalating to {stronger.model_key} after tool errors")
        return stronger

    def _verify_and_repair(self, result: AgentResult) -> VerificationResult:
        """Run the project's tests; on failure, hand the output back for repair."""
        self._emit("phase", "verifying")
        verification = run_verification(
            self.toolbox, configured_command=self.config.agent.verify_command
        )
        if verification.skipped or verification.passed:
            self._emit("verify", verification.output if verification.skipped else "passed")
            return verification

        self._verification_failed = True
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
            # Repair always runs on the strongest tier: ground truth has already
            # said the cheap answer was wrong.
            repair_route = self.pool.router.route(
                Role.REPAIR, self._signals(attempt=attempt)
            )
            repair = self._act_with_route(repair_route)
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

    def _act_with_route(self, route: Route) -> AgentResult:
        """Run the acting loop pinned to a specific route."""
        self._emit("phase", f"repairing via {route.model_key}")
        tools = self.toolbox.schemas()
        steps = 0

        while steps < self.config.agent.max_steps:
            self._trim_history()
            completion, _ = self._call(Role.REPAIR, self.messages, tools=tools, route=route)
            steps += 1

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
                if result.is_error:
                    self._tool_errors += 1
                self._emit("tool_result", _preview(result))
                self.messages.append(_tool_message(call, result))

        return AgentResult(text="repair loop hit the step limit", steps=steps,
                           hit_step_limit=True)

    # ---- message construction -------------------------------------------------

    def _user_message(self, text: str, attachments: list[Attachment]) -> dict[str, Any]:
        if not attachments:
            return {"role": "user", "content": text}

        parts: list[dict[str, Any]] = []
        for attachment in attachments:
            parts.append(
                {"type": "text", "text": f"[attached {attachment.kind}: {attachment.summary}]"}
            )
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
        limit = self.config.generation.context_length - self.config.generation.max_output_tokens
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
        self.usage_by_model = {}


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
