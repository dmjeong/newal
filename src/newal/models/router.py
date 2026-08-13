"""Query routing: which model, and how hard should it think.

Two decisions, both made before every model call:

**Which tier.** A cascade (arXiv:2606.27457, arXiv:2605.18796): serve cheaply by
default, escalate on evidence. Note what this buys and what it does not -- on a
single machine the cascade is a *latency and VRAM* optimisation, not an accuracy
one. Accuracy comes from escalating on real failure signals, not from asking
more models and voting.

**How hard to think.** Qwen's thinking mode is expensive and mostly wasted on
mechanical work. Following the inhibitory-deliberation idea (arXiv:2606.06745),
a switch score decides per call rather than once in config.

The scoring is deliberately a transparent heuristic rather than a learned
router: it is inspectable, deterministic, adds no latency, and needs no
training data the user does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .roles import HEAVY_ONLY_ROLES, ROLE_DIFFICULTY, Role

# Words that mark a request as investigative rather than mechanical. Korean
# terms are included because the assistant answers in the user's language and
# these signals must work there too.
_HARD_MARKERS = re.compile(
    r"\b(why|debug|diagnos\w*|root\s*cause|race|deadlock|leak|flaky|intermittent|"
    r"regression|refactor|architect\w*|design|migrat\w*|optimi[sz]e|performance|"
    r"security|concurren\w*|thread\s*safe|corrupt\w*)\b"
    r"|왜|원인|디버\w*|버그|경합|교착|누수|리팩터\w*|리팩토\w*|아키텍\w*|설계|"
    r"마이그레이션|최적화|성능|보안|동시성|race\s*condition",
    re.IGNORECASE,
)

# Words that mark a request as mechanical: cheap tier, no thinking.
# Korean verb stems change under conjugation (바꾸다 -> 바꿔, 고치다 -> 고쳐), so
# each stem lists the forms that actually appear in a request rather than the
# dictionary form alone.
_EASY_MARKERS = re.compile(
    r"\b(rename|format|typo|comment|docstring|import|lint|indent|"
    r"add\s+a?\s*log|bump|spelling)\b"
    r"|(이름|변수명|함수명)\s*(바꾸|바꿔|변경|고치|고쳐)"
    r"|리네임|포맷|오타|주석|들여쓰기|로그\s*추가|정렬해",
    re.IGNORECASE,
)

_LONG_PROMPT_CHARS = 400
_VERY_LONG_PROMPT_CHARS = 1200


@dataclass
class RouteSignals:
    """Everything the router knows about the call it is about to make."""

    prompt: str = ""
    has_attachments: bool = False
    has_video: bool = False
    #: Escalation round. 0 is the first attempt at this step.
    attempt: int = 0
    #: Tool calls that returned an error since the last successful step.
    tool_errors: int = 0
    #: Verification has failed at least once for the current change.
    verification_failed: bool = False
    #: Distinct files already modified in this turn.
    files_touched: int = 0


@dataclass
class Route:
    """The routing decision for one model call."""

    model_key: str
    role: Role
    enable_thinking: bool
    tier: int
    score: float
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        thinking = "thinking" if self.enable_thinking else "direct"
        return f"{self.model_key} ({self.role}, {thinking}, score={self.score:.2f})"


def difficulty_score(role: Role, signals: RouteSignals) -> tuple[float, list[str]]:
    """Score how much deliberation this call warrants, in [0, 1].

    Returns the score and the human-readable reasons that produced it, so a
    routing decision can always be explained.
    """
    score = ROLE_DIFFICULTY.get(role, 0.5)
    reasons = [f"role={role} ({score:.2f})"]

    def bump(delta: float, why: str) -> None:
        nonlocal score
        score = min(1.0, max(0.0, score + delta))
        reasons.append(f"{why} ({delta:+.2f})")

    # Hard evidence first: something already went wrong. These dominate,
    # because they are observations rather than guesses about the prompt.
    if signals.verification_failed:
        bump(0.40, "verification failed")
    if signals.attempt > 0:
        bump(min(0.30, 0.15 * signals.attempt), f"escalation attempt {signals.attempt}")
    if signals.tool_errors:
        bump(min(0.20, 0.07 * signals.tool_errors), f"{signals.tool_errors} tool error(s)")

    # Softer prompt-shape signals.
    prompt = signals.prompt or ""
    if _HARD_MARKERS.search(prompt):
        bump(0.20, "investigative wording")
    elif _EASY_MARKERS.search(prompt):
        bump(-0.20, "mechanical wording")

    if len(prompt) > _VERY_LONG_PROMPT_CHARS:
        bump(0.15, "very long request")
    elif len(prompt) > _LONG_PROMPT_CHARS:
        bump(0.08, "long request")

    if signals.has_video:
        bump(0.15, "video attached")
    elif signals.has_attachments:
        bump(0.10, "media attached")

    if signals.files_touched >= 3:
        bump(0.12, f"{signals.files_touched} files in play")

    return score, reasons


class Router:
    """Chooses a pool member and a thinking mode for each call."""

    def __init__(
        self,
        *,
        tiers: list[tuple[str, int]],
        strategy: str = "cascade",
        escalate_threshold: float = 0.45,
        thinking_mode: str = "adaptive",
        thinking_threshold: float = 0.55,
    ) -> None:
        """
        ``tiers`` is ``[(model_key, tier)]`` for every generation-capable member,
        which the router sorts from cheapest to strongest.
        """
        if not tiers:
            raise ValueError("router needs at least one generation model")
        self._tiers = sorted(tiers, key=lambda item: item[1])
        self.strategy = strategy
        self.escalate_threshold = escalate_threshold
        self.thinking_mode = thinking_mode
        self.thinking_threshold = thinking_threshold

    @property
    def strongest(self) -> str:
        return self._tiers[-1][0]

    @property
    def cheapest(self) -> str:
        return self._tiers[0][0]

    def _tier_of(self, model_key: str) -> int:
        for key, tier in self._tiers:
            if key == model_key:
                return tier
        return self._tiers[-1][1]

    def _wants_thinking(self, score: float) -> bool:
        if self.thinking_mode == "always":
            return True
        if self.thinking_mode == "never":
            return False
        return score >= self.thinking_threshold

    def route(self, role: Role, signals: RouteSignals | None = None) -> Route:
        """Pick the model and thinking mode for one call."""
        signals = signals or RouteSignals()
        score, reasons = difficulty_score(role, signals)

        if self.strategy == "single" or len(self._tiers) == 1:
            model_key = self.strongest
            reasons.append("single-model strategy" if self.strategy == "single"
                           else "only one model in pool")
        elif role in HEAVY_ONLY_ROLES:
            model_key = self.strongest
            reasons.append(f"{role} never runs on the cheap tier")
        elif score >= self.escalate_threshold:
            model_key = self.strongest
            reasons.append(f"score >= escalate_threshold ({self.escalate_threshold:.2f})")
        else:
            model_key = self.cheapest
            reasons.append("below escalation threshold")

        return Route(
            model_key=model_key,
            role=role,
            enable_thinking=self._wants_thinking(score),
            tier=self._tier_of(model_key),
            score=score,
            reasons=reasons,
        )

    def escalate(self, current: Route, why: str) -> Route | None:
        """Move one step up the cascade, or ``None`` if already at the top.

        Escalation is driven by observed failure -- a tool error, a failed test
        run -- not by disagreement between models. Execution results are the one
        signal in the system that is genuinely uncorrelated with model error.
        """
        higher = [(key, tier) for key, tier in self._tiers if tier > current.tier]
        if not higher:
            return None

        model_key, tier = higher[0]
        score = min(1.0, current.score + 0.25)
        return Route(
            model_key=model_key,
            role=current.role,
            # An escalated call always thinks: the cheap path already missed.
            enable_thinking=self.thinking_mode != "never",
            tier=tier,
            score=score,
            reasons=current.reasons + [f"escalated: {why}"],
        )
