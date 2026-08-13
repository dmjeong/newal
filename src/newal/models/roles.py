"""Roles a model can be asked to fill.

Roles are *jobs*, not sizes. The distinction matters: the co-failure work on
ensembling (arXiv:2606.27288) finds that gains come from members failing on
different inputs, and that stacking same-family models -- Self-MoA -- buys
little because their errors are correlated. Assigning genuinely different
tasks is what keeps pool members decorrelated.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    #: Cheap classification and routing decisions.
    TRIAGE = "triage"
    #: Multi-step reasoning before touching anything.
    PLAN = "plan"
    #: Writing new code or non-trivial edits.
    CODE = "code"
    #: Mechanical edits: renames, formatting, single-line changes.
    EDIT = "edit"
    #: Fixing a change after verification failed. Always the strongest tier.
    REPAIR = "repair"
    #: Compressing tool output or history.
    SUMMARIZE = "summarize"
    #: Turns carrying images or video frames.
    VISION = "vision"

    def __str__(self) -> str:
        return self.value


#: How much deliberation each role warrants before other signals are weighed.
#: Repair sits highest: it runs only after ground truth said the work was wrong.
ROLE_DIFFICULTY: dict[Role, float] = {
    Role.TRIAGE: 0.0,
    Role.SUMMARIZE: 0.05,
    Role.EDIT: 0.2,
    Role.VISION: 0.5,
    Role.CODE: 0.55,
    Role.PLAN: 0.75,
    Role.REPAIR: 0.9,
}

#: Roles that must never be served by the cheapest tier, regardless of routing.
#: A wrong plan or a wrong repair costs more than the tokens it saves.
HEAVY_ONLY_ROLES: frozenset[Role] = frozenset({Role.PLAN, Role.REPAIR})
