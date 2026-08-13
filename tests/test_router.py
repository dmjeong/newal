"""Routing decisions: which tier, and whether to think."""

from __future__ import annotations

import pytest

from newal.models.roles import Role
from newal.models.router import Route, RouteSignals, Router, difficulty_score


@pytest.fixture
def router():
    return Router(
        tiers=[("light", 1), ("heavy", 3)],
        strategy="cascade",
        escalate_threshold=0.45,
        thinking_mode="adaptive",
        thinking_threshold=0.55,
    )


# ---- difficulty scoring ------------------------------------------------------


def test_role_sets_the_baseline():
    edit, _ = difficulty_score(Role.EDIT, RouteSignals())
    plan, _ = difficulty_score(Role.PLAN, RouteSignals())
    assert edit < plan


def test_verification_failure_dominates_the_score():
    signals = RouteSignals(prompt="rename the variable")
    calm, _ = difficulty_score(Role.EDIT, signals)

    failed, reasons = difficulty_score(
        Role.EDIT, RouteSignals(prompt="rename the variable", verification_failed=True)
    )
    assert failed > calm
    assert any("verification failed" in r for r in reasons)


def test_investigative_wording_raises_the_score():
    easy, _ = difficulty_score(Role.CODE, RouteSignals(prompt="add a docstring"))
    hard, _ = difficulty_score(
        Role.CODE, RouteSignals(prompt="why does this deadlock under load?")
    )
    assert hard > easy


def test_korean_investigative_wording_is_recognised():
    """The assistant answers in the user's language; signals must work there too."""
    neutral, _ = difficulty_score(Role.CODE, RouteSignals(prompt="이 파일 좀 봐줘"))
    hard, _ = difficulty_score(Role.CODE, RouteSignals(prompt="이 버그 원인이 뭐야?"))
    assert hard > neutral


def test_korean_mechanical_wording_lowers_the_score():
    neutral, _ = difficulty_score(Role.CODE, RouteSignals(prompt="이 파일 좀 봐줘"))
    easy, _ = difficulty_score(Role.CODE, RouteSignals(prompt="변수 이름 바꿔줘"))
    assert easy < neutral


def test_video_counts_for_more_than_a_still_image():
    image, _ = difficulty_score(Role.VISION, RouteSignals(has_attachments=True))
    video, _ = difficulty_score(
        Role.VISION, RouteSignals(has_attachments=True, has_video=True)
    )
    assert video > image


def test_score_stays_within_bounds_under_every_signal():
    worst = RouteSignals(
        prompt="why " * 500,
        has_attachments=True,
        has_video=True,
        attempt=9,
        tool_errors=20,
        verification_failed=True,
        files_touched=40,
    )
    score, _ = difficulty_score(Role.REPAIR, worst)
    assert 0.0 <= score <= 1.0


def test_reasons_explain_every_adjustment():
    _, reasons = difficulty_score(
        Role.CODE, RouteSignals(prompt="debug this race condition", tool_errors=2)
    )
    assert any("role=" in r for r in reasons)
    assert any("tool error" in r for r in reasons)


# ---- tier selection ----------------------------------------------------------


def test_mechanical_edit_goes_to_the_cheap_tier(router):
    route = router.route(Role.EDIT, RouteSignals(prompt="fix the typo in the comment"))
    assert route.model_key == "light"


def test_planning_never_uses_the_cheap_tier(router):
    route = router.route(Role.PLAN, RouteSignals(prompt="fix the typo"))
    assert route.model_key == "heavy"
    assert any("never runs on the cheap tier" in r for r in route.reasons)


def test_repair_never_uses_the_cheap_tier(router):
    route = router.route(Role.REPAIR, RouteSignals(prompt="fix the typo"))
    assert route.model_key == "heavy"


def test_hard_request_goes_straight_to_the_strong_tier(router):
    route = router.route(
        Role.CODE, RouteSignals(prompt="diagnose the intermittent deadlock in the scheduler")
    )
    assert route.model_key == "heavy"


def test_single_strategy_always_uses_the_strongest():
    router = Router(tiers=[("light", 1), ("heavy", 3)], strategy="single")
    route = router.route(Role.EDIT, RouteSignals(prompt="fix the typo"))
    assert route.model_key == "heavy"


def test_one_model_pool_routes_everything_to_it():
    router = Router(tiers=[("only", 1)])
    assert router.route(Role.EDIT).model_key == "only"
    assert router.route(Role.PLAN).model_key == "only"


def test_router_requires_at_least_one_model():
    with pytest.raises(ValueError):
        Router(tiers=[])


# ---- thinking mode -----------------------------------------------------------


def test_adaptive_thinking_is_off_for_mechanical_work(router):
    route = router.route(Role.EDIT, RouteSignals(prompt="rename this variable"))
    assert route.enable_thinking is False


def test_adaptive_thinking_is_on_for_repair(router):
    route = router.route(Role.REPAIR, RouteSignals(verification_failed=True))
    assert route.enable_thinking is True


def test_thinking_mode_never_overrides_everything():
    router = Router(tiers=[("heavy", 3)], thinking_mode="never")
    route = router.route(Role.REPAIR, RouteSignals(verification_failed=True))
    assert route.enable_thinking is False


def test_thinking_mode_always_overrides_everything():
    router = Router(tiers=[("heavy", 3)], thinking_mode="always")
    route = router.route(Role.EDIT, RouteSignals(prompt="fix typo"))
    assert route.enable_thinking is True


# ---- escalation --------------------------------------------------------------


def test_escalation_moves_up_one_tier(router):
    start = router.route(Role.EDIT, RouteSignals(prompt="fix the typo"))
    assert start.model_key == "light"

    stronger = router.escalate(start, "tool errors")
    assert stronger is not None
    assert stronger.model_key == "heavy"
    assert stronger.tier > start.tier


def test_escalation_forces_thinking_on(router):
    start = router.route(Role.EDIT, RouteSignals(prompt="fix the typo"))
    stronger = router.escalate(start, "tool errors")
    assert stronger.enable_thinking is True


def test_escalation_records_why(router):
    start = router.route(Role.EDIT, RouteSignals(prompt="fix the typo"))
    stronger = router.escalate(start, "2 tool errors")
    assert any("2 tool errors" in r for r in stronger.reasons)


def test_escalation_from_the_top_tier_returns_none(router):
    top = router.route(Role.REPAIR, RouteSignals())
    assert router.escalate(top, "still failing") is None


def test_route_describe_is_human_readable(router):
    route = router.route(Role.PLAN, RouteSignals(prompt="redesign the cache layer"))
    described = route.describe()
    assert "heavy" in described
    assert "plan" in described


def test_middle_tier_escalates_one_step_at_a_time():
    router = Router(tiers=[("small", 1), ("medium", 2), ("large", 3)])
    start = Route(model_key="small", role=Role.CODE, enable_thinking=False, tier=1, score=0.1)

    first = router.escalate(start, "errors")
    assert first.model_key == "medium"
    second = router.escalate(first, "still failing")
    assert second.model_key == "large"
    assert router.escalate(second, "give up") is None
