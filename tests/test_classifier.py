"""Learned query classification and outcome-driven learning."""

from __future__ import annotations

import pytest

from newal.backends.base import Completion
from newal.models.classifier import (
    CHEAP,
    SEED_EXEMPLARS,
    STRONG,
    LayeredClassifier,
    LLMClassifier,
    SemanticClassifier,
    Verdict,
    label_from_outcome,
)
from newal.models.roles import Role
from newal.models.router import Router, RouteSignals


class WordEmbedder:
    """Bag-of-words embedding over a fixed vocabulary.

    Crude, but it puts texts that share words near each other, which is all the
    kNN vote needs in order to be tested.
    """

    VOCAB = [
        "rename", "typo", "comment", "format", "variable", "docstring",
        "why", "deadlock", "race", "leak", "refactor", "root", "cause",
        "이름", "오타", "주석", "원인", "누수", "리팩터링", "설계",
    ]

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embed(self, texts, batch_size: int = 32):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding server is down")
        vectors = []
        for text in texts:
            lowered = text.lower()
            vec = [float(word in lowered) for word in self.VOCAB]
            if not any(vec):
                vec[0] = 0.01  # avoid an all-zero row
            vectors.append(vec)
        return vectors


class ScriptedBackend:
    def __init__(self, reply: str, *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0
        self.last_kwargs: dict = {}

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.fail:
            raise RuntimeError("model is down")
        return Completion(text=self.reply)


# ---- semantic classifier -----------------------------------------------------


def test_semantic_classifier_matches_mechanical_requests():
    clf = SemanticClassifier(WordEmbedder())
    verdict = clf.classify("rename this variable")
    assert verdict is not None
    assert verdict.label == CHEAP


def test_semantic_classifier_matches_investigative_requests():
    clf = SemanticClassifier(WordEmbedder())
    verdict = clf.classify("find the root cause of the deadlock")
    assert verdict is not None
    assert verdict.label == STRONG


def test_semantic_classifier_generalises_beyond_the_heuristic_patterns():
    """A phrasing with no regex marker should still classify correctly."""
    clf = SemanticClassifier(WordEmbedder())
    verdict = clf.classify("there is a race somewhere, track it down")
    assert verdict is not None
    assert verdict.label == STRONG


def test_semantic_confidence_is_the_margin_between_labels():
    clf = SemanticClassifier(WordEmbedder())
    clear = clf.classify("rename the variable")
    assert clear is not None
    assert 0.0 <= clear.confidence <= 1.0


def test_semantic_classifier_reports_its_exemplar_count():
    clf = SemanticClassifier(WordEmbedder())
    assert clf.size == len(SEED_EXEMPLARS)


def test_semantic_classifier_abstains_when_nothing_resembles_the_query():
    """No overlap with any exemplar must abstain, not guess at zero confidence."""
    clf = SemanticClassifier(WordEmbedder(), exemplars=[("typo in comment", CHEAP)])
    assert clf.classify("deadlock leak race") is None


def test_semantic_classifier_survives_a_dead_embedder():
    clf = SemanticClassifier(WordEmbedder(fail=True))
    assert clf.size == 0
    assert clf.classify("anything") is None


def test_semantic_classifier_ignores_empty_prompts():
    clf = SemanticClassifier(WordEmbedder())
    assert clf.classify("   ") is None


def test_learned_exemplars_override_the_seed_set():
    """A user-specific exemplar should be able to pull a query the other way."""
    embedder = WordEmbedder()
    clf = SemanticClassifier(
        embedder,
        exemplars=[("rename this variable", STRONG)],  # deliberately contrary
        neighbours=1,
    )
    verdict = clf.classify("rename this variable")
    assert verdict is not None
    assert verdict.label == STRONG


def test_refitting_replaces_the_exemplar_set():
    clf = SemanticClassifier(WordEmbedder(), exemplars=[("typo", CHEAP)])
    assert clf.size == 1
    clf.fit([("typo", CHEAP), ("deadlock", STRONG)])
    assert clf.size == 2


# ---- llm classifier ----------------------------------------------------------


def test_llm_classifier_parses_complex():
    backend = ScriptedBackend("COMPLEX")
    verdict = LLMClassifier(backend).classify("why is this flaky?")
    assert verdict is not None
    assert verdict.label == STRONG
    assert verdict.source == "llm"


def test_llm_classifier_parses_simple_in_a_sentence():
    verdict = LLMClassifier(ScriptedBackend("This looks SIMPLE to me.")).classify("x")
    assert verdict is not None
    assert verdict.label == CHEAP


def test_llm_classifier_never_burns_tokens_on_thinking():
    backend = ScriptedBackend("SIMPLE")
    LLMClassifier(backend).classify("rename x")
    assert backend.last_kwargs["enable_thinking"] is False
    assert backend.last_kwargs["max_tokens"] <= 16


def test_llm_classifier_returns_none_on_unparseable_output():
    assert LLMClassifier(ScriptedBackend("I'm not sure, maybe?")).classify("x") is None


def test_llm_classifier_survives_a_dead_backend():
    assert LLMClassifier(ScriptedBackend("COMPLEX", fail=True)).classify("x") is None


# ---- layering ----------------------------------------------------------------


class StubClassifier:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def classify(self, prompt):
        self.calls += 1
        return self.verdict


def test_layered_stops_at_the_first_confident_verdict():
    first = StubClassifier(Verdict(STRONG, 0.9, "semantic"))
    second = StubClassifier(Verdict(CHEAP, 1.0, "llm"))

    verdict = LayeredClassifier([first, second]).classify("x")
    assert verdict.source == "semantic"
    assert second.calls == 0  # the expensive layer was never reached


def test_layered_falls_through_when_the_first_is_unsure():
    first = StubClassifier(Verdict(STRONG, 0.05, "semantic"))
    second = StubClassifier(Verdict(CHEAP, 1.0, "llm"))

    verdict = LayeredClassifier([first, second], min_confidence=0.3).classify("x")
    assert verdict.source == "llm"
    assert second.calls == 1


def test_layered_falls_through_when_the_first_abstains():
    first = StubClassifier(None)
    second = StubClassifier(Verdict(STRONG, 1.0, "llm"))
    assert LayeredClassifier([first, second]).classify("x").source == "llm"


def test_layered_returns_none_when_everyone_abstains():
    assert LayeredClassifier([StubClassifier(None)]).classify("x") is None


# ---- router integration ------------------------------------------------------


@pytest.fixture
def router_with(request):
    def build(classifier, **kwargs):
        kwargs.setdefault("uncertainty_band", 0.12)
        return Router(
            tiers=[("light", 1), ("heavy", 3)],
            escalate_threshold=0.45,
            classifier=classifier,
            **kwargs,
        )

    return build


def test_classifier_is_skipped_when_the_heuristic_is_confident(router_with):
    stub = StubClassifier(Verdict(STRONG, 1.0, "llm"))
    router = router_with(stub)

    # Strongly mechanical wording scores 0.35, well outside the 0.33-0.57 band...
    router.route(Role.EDIT, RouteSignals(prompt="fix the typo"))
    assert stub.calls == 0


def test_classifier_breaks_the_tie_inside_the_uncertainty_band(router_with):
    stub = StubClassifier(Verdict(STRONG, 1.0, "semantic"))
    router = router_with(stub)

    # Role.VISION scores 0.50, inside the band around 0.45.
    route = router.route(Role.VISION, RouteSignals(prompt="what is happening here"))
    assert stub.calls == 1
    assert route.model_key == "heavy"


def test_classifier_can_route_down_as_well_as_up(router_with):
    stub = StubClassifier(Verdict(CHEAP, 1.0, "semantic"))
    router = router_with(stub)

    route = router.route(Role.VISION, RouteSignals(prompt="what is happening here"))
    assert route.model_key == "light"


def test_router_keeps_the_heuristic_when_the_classifier_abstains(router_with):
    router = router_with(StubClassifier(None))
    route = router.route(Role.VISION, RouteSignals(prompt="what is happening"))
    # 0.50 >= 0.45, so the heuristic sends it strong on its own.
    assert route.model_key == "heavy"
    assert any("no confident opinion" in r for r in route.reasons)


def test_router_survives_a_classifier_that_raises(router_with):
    class Exploding:
        def classify(self, prompt):
            raise RuntimeError("boom")

    route = router_with(Exploding()).route(Role.VISION, RouteSignals(prompt="hm"))
    assert route.model_key in {"light", "heavy"}
    assert any("classifier error" in r for r in route.reasons)


def test_zero_band_disables_the_classifier_entirely(router_with):
    stub = StubClassifier(Verdict(STRONG, 1.0, "llm"))
    router = router_with(stub, uncertainty_band=0.0)
    router.route(Role.VISION, RouteSignals(prompt="what is happening"))
    assert stub.calls == 0


def test_heavy_only_roles_skip_the_classifier(router_with):
    stub = StubClassifier(Verdict(CHEAP, 1.0, "llm"))
    router = router_with(stub)
    route = router.route(Role.REPAIR, RouteSignals(prompt="fix it"))
    assert route.model_key == "heavy"
    assert stub.calls == 0


def test_routing_reason_records_the_classifier_verdict(router_with):
    router = router_with(StubClassifier(Verdict(STRONG, 0.8, "semantic")))
    route = router.route(Role.VISION, RouteSignals(prompt="what is happening"))
    assert any("semantic classifier says strong" in r for r in route.reasons)


# ---- outcome labelling -------------------------------------------------------


def test_cheap_route_that_escalated_is_labelled_strong():
    assert label_from_outcome(
        routed_to_strong=False, escalated=True, verification_failed=False
    ) == STRONG


def test_cheap_route_that_failed_verification_is_labelled_strong():
    assert label_from_outcome(
        routed_to_strong=False, escalated=False, verification_failed=True
    ) == STRONG


def test_clean_cheap_route_confirms_cheap_was_enough():
    assert label_from_outcome(
        routed_to_strong=False, escalated=False, verification_failed=False
    ) == CHEAP


def test_clean_strong_route_teaches_nothing():
    """We never learn whether the cheap model would also have managed."""
    assert label_from_outcome(
        routed_to_strong=True, escalated=False, verification_failed=False
    ) is None


def test_strong_route_that_still_failed_is_labelled_strong():
    assert label_from_outcome(
        routed_to_strong=True, escalated=False, verification_failed=True
    ) == STRONG
