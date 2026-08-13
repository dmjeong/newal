"""Turning ordinary use into fine-tuning data.

The bottleneck in fine-tuning is rarely the framework -- it is having labelled,
domain-specific data. newal produces that as a by-product: every turn is a real
request against a real repository, and the verify-and-repair loop attaches an
objective label to it by running the project's tests.

Two things fall out for free:

* **SFT material** -- turns the test suite agreed were correct.
* **DPO pairs** -- the edit that failed the tests versus the one that passed,
  with no human annotator and no LLM judge in the labelling path.

Capture is local-only and can be switched off with ``training.enabled``.
"""

from __future__ import annotations

from .export import FORMATS, ExportStats, Format, export, iter_dpo, iter_routing, iter_sft

__all__ = [
    "FORMATS",
    "ExportStats",
    "Format",
    "export",
    "iter_dpo",
    "iter_routing",
    "iter_sft",
]
