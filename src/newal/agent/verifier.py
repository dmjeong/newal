"""Verification: figure out how to check the work, and check it.

This is the piece that closes the capability gap with a stronger model. A local
model's first attempt is less reliable, but "run the tests, read the failure,
fix it, repeat" converts a weaker one-shot into a stronger final answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .tools import Toolbox, ToolResult

log = logging.getLogger(__name__)

# Ordered by specificity: a marker file implies its command.
AUTO_VERIFY_RULES: list[tuple[str, str]] = [
    ("pytest.ini", "python -m pytest -q"),
    ("tox.ini", "python -m pytest -q"),
    ("pyproject.toml", "python -m pytest -q"),
    ("setup.cfg", "python -m pytest -q"),
    ("package.json", "npm test --silent"),
    ("Cargo.toml", "cargo test --quiet"),
    ("go.mod", "go test ./..."),
    ("Makefile", "make test"),
]


@dataclass
class VerificationResult:
    command: str | None
    passed: bool
    output: str

    @property
    def skipped(self) -> bool:
        return self.command is None


def detect_verify_command(root: Path) -> str | None:
    """Infer a test command from the project's marker files.

    ``pyproject.toml`` only implies pytest when there is somewhere for tests to
    live -- otherwise every Python project would "verify" by erroring out with
    no tests collected.
    """
    for marker, command in AUTO_VERIFY_RULES:
        if not (root / marker).is_file():
            continue
        if command.startswith("python -m pytest"):
            has_tests = any(
                (root / d).is_dir() for d in ("tests", "test")
            ) or any(root.glob("test_*.py"))
            if not has_tests:
                continue
        return command
    return None


def run_verification(
    toolbox: Toolbox,
    *,
    configured_command: str | None,
    timeout_s: int = 600,
) -> VerificationResult:
    """Run the verification command, if one can be determined."""
    command = configured_command or detect_verify_command(toolbox.workspace.root)
    if not command:
        log.info("no verification command could be determined; skipping")
        return VerificationResult(
            command=None,
            passed=True,
            output="No test command detected. Set agent.verify_command to enable "
                   "automatic verification.",
        )

    log.info("verifying with: %s", command)
    # Bypass the interactive approval prompt: this command comes from config or
    # from the project's own layout, not from the model.
    result: ToolResult = toolbox._execute(  # noqa: SLF001 - deliberate internal use
        command, shell=True, timeout_s=timeout_s
    )
    return VerificationResult(
        command=command,
        passed=not result.is_error,
        output=result.content,
    )
