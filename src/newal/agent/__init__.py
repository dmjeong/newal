"""The agent: planning, tool use, and verification."""

from __future__ import annotations

from .loop import Agent, AgentResult, EventCallback, EventKind
from .prompts import build_system_prompt, load_project_doc
from .tools import PermissionDenied, Toolbox, ToolError, ToolResult, describe_command
from .verifier import VerificationResult, detect_verify_command, run_verification

__all__ = [
    "Agent",
    "AgentResult",
    "EventCallback",
    "EventKind",
    "PermissionDenied",
    "ToolError",
    "ToolResult",
    "Toolbox",
    "VerificationResult",
    "build_system_prompt",
    "describe_command",
    "detect_verify_command",
    "load_project_doc",
    "run_verification",
]
