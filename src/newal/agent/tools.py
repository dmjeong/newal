"""Tools the model can call, with a sandbox that keeps them inside the workspace."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ToolsConfig
from ..memory import RepoIndex, format_context

log = logging.getLogger(__name__)

# Commands that are destructive enough that we refuse them even under
# shell_policy: allow. The approval callback covers everything else.
BLOCKED_SHELL_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]",
    r"\bmkfs(\.|\s)",
    r"\bdd\s+.*\bof=/dev/",
    r":\(\)\s*\{.*\};:",           # fork bomb
    r"\bshutdown\b|\breboot\b",
    r">\s*/dev/(sd|nvme|hd)",
    r"\bgit\s+push\b.*--force",
    r"\bchmod\s+-R\s+777\s+/",
]


class ToolError(Exception):
    """Recoverable failure; the message is handed back to the model verbatim."""


class PermissionDenied(ToolError):
    pass


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


ApprovalCallback = Callable[[str, str], bool]
"""Called as ``approve(tool_name, detail)``; return True to permit."""


class Workspace:
    """Path resolution and guardrails for every filesystem-touching tool."""

    def __init__(self, config: ToolsConfig) -> None:
        self.config = config
        self.root = Path(config.workspace_root).expanduser().resolve()
        if not self.root.is_dir():
            raise ToolError(f"workspace_root is not a directory: {self.root}")

    def resolve(self, relative: str, *, must_exist: bool = False) -> Path:
        """Resolve a user/model-supplied path, refusing escapes and denied paths."""
        if not relative or not relative.strip():
            raise ToolError("path must not be empty")

        candidate = Path(relative).expanduser()
        target = candidate if candidate.is_absolute() else self.root / candidate

        # resolve() before the containment check so symlinks and ".." cannot
        # be used to step outside the workspace.
        resolved = target.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PermissionDenied(f"path escapes the workspace: {relative}")

        rel_posix = resolved.relative_to(self.root).as_posix() if resolved != self.root else "."
        for denied in self.config.denied_paths:
            if rel_posix == denied or rel_posix.startswith(denied.rstrip("/") + "/"):
                raise PermissionDenied(f"path is on the deny list: {rel_posix}")
            if fnmatch.fnmatch(rel_posix, denied):
                raise PermissionDenied(f"path is on the deny list: {rel_posix}")

        if must_exist and not resolved.exists():
            raise ToolError(f"no such file or directory: {rel_posix}")
        return resolved

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)


class Toolbox:
    """Dispatches tool calls and exposes their JSON schemas."""

    def __init__(
        self,
        config: ToolsConfig,
        *,
        index: RepoIndex | None = None,
        approve: ApprovalCallback | None = None,
    ) -> None:
        self.config = config
        self.workspace = Workspace(config)
        self.index = index
        self._approve = approve or (lambda _name, _detail: False)
        self.files_written: set[str] = set()

    # ---- dispatch -------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if "__parse_error__" in arguments:
            return ToolResult(
                f"Your tool arguments were not valid JSON: {arguments['__parse_error__']!r}. "
                "Re-issue the call with a well-formed JSON object.",
                is_error=True,
            )

        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            known = ", ".join(sorted(s["function"]["name"] for s in self.schemas()))
            return ToolResult(f"unknown tool {name!r}. Available: {known}", is_error=True)

        try:
            return handler(**arguments)
        except TypeError as exc:
            return ToolResult(f"bad arguments for {name}: {exc}", is_error=True)
        except ToolError as exc:
            return ToolResult(str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 - never kill the loop on a tool bug
            log.exception("tool %s crashed", name)
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)

    # ---- individual tools -----------------------------------------------------

    def _tool_read_file(
        self, path: str, start_line: int = 1, end_line: int | None = None
    ) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        if target.is_dir():
            raise ToolError(f"{path} is a directory; use list_dir")

        size = target.stat().st_size
        if size > self.config.max_file_read_bytes and end_line is None:
            raise ToolError(
                f"{path} is {size} bytes, over the {self.config.max_file_read_bytes} limit. "
                "Pass start_line/end_line to read a slice."
            )
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{path} is not UTF-8 text") from exc

        lines = text.splitlines()
        first = max(1, start_line)
        last = min(len(lines), end_line or len(lines))
        if first > len(lines):
            raise ToolError(f"start_line {first} is past the end ({len(lines)} lines)")

        numbered = "\n".join(
            f"{n:>6}\t{line}" for n, line in enumerate(lines[first - 1 : last], start=first)
        )
        return ToolResult(numbered or "(empty)")

    def _tool_write_file(self, path: str, content: str) -> ToolResult:
        target = self.workspace.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        rel = self.workspace.relative(target)
        self.files_written.add(rel)
        verb = "overwrote" if existed else "created"
        return ToolResult(f"{verb} {rel} ({len(content.splitlines())} lines)")

    def _tool_edit_file(
        self, path: str, old_text: str, new_text: str, replace_all: bool = False
    ) -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        text = target.read_text(encoding="utf-8")

        occurrences = text.count(old_text)
        if occurrences == 0:
            raise ToolError(
                f"old_text not found in {path}. Read the file again -- it must match "
                "byte for byte, including indentation."
            )
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_text appears {occurrences} times in {path}. Include more "
                "surrounding context to make it unique, or pass replace_all=true."
            )

        target.write_text(
            text.replace(old_text, new_text) if replace_all
            else text.replace(old_text, new_text, 1),
            encoding="utf-8",
        )
        rel = self.workspace.relative(target)
        self.files_written.add(rel)
        count = occurrences if replace_all else 1
        return ToolResult(f"edited {rel} ({count} replacement(s))")

    def _tool_list_dir(self, path: str = ".") -> ToolResult:
        target = self.workspace.resolve(path, must_exist=True)
        if not target.is_dir():
            raise ToolError(f"{path} is not a directory")

        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name)):
            try:
                self.workspace.resolve(str(child))
            except PermissionDenied:
                continue  # hide denied paths entirely
            if child.is_dir():
                entries.append(f"{child.name}/")
            else:
                entries.append(f"{child.name}  ({child.stat().st_size} B)")
        return ToolResult("\n".join(entries) or "(empty directory)")

    def _tool_grep(
        self, pattern: str, path: str = ".", glob: str = "*", max_results: int = 60
    ) -> ToolResult:
        root = self.workspace.resolve(path, must_exist=True)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc

        candidates = [root] if root.is_file() else sorted(root.rglob(glob))
        matches: list[str] = []
        for candidate in candidates:
            if len(matches) >= max_results:
                matches.append("... (truncated; narrow the pattern or glob)")
                break
            if not candidate.is_file():
                continue
            try:
                self.workspace.resolve(str(candidate))
                text = candidate.read_text(encoding="utf-8")
            except (PermissionDenied, UnicodeDecodeError, OSError):
                continue
            rel = self.workspace.relative(candidate)
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{rel}:{number}: {line.strip()[:200]}")
                    if len(matches) >= max_results:
                        break
        return ToolResult("\n".join(matches) or f"no matches for {pattern!r}")

    def _tool_search_memory(self, query: str, top_k: int = 8) -> ToolResult:
        if self.index is None:
            raise ToolError("memory is disabled (memory.enabled = false)")
        results = self.index.search(query, top_k=top_k)
        if not results:
            return ToolResult(f"nothing indexed matches {query!r}")
        return ToolResult(format_context(results))

    def _tool_remember(self, topic: str, content: str) -> ToolResult:
        if self.index is None:
            raise ToolError("memory is disabled (memory.enabled = false)")
        note_id = self.index.store.add_note(topic, content)
        return ToolResult(f"stored note #{note_id} under {topic!r}")

    def _tool_run_shell(self, command: str, timeout_s: int | None = None) -> ToolResult:
        self._authorize_execution("run_shell", command)
        return self._execute(
            command,
            shell=True,
            timeout_s=timeout_s or self.config.shell_timeout_s,
        )

    def _tool_run_python(self, code: str, timeout_s: int | None = None) -> ToolResult:
        self._authorize_execution("run_python", code)
        return self._execute(
            [sys.executable, "-c", code],
            shell=False,
            timeout_s=timeout_s or self.config.shell_timeout_s,
        )

    # ---- execution plumbing ---------------------------------------------------

    def _authorize_execution(self, tool_name: str, payload: str) -> None:
        if self.config.shell_policy == "deny":
            raise PermissionDenied(
                f"{tool_name} is disabled (tools.shell_policy = deny). "
                "Describe the command for the user to run instead."
            )
        for pattern in BLOCKED_SHELL_PATTERNS:
            if re.search(pattern, payload):
                raise PermissionDenied(
                    f"refused: the command matches a destructive pattern ({pattern}). "
                    "If you genuinely need this, ask the user to run it themselves."
                )
        if self.config.shell_policy == "ask" and not self._approve(tool_name, payload):
            raise PermissionDenied("the user declined to run this command")

    def _execute(
        self, command: str | list[str], *, shell: bool, timeout_s: int
    ) -> ToolResult:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            completed = subprocess.run(
                command,
                shell=shell,
                cwd=self.workspace.root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"timed out after {timeout_s}s") from None
        except OSError as exc:
            raise ToolError(f"could not execute: {exc}") from exc

        return ToolResult(
            _format_process_output(completed.returncode, completed.stdout, completed.stderr),
            is_error=completed.returncode != 0,
        )

    # ---- schemas --------------------------------------------------------------

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-format tool schemas for the tools currently enabled."""
        schemas = [
            _schema(
                "read_file",
                "Read a UTF-8 text file from the workspace, with line numbers.",
                {
                    "path": _p("string", "Path relative to the workspace root."),
                    "start_line": _p("integer", "First line to read (1-based)."),
                    "end_line": _p("integer", "Last line to read, inclusive."),
                },
                ["path"],
            ),
            _schema(
                "write_file",
                "Create a file or replace its entire contents. Prefer edit_file for "
                "changes to an existing file.",
                {
                    "path": _p("string", "Path relative to the workspace root."),
                    "content": _p("string", "Full file contents."),
                },
                ["path", "content"],
            ),
            _schema(
                "edit_file",
                "Replace an exact string in a file. old_text must match byte for byte "
                "and be unique unless replace_all is true.",
                {
                    "path": _p("string", "Path relative to the workspace root."),
                    "old_text": _p("string", "Exact text to find."),
                    "new_text": _p("string", "Replacement text."),
                    "replace_all": _p("boolean", "Replace every occurrence."),
                },
                ["path", "old_text", "new_text"],
            ),
            _schema(
                "list_dir",
                "List the entries of a directory.",
                {"path": _p("string", "Directory path; defaults to the workspace root.")},
                [],
            ),
            _schema(
                "grep",
                "Search file contents with a Python regular expression.",
                {
                    "pattern": _p("string", "Python regex."),
                    "path": _p("string", "File or directory to search."),
                    "glob": _p("string", "Filename glob filter, e.g. '*.py'."),
                    "max_results": _p("integer", "Cap on returned matches."),
                },
                ["pattern"],
            ),
        ]

        if self.index is not None:
            schemas += [
                _schema(
                    "search_memory",
                    "Semantic/keyword search over the indexed repository. Use this "
                    "before grep when you do not know which file to look in.",
                    {
                        "query": _p("string", "What you are looking for."),
                        "top_k": _p("integer", "How many chunks to return."),
                    },
                    ["query"],
                ),
                _schema(
                    "remember",
                    "Save a durable note about this project for future sessions. Use "
                    "for conventions, gotchas, and architecture decisions -- not for "
                    "transient task state.",
                    {
                        "topic": _p("string", "Short topic key."),
                        "content": _p("string", "The fact to remember."),
                    },
                    ["topic", "content"],
                ),
            ]

        if self.config.shell_policy != "deny":
            schemas += [
                _schema(
                    "run_shell",
                    "Run a shell command in the workspace root. Use this to run tests, "
                    "linters, and build steps.",
                    {
                        "command": _p("string", "The command line to run."),
                        "timeout_s": _p("integer", "Timeout in seconds."),
                    },
                    ["command"],
                ),
                _schema(
                    "run_python",
                    "Run a short Python snippet in a fresh subprocess.",
                    {
                        "code": _p("string", "Python source to execute."),
                        "timeout_s": _p("integer", "Timeout in seconds."),
                    },
                    ["code"],
                ),
            ]
        return schemas


def _p(json_type: str, description: str) -> dict[str, str]:
    return {"type": json_type, "description": description}


def _schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _format_process_output(
    returncode: int, stdout: str, stderr: str, *, max_chars: int = 20000
) -> str:
    """Render process output, keeping the tail when it is too long.

    The tail matters more than the head: stack traces and assertion failures
    land at the end of the output.
    """
    sections = [f"exit code: {returncode}"]
    for label, stream in (("stdout", stdout), ("stderr", stderr)):
        text = (stream or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = f"... (truncated {len(text) - max_chars} chars)\n{text[-max_chars:]}"
        sections.append(f"--- {label} ---\n{text}")
    return "\n".join(sections)


def describe_command(command: str) -> str:
    """One-line rendering of a command for an approval prompt."""
    collapsed = " ".join(command.split())
    return collapsed if len(collapsed) <= 200 else collapsed[:197] + "..."


__all__ = [
    "ApprovalCallback",
    "PermissionDenied",
    "ToolError",
    "ToolResult",
    "Toolbox",
    "Workspace",
    "describe_command",
]
