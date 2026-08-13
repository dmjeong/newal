"""Tool sandbox, file operations, and shell gating."""

from __future__ import annotations

import pytest

from newal.agent.tools import PermissionDenied, Toolbox, Workspace
from newal.config import ToolsConfig


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# notes\nalpha\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    return tmp_path


@pytest.fixture
def toolbox(workspace):
    return Toolbox(ToolsConfig(workspace_root=str(workspace), shell_policy="deny"))


def test_resolve_rejects_parent_traversal(workspace):
    ws = Workspace(ToolsConfig(workspace_root=str(workspace)))
    with pytest.raises(PermissionDenied):
        ws.resolve("../../etc/passwd")


def test_resolve_rejects_absolute_paths_outside_root(workspace):
    ws = Workspace(ToolsConfig(workspace_root=str(workspace)))
    with pytest.raises(PermissionDenied):
        ws.resolve("/etc/passwd")


def test_resolve_rejects_symlink_escape(workspace, tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform")

    ws = Workspace(ToolsConfig(workspace_root=str(workspace)))
    with pytest.raises(PermissionDenied):
        ws.resolve("link.txt")


def test_denied_paths_are_blocked(toolbox):
    result = toolbox.call("read_file", {"path": ".git/config"})
    assert result.is_error
    assert "deny list" in result.content


def test_read_file_is_line_numbered(toolbox):
    result = toolbox.call("read_file", {"path": "src/app.py"})
    assert not result.is_error
    assert "def run():" in result.content
    assert "1\t" in result.content


def test_read_file_slice(toolbox):
    result = toolbox.call("read_file", {"path": "src/app.py", "start_line": 2, "end_line": 2})
    assert "return 1" in result.content
    assert "def run" not in result.content


def test_read_missing_file_reports_cleanly(toolbox):
    result = toolbox.call("read_file", {"path": "nope.py"})
    assert result.is_error
    assert "no such file" in result.content


def test_write_then_read_round_trip(toolbox, workspace):
    write = toolbox.call("write_file", {"path": "new/mod.py", "content": "X = 1\n"})
    assert not write.is_error
    assert (workspace / "new" / "mod.py").read_text(encoding="utf-8") == "X = 1\n"
    assert "new/mod.py" in toolbox.files_written


def test_edit_file_replaces_unique_text(toolbox, workspace):
    result = toolbox.call(
        "edit_file",
        {"path": "src/app.py", "old_text": "return 1", "new_text": "return 42"},
    )
    assert not result.is_error
    assert "return 42" in (workspace / "src" / "app.py").read_text(encoding="utf-8")


def test_edit_file_refuses_ambiguous_match(toolbox, workspace):
    (workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = toolbox.call(
        "edit_file", {"path": "dup.py", "old_text": "x = 1", "new_text": "x = 2"}
    )
    assert result.is_error
    assert "appears 2 times" in result.content


def test_edit_file_replace_all(toolbox, workspace):
    (workspace / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = toolbox.call(
        "edit_file",
        {"path": "dup.py", "old_text": "x = 1", "new_text": "x = 2", "replace_all": True},
    )
    assert not result.is_error
    assert (workspace / "dup.py").read_text(encoding="utf-8") == "x = 2\nx = 2\n"


def test_edit_file_missing_old_text(toolbox):
    result = toolbox.call(
        "edit_file", {"path": "src/app.py", "old_text": "nope", "new_text": "x"}
    )
    assert result.is_error
    assert "not found" in result.content


def test_grep_finds_matches_with_locations(toolbox):
    result = toolbox.call("grep", {"pattern": r"def \w+", "glob": "*.py"})
    assert "src/app.py:1" in result.content


def test_grep_reports_invalid_regex(toolbox):
    result = toolbox.call("grep", {"pattern": "([unclosed"})
    assert result.is_error
    assert "invalid regex" in result.content


def test_list_dir_hides_denied_entries(toolbox):
    result = toolbox.call("list_dir", {"path": "."})
    assert "src/" in result.content
    assert ".git" not in result.content


def test_shell_is_blocked_under_deny_policy(toolbox):
    result = toolbox.call("run_shell", {"command": "echo hi"})
    assert result.is_error
    assert "disabled" in result.content


def test_deny_policy_hides_shell_from_schemas(toolbox):
    names = {schema["function"]["name"] for schema in toolbox.schemas()}
    assert "run_shell" not in names
    assert "read_file" in names


def test_destructive_commands_are_refused_even_when_allowed(workspace):
    box = Toolbox(ToolsConfig(workspace_root=str(workspace), shell_policy="allow"))
    result = box.call("run_shell", {"command": "rm -rf /"})
    assert result.is_error
    assert "destructive" in result.content


def test_ask_policy_consults_the_approver(workspace):
    denied = Toolbox(
        ToolsConfig(workspace_root=str(workspace), shell_policy="ask"),
        approve=lambda name, detail: False,
    )
    assert denied.call("run_shell", {"command": "echo hi"}).is_error

    allowed = Toolbox(
        ToolsConfig(workspace_root=str(workspace), shell_policy="ask"),
        approve=lambda name, detail: True,
    )
    result = allowed.call("run_shell", {"command": "echo hi"})
    assert not result.is_error
    assert "hi" in result.content


def test_shell_failure_surfaces_exit_code(workspace):
    box = Toolbox(ToolsConfig(workspace_root=str(workspace), shell_policy="allow"))
    result = box.call("run_python", {"code": "import sys; sys.exit(3)"})
    assert result.is_error
    assert "exit code: 3" in result.content


def test_unknown_tool_lists_alternatives(toolbox):
    result = toolbox.call("teleport", {})
    assert result.is_error
    assert "read_file" in result.content


def test_malformed_arguments_are_reported_back(toolbox):
    result = toolbox.call("read_file", {"__parse_error__": "{broken"})
    assert result.is_error
    assert "not valid JSON" in result.content
