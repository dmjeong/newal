"""The VS Code entry points.

These are the first thing anyone touches after cloning, and nothing else fails
when they are wrong -- the editor just does something other than what the
README promised. So they are asserted here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VSCODE = Path(__file__).resolve().parent.parent / ".vscode"


def _jsonc(name: str) -> dict:
    """Read a VS Code config. They are JSON with // comments allowed."""
    raw = (VSCODE / name).read_text(encoding="utf-8")
    return json.loads(re.sub(r"^\s*//.*$", "", raw, flags=re.M))


def test_f5_starts_the_browser_ui():
    """F5 runs the first configuration, and the README says a website appears."""
    first = _jsonc("launch.json")["configurations"][0]
    assert first["module"] == "newal"
    assert first["args"][0] == "web"


def test_there_is_a_way_to_run_the_ui_without_a_gpu():
    configs = _jsonc("launch.json")["configurations"]
    web = [c for c in configs if c["args"][:1] == ["web"]]
    assert any("--no-autostart" in c["args"] for c in web)


def test_launch_configs_can_import_the_package_before_install():
    for config in _jsonc("launch.json")["configurations"]:
        assert "src" in config["env"]["PYTHONPATH"], config["name"]


def test_setup_installs_what_the_browser_ui_needs():
    """Without the web extra there is no uvicorn, so `newal web` cannot start."""
    setup = next(
        task for task in _jsonc("tasks.json")["tasks"] if task["label"].startswith("setup:")
    )
    for command in (setup["command"], setup["windows"]["command"]):
        assert "web" in command, "the setup task must install the web extra"


def test_every_task_works_on_windows_too():
    """The default commands use .venv/bin, which does not exist on Windows."""
    for task in _jsonc("tasks.json")["tasks"]:
        assert "windows" in task, task["label"]
