"""Runtime-editable settings.

The panel writes into a live config and into configs/local.yaml, so the rules
worth pinning are: only declared settings are writable, values are validated
before anything is touched, and the saved file stays a small diff rather than a
frozen copy of the defaults.
"""

from __future__ import annotations

import pytest
import yaml

from newal.config import load_config
from newal.web import settings as s


@pytest.fixture
def config(tmp_path):
    return load_config(
        use_env=False, overrides={"tools": {"workspace_root": str(tmp_path)}}
    )


# ---- the schema --------------------------------------------------------------


def test_every_declared_setting_resolves_against_the_config(config):
    """A typo in a path would otherwise only surface when someone opened the panel."""
    for setting in s.SETTINGS:
        setting.get(config)


def test_startup_only_config_is_not_exposed():
    """Model pool, ports and database paths are read once; editing them would lie."""
    paths = {setting.path for setting in s.SETTINGS}
    for forbidden in ("models", "runtime.engine", "runtime.max_model_len", "memory.db_path"):
        assert not any(p == forbidden or p.startswith(forbidden + ".") for p in paths)


def test_describe_reports_current_values(config):
    described = {item["path"]: item for item in s.describe(config)}
    assert described["agent.verify"]["value"] is config.agent.verify
    assert described["router.mode"]["value"] == config.router.mode
    assert described["tools.shell_policy"]["sensitive"] is True


# ---- validation --------------------------------------------------------------


def test_a_bad_enum_is_rejected(config):
    result = s.apply(config, {"router.mode": "telepathy"}, persist=False)
    assert result.errors
    assert result.changed == {}


def test_out_of_range_numbers_are_rejected(config):
    result = s.apply(config, {"router.escalate_threshold": 4.2}, persist=False)
    assert result.errors


def test_unknown_paths_are_rejected(config):
    result = s.apply(config, {"models.heavy.id": "something/else"}, persist=False)
    assert result.errors
    assert "편집할 수 없는" in result.errors[0]


def test_one_bad_value_blocks_the_whole_update(config):
    """A half-applied settings form is worse than one that failed loudly."""
    before = config.agent.max_steps
    result = s.apply(
        config,
        {"agent.max_steps": 12, "router.mode": "nonsense"},
        persist=False,
    )
    assert result.errors
    assert config.agent.max_steps == before


def test_checkbox_style_values_coerce(config):
    s.apply(config, {"agent.verify": "false"}, persist=False)
    assert config.agent.verify is False
    s.apply(config, {"agent.verify": "on"}, persist=False)
    assert config.agent.verify is True


# ---- applying ----------------------------------------------------------------


def test_changes_apply_to_the_live_config(config):
    result = s.apply(
        config, {"router.mode": "heuristic", "generation.temperature": 0.2}, persist=False
    )
    assert result.changed == {"router.mode": "heuristic", "generation.temperature": 0.2}
    assert config.router.mode == "heuristic"
    assert config.generation.temperature == 0.2


def test_unchanged_values_are_not_reported(config):
    result = s.apply(config, {"router.mode": config.router.mode}, persist=False)
    assert result.changed == {}


def test_nested_paths_apply(config):
    s.apply(config, {"media.video.max_frames": 8}, persist=False)
    assert config.media.video.max_frames == 8


# ---- persistence -------------------------------------------------------------


def test_saving_writes_only_what_changed(tmp_path, config):
    s.apply(config, {"router.mode": "llm"})

    saved = yaml.safe_load((tmp_path / "configs" / "local.yaml").read_text(encoding="utf-8"))
    assert saved == {"router": {"mode": "llm"}}
    # The rest of the defaults must stay live rather than being frozen into the file.
    assert "models" not in saved
    assert "generation" not in saved


def test_saving_merges_with_an_existing_file(tmp_path, config):
    local = tmp_path / "configs" / "local.yaml"
    local.parent.mkdir(parents=True)
    local.write_text("router:\n  explain: true\nui:\n  show_thinking: true\n", encoding="utf-8")

    s.apply(config, {"router.mode": "semantic"})

    saved = yaml.safe_load(local.read_text(encoding="utf-8"))
    assert saved["router"] == {"explain": True, "mode": "semantic"}
    assert saved["ui"] == {"show_thinking": True}


def test_a_corrupt_local_file_is_replaced_not_fatal(tmp_path, config):
    local = tmp_path / "configs" / "local.yaml"
    local.parent.mkdir(parents=True)
    local.write_text("{{{ not yaml", encoding="utf-8")

    s.apply(config, {"agent.plan_first": False})
    assert yaml.safe_load(local.read_text(encoding="utf-8")) == {"agent": {"plan_first": False}}


def test_saved_settings_survive_a_reload(tmp_path, config, monkeypatch):
    s.apply(config, {"router.mode": "heuristic", "agent.max_steps": 7})

    monkeypatch.chdir(tmp_path)
    reloaded = load_config(use_env=False)
    assert reloaded.router.mode == "heuristic"
    assert reloaded.agent.max_steps == 7
