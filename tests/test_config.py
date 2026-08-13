"""Config layering and validation."""

from __future__ import annotations

import pytest

from newal.config import Config, _deep_merge, _env_overrides, load_config


def test_deep_merge_is_recursive_and_non_destructive():
    base = {"model": {"id": "a", "temperature": 0.7}, "ui": {"show_thinking": False}}
    overlay = {"model": {"id": "b"}}
    merged = _deep_merge(base, overlay)

    assert merged["model"] == {"id": "b", "temperature": 0.7}
    assert merged["ui"] == {"show_thinking": False}
    assert base["model"]["id"] == "a"  # original untouched


def test_env_overrides_are_nested_and_typed():
    overrides = _env_overrides(
        {
            "NEWAL_MODEL__ID": "Qwen/Qwen3.5-9B",
            "NEWAL_MODEL__ENABLE_THINKING": "false",
            "NEWAL_BACKEND__MAX_MODEL_LEN": "32768",
            "UNRELATED": "ignored",
        }
    )
    assert overrides["model"]["id"] == "Qwen/Qwen3.5-9B"
    assert overrides["model"]["enable_thinking"] is False
    assert overrides["backend"]["max_model_len"] == 32768
    assert "unrelated" not in overrides


def test_defaults_load_from_the_shipped_config():
    config = load_config(use_env=False)
    assert config.model.id.startswith("Qwen/")
    assert config.backend.kind == "openai_compat"


def test_overrides_win_over_file_values():
    config = load_config(use_env=False, overrides={"model": {"id": "custom/model"}})
    assert config.model.id == "custom/model"


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap_lines"):
        Config.model_validate({"memory": {"chunk_lines": 20, "chunk_overlap_lines": 20}})


def test_invalid_enum_value_is_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"tools": {"shell_policy": "yolo"}})
