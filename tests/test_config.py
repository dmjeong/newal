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
    key, spec = config.strongest_model()
    assert spec.id.startswith("Qwen/")
    assert config.runtime.kind == "openai_compat"
    assert key in config.models


def test_only_one_model_is_enabled_by_default():
    """A fresh install must fit on one GPU, so extra pool members ship off."""
    config = load_config(use_env=False)
    assert len(config.enabled_models()) == 1
    assert config.enabled_models(task="generate")


def test_overrides_win_over_file_values():
    config = load_config(
        use_env=False, overrides={"models": {"heavy": {"id": "custom/model"}}}
    )
    assert config.models["heavy"].id == "custom/model"


def test_config_rejects_a_pool_with_no_generation_model():
    with pytest.raises(ValueError, match="task 'generate'"):
        Config.model_validate(
            {
                "models": {
                    "embedding": {
                        "id": "Qwen/Qwen3-Embedding-0.6B",
                        "base_url": "http://127.0.0.1:8002/v1",
                        "task": "embed",
                    }
                }
            }
        )


def test_disabled_models_are_not_in_the_pool():
    config = load_config(
        use_env=False, overrides={"models": {"light": {"enabled": True}}}
    )
    assert "light" in config.enabled_models()
    assert "embedding" not in config.enabled_models()


def test_strongest_model_is_the_highest_tier():
    config = load_config(
        use_env=False, overrides={"models": {"light": {"enabled": True}}}
    )
    key, spec = config.strongest_model()
    assert key == "heavy"
    assert spec.tier > config.models["light"].tier


def test_serving_params_inherit_from_runtime_and_can_be_overridden():
    config = load_config(
        use_env=False,
        overrides={
            "runtime": {"max_model_len": 40000, "gpu_memory_utilization": 0.8},
            "models": {"heavy": {"max_model_len": 16384}},
        },
    )
    params = config.serving_params(config.models["heavy"])
    assert params["max_model_len"] == 16384          # spec wins
    assert params["gpu_memory_utilization"] == 0.8   # inherited


def test_thresholds_must_be_probabilities():
    with pytest.raises(ValueError):
        Config.model_validate(
            {
                "models": {
                    "m": {"id": "x", "base_url": "http://localhost:8000/v1"}
                },
                "router": {"escalate_threshold": 1.7},
            }
        )


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap_lines"):
        Config.model_validate({"memory": {"chunk_lines": 20, "chunk_overlap_lines": 20}})


def test_invalid_enum_value_is_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"tools": {"shell_policy": "yolo"}})


# ---- the defaults must travel with the package ------------------------------


def test_shipped_defaults_live_inside_the_package():
    """Deriving the path from the repo layout breaks a plain `pip install`.

    The old code used PACKAGE_ROOT.parent.parent / "configs", which only
    resolves in a source checkout. An installed copy landed on a path that did
    not exist and failed validation before the first prompt.
    """
    from newal.config import DEFAULT_CONFIG_PATH, PACKAGE_ROOT

    assert DEFAULT_CONFIG_PATH.is_file()
    assert PACKAGE_ROOT in DEFAULT_CONFIG_PATH.parents


def test_defaults_load_without_any_repo_files_present(tmp_path, monkeypatch):
    """Running from an unrelated directory must still produce a valid config."""
    monkeypatch.chdir(tmp_path)
    config = load_config(use_env=False)
    assert config.enabled_models(task="generate")


def test_local_overrides_are_read_from_the_working_directory(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "local.yaml").write_text(
        "router:\n  mode: heuristic\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    assert load_config(use_env=False).router.mode == "heuristic"
