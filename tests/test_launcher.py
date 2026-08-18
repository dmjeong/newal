"""Engine command construction.

These flags are easy to get subtly wrong and expensive to debug against a real
GPU, so they are asserted here instead.
"""

from __future__ import annotations

import json

import pytest

from newal.backends.launcher import build_command
from newal.config import load_config


def _config(**overrides):
    return load_config(use_env=False, overrides=overrides)


def test_vllm_command_carries_qwen_parsers():
    config = _config()
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)

    assert "--reasoning-parser" in cmd
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd


def test_speculative_config_is_valid_json_with_the_draft_model():
    config = _config()
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)

    assert "--speculative-config" in cmd
    payload = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert payload["model"] == spec.speculative_draft
    assert payload["num_speculative_tokens"] == spec.speculative_tokens


def test_speculation_is_omitted_when_no_draft_model_is_set():
    config = _config(models={"heavy": {"speculative_draft": None}})
    key, spec = config.strongest_model()
    assert "--speculative-config" not in build_command(config, key, spec)


def test_port_comes_from_the_model_base_url():
    config = _config(models={"heavy": {"base_url": "http://127.0.0.1:9123/v1"}})
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)
    assert cmd[cmd.index("--port") + 1] == "9123"


def test_embedding_model_is_served_as_an_embedding_task():
    config = _config(models={"embedding": {"enabled": True}})
    spec = config.models["embedding"]
    cmd = build_command(config, "embedding", spec)

    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "embed"
    # Generation-only flags must not leak onto a non-generation server.
    assert "--enable-auto-tool-choice" not in cmd
    assert "--speculative-config" not in cmd


def test_reranker_model_is_served_as_a_scoring_task():
    config = _config(models={"reranker": {"enabled": True}})
    cmd = build_command(config, "reranker", config.models["reranker"])
    assert cmd[cmd.index("--task") + 1] == "score"


def test_per_model_serving_overrides_reach_the_command_line():
    config = _config(
        runtime={"max_model_len": 65536},
        models={"heavy": {"max_model_len": 8192, "quantization": "fp8"}},
    )
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)

    assert cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert cmd[cmd.index("--quantization") + 1] == "fp8"


def test_sglang_command_uses_its_own_flag_names():
    config = _config(runtime={"engine": "sglang"})
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)

    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "--context-length" in cmd
    assert "--max-model-len" not in cmd


def test_extra_args_are_appended_last():
    config = _config(
        runtime={"extra_args": ["--seed", "7"]},
        models={"heavy": {"extra_args": ["--disable-log-requests"]}},
    )
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)

    assert cmd[-3:] == ["7", "--disable-log-requests"] or cmd[-1] == "--disable-log-requests"
    assert "--seed" in cmd


def test_each_pool_member_gets_a_distinct_port():
    config = _config(
        models={"light": {"enabled": True}, "embedding": {"enabled": True}}
    )
    ports = set()
    for key, spec in config.enabled_models().items():
        cmd = build_command(config, key, spec)
        ports.add(cmd[cmd.index("--port") + 1])
    assert len(ports) == len(config.enabled_models())


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_both_engines_produce_a_runnable_looking_command(engine):
    config = _config(runtime={"engine": engine})
    key, spec = config.strongest_model()
    cmd = build_command(config, key, spec)
    assert cmd[1] == "-m"
    assert all(isinstance(part, str) for part in cmd)


# ---- an unreachable member ---------------------------------------------------


@pytest.fixture
def no_server(monkeypatch):
    """No inference server anywhere, and nothing may be started."""
    monkeypatch.setattr("newal.models.pool.is_server_up", lambda _url: False)
    return _config(runtime={"autostart": False})


def test_a_terminal_pool_refuses_to_start_without_a_model(no_server):
    """The CLI has nothing to offer without a model, so it must fail loudly."""
    from newal.backends.launcher import ServerStartError
    from newal.models import ModelPool

    with pytest.raises(ServerStartError):
        ModelPool(no_server)


def test_a_tolerant_pool_starts_and_records_why(no_server):
    """The browser UI still has pages, settings and training data to serve."""
    from newal.models import ModelPool

    pool = ModelPool(no_server, tolerate_unavailable=True)

    assert set(pool.unavailable) == set(no_server.enabled_models())
    # The router still exists, so routing and the settings panel keep working.
    assert pool.router.strongest
    assert pool.describe()


def test_a_tolerated_member_still_refuses_to_serve_a_turn(no_server):
    """Tolerating the failure at startup must not hide it at request time."""
    from newal.backends.base import BackendError
    from newal.models import ModelPool

    pool = ModelPool(no_server, tolerate_unavailable=True)

    with pytest.raises(BackendError, match="autostart is off"):
        pool.backend(pool.router.strongest)
