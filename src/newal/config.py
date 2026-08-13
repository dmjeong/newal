"""Configuration loading and validation.

Layering, lowest priority first:
  configs/default.yaml  ->  configs/local.yaml  ->  --config FILE  ->  NEWAL_* env vars

Env overrides use double underscores for nesting, e.g.
``NEWAL_MODEL__ID=Qwen/Qwen3.5-9B`` or ``NEWAL_BACKEND__BASE_URL=...``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

ENV_PREFIX = "NEWAL_"


class ModelConfig(BaseModel):
    id: str = "Qwen/Qwen3.6-35B-A3B"
    context_length: int = 262144
    enable_thinking: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 32768


class BackendConfig(BaseModel):
    kind: Literal["openai_compat", "transformers"] = "openai_compat"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    request_timeout_s: int = 600
    autostart: bool = True
    engine: Literal["vllm", "sglang"] = "vllm"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 65536
    quantization: str | None = None
    startup_timeout_s: int = 900
    extra_args: list[str] = Field(default_factory=list)


class VideoConfig(BaseModel):
    max_frames: int = 32
    strategy: Literal["uniform", "scene"] = "scene"
    scene_threshold: float = 0.28
    frame_max_pixels: int = 409600
    include_timestamps: bool = True


class MediaConfig(BaseModel):
    visual_token_budget: int = 16384
    image_max_pixels: int = 1638400
    video: VideoConfig = Field(default_factory=VideoConfig)


class AgentConfig(BaseModel):
    max_steps: int = 40
    plan_first: bool = True
    verify: bool = True
    verify_command: str | None = None
    max_verify_retries: int = 2


class ToolsConfig(BaseModel):
    workspace_root: str = "."
    shell_policy: Literal["ask", "allow", "deny"] = "ask"
    shell_timeout_s: int = 180
    max_file_read_bytes: int = 262144
    denied_paths: list[str] = Field(
        default_factory=lambda: [".git", ".env", ".venv", "node_modules", "__pycache__"]
    )


class MemoryConfig(BaseModel):
    enabled: bool = True
    db_path: str = ".newal/memory.db"
    index_globs: list[str] = Field(default_factory=lambda: ["**/*.py", "**/*.md"])
    index_exclude: list[str] = Field(
        default_factory=lambda: [".venv/**", "node_modules/**", ".git/**", ".newal/**"]
    )
    chunk_lines: int = 80
    chunk_overlap_lines: int = 15
    retrieve_top_k: int = 8
    dense_rerank: bool = False
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @field_validator("chunk_overlap_lines")
    @classmethod
    def _overlap_fits(cls, v: int, info: Any) -> int:
        chunk = info.data.get("chunk_lines", 80)
        if v >= chunk:
            raise ValueError(
                f"chunk_overlap_lines ({v}) must be smaller than chunk_lines ({chunk}); "
                "otherwise chunking never advances"
            )
        return v


class UIConfig(BaseModel):
    show_thinking: bool = False
    show_token_usage: bool = True
    transcript_dir: str = ".newal/transcripts"


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    def workspace_path(self) -> Path:
        return Path(self.tools.workspace_root).expanduser().resolve()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    """Parse an env var value as YAML so ints/bools/null survive the round trip."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    out: dict[str, Any] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # conflicting scalar already set
                break
        else:
            cursor[path[-1]] = _coerce_scalar(raw_value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(
    config_path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Config:
    """Build a :class:`Config` from the layered sources described above."""
    merged = _read_yaml(DEFAULT_CONFIG_PATH)
    merged = _deep_merge(merged, _read_yaml(LOCAL_CONFIG_PATH))

    if config_path is not None:
        explicit = Path(config_path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"config file not found: {explicit}")
        merged = _deep_merge(merged, _read_yaml(explicit))

    if use_env:
        merged = _deep_merge(merged, _env_overrides())
    if overrides:
        merged = _deep_merge(merged, overrides)

    return Config.model_validate(merged)
