"""Configuration loading and validation.

Layering, lowest priority first:
  configs/default.yaml  ->  configs/local.yaml  ->  --config FILE  ->  NEWAL_* env vars

Env overrides use double underscores for nesting, e.g.
``NEWAL_MODELS__HEAVY__ID=Qwen/Qwen3.5-27B`` or ``NEWAL_ROUTER__STRATEGY=single``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "local.yaml"

ENV_PREFIX = "NEWAL_"

ModelTask = Literal["generate", "embed", "rerank"]


class GenerationConfig(BaseModel):
    """Sampling defaults shared by every generation model in the pool."""

    context_length: int = 262144
    temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 32768


class ModelSpec(BaseModel):
    """One member of the model pool.

    Serving fields left as ``None`` inherit from :class:`RuntimeConfig`, so a
    pool member only has to state what makes it different.
    """

    id: str
    base_url: str
    api_key: str = "EMPTY"
    task: ModelTask = "generate"
    #: Cascade position. Higher is stronger and more expensive.
    tier: int = 1
    enabled: bool = True
    autostart: bool = True

    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    tensor_parallel_size: int | None = None
    quantization: str | None = None

    #: Draft model for speculative decoding. Output is distribution-identical
    #: to decoding without it, so this is speed with no quality tradeoff.
    speculative_draft: str | None = None
    speculative_tokens: int = 3

    extra_args: list[str] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    """Serving defaults applied to every pool member that does not override them."""

    kind: Literal["openai_compat", "transformers"] = "openai_compat"
    engine: Literal["vllm", "sglang"] = "vllm"
    autostart: bool = True
    request_timeout_s: int = 600
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int = 65536
    quantization: str | None = None
    startup_timeout_s: int = 900
    extra_args: list[str] = Field(default_factory=list)


class ThinkingConfig(BaseModel):
    #: always | never | adaptive (decide per call from a difficulty score)
    mode: Literal["always", "never", "adaptive"] = "adaptive"
    threshold: float = 0.55

    @field_validator("threshold")
    @classmethod
    def _in_unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("thinking.threshold must be between 0 and 1")
        return v


class RouterConfig(BaseModel):
    #: single -> always use the strongest model; cascade -> route by difficulty.
    strategy: Literal["single", "cascade"] = "cascade"
    #: How the tier is chosen for borderline queries:
    #:   heuristic -> patterns and signals only (free, deterministic)
    #:   semantic  -> nearest labelled exemplars via embeddings (needs `embed`)
    #:   llm       -> ask the cheapest model (one short call)
    #:   auto      -> heuristic, falling back to semantic then llm when unsure
    mode: Literal["heuristic", "semantic", "llm", "auto"] = "auto"
    escalate_threshold: float = 0.45
    #: Consult a classifier only when the score is this close to the threshold.
    #: 0 disables classifiers entirely, whatever `mode` says.
    uncertainty_band: float = 0.12
    #: Verdicts below this confidence are ignored and the heuristic stands.
    min_classifier_confidence: float = 0.3
    #: Nearest exemplars the semantic classifier votes over.
    semantic_neighbours: int = 5
    max_escalations: int = 2
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    #: Record outcomes and feed them back as exemplars, so routing adapts to
    #: this repository over time.
    learn_from_outcomes: bool = True
    #: Cap on stored outcomes replayed into the classifier at startup.
    max_learned_exemplars: int = 200
    #: Print the routing decision for every call.
    explain: bool = False

    @field_validator("escalate_threshold", "uncertainty_band", "min_classifier_confidence")
    @classmethod
    def _in_unit_range(cls, v: float, info: Any) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"router.{info.field_name} must be between 0 and 1")
        return v


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
    #: Blend dense embedding similarity with BM25 (needs an `embed` pool member).
    hybrid_retrieval: bool = True
    #: Weight of the dense score in the hybrid blend; BM25 takes the remainder.
    dense_weight: float = 0.5
    #: Cross-encoder rerank of the merged candidates (needs a `rerank` member).
    use_reranker: bool = True
    #: Candidates to retrieve before reranking trims to retrieve_top_k.
    rerank_candidates: int = 30

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

    @field_validator("dense_weight")
    @classmethod
    def _weight_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("memory.dense_weight must be between 0 and 1")
        return v


class TrainingConfig(BaseModel):
    """Capture of turns and repair pairs as fine-tuning material.

    Local only -- nothing is uploaded anywhere. The captured turns contain the
    source code discussed in the conversation, so the memory DB should be
    treated as sensitive. Attachments are always redacted before storage.
    """

    enabled: bool = True
    #: Skip capturing turns longer than this; a runaway loop is not useful data.
    max_messages_per_turn: int = 80
    #: Cap on stored context messages per DPO pair, counted from the end.
    max_context_messages: int = 40


class UIConfig(BaseModel):
    show_thinking: bool = False
    show_token_usage: bool = True
    #: Write one JSONL file per session. Attachments are redacted before
    #: writing; source code from the conversation is not.
    save_transcripts: bool = True
    transcript_dir: str = ".newal/transcripts"


class Config(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @model_validator(mode="after")
    def _needs_a_generation_model(self) -> Config:
        if not self.enabled_models(task="generate"):
            raise ValueError(
                "at least one enabled model with task 'generate' is required; "
                "check the `models:` section of your config"
            )
        return self

    # ---- pool access ----------------------------------------------------------

    def enabled_models(self, *, task: ModelTask | None = None) -> dict[str, ModelSpec]:
        return {
            key: spec
            for key, spec in self.models.items()
            if spec.enabled and (task is None or spec.task == task)
        }

    def first_model(self, task: ModelTask) -> tuple[str, ModelSpec] | None:
        """The lowest-tier enabled model for a task, if any."""
        candidates = sorted(self.enabled_models(task=task).items(), key=lambda i: i[1].tier)
        return candidates[0] if candidates else None

    def strongest_model(self) -> tuple[str, ModelSpec]:
        candidates = sorted(
            self.enabled_models(task="generate").items(), key=lambda i: i[1].tier
        )
        return candidates[-1]

    def serving_params(self, spec: ModelSpec) -> dict[str, Any]:
        """Merge a spec's serving fields over the shared runtime defaults."""
        return {
            "max_model_len": spec.max_model_len or self.runtime.max_model_len,
            "gpu_memory_utilization": (
                spec.gpu_memory_utilization
                if spec.gpu_memory_utilization is not None
                else self.runtime.gpu_memory_utilization
            ),
            "tensor_parallel_size": (
                spec.tensor_parallel_size or self.runtime.tensor_parallel_size
            ),
            "quantization": spec.quantization or self.runtime.quantization,
            "extra_args": list(self.runtime.extra_args) + list(spec.extra_args),
        }

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
