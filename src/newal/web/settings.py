"""Which configuration a user may change from the browser, and how.

Not everything in the config is safe to edit from a running session. The model
pool, ports and database paths are read once at startup, so changing them
mid-flight would silently disagree with what is actually running. Only settings
the agent re-reads on every call are exposed here; each one is declared with
its type and range so the server can validate rather than trust the page.

Changes apply to the live session immediately and are also written to
``configs/local.yaml`` so they survive a restart. Only the keys that actually
changed are written -- the file stays a small diff against the shipped
defaults rather than a full copy that would freeze them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ..config import LOCAL_CONFIG_RELATIVE, Config

log = logging.getLogger(__name__)

Kind = Literal["bool", "int", "float", "enum"]


@dataclass(frozen=True)
class Setting:
    """One editable knob, described well enough to validate and to render."""

    path: str
    label: str
    kind: Kind
    group: str
    help: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    #: Security-relevant, so the page can mark it rather than bury it in a list.
    sensitive: bool = False

    def get(self, config: Config) -> Any:
        target: Any = config
        for part in self.path.split("."):
            target = getattr(target, part)
        return target

    def coerce(self, raw: Any) -> Any:
        """Convert and range-check a value coming from the browser."""
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        if self.kind == "enum":
            value = str(raw)
            if value not in self.choices:
                raise ValueError(f"{self.path}: expected one of {', '.join(self.choices)}")
            return value

        try:
            number = int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.path}: {raw!r} is not a {self.kind}") from exc

        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"{self.path}: must be at least {self.minimum}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"{self.path}: must be at most {self.maximum}")
        return number


#: Everything the browser may change. Anything absent is deliberately absent.
SETTINGS: tuple[Setting, ...] = (
    # -- how the agent works ---------------------------------------------------
    Setting(
        "agent.plan_first", "먼저 계획 세우기", "bool", "에이전트",
        help="편집 전에 계획을 세웁니다. 호출이 한 번 늘지만 다중 파일 작업이 정확해집니다.",
    ),
    Setting(
        "agent.verify", "테스트로 검증", "bool", "에이전트",
        help="편집 후 테스트를 돌리고 실패하면 스스로 고칩니다. 품질에 가장 크게 기여합니다.",
    ),
    Setting(
        "agent.max_verify_retries", "수리 재시도 횟수", "int", "에이전트",
        minimum=0, maximum=5, step=1,
    ),
    Setting(
        "agent.max_steps", "턴당 최대 도구 호출", "int", "에이전트",
        minimum=1, maximum=200, step=1,
    ),
    # -- routing ---------------------------------------------------------------
    Setting(
        "router.strategy", "라우팅 전략", "enum", "라우팅",
        choices=("cascade", "single"),
        help="cascade는 쉬운 요청을 싼 모델로 보냅니다. single은 항상 최상위 모델을 씁니다.",
    ),
    Setting(
        "router.mode", "티어 결정 방식", "enum", "라우팅",
        choices=("heuristic", "semantic", "llm", "auto"),
        help="auto는 휴리스틱으로 시작해 애매할 때만 분류기를 부릅니다.",
    ),
    Setting(
        "router.escalate_threshold", "에스컬레이션 임계값", "float", "라우팅",
        minimum=0.0, maximum=1.0, step=0.01,
    ),
    Setting(
        "router.uncertainty_band", "불확실 구간 폭", "float", "라우팅",
        minimum=0.0, maximum=0.5, step=0.01,
        help="0으로 두면 분류기를 쓰지 않습니다.",
    ),
    Setting(
        "router.thinking.mode", "thinking 모드", "enum", "라우팅",
        choices=("adaptive", "always", "never"),
    ),
    Setting(
        "router.explain", "라우팅 근거 표시", "bool", "라우팅",
        help="매 호출마다 어느 모델을 왜 골랐는지 보여줍니다.",
    ),
    # -- generation ------------------------------------------------------------
    Setting(
        "generation.temperature", "temperature", "float", "생성",
        minimum=0.0, maximum=2.0, step=0.05,
    ),
    Setting(
        "generation.top_p", "top_p", "float", "생성",
        minimum=0.0, maximum=1.0, step=0.01,
    ),
    Setting(
        "generation.max_output_tokens", "최대 출력 토큰", "int", "생성",
        minimum=256, maximum=131072, step=256,
    ),
    # -- media -----------------------------------------------------------------
    Setting(
        "media.visual_token_budget", "시각 토큰 예산", "int", "미디어",
        minimum=512, maximum=131072, step=512,
        help="이미지와 동영상 프레임이 이 예산을 나눠 씁니다.",
    ),
    Setting(
        "media.video.max_frames", "동영상 최대 프레임", "int", "미디어",
        minimum=1, maximum=128, step=1,
    ),
    Setting(
        "media.video.strategy", "프레임 선택", "enum", "미디어",
        choices=("scene", "uniform"),
        help="scene은 장면이 바뀌는 지점을 고릅니다.",
    ),
    Setting(
        "media.video.scene_threshold", "장면 전환 민감도", "float", "미디어",
        minimum=0.05, maximum=0.95, step=0.01,
        help="낮을수록 더 많은 전환을 잡습니다.",
    ),
    # -- safety ----------------------------------------------------------------
    Setting(
        "tools.shell_policy", "셸 실행 정책", "enum", "안전",
        choices=("ask", "allow", "deny"),
        sensitive=True,
        help="allow로 두면 확인 없이 명령이 실행됩니다. 파괴적 명령 차단은 그대로 유지됩니다.",
    ),
    Setting(
        "tools.shell_timeout_s", "셸 타임아웃(초)", "int", "안전",
        minimum=5, maximum=3600, step=5,
    ),
    # -- what is kept ----------------------------------------------------------
    Setting(
        "training.enabled", "학습 데이터 수집", "bool", "기록",
        sensitive=True,
        help="대화에 등장한 소스 코드가 로컬 DB에 저장됩니다. 전송되지는 않습니다.",
    ),
    Setting(
        "router.learn_from_outcomes", "라우팅 결과 학습", "bool", "기록",
    ),
    Setting(
        "ui.save_transcripts", "세션 기록 저장", "bool", "기록",
        help="첨부는 제거되지만 대화 내용은 저장됩니다.",
    ),
    Setting(
        "ui.show_thinking", "thinking 과정 표시", "bool", "기록",
    ),
)

BY_PATH: dict[str, Setting] = {s.path: s for s in SETTINGS}


@dataclass
class ApplyResult:
    changed: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    saved_to: str | None = None


def describe(config: Config) -> list[dict[str, Any]]:
    """Render every editable setting with its current value, for the page."""
    return [
        {
            "path": s.path,
            "label": s.label,
            "kind": s.kind,
            "group": s.group,
            "help": s.help,
            "choices": list(s.choices),
            "minimum": s.minimum,
            "maximum": s.maximum,
            "step": s.step,
            "sensitive": s.sensitive,
            "value": s.get(config),
        }
        for s in SETTINGS
    ]


def _assign(config: Config, path: str, value: Any) -> None:
    target: Any = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _nest(path: str, value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cursor = out
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return out


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def apply(config: Config, updates: dict[str, Any], *, persist: bool = True) -> ApplyResult:
    """Validate, apply to the live config, and optionally write local.yaml.

    Rejected values leave the running config untouched: a partially applied
    settings form is worse than one that failed loudly.
    """
    result = ApplyResult()
    accepted: dict[str, Any] = {}

    for path, raw in updates.items():
        setting = BY_PATH.get(path)
        if setting is None:
            result.errors.append(f"{path}: 편집할 수 없는 설정입니다")
            continue
        try:
            accepted[path] = setting.coerce(raw)
        except ValueError as exc:
            result.errors.append(str(exc))

    if result.errors:
        return result

    for path, value in accepted.items():
        setting = BY_PATH[path]
        if setting.get(config) != value:
            _assign(config, path, value)
            result.changed[path] = value

    if persist and result.changed:
        result.saved_to = str(save(config, result.changed))
    return result


def save(config: Config, changed: dict[str, Any]) -> Path:
    """Merge the changed keys into configs/local.yaml, keeping what is there."""
    path = config.workspace_path() / LOCAL_CONFIG_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            log.warning("%s is not valid YAML; it will be replaced", path)
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    for key, value in changed.items():
        existing = _merge(existing, _nest(key, value))

    header = (
        "# newal local overrides.\n"
        "# Written by the settings panel; merged over the shipped defaults.\n"
        "# Only the keys you changed appear here, so defaults stay live.\n"
    )
    path.write_text(
        header + yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
