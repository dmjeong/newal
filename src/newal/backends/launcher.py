"""Start and health-check local vLLM / SGLang servers.

Each pool member is its own server process on its own port, so the launcher
must be able to bring up several and shut them all down together.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from ..config import Config, ModelSpec

log = logging.getLogger(__name__)

# Reasoning/tool parsers Qwen3.5+ ships with. Without these, vLLM returns the
# raw <think> block as content and tool calls arrive as unparsed text.
QWEN_REASONING_PARSER = "qwen3"
QWEN_TOOL_PARSER = "hermes"


class ServerStartError(RuntimeError):
    pass


def is_server_up(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if an OpenAI-compatible server answers at ``base_url``."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _host_port(base_url: str) -> tuple[str, str]:
    parsed = urlparse(base_url)
    return parsed.hostname or "127.0.0.1", str(parsed.port or 8000)


def build_command(config: Config, key: str, spec: ModelSpec) -> list[str]:
    """Build the engine command line for one pool member.

    Kept pure so the flags -- especially the speculative-decoding and task
    settings, which are easy to get subtly wrong -- can be asserted in tests.
    """
    host, port = _host_port(spec.base_url)
    params = config.serving_params(spec)

    if config.runtime.engine == "vllm":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            spec.id,
            "--served-model-name",
            spec.id,
            "--host",
            host,
            "--port",
            port,
            "--max-model-len",
            str(params["max_model_len"]),
            "--gpu-memory-utilization",
            str(params["gpu_memory_utilization"]),
            "--tensor-parallel-size",
            str(params["tensor_parallel_size"]),
        ]

        if spec.task == "generate":
            cmd += [
                "--reasoning-parser",
                QWEN_REASONING_PARSER,
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                QWEN_TOOL_PARSER,
            ]
            if spec.speculative_draft:
                # Draft-model speculation: the small model proposes
                # `num_speculative_tokens` ahead and the target verifies them in
                # one forward pass. Accepted tokens are sampled from the target's
                # own distribution, so quality is unchanged.
                cmd += [
                    "--speculative-config",
                    json.dumps(
                        {
                            "model": spec.speculative_draft,
                            "num_speculative_tokens": spec.speculative_tokens,
                        }
                    ),
                ]
        elif spec.task == "embed":
            cmd += ["--task", "embed"]
        elif spec.task == "rerank":
            cmd += ["--task", "score"]

        if params["quantization"]:
            cmd += ["--quantization", params["quantization"]]

    elif config.runtime.engine == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            spec.id,
            "--host",
            host,
            "--port",
            port,
            "--context-length",
            str(params["max_model_len"]),
            "--mem-fraction-static",
            str(params["gpu_memory_utilization"]),
            "--tp-size",
            str(params["tensor_parallel_size"]),
        ]
        if spec.task == "generate":
            cmd += [
                "--reasoning-parser",
                QWEN_REASONING_PARSER,
                "--tool-call-parser",
                QWEN_TOOL_PARSER,
            ]
            if spec.speculative_draft:
                cmd += [
                    "--speculative-algorithm",
                    "EAGLE",
                    "--speculative-draft-model-path",
                    spec.speculative_draft,
                    "--speculative-num-steps",
                    str(spec.speculative_tokens),
                ]
        elif spec.task in ("embed", "rerank"):
            cmd += ["--is-embedding"]

        if params["quantization"]:
            cmd += ["--quantization", params["quantization"]]
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ServerStartError(f"unknown engine: {config.runtime.engine}")

    return cmd + list(params["extra_args"])


class ServerProcess:
    """Owns a spawned engine process and shuts it down on exit."""

    def __init__(self, process: subprocess.Popen, key: str, base_url: str) -> None:
        self._process = process
        self.key = key
        self.base_url = base_url
        atexit.register(self.stop)

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_running(self) -> bool:
        return self._process.poll() is None

    def stop(self, timeout: float = 20.0) -> None:
        if self._process.poll() is not None:
            return
        log.info("stopping %s server (pid %s)", self.key, self._process.pid)
        # Kill the whole group: vLLM spawns worker children that outlive the
        # parent if signalled individually.
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            self._process.terminate()

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("%s did not exit in %ss, killing", self.key, timeout)
            self._process.kill()


def ensure_server(
    config: Config,
    key: str,
    spec: ModelSpec,
    *,
    log_dir: str = ".newal",
) -> ServerProcess | None:
    """Make sure a server for ``spec`` is reachable, starting one if needed.

    Returns the spawned process, or ``None`` when a server was already up (in
    which case we must not manage its lifetime).
    """
    if is_server_up(spec.base_url):
        log.info("reusing server already listening at %s for %s", spec.base_url, key)
        return None

    if not (spec.autostart and config.runtime.autostart):
        raise ServerStartError(
            f"no server at {spec.base_url} for model {key!r} and autostart is off. "
            f"Start one manually:\n  {' '.join(build_command(config, key, spec))}"
        )

    cmd = build_command(config, key, spec)
    log.info("starting %s: %s", key, " ".join(cmd))

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"server-{key}.log")
    stdout = open(log_path, "ab", buffering=0)  # noqa: SIM115 - lives with process

    popen_kwargs: dict[str, object] = {"stdout": stdout, "stderr": subprocess.STDOUT}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        engine = config.runtime.engine
        raise ServerStartError(
            f"could not launch {engine}. Install it with: pip install {engine}"
        ) from exc

    server = ServerProcess(process, key, spec.base_url)
    _wait_until_ready(server, config, spec, log_path)
    return server


def _wait_until_ready(
    server: ServerProcess, config: Config, spec: ModelSpec, log_path: str
) -> None:
    deadline = time.monotonic() + config.runtime.startup_timeout_s
    while time.monotonic() < deadline:
        if not server.is_running():
            raise ServerStartError(
                f"server for {server.key!r} exited during startup. See {log_path}. "
                "Common causes: not enough VRAM (lower runtime.max_model_len or "
                "gpu_memory_utilization, or disable a pool member), or a wrong model id."
            )
        if is_server_up(spec.base_url):
            log.info("%s ready at %s", server.key, spec.base_url)
            return
        time.sleep(2.0)

    server.stop()
    raise ServerStartError(
        f"server for {server.key!r} did not become ready within "
        f"{config.runtime.startup_timeout_s}s. First run downloads weights, which "
        "can take a while -- raise runtime.startup_timeout_s and try again."
    )
