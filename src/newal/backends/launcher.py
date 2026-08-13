"""Start and health-check a local vLLM / SGLang server.

The assistant is meant to be launched with one command, so if nothing is
listening on ``backend.base_url`` we spawn the engine ourselves and wait for
the OpenAI-compatible endpoint to come up.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse

import httpx

from ..config import BackendConfig, ModelConfig

log = logging.getLogger(__name__)

# Reasoning/tool parsers Qwen3.5+ ships with. Without these, vLLM returns the
# raw <think> block as content and tool calls arrive as unparsed text.
QWEN_REASONING_PARSER = "qwen3"
QWEN_TOOL_PARSER = "hermes"


class ServerStartError(RuntimeError):
    pass


def _root_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_server_up(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if an OpenAI-compatible server answers at ``base_url``."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def build_command(model: ModelConfig, backend: BackendConfig) -> list[str]:
    """Build the engine command line. Kept pure so it is easy to test."""
    parsed = urlparse(backend.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = str(parsed.port or 8000)

    if backend.engine == "vllm":
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model.id,
            "--host",
            host,
            "--port",
            port,
            "--max-model-len",
            str(backend.max_model_len),
            "--gpu-memory-utilization",
            str(backend.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(backend.tensor_parallel_size),
            "--reasoning-parser",
            QWEN_REASONING_PARSER,
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            QWEN_TOOL_PARSER,
        ]
        if backend.quantization:
            cmd += ["--quantization", backend.quantization]
    elif backend.engine == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model.id,
            "--host",
            host,
            "--port",
            port,
            "--context-length",
            str(backend.max_model_len),
            "--mem-fraction-static",
            str(backend.gpu_memory_utilization),
            "--tp-size",
            str(backend.tensor_parallel_size),
            "--reasoning-parser",
            QWEN_REASONING_PARSER,
            "--tool-call-parser",
            QWEN_TOOL_PARSER,
        ]
        if backend.quantization:
            cmd += ["--quantization", backend.quantization]
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ServerStartError(f"unknown engine: {backend.engine}")

    return cmd + list(backend.extra_args)


class ServerProcess:
    """Owns a spawned engine process and shuts it down on exit."""

    def __init__(self, process: subprocess.Popen, base_url: str) -> None:
        self._process = process
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
        log.info("stopping inference server (pid %s)", self._process.pid)
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
            log.warning("server did not exit in %ss, killing", timeout)
            self._process.kill()


def ensure_server(
    model: ModelConfig,
    backend: BackendConfig,
    *,
    log_path: str | None = None,
) -> ServerProcess | None:
    """Make sure a server is reachable, starting one if needed.

    Returns the spawned process, or ``None`` when a server was already up (in
    which case we must not manage its lifetime).
    """
    if is_server_up(backend.base_url):
        log.info("reusing inference server already listening at %s", backend.base_url)
        return None

    if not backend.autostart:
        raise ServerStartError(
            f"no server at {backend.base_url} and backend.autostart is false. "
            f"Start one manually, e.g.:\n  {' '.join(build_command(model, backend))}"
        )

    module = "vllm" if backend.engine == "vllm" else "sglang"
    if shutil.which(sys.executable) is None:  # pragma: no cover - paranoia
        raise ServerStartError("cannot locate the current Python interpreter")

    cmd = build_command(model, backend)
    log.info("starting %s: %s", module, " ".join(cmd))

    stdout: int | object = subprocess.DEVNULL
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        stdout = open(log_path, "ab", buffering=0)  # noqa: SIM115 - lives with process

    popen_kwargs: dict[str, object] = {
        "stdout": stdout,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        raise ServerStartError(
            f"could not launch {module}. Install it with: pip install {module}"
        ) from exc

    server = ServerProcess(process, backend.base_url)
    _wait_until_ready(server, backend, log_path)
    return server


def _wait_until_ready(
    server: ServerProcess, backend: BackendConfig, log_path: str | None
) -> None:
    deadline = time.monotonic() + backend.startup_timeout_s
    while time.monotonic() < deadline:
        if not server.is_running():
            hint = f" See {log_path} for details." if log_path else ""
            raise ServerStartError(
                f"inference server exited during startup.{hint} "
                "Common causes: not enough VRAM (lower backend.max_model_len or "
                "gpu_memory_utilization), or the model id is wrong."
            )
        if is_server_up(backend.base_url):
            log.info("inference server ready at %s", backend.base_url)
            return
        time.sleep(2.0)

    server.stop()
    raise ServerStartError(
        f"server did not become ready within {backend.startup_timeout_s}s. "
        "First run downloads weights, which can take a while -- raise "
        "backend.startup_timeout_s and try again."
    )
