"""Interactive terminal front end."""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .agent import Agent, Toolbox, TranscriptWriter, describe_command
from .backends import BackendError
from .config import Config, load_config
from .media import Attachment, AttachmentError, prepare_attachments
from .memory import RepoIndex, build_index
from .models import ModelPool
from .training import FORMATS, export

app = typer.Typer(add_completion=False, help="newal -- local multimodal coding assistant")
console = Console()

HELP_TEXT = """\
[bold]Commands[/bold]
  /img PATH [PATH...]   attach image(s) to the next message
  /vid PATH [PATH...]   attach video(s) to the next message
  /files                show what is currently attached
  /clear                drop pending attachments
  /index                re-index the workspace
  /models               show the model pool and per-model token usage
  /routing              show how routing is configured and what it has learned
  /notes                show what the assistant remembers about this project
  /reset                start a fresh conversation (memory is kept)
  /usage                show token usage for this session
  /help                 this list
  /exit                 quit

Anything else is sent to the model. Paths may be quoted if they contain spaces.
"""


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _make_event_printer(config: Config):
    styles = {
        "phase": "bold cyan",
        "route": "blue",
        "thinking": "dim italic",
        "tool_call": "yellow",
        "tool_result": "dim",
        "verify": "bold magenta",
        "warning": "bold red",
    }

    def emit(kind: str, text: str) -> None:
        if kind == "assistant":
            console.print(Markdown(text))
            return
        if kind == "thinking" and not config.ui.show_thinking:
            return
        prefix = {"tool_call": "→", "tool_result": "  ", "warning": "!",
                  "route": "⇢"}.get(kind, "·")
        console.print(f"[{styles.get(kind, 'dim')}]{prefix} {text}[/]")

    return emit


def _make_approver() -> Any:
    def approve(tool_name: str, detail: str) -> bool:
        console.print(
            Panel(
                describe_command(detail),
                title=f"[bold yellow]{tool_name}[/] wants to run this",
                border_style="yellow",
            )
        )
        try:
            answer = console.input("[bold]Allow? [y/N][/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    return approve


def _build_session(
    config: Config, *, autostart: bool
) -> tuple[Agent, ModelPool, RepoIndex | None]:
    root = config.workspace_path()

    config.runtime.autostart = autostart
    with console.status("[cyan]starting model pool..."):
        pool = ModelPool(config, on_progress=lambda msg: console.print(f"[dim]{msg}[/]"))
    for line in pool.describe():
        console.print(f"[dim]  {line}[/]")

    index: RepoIndex | None = None
    if config.memory.enabled:
        index = build_index(root, config.memory)
        # Retrieval models come from the pool, so attach before the first pass.
        index.attach_models(embedder=pool.embedder, reranker=pool.reranker)
        with console.status("[cyan]indexing workspace..."):
            stats = index.refresh()
        detail = f"{stats.chunks_embedded} embedded, " if stats.chunks_embedded else ""
        console.print(
            f"[dim]indexed {stats.files_indexed} file(s), "
            f"{index.store.chunk_count()} chunks, {detail}"
            f"{stats.files_skipped} unchanged[/]"
        )

    # The classifier needs the store open so past outcomes join the seed set.
    learned = index.store.routing_exemplars(config.router.max_learned_exemplars) if index else []
    if pool.build_classifier(learned) is not None:
        detail = f" (+{len(learned)} learned)" if learned else ""
        console.print(f"[dim]router: {config.router.mode} mode{detail}[/]")

    toolbox = Toolbox(config.tools, index=index, approve=_make_approver())
    agent = Agent(config, pool, toolbox, index=index, on_event=_make_event_printer(config))
    return agent, pool, index


def _resolve_attachment_paths(raw: str) -> list[str]:
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _handle_command(
    line: str, agent: Agent, index: RepoIndex | None, pending: list[str]
) -> bool:
    """Handle a /command. Returns False when the session should end."""
    command, _, rest = line.partition(" ")
    command = command.lower()

    if command in {"/exit", "/quit"}:
        return False

    if command == "/help":
        console.print(HELP_TEXT)

    elif command in {"/img", "/vid"}:
        paths = _resolve_attachment_paths(rest)
        if not paths:
            console.print("[red]usage: /img PATH [PATH...][/]")
        for path in paths:
            resolved = Path(path).expanduser()
            if not resolved.is_file():
                console.print(f"[red]not found: {resolved}[/]")
                continue
            pending.append(str(resolved))
            console.print(f"[green]attached[/] {resolved.name}")

    elif command == "/files":
        console.print("\n".join(pending) if pending else "[dim]nothing attached[/]")

    elif command == "/clear":
        pending.clear()
        console.print("[dim]attachments cleared[/]")

    elif command == "/index":
        if index is None:
            console.print("[red]memory is disabled[/]")
        else:
            with console.status("[cyan]re-indexing..."):
                stats = index.refresh()
            console.print(
                f"[green]indexed[/] {stats.files_indexed} changed, "
                f"{stats.files_removed} removed, {stats.chunks_written} chunks written"
            )

    elif command == "/notes":
        if index is None:
            console.print("[red]memory is disabled[/]")
        else:
            notes = index.store.recent_notes(limit=30)
            if not notes:
                console.print("[dim]no notes stored yet[/]")
            else:
                table = Table("topic", "note", box=None, show_header=True)
                for note in notes:
                    table.add_row(note.topic, note.content)
                console.print(table)

    elif command == "/reset":
        agent.reset()
        pending.clear()
        console.print("[dim]conversation reset[/]")

    elif command == "/models":
        table = Table("model", "prompt", "completion", "total", box=None)
        for line in agent.pool.describe():
            console.print(f"[dim]{line}[/]")
        for key, usage in sorted(agent.usage_by_model.items()):
            table.add_row(
                key, f"{usage.prompt_tokens:,}",
                f"{usage.completion_tokens:,}", f"{usage.total_tokens:,}"
            )
        if agent.usage_by_model:
            console.print(table)

    elif command == "/routing":
        settings = agent.config.router
        console.print(
            f"strategy: {settings.strategy}  mode: {settings.mode}  "
            f"threshold: {settings.escalate_threshold:.2f} "
            f"±{settings.uncertainty_band:.2f}"
        )
        console.print(
            f"classifier: "
            f"{'active' if agent.pool.router.classifier else 'not in use (heuristic only)'}"
        )
        if index is not None:
            counts = index.store.routing_outcome_counts()
            learned = sum(counts.values())
            console.print(
                f"learned from {learned} outcome(s): "
                + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none yet")
            )

    elif command == "/usage":
        usage = agent.usage
        console.print(
            f"prompt: {usage.prompt_tokens:,}  "
            f"completion: {usage.completion_tokens:,}  "
            f"total: {usage.total_tokens:,}"
        )
        for key, per in sorted(agent.usage_by_model.items()):
            console.print(f"[dim]  {key}: {per.total_tokens:,}[/]")

    else:
        console.print(f"[red]unknown command {command}[/] -- try /help")

    return True


def _prepare_pending(config: Config, pending: list[str]) -> list[Attachment]:
    if not pending:
        return []
    with console.status("[cyan]preparing attachments..."):
        try:
            attachments = prepare_attachments(list(pending), config.media)
        except (AttachmentError, FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]attachment error:[/] {exc}")
            return []
    for attachment in attachments:
        console.print(f"[dim]  {attachment.kind}: {attachment.summary}[/]")
    return attachments


@app.command()
def chat(
    config_path: Path = typer.Option(None, "--config", "-c", help="Config file to load."),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Workspace root."),
    model: str = typer.Option(None, "--model", "-m", help="Override the model id."),
    no_autostart: bool = typer.Option(
        False, "--no-autostart", help="Do not launch an inference server."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start an interactive session."""
    _configure_logging(verbose)

    overrides: dict[str, Any] = {}
    if workspace:
        overrides["tools"] = {"workspace_root": str(workspace)}
    if model:
        # --model overrides the strongest tier, which is what "the model" means
        # to someone who has not opened the pool config.
        base = load_config(config_path, overrides=overrides, use_env=True)
        key, _ = base.strongest_model()
        overrides["models"] = {key: {"id": model}}
    config = load_config(config_path, overrides=overrides)

    strongest_key, strongest = config.strongest_model()
    console.print(
        Panel(
            f"[bold]newal[/] v{__version__}\n"
            f"pool: [cyan]{len(config.enabled_models())} model(s)[/], "
            f"routing: [cyan]{config.router.strategy}[/]\n"
            f"primary: [cyan]{strongest.id}[/] ({strongest_key})\n"
            f"workspace: [cyan]{config.workspace_path()}[/]\n"
            f"[dim]/help for commands[/]",
            border_style="cyan",
        )
    )

    try:
        agent, pool, index = _build_session(config, autostart=not no_autostart)
    except (BackendError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]startup failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    transcript: TranscriptWriter | None = None
    if config.ui.save_transcripts:
        root = config.workspace_path()
        directory = Path(config.ui.transcript_dir)
        transcript = TranscriptWriter.create(
            directory if directory.is_absolute() else root / directory
        )
        console.print(f"[dim]transcript: {transcript.path}[/]")

    pending: list[str] = []
    try:
        while True:
            try:
                line = console.input("\n[bold green]>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue

            if line.startswith("/"):
                if not _handle_command(line, agent, index, pending):
                    break
                continue

            attachments = _prepare_pending(config, pending)
            pending.clear()

            try:
                result = agent.run(line, attachments)
            except KeyboardInterrupt:
                console.print("[yellow]interrupted[/]")
                continue
            except BackendError as exc:
                console.print(f"[bold red]backend error:[/] {exc}")
                continue

            if result.escalations:
                console.print(f"[dim]escalated {result.escalations}x[/]")
            if result.files_written:
                console.print(f"[dim]changed: {', '.join(result.files_written)}[/]")
            if result.verification and not result.verification.skipped:
                status = "[green]passed[/]" if result.verification.passed else "[red]FAILED[/]"
                console.print(f"[dim]verify ({result.verification.command}):[/] {status}")
            if config.ui.show_token_usage:
                console.print(f"[dim]{result.usage.total_tokens:,} tokens · "
                              f"{result.steps} step(s)[/]")

            if transcript is not None:
                verification = None
                if result.verification and not result.verification.skipped:
                    verification = (
                        result.verification.command,
                        result.verification.passed,
                    )
                transcript.record(
                    prompt=line,
                    response=result.text,
                    routes=result.routes,
                    attachments=[a.source for a in attachments],
                    files_written=result.files_written,
                    verification=verification,
                    usage_by_model={
                        key: usage.total_tokens
                        for key, usage in result.usage_by_model.items()
                    },
                    steps=result.steps,
                    escalations=result.escalations,
                )
    finally:
        pool.close()
        console.print("[dim]bye[/]")


@app.command()
def serve(
    config_path: Path = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the inference server only, and keep it running."""
    _configure_logging(verbose)
    config = load_config(config_path)

    console.print(f"[cyan]starting {len(config.enabled_models())} server(s)...[/]")
    pool = ModelPool(config, on_progress=lambda msg: console.print(f"[dim]{msg}[/]"))
    for line in pool.describe():
        console.print(f"[green]ready[/] {line}")
    console.print("[dim]Ctrl-C to stop[/]")
    try:
        import time

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        pool.close()


@app.command()
def index(
    config_path: Path = typer.Option(None, "--config", "-c"),
    workspace: Path = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Index the workspace without starting a chat session."""
    overrides: dict[str, Any] = {}
    if workspace:
        overrides["tools"] = {"workspace_root": str(workspace)}
    config = load_config(config_path, overrides=overrides)

    repo_index = build_index(config.workspace_path(), config.memory)
    stats = repo_index.refresh()
    console.print(
        f"indexed {stats.files_indexed} file(s), skipped {stats.files_skipped}, "
        f"removed {stats.files_removed}\n"
        f"{repo_index.store.chunk_count()} chunks in {repo_index.store.db_path}"
    )


@app.command()
def config(
    config_path: Path = typer.Option(None, "--config", "-c"),
    section: str = typer.Option(None, "--section", "-s", help="Show one section only."),
) -> None:
    """Print the merged configuration.

    Config comes from four layers, so "why is it using that model?" is a common
    question. This prints what they actually resolved to.
    """
    try:
        resolved = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]invalid config:[/] {exc}")
        raise typer.Exit(code=1) from exc

    data = resolved.model_dump()
    if section:
        if section not in data:
            console.print(
                f"[red]no section {section!r}[/] -- try one of: {', '.join(data)}"
            )
            raise typer.Exit(code=1)
        data = {section: data[section]}

    console.print(Syntax(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                         "yaml", theme="ansi_dark", background_color="default"))

    if section is None:
        pool = ", ".join(resolved.enabled_models()) or "(none)"
        console.print(f"[dim]enabled pool members: {pool}[/]")


@app.command()
def web(
    config_path: Path = typer.Option(None, "--config", "-c"),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Workspace root."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8800, "--port", "-p"),
    no_autostart: bool = typer.Option(
        False, "--no-autostart", help="Do not launch an inference server."
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Serve the browser UI."""
    _configure_logging(verbose)

    overrides: dict[str, Any] = {}
    if workspace:
        overrides["tools"] = {"workspace_root": str(workspace)}
    config = load_config(config_path, overrides=overrides)

    from .web.server import WebUnavailable, serve

    if host not in ("127.0.0.1", "localhost", "::1"):
        # The agent edits files and, under shell_policy, runs commands. Nothing
        # here authenticates, so a non-loopback bind hands that to the network.
        console.print(
            Panel(
                f"[bold]{host}[/] 로 바인딩합니다. 이 UI에는 인증이 없고,\n"
                f"접근할 수 있는 사람은 워크스페이스 파일 수정과\n"
                f"셸 실행(shell_policy={config.tools.shell_policy})을 그대로 물려받습니다.",
                title="[bold red]경고[/]",
                border_style="red",
            )
        )

    console.print(f"[cyan]http://localhost:{port}[/] 에서 실행합니다. Ctrl-C로 종료.")
    try:
        serve(
            config,
            host=host,
            port=port,
            autostart=not no_autostart,
            open_browser=not no_browser,
        )
    except WebUnavailable as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        pass


@app.command(name="export")
def export_data(
    fmt: str = typer.Option(
        None, "--format", "-f", help=f"One of: {', '.join(FORMATS)}."
    ),
    out: Path = typer.Option(None, "--out", "-o", help="Destination JSONL file."),
    config_path: Path = typer.Option(None, "--config", "-c"),
    workspace: Path = typer.Option(None, "--workspace", "-w"),
    include_unverified: bool = typer.Option(
        False,
        "--include-unverified",
        help="SFT only: also export turns the test suite never confirmed.",
    ),
    stats: bool = typer.Option(False, "--stats", help="Show what has been captured."),
    clear: bool = typer.Option(False, "--clear", help="Delete all captured turns."),
) -> None:
    """Export captured interactions as a fine-tuning dataset."""
    overrides: dict[str, Any] = {}
    if workspace:
        overrides["tools"] = {"workspace_root": str(workspace)}
    config = load_config(config_path, overrides=overrides)

    repo_index = build_index(config.workspace_path(), config.memory)
    store = repo_index.store

    if stats or (not fmt and not clear):
        counts = store.training_counts()
        table = Table("captured", "count", box=None)
        for key, value in counts.items():
            table.add_row(key.replace("_", " "), f"{value:,}")
        console.print(table)
        console.print(f"[dim]{store.db_path}[/]")
        if not fmt and not clear:
            console.print(
                f"[dim]export with: newal export --format {'|'.join(FORMATS)} "
                "--out data.jsonl[/]"
            )
        if not clear:
            return

    if clear:
        counts = store.training_counts()
        total = counts["turns"] + counts["repair_pairs"]
        if total == 0:
            console.print("[dim]nothing captured to clear[/]")
            return
        answer = console.input(
            f"[bold yellow]delete {total:,} captured record(s)? [y/N][/] "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            console.print("[dim]kept[/]")
            return
        store.clear_training_data()
        console.print("[green]cleared[/] (notes and the routing set were kept)")
        return

    if fmt not in FORMATS:
        console.print(f"[red]--format must be one of:[/] {', '.join(FORMATS)}")
        raise typer.Exit(code=1)
    if out is None:
        console.print("[red]--out is required[/]")
        raise typer.Exit(code=1)

    written = export(store, fmt, out, include_unverified=include_unverified)
    if written == 0:
        console.print(
            f"[yellow]no {fmt} samples to export.[/] "
            "Keep using newal -- capture happens as you work."
        )
        return
    console.print(f"[green]wrote {written:,} {fmt} sample(s)[/] to {out}")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"newal {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
