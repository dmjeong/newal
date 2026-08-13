"""Interactive terminal front end."""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent import Agent, Toolbox, describe_command
from .backends import BackendError, build_backend
from .config import Config, load_config
from .media import Attachment, AttachmentError, prepare_attachments
from .memory import RepoIndex, build_index

app = typer.Typer(add_completion=False, help="newal -- local multimodal coding assistant")
console = Console()

HELP_TEXT = """\
[bold]Commands[/bold]
  /img PATH [PATH...]   attach image(s) to the next message
  /vid PATH [PATH...]   attach video(s) to the next message
  /files                show what is currently attached
  /clear                drop pending attachments
  /index                re-index the workspace
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
        prefix = {"tool_call": "→", "tool_result": "  ", "warning": "!"}.get(kind, "·")
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


def _build_session(config: Config, *, autostart: bool) -> tuple[Agent, RepoIndex | None]:
    root = config.workspace_path()

    index: RepoIndex | None = None
    if config.memory.enabled:
        index = build_index(root, config.memory)
        with console.status("[cyan]indexing workspace..."):
            stats = index.refresh()
        console.print(
            f"[dim]indexed {stats.files_indexed} file(s), "
            f"{index.store.chunk_count()} chunks "
            f"({stats.files_skipped} unchanged)[/]"
        )

    config.backend.autostart = autostart
    with console.status(f"[cyan]connecting to {config.model.id}..."):
        backend = build_backend(config)

    toolbox = Toolbox(config.tools, index=index, approve=_make_approver())
    agent = Agent(config, backend, toolbox, index=index, on_event=_make_event_printer(config))
    return agent, index


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

    elif command == "/usage":
        usage = agent.usage
        console.print(
            f"prompt: {usage.prompt_tokens:,}  "
            f"completion: {usage.completion_tokens:,}  "
            f"total: {usage.total_tokens:,}"
        )

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
        overrides["model"] = {"id": model}
    config = load_config(config_path, overrides=overrides)

    console.print(
        Panel(
            f"[bold]newal[/] v{__version__}\n"
            f"model: [cyan]{config.model.id}[/]\n"
            f"workspace: [cyan]{config.workspace_path()}[/]\n"
            f"[dim]/help for commands[/]",
            border_style="cyan",
        )
    )

    try:
        agent, index = _build_session(config, autostart=not no_autostart)
    except (BackendError, RuntimeError) as exc:
        console.print(f"[bold red]startup failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

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

            if result.files_written:
                console.print(f"[dim]changed: {', '.join(result.files_written)}[/]")
            if result.verification and not result.verification.skipped:
                status = "[green]passed[/]" if result.verification.passed else "[red]FAILED[/]"
                console.print(f"[dim]verify ({result.verification.command}):[/] {status}")
            if config.ui.show_token_usage:
                console.print(f"[dim]{result.usage.total_tokens:,} tokens · "
                              f"{result.steps} step(s)[/]")
    finally:
        agent.backend.close()
        console.print("[dim]bye[/]")


@app.command()
def serve(
    config_path: Path = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the inference server only, and keep it running."""
    _configure_logging(True if verbose else False)
    config = load_config(config_path)

    from .backends.launcher import ensure_server

    console.print(f"[cyan]starting {config.backend.engine} for {config.model.id}...[/]")
    server = ensure_server(config.model, config.backend, log_path=".newal/server.log")
    if server is None:
        console.print(f"[yellow]a server is already running at {config.backend.base_url}[/]")
        return

    console.print(f"[green]ready[/] at {config.backend.base_url} (pid {server.pid})")
    console.print("[dim]Ctrl-C to stop[/]")
    try:
        while server.is_running():
            import time

            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


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
def version() -> None:
    """Print the version."""
    console.print(f"newal {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
