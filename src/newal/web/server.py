"""Running the web UI."""

from __future__ import annotations

import logging
import threading
import webbrowser

from ..config import Config

log = logging.getLogger(__name__)


class WebUnavailable(RuntimeError):
    pass


def serve(
    config: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    autostart: bool = True,
    open_browser: bool = True,
) -> None:
    """Run the UI until interrupted.

    Binds to loopback by default and stays there unless told otherwise. The
    agent can edit files and, depending on `tools.shell_policy`, run commands;
    anyone who can reach this port inherits that, with no authentication in
    front of it.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise WebUnavailable(
            "the web UI needs extra packages: pip install 'newal[web]'"
        ) from exc

    from .app import create_app

    app = create_app(config, autostart=autostart)

    if open_browser:
        url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
