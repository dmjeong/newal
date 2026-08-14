"""Browser front end for newal.

The same `Agent` the terminal drives, reached over HTTP instead. The UI is
plain HTML, CSS and JavaScript served straight from the package -- there is no
bundler and no Node toolchain, so `pip install` remains the whole setup.
"""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def create_app(*args, **kwargs):
    """Lazily build the FastAPI app so the web extra stays optional."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


def serve(*args, **kwargs):
    from .server import serve as _serve

    return _serve(*args, **kwargs)
