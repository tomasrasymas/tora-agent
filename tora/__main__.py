"""Entrypoint: `python -m tora` (or the installed `tora` console script).

Run with:
  uv run tora
"""

import uvicorn

from .app import create_app
from .config import load_settings


def main() -> None:
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
