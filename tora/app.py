"""Application factory: wires settings, routes, and the static frontend together.

A thin FastAPI app that:
  1. Serves the chat frontend (web/index.html).
  2. Proxies chat requests to the llama.cpp server and streams tokens back.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .llm import LLMClient
from .routes import create_chat_router, create_model_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and return the FastAPI application.

    Pass `settings` explicitly (e.g. in tests); otherwise they're loaded from
    the environment.
    """

    settings = settings or load_settings()

    app = FastAPI(title="TORA agent")

    llm = LLMClient(settings.llm_base_url)
    app.include_router(create_chat_router(llm))
    app.include_router(create_model_router(llm))

    # Mounted last so the API routes above take precedence over "/".
    app.mount(
        "/",
        StaticFiles(directory=settings.web_dir, html=True),
        name="web",
    )

    return app
