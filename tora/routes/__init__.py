"""HTTP route factories for the TORA agent.

Each module exposes a ``create_*_router(llm)`` factory that returns an
``APIRouter`` wired to a given LLM client. The app factory includes them all.
"""

from .chat import create_chat_router
from .model import create_model_router

__all__ = ["create_chat_router", "create_model_router"]
