"""HTTP route factories for the TORA agent.

Each module exposes a ``create_*_router(...)`` factory that returns an
``APIRouter`` wired to its dependency — the chat router to an ``Agent`` + store,
the conversations router to the store, the model router to an ``LLMClient``. The
app factory includes them all.
"""

from .chat import create_chat_router
from .conversations import create_conversations_router
from .model import create_model_router

__all__ = [
    "create_chat_router",
    "create_conversations_router",
    "create_model_router",
]
