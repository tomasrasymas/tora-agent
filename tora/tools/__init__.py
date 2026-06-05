"""Tools the model can call.

Each tool lives in its own module and exposes a `Tool` instance. This package
assembles them into the single `registry` that `LLMClient` always uses.

To add a tool: create `tools/<name>.py` with a `Tool`, then add it here.
"""

from .base import Tool, ToolRegistry
from .bash import bash_tool
from .memory import forget_tool, recall_tool, remember_tool
from .skill import load_skill_tool
from .web import web_fetch_tool, web_search_tool

registry = ToolRegistry(
    [
        bash_tool,
        load_skill_tool,
        remember_tool,
        recall_tool,
        forget_tool,
        web_search_tool,
        web_fetch_tool,
    ]
)

__all__ = ["Tool", "ToolRegistry", "registry"]
