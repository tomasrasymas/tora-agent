"""Tools the model can call.

Each tool lives in its own module and exposes a `Tool` instance. This package
assembles them into `default_registry`, which `LLMClient` uses by default.

To add a tool: create `tools/<name>.py` with a `Tool`, then add it here.
"""

from .base import Tool, ToolRegistry
from .shell import shell_tool
from .weather import weather_tool

default_registry = ToolRegistry([weather_tool, shell_tool])

__all__ = ["Tool", "ToolRegistry", "default_registry"]
