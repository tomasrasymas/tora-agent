"""Tool abstraction and registry.

A `Tool` bundles an OpenAI function schema with the Python callable that
implements it. A `ToolRegistry` holds a set of tools and knows how to expose
their schemas to the model and dispatch a call by name.
"""

from collections.abc import Callable
from dataclasses import dataclass

# A tool implementation: called with the JSON arguments as keywords, returns
# anything JSON-serializable (the result is sent back to the model).
ToolFn = Callable[..., object]


@dataclass(frozen=True)
class Tool:
    """One callable tool plus the metadata the model needs to call it."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function arguments
    fn: ToolFn

    @property
    def schema(self) -> dict:
        """The OpenAI `tools=[...]` entry describing this tool."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A named collection of tools, with schema export and dispatch."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def schemas(self) -> list[dict]:
        """All tool schemas, ready to pass as the `tools` argument."""

        return [tool.schema for tool in self._tools.values()]

    def run(self, name: str, args: dict) -> object:
        """Execute tool `name` with `args`, returning its result or an error.

        Errors are returned (not raised) so they can be fed back to the model
        as the tool result.
        """

        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        try:
            return tool.fn(**args)
        except Exception as e:
            return {"error": str(e)}
