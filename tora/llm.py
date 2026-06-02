"""Client for the llama.cpp server (OpenAI-compatible).

This module is only about *talking to the model*: opening the connection,
streaming a single turn, and accumulating the tool calls the model emits as it
streams. The agent loop that decides what to do with those turns — building the
prompt, running tools, looping — lives in ``agent.py``.
"""

from collections.abc import Generator
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from .schemas import EventType, StreamEvent


class FunctionFragment(Protocol):
    """The function part of a streamed tool-call delta.

    A structural type (not tied to any SDK): anything exposing optional `name`
    and `arguments` attributes works, so the accumulator stays provider-neutral.
    """

    name: str | None
    arguments: str | None


@dataclass
class ToolCall:
    """A tool call accumulated from streamed fragments.

    Streaming APIs split a tool call across chunks: the id and name usually
    arrive once, while the JSON arguments come piece by piece. Callers feed each
    fragment to `update`, which tolerates absent or partial fragments.
    """

    id: str | None = None
    name: str | None = None
    arguments: str = ""

    def update(
        self,
        *,
        id: str | None = None,
        function: FunctionFragment | None = None,
    ) -> None:
        """Merge one streamed fragment into this call.

        Each field is applied only when present, so empty fragments (including a
        missing `function`) are no-ops and `arguments` accumulates across chunks.
        """

        if id:
            self.id = id
        if function is not None:
            if function.name:
                self.name = function.name
            if function.arguments:
                self.arguments += function.arguments

    @property
    def complete(self) -> bool:
        """True once we have the id and name needed to dispatch the call."""

        return bool(self.id and self.name)


class LLMClient:
    """Thin wrapper around the OpenAI SDK pointed at our llama.cpp server."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)

    def model_id(self) -> str:
        try:
            return self._client.models.list().data[0].id
        except Exception:
            return "unknown"

    def stream_turn(
        self,
        conversation: list[dict],
        tool_schemas: list[dict],
    ) -> Generator[str, None, tuple[str, dict[int, ToolCall]]]:
        """Stream one model turn.

        Yields the turn's content/reasoning events (as NDJSON) and *returns* (via
        ``return``, consumed by ``yield from``) a ``(content, tool_calls)`` pair:
        the full answer text accumulated this turn, and the tool calls the model
        requested (keyed by stream index). The caller supplies the full
        conversation and the tool schemas to expose — this method knows nothing
        about the prompt, the tool registry, or the surrounding loop.
        """

        stream = self._client.chat.completions.create(
            model=self.model_id(),
            messages=conversation,
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.1,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        content = ""
        tool_calls: dict[int, ToolCall] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            thinking = getattr(delta, "reasoning_content", None)
            if thinking:
                yield StreamEvent(type=EventType.THINKING, text=thinking)()

            if delta.content:
                content += delta.content
                yield StreamEvent(type=EventType.CONTENT, text=delta.content)()

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    call = tool_calls.setdefault(tool_call_delta.index, ToolCall())
                    call.update(
                        id=tool_call_delta.id,
                        function=tool_call_delta.function,
                    )

        return content, tool_calls
