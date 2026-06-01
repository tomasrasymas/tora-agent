"""Pydantic models for the HTTP API."""

from enum import Enum

from pydantic import BaseModel


class Message(BaseModel):
    """A chat message in the OpenAI shape.

    Plain user/assistant turns use `role` + `content`. Tool use adds two more
    shapes, which the frontend stores in its history and sends back so the model
    keeps track of what it called:

      - assistant requesting tools: `content` (maybe None) + `tool_calls`
      - a tool's result: `role="tool"` + `tool_call_id` + `content`
    """

    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class Messages(BaseModel):
    """Body of POST /api/chat — the full conversation so far."""

    messages: list[Message]


class EventType(str, Enum):
    """Kind of streaming event sent to the browser over NDJSON.

    - ``thinking``     — the model's reasoning (collapsible block)
    - ``content``      — answer tokens
    - ``tool_call``    — tools the model decided to call this turn
    - ``tool_result``  — the result of running one tool
    """

    THINKING = "thinking"
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class StreamEvent(BaseModel):
    """One streaming event in the NDJSON response of POST /api/chat.

    Which fields are set depends on `type`: `text` for thinking/content (and the
    JSON result of a tool_result), `calls` for tool_call, and `id`/`name` to tie
    a tool_result back to its call. Unused fields are dropped on serialization.
    """

    type: EventType
    text: str | None = None
    calls: list[dict] | None = None  # tool_call: [{id, name, arguments}, ...]
    id: str | None = None  # tool_result: which call it answers
    name: str | None = None  # tool_result: the tool's name

    def to_ndjson(self) -> str:
        """Serialize as a single newline-terminated JSON line."""

        return self.model_dump_json() + "\n"

    def __call__(self) -> str:
        """Shorthand for `to_ndjson()` — `StreamEvent(...)()` yields the line."""

        return self.to_ndjson()
