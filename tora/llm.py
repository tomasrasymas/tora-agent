"""Client for the llama.cpp server (OpenAI-compatible)."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from .schemas import EventType, Message, StreamEvent
from .tools import ToolRegistry, default_registry


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


SYSTEM_PROMPT = """\
You are TORA, a personal AI assistant for athletes. You help them manage the \
everyday details of athletic life — things like supplements, training, and \
nutrition — using your built-in tools.

You're an assistant, not a coach: help the user get things done and answer what \
they ask, rather than prescribing training programs or pushing advice they \
didn't request. Be concise, practical, and friendly. Use the available tools \
when they help, and when a request is missing detail you need, ask one brief \
clarifying question first. Don't give medical diagnoses — for injuries or health \
concerns, recommend seeing a qualified professional."""


class LLMClient:
    """Thin wrapper around the OpenAI SDK pointed at our llama.cpp server."""

    # Safety cap on chained tool calls, so a model that keeps requesting tools
    # (or loops on a failing one) can't stream forever.
    MAX_TOOL_ROUNDS = 8

    def __init__(
        self,
        base_url: str,
        timeout: float = 600.0,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)
        self._tools = tools or default_registry

    def model_id(self) -> str:
        try:
            return self._client.models.list().data[0].id
        except Exception:
            return "unknown"

    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Stream a chat completion as NDJSON, running tools as the model asks.

        Stateless: the conversation comes from the client each request (the
        frontend stores tool calls and results in its history and sends them
        back). Each loop iteration is one model turn — if it requests tool
        calls, we run them, append the results, and let it respond again, up to
        ``MAX_TOOL_ROUNDS``.
        """

        try:
            # The system prompt is injected here each request — it's not part of
            # the history the frontend stores and sends back.
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *(m.model_dump(exclude_none=True) for m in messages),
            ]

            for round_num in range(self.MAX_TOOL_ROUNDS + 1):
                tool_calls = yield from self._stream_turn(conversation)

                # Only complete calls (id + name present) are runnable.
                calls = [call for call in tool_calls.values() if call.complete]
                if not calls:
                    return  # the model gave a final answer

                if round_num == self.MAX_TOOL_ROUNDS:
                    yield StreamEvent(
                        type=EventType.CONTENT,
                        text="\n[stopped: too many tool-call rounds]\n",
                    )()
                    return

                yield from self._run_tools(conversation, calls)

        except Exception as e:
            yield StreamEvent(
                type=EventType.CONTENT,
                text=f"[error talking to model at {self.base_url}: {e}]",
            )()

    def _stream_turn(self, conversation: list[dict]) -> Iterator[str]:
        """Stream one model turn.

        Yields the turn's content/reasoning events and *returns* (via
        ``return``, consumed by ``yield from``) the tool calls the model
        requested, keyed by their stream index.
        """

        stream = self._client.chat.completions.create(
            model=self.model_id(),
            messages=conversation,
            tools=self._tools.schemas,
            tool_choice="auto",
            temperature=0.1,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        tool_calls: dict[int, ToolCall] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            thinking = getattr(delta, "reasoning_content", None)
            if thinking:
                yield StreamEvent(type=EventType.THINKING, text=thinking)()

            if delta.content:
                yield StreamEvent(type=EventType.CONTENT, text=delta.content)()

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    call = tool_calls.setdefault(tool_call_delta.index, ToolCall())
                    call.update(
                        id=tool_call_delta.id,
                        function=tool_call_delta.function,
                    )

        return tool_calls

    def _run_tools(
        self,
        conversation: list[dict],
        calls: list[ToolCall],
    ) -> Iterator[str]:
        """Run `calls`, appending the assistant + tool messages to `conversation`.

        Emits a `tool_call` event (so the UI can show what's running) and a
        `tool_result` event per call. The conversation is mutated in place so the
        outer loop just calls the model again on the next iteration.
        """

        assistant_tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments or "{}",
                },
            }
            for call in calls
        ]
        conversation.append(
            {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls}
        )

        yield StreamEvent(
            type=EventType.TOOL_CALL,
            calls=[
                {"id": call.id, "name": call.name, "arguments": call.arguments or "{}"}
                for call in calls
            ],
        )()

        for call in calls:
            raw_arguments = call.arguments or "{}"

            try:
                args = json.loads(raw_arguments)
            except json.JSONDecodeError as e:
                result = {"error": f"Invalid tool arguments: {e}"}
            else:
                result = self._tools.run(call.name, args)

            result_text = json.dumps(result)
            conversation.append(
                {"role": "tool", "tool_call_id": call.id, "content": result_text}
            )

            yield StreamEvent(
                type=EventType.TOOL_RESULT,
                id=call.id,
                name=call.name,
                text=result_text,
            )()
