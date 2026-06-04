"""The main agent flow.

Orchestrates a chat turn: build the system prompt, ask the model (via
``LLMClient``) to stream a turn, and — when the model requests tools — run them
against the tool registry, feed the results back, and loop until the model
produces a final answer (bounded by ``MAX_TOOL_ROUNDS``).

Division of labour: ``llm.py`` handles *talking to the model*; this module
handles *what to do with the conversation*.
"""

import json
from collections.abc import Generator, Iterator
from datetime import datetime
from pathlib import Path

from . import memory, skills
from .llm import LLMClient, ToolCall
from .schemas import EventType, Message, StreamEvent
from .tools import registry

# The base system prompt lives in a Markdown file (tora/prompts/system.md) so its
# structure is easy to read and edit without touching Python. Loaded once at
# import — i.e. on startup — and reused for every request; the dynamic bits are
# spliced in by build_system_prompt() via {{...}} placeholders.
_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"
SYSTEM_PROMPT_TEMPLATE = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 chars/token) over the serialized messages.

    Used only to decide *when* to trim history before a request — the exact
    count comes back from the model as a usage event afterwards. Deliberately
    cheap (no tokenizer round-trip) and slightly conservative.
    """
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return chars // 4


def trim_to_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    """Drop the oldest turns until the history fits ``max_tokens``.

    Truncation only ever happens at a user message — the start of a turn — so we
    never orphan a tool result or leave an assistant ``tool_calls`` message
    without the tool outputs that answer it (a shape the model API rejects).
    Keeps the largest such suffix that fits; if even the last user turn is over
    budget, keeps that turn anyway (better than sending nothing — let the model
    server deal with it, and lean on the memory system / summarization later).
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages
    # User messages are the only safe truncation boundaries.
    user_starts = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    for start in user_starts:  # earliest boundary that fits wins (keeps the most)
        if _estimate_tokens(messages[start:]) <= max_tokens:
            return messages[start:]
    return messages[user_starts[-1] :] if user_starts else messages


def build_system_prompt() -> str:
    """Assemble the full system prompt for one request.

    The template (loaded once at startup) is static; the date/time, remembered
    facts, and skills listing are resolved live and substituted into their
    ``{{...}}`` placeholders so they reflect "now" and the currently discovered
    skills/memories. The memory and skills sections carry their own headers and
    collapse to "" when empty, so the placeholders just splice in cleanly.
    """
    now = datetime.now().astimezone()
    return (
        SYSTEM_PROMPT_TEMPLATE.replace("{{datetime}}", f"{now:%A, %d %B %Y, %H:%M %Z}")
        .replace("{{memory}}", memory.store.prompt_section())
        .replace("{{skills}}", skills.registry.prompt_section())
    )


class Agent:
    """Runs the chat/tool loop on top of an ``LLMClient``."""

    # Safety cap on chained tool calls, so a model that keeps requesting tools
    # (or loops on a failing one) can't stream forever.
    MAX_TOOL_ROUNDS = 8

    # Fraction of the model's context window to fill with the system prompt +
    # history, leaving headroom for the model's reply, this request's tool
    # rounds, and slack in the token estimate.
    CONTEXT_BUDGET = 0.7

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._tools = registry

    def stream_chat(
        self, messages: list[Message], thinking: bool = False
    ) -> Generator[str, None, list[dict]]:
        """Stream a chat completion as NDJSON, running tools as the model asks.

        ``messages`` is the conversation so far (loaded from storage by the
        caller). Each loop iteration is one model turn — if it requests tool
        calls, we run them, append the results, and let it respond again, up to
        ``MAX_TOOL_ROUNDS``. ``thinking`` enables the model's reasoning step for
        every turn of this request.

        *Returns* (via ``return``, captured by ``yield from``) the list of new
        messages produced this request — the assistant turn(s) and any tool
        results — so the caller can persist ``messages + new`` without re-parsing
        the stream.
        """

        new_messages: list[dict] = []
        try:
            # The system prompt is injected here each request — it's not stored.
            system = build_system_prompt()
            history = [m.model_dump(exclude_none=True) for m in messages]

            # Keep the prompt within the model's context window: the system
            # prompt is always kept; trim the oldest turns from history to fit.
            window = self._llm.context_window()
            budget = int(window * self.CONTEXT_BUDGET) - _estimate_tokens(
                [{"role": "system", "content": system}]
            )
            history = trim_to_budget(history, budget)

            conversation = [{"role": "system", "content": system}, *history]

            for round_num in range(self.MAX_TOOL_ROUNDS + 1):
                content, tool_calls = yield from self._llm.stream_turn(
                    conversation, self._tools.schemas, thinking=thinking
                )

                # Only complete calls (id + name present) are runnable.
                calls = [call for call in tool_calls.values() if call.complete]
                if not calls:
                    # Final answer: record the assistant turn and stop.
                    if content:
                        new_messages.append({"role": "assistant", "content": content})
                    return new_messages

                if round_num == self.MAX_TOOL_ROUNDS:
                    yield StreamEvent(
                        type=EventType.CONTENT,
                        text="\n[stopped: too many tool-call rounds]\n",
                    )()
                    return new_messages

                yield from self._run_tools(conversation, content, calls, new_messages)

        except Exception as e:
            yield StreamEvent(
                type=EventType.CONTENT,
                text=f"[error talking to model at {self._llm.base_url}: {e}]",
            )()
        return new_messages

    def _run_tools(
        self,
        conversation: list[dict],
        content: str,
        calls: list[ToolCall],
        new_messages: list[dict],
    ) -> Iterator[str]:
        """Run `calls`, appending the assistant + tool messages.

        Each new message is appended to both `conversation` (so the next model
        turn sees it) and `new_messages` (so the caller can persist it). Emits a
        `tool_call` event and a `tool_result` event per call.
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
        assistant_msg = {
            "role": "assistant",
            "content": content or None,
            "tool_calls": assistant_tool_calls,
        }
        conversation.append(assistant_msg)
        new_messages.append(assistant_msg)

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
            tool_msg = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result_text,
            }
            conversation.append(tool_msg)
            new_messages.append(tool_msg)

            yield StreamEvent(
                type=EventType.TOOL_RESULT,
                id=call.id,
                name=call.name,
                text=result_text,
            )()
