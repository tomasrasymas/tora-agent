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

from . import memory, skills
from .llm import LLMClient, ToolCall
from .schemas import EventType, Message, StreamEvent
from .tools import registry

SYSTEM_PROMPT = """\
You are TORA, a personal AI assistant for Tomas. You help him manage the \
everyday details — things like supplements schedule, reminding things, \
helping on day-to-day tasks — using your built-in tools.

Help Tomas get things done and answer what he ask, rather than pushing advice he \
didn't request. Be concise, practical, and friendly. Use the available tools \
when they help, and when a request is missing detail your need, ask one brief \
clarifying question first.

Besids tools you have access to different skills. Skills are additional capabilities for you \
to serve Tomas needs.

## Remembering
You keep durable notes about Tomas between conversations with the `remember`, `recall`, and \
`forget` tools. Saving is cheap and expected — be proactive, you don't need permission and you \
don't need to mention that you saved something.

Your rule: whenever a message reveals something lasting about Tomas — a preference, habit, \
routine, goal, constraint, relationship, or fact about his life — and it isn't already under \
"Memory" below, call `remember` for it in the same turn, alongside your normal reply. Don't \
save one-off task details, passing context, or anything that won't matter next time.

Store the durable fact behind the message, not the message itself:
- "I had a long run today" → he runs; remember "Tomas is a runner".
- "What supplements should I take today?" → he takes supplements; remember that.
- "My wife and I are flying to Italy in July" → remember he's married, and the trip.
Always check the "Memory" list first so you don't save a duplicate. If a fact changes or turns \
out wrong, correct it: `forget` the stale one and `remember` the new version.

About you - you run on a local server in in a docker container. LLM model that is core part \
of you runs on separate device (DGX Spark).
"""


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

    The base prompt is static, but the date/time and the skills listing are
    resolved live so they reflect "now" and the currently discovered skills.
    All system-prompt formatting lives here, in one place.
    """
    now = datetime.now().astimezone()
    date_line = f"\n\nThe current date and time is {now:%A, %d %B %Y, %H:%M %Z}."
    return (
        SYSTEM_PROMPT
        + date_line
        + memory.store.prompt_section()
        + skills.registry.prompt_section()
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
