"""remember / recall / forget tools: TORA's durable memory about Tomas.

Thin wrappers over `tora.memory`, which owns the on-disk format. The model calls
`remember` when Tomas shares something worth keeping for future conversations,
`recall` to search past memories, and `forget` to drop one that's wrong or stale.
"""

from ..memory import forget, recall, remember
from .base import Tool

remember_tool = Tool(
    name="remember",
    description=(
        "Save a durable fact about Tomas so you'll have it in future "
        "conversations — preferences, stable personal facts, ongoing goals or "
        "situations (e.g. 'Tomas is allergic to shellfish', 'prefers to train in "
        "the mornings'). Use this only for things worth keeping long-term; do NOT "
        "save one-off task details, trivia, or anything that won't matter later. "
        "Each memory should capture a single coherent topic — a few sentences is "
        "fine if the detail matters; split unrelated facts into separate memories."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The fact to remember. Can be a sentence or a short paragraph "
                    "— include whatever detail makes it useful later."
                ),
            },
            "title": {
                "type": "string",
                "description": (
                    "Optional short label for the memory (used to name the file "
                    "and as its id). Derived from the content if omitted."
                ),
            },
        },
        "required": ["content"],
    },
    fn=remember,
)

recall_tool = Tool(
    name="recall",
    description=(
        "Search your stored memories about Tomas by keyword and return the "
        "matches. Your current memories are already listed in the system prompt, "
        "so you usually don't need this — reach for it only to look something up "
        "explicitly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to search remembered facts for.",
            },
        },
        "required": ["query"],
    },
    fn=recall,
)

forget_tool = Tool(
    name="forget",
    description=(
        "Delete a stored memory by its id (the value in brackets next to each "
        "memory in the system prompt). Use this when a remembered fact is wrong, "
        "outdated, or Tomas asks you to forget it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The id of the memory to delete.",
            },
        },
        "required": ["name"],
    },
    fn=forget,
)
