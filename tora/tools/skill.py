"""load_skill tool: fetch a skill's full instructions on demand.

The system prompt lists available skills by name + description; when the model
decides one applies, it calls this tool to pull the full instruction body. This
keeps the up-front context small (progressive disclosure) — see `tora.skills`.
"""

from ..skills import load_skill
from .base import Tool

load_skill_tool = Tool(
    name="load_skill",
    description=(
        "Load the full instructions for one of the available skills (listed in "
        "the system prompt under 'Skills'). Call this with the skill's name when "
        "a request matches it, then follow the returned instructions. The result "
        "also includes the skill's absolute 'directory'; any files the "
        "instructions reference by relative path (e.g. reference/run.py) live "
        "under it — read or run them with the bash tool, e.g. "
        "`python <directory>/reference/run.py`."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the skill to load.",
            },
        },
        "required": ["name"],
    },
    fn=load_skill,
)
