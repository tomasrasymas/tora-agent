"""Skill discovery and lookup.

A *skill* is a folder containing a `SKILL.md` file: YAML frontmatter (at least a
`description`) followed by a markdown body of instructions. Skills are discovered
across an ordered list of roots — system skills shipped inside the package, then
the user's `~/.tora/skills` — with **user skills overriding system skills** of
the same name.

Progressive disclosure: only each skill's name + description are surfaced to the
model up front (see `SkillRegistry.prompt_section`). The full body is loaded on
demand when the model calls the `load_skill` tool, keeping the context small no
matter how many skills exist.

Discovery is fail-soft: a malformed or unreadable skill is skipped with a logged
warning rather than crashing startup, because the user's skill dir is untrusted
input.
"""

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import SYSTEM_SKILLS_DIR
from .frontmatter import split_frontmatter

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    """A single discovered skill.

    Holds only the cheap metadata; the instruction body lives on disk and is
    read lazily by `load_body` so it never sits in memory or context until used.
    """

    name: str
    description: str
    path: Path  # the SKILL.md file itself

    def load_body(self) -> str:
        """Return the markdown instruction body (frontmatter stripped)."""
        return split_frontmatter(self.path.read_text(encoding="utf-8"))[1]


def _load_skill(skill_dir: Path) -> Skill | None:
    """Load one skill from its directory, or None if it isn't a valid skill.

    Returns None (with a warning) for anything malformed; never raises, so one
    bad skill can't take down discovery.
    """
    skill_file = skill_dir / SKILL_FILE
    if not skill_file.is_file():
        return None  # not a skill folder; silently ignore

    try:
        meta, _ = split_frontmatter(skill_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as e:
        logger.warning("Skipping skill %s: %s", skill_dir.name, e)
        return None

    description = meta.get("description")
    if not description:
        logger.warning(
            "Skipping skill %s: missing 'description' in frontmatter", skill_dir.name
        )
        return None

    # Folder name is the default id; frontmatter `name` may override it.
    name = meta.get("name") or skill_dir.name
    return Skill(name=name, description=str(description), path=skill_file)


class SkillRegistry:
    """The set of skills available to the model, keyed by name."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def __len__(self) -> int:
        return len(self._skills)

    def prompt_section(self) -> str:
        """A system-prompt block listing skills, or "" when there are none.

        Lists only name + description (progressive disclosure); the model calls
        `load_skill` to pull the full instructions when it picks one.
        """
        if not self._skills:
            return ""
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        listing = "\n".join(lines)
        return (
            "\n\n## Skills\n"
            "You have access to the following skills — focused instructions for "
            "specific tasks. When a request matches one, call the `load_skill` "
            "tool with its name to get the full instructions, then follow them:\n"
            f"{listing}"
        )


def discover_skills(roots: Iterable[Path]) -> SkillRegistry:
    """Discover skills across `roots` in order, later roots overriding earlier.

    Each root's immediate subdirectories are inspected for a `SKILL.md`. Because
    roots are applied in order and a later same-named skill replaces an earlier
    one, passing (system_root, user_root) gives "user overrides system".
    """
    found: dict[str, Skill] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            skill = _load_skill(entry)
            if skill is None:
                continue
            if skill.name in found:
                logger.info(
                    "Skill '%s' from %s overrides an earlier one", skill.name, root
                )
            found[skill.name] = skill
    return SkillRegistry(found.values())


def _skill_roots() -> tuple[Path, Path]:
    """System skills (shipped) first, then the user's skills under TORA_HOME."""
    tora_home = Path(os.environ.get("TORA_HOME", "~/.tora")).expanduser()
    return (SYSTEM_SKILLS_DIR, tora_home / "skills")


# Discovered once, at import: every skill is active from the start. Shared by the
# system prompt (which lists skills) and the `load_skill` tool (which reads a
# body), the same way `tools.registry` is a single module-level registry.
registry = discover_skills(_skill_roots())


def load_skill(name: str) -> dict:
    """Tool implementation: return a skill's full instructions by name.

    Also returns the skill's absolute `directory` so the model can read or run
    files the instructions reference by relative path (e.g. `reference/run.py`)
    via the bash tool — the body itself stays the only thing loaded into context.
    """
    skill = registry.get(name)
    if skill is None:
        return {"error": f"Unknown skill: {name}"}
    try:
        return {
            "name": skill.name,
            "directory": str(skill.path.parent),
            "instructions": skill.load_body(),
        }
    except OSError as e:
        return {"error": f"Could not read skill '{name}': {e}"}
