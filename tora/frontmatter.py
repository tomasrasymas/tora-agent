"""Parse the leading YAML frontmatter block shared by skills and memories.

Both ``tora.skills`` and ``tora.memory`` store markdown files as a ``---``-delimited
YAML header followed by a body; this is the one place that knows how to split them.
"""

import yaml


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split file text into (frontmatter dict, body).

    Frontmatter is a leading ``---``-delimited YAML block. If absent, returns an
    empty dict and the whole text as the body. Raises ValueError if the block is
    present but isn't a YAML mapping.
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            if not isinstance(meta, dict):
                raise ValueError("frontmatter is not a mapping")
            return meta, parts[2].lstrip("\n")
    return {}, text
