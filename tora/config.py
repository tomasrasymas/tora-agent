"""Runtime configuration, loaded once from the environment.

Settings come from environment variables (optionally populated by a repo-local
`.env` and a `$TORA_HOME/.env`). Keeping all of this in one place means the rest
of the app never touches `os.environ` directly.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Static frontend, shipped inside the package (tora/web/).
WEB_DIR = Path(__file__).resolve().parent / "web"

# Built-in skills shipped inside the package (tora/skills/). User skills live
# under TORA_HOME and take precedence over these (see `tora.skills`).
SYSTEM_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


@dataclass(frozen=True)
class Settings:
    """Everything the app needs to run, resolved from the environment."""

    llm_base_url: str  # OpenAI-compatible llama.cpp server, e.g. http://dgx:8033/v1
    web_dir: Path  # static frontend served at "/"
    skills_dir: Path  # writable user skill dir (under TORA_HOME)
    db_path: Path  # SQLite file for persisted conversations (under TORA_HOME)
    host: str  # backend bind address
    port: int  # backend bind port


def load_settings() -> Settings:
    """Read configuration from the environment and return immutable settings.

    Loads `.env` files for convenience, ensures the skills directory exists, and
    fails loudly (KeyError) if a required variable is missing.
    """

    # Repo-local .env first, then the user's TORA_HOME/.env (which may override).
    load_dotenv()

    tora_home = Path(os.environ["TORA_HOME"]).expanduser()
    skills_dir = tora_home / "skills"
    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # mount may be read-only; that's fine, we only read from it

    return Settings(
        llm_base_url=os.environ["TORA_LLM_BASE_URL"],
        web_dir=WEB_DIR,
        skills_dir=skills_dir,
        db_path=tora_home / "tora.db",
        host=os.environ["TORA_HOST"],
        port=int(os.environ["TORA_PORT"]),
    )
