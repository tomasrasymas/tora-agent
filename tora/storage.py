"""Conversation persistence (SQLite).

A single ``conversations`` table in ``$TORA_HOME/tora.db``. Each conversation
stores its whole message list as a JSON blob (the OpenAI-shaped messages the
frontend and agent already use), so there's no row-per-message bookkeeping —
load the list, append, save.

The backend owns history: ``/api/chat`` loads a conversation, runs the agent,
and saves the user + assistant turns. A connection is opened per call so it's
safe to use from FastAPI's threadpool.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path

_TITLE_MAX = 60


def _derive_title(messages: list[dict]) -> str:
    """A short title from the first user message, or '' if there isn't one yet."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            text = " ".join(m["content"].split())
            return text[:_TITLE_MAX]
    return ""


class Storage:
    """SQLite-backed conversation store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id         TEXT PRIMARY KEY,
                    title      TEXT NOT NULL DEFAULT '',
                    messages   TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self) -> str:
        """Create an empty conversation and return its id."""
        cid = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, created_at, updated_at) VALUES (?, ?, ?)",
                (cid, now, now),
            )
        return cid

    def get(self, cid: str) -> list[dict] | None:
        """Return a conversation's messages, or None if it doesn't exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages FROM conversations WHERE id = ?", (cid,)
            ).fetchone()
        return None if row is None else json.loads(row["messages"])

    def save(self, cid: str, messages: list[dict]) -> None:
        """Overwrite a conversation's messages; set its title on first save."""
        payload = json.dumps(messages, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                   SET messages = ?,
                       updated_at = ?,
                       title = CASE WHEN title = '' THEN ? ELSE title END
                 WHERE id = ?
                """,
                (payload, time.time(), _derive_title(messages), cid),
            )

    def list(self) -> list[dict]:
        """All conversations as {id, title, created_at, updated_at}, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, cid: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
