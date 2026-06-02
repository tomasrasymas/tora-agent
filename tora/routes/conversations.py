"""Conversation routes: list, create, load, and delete stored chats."""

from fastapi import APIRouter, HTTPException

from ..schemas import ConversationInfo, Message, Messages, NewConversation
from ..storage import Storage


def create_conversations_router(store: Storage) -> APIRouter:
    """Build the conversations router, wired to a given store."""

    router = APIRouter(prefix="/api/conversations")

    @router.get("")
    def list_conversations() -> list[ConversationInfo]:
        """All conversations, newest first, for the sidebar."""
        return [ConversationInfo(**c) for c in store.list()]

    @router.post("")
    def create_conversation() -> NewConversation:
        """Create an empty conversation and return its id."""
        return NewConversation(id=store.create())

    @router.get("/{conversation_id}")
    def get_conversation(conversation_id: str) -> Messages:
        """Return a conversation's messages so the frontend can render it."""
        messages = store.get(conversation_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return Messages(messages=[Message(**m) for m in messages])

    @router.delete("/{conversation_id}")
    def delete_conversation(conversation_id: str) -> dict[str, bool]:
        store.delete(conversation_id)
        return {"ok": True}

    return router
