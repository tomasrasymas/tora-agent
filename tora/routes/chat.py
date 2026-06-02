"""Chat route: streams completions from the model to the browser.

The backend owns history: the request names a conversation and carries only the
new user message. We load the prior messages from storage, stream the agent's
turn, and persist the user + assistant turns once the stream completes.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..agent import Agent
from ..schemas import ChatRequest, Message
from ..storage import Storage


def create_chat_router(agent: Agent, store: Storage) -> APIRouter:
    """Build the chat router, wired to a given agent and conversation store."""

    router = APIRouter(prefix="/api")

    @router.post("/chat")
    def chat(body: ChatRequest) -> StreamingResponse:
        """Stream a chat completion back to the browser as NDJSON events."""

        prior = store.get(body.conversation_id)
        if prior is None:
            raise HTTPException(status_code=404, detail="conversation not found")

        user_msg = body.message.model_dump(exclude_none=True)
        history = prior + [user_msg]
        models = [Message(**m) for m in history]

        def stream():
            # `yield from` forwards the agent's NDJSON events and captures its
            # return value (the new assistant/tool messages). We persist once the
            # stream is fully consumed.
            new_messages = yield from agent.stream_chat(models)
            store.save(body.conversation_id, history + new_messages)

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return router
