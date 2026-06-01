"""Chat route: streams completions from the model to the browser."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..llm import LLMClient
from ..schemas import Messages


def create_chat_router(llm: LLMClient) -> APIRouter:
    """Build the chat router, wired to a given LLM client.

    Taking the client as an argument (rather than reaching for a global) keeps
    the route easy to test and its dependencies explicit.
    """

    router = APIRouter(prefix="/api")

    @router.post("/chat")
    def chat(body: Messages) -> StreamingResponse:
        """Stream a chat completion back to the browser as NDJSON events."""

        return StreamingResponse(
            llm.stream_chat(body.messages),
            media_type="application/x-ndjson",
        )

    return router
