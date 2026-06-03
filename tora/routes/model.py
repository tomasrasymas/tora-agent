"""Model route: reports the currently loaded model."""

from fastapi import APIRouter

from ..llm import LLMClient


def create_model_router(llm: LLMClient) -> APIRouter:
    """Build the model router, wired to a given LLM client.

    Taking the client as an argument (rather than reaching for a global) keeps
    the route easy to test and its dependencies explicit.
    """

    router = APIRouter(prefix="/api")

    @router.get("/model")
    def model() -> dict:
        """Return the loaded model id and context window for the frontend.

        ``context_window`` is the per-conversation token budget; the UI uses it
        as the denominator of the live token meter.
        """

        return {"id": llm.model_id(), "context_window": llm.context_window()}

    return router
