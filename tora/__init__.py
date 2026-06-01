"""TORA agent — a minimal, fully-local personal AI assistant.

The package is split into small, single-purpose modules:

  config   — runtime settings loaded from the environment
  schemas  — Pydantic models for the HTTP API
  llm      — thin client over the llama.cpp OpenAI-compatible server
  api      — FastAPI routes
  app      — application factory that wires everything together
  __main__ — uvicorn entrypoint (`python -m tora` / the `tora` script)
"""

__version__ = "0.1.0"
