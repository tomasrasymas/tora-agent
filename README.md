# TORA personal ai agent

Building a personal AI assistant from scratch — runs locally on my DGX Spark, handles small but annoying tasks, and gives me a real understanding of how agents actually work under the hood.

This is a learning project as much as a product. I'm documenting every decision, dead end, and architecture choice in a [blog series](https://tomasrasymas.substack.com).

## Status

Early but usable. Model selection and benchmarking are done (16 model × quant combos; winner: `gemma-4-26B-A4B-it` at Q8), and the agent itself runs: a streaming chat backend with a tool loop, a `bash` tool, web search and fetch (self-hosted SearXNG), on-demand skills, long-term memory, and persisted conversations, all behind a single-page web UI.

## Hardware

NVIDIA DGX Spark — GB10 Grace Blackwell, 128GB unified memory (CPU+GPU).

## Constraints

- Open-weight models only
- Runs on local network, no external API calls
- Everything self-hosted

## Stack (so far)

| Layer | Choice | Notes |
|---|---|---|
| Inference | llama.cpp | Needed for MTP support and full control over the runtime |
| Model families | Gemma 4, Qwen3.6 | GGUF builds from Unsloth, run via llama.cpp |
| Backend | FastAPI, Python 3.13, `uv` | Streaming chat, the tool loop, SQLite persistence |
| Frontend | Single-page HTML/JS | Served by the backend; reads the NDJSON token stream |
| Web search | SearXNG, trafilatura | Self-hosted metasearch backend; readable-text extraction for fetched pages |
| Packaging | Docker | Agent + SearXNG containerized on the workstation; the model stays on the DGX |

## How the agent works

TORA is a FastAPI backend plus a single-page chat UI. The model runs remotely on
the DGX (llama.cpp, OpenAI-compatible API); everything else runs in the backend
on my workstation.

A chat turn flows like this:

1. The frontend posts the conversation to `/api/chat`.
2. The agent assembles the system prompt — base instructions + live date/time +
   the current memory and skills listings — and streams a model turn.
3. If the model calls tools, the agent runs them, feeds the results back, and
   lets the model continue. This loops until it produces a final answer (capped
   at 8 tool rounds, so a misbehaving model can't stream forever).
4. Tokens and tool events stream to the UI as NDJSON; the finished turn is
   persisted to SQLite so conversations survive a restart.

The system prompt is rebuilt every request and never stored — so edits to it,
the date, memories, and skills are always live.

Key modules (all under `tora/`):

| Module | Responsibility |
|---|---|
| `llm.py` | Talking to the model: streaming turns, parsing tool calls |
| `agent.py` | The chat/tool loop and system-prompt assembly |
| `tools/` | The tool registry and individual tools (`bash`, `load_skill`, memory, web search/fetch) |
| `skills.py` | Skill discovery and progressive disclosure |
| `memory.py` | Long-term memory — durable facts about me |
| `storage.py` | Conversation persistence (SQLite, one JSON blob per conversation) |
| `frontmatter.py` | Shared YAML-frontmatter parser for skills and memories |
| `routes/` | HTTP endpoints (chat, conversations, model) |
| `web/` | Single-page chat frontend |
| `config.py` | Settings resolved once from the environment |

### HTTP API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat` | Stream a chat turn (NDJSON) |
| GET | `/api/model` | The currently loaded model id |
| GET | `/api/conversations` | List saved conversations |
| POST | `/api/conversations` | Start a new conversation |
| GET | `/api/conversations/{id}` | Load a conversation |
| DELETE | `/api/conversations/{id}` | Delete a conversation |

## Tools

The model has a small set of built-in tools, dispatched by the agent loop:

- **`bash`** — runs shell commands on the machine the backend runs on: read and
  write files, inspect the system, run programs. Guard-railed — commands are
  parsed with `bashlex` and rejected if any of them names a known-dangerous
  binary (`rm`, `dd`, `mkfs`, `kill`, `shutdown`, …), and anything the parser
  can't understand is refused (fail closed). It's a footgun guard, **not** a
  security sandbox — only enable it on a machine you trust the model to operate.
- **`load_skill`** — pulls the full instructions for a skill on demand (see
  [Skills](#skills)).
- **`remember` / `recall` / `forget`** — long-term memory (see [Memory](#memory)).
- **`web_search` / `web_fetch`** — search the live web and read pages (see
  [Web search](#web-search)).

## Web search

TORA can reach the live web for current events, prices, and anything that's
changed since the model's training cutoff. It's deliberately two tools so the
model only pays for the content it actually needs:

- **`web_search`** — queries a self-hosted [SearXNG](https://docs.searxng.org/)
  instance and returns ranked result snippets (title, URL, short summary).
  Cheap — gives the model candidates to choose from. Pass `time_range`
  (`day`/`week`/`month`/`year`) when freshness matters.
- **`web_fetch`** — downloads one URL and extracts the readable article text
  with [trafilatura](https://trafilatura.readthedocs.io/) (navigation, ads, and
  boilerplate stripped). This is where the real content comes from, once the
  model picks a result worth reading.

Snippets are clipped to 500 chars and fetched pages to 10k chars to keep them
from swamping the context window. Errors (SearXNG down, a page that won't
extract) are *returned* to the model as text rather than raised, so it can read
the problem and adapt.

SearXNG runs as a sibling container defined in `docker-compose.yml`; its config
lives in [`searxng/settings.yml`](./searxng/settings.yml). Two settings there are
load-bearing: `search.formats` must include `json` (otherwise the API returns a
403/HTML), and `server.limiter` is off so programmatic calls aren't bot-blocked.
The backend finds it via `TORA_SEARXNG_URL` (defaults to the compose service
name `http://searxng:8080`); for local `uv run tora`, point it at wherever you
run SearXNG, e.g. `http://localhost:8088`.

## Memory

TORA keeps durable facts about me across conversations — preferences, habits,
goals, life facts. The model decides what's worth keeping and saves it
proactively, so I never have to ask it to remember.

- Each memory is a markdown file under `~/.tora/memory/`, with YAML frontmatter
  (`name`, `created_at`) and the fact as the body. Human-readable and
  hand-editable — fix or delete a file and the change is live on the next turn
  (the store re-reads disk every time, nothing is cached).
- Recall is deliberately simple while the set is small: every memory is injected
  into the system prompt each turn. `recall` does a keyword search for when the
  set outgrows that; `forget` drops a fact that's wrong or stale.

```markdown
---
name: tomas-is-a-runner
created_at: 2026-06-02T17:00:00+03:00
---

Tomas runs regularly and prefers morning runs.
```

## Benchmark matrix

All models are pulled from [Unsloth](https://huggingface.co/unsloth) as GGUF. Each is benchmarked at three precisions: `BF16`, `UD-Q8_K_XL`, `UD-Q4_K_XL`.

| Family | Variant | Notes |
|---|---|---|
| Gemma 4 | `gemma-4-31B-it` | Dense 31B |
| Gemma 4 | `gemma-4-26B-A4B-it` | MoE, 4B active |
| Qwen3.6 | `Qwen3.6-35B-A3B-MTP` | MoE, 3B active, MTP |
| Qwen3.6 | `Qwen3.6-35B-A3B` | MoE, 3B active |
| Qwen3.6 | `Qwen3.6-27B-MTP` | Dense 27B, MTP |
| Qwen3.6 | `Qwen3.6-27B` | Dense 27B |

16 model × quant combinations benchmarked. The two Qwen3.6-35B-A3B BF16 variants (MoE and MTP) didn't fit on hardware, so they're excluded. See [`benchmarks/`](./benchmarks) for the runner, the full test suite, and per-category results.

## Results

Run on the DGX Spark with `llama-server` (`--parallel 1 --no-cont-batching`, full GPU offload), `temperature=0.0`, thinking disabled. Judge: Claude Sonnet 4.6. `Score` is a weighted composite across function calling, tool selection, structured output, instruction following, context, reasoning, robustness, latency, and throughput. `tok/s` is the mean of the two throughput tests; `Latency` is the mean of the two latency tests (lower is better). Top of the leaderboard:

| Model | Quant | Score | tok/s | Latency (s) |
|---|---|--:|--:|--:|
| gemma-4-26B-A4B-it | Q8 | 95.1 | 36.7 | 0.42 |
| gemma-4-31B-it | BF16 | 95.1 | 3.8 | 3.23 |
| gemma-4-26B-A4B-it | BF16 | 92.6 | 24.1 | 0.62 |
| gemma-4-26B-A4B-it | Q4 | 92.6 | 52.4 | 0.30 |
| gemma-4-31B-it | Q4 | 91.6 | 10.0 | 1.22 |
| Qwen3.6-35B-A3B | Q8 | 88.4 | 45.2 | 0.35 |

Full 16-row leaderboard, per-category breakdown, and MTP speculative-decoding numbers are in [`benchmarks/BENCHMARKS.md`](./benchmarks/BENCHMARKS.md).

### Takeaways

- **Gemma 4 leads on quality.** `gemma-4-26B-A4B-it` at Q8 (95.1%, 36.7 tok/s) is the best quality-for-speed pick — its MoE 4B-active design keeps it fast despite the top score. `gemma-4-31B-it` BF16 ties on quality but is unusably slow (3.8 tok/s).
- **Qwen3.6-35B-A3B is the throughput champion** among the strong models: 85–88% while sustaining 45–78 tok/s thanks to 3B active params.
- **The dense Qwen3.6-27B underperforms** (~78–80%), held back entirely by Reasoning = 0% with thinking disabled.
- **MTP is a near-free speedup** — ~2.2–2.5× tok/s on the dense 27B, ~1.25× on the MoE 35B, with quality within noise. Q4 is generally within ~2–3 points of Q8.

## Blog series

- [#1 — Picking the model](https://tomasrasymas.substack.com) — hardware constraints, model selection, getting llama.cpp running on Blackwell
- [#2 — Benchmarking the models](https://tomasrasymas.substack.com/p/building-a-personal-ai-agent-that-f36) — the test suite, MTP speculative decoding, and which model/quant wins

More posts coming as the project progresses.

## Goals

- Scheduling tasks
- Sending reminders
- Eventually: more interesting agentic stuff

## Running locally

The agent is a FastAPI backend (the `tora/` package, launched with `uv run tora`)
that talks to the llama.cpp server over the OpenAI protocol and streams tokens
back, plus a single-page chat frontend (`tora/web/index.html`).

1. Start the llama.cpp server on the DGX with the chosen model (see
   `serve_model.sh`).
2. Copy `.env.example` to `.env` and point `TORA_LLM_BASE_URL` at it. For web
   search, also run a SearXNG instance and set `TORA_SEARXNG_URL` (e.g.
   `http://localhost:8088`); without it `web_search`/`web_fetch` just return an
   error and the rest of the agent works fine.
3. Run the backend:

   ```bash
   uv run tora
   ```

4. Open http://127.0.0.1:8888 and chat.

## Running in Docker

The agent runs on your workstation in Docker; only the model runs on the DGX.
`docker compose up` brings up two services — the agent and a SearXNG container
that powers web search (reached over the compose network, no host port by
default). The image is built with `uv` on Python 3.13. Your custom skills and
configs live in `~/.tora` on the workstation and are bind-mounted into the
container, so you can edit them without rebuilding.

The container reaches the model over the LAN, so it needs the DGX's IP — point
`TORA_LLM_BASE_URL` at it (compose errors out if it's unset):

```bash
TORA_LLM_BASE_URL=http://<dgx-ip>:8033/v1 docker compose up --build
```

Or set `TORA_LLM_BASE_URL=http://<dgx-ip>:8033/v1` in `.env` (compose loads it
automatically) and just run `docker compose up --build`. Then open
http://127.0.0.1:8888.

Notes:

- `localhost` / `host.docker.internal` won't work for the model URL — both point
  at the workstation, not the DGX.
- `~/.tora` is mounted at `/root/.tora` (`TORA_HOME`). The backend reads
  `~/.tora/.env` for extra config, looks for skills under `~/.tora/skills`, and
  writes memories to `~/.tora/memory`. You can mount it read-only (`:ro` in
  `docker-compose.yml`), but then the agent can't save new memories.
- SearXNG has no host port by default. To poke at its web UI for debugging,
  uncomment the `ports` mapping under the `searxng` service in
  `docker-compose.yml` (exposes it on `127.0.0.1:8088`).

## Skills

A *skill* is a focused set of instructions for a specific task. Skills are
discovered from two roots, low→high precedence, so a user skill overrides a
built-in one of the same name:

- **System skills**: shipped in the package at `tora/skills/`.
- **User skills**: dropped into `~/.tora/skills/` (the Docker bind mount), no
  rebuild needed.

Each skill is a folder containing a `SKILL.md`: YAML frontmatter plus a markdown
body of instructions.

```
~/.tora/skills/
  my-skill/
    SKILL.md            # frontmatter (name, description) + instructions
    reference/run.py    # optional supporting files
```

```markdown
---
name: my-skill          # optional; defaults to the folder name
description: One line the model uses to decide when to apply this skill.
---

# My skill

Step-by-step instructions go here…
```

Only each skill's name + description are shown to the model up front; it calls
the `load_skill` tool to pull the full body when a request matches (progressive
disclosure). `load_skill` also returns the skill's absolute `directory`, so
**reference bundled files by relative path** (e.g. `reference/run.py`) — the
model reads or runs them under that directory via the bash tool. A malformed
skill (bad frontmatter, missing `description`) is skipped with a warning, never
crashing startup.

## Development

Install the git pre-commit hooks once after cloning:

```bash
uv run pre-commit install
```

They run on every commit (ruff lint + format, file hygiene, and gitleaks secret
scanning). To run them across the whole repo on demand:

```bash
uv run pre-commit run --all-files
```
