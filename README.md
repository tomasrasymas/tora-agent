# TORA personal ai agent

Building a personal AI assistant from scratch — runs locally on my DGX Spark, handles small but annoying tasks, and gives me a real understanding of how agents actually work under the hood.

This is a learning project as much as a product. I'm documenting every decision, dead end, and architecture choice in a [blog series](https://tomasrasymas.substack.com).

## Status

Early. Model selection done, inference stack benchmarked (16 model × quant combos). Winner picked: `gemma-4-26B-A4B-it` at Q8.

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

The agent is a thin FastAPI backend (`server.py`) that proxies to the llama.cpp
server over the OpenAI protocol and streams tokens back, plus a single-page chat
frontend (`web/index.html`).

1. Start `model-server` on the DGX with the chosen model.
2. Copy `.env.example` to `.env` and point `TORA_LLM_BASE_URL` at it.
3. Run the backend:

   ```bash
   uv run server.py
   ```

4. Open http://127.0.0.1:8000 and chat.

## Running in Docker

The agent runs on your workstation in Docker; only the model runs on the DGX.
The image is built with `uv` on Python 3.13. Your custom skills and configs
live in `~/.tora` on the workstation and are bind-mounted into the container,
so you can edit them without rebuilding.

The container reaches the model over the LAN, so it needs the DGX's IP — point
`TORA_LLM_BASE_URL` at it (compose errors out if it's unset):

```bash
TORA_LLM_BASE_URL=http://<dgx-ip>:8033/v1 docker compose up --build
```

Or set `TORA_LLM_BASE_URL=http://<dgx-ip>:8033/v1` in `.env` (compose loads it
automatically) and just run `docker compose up --build`. Then open
http://127.0.0.1:8000.

Notes:

- `localhost` / `host.docker.internal` won't work for the model URL — both point
  at the workstation, not the DGX.
- `~/.tora` is mounted at `/root/.tora` (`TORA_HOME`). The backend reads
  `~/.tora/.env` for extra config and looks for skills under `~/.tora/skills`.
  Add `:ro` to the volume in `docker-compose.yml` to mount it read-only.
