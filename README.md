# TORA personal ai agent

Building a personal AI assistant from scratch — runs locally on my DGX Spark, handles small but annoying tasks, and gives me a real understanding of how agents actually work under the hood.

This is a learning project as much as a product. I'm documenting every decision, dead end, and architecture choice in a [blog series](https://tomasrasymas.substack.com).

## Status

Early. Model selection done, inference stack being benchmarked.

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

18 model × quant combinations in total. See [`benchmarks/`](./benchmarks) for the runner and results.

## Blog series

- [#1 — Picking the model](https://tomasrasymas.substack.com) — hardware constraints, model selection, getting llama.cpp running on Blackwell

More posts coming as the project progresses.

## Goals

- Scheduling tasks
- Sending reminders
- Eventually: more interesting agentic stuff

## Running locally

Nothing to run yet. Will update as the stack takes shape.
