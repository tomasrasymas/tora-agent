# Benchmarks

Folder for model benchmarks and the code that runs them.

## Runtime

**llama.cpp only.** Ollama is not used.

## Usage

Two pieces: `serve.sh` boots a `llama-server` for one model; `run_benchmarks.py` runs the test suite against whatever is currently loaded on that server.

**1. Start the server** (foreground, Ctrl-C to stop):

```bash
./serve.sh /path/to/model.gguf
```

Listens on `0.0.0.0:8033` with single-slot, GPU-offloaded defaults. Edit `serve.sh` if you need different flags.

**2. Run the benchmarks** (from the project root, in another terminal):

```bash
uv run benchmarks/run_benchmarks.py --runs 3 --warmup 1 --base-url http://localhost:8033
```

Flags:
- `--runs` measured runs per test (default 5)
- `--warmup` discarded warmup runs per test (default 1)
- `--base-url` llama.cpp server URL (default `http://localhost:8080`)

Results in this folder were generated with **`--runs 3 --warmup 1`**. At `temperature=0.0` with `--parallel 1 --no-cont-batching` on the server, accuracy is near-deterministic so 3 runs is sufficient for go/no-go; timing statistics (mean, median) are usable, but stdev/p95 are noisy with n=3.

Results are written to `results/<model>-<timestamp>.json`.

## Judge

Tests with a `judge` config are scored by Claude Sonnet 4.6. Set `ANTHROPIC_API_KEY` in the repo-root `.env` to enable; without it, judge evals are skipped and only deterministic `eval` checks count.

## Test suite

Defined in `benchmarks.py`. Categories and weights:

| Category | Weight |
|---|---|
| Function Calling | 2.0 |
| Tool Selection | 1.5 |
| Structured Output | 1.5 |
| Instruction Following | 1.2 |
| Context | 1.2 |
| Reasoning | 1.0 |
| Robustness | 1.0 |
| Latency | 0.5 |
| Throughput | 0.3 |

Each test sets `eval` (deterministic), `judge` (LLM scored), both, or neither (latency/throughput only). Throughput and reasoning tests pin `max_tokens` per-test; everything else uses the 2048 default.

## Models under test

All from [Unsloth](https://huggingface.co/unsloth) as GGUF. Each model is benchmarked at three quantizations: `BF16`, `UD-Q8_K_XL`, `UD-Q4_K_XL`.

### Gemma 4
- `unsloth/gemma-4-31B-it-GGUF` — dense 31B
- `unsloth/gemma-4-26B-A4B-it-GGUF` — MoE, 4B active

### Qwen3.6
- `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` — MoE, 3B active, MTP
- `unsloth/Qwen3.6-35B-A3B-GGUF` — MoE, 3B active
- `unsloth/Qwen3.6-27B-MTP-GGUF` — dense 27B, MTP
- `unsloth/Qwen3.6-27B-GGUF` — dense 27B

Total: **16 model × quant combinations** benchmarked. The two `Qwen3.6-35B-A3B` BF16 variants (MoE and MTP) didn't fit on hardware, so they're excluded.

## Results

Run on the DGX Spark (GB10, 128GB unified memory) with `llama-server` (`--parallel 1 --no-cont-batching`, full GPU offload). Config: `--runs 3 --warmup 1`, `temperature=0.0`, thinking disabled. Judge: Claude Sonnet 4.6. Raw JSON in `results/`. Dated 2026-05-29/30.

`Score` is the weighted composite (weights above). `tok/s` is the mean of the two throughput tests; `Latency` is the mean of the two latency tests (lower is better).

### Leaderboard (by composite score)

| Model | Quant | Score | tok/s | Latency (s) |
|---|---|--:|--:|--:|
| gemma-4-26B-A4B-it | Q8 | 95.1 | 36.7 | 0.42 |
| gemma-4-31B-it | BF16 | 95.1 | 3.8 | 3.23 |
| gemma-4-26B-A4B-it | BF16 | 92.6 | 24.1 | 0.62 |
| gemma-4-26B-A4B-it | Q4 | 92.6 | 52.4 | 0.30 |
| gemma-4-31B-it | Q4 | 91.6 | 10.0 | 1.22 |
| gemma-4-31B-it | Q8 | 90.8 | 6.0 | 2.03 |
| Qwen3.6-35B-A3B | Q8 | 88.4 | 45.2 | 0.35 |
| Qwen3.6-35B-A3B-MTP | Q8 | 86.5 | 58.6 | 0.34 |
| Qwen3.6-35B-A3B | Q4 | 85.9 | 63.3 | 0.27 |
| Qwen3.6-35B-A3B-MTP | Q4 | 82.0 | 78.2 | 0.27 |
| Qwen3.6-27B | Q4 | 80.5 | 11.2 | 1.13 |
| Qwen3.6-27B | BF16 | 78.9 | 4.5 | 2.72 |
| Qwen3.6-27B | Q8 | 78.1 | 6.6 | 1.88 |
| Qwen3.6-27B-MTP | Q8 | 78.1 | 14.9 | 1.06 |
| Qwen3.6-27B-MTP | BF16 | 78.1 | 11.2 | 1.42 |
| Qwen3.6-27B-MTP | Q4 | 77.2 | 24.7 | 0.67 |

### Per-category breakdown

Accuracy (%) per category. FC = Function Calling, Tool = Tool Selection, SO = Structured Output, IF = Instruction Following, Ctx = Context, Reas = Reasoning, Rob = Robustness.

| Model | Quant | Score | FC | Tool | SO | IF | Ctx | Reas | Rob | tok/s | Lat |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gemma-4-31B-it | BF16 | 95.1 | 100 | 100 | 100 | 100 | 100 | 50 | 100 | 3.8 | 3.23 |
| gemma-4-31B-it | Q8 | 90.8 | 100 | 100 | 87.5 | 100 | 100 | 50 | 75 | 6.0 | 2.03 |
| gemma-4-31B-it | Q4 | 91.6 | 100 | 100 | 87.5 | 100 | 100 | 50 | 83.3 | 10.0 | 1.22 |
| gemma-4-26B-A4B-it | BF16 | 92.6 | 100 | 100 | 100 | 100 | 100 | 50 | 75 | 24.1 | 0.62 |
| gemma-4-26B-A4B-it | Q8 | 95.1 | 100 | 100 | 100 | 100 | 100 | 50 | 100 | 36.7 | 0.42 |
| gemma-4-26B-A4B-it | Q4 | 92.6 | 100 | 100 | 100 | 100 | 100 | 50 | 75 | 52.4 | 0.30 |
| Qwen3.6-35B-A3B | Q8 | 88.4 | 100 | 100 | 87.5 | 100 | 100 | 50 | 75 | 45.2 | 0.35 |
| Qwen3.6-35B-A3B | Q4 | 85.9 | 100 | 100 | 87.5 | 100 | 100 | 50 | 50 | 63.3 | 0.27 |
| Qwen3.6-35B-A3B-MTP | Q8 | 86.5 | 100 | 100 | 75 | 100 | 100 | 50 | 75 | 58.6 | 0.34 |
| Qwen3.6-35B-A3B-MTP | Q4 | 82.0 | 100 | 100 | 87.5 | 66.7 | 100 | 50 | 50 | 78.2 | 0.27 |
| Qwen3.6-27B | BF16 | 78.9 | 100 | 100 | 87.5 | 100 | 75 | 0 | 58.3 | 4.5 | 2.72 |
| Qwen3.6-27B | Q8 | 78.1 | 100 | 100 | 87.5 | 100 | 75 | 0 | 50 | 6.6 | 1.88 |
| Qwen3.6-27B | Q4 | 80.5 | 100 | 100 | 87.5 | 100 | 75 | 0 | 75 | 11.2 | 1.13 |
| Qwen3.6-27B-MTP | BF16 | 78.1 | 100 | 100 | 87.5 | 100 | 75 | 0 | 50 | 11.2 | 1.42 |
| Qwen3.6-27B-MTP | Q8 | 78.1 | 100 | 100 | 87.5 | 100 | 75 | 0 | 50 | 14.9 | 1.06 |
| Qwen3.6-27B-MTP | Q4 | 77.2 | 100 | 100 | 87.5 | 100 | 75 | 0 | 41.7 | 24.7 | 0.67 |

### MTP speculative decoding

MTP variants were served with `--spec-type draft-mtp --spec-draft-n-max 3 --draft-p-min 0`. They produce a near-identical quality profile while decoding faster — the gain is largest on the dense 27B (~2.2–2.5×) and smaller on the already-fast 35B-A3B MoE (~1.25×).

| Base | Quant | Score (base → MTP) | tok/s (base → MTP) | Speedup |
|---|---|--:|--:|--:|
| Qwen3.6-27B | BF16 | 78.9 → 78.1 | 4.5 → 11.2 | 2.49× |
| Qwen3.6-27B | Q8 | 78.1 → 78.1 | 6.6 → 14.9 | 2.26× |
| Qwen3.6-27B | Q4 | 80.5 → 77.2 | 11.2 → 24.7 | 2.21× |
| Qwen3.6-35B-A3B | Q8 | 88.4 → 86.5 | 45.2 → 58.6 | 1.30× |
| Qwen3.6-35B-A3B | Q4 | 85.9 → 82.0 | 63.3 → 78.2 | 1.24× |

### Takeaways

- **Gemma 4 leads on quality.** `gemma-4-26B-A4B-it` at Q8 (95.1%, 36.7 tok/s) is the best quality-for-speed pick; its MoE 4B-active design keeps it fast despite the high score. `gemma-4-31B-it` BF16 ties on quality but is unusably slow (3.8 tok/s) — its Q4 is the practical dense option.
- **Qwen3.6-35B-A3B is the throughput champion** among the strong models: 85–88% while sustaining 45–78 tok/s thanks to 3B active params.
- **The dense Qwen3.6-27B underperforms** (~78–80%), held back entirely by **Reasoning = 0%** — with thinking disabled it fails both multi-step arithmetic and day-of-week tests, where every other family scores 50%. Function calling, tool selection, and instruction following are perfect across all Qwen variants.
- **MTP is a free speedup** on the dense 27B (~2.2–2.5× tok/s, quality within noise) and a modest one on the MoE 35B. Quantization quality loss is small everywhere; Q4 is generally within ~2–3 points of Q8.

## All tests

Generated from `benchmarks.py`. `det` = deterministic `eval`, `judge` = LLM judge.

### Latency
| Test | Description | Checks |
|---|---|---|
| `latency_short` | Short prompt, short answer. TTFT + decode. | det |
| `latency_system_prompt` | System prompt overhead on short task. | det |

### Throughput
| Test | Description | Checks |
|---|---|---|
| `throughput_512` | 512-token generation. Sustained tok/s benchmark. | timing only |
| `throughput_code` | Long code generation. Measures tok/s on structured output. | timing only |

### Function Calling
| Test | Description | Checks |
|---|---|---|
| `fc_single_no_ambiguity` | One tool, unambiguous request. Baseline pass/fail. | det |
| `fc_arg_enum_unit` | Enum argument: unit must be `celsius`. | det |
| `fc_calendar_duration` | Duration parsed: '90-minute' → 90. | det |
| `fc_send_email` | Email: to/subject/body extracted. Judge checks body coherence. | det + judge |
| `fc_reminder` | Set reminder: message + datetime populated. | det |
| `fc_no_tool_needed` | Tools available but not needed — answer from knowledge. | det + judge |

### Tool Selection
| Test | Description | Checks |
|---|---|---|
| `ts_calendar_vs_email` | Must pick calendar, not email. | det |
| `ts_reminder_vs_calendar` | Reminder (not event) for personal alert without attendees. | det |
| `ts_file_vs_search` | Read local file, not web search, when path is given. | det |
| `ts_search_vs_static` | Real-time info (stock price) needs search. | det |
| `ts_five_tools_right_one` | 5 tools, only `send_email` is correct. | det |
| `ts_no_tool_five_available` | 5 tools available, none needed. Confident correct answer. | det + judge |
| `fc_contacts_vs_email` | Look up contact first, not email — correct tool ordering. | det |

### Instruction Following
| Test | Description | Checks |
|---|---|---|
| `if_word_limit_strict` | Hard 20-word limit. | det |
| `if_system_persona` | System persona: CLI tool. Full compliance. | det + judge |
| `if_negative_constraint` | Must NOT use the word 'programming'. Quality + compliance. | det + judge |

### Context
| Test | Description | Checks |
|---|---|---|
| `ctx_recall_two_facts` | Recall two different facts from earlier turns. | det |
| `ctx_override_fact` | Updated fact must overwrite earlier one. | det |
| `ctx_ignore_distraction` | Distractor turn — model must recall correct earlier fact. | det |
| `ctx_five_turn` | 5-turn conversation — recall from turn 1. | det |

### Reasoning
| Test | Description | Checks |
|---|---|---|
| `reasoning_arithmetic` | Multi-step arithmetic — correct numerical answer. | det |
| `reasoning_calendar` | Day-of-week calculation (Wed + 10 days = Saturday). | det |

### Robustness
| Test | Description | Checks |
|---|---|---|
| `rob_no_hallucinate_person` | Should not invent facts about a fabricated person. | judge |
| `rob_refuse_harmful` | Must refuse harmful request. Genuine refusal. | judge |
| `rob_ambiguous_request` | Ambiguous request — must ask clarification. | det + judge |
| `rob_prompt_injection_resist` | Prompt injection attempt. Resistance quality. | det + judge |

### Structured Output
| Test | Description | Checks |
|---|---|---|
| `so_json_object_mode` | `response_format=json_object`. Output must be valid JSON. | det + judge |
| `so_schema_flat` | `json_schema`: flat object, required fields with correct types. | det |
| `so_schema_nested` | `json_schema`: nested object with address sub-schema. | det |
| `so_schema_array` | `json_schema`: top-level array of typed objects. | det |
| `so_schema_enum` | `json_schema`: enum field must be one of allowed values. | det + judge |
| `so_schema_optional_fields` | `json_schema`: required fields present; optional handled cleanly. | det |
| `so_schema_extraction` | `json_schema`: extract structured data from unstructured prose. | det + judge |
| `so_schema_conflict` | Schema conflicts with prompt intent — schema wins, no hallucination. | det + judge |
