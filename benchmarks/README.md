# Benchmarks

Folder for model benchmarks and the code that runs them.

## Runtime

**llama.cpp only.** Ollama is not used.

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

Total: **18 model × quant combinations**.