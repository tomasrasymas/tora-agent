#!/bin/bash

set -e

PORT=8033
HOST=0.0.0.0
BENCH_CMD="uv run run_benchmarks.py --runs 4 --warmup 2 --base-url http://localhost:${PORT}"

BASE_FLAGS="--host $HOST --port $PORT --n-gpu-layers 99 --parallel 1 --threads 10 --threads-batch 10 --no-cont-batching --cache-reuse 0"

# All 18 models: 6 families × 3 quants (Q4, Q8, BF16-first-shard)
ALL_MODELS=(
  "gemma-4-26B-A4B-it-GGUF/UD-Q4_K_XL/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
  "gemma-4-26B-A4B-it-GGUF/UD-Q8_K_XL/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf"
  "gemma-4-26B-A4B-it-GGUF/BF16/BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf"

  "gemma-4-31B-it-GGUF/UD-Q4_K_XL/gemma-4-31B-it-UD-Q4_K_XL.gguf"
  "gemma-4-31B-it-GGUF/UD-Q8_K_XL/gemma-4-31B-it-UD-Q8_K_XL.gguf"
  "gemma-4-31B-it-GGUF/BF16/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"

  "Qwen3.6-27B-GGUF/UD-Q4_K_XL/Qwen3.6-27B-UD-Q4_K_XL.gguf"
  "Qwen3.6-27B-GGUF/UD-Q8_K_XL/Qwen3.6-27B-UD-Q8_K_XL.gguf"
  "Qwen3.6-27B-GGUF/BF16/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"

  "Qwen3.6-35B-A3B-GGUF/UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  "Qwen3.6-35B-A3B-GGUF/UD-Q8_K_XL/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
  "Qwen3.6-35B-A3B-GGUF/BF16/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf"

  "Qwen3.6-35B-A3B-MTP-GGUF/UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
  "Qwen3.6-35B-A3B-MTP-GGUF/UD-Q8_K_XL/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
  "Qwen3.6-35B-A3B-MTP-GGUF/BF16/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf"

  "Qwen3.6-27B-MTP-GGUF/UD-Q4_K_XL/Qwen3.6-27B-UD-Q4_K_XL.gguf"
  "Qwen3.6-27B-MTP-GGUF/UD-Q8_K_XL/Qwen3.6-27B-UD-Q8_K_XL.gguf"
  "Qwen3.6-27B-MTP-GGUF/BF16/BF16/Qwen3.6-27B-BF16-00001-of-00002.gguf"
)

# -----------------------------------------------------------------------

wait_for_server() {
  echo "  Waiting for server on port $PORT..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
      echo "  Server ready."
      return 0
    fi
    sleep 2
  done
  echo "  ERROR: server did not start in 60s"
  return 1
}

run_model() {
  local model_path="models/$1"
  echo ""
  echo "========================================"
  echo "Model: $model_path"
  echo "========================================"

  # start server in background
  llama-server -m "$model_path" $BASE_FLAGS > /tmp/llama-server.log 2>&1 &
  SERVER_PID=$!

  # wait for it to be ready
  if ! wait_for_server; then
    echo "  Server failed to start. Log:"
    cat /tmp/llama-server.log
    kill $SERVER_PID 2>/dev/null || true
    return 1
  fi

  # run benchmarks
  echo "  Running benchmarks..."
  $BENCH_CMD || echo "  WARNING: benchmark exited with error"

  # stop server
  echo "  Stopping server (PID $SERVER_PID)..."
  kill $SERVER_PID 2>/dev/null || true
  wait $SERVER_PID 2>/dev/null || true
  sleep 3
}

# -----------------------------------------------------------------------
# Entry point: single model or full sweep

if [ -n "$1" ]; then
  # single model mode — pass relative path under models/
  # e.g. ./run_evals.sh Qwen3.6-27B-GGUF/UD-Q8_K_XL/Qwen3.6-27B-UD-Q8_K_XL.gguf
  run_model "$1"
else
  total=${#ALL_MODELS[@]}
  echo "Starting full eval sweep: $total models"
  for i in "${!ALL_MODELS[@]}"; do
    echo ""
    echo "[$(( i + 1 ))/$total]"
    run_model "${ALL_MODELS[$i]}" || echo "  Skipping to next model..."
  done
  echo ""
  echo "========================================"
  echo "All done."
  echo "========================================"
fi