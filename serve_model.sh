#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <model-path>"
  exit 1
fi

llama-server \
  -m "$1" \
  --host 0.0.0.0 \
  --port 8033 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --parallel 4 \
  --ctx-size 131072 \
  --cache-reuse 256 \
  --threads 10 \
  --threads-batch 10 \
  --metrics