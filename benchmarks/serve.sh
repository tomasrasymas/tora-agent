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
  --parallel 1 \
  --threads 10 \
  --threads-batch 10 \
  --no-cont-batching \
  --cache-reuse 0
