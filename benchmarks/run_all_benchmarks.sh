#!/bin/bash

set -e

if [ -z "$1" ]; then
  cat <<EOF
Usage:
  $0 <models-dir>              Benchmark every model under <models-dir>.
  $0 <models-dir> <path ...>   Benchmark specific models (paths relative to
                               <models-dir> or absolute).
  $0 <path ...>                Benchmark specific .gguf files directly.

  MTP variants (under a *-MTP-* folder) are auto-detected from the path.
EOF
  exit 1
fi

# First arg may be a models-dir root (discover / resolve relative paths) OR a
# .gguf file (in which case every arg is treated as an explicit model path).
MODELS_DIR=""
if [[ -d "$1" ]]; then
    MODELS_DIR="$(realpath "$1")"
    shift  # remaining args (if any) are explicit model paths
fi
BENCHMARK_DIR="$(dirname "$(realpath "$0")")"
SERVER_PORT=8033
SERVER_HOST="0.0.0.0"
SERVER_URL="http://localhost:${SERVER_PORT}"
BENCHMARK_RUNS=3
BENCHMARK_WARMUP=1
HEALTH_CHECK_TIMEOUT=200  # seconds to wait for server to become ready
HEALTH_CHECK_INTERVAL=5   # seconds between health checks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()     { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }

SERVER_PID=""

stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping llama-server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        # Wait for process to actually die
        local waited=0
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            sleep 1
            waited=$((waited + 1))
            if [ $waited -ge 15 ]; then
                warn "Server didn't stop gracefully, sending SIGKILL..."
                kill -9 "$SERVER_PID" 2>/dev/null || true
                break
            fi
        done
        success "Server stopped."
        SERVER_PID=""
    fi

    # Also kill any stray llama-server processes on our port
    local stray
    stray=$(lsof -ti:"${SERVER_PORT}" 2>/dev/null || true)
    if [ -n "$stray" ]; then
        warn "Killing stray process on port ${SERVER_PORT}: $stray"
        kill -9 $stray 2>/dev/null || true
        sleep 1
    fi
}

# Ensure server is stopped on script exit
trap stop_server EXIT INT TERM

wait_for_server() {
    local model_label="$1"
    log "Waiting for server to be ready (timeout: ${HEALTH_CHECK_TIMEOUT}s)..."
    local elapsed=0
    while [ $elapsed -lt $HEALTH_CHECK_TIMEOUT ]; do
        if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
            success "Server is ready for: ${model_label}"
            return 0
        fi
        sleep $HEALTH_CHECK_INTERVAL
        elapsed=$((elapsed + HEALTH_CHECK_INTERVAL))
    done
    error "Server did not become ready within ${HEALTH_CHECK_TIMEOUT}s for: ${model_label}"
    return 1
}

start_server() {
    local model_path="$1"
    local model_label="$2"
    local model_alias="$3"
    local is_mtp="$4"

    log "Starting llama-server for: ${model_label}"
    log "  Model path: ${model_path}"
    log "  Model alias: ${model_alias}"

    # The reported model id (alias) becomes the results filename prefix. For MTP
    # models we inject "MTP" into the alias and enable MTP speculative decoding.
    local extra_args=(--alias "$model_alias")
    if [[ "$is_mtp" == "1" ]]; then
        log "  MTP model detected — enabling MTP speculative decoding"
        extra_args+=(--spec-type draft-mtp --spec-draft-n-max 3 --draft-p-min 0)
    fi

    llama-server \
        -m "$model_path" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --n-gpu-layers 99 \
        --parallel 1 \
        --threads 10 \
        --threads-batch 10 \
        --no-cont-batching \
        --cache-reuse 0 \
        "${extra_args[@]}" \
        > "/tmp/llama_server_${model_label//\//_}.log" 2>&1 &

    SERVER_PID=$!
    log "Server started with PID: ${SERVER_PID}"
}

run_benchmark() {
    local model_label="$1"
    log "Running benchmark for: ${model_label}"
    (
        cd "$BENCHMARK_DIR"
        uv run run_benchmarks.py \
            --runs "$BENCHMARK_RUNS" \
            --warmup "$BENCHMARK_WARMUP" \
            --base-url "$SERVER_URL"
    )
}

# ---------------------------------------------------------------------------
# Discover models
# Rules:
#   - Find all .gguf files under MODELS_DIR
#   - Skip mmproj-BF16.gguf files
#   - For split models (*-00001-of-*.gguf), use the 00001 part as entry point
#   - Skip the remaining parts (*-00002-of-*, etc.)
# ---------------------------------------------------------------------------
declare -a MODEL_PATHS   # ordered list of model paths to benchmark
declare -a MODEL_LABELS  # human-readable labels
declare -a MODEL_ALIASES # reported model id → results filename prefix
declare -a MODEL_IS_MTP  # 1 if the model is an MTP variant, else 0

# Validate a single .gguf path and append its path/label/alias/MTP-flag to the
# model arrays. Skips mmproj sidecars and split-model shards other than 00001.
add_model() {
    local gguf_file="$1"
    local filename
    filename=$(basename "$gguf_file")

    # Skip mmproj files
    if [[ "$filename" == mmproj-* ]]; then
        return
    fi

    # For split models keep only part 00001 (the entry point); skip the rest.
    if [[ "$filename" =~ -([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
        if [[ "${BASH_REMATCH[1]}" != "00001" ]]; then
            warn "Skipping split-model shard (use the 00001 part): $filename"
            return
        fi
    fi

    # Label: path relative to MODELS_DIR when known, otherwise the full path.
    local label
    if [[ -n "$MODELS_DIR" && "$gguf_file" == "${MODELS_DIR}/"* ]]; then
        label="${gguf_file#${MODELS_DIR}/}"
    else
        label="$gguf_file"
    fi

    # Detect MTP variants from the model folder in the path (e.g. Qwen3.6-27B-MTP-GGUF).
    # MTP and non-MTP folders hold identically-named gguf files, so we inject "MTP"
    # into the reported alias to keep their results distinguishable:
    #   Qwen3.6-27B-UD-Q8_K_XL.gguf  →  Qwen3.6-27B-MTP-UD-Q8_K_XL.gguf
    local alias is_mtp=0
    local mtp_dir=""
    local IFS='/'
    local part
    for part in $gguf_file; do
        [[ "$part" == *MTP* ]] && mtp_dir="$part"
    done
    unset IFS
    if [[ -n "$mtp_dir" ]]; then
        local base_mtp="${mtp_dir%-GGUF}"      # Qwen3.6-27B-MTP
        local base_plain="${base_mtp/-MTP/}"   # Qwen3.6-27B
        alias="${base_mtp}${filename#${base_plain}}"
        is_mtp=1
    else
        alias="$filename"
    fi

    MODEL_PATHS+=("$gguf_file")
    MODEL_LABELS+=("$label")
    MODEL_ALIASES+=("$alias")
    MODEL_IS_MTP+=("$is_mtp")
}

if [ "$#" -gt 0 ]; then
    # Explicit model paths — resolve each (absolute, or relative to MODELS_DIR).
    log "Benchmarking ${#} explicitly-specified model(s)."
    for m in "$@"; do
        if [[ -f "$m" ]]; then
            add_model "$(realpath "$m")"
        elif [[ -n "$MODELS_DIR" && -f "${MODELS_DIR}/${m}" ]]; then
            add_model "$(realpath "${MODELS_DIR}/${m}")"
        else
            error "Model path not found: $m"
            exit 1
        fi
    done
elif [[ -n "$MODELS_DIR" ]]; then
    # No explicit paths — discover every model under MODELS_DIR.
    log "No model paths given — discovering all models under ${MODELS_DIR}."
    while IFS= read -r -d '' gguf_file; do
        add_model "$gguf_file"
    done < <(find "$MODELS_DIR" -name "*.gguf" -type f -print0 | sort -z)
else
    error "No models to benchmark. Pass a models directory or .gguf file path(s)."
    exit 1
fi

total=${#MODEL_PATHS[@]}

if [ $total -eq 0 ]; then
    error "No models found under ${MODELS_DIR}"
    exit 1
fi

log "Found ${total} model(s) to benchmark:"
for i in "${!MODEL_LABELS[@]}"; do
    mtp_tag=""
    [[ "${MODEL_IS_MTP[$i]}" == "1" ]] && mtp_tag=" ${YELLOW}[MTP]${NC}"
    echo -e "  $((i+1)). ${MODEL_LABELS[$i]} → ${MODEL_ALIASES[$i]}${mtp_tag}"
done
echo ""

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()

for i in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$i]}"
    model_label="${MODEL_LABELS[$i]}"
    model_alias="${MODEL_ALIASES[$i]}"
    model_is_mtp="${MODEL_IS_MTP[$i]}"
    current=$((i+1))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Model ${current}/${total}: ${model_label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Ensure clean state before starting
    stop_server
    sleep 2

    start_server "$model_path" "$model_label" "$model_alias" "$model_is_mtp"

    if wait_for_server "$model_label"; then
        if run_benchmark "$model_label"; then
            success "Benchmark completed for: ${model_label}"
            PASSED+=("$model_label")
        else
            error "Benchmark FAILED for: ${model_label}"
            FAILED+=("$model_label")
        fi
    else
        error "Server failed to start for: ${model_label}"
        error "Server log:"
        cat "/tmp/llama_server_${model_label//\//_}.log" 2>/dev/null | tail -20 || true
        FAILED+=("$model_label")
    fi

    stop_server
    sleep 3  # brief cooldown between models
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "BENCHMARK RUN COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
success "Passed (${#PASSED[@]}):"
for m in "${PASSED[@]}"; do echo "  ✓ $m"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    error "Failed (${#FAILED[@]}):"
    for m in "${FAILED[@]}"; do echo "  ✗ $m"; done
    exit 1
fi

echo ""
success "All ${total} models benchmarked successfully."