#!/usr/bin/env bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration. Override these env vars before running the script.
###############################################################################
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p "$LOG_PATH"

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_DECODE_PORT="${PREFILL_DECODE_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"
SERVER_NAME="${SERVER_NAME:-127.0.0.1}"

GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"
TP_E="${TP_E:-1}"
TP_PD="${TP_PD:-1}"

ENCODER_TRANSFER_BACKEND="${ENCODER_TRANSFER_BACKEND:-dlslime_rdma}"

MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
SESSION_LEN="${SESSION_LEN:-32768}"
CACHE_MAX_ENTRY_COUNT="${CACHE_MAX_ENTRY_COUNT:-0.75}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"

# Optional smoke request. Example:
# IMAGE_URL=file:///path/to/image.jpg bash examples/epd/disagg_1e1pd_example.sh
IMAGE_URL="${IMAGE_URL:-}"

START_TIME="$(date +"%Y%m%d_%H%M%S")"
PROXY_LOG="${LOG_PATH}/epd_proxy_${START_TIME}.log"
ENC_LOG="${LOG_PATH}/epd_encoder_${START_TIME}.log"
PD_LOG="${LOG_PATH}/epd_pd_${START_TIME}.log"

###############################################################################
# Helpers
###############################################################################
cleanup() {
    echo "Stopping EPD example services..."
    trap - INT TERM
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    wait || true
}

trap cleanup EXIT
trap exit INT TERM

wait_for_url() {
    local url="$1"
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -fsS '$url' >/dev/null 2>&1; do
            sleep 1
        done"
}

###############################################################################
# Proxy. LMDeploy API servers register themselves with this process.
###############################################################################
lmdeploy serve proxy \
    --server-name "$SERVER_NAME" \
    --server-port "$PROXY_PORT" \
    --serving-strategy Hybrid \
    --disable-cache-status \
    >"$PROXY_LOG" 2>&1 &
PIDS+=("$!")

wait_for_url "http://${SERVER_NAME}:${PROXY_PORT}/nodes/status"

###############################################################################
# Encoder worker. This node computes multimodal encoder outputs only.
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" lmdeploy serve api_server "$MODEL" \
    --backend pytorch \
    --tp "$TP_E" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --cache-max-entry-count "$CACHE_MAX_ENTRY_COUNT" \
    --session-len "$SESSION_LEN" \
    --trust-remote-code \
    --server-name "$SERVER_NAME" \
    --server-port "$ENCODE_PORT" \
    --proxy-url "http://${SERVER_NAME}:${PROXY_PORT}" \
    --role Encoder \
    --encoder-only \
    --encoder-transfer-backend "$ENCODER_TRANSFER_BACKEND" \
    >"$ENC_LOG" 2>&1 &
PIDS+=("$!")

###############################################################################
# Prefill+Decode worker. This node consumes encoder output and runs the LLM.
###############################################################################
pd_args=(
    lmdeploy serve api_server "$MODEL"
    --backend pytorch
    --tp "$TP_PD"
    --max-batch-size "$MAX_BATCH_SIZE"
    --cache-max-entry-count "$CACHE_MAX_ENTRY_COUNT"
    --session-len "$SESSION_LEN"
    --trust-remote-code
    --server-name "$SERVER_NAME"
    --server-port "$PREFILL_DECODE_PORT"
    --proxy-url "http://${SERVER_NAME}:${PROXY_PORT}"
    --role Hybrid
    --language-only
    --encoder-transfer-backend "$ENCODER_TRANSFER_BACKEND"
)

CUDA_VISIBLE_DEVICES="$GPU_PD" "${pd_args[@]}" >"$PD_LOG" 2>&1 &
PIDS+=("$!")

wait_for_url "http://${SERVER_NAME}:${ENCODE_PORT}/v1/models"
wait_for_url "http://${SERVER_NAME}:${PREFILL_DECODE_PORT}/v1/models"

echo "EPD E+PD services are up."
echo "Proxy: http://${SERVER_NAME}:${PROXY_PORT}/v1/chat/completions"
echo "Logs: $PROXY_LOG $ENC_LOG $PD_LOG"

if [[ -n "$IMAGE_URL" ]]; then
    curl "http://${SERVER_NAME}:${PROXY_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"${MODEL}"'",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": "'"${IMAGE_URL}"'"}}
                ]
            }],
            "max_tokens": 96,
            "stream": false,
            "chat_template_kwargs": {"enable_thinking": false}
        }'
    echo
else
    echo "Set IMAGE_URL to run a multimodal smoke request."
    echo "Press Ctrl-C to stop the services."
    while true; do
        sleep 3600
    done
fi
