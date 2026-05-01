#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    # Export variables from .env, ignoring comments and handling complex strings
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z $key ]] && continue

        # Remove quotes from value if present
        value=$(echo "$value" | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/")

        # Export the variable
        export "$key=$value"
    done < .env
else
    echo ".env file not found! Please create one with MODEL_PATH defined."
    exit 1
fi

if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is not set in the .env file."
    exit 1
fi

echo "Serving model: $MODEL_PATH"

# Determine the executable to use
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

LLAMA_SERVER="llama-server"
if ! command_exists "$LLAMA_SERVER"; then
    if command_exists "server"; then
        LLAMA_SERVER="server"
    elif command_exists "llama-server.exe"; then
        LLAMA_SERVER="llama-server.exe"
    elif command_exists "server.exe"; then
        LLAMA_SERVER="server.exe"
    else
        # Fallback to just running llama-server and hoping it's accessible or aliased
        LLAMA_SERVER="llama-server"
    fi
fi

# Snapshot help text once so we can conditionally use newer flags.
LLAMA_HELP_TEXT="$($LLAMA_SERVER --help 2>/dev/null || true)"
supports_flag() {
    local flag="$1"
    echo "$LLAMA_HELP_TEXT" | grep -q -- "$flag"
}

cpu_model_name() {
    if command_exists "lscpu"; then
        lscpu | awk -F: '/Model name/ {gsub(/^[[:space:]]+/, "", $2); print $2; exit}'
        return
    fi
    echo ""
}

detect_physical_cores() {
    local os_name
    os_name="$(uname -s)"
    if [ "$os_name" = "Linux" ] && command_exists "lscpu"; then
        lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l
        return
    fi

    if [[ "$os_name" == "MINGW"* ]] || [[ "$os_name" == "MSYS"* ]] || [[ "$os_name" == "CYGWIN"* ]]; then
        echo "$NUMBER_OF_PROCESSORS"
        return
    fi

    echo "4"
}

detect_logical_cores() {
    if command_exists "getconf"; then
        getconf _NPROCESSORS_ONLN
        return
    fi
    if command_exists "nproc"; then
        nproc
        return
    fi
    echo "8"
}

# Detect number of cores for efficient CPU inference.
CPU_MODEL="$(cpu_model_name)"
IS_GEN12_I7=0
if echo "$CPU_MODEL" | grep -Ei '12th Gen Intel.*Core.*i7|i7-12[0-9]{3}' >/dev/null 2>&1; then
    IS_GEN12_I7=1
fi

if [ -n "$LLAMA_THREADS" ]; then
    THREADS=$LLAMA_THREADS
    echo "Using LLAMA_THREADS=$THREADS from .env settings."
else
    PHYSICAL_CORES="$(detect_physical_cores)"
    THREADS="$PHYSICAL_CORES"

    # Ensure THREADS is a valid number, fallback to 4 if detection fails.
    if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
        THREADS=4
    fi

    # Alder Lake i7 is hybrid (P+E cores). For token latency, cap generation
    # threads to a P-core-friendly value unless user overrides explicitly.
    if [ "$IS_GEN12_I7" -eq 1 ] && [ "$THREADS" -gt 8 ]; then
        THREADS=8
    fi

    echo "Auto thread tuning -> model='$CPU_MODEL', generation threads=$THREADS"
fi

if [ -n "$LLAMA_BATCH_THREADS" ]; then
    BATCH_THREADS=$LLAMA_BATCH_THREADS
    echo "Using LLAMA_BATCH_THREADS=$BATCH_THREADS for prompt evaluation."
else
    LOGICAL_CORES="$(detect_logical_cores)"
    if ! [[ "$LOGICAL_CORES" =~ ^[0-9]+$ ]]; then
        LOGICAL_CORES=$THREADS
    fi

    if [ "$IS_GEN12_I7" -eq 1 ]; then
        # Use a few extra threads for prompt/batch work while keeping token
        # generation pinned to a lower-latency thread count.
        BATCH_THREADS=$((THREADS + 4))
        if [ "$BATCH_THREADS" -gt "$LOGICAL_CORES" ]; then
            BATCH_THREADS=$LOGICAL_CORES
        fi
    else
        BATCH_THREADS=$THREADS
    fi
fi

CTX_SIZE="${LLAMA_CTX_SIZE:-2048}"
BATCH_SIZE="${LLAMA_BATCH_SIZE:-512}"
UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-128}"
PARALLEL_SLOTS="${LLAMA_PARALLEL_SLOTS:-2}"
HTTP_THREADS="${LLAMA_HTTP_THREADS:-4}"
POLL_LEVEL="${LLAMA_POLL_LEVEL:-0}"
VERBOSE_LOGS="${LLAMA_VERBOSE:-0}"
LOG_VERBOSE="${LLAMA_LOG_VERBOSE:-0}"
THINKING="${LLAMA_THINKING:-1}"

# Dynamically infer the chat template based on the model filename
MODEL_BASENAME=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
CHAT_TEMPLATE=""

case "$MODEL_BASENAME" in
    *llama-3*|*llama3*|*llama_3*)
        # Llama-3 models typically need their own template
        CHAT_TEMPLATE="llama3"
        ;;
    *gemma*)
        CHAT_TEMPLATE="gemma"
        ;;
    *phi-3*|*phi3*)
        CHAT_TEMPLATE="phi3"
        ;;
    *chatml*|*wizard*|*zephyr*)
        # Many models use ChatML format (WizardLM, Zephyr, Dolphin)
        CHAT_TEMPLATE="chatml"
        ;;
    *qwen3*)
        # Qwen3 has its own embedded chat template with thinking/non-thinking
        # mode support. Let llama-server read it from the GGUF metadata.
        CHAT_TEMPLATE=""
        ;;
    *qwen*)
        # Older Qwen models (Qwen1, Qwen2, Qwen2.5) use ChatML format
        CHAT_TEMPLATE="chatml"
        ;;
    *mistral*|*mixtral*)
        # Some mistral iterations use chatml, others might just rely on auto-detection or 'mistral' template if supported later.
        # It's safest to leave blank to let llama.cpp extract the embedded template for standard Mistral models, 
        # unless it explicitly says chatml.
        # But for generic fallback we will use chatml if not embedded.
        # Let's leave empty by default to prefer the embedded metadata for mistral.
        ;;
    *)
        # Default: empty (let llama-server try to read it from the embedded GGUF metadata)
        ;;
esac

# llama.cpp server flags for efficient CPU inference.
CMD="$LLAMA_SERVER -m \"$MODEL_PATH\" -c $CTX_SIZE -t $THREADS -tb $BATCH_THREADS -b $BATCH_SIZE"

if supports_flag --ubatch-size || supports_flag --ubatch; then
    CMD="$CMD -ub $UBATCH_SIZE"
fi

if supports_flag --parallel; then
    CMD="$CMD -np $PARALLEL_SLOTS"
fi

if supports_flag --threads-http; then
    CMD="$CMD --threads-http $HTTP_THREADS"
fi

if supports_flag --poll; then
    CMD="$CMD --poll $POLL_LEVEL"
fi

if supports_flag --cache-prompt; then
    CMD="$CMD --cache-prompt"
fi

if supports_flag --reasoning-budget; then
    if [ "$THINKING" = "0" ]; then
        CMD="$CMD --reasoning-budget 0"
        echo "Thinking mode: DISABLED"
    else
        CMD="$CMD --reasoning-budget -1"
        echo "Thinking mode: ENABLED"
    fi
fi

if [ "${LLAMA_MLOCK:-0}" = "1" ] && supports_flag --mlock; then
    CMD="$CMD --mlock"
fi

CPU_MASK="${LLAMA_CPU_MASK:-}"
if [ -n "$CPU_MASK" ] && supports_flag --cpu-mask; then
    CMD="$CMD --cpu-mask $CPU_MASK"
    echo "CPU affinity mask: $CPU_MASK"
fi

if supports_flag --flash-attn; then
    CMD="$CMD --flash-attn off"
fi

if [ "$VERBOSE_LOGS" = "1" ]; then
    if supports_flag --verbose; then
        CMD="$CMD --verbose"
    fi
    if [ "$LOG_VERBOSE" = "1" ] && supports_flag --log-verbose; then
        CMD="$CMD --log-verbose"
    fi
    if supports_flag --log-timestamps; then
        CMD="$CMD --log-timestamps"
    fi
fi

if [ -n "$CHAT_TEMPLATE" ] && supports_flag --chat-template; then
    echo "Detected template heuristics. Forcing chat template: $CHAT_TEMPLATE"
    CMD="$CMD --chat-template $CHAT_TEMPLATE"
else
    echo "No explicit chat template heuristic matched (or relying on auto-detect). Letting llama-server read embedded metadata."
fi

echo "Runtime tuning: ctx=$CTX_SIZE, batch=$BATCH_SIZE, ubatch=$UBATCH_SIZE, parallel_slots=$PARALLEL_SLOTS"
echo "CPU tuning: threads=$THREADS, batch_threads=$BATCH_THREADS, http_threads=$HTTP_THREADS, poll=$POLL_LEVEL"
echo "Verbose llama logs: $VERBOSE_LOGS"
echo "Token-level verbose logs: $LOG_VERBOSE"

echo "Executing: $CMD"
eval "$CMD"
