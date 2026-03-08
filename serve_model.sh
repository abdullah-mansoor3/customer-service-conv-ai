#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    # Export variables from .env, ignoring comments
    export $(grep -v '^#' .env | xargs)
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

# Detect number of physical cores for efficient CPU inference
if [ -n "$LLAMA_THREADS" ]; then
    THREADS=$LLAMA_THREADS
    echo "Using LLAMA_THREADS=$THREADS from .env settings."
else
    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        # Get physical cores on Linux to avoid hyperthreading contention
        THREADS=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,4 | wc -l)
    elif [[ "$OS" == "MINGW"* ]] || [[ "$OS" == "MSYS"* ]] || [[ "$OS" == "CYGWIN"* ]]; then
        # Windows native or via bash
        THREADS=$NUMBER_OF_PROCESSORS
    else
        THREADS=4 # Fallback
    fi

    # Ensure THREADS is a valid number, fallback to 4 if detection fails
    if ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
        THREADS=4
    fi
    echo "Detected $THREADS physical CPU cores. Using this for inference threads."
fi

if [ -n "$LLAMA_BATCH_THREADS" ]; then
    BATCH_THREADS=$LLAMA_BATCH_THREADS
    echo "Using LLAMA_BATCH_THREADS=$BATCH_THREADS for prompt evaluation."
else
    BATCH_THREADS=$THREADS
fi

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

# llama.cpp server flags for efficient CPU inference:
# -m: Model path (loaded from .env)
# -c: Context size (2048 is a safe default, adjust as needed)
# -t: Number of threads for generation (optimally set to physical cores)
# -tb: Number of threads for prompt processing (batch processing)
# -b: Batch size for prompt processing
# --mlock: Force system to keep model in RAM (optional, uncomment if you have enough RAM and want to avoid swapping)

CMD="$LLAMA_SERVER -m \"$MODEL_PATH\" -c 2048 -t $THREADS -tb $BATCH_THREADS -b 512"

if [ -n "$CHAT_TEMPLATE" ]; then
    echo "Detected template heuristics. Forcing chat template: $CHAT_TEMPLATE"
    CMD="$CMD --chat-template $CHAT_TEMPLATE"
else
    echo "No explicit chat template heuristic matched (or relying on auto-detect). Letting llama-server read embedded metadata."
fi

# Uncomment the following line to enable mlock for potentially better performance
# CMD="$CMD --mlock"

echo "Executing: $CMD"
eval "$CMD"
