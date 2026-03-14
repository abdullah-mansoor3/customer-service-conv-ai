#!/bin/bash

# Download voice models for ASR and TTS
# This script downloads the required models for voice functionality

set -e

echo "📥 Downloading voice models..."

# Create voice models directory
mkdir -p models/voice

# Download Faster Whisper tiny model (ASR)
echo "📥 Downloading Faster Whisper tiny model..."
python3 -c "
import os
os.makedirs('models/voice/whisper', exist_ok=True)
from faster_whisper import WhisperModel
try:
    model = WhisperModel('tiny', download_root='models/voice/whisper')
    print('✅ Whisper model downloaded')
except Exception as e:
    print(f'❌ Failed to download Whisper model: {e}')
    print('Note: Model will be downloaded automatically on first use')
"

# Download Piper TTS model (English)
echo "📥 Downloading Piper TTS model..."
mkdir -p models/voice/tts
curl -L -o models/voice/tts/en_US-lessac-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
if [ $? -eq 0 ]; then
    echo "✅ Piper TTS model downloaded"
else
    echo "❌ Failed to download Piper TTS model"
    echo "Note: Model will be downloaded automatically on first use"
fi

echo "📥 Downloading Piper TTS config..."
curl -L -o models/voice/tts/en_US-lessac-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
if [ $? -eq 0 ]; then
    echo "✅ Piper TTS config downloaded"
else
    echo "❌ Failed to download Piper TTS config"
fi

echo "✅ Voice models download script completed!"
echo "Models are stored in: models/voice/"
echo "Note: If download failed, models will be downloaded automatically on first use."