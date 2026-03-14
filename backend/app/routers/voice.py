"""
Voice processing endpoints for ASR and TTS.

Provides endpoints for:
- Converting speech to text (ASR)
- Converting text to speech (TTS)
- Streaming voice chat via WebSocket
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import wave
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from faster_whisper import WhisperModel
from piper import PiperVoice

from app.config import LLAMA_SERVER_URL
from app.store import sessions

router = APIRouter()

# Global model instances (lazy loaded)
_whisper_model = None
_piper_voice = None

REPO_ROOT = Path(__file__).resolve().parents[3]
VOICE_MODELS_DIR = REPO_ROOT / "models" / "voice"
PIPER_MODEL_FILENAME = "en_US-lessac-medium.onnx"
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
)

def get_whisper_model():
    """Lazy load Whisper model for ASR."""
    global _whisper_model
    if _whisper_model is None:
        model_path = str(VOICE_MODELS_DIR / "whisper")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Whisper model not found. Run download_voice_models.sh")
        _whisper_model = WhisperModel("tiny", download_root=model_path, device="cpu", compute_type="int8")
    return _whisper_model

def get_piper_voice():
    """Lazy load Piper voice for TTS."""
    global _piper_voice
    if _piper_voice is None:
        model_path = VOICE_MODELS_DIR / "tts" / PIPER_MODEL_FILENAME
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Piper TTS model not found. Run download_voice_models.sh")

        # Piper requires a sibling JSON config: <model>.json
        config_path = Path(f"{model_path}.json")
        if not config_path.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(PIPER_CONFIG_URL, str(config_path))
                logging.info("Downloaded missing Piper config: %s", config_path)
            except URLError as exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Piper config file is missing and could not be downloaded. "
                        "Expected file: "
                        f"{config_path}. "
                        "Please run download_voice_models.sh to fetch both "
                        ".onnx and .onnx.json files."
                    ),
                ) from exc

        _piper_voice = PiperVoice.load(str(model_path))
    return _piper_voice


def transcribe_audio_bytes(audio_data: bytes, suffix: str = ".webm") -> str:
    """Transcribe raw audio bytes by writing to a temp file first.

    Browser MediaRecorder commonly sends webm/opus chunks that soundfile
    cannot decode directly from a BytesIO buffer. Faster Whisper can decode
    these via ffmpeg when given a file path.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(temp_audio_path, language="en")
        return " ".join(segment.text for segment in segments).strip()
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


def synthesize_wav_bytes(text: str) -> bytes:
    """Run Piper synthesis and return WAV bytes."""
    voice = get_piper_voice()
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return wav_buffer.getvalue()

@router.post("/asr")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert uploaded audio file to text using Whisper.
    Accepts common formats (webm/wav/mp3/ogg) if ffmpeg is available.
    """
    try:
        audio_data = await audio.read()
        filename = (audio.filename or "audio.webm").lower()
        suffix = Path(filename).suffix or ".webm"
        text = transcribe_audio_bytes(audio_data, suffix=suffix)
        return {"text": text, "language": "en"}

    except Exception as e:
        logging.error(f"ASR error: {e}")
        raise HTTPException(status_code=500, detail=f"Speech recognition failed: {str(e)}")

@router.post("/tts")
async def text_to_speech(data: dict):
    """
    Convert text to speech using Piper TTS.
    Returns WAV audio data.
    """
    try:
        text = data.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        audio_data = await asyncio.to_thread(synthesize_wav_bytes, text)
        return Response(content=audio_data, media_type="audio/wav")

    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@router.websocket("/ws/voice-chat")
async def voice_chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice chat.
    Handles voice input -> ASR -> LLM -> TTS -> voice output
    """
    await websocket.accept()

    try:
        while True:
            # Receive audio data from client
            audio_data = await websocket.receive_bytes()

            try:
                # Step 1: ASR - Convert speech to text
                user_text = transcribe_audio_bytes(audio_data, suffix=".webm")

                if not user_text:
                    continue  # Skip if no speech detected

                # Send transcription to client for display
                await websocket.send_json({
                    "type": "transcription",
                    "text": user_text
                })

                # Step 2: Process through LLM (reuse existing chat logic)
                import httpx
                from app.config import DEFAULT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, MAX_HISTORY

                # Get or create session
                session_id = "voice_session"  # Could be made dynamic

                if session_id not in sessions:
                    sessions[session_id] = {
                        "created_at": "voice_session",
                        "messages": [],
                        "state": {"router_model": None, "lights_status": None, "error_message": None,
                                "connection_type": None, "has_restarted": None}
                    }

                # Add user message
                sessions[session_id]["messages"].append({"role": "user", "content": user_text})

                # Build prompt like in chat.py
                state_str = json.dumps(sessions[session_id]["state"])
                system_prompt = (
                    f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                    "/no_think\n\n"
                    "You MUST begin EVERY response with a <STATE> JSON block "
                    "that tracks what you know so far, then a newline, then "
                    "your short reply (1-2 sentences max).\n\n"
                    "The STATE JSON has exactly these 5 keys:\n"
                    '  router_model, lights_status, error_message, '
                    'connection_type, has_restarted\n'
                    'Use null for anything the user has NOT mentioned yet. '
                    'Only fill a field when the user explicitly states it.\n\n'
                    "=== FORMAT EXAMPLE (not real data, ignore these values) ===\n"
                    'If a user said their BrandX router has green lights:\n'
                    '<STATE>{"router_model": "BrandX", "lights_status": "green", '
                    '"error_message": null, "connection_type": null, '
                    '"has_restarted": null}</STATE>\n'
                    "I see your BrandX has green lights. What error are you getting?\n"
                    "=== END FORMAT EXAMPLE ===\n\n"
                    f"CURRENT KNOWN STATE (carry forward and update):\n{state_str}"
                )

                # Prepare messages for LLM
                history = sessions[session_id]["messages"][-MAX_HISTORY:]
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(FEW_SHOT_EXAMPLES)
                messages.extend(history)

                payload = {
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_tokens": -1,
                    "stream": False,  # Not streaming for voice
                }

                # Get LLM response
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{LLAMA_SERVER_URL}/v1/chat/completions",
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()

                ai_response_full = result["choices"][0]["message"]["content"]

                # Extract clean response (remove STATE block)
                import re
                match = re.search(r"<STATE>\s*(\{.*?\})\s*</STATE>\s*(.*)", ai_response_full, re.DOTALL)
                if match:
                    try:
                        new_state = json.loads(match.group(1))
                        if isinstance(new_state, dict):
                            for key, value in new_state.items():
                                if value is not None and str(value).strip().lower() not in ("", "null", "none"):
                                    if key in sessions[session_id]["state"]:
                                        sessions[session_id]["state"][key] = value
                        ai_response = match.group(2).strip()
                    except json.JSONDecodeError:
                        ai_response = ai_response_full.replace("<STATE>", "").replace("</STATE>", "").strip()
                else:
                    ai_response = ai_response_full.replace("<STATE>", "").replace("</STATE>", "").strip()

                # Add AI response to session
                sessions[session_id]["messages"].append({"role": "assistant", "content": ai_response})

                # Send assistant text immediately so UI stops waiting even if TTS is slow.
                await websocket.send_json({
                    "type": "assistant_text",
                    "text": ai_response,
                })

                # Step 3: TTS - Convert response to speech
                try:
                    # Piper synthesis is blocking/CPU-heavy; run it off the event loop.
                    audio_bytes = await asyncio.wait_for(
                        asyncio.to_thread(synthesize_wav_bytes, ai_response),
                        timeout=20.0,
                    )

                    await websocket.send_bytes(audio_bytes)

                except asyncio.TimeoutError:
                    await websocket.send_json({
                        "type": "error",
                        "error": "TTS generation timed out. Please try a shorter prompt.",
                    })

            except Exception as e:
                logging.error(f"Voice processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Voice processing failed: {str(e)}"
                })

    except WebSocketDisconnect:
        logging.info("Voice chat WebSocket disconnected")