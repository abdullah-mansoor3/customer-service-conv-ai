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
import uuid
import wave
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from faster_whisper import WhisperModel
from piper import PiperVoice

from app.config import LLAMA_SERVER_URL
from app.store import DEFAULT_STATE, sessions

router = APIRouter()

# Global model instances (lazy loaded)
_whisper_model = None
_piper_voices: dict[str, PiperVoice] = {}

REPO_ROOT = Path(__file__).resolve().parents[3]
VOICE_MODELS_DIR = REPO_ROOT / "models" / "voice"
VOICE_CATALOG = {
    "ella": {
        "name": "Ella",
        "description": "US English female, warm and clear",
        "hf_path": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    },
    "john": {
        "name": "John",
        "description": "US English male, neutral and calm",
        "hf_path": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    },
    "olivia": {
        "name": "Olivia",
        "description": "US English female, expressive",
        "hf_path": "en/en_US/lessac/high/en_US-lessac-high.onnx",
    },
    "mason": {
        "name": "Mason",
        "description": "US English male, deeper voice",
        "hf_path": "en/en_US/ryan/high/en_US-ryan-high.onnx",
    },
    "alba": {
        "name": "Alba",
        "description": "British English female",
        "hf_path": "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
    },
}
DEFAULT_VOICE_ID = "ella"
HF_VOICES_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

def get_whisper_model():
    """Lazy load Whisper model for ASR."""
    global _whisper_model
    if _whisper_model is None:
        model_path = str(VOICE_MODELS_DIR / "whisper")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Whisper model not found. Run download_voice_models.sh")
        _whisper_model = WhisperModel("tiny", download_root=model_path, device="cpu", compute_type="int8")
    return _whisper_model

def _resolve_voice_id(voice_id: str | None) -> str:
    if voice_id and voice_id in VOICE_CATALOG:
        return voice_id
    return DEFAULT_VOICE_ID


def _ensure_voice_assets(voice_id: str) -> Path:
    voice_meta = VOICE_CATALOG[voice_id]
    model_filename = Path(voice_meta["hf_path"]).name
    tts_dir = VOICE_MODELS_DIR / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    model_path = tts_dir / model_filename
    config_path = Path(f"{model_path}.json")
    model_url = f"{HF_VOICES_BASE_URL}{voice_meta['hf_path']}"
    config_url = f"{model_url}.json"

    if not model_path.exists():
        try:
            urlretrieve(model_url, str(model_path))
            logging.info("Downloaded Piper model for voice '%s': %s", voice_id, model_path)
        except URLError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not download voice model '{voice_id}'. "
                    "Please check internet and try again."
                ),
            ) from exc

    if not config_path.exists():
        try:
            urlretrieve(config_url, str(config_path))
            logging.info("Downloaded Piper config for voice '%s': %s", voice_id, config_path)
        except URLError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not download voice config for '{voice_id}'. "
                    "Please check internet and try again."
                ),
            ) from exc

    return model_path


def get_piper_voice(voice_id: str | None = None) -> PiperVoice:
    """Lazy load and cache Piper voice by voice id."""
    resolved_voice_id = _resolve_voice_id(voice_id)
    if resolved_voice_id in _piper_voices:
        return _piper_voices[resolved_voice_id]

    model_path = _ensure_voice_assets(resolved_voice_id)
    _piper_voices[resolved_voice_id] = PiperVoice.load(str(model_path))
    return _piper_voices[resolved_voice_id]


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


def synthesize_wav_bytes(text: str, voice_id: str | None = None) -> bytes:
    """Run Piper synthesis and return WAV bytes."""
    voice = get_piper_voice(voice_id)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return wav_buffer.getvalue()


def iter_transcription_segments(audio_data: bytes, suffix: str = ".webm"):
    """Yield transcription segments progressively for live UX updates."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
        temp_audio.write(audio_data)
        temp_audio_path = temp_audio.name

    try:
        model = get_whisper_model()
        segments, _ = model.transcribe(temp_audio_path, language="en")
        for segment in segments:
            text = segment.text.strip()
            if text:
                yield text
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

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


@router.get("/api/voices")
async def list_voices():
    """List available voice profiles for frontend selection."""
    return {
        "default_voice_id": DEFAULT_VOICE_ID,
        "voices": [
            {
                "id": voice_id,
                "name": meta["name"],
                "description": meta["description"],
            }
            for voice_id, meta in VOICE_CATALOG.items()
        ],
    }

@router.post("/tts")
async def text_to_speech(data: dict):
    """
    Convert text to speech using Piper TTS.
    Returns WAV audio data.
    """
    try:
        text = data.get("text", "").strip()
        voice_id = data.get("voice_id")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        audio_data = await asyncio.to_thread(synthesize_wav_bytes, text, voice_id)
        return Response(content=audio_data, media_type="audio/wav")

    except Exception as e:
        logging.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@router.websocket("/ws/voice-chat")
async def voice_chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice chat with cancellation support."""
    await websocket.accept()

    current_task: asyncio.Task | None = None
    cancel_event: asyncio.Event | None = None
    selected_voice_id = DEFAULT_VOICE_ID
    selected_session_id: str | None = None
    active_preload_task: asyncio.Task | None = None
    active_preload_voice_id: str | None = None
    preload_generation = 0

    def ensure_voice_preload_task(voice_id: str) -> asyncio.Task:
        nonlocal active_preload_task, active_preload_voice_id, preload_generation

        # If this voice is already cached, no background preload is needed.
        if voice_id in _piper_voices:
            done_task = asyncio.create_task(asyncio.sleep(0))
            active_preload_task = done_task
            active_preload_voice_id = voice_id
            return done_task

        # Same voice already loading -> reuse.
        if (
            active_preload_task is not None
            and not active_preload_task.done()
            and active_preload_voice_id == voice_id
        ):
            return active_preload_task

        # Different voice was loading -> cancel stale preload request.
        if active_preload_task is not None and not active_preload_task.done():
            active_preload_task.cancel()

        preload_generation += 1
        active_preload_voice_id = voice_id
        active_preload_task = asyncio.create_task(asyncio.to_thread(get_piper_voice, voice_id))
        return active_preload_task

    await websocket.send_json({
        "type": "voice_catalog",
        "default_voice_id": DEFAULT_VOICE_ID,
        "voices": [
            {"id": voice_id, "name": meta["name"], "description": meta["description"]}
            for voice_id, meta in VOICE_CATALOG.items()
        ],
    })
    await websocket.send_json({"type": "voice_selected", "voice_id": selected_voice_id})

    # Preload default voice immediately to reduce first-turn latency.
    if selected_voice_id in _piper_voices:
        await websocket.send_json({"type": "voice_ready", "voice_id": selected_voice_id})
    else:
        preload_task = ensure_voice_preload_task(selected_voice_id)
        request_generation = preload_generation
        await websocket.send_json({"type": "voice_loading", "voice_id": selected_voice_id})

        def _notify_default_preload_done(task: asyncio.Task, voice_id: str = selected_voice_id):
            if task.cancelled():
                return
            if request_generation != preload_generation or voice_id != selected_voice_id:
                return
            try:
                task.result()
                asyncio.create_task(websocket.send_json({"type": "voice_ready", "voice_id": voice_id}))
            except Exception as exc:  # noqa: BLE001
                asyncio.create_task(
                    websocket.send_json({
                        "type": "error",
                        "error": f"Voice preload failed for '{voice_id}': {exc}",
                    })
                )

        preload_task.add_done_callback(_notify_default_preload_done)

    async def process_voice_turn(
        turn_audio_data: bytes,
        turn_cancel_event: asyncio.Event,
        turn_voice_id: str,
        turn_session_id: str,
    ):
        try:
            preload_task = ensure_voice_preload_task(turn_voice_id)

            # Step 1: ASR - stream partial transcription updates
            partial_text = ""
            for segment_text in iter_transcription_segments(turn_audio_data, suffix=".webm"):
                if turn_cancel_event.is_set():
                    return
                partial_text = f"{partial_text} {segment_text}".strip()
                await websocket.send_json({"type": "transcription_partial", "text": partial_text})
                await asyncio.sleep(0)

            user_text = partial_text.strip()
            if not user_text:
                await websocket.send_json({
                    "type": "error",
                    "error": "No speech detected. Please try again.",
                })
                return

            await websocket.send_json({"type": "transcription_final", "text": user_text})
            if turn_cancel_event.is_set():
                return

            # Step 2: Process through LLM (reuse existing chat logic)
            import httpx
            import re
            from app.config import DEFAULT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, MAX_HISTORY

            session_id = turn_session_id
            if session_id not in sessions:
                sessions[session_id] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "messages": [],
                    "state": {**DEFAULT_STATE},
                }

            sessions[session_id]["messages"].append({"role": "user", "content": user_text})

            state_str = json.dumps(sessions[session_id]["state"])
            system_prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                "/no_think\n\n"
                "You MUST begin EVERY response with a <STATE> JSON block "
                "that tracks what you know so far, then a newline, then "
                "your short reply (1-2 sentences max).\n\n"
                "The STATE JSON has exactly these 5 keys:\n"
                "  router_model, lights_status, error_message, connection_type, has_restarted\n"
                "Use null for anything the user has NOT mentioned yet. "
                "Only fill a field when the user explicitly states it.\n\n"
                "=== FORMAT EXAMPLE (not real data, ignore these values) ===\n"
                "If a user said their BrandX router has green lights:\n"
                '<STATE>{"router_model": "BrandX", "lights_status": "green", '
                '"error_message": null, "connection_type": null, '
                '"has_restarted": null}</STATE>\n'
                "I see your BrandX has green lights. What error are you getting?\n"
                "=== END FORMAT EXAMPLE ===\n\n"
                f"CURRENT KNOWN STATE (carry forward and update):\n{state_str}"
            )

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
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{LLAMA_SERVER_URL}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            if turn_cancel_event.is_set():
                return

            ai_response_full = result["choices"][0]["message"]["content"]
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
                    print(
                        f"✅ Extracted State Update (voice, session={session_id}): "
                        f"{sessions[session_id]['state']}"
                    )
                except json.JSONDecodeError:
                    ai_response = ai_response_full.replace("<STATE>", "").replace("</STATE>", "").strip()
            else:
                ai_response = ai_response_full.replace("<STATE>", "").replace("</STATE>", "").strip()
                print("⚠️  No valid <STATE>{…}</STATE> block found in voice LLM response.")

            sessions[session_id]["messages"].append({"role": "assistant", "content": ai_response})

            await websocket.send_json({"type": "assistant_text", "text": ai_response})
            if turn_cancel_event.is_set():
                return

            # Step 3: TTS - run off event loop
            try:
                # Wait for selected voice preload (likely already in progress while user is speaking).
                await asyncio.wait_for(asyncio.shield(preload_task), timeout=45.0)
                if turn_cancel_event.is_set():
                    return

                audio_bytes = await asyncio.wait_for(
                    asyncio.to_thread(synthesize_wav_bytes, ai_response, turn_voice_id),
                    timeout=35.0,
                )
                if turn_cancel_event.is_set():
                    return
                await websocket.send_bytes(audio_bytes)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "error",
                    "error": "TTS generation timed out while preparing selected voice. Please retry.",
                })

        except asyncio.CancelledError:
            # Task cancelled by user; ignore quietly.
            return
        except Exception as e:
            logging.error(f"Voice processing error: {e}")
            await websocket.send_json({
                "type": "error",
                "error": f"Voice processing failed: {str(e)}",
            })

    try:
        while True:
            message = await websocket.receive()

            if current_task is not None and current_task.done():
                current_task = None
                cancel_event = None

            if message.get("type") == "websocket.disconnect":
                break

            text_payload = message.get("text")
            bytes_payload = message.get("bytes")

            if text_payload is not None:
                try:
                    data = json.loads(text_payload)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "Invalid voice control payload."})
                    continue

                if data.get("type") == "cancel_current":
                    if cancel_event is not None:
                        cancel_event.set()
                    if current_task is not None and not current_task.done():
                        current_task.cancel()
                    await websocket.send_json({"type": "cancelled", "stage": "processing"})
                    continue

                if data.get("type") == "set_voice":
                    requested_voice_id = data.get("voice_id")
                    if requested_voice_id not in VOICE_CATALOG:
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Unknown voice: {requested_voice_id}",
                        })
                        continue
                    selected_voice_id = requested_voice_id
                    await websocket.send_json({"type": "voice_selected", "voice_id": selected_voice_id})

                    if selected_voice_id in _piper_voices:
                        await websocket.send_json({"type": "voice_ready", "voice_id": selected_voice_id})
                    else:
                        preload_task = ensure_voice_preload_task(selected_voice_id)
                        request_generation = preload_generation
                        await websocket.send_json({"type": "voice_loading", "voice_id": selected_voice_id})

                        def _notify_preload_done(task: asyncio.Task, voice_id: str = selected_voice_id):
                            if task.cancelled():
                                return
                            # Ignore stale completions if user switched voices again.
                            if request_generation != preload_generation or voice_id != selected_voice_id:
                                return
                            try:
                                task.result()
                                asyncio.create_task(
                                    websocket.send_json({"type": "voice_ready", "voice_id": voice_id})
                                )
                            except Exception as exc:  # noqa: BLE001
                                asyncio.create_task(
                                    websocket.send_json({
                                        "type": "error",
                                        "error": f"Voice preload failed for '{voice_id}': {exc}",
                                    })
                                )

                        preload_task.add_done_callback(_notify_preload_done)
                    continue

                if data.get("type") == "set_session":
                    requested_session_id = data.get("session_id")
                    if not requested_session_id or requested_session_id == "new_session":
                        selected_session_id = str(uuid.uuid4())
                    else:
                        selected_session_id = str(requested_session_id)

                    if selected_session_id not in sessions:
                        sessions[selected_session_id] = {
                            "created_at": datetime.utcnow().isoformat(),
                            "messages": [],
                            "state": {**DEFAULT_STATE},
                        }

                    await websocket.send_json({"type": "session_id", "session_id": selected_session_id})
                    continue

                await websocket.send_json({"type": "error", "error": "Unknown voice command."})
                continue

            if bytes_payload is not None:
                if current_task is not None and not current_task.done():
                    await websocket.send_json({
                        "type": "error",
                        "error": "A voice request is already running. Cancel it before starting another.",
                    })
                    continue

                if not selected_session_id:
                    selected_session_id = str(uuid.uuid4())
                    if selected_session_id not in sessions:
                        sessions[selected_session_id] = {
                            "created_at": datetime.utcnow().isoformat(),
                            "messages": [],
                            "state": {**DEFAULT_STATE},
                        }
                    await websocket.send_json({"type": "session_id", "session_id": selected_session_id})

                cancel_event = asyncio.Event()
                current_task = asyncio.create_task(
                    process_voice_turn(
                        bytes_payload,
                        cancel_event,
                        selected_voice_id,
                        selected_session_id,
                    )
                )
                continue

    except WebSocketDisconnect:
        logging.info("Voice chat WebSocket disconnected")
    finally:
        if cancel_event is not None:
            cancel_event.set()
        if current_task is not None and not current_task.done():
            current_task.cancel()
        if active_preload_task is not None and not active_preload_task.done():
            active_preload_task.cancel()