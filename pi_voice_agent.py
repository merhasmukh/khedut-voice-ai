#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import uuid

import pyaudio
from dotenv import load_dotenv

load_dotenv()

# -- Optional pygame for experience MP3 ---------------------------------------
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("WARNING: pygame not installed. Experience audio (MP3) will be skipped.")
    print("         Install with: pip install pygame")

# -- websockets ---------------------------------------------------------------
try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# =============================================================================
# Import ALL shared code from existing modules — zero duplication
# =============================================================================
from ai_services.gemini_api import (
    BASE_SYSTEM_INSTRUCTION,       # same Gujarati system prompt
    KNOWLEDGE_BASE_TOOL,           # same tool declarations (search + audio)
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    GEMINI_WS_BASE_URL,
    get_gemini_ws_url,             # builds wss://...?key=...
    build_setup_message,           # builds full Gemini setup payload
)
from rag.retriever import (
    retrieve_relevant_knowledge,   # Qdrant search + local JSON fallback
    build_rag_context,             # formats results into context string
)
from database.connection import AsyncSessionLocal, init_db
from database import crud
from ai_services.profile_extractor import extract_profile_from_conversation

# =============================================================================
# Audio Device Configuration
# Leave as None to use the system default (Bluetooth auto-detected by PulseAudio).
# Override via: --input-device N   --output-device N
# =============================================================================
AUDIO_INPUT_DEVICE_INDEX  = None   # USB Microphone
AUDIO_OUTPUT_DEVICE_INDEX = None   # Bluetooth Speaker (system default)

# -- Audio Parameters ---------------------------------------------------------
MIC_SAMPLE_RATE      = 16000   # Gemini Live expects 16 kHz
MIC_CHANNELS         = 1
MIC_CHUNK_FRAMES     = 512     # ~32 ms per chunk (low latency)
SPEAKER_SAMPLE_RATE  = 24000   # Gemini Live outputs 24 kHz PCM
SPEAKER_CHANNELS     = 1
AUDIO_FORMAT         = pyaudio.paInt16

# -- Voice Activity Detection (Barge-in / Interruption) ------------------------
INTERRUPT_RMS_THRESHOLD = 4500    # int16 RMS: user voice threshold to interrupt during playback
VAD_STREAK_TRIGGER      = 2       # consecutive frames (~64ms) above threshold to trigger interrupt

# -- Paths --------------------------------------------------------------------
PROJECT_DIR          = os.path.dirname(os.path.abspath(__file__))
AUDIO_EXPERIENCE_DIR = os.path.join(PROJECT_DIR, "audio_experiences")


# =============================================================================
# Audio device listing
# =============================================================================

def list_audio_devices():
    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    print(f"\nAvailable Audio Devices ({count} total):")
    print("-" * 68)
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        tags = []
        if info["maxInputChannels"]  > 0: tags.append("INPUT")
        if info["maxOutputChannels"] > 0: tags.append("OUTPUT")
        tag_str = " | ".join(tags) if tags else "--"
        print(f"  [{i:2d}] {info['name'][:44]:<44}  {tag_str}")
    print("-" * 68)
    try:
        di = pa.get_default_input_device_info()
        print(f"  Default Input  -> [{di['index']}] {di['name']}")
    except Exception:
        print("  Default Input  -> (none)")
    try:
        do = pa.get_default_output_device_info()
        print(f"  Default Output -> [{do['index']}] {do['name']}")
    except Exception:
        print("  Default Output -> (none)")
    print()
    pa.terminate()


# =============================================================================
# RMS (Voice Activity Detection helper)
# =============================================================================

def compute_rms(pcm_bytes: bytes) -> float:
    n = len(pcm_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm_bytes[:n * 2])
    return math.sqrt(sum(s * s for s in samples) / n)


# =============================================================================
# Thread-safe PCM audio player  (PyAudio callback mode)
# =============================================================================

class AudioPlayer:
    """
    Plays 24 kHz int16 PCM audio on the speaker.
    Fed from an asyncio task; callback runs on a PortAudio thread.
    Both sides share a threading.Lock for safety.
    """

    def __init__(self, pa: pyaudio.PyAudio, device_index=None):
        self._buf            = collections.deque()
        self._lock           = threading.Lock()
        self._last_play_time = 0.0
        self._stream         = pa.open(
            format=AUDIO_FORMAT,
            channels=SPEAKER_CHANNELS,
            rate=SPEAKER_SAMPLE_RATE,
            output=True,
            output_device_index=device_index,
            frames_per_buffer=1024,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        needed = frame_count * 2          # int16 = 2 bytes/sample
        out    = bytearray()
        with self._lock:
            while len(out) < needed and self._buf:
                chunk = self._buf[0]
                take  = needed - len(out)
                out  += chunk[:take]
                if take >= len(chunk):
                    self._buf.popleft()
                else:
                    self._buf[0] = chunk[take:]
            if out:
                self._last_play_time = time.time()
        out += b"\x00" * (needed - len(out))   # silence pad
        return (bytes(out), pyaudio.paContinue)

    def feed(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        with self._lock:
            self._buf.append(pcm_bytes)
            self._last_play_time = time.time()

    def is_playing(self) -> bool:
        """Returns True if audio is currently playing or recently finished draining."""
        with self._lock:
            if bool(self._buf):
                return True
            return (time.time() - self._last_play_time) < 0.4

    def is_active(self) -> bool:
        return self.is_playing()

    def clear(self):
        with self._lock:
            self._buf.clear()
            self._last_play_time = 0.0

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:
            pass


# =============================================================================
# Experience audio playback  (pygame MP3/WAV — same files as web agent)
# =============================================================================

_exp_playing = threading.Event()


def play_experience_audio(audio_url: str) -> bool:
    """
    Resolve audio_url to local file in audio_experiences/ and play via pygame.
    Mirrors the browser's <audio> element behaviour for Pi mode.
    """
    if not PYGAME_AVAILABLE:
        print("WARNING: pygame unavailable -- skipping experience audio")
        return False

    # URL may look like  /api/experiences/audio/valsad_asmukhbhai_experience.mp3
    filename = audio_url.replace("\\", "/").split("/")[-1]
    filepath = os.path.join(AUDIO_EXPERIENCE_DIR, filename)

    if not os.path.isfile(filepath):
        print(f"WARNING: Audio file not found: {filepath}")
        return False

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        _exp_playing.set()
        print(f"Playing experience audio: {filename}")

        def _watch():
            while pygame.mixer.music.get_busy():
                time.sleep(0.2)
            _exp_playing.clear()

        threading.Thread(target=_watch, daemon=True).start()
        return True
    except Exception as exc:
        print(f"pygame error: {exc}")
        _exp_playing.clear()
        return False


def stop_experience_audio():
    if not PYGAME_AVAILABLE:
        return
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            print("Experience audio stopped.")
    except Exception:
        pass
    _exp_playing.clear()


def is_experience_playing() -> bool:
    return _exp_playing.is_set()


# =============================================================================
# DB persistence helpers  (same logic as gemini_api.py background tasks)
# =============================================================================

async def _persist_message(session_id: str, role: str, content: str):
    """Best-effort DB write — voice agent keeps working even if this fails."""
    try:
        async with AsyncSessionLocal() as db:
            conv = await crud.get_or_create_conversation(db, conversation_id=session_id)
            await crud.add_message(db, conv.id, role, content)
    except Exception as exc:
        print(f"  [DB] {exc}")


async def _update_farmer_profile(session_id: str):
    """Same background profile extraction as used in the web agent."""
    try:
        async with AsyncSessionLocal() as db:
            conv = await crud.get_conversation(db, session_id)
            if not conv or not conv.messages:
                return
            recent = [
                {"role": m.role, "content": m.content}
                for m in conv.messages[-10:]
                if m.content.strip()
            ]
            extracted = await extract_profile_from_conversation(recent)
            if not extracted:
                return
            profile_id = conv.farmer_id
            if not profile_id:
                default_p  = await crud.get_or_create_default_profile(db)
                profile_id = default_p.id
            await crud.update_farmer_profile(db, profile_id, extracted)
            print(f"  [Profile] Updated: {list(extracted.keys())}")
    except Exception as exc:
        print(f"  [Profile] {exc}")


# =============================================================================
# Handle a single Gemini Live tool call  (identical logic to web agent)
# =============================================================================

async def _handle_tool_calls(gemini_ws, fn_calls: list) -> None:
    """
    Process all tool calls from one Gemini message and send back responses.
    Mirrors gemini_to_browser() in gemini_api.py — search + experience audio.
    """
    fn_responses = []

async def _handle_tool_calls(gemini_ws, fn_calls: list) -> str | None:
    """
    Execute tool calls and return any queued experience audio URL to play AFTER AI finishes speaking.
    """
    pending_audio = None
    fn_responses = []

    for call in fn_calls:
        fn_name  = call.get("name", "")
        fn_args  = call.get("args", {})
        call_id  = call.get("id", "")

        # ── Tool: search_agricultural_knowledge_base ──────────────────────────
        if fn_name == "search_agricultural_knowledge_base":
            query = fn_args.get("query", "")
            print(f"RAG search: {query}")
            try:
                matches = await retrieve_relevant_knowledge(query, limit=3, score_threshold=0.45)
                if matches:
                    results_text = "\n\n".join([
                        f"- [{m.get('title', '')}]: {m.get('text', '')}"
                        for m in matches
                    ])
                    # If a result has an audio experience, queue it to play AFTER AI introduction
                    for m in matches:
                        if m.get("audio_url"):
                            pending_audio = m["audio_url"]
                            break
                else:
                    results_text = "આ વિષય પર કોઈ ચોક્કસ માહિતી મળી નથી."
            except Exception as exc:
                results_text = f"Search error: {exc}"
                print(f"  RAG error: {exc}")

            fn_responses.append({
                "response": {
                    "name": fn_name,
                    "content": {"results": results_text},
                },
                "id": call_id,
            })

        # ── Tool: play_farmer_experience_audio ────────────────────────────────
        elif fn_name == "play_farmer_experience_audio":
            farmer_name      = fn_args.get("farmer_name")  or "હસમુખભાઈ પટેલ"
            district         = fn_args.get("district")     or "વલસાડ"
            audio_url        = fn_args.get("audio_url")    or "/api/experiences/audio/valsad_asmukhbhai_experience.mp3"
            experience_years = fn_args.get("experience_years") or "૫ વર્ષ"

            print(f"Queued experience audio: {farmer_name} ({district}) [{experience_years}]")
            pending_audio = audio_url

            fn_responses.append({
                "response": {
                    "name": fn_name,
                    "content": {
                        "status":  "queued",
                        "message": f"{farmer_name} ({district}) નો ઓડિયો અનુભવ AI બોલવાનું પૂરું કરે પછી શરૂ થશે.",
                    },
                },
                "id": call_id,
            })

    if fn_responses:
        await gemini_ws.send(json.dumps({
            "toolResponse": {"functionResponses": fn_responses}
        }))

    return pending_audio


# =============================================================================
# Main Gemini Live voice loop — Pi mode
# =============================================================================

async def run_pi_session(
    api_key: str,
    input_device=None,
    output_device=None,
    model: str  = DEFAULT_MODEL,
    voice: str  = DEFAULT_VOICE,
    session_id: str | None = None,
    interrupt_threshold: int = INTERRUPT_RMS_THRESHOLD,
):
    """
    Full always-listening voice loop for Raspberry Pi with default barge-in interruption.
    Uses the same system prompt, tools, RAG, and DB as the web agent.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # 1. Initialize SQLite database schema if not already initialized
    try:
        await init_db()
    except Exception as exc:
        print(f"  [DB init] {exc}")

    # 2. Build context from DB if a previous session exists
    full_instruction = BASE_SYSTEM_INSTRUCTION
    try:
        async with AsyncSessionLocal() as db:
            conv       = await crud.get_or_create_conversation(db, conversation_id=session_id)
            session_id = conv.id
            ctx_str    = await crud.build_conversation_context(db, conversation_id=session_id)
            if ctx_str.strip():
                full_instruction += f"\n\n{ctx_str}"
    except Exception as exc:
        print(f"  [DB context] {exc}")

    # Init pygame
    if PYGAME_AVAILABLE:
        try:
            pygame.mixer.init()
        except Exception as exc:
            print(f"WARNING: pygame init failed: {exc}")

    pa     = pyaudio.PyAudio()
    player = AudioPlayer(pa, device_index=output_device)

    mic_q    = queue.Queue(maxsize=300)
    shutdown = threading.Event()

    def mic_callback(in_data, frame_count, time_info, status):
        if not shutdown.is_set():
            try:
                mic_q.put_nowait(in_data)
            except queue.Full:
                pass
        return (None, pyaudio.paContinue)

    mic_stream = pa.open(
        format=AUDIO_FORMAT,
        channels=MIC_CHANNELS,
        rate=MIC_SAMPLE_RATE,
        input=True,
        input_device_index=input_device,
        frames_per_buffer=MIC_CHUNK_FRAMES,
        stream_callback=mic_callback,
    )
    mic_stream.start_stream()
    print("Microphone open -- always listening...\n")

    # Reuse the exact same Gemini URL + setup message as the web agent
    url       = get_gemini_ws_url(api_key)
    setup_msg = build_setup_message(model, full_instruction, voice)

    loop = asyncio.get_event_loop()

    try:
        async with websockets.connect(
            url,
            open_timeout=30,
            close_timeout=15,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as gemini_ws:
            print("Gemini Live connected!")

            # Handshake
            await gemini_ws.send(json.dumps(setup_msg))
            raw = await asyncio.wait_for(gemini_ws.recv(), timeout=15.0)
            if "setupComplete" in json.loads(raw):
                print("Setup complete. Speak now!\n" + "-" * 50)
            else:
                print(f"Unexpected setup response: {raw[:200]}")

            speech_streak  = 0
            ai_is_speaking = False

            # -----------------------------------------------------------------
            # Task A: mic -> Gemini  (Clean Half-Duplex Listening)
            # -----------------------------------------------------------------
            async def mic_to_gemini():
                nonlocal speech_streak
                while True:
                    pcm = await loop.run_in_executor(None, mic_q.get)
                    if pcm is None:
                        break

                    # 1. While Farmer Experience MP3 is playing: allow user to interrupt it
                    if is_experience_playing():
                        rms = compute_rms(pcm)
                        if rms >= interrupt_threshold:
                            speech_streak += 1
                            if speech_streak >= VAD_STREAK_TRIGGER:
                                print(f"\n⚡ User voice detected (RMS: {int(rms)}) -> stopping experience audio")
                                stop_experience_audio()
                                speech_streak = 0
                        else:
                            speech_streak = max(0, speech_streak - 1)
                        # Do not stream MP3 sound into Gemini Live
                        continue

                    # 2. While AI is actively speaking / outputting audio:
                    # Do NOT stream mic to Gemini so Gemini never receives its own acoustic echo
                    if ai_is_speaking or player.is_playing():
                        speech_streak = 0
                        continue

                    # 3. Idle / Listening to Farmer: Stream mic audio to Gemini Live
                    speech_streak = 0
                    await gemini_ws.send(json.dumps({
                        "realtimeInput": {
                            "audio": {
                                "mimeType": "audio/pcm;rate=16000",
                                "data": base64.b64encode(pcm).decode(),
                            }
                        }
                    }))

            # -----------------------------------------------------------------
            # Task B: Gemini -> speaker + tool handling
            # -----------------------------------------------------------------
            async def gemini_to_speaker():
                nonlocal ai_is_speaking
                ai_text_parts = []
                queued_experience_audio = None

                async for raw_msg in gemini_ws:
                    try:
                        msg = json.loads(raw_msg)
                    except Exception:
                        continue

                    sc = msg.get("serverContent", {})

                    # -- Interrupted ------------------------------------------
                    if sc.get("interrupted"):
                        print("\nAI interrupted by user.")
                        player.clear()
                        stop_experience_audio()
                        ai_is_speaking = False
                        queued_experience_audio = None
                        partial = "".join(ai_text_parts).strip()
                        ai_text_parts.clear()
                        if partial:
                            asyncio.create_task(
                                _persist_message(session_id, "assistant",
                                                 f"{partial} ... [અટકાવેલ]")
                            )
                        continue

                    # -- Tool calls -------------------------------------------
                    tool_call = msg.get("toolCall") or sc.get("toolCall")
                    if tool_call:
                        fn_calls = tool_call.get("functionCalls", [])
                        if fn_calls:
                            new_audio = await _handle_tool_calls(gemini_ws, fn_calls)
                            if new_audio:
                                queued_experience_audio = new_audio

                    # -- PCM audio -> speaker ---------------------------------
                    model_turn = sc.get("modelTurn", {})
                    for part in model_turn.get("parts", []):
                        inline = part.get("inlineData", {})
                        if inline.get("data"):
                            ai_is_speaking = True
                            player.feed(base64.b64decode(inline["data"]))
                        if part.get("text"):
                            ai_text_parts.append(part["text"])
                            print(f"AI: {part['text']}", end="", flush=True)

                    # -- Output transcription ---------------------------------
                    out_tr = sc.get("outputTranscription", {})
                    if out_tr.get("text"):
                        ai_text_parts.append(out_tr["text"])
                        print(f"AI: {out_tr['text']}", end="", flush=True)

                    # -- Turn complete ----------------------------------------
                    if sc.get("turnComplete"):
                        full_text = "".join(ai_text_parts).strip()
                        ai_text_parts.clear()
                        print("\n" + "-" * 50)

                        # Wait until all buffered speaker audio finishes playing out loud
                        while player.is_playing():
                            await asyncio.sleep(0.08)

                        ai_is_speaking = False

                        # If an experience audio was queued by tools, play it NOW after AI speech finishes!
                        if queued_experience_audio:
                            audio_to_play = queued_experience_audio
                            queued_experience_audio = None
                            print(f"\n▶️ [AI બોલી લીધું છે. હવે ખેડૂત અનુભવ ઓડિયો શરૂ થઈ રહ્યો છે...]")
                            play_experience_audio(audio_to_play)
                        else:
                            print("🎤 [તમારો પ્રશ્ન પૂછો / Please speak now...]")

                        if full_text:
                            asyncio.create_task(
                                _persist_message(session_id, "assistant", full_text)
                            )
                            asyncio.create_task(
                                _update_farmer_profile(session_id)
                            )

            await asyncio.gather(mic_to_gemini(), gemini_to_speaker())

    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as exc:
        err_str = str(exc)
        print(f"\nSession error: {err_str}")
        if "1008" in err_str or "authentication" in err_str.lower() or "1007" in err_str:
            print("\n" + "=" * 60)
            print("🔑 GEMINI API KEY AUTHENTICATION ERROR")
            print("=" * 60)
            print("Google rejected the Gemini Live connection due to invalid authentication.")
            print("\nCommon Causes & Solutions:")
            print("1. Check your .env file:")
            print("   Make sure GEMINI_API_KEY is set without quotes or extra spaces:")
            print("   GEMINI_API_KEY=AIzaSy...")
            print("2. Google AI Studio Key vs Google Cloud Key:")
            print("   Gemini Live requires a key from Google AI Studio:")
            print("   👉 https://aistudio.google.com/app/apikey")
            print("3. If using Google Cloud Console:")
            print("   Ensure 'Generative Language API' is enabled and API key has NO HTTP/IP restrictions.")
            print("=" * 60 + "\n")
    finally:
        shutdown.set()
        mic_q.put(None)
        mic_stream.stop_stream()
        mic_stream.close()
        player.close()
        pa.terminate()
        stop_experience_audio()
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass
        print("Cleanup complete.")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Khedut Voice AI -- Raspberry Pi standalone always-listening agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python pi_voice_agent.py                      # Start with default devices
  python pi_voice_agent.py --list-devices       # See all PyAudio devices
  python pi_voice_agent.py --input-device 2     # Use mic at device index 2
  python pi_voice_agent.py --output-device 6    # Use speaker at device index 6

Web / browser mode (runs independently alongside Pi mode):
  uvicorn main:app --host 0.0.0.0 --port 8000
""",
    )
    parser.add_argument("--list-devices",  action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--input-device",  type=int, default=None, metavar="N",
                        help="PyAudio input device index (USB mic)")
    parser.add_argument("--output-device", type=int, default=None, metavar="N",
                        help="PyAudio output device index (speaker/Bluetooth)")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--voice",  default=DEFAULT_VOICE)
    parser.add_argument("--session", default=None,
                        help="Resume a previous session ID (optional)")
    parser.add_argument("--interrupt-threshold", type=int, default=INTERRUPT_RMS_THRESHOLD,
                        help=f"RMS energy threshold to interrupt AI playback (default: {INTERRUPT_RMS_THRESHOLD})")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    in_dev  = args.input_device  if args.input_device  is not None else AUDIO_INPUT_DEVICE_INDEX
    out_dev = args.output_device if args.output_device is not None else AUDIO_OUTPUT_DEVICE_INDEX

    print("=" * 50)
    print("   Khedut Voice AI -- Raspberry Pi Mode")
    print("=" * 50)
    print(f"  Model               : {args.model}")
    print(f"  Voice               : {args.voice}")
    print(f"  Mic                 : {'System default' if in_dev  is None else f'Device {in_dev}'}")
    print(f"  Speaker             : {'System default (Bluetooth if set)' if out_dev is None else f'Device {out_dev}'}")
    print(f"  Interrupt Threshold : {args.interrupt_threshold}")
    print(f"  Voice Interruption  : Enabled by default ⚡")
    if args.session:
        print(f"  Session             : {args.session} (resuming)")
    print()

    try:
        asyncio.run(run_pi_session(
            api_key=api_key,
            input_device=in_dev,
            output_device=out_dev,
            model=args.model,
            voice=args.voice,
            session_id=args.session,
            interrupt_threshold=args.interrupt_threshold,
        ))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
