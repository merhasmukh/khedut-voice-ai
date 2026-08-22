"""
Gemini Live Realtime Voice AI Service
Handles bidirectional real-time audio and text communication with the Gemini Live API.
"""

import asyncio
import base64
import json
import os
import websockets
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_WS_BASE_URL = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "તમે એક અનુભવી ઓર્ગેનિક ખેડૂત છો જેઓ ઓર્ગેનિક ખેતી વિશે ઊંડી સમજ ધરાવો છો.\n"
    "તમારી ભૂમિકા:\n"
    "- ખેડૂતોને ઓર્ગેનિક ખેતીની સલાહ આપવી\n"
    "- કુદરતી ખાતર, જૈવ જંતુનાશક, અને ટકાઉ ખેતી પ્રણાલી વિશે માર્ગદર્શન આપવું\n"
    "- ગુજરાતની સ્થાનિક ફસલો — કપાસ, મગફળી, ઘઉં, બાજરી, શાકભાજી — વિશે જ્ઞાન આપવું\n"
    "- જમીન, પાણી, ઋતુ અનુસાર ખેતી અંગે સૂચનો આપવા\n\n"
    "નિયમો:\n"
    "- હંમેશા માત્ર ગુજરાતી ભાષામાં જ જવાબ આપો\n"
    "- સરળ, સ્થાનિક ભાષા વાપરો જે સામાન્ય ખેડૂત સમજી શકે\n"
    "- ઓર્ગેનિક અને કુદરતી ઉપાયો જ સૂચવો, રાસાયણિક ઉત્પાદનો નહીં\n"
    "- ગરમ, મૈત્રીપૂર્ણ અને વ્યવહારુ અભિગમ રાખો"
)


DEFAULT_VOICE = "Sadaltager"


def get_gemini_ws_url(api_key: str | None = None) -> str:
    """Constructs the WebSocket URL for Gemini Live API."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")
    return f"{GEMINI_WS_BASE_URL}?key={key}"


def build_setup_message(
    model: str = DEFAULT_MODEL,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    voice_name: str = DEFAULT_VOICE,
) -> dict:
    """Constructs the initial setup configuration for the Gemini Live WebSocket session."""
    return {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice_name}
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
        }
    }


async def handle_gemini_live_session(
    browser_ws,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    voice_name: str = DEFAULT_VOICE,
):
    """
    Bridges a FastAPI WebSocket connection from the browser to the Gemini Live WebSocket API.
    Handles:
      1. Gemini setup handshake
      2. Mic audio streaming (Browser -> Gemini)
      3. PCM audio & transcript streaming (Gemini -> Browser)
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        await browser_ws.send_text(json.dumps({"error": "GEMINI_API_KEY not set"}))
        await browser_ws.close()
        return

    gemini_url = get_gemini_ws_url(key)
    setup_message = build_setup_message(model, system_instruction, voice_name)

    try:
        async with websockets.connect(
            gemini_url,
            additional_headers={"Content-Type": "application/json"},
        ) as gemini_ws:
            print("🤖 Connected to Gemini Live API")

            # 1. Send setup message
            await gemini_ws.send(json.dumps(setup_message))
            print("📤 Setup sent to Gemini")

            # Wait for setup confirmation
            msg = await gemini_ws.recv()
            data = json.loads(msg)
            if "setupComplete" in data:
                print("✅ Gemini setup complete")

            # 2. Forward microphone audio from browser to Gemini
            async def browser_to_gemini():
                try:
                    while True:
                        raw_pcm = await browser_ws.receive_bytes()
                        payload = {
                            "realtimeInput": {
                                "audio": {
                                    "mimeType": "audio/pcm;rate=16000",
                                    "data": base64.b64encode(raw_pcm).decode("utf-8"),
                                }
                            }
                        }
                        await gemini_ws.send(json.dumps(payload))
                except Exception:
                    pass

            # 3. Forward Gemini response (PCM audio chunks and text) to browser
            async def gemini_to_browser():
                try:
                    async for raw_msg in gemini_ws:
                        resp = json.loads(raw_msg)
                        sc = resp.get("serverContent", {})
                        if not sc:
                            continue

                        # Model turn parts
                        model_turn = sc.get("modelTurn", {})
                        for part in model_turn.get("parts", []):
                            inline = part.get("inlineData", {})
                            if inline.get("data"):
                                audio_bytes = base64.b64decode(inline["data"])
                                await browser_ws.send_bytes(audio_bytes)
                            if part.get("text"):
                                await browser_ws.send_text(
                                    json.dumps({"text": part["text"]})
                                )

                        # Output transcription if available
                        out_transcription = sc.get("outputTranscription", {})
                        if out_transcription.get("text"):
                            await browser_ws.send_text(
                                json.dumps({"text": out_transcription["text"]})
                            )
                except Exception:
                    pass

            await asyncio.gather(browser_to_gemini(), gemini_to_browser())

    except Exception as e:
        print(f"Gemini live session error: {e}")
        try:
            await browser_ws.close()
        except Exception:
            pass
