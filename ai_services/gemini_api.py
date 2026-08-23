"""
Gemini Live Realtime Voice AI Service
- Short, to-the-point Gujarati answers
- Farmer profile auto-extracted from conversation and persisted in SQLite
"""

import asyncio
import base64
import json
import os
from typing import Optional

import websockets
from dotenv import load_dotenv

from database.connection import AsyncSessionLocal
from database import crud
from ai_services.profile_extractor import extract_profile_from_conversation

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_WS_BASE_URL = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_VOICE = "Sadaltager"

# ─── System Prompt ────────────────────────────────────────────────────────────
# Keep it short: answer only what was asked, in simple Gujarati.
BASE_SYSTEM_INSTRUCTION = """\
તમે એક અનુભવી ઓર્ગેનિક ખેડૂત છો જેઓ ઓર્ગેનિક ખેતી વિશે ઊંડી સમજ ધરાવો છો.\n
તમારી ભૂમિકા:\n"
- ખેડૂતોને ઓર્ગેનિક ખેતીની સલાહ આપવી\n
- કુદરતી ખાતર, જૈવ જંતુનાશક, અને ટકાઉ ખેતી પ્રણાલી વિશે માર્ગદર્શન આપવું\n
- ગુજરાતની સ્થાનિક ફસલો — કપાસ, મગફળી, ઘઉં, બાજરી, શાકભાજી — વિશે જ્ઞાન આપવું\n
- જમીન, પાણી, ઋતુ અનુસાર ખેતી અંગે સૂચનો આપવા\n\n

નિયમ:
- ફક્ત ખેડૂતે જે પૂછ્યું છે, ફક્ત તેટલો જ જવાબ આપો.
- ટૂંકો, સ્પષ્ટ, સ્થાનિક ગુજરાતીમાં જવાબ આપો.
- ખેડૂત "વધારે જણાવો" અથવા "detail" માગે ત્યારે જ વિગત આપો.
- ઓર્ગેનિક / પ્રાકૃતિક ઉપાય જ સૂચવો, રાસાયણ નહીં.
- ગરમ, સ્નેહાળ ભાષા વાપરો.\
"""


def get_gemini_ws_url(api_key: str | None = None) -> str:
    """Constructs the Gemini Live WebSocket URL."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return f"{GEMINI_WS_BASE_URL}?key={key}"


def build_setup_message(
    model: str = DEFAULT_MODEL,
    system_instruction: str = BASE_SYSTEM_INSTRUCTION,
    voice_name: str = DEFAULT_VOICE,
) -> dict:
    """Constructs the Gemini Live setup message."""
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
    conversation_id: Optional[str] = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    voice_name: str = DEFAULT_VOICE,
):
    """
    Bridges browser WebSocket ↔ Gemini Live API.
    1. Loads known farmer profile + recent history from DB and injects into prompt.
    2. Streams mic audio to Gemini; streams PCM audio + transcripts back to browser.
    3. After each AI turn: persists messages in DB + auto-extracts farmer profile.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        await browser_ws.send_text(json.dumps({"error": "GEMINI_API_KEY not set"}))
        await browser_ws.close()
        return

    # ── Load context from DB ──────────────────────────────────────────────────
    async with AsyncSessionLocal() as db:
        conv = await crud.get_or_create_conversation(db, conversation_id=conversation_id)
        active_conversation_id = conv.id
        context_str = await crud.build_conversation_context(db, conversation_id=active_conversation_id)

    # Inject known context into prompt only if something is known
    full_instruction = BASE_SYSTEM_INSTRUCTION
    if context_str.strip():
        full_instruction += f"\n\n{context_str}"

    # Tell browser the session ID
    await browser_ws.send_text(json.dumps({
        "type": "session_info",
        "conversation_id": active_conversation_id,
        "title": conv.title,
    }))

    gemini_url = get_gemini_ws_url(key)
    setup_message = build_setup_message(model, full_instruction, voice_name)

    try:
        async with websockets.connect(
            gemini_url,
            additional_headers={"Content-Type": "application/json"},
        ) as gemini_ws:
            print(f"🤖 Gemini Live connected (session: {active_conversation_id})")

            # Setup handshake
            await gemini_ws.send(json.dumps(setup_message))
            msg = await gemini_ws.recv()
            if "setupComplete" in json.loads(msg):
                print("✅ Gemini setup complete")

            # ── Browser → Gemini (microphone PCM) ────────────────────────────
            async def browser_to_gemini():
                try:
                    while True:
                        ws_msg = await browser_ws.receive()
                        if "bytes" in ws_msg and ws_msg["bytes"]:
                            raw_pcm = ws_msg["bytes"]
                            await gemini_ws.send(json.dumps({
                                "realtimeInput": {
                                    "audio": {
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": base64.b64encode(raw_pcm).decode("utf-8"),
                                    }
                                }
                            }))
                        elif "text" in ws_msg and ws_msg["text"]:
                            try:
                                parsed = json.loads(ws_msg["text"])
                                if parsed.get("type") == "text_prompt" and parsed.get("text"):
                                    user_text = parsed["text"].strip()
                                    async with AsyncSessionLocal() as db:
                                        await crud.add_message(db, active_conversation_id, "user", user_text)
                                    await gemini_ws.send(json.dumps({
                                        "clientContent": {
                                            "turns": [{"role": "user", "parts": [{"text": user_text}]}],
                                            "turnComplete": True
                                        }
                                    }))
                            except Exception:
                                pass
                except Exception:
                    pass

            # ── Gemini → Browser (audio + transcript) + DB persistence ────────
            async def gemini_to_browser():
                ai_text_parts: list[str] = []
                try:
                    async for raw_msg in gemini_ws:
                        resp = json.loads(raw_msg)
                        sc = resp.get("serverContent", {})
                        # Interruption detection (Farmer spoke while AI was answering)
                        if sc.get("interrupted"):
                            print(f"⚡ [Session: {active_conversation_id}] AI interrupted by farmer.")
                            # Notify browser immediately to cut off any queued/playing audio
                            await browser_ws.send_text(json.dumps({"type": "interrupted"}))

                            # Persist partial turn if text was generated
                            partial_text = "".join(ai_text_parts).strip()
                            ai_text_parts.clear()
                            if partial_text:
                                async with AsyncSessionLocal() as db:
                                    await crud.add_message(
                                        db, active_conversation_id, "assistant",
                                        f"{partial_text} ... [અટકાવેલ]", audio_format="pcm_24000"
                                    )

                        # Audio / text parts
                        model_turn = sc.get("modelTurn", {})
                        for part in model_turn.get("parts", []):
                            inline = part.get("inlineData", {})
                            if inline.get("data"):
                                await browser_ws.send_bytes(base64.b64decode(inline["data"]))
                            if part.get("text"):
                                ai_text_parts.append(part["text"])
                                await browser_ws.send_text(json.dumps({"type": "text", "text": part["text"]}))

                        # Output transcription
                        out_tr = sc.get("outputTranscription", {})
                        if out_tr.get("text"):
                            ai_text_parts.append(out_tr["text"])
                            await browser_ws.send_text(json.dumps({"type": "text", "text": out_tr["text"]}))

                        # Turn complete
                        if sc.get("turnComplete"):
                            full_ai_text = "".join(ai_text_parts).strip()
                            ai_text_parts.clear()

                            if full_ai_text:
                                # Persist AI turn in DB
                                async with AsyncSessionLocal() as db:
                                    await crud.add_message(
                                        db, active_conversation_id, "assistant",
                                        full_ai_text, audio_format="pcm_24000"
                                    )

                                # Background: extract farmer profile from recent dialogue
                                asyncio.create_task(
                                    _update_profile_from_conversation(active_conversation_id)
                                )

                            await browser_ws.send_text(json.dumps({"type": "turn_complete"}))

                except Exception:
                    pass

            await asyncio.gather(browser_to_gemini(), gemini_to_browser())

    except Exception as e:
        print(f"Gemini live session error: {e}")
        try:
            await browser_ws.close()
        except Exception:
            pass


async def _update_profile_from_conversation(conversation_id: str):
    """
    Background task: extract farmer profile fields from recent conversation turns
    and persist any newly discovered fields into the FarmerProfile table.
    Runs silently after each AI turn — does not block streaming.
    """
    try:
        async with AsyncSessionLocal() as db:
            conv = await crud.get_conversation(db, conversation_id)
            if not conv or not conv.messages:
                return

            # Use last 10 messages for extraction context
            recent_msgs = [
                {"role": m.role, "content": m.content}
                for m in conv.messages[-10:]
                if m.content.strip()
            ]

            extracted = await extract_profile_from_conversation(recent_msgs)
            if not extracted:
                return

            # Only update fields that are genuinely new (don't overwrite existing data)
            profile = await crud.get_or_create_default_profile(db)
            update_data = {}
            for field, value in extracted.items():
                current = getattr(profile, field, None)
                if field == "crops":
                    # Merge crop lists
                    if value:
                        update_data[field] = value
                elif not current and value:
                    # Only set if field is currently empty
                    update_data[field] = value

            if update_data:
                await crud.update_farmer_profile(db, profile.id, update_data)
                print(f"🌱 Profile updated from conversation: {list(update_data.keys())}")

    except Exception as e:
        print(f"[Profile update] Non-critical error: {e}")
