"""
Gemini Live Realtime Voice AI Service
- Short, to-the-point Gujarati answers
- Farmer profile auto-extracted from conversation and persisted in SQLite
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Optional, Union

import websockets
from dotenv import load_dotenv

from database.connection import AsyncSessionLocal
from database import crud
from ai_services.profile_extractor import extract_profile_from_conversation
from rag.retriever import retrieve_relevant_knowledge, build_rag_context
from ai_services.whatsapp_client import send_whatsapp_message_async

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_WS_BASE_URL = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_VOICE = "Sadaltager"

# ─── System Prompt ────────────────────────────────────────────────────────────
BASE_SYSTEM_INSTRUCTION = """\
તમે ગુજરાતના ખેડૂતો માટે એક અનુભવી, નમ્ર અને સહાયક પ્રાકૃતિક ખેતી સલાહકાર (AI કિસાન મિત્ર) છો.

મહત્વપૂર્ણ નિયમો:
1. શરૂઆતનું અભિવાદન: જ્યારે પણ નવી વાતચીત શરૂ થાય, ત્યારે હંમેશા આ રીતે પૂછો: "રામ રામ, કેમ છો? મજામાં? પ્રાકૃતિક ખેતી વિશે તમારે શું જાણવું છે?"
2. કોઈ ખોટું નામ ન વાપરવું: જ્યાં સુધી ખેડૂત પોતે પોતાનું નામ ન જણાવે, ત્યાં સુધી કોઈ પણ નામ (જેમ કે મિનીબેન, મીની, રમેશભાઈ વગેરે) પોતાની મેળે ક્યારેય ધારવું કે બોલવું નહીં.
3. ખેડૂત જાત-અનુભવ (Experience Audio): જ્યારે પણ ખેડૂત જણાવે કે તેઓ કોઈ ચોક્કસ જિલ્લાથી છે (જેમ કે 'હું વલસાડથી છું' અથવા 'વલસાડમાં કોઈ ખેડૂતનો અનુભવ છે?'):
   - `search_agricultural_knowledge_base` દ્વારા તે જિલ્લાના અનુભવી ખેડૂતની વિગત શોધો.
   - ખેડૂતને સ્નેહપૂર્વક ૧-૨ વાક્યમાં કહો: "અરે વાહ! તમે વલસાડથી છો? વલસાડ જિલ્લામાં હસમુખભાઈ પટેલ છેલ્લા ૫ વર્ષથી પ્રાકૃતિક ખેતી કરે છે. ચાલો હું તમને એમનો પોતાનો અનુભવ સંભળાવું."
   - સાથે `play_farmer_experience_audio` ટૂલ બોલાવો. ટૂલ બોલાવ્યા પછી વધારાનું બીજું કંઈ ન બોલવું જેથી ખેડૂત શાંતિથી ઓડિયો સાંભળી શકે.
4. વેક્ટર સ્ટોરમાંથી માહિતી મેળવવી: જ્યારે પણ ખેડૂત કોઈ ચોક્કસ આંકડા (જેમ કે કોઈ જિલ્લામાં પ્રાકૃતિક ખેતી કરતા ખેડૂતોની સંખ્યા), ખાતર/દવાના ડોઝ, બનાવવાની રીત કે પાક વિષયક માહિતી પૂછે, ત્યારે `search_agricultural_knowledge_base` ટૂલનો ઉપયોગ કરી વેક્ટર સ્ટોરમાંથી સાચી માહિતી શોધીને જ ચોક્કસ જવાબ આપો.
5. ટૂંકો અને સરળ જવાબ: ખેડૂત જે પ્રશ્ન પૂછે, તેનો જ સીધો, સાચો અને સરળ ૧ થી ૩ વાક્યોમાં દેશી ગુજરાતીમાં જવાબ આપો.
6. વિગત ત્યારે જ આપવી: ખેડૂત જ્યારે "વિગતે સમજાવો" અથવા "detail માં કહો" એમ કહે ત્યારે જ વિસ્તારથી જવાબ આપો.
7. પ્રાકૃતિક/ઓર્ગેનિક ઉપાય: ફક્ત દેશી ખાતર (જીવામૃત, ઘનજીવામૃત) અને જૈવિક કીટ નિયંત્રણ (નીમાસ્ત્ર, બ્રહ્માસ્ત્ર, અગ્નિઅસ્ત્ર, દશપર્ણી અર્ક, ખાટી છાશ) સૂચવો. કોઈ રાસાયણિક દવા કે ખાતર ન જણાવવા.
8. સ્નેહાળ ગ્રામીણ ભાષા: ગુજરાત-સૌરાષ્ટ્રની આત્મીય, મીઠી અને સન્માનજનક ગ્રામીણ ગુજરાતી ભાષા વાપરો.
9. WhatsApp પર જવાબ મોકલવો: જ્યારે ખેડૂત "WhatsApp પર મોકલો", "WhatsApp ma moklo", "WhatsApp par mokli do" અથવા similar request કરે:
   - પ્રથમ ખેડૂતનો WhatsApp નંબર પૂછો: "ભાઈ, તમારો WhatsApp નંબર આપો."
   - ખેડૂત નંબર આપ્યા પછી `send_whatsapp_answer` ટૂલ બોલાવો.
   - answer_text = વાતચીતમાં છેલ્લો આપેલ જ્ઞાન-ભરેલો AI જવાબ (૧-૩ વાક્ય, ગુજરાતી).
   - ટૂલ સફળ થાય ત્યારે AI બોલે: "WhatsApp પર મોકલ્યો! ચેક કરી લેજો."
10. પુસ્તકો, સાહિત્ય અને YouTube વિડીયો: જ્યારે પણ ખેડૂત પુસ્તક, સાહિત્ય, PDF કે YouTube વિડીયો વિશે પૂછે, ત્યારે `search_agricultural_knowledge_base` દ્વારા માહિતી શોધીને શ્રી આચાર્ય દેવવ્રતજી (માનનીય રાજ્યપાલશ્રી) લિખિત 'પ્રાકૃતિક કૃષિ' પુસ્તકની કામધેનુ યુનિવર્સિટી PDF લિંક અથવા સંબંધિત YouTube વિડીયો લિંક સચોટ રીતે આપો.
"""

KNOWLEDGE_BASE_TOOL = {
    "functionDeclarations": [
        {
            "name": "search_agricultural_knowledge_base",
            "description": "ગુજરાતના તમામ ૩૩ જિલ્લાઓમાં પ્રાકૃતિક ખેતી કરતા ખેડૂતોની સત્તાવાર સંખ્યા, જીવામૃત, બીજામૃત, ઘનજીવામૃત, નીમાસ્ત્ર, બ્રહ્માસ્ત્ર, અગ્નિઅસ્ત્ર, દશપર્ણી અર્ક, પાક સંરક્ષણ, ખેડૂતોના જાત-અનુભવ (જેમ કે વલસાડના હસમુખભાઈ પટેલ), તેમજ પ્રાકૃતિક ખેતીના પુસ્તકો (માનનીય રાજ્યપાલ શ્રી આચાર્ય દેવવ્રતજી લિખિત પુસ્તક PDF) અને YouTube વિડીયો લિંક્સ Qdrant વેક્ટર સ્ટોરમાંથી શોધવા માટેનું ટૂલ.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "શોધવા માટેનો પ્રશ્ન કે કીવર્ડ્સ (દા.ત. 'પ્રાકૃતિક ખેતી પુસ્તક PDF', 'આચાર્ય દેવવ્રત વિડીયો', 'વલસાડ ઓર્ગેનિક ખેડૂત અનુભવ', 'બોટાદ પ્રાકૃતિક ખેડૂતોની સંખ્યા')"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "play_farmer_experience_audio",
            "description": "જ્યારે પણ વાતચીતમાં ખેડૂત કોઈ ચોક્કસ જિલ્લા (જેમ કે વલસાડ) વિશે કહે અથવા કોઈ ખેડૂતનો જાત-અનુભવ સાંભળવા માંગે, ત્યારે આ ટૂલ બોલાવીને ખેડૂતનો રેકોર્ડ કરેલો ઓરિજિનલ ઓડિયો અનુભવ શરૂ કરો.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "farmer_name": {
                        "type": "STRING",
                        "description": "અનુભવી ખેડૂતનું નામ (દા.ત. 'હસમુખભાઈ પટેલ')"
                    },
                    "district": {
                        "type": "STRING",
                        "description": "જિલ્લો (દા.ત. 'વલસાડ')"
                    },
                    "audio_url": {
                        "type": "STRING",
                        "description": "ઓડિયો ફાઇલની URL (દા.ત. '/api/experiences/audio/valsad.mp3')"
                    },
                    "experience_years": {
                        "type": "STRING",
                        "description": "ખેતી અનુભવના વર્ષો (દા.ત. '૫ વર્ષ')"
                    }
                },
                "required": ["district", "audio_url"]
            }
        },
        {
            "name": "send_whatsapp_answer",
            "description": "જ્યારે ખેડૂત WhatsApp પર AI નો જવાબ મેળવવા ઈચ્છે ત્યારે આ ટૂલ બોલાવો. ખેડૂત નંબર આપ્યા પછી છેલ્લો AI જ્ઞાન-ભરેલો જવાબ WhatsApp ટેમ્પ્લેટ દ્વારા તે નંબર પર મોકલો.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "phone_number": {
                        "type": "STRING",
                        "description": "ખેડૂતનો WhatsApp નંબર (દા.ત. '919724455986' અથવા '97244 55986' — 10 કે 12 અંક)"
                    },
                    "answer_text": {
                        "type": "STRING",
                        "description": "WhatsApp પર મોકલવાનો AI નો જ્ઞાન-ભરેલો જવાબ (ગુજરાતી, ૧-૩ વાક્ય)"
                    }
                },
                "required": ["phone_number", "answer_text"]
            }
        }
    ]
}


def get_gemini_ws_url(api_key: str | None = None) -> str:
    """Constructs the Gemini Live WebSocket URL."""
    key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return f"{GEMINI_WS_BASE_URL}?key={key}"


def _normalize_phone(raw: str) -> str:
    """
    Normalise a user-spoken/typed phone number to WhatsApp-ready format.
    - Strips all non-digit characters (spaces, +, -, (, ))
    - If 10 digits remain, prepends '91' (India country code)
    - Returns empty string if result is not 10-12 digits (invalid)
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        digits = "91" + digits
    if 10 <= len(digits) <= 12:
        return digits
    return ""


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
            "tools": [KNOWLEDGE_BASE_TOOL],
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

    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            async with websockets.connect(
                gemini_url,
                open_timeout=30,
                close_timeout=15,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,
            ) as gemini_ws:
                print(f"🤖 Gemini Live connected (session: {active_conversation_id})")

                # Setup handshake
                await gemini_ws.send(json.dumps(setup_message))
                msg = await asyncio.wait_for(gemini_ws.recv(), timeout=15.0)
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

                                        # Query Qdrant vector database for text prompts
                                        rag_ctx = await build_rag_context(user_text, max_chunks=2)
                                        prompt_to_send = f"{user_text}\n\n[પ્રમાણિત કૃષિ રેકોર્ડ:\n{rag_ctx}]" if rag_ctx else user_text

                                        await gemini_ws.send(json.dumps({
                                            "clientContent": {
                                                "turns": [{"role": "user", "parts": [{"text": prompt_to_send}]}],
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

                            # Handle Function / Tool Call from Gemini Live (Qdrant Vector Search / Experience Audio)
                            tool_call = resp.get("toolCall") or sc.get("toolCall")
                            if tool_call:
                                function_calls = tool_call.get("functionCalls", [])
                                function_responses = []
                                for fc in function_calls:
                                    call_id = fc.get("id")
                                    func_name = fc.get("name")
                                    args = fc.get("args", {})

                                    if func_name == "play_farmer_experience_audio":
                                        farmer_name = args.get("farmer_name") or "હસમુખભાઈ પટેલ"
                                        district = args.get("district") or "વલસાડ"
                                        audio_url = args.get("audio_url") or "/api/experiences/audio/valsad.mp3"
                                        experience_years = args.get("experience_years") or "૫ વર્ષ"

                                        print(f"🎙️ [Experience Audio Triggered] {farmer_name} ({district}) -> {audio_url}")
                                        await browser_ws.send_text(json.dumps({
                                            "type": "play_experience_audio",
                                            "farmer_name": farmer_name,
                                            "district": district,
                                            "audio_url": audio_url,
                                            "experience_years": experience_years,
                                        }))

                                        function_responses.append({
                                            "response": {
                                                "name": func_name,
                                                "content": {
                                                    "status": "playing",
                                                    "message": f"{farmer_name} (જિલ્લો: {district}) નો ઓડિયો અનુભવ પ્લે થઈ રહ્યો છે."
                                                }
                                            },
                                            "id": call_id
                                        })
                                    elif func_name == "send_whatsapp_answer":
                                        phone_raw   = args.get("phone_number", "")
                                        answer_text = args.get("answer_text", "")

                                        phone = _normalize_phone(phone_raw)
                                        if not phone:
                                            wa_status  = "error"
                                            wa_message = f"અમાન્ય WhatsApp નંબર: {phone_raw}. કૃપા કરીને ૧૦ અંકનો નંબર આપો."
                                            print(f"📲 [WhatsApp] Invalid phone: {phone_raw!r}")
                                        else:
                                            try:
                                                wa_result = await send_whatsapp_message_async(
                                                    to=phone,
                                                    body_params=[answer_text],
                                                    template_name="natural_farming_ai_bot_response",
                                                    language="gu",
                                                )
                                                if wa_result.get("status") == "error":
                                                    wa_status  = "error"
                                                    wa_message = f"WhatsApp ન ગયો: {wa_result.get('error', 'Unknown error')}"
                                                    print(f"📲 [WhatsApp] Send failed to {phone}: {wa_result}")
                                                else:
                                                    wa_status  = "sent"
                                                    wa_message = f"✅ {phone} ને WhatsApp મોકલ્યો."
                                                    print(f"📲 [WhatsApp] Sent to {phone}: {answer_text[:60]}...")
                                            except Exception as wa_exc:
                                                wa_status  = "error"
                                                wa_message = f"WhatsApp error: {wa_exc}"
                                                print(f"📲 [WhatsApp] Exception: {wa_exc}")

                                        function_responses.append({
                                            "response": {
                                                "name": func_name,
                                                "content": {
                                                    "status":  wa_status,
                                                    "message": wa_message,
                                                }
                                            },
                                            "id": call_id
                                        })

                                    else:
                                        # search_agricultural_knowledge_base (or unknown tool)
                                        query = args.get("query", "")
                                        print(f"🔍 [Gemini Live ToolCall: {func_name}] Querying Qdrant for: {query}")

                                        matches = await retrieve_relevant_knowledge(query, limit=3, score_threshold=0.45)
                                        if matches:
                                            results_text = "\n\n".join([f"• [{m.get('title', '')}]: {m.get('text', '')}" for m in matches])
                                            # If experience with audio is retrieved, notify frontend
                                            for m in matches:
                                                if m.get("audio_url"):
                                                    await browser_ws.send_text(json.dumps({
                                                        "type": "experience_available",
                                                        "farmer_name": m.get("farmer_name", "હસમુખભાઈ પટેલ"),
                                                        "district": m.get("district", "વલસાડ"),
                                                        "audio_url": m.get("audio_url"),
                                                        "experience_years": m.get("experience_years", "૫ વર્ષ"),
                                                    }))
                                                    break
                                        else:
                                            results_text = "આ વિષય પર કોઈ ચોક્કસ માહિતી મળી નથી."

                                        function_responses.append({
                                            "response": {
                                                "name": func_name,
                                                "content": {
                                                    "results": results_text
                                                }
                                            },
                                            "id": call_id
                                        })

                                tool_resp_msg = {
                                    "toolResponse": {
                                        "functionResponses": function_responses
                                    }
                                }
                                await gemini_ws.send(json.dumps(tool_resp_msg))
                                print(f"✅ Sent tool response back to Gemini Live ({len(function_responses)} results).")

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
                break  # Session ended normally

        except (websockets.exceptions.InvalidHandshake, TimeoutError, asyncio.TimeoutError) as e:
            if attempt < max_retries:
                print(f"⚠️ Gemini Live handshake attempt {attempt} failed ({e}), retrying in 1s...")
                await asyncio.sleep(1.0)
                continue
            print(f"Gemini live session handshake error: {e}")
            try:
                await browser_ws.send_text(json.dumps({"error": "Gemini લાઈવ સર્વર સાથે જોડાવામાં વિલંબ થયો. કૃપા કરીને ફરી પ્રયાસ કરો."}))
                await browser_ws.close()
            except Exception:
                pass
            break

        except Exception as e:
            print(f"Gemini live session error: {e}")
            try:
                await browser_ws.send_text(json.dumps({"error": f"કનેક્શન ક્ષતિ: {str(e)}"}))
                await browser_ws.close()
            except Exception:
                pass
            break


async def _update_profile_from_conversation(conversation_id: str):
    """
    Background task: extract farmer profile fields from recent conversation turns
    and persist any newly discovered fields into this specific session's FarmerProfile.
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

            profile_id = conv.farmer_id
            if not profile_id:
                default_p = await crud.get_or_create_default_profile(db)
                profile_id = default_p.id

            await crud.update_farmer_profile(db, profile_id, extracted)
            print(f"🌱 Profile updated for session {conversation_id}: {list(extracted.keys())}")

    except Exception as e:
        print(f"[Profile update] Non-critical error: {e}")
