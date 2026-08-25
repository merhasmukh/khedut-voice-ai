"""
Profile Extractor — automatically identifies farmer details from conversation text.
Uses the Gemini REST API to extract structured profile fields as JSON.
Runs as a background task after each voice turn completes.
100% compatible with Python 3.9+ via httpx (no google-genai SDK needed).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional
import httpx

GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"
EXTRACTION_MODEL = "gemini-3.6-flash"


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY not set")
    return key


EXTRACTION_PROMPT = """\
You are analyzing a dialogue between a Gujarati farmer (ખેડૂત) and an AI farming assistant.
Your goal is to extract personal farming profile details that the FARMER (ખેડૂત) EXPLICITLY and CLEARLY stated about themselves.

STRICT EXTRACTION RULES:
1. Extract details ONLY from what the FARMER (ખેડૂત) said in their own messages. NEVER extract or guess anything from the AI's messages.
2. "name": Extract the farmer's personal name ONLY if the farmer explicitly introduced themselves (e.g., "મારું નામ રમેશભાઈ છે" or "હું મુકેશભાઈ બોલું છું").
   - NEVER extract greetings, titles, or names said by the AI (e.g. "મિની", "મિનીબેન", "ભાઈ", "બેન", "મિત્ર").
   - If the farmer did not introduce themselves explicitly, "name" MUST be null.
3. "village" and "district": Extract ONLY if the farmer explicitly mentioned their own village, town, or district.
4. "crops": Extract ONLY if the farmer explicitly mentioned the crops they grow or own.
5. "land_acres": Extract ONLY if the farmer mentioned their farm size (convert bigha/hectare to acres if needed).
6. If you are not 100% certain that the farmer stated it, set the field to null. DO NOT guess or assume.

JSON Output Format (use null for any unmentioned field):
{
  "name": "farmer's name if explicitly stated by farmer, else null",
  "village": "village name if stated by farmer, else null",
  "district": "district name if stated by farmer, else null",
  "land_acres": number or null,
  "crops": ["crop1", "crop2"] or null,
  "soil_type": "soil type or null",
  "farming_type": "farming type or null",
  "notes": "specific note or null"
}

Return ONLY valid JSON.

Conversation:
"""


async def extract_profile_from_conversation(messages: list) -> dict:
    """
    Calls Gemini REST API to extract farmer profile fields from recent messages.
    Returns a dict of non-null fields found. Returns {} if extraction fails.
    
    messages: list of {"role": "user"|"assistant", "content": "text"}
    """
    if not messages:
        return {}

    # Must contain at least one user message
    user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content", "").strip()]
    if not user_msgs:
        return {}

    # Build conversation text clearly labeling farmer vs assistant
    conv_text = "\n".join(
        f"{'ખેડૂત' if m['role'] == 'user' else 'AI'}: {m['content']}"
        for m in messages
        if m.get("content", "").strip()
    )
    if not conv_text.strip():
        return {}

    try:
        api_key = _get_api_key()
        url = f"{GEMINI_REST_BASE}/models/{EXTRACTION_MODEL}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"parts": [{"text": EXTRACTION_PROMPT + conv_text}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return {}

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return {}

        raw = parts[0].get("text", "").strip()
        if not raw:
            return {}

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        extracted = json.loads(raw)

        # Remove null / empty values — only keep what is genuinely found
        return {
            k: v for k, v in extracted.items()
            if v is not None and v != "" and v != []
        }

    except Exception as e:
        print(f"[Profile extractor] Extraction failed (non-critical): {e}")
        return {}
