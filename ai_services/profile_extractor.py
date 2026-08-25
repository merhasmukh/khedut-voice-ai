"""
Profile Extractor — automatically identifies farmer details from conversation text.
Uses the Gemini non-live API to extract structured profile fields as JSON.
Runs as a background task after each voice turn completes.
"""

import json
import os
import re
from typing import Optional
from google import genai
from google.genai import types

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


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


async def extract_profile_from_conversation(messages: list[dict]) -> dict:
    """
    Calls Gemini to extract farmer profile fields from recent messages.
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
        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=EXTRACTION_PROMPT + conv_text,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip() if response.text else ""
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
