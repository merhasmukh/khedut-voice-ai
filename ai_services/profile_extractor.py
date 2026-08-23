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
Read this conversation between a farmer and an AI farming assistant in Gujarati.
Extract any personal farming details the farmer revealed about themselves.

Return a JSON object with ONLY the fields you are confident about from the conversation.
Omit any field you are not sure about — do not guess or assume anything.

Fields to extract (use null if not mentioned):
{
  "name": "farmer's name if they said their name, else null",
  "village": "village name if mentioned, else null",
  "district": "district/taluka if mentioned, else null",
  "land_acres": a number if land size was mentioned in acres/bigha/hectare (convert to acres), else null,
  "crops": ["list", "of", "crops"] mentioned by the farmer as their current crops, else null,
  "soil_type": "soil type if described (e.g. black, sandy, loamy), else null",
  "farming_type": "e.g. organic, natural, chemical, mixed — if mentioned, else null",
  "notes": "any other specific farming detail worth remembering, else null"
}

Return ONLY valid JSON — no explanation, no markdown, no extra text.

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

    # Build conversation text
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
