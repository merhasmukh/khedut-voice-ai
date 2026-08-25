"""
Embeddings generator using Google Gemini text-embedding-001 / text-embedding-004.
Supports single query embedding and batch embedding for document chunks.
Native support for Gujarati, Hindi, and English text.
100% compatible with Python 3.9+ via direct httpx REST calls.
"""

from __future__ import annotations

import os
from typing import List, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 768
GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return key


async def get_embedding(text: str) -> List[float]:
    """
    Generate 768-dim embedding for a single text string (e.g. search query).
    Works on Python 3.9+ using async httpx without requiring google-genai SDK.
    """
    clean_text = text.strip()
    if not clean_text:
        return [0.0] * VECTOR_SIZE

    api_key = _get_api_key()
    url = f"{GEMINI_REST_BASE}/models/{EMBEDDING_MODEL}:embedContent?key={api_key}"
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": clean_text}]},
        "outputDimensionality": VECTOR_SIZE,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            values = data.get("embedding", {}).get("values", [])
            if values:
                return values
    except Exception as exc:
        print(f"⚠️ Embedding API error: {exc}")

    return [0.0] * VECTOR_SIZE


async def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    Works on Python 3.9+ using async httpx without requiring google-genai SDK.
    """
    if not texts:
        return []

    api_key = _get_api_key()
    url = f"{GEMINI_REST_BASE}/models/{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    all_embeddings: List[List[float]] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            requests_payload = [
                {
                    "model": f"models/{EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": t.strip() if t.strip() else " "}]},
                    "outputDimensionality": VECTOR_SIZE,
                }
                for t in chunk
            ]

            try:
                resp = await client.post(url, json={"requests": requests_payload})
                resp.raise_for_status()
                data = resp.json()
                embeddings_list = data.get("embeddings", [])
                for emb in embeddings_list:
                    all_embeddings.append(emb.get("values", [0.0] * VECTOR_SIZE))
            except Exception as exc:
                print(f"⚠️ Batch embedding error on batch {i // batch_size + 1}: {exc}")
                all_embeddings.extend([[0.0] * VECTOR_SIZE for _ in chunk])

    return all_embeddings
