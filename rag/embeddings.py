"""
Embeddings generator using Google Gemini text-embedding-004.
Supports single query embedding and batch embedding for document chunks.
Native support for Gujarati, Hindi, and English text.
"""

import os
from typing import List, Optional
from google import genai
from google.genai import types

_client: Optional[genai.Client] = None
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 768


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


async def get_embedding(text: str) -> List[float]:
    """
    Generate 768-dim embedding for a single text string (e.g. search query).
    """
    clean_text = text.strip()
    if not clean_text:
        return [0.0] * VECTOR_SIZE

    client = _get_client()
    response = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=clean_text,
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
    )
    if response.embeddings and len(response.embeddings) > 0:
        return response.embeddings[0].values
    return [0.0] * VECTOR_SIZE


async def get_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    """
    if not texts:
        return []

    client = _get_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        cleaned_chunk = [t.strip() if t.strip() else " " for t in chunk]
        
        response = await client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=cleaned_chunk,
            config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE),
        )
        if response.embeddings:
            for emb in response.embeddings:
                all_embeddings.append(emb.values)
        else:
            all_embeddings.extend([[0.0] * VECTOR_SIZE for _ in chunk])

    return all_embeddings
