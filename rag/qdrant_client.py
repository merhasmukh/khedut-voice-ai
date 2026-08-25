"""
Qdrant Vector Database Client Manager for Khedut Voice AI.
Connects to Qdrant running via Docker (localhost:6333).
Manages collection lifecycle, vector upserts, and similarity searches.
"""

import os
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels

from .embeddings import VECTOR_SIZE

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "khedut_knowledge")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_URL = os.environ.get("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")

import asyncio

_async_client: Optional[AsyncQdrantClient] = None
_async_client_loop = None
_sync_client: Optional[QdrantClient] = None


def get_qdrant_client() -> AsyncQdrantClient:
    """Get or create AsyncQdrantClient bound to current running event loop."""
    global _async_client, _async_client_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _async_client is None or _async_client_loop != current_loop:
        _async_client = AsyncQdrantClient(
            url=QDRANT_URL,
            timeout=10.0,
            check_compatibility=False,
        )
        _async_client_loop = current_loop
    return _async_client


def get_sync_qdrant_client() -> QdrantClient:
    """Get or create singleton sync QdrantClient."""
    global _sync_client
    if _sync_client is None:
        _sync_client = QdrantClient(
            url=QDRANT_URL,
            timeout=10.0,
            check_compatibility=False,
        )
    return _sync_client


async def is_qdrant_available() -> bool:
    """Check if Qdrant service is reachable."""
    try:
        client = get_qdrant_client()
        collections = await client.get_collections()
        return collections is not None
    except Exception:
        return False


async def init_qdrant_collection() -> bool:
    """
    Ensures the khedut_knowledge collection exists in Qdrant with
    768 vector dimensions and Cosine distance metric.
    """
    try:
        client = get_qdrant_client()
        collections = await client.get_collections()
        existing = [c.name for c in collections.collections]

        if COLLECTION_NAME not in existing:
            print(f"📦 Creating Qdrant collection '{COLLECTION_NAME}' (768-dim, Cosine)...")
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            print(f"✅ Qdrant collection '{COLLECTION_NAME}' created successfully.")
        return True
    except Exception as e:
        print(f"⚠️ Qdrant collection initialization skipped (service might be offline): {e}")
        return False


async def upsert_documents(
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """
    Upsert a batch of document chunks and their vectors into Qdrant.
    Each document dict should contain:
      - text: str (the text chunk)
      - title: str
      - category: str (e.g. 'જીવામૃત', 'કીટ નિયંત્રણ', 'ખાતર')
      - crop: Optional[str] (e.g. 'કપાસ', 'મગફળી', 'બધા')
      - source_file: Optional[str]
      - extra metadata...
    """
    if not documents or not embeddings or len(documents) != len(embeddings):
        return 0

    await init_qdrant_collection()
    client = get_qdrant_client()

    points = []
    for doc, emb in zip(documents, embeddings):
        doc_id = doc.get("id") or str(uuid.uuid4())
        payload = {
            "text": doc.get("text", ""),
            "title": doc.get("title", ""),
            "category": doc.get("category", "સામાન્ય"),
            "crop": doc.get("crop", "બધા"),
            "source_file": doc.get("source_file", ""),
            "language": doc.get("language", "gu"),
            **doc.get("metadata", {}),
        }
        points.append(
            qmodels.PointStruct(
                id=doc_id,
                vector=emb,
                payload=payload,
            )
        )

    # Upsert in batches of 50
    batch_size = 50
    total_upserted = 0
    for i in range(0, len(points), batch_size):
        chunk = points[i : i + batch_size]
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=chunk,
        )
        total_upserted += len(chunk)

    return total_upserted


async def search_knowledge(
    query_vector: List[float],
    limit: int = 4,
    score_threshold: float = 0.55,
    crop_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Qdrant for the most similar agricultural knowledge chunks.
    Returns list of matched payloads with relevance scores.
    """
    if not await is_qdrant_available():
        return []

    client = get_qdrant_client()
    query_filter = None

    if crop_filter:
        # Match crop specifically or general knowledge ("બધા" / "all")
        query_filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="crop",
                    match=qmodels.MatchValue(value=crop_filter),
                ),
                qmodels.FieldCondition(
                    key="crop",
                    match=qmodels.MatchValue(value="બધા"),
                ),
            ]
        )

    try:
        # Use query_points for modern qdrant-client
        results = await client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        matches = []
        for point in results.points:
            p = point.payload or {}
            matches.append({
                "id": str(point.id),
                "score": float(point.score),
                "text": p.get("text", ""),
                "title": p.get("title", ""),
                "category": p.get("category", ""),
                "crop": p.get("crop", ""),
                "source_file": p.get("source_file", ""),
                "farmer_name": p.get("farmer_name", ""),
                "district": p.get("district", ""),
                "experience_years": p.get("experience_years", ""),
                "audio_url": p.get("audio_url", ""),
                "is_experience": p.get("is_experience", False),
            })
        return matches
    except Exception as e:
        print(f"⚠️ Qdrant search error: {e}")
        return []


async def get_collection_stats() -> Dict[str, Any]:
    """Retrieve collection status and total vector count."""
    try:
        if not await is_qdrant_available():
            return {"status": "offline", "points_count": 0, "vectors_count": 0}
        
        client = get_qdrant_client()
        info = await client.get_collection(collection_name=COLLECTION_NAME)
        
        points = getattr(info, "points_count", 0) or 0
        vectors = getattr(info, "indexed_vectors_count", points) or points
        
        vector_size = VECTOR_SIZE
        distance = "Cosine"
        try:
            if hasattr(info.config.params, "vectors"):
                v_params = info.config.params.vectors
                if hasattr(v_params, "size"):
                    vector_size = v_params.size
                if hasattr(v_params, "distance"):
                    distance = str(v_params.distance.value if hasattr(v_params.distance, "value") else v_params.distance)
        except Exception:
            pass

        return {
            "status": "online",
            "collection_name": COLLECTION_NAME,
            "points_count": points,
            "vectors_count": vectors,
            "vector_size": vector_size,
            "distance": distance,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "points_count": 0}
